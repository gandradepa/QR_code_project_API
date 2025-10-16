#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to extract structured data from industrial asset photographs using
an advanced ensemble of local OCR (Tesseract) and a multimodal LLM (GPT-4o).

This version introduces a critical architectural change for improved accuracy:
- Decoupled image processing: The LLM now receives the original image, while
  Tesseract receives a specifically pre-processed version.
- Stricter post-processing of LLM results to handle 'None'/'N/A' values.
- Advanced pre-processing, ensemble model, CoT prompting, and validation remain.
"""

import os
import json
import base64
import re
import time
import logging
import platform
import shutil
import sqlite3
from collections import defaultdict
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any, Optional, Set

# Third-party libraries
import cv2
import numpy as np
import pytesseract
from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, RateLimitError, APIStatusError


# --- [1] Configuration ---
class Config:
    """Centralized configuration for the asset processing script."""
    # --- Paths ---
    ROOT_DEV_PATH = os.getenv("DEV_PATH", "/home/developer")
    IMAGE_FOLDER = os.path.join(ROOT_DEV_PATH, "Capture_photos_upload")
    OUTPUT_FOLDER = os.path.join(ROOT_DEV_PATH, "Output_jason_api")
    DEBUG_FOLDER = os.path.join(OUTPUT_FOLDER, "debug_ubc_tag")
    DB_PATH = os.path.join(ROOT_DEV_PATH, "asset_capture_app_dev/data/QR_codes.db")
    ENV_PATH = os.path.join(ROOT_DEV_PATH, "API/OpenAI_key_bryan.env")

    # --- Database ---
    DB_TABLE = "sdi_dataset"

    # --- File Matching ---
    VALID_SUFFIXES = {"0", "1", "3"}
    VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    FILENAME_PATTERN = re.compile(
        r"^(\d+)\s+" r"(\d+(?:-\d+)?)\s+" r"([A-Z]{2})\s*-\s*([0-3])$", re.IGNORECASE
    )

    # --- OCR & Image Processing ---
    UBC_TAG_PATTERNS: List[str] = [
        # This new pattern accepts a hyphen OR a space as the separator.
        # It will now match 'HUM 5', 'FC-6.32', and 'FH-B124-1'.
        r"([A-Z]{1,4}[-\s][\w\.-]+)",
        
        # Kept as a fallback.
        r"([A-Z]{2,3}-[A-Z0-9]+-[0-9]+)"
    ]
    TESSERACT_MIN_CONFIDENCE = 75.0

    # --- OpenAI API ---
    OPENAI_MODEL = "gpt-4o"
    API_MAX_RETRIES = 3
    API_RETRY_DELAY = 1.5

    # --- Concurrency ---
    MAX_WORKERS = 8

    # --- Field Mapping & Validation ---
    FIELD_SOURCES: Dict[str, List[str]] = {
        "Manufacturer": ["0"], "Model": ["0"], "Serial Number": ["0"],
        "Year": ["0"], "UBC Tag": ["1"], "Technical Safety BC": ["3"],
    }
    EXPECTED_FIELDS: List[str] = list(FIELD_SOURCES.keys())
    YEAR_VALIDATION_RANGE = (1950, 2025)
    
    # --- Completeness Score ---
    COMPLETENESS_SCORE_FIELDS: List[str] = [
        "Manufacturer", "Model", "Serial Number", "Year", "UBC Tag"
    ]


# --- [2] Setup Logging and Environment ---
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def setup_environment():
    """Loads environment variables and configures Tesseract."""
    load_dotenv(dotenv_path=Config.ENV_PATH)
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set.")
    
    if platform.system() == "Windows":
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    else:
        pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract") or "tesseract"
    
    os.makedirs(Config.OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(Config.DEBUG_FOLDER, exist_ok=True)


# --- [3] Asset Processing Class ---
class AssetProcessor:
    """Orchestrates asset data extraction using an advanced ensemble methodology."""

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.qrs_to_ignore = self._load_qrs_to_ignore()
        logging.info(f"Loaded {len(self.qrs_to_ignore)} QR codes to ignore (approved or flagged).")

    def _load_qrs_to_ignore(self) -> Set[str]:
        """Loads QR codes that are already approved or have been flagged."""
        to_ignore = set()
        if not os.path.exists(Config.DB_PATH):
            logging.warning(f"Database not found: {Config.DB_PATH}. Proceeding without filtering processed assets.")
            return to_ignore
        try:
            with closing(sqlite3.connect(Config.DB_PATH)) as conn:
                conn.row_factory = sqlite3.Row
                with closing(conn.cursor()) as cur:
                    query = f'SELECT "QR Code" FROM {Config.DB_TABLE} WHERE Approved = 1 OR Approved = \'1\' OR Flagged = 1 OR Flagged = \'1\''
                    cur.execute(query)
                    for row in cur.fetchall():
                        if qrid := str(row["QR Code"]).strip():
                            to_ignore.add(qrid)
        except sqlite3.Error as e:
            logging.error(f"Error reading DB to filter assets: {e}.")
        return to_ignore
        
    def discover_assets(self) -> Dict[str, Dict[str, Any]]:
        grouped = defaultdict(lambda: {"images": {}, "building": "", "asset_type": ""})
        logging.info(f"Scanning for images in: {Config.IMAGE_FOLDER}")
        for filename in sorted(os.listdir(Config.IMAGE_FOLDER)):
            base, ext = os.path.splitext(filename)
            if ext.lower() not in Config.VALID_EXTS: continue
            match = Config.FILENAME_PATTERN.match(base)
            if not match: continue
            qr, building, asset_type, seq = match.groups()
            if asset_type.upper() != "ME" or seq not in Config.VALID_SUFFIXES: continue
            if qr in self.qrs_to_ignore: continue
            grouped[qr]["building"] = building
            grouped[qr]["asset_type"] = asset_type.upper()
            grouped[qr]["images"][seq] = os.path.join(Config.IMAGE_FOLDER, filename)
        logging.info(f"Found {len(grouped)} new assets to process.")
        return grouped

    def run(self):
        """Main execution flow: discover assets and process them concurrently."""
        assets = self.discover_assets()
        if not assets:
            logging.info("No new assets found. Exiting.")
            return

        saved_count = 0
        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            future_to_qr = {executor.submit(self.process_single_asset, qr, info): qr for qr, info in assets.items()}
            for future in as_completed(future_to_qr):
                qr = future_to_qr[future]
                try:
                    if output_data := future.result():
                        self._save_result(output_data)
                        saved_count += 1
                        logging.info(f"Successfully processed and saved asset QR: {qr} (Completeness: {output_data['completeness_score']:.0f}%)")
                except Exception as e:
                    logging.error(f"Failed to process asset QR {qr}: {e}", exc_info=True)
        
        logging.info(f"\n--- SUMMARY ---\nTotal assets processed: {len(assets)}\nSuccessfully saved: {saved_count}")

    def process_single_asset(self, qr: str, info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Processes all images for a single asset to extract data using the ensemble method."""
        final_data = {key: "" for key in Config.EXPECTED_FIELDS}

        for seq, path in info["images"].items():
            fields_for_seq = [f for f, srcs in Config.FIELD_SOURCES.items() if seq in srcs]
            if not fields_for_seq: continue

            logging.info(f"Processing image {os.path.basename(path)} for fields: {fields_for_seq}")
            
            original_img = cv2.imread(path)
            if original_img is None:
                logging.warning(f"Could not read image {path}, skipping.")
                continue

            # 1. Get OCR opinion using a pre-processed image optimized for Tesseract
            tesseract_img = self._preprocess_for_ocr(original_img, os.path.basename(path))
            tesseract_results = self._tesseract_read_all(tesseract_img, fields_for_seq)

            # 2. Get LLM opinion using the ORIGINAL image for best results
            llm_results = self._llm_extract_fields(original_img, fields_for_seq, path)

            # 3. Use Decision Engine to choose the best result for each field
            for field in fields_for_seq:
                tess_result = tesseract_results.get(field, ("", 0.0))
                llm_result = llm_results.get(field, {"value": "", "confidence": 0})
                
                best_value = self._decision_engine(field, tess_result, llm_result)
                final_data[field] = best_value
        
        final_data = self._validate_and_normalize(final_data)
        
        completeness_score = self._calculate_completeness_score(final_data)
        
        return {
            "qr_code": qr, "building_number": info.get("building", ""),
            "asset_type": f"- {info.get('asset_type', 'ME').upper()}",
            "structured_data": final_data,
            "completeness_score": completeness_score,
        }
    
    def _calculate_completeness_score(self, data: Dict[str, str]) -> float:
        """Calculates the percentage of key fields that are present."""
        if not Config.COMPLETENESS_SCORE_FIELDS:
            return 100.0
            
        present_count = sum(1 for field in Config.COMPLETENESS_SCORE_FIELDS if data.get(field, "").strip())
        total_fields = len(Config.COMPLETENESS_SCORE_FIELDS)
        
        return (present_count / total_fields) * 100

    def _decision_engine(self, field: str, tess_result: Tuple[str, float], llm_result: Dict[str, Any]) -> str:
        """Intelligently chooses the best value between Tesseract and LLM outputs."""
        tess_val, tess_conf = tess_result
        llm_val, llm_conf = llm_result.get("value", ""), llm_result.get("confidence", 0)

        tess_norm = re.sub(r'[\s-]', '', (tess_val or "")).lower()
        llm_norm = re.sub(r'[\s-]', '', (llm_val or "")).lower()

        if llm_conf >= 80:
            logging.info(f"[{field}] High confidence LLM result: '{llm_val}' (Conf: {llm_conf}%)")
            return llm_val
            
        if tess_norm and tess_norm == llm_norm:
            logging.info(f"[{field}] Agreement between OCR & LLM: '{tess_val}'")
            return tess_val

        if tess_conf >= Config.TESSERACT_MIN_CONFIDENCE:
            logging.info(f"[{field}] High confidence OCR result: '{tess_val}' (Conf: {tess_conf:.1f}%)")
            return tess_val
        
        final_choice = llm_val or tess_val
        logging.warning(f"[{field}] Conflicting results. OCR: '{tess_val}' ({tess_conf:.1f}%), "
                        f"LLM: '{llm_val}' ({llm_conf}%). Choosing: '{final_choice}'")
        return final_choice

    def _validate_and_normalize(self, data: Dict[str, str]) -> Dict[str, str]:
        """Performs final validation and normalization on the extracted data."""
        if year_str := data.get("Year"):
            try:
                year = int(year_str)
                min_y, max_y = Config.YEAR_VALIDATION_RANGE
                if not (min_y <= year <= max_y):
                    logging.warning(f"Year '{year}' is outside valid range {Config.YEAR_VALIDATION_RANGE}. Discarding.")
                    data["Year"] = ""
            except (ValueError, TypeError):
                data["Year"] = self._normalize_year(year_str)
        
        if ubc_tag := data.get("UBC Tag"):
            data["UBC Tag"] = self._canonicalize_ubc_tag(ubc_tag)
            
        return data

    def _save_result(self, data: Dict[str, Any]):
        qr, asset_type, building = data["qr_code"], data["asset_type"].replace("- ", ""), data["building_number"]
        json_filename = f"{qr}_{asset_type}_{building}.json"
        out_path = os.path.join(Config.OUTPUT_FOLDER, json_filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # --- [4] Image Processing and Extraction Methods ---

    def _preprocess_for_ocr(self, image_data: np.ndarray, original_filename: str) -> np.ndarray:
        """Applies a suite of pre-processing filters optimized for Tesseract OCR."""
        corrected_img = self._correct_perspective(image_data)
        gray = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        _, bw_img = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        debug_path = os.path.join(Config.DEBUG_FOLDER, f"preprocessed_{original_filename}")
        cv2.imwrite(debug_path, bw_img)
        return bw_img
        
    def _correct_perspective(self, image: np.ndarray) -> np.ndarray:
        """Finds the largest quadrilateral and transforms it to a top-down view."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return image
        
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        screenCnt = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                screenCnt = approx
                break
        
        if screenCnt is None:
            return image

        pts = screenCnt.reshape(4, 2)
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        (tl, tr, br, bl) = rect
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        dst = np.array([
            [0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]
        ], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    def _tesseract_read_all(self, bw_img: np.ndarray, fields: List[str]) -> Dict[str, Tuple[str, float]]:
        """Runs Tesseract once and attempts to parse all required fields."""
        results = {}
        try:
            full_text = pytesseract.image_to_string(bw_img, config='--psm 6')

            for field in fields:
                if field == "UBC Tag":
                    val = self._canonicalize_ubc_tag(full_text)
                    results[field] = (val, 80.0 if val else 0.0)
                
        except pytesseract.TesseractError as e:
            logging.warning(f"Tesseract error: {e}")
        return results

    def _llm_extract_fields(self, original_image: np.ndarray, fields: List[str], original_path: str) -> Dict[str, Any]:
        """Uses the LLM with the original image for robust extraction."""
        fields_list = ", ".join(f'"{f}"' for f in fields)
        
        prompt = f"""
Analyze the provided image of an industrial asset. Your primary task is to extract information for these fields: {fields_list}.

Follow these steps with high precision:
1.  **Reasoning**: Systematically scan the entire image. Describe what you see, including the main nameplate AND any other stickers, tags, or labels. Note any ambiguities, glare, or unreadable text. If a field is not present, state that clearly.

2.  **Special Instructions for Key Fields**:
    * **'Year'**: This may be labeled 'Year', 'Mfg. Date', 'Manufactured Date', or '**Production Date**'. You must extract only the four-digit year from the value (e.g., if the date is '2023/07', the year is '2023').
    * **'Serial Number'**: This may be labeled as 'Serial No.', 'S/N', 'Serial', or similar. **Crucially, if the field next to the label is blank, look for a barcode. The number printed directly below a barcode is almost always the Serial Number.**
    * **'UBC Tag'**: This field is CRITICAL. It is usually on a separate sticker (white, silver, yellow) and not on the main metal nameplate. Look for formats like 'FH-B124-1', 'FC-6.32', or '**HUM 5**' (note the space instead of a hyphen).

3.  **Confidence Score**: For each field, provide a confidence score from 0 (not found/guess) to 100 (perfectly clear and certain).

4.  **Extraction**: Provide the final extracted data in a strict JSON object. If a value cannot be found for any reason, use an empty string "" for that field.

Your final output MUST be a single JSON object with three keys: "reasoning", "confidence_scores", and "extracted_data".

Example format:
{{
  "reasoning": "The image shows a main nameplate. The 'Production Date' is listed as 2023/07, so I will use 2023 for the Year. The 'Serial No.' field is blank, but a barcode number is present below it.",
  "confidence_scores": {{
    "Manufacturer": 95,
    "Model": 100,
    "Serial Number": 98,
    "Year": 100,
    "UBC Tag": 0
  }},
  "extracted_data": {{
    "Manufacturer": "Polar Air",
    "Model": "PDWA(4R)-800-VX-W-AECM-L",
    "Serial Number": "6902307100180",
    "Year": "2023",
    "UBC Tag": ""
  }}
}}
"""
        response = self._call_vision_api(prompt, original_path, original_image, max_tokens=600)
        
        data = response.get("extracted_data", {})
        confidences = response.get("confidence_scores", {})
        
        # Clean the returned data from the LLM
        cleaned_data = {}
        if isinstance(data, dict):
            for field, value in data.items():
                str_val = str(value).strip()
                if str_val.lower() in ["none", "n/a", "null", ""]:
                    cleaned_data[field] = ""
                else:
                    cleaned_data[field] = str_val
        else:
            logging.warning(f"LLM returned malformed 'extracted_data': {data}")

        return {
            field: {"value": cleaned_data.get(field, ""), "confidence": confidences.get(field, 0)}
            for field in fields
        }

    def _call_vision_api(self, prompt: str, image_path: str, image_data: np.ndarray, max_tokens: int) -> Dict[str, Any]:
        """Robust wrapper for OpenAI Vision API calls with retry logic."""
        try:
            b64_image = self._encode_image_from_data(image_data)
        except Exception as e:
            logging.error(f"Could not encode image data from {image_path}: {e}")
            return {}

        content = [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": b64_image}}]
        
        for attempt in range(Config.API_MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=Config.OPENAI_MODEL,
                    messages=[{"role": "user", "content": content}],
                    temperature=0.0, max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                )
                raw_json = (response.choices[0].message.content or "").strip()
                return json.loads(raw_json)
            except (APIConnectionError, RateLimitError, APIStatusError, json.JSONDecodeError) as e:
                logging.warning(f"API/JSON error on attempt {attempt + 1}: {e}. Retrying...")
                time.sleep(Config.API_RETRY_DELAY * (attempt + 1))
            except Exception as e:
                logging.error(f"Unexpected error in API call: {e}", exc_info=True)
                break
        
        logging.error(f"API call failed after {Config.API_MAX_RETRIES} attempts for {image_path}.")
        return {}

    @staticmethod
    def _encode_image_from_data(image_data: np.ndarray, format: str = ".jpg") -> str:
        """Encodes an in-memory np.ndarray image to a base64 data URI."""
        success, buffer = cv2.imencode(format, image_data)
        if not success:
            raise IOError("Could not encode image data.")
        encoded = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    @staticmethod
    def _normalize_year(value: str) -> str:
        if not value: return ""
        match = re.search(r"\b(19\d{2}|20\d{2})\b", str(value))
        return match.group(0) if match else ""

    @staticmethod
    def _canonicalize_ubc_tag(text: str) -> str:
        """Finds a UBC tag in text by trying multiple regex patterns."""
        if not text: return ""
        
        for pattern in Config.UBC_TAG_PATTERNS:
            if match := re.search(pattern, text, re.IGNORECASE):
                return match.group(1).upper().replace(" ", "")
        
        return ""

# --- [5] Main Execution Block ---
if __name__ == "__main__":
    try:
        setup_environment()
        processor = AssetProcessor()
        processor.run()
    except Exception as e:
        logging.critical(f"A critical error terminated the script: {e}", exc_info=True)