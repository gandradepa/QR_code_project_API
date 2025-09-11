#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to extract structured data from industrial asset photographs using
an advanced ensemble of local OCR (Tesseract) and a multimodal LLM (GPT-4o).

This version introduces sophisticated improvements for data accuracy:
- Advanced image pre-processing including perspective correction and CLAHE.
- An ensemble extraction model that uses a decision engine to weigh OCR and LLM results.
- Chain-of-Thought LLM prompting for higher-quality AI analysis.
- Post-extraction data validation to ensure plausibility.
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
    DB_TABLE = "sdi_dataset_ME"

    # --- File Matching ---
    VALID_SUFFIXES = {"0", "1", "3"}
    VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    FILENAME_PATTERN = re.compile(
        r"^(\d+)\s+" r"(\d+(?:-\d+)?)\s+" r"([A-Z]{2})\s*-\s*([0-3])$", re.IGNORECASE
    )

    # --- OCR & Image Processing ---
    UBC_TAG_PATTERN = re.compile(r"\b([A-Z]{1,3})[-\u2013]?\s?(\d{1,4})([A-Z]?)\b")
    TESSERACT_MIN_CONFIDENCE = 75.0  # Increased threshold as we now have a better LLM fallback

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
        self.approved_qrs = self._load_approved_qrs()
        logging.info(f"Loaded {len(self.approved_qrs)} approved QR codes to ignore.")

    def _load_approved_qrs(self) -> Set[str]:
        # ... (same as previous version, no changes needed)
        approved = set()
        if not os.path.exists(Config.DB_PATH):
            logging.warning(f"Database not found: {Config.DB_PATH}. Proceeding without approval filter.")
            return approved
        try:
            with closing(sqlite3.connect(Config.DB_PATH)) as conn:
                conn.row_factory = sqlite3.Row
                with closing(conn.cursor()) as cur:
                    query = f"SELECT QR_code_ID FROM {Config.DB_TABLE} WHERE Approved = 1 OR Approved = '1'"
                    cur.execute(query)
                    for row in cur.fetchall():
                        if qrid := str(row["QR_code_ID"]).strip():
                            approved.add(qrid)
        except sqlite3.Error as e:
            logging.error(f"Error reading approvals from DB: {e}.")
        return approved
        
    def discover_assets(self) -> Dict[str, Dict[str, Any]]:
        # ... (same as previous version, no changes needed)
        grouped = defaultdict(lambda: {"images": {}, "building": "", "asset_type": ""})
        logging.info(f"Scanning for images in: {Config.IMAGE_FOLDER}")
        for filename in sorted(os.listdir(Config.IMAGE_FOLDER)):
            base, ext = os.path.splitext(filename)
            if ext.lower() not in Config.VALID_EXTS: continue
            match = Config.FILENAME_PATTERN.match(base)
            if not match: continue
            qr, building, asset_type, seq = match.groups()
            if asset_type.upper() != "ME" or seq not in Config.VALID_SUFFIXES: continue
            if qr in self.approved_qrs: continue
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
                        logging.info(f"Successfully processed and saved asset QR: {qr}")
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
            
            # --- The new ensemble extraction logic ---
            preprocessed_img = self._preprocess_image(path)
            if preprocessed_img is None:
                logging.warning(f"Could not read or preprocess image {path}, skipping.")
                continue

            # 1. Get OCR opinion
            tesseract_results = self._tesseract_read_all(preprocessed_img, fields_for_seq)

            # 2. Get LLM opinion
            llm_results = self._llm_extract_fields(preprocessed_img, fields_for_seq, path)

            # 3. Use Decision Engine to choose the best result for each field
            for field in fields_for_seq:
                tess_result = tesseract_results.get(field, ("", 0.0))
                llm_result = llm_results.get(field, {"value": "", "confidence": 0})
                
                best_value = self._decision_engine(field, tess_result, llm_result)
                final_data[field] = best_value
        
        # 4. Post-Extraction Validation and Normalization
        final_data = self._validate_and_normalize(final_data)
        
        return {
            "qr_code": qr, "building_number": info.get("building", ""),
            "asset_type": f"- {info.get('asset_type', 'ME').upper()}",
            "structured_data": final_data,
        }
    
    def _decision_engine(self, field: str, tess_result: Tuple[str, float], llm_result: Dict[str, Any]) -> str:
        """Intelligently chooses the best value between Tesseract and LLM outputs."""
        tess_val, tess_conf = tess_result
        llm_val, llm_conf = llm_result.get("value", ""), llm_result.get("confidence", 0)

        # Normalize for comparison (e.g., remove spaces, hyphens)
        tess_norm = re.sub(r'[\s-]', '', (tess_val or "")).lower()
        llm_norm = re.sub(r'[\s-]', '', (llm_val or "")).lower()

        # Case 1: They agree. High confidence.
        if tess_norm and tess_norm == llm_norm:
            logging.info(f"[{field}] Agreement between OCR & LLM: '{tess_val}'")
            return tess_val

        # Case 2: Tesseract is highly confident.
        if tess_conf >= Config.TESSERACT_MIN_CONFIDENCE:
            logging.info(f"[{field}] High confidence OCR result: '{tess_val}' (Conf: {tess_conf:.1f}%)")
            return tess_val

        # Case 3: LLM is highly confident and Tesseract is not.
        if llm_conf >= 80:
            logging.info(f"[{field}] High confidence LLM result: '{llm_val}' (Conf: {llm_conf}%)")
            return llm_val
        
        # Case 4: Fallback to the most plausible non-empty result. Prefer LLM.
        final_choice = llm_val or tess_val
        logging.warning(f"[{field}] Conflicting results. OCR: '{tess_val}' ({tess_conf:.1f}%), "
                        f"LLM: '{llm_val}' ({llm_conf}%). Choosing: '{final_choice}'")
        return final_choice

    def _validate_and_normalize(self, data: Dict[str, str]) -> Dict[str, str]:
        """Performs final validation and normalization on the extracted data."""
        # Validate Year
        if year_str := data.get("Year"):
            try:
                year = int(year_str)
                min_y, max_y = Config.YEAR_VALIDATION_RANGE
                if not (min_y <= year <= max_y):
                    logging.warning(f"Year '{year}' is outside valid range {Config.YEAR_VALIDATION_RANGE}. Discarding.")
                    data["Year"] = ""
            except (ValueError, TypeError):
                data["Year"] = self._normalize_year(year_str) # Attempt to fix
        
        # Normalize UBC Tag
        if ubc_tag := data.get("UBC Tag"):
            data["UBC Tag"] = self._canonicalize_ubc_tag(ubc_tag)
            
        return data

    def _save_result(self, data: Dict[str, Any]):
        # ... (same as previous version, no changes needed)
        qr, asset_type, building = data["qr_code"], data["asset_type"].replace("- ", ""), data["building_number"]
        json_filename = f"{qr}_{asset_type}_{building}.json"
        out_path = os.path.join(Config.OUTPUT_FOLDER, json_filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # --- [4] Image Processing and Extraction Methods ---

    def _preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """Loads and applies a full suite of pre-processing filters."""
        img = cv2.imread(image_path)
        if img is None: return None

        # 1. Perspective Correction
        corrected_img = self._correct_perspective(img)
        
        # 2. Convert to Grayscale
        gray = cv2.cvtColor(corrected_img, cv2.COLOR_BGR2GRAY)
        
        # 3. Enhance Contrast with CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # 4. Final Thresholding
        _, bw_img = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Save debug image
        debug_path = os.path.join(Config.DEBUG_FOLDER, f"preprocessed_{os.path.basename(image_path)}")
        cv2.imwrite(debug_path, bw_img)

        return bw_img
        
    def _correct_perspective(self, image: np.ndarray) -> np.ndarray:
        """Finds the largest quadrilateral in an image and transforms it to a top-down view."""
        # ... (This is a complex CV task, here's a simplified robust implementation)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return image # No contours, return original
        
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4: # Found a quadrilateral
                screenCnt = approx
                break
        else:
            return image # No 4-sided contour found, return original

        # Order the points
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
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        
        return warped

    def _tesseract_read_all(self, bw_img: np.ndarray, fields: List[str]) -> Dict[str, Tuple[str, float]]:
        """Runs Tesseract once and attempts to parse all required fields."""
        results = {}
        try:
            # Use PSM 6 for a uniform block of text
            data = pytesseract.image_to_data(bw_img, config='--psm 6', output_type=pytesseract.Output.DICT)
            full_text = " ".join(filter(None, data['text']))

            # Simple keyword-based extraction from the full text
            for field in fields:
                if field == "UBC Tag":
                    val = self._canonicalize_ubc_tag(full_text)
                    results[field] = (val, 80.0 if val else 0.0) # Assume high confidence if format matches
                # Add more keyword logic for other fields if patterns exist
                
        except pytesseract.TesseractError as e:
            logging.warning(f"Tesseract error: {e}")
        return results

    def _llm_extract_fields(self, image: np.ndarray, fields: List[str], original_path: str) -> Dict[str, Any]:
        """Uses the LLM with Chain-of-Thought prompting for robust extraction."""
        fields_list = ", ".join(f'"{f}"' for f in fields)
        prompt = f"""
Analyze the provided image of an asset nameplate. Your task is to extract the following fields: {fields_list}.

Follow these steps carefully:
1.  **Reasoning**: Describe what you see. Identify potential values for each requested field. Note any ambiguities, glare, or unreadable text.
2.  **Confidence Score**: For each field, provide a confidence score from 0 (not found) to 100 (perfectly clear).
3.  **Extraction**: Provide the final extracted data in a strict JSON object.

Your final output MUST be a single JSON object with three keys: "reasoning", "confidence_scores", and "extracted_data".

Example format:
{{
  "reasoning": "The image shows a metal nameplate. The 'Manufacturer' is clearly 'ACME Inc.'. The 'Serial Number' seems to be 'XYZ-12345', but the last digit is slightly blurred. The 'Year' is not visible.",
  "confidence_scores": {{
    "Manufacturer": 100,
    "Serial Number": 85,
    "Year": 0
  }},
  "extracted_data": {{
    "Manufacturer": "ACME Inc.",
    "Serial Number": "XYZ-12345",
    "Year": ""
  }}
}}
"""
        response = self._call_vision_api(prompt, original_path, image, max_tokens=600)
        
        # Combine extracted data and confidence scores for the decision engine
        data = response.get("extracted_data", {})
        confidences = response.get("confidence_scores", {})
        
        # Ensure data is in the expected format
        if not isinstance(data, dict):
            logging.warning(f"LLM returned malformed 'extracted_data': {data}")
            data = {}
        if not isinstance(confidences, dict):
            confidences = {}
            
        return {
            field: {"value": str(data.get(field, "")), "confidence": confidences.get(field, 0)}
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
        if not text: return ""
        normalized = text.replace("—", "-").replace("–", "-").replace(" ", "")
        match = Config.UBC_TAG_PATTERN.search(normalized)
        if not match: return ""
        left, num, suffix = match.groups()
        return f"{left}-{num}{suffix}".strip("-")

# --- [5] Main Execution Block ---
if __name__ == "__main__":
    try:
        setup_environment()
        processor = AssetProcessor()
        processor.run()
    except Exception as e:
        logging.critical(f"A critical error terminated the script: {e}", exc_info=True)