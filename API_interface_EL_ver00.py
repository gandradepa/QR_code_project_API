#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/developer/API/API_interface_EL_ver00.py

"""
EL interface with a structured logging output.

This version is refactored for robustness with an adaptive processing strategy.
It now detects if a single image or multiple images are provided for an asset and
uses a tailored, hyper-focused AI prompt for each scenario to ensure reliability.
"""

import os
import re
import io
import json
import base64
import shutil
import sqlite3
import argparse
import platform
import logging
from typing import Dict, List, Tuple, Optional, Set, Any
from contextlib import closing
from collections import defaultdict

# Imaging / OCR
try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None
import numpy as np
from PIL import Image
import pytesseract as pt

# OpenAI
from dotenv import load_dotenv
from openai import OpenAI

# -------------------- Constants --------------------
DEFAULT_IMAGE_DIR  = r"/home/developer/Capture_photos_upload"
DEFAULT_OUTPUT_DIR = r"/home/developer/Output_jason_api"
DEFAULT_DB_PATH    = r"/home/developer/asset_capture_app_dev/data/QR_codes.db"
DB_TABLE           = "sdi_dataset_EL"

VALID_SUFFIXES: Set[str] = {"0", "1", "2"}
VALID_EXTS: Set[str]     = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# A simpler validation rule: must contain at least one letter and one digit.
SIMPLE_TAG_VALIDATION = re.compile(r"^(?=.*[a-zA-Z])(?=.*\d).+$")

# Fields for completeness score calculation
COMPLETENESS_FIELDS: List[str] = ["UBC Asset Tag", "Ampere", "Supply From", "Location", "Volts"]


# -------------------- Logging Setup --------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# -------------------- Environment --------------------

def init_openai(dotenv_path: Optional[str]) -> OpenAI:
    if dotenv_path and os.path.isfile(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
    elif os.path.isfile("/home/developer/API/OpenAI_key_bryan.env"):
        load_dotenv(dotenv_path="/home/developer/API/OpenAI_key_bryan.env")
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found")
    return OpenAI(api_key=api_key)


def init_tesseract_quiet() -> None:
    # This function is not used by the core logic but kept for potential future use.
    pass

# -------------------- Utilities --------------------

def _normalize_qr(s: str) -> str:
    s = str(s).strip()
    m = re.match(r"\d+", s)
    if not m:
        return s
    core = m.group(0).lstrip("0")
    return core or "0"


def _detect_columns(conn: sqlite3.Connection, table: str) -> Tuple[str, str]:
    wanted_qr_names = {"QR Code", "QR_code_ID", "QRCode", "QR", "QR_code"}
    cols: Dict[str, str] = {}
    for _, name, *_ in conn.execute(f'PRAGMA table_info("{table}")').fetchall():
        cols[name.lower()] = name
    qr_col = next((cols[c.lower()] for c in wanted_qr_names if c.lower() in cols), None)
    if not qr_col:
        raise RuntimeError(f"QR column not found in {table}")
    if "approved" not in cols:
        raise RuntimeError("Approved column not found")
    return qr_col, cols["approved"]


def load_eligible_qrs(db_path: str, table: str) -> Set[str]:
    eligible: Set[str] = set()
    if not os.path.exists(db_path):
        return eligible
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            qr_col, approved_col = _detect_columns(conn, table)
            sql = f'''
                SELECT "{qr_col}" AS qr
                FROM "{table}"
                WHERE COALESCE(CAST({approved_col} AS INTEGER), 0) <> 1
            '''
            for row in conn.execute(sql):
                qr_raw = str(row["qr"]).strip()
                if qr_raw:
                    eligible.add(_normalize_qr(qr_raw))
    except Exception as e:
        logging.error(f"Failed to load eligible QRs from database: {e}")
        return eligible
    return eligible


def _calculate_completeness(data: Dict[str, str], fields: List[str]) -> float:
    """Calculates the percentage of specified fields that are non-empty."""
    if not fields:
        return 100.0
    present_count = sum(1 for field in fields if data.get(field, "").strip())
    return (present_count / len(fields)) * 100.0


def _get_case_insensitive(data: Dict[str, Any], key: str, default: Any = "") -> Any:
    """Gets a value from a dict using a case-insensitive and space/underscore insensitive key."""
    if not isinstance(data, dict):
        return default
    key_norm = key.lower().replace(" ", "").replace("_", "")
    for k, v in data.items():
        k_norm = k.lower().replace(" ", "").replace("_", "")
        if k_norm == key_norm:
            return v
    return default


def encode_image_from_path(image_path: str) -> Optional[str]:
    try:
        ext = os.path.splitext(image_path)[1].lower()
        mime = f"image/{ext.replace('.', '')}"
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None

# -------------------- Grouping --------------------

def build_groups(image_dir: str, eligible_qrs: Set[str], qr_filter: Optional[str]) -> Dict[str, Dict]:
    pattern = re.compile(r"^(\d+)\s+(\d+(?:-\d+)?)\s+(EL)\s*-\s*([012])$", re.IGNORECASE)
    groups: Dict[str, Dict] = defaultdict(lambda: {"building": "", "images": {}, "asset": "EL"})
    for fn in os.listdir(image_dir):
        base, ext = os.path.splitext(fn)
        if ext.lower() not in VALID_EXTS:
            continue
        m = pattern.match(base)
        if not m:
            continue
        qr, building, asset_type, seq = m.groups()
        if seq not in VALID_SUFFIXES or asset_type.upper() != "EL":
            continue
        if qr_filter and _normalize_qr(qr) != _normalize_qr(qr_filter):
            continue
        if eligible_qrs and _normalize_qr(qr) not in eligible_qrs:
            continue
        groups[qr]["building"] = building
        groups[qr]["images"][seq] = os.path.join(image_dir, fn)
    return groups

# -------------------- OpenAI Extraction (Refactored) --------------------

def _call_openai(client: OpenAI, content: List[Dict], debug: bool = False) -> Dict[str, Any]:
    """Generic wrapper for OpenAI API calls."""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": content}],
            temperature=0.0,
            max_tokens=800,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        if debug:
            logging.debug(f"RAW AI RESPONSE:\n{raw}")
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logging.error(f"AI call failed: {e}")
        return {}


def ask_model_for_single_tag(client: OpenAI, image_b64: str, debug: bool = False) -> Dict[str, Any]:
    """Makes a hyper-focused API call to extract only the asset tag from a single image."""
    prompt = (
        "Your single task is to identify and extract the most prominent alphanumeric identifier from the provided image. "
        "This is the 'UBC Asset Tag'.\n\n"
        "**Rules**:\n"
        "1.  Find the main ID. It is often on a colored physical label (e.g., a yellow sticker).\n"
        "2.  Your entire response must be a single JSON object with ONE key: \"UBC Asset Tag\".\n"
        "   Example: {\"UBC Asset Tag\": \"CDP 2S0D1\"}"
    )
    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_b64, "detail": "high"}},
    ]
    return _call_openai(client, content, debug)


def ask_model_multi_image(client: OpenAI,
                          el0_b64: Optional[str],
                          el1_b64: Optional[str],
                          el2_b64: Optional[str],
                          debug: bool = False) -> Dict[str, Any]:
    """Makes a unified, multi-image API call with clear instructions for sourcing data."""
    content: List[Dict] = []
    prompt = (
        "You are an expert at extracting data from electrical panel labels. Analyze the provided images and extract the following fields into a strict JSON object. Use an empty string \"\" if a value is missing.\n\n"
        "**Fields to Extract**:\n"
        "- UBC Asset Tag\n- Branch Panel\n- Ampere\n- Supply From\n- Volts\n- Location\n\n"
        "**Instructions**:\n"
        "1.  **Examine all images** to get the full context.\n"
        "2.  The **panel schedule (Image 1)** contains most of the data, like 'Location' (top-right), 'Volts', and 'Ampere'.\n"
        "3.  The **asset tag label (Image 2)** is the primary source for the 'UBC Asset Tag'.\n"
        "4.  Fill in all the fields you can find.\n"
        "5.  **Output**: Your entire response MUST be ONLY the JSON object."
    )
    content.append({"type": "text", "text": prompt})
    
    if el0_b64:
        content.append({"type": "text", "text": "\n--- Image 1: Panel Schedule (EL-0) ---"})
        content.append({"type": "image_url", "image_url": {"url": el0_b64, "detail": "high"}})
    if el1_b64:
        content.append({"type": "text", "text": "\n--- Image 2: Asset Tag Label (EL-1) ---"})
        content.append({"type": "image_url", "image_url": {"url": el1_b64, "detail": "high"}})
    if el2_b64:
        content.append({"type": "text", "text": "\n--- Image 3: Context (EL-2) ---"})
        content.append({"type": "image_url", "image_url": {"url": el2_b64, "detail": "auto"}})
    
    return _call_openai(client, content, debug)


# -------------------- Processing --------------------

def process_group(client: OpenAI, qr: str, building: str, paths: Dict[str, str], output_dir: str, debug: bool = False) -> bool:
    logging.info(f"Processing asset QR: {qr}...")
    
    raw_data = {}
    
    # Adaptive Strategy: Check number of images and choose the best AI prompt.
    if len(paths) == 1:
        single_image_path = list(paths.values())[0]
        if single_image_b64 := encode_image_from_path(single_image_path):
             if debug: logging.info("Single image 'Tag-Only' mode: Querying AI...")
             raw_data = ask_model_for_single_tag(client, single_image_b64, debug)
    else:
        el0_b64 = encode_image_from_path(paths.get("0", "")) if paths.get("0") else None
        el1_b64 = encode_image_from_path(paths.get("1", "")) if paths.get("1") else None
        el2_b64 = encode_image_from_path(paths.get("2", "")) if paths.get("2") else None
        if debug: logging.info("Multi-image 'Unified Context' mode: Querying AI...")
        raw_data = ask_model_multi_image(client, el0_b64, el1_b64, el2_b64, debug)

    if not raw_data:
        logging.error(f"Failed to get a valid response from AI for QR: {qr}")
        return False

    # Use case-insensitive parsing for all fields
    all_fields = COMPLETENESS_FIELDS + ["Branch Panel", "Description"]
    data: Dict[str, str] = {
        field: str(_get_case_insensitive(raw_data, field)).strip()
        for field in all_fields
    }

    # Post-processing and validation
    ubc_val = data.get("UBC Asset Tag", "").upper().strip()
    ubc_val = ubc_val.replace("EQUIPMENT NAME:", "").replace("MAIN", "").strip()
    if not SIMPLE_TAG_VALIDATION.search(ubc_val):
        ubc_val = ""

    branch_panel = data.get("Branch Panel", "").upper().strip()
    if not ubc_val and SIMPLE_TAG_VALIDATION.search(branch_panel):
        ubc_val = branch_panel
    
    data["UBC Asset Tag"] = ubc_val
    data["Description"] = f"Panel - {ubc_val}" if ubc_val else "Panel - "
    data["Branch Panel"] = branch_panel

    for k in ["Ampere", "Supply From", "Volts", "Location"]:
        v = str(data.get(k, "") or "").strip()
        if v == ".": v = ""
        data[k] = v
        
    completeness_score = _calculate_completeness(data, COMPLETENESS_FIELDS)

    result = {
        "qr_code": qr,
        "building_number": building,
        "asset_type": "- EL",
        "structured_data": {
            "Description": data.get("Description", ""),
            "UBC Asset Tag": data.get("UBC Asset Tag", ""),
            "Branch Panel": data.get("Branch Panel", ""),
            "Ampere": data.get("Ampere", ""),
            "Supply From": data.get("Supply From", ""),
            "Volts": data.get("Volts", ""),
            "Location": data.get("Location", ""),
        },
        "completeness_score": completeness_score,
    }

    json_filename = f"{qr}_EL_{building}.json"
    out_path = os.path.join(output_dir, json_filename)

    try:
        with open(out_path, "w", encoding="utf-8") as jf:
            json.dump(result, jf, ensure_ascii=True, indent=2)
    except Exception as e:
        logging.error(f"Failed to save JSON file for QR {qr}: {e}")
        return False

    logging.info(f"Successfully processed and saved asset QR: {qr} (Completeness: {completeness_score:.0f}%)")
    return True

# -------------------- Main --------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="EL OCR/Extraction with structured logging")
    ap.add_argument("--qr", dest="qr_filter", default=None, help="Process a single QR only (e.g., 0000183710)")
    ap.add_argument("--db", dest="db_path", default=DEFAULT_DB_PATH, help="Path to QR_codes.db")
    ap.add_argument("--images-dir", dest="images_dir", default=DEFAULT_IMAGE_DIR, help="Path to image directory")
    ap.add_argument("--output-dir", dest="output_dir", default=DEFAULT_OUTPUT_DIR, help="Path to output JSON dir")
    ap.add_argument("--env", dest="dotenv_path", default=None, help="Path to .env containing OPENAI_API_KEY")
    ap.add_argument("--debug", action="store_true", help="Enable debug mode to print raw AI responses.")
    args = ap.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    os.makedirs(args.output_dir, exist_ok=True)
    client = init_openai(args.dotenv_path)

    eligible = load_eligible_qrs(args.db_path, DB_TABLE)
    groups = build_groups(args.images_dir, eligible, qr_filter=args.qr_filter)

    logging.info(f"Found {len(groups)} new assets to process.")
    
    saved_count = 0
    for qr, info in groups.items():
        success = process_group(
            client=client,
            qr=qr,
            building=info.get("building", ""),
            paths=info.get("images", {}),
            output_dir=args.output_dir,
            debug=args.debug,
        )
        if success:
            saved_count += 1
            
    # Print final summary
    print("\n--- SUMMARY ---")
    print(f"Successfully saved: {saved_count}")


if __name__ == "__main__":
    main()