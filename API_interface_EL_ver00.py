#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: /home/developer/API/API_interface_EL_ver00.py

"""
EL interface with ASCII-only summary output matching BF template

Summary format (ASCII-only):

Total assets found (after approval filter): N
Processing QR {QR} ...
Saved /home/developer/Output_jason_api/{QR}_EL_{BUILDING}.json

Keeps existing setups: paths, DB table sdi_dataset_EL, images dir, output dir.
Filters on Approved <> 1. Uses EL-0, EL-1, EL-2 images. Minimal console output.
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
from typing import Dict, List, Tuple, Optional, Set
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
HEADER_FRACTION: float   = 0.25
UBC_TAG_CANDIDATE = re.compile(r"\b([A-Z]{2,5})[ -]?(?=[A-Z0-9]*\d)[A-Z0-9]{2,12}\b")

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
    cmd: Optional[str] = None
    if platform.system() == "Windows":
        for c in (r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
                  r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"):
            if os.path.exists(c):
                cmd = c
                break
    cmd = cmd or shutil.which("tesseract") or "/usr/bin/tesseract"
    pt.pytesseract.tesseract_cmd = cmd
    for td in ("/usr/share/tesseract-ocr/5/tessdata",
               "/usr/share/tesseract-ocr/4.00/tessdata",
               "/usr/share/tesseract-ocr/tessdata"):
        if os.path.isdir(td):
            os.environ.setdefault("TESSDATA_PREFIX", td)
            break


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
    except Exception:
        return eligible
    return eligible


def encode_image_from_path(image_path: str) -> Optional[str]:
    try:
        mime = "image/jpeg"
        ext = os.path.splitext(image_path)[1].lower()
        if ext == ".png":
            mime = "image/png"
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


def _save_array_as_jpeg(arr: np.ndarray, path: str) -> None:
    try:
        if cv2 is not None:
            cv2.imwrite(path, arr)
        else:
            img = Image.fromarray(arr.astype(np.uint8), mode=("L" if arr.ndim == 2 else "RGB"))
            img.save(path, format="JPEG", quality=90)
    except Exception:
        pass


def encode_image_from_ndarray(img: np.ndarray) -> Optional[str]:
    try:
        if cv2 is not None:
            ok, buf = cv2.imencode(".jpg", img)
            if not ok:
                return None
            b = buf.tobytes()
        else:
            pil = Image.fromarray(img.astype(np.uint8), mode=("L" if img.ndim == 2 else "RGB"))
            bio = io.BytesIO()
            pil.save(bio, format="JPEG", quality=90)
            b = bio.getvalue()
        b64 = base64.b64encode(b).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def crop_header_top(image_path: str, fraction: float = HEADER_FRACTION) -> Optional[np.ndarray]:
    try:
        if cv2 is not None:
            img = cv2.imread(image_path)
            if img is None:
                return None
            h = img.shape[0]
            crop_h = max(1, int(h * fraction))
            return img[:crop_h, :, :]
        img = Image.open(image_path).convert("RGB")
        arr = np.array(img)
        h = arr.shape[0]
        crop_h = max(1, int(h * fraction))
        return arr[:crop_h, :, :]
    except Exception:
        return None


def _binarize_gray(gray: np.ndarray) -> np.ndarray:
    thresh = float(np.mean(gray)) if gray.size else 127.0
    return ((gray > thresh).astype(np.uint8) * 255)


def quick_ocr_text(img_path: str) -> str:
    try:
        if cv2 is not None:
            img = cv2.imread(img_path)
            if img is None:
                return ""
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
            return pt.image_to_string(bw, config="--oem 3 --psm 6 -l eng") or ""
        img = Image.open(img_path).convert("L")
        arr = np.array(img)
        bw = _binarize_gray(arr)
        return pt.image_to_string(bw, config="--oem 3 --psm 6 -l eng") or ""
    except Exception:
        return ""


def find_ubc_tag_hint_from_el1(img_path: Optional[str]) -> str:
    if not img_path:
        return ""
    t = quick_ocr_text(img_path).upper()
    m = UBC_TAG_CANDIDATE.search(t)
    return m.group(0) if m else ""

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

# -------------------- OpenAI Extraction --------------------

def ask_model(client: OpenAI, header_img_b64: Optional[str], el1_b64: Optional[str], el2_b64: Optional[str], ubc_hint: str) -> Dict[str, str]:
    content: List[Dict] = []
    prompt = (
        "You will see up to three images for an ELECTRICAL PANEL.\n\n"
        "- First image: HEADER CROP (EL-0). Extract header-only fields.\n"
        "- Second image: UBC Asset Tag label (EL-1).\n"
        "- Third image: optional context (EL-2).\n\n"
        "Extract EXACT fields as strict JSON (empty string if missing):\n"
        "- Description\n- UBC Asset Tag\n- Branch Panel\n- Ampere\n- Supply From\n- Volts\n- Location\n\n"
        "Rules:\n"
        "1) UBC Asset Tag primarily from EL-1; if none or no digits, leave empty.\n"
        "2) Other fields from EL-0 header.\n"
        "3) Description must be 'Panel - <UBC Asset Tag>'.\n"
        "4) Strict JSON only."
    )
    content.append({"type": "text", "text": prompt + f"\nHint: {ubc_hint}"})
    if header_img_b64:
        content.append({"type": "image_url", "image_url": {"url": header_img_b64}})
    if el1_b64:
        content.append({"type": "image_url", "image_url": {"url": el1_b64}})
    if el2_b64:
        content.append({"type": "image_url", "image_url": {"url": el2_b64}})

    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": content}],
            temperature=0.1,
            max_tokens=800,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception:
        return {}

    if raw.startswith("```json"):
        raw = raw[7:].strip()
    elif raw.startswith("```"):
        raw = raw[3:].strip()
    if raw.endswith("```"):
        raw = raw[:-3].strip()

    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return {}
        out: Dict[str, str] = {k: str(data.get(k, "") or "").strip() for k in [
            "Description", "UBC Asset Tag", "Branch Panel", "Ampere", "Supply From", "Volts", "Location"
        ]}
        return out
    except Exception:
        return {}

# -------------------- Processing --------------------

def process_group(client: OpenAI, qr: str, building: str, paths: Dict[str, str], output_dir: str) -> Optional[str]:
    print(f"Processing QR {qr} ...", flush=True)

    # Build image inputs
    header_img = crop_header_top(paths.get("0", ""), HEADER_FRACTION) if paths.get("0") else None
    header_b64 = encode_image_from_ndarray(header_img) if header_img is not None else (
        encode_image_from_path(paths.get("0", "")) if paths.get("0") else None
    )
    el1_b64 = encode_image_from_path(paths.get("1", "")) if paths.get("1") else None
    el2_b64 = encode_image_from_path(paths.get("2", "")) if paths.get("2") else None

    ubc_hint = find_ubc_tag_hint_from_el1(paths.get("1"))

    data = ask_model(client, header_b64, el1_b64, el2_b64, ubc_hint)

    # Minimal fallback structure
    if not isinstance(data, dict):
        data = {}
    ubc_val = (data.get("UBC Asset Tag", "") or "").upper().strip()
    if not UBC_TAG_CANDIDATE.search(ubc_val):
        ubc_val = ""
    branch_panel = (data.get("Branch Panel", "") or "").upper().strip()
    if not ubc_val and branch_panel:
        ubc_val = branch_panel

    data["UBC Asset Tag"] = ubc_val
    data["Description"] = f"Panel - {ubc_val}" if ubc_val else "Panel - "
    for k in ["Ampere", "Supply From", "Volts", "Location", "Branch Panel"]:
        v = str(data.get(k, "") or "").strip()
        if v == ".":
            v = ""
        data[k] = v

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
    }

    # File name pattern to match BF dashboard
    json_filename = f"{qr}_EL_{building}.json"
    out_path = os.path.join(output_dir, json_filename)

    try:
        with open(out_path, "w", encoding="utf-8") as jf:
            json.dump(result, jf, ensure_ascii=True, indent=2)
    except Exception:
        return None

    print(f"Saved {out_path}", flush=True)
    return out_path

# -------------------- Main --------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="EL OCR/Extraction (ASCII summary)")
    ap.add_argument("--qr", dest="qr_filter", default=None, help="Process a single QR only (e.g., 0000183710)")
    ap.add_argument("--db", dest="db_path", default=DEFAULT_DB_PATH, help="Path to QR_codes.db")
    ap.add_argument("--images-dir", dest="images_dir", default=DEFAULT_IMAGE_DIR, help="Path to image directory")
    ap.add_argument("--output-dir", dest="output_dir", default=DEFAULT_OUTPUT_DIR, help="Path to output JSON dir")
    ap.add_argument("--env", dest="dotenv_path", default=None, help="Path to .env containing OPENAI_API_KEY")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    init_tesseract_quiet()
    client = init_openai(args.dotenv_path)

    eligible = load_eligible_qrs(args.db_path, DB_TABLE)
    groups = build_groups(args.images_dir, eligible, qr_filter=args.qr_filter)

    print(f"Total assets found (after approval filter): {len(groups)}", flush=True)

    for qr, info in groups.items():
        process_group(
            client=client,
            qr=qr,
            building=info.get("building", ""),
            paths=info.get("images", {}),
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
