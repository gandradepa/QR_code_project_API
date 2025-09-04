import os
import json
import base64
import re
import time
from collections import defaultdict
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI

# --- OCR / image libs ---
import cv2
import numpy as np
from PIL import Image  # noqa: F401
import pytesseract

# NEW: database
import sqlite3
from contextlib import closing

import platform, shutil, pytesseract
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract") or "tesseract"


# --- [1] Load API key ---
load_dotenv(dotenv_path=r"/home/developer/API/OpenAI_key_bryan.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- [2] Paths & constants ---
image_folder  = r"/home/developer/Capture_photos_upload"
output_folder = r"/home/developer/Output_jason_api"
debug_folder  = os.path.join(output_folder, "debug_ubc_tag")
os.makedirs(output_folder, exist_ok=True)
os.makedirs(debug_folder, exist_ok=True)

# NEW: DB path (SQLite)
DB_PATH = r"/home/developer/asset_capture_app_dev/data/QR_codes.db"
DB_TABLE = "QR_codes"  # precisa conter QR_code_ID e Approved

VALID_SUFFIXES = {"0", "1", "3"}
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

FIELD_SOURCES: Dict[str, List[str]] = {
    "Manufacturer": ["0"],
    "Model": ["0"],
    "Serial Number": ["0"],
    "Year": ["0"],
    "UBC Tag": ["1"],
    "Technical Safety BC": ["3"],
}

UBC_TAG_PATTERN = re.compile(r"\b([A-Z]{1,3})[-\u2013]?\s?(\d{1,4})([A-Z]?)\b")  # e.g., P-12A, AH-203

# --- [2.1] DB helpers (NEW) ---
def load_approved_qrs(db_path: str, table: str = "QR_codes") -> set:
    """
    Retorna um set de QR_code_ID em que Approved == 1.
    Funciona se Approved for INTEGER 1 ou TEXT '1'.
    """
    approved = set()
    if not os.path.exists(db_path):
        print(f"⚠ DB não encontrado: {db_path}. Seguindo sem filtro de aprovação.")
        return approved
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            with closing(conn.cursor()) as cur:
                cur.execute(f"""
                    SELECT QR_code_ID
                    FROM {table}
                    WHERE (Approved = 1 OR Approved = '1')
                """)
                for row in cur.fetchall():
                    qrid = str(row["QR_code_ID"]).strip()
                    if qrid:
                        approved.add(qrid)
    except Exception as e:
        print(f"⚠ Erro ao ler aprovações do DB: {e}. Seguindo sem filtro de aprovação.")
    return approved

APPROVED_QRS = load_approved_qrs(DB_PATH, DB_TABLE)
if APPROVED_QRS:
    print(f"Filtro de aprovação carregado: {len(APPROVED_QRS)} QR(s) serão ignorados.")

# --- [3] Group files by QR ---
# "<QR> <Building> ME - <Sequence>"
pattern = re.compile(
    r"^(\d+)\s+"
    r"(\d+(?:-\d+)?)\s+"
    r"(ME)\s*-\s*([013])$",
    re.IGNORECASE
)

grouped = defaultdict(lambda: {"images": {}, "building": "", "asset_type": "ME"})

for fn in os.listdir(image_folder):
    base, ext = os.path.splitext(fn)
    if ext.lower() not in VALID_EXTS:
        continue
    m = pattern.match(base)
    if not m:
        print(f"⚠ Skipping unrecognized filename: {fn}")
        continue
    qr, building, asset_type, seq = m.groups()
    if seq not in VALID_SUFFIXES or asset_type.upper() != "ME":
        continue

    # NEW: pular logo no agrupamento se Approved=1
    if qr in APPROVED_QRS:
        print(f"⏭️  Skipping QR {qr} (Approved=1 no DB)")
        continue

    grouped[qr]["building"] = building
    grouped[qr]["images"][seq] = os.path.join(image_folder, fn)

print(f"\nTotal assets found (após filtro de aprovação): {len(grouped)}")

# --- [4] Utilities ---
def encode_image(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    mime = {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".bmp":  "image/bmp",
        ".webp": "image/webp"
    }.get(ext, "application/octet-stream")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def normalize_year(value: str) -> str:
    if not value:
        return ""
    m = re.search(r"\b(19\d{2}|20\d{2})\b", value)
    return m.group(0) if m else ""

def canonicalize_ubc_tag(text: str) -> str:
    if not text:
        return ""
    m = UBC_TAG_PATTERN.search(text.replace("—", "-").replace("–", "-"))
    if not m:
        return ""
    left, num, suffix = m.groups()
    return f"{left}-{num}{suffix}".strip("-")

# --- [5] UBC Tag OCR pipeline ---
def find_white_plate_roi(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 31, 5)
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_area, best_box = 0, None
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        aspect = cw / max(ch, 1)
        if area > 0.02 * w * h and 1.8 <= aspect <= 8.0:
            if area > best_area:
                best_area, best_box = area, (x, y, cw, ch)

    if best_box is None:
        return img_bgr

    x, y, cw, ch = best_box
    pad = int(0.05 * (cw + ch) / 2)
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(w, x + cw + pad); y1 = min(h, y + ch + pad)
    roi = img_bgr[y0:y1, x0:x1]
    return roi if roi.size > 0 else img_bgr

def preprocess_for_ocr(roi_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((2, 2), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
    return bw

def tesseract_read_tag(img_bgr: np.ndarray) -> Tuple[str, float]:
    roi = find_white_plate_roi(img_bgr)
    bw = preprocess_for_ocr(roi)
    debug_name = os.path.join(debug_folder, f"roi_{int(time.time()*1000)}.png")
    cv2.imwrite(debug_name, bw)

    config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
    data = pytesseract.image_to_data(bw, output_type=pytesseract.Output.DICT, config=config)

    texts, confs = [], []
    for txt, conf in zip(data.get("text", []), data.get("conf", [])):
        if not txt or txt.strip() == "" or conf in ("-1", -1):
            continue
        texts.append(txt.strip())
        try:
            confs.append(float(conf))
        except Exception:
            pass

    raw = " ".join(texts).upper().replace(" ", "")
    raw = raw.replace("–", "-").replace("—", "-")
    text = canonicalize_ubc_tag(raw)
    mean_conf = (sum(confs) / len(confs)) if confs else 0.0
    return text, mean_conf

def ask_model_for_ubc_tag(image_path: str) -> str:
    prompt = """
You will see ONE image that contains an equipment identifier printed in large bold black text
on a white rectangular plate (e.g., P-12A). Extract ONLY that identifier.

Rules:
- Return a STRICT JSON object with key "UBC Tag" only.
- Allowed format examples: "P-12A", "AH-203", "P-12". A letter or 1–3 letters, a hyphen, 1–4 digits, optional trailing letter.
- If not visible, return {"UBC Tag": ""}.
Output only JSON.
""".strip()

    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": encode_image(image_path)}},
    ]
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": content}],
                temperature=0.0,
                max_tokens=60
            )
            raw = (resp.choices[0].message.content or "").strip()
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
            data = json.loads(raw)
            val = str(data.get("UBC Tag", "")).upper()
            return canonicalize_ubc_tag(val)
        except Exception:
            time.sleep(1.2 * (attempt + 1))
    return ""

def ask_model_for_fields(image_path: str, fields: List[str]) -> dict:
    fields_list = "\n".join([f"- {f}" for f in fields])
    prompt = f"""
You will see ONE image. Extract ONLY the requested fields below.
Return a STRICT JSON object with EXACT keys, using empty string if missing/unclear.

{fields_list}

Formatting rules:
- "Year" must be a 4-digit year (e.g., "2022"); if not 4-digit, return "".
Do not include any text before or after the JSON.
""".strip()

    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": encode_image(image_path)}},
    ]
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": content}],
                temperature=0.0,
                max_tokens=300
            )
            raw = (resp.choices[0].message.content or "").strip()
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.IGNORECASE)
            data = json.loads(raw)
            return {k: (data.get(k, "") if isinstance(data.get(k, ""), str) else "") for k in fields}
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return {k: "" for k in fields}

# --- [6] Process each asset ---
for qr, info in grouped.items():
    # Double-guard (caso algo mude no agrupamento)
    if qr in APPROVED_QRS:
        print(f"⏭️  Skipping QR {qr} (Approved=1 no DB)")
        continue

    print(f"\nProcessing QR {qr} …")

    result = {
        "Manufacturer": "",
        "Model": "",
        "Serial Number": "",
        "Year": "",
        "UBC Tag": "",
        "Technical Safety BC": ""
    }

    for seq, path in info["images"].items():
        fields_for_seq = [f for f, srcs in FIELD_SOURCES.items() if seq in srcs]
        if not fields_for_seq:
            continue

        if seq == "1" and "UBC Tag" in fields_for_seq:
            img_bgr = cv2.imread(path)
            tag_text, mean_conf = tesseract_read_tag(img_bgr)

            if not tag_text or mean_conf < 65.0:
                model_tag = ask_model_for_ubc_tag(path)
                tag_final = canonicalize_ubc_tag(tag_text) or canonicalize_ubc_tag(model_tag)
            else:
                tag_final = tag_text

            result["UBC Tag"] = tag_final or ""
            print(f"  → UBC Tag (seq 1): '{result['UBC Tag']}' (tesseract_conf≈{mean_conf:.1f})")
            continue

        partial = ask_model_for_fields(path, fields_for_seq)
        if "Year" in partial:
            partial["Year"] = normalize_year(partial.get("Year", ""))

        for k, v in partial.items():
            if isinstance(v, str):
                result[k] = v.strip()

    output_data = {
        "qr_code":         qr,
        "building_number": info.get("building", ""),
        "asset_type":      f"- {info.get('asset_type', 'ME').upper()}",
        "structured_data": result
    }

    json_filename = f"{qr}_{info.get('asset_type', 'ME').upper()}_{info.get('building', '')}.json"
    out_path = os.path.join(output_folder, json_filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"  ✅ Saved {out_path}")
