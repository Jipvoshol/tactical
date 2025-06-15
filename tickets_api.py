
ACCESS_TOKEN = (
    "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJhdWQiOiI0SU5oVjJxU1NGSmpxZjF4bkpVSzdMQ1cwbXBZRWVpOXNUMmt0V3hVIiwianRpIjoiMjE1ZTRhZTNmZTNiOTE4MWY5YTA4NDVlMTgxM2ZkYTA1YWQ5NGJlMjdiZDgxMzY0NmEyMTQzMzFmMDkyZmFlMDNlZDJmOTFkNjlmODQyMGYiLCJpYXQiOjE3NDcwNDAwMzEsIm5iZiI6MTc0NzA0MDAzMSwiZXhwIjoxNzQ3Mjk5MjMxLCJzdWIiOiJhNWFlMGE0MC0zOGNkLTExZWQtYWFmOC04YmQzZmQ4ZTRmNWQiLCJzY29wZXMiOltdLCJhZG1pbiI6ZmFsc2UsImNvbXBhbnkiOiIzNzk0ZDE5MC02NTk0LTQ2YTAtODg4OC0yYjZmNjIyODc4MTQiLCJ2ZXIiOjIsIndsIjoiMmU2ZDExZjAtMzY2OC0xMWU2LThhMGItOTEwN2I4ZGRiYzU1In0.qVGRHlbU5XPfb4lhmv7AUHGXq1uM-Pu8AyqtNB6VPrlVTf-6DVWaPy1oIjRTJRK0a49zexQsQRWSJAHcj2jRS7LwDLacy1gCC_uMXs1Jz93OF-mMWu0wRajnUy5SuJ4xkk74js2UZnTfW6QANNwBilo__LAGyA1rSkfJSKyNPxORGeKGq7fkG-idPQaObK0w7HvycTMPQvc4OwgtInAIimDR8rfsZWt03dONc_9-yGndfLmkoKZWuLFUXmJFoV9rXwV_y6Gb9Tg4K9i77qtsyN9sA3zhRbLAGuTOr8BFx8LwXsslkvyQRUtPa62w3vkDiLAykZcRVo9w7zkvEhF7B6KqXTduF1Y7INbaAOpRlZnONXEV2E8MmF8zPv_TSwVIRdg335Vnqg4XhrAFxCsDE-cqB8_1Yqh292uf2px8F7JUyUDOa8M6uXbndB_aTcNJ7nJWGt36L-yaVdq56CrwpCeGLejjE_A1a5Bsh1Q_QuxQWtc3NBFSPruHChQvIyA1VxTG-8-CH9hJAtP3CzdlPYRcPbTZ8w7WP4ZdZfo1S6GoB3NeFDIAUubm2-hv15w9PTUDWyXKqwqK94tA9CYBH1gTbcMbBhAhXedpuaJh09xsQBPTrngKBLddiWtefez6Wz0XkTJgCK6zW0NoWSzuTU0_JXy_DRKQm2s1WZJmi04"
)
import argparse, base64, json, os, sys, time, csv, math, random, requests
from pathlib import Path
from datetime import datetime, date
from typing import List, Dict, Any, Optional
import pandas as pd
import chardet

# ACCESS_TOKEN = (
#     "your-token-here"
# )

API_BASE = "https://api.weeztix.com"

COLUMNS = [
    "order_id", "shop_name", "event_name", "event_category", "event_subcategories",
    "first_event_date_start", "last_event_date_end", "ticket_name", "barcode",
    "product_name", "is_optional", "order_status", "created_at", "paid_currency",
    "product_value", "product_fees", "product_refunded_amount", "payment_id",
    "payment_method", "device", "product_is_scanned",
    "order.metadata.date_of_birth", "order.metadata.city", "order.metadata.gender",
    "geolocation.locality", "geolocation.admin_level_1", "geolocation.admin_level_2",
    "geolocation.country_code", "geolocation.latitude", "geolocation.longitude",
    "ticket_pdf_link", "days_to_event_start", "days_to_event_end", "purchase_dow",
    "purchase_hour", "age_at_purchase", "age_bin", "gender_filled", "country_code",
    "city_cleaned", "total_price", "refunded", "payment_method_cat",
    "tickets_in_order", "lineup", "artist_mentioned", "ticket_type", "lineup_score",
    "lineup_bin", "event_date", "total_tickets_sold", "line_up_id",
    "daily_events_city", "daily_events_province", "sales_period", "campaign_period",
    "pre_sale", "max_capacity", "full_club_ratio"
]

def _hdr(tok: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {tok}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }

def _company_from_jwt(tok: str) -> Optional[str]:
    try:
        p = json.loads(base64.urlsafe_b64decode(tok.split(".")[1] + "===").decode())
        return p.get("company")
    except Exception:
        return None

def _future_event_guids(tok: str, cid: str) -> List[str]:
    qs = {"as": cid} if cid else {}
    data = requests.get(f"{API_BASE}/event", headers=_hdr(tok), params=qs, timeout=10).json()
    evs = data if isinstance(data, list) else data.get("data", [])
    today = date.today()
    return [e["guid"] for e in evs if "start" in e and date.fromisoformat(e["start"][:10]) >= today]

def _link_from(obj: dict) -> Optional[str]:
    dlinks = obj.get("download_links") or {}
    if isinstance(dlinks, dict) and dlinks.get("csv"):
        return dlinks["csv"]
    for k in ("download_link", "file_link", "csv"):
        if isinstance(obj.get(k), str) and obj[k].startswith("http"):
            return obj[k]
    for f in obj.get("files", []):
        if (f.get("format") or "").lower() == "csv":
            return f.get("download_link") or f.get("file_link") or f.get("link")
    return None

def _wait_csv_link(exp_guid: str, tok: str, cid: str, poll: int, ttl: int) -> str:
    qs, hdr = ({"as": cid} if cid else {}), _hdr(tok)
    waited = 0
    while waited < ttl:
        time.sleep(poll)
        waited += poll
        lst = requests.get(f"{API_BASE}/exports", headers=hdr, params=qs, timeout=10).json()
        exports = lst if isinstance(lst, list) else lst.get("data", [])
        cur = next((e for e in exports if e.get("guid") == exp_guid), {})
        link = _link_from(cur)
        if link:
            return link
        try:
            det = requests.get(f"{API_BASE}/export/{exp_guid}", headers=hdr, params=qs, timeout=10).json()
            link = _link_from(det)
            if link:
                return link
        except requests.HTTPError:
            pass
    raise TimeoutError("Timeout – CSV link not available")

def detect_encoding(path: Path) -> str:
    raw = path.read_bytes()[:100_000]
    enc = chardet.detect(raw)["encoding"] or "latin-1"
    return enc

def safe_csv_to_df(path: Path, encoding: str) -> pd.DataFrame:
    with path.open("r", encoding=encoding, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if header and header[0] == "":
            header = header[1:]
        header_len = len(header)
        rows = []
        for row in reader:
            if len(row) < header_len:
                row += [""] * (header_len - len(row))
            elif len(row) > header_len:
                row = row[:header_len]
            rows.append(row)
    return pd.DataFrame(rows, columns=header)

def postprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    for dcol in ["first_event_date_start", "last_event_date_end", "created_at"]:
        df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df["days_to_event_start"] = (df["first_event_date_start"] - df["created_at"]).dt.total_seconds() / 86_400
    df["days_to_event_end"] = (df["last_event_date_end"] - df["created_at"]).dt.total_seconds() / 86_400
    df["purchase_dow"] = df["created_at"].dt.dayofweek
    df["purchase_hour"] = df["created_at"].dt.hour
    dob = pd.to_datetime(df["order.metadata.date_of_birth"], errors="coerce")
    age = (df["created_at"] - dob).dt.total_seconds() / (365.25 * 24 * 3600)
    df["age_at_purchase"] = age.round(2)
    bins = [0, 18, 25, 35, 45, 60, 120]
    labels = ["<18", "18-24", "25-34", "35-44", "45-59", "60+"]
    df["age_bin"] = pd.cut(df["age_at_purchase"], bins=bins, labels=labels, right=False)
    df["gender_filled"] = df["order.metadata.gender"].fillna("Unknown")
    df["payment_method_cat"] = df["payment_method"].str.lower().map({
        "ideal": "iDeal", "creditcard": "Card", "paypal": "PayPal"
    }).fillna("Other")
    df["product_value"] = pd.to_numeric(df["product_value"], errors="coerce")
    df["product_fees"] = pd.to_numeric(df["product_fees"], errors="coerce")
    df["product_refunded_amount"] = pd.to_numeric(df["product_refunded_amount"], errors="coerce")
    df["total_price"] = df["product_value"] + df["product_fees"]
    df["refunded"] = df["product_refunded_amount"] > 0
    df["tickets_in_order"] = df.groupby("order_id")["order_id"].transform("count")
    df["country_code"] = df["geolocation.country_code"].fillna(df["order.metadata.country"])
    df["city_cleaned"] = df["geolocation.locality"].fillna(df["order.metadata.city"])
    df = df.reindex(columns=COLUMNS)
    df.reset_index(inplace=True)
    df.rename(columns={"index": ""}, inplace=True)
    return df

def generate_unique_filename(prefix: str = "weeztix_export_final", extension: str = ".csv") -> Path:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = random.randint(1000, 9999)
    filename = f"{prefix}_{now}_{suffix}{extension}"
    return Path(filename).resolve()

# ---------------- MAIN ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--access-token")
    ap.add_argument("-c", "--company-id")
    ap.add_argument("-o", "--outfile")
    args = ap.parse_args()

    token = ACCESS_TOKEN
    if not token:
        sys.exit("⚠️  Provide JWT token via -t or env variable.")

    company_id = args.company_id or _company_from_jwt(token)
    ev_guids = _future_event_guids(token, company_id)
    if not ev_guids:
        sys.exit("ℹ️  No upcoming events.")

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    body = {
        "name": f"api_export_{now}",
        "type": "orders_and_order_tickets_and_order_products",
        "filetypes": ["csv"],
        "events": ev_guids,
        "timezone": "Europe/Amsterdam",
    }
    qs = {"as": company_id} if company_id else {}
    exp_resp = requests.post(f"{API_BASE}/export", headers=_hdr(token), params=qs, json=body, timeout=15).json()
    exp_guid = exp_resp["guid"]
    print(f"⌛ Job GUID = {exp_guid}")

    try:
        csv_url = _wait_csv_link(exp_guid, token, company_id, poll=5, ttl=600)
    except Exception as e:
        sys.exit(str(e))

    raw_path = Path(f"weeztix_raw_{now}.csv").resolve()
    raw_path.write_bytes(requests.get(csv_url, timeout=60).content)
    print(f"📦 Raw CSV saved → {raw_path}")

    enc = detect_encoding(raw_path)
    raw_df = safe_csv_to_df(raw_path, enc)
    final_df = postprocess(raw_df)

    out_path = generate_unique_filename()
    final_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✔ Final file saved as → {out_path}")

if __name__ == "__main__":
    main()
