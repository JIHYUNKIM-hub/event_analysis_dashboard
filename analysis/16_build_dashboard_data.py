"""대시보드(07_dashboard.py) 배포용 압축 데이터 만들기.

events_with_category.csv(08_join_category.py 결과)에서 대시보드가 쓰는 열만 남겨 gzip CSV로 저장한다.
행은 그대로이며(정제·카테고리 조인 기준 유지), GitHub 파일 크기 제한(100MB) 안에 들도록 열만 줄인다.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "processed" / "events_with_category.csv"
CLEAN = ROOT / "data" / "processed" / "events_clean.csv"
OUT = ROOT / "data" / "processed" / "dashboard_events.csv.gz"
COLS = ["timestamp", "visitorid", "event", "itemid", "transactionid", "categoryid", "root_categoryid"]

df = pd.read_csv(SRC, usecols=COLS, dtype={"transactionid": "Int64", "categoryid": "Int64", "root_categoryid": "Int64"})
df = df[COLS]

# 정제 데이터와 행이 같은지 확인 (08번 조인은 left join이라 행 수·순서가 같아야 한다)
clean = pd.read_csv(CLEAN, usecols=["timestamp", "visitorid", "event", "itemid", "transactionid"],
                    dtype={"transactionid": "Int64"})
assert len(df) == len(clean)
assert df[clean.columns].reset_index(drop=True).equals(clean[clean.columns].reset_index(drop=True))

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False, compression={"method": "gzip", "mtime": 0})
print(f"저장: {OUT.relative_to(ROOT)} ({len(df):,}행, {OUT.stat().st_size / 1e6:.1f}MB)")
