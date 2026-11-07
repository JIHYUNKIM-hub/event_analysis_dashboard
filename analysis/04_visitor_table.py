"""방문자 단위 파생 테이블 (02_clean_events.py 정제 데이터 기준, UTC 날짜).

열 정의
- purchase_count: 고유 transactionid 수 (한 거래에 상품이 여러 개여도 1회)
- purchase_amount: 구매 총 금액. 현재 데이터(events, 카테고리 파일)에 가격 정보가 없어 전부 결측으로 둔다.
- first_visit_date: 첫 이벤트(view/addtocart/transaction 무관) 발생일. 관측 시작(2015-05-03) 이전 이력은 알 수 없다.
- last_purchase_date: 마지막 transaction 발생일. 구매가 없으면 결측
- view_count / addtocart_count: view / addtocart 이벤트 행 수
- first_visit_in_first_4weeks: 첫 방문일이 첫 4주(UTC 2015-05-03 ~ 2015-05-30)에 속하는지. 좌측 절단 구분용
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "processed" / "events_clean.csv"
OUT = ROOT / "data" / "processed" / "visitor_summary.csv"

FIRST_4WEEKS_END = pd.Timestamp("2015-05-31", tz="UTC")

df = pd.read_csv(CLEAN, usecols=["event_time", "visitorid", "event", "transactionid"],
                 dtype={"transactionid": "Int64"}, parse_dates=["event_time"])

g = df.groupby("visitorid")
tx = df[df["event"] == "transaction"].groupby("visitorid")
first_visit = g["event_time"].min()

visitors = pd.DataFrame({
    "purchase_count": tx["transactionid"].nunique(),
    "purchase_amount": pd.NA,
    "first_visit_date": first_visit.dt.date,
    "last_purchase_date": tx["event_time"].max().dt.date,
    "view_count": df[df["event"] == "view"].groupby("visitorid").size(),
    "addtocart_count": df[df["event"] == "addtocart"].groupby("visitorid").size(),
    "first_visit_in_first_4weeks": first_visit < FIRST_4WEEKS_END,
}, index=first_visit.index)

count_cols = ["purchase_count", "view_count", "addtocart_count"]
visitors[count_cols] = visitors[count_cols].fillna(0).astype("int64")
visitors["purchase_amount"] = visitors["purchase_amount"].astype("Float64")
visitors = visitors.reset_index()

OUT.parent.mkdir(parents=True, exist_ok=True)
visitors.to_csv(OUT, index=False)

print(f"방문자 수: {len(visitors):,} (정제 데이터 고유 visitorid {df['visitorid'].nunique():,})")
print(visitors.head().to_string(index=False))
print("\n[요약]")
print(visitors[count_cols].describe().round(2).to_string())
buyers = visitors["purchase_count"] > 0
print(f"\n구매 방문자: {buyers.sum():,} ({buyers.mean() * 100:.2f}%)")
print(f"첫 방문이 첫 4주인 방문자: {visitors['first_visit_in_first_4weeks'].sum():,}")
print(f"결측: {visitors.isna().sum().to_dict()}")
print(f"저장: {OUT.relative_to(ROOT)}")
