"""events.csv 데이터 품질 점검 (원본은 읽기만 하고 수정하지 않는다)."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "events.csv"

df = pd.read_csv(RAW, dtype={"transactionid": "Int64"})
print(f"행 수: {len(df):,} / 열: {list(df.columns)}")
print("\n[dtypes]\n", df.dtypes, sep="")

# 1. 이벤트 종류별 건수
print("\n[이벤트 종류별 건수]")
cnt = df["event"].value_counts(dropna=False)
print(pd.DataFrame({"count": cnt, "ratio(%)": (cnt / len(df) * 100).round(3)}))

# 2. 열별 결측
print("\n[열별 결측]")
na = df.isna().sum()
print(pd.DataFrame({"missing": na, "ratio(%)": (na / len(df) * 100).round(3)}))

# 3. event와 transactionid 정합성
has_tid = df["transactionid"].notna()
is_tx = df["event"] == "transaction"
print("\n[event x transactionid 유무]")
print(pd.crosstab(df["event"], has_tid.rename("has_transactionid")))
print(f"구매가 아닌데 transactionid 있음: {(~is_tx & has_tid).sum():,}")
print(f"구매인데 transactionid 없음: {(is_tx & ~has_tid).sum():,}")

# 4. 기타 품질 점검
print("\n[완전 중복 행]")
dup_all = df.duplicated()
print(f"완전 중복: {dup_all.sum():,}")
print("이벤트별 완전 중복:\n", df[dup_all]["event"].value_counts(), sep="")

key = ["visitorid", "event", "itemid", "transactionid"]
print(f"\n[timestamp 제외 동일 행(같은 방문자·이벤트·상품·거래)] {df.duplicated(key).sum():,}")

print("\n[timestamp 범위/값]")
ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
print(f"min: {ts.min()}  max: {ts.max()}")
print(f"timestamp 음수/0: {(df['timestamp'] <= 0).sum():,}")
daily = ts.dt.date.value_counts().sort_index()
print(f"일수: {len(daily)}, 일별 건수 min {daily.min():,} / median {int(daily.median()):,} / max {daily.max():,}")
print("일별 건수 하위 5일:\n", daily.nsmallest(5), sep="")

print("\n[식별자 값 범위·고유수]")
for c in ["visitorid", "itemid", "transactionid"]:
    s = df[c].dropna()
    print(f"{c}: nunique {s.nunique():,}, min {s.min()}, max {s.max()}, 음수 {(s < 0).sum()}")

print("\n[event 값 표기 (공백/대소문자)]")
print(df["event"].unique())

# 거래 단위 점검: 하나의 transactionid가 여러 방문자에 걸쳐 있는지
tx = df[is_tx]
tid_visitors = tx.groupby("transactionid")["visitorid"].nunique()
print("\n[transactionid 점검]")
print(f"거래 수: {tx['transactionid'].nunique():,}, 구매 행: {len(tx):,}")
print(f"transactionid당 방문자 2명 이상: {(tid_visitors > 1).sum():,}")
items_per_tx = tx.groupby("transactionid")["itemid"].size()
print(f"거래당 행 수: mean {items_per_tx.mean():.2f}, max {items_per_tx.max()}")
tid_ts = tx.groupby("transactionid")["timestamp"].nunique()
print(f"transactionid 내 timestamp 2개 이상: {(tid_ts > 1).sum():,}")
print(f"transactionid+itemid 중복 행: {tx.duplicated(['transactionid', 'itemid']).sum():,}")

# UTC 날짜로 자르면 시작/끝이 03:00이므로 UTC-3 기준 범위도 확인
ts_local = ts - pd.Timedelta(hours=3)
print(f"\n[UTC-3 기준 범위] {ts_local.min()} ~ {ts_local.max()}")

# 퍼널 순서 점검: 구매/장바구니 전에 해당 상품 view가 있었는지
first = df.groupby(["visitorid", "itemid", "event"])["timestamp"].min().unstack()
print("\n[방문자·상품 단위 퍼널 선행 이벤트 점검]")
for ev in ["addtocart", "transaction"]:
    sub = first[first[ev].notna()]
    no_view = sub["view"].isna().sum()
    view_after = (sub["view"] > sub[ev]).sum()
    print(f"{ev} 발생 쌍 {len(sub):,}: view 없음 {no_view:,}, 첫 view가 이후 {view_after:,}")
sub = first[first["transaction"].notna()]
print(f"transaction 쌍 중 addtocart 없음: {sub['addtocart'].isna().sum():,}, "
      f"첫 addtocart가 이후: {(sub['addtocart'] > sub['transaction']).sum():,}")

# 방문자 활동량 이상치(봇 의심)
per_visitor = df["visitorid"].value_counts()
print("\n[방문자당 이벤트 수]")
print(per_visitor.describe(percentiles=[0.5, 0.9, 0.99, 0.999]).round(1))
print("상위 5명:\n", per_visitor.head(), sep="")
print(f"이벤트 1건뿐인 방문자: {(per_visitor == 1).sum():,} ({(per_visitor == 1).mean() * 100:.1f}%)")
top = df[df["visitorid"].isin(per_visitor.head(5).index)]
print("상위 5명 이벤트 구성:\n", pd.crosstab(top["visitorid"], top["event"]), sep="")
