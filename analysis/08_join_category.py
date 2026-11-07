"""events_clean.csv 기준 left join으로 상품 카테고리 붙이기.

조인 기준 (data/raw/SOURCE_category.txt: "Match only records at or before the event")
- 같은 itemid에서 effective_timestamp <= 이벤트 timestamp 인 기록 중 가장 최근 것 1개를 붙인다 (as-of join).
- itemid만으로 단순 조인하면 상품당 카테고리 기록이 여러 개(최대 18개)라 이벤트 행이 늘어나므로 쓰지 않는다.
- 이벤트 이후에 생긴 카테고리 기록은 쓰지 않는다. 맞는 기록이 없으면 categoryid/root_categoryid는 결측으로 둔다.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "processed" / "events_clean.csv"
HISTORY = ROOT / "data" / "raw" / "item_category_history.csv"
OUT = ROOT / "data" / "processed" / "events_with_category.csv"
LOG = ROOT / "outputs" / "tables" / "category_join_log.csv"

ev = pd.read_csv(EVENTS, dtype={"transactionid": "Int64"}, parse_dates=["event_time"])
hist = pd.read_csv(HISTORY, dtype={"categoryid": "Int64", "root_categoryid": "Int64"})
assert not hist.duplicated(["itemid", "effective_timestamp"]).any()

ev["_row"] = range(len(ev))
joined = pd.merge_asof(
    ev.sort_values("timestamp"),
    hist.sort_values("effective_timestamp"),
    left_on="timestamp", right_on="effective_timestamp",
    by="itemid", direction="backward",
)
joined = joined.sort_values("_row").drop(columns="_row").reset_index(drop=True)
joined["effective_timestamp"] = joined["effective_timestamp"].astype("Int64")
assert len(joined) == len(ev)

# 미매칭 사유 구분
hist_items = set(hist["itemid"])
first_eff = hist.groupby("itemid")["effective_timestamp"].min()
no_cat = joined["categoryid"].isna()
item_absent = ~joined["itemid"].isin(hist_items)
before_first = no_cat & ~item_absent & (joined["timestamp"] < joined["itemid"].map(first_eff))

log = pd.DataFrame([
    ("전체 이벤트 행", len(joined)),
    ("카테고리 매칭", int((~no_cat).sum())),
    ("미매칭: 카테고리 이력에 없는 상품", int(item_absent.sum())),
    ("미매칭: 해당 상품의 첫 카테고리 기록 이전 이벤트", int(before_first.sum())),
    ("매칭됐지만 root_categoryid 결측 (트리에 없는 categoryid)", int((~no_cat & joined["root_categoryid"].isna()).sum())),
], columns=["item", "rows"])
log["ratio_pct"] = (log["rows"] / len(joined) * 100).round(2)
assert log.loc[1, "rows"] + log.loc[2, "rows"] + log.loc[3, "rows"] == len(joined)

OUT.parent.mkdir(parents=True, exist_ok=True)
joined.to_csv(OUT, index=False)
log.to_csv(LOG, index=False, encoding="utf-8-sig")

print(log.to_string(index=False))
print("\n[이벤트 종류별 카테고리 매칭률(%)]")
print((joined.groupby("event")["categoryid"].apply(lambda s: s.notna().mean() * 100)).round(2).to_string())
print(f"\n열: {list(joined.columns)}")
print(f"저장: {OUT.relative_to(ROOT)} ({len(joined):,}행), {LOG.relative_to(ROOT)}")
