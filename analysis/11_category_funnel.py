"""대분류·카테고리별 24시간 퍼널 비교 (funnel_utils.first_view_funnel 정의, 진입 카테고리 기준).

산출 표
- funnel_24h_overall.csv: 전체 퍼널
- funnel_24h_by_root.csv: 대분류별 퍼널 ('카테고리 없음' 포함)
- funnel_24h_by_category.csv: 카테고리별 퍼널 (전체)
- funnel_24h_time_distribution.csv: 단계 간 소요 시간 분포
- funnel_24h_prev_stage_counts.csv: 단계 도달 전 이전 단계 수행 횟수 분포
비교 시에는 표본이 작은 항목의 우연한 변동을 줄이기 위해 첫 조회 방문자 MIN_VIEW명 이상만 본다.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from funnel_utils import (count_distribution, first_view_funnel, funnel_counts, group_funnel,
                          time_distribution)

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "processed" / "events_with_category.csv"
TABLES = ROOT / "outputs" / "tables"
MIN_VIEW = 1000

ev = pd.read_csv(EVENTS, usecols=["timestamp", "visitorid", "event", "itemid", "categoryid", "root_categoryid"],
                 dtype={"categoryid": "Int64", "root_categoryid": "Int64"})
fv = first_view_funnel(ev)

overall = pd.DataFrame([funnel_counts(fv)])
by_root = group_funnel(fv, ["root_categoryid"])
by_cat = group_funnel(fv, ["categoryid", "root_categoryid"])
time_dist = pd.DataFrame({"view_to_cart": time_distribution(fv["cart_minutes"]),
                          "cart_to_tx": time_distribution(fv["tx_minutes"])})
prev_counts = pd.DataFrame({"views_before_cart": count_distribution(fv["n_views_before_cart"]),
                            "carts_before_tx": count_distribution(fv["n_carts_before_tx"])})

TABLES.mkdir(parents=True, exist_ok=True)
for name, d in [("overall", overall), ("by_root", by_root), ("by_category", by_cat)]:
    d.round(3).to_csv(TABLES / f"funnel_24h_{name}.csv", index=False, encoding="utf-8-sig")
time_dist.rename_axis("bucket").to_csv(TABLES / "funnel_24h_time_distribution.csv", encoding="utf-8-sig")
prev_counts.rename_axis("bucket").to_csv(TABLES / "funnel_24h_prev_stage_counts.csv", encoding="utf-8-sig")

pd.set_option("display.width", 200)
cols = ["view", "addtocart", "transaction", "view_to_cart_pct", "cart_to_tx_pct", "view_to_tx_pct",
        "cart_minutes_median", "tx_minutes_median", "views_before_cart_median", "carts_before_tx_median"]
print(f"코호트: 첫 조회 ≤ 2015-09-14 21:00 UTC 방문자 {len(fv):,}명")
print(overall.round(2).to_string(index=False))
print(f"\n중앙값 - View→AddToCart {fv['cart_minutes'].median():.1f}분, AddToCart→Transaction {fv['tx_minutes'].median():.1f}분")
print(f"중앙값 - 장바구니 전 view {fv['n_views_before_cart'].median():.0f}회, 구매 전 장바구니 {fv['n_carts_before_tx'].median():.0f}회")
print("\n[대분류별]")
print(by_root[by_root["view"] >= MIN_VIEW][["root_categoryid", *cols]].round(2).to_string(index=False))
big_cat = by_cat[(by_cat["view"] >= MIN_VIEW) & by_cat["categoryid"].notna()]
print(f"\n[카테고리별, 첫 조회 {MIN_VIEW}명 이상 {len(big_cat)}개] View→Transaction 상위·하위 10")
show = ["categoryid", "root_categoryid", "view", "addtocart", "transaction", "view_to_cart_pct", "cart_to_tx_pct", "view_to_tx_pct"]
print(big_cat.sort_values("view_to_tx_pct", ascending=False)[show].head(10).round(2).to_string(index=False))
print(big_cat.sort_values("view_to_tx_pct")[show].head(10).round(2).to_string(index=False))
print(f"\n카테고리 간 View→Transaction 분포(%): {big_cat['view_to_tx_pct'].describe().round(2).to_dict()}")
print(f"구매 0명 카테고리(1000명 이상 중): {(big_cat['transaction'] == 0).sum()}")
print("\n[시간 분포]\n", time_dist.to_string(), sep="")
print("\n[이전 단계 수행 횟수]\n", prev_counts.to_string(), sep="")
corr = big_cat[["view_to_cart_pct", "cart_to_tx_pct"]].corr().iat[0, 1]
print(f"\n카테고리 수준 View→Cart vs Cart→Tx 상관: {corr:.2f}")
