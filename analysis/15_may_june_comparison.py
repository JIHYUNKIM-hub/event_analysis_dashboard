"""5월 vs 6월 비교 (6월 마케팅 전략 기획서용, UTC 월 기준).

산출 표 (outputs/tables/)
- may_june_key_metrics.csv: 주요 지표 (06_monthly_key_metrics.py와 같은 정의) + 일평균
- may_june_weekly_active.csv: 주별(월요일 시작) 활성 사용자 수, 2015-05-04 ~ 2015-06-28 주
- may_june_funnel.csv: 퍼널 두 가지 정의
    · 월 내 고유 방문자: 해당 월에 view / addtocart / transaction을 1회 이상 한 고유 방문자 (순서 무관)
    · 24시간 퍼널: funnel_utils.first_view_funnel 정의, 첫 조회 날짜가 해당 월인 방문자
- may_june_root.csv: 대분류별 월 내 고유 방문자 전환율 (이벤트 상품의 대분류 기준, 6월 조회자 MIN_VIEW명 이상)
- may_june_items.csv: 상품별 월별 조회·장바구니·구매 방문자 수와 구매 순위
- may_june_overlap.csv: 5월·6월 방문자·구매자 중복

주의
- 5월은 2015-05-03 03:00부터 관측된 부분 월(29일)이고 그 이전 행동은 보이지 않는다(좌측 절단).
  24시간 퍼널의 5월 '첫 조회'에는 5/3 이전에 방문했던 사람이 섞여 있다.
- 카테고리 매핑이 없는 이벤트 비율이 5월에 훨씬 높아(매핑 공백) 대분류별 방문자 수 증감은 비교하지 않고 전환율만 본다.
- 구매(상품 단위)는 transaction 이벤트 행 기준이다. 한 거래에 여러 상품이 있으면 상품마다 센다.
"""
from pathlib import Path

import pandas as pd

from funnel_utils import first_view_funnel, funnel_counts

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "processed" / "events_with_category.csv"
TABLES = ROOT / "outputs" / "tables"
MONTHS = [5, 6]
MIN_VIEW = 5000
TOP_N = 20

ev = pd.read_csv(EVENTS, usecols=["timestamp", "event_time", "month", "visitorid", "event", "itemid",
                                  "transactionid", "categoryid", "root_categoryid"],
                 dtype={"transactionid": "Int64", "categoryid": "Int64", "root_categoryid": "Int64"},
                 parse_dates=["event_time"])
mm = ev[ev["month"].isin(MONTHS)]


def uniq(df: pd.DataFrame, event: str, keys: list[str]) -> pd.Series:
    return df[df["event"] == event].groupby(keys)["visitorid"].nunique()


# 주요 지표
g = mm.groupby("month")
key = pd.DataFrame({
    "active_users": g["visitorid"].nunique(),
    "buyers": uniq(mm, "transaction", ["month"]),
    "purchases": g["transactionid"].nunique(),
    "addtocart_count": mm[mm["event"] == "addtocart"].groupby("month").size(),
    "event_count": g.size(),
    "observed_days": g["event_time"].agg(lambda s: s.dt.date.nunique()),
})
for col in ["active_users", "buyers", "purchases", "addtocart_count", "event_count"]:
    key[f"{col}_per_day"] = key[col] / key["observed_days"]
key["buyer_rate_pct"] = key["buyers"] / key["active_users"] * 100
key["purchases_per_buyer"] = key["purchases"] / key["buyers"]
key.to_csv(TABLES / "may_june_key_metrics.csv", encoding="utf-8-sig")

# 주별 활성 사용자 (월요일 시작, 두 달에 걸친 주는 포함하고 5/3(일) 하루짜리 주는 뺀다)
d = mm["event_time"].dt.tz_localize(None).dt.normalize()
week = d - pd.to_timedelta(d.dt.day_of_week, unit="D")
weekly = mm.groupby(week)["visitorid"].nunique().rename("active_users").rename_axis("week_start")
weekly = weekly[(weekly.index >= "2015-05-04") & (weekly.index <= "2015-06-22")]
weekly.to_csv(TABLES / "may_june_weekly_active.csv", encoding="utf-8-sig")

# 퍼널
rows = []
for m in MONTHS:
    e = mm[mm["month"] == m]
    v, c, t = (e.loc[e["event"] == x, "visitorid"].nunique() for x in ["view", "addtocart", "transaction"])
    rows.append({"month": m, "definition": "월 내 고유 방문자", "view": v, "addtocart": c, "transaction": t,
                 "view_to_cart_pct": c / v * 100, "cart_to_tx_pct": t / c * 100, "view_to_tx_pct": t / v * 100})
fv = first_view_funnel(ev[["timestamp", "visitorid", "event", "itemid", "categoryid", "root_categoryid"]])
fv_month = pd.to_datetime(fv["t0"], unit="ms").dt.month
for m in MONTHS:
    rows.append({"month": m, "definition": "24시간 퍼널(첫 조회 월)", **funnel_counts(fv[fv_month == m])})
pd.DataFrame(rows).to_csv(TABLES / "may_june_funnel.csv", index=False, encoding="utf-8-sig")

# 대분류별 전환율
keys = ["root_categoryid", "month"]
root = pd.DataFrame({x: uniq(mm, x, keys) for x in ["view", "addtocart", "transaction"]}).fillna(0)
root = root.unstack("month")
root.columns = [f"{a}_{b}" for a, b in root.columns]
for m in MONTHS:
    root[f"view_to_cart_pct_{m}"] = root[f"addtocart_{m}"] / root[f"view_{m}"] * 100
    root[f"cart_to_tx_pct_{m}"] = root[f"transaction_{m}"] / root[f"addtocart_{m}"] * 100
no_cat_share = {m: mm[(mm["month"] == m) & (mm["event"] == "view")]["root_categoryid"].isna().mean() * 100
                for m in MONTHS}
print("카테고리 없는 조회 이벤트 비율(%):", {m: round(s, 1) for m, s in no_cat_share.items()})
root = root[root["view_6"] >= MIN_VIEW].sort_values("view_6", ascending=False)
root.reset_index().to_csv(TABLES / "may_june_root.csv", index=False, encoding="utf-8-sig")

# 상품별
keys = ["itemid", "month"]
items = pd.DataFrame({x: uniq(mm, x, keys) for x in ["view", "addtocart", "transaction"]}).fillna(0).astype(int)
items = items.unstack("month", fill_value=0)
items.columns = [f"{a}_{b}" for a, b in items.columns]
for m in MONTHS:
    items[f"tx_rank_{m}"] = items[f"transaction_{m}"].rank(method="min", ascending=False).astype(int)
    items[f"view_to_cart_pct_{m}"] = items[f"addtocart_{m}"] / items[f"view_{m}"].where(items[f"view_{m}"] > 0) * 100
items = items.join(mm.drop_duplicates("itemid", keep="last").set_index("itemid")["root_categoryid"])
items.reset_index().to_csv(TABLES / "may_june_items.csv", index=False, encoding="utf-8-sig")

# 방문자·구매자 중복
ids = {m: set(mm.loc[mm["month"] == m, "visitorid"]) for m in MONTHS}
buy = {m: set(mm.loc[(mm["month"] == m) & (mm["event"] == "transaction"), "visitorid"]) for m in MONTHS}
overlap = pd.DataFrame([
    {"metric": "6월 활성 사용자", "value": len(ids[6])},
    {"metric": "6월 활성 사용자 중 5월에도 활성", "value": len(ids[6] & ids[5])},
    {"metric": "5월 활성 사용자 중 6월에도 활성", "value": len(ids[5] & ids[6])},
    {"metric": "6월 구매자", "value": len(buy[6])},
    {"metric": "6월 구매자 중 5월에도 활성", "value": len(buy[6] & ids[5])},
    {"metric": "5월 구매자", "value": len(buy[5])},
    {"metric": "5월 구매자 중 6월에도 활성", "value": len(buy[5] & ids[6])},
    {"metric": "5월 구매자 중 6월에도 구매", "value": len(buy[5] & buy[6])},
])
overlap.to_csv(TABLES / "may_june_overlap.csv", index=False, encoding="utf-8-sig")

# 요약 출력
pd.set_option("display.width", 200)
print(key.round(3).T, weekly, pd.DataFrame(rows).round(2), sep="\n\n")
print(root.round(1).to_string())
print(overlap.to_string())
for m in MONTHS:
    tx = items[f"transaction_{m}"]
    top = tx.nlargest(TOP_N)
    print(f"{m}월 구매 상품 수 {int((tx > 0).sum())}, 상위 {TOP_N}개 상품 구매 비중 {top.sum() / tx.sum() * 100:.1f}%")
top5, top6 = set(items["transaction_5"].nlargest(TOP_N).index), set(items["transaction_6"].nlargest(TOP_N).index)
print(f"상위 {TOP_N} 중 두 달 모두 포함: {len(top5 & top6)}")
cols = ["view_5", "view_6", "addtocart_5", "addtocart_6", "transaction_5", "transaction_6", "tx_rank_5", "tx_rank_6", "root_categoryid"]
print(items.sort_values("transaction_6", ascending=False)[cols].head(10).to_string())
low = items[items["view_6"] >= 1000].sort_values("view_to_cart_pct_6")
print(low[cols + ["view_to_cart_pct_6"]].head(10).to_string())
print("6월 조회 상위 10 상품:")
print(items.sort_values("view_6", ascending=False)[cols + ["view_to_cart_pct_6"]].head(10).to_string())
