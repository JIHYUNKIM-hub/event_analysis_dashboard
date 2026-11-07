"""카테고리별 첫 상품 조회 집계 (events_with_category.csv 기준).

정의
- 첫 상품 조회: 방문자별 가장 이른 view 이벤트 1건 (관측 기간 내 첫 조회)
- 같은 시각에 view가 여러 건이면 itemid가 작은 것을 고른다 (재실행 시 같은 결과)
- 집계 단위: categoryid. 라벨에 대분류(root_categoryid)를 함께 표기한다.
- 카테고리는 08_join_category.py에서 이벤트 시점 이전 기록만 붙인 값이며, 없으면 '카테고리 없음'
- 첫 조회가 첫 4주(UTC 2015-05-03 ~ 2015-05-30)인 방문자는 그 이전 조회가 있었을 수 있어 열을 나눠 둔다
- 표에는 전체 카테고리를, 그래프에는 상위 TOP_N개와 '카테고리 없음'을 표시한다.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "processed" / "events_with_category.csv"
FIG = ROOT / "outputs" / "figures" / "first_view_by_category.png"
TABLE = ROOT / "outputs" / "tables" / "first_view_by_category.csv"

FIRST_4WEEKS_END = pd.Timestamp("2015-05-31", tz="UTC")
TOP_N = 20
NO_CAT = "카테고리 없음"

ev = pd.read_csv(EVENTS, usecols=["event_time", "timestamp", "visitorid", "event", "itemid",
                                  "categoryid", "root_categoryid"],
                 dtype={"categoryid": "Int64", "root_categoryid": "Int64"}, parse_dates=["event_time"])
views = ev[ev["event"] == "view"]
first = (views.sort_values(["visitorid", "timestamp", "itemid"])
         .drop_duplicates("visitorid", keep="first"))
first = first.assign(in_first_4weeks=first["event_time"] < FIRST_4WEEKS_END)

table = (first.groupby(["categoryid", "root_categoryid"], dropna=False)
         .agg(first_view_visitors=("visitorid", "size"), first_4weeks=("in_first_4weeks", "sum"))
         .reset_index()
         .sort_values("first_view_visitors", ascending=False, ignore_index=True))
table["after_first_4weeks"] = table["first_view_visitors"] - table["first_4weeks"]
table["share_pct"] = (table["first_view_visitors"] / len(first) * 100).round(3)
table["label"] = [NO_CAT if pd.isna(c) else f"카테고리 {c} (대분류 {r})"
                  for c, r in zip(table["categoryid"], table["root_categoryid"])]
assert table["categoryid"].dropna().is_unique  # categoryid 하나에 대분류는 하나

TABLE.parent.mkdir(parents=True, exist_ok=True)
table.to_csv(TABLE, index=False, encoding="utf-8-sig")
matched = table[table["categoryid"].notna()]
print(f"첫 조회 방문자: {len(first):,} / 첫 조회에 나타난 카테고리: {len(matched):,}개")
print(f"상위 {TOP_N}개 카테고리 비중: {matched.head(TOP_N)['share_pct'].sum():.1f}%")
print(table.head(TOP_N + 1).to_string(index=False))

# 그래프: 상위 TOP_N개 크기순. '카테고리 없음'은 막대 하나가 다른 막대를 눌러 보이지 않게 하므로 각주로 표기
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
SURFACE, BAR, INK, INK2, GRID = "#fcfcfb", "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"

plot = matched.head(TOP_N)
no_cat = table[table["categoryid"].isna()]
no_cat_n = int(no_cat["first_view_visitors"].sum())
no_cat_pct = float(no_cat["share_pct"].sum())
n = len(plot)
fig, ax = plt.subplots(figsize=(9.5, 0.36 * n + 2.1), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
y = list(range(n))[::-1]
ax.barh(y, plot["first_view_visitors"], height=0.68, color=BAR)
xmax = plot["first_view_visitors"].max()
for yi, v, s in zip(y, plot["first_view_visitors"], plot["share_pct"]):
    ax.text(v + xmax * 0.01, yi, f"{v:,}명 ({s:.1f}%)", va="center", fontsize=8.5, color=INK)
ax.set_yticks(y, plot["label"], fontsize=9, color=INK)
ax.set_xlim(0, xmax * 1.22)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.tick_params(axis="x", labelsize=8.5, colors=INK2, length=0)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for s in ["top", "right", "bottom"]:
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color("#d0cfca")
ax.set_title(f"카테고리별 첫 상품 조회 방문자 수 · 상위 {TOP_N}개 (전체 {len(matched):,}개 카테고리, {len(first):,}명)",
             loc="left", fontsize=12.5, color=INK, pad=12)
fig.text(0.01, 0.035, f"그래프 제외: {NO_CAT} {no_cat_n:,}명 ({no_cat_pct:.1f}%) — 조회 시점 이전 카테고리 기록이 없는 방문자",
         fontsize=8, color=INK2)
fig.text(0.01, 0.01, "첫 상품 조회 = 방문자별 관측 기간 내 첫 view · 괄호는 전체 첫 조회 방문자 대비 비율 · "
         "UTC, ~2015-09-15 21:00 · 전체 카테고리는 outputs/tables/first_view_by_category.csv",
         fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.05, 1, 1))
FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG, dpi=150)
print(f"\n저장: {FIG.relative_to(ROOT)}, {TABLE.relative_to(ROOT)}")
