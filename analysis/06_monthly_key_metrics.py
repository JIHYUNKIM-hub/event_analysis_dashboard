"""주요 지표 월별 추이 (05_key_metrics.py와 같은 지표 정의를 월 단위로 적용, UTC 월 기준).

- 활성 사용자 수: 해당 월에 이벤트를 1회 이상 수행한 고유 방문자
- 구매자 수: 해당 월에 transaction을 1회 이상 수행한 고유 방문자
- 구매 횟수: 해당 월의 고유 transactionid 수
- 장바구니 담기 횟수: 해당 월 addtocart 행 수
- 이벤트 횟수: 해당 월 전체 이벤트 행 수
5월(2015-05-03 03:00~)과 9월(~2015-09-15 21:00)은 관측 기간이 일부뿐인 부분 월이다.
고유 방문자 수는 월마다 따로 세므로 월별 값을 더해도 전체 기간 값과 같지 않다.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "processed" / "events_clean.csv"
FIG = ROOT / "outputs" / "figures" / "monthly_key_metrics.png"
TABLE = ROOT / "outputs" / "tables" / "monthly_key_metrics.csv"

ev = pd.read_csv(EVENTS, usecols=["event_time", "month", "visitorid", "event", "transactionid"],
                 dtype={"transactionid": "Int64"}, parse_dates=["event_time"])

g = ev.groupby("month")
is_tx = ev["event"] == "transaction"
monthly = pd.DataFrame({
    "active_users": g["visitorid"].nunique(),
    "buyers": ev[is_tx].groupby("month")["visitorid"].nunique(),
    "purchases": g["transactionid"].nunique(),
    "addtocart_count": ev[ev["event"] == "addtocart"].groupby("month").size(),
    "event_count": g.size(),
    "observed_days": g["event_time"].agg(lambda s: s.dt.date.nunique()),
    "first_event": g["event_time"].min(),
    "last_event": g["event_time"].max(),
})
monthly["partial_month"] = monthly.index.isin([5, 9])

# 한 거래가 두 달에 걸치면 월별 구매 횟수 합계가 전체보다 커진다
assert monthly["purchases"].sum() == ev["transactionid"].nunique()

TABLE.parent.mkdir(parents=True, exist_ok=True)
monthly.reset_index().to_csv(TABLE, index=False, encoding="utf-8-sig")
print(monthly.drop(columns=["first_event", "last_event"]).to_string())

# 그래프: 지표마다 단위가 달라 패널을 나눈다 (한 패널에 한 축)
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
SURFACE, BAR, INK, INK2, GRID = "#fcfcfb", "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df"

metrics = [
    ("active_users", "활성 사용자 수", "명"),
    ("buyers", "구매자 수", "명"),
    ("purchases", "구매 횟수", "회"),
    ("addtocart_count", "장바구니 담기 횟수", "회"),
    ("event_count", "이벤트 횟수", "건"),
]
labels = [f"{m}월" + ("\n(5/3~)" if m == 5 else "\n(~9/15)" if m == 9 else "\n") for m in monthly.index]
x = range(len(monthly))

fig, axes = plt.subplots(len(metrics), 1, figsize=(8, 13), facecolor=SURFACE, sharex=True)
for ax, (col, title, unit) in zip(axes, metrics):
    ax.set_facecolor(SURFACE)
    vals = monthly[col]
    for xi, v, partial in zip(x, vals, monthly["partial_month"]):
        ax.bar(xi, v, width=0.6, color=SURFACE if partial else BAR, edgecolor=BAR,
               hatch="///" if partial else None, linewidth=1.2)
        ax.text(xi, v * 1.02, f"{v:,}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_title(f"{title} ({unit})", loc="left", fontsize=11.5, color=INK, pad=6)
    ax.set_ylim(0, vals.max() * 1.18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.tick_params(axis="y", labelsize=8.5, colors=INK2, length=0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d0cfca")
axes[-1].set_xticks(list(x), labels, fontsize=10, color=INK)

fig.suptitle("주요 지표 월별 추이 (UTC, 2015-05-03 ~ 2015-09-15 21:00)", x=0.02, ha="left",
             fontsize=13.5, color=INK)
fig.text(0.02, 0.008, "빗금 막대: 관측 기간이 일부인 부분 월(5월 29일, 9월 15일) · 월 고유 방문자는 월별로 따로 집계",
         fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.02, 1, 0.98))
FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG, dpi=150)
print(f"\n저장: {FIG.relative_to(ROOT)}, {TABLE.relative_to(ROOT)}")
