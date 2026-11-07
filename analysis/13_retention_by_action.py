"""행동 기준 코호트별 N-day 리텐션 비교 (UTC 날짜 기준, 12_retention_curve.py와 같은 N-day 정의).

코호트 (방문자 단위, 겹칠 수 있다)
- 구매: Day 0 = 첫 transaction UTC 날짜
- 장바구니: Day 0 = 첫 addtocart UTC 날짜
- 장바구니+구매: addtocart와 transaction을 모두 한 방문자. Day 0 = 두 행동을 모두 마친 날짜
  = max(첫 addtocart 날짜, 첫 transaction 날짜)
- (참고) 전체: Day 0 = 첫 관측 날짜 (12번과 같음)
Day 0은 행동이 일어난 날로 잡아 코호트 소속이 Day 0 이후 정보에 의존하지 않게 한다.

재방문: Day 0으로부터 정확히 N일째 UTC 날짜에 이벤트(view·addtocart·transaction 무관)가 1건 이상 있는 경우.
분모: Day 0 + N <= 2015-09-14(마지막 완전 관측일)인 방문자.
좌측 절단: Day 0이 첫 4주(2015-05-03 ~ 2015-05-30)인 경우 그 이전에 같은 행동을 했을 수 있어 '첫 행동'으로 단정할 수 없다.
그래프는 Day 0 >= 2015-05-31 코호트만 그리고, 표에는 첫 4주 코호트를 따로 남긴다.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "processed" / "events_clean.csv"
FIG = ROOT / "outputs" / "figures" / "retention_by_action.png"
TABLE = ROOT / "outputs" / "tables" / "retention_by_action.csv"

DAY_MS = 86_400_000
LAST_FULL_DAY = pd.Timestamp("2015-09-14").value // 10**6 // DAY_MS
EARLY_END = pd.Timestamp("2015-05-30").value // 10**6 // DAY_MS

ev = pd.read_csv(EVENTS, usecols=["timestamp", "visitorid", "event"])
ev["day"] = ev["timestamp"] // DAY_MS
active = ev[["visitorid", "day"]].drop_duplicates()
active = active[active["day"] <= LAST_FULL_DAY]

first_any = ev.groupby("visitorid")["day"].min()
first_cart = ev[ev["event"] == "addtocart"].groupby("visitorid")["day"].min()
first_tx = ev[ev["event"] == "transaction"].groupby("visitorid")["day"].min()
both = pd.concat([first_cart, first_tx], axis=1, join="inner").max(axis=1)

COHORTS = [
    ("transaction", "구매", first_tx),
    ("addtocart", "장바구니", first_cart),
    ("cart_and_tx", "장바구니+구매", both),
    ("all", "전체 (참고)", first_any),
]


def nday_retention(day0: pd.Series) -> pd.DataFrame:
    a = active.join(day0.rename("day0"), on="visitorid", how="inner")
    a["n"] = a["day"] - a["day0"]
    returned = a[a["n"] >= 1].groupby("n")["visitorid"].nunique()
    day0_counts = day0.value_counts().sort_index()
    rows = []
    for n in range(1, int(LAST_FULL_DAY - day0.min()) + 1):
        eligible = int(day0_counts[day0_counts.index <= LAST_FULL_DAY - n].sum())
        ret = int(returned.get(n, 0))
        rows.append({"n_day": n, "eligible_users": eligible, "returned_users": ret,
                     "retention_pct": ret / eligible * 100})
    return pd.DataFrame(rows)


parts = []
for key, label, day0 in COHORTS:
    for period, mask in [("after_0531", day0 > EARLY_END), ("early_4w", day0 <= EARLY_END)]:
        r = nday_retention(day0[mask])
        r.insert(0, "day0_period", period)
        r.insert(0, "cohort_label", label)
        r.insert(0, "cohort", key)
        parts.append(r)
ret_df = pd.concat(parts, ignore_index=True)
TABLE.parent.mkdir(parents=True, exist_ok=True)
ret_df.to_csv(TABLE, index=False, encoding="utf-8-sig")

for key, label, day0 in COHORTS:
    print(f"{label}: 전체 {len(day0):,}명 / Day 0 05-31 이후 {(day0 > EARLY_END).sum():,}명")
main = ret_df[ret_df["day0_period"] == "after_0531"]
print("\n[Day 0 >= 2015-05-31 코호트 재방문율 %]")
print(main[main["n_day"].isin([1, 2, 3, 7, 14, 30, 60, 90])]
      .pivot(index="n_day", columns="cohort_label", values="retention_pct").round(2).to_string())
print("\n[Day 1 분모]")
print(main[main["n_day"] == 1].set_index("cohort_label")[["eligible_users", "returned_users"]].to_string())

# 그래프: Day 0 >= 05-31 코호트. 앞쪽 변화가 커서 전체 구간 + 0~30일 확대 두 패널
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
STYLE = {"transaction": ("#2a78d6", "-"), "addtocart": ("#e07b39", "-"),
         "cart_and_tx": ("#1f9e6e", "--"), "all": ("#9a9893", ":")}

fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.6, 1]})
for ax, xmax, title in [(axes[0], None, "전체 구간"), (axes[1], 30, "0~30일 확대")]:
    ax.set_facecolor(SURFACE)
    for key, label, _ in COHORTS:
        d = main[main["cohort"] == key]
        if xmax:
            d = d[d["n_day"] <= xmax]
        color, ls = STYLE[key]
        n1 = main.loc[(main["cohort"] == key) & (main["n_day"] == 1), "eligible_users"].iloc[0]
        ax.plot(d["n_day"], d["retention_pct"], color=color, ls=ls, lw=1.5,
                label=f"{label} (Day 1 분모 {n1:,}명)")
        if xmax:
            for n in [1, 7, 30]:
                v = d.loc[d["n_day"] == n, "retention_pct"].iloc[0]
                ax.scatter(n, v, color=color, s=16, zorder=3)
                if key != "all":
                    dy = 3 if key == "cart_and_tx" else -11
                    ax.annotate(f"{v:.1f}%", (n, v), xytext=(5, dy), textcoords="offset points",
                                fontsize=8, color=color)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel("N일 (행동일 = Day 0, UTC 날짜)", fontsize=9.5, color=INK)
    ax.set_ylim(0, None)
    ax.set_xlim(0, xmax + 3 if xmax else main["n_day"].max() + 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.tick_params(labelsize=9, colors=INK2, length=0)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color("#d0cfca")
axes[0].set_ylabel("재방문율 (%, 전체 이벤트 기준)", fontsize=9.5, color=INK)
axes[0].legend(frameon=False, fontsize=9, loc="upper right")

fig.suptitle("행동별 N-day 리텐션 비교 (Day 0 = 해당 행동을 한 날, Day 0 ≥ 2015-05-31)", x=0.01, ha="left",
             fontsize=13, color=INK)
fig.text(0.01, 0.01, "코호트는 겹친다(구매자 대부분이 장바구니 코호트에도 포함) · 장바구니+구매의 Day 0 = 두 행동을 모두 마친 날 · "
         "분모: Day N이 2015-09-14 UTC 이내인 방문자 · 구매 코호트는 인원이 적어 뒤쪽 구간 변동이 큼",
         fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 0.96))
FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG, dpi=150)
print(f"\n저장: {FIG.relative_to(ROOT)}, {TABLE.relative_to(ROOT)}")
