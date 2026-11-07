"""N-day 리텐션 곡선 (UTC 날짜 기준).

정의
- Day 0: 방문자의 첫 관측 UTC 날짜 (이벤트 종류 무관, view·addtocart·transaction 모두 활동으로 본다)
- Day N 재방문: 첫 관측일로부터 정확히 N일째 UTC 날짜에 이벤트가 1건 이상 있는 경우.
  그 날짜에 돌아온 사용자만 센다 (N일 이후 아무 때나 돌아온 사용자를 누적하지 않음, unbounded 아님).
- N-day 리텐션 = Day N 재방문자 수 / Day N을 관측할 수 있는 코호트 방문자 수
- 우측 절단: 2015-09-15는 21:00까지만 관측되므로 마지막 완전 관측일은 2015-09-14다.
  첫 관측일 + N <= 2015-09-14 인 방문자만 Day N 분모에 넣는다 (N마다 분모가 다르다).
- 좌측 절단: 첫 4주(2015-05-03 ~ 2015-05-30)에 첫 관측된 방문자는 그 이전 이력이 없어 신규로 단정할 수 없다.
  그래서 첫 관측일이 2015-05-31 이후인 코호트(주 곡선)와 첫 4주 코호트를 나눠 계산한다.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "processed" / "events_clean.csv"
FIG = ROOT / "outputs" / "figures" / "retention_curve_nday.png"
TABLE = ROOT / "outputs" / "tables" / "retention_nday.csv"

DAY_MS = 86_400_000
LAST_FULL_DAY = pd.Timestamp("2015-09-14").value // 10**6 // DAY_MS  # 완전히 관측된 마지막 UTC 날짜
EARLY_END = pd.Timestamp("2015-05-30").value // 10**6 // DAY_MS      # 첫 4주 마지막 날짜

ev = pd.read_csv(EVENTS, usecols=["timestamp", "visitorid"])
ev["day"] = ev["timestamp"] // DAY_MS  # UTC 날짜 번호 (epoch 일수)
active = ev[["visitorid", "day"]].drop_duplicates()

first = active.groupby("visitorid")["day"].min().rename("day0")
active = active.join(first, on="visitorid")
active["n"] = active["day"] - active["day0"]
users = first.to_frame()
users["cohort"] = np.where(users["day0"] <= EARLY_END, "early_4w", "after_0531")
active["cohort"] = active["visitorid"].map(users["cohort"])

rows = []
for cohort, label in [("after_0531", "2015-05-31 이후 첫 관측"), ("early_4w", "첫 4주(05-03~05-30) 첫 관측"),
                      ("all", "전체")]:
    u = users if cohort == "all" else users[users["cohort"] == cohort]
    a = active if cohort == "all" else active[active["cohort"] == cohort]
    returned = a[a["day"] <= LAST_FULL_DAY].groupby("n")["visitorid"].nunique()
    max_n = int(LAST_FULL_DAY - u["day0"].min())
    # Day N을 관측할 수 있는 방문자 수: day0 <= LAST_FULL_DAY - N
    day0_counts = u["day0"].value_counts().sort_index()
    for n in range(0, max_n + 1):
        eligible = int(day0_counts[day0_counts.index <= LAST_FULL_DAY - n].sum())
        ret = int(returned.get(n, 0))
        rows.append({"cohort": cohort, "cohort_label": label, "n_day": n, "eligible_users": eligible,
                     "returned_users": ret, "retention_pct": ret / eligible * 100})

ret_df = pd.DataFrame(rows)
TABLE.parent.mkdir(parents=True, exist_ok=True)
ret_df.to_csv(TABLE, index=False, encoding="utf-8-sig")

print(f"방문자 수: 전체 {len(users):,} / 05-31 이후 첫 관측 {(users['cohort'] == 'after_0531').sum():,} "
      f"/ 첫 4주 첫 관측 {(users['cohort'] == 'early_4w').sum():,}")
show = ret_df[ret_df["n_day"].isin([1, 2, 3, 7, 14, 30, 60, 90])]
print(show.pivot(index="n_day", columns="cohort", values="retention_pct").round(3).to_string())

# 그래프: x = N일, y = 재방문율(%). Day 0(100%)은 축 범위를 눌러서 제외한다.
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
SURFACE, MAIN, SUB, INK, INK2, GRID = "#fcfcfb", "#2a78d6", "#e07b39", "#0b0b0b", "#52514e", "#e4e3df"

fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
for cohort, color, ls, dy in [("after_0531", MAIN, "-", -12), ("early_4w", SUB, "--", 4)]:
    d = ret_df[(ret_df["cohort"] == cohort) & (ret_df["n_day"] >= 1)]
    n_users = d["eligible_users"].iloc[0]
    ax.plot(d["n_day"], d["retention_pct"], color=color, ls=ls, lw=1.6,
            label=f"{d['cohort_label'].iloc[0]} (Day 1 분모 {n_users:,}명)")
    for n in [1, 7, 30]:
        v = d.loc[d["n_day"] == n, "retention_pct"].iloc[0]
        ax.scatter(n, v, color=color, s=18, zorder=3)
        ax.annotate(f"D{n} {v:.2f}%", (n, v), xytext=(6, dy), textcoords="offset points", fontsize=8.5, color=color)

ax.set_xlabel("N일 (첫 관측일 = Day 0, UTC 날짜)", fontsize=10, color=INK)
ax.set_ylabel("재방문율 (%)", fontsize=10, color=INK)
ax.set_xlim(0, ret_df["n_day"].max() + 1)
ax.set_ylim(0, None)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}%"))
ax.tick_params(labelsize=9, colors=INK2, length=0)
ax.grid(color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
for s in ["left", "bottom"]:
    ax.spines[s].set_color("#d0cfca")
ax.legend(frameon=False, fontsize=9.5, loc="upper right")
ax.set_title("N-day 리텐션 곡선 (해당 N일째 날짜에 재방문한 사용자 비율)", loc="left", fontsize=13, color=INK, pad=10)
fig.text(0.01, 0.01, "분모: Day N이 완전 관측일(~2015-09-14 UTC) 안에 드는 방문자 · "
         "첫 4주 코호트는 이전 이력이 관측되지 않아(좌측 절단) 신규 방문자로 단정할 수 없음",
         fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1))
FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG, dpi=150)
print(f"\n저장: {FIG.relative_to(ROOT)}, {TABLE.relative_to(ROOT)}")
