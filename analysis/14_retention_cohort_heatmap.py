"""일별 방문 코호트 리텐션 히트맵 (UTC 날짜 기준, 12_retention_curve.py와 같은 N-day 정의).

- 행(코호트): 방문자의 첫 관측 UTC 날짜 (Day 0). 이벤트 종류 무관.
- 열: N일 (Day 0 ~ Day 30). 각 칸 = Day 0으로부터 정확히 N일째 날짜에 이벤트가 1건 이상 있는 방문자 수와 비율.
  Day 0 칸은 코호트 크기(100%)다.
- 관측 범위: 2015-09-15는 21:00까지만 관측되므로 마지막 완전 관측일은 2015-09-14다.
  Day 0 + N > 2015-09-14 인 칸은 비워 둔다. 9/15 코호트는 Day 0 자체가 부분 관측이라 행에서 뺀다.
- 좌측 절단: 첫 4주(2015-05-03 ~ 2015-05-30) 코호트는 그 이전 이력이 관측되지 않아 신규 방문자로 단정할 수 없다.
  행 이름에 *를 붙여 구분하고, 맨 위 요약 행은 05-31 이후 코호트만 합산한다.
- 표(CSV)에는 그림에 다 넣지 못한 Day 31 이후까지 모든 관측 가능한 칸을 남긴다.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retention_utils import DAY_MS, EARLY_END, LAST_FULL_DAY, active_days, return_offsets  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "processed" / "events_clean.csv"
FIG = ROOT / "outputs" / "figures" / "retention_cohort_heatmap.png"
TABLE = ROOT / "outputs" / "tables" / "retention_cohort_daily.csv"
N_SHOW = 30

ev = pd.read_csv(EVENTS, usecols=["timestamp", "visitorid"])
cohort = (ev.groupby("visitorid")["timestamp"].min() // DAY_MS).rename("day0").to_frame()
cohort = cohort[cohort["day0"] <= LAST_FULL_DAY]
back = return_offsets(cohort, active_days(ev)).join(cohort["day0"], on="visitorid")

size = cohort.groupby("day0").size()
returned = back.groupby(["day0", "n"]).size().unstack(fill_value=0)
n_cols = np.arange(0, int(LAST_FULL_DAY - size.index.min()) + 1)
counts = returned.reindex(index=size.index, columns=n_cols, fill_value=0)
counts[0] = size
observable = (size.index.to_numpy()[:, None] + n_cols[None, :]) <= LAST_FULL_DAY
counts = counts.where(observable)
rates = counts.div(size, axis=0) * 100

dates = pd.to_datetime(size.index * DAY_MS, unit="ms")
early = size.index.to_numpy() <= EARLY_END

# 요약 행: 05-31 이후 코호트를 칸마다 합산 (관측 가능한 코호트만 분모에 넣는 가중 평균)
late_counts = counts[~early]
late_elig = observable[~early] * size[~early].to_numpy()[:, None]
summary_rate = late_counts.sum() / late_elig.sum(axis=0) * 100

# CSV: 코호트·N 단위 긴 표
counts = counts.rename_axis(index="day0_num", columns="n_day")
rates = rates.rename_axis(index="day0_num", columns="n_day")
long = (counts.stack().dropna().rename("returned_users").to_frame()
        .join(rates.stack().dropna().rename("retention_pct")).reset_index())
long.insert(0, "cohort_date", pd.to_datetime(long["day0_num"] * DAY_MS, unit="ms").dt.date)
long.insert(1, "first_4_weeks", long["day0_num"] <= EARLY_END)
long.insert(2, "cohort_size", long["day0_num"].map(size))
long = long.drop(columns="day0_num")
long["returned_users"] = long["returned_users"].astype(int)
TABLE.parent.mkdir(parents=True, exist_ok=True)
long.to_csv(TABLE, index=False, encoding="utf-8-sig")

print(f"코호트 {len(size)}개 ({dates.min():%Y-%m-%d} ~ {dates.max():%Y-%m-%d}), 방문자 {size.sum():,}명")
print(f"05-31 이후 코호트 요약 재방문율(%): " +
      ", ".join(f"D{n} {summary_rate[n]:.2f}" for n in [1, 7, 14, 30]))

# 그림: Day 0 ~ Day N_SHOW. 색은 Day 1 이후 재방문율(%)에만 입힌다 (Day 0 = 100%는 회색)
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
SURFACE, INK, INK2, DAY0 = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CMAP = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fe", "#b7d3f6", "#5598e7", "#1c5cab"])

show_n = n_cols[: N_SHOW + 1]
r = rates[show_n].to_numpy()
c = counts[show_n].to_numpy()
row_labels = [f"{d:%m-%d}" + (" *" if e else "") for d, e in zip(dates, early)]
row_labels = [f"요약 (05-31~)"] + row_labels
r = np.vstack([np.r_[100, summary_rate[show_n[1:]].to_numpy()], r])
c = np.vstack([np.r_[size[~early].sum(), late_counts[show_n[1:]].sum().to_numpy()], c])

vals = r[:, 1:][~np.isnan(r[:, 1:])]
vmax = np.percentile(vals, 98)  # 소수 극단값이 색 범위를 차지하지 않도록 98 백분위에서 자른다

n_rows, n_c = r.shape
fig, ax = plt.subplots(figsize=(0.62 * n_c + 1.6, 0.27 * n_rows + 1.8), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
img = np.ma.masked_invalid(np.where(np.arange(n_c)[None, :] == 0, np.nan, r))
mesh = ax.pcolormesh(img, cmap=CMAP, vmin=0, vmax=vmax, edgecolors=SURFACE, linewidth=0.6)
ax.pcolormesh(np.ma.masked_where(~np.isfinite(r[:, :1]), np.zeros((n_rows, 1))),
              cmap=LinearSegmentedColormap.from_list("d0", [DAY0, DAY0]), edgecolors=SURFACE, linewidth=0.6)
for i in range(n_rows):
    for j in range(n_c):
        if np.isnan(r[i, j]):
            continue
        dark = j > 0 and r[i, j] > vmax * 0.55
        ax.text(j + 0.5, i + 0.5, f"{int(c[i, j]):,}\n{r[i, j]:.1f}%" if j else f"{int(c[i, j]):,}",
                ha="center", va="center", fontsize=4.6, color="white" if dark else INK, linespacing=1.0)
ax.axhline(1, color=INK, linewidth=1.0)
ax.invert_yaxis()
ax.set_xticks(np.arange(n_c) + 0.5, [f"D{n}" for n in show_n], fontsize=7)
ax.set_yticks(np.arange(n_rows) + 0.5, row_labels, fontsize=6.2)
ax.xaxis.tick_top()
ax.tick_params(length=0, colors=INK2)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(mesh, ax=ax, fraction=0.02, pad=0.01, aspect=50)
cb.set_label(f"재방문율 (%) · {vmax:.1f}% 이상은 가장 진한 색", fontsize=8, color=INK2)
cb.ax.tick_params(labelsize=7, colors=INK2)
cb.outline.set_visible(False)
ax.set_title("일별 방문 코호트 리텐션 (행 = 첫 방문일 UTC, 열 = N일, 칸 = N일째 재방문자 수 / 재방문율)",
             loc="left", fontsize=12, color=INK, pad=24)
fig.text(0.01, 0.004, "Day 0 칸 = 코호트 크기 · 빈 칸 = 9/14(마지막 완전 관측일) 이후라 관측 불가 · "
         "* = 첫 4주 코호트(이전 이력 미관측, 신규로 단정 불가) · 요약 행 = 05-31 이후 코호트 합산 · "
         f"Day {N_SHOW + 1} 이후는 {TABLE.relative_to(ROOT)}",
         fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.006, 1, 1))
FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG, dpi=170)
print(f"\n저장: {FIG.relative_to(ROOT)}, {TABLE.relative_to(ROOT)}")
