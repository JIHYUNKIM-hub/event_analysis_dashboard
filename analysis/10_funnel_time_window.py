"""View → AddToCart → Transaction 퍼널을 첫 조회 후 24시간·7일 이내로 제한해 비교 (03_funnel.py와 같은 순차 정의).

정의 (방문자 단위)
- View: 방문자의 첫 view 시각 t0
- AddToCart: t0 이후(같은 시각 포함) 첫 addtocart 시각 t1, t1 - t0 <= 제한 시간
- Transaction: t1 이후(같은 시각 포함) transaction이 있고, 그 시각 - t0 <= 제한 시간 (조회부터 구매까지)
- 제한 없음: 기준 시점(2015-09-15 21:00 UTC)까지 관측된 전체
공정한 비교를 위한 코호트
- 첫 조회 후 7일을 모두 관측할 수 있는 방문자만 쓴다: t0 <= 기준 시점 - 7일.
  세 조건(24시간·7일·제한 없음)에 같은 코호트를 써서 차이가 시간 제한에서만 나오게 한다.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "processed" / "events_clean.csv"
FIG = ROOT / "outputs" / "figures" / "funnel_time_window_comparison.png"
TABLE = ROOT / "outputs" / "tables" / "funnel_time_window_comparison.csv"

CUTOFF_MS = int(pd.Timestamp("2015-09-15 21:00:00", tz="UTC").timestamp() * 1000)
HOUR_MS = 3_600_000
WINDOWS = {"24시간 이내": 24 * HOUR_MS, "7일 이내": 7 * 24 * HOUR_MS, "제한 없음": None}
MAX_WINDOW_MS = 7 * 24 * HOUR_MS

df = pd.read_csv(CLEAN, usecols=["timestamp", "visitorid", "event"])

t0 = df[df["event"] == "view"].groupby("visitorid")["timestamp"].min().rename("t0")
n_all = len(t0)
t0 = t0[t0 <= CUTOFF_MS - MAX_WINDOW_MS]
n_excluded = n_all - len(t0)

cart = df[df["event"] == "addtocart"].merge(t0, on="visitorid")
t1 = cart[cart["timestamp"] >= cart["t0"]].groupby("visitorid")["timestamp"].min().rename("t1")
tx = df[df["event"] == "transaction"].merge(pd.concat([t0, t1], axis=1).dropna(), on="visitorid")
tx = tx[tx["timestamp"] >= tx["t1"]]
t2 = tx.groupby("visitorid")["timestamp"].min().rename("t2")  # 첫 장바구니 이후 첫 구매

v = pd.concat([t0, t1, t2], axis=1)
rows = []
for name, w in WINDOWS.items():
    limit = np.inf if w is None else w
    carted = v["t1"].notna() & ((v["t1"] - v["t0"]) <= limit)
    bought = carted & v["t2"].notna() & ((v["t2"] - v["t0"]) <= limit)
    counts = [len(v), int(carted.sum()), int(bought.sum())]
    for i, stage in enumerate(["View", "AddToCart", "Transaction"]):
        rows.append({
            "window": name, "stage": stage, "visitors": counts[i],
            "step_conversion_pct": 100.0 if i == 0 else counts[i] / counts[i - 1] * 100,
            "overall_conversion_pct": counts[i] / counts[0] * 100,
        })
result = pd.DataFrame(rows).round(2)

# 구매까지 걸린 시간 분포 (제한 없음 기준 구매자)
hours = (v["t2"].dropna() - v.loc[v["t2"].notna(), "t0"]) / HOUR_MS
time_stats = hours.describe(percentiles=[0.25, 0.5, 0.75, 0.9]).round(2)

TABLE.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(TABLE, index=False, encoding="utf-8-sig")
print(f"코호트: 첫 조회 {pd.to_datetime(CUTOFF_MS - MAX_WINDOW_MS, unit='ms', utc=True)} 이전 방문자 "
      f"{len(t0):,}명 (7일 관측 불가로 제외 {n_excluded:,}명)")
print(result.to_string(index=False))
print("\n[첫 조회 → 구매 소요 시간(시간), 제한 없음 구매자]")
print(time_stats.to_string())
print(f"24시간 이내 구매 비중: {(hours <= 24).mean() * 100:.1f}%, 7일 이내: {(hours <= 168).mean() * 100:.1f}%")

# 그래프: 왼쪽 단계별 방문자, 오른쪽 단계 전환율 (조건별 막대)
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"24시간 이내": "#2a78d6", "7일 이내": "#eb6834", "제한 없음": "#1baf7a"}  # 범주 팔레트 1~3번

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5.4), facecolor=SURFACE, gridspec_kw={"width_ratios": [1, 1]})
bar_h = 0.26

# 왼쪽: AddToCart·Transaction 방문자 수 (View는 세 조건 모두 같아 제목에 표기)
stages = ["AddToCart", "Transaction"]
for k, name in enumerate(WINDOWS):
    sub = result[(result["window"] == name) & result["stage"].isin(stages)]
    ys = np.arange(len(stages))[::-1] + (1 - k) * bar_h
    ax_l.barh(ys, sub["visitors"], height=bar_h - 0.03, color=COLORS[name], label=name)
    for yv, n, p in zip(ys, sub["visitors"], sub["overall_conversion_pct"]):
        ax_l.text(n * 1.01, yv, f"{n:,}명 ({p:.2f}%)", va="center", fontsize=8.5, color=INK)
ax_l.set_yticks(np.arange(len(stages))[::-1], stages, fontsize=11, color=INK)
ax_l.set_xlim(0, result.loc[result["stage"] == "AddToCart", "visitors"].max() * 1.35)
ax_l.set_title(f"단계별 방문자 수 (View {len(v):,}명 기준, 괄호: View 대비)", loc="left", fontsize=11.5, color=INK)

# 오른쪽: 전 단계 대비 전환율
steps = [("View→AddToCart", "AddToCart", "step_conversion_pct"),
         ("AddToCart→Transaction", "Transaction", "step_conversion_pct"),
         ("View→Transaction (전체)", "Transaction", "overall_conversion_pct")]
for k, name in enumerate(WINDOWS):
    vals = [result.loc[(result["window"] == name) & (result["stage"] == st_), col].iat[0] for _, st_, col in steps]
    ys = np.arange(len(steps))[::-1] + (1 - k) * bar_h
    ax_r.barh(ys, vals, height=bar_h - 0.03, color=COLORS[name])
    for yv, val in zip(ys, vals):
        ax_r.text(val + 0.4, yv, f"{val:.2f}%", va="center", fontsize=8.5, color=INK)
ax_r.set_yticks(np.arange(len(steps))[::-1], [s[0] for s in steps], fontsize=10.5, color=INK)
ax_r.set_xlim(0, 36)
ax_r.set_title("전환율 (전 단계 대비, 마지막 줄은 View 대비)", loc="left", fontsize=11.5, color=INK)

for ax in (ax_l, ax_r):
    ax.set_facecolor(SURFACE)
    ax.tick_params(axis="x", labelsize=8.5, colors=INK2, length=0)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ["top", "right", "bottom"]:
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#d0cfca")
ax_l.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax_r.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))

fig.legend(*ax_l.get_legend_handles_labels(), loc="upper right", ncol=3, frameon=False, fontsize=10,
           bbox_to_anchor=(0.99, 0.99))
fig.suptitle("조회→구매 시간 제한별 퍼널 비교 (View → AddToCart → Transaction)", x=0.01, ha="left",
             fontsize=13.5, color=INK)
fig.text(0.01, 0.012,
         f"코호트: 첫 조회 후 7일을 모두 관측할 수 있는 방문자(첫 조회 ≤ 2015-09-08 21:00 UTC) {len(v):,}명, 세 조건 공통 · "
         "시간 제한은 첫 조회부터 각 단계까지 걸린 시간 · 순차 기준, 상품 일치 무관",
         fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 0.93))
FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG, dpi=150)
print(f"\n저장: {FIG.relative_to(ROOT)}, {TABLE.relative_to(ROOT)}")
