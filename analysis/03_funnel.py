"""View → AddToCart → Transaction 퍼널 (02_clean_events.py 정제 데이터 기준, ~2015-09-15 21:00 UTC).

퍼널 정의 (방문자 단위, 순차형)
- 1단계 View: view 이벤트가 1건 이상인 방문자
- 2단계 장바구니 담기: 첫 view 시점 이후(같은 시각 포함) addtocart가 있는 방문자
- 3단계 구매: 위 첫 addtocart 시점 이후(같은 시각 포함) transaction이 있는 방문자
상품 일치 여부는 따지지 않는다. 비교용으로 순서를 따지지 않는 방문자 수도 함께 저장한다.
단계 전환율 = 전 단계 방문자 중 다음 단계까지 간 방문자 비율
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "data" / "processed" / "events_clean.csv"
FIG = ROOT / "outputs" / "figures" / "funnel_view_cart_purchase.png"
TABLE = ROOT / "outputs" / "tables" / "funnel_conversion.csv"

df = pd.read_csv(CLEAN, usecols=["timestamp", "visitorid", "event"])

# 순차형 퍼널
first_view = df[df["event"] == "view"].groupby("visitorid")["timestamp"].min().rename("first_view")
cart = df[df["event"] == "addtocart"].merge(first_view, on="visitorid")
first_cart = (cart[cart["timestamp"] >= cart["first_view"]]
              .groupby("visitorid")["timestamp"].min().rename("first_cart"))
tx = df[df["event"] == "transaction"].merge(first_cart, on="visitorid")
purchasers = tx.loc[tx["timestamp"] >= tx["first_cart"], "visitorid"].unique()

stages = ["View", "AddToCart", "Transaction"]
ordered = [first_view.size, first_cart.size, len(purchasers)]
# 순서 무관 (각 이벤트를 한 번이라도 한 방문자)
unordered = [df.loc[df["event"] == e, "visitorid"].nunique() for e in ["view", "addtocart", "transaction"]]

result = pd.DataFrame({
    "stage": stages,
    "visitors": ordered,
    "step_conversion_pct": [100.0] + [ordered[i] / ordered[i - 1] * 100 for i in (1, 2)],
    "overall_conversion_pct": [v / ordered[0] * 100 for v in ordered],
    "visitors_unordered": unordered,
}).round(2)
TABLE.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(TABLE, index=False, encoding="utf-8-sig")
print(result.to_string(index=False))

# 그래프
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
BAR, INK, INK2 = "#2a78d6", "#0b0b0b", "#52514e"

fig, ax = plt.subplots(figsize=(9, 4.6), facecolor="#fcfcfb")
ax.set_facecolor("#fcfcfb")
y = range(len(stages))[::-1]
ax.barh(list(y), ordered, height=0.55, color=BAR)
for yi, n, pct in zip(y, ordered, result["overall_conversion_pct"]):
    ax.text(n + ordered[0] * 0.01, yi, f"{n:,}명 ({pct:.2f}%)", va="center", fontsize=11, color=INK)
for i in (1, 2):
    ax.text(ordered[0] * 0.30, (y[i - 1] + y[i]) / 2,
            f"↓ 전 단계 대비 전환율 {result['step_conversion_pct'][i]:.2f}%  ({ordered[i - 1]:,}명 중 {ordered[i]:,}명)",
            va="center", fontsize=10.5, color=INK)
ax.set_yticks(list(y), stages, fontsize=12, color=INK)
ax.set_xlim(0, ordered[0] * 1.3)
ax.xaxis.set_visible(False)
for s in ["top", "right", "bottom"]:
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color("#d0cfca")
ax.tick_params(axis="y", length=0)
ax.set_title("View → AddToCart → Transaction 퍼널 (방문자 기준, 순차)", loc="left", fontsize=13, color=INK, pad=12)
fig.text(0.01, 0.01, "괄호: View 대비 전체 전환율 · 기간 2015-05-03 ~ 2015-09-15 21:00 (UTC) · 상품 일치 여부 무관",
         fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
FIG.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG, dpi=150)
print(f"\n저장: {FIG.relative_to(ROOT)}, {TABLE.relative_to(ROOT)}")
