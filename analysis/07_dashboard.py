"""주요 지표 Streamlit 대시보드.

실행: streamlit run analysis/07_dashboard.py
- 주요 지표 탭: 지표 정의는 05_key_metrics.py, 월 기준은 06_monthly_key_metrics.py와 같다 (UTC).
  선택한 기간 안의 이벤트만으로 계산하므로 고유 방문자 수는 기간마다 새로 센다.
- 카테고리별 첫 조회 성과 탭: 첫 조회 정의는 09_first_view_by_category.py와 같다.
  방문자별 첫 조회는 전체 관측 기간 기준으로 정하고, 그 첫 조회 날짜가 선택 기간에 들어가는 방문자만 집계한다.
  장바구니·구매 전환율은 첫 조회 방문자 중 첫 조회 시각 이후(같은 시각 포함) 해당 이벤트를
  1회 이상 한 방문자 비율이다 (상품·카테고리 무관, 기준 시점까지 관측).
- 퍼널 탭: funnel_utils.first_view_funnel 정의 (첫 조회 후 24시간 이내, 진입 카테고리 기준).
  11_category_funnel.py와 같은 계산이며, 첫 조회 날짜가 선택 기간에 들어가는 방문자만 집계한다.
- 리텐션 탭과 카테고리 탭의 30일 재방문율: retention_utils 정의 (12·13번 스크립트와 같은 N-day 리텐션).
  Day 0 날짜가 선택 기간에 들어가는 방문자만 집계한다.
  리텐션 탭은 일별·주별(월요일 시작)·월별 단위를 고를 수 있고, 일별 코호트 히트맵은 14_retention_cohort_heatmap.py와 같은 정의다.
"""
import calendar
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from funnel_utils import (COUNT_LABELS, TIME_LABELS, count_distribution, first_view_funnel,  # noqa: E402
                          funnel_counts, group_funnel, time_distribution)
from retention_utils import (DAY_MS, EARLY_END, LAST_FULL_DAY, UNIT_NAMES, active_days, cohort_matrix,  # noqa: E402
                             curve, first_action, group_rates, last_full_period, period_bounds, period_index,
                             period_label, return_offsets, to_day, to_unit)

ROOT = Path(__file__).resolve().parents[1]
# 배포용 압축 데이터 (16_build_dashboard_data.py가 events_with_category.csv에서 필요한 열만 추려 만든다)
DATA = ROOT / "data" / "processed" / "dashboard_events.csv.gz"

BLUE = "#2a78d6"
NO_CAT = "카테고리 없음"
TOP_N = 10
METRICS = [
    ("active_users", "활성 사용자 수", "명", "이벤트 1회 이상 고유 방문자"),
    ("buyers", "구매자 수", "명", "transaction 1회 이상 고유 방문자"),
    ("purchases", "구매 횟수", "회", "고유 transactionid 수"),
    ("addtocart_count", "장바구니 담기 횟수", "회", "addtocart 이벤트 수"),
    ("event_count", "이벤트 횟수", "건", "전체 이벤트 수"),
]

st.set_page_config(page_title="주요 지표 대시보드", layout="wide")


def ms_to_date(ts: pd.Series) -> pd.Series:
    """Unix ms → UTC 날짜 (시간대 정보 없는 자정 값)."""
    return pd.to_datetime(ts, unit="ms").dt.normalize()


@st.cache_resource(show_spinner="이벤트 데이터를 불러오는 중...")
def load_cat_events() -> pd.DataFrame:
    """카테고리가 붙은 정제 이벤트 (모든 탭이 읽기 전용으로 함께 사용)."""
    return pd.read_csv(DATA, dtype={"transactionid": "Int64", "categoryid": "Int64", "root_categoryid": "Int64",
                                    "event": "category"})


@st.cache_data(show_spinner="이벤트 데이터를 정리하는 중...")
def load_events() -> pd.DataFrame:
    """주요 지표 탭용: 방문자·이벤트·거래와 UTC 날짜·월."""
    ev = load_cat_events()[["timestamp", "visitorid", "event", "transactionid"]].copy()
    ev["date"] = ms_to_date(ev["timestamp"])
    ev["month"] = ev["date"].dt.to_period("M")
    return ev.drop(columns="timestamp")


@st.cache_data(show_spinner="첫 조회 데이터를 만드는 중...")
def load_first_views() -> pd.DataFrame:
    """방문자별 첫 view 1건 (같은 시각이면 itemid가 작은 것)."""
    ev = load_cat_events()
    views = ev[ev["event"] == "view"]
    first = views.sort_values(["visitorid", "timestamp", "itemid"]).drop_duplicates("visitorid", keep="first")
    first = first.assign(date=ms_to_date(first["timestamp"]))
    # 전환: 첫 조회 시각 이후(같은 시각 포함) 해당 이벤트가 1회 이상 있으면 전환 (상품·카테고리 무관)
    for event, col in [("addtocart", "carted"), ("transaction", "purchased")]:
        last_ts = ev[ev["event"] == event].groupby("visitorid")["timestamp"].max()
        first[col] = first["visitorid"].map(last_ts).ge(first["timestamp"]).fillna(False).astype(bool)
    # 30일 재방문: 첫 조회 날짜(Day 0)로부터 정확히 30일째 UTC 날짜에 이벤트가 1건 이상 (retention_utils 정의)
    first["day0"] = first["timestamp"] // DAY_MS
    first["elig30"] = first["day0"] <= LAST_FULL_DAY - 30
    back = return_offsets(first.set_index("visitorid")[["day0"]], active_days(ev))
    first["ret30"] = first["visitorid"].isin(back.loc[back["n"] == 30, "visitorid"])
    return first[["date", "visitorid", "itemid", "categoryid", "root_categoryid", "carted", "purchased",
                  "elig30", "ret30"]]


@st.cache_data(show_spinner="24시간 퍼널을 계산하는 중...")
def load_funnel() -> pd.DataFrame:
    """funnel_utils.first_view_funnel 결과 (24시간, 진입 카테고리 기준)."""
    fv = first_view_funnel(load_cat_events()).reset_index()
    fv["date"] = ms_to_date(fv["t0"])
    return fv


def summarize(ev: pd.DataFrame, key: str | None = None):
    """key가 None이면 전체 값(dict), 있으면 key별 집계(DataFrame)."""
    if key is None:
        return {
            "active_users": ev["visitorid"].nunique(),
            "buyers": ev.loc[ev["event"] == "transaction", "visitorid"].nunique(),
            "purchases": ev["transactionid"].nunique(),
            "addtocart_count": int((ev["event"] == "addtocart").sum()),
            "event_count": len(ev),
        }
    g = ev.groupby(key, observed=True)
    out = pd.DataFrame({
        "active_users": g["visitorid"].nunique(),
        "buyers": ev[ev["event"] == "transaction"].groupby(key, observed=True)["visitorid"].nunique(),
        "purchases": g["transactionid"].nunique(),
        "addtocart_count": ev[ev["event"] == "addtocart"].groupby(key, observed=True).size(),
        "event_count": g.size(),
        "observed_days": g["date"].nunique(),
    })
    return out.fillna(0).astype("int64")


def line_chart(x, y, unit, hover_x, height, marker_symbols=None, extra_hover=None):
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="lines+markers",
        line=dict(color=BLUE, width=2),
        marker=dict(size=8 if marker_symbols is not None else 4, color=BLUE,
                    symbol=marker_symbols if marker_symbols is not None else "circle",
                    line=dict(color=BLUE, width=2)),
        customdata=extra_hover,
        hovertemplate=f"{hover_x}<br>%{{y:,}}{unit}" + ("%{customdata}" if extra_hover is not None else "") + "<extra></extra>",
    ))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      hovermode="x unified" if marker_symbols is None else "closest")
    fig.update_yaxes(rangemode="tozero", tickformat=",", showgrid=True)
    fig.update_xaxes(showgrid=False, type="category" if marker_symbols is not None else "date")
    return fig


def rank_first_views(df: pd.DataFrame, keys: list[str], total: int) -> pd.DataFrame:
    """keys별 첫 조회 방문자 수 순위. total 대비 비율(%)과 장바구니·구매 전환율(%) 포함."""
    out = (df.groupby(keys, dropna=False)
           .agg(visitors=("visitorid", "size"), carted=("carted", "sum"), purchased=("purchased", "sum"),
                elig30=("elig30", "sum"), ret30=("ret30", "sum"))
           .reset_index()
           .sort_values(["visitors"] + keys, ascending=[False] + [True] * len(keys), ignore_index=True))
    out["share"] = out["visitors"] / total * 100
    out["cart_rate"] = out["carted"] / out["visitors"] * 100
    out["purchase_rate"] = out["purchased"] / out["visitors"] * 100
    out["retention30"] = out["ret30"] / out["elig30"].where(out["elig30"] > 0) * 100
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


# 드릴다운 그래프 지표: (열, 비율 여부, 분모 열)
CHART_METRICS = {
    "첫 조회 방문자": ("visitors", False, "visitors"),
    "장바구니 전환율": ("cart_rate", True, "visitors"),
    "구매 전환율": ("purchase_rate", True, "visitors"),
    "30일 재방문율": ("retention30", True, "elig30"),
}


def metric_bar(df: pd.DataFrame, labels, metric: str, base_total: int, min_n: int, key: str):
    """metric 기준 상위 20개 막대. 비율은 분모(첫 조회 방문자, 30일 재방문율은 Day 30 관측 가능 방문자)가
    min_n명 이상인 항목만 비교한다."""
    col, is_rate, base = CHART_METRICS[metric]
    d = df.assign(label=labels)
    if is_rate:
        d = d[d[base] >= min_n].sort_values([col, base], ascending=[False, False])
        if d.empty:
            st.info(f"분모가 {min_n}명 이상인 항목이 없어 비율을 비교하지 않습니다.")
            return
        text = [f"{v:.2f}% ({n:,}명)" for v, n in zip(d[col], d[base])]
        hover = "%{y}<br>%{x:.2f}%<extra></extra>"
        xmax = max(d[col].max() * 1.3, 0.1)
    else:
        text = [f"{v:,}명 ({v / base_total * 100:.1f}%)" for v in d[col]]
        hover = "%{y}<br>%{x:,}명<extra></extra>"
        xmax = d[col].max() * 1.2
    d = d.head(20)
    fig = go.Figure(go.Bar(x=d[col], y=d["label"], orientation="h", marker_color=BLUE,
                           text=text[:len(d)], textposition="outside", cliponaxis=False, hovertemplate=hover))
    fig.update_layout(height=max(220, 30 * len(d) + 60), margin=dict(l=10, r=100, t=10, b=10),
                      showlegend=False, bargap=0.3)
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_xaxes(tickformat=".1f" if is_rate else ",", ticksuffix="%" if is_rate else "",
                     showgrid=True, range=[0, xmax])
    st.plotly_chart(fig, use_container_width=True, key=key, config={"displayModeBar": False})


RATE_COLS = {"cart_rate": "장바구니%", "purchase_rate": "구매%", "retention30": "30일 재방문%"}


# 히트맵 표: 수치 열마다 따로 정규화해 값이 클수록 진한 파랑 (단색 순차 팔레트)
HEAT_CMAP = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fe", "#b7d3f6", "#5598e7", "#1c5cab"])


def heat_table(df: pd.DataFrame, label_cols=(), container=st, height: int | None = None, column_config=None):
    """label_cols(순위·ID·날짜 등)를 뺀 수치 열에 열별 배경 그라데이션을 입힌 표."""
    heat = [c for c in df.columns if c not in label_cols
            and pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    fmt = {c: "{:,.0f}" if pd.api.types.is_integer_dtype(df[c]) else "{:,.2f}" for c in heat}
    df = df.copy()
    df[heat] = df[heat].astype("float64")
    sty = df.style.format(na_rep="-").format(fmt, na_rep="-")
    # 표본이 작은 항목의 극단값이 색 범위를 차지하지 않도록 열마다 5~95 백분위로 색 범위를 자른다
    for c in heat:
        lo, hi = df[c].quantile(0.05), df[c].quantile(0.95)
        if pd.notna(lo) and hi > lo:
            sty = sty.background_gradient(cmap=HEAT_CMAP, subset=[c], vmin=lo, vmax=hi)
    kwargs = {"hide_index": True, "use_container_width": True, "column_config": column_config}
    if height:
        kwargs["height"] = height
    container.dataframe(sty, **kwargs)


def top_table(df: pd.DataFrame, cols: dict[str, str]):
    view = df.head(TOP_N)[["rank", *cols, "visitors", *RATE_COLS]].rename(
        columns={"rank": "순위", **cols, "visitors": "방문자", **RATE_COLS})
    heat_table(view, label_cols=["순위", *cols.values()], height=35 * TOP_N + 40,
               column_config={"순위": st.column_config.NumberColumn(width="small")})


ev = load_events()
min_date, max_date = ev["date"].min().date(), ev["date"].max().date()

# ── 상단: 제목 + 기간 선택 (두 탭 공통) ──────────────────
title_col, date_col = st.columns([3, 2], vertical_alignment="bottom")
with title_col:
    st.title("주요 지표 대시보드")
    st.caption("Retailrocket 이벤트 로그 · UTC 기준 · 분석 기준 시점 2015-09-15 21:00 UTC "
               "· 표의 배경색은 같은 열 안에서 값이 클수록 진합니다 (열의 상·하위 5% 극단값은 가장 진한·연한 색으로 표시)")
    st.caption("데이터: [Retailrocket recommender system dataset](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset) "
               "(Retailrocket / Roman Zykov), [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). "
               "정제·카테고리 조인 등 가공한 데이터이며 비상업적 교육 목적으로만 사용합니다.")
with date_col:
    picked = st.date_input("기간 선택 (UTC)", value=(min_date, max_date),
                           min_value=min_date, max_value=max_date, format="YYYY-MM-DD")

if not isinstance(picked, tuple) or len(picked) != 2:
    st.info("캘린더에서 시작일과 종료일을 모두 선택해 주세요.")
    st.stop()
start, end = pd.Timestamp(picked[0]), pd.Timestamp(picked[1])
n_days = (end - start).days + 1

tab_kpi, tab_cat, tab_funnel, tab_ret = st.tabs(["주요 지표", "카테고리별 첫 조회 성과", "퍼널", "리텐션"])

# ════════════════════ 탭 1: 주요 지표 ════════════════════
with tab_kpi:
    sel = ev[(ev["date"] >= start) & (ev["date"] <= end)]

    totals = summarize(sel)
    st.subheader(f"{start:%Y-%m-%d} ~ {end:%Y-%m-%d} ({n_days}일)")
    cards = "".join(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{totals[key]:,}<span class="kpi-unit">{unit}</span></div>'
        f'<div class="kpi-desc">{desc}</div></div>'
        for key, label, unit, desc in METRICS
    )
    st.markdown(f"""
<style>
.kpi-grid {{display:grid; grid-template-columns:repeat(auto-fit, minmax(170px, 1fr)); gap:12px; margin-bottom:8px;}}
.kpi {{border:1px solid rgba(128,128,128,.28); border-radius:10px; padding:14px 16px;}}
.kpi-label {{font-size:.9rem; opacity:.75;}}
.kpi-value {{font-size:1.9rem; font-weight:700; line-height:1.25; margin-top:4px; font-variant-numeric:tabular-nums;}}
.kpi-unit {{font-size:1rem; font-weight:500; margin-left:3px; opacity:.75;}}
.kpi-desc {{font-size:.75rem; opacity:.6; margin-top:4px;}}
</style>
<div class="kpi-grid">{cards}</div>
""", unsafe_allow_html=True)

    notes = []
    if start.date() == min_date:
        notes.append("5/3은 03:00 UTC부터 기록되어 있고, 첫 4주(5/3~5/30)는 그 이전 이력이 관측되지 않습니다.")
    if end.date() == max_date:
        notes.append("9/15는 21:00 UTC(분석 기준 시점)까지만 포함됩니다.")
    notes.append("고유 방문자 수는 선택 기간 안에서 새로 센 값이라 일별·월별 값을 더해도 기간 값과 같지 않습니다. 봇 의심 방문자도 포함되어 있습니다.")
    st.caption(" · ".join(notes))

    st.divider()

    # 추이: 지표마다 일자별(왼쪽) · 월별(오른쪽)
    st.subheader("추이")
    daily = summarize(sel, "date")
    monthly = summarize(sel, "month")
    days_in_month = [calendar.monthrange(p.year, p.month)[1] for p in monthly.index]
    partial = monthly["observed_days"].to_numpy() < days_in_month
    month_labels = [f"{p.month}월" for p in monthly.index]
    month_hover = [f"<br>관측 {d}일" + (" (부분 월)" if p else "") for d, p in zip(monthly["observed_days"], partial)]
    symbols = ["circle-open" if p else "circle" for p in partial]

    head_l, head_r = st.columns([3, 1])
    head_l.markdown("**일자별**")
    head_r.markdown("**월별** · 빈 원: 선택 기간에 한 달이 다 포함되지 않은 부분 월")

    for key, label, unit, _ in METRICS:
        st.markdown(f"##### {label} ({unit})")
        left, right = st.columns([3, 1])
        with left:
            st.plotly_chart(line_chart(daily.index, daily[key], unit, "%{x|%Y-%m-%d (%a)}", 240),
                            use_container_width=True, key=f"d_{key}")
        with right:
            st.plotly_chart(line_chart(month_labels, monthly[key], unit, "%{x}", 240,
                                       marker_symbols=symbols, extra_hover=month_hover),
                            use_container_width=True, key=f"m_{key}")

    with st.expander("표로 보기"):
        tab_d, tab_m = st.tabs(["일자별", "월별"])
        names = {k: f"{lbl}({u})" for k, lbl, u, _ in METRICS} | {"observed_days": "관측일"}
        heat_table(daily.drop(columns="observed_days").rename(columns=names)
                   .rename_axis("날짜").reset_index().assign(날짜=lambda d: d["날짜"].dt.date),
                   label_cols=["날짜"], container=tab_d)
        heat_table(monthly.rename(columns=names).set_axis(month_labels).rename_axis("월").reset_index(),
                   label_cols=["월"], container=tab_m)

    # 카테고리별 전환율 산포도 (첫 조회 기준 장바구니·구매 전환율, 카테고리별 첫 조회 성과 탭과 같은 정의)
    st.divider()
    st.subheader("카테고리별 장바구니·구매 전환율")
    sc_all = load_first_views()
    sc = sc_all[(sc_all["date"] >= start) & (sc_all["date"] <= end)].dropna(subset=["categoryid"])
    cat_pts = (sc.groupby(["categoryid", "root_categoryid"])
               .agg(visitors=("visitorid", "size"), carted=("carted", "mean"), purchased=("purchased", "mean"))
               .reset_index())
    cat_pts[["carted", "purchased"]] *= 100

    # 색은 대분류에 고정: 전체 기간 첫 조회 규모 순서로 범주 팔레트를 배정해 기간을 바꿔도 같은 대분류는 같은 색
    root_order = (sc_all.dropna(subset=["root_categoryid"]).groupby("root_categoryid").size()
                  .sort_values(ascending=False).index.tolist())
    PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
    OTHER_GRAY = "#a3a29c"
    opt_l, opt_r = st.columns([3, 1])
    with opt_l:
        colored = st.multiselect("색으로 구분할 대분류 (최대 8개, 나머지는 회색 '기타')", root_order,
                                 default=root_order[:8], max_selections=8, format_func=lambda r: f"대분류 {r}",
                                 key="sc_roots")
    with opt_r:
        min_n = st.number_input("최소 첫 조회 방문자 수", min_value=1, value=100, step=50, key="sc_min_n",
                                help="표본이 작은 카테고리는 전환율이 크게 흔들려 제외합니다.")
    color_of = {r: PALETTE[root_order.index(r) % len(PALETTE)] for r in colored}
    pts = cat_pts[cat_pts["visitors"] >= min_n]

    if pts.empty:
        st.info("조건에 맞는 카테고리가 없습니다. 최소 첫 조회 방문자 수를 낮춰 보세요.")
    else:
        size_ref = 2.0 * pts["visitors"].max() / (48 ** 2)  # 가장 큰 풍선 지름 약 48px, 면적 비례
        fig = go.Figure()
        groups = [(r, f"대분류 {r}", color_of[r]) for r in colored if r in set(pts["root_categoryid"])]
        others = pts[~pts["root_categoryid"].isin(colored)]
        if not others.empty:
            groups.append((None, "기타", OTHER_GRAY))
        for r, name, color in groups:
            g = others if r is None else pts[pts["root_categoryid"] == r]
            fig.add_trace(go.Scatter(
                x=g["carted"], y=g["purchased"], mode="markers", name=f"{name} ({len(g)})",
                marker=dict(size=g["visitors"], sizemode="area", sizeref=size_ref, sizemin=3,
                            color=color, opacity=0.75, line=dict(color="white", width=1)),
                customdata=g[["categoryid", "root_categoryid", "visitors"]].to_numpy(),
                hovertemplate="카테고리 %{customdata[0]} (대분류 %{customdata[1]})<br>"
                              "첫 조회 방문자 %{customdata[2]:,}명<br>"
                              "장바구니 전환율 %{x:.2f}%<br>구매 전환율 %{y:.2f}%<extra></extra>",
            ))
        avg_cart, avg_buy = sc["carted"].mean() * 100, sc["purchased"].mean() * 100
        fig.add_vline(x=avg_cart, line=dict(color="#8a8984", width=1, dash="dash"),
                      annotation_text=f"평균 {avg_cart:.2f}%", annotation_position="top")
        fig.add_hline(y=avg_buy, line=dict(color="#8a8984", width=1, dash="dash"),
                      annotation_text=f"평균 {avg_buy:.2f}%", annotation_position="right")
        fig.update_layout(height=560, margin=dict(l=10, r=10, t=30, b=10),
                          legend=dict(title="대분류 (카테고리 수)", itemsizing="constant"))
        fig.update_xaxes(title="장바구니 전환율 (%)", ticksuffix="%", rangemode="tozero", showgrid=True)
        fig.update_yaxes(title="구매 전환율 (%)", ticksuffix="%", rangemode="tozero", showgrid=True)
        st.plotly_chart(fig, use_container_width=True, key="cat_scatter")
    st.caption(
        f"점 하나 = 카테고리 하나 (첫 조회 방문자 {min_n:,}명 이상 {len(pts):,}개, 카테고리 기록이 있는 첫 조회만). "
        "풍선 크기 = 그 카테고리의 첫 조회 방문자 수. 색 = 대분류. "
        "장바구니·구매 전환율 = 첫 조회 방문자 중 첫 조회 이후(같은 시각 포함) 장바구니 담기·구매를 1회 이상 한 비율 "
        "(상품·카테고리 무관, 기준 시점까지 관측). 점선 = 선택 기간 카테고리 있는 첫 조회 전체 평균. "
        "기간은 첫 조회 날짜 기준이며 범례를 눌러 대분류를 숨기거나 더블클릭해 하나만 볼 수 있습니다."
    )

# ════════════════ 탭 3: 퍼널 (24시간) ════════════════
# 탭 2는 데이터가 없으면 st.stop()으로 끝나므로, 탭 3·4를 먼저 그린다 (화면의 탭 순서는 st.tabs 순서를 따른다).
STAGES = ["View", "AddToCart", "Transaction"]


def funnel_chart(c: dict, height: int = 300):
    vals = [c["view"], c["addtocart"], c["transaction"]]
    prev = [None, c["view_to_cart_pct"], c["cart_to_tx_pct"]]
    text = [f"{vals[0]:,}명"] + [
        "-" if not vals[0] or pd.isna(p) else f"{v:,}명<br>이전 대비 {p:.2f}%<br>View 대비 {v / vals[0] * 100:.2f}%"
        for v, p in zip(vals[1:], prev[1:])]
    fig = go.Figure(go.Funnel(
        y=STAGES, x=vals, text=text, textinfo="text", textposition=["inside", "outside", "outside"],
        insidetextfont=dict(color="white", size=13), outsidetextfont=dict(size=11),
        marker=dict(color=BLUE), connector=dict(fillcolor="rgba(42,120,214,0.15)", line=dict(width=0)),
        hovertemplate="%{y}<br>%{x:,}명<extra></extra>", cliponaxis=False,
    ))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    fig.update_yaxes(tickfont=dict(size=13))
    return fig


def dist_chart(counts: pd.Series, unit_label: str, key: str):
    total_n = int(counts.sum())
    pct = counts / total_n * 100 if total_n else counts * 0
    fig = go.Figure(go.Bar(
        x=counts.index.astype(str), y=counts.values, marker_color=BLUE,
        text=[f"{p:.1f}%" for p in pct], textposition="outside", cliponaxis=False,
        hovertemplate=f"%{{x}}<br>%{{y:,}}명<extra></extra>",
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), showlegend=False, bargap=0.25)
    fig.update_yaxes(tickformat=",", title=unit_label, range=[0, max(counts.max(), 1) * 1.18])
    fig.update_xaxes(type="category")
    st.plotly_chart(fig, use_container_width=True, key=key, config={"displayModeBar": False})


def root_name(r) -> str:
    return NO_CAT if r == -1 else f"대분류 {r}"


FUNNEL_COLS = {
    "view": "View", "addtocart": "AddToCart", "transaction": "Transaction",
    "view_to_cart_pct": "View→Cart(%)", "cart_to_tx_pct": "Cart→Tx(%)", "view_to_tx_pct": "View→Tx(%)",
    "cart_minutes_median": "Cart까지 중앙값(분)", "tx_minutes_median": "Tx까지 중앙값(분)",
}

with tab_funnel:
    fn_all = load_funnel()
    fn = fn_all[(fn_all["date"] >= start) & (fn_all["date"] <= end)].copy()
    st.subheader(f"{start:%Y-%m-%d} ~ {end:%Y-%m-%d} 퍼널 · 첫 조회 후 24시간 이내")
    if fn.empty:
        st.info("선택 기간에 24시간을 모두 관측할 수 있는 첫 조회 방문자가 없습니다. (9/14 21:00 UTC 이후 첫 조회는 제외)")
    else:
        fn["root_key"] = fn["root_categoryid"].fillna(-1).astype(int)
        fn["cat_key"] = fn["categoryid"].fillna(-1).astype(int)
        overall = funnel_counts(fn)

        kpis = [
            ("첫 조회 방문자 (View)", f"{overall['view']:,}", "명", "24시간 관측 가능한 코호트"),
            ("View → AddToCart", f"{overall['view_to_cart_pct']:.2f}", "%", f"{overall['addtocart']:,}명 장바구니 담기"),
            ("AddToCart → Transaction", f"{overall['cart_to_tx_pct']:.2f}", "%", f"{overall['transaction']:,}명 구매"),
            ("View → Transaction", f"{overall['view_to_tx_pct']:.2f}", "%", "첫 조회 대비 24시간 내 구매"),
        ]
        st.markdown('<div class="kpi-grid">' + "".join(
            f'<div class="kpi"><div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}<span class="kpi-unit">{unit}</span></div>'
            f'<div class="kpi-desc">{desc}</div></div>' for label, value, unit, desc in kpis) + "</div>",
            unsafe_allow_html=True)
        st.caption(
            "View = 방문자의 첫 조회, AddToCart = 첫 조회 후 24시간 안의 첫 장바구니 담기, "
            "Transaction = 그 장바구니 담기 이후이면서 첫 조회 후 24시간 안의 구매. 상품 일치는 따지지 않습니다. "
            "· 대분류·카테고리는 첫 조회 상품의 카테고리(진입 카테고리)입니다. "
            "· 24시간을 모두 관측할 수 있도록 첫 조회가 2015-09-14 21:00 UTC 이전인 방문자만 셉니다. "
            "· 봇 의심 방문자 포함, 첫 4주 방문자는 이전 이력이 관측되지 않습니다."
        )

        by_root = group_funnel(fn, ["root_key"])
        # '카테고리 없음'은 비교 기준이 아니므로 목록 맨 뒤로
        by_root = pd.concat([by_root[by_root["root_key"] != -1], by_root[by_root["root_key"] == -1]], ignore_index=True)
        sel_l, sel_r = st.columns(2)
        with sel_l:
            f_root = st.selectbox("대분류 선택", by_root["root_key"].tolist(), key="f_root",
                                  format_func=lambda r: f"{root_name(r)} · {int(by_root.loc[by_root['root_key'] == r, 'view'].iat[0]):,}명")
        sub = fn[fn["root_key"] == f_root]
        by_cat = group_funnel(sub, ["cat_key"])
        with sel_r:
            f_cat = st.selectbox(f"카테고리 선택 ({root_name(f_root)}의 하위 {len(by_cat)}개)", by_cat["cat_key"].tolist(),
                                 key="f_cat", disabled=f_root == -1,
                                 format_func=lambda c: f"{NO_CAT if c == -1 else f'카테고리 {c}'} · "
                                                       f"{int(by_cat.loc[by_cat['cat_key'] == c, 'view'].iat[0]):,}명")
        sub_cat = sub[sub["cat_key"] == f_cat]

        st.markdown("#### 퍼널")
        f1, f2, f3 = st.columns(3)
        for col, (title, data, key) in zip((f1, f2, f3), [
            ("전체", fn, "fun_all"), (root_name(f_root), sub, "fun_root"),
            ("카테고리 없음" if f_cat == -1 else f"카테고리 {f_cat} ({root_name(f_root)})", sub_cat, "fun_cat")]):
            with col:
                st.markdown(f"**{title}**")
                st.plotly_chart(funnel_chart(funnel_counts(data)), use_container_width=True, key=key,
                                config={"displayModeBar": False})
        st.caption("막대 폭은 방문자 수에 비례합니다. AddToCart·Transaction은 View의 2% 안팎이라 막대가 아주 얇게 보입니다.")

        st.markdown("#### 대분류별 퍼널 비교")
        heat_table(by_root.assign(대분류=by_root["root_key"].map(root_name))[["대분류", *FUNNEL_COLS]]
                   .rename(columns=FUNNEL_COLS), label_cols=["대분류"])
        st.markdown(f"#### {root_name(f_root)}의 카테고리별 퍼널 비교")
        heat_table(by_cat.assign(카테고리=by_cat["cat_key"].map(lambda c: NO_CAT if c == -1 else str(c)))
                   [["카테고리", *FUNNEL_COLS]].rename(columns=FUNNEL_COLS), label_cols=["카테고리"])
        st.caption("방문자가 적은 항목은 전환율이 크게 흔들립니다. View 1,000명 이상 항목 위주로 비교하세요.")

        st.divider()
        head_l, head_r = st.columns([1, 1], vertical_alignment="bottom")
        head_l.markdown("#### 단계 전환 시간 · 이전 단계 수행 횟수")
        scope = head_r.radio("범위", ["전체", "선택한 대분류", "선택한 카테고리"], horizontal=True, key="f_scope")
        data = {"전체": fn, "선택한 대분류": sub, "선택한 카테고리": sub_cat}[scope]

        t_l, t_r = st.columns(2)
        with t_l:
            st.markdown(f"**View → AddToCart 걸린 시간** · {int(data['carted'].sum()):,}명")
            dist_chart(time_distribution(data["cart_minutes"]), "방문자", "time_cart")
        with t_r:
            st.markdown(f"**AddToCart → Transaction 걸린 시간** · {int(data['purchased'].sum()):,}명")
            dist_chart(time_distribution(data["tx_minutes"]), "방문자", "time_tx")
        st.caption("View→AddToCart는 첫 조회부터 첫 장바구니 담기까지, AddToCart→Transaction은 첫 장바구니 담기부터 구매까지 걸린 시간입니다. "
                   "막대 위 비율은 해당 단계 도달자 대비입니다.")

        c_l, c_r = st.columns(2)
        with c_l:
            st.markdown("**AddToCart 도달 전 View 수행 횟수**")
            dist_chart(count_distribution(data["n_views_before_cart"]), "방문자", "cnt_view")
        with c_r:
            st.markdown("**Transaction 도달 전 AddToCart 수행 횟수**")
            dist_chart(count_distribution(data["n_carts_before_tx"]), "방문자", "cnt_cart")
        st.caption("첫 조회부터 첫 장바구니 담기까지의 view 수(첫 조회 포함), 첫 조회부터 구매까지의 장바구니 담기 수입니다. 상품 무관.")

# ════════════════ 탭 4: 리텐션 ════════════════
# 지표: (Day 0 코호트, 복귀 이벤트, 설명) · 코호트 any = 첫 이벤트, tx = 첫 구매
RET_METRICS = {
    "재방문율": ("any", "any", "0 = 첫 이벤트 기간 · 복귀 = 모든 이벤트"),
    "재구매율": ("tx", "tx", "0 = 첫 구매 기간 · 복귀 = 구매"),
    "구매자 재방문율": ("tx", "any", "0 = 첫 구매 기간 · 복귀 = 모든 이벤트"),
}
PALETTE = ["#2a78d6", "#e07b39", "#1f9e6e", "#c2417a", "#7a5cc7", "#b8921c", "#3aa6b9", "#7d7a73"]
UNITS = {"일별": "day", "주별": "week", "월별": "month"}
PREFIX = {"day": "D", "week": "W", "month": "M"}
SUFFIX = {"day": "일", "week": "주", "month": "개월"}
RET_N = {"day": (1, 7, 14, 30), "week": (1, 2, 4, 8), "month": (1, 2, 3)}  # 표에 보일 N
KPI_N = {"day": (30, 1, 7), "week": (4, 1, 2), "month": (1, 2, 3)}       # 카드: 큰 값, 보조 값
N_DEFAULT = {"day": 30, "week": 12, "month": 3}
DATA_START = to_day("2015-05-03")


@st.cache_data(show_spinner="리텐션 코호트를 만드는 중...")
def load_retention():
    """retention_utils 정의. 코호트(첫 행동)와 지표별 복귀 (visitorid, n), 일 단위."""
    ev = load_cat_events()
    cohorts = {"any": first_action(ev), "tx": first_action(ev, "transaction")}
    days = {"any": active_days(ev), "tx": active_days(ev, "transaction")}
    returns = {name: return_offsets(cohorts[c], days[r]) for name, (c, r, _) in RET_METRICS.items()}
    return cohorts, returns


@st.cache_data(show_spinner="기간 단위로 바꾸는 중...")
def load_retention_unit(unit: str) -> dict:
    """지표별 (코호트, 복귀)를 unit 단위로. 코호트 first_day = 원래 첫 행동 날짜."""
    cohorts, returns = load_retention()
    return {name: to_unit(cohorts[c], returns[name], unit) for name, (c, _, _) in RET_METRICS.items()}


def retention_chart(curves: dict[str, pd.DataFrame], unit: str, height: int = 360):
    p = PREFIX[unit]
    fig = go.Figure()
    for i, (name, c) in enumerate(curves.items()):
        fig.add_trace(go.Scatter(
            x=c["n_day"], y=c["rate"], mode="lines+markers" if unit != "day" else "lines", name=name,
            line=dict(color=PALETTE[i % len(PALETTE)], width=2), marker=dict(size=5),
            customdata=np.c_[c["returned"], c["eligible"]],
            hovertemplate=f"{name}<br>{p}%{{x}}: %{{y:.2f}}% (%{{customdata[0]:,}} / %{{customdata[1]:,}}명)<extra></extra>",
        ))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    fig.update_yaxes(rangemode="tozero", ticksuffix="%", showgrid=True, title="재방문율")
    fig.update_xaxes(showgrid=False, title=f"N{SUFFIX[unit]} (0 = 시작 행동 {UNIT_NAMES[unit]}, UTC)",
                     dtick=1 if unit != "day" else None)
    return fig


def cohort_heatmap(cnt: pd.DataFrame, rate: pd.DataFrame, labels: list[str], unit: str):
    """행 = 코호트, 열 = N. 0열은 코호트 크기(회색), N >= 1 칸은 재방문율로 색칠."""
    xs = [f"{PREFIX[unit]}{n}" for n in cnt.columns]
    c, r = cnt.to_numpy(), rate.to_numpy()
    z = r.copy()
    z[:, 0] = np.nan
    vals = z[~np.isnan(z)]
    zmax = max(np.percentile(vals, 98), 0.1) if vals.size else 1  # 소수 극단값이 색 범위를 차지하지 않게
    text = [["" if j == 0 or np.isnan(cv) else f"{int(cv):,}<br>{rv:.1f}%" for j, (cv, rv) in enumerate(zip(cr, rr))]
            for cr, rr in zip(c, r)]
    hover = [[("" if np.isnan(cv) else f"{int(cv):,}명 / {rv:.2f}%") for cv, rv in zip(cr, rr)] for cr, rr in zip(c, r)]
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=[[1] for _ in labels], x=xs[:1], y=labels, colorscale=[[0, "#e4e3df"], [1, "#e4e3df"]], showscale=False,
        text=[[f"{int(v):,}"] for v in c[:, 0]], texttemplate="%{text}", textfont=dict(size=10, color="#0b0b0b"),
        xgap=1, ygap=1, hovertemplate="%{y}<br>코호트 크기 %{text}명<extra></extra>"))
    fig.add_trace(go.Heatmap(
        z=z, x=xs, y=labels, zmin=0, zmax=zmax, hoverongaps=False, xgap=1, ygap=1,
        colorscale=[[0, "#f4f8fe"], [0.33, "#b7d3f6"], [0.66, "#5598e7"], [1, "#1c5cab"]],
        text=text, texttemplate="%{text}", textfont=dict(size=10), customdata=hover,
        colorbar=dict(title="재방문율", ticksuffix="%", thickness=12),
        hovertemplate="%{y} · %{x}<br>%{customdata}<extra></extra>"))
    # 칸 안 글자(방문자 수·비율)가 겹치지 않도록 열마다 최소 폭을 둔다
    fig.update_layout(height=34 * len(labels) + 90, width=max(54 * len(xs) + 260, 700), margin=dict(l=110, r=10, t=40, b=10),
                      font=dict(family="sans-serif"), paper_bgcolor="white", plot_bgcolor="white")
    fig.update_yaxes(autorange="reversed", type="category")
    fig.update_xaxes(side="top", type="category")
    return fig


def rate_text(c: pd.DataFrame, n: int) -> str:
    r = c.loc[c["n_day"] == n, "rate"]
    return "-" if r.empty or pd.isna(r.iat[0]) else f"{r.iat[0]:.2f}"


def ret_table(g: pd.DataFrame, key: str, label: str, fmt, unit: str):
    p, ns = PREFIX[unit], RET_N[unit]
    cols = {key: label, "cohort": f"{p}0 방문자", **{f"d{n}": f"{p}{n}%" for n in ns}, f"elig{ns[-1]}": f"{p}{ns[-1]} 분모"}
    heat_table(g.assign(**{key: g[key].map(fmt)})[list(cols)].rename(columns=cols), label_cols=[label])


def cat_name(c) -> str:
    return NO_CAT if c == -1 else f"카테고리 {c}"


with tab_ret:
    # 리텐션 탭 전용 기간: 첫 행동(첫 방문·첫 구매) 날짜가 이 기간에 있는 방문자만 코호트로 쓴다. 상단 공통 기간은 쓰지 않는다.
    cal_col, unit_col = st.columns([3, 2], vertical_alignment="bottom")
    r_picked = cal_col.date_input("첫 방문일 선택 (UTC, 리텐션 탭 전용)", value=(min_date, max_date),
                                  min_value=min_date, max_value=max_date, format="YYYY-MM-DD", key="ret_dates")
    unit = UNITS[unit_col.segmented_control("기준", list(UNITS), default="일별", key="ret_unit") or "일별"]
    r_dates = r_picked if isinstance(r_picked, tuple) else (r_picked,)
    if len(r_dates) == 1:
        cal_col.caption("종료일을 고르지 않아 시작일 하루만 봅니다.")
    r_start, r_end = pd.Timestamp(r_dates[0]), pd.Timestamp(r_dates[-1])
    st.subheader(f"{r_start:%Y-%m-%d} ~ {r_end:%Y-%m-%d} 첫 방문 코호트 리텐션")
    p, sfx = PREFIX[unit], SUFFIX[unit]
    excl = st.checkbox("첫 행동이 첫 4주(5/3~5/30)인 방문자 제외", value=True,
                       help="첫 4주에는 그 이전 이력이 관측되지 않아 첫 방문·첫 구매로 단정할 수 없습니다.")
    data = load_retention_unit(unit)
    last = last_full_period(unit)
    # 선택 기간은 첫 행동 날짜에 적용한다 (주·월 코호트는 기간 경계에서 일부 날짜만 들어갈 수 있다)
    lo, hi = to_day(r_start), min(to_day(r_end), LAST_FULL_DAY - 1)
    if excl:
        lo = max(lo, EARLY_END + 1)
    # 다음 기간(N = 1)을 끝까지 관측할 수 없는 코호트(예: 월별의 9월)는 뺀다
    sel_ret = {name: (co[co["first_day"].between(lo, hi) & (co["day0"] < last)], ret) for name, (co, ret) in data.items()}
    n_max_all = int(last - period_index([lo], unit)[0]) if lo <= hi else 0

    if lo > hi or sel_ret["재방문율"][0].empty or n_max_all < 1:
        st.info(f"선택 기간에 {p}1을 끝까지 관측할 수 있는 코호트가 없습니다. 마지막 완전 관측일은 9/14이고 "
                f"(주별은 9/7~9/13 주, 월별은 8월까지), 첫 4주 제외를 켜면 5/30 이전 첫 행동도 빠집니다.")
    else:
        n_max = (st.slider(f"N 최대 ({sfx})", 1, n_max_all, min(N_DEFAULT[unit], n_max_all), key=f"ret_nmax_{unit}")
                 if n_max_all > 1 else 1)
        overall = {name: curve(co, ret, n_max, last) for name, (co, ret) in sel_ret.items()}
        kn = KPI_N[unit]
        kpi = {name: curve(co, ret, max(kn), last) for name, (co, ret) in sel_ret.items()}

        st.markdown('<div class="kpi-grid">' + "".join(
            f'<div class="kpi"><div class="kpi-label">{name} · {kn[0]}{sfx}째</div>'
            f'<div class="kpi-value">{rate_text(kpi[name], kn[0])}<span class="kpi-unit">%</span></div>'
            f'<div class="kpi-desc">{kn[1]}{sfx}째 {rate_text(kpi[name], kn[1])}% · {kn[2]}{sfx}째 {rate_text(kpi[name], kn[2])}% '
            f'· 코호트 {len(sel_ret[name][0]):,}명<br>{p}{RET_METRICS[name][2]}</div></div>'
            for name in RET_METRICS) + "</div>", unsafe_allow_html=True)
        st.caption(
            f"N{sfx}째 재방문율 = 첫 행동이 속한 {UNIT_NAMES[unit]}({p}0)로부터 정확히 N번째 {UNIT_NAMES[unit]}에 복귀 이벤트가 있는 방문자 비율 "
            "(그 기간만 셈, 누적 아님). 주는 월요일 시작, 월은 달력 월 기준(UTC)입니다. "
            "· 분모는 N번째 기간이 끝까지 관측된(마지막 날이 9/14 이내) 방문자라 N마다 다릅니다. "
            "· 복귀 이벤트는 카테고리와 무관하게 셉니다. 위 캘린더의 기간은 첫 행동 날짜(첫 방문·첫 구매)에 적용합니다. · 봇 의심 방문자 포함."
        )

        st.markdown("#### 지표별 리텐션 곡선")
        st.plotly_chart(retention_chart(overall, unit), use_container_width=True, key="ret_overall",
                        config={"displayModeBar": False})

        # 코호트 히트맵
        st.divider()
        head_l, head_r = st.columns([1, 1], vertical_alignment="bottom")
        head_l.markdown(f"#### {UNIT_NAMES[unit]}별 코호트 리텐션")
        h_metric = head_r.radio("지표", list(RET_METRICS), horizontal=True, key="ret_hm_metric")
        hco, hret = sel_ret[h_metric]
        cnt, size = cohort_matrix(hco, hret, n_max, last)
        rate = cnt.div(size, axis=0) * 100
        # 맨 위 요약 행: 칸마다 관측 가능한 코호트만 합산
        elig = cnt.notna().mul(size, axis=0).sum()
        total_rate = (cnt.sum() / elig.where(elig > 0) * 100).where(cnt.notna().any())
        total_cnt = cnt.sum(min_count=1)
        total_rate[0], total_cnt[0] = 100.0, size.sum()
        cnt_all = pd.concat([total_cnt.to_frame().T, cnt])
        rate_all = pd.concat([total_rate.to_frame().T, rate])
        # 기간 일부만 들어간 코호트(선택 기간·데이터 시작일 경계)는 †, 첫 4주가 걸친 코호트는 *
        p_start, p_end = period_bounds(size.index, unit)
        partial = (p_start < max(lo, DATA_START)) | (p_end > hi)
        early_mark = p_start <= EARLY_END
        labels = ["전체"] + [lab + (" *" if e else "") + (" †" if pt else "")
                             for lab, e, pt in zip(period_label(size.index, unit), early_mark, partial)]
        # 열이 많으면 화면 폭을 넘으므로 가로 스크롤되는 프레임 안에 그린다 (st.plotly_chart는 화면 폭에 맞춰 줄어든다)
        hm = cohort_heatmap(cnt_all, rate_all, labels, unit)
        components.html(hm.to_html(include_plotlyjs="cdn", full_html=False, config={"displayModeBar": False}),
                        height=hm.layout.height + 30, scrolling=True)
        st.caption(f"행 = 첫 행동 {UNIT_NAMES[unit]}(UTC), 열 = N{sfx}, 칸 = N번째 {UNIT_NAMES[unit]}에 돌아온 방문자 수 / 재방문율. "
                   f"{p}0 칸은 코호트 크기입니다. 빈 칸은 아직 끝까지 관측되지 않은 기간입니다. "
                   "색은 같은 표 안에서 재방문율이 높을수록 진하며, 상위 2% 극단값은 가장 진한 색으로 표시합니다. "
                   "· 전체 = 칸마다 관측 가능한 코호트를 합산한 값 · * = 첫 4주가 걸친 코호트(이전 이력 미관측) "
                   "· † = 선택 기간·데이터 시작일(5/3) 경계 때문에 기간 중 일부 날짜만 들어간 코호트")

        st.divider()
        head_l, head_r = st.columns([1, 1], vertical_alignment="bottom")
        head_l.markdown("#### 카테고리별 리텐션")
        r_metric = head_r.radio("지표", list(RET_METRICS), horizontal=True, key="ret_metric")
        co, ret = sel_ret[r_metric]
        st.caption(f"{p}{RET_METRICS[r_metric][2]} · 카테고리는 첫 행동 상품의 카테고리입니다 "
                   "(재방문율은 첫 이벤트 상품, 재구매율·구매자 재방문율은 첫 구매 상품. 한 구매에 상품이 여럿이면 itemid가 가장 작은 상품). "
                   "곡선은 코호트 방문자가 적으면 크게 흔들리므로 표의 분모를 함께 확인하세요.")

        ns = RET_N[unit]
        by_root = group_rates(co, ret, "root_key", ns, last)
        by_root = pd.concat([by_root[by_root["root_key"] != -1], by_root[by_root["root_key"] == -1]], ignore_index=True)
        root_n = by_root.set_index("root_key")["cohort"]
        roots = st.multiselect("대분류 선택 (최대 8개)", by_root["root_key"].tolist(),
                               default=by_root.loc[by_root["root_key"] != -1, "root_key"].head(5).tolist(),
                               max_selections=8, key=f"ret_roots_{r_metric}",
                               format_func=lambda r: f"{root_name(r)} · {root_n.get(r, 0):,}명")
        if roots:
            st.plotly_chart(retention_chart({root_name(r): curve(co[co["root_key"] == r], ret, n_max, last) for r in roots}, unit),
                            use_container_width=True, key="ret_root_chart", config={"displayModeBar": False})
        st.markdown("**대분류별 리텐션 표**")
        ret_table(by_root, "root_key", "대분류", root_name, unit)

        st.markdown("#### 대분류 안 카테고리별 리텐션")
        r_root = st.selectbox("대분류 선택", by_root["root_key"].tolist(), key=f"ret_root_{r_metric}",
                              format_func=lambda r: f"{root_name(r)} · {root_n.get(r, 0):,}명")
        sub_co = co[co["root_key"] == r_root]
        by_cat = group_rates(sub_co, ret, "cat_key", ns, last)
        cat_n = by_cat.set_index("cat_key")["cohort"]
        cats = st.multiselect(f"카테고리 선택 ({root_name(r_root)}의 하위 {len(by_cat)}개, 최대 8개)", by_cat["cat_key"].tolist(),
                              default=by_cat.loc[by_cat["cat_key"] != -1, "cat_key"].head(5).tolist(),
                              max_selections=8, key=f"ret_cats_{r_metric}_{r_root}",
                              format_func=lambda c: f"{cat_name(c)} · {cat_n.get(c, 0):,}명")
        if cats:
            st.plotly_chart(retention_chart({cat_name(c): curve(sub_co[sub_co["cat_key"] == c], ret, n_max, last) for c in cats}, unit),
                            use_container_width=True, key="ret_cat_chart", config={"displayModeBar": False})
        st.markdown(f"**{root_name(r_root)}의 카테고리별 리텐션 표**")
        ret_table(by_cat, "cat_key", "카테고리", lambda c: NO_CAT if c == -1 else str(c), unit)
        st.caption(f"{p}N% = N{sfx}째 재방문율, {p}{ns[-1]} 분모 = {ns[-1]}{sfx}째를 끝까지 관측할 수 있는 코호트 방문자 수. "
                   "관측할 수 없으면 '-'입니다.")

# ════════════════ 탭 2: 카테고리별 첫 조회 성과 ════════════════
with tab_cat:
    fv_all = load_first_views()
    fv = fv_all[(fv_all["date"] >= start) & (fv_all["date"] <= end)]
    total = len(fv)
    st.subheader(f"{start:%Y-%m-%d} ~ {end:%Y-%m-%d} 첫 조회 성과")
    if total == 0:
        st.info("선택 기간에 첫 조회한 방문자가 없습니다.")
        st.stop()

    kpis = [
        ("첫 조회 방문자", f"{total:,}", "명", "첫 조회 날짜가 선택 기간에 있는 방문자"),
        ("장바구니 전환율", f"{fv['carted'].mean() * 100:.2f}", "%", f"첫 조회 후 장바구니 담기 {int(fv['carted'].sum()):,}명"),
        ("구매 전환율", f"{fv['purchased'].mean() * 100:.2f}", "%", f"첫 조회 후 구매 {int(fv['purchased'].sum()):,}명"),
    ]
    elig30, ret30 = int(fv["elig30"].sum()), int(fv["ret30"].sum())
    kpis.append(("30일 재방문율", f"{ret30 / elig30 * 100:.2f}" if elig30 else "-", "%",
                 f"30일째 관측 가능 {elig30:,}명 중 {ret30:,}명 재방문"))
    st.markdown('<div class="kpi-grid">' + "".join(
        f'<div class="kpi"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}<span class="kpi-unit">{unit}</span></div>'
        f'<div class="kpi-desc">{desc}</div></div>' for label, value, unit, desc in kpis) + "</div>",
        unsafe_allow_html=True)

    no_cat_n = int(fv["categoryid"].isna().sum())
    st.caption(
        "첫 조회 = 방문자별 전체 관측 기간에서 가장 먼저 한 view 1건. 그 날짜가 선택 기간에 들어가는 방문자만 셉니다. "
        "· 장바구니·구매 전환율 = 첫 조회 방문자 중 첫 조회 시각 이후(같은 시각 포함) 장바구니 담기·구매를 1회 이상 한 방문자 비율. "
        "어떤 상품이든 셉니다(첫 조회 상품·카테고리와 같을 필요 없음). 기준 시점(9/15 21:00)까지만 관측하므로 늦게 첫 조회한 방문자일수록 전환할 시간이 짧습니다. "
        "· 30일 재방문율 = 첫 조회 날짜(Day 0)로부터 정확히 30일째 UTC 날짜에 이벤트(종류·카테고리 무관)가 있는 방문자 비율. "
        "분모는 30일째가 마지막 완전 관측일(9/14) 안에 드는 방문자, 즉 첫 조회가 8/15 이전인 방문자뿐입니다. "
        f"· {NO_CAT} {no_cat_n:,}명({no_cat_n / total * 100:.1f}%)은 조회 시점 이전 카테고리 기록이 없어 대분류·카테고리 순위에서 뺐습니다. "
        "· 첫 4주(5/3~5/30) 방문자는 그 이전 조회가 있었을 수 있습니다."
    )

    root_rank = rank_first_views(fv.dropna(subset=["root_categoryid"]), ["root_categoryid"], total)
    cat_rank = rank_first_views(fv.dropna(subset=["categoryid"]), ["categoryid", "root_categoryid"], total)
    item_rank = rank_first_views(fv, ["itemid"], total)
    # 아이템의 카테고리는 첫 조회 시점 값이 여러 개일 수 있어 가장 많이 나온 값을 표기 (상위 항목만 계산)
    top_items = fv[fv["itemid"].isin(item_rank["itemid"].head(TOP_N))]
    item_cat = (top_items.groupby(["itemid", "categoryid", "root_categoryid"], dropna=False).size()
                .reset_index(name="n").sort_values(["itemid", "n"], ascending=[True, False])
                .drop_duplicates("itemid").set_index("itemid"))
    item_rank["cat_root"] = item_rank["itemid"].map(
        {i: "없음" if pd.isna(r.categoryid) else f"{r.categoryid} ({r.root_categoryid})" for i, r in item_cat.iterrows()})

    # 주요 성과: 첫 조회 상위 10개와 각 항목의 전환율
    st.markdown(f"#### 주요 성과 · 첫 조회 상위 {TOP_N}")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**대분류 기준**")
        top_table(root_rank, {"root_categoryid": "대분류"})
    with c2:
        st.markdown("**카테고리 기준**")
        top_table(cat_rank, {"categoryid": "카테고리", "root_categoryid": "대분류"})
    with c3:
        st.markdown("**아이템 기준**")
        top_table(item_rank, {"itemid": "아이템", "cat_root": "카테고리(대분류)"})
    st.caption("순위는 첫 조회 방문자 수 기준입니다. 장바구니%·구매%는 각 항목의 장바구니·구매 전환율, 30일 재방문%는 30일 재방문율입니다 "
               "(30일째를 관측할 수 없는 항목은 '-'). 아이템의 카테고리는 첫 조회 시점 기준이며, 시점마다 다르면 가장 많이 나온 값을 표기합니다. "
               "'없음'은 조회 시점 이전 카테고리 기록이 없는 경우입니다.")

    st.divider()

    # 드릴다운: 대분류 → 카테고리 → 아이템
    head_l, head_r = st.columns([1, 1], vertical_alignment="bottom")
    head_l.markdown("#### 단계별로 살펴보기")
    metric = head_r.radio("그래프 지표", list(CHART_METRICS), horizontal=True)
    if metric == "30일 재방문율":
        # 30일 재방문율은 0.1%대라 분모가 작으면 0% 또는 극단값이 되므로 기준을 높게 잡는다
        min_n = {"root": 1000, "cat": 1000, "item": 300}
        st.caption(f"30일 재방문율 그래프는 30일째 관측 가능 방문자가 대분류·카테고리 {min_n['cat']:,}명, "
                   f"아이템 {min_n['item']}명 이상인 항목만 재방문율 순으로 보여 줍니다. 괄호는 30일째 관측 가능 방문자 수입니다.")
    else:
        min_n = {"root": 100, "cat": 100, "item": 30}
        if CHART_METRICS[metric][1]:
            st.caption(f"전환율 그래프는 우연한 변동을 줄이기 위해 첫 조회 방문자가 대분류·카테고리 {min_n['cat']}명, "
                       f"아이템 {min_n['item']}명 이상인 항목만 전환율 순으로 보여 줍니다. 괄호는 첫 조회 방문자 수입니다.")

    st.markdown(f"**① 대분류별 {metric}**")
    metric_bar(root_rank, [f"대분류 {r}" for r in root_rank["root_categoryid"]], metric, total, min_n["root"], "root_bar")

    sel_l, sel_r = st.columns(2)
    with sel_l:
        root_sel = st.selectbox("대분류 선택", root_rank["root_categoryid"].tolist(),
                                format_func=lambda r: f"대분류 {r} · {int(root_rank.loc[root_rank['root_categoryid'] == r, 'visitors'].iat[0]):,}명")
    sub_cats = cat_rank[cat_rank["root_categoryid"] == root_sel].reset_index(drop=True)
    with sel_r:
        cat_sel = st.selectbox(f"카테고리 선택 (대분류 {root_sel}의 하위 {len(sub_cats)}개)", sub_cats["categoryid"].tolist(),
                               format_func=lambda c: f"카테고리 {c} · {int(sub_cats.loc[sub_cats['categoryid'] == c, 'visitors'].iat[0]):,}명")

    root_total = int(sub_cats["visitors"].sum())
    st.markdown(f"**② 대분류 {root_sel}의 카테고리별 {metric}** · 대분류 안 {root_total:,}명 (상위 20개 표시)")
    metric_bar(sub_cats, [f"카테고리 {c}" for c in sub_cats["categoryid"]], metric, root_total, min_n["cat"], "cat_bar")

    items = rank_first_views(fv[fv["categoryid"] == cat_sel], ["itemid"], total)
    cat_total = int(items["visitors"].sum())
    st.markdown(f"**③ 카테고리 {cat_sel}의 아이템별 {metric}** · 카테고리 안 아이템 {len(items):,}개, {cat_total:,}명 (상위 20개 표시)")
    metric_bar(items, [f"아이템 {i}" for i in items["itemid"]], metric, cat_total, min_n["item"], "item_bar")
    if not CHART_METRICS[metric][1]:
        st.caption("②·③ 막대의 괄호 비율은 선택한 대분류·카테고리 안의 첫 조회 방문자 대비입니다.")

    detail_cols = {"rank": "순위", "visitors": "첫 조회 방문자", "carted": "장바구니 전환 방문자",
                   "purchased": "구매 전환 방문자", "elig30": "30일째 관측 가능", "ret30": "30일째 재방문", **RATE_COLS}
    with st.expander(f"대분류 {root_sel}의 전체 카테고리 표"):
        heat_table(sub_cats[["categoryid", *detail_cols]].rename(columns={"categoryid": "카테고리", **detail_cols}),
                   label_cols=["카테고리", "순위"])
    with st.expander(f"카테고리 {cat_sel}의 전체 아이템 표"):
        heat_table(items[["itemid", *detail_cols]].rename(columns={"itemid": "아이템", **detail_cols}),
                   label_cols=["아이템", "순위"])
