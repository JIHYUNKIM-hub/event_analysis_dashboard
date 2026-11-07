"""첫 조회 기준 시간 제한 퍼널 계산 (11_category_funnel.py와 07_dashboard.py가 함께 사용).

정의 (방문자 단위, 순차형, 상품 일치 무관)
- View: 방문자의 첫 view 시각 t0. 카테고리는 그 첫 조회 상품의 카테고리(진입 카테고리)로 정한다.
  같은 시각 view가 여러 건이면 itemid가 작은 것 (09_first_view_by_category.py와 같음)
- AddToCart: t0 이후(같은 시각 포함) 제한 시간 안의 첫 addtocart 시각 t1
- Transaction: t1 이후(같은 시각 포함)이면서 t0로부터 제한 시간 안의 첫 transaction 시각 t2
- 코호트: 제한 시간을 모두 관측할 수 있도록 t0 <= 기준 시점 - 제한 시간 인 방문자만 쓴다.
- 이전 단계 수행 횟수: AddToCart 도달자는 t0~t1 사이 view 수, Transaction 도달자는 t0~t2 사이 addtocart 수
"""
import numpy as np
import pandas as pd

CUTOFF_MS = int(pd.Timestamp("2015-09-15 21:00:00", tz="UTC").timestamp() * 1000)
WINDOW_24H_MS = 24 * 3_600_000

TIME_BINS_MIN = [0, 1, 5, 15, 30, 60, 180, 360, 720, 1440 + 1e-6]  # 마지막 구간은 정확히 24시간 포함
TIME_LABELS = ["1분 미만", "1~5분", "5~15분", "15~30분", "30분~1시간", "1~3시간", "3~6시간", "6~12시간", "12~24시간"]
COUNT_BINS = [0, 1, 2, 3, 4, 5, 10, 20, np.inf]
COUNT_LABELS = ["1회", "2회", "3회", "4회", "5회", "6~10회", "11~20회", "21회 이상"]


def first_view_funnel(ev: pd.DataFrame, window_ms: int = WINDOW_24H_MS, cutoff_ms: int = CUTOFF_MS) -> pd.DataFrame:
    """ev: timestamp, visitorid, event, itemid, categoryid, root_categoryid 열을 가진 이벤트 표."""
    views = ev[ev["event"] == "view"]
    first = (views.sort_values(["visitorid", "timestamp", "itemid"])
             .drop_duplicates("visitorid", keep="first")
             .set_index("visitorid")[["timestamp", "itemid", "categoryid", "root_categoryid"]]
             .rename(columns={"timestamp": "t0"}))
    first = first[first["t0"] <= cutoff_ms - window_ms]

    def within(event: str, start_col: str, frame: pd.DataFrame) -> pd.DataFrame:
        e = ev.loc[ev["event"] == event, ["visitorid", "timestamp"]].merge(
            frame[list(dict.fromkeys(["t0", start_col]))].dropna(), left_on="visitorid", right_index=True)
        return e[(e["timestamp"] >= e[start_col]) & (e["timestamp"] - e["t0"] <= window_ms)]

    first["t1"] = within("addtocart", "t0", first).groupby("visitorid")["timestamp"].min()
    first["t2"] = within("transaction", "t1", first).groupby("visitorid")["timestamp"].min()
    first["carted"] = first["t1"].notna()
    first["purchased"] = first["t2"].notna()
    first["cart_minutes"] = (first["t1"] - first["t0"]) / 60_000
    first["tx_minutes"] = (first["t2"] - first["t1"]) / 60_000

    def count_between(event: str, end_col: str) -> pd.Series:
        e = ev.loc[ev["event"] == event, ["visitorid", "timestamp"]].merge(
            first[["t0", end_col]].dropna(), left_on="visitorid", right_index=True)
        return e[(e["timestamp"] >= e["t0"]) & (e["timestamp"] <= e[end_col])].groupby("visitorid").size()

    first["n_views_before_cart"] = count_between("view", "t1")
    first["n_carts_before_tx"] = count_between("addtocart", "t2")
    return first


def funnel_counts(fv: pd.DataFrame) -> dict:
    n0, n1, n2 = len(fv), int(fv["carted"].sum()), int(fv["purchased"].sum())
    return {
        "view": n0, "addtocart": n1, "transaction": n2,
        "view_to_cart_pct": n1 / n0 * 100 if n0 else np.nan,
        "cart_to_tx_pct": n2 / n1 * 100 if n1 else np.nan,
        "view_to_tx_pct": n2 / n0 * 100 if n0 else np.nan,
    }


def group_funnel(fv: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    g = fv.groupby(keys, dropna=False)
    out = g.agg(view=("t0", "size"), addtocart=("carted", "sum"), transaction=("purchased", "sum"),
                cart_minutes_median=("cart_minutes", "median"), tx_minutes_median=("tx_minutes", "median"),
                views_before_cart_median=("n_views_before_cart", "median"),
                carts_before_tx_median=("n_carts_before_tx", "median")).reset_index()
    out["view_to_cart_pct"] = out["addtocart"] / out["view"] * 100
    out["cart_to_tx_pct"] = (out["transaction"] / out["addtocart"].where(out["addtocart"] > 0)) * 100
    out["view_to_tx_pct"] = out["transaction"] / out["view"] * 100
    return out.sort_values("view", ascending=False, ignore_index=True)


def time_distribution(minutes: pd.Series) -> pd.Series:
    cut = pd.cut(minutes.dropna(), bins=TIME_BINS_MIN, labels=TIME_LABELS, right=False, include_lowest=True)
    return cut.value_counts().reindex(TIME_LABELS, fill_value=0)


def count_distribution(counts: pd.Series) -> pd.Series:
    cut = pd.cut(counts.dropna(), bins=COUNT_BINS, labels=COUNT_LABELS, right=True)
    return cut.value_counts().reindex(COUNT_LABELS, fill_value=0)
