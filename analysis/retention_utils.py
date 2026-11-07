"""N-day 리텐션 계산 (07_dashboard.py의 카테고리·리텐션 탭이 사용, 12·13번 스크립트와 같은 정의).

정의 (방문자 단위, UTC 날짜 기준)
- Day 0: 코호트 시작 행동을 처음 한 UTC 날짜. 카테고리는 그 첫 행동 행(같은 시각이면 itemid가 작은 것)의
  상품 카테고리로 정한다. 한 구매에 상품이 여러 개면 itemid가 가장 작은 상품의 카테고리를 쓴다.
- Day N 복귀: Day 0으로부터 정확히 N일째 UTC 날짜에 복귀 이벤트가 1건 이상 있는 경우 (누적 아님).
  복귀 이벤트는 카테고리와 무관하게 센다.
- 분모: Day 0 + N <= 2015-09-14 (2015-09-15는 21:00까지만 관측되므로 마지막 완전 관측일은 9/14)
- 좌측 절단: Day 0이 첫 4주(2015-05-03 ~ 2015-05-30)이면 그 이전 행동이 관측되지 않아 '첫 행동'으로 단정할 수 없다.
"""
import numpy as np
import pandas as pd

DAY_MS = 86_400_000
LAST_FULL_DAY = pd.Timestamp("2015-09-14").value // 10**6 // DAY_MS
EARLY_END = pd.Timestamp("2015-05-30").value // 10**6 // DAY_MS


def to_day(date: pd.Timestamp) -> int:
    """UTC 날짜 → 날짜 번호 (epoch 일수)."""
    return pd.Timestamp(date).value // 10**6 // DAY_MS


def first_action(ev: pd.DataFrame, event: str | None = None) -> pd.DataFrame:
    """방문자별 첫 행동 1건 (event가 None이면 모든 이벤트). index=visitorid, 열: day0, root_key, cat_key (-1 = 카테고리 없음)."""
    e = ev if event is None else ev[ev["event"] == event]
    first = (e.sort_values(["visitorid", "timestamp", "itemid"]).drop_duplicates("visitorid")
             .set_index("visitorid"))
    return pd.DataFrame({
        "day0": first["timestamp"] // DAY_MS,
        "root_key": first["root_categoryid"].fillna(-1).astype(int),
        "cat_key": first["categoryid"].fillna(-1).astype(int),
    })


def active_days(ev: pd.DataFrame, event: str | None = None) -> pd.DataFrame:
    """복귀 이벤트가 있는 (visitorid, day) 쌍. 마지막 완전 관측일 이후는 뺀다."""
    e = ev if event is None else ev[ev["event"] == event]
    d = pd.DataFrame({"visitorid": e["visitorid"].to_numpy(), "day": (e["timestamp"] // DAY_MS).to_numpy()})
    return d[d["day"] <= LAST_FULL_DAY].drop_duplicates()


def return_offsets(cohort: pd.DataFrame, days: pd.DataFrame) -> pd.DataFrame:
    """코호트 방문자의 복귀 (visitorid, n), n >= 1."""
    r = days.join(cohort["day0"], on="visitorid", how="inner")
    r["n"] = r["day"] - r["day0"]
    return r.loc[r["n"] >= 1, ["visitorid", "n"]].reset_index(drop=True)


def eligible_counts(day0: pd.Series, ns, last: int = LAST_FULL_DAY) -> np.ndarray:
    """각 N에 대해 Day N을 관측할 수 있는 방문자 수 (day0 <= last - N)."""
    s = np.sort(day0.to_numpy())
    return np.searchsorted(s, last - np.asarray(ns), side="right")


def curve(cohort: pd.DataFrame, returns: pd.DataFrame, n_max: int, last: int = LAST_FULL_DAY) -> pd.DataFrame:
    """N = 1..n_max 리텐션 곡선. 열: n_day, eligible, returned, rate(%)."""
    ns = np.arange(1, n_max + 1)
    r = returns[returns["visitorid"].isin(cohort.index) & (returns["n"] <= n_max)]
    returned = r["n"].value_counts().reindex(ns, fill_value=0).to_numpy()
    eligible = eligible_counts(cohort["day0"], ns, last)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(eligible > 0, returned / np.maximum(eligible, 1) * 100, np.nan)
    return pd.DataFrame({"n_day": ns, "eligible": eligible, "returned": returned, "rate": rate})


def group_rates(cohort: pd.DataFrame, returns: pd.DataFrame, key: str, ns=(1, 7, 14, 30),
                last: int = LAST_FULL_DAY) -> pd.DataFrame:
    """key별 Day 0 방문자 수와 Day N 리텐션(%)·분모. 열: key, cohort, d{N}, elig{N}."""
    out = cohort.groupby(key).size().rename("cohort").to_frame()
    r = returns[returns["n"].isin(ns)].join(cohort[key], on="visitorid", how="inner")
    ret = r.groupby([key, "n"]).size().unstack(fill_value=0)
    for n in ns:
        elig = cohort[cohort["day0"] <= last - n].groupby(key).size()
        out[f"elig{n}"] = elig.reindex(out.index, fill_value=0)
        got = ret[n].reindex(out.index, fill_value=0) if n in ret.columns else 0
        out[f"d{n}"] = (got / out[f"elig{n}"].where(out[f"elig{n}"] > 0)) * 100
    return out.reset_index().sort_values("cohort", ascending=False, ignore_index=True)


# ── 일·주·월 단위 ──────────────────
# 주는 월요일 시작 UTC 주, 월은 UTC 달력 월. 단위를 바꾸면 Day 0·N도 그 단위가 된다:
# Day 0 = 첫 행동이 속한 기간, N기간째 복귀 = 그 기간으로부터 정확히 N번째 기간에 복귀 이벤트가 1건 이상.
# 분모는 N번째 기간 전체가 관측된(마지막 날이 2015-09-14 이내) 방문자.
UNIT_NAMES = {"day": "일", "week": "주", "month": "월"}


def period_index(days, unit: str) -> np.ndarray:
    """날짜 번호(epoch 일수) → 기간 번호."""
    days = np.asarray(days, dtype="int64")
    if unit == "day":
        return days
    if unit == "week":
        return (days + 3) // 7  # 1970-01-01은 목요일이므로 +3 하면 월요일 시작 주
    d = pd.DatetimeIndex(days * DAY_MS * 10**6)
    return (d.year * 12 + d.month - 1).to_numpy().astype("int64")


def period_bounds(p, unit: str) -> tuple[np.ndarray, np.ndarray]:
    """기간 번호 → (첫날, 마지막날) 날짜 번호."""
    p = np.asarray(p, dtype="int64")
    if unit == "day":
        return p, p
    if unit == "week":
        return p * 7 - 3, p * 7 + 3
    first = pd.PeriodIndex.from_fields(year=p // 12, month=p % 12 + 1, freq="M")
    to_num = lambda ts: ((ts - pd.Timestamp(0)) // pd.Timedelta(days=1)).to_numpy().astype("int64")  # noqa: E731
    return to_num(first.start_time), to_num(first.end_time.normalize())


def period_label(p, unit: str) -> list[str]:
    start, _ = period_bounds(p, unit)
    d = pd.DatetimeIndex(start * DAY_MS * 10**6)
    return list(d.strftime({"day": "%m-%d", "week": "%m-%d 주", "month": "%Y-%m"}[unit]))


def last_full_period(unit: str) -> int:
    """끝까지 관측된 마지막 기간 번호."""
    p = period_index([LAST_FULL_DAY, LAST_FULL_DAY + 1], unit)
    return int(p[0] if p[1] > p[0] else p[0] - 1)


def to_unit(cohort: pd.DataFrame, returns: pd.DataFrame, unit: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """일 단위 코호트·복귀를 unit 단위로 바꾼다. 코호트의 day0은 기간 번호가 되고, 원래 날짜는 first_day에 남긴다.
    복귀 n은 기간 차이(>= 1)이며, 끝까지 관측되지 않은 기간의 복귀는 뺀다."""
    co = cohort.assign(first_day=cohort["day0"])
    if unit == "day":
        return co, returns
    co["day0"] = period_index(co["first_day"], unit)
    d0 = cohort["day0"].reindex(returns["visitorid"]).to_numpy()
    p0 = co["day0"].reindex(returns["visitorid"]).to_numpy()
    pr = period_index(d0 + returns["n"].to_numpy(), unit)
    r = pd.DataFrame({"visitorid": returns["visitorid"].to_numpy(), "n": pr - p0})
    r = r[(r["n"] >= 1) & (pr <= last_full_period(unit))].drop_duplicates(ignore_index=True)
    return co, r


def cohort_matrix(cohort: pd.DataFrame, returns: pd.DataFrame, n_max: int,
                  last: int = LAST_FULL_DAY) -> tuple[pd.DataFrame, pd.Series]:
    """코호트(day0)별 N = 0..n_max 복귀 방문자 수. N = 0은 코호트 크기, 관측할 수 없는 칸은 NaN."""
    size = cohort.groupby("day0").size()
    r = returns[returns["n"] <= n_max].join(cohort["day0"], on="visitorid", how="inner")
    cnt = (r.groupby(["day0", "n"]).size().unstack(fill_value=0)
           .reindex(index=size.index, columns=range(n_max + 1), fill_value=0))
    cnt[0] = size
    observable = size.index.to_numpy()[:, None] + np.arange(n_max + 1)[None, :] <= last
    return cnt.where(observable), size
