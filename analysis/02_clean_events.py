"""events.csv 정제 (CLAUDE.md '이번 분석 기준' 적용).

적용 기준
1. 완전 중복 행 제거: timestamp를 포함한 5개 열 값이 모두 같은 행은 첫 행만 유지
2. 분석 기준 시점 적용: event_time < 2015-09-15 21:00:00 UTC 인 행만 유지
3. 자료형 정리: transactionid는 Int64, event_time(UTC datetime) 열 추가. 시간대 변환은 하지 않는다.
   날짜 파생 열 year/month/day/weekday(요일)/day_type(주말·평일, 토·일=주말)을 UTC 날짜 기준으로 추가한다.
첫 4주(UTC 2015-05-03 ~ 2015-05-30)는 제외하지 않는다. 좌측 절단은 분석 단계에서 반영한다.
원본 data/raw/events.csv는 수정하지 않는다.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "events.csv"
OUT = ROOT / "data" / "processed" / "events_clean.csv"
LOG = ROOT / "outputs" / "tables" / "cleaning_log.csv"

CUTOFF = pd.Timestamp("2015-09-15 21:00:00", tz="UTC")

df = pd.read_csv(RAW, dtype={"transactionid": "Int64"})
log = [{"step": "0_raw", "criterion": "원본", "removed_rows": 0, "rows_after": len(df)}]

# 1. 완전 중복 제거
before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
log.append({"step": "1_drop_exact_duplicates", "criterion": "timestamp 포함 5개 열 값이 모두 같은 행 (첫 행 유지)",
            "removed_rows": before - len(df), "rows_after": len(df)})

# 2. 분석 기준 시점 이후 제외
df["event_time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
before = len(df)
df = df[df["event_time"] < CUTOFF].reset_index(drop=True)
log.append({"step": "2_apply_cutoff", "criterion": f"event_time >= {CUTOFF} 행 제외",
            "removed_rows": before - len(df), "rows_after": len(df)})

# 3. 날짜 파생 열 (UTC 기준, timestamp 원값은 유지)
WEEKDAYS = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
df["year"] = df["event_time"].dt.year
df["month"] = df["event_time"].dt.month
df["day"] = df["event_time"].dt.day
dow = df["event_time"].dt.day_of_week
df["weekday"] = dow.map(dict(enumerate(WEEKDAYS)))
df["day_type"] = dow.map(lambda d: "주말" if d >= 5 else "평일")

# 4. 열 정리
df = df[["timestamp", "event_time", "year", "month", "day", "weekday", "day_type",
         "visitorid", "event", "itemid", "transactionid"]]
df = df.sort_values(["visitorid", "timestamp", "event"], kind="stable").reset_index(drop=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
log_df = pd.DataFrame(log)
log_df.to_csv(LOG, index=False, encoding="utf-8-sig")

print(log_df.to_string(index=False))
print("\n[정제 후 이벤트 종류별 건수]")
print(df["event"].value_counts())
print(f"\n기간(UTC): {df['event_time'].min()} ~ {df['event_time'].max()}")
print(f"저장: {OUT.relative_to(ROOT)} ({len(df):,}행)")
print(f"기록: {LOG.relative_to(ROOT)}")
