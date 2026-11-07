"""주요 지표 (이벤트 테이블 events_clean.csv, 방문자 테이블 visitor_summary.csv 기준).

지표 정의
1. 활성 사용자 수: 이벤트 종류와 무관하게(view/addtocart/transaction) 1회 이상 수행한 고유 visitorid 수
2. 구매자 수: transaction을 1회 이상 수행한 고유 visitorid 수
3. 구매 횟수: 고유 transactionid 수 (방문자 테이블 purchase_count와 같은 기준)
4. 장바구니 담기 횟수: addtocart 이벤트 행 수
5. 이벤트 횟수: 전체 이벤트 행 수 (view + addtocart + transaction)
두 테이블에서 각각 계산해 값이 일치하는지 확인한다.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "processed" / "events_clean.csv"
VISITORS = ROOT / "data" / "processed" / "visitor_summary.csv"
OUT = ROOT / "outputs" / "tables" / "key_metrics.csv"

ev = pd.read_csv(EVENTS, usecols=["visitorid", "event", "transactionid"], dtype={"transactionid": "Int64"})
vs = pd.read_csv(VISITORS)

from_events = {
    "활성 사용자 수": ev["visitorid"].nunique(),
    "구매자 수": ev.loc[ev["event"] == "transaction", "visitorid"].nunique(),
    "구매 횟수": ev["transactionid"].nunique(),
    "장바구니 담기 횟수": int((ev["event"] == "addtocart").sum()),
    "이벤트 횟수": len(ev),
}
from_visitors = {
    "활성 사용자 수": vs["visitorid"].nunique(),
    "구매자 수": int((vs["purchase_count"] > 0).sum()),
    "구매 횟수": int(vs["purchase_count"].sum()),
    "장바구니 담기 횟수": int(vs["addtocart_count"].sum()),
    # 방문자 테이블에는 transaction 행 수가 없어 이벤트 횟수는 이벤트 테이블에서만 계산
    "이벤트 횟수": pd.NA,
}
definitions = {
    "활성 사용자 수": "이벤트(view/addtocart/transaction) 1회 이상 고유 방문자",
    "구매자 수": "transaction 1회 이상 고유 방문자",
    "구매 횟수": "고유 transactionid 수",
    "장바구니 담기 횟수": "addtocart 이벤트 행 수",
    "이벤트 횟수": "전체 이벤트 행 수",
}

result = pd.DataFrame({
    "metric": list(from_events),
    "value": list(from_events.values()),
    "definition": [definitions[k] for k in from_events],
    "value_from_visitor_table": pd.array([from_visitors[k] for k in from_events], dtype="Int64"),
})
match = (result["value"] == result["value_from_visitor_table"]).fillna(True)
assert match.all(), result[~match]

OUT.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUT, index=False, encoding="utf-8-sig")
print(result.to_string(index=False))
print(f"\n참고 - transaction 이벤트 행 수(구매 상품 행): {(ev['event'] == 'transaction').sum():,}")
print(f"참고 - view 1회 이상 고유 방문자: {ev.loc[ev['event'] == 'view', 'visitorid'].nunique():,}")
print(f"저장: {OUT.relative_to(ROOT)}")
