from __future__ import annotations


SYSTEM = """あなたは愛媛旅行の日本語プランナーです。
候補リストにある spot_id だけを使って旅程を作成してください。
URL、住所、出典文字列は生成しません。施設の事実は候補抜粋だけを根拠にしてください。
移動時間を考慮し、同日に離れすぎた地域を不自然に詰め込まないでください。
出力は指定JSON Schemaに厳密に従います。"""


def _conditions(
    *,
    trip_days: int,
    start_date: str,
    party: str,
    transport: str,
    interests: list[str],
    start_area: str,
    with_kids: bool,
    pace: str,
    start_end_point: str,
) -> str:
    return "\n".join(
        [
            f"日数={trip_days}",
            f"開始日={start_date}",
            f"同行者={party}",
            f"移動手段={transport}",
            f"関心={','.join(interests) if interests else '指定なし'}",
            f"主なエリア={start_area or '指定なし'}",
            f"子連れ配慮={'必要' if with_kids else '指定なし'}",
            f"ペース={pace}",
            f"発着地={start_end_point or '指定なし'}",
        ]
    )


def build_spot_plan_prompt(
    *,
    trip_days: int,
    start_date: str,
    party: str,
    transport: str,
    interests: list[str],
    start_area: str,
    with_kids: bool,
    pace: str,
    start_end_point: str,
    candidate_context: list[str],
) -> str:
    catalog = "\n".join(candidate_context)
    conditions = _conditions(
        trip_days=trip_days,
        start_date=start_date,
        party=party,
        transport=transport,
        interests=interests,
        start_area=start_area,
        with_kids=with_kids,
        pace=pace,
        start_end_point=start_end_point,
    )
    return f"""{SYSTEM}

【旅行条件】
{conditions}

【候補】
{catalog}

【ルール】
- day は1から{trip_days}まで欠番なく作る。
- 各日は1〜2件。spot_id は候補のIDだけを使う。
- 同じ日に同じspot_idを重複させない。
- time は原則 HH:MM-HH:MM。
- activity と tip は短く、候補抜粋にない料金・営業時間等を作らない。
- 発着地が指定されている場合、初日と最終日の動線を考慮する。
- CompactDayBundle JSONだけを返す。URL、住所、Markdownは出力しない。
"""


def build_spot_segment_prompt(
    *,
    start_day: int,
    end_day: int,
    trip_days: int,
    start_date: str,
    party: str,
    transport: str,
    interests: list[str],
    start_area: str,
    with_kids: bool,
    pace: str,
    start_end_point: str,
    candidate_context: list[str],
    previous_spot_id: str | None,
) -> str:
    catalog = "\n".join(candidate_context)
    conditions = _conditions(
        trip_days=trip_days,
        start_date=start_date,
        party=party,
        transport=transport,
        interests=interests,
        start_area=start_area,
        with_kids=with_kids,
        pace=pace,
        start_end_point=start_end_point,
    )
    return f"""{SYSTEM}

【旅行条件】
{conditions}

【候補】
{catalog}

今回は Day {start_day}〜Day {end_day} だけを作成する。
直前区間の最後のspot_id={previous_spot_id or 'なし'}

【ルール】
- daysには day={start_day} から day={end_day} だけを欠番なく入れる。
- 各日は1〜2件。spot_id は候補のIDだけを使う。
- 直前spotがある場合は自然な移動順にする。
- Day {trip_days} を含む最終区間では発着地への戻りを考慮する。
- time は原則 HH:MM-HH:MM。activity と tip は短くする。
- CompactDayBundle JSONだけを返す。URL、住所、Markdownは出力しない。
"""
