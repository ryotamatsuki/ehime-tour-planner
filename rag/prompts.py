from __future__ import annotations

import json


SYSTEM_GUARDRAILS = """あなたは愛媛旅行の日本語プランナーです。
与えられた検索コンテキストだけを根拠として旅程を作成してください。
原文の長い引用は禁止し、必ず要約・言い換えを行ってください。
営業時間・料金・休館日など最新性が重要な情報は、根拠にない場合は断定しないでください。
存在しないURLを作らないでください。
移動時間を考慮し、同日に離れすぎた地域を不自然に詰め込まないでください。
出力は指定JSON Schemaに厳密に従います。"""


def build_plan_prompt(
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
    context: list[str],
) -> str:
    ctx = "\n\n".join(context)
    start_end = start_end_point if start_end_point and start_end_point != "指定なし" else "指定なし"
    return f"""{SYSTEM_GUARDRAILS}

【旅行条件】
- 日数: {trip_days}日
- 開始日: {start_date}
- 同行者: {party}
- 移動手段: {transport}
- 関心テーマ: {", ".join(interests) if interests else "指定なし"}
- 主な訪問エリア: {start_area or "指定なし"}
- 子連れ配慮: {"必要" if with_kids else "指定なし"}
- ペース: {pace}
- 発着地: {start_end}

【検索コンテキスト】
{ctx}

【作成ルール】
1. day は1から{trip_days}まで欠番なく作成する。
2. 1日目は発着地が指定されていればそこから始め、最終日はそこへ戻る。
3. 各日2〜3件程度を目安に、移動時間を含めて無理のない順序にする。
4. schedule.time は原則 HH:MM-HH:MM 形式にする。
5. source_urls と schedule.url は上記コンテキストに明示されたURLだけを使う。
6. 根拠にない詳細は推測で補わず、tip に「要確認」と明示する。
7. 省スペースJSONにする。各文字列は簡潔にし、activity/spot/address/tip/notes は各50文字以内にする。
8. title/summary/audience/transport/sources を含む完全な旅程JSONを返す。
"""


def build_segment_prompt(
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
    previous_day: dict | None,
    context: list[str],
) -> str:
    prev = json.dumps(previous_day, ensure_ascii=False, indent=2) if previous_day else "なし"
    ctx = "\n\n".join(context)
    return f"""{SYSTEM_GUARDRAILS}

長期旅程を分割して作成します。今回は Day {start_day}〜Day {end_day} だけを作成してください。

【旅行条件】
- 全体日数: {trip_days}日
- 開始日: {start_date}
- 同行者: {party}
- 移動手段: {transport}
- 関心テーマ: {", ".join(interests) if interests else "指定なし"}
- 主な訪問エリア: {start_area or "指定なし"}
- 子連れ配慮: {"必要" if with_kids else "指定なし"}
- ペース: {pace}
- 発着地: {start_end_point or "指定なし"}

【直前の最終日】
{prev}

【検索コンテキスト】
{ctx}

【作成ルール】
1. days には day={start_day} から day={end_day} までだけを欠番なく入れる。
2. 直前の最終日がある場合は、その終点から自然につながるようにする。
3. 最終区間に Day {trip_days} が含まれ、発着地が指定されている場合は最終的にそこへ戻る。
4. 各日2〜3件程度。schedule.time は原則 HH:MM-HH:MM。各文字列は50文字以内。
5. URLは検索コンテキストに明示されたものだけを使う。
6. DayBundle JSONだけを返し、説明文やMarkdownコードフェンスを付けない。
"""


def build_refine_patch_prompt(
    *,
    existing_plan: dict,
    user_request: str,
    context: list[str],
) -> str:
    plan_str = json.dumps(existing_plan, ensure_ascii=False, indent=2)
    ctx = "\n\n".join(context)
    return f"""{SYSTEM_GUARDRAILS}

既存旅程に対するユーザーの修正依頼です。
旅程全体を再生成せず、変更が必要な日だけを PlanPatch として返してください。
たとえば「2日目をゆったり」に対しては days に day=2 だけを返します。
全日程に影響する依頼なら必要な全日を返して構いません。

【既存旅程】
{plan_str}

【追加の検索コンテキスト】
{ctx}

【修正依頼】
{user_request}

【ルール】
- 変更しない日は days に含めない。
- title/summary を変えない場合は null。
- dayの値は既存旅程のday番号を維持する。
- URLは既存旅程または検索コンテキストに存在するものだけを使う。
- PlanPatch JSONだけを返す。
"""


def build_repair_prompt(*, invalid_plan: dict, violations: list[str], context: list[str]) -> str:
    return f"""{SYSTEM_GUARDRAILS}

次の旅程JSONには検証エラーがあります。内容を最小限修正し、完全な旅程JSONを返してください。

【検証エラー】
{chr(10).join("- " + v for v in violations)}

【現在の旅程】
{json.dumps(invalid_plan, ensure_ascii=False, indent=2)}

【利用可能な検索コンテキスト】
{chr(10).join(context)}

URLを新しく作らず、日数・day番号・空のschedule・根拠URLの問題を優先して直してください。
"""
