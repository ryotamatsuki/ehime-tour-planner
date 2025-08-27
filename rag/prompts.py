from google.genai import types

ITINERARY_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    required=["title", "days", "audience", "transport", "sources"],
    properties={
        "title": types.Schema(type=types.Type.STRING),
        "summary": types.Schema(type=types.Type.STRING),
        "audience": types.Schema(type=types.Type.STRING),
        "transport": types.Schema(type=types.Type.STRING),
        "days": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                required=["day", "theme", "schedule", "notes", "source_urls"],
                properties={
                    "day": types.Schema(type=types.Type.INTEGER),
                    "theme": types.Schema(type=types.Type.STRING),
                    "area": types.Schema(type=types.Type.STRING),
                    "schedule": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(
                            type=types.Type.OBJECT,
                            required=["time", "activity", "spot", "tip"],
                            properties={
                                "time": types.Schema(type=types.Type.STRING),
                                "activity": types.Schema(type=types.Type.STRING),
                                "spot": types.Schema(type=types.Type.STRING),
                                "address": types.Schema(type=types.Type.STRING),
                                "url": types.Schema(type=types.Type.STRING),
                                "tip": types.Schema(type=types.Type.STRING),
                            },
                        ),
                    ),
                    "notes": types.Schema(type=types.Type.STRING),
                    "source_urls": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.STRING),
                    ),
                },
            ),
        ),
        "sources": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                required=["title", "url", "site"],
                properties={
                    "title": types.Schema(type=types.Type.STRING),
                    "url": types.Schema(type=types.Type.STRING),
                    "site": types.Schema(type=types.Type.STRING),
                },
            ),
        ),
    },
)

SYSTEM_GUARDRAILS = (
    "あなたは愛媛旅行の日本語プランナー。いよ観ネットの要約から**パラフレーズ**で情報を統合する。n"
    "禁止: 原文の長い引用・転載、無根拠の情報。n"
    "必須: 各日ごとに根拠URLを列挙し、家族/高齢者/子連れなど安全面と季節性を明示。n"
)

TEMPLATE = """{system}
【旅行条件】
- 日数: {trip_days}日
- 開始日: {start_date}
- 同行者: {party}
- 移動手段: {transport}
- 関心テーマ: {interests}
- 主な訪問エリア: {start_area}
- 子連れ配慮: {with_kids}
- ペース: {pace}
- 発着地: {start_end_point}

【参考要点】
{context}

【指示】
上記条件に基づき、以下の点を厳守して現実的な旅行プランをJSON形式で作成してください。

1.  **行程の起点と終点:** 1日目は「{start_end_point}」から出発し、最終日({trip_days}日目)は「{start_end_point}」に到着して解散する行程とします。
2.  **宿泊地の最適化:** **毎日、発着地に戻る必要はありません。** 各日の宿泊地は、その日の観光エリアや翌日の移動を考慮して、最も効率的で現実的な場所（例: 松山市内、道後温泉、今治市、宇和島市など）を設定してください。
3.  **時間配分:** 各アクティビティの所要時間と、エリア間の移動時間を考慮した、現実的な時間割を作成してください。
4.  **出力形式:** 必ず指定されたJSONスキーマに従ってください。
"""

def build_plan_prompt(
    trip_days: int,
    start_date: str,
    party: str,
    transport: str,
    interests: list[str],
    start_area: str,
    with_kids: bool,
    pace: str,
    start_end_point: str,
    sources: list[dict],
    context: list[str],
) -> str:
    ctx = "\n\n".join(context)
    system = SYSTEM_GUARDRAILS
    
    start_end_prompt_val = start_end_point if start_end_point and start_end_point != "指定なし" else "指定なし"

    return TEMPLATE.format(
        system=system,
        trip_days=trip_days,
        start_date=start_date,
        party=party,
        transport=transport,
        interests=",".join(interests),
        start_area=start_area,
        with_kids=with_kids,
        pace=pace,
        start_end_point=start_end_prompt_val,
        context=ctx,
    )


def build_plan_prompt(
    trip_days: int,
    start_date: str,
    party: str,
    transport: str,
    interests: list[str],
    start_area: str,
    with_kids: bool,
    pace: str,
    start_end_point: str,
    sources: list[dict],
    context: list[str],
) -> str:
    ctx = "\n\n".join(context)
    system = SYSTEM_GUARDRAILS
    
    start_end_prompt_val = start_end_point if start_end_point and start_end_point != "指定なし" else "指定なし"

    return TEMPLATE.format(
        system=system,
        trip_days=trip_days,
        start_date=start_date,
        party=party,
        transport=transport,
        interests=",".join(interests),
        start_area=start_area,
        with_kids=with_kids,
        pace=pace,
        start_end_point=start_end_prompt_val,
        context=ctx,
    )
