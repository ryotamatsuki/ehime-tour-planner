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

TEMPLATE = (
    "{system}n"
    "【旅行条件】日数={trip_days}日, 開始日={start_date}, 同行者={party}, 移動={transport}, 関心={interests}, エリア={start_area}, 子連れ配慮={with_kids}, ペース={pace}nn"
    "【参考要点】n{context}nn"
    "上記を踏まえて、現実的な**日次の時間割**(午前/午後/夜, 所要時間の目安や注意点) を作成しなさい。n"
    "各日の区切りは JSON スキーマに従うこと。n"
    "**交通の現実性**(松山空港/松山駅/今治/大洲/内子/宇和島/西条 などの移動時間の感覚)を反映。n"
    "地名や施設は日本語で、URL は可能な限り いよ観ネット のものを優先。n"
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
    sources: list[dict],
    context: list[str],
) -> str:
    ctx = "nn".join(context)
    system = SYSTEM_GUARDRAILS
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
        context=ctx,
    )