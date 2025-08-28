import json
# rag/prompts.py（先頭の import 部）
# 旧: from google.generativeai import types
try:
    from google.genai import types  # 新SDK（google-genai）
except ModuleNotFoundError:
    # ローカルで旧SDKしかない場合のフォールバック（任意）
    from google.generativeai import types  # 旧SDK（google-generativeai）


# Converted to a dictionary to be compatible with the current library version
ITINERARY_SCHEMA = {
    "type": "object",
    "required": ["title", "days", "audience", "transport", "sources"],
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "audience": {"type": "string"},
        "transport": {"type": "string"},
        "days": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["day", "theme", "schedule", "notes", "source_urls"],
                "properties": {
                    "day": {"type": "integer"},
                    "theme": {"type": "string"},
                    "area": {"type": "string"},
                    "schedule": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["time", "activity", "spot", "tip"],
                            "properties": {
                                "time": {"type": "string"},
                                "activity": {"type": "string"},
                                "spot": {"type": "string"},
                                "address": {"type": "string"},
                                "url": {"type": "string"},
                                "tip": {"type": "string"},
                            },
                        },
                    },
                    "notes": {"type": "string"},
                    "source_urls": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["title", "url", "site"],
                "properties": {
                    "title": {"type": "string"},
                    "url": {"type": "string"},
                    "site": {"type": "string"},
                },
            },
        },
    },
}

SYSTEM_GUARDRAILS = (
    "あなたは愛媛旅行の日本語プランナー。いよ観ネットの要約から**パラフレーズ**で情報を統合する。\n"
    "禁止: 原文の長い引用・転載、無根拠の情報。\n"
    "必須: 各日ごとに根拠URLを列挙し、家族/高齢者/子連れなど安全面と季節性を明示。\n"
)

TEMPLATE = '''{system}
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
'''

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

def build_refine_plan_prompt(existing_plan: dict, user_request: str) -> str:
    """
    既存のプランとユーザーの修正依頼から、プラン修正用のプロンプトを生成する。
    """
    plan_str = json.dumps(existing_plan, indent=2, ensure_ascii=False)
    
    return f'''
あなたは優秀な旅行プランナーです。

以下の既存の旅行プランが提示されています。
ユーザーからの修正依頼を基に、このプランを更新してください。

制約:
- 必ず `ITINERARY_SCHEMA` に準拠したJSON形式で出力してください。
- 元のプランの構造を維持し、必要な箇所だけを修正してください。
- 修正が難しい場合でも、何らかの形で依頼に応えようと試みてください。

# 既存の旅行プラン (JSON)
```json
{plan_str}
```

# ユーザーからの修正依頼
{user_request}

# 修正後の旅行プラン (JSON)
'''
