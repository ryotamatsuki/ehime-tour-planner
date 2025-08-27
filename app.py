import os
import json
import time
from datetime import date, datetime
from dateutil.relativedelta import relativedelta

import streamlit as st
import pandas as pd

from google import genai
from google.genai import types

from rag.retriever import EhimeRetriever, RetrievalItem
from rag.prompts import build_plan_prompt, ITINERARY_SCHEMA
from utils.formatting import plan_json_to_markdown

st.set_page_config(
    page_title="Ehime Tour Planner — RAG × Tavily × Gemini",
    layout="wide",
)

st.title("Ehime Tour Planner (愛媛RAGプランナー)")
st.caption("Tavily検索 + Gemini 2.5 Flash で いよ観ネットを参照しながら旅程を自動作成。出典URLを明示し、原文転載は行いません。")

# --- Secrets / Clients ---
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY", os.getenv("TAVILY_API_KEY"))

if not GEMINI_API_KEY or not TAVILY_API_KEY:
    st.error("GEMINI_API_KEY と TAVILY_API_KEY を Secrets に設定してください")
    st.stop()

# Gemini Client
client = genai.Client(api_key=GEMINI_API_KEY)
retriever = EhimeRetriever(api_key=TAVILY_API_KEY)

# --- Sidebar: 条件入力 ---
st.sidebar.header("プラン条件")
with st.sidebar:
    trip_days = st.number_input("旅行日数（日）", 1, 14, 2)
    start_date = st.date_input("開始日 (任意)", value=date.today())
    party = st.text_input("同行者（例: 大人2・小学生1）", "大人2")
    transport = st.selectbox("移動手段", ["公共交通", "自家用車", "レンタカー", "自転車"], index=0)
    
    st.divider()
    st.markdown("##### 発着地 (任意)")
    start_end_options = ["指定なし", "松山空港", "JR松山駅", "松山市駅", "松山観光港", "その他（自由記述）"]
    start_end_choice = st.selectbox("場所を選択", start_end_options, index=0)
    
    start_end_point = ""
    if start_end_choice == "その他（自由記述）":
        start_end_point = st.text_input("自由記述欄", placeholder="例: 今治港、自宅など")
    else:
        start_end_point = start_end_choice
    st.divider()

    interests = st.multiselect(
        "関心テーマ",
        ["温泉", "城・歴史", "サイクリング", "自然景観", "島めぐり", "グルメ", "アート", "祭り・イベント", "体験・アクティビティ"],
        default=["温泉", "城・歴史"],
    )
    area_options = ["指定なし", "中予(松山・道後)", "東予(今治・西条など)", "南予(大洲・内子・宇和島など)", "その他（自由記述）"]
    area_choice = st.selectbox("主な訪問エリア", area_options, index=0)
    
    start_area = ""
    if area_choice == "その他（自由記述）":
        start_area = st.text_input("エリアを自由に入力", placeholder="例: 愛南町、鬼北町")
    else:
        start_area = area_choice
    with_kids = st.checkbox("子連れ考慮")
    pace = st.select_slider("1日の詰め込み度", options=["ゆったり", "標準", "ぎっしり"], value="標準")
    generate_btn = st.button("プラン生成", type="primary")

# --- Main: 検索 + 生成 ---
colL, colR = st.columns([0.55, 0.45])

with colL:
    st.subheader("1) 関連ソース検索")
    add_web_search = st.checkbox("ウェブ検索の結果も追加する", value=False, help="「いよ観ネット」に加えて、Web全体からも関連情報を検索します。")
    q_default = "愛媛 観光 モデルコース 道後温泉 松山城"
    query = st.text_input("検索キーワード（必要に応じて編集）", q_default)
    max_results = st.slider("最大取得サイト数", 3, 15, 8)
    if st.button("関連ページを収集"):
        with st.spinner("Tavilyで検索・要約中..."):
            items = retriever.search_and_prepare(
                query=query, 
                max_results=max_results,
                add_web_search=add_web_search
            )
        st.session_state["items"] = [i.model_dump() for i in items]
        st.success(f"{len(items)} 件の候補を取り込みました。右ペインで内容を確認できます。")

with colR:
    st.subheader("候補リスト（出典URL明示）")
    items_state = st.session_state.get("items", [])
    if items_state:
        df = pd.DataFrame(items_state)[["title", "url", "site", "content_chars"]]
        df.rename(columns={"content_chars": "要約文字数(概算)"}, inplace=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("左側で『関連ページを収集』を実行すると候補が表示されます。")

st.divider()

# --- 旅程生成 ---
if generate_btn:
    items_state = st.session_state.get("items", [])
    if not items_state:
        st.warning("まず関連ページを収集してください。")
        st.stop()

    # 類似検索で上位チャンクを抽出
    with st.spinner("RAG で関連チャンクを選別中..."):
        top_chunks, used_sources = retriever.retrieve_for_plan(
            items=[RetrievalItem(**i) for i in items_state],
            user_query=query,
            k=8,
        )

    # Gemini に構造化 JSON で旅程を生成させる
    with st.spinner("Gemini で旅程を構成中..."):
        prompt = build_plan_prompt(
            trip_days=trip_days,
            start_date=str(start_date),
            party=party,
            transport=transport,
            interests=interests,
            start_area=start_area,
            with_kids=with_kids,
            pace=pace,
            start_end_point=start_end_point,
            sources=used_sources,
            context=top_chunks,
        )
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ITINERARY_SCHEMA,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        plan_json = json.loads(resp.text)

    # 表示 + エクスポート
    st.subheader("2) 旅程プラン（ドラフト）")
    st.caption("※ 原文転載は行わず、要約とパラフレーズのみ。各日の根拠URLを併記。")
    md = plan_json_to_markdown(plan_json)
    st.markdown(md)

    st.download_button(
        label="Markdown をダウンロード",
        file_name=f"ehime_plan_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown",
        data=md,
    )

    st.subheader("3) 参照元（いよ観ネット等）")
    for s in plan_json.get("sources", []):
        st.markdown(f"- [{s['title']}]({s['url']}) — {s.get('site','')}")

st.divider()

st.subheader("チューニングメモ")
st.markdown(
    "- Tavily の `search_depth='advanced'` + `include_domains=['iyokannet.jp']` を基本とし、必要に応じて `chunks_per_source` を増減n"
    "- 埋め込み次元は 768（Gemini Embedding MRL）; 速度重視で 256/512 に縮小可n"
    "- 生成は Structured Output（response_mime_type='application/json' + response_schema）で安定化n"
)