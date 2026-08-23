import os
from datetime import date, datetime

import pandas as pd
import streamlit as st

from llm.sarashina_client import SarashinaClient
from rag.fast_retriever import CachedSpotRetriever
from utils.formatting import plan_json_to_markdown
from utils.runtime_metrics import (
    ensure_workflow_metrics_compatibility,
    snapshot_workflow_metrics,
)
from workflow.fast_planner import FastPlannerWorkflow


st.set_page_config(
    page_title="Ehime Tour Planner — Hybrid RAG × Sarashina",
    layout="wide",
)

st.title("Ehime Tour Planner（愛媛RAGプランナー）")
st.caption(
    "Tavilyで情報を取得し、Ruri + BM25 のHybrid RAGで絞り込み、"
    "Modal上のSarashinaで旅程を作成します。"
)

TAVILY_API_KEY = st.secrets.get("TAVILY_API_KEY", os.getenv("TAVILY_API_KEY"))
SARASHINA_BASE_URL = st.secrets.get(
    "SARASHINA_BASE_URL", os.getenv("SARASHINA_BASE_URL")
)
SARASHINA_API_KEY = st.secrets.get(
    "SARASHINA_API_KEY", os.getenv("SARASHINA_API_KEY")
)
SARASHINA_MODEL = st.secrets.get(
    "SARASHINA_MODEL", os.getenv("SARASHINA_MODEL", "sarashina")
)
# Explicit cache versions prevent a long-lived Streamlit process from reusing
# pre-release workflow/retriever objects after code-only deployments.
WORKFLOW_CONFIG_VERSION = "spot-id-cache-v3-quality"
RETRIEVER_CONFIG_VERSION = "canonical-spot-dedupe-v1"

missing = []
if not TAVILY_API_KEY:
    missing.append("TAVILY_API_KEY")
if not SARASHINA_BASE_URL:
    missing.append("SARASHINA_BASE_URL")
if not SARASHINA_API_KEY:
    missing.append("SARASHINA_API_KEY")

if missing:
    st.error("Secrets に次を設定してください: " + ", ".join(missing))
    st.stop()


@st.cache_resource
def get_retriever(
    api_key: str,
    retriever_config_version: str,
) -> CachedSpotRetriever:
    del retriever_config_version
    return CachedSpotRetriever(api_key=api_key)


@st.cache_resource
def get_workflow(
    tavily_key: str,
    sarashina_base_url: str,
    sarashina_api_key: str,
    sarashina_model: str,
    workflow_config_version: str,
    retriever_config_version: str,
) -> FastPlannerWorkflow:
    del workflow_config_version
    retriever = get_retriever(tavily_key, retriever_config_version)
    llm = SarashinaClient(
        base_url=sarashina_base_url,
        api_key=sarashina_api_key,
        model=sarashina_model,
    )
    return FastPlannerWorkflow(retriever=retriever, llm=llm)


retriever = get_retriever(TAVILY_API_KEY, RETRIEVER_CONFIG_VERSION)
workflow = get_workflow(
    TAVILY_API_KEY,
    SARASHINA_BASE_URL,
    SARASHINA_API_KEY,
    SARASHINA_MODEL,
    WORKFLOW_CONFIG_VERSION,
    RETRIEVER_CONFIG_VERSION,
)
# A successful itinerary must never be turned into a failure only because an
# older cached object does not yet expose telemetry fields.
ensure_workflow_metrics_compatibility(workflow)

if "items" not in st.session_state:
    st.session_state.items = []
if "plan_json" not in st.session_state:
    st.session_state.plan_json = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_metrics" not in st.session_state:
    st.session_state.last_metrics = None


def snapshot_metrics(final_state: dict) -> dict:
    return snapshot_workflow_metrics(workflow, final_state)


def seconds_label(value) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    return f"{value / 1000:.1f}s"


st.sidebar.header("プラン条件")
with st.sidebar:
    trip_days = st.number_input("旅行日数（日）", 1, 14, 2)
    start_date = st.date_input("開始日（任意）", value=date.today())
    party = st.text_input("同行者（例: 大人2・小学生1）", "大人2")
    transport = st.selectbox(
        "移動手段",
        ["公共交通", "自家用車", "レンタカー", "自転車"],
        index=1,
    )

    st.divider()
    st.markdown("##### 発着地（任意）")
    start_end_options = [
        "指定なし",
        "松山空港",
        "JR松山駅",
        "松山市駅",
        "松山観光港",
        "その他（自由記述）",
    ]
    start_end_choice = st.selectbox("場所を選択", start_end_options, index=0)
    if start_end_choice == "その他（自由記述）":
        start_end_point = st.text_input(
            "自由記述欄", placeholder="例: 今治港、自宅など"
        )
    else:
        start_end_point = start_end_choice

    st.divider()
    interests = st.multiselect(
        "関心テーマ",
        [
            "温泉",
            "城・歴史",
            "サイクリング",
            "自然景観",
            "島めぐり",
            "グルメ",
            "アート",
            "祭り・イベント",
            "体験・アクティビティ",
        ],
        default=["温泉", "グルメ"],
    )

    area_options = [
        "指定なし",
        "中予(松山・道後)",
        "東予(今治・西条など)",
        "南予(大洲・内子・宇和島など)",
        "その他（自由記述）",
    ]
    area_choice = st.selectbox("主な訪問エリア", area_options, index=0)
    if area_choice == "その他（自由記述）":
        start_area = st.text_input(
            "エリアを自由に入力", placeholder="例: 愛南町、鬼北町"
        )
    else:
        start_area = area_choice

    with_kids = st.checkbox("子連れ考慮")
    pace = st.select_slider(
        "1日の詰め込み度",
        options=["ゆったり", "標準", "ぎっしり"],
        value="標準",
    )

    if trip_days == 4:
        st.caption("4日は入力サイズに余裕があれば1回生成し、超える場合だけ分割します。")
    elif trip_days >= 5:
        st.caption("5日以上は3日単位に分割し、各区間をcompactなspot-ID形式で生成します。")

    generate_btn = st.button("プラン生成", type="primary")


st.subheader("1) 関連ソース")
col_l, col_r = st.columns([0.55, 0.45])
with col_l:
    add_web_search = st.checkbox(
        "いよ観ネット以外のWeb検索も追加",
        value=False,
        help="通常は、いよ観ネットを優先して検索します。",
    )
    q_default = "愛媛 観光 モデルコース 道後温泉 松山城"
    query = st.text_input("検索キーワード", q_default)
    max_results = st.slider("最大取得サイト数", 3, 12, 8)

    if st.button("関連ページを事前収集"):
        try:
            with st.spinner("関連ページを取得中..."):
                items = retriever.search_and_prepare(
                    query=query,
                    max_results=max_results,
                    add_web_search=add_web_search,
                )
            st.session_state.items = [item.model_dump() for item in items]
            st.success(f"{len(items)}件を取り込みました。")
        except Exception as exc:
            st.error(f"検索に失敗しました: {exc}")

with col_r:
    st.markdown("**候補ソース**")
    items_state = st.session_state.get("items", [])
    if items_state:
        df = pd.DataFrame(items_state)[["title", "url", "site", "content_chars"]]
        df.rename(columns={"content_chars": "取得文字数"}, inplace=True)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info(
            "事前収集は任意です。未収集のまま「プラン生成」を押すと、"
            "Workflowが必要な検索を自動実行します。"
        )


def current_conditions() -> dict:
    return {
        "query": query,
        "trip_days": int(trip_days),
        "start_date": str(start_date),
        "party": party,
        "transport": transport,
        "interests": list(interests),
        "start_area": start_area,
        "with_kids": bool(with_kids),
        "pace": pace,
        "start_end_point": start_end_point,
        "add_web_search": bool(add_web_search),
        "max_results": int(max_results),
        "items": st.session_state.get("items", []),
    }


if generate_btn:
    try:
        with st.spinner(
            "Hybrid RAGで候補を絞り込み、Sarashinaで旅程を作成しています..."
        ):
            final_state = workflow.run_plan(**current_conditions())
        st.session_state.plan_json = final_state["result"]
        st.session_state.items = final_state.get(
            "items", st.session_state.get("items", [])
        )
        st.session_state.last_metrics = snapshot_metrics(final_state)
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "プランの初稿を作成しました。変更したい点があれば、"
                    "下のチャット欄から具体的に教えてください。"
                ),
            }
        ]
        st.rerun()
    except Exception as exc:
        st.error(f"プラン生成に失敗しました: {exc}")


if st.session_state.plan_json:
    st.divider()
    st.subheader("2) 旅程プラン")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    md = plan_json_to_markdown(st.session_state.plan_json)
    st.markdown(md)
    st.download_button(
        label="Markdown をダウンロード",
        file_name=f"ehime_plan_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown",
        data=md,
    )

    st.subheader("3) 参照元")
    for source in st.session_state.plan_json.get("sources", []):
        st.markdown(
            f"- [{source['title']}]({source['url']}) — {source.get('site', '')}"
        )

    metrics = st.session_state.get("last_metrics")
    if metrics:
        with st.expander("4) 性能計測（直近実行）", expanded=True):
            planner_metrics = metrics.get("planner", {})
            retrieval_metrics = metrics.get("retrieval", {})
            llm_metrics = metrics.get("llm", {})

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("総所要時間", seconds_label(planner_metrics.get("total_ms")))
            c2.metric("生成", seconds_label(planner_metrics.get("generation_ms")))
            c3.metric("RAG", seconds_label(planner_metrics.get("retrieval_ms")))
            c4.metric("生成戦略", metrics.get("strategy", "unknown"))

            r1, r2, r3, r4 = st.columns(4)
            r1.metric(
                "Embedding cache",
                "HIT" if retrieval_metrics.get("cache_hit") else "MISS",
            )
            r2.metric(
                "文書embedding",
                seconds_label(retrieval_metrics.get("doc_embedding_ms")),
            )
            r3.metric(
                "Query embedding",
                seconds_label(retrieval_metrics.get("query_embedding_ms")),
            )
            r4.metric("候補spot数", retrieval_metrics.get("candidate_count", "—"))

            l1, l2, l3, l4, l5 = st.columns(5)
            l1.metric("LLM API", seconds_label(llm_metrics.get("elapsed_ms")))
            l2.metric("LLM phase", llm_metrics.get("phase", "—"))
            l3.metric("retry", llm_metrics.get("retries", "—"))
            l4.metric("retry待機", seconds_label(llm_metrics.get("retry_wait_ms")))
            l5.metric(
                "tokens",
                (
                    f"{llm_metrics.get('prompt_tokens', '—')} → "
                    f"{llm_metrics.get('completion_tokens', '—')}"
                ),
            )
            if not llm_metrics:
                st.caption(
                    "LLM詳細計測は旧キャッシュ互換モードのため未取得です。"
                    "次回生成から新しいSarashinaClientの計測値を表示します。"
                )
            st.caption(
                "cache HIT時は同じ取得文書のdense embeddingを再計算せず、"
                "query embeddingだけを更新します。"
            )

    st.divider()


if prompt := st.chat_input("プランの修正点を入力してください"):
    if not st.session_state.plan_json:
        st.warning("まずプランを生成してください。")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("必要な情報を確認し、変更箇所だけを再生成しています..."):
                refine_args = current_conditions()
                refine_args["query"] = prompt
                refine_args["existing_plan"] = st.session_state.plan_json
                final_state = workflow.run_refine(**refine_args)
            st.session_state.plan_json = final_state["result"]
            st.session_state.items = final_state.get(
                "items", st.session_state.get("items", [])
            )
            st.session_state.last_metrics = snapshot_metrics(final_state)
            response_text = "プランを修正しました。"
        except Exception as exc:
            response_text = f"プランの修正に失敗しました: {exc}"

        st.markdown(response_text)
        st.session_state.messages.append(
            {"role": "assistant", "content": response_text}
        )
        st.rerun()


with st.sidebar:
    st.divider()
    if st.button("会話とプランをリセット"):
        st.session_state.messages = []
        st.session_state.plan_json = None
        st.session_state.items = []
        st.session_state.last_metrics = None
        st.rerun()

    st.caption(
        "生成AIは候補spotの選択・順序・短い説明に利用します。URL等は検索結果から"
        "Python側で復元します。営業時間・運休・料金などはリンク先で確認してください。"
    )
