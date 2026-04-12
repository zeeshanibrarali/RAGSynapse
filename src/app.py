import os
from dotenv import load_dotenv
from pathlib import Path

# Load env FIRST — before any other imports
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=BASE_DIR / ".env")
load_dotenv(dotenv_path="./.env")

import streamlit as st
from llama_index.core import VectorStoreIndex
from ragsynapse.llm.model_factory import get_llm

st.set_page_config(
    page_title="RAGSynapse", page_icon="🧠", layout="wide"
)

from ragsynapse.HTMLTemplates import css
from ragsynapse.stcomp import initialize_session_state, file_processing, handle_user_input

PROVIDER_FALLBACK = ["openai", "anthropic", "ollama"]

DISPLAY_LABELS = {
    "ollama":    "🟢 Ollama (local)",
    "openai":    "⚡ OpenAI",
    "anthropic": "🔷 Anthropic",
}


def render_sidebar():
    with st.sidebar:
        st.markdown("## 🧠 RAGSynapse")
        st.caption("Multi-model document intelligence")
        st.divider()

        # ── Previously stored documents ───────────────────────────────────────
        st.markdown("### 📂 Documents")

        from ragsynapse.pipeline import get_stored_documents
        stored_docs = get_stored_documents()

        if stored_docs:
            st.caption(f"{len(stored_docs)} document(s) in memory")

            # Show stored docs as selectable options
            options = ["— select a document —"] + stored_docs
            selected_doc = st.selectbox(
                label="Stored documents",
                options=options,
                label_visibility="collapsed",
                key="stored_doc_select"
            )

            if selected_doc != "— select a document —":
                if st.button(
                    f"✅ Use: {selected_doc[:30]}",
                    use_container_width=True,
                    type="primary"
                ):
                    st.session_state.documents_processed = True
                    st.session_state.active_document = selected_doc
                    # Re-init conversation so it queries fresh
                    if "conversation" in st.session_state:
                        del st.session_state["conversation"]
                    st.success(f"Active: {selected_doc}")
                    st.rerun()

            # Show currently active document
            if st.session_state.get("active_document"):
                st.info(f"Active: `{st.session_state.active_document}`")

            st.divider()

        # ── Upload new document ───────────────────────────────────────────────
        with st.expander(
            "➕ Upload new document",
            expanded=not bool(stored_docs)   # open by default if nothing stored
        ):
            files = st.file_uploader(
                label="Upload PDF, DOCX or TXT:",
                accept_multiple_files=True,
                type=["pdf", "docx", "txt"],
                label_visibility="collapsed"
            )

            if st.button("Analyze", use_container_width=True, type="primary"):
                if not files:
                    st.warning("Select at least one file first.")
                else:
                    file_processing(files)
                    # After processing, refresh stored docs
                    st.rerun()

        st.divider()

        # ── Status ────────────────────────────────────────────────────────────
        if st.session_state.get("documents_processed"):
            st.success("✅ Ready — ask a question")
        else:
            st.info("⬆️ Select or upload a document")

        active = st.session_state.get("llm_provider", "ollama")
        st.caption(f"Model: `{DISPLAY_LABELS.get(active, active)}`")

        st.divider()
        st.markdown(
            "<small>RAGSynapse v2 · Built by "
            "[Zeeshan Ibrar](https://github.com/zeeshanibrarali)</small>",
            unsafe_allow_html=True
        )

def render_chat_input(current_provider: str) -> tuple[str | None, str]:
    """
    Renders provider dropdown + query input in one row.
    No keys used — avoids DuplicateWidgetID on reruns.
    """
    provider_options = ["ollama", "openai", "anthropic"]
    default_idx = provider_options.index(current_provider) \
        if current_provider in provider_options else 0

    col_provider, col_input = st.columns([1, 5])

    with col_provider:
        selected_provider = st.selectbox(
            label="provider",
            options=provider_options,
            format_func=lambda p: DISPLAY_LABELS.get(p, p),
            index=default_idx,
            label_visibility="collapsed",
        )

    with col_input:
        if not st.session_state.get("documents_processed"):
            st.text_input(
                label="query",
                placeholder="⬆️  Upload and analyze a document first...",
                disabled=True,
                label_visibility="collapsed",
            )
            return None, selected_provider

        user_query = st.text_input(
            label="query",
            placeholder="Ask a question about your documents...",
            label_visibility="collapsed",
        )
        return user_query, selected_provider

def render_evaluation_tab():
    """RAGAS evaluation dashboard — RAGSynapse v2 addition."""
    from ragsynapse.evaluation import run_evaluation, EvalResult

    st.markdown("## 📊 RAG Evaluation")
    st.caption(
        "Test your RAG pipeline quality using RAGAS metrics. "
        "Provide question + expected answer pairs, run evaluation, "
        "see faithfulness and relevancy scores."
    )

    if not st.session_state.get("documents_processed"):
        st.warning("⬆️ Upload and analyze documents first before running evaluation.")
        return

    st.divider()

    # ── Test case input ───────────────────────────────────────────────────────
    st.markdown("### 📝 Test cases")
    st.caption("Enter one question and expected answer per row.")

    # Prefill with 3 empty rows
    if "eval_rows" not in st.session_state:
        st.session_state.eval_rows = [
            {"question": "", "ground_truth": ""},
            {"question": "", "ground_truth": ""},
            {"question": "", "ground_truth": ""},
        ]

    updated_rows = []
    for i, row in enumerate(st.session_state.eval_rows):
        col_q, col_a = st.columns([1, 1])
        with col_q:
            q = st.text_input(
                f"Question {i+1}",
                value=row["question"],
                key=f"eval_q_{i}",
                placeholder="e.g. What is the main topic?"
            )
        with col_a:
            a = st.text_input(
                f"Expected answer {i+1}",
                value=row["ground_truth"],
                key=f"eval_a_{i}",
                placeholder="e.g. The document covers..."
            )
        updated_rows.append({"question": q, "ground_truth": a})

    st.session_state.eval_rows = updated_rows

    col_add, col_run = st.columns([1, 3])
    with col_add:
        if st.button("+ Add row"):
            st.session_state.eval_rows.append({"question": "", "ground_truth": ""})
            st.rerun()
    with col_run:
        run_name = st.text_input(
            "Run name (optional)",
            placeholder="e.g. test-v1",
            label_visibility="collapsed"
        )

    st.divider()

    if st.button("▶ Run Evaluation", type="primary", use_container_width=True):
        # Filter out empty rows
        valid = [r for r in st.session_state.eval_rows
                 if r["question"].strip() and r["ground_truth"].strip()]

        if not valid:
            st.warning("Add at least one question and expected answer.")
            return

        questions = [r["question"] for r in valid]
        ground_truths = [r["ground_truth"] for r in valid]

        with st.spinner(f"Running RAGAS evaluation on {len(questions)} questions..."):
            # Use a query engine for evaluation (not chat engine)
            from ragsynapse.stcomp import load_pipeline
            from llama_index.core import VectorStoreIndex

            _pipeline, _embed_model = load_pipeline()
            # query_engine = VectorStoreIndex.from_vector_store(
            #     _pipeline.vector_store,
            #     embed_model=_embed_model
            # ).as_query_engine(similarity_top_k=2)

            _llm = get_llm(
                provider=st.session_state.get("llm_provider", "ollama"),
                model=None
            )

            from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator

            active_doc = st.session_state.get("active_document")
            filters = None
            if active_doc:
                filters = MetadataFilters(filters=[
                    MetadataFilter(key="source", value=active_doc, operator=FilterOperator.EQ)
                ])
            
            query_engine = VectorStoreIndex.from_vector_store(
                _pipeline.vector_store,
                embed_model=_embed_model
            ).as_query_engine(
                llm=_llm,
                similarity_top_k=3,
                filters=filters,
            )
            
            result = run_evaluation(
                questions=questions,
                ground_truths=ground_truths,
                chat_engine=query_engine,
                run_name=run_name or None,
                log_to_mlflow=True,
            )

        if result.error:
            st.error(f"Evaluation failed: {result.error}")
            return

        # ── Results dashboard ─────────────────────────────────────────────────
        st.success(f"✅ Evaluation complete — {result.num_questions} questions scored")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Overall", f"{result.overall_score():.2f}")
        col2.metric("Faithfulness", f"{result.faithfulness:.2f}",
                    help="Did the answer stick to retrieved context?")
        col3.metric("Answer relevancy", f"{result.answer_relevancy:.2f}",
                    help="Did the answer actually address the question?")
        col4.metric("Context precision", f"{result.context_precision:.2f}",
                    help="Were the retrieved chunks relevant?")

        if result.run_id:
            st.caption(f"Logged to MLflow · run_id: `{result.run_id}` · "
                       f"[View dashboard](http://localhost:5000)")
        else:
            st.caption("MLflow not available — scores shown above only")

        # Score interpretation
        overall = result.overall_score()
        if overall >= 0.8:
            st.success("🟢 Excellent RAG quality")
        elif overall >= 0.6:
            st.warning("🟡 Moderate — consider improving chunking or retrieval")
        else:
            st.error("🔴 Low quality — check document ingestion and chunk size")


def main() -> None:
    st.write(css, unsafe_allow_html=True)

    # Determine current provider from session state or env
    current_provider = st.session_state.get(
        "llm_provider",
        os.getenv("LLM_PROVIDER", "ollama")
    )

    # Initialize session state
    initialize_session_state(provider=current_provider, model=None)

    render_sidebar()

    st.title("🧠 RAGSynapse")
    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_chat, tab_eval = st.tabs(["💬 Chat", "📊 Evaluation"])

    with tab_chat:
        chat_area = st.container()
        st.divider()
        user_query, selected_provider = render_chat_input(current_provider)

        if selected_provider != current_provider:
            st.session_state["llm_provider"] = selected_provider
            if "conversation" in st.session_state:
                del st.session_state["conversation"]
            st.rerun()

        if user_query and user_query.strip():
            with chat_area:
                handle_user_input(
                    user_query,
                    provider=selected_provider,
                    fallback_chain=PROVIDER_FALLBACK
                )

    with tab_eval:
        render_evaluation_tab()



if __name__ == "__main__":
    main()