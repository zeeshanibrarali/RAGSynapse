import os
import streamlit as st

st.set_page_config(
    page_title="RAGSynapse", page_icon="🧠", layout="wide"
)

from dotenv import load_dotenv
from ragsynapse.HTMLTemplates import css
from ragsynapse.stcomp import initialize_session_state, file_processing, handle_user_input
from ragsynapse.llm.model_factory import get_available_models, get_llm

load_dotenv(dotenv_path="../.env", verbose=True)
load_dotenv(dotenv_path="./.env", verbose=True)


# ── Model config per provider ─────────────────────────────────────────────────
PROVIDER_MODELS = {
    "ollama":    ["llama3.2", "mistral", "codellama", "phi3"],
    "openai":    ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    "anthropic": ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022"],
}


def render_sidebar() -> tuple[str, str]:
    """
    Renders the full sidebar: model selector + document uploader.
    Returns selected (provider, model) so app.py can pass them to session state.
    """
    with st.sidebar:

        # ── 1. Model selector (RAGSynapse v2 addition) ────────────────────────
        st.markdown("### 🤖 LLM Backend")

        available = get_available_models()
        provider_options = list(available.keys())

        # Default to env var, fall back to ollama
        default_provider = os.getenv("LLM_PROVIDER", "ollama")
        default_idx = provider_options.index(default_provider) \
            if default_provider in provider_options else 0

        selected_provider = st.selectbox(
            label="Provider",
            options=provider_options,
            index=default_idx,
            help="Switch between OpenAI, Anthropic Claude, or local Ollama models"
        )

        model_options = PROVIDER_MODELS.get(selected_provider, ["default"])
        selected_model = st.selectbox(
            label="Model",
            options=model_options,
            help="Select the specific model to use for answering questions"
        )

        # Show status badge
        if selected_provider == "ollama":
            st.success("🟢 Local — no API key needed")
        elif selected_provider == "openai" and os.getenv("OPENAI_API_KEY"):
            st.success("🟢 OpenAI key detected")
        elif selected_provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
            st.success("🟢 Anthropic key detected")
        else:
            st.warning(f"⚠️ No API key found for {selected_provider}")

        # Apply model change button
        if st.button("Apply Model", use_container_width=True):
            # Force re-init conversation with new model
            if "conversation" in st.session_state:
                del st.session_state["conversation"]
            if "llm_provider" in st.session_state:
                del st.session_state["llm_provider"]
            st.success(f"Switched to {selected_provider} / {selected_model}")
            st.rerun()

        st.divider()

        # ── 2. Document uploader ──────────────────────────────────────────────
        st.markdown("### 📄 Upload Documents")
        files = st.file_uploader(
            label="Upload PDF, DOCX or TXT files:",
            accept_multiple_files=True,
            type=["pdf", "docx", "txt"],
        )

        if st.button(label="Analyze", use_container_width=True):
            if st.session_state.get("documents_processed"):
                st.warning(
                    "Documents already processed. "
                    "Uploading more will add to existing data."
                )
            if files:
                file_processing(files)

        st.divider()

        # ── 3. Footer ─────────────────────────────────────────────────────────
        st.markdown(
            "<small>RAGSynapse v2 · Built by "
            "[Zeeshan Ibrar](https://github.com/zeeshanibrarali)</small>",
            unsafe_allow_html=True
        )

    return selected_provider, selected_model


def main() -> None:
    st.write(css, unsafe_allow_html=True)

    # Render sidebar and get selected model config
    selected_provider, selected_model = render_sidebar()

    # Initialize session state with selected model
    initialize_session_state(
        provider=selected_provider,
        model=selected_model
    )

    # ── Main area ─────────────────────────────────────────────────────────────
    st.title("🧠 RAGSynapse")
    st.caption(
        "Multi-model document intelligence · "
        f"Running: `{selected_provider}` / `{selected_model}`"
    )

    st.divider()

    try:
        user_query = st.text_input(
            "Ask a question about your documents:",
            placeholder="e.g. What are the key findings in this report?"
        )
        if user_query:
            handle_user_input(user_query)
    except Exception as e:
        st.error(f"Query Error: {e}")


if __name__ == "__main__":
    main()
