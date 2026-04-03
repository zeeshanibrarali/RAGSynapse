# Standard Libraries
from time import perf_counter
from typing import Any
import copy
import os
import hashlib
import time
import subprocess
import io
from urllib import response
import toml
from pathlib import Path
import redis as redis_client
from ragsynapse.llm.model_factory import DEFAULT_MODELS, LLMProvider
from ragsynapse.llm.model_factory import DEFAULT_MODELS
from ragsynapse.observability.tracker import log_query

# Third-Party Libraries
import streamlit as st
import nltk
from docx import Document
from fpdf import FPDF
try:
    nltk.download('punkt_tab')
except:
    # Fallback to older punkt if punkt_tab is not available
    nltk.download('punkt')

# Module Imports
from ..chat import get_conversation_engine
from ..pipeline import get_pipeline
from ..pdf_ingest import get_pdf_text, get_pdf_text_ocr, get_text_nodes
from ..HTMLTemplates import bot_template, user_template
from ..display_image import show_image
from nltk.tokenize import sent_tokenize

# Testing
import re

# Initialising pipeline and embed model
# pipeline = get_pipeline()['pipeline']
# embed_model = get_pipeline()['embed_model']

@st.cache_resource
def load_pipeline():
    result = get_pipeline()
    return result['pipeline'], result['embed_model']


# Get the directory of the current file and construct path to config.toml
current_dir = Path(__file__).parent
config_path = current_dir / '..' / '..' / '..' / 'config.toml'
with open(config_path, 'r') as f:
    params = toml.load(f)
# Create the directories if they dont exist
data_path = params['paths']['data_path']
os.makedirs(data_path, exist_ok=True)
dir_path = data_path



class CustomUploadedFile(io.BytesIO):
    def __init__(self, data, name):
        super().__init__(data)
        self.name = name

    def __repr__(self):
        return f"CustomUploadedFile(name={self.name}, size={len(self.getvalue())})"
    


def convert_docx_to_pdf(input_file, output_file):
    subprocess.run(["pandoc", "-o", output_file, input_file])



def split_into_sentences(text):
    sentences = sent_tokenize(text)
    return sentences


    



# Function to save the uploaded file
def save_uploaded_file(uploaded_file):
    try:
        # Create the directory if it doesn't exist
        os.makedirs(dir_path, exist_ok=True)
        
        # Save the file to the specified directory
        file_path = os.path.join(dir_path, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
    except Exception as e:
        st.error(f'Error saving file: {e}')





def txt_to_pdf(txt_file_path, pdf_file_path):
    # Create a new PDF object
    pdf = FPDF()
    # Open the text file for reading
    with open(txt_file_path, 'r', encoding='latin-1') as txt_file:
        # Get the content of the text file
        txt_content = txt_file.read()
    # Split the text content into lines
    lines = txt_content.splitlines()
    # Add a new page to the PDF
    pdf.add_page()
    # Set the font and font size
    pdf.set_font('Arial', size=12)
    # Loop through the lines and add them to the PDF
    for line in lines:
        pdf.cell(w=200, h=10, txt=line, ln=1, align='L') # type: ignore
    # Save the PDF
    pdf.output(pdf_file_path)


def initialize_session_state(provider: str = None, model: str = None):
    if "conversation" not in st.session_state:
        _pipeline, _embed_model = load_pipeline()
        active_doc = st.session_state.get("active_document")  # ← ADD THIS
        st.session_state.conversation = get_conversation_engine(
            _embed_model,
            _pipeline.vector_store,
            provider=provider,
            model=model,
            active_document=active_doc,  # ← ADD THIS
        )
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = provider or os.getenv("LLM_PROVIDER", "ollama")
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = model
    if not st.session_state.get("documents_processed"):
        from ragsynapse.pipeline import get_stored_documents
        if get_stored_documents():
            st.session_state.documents_processed = True




def file_processing(files: list[Any]) -> None:
    """
    Process uploaded PDF files: Extract text, segment them, and add them to the vector store.
 
    Args:
    - files (list[Any]): A list of uploaded PDF files to be processed.
 
    Notes:
    - The function provides user feedback using Streamlit's info and spinner functionalities.
    - It updates the session state to indicate that documents have been processed.
    - Passes PDFs for performing OCR on them if they dont contain any text.
    - Any exceptions raised during processing are caught and displayed as errors in Streamlit.
    """
    try:
        # While Everything is being processed run the spinner
        with st.spinner("Processing your documents..."):
            copy_files = copy.deepcopy(files)
            # Initialise documents to store all the Documents in the list
            documents = []


            

            # Loop across every file that has been uploaded
            for i,file in enumerate(files):
                if file.name.endswith(".pdf") :
                    save_uploaded_file(file)
                    document_list = get_pdf_text(file)
                    # Check if document contains an error
                    # Fallback to OCR if extracting text from PDF fails
                    if document_list[0].text == 'Error':
                        st.warning("Error extracting text from PDFs using the first method. Trying OCR...")
                        document_list = get_pdf_text_ocr(copy_files[i])
                    
                    # documents.extend(document_list)
                
                elif file.name.endswith(".docx"):
                    docx_file = file
                    docx_path = os.path.join(data_path, docx_file.name)
                    pdf_path = os.path.join(data_path, os.path.splitext(docx_file.name)[0] + ".pdf")
                    
                    # Save the docx file
                    with open(docx_path, "wb") as f:
                        f.write(docx_file.read())
                        # Convert the docx file to pdf
                        convert_docx_to_pdf(docx_path, pdf_path)

                        # Read the PDF file
                        if os.path.exists(pdf_path):
                            with open(pdf_path, "rb") as f:
                                pdf_bytes = f.read()
                            uploaded_file = CustomUploadedFile(pdf_bytes, os.path.splitext(docx_file.name)[0] + ".pdf")
                            document_list = get_pdf_text(uploaded_file)  
                        else:
                            st.write(f"PDF file {os.path.basename(pdf_path)} not found.")


                                
                elif file.name.endswith(".txt"):
                    txt_file = file
                    txt_path = os.path.join(data_path, txt_file.name)
                    pdf_path = os.path.join(data_path, os.path.splitext(txt_file.name)[0] + ".pdf")
                    # Save the txt file
                    with open(txt_path, "wb") as f:
                        f.write(txt_file.read())
                    # Convert the txt file to pdf
                    txt_to_pdf(txt_path, pdf_path)
                    # Read the PDF file
                    if os.path.exists(pdf_path):
                        with open(pdf_path, "rb") as f:
                            pdf_bytes = f.read()
                        uploaded_file = CustomUploadedFile(pdf_bytes, os.path.splitext(txt_file.name)[0] + ".pdf")
                        document_list = get_pdf_text(uploaded_file)
                    else:
                        st.write(f"PDF file {os.path.basename(pdf_path)} not found.")

                # Append all the extracted pages in main document list
                documents.extend(document_list)
            
            st.info(
                f"Document processing completed."
            ) 
            
            # Initialise performance counter time
            t0 = perf_counter()
            # Get text nodes
            # nodes = get_text_nodes(documents, pipeline)
            _pipeline, _embed_model = load_pipeline()


            # nodes = get_text_nodes(documents, _pipeline)
            # t_delta = (perf_counter() - t0) / 60

            # #  if Every thing moves smoothly Update session state
            # if nodes is not None:
            #     st.success(
            #         f"Data preparation complete in {t_delta:.2f} minutes. You can now initiate queries."
            #     )
            #     st.session_state.documents_processed = True
            # else: 
            #     st.info(
            #         "An Error occured. You can try passing the documents again..."
            #     )


            nodes = get_text_nodes(documents, _pipeline)
            t_delta = (perf_counter() - t0) / 60

            if nodes is not None and len(nodes) > 0:
                st.success(f"✅ {len(nodes)} chunks embedded and stored in Redis ({t_delta:.2f} min)")
                st.session_state.documents_processed = True

            elif nodes is not None and len(nodes) == 0:
                st.error(
                    "⚠️ Pipeline ran but stored 0 chunks. "
                    "Likely cause: document already exists in Redis (duplicate). "
                    "Fix: run `docker compose down -v && docker compose up` to clear Redis, then re-upload."
            )
            # Don't set documents_processed = True — nothing was stored

            else:
                st.error("❌ Pipeline returned None — check logs for exception.")

            health = check_redis_health()
            if health["connected"]:
                st.caption(f"Redis: {health['total_keys']} total keys stored")
            else:
                st.warning(f"Redis connection issue: {health['error']}")

    except Exception as e:
         st.error(f"An error occurred: {e}")




def get_page_num(text: str):
    # Regular expression pattern to find page number references
    pattern = r'PAGE_NUM=(\d+)'

    # Find all matches of the pattern in the text
    matches = re.findall(pattern, text)

    # Use a set to store unique page numbers
    unique_page_numbers = set()

    # Extract the numbers from the matches and store them in a set to ensure uniqueness
    for match in matches:
        unique_page_numbers.add(int(match))

    return list(unique_page_numbers)

def handle_user_input(
    user_query: str,
    provider: str = None,
    fallback_chain: list = None,
) -> None:
    """
    Process user input with automatic provider fallback on rate limit errors.
    Logs every query to MLflow for observability.
    """
    from ragsynapse.observability.tracker import log_query
    from ragsynapse.llm.model_factory import DEFAULT_MODELS, LLMProvider

    fallback_chain = fallback_chain or ["openai", "anthropic", "ollama"]
    current_provider = provider or st.session_state.get("llm_provider", "ollama")

    if current_provider in fallback_chain:
        start = fallback_chain.index(current_provider)
        ordered = fallback_chain[start:] + fallback_chain[:start]
    else:
        ordered = [current_provider] + fallback_chain

    response = None
    used_provider = current_provider
    chat_history_snapshot = None
    total_latency_ms = 0.0

    for attempt_provider in ordered:
        try:
            if st.session_state.get("llm_provider") != attempt_provider:
                if "conversation" in st.session_state:
                    del st.session_state["conversation"]
                st.session_state["llm_provider"] = attempt_provider
                _pipeline, _embed_model = load_pipeline()
                st.session_state.conversation = get_conversation_engine(
                    _embed_model,
                    _pipeline.vector_store,
                    provider=attempt_provider,
                    model=None,
                )

            t0 = perf_counter()
            response = st.session_state.conversation.chat(user_query)
            total_latency_ms = (perf_counter() - t0) * 1000

            # Capture history immediately before anything else runs
            chat_history_snapshot = list(
                st.session_state.conversation.chat_history
            )
            used_provider = attempt_provider
            break

        except Exception as e:
            err = str(e).lower()
            is_quota = any(x in err for x in [
                "rate limit", "ratelimit", "quota", "429",
                "insufficient_quota", "too many requests"
            ])
            if is_quota and attempt_provider != ordered[-1]:
                next_p = ordered[ordered.index(attempt_provider) + 1]
                st.warning(
                    f"⚠️ `{attempt_provider}` rate limit hit — "
                    f"switching to `{next_p}` automatically..."
                )
                continue
            else:
                st.error(f"Query Error: {e}")
                return

    if response is None or chat_history_snapshot is None:
        st.error("All providers failed. Please try again later.")
        return

    # ── MLflow tracking ───────────────────────────────────────────────────────
    used_model = st.session_state.get("llm_model") or DEFAULT_MODELS.get(
        LLMProvider(used_provider), "unknown"
    )
    context_texts = [
        node.text for node in response.source_nodes
    ] if hasattr(response, "source_nodes") and response.source_nodes else []

    log_query(
        question=user_query,
        answer=str(response),
        provider=used_provider,
        model=used_model,
        total_latency_ms=total_latency_ms,
        num_source_nodes=len(context_texts),
        context_texts=context_texts,
    )

    if used_provider != current_provider:
        st.info(f"ℹ️ Response from `{used_provider}` (auto-switched)")

    # ── Use snapshot — never touch session_state.conversation again ───────────
    st.session_state.chat_history = chat_history_snapshot

    for idx, msg in enumerate(chat_history_snapshot):
        if msg.content is None:
            continue

        if msg.role.name == 'ASSISTANT':
            bot_response = msg.content
            bot_sentences = split_into_sentences(bot_response)

            if len(bot_sentences) > 2:
                truncated_response = ' '.join(bot_sentences[:2])
                st.write(bot_template.replace("{{MSG}}", truncated_response), unsafe_allow_html=True)
                with st.expander(label="More Options", expanded=False):
                    tab1, tab2, tab3 = st.tabs(["Know More", "Get Full Context", "View Page"])
                    with tab1:
                        st.write(bot_template.replace("{{MSG}}", bot_response), unsafe_allow_html=True)
                    with tab2:
                        if response.source_nodes:
                            file_names = list({n.metadata['source'] for n in response.source_nodes})
                            st.write("#### Source document")
                            st.write(f"**File:** {file_names[0]}")
                            st.write("#### Relevant excerpt")
                            node = response.source_nodes[0]
                            context_text = f"*Source: {node.metadata.get('source', 'Unknown')}*\n\n{node.text[:400]}..."
                            st.markdown(context_text)
                    with tab3:
                        for node_idx, node in enumerate(response.source_nodes):
                            path = data_path + node.metadata['source']
                            page_nums = get_page_num(node.text)
                            for page_idx, page_num in enumerate(page_nums):
                                st.write(f"Page {page_num}")
                                images, image_path = show_image(path, page_num)
                                st.image(images, caption=f"Page {page_num}", use_column_width=True)
                                with open(image_path, "rb") as file:
                                    key = hashlib.sha256(
                                        (image_path + str(idx) + str(node_idx) +
                                         str(page_idx) + str(page_num) + user_query).encode()
                                    ).hexdigest()
                                    st.download_button(
                                        label="Download Page ⬇️",
                                        data=file,
                                        file_name=image_path,
                                        mime="image/png",
                                        key=key
                                    )
                break
            else:
                st.write(bot_template.replace("{{MSG}}", bot_response), unsafe_allow_html=True)
                with st.expander(label="More Options", expanded=False):
                    tab1, tab2 = st.tabs(["Get Full Context", "View Page"])
                    with tab1:
                        if response.source_nodes:
                            file_names = list({n.metadata['source'] for n in response.source_nodes})
                            st.write("#### Source document")
                            st.write(f"**File:** {file_names[0]}")
                            node = response.source_nodes[0]
                            context_text = f"*Source: {node.metadata.get('source', 'Unknown')}*\n\n{node.text[:400]}..."
                            st.markdown(context_text)
                    with tab2:
                        for node_idx, node in enumerate(response.source_nodes):
                            path = data_path + node.metadata['source']
                            page_nums = get_page_num(node.text)
                            for page_idx, page_num in enumerate(page_nums):
                                st.write(f"Page {page_num}")
                                images, image_path = show_image(path, page_num)
                                st.image(images, caption=f"Page {page_num}", use_column_width=True)
                                with open(image_path, "rb") as file:
                                    key = hashlib.sha256(
                                        (image_path + str(idx) + str(node_idx) +
                                         str(page_idx) + str(page_num) + user_query).encode()
                                    ).hexdigest()
                                    st.download_button(
                                        label="Download Page ⬇️",
                                        data=file,
                                        file_name=image_path,
                                        mime="image/png",
                                        key=key
                                    )

        elif msg.role.name == 'USER':
            st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)


def check_redis_health() -> dict:
    """Returns count of vectors and docs currently in Redis."""
    try:
        r = redis_client.Redis(host='redis', port=6379)
        info = r.info('keyspace')
        keys = r.keys('*')
        return {
            "connected": True,
            "total_keys": len(keys),
            "keyspace": info
        }
    except Exception as e:
        return {"connected": False, "error": str(e)}
# def handle_user_input(user_query: str) -> None:

    """
    Process user input, retrieve relevant responses, and display them in Streamlit.

    Args:
    - user_query (str): The query or question input by the user.

    Notes:
    - The function retrieves a response using the conversation chain from the session state.
    - It updates the chat history in the session state.
    - Displays user input and bot responses using predefined HTML templates.
    """
    # Get response
    # response = st.session_state.conversation.chat(user_query, tool_choice="query_engine_tool")


    response = st.session_state.conversation.chat(user_query)
    # Create new session state Variable
    st.session_state.chat_history = st.session_state.conversation.chat_history
    for idx, msg in enumerate(st.session_state.chat_history):
        # Output empty string if response is None (handling an exception)
        if msg.content is None: 
            continue

        if msg.role.name == 'ASSISTANT':
            # Split the bot's response into sentences
            bot_response = msg.content
            bot_sentences = split_into_sentences(bot_response)
            # Concatenate the first two sentences into a single response
            if len(bot_sentences) > 2:
                truncated_response = ' '.join(bot_sentences[:2])
                st.write(
                    bot_template.replace("{{MSG}}", truncated_response), unsafe_allow_html=True
                )  
                with st.expander(label = "More Options",expanded = False):
                    tab1 , tab2, tab3 = st.tabs(["Know More","Get Full Context","View Page"])
                    with tab1:
                        # Display the entire response
                            st.write(
                                bot_template.replace("{{MSG}}", bot_response), unsafe_allow_html=True
                            )
                    with tab2:
                        # Initialize empty lists to store file names and page numbers
                        file_names = []
                        # Iterate through response.source_nodes
                        for node in response.source_nodes:
                            # Append file name and page number to respective lists
                            file_names.append(node.metadata['source'])
                        # Use set to remove duplicates
                        unique_file_names = list(set(file_names))

                        # Display unique file names and page numbers
                        st.write(f"#### The answer generated by Document Insight is from below file")
                        st.write(f"File Name: {unique_file_names[0]}")

                        # Display additional context only once
                        if len(response.source_nodes) > 0:
                            st.write("#### Exact Paragraph from where answer is derived: ")
                            node = response.source_nodes[0]  # Only consider the first node
                            # st.markdown(get_context(node.text, bot_response),unsafe_allow_html=True)
                            # Replace get_context calls with this temporarily
                            # context_text = f"*Source: {node.metadata.get('source', 'Unknown')} | Chunk preview: {node.text[:300]}...*"
                            context_text = f"*Source: {node.metadata.get('source', 'Unknown')}*\n\n{node.text[:400]}..."
                            st.markdown(context_text)

                    with tab3:
                        for node_idx, node in enumerate(response.source_nodes):
                            path = data_path+node.metadata['source']
                            print(path)
                            # page_num = int(node.metadata['page_num'])-1
                            page_nums = get_page_num(node.text)
                            if page_nums is []:
                                st.error('NO PAGE NUM FOUND')
                            for page_idx, page_num in enumerate(page_nums):
                                st.write('PAGE_NUM found:' + str(page_num))
                                images,image_path = show_image(path,page_num)
                                st.image(images, caption=f"Page {page_num}", use_column_width=True)
                                with open(image_path, "rb") as file:
                                    key = hashlib.sha256((image_path + str(idx) + str(node_idx) + str(page_idx) + str(page_num) + "_tab3_long" + user_query).encode()).hexdigest() 
                                    st.download_button(
                                        label="Download Page⬇️",
                                        data=file,
                                        file_name=image_path,
                                        mime="image/png",
                                        key=key
                                    )
                    break
            else:
                st.write(
                    bot_template.replace("{{MSG}}", bot_response), unsafe_allow_html=True
                )
                with st.expander(label = "More Options",expanded = False):
                    tab1,tab2= st.tabs(["Get Full Context","View Page"])
                    with tab1:
                        # Initialize empty lists to store file names and page numbers
                        file_names = []
                        # Iterate through response.source_nodes
                        for node in response.source_nodes:
                            # Append file name and page number to respective lists
                            file_names.append(node.metadata['source'])
                        # Use set to remove duplicates
                        unique_file_names = list(set(file_names))

                        # Display unique file names and page numbers
                        st.write(f"#### The answer generated by Document Insight is from below file")
                        st.write(f"File Name: {unique_file_names[0]}")

                        # Display additional context only once
                        if len(response.source_nodes) > 0:
                            st.write("#### Exact Paragraph from where answer is derived: ")
                            node = response.source_nodes[0]  # Only consider the first node
                            # st.markdown(get_context(node.text, bot_response),unsafe_allow_html=True)
                            # Replace get_context calls with this temporarily
                            # context_text = f"*Source: {node.metadata.get('source', 'Unknown')} | Chunk preview: {node.text[:300]}...*"
                            context_text = f"*Source: {node.metadata.get('source', 'Unknown')}*\n\n{node.text[:400]}..."
                            st.markdown(context_text)

                    with tab2:
                        for node_idx, node in enumerate(response.source_nodes):
                            path = data_path+node.metadata['source']
                            # page_num = int(node.metadata['page_num'])-1
                            page_nums = get_page_num(node.text)
                            if page_nums is []:
                                st.error('NO PAGE NUM FOUND')
                            for page_idx, page_num in enumerate(page_nums):
                                st.write('PAGE_NUM found:' + str(page_num))
                                images,image_path = show_image(path,page_num)
                                st.image(images, caption=f"Page {page_num}", use_column_width=True)
                                with open(image_path, "rb") as file:
                                    key = hashlib.sha256((image_path + str(idx) + str(node_idx) + str(page_idx) + str(page_num) + "_tab2_short" + user_query).encode()).hexdigest() 
                                    st.download_button(
                                        label="Download Page⬇️",
                                        data=file,
                                        file_name=image_path,
                                        mime="image/png",
                                        key=key
                                    )               
        elif msg.role.name == 'USER':
            # Adding styles to Chat Boxes and messages
            st.write(
                user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True
            )
        # Skipping messages from other roles (TOOL)
        else:
            continue