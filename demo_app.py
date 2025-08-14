# In app.py (Final Version with Demo Mode Toggle)
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import streamlit as st
import os
import json
import pandas as pd
import shutil

# Import logic from BOTH pipeline files
from pipeline import process_single_pdf, update_knowledge_base
from demo_pipeline import process_single_pdf_fast, update_knowledge_base as update_knowledge_base_demo

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI

st.set_page_config(layout="wide", page_title="Document Intelligence Pipeline", page_icon="🤖")
st.markdown("""<style> ... </style>""", unsafe_allow_html=True) # Your CSS here

def retrieve_relevant_documents(query_text, n_results=15):
    embedding_function = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small")
    client = chromadb.PersistentClient(path="vector_db/")
    collection = client.get_collection(name="utility_docs", embedding_function=embedding_function)
    results = collection.query(query_texts=[query_text], n_results=n_results)
    return results['documents'][0] if results and results.get('documents') else []

def generate_final_answer(query_text, retrieved_docs):
    if not retrieved_docs: return "I couldn't find any relevant documents to answer that."
    client = OpenAI()
    context_str = "\n\n---\n\n".join(retrieved_docs)
    system_prompt = "You are an expert Q&A system. Answer the user's question based ONLY on the provided context documents. If the answer isn't in the context, say so."
    user_prompt = f"CONTEXT DOCUMENTS:\n{context_str}\n\nUSER'S QUESTION:\n{query_text}\n\nANSWER:"
    response = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0.0
    )
    return response.choices[0].message.content

def cleanup_intermediate_folders():
    st.write("Preparing a clean workspace...")
    dirs_to_clean = ["source_pdfs", "corrected_pdfs", "generic_json_outputs"]
    for directory in dirs_to_clean:
        if os.path.isdir(directory): shutil.rmtree(directory)
        os.makedirs(directory)
    st.write("Workspace is clean.")

st.title("Document Intelligence Pipeline")
st.write("An end-to-end solution for ingesting, structuring, and querying utility documents.")

if 'OPENAI_API_KEY' not in os.environ or 'VISION_AGENT_API_KEY' not in os.environ:
    st.error("FATAL ERROR: One or more API keys are not set.")
    st.stop()

if 'processed_data' not in st.session_state: st.session_state.processed_data = []

tab1, tab2 = st.tabs(["Query Knowledge Base", "Add New Documents"])

with tab1:
    st.header("Ask a Question")
    st.info("The knowledge base is ready. Ask any question about the ingested documents.")
    query = st.text_input("Enter your question:", key="rag_query")
    if query:
        with st.spinner("Searching..."):
            retrieved = retrieve_relevant_documents(query)
            answer = generate_final_answer(query, retrieved)
            st.success("Answer:")
            st.markdown(answer)

with tab2:
    st.header("Upload and Process New PDFs")
    
    demo_mode = st.checkbox("🚀 Enable Fast Demo Mode (skips slow rotation fix)", value=True)
    if demo_mode:
        st.warning("Demo Mode is ON. Results for rotated or skewed documents may be inaccurate.", icon="⚠️")
    else:
        st.success("Full Accuracy Mode is ON. All processing steps will run.", icon="✅")

    UPLOAD_DIR = "source_pdfs/"
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)

    if st.button("Process Uploaded Files"):
        if uploaded_files:
            cleanup_intermediate_folders()
            saved_file_paths = []
            for uploaded_file in uploaded_files:
                file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                saved_file_paths.append(file_path)

            st.session_state.processed_data = []
            processed_json_list = []
            progress_bar = st.progress(0, text="Starting pipeline...")
            status_text = st.empty()

            if demo_mode:
                processing_function = process_single_pdf_fast
                db_update_function = update_knowledge_base_demo
            else:
                processing_function = process_single_pdf
                db_update_function = update_knowledge_base

            for i, file_path in enumerate(saved_file_paths):
                filename = os.path.basename(file_path)
                progress_percentage = (i) / len(saved_file_paths)
                progress_bar.progress(progress_percentage, text=f"Processing file {i+1}/{len(saved_file_paths)}: {filename}")
                def status_callback(message): status_text.info(f"File: {filename} - {message}")
                final_json = processing_function(file_path, status_callback)
                processed_json_list.append(final_json)

            status_text.info("All files processed. Updating knowledge base...")
            progress_bar.progress(95, text="Updating knowledge base...")
            db_update_function(processed_json_list)
            
            st.session_state.processed_data = processed_json_list
            progress_bar.progress(100, text="Pipeline complete!")
            status_text.success("Pipeline complete! The knowledge base has been updated.")
            st.balloons()
        else:
            st.warning("Please upload at least one PDF file.")

    if st.session_state.processed_data:
        st.divider()
        st.header("Extracted Information from Last Run")
        for item in st.session_state.processed_data:
            with st.expander(f"**{item.get('documentId')}** - Issuer: {item.get('issuer')}"):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Customer", item.get('customerName', 'N/A'))
                col2.metric("Statement Date", item.get('statementDate', 'N/A'))
                col3.metric("Total Usage", f"{item.get('totalUsage', 0)} {item.get('unit', '')}")
                if item.get('_qc_flag'):
                    col4.error(f"QC Flag: {item.get('_qc_reason')}")
                else:
                    col4.success("QC Passed")
                st.write("**Usage History:**")
                if item.get('usageHistory'):
                    df = pd.DataFrame(item['usageHistory'])
                    st.dataframe(df, use_container_width=True)
                else:
                    st.write("No usage history found.")
                with st.popover("View Full Extracted JSON"):
                    st.json(item)