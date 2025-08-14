# In demo_pipeline.py
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import os
import json
import shutil
from agentic_doc.parse import parse
from openai import OpenAI
import chromadb
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# --- Configuration (Same as main pipeline) ---
SOURCE_PDF_DIR = "source_pdfs/"
GENERIC_JSON_DIR = "generic_json_outputs/"
FINAL_JSON_DIR = "final_json_outputs/"
REVIEW_DIR = "manual_review_needed/"
DB_DIR = "vector_db/"
COLLECTION_NAME = "utility_docs"

def process_single_pdf_fast(pdf_path, status_callback):
    """
    A lightweight version of the pipeline that SKIPS the slow rotation fix.
    Reads directly from the source PDF.
    """
    filename = os.path.basename(pdf_path)
    base_name = os.path.splitext(filename)[0]

    # --- Step 1 is SKIPPED ---
    # We use the original pdf_path directly
    status_callback(f"Step 1/4: Skipping rotation fix (Demo Mode)...")
    
    # --- Step 2: PDF -> Generic JSON ---
    status_callback(f"Step 2/4: Running AgenticDoc for {filename}...")
    os.makedirs(GENERIC_JSON_DIR, exist_ok=True)
    # IMPORTANT: We parse the original `pdf_path`, not a corrected one.
    results = parse([pdf_path], include_marginalia=False, include_metadata_in_markdown=False)
    output_generic_json_path = os.path.join(GENERIC_JSON_DIR, f"{base_name}.json")
    chunks_data = [chunk.dict() for chunk in results[0].chunks]
    with open(output_generic_json_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2)

    # --- Step 3: Generic JSON -> Final JSON ---
    status_callback(f"Step 3/4: Structuring data with OpenAI for {filename}...")
    # (The rest of this function is identical to the main pipeline.py)
    os.makedirs(FINAL_JSON_DIR, exist_ok=True)
    with open("master_prompt.txt", "r") as f:
        master_prompt_template = f.read()
    client = OpenAI()
    with open(output_generic_json_path, 'r') as f:
        content = f.read()
    prompt = master_prompt_template.replace("{{generic_json_content}}", content)
    prompt = prompt.replace("{{document_id_placeholder}}", base_name)
    response = client.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}],
        temperature=0.0, response_format={"type": "json_object"}
    )
    final_json_obj = json.loads(response.choices[0].message.content)
    with open(os.path.join(FINAL_JSON_DIR, f"{base_name}.json"), 'w') as f:
        json.dump(final_json_obj, f, indent=2)

    # --- Step 4: QC Flagging ---
    status_callback(f"Step 4/4: Running QC check for {filename}...")
    os.makedirs(REVIEW_DIR, exist_ok=True)
    if final_json_obj.get('_qc_flag') is True:
        original_uncorrected_path = os.path.join(SOURCE_PDF_DIR, filename)
        destination_path = os.path.join(REVIEW_DIR, filename)
        if os.path.exists(original_uncorrected_path):
            shutil.move(original_uncorrected_path, destination_path)
            
    return final_json_obj

# We still need the update_knowledge_base function for the app to call
def update_knowledge_base(list_of_final_json):
    # This function is identical to the one in pipeline.py
    if not list_of_final_json: return
    db_client = chromadb.PersistentClient(path=DB_DIR)
    embedding_function = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small")
    collection = db_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=embedding_function)
    docs_to_ingest = [Document(page_content=json.dumps(obj, indent=2), metadata={"source": f"{obj.get('documentId')}.json"}) for obj in list_of_final_json]
    if docs_to_ingest:
        langchain_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        embeddings = langchain_embeddings.embed_documents([doc.page_content for doc in docs_to_ingest])
        current_count = collection.count()
        ids = [f"doc_{current_count + i}" for i in range(len(docs_to_ingest))]
        collection.add(ids=ids, embeddings=embeddings, documents=[doc.page_content for doc in docs_to_ingest], metadatas=[doc.metadata for doc in docs_to_ingest])
    print(f"Successfully added {len(docs_to_ingest)} new document(s) to the knowledge base.")