import os
import uuid
import base64
from pathlib import Path
from typing import Any, Dict, List

from unstructured.partition.auto import partition

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever

# --- GLOBAL CONFIGURATION ---

# Define the paths for storing data.
# This makes it easy to change the storage location in one place.
DOCSTORE_PATH = Path("./my_multimodal_rag/docstore")
IMAGE_OUTPUT_PATH = Path("./my_multimodal_rag/figures")
VECTORSTORE_PATH = Path("./my_multimodal_rag/vectorstore") # ChromaDB will store its data here

# Ensure the output directories exist.
IMAGE_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
DOCSTORE_PATH.mkdir(parents=True, exist_ok=True)


# --- CORE INGESTION LOGIC ---

def partition_document(file_path: str) -> List[Any]:
    """
    Partitions a document using Unstructured.io. This function automatically
    handles different file types (PDF, DOCX, etc.) and extracts text,
    tables, and images.

    Args:
        file_path: The path to the document file.

    Returns:
        A list of Unstructured 'Element' objects.
    """
    print(f"-> Partitioning document: {file_path}")
    return partition(
        filename=file_path,
        # Unstructured will use Detectron2 for 'YOLOX' model for table detection.
        # It will also use Tesseract for OCR on scanned documents or images.
        infer_table_structure=True,
        extract_images_in_pdf=True,
        image_output_dir_path=str(IMAGE_OUTPUT_PATH),
        # Chunking strategy helps in breaking down large documents.
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
    )

def classify_elements(elements: List[Any]) -> Dict[str, List[str]]:
    """
    Classifies partitioned elements into text, tables, and images.
    Tables are returned as their text representation.
    Images are returned as their file paths.

    Args:
        elements: A list of Unstructured 'Element' objects.

    Returns:
        A dictionary with keys 'texts', 'tables', and 'image_paths'.
    """
    print("-> Classifying partitioned elements...")
    texts = []
    tables = []
    
    for element in elements:
        if "CompositeElement" in str(type(element)) or "TopLevel" in str(type(element)):
            # This captures narrative text, titles, etc.
            texts.append(element.text)
        elif "Table" in str(type(element)):
            # This captures tables as their text representation.
            tables.append(element.text)
            
    # Find all saved image files in the output directory
    image_paths = sorted([
        str(f) for f in IMAGE_OUTPUT_PATH.iterdir() if f.is_file()
    ])
    
    print(f"   - Found {len(texts)} text elements.")
    print(f"   - Found {len(tables)} table elements.")
    print(f"   - Found {len(image_paths)} image files.")
    
    return {"texts": texts, "tables": tables, "image_paths": image_paths}

def generate_summaries(elements: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Generates summaries for text, tables, and images using GPT-4o.

    Args:
        elements: A dictionary containing lists of texts, tables, and image paths.

    Returns:
        A dictionary with keys 'text_summaries', 'table_summaries', and 'image_summaries'.
    """
    print("-> Generating summaries and descriptions with GPT-4o...")
    
    # Initialize the OpenAI model
    # gpt-4o is a great choice for its multimodal capabilities and performance.
    llm = ChatOpenAI(model="gpt-4o", max_tokens=1024)

    # --- Text and Table Summaries ---
    text_summaries = llm.batch([
        f"Summarize the following text:\n\n{text}" for text in elements["texts"]
    ])
    table_summaries = llm.batch([
        f"Summarize the following table:\n\n{table}" for table in elements["tables"]
    ])
    
    # --- Image Summaries ---
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    image_summaries = []
    for image_path in elements["image_paths"]:
        base64_image = encode_image(image_path)
        response = llm.invoke([
            SystemMessage(content="You are a bot that is good at analyzing images."),
            HumanMessage(content=[
                {"type": "text", "text": "Describe the contents of this image in detail. If it's a chart or table, summarize its key findings."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ])
        ])
        image_summaries.append(response.content)
        
    print(f"   - Generated {len(text_summaries)} text summaries.")
    print(f"   - Generated {len(table_summaries)} table summaries.")
    print(f"   - Generated {len(image_summaries)} image descriptions.")

    return {
        "text_summaries": [s.content for s in text_summaries],
        "table_summaries": [s.content for s in table_summaries],
        "image_summaries": image_summaries,
    }


def initialize_and_store(
    summaries: Dict[str, List[str]],
    original_contents: Dict[str, List[str]]
):
    """
    Initializes the storage layer (ChromaDB and FileStore) and stores the
    documents using the MultiVectorRetriever pattern.

    Args:
        summaries: A dictionary of summaries for each content type.
        original_contents: A dictionary of the original content for each type.
    """
    print("-> Initializing storage and adding documents...")

    # Initialize the storage components
    docstore = LocalFileStore(str(DOCSTORE_PATH))
    vectorstore = Chroma(
        collection_name="multimodal_rag_store",
        embedding_function=OpenAIEmbeddings(),
        persist_directory=str(VECTORSTORE_PATH) # Persist ChromaDB to disk
    )
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key="doc_id"
    )

    def add_documents_to_retriever(summary_list, original_list, content_type):
        if not summary_list:
            return
            
        doc_ids = [str(uuid.uuid4()) for _ in summary_list]
        
        summary_docs = [
            Document(page_content=s, metadata={"doc_id": doc_ids[i], "content_type": content_type})
            for i, s in enumerate(summary_list)
        ]
        
        retriever.vectorstore.add_documents(summary_docs)
        
        # --- FIX APPLIED HERE ---
        # We must encode strings to bytes before storing them in FileStore.
        # We assume 'text' and 'table' content are strings that need encoding.
        # We assume 'image' content is a path (string), which also needs encoding to be stored.
        encoded_original_list = [
            item.encode('utf-8') if isinstance(item, str) else item 
            for item in original_list
        ]
        
        retriever.docstore.mset(list(zip(doc_ids, encoded_original_list)))
        print(f"   - Added {len(summary_list)} {content_type} documents to storage.")

    # Add each type of content to the retriever
    add_documents_to_retriever(summaries["text_summaries"], original_contents["texts"], "text")
    add_documents_to_retriever(summaries["table_summaries"], original_contents["tables"], "table")
    add_documents_to_retriever(summaries["image_summaries"], original_contents["image_paths"], "image")
    
    # Persist the changes to ChromaDB
    vectorstore.persist()
    print("-> Storage initialization and document addition complete.")


# --- MAIN ORCHESTRATION FUNCTION ---

def ingest_and_process_file(file_path: str):
    """
    The main orchestration function that runs the complete ingestion pipeline
    for a single file.

    Args:
        file_path: The path to the document file to be processed.
    """
    try:
        # Step 1: Partition the document
        raw_elements = partition_document(file_path)
        
        # Step 2: Classify elements into texts, tables, and images
        classified_elements = classify_elements(raw_elements)
        
        # Step 3: Generate summaries and descriptions
        summaries = generate_summaries(classified_elements)
        
        # Step 4: Initialize storage and add documents
        initialize_and_store(summaries, classified_elements)

        print(f"\n✅ Successfully processed and indexed file: {os.path.basename(file_path)}")
    
    except Exception as e:
        print(f"\n💥 An error occurred while processing {os.path.basename(file_path)}: {e}")


# This block allows you to run this file directly for testing purposes.
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Example usage:
    # Make sure you have a file named 'test_document.pdf' in the 'source_documents' folder.
    test_file = "./my_multimodal_rag/source_documents/test_document.pdf"
    
    if os.path.exists(test_file):
        print("--- Starting standalone ingestion test ---")
        ingest_and_process_file(test_file)
        print("--- Ingestion test complete ---")
    else:
        print(f"Test file not found: {test_file}. Please add it to run a standalone test.")