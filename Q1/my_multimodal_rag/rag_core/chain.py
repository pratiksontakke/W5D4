import base64
from pathlib import Path
from typing import Dict, List

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.storage import FileStore
from langchain.vectorstores import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever

# --- GLOBAL CONFIGURATION (Must match ingestion.py) ---

# Define the paths for loading the stored data.
DOCSTORE_PATH = Path("./docstore")
VECTORSTORE_PATH = Path("./vectorstore")


# --- HELPER FUNCTIONS FOR CONTEXT FORMATTING ---

def format_context(context: Dict) -> str:
    """
    Formats the retrieved context into a readable string. This function
    handles both text documents and images (by encoding them as base64).
    """
    formatted_docs = []
    
    # Check if the context contains retrieved documents
    if "context" in context and context["context"]:
        for i, doc in enumerate(context["context"]):
            # doc is a Document object, doc.page_content contains the original content
            content = doc.page_content
            doc_type = "text"
            
            # Check if the content is a path to an image file
            if isinstance(content, str) and content.endswith(('.png', '.jpg', '.jpeg')):
                doc_type = "image"
                try:
                    with open(content, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                    content = f"data:image/jpeg;base64,{base64_image}"
                except FileNotFoundError:
                    content = "Image not found."
            
            # Format the document with its index and type
            formatted_docs.append(f"--- Document {i+1} ({doc_type}) ---\n{content}")

    return "\n\n".join(formatted_docs)

def create_final_prompt(context: Dict) -> List:
    """
    Creates the final list of messages for the LLM, including formatted
    context (text and images) and the user's question.
    """
    question = context.get("question", "No question provided.")
    formatted_context = format_context(context)

    # If the context contains images, the prompt must be a list of dicts.
    # Otherwise, it can be a simple string. For consistency, we always use the list format.
    
    final_prompt_content = [
        {"type": "text", "text": f"""
Answer the user's question based only on the following context, which can include text and images. If the context is empty or irrelevant, say you do not have enough information to answer.

## Context:
{formatted_context}

## User's Question:
{question}
"""}
    ]
    
    # Add image URLs if any were found and formatted
    if "context" in context and context["context"]:
        for doc in context["context"]:
            content = doc.page_content
            if isinstance(content, str) and content.endswith(('.png', '.jpg', '.jpeg')):
                try:
                    with open(content, "rb") as image_file:
                        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                    final_prompt_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    })
                except FileNotFoundError:
                    continue # Skip if image not found

    return [HumanMessage(content=final_prompt_content)]


# --- MAIN CHAIN CREATION FUNCTION ---

def create_multimodal_rag_chain():
    """
    Creates the complete RAG chain by loading the existing vector store
    and document store, and assembling the components using LCEL.

    Returns:
        A runnable LangChain object that takes a question and returns an answer.
    """
    print("-> Loading existing storage...")

    # Initialize the storage components to load the existing data
    docstore = FileStore(str(DOCSTORE_PATH))
    vectorstore = Chroma(
        collection_name="multimodal_rag_store",
        embedding_function=OpenAIEmbeddings(),
        persist_directory=str(VECTORSTORE_PATH)
    )
    
    # Initialize the retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key="doc_id"
    )

    print("-> Assembling the RAG chain...")

    # Initialize the language model
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Define the RAG chain using LangChain Expression Language (LCEL)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | RunnableLambda(create_final_prompt)
        | model
        | StrOutputParser()
    )

    print("-> RAG chain created successfully.")
    return chain


# This block allows you to run this file directly for testing purposes.
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("--- Starting standalone chain test ---")
    
    # Create the chain
    rag_chain = create_multimodal_rag_chain()
    
    # Test with a question
    test_question = "What is the key finding from any charts in the document?"
    print(f"\nTesting chain with question: '{test_question}'")
    
    # Invoke the chain
    try:
        answer = rag_chain.invoke(test_question)
        print("\n--- Answer ---")
        print(answer)
    except Exception as e:
        print(f"\n💥 An error occurred during chain invocation: {e}")

    print("\n--- Standalone chain test complete ---")