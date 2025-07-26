import base64
from pathlib import Path
from typing import Dict, List, Any

# Corrected imports for Chroma and LocalFileStore
from langchain_chroma import Chroma
from langchain.storage import LocalFileStore
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever

# --- GLOBAL CONFIGURATION ---
# These paths must match the ones used in ingestion.py
DOCSTORE_PATH = Path("./my_multimodal_rag/docstore")
VECTORSTORE_PATH = Path("./my_multimodal_rag/vectorstore")


class RAGChainLoader:
    """
    A class to encapsulate the loading and creation of the multimodal RAG chain.

    This approach makes the code more modular and easier to manage. The main
    script will just need to create an instance of this class and call a method
    to get the fully assembled chain.
    """
    def __init__(self):
        print("-> Initializing RAGChainLoader...")
        self.docstore = LocalFileStore(str(DOCSTORE_PATH))
        self.vectorstore = Chroma(
            collection_name="multimodal_rag_store",
            embedding_function=OpenAIEmbeddings(),
            persist_directory=str(VECTORSTORE_PATH)
        )
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.docstore,
            id_key="doc_id"
        )
        self.model = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=1024)
        print("-> Storage and models loaded.")

    def _wrap_retrieved_documents(self, retrieved_items: List[Any]) -> List[Document]:
        """
        Private method to wrap raw retrieved data into Document objects.
        """
        docs = []
        for item in retrieved_items:
            if isinstance(item, bytes):
                docs.append(Document(page_content=item.decode('utf-8', errors='ignore')))
            else:
                docs.append(Document(page_content=str(item)))
        return docs

    def _create_multimodal_prompt(self, context: Dict) -> List[HumanMessage]:
        """
        Private method to create the final multimodal prompt for the LLM.
        """
        question = context.get("question", "No question provided.")
        context_docs = context.get("context", [])
        
        text_context = [doc.page_content for doc in context_docs if not str(doc.page_content).endswith(('.png', '.jpg', '.jpeg'))]
        formatted_text = "\n\n---\n\n".join(text_context)
        
        prompt_text_template = f"""
Answer the user's question based ONLY on the following context, which can include text and images. If the context is empty or irrelevant, say you do not have enough information to answer.

## Text & Table Context:
{formatted_text}

## User's Question:
{question}
"""
        
        final_prompt_payload = [{"type": "text", "text": prompt_text_template}]

        image_docs = [doc for doc in context_docs if str(doc.page_content).endswith(('.png', '.jpg', '.jpeg'))]
        for doc in image_docs:
            try:
                with open(doc.page_content, "rb") as image_file:
                    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                final_prompt_payload.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                })
            except FileNotFoundError:
                # Add a note to the text part if an image file is missing
                final_prompt_payload[0]["text"] += f"\n\n[Note: Image '{doc.page_content}' was not found on disk.]"

        return [HumanMessage(content=final_prompt_payload)]

    def get_chain(self):
        """
        Public method to assemble and return the final, runnable RAG chain.
        """
        print("-> Assembling the RAG chain...")
        
        chain = (
            {
                "context": self.retriever | RunnableLambda(self._wrap_retrieved_documents),
                "question": RunnablePassthrough()
            }
            | RunnableLambda(self._create_multimodal_prompt)
            | self.model
            | StrOutputParser()
        )
        
        print("-> RAG chain created successfully.")
        return chain


# --- Standalone Test Block ---
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    print("--- Starting standalone chain test using RAGChainLoader class ---")
    
    # 1. Create an instance of the class
    chain_loader = RAGChainLoader()
    
    # 2. Get the runnable chain from the class instance
    rag_chain = chain_loader.get_chain()
    
    # 3. Test with a question
    test_question = "Types of Screening Tests?"
    print(f"\nTesting chain with question: '{test_question}'")
    
    try:
        answer = rag_chain.invoke(test_question)
        print("\n--- Answer ---")
        print(answer)
    except Exception as e:
        print(f"\n💥 An error occurred during chain invocation: {e}")

    print("\n--- Standalone chain test complete ---")
