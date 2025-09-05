"""
Arkadiusz's RAG Module
=============================================================

Author: Arkadiusz
Integration: Overfit and Underpaid Team
Date: September 2025
"""

import hashlib
import uuid
import os
import streamlit as st
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Iterable

# --- LangChain core
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# --- Loaders
from langchain_community.document_loaders import TextLoader, PyPDFLoader

# --- OpenAI embeddings (LangChain)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# --- Additional LLM providers for LangChain
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from langchain_qdrant import QdrantVectorStore

# --- LangChain RAG components
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Import RiskRadar logging
from src.utils.debug_logger import get_logger, log_info, log_error, log_debug, log_warning
import config

# Get logger instance
logger = get_logger()


class ArkadiuszRAGSystem:
    """
    Komentarze w języku polskim są zachowane zgodnie z oryginałem
    """
    
    def __init__(self, chat_model=None):
        """Initialize the RAG system with configuration from config.py
        
        Args:
            chat_model: Optional model name to use for chat. If not provided, uses config default.
        """
        log_info("[RAG] Initializing RAG System")
        
        # Set environment variables for the module
        # we set environment variables while the program is running
        if config.OPENAI_API_KEY:
            os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
        
        # Qdrant configuration
        self.qdrant_url = config.QDRANT_URL
        self.qdrant_api_key = config.QDRANT_API_KEY
        
        # Data directory
        self.DATA_DIR = Path(config.RAG_DATA_DIR)
        self.COLLECTION_NAME = config.QDRANT_COLLECTION_NAME
        
        # Chunking params
        self.CHUNK_SIZE = config.RAG_CHUNK_SIZE          # number of characters
        self.CHUNK_OVERLAP = config.RAG_CHUNK_OVERLAP    # number of characters
        self.SEPARATORS = config.RAG_SEPARATORS  # To be specified after analyzing the file structure
        
        # Embeddings (OpenAI)
        self.EMBEDDING_MODEL = config.RAG_EMBEDDING_MODEL  # -small (cheaper)
        
        # Chat model - use provided model or fallback to config
        self.CHAT_MODEL = chat_model if chat_model else config.RAG_CHAT_MODEL
        log_info(f"[RAG] Using chat model: {self.CHAT_MODEL}")
        
        # Batching writes (performance)
        self.WRITE_BATCH_SIZE = config.RAG_WRITE_BATCH_SIZE
        
        # Initialize client
        self.client = None
        self.vectorstore = None
        
        log_debug(f"[RAG] Configuration loaded - Collection: {self.COLLECTION_NAME}, Model: {self.EMBEDDING_MODEL}")
        log_info(f"[RAG] Data directory: {self.DATA_DIR}")
    
    def get_qdrant_client(self) -> QdrantClient:
        """Get or create Qdrant client with proper configuration"""
        if self.client:
            return self.client
            
        log_info("[RAG-QDRANT] Connecting to Qdrant")
        
        try:
            # Try to use Streamlit secrets if available (for Streamlit Cloud deployment)
            try:
                if hasattr(st, 'secrets') and 'QDRANT_URL' in st.secrets:
                    log_debug("[RAG-QDRANT] Using Streamlit secrets for Qdrant Cloud")
                    self.client = QdrantClient(
                        url=st.secrets["QDRANT_URL"],
                        api_key=st.secrets.get("QDRANT_API_KEY"),
                        timeout=60
                    )
                    log_info("[RAG-QDRANT] Successfully connected to Qdrant via Streamlit secrets")
                    return self.client
            except Exception as secrets_error:
                log_debug(f"[RAG-QDRANT] Streamlit secrets not available: {secrets_error}")
                
            # Use environment variables (primary method for local development)
            if self.qdrant_api_key:
                log_debug(f"[RAG-QDRANT] Using Qdrant Cloud at {self.qdrant_url}")
                self.client = QdrantClient(
                    url=self.qdrant_url,
                    api_key=self.qdrant_api_key,
                    timeout=60
                )
            # Fallback to local Qdrant
            else:
                log_debug(f"[RAG-QDRANT] Using local Qdrant at {self.qdrant_url}")
                self.client = QdrantClient(
                    url=self.qdrant_url,
                    timeout=60
                )
            
            log_info("[RAG-QDRANT] Successfully connected to Qdrant")
            return self.client
            
        except Exception as e:
            log_error(f"[RAG-QDRANT] Failed to connect to Qdrant: {str(e)}")
            raise
    
    # ============= ORIGINAL UTILITY FUNCTIONS =============
    
    def load_documents(self, data_dir: Path, selected_files: List[str] = None) -> List[Document]:
        """
        Loads documents from the data_dir directory or specific selected files.
        Supports PDF files.
        Each Document has the following metadata: source, page (for PDF), filetype.
        
        """
        log_info(f"[RAG-INDEX] Loading documents from {data_dir}")
        docs: List[Document] = []

        # If specific files are selected, only load those
        if selected_files:
            pdf_files = [Path(f) for f in selected_files if f.endswith('.pdf')]
            log_info(f"[RAG-INDEX] Loading {len(pdf_files)} selected PDF files")
        else:
            # Original behavior: load all PDFs from directory
            pdf_files = list(data_dir.rglob("*.pdf"))
            log_debug(f"[RAG-INDEX] Found {len(pdf_files)} PDF files in directory")
        
        for p in pdf_files:
            if not p.exists():
                log_warning(f"[RAG-INDEX] File not found: {p}")
                continue
            log_debug(f"[RAG-INDEX] Processing PDF: {p}")
            loader = PyPDFLoader(str(p))
            pdf_docs = loader.load()
            for d in pdf_docs:
                d.metadata = {**d.metadata, "source": str(p), "filetype": ".pdf", "page": d.metadata.get("page")}
            docs.extend(pdf_docs)
            log_debug(f"[RAG-INDEX] Loaded {len(pdf_docs)} pages from {p.name}")

        log_info(f"[RAG-INDEX] Total documents loaded: {len(docs)}")
        return docs

    def make_chunks(self, docs: List[Document]) -> List[Document]:
        """
        Chunking documents based on RecursiveCharacterTextSplitter.
        
        """
        log_info(f"[RAG-INDEX] Creating chunks from {len(docs)} documents")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHUNK_SIZE,
            chunk_overlap=self.CHUNK_OVERLAP,
            separators=self.SEPARATORS,
            add_start_index=True, 
        )
        chunks = splitter.split_documents(docs)
        
        log_info(f"[RAG-INDEX] Created {len(chunks)} chunks")
        log_debug(f"[RAG-INDEX] Chunk size: {self.CHUNK_SIZE}, Overlap: {self.CHUNK_OVERLAP}")
        
        return chunks

    def deterministic_id(self, text: str, metadata: dict) -> str:
        """
        Returns a stable, deterministic UUID v5 based on the chunk content
        and key metadata. Compliant with Qdrant requirements (UUID or int).
        
        """
        source = str(metadata.get("source", ""))
        page = str(metadata.get("page", ""))
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        name = f"{source}|{page}|{text_hash}"

        return str(uuid.uuid5(uuid.NAMESPACE_URL, name))

    def ensure_collection(self, client: QdrantClient, collection_name: str, vector_size: int, clear_existing: bool = True) -> None:
        """
        Creates or recreates a collection. By default, clears existing collection to ensure clean state.
        
        Args:
            client: Qdrant client
            collection_name: Name of the collection
            vector_size: Size of the embedding vectors
            clear_existing: If True, recreates collection even if it exists (default True for clean indexing)
        
        """
        import time
        log_info(f"[RAG-QDRANT] Checking collection: {collection_name}")
        
        exists = client.collection_exists(collection_name)
        
        if exists and not clear_existing:
            log_debug(f"[RAG-QDRANT] Collection {collection_name} already exists, keeping existing data")
            return
        
        if exists and clear_existing:
            log_info(f"[RAG-QDRANT] Collection exists, will delete and recreate for clean index")
            
            # Get current stats for logging
            try:
                info = client.get_collection(collection_name)
                log_info(f"[RAG-QDRANT] Current collection has {info.points_count} points, deleting...")
            except:
                pass
            
            # Delete the collection explicitly
            client.delete_collection(collection_name)
            log_info(f"[RAG-QDRANT] Collection {collection_name} deleted")
            
            # Small delay to ensure deletion is complete
            time.sleep(0.5)
            
            # Verify deletion
            if client.collection_exists(collection_name):
                log_warning(f"[RAG-QDRANT] Collection still exists after delete, forcing recreate")
                
        # Create or recreate the collection
        log_info(f"[RAG-QDRANT] Creating collection: {collection_name}")
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
        )
        
        # Verify the collection is empty
        time.sleep(0.2)  # Small delay for creation to complete
        try:
            info = client.get_collection(collection_name)
            if info.points_count > 0:
                log_warning(f"[RAG-QDRANT] New collection has {info.points_count} points, should be 0!")
                # Try one more time with delete
                client.delete_collection(collection_name)
                time.sleep(0.5)
                client.recreate_collection(
                    collection_name=collection_name,
                    vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
                )
            else:
                log_info(f"[RAG-QDRANT] Collection created successfully with 0 points")
        except Exception as e:
            log_warning(f"[RAG-QDRANT] Could not verify collection: {e}")
            
        log_info(f"[RAG-QDRANT] Collection {'recreated' if exists else 'created'} and verified")
    
    # ============= MAIN POPULATE FUNCTION =============
    
    def populate_vector_store(self, selected_files: List[str] = None) -> Dict[str, Any]:
        """
        Populate the vector store with documents from the data directory
                
        Returns:
            Dictionary with indexing statistics
        """
        log_info("[RAG] Starting vector store population")
        
        try:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                log_error("[RAG] OPENAI_API_KEY not set")
                raise RuntimeError("Setup OPENAI_API_KEY.")

            embeddings = OpenAIEmbeddings(model=self.EMBEDDING_MODEL, api_key=openai_api_key)
            log_debug(f"[RAG] Using embedding model: {self.EMBEDDING_MODEL}")

            if self.EMBEDDING_MODEL == "text-embedding-3-large":
                dim = 3072
            elif self.EMBEDDING_MODEL == "text-embedding-3-small":
                dim = 1536
            else:
                raise RuntimeError("Setup embeddings shape.")
            
            log_debug(f"[RAG] Embedding dimension: {dim}")

            client = self.get_qdrant_client()

            # Load pdfs
            if not self.DATA_DIR.exists():
                log_error(f"[RAG] Directory {self.DATA_DIR} does not exist")
                raise RuntimeError(f"Directory {self.DATA_DIR} does not exist. Create {self.DATA_DIR} and add some files!")

            base_docs = self.load_documents(self.DATA_DIR, selected_files)
            if not base_docs:
                log_warning(f"[RAG] Empty directory {self.DATA_DIR}")
                raise RuntimeError(f"Empty directory {self.DATA_DIR}. Add some .pdf files.")

            # Chunking
            chunks = self.make_chunks(base_docs)

            # Idempotentne IDs
            ids = [self.deterministic_id(doc.page_content, doc.metadata) for doc in chunks]
            log_debug(f"[RAG] Generated {len(ids)} deterministic IDs")

            # Setup Qdrant collection
            self.ensure_collection(client, self.COLLECTION_NAME, dim)
            
            # Force client reconnection after collection recreation to avoid cached state
            self.client = None  # Clear cached client
            client = self.get_qdrant_client()  # Get fresh client
            log_debug("[RAG] Forced client reconnection after collection setup")
            
            vectorstore = QdrantVectorStore(
                client=client,
                collection_name=self.COLLECTION_NAME,
                embedding=embeddings,
            )
            self.vectorstore = vectorstore

            # Verify collection is empty before upload
            try:
                pre_info = client.get_collection(self.COLLECTION_NAME)
                log_info(f"[RAG] Collection state before upload: {pre_info.points_count} points")
                if pre_info.points_count > 0:
                    log_warning(f"[RAG] WARNING: Collection not empty before upload! Has {pre_info.points_count} points")
            except Exception as e:
                log_warning(f"[RAG] Could not check collection state: {e}")
            
            # Save chunks to Qdrant
            log_info(f"[RAG] Starting batch upload of {len(chunks)} chunks from {len(base_docs)} pages")
            for i in range(0, len(chunks), self.WRITE_BATCH_SIZE):
                batch_docs = chunks[i : i + self.WRITE_BATCH_SIZE]
                batch_ids = ids[i : i + self.WRITE_BATCH_SIZE]
                vectorstore.add_documents(batch_docs, ids=batch_ids)
                log_debug(f"[RAG] Uploaded batch {i//self.WRITE_BATCH_SIZE + 1}/{(len(chunks)-1)//self.WRITE_BATCH_SIZE + 1}")

            # Verify final count
            try:
                post_info = client.get_collection(self.COLLECTION_NAME)
                log_info(f"[RAG] Collection state after upload: {post_info.points_count} points (expected: {len(chunks)})")
                if post_info.points_count != len(chunks):
                    log_warning(f"[RAG] WARNING: Point count mismatch! Expected {len(chunks)}, got {post_info.points_count}")
            except Exception as e:
                log_warning(f"[RAG] Could not verify final collection state: {e}")
                
            log_info(f"[RAG] ✅ Finished. Collection: {self.COLLECTION_NAME}, new vectors in collection: {len(chunks)}")
            
            return {
                "success": True,
                "documents_processed": len(base_docs),
                "chunks_created": len(chunks),
                "collection_name": self.COLLECTION_NAME,
                "embedding_model": self.EMBEDDING_MODEL
            }
            
        except Exception as e:
            log_error(f"[RAG-ERROR] Failed to populate vector store: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    # ============= RAG QUERY FUNCTIONS =============
    
    def _format_docs(self, docs: List[Document]) -> str:
        """Combines chunks into a single context with short [i] tags corresponding to sources.
        """
        parts: List[str] = []
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("source", "unknown")
            page = d.metadata.get("page")
            tag = f"{src}" + (f":{page}" if page is not None else "")
            parts.append(f"{d.page_content}\n[[{i}] {tag}]")
        return "\n\n---\n\n".join(parts)

    def _pretty_sources(self, docs_with_scores: List[Tuple[Document, float]], topk: int = 4) -> List[str]:
        """Returns a unique list of sources (file[:page]) with score.
        """
        out: List[str] = []
        seen = set()
        for doc, score in docs_with_scores:
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page")
            label = f"{src}" + (f":{page}" if page is not None else "")
            if label in seen:
                continue
            seen.add(label)
            out.append(f"{label} (score={score:.4f})")
            if len(out) >= topk:
                break
        return out

    def _build_vectorstore(self) -> QdrantVectorStore:
        """Creates a VectorStore on an existing Qdrant collection.
        """
        log_debug("[RAG-QUERY] Building vectorstore")
        
        client = self.get_qdrant_client()
        embeddings = OpenAIEmbeddings(
            model=self.EMBEDDING_MODEL, 
            api_key=os.environ.get("OPENAI_API_KEY")
        )
        vs = QdrantVectorStore(
            client=client, 
            collection_name=self.COLLECTION_NAME, 
            embedding=embeddings
        )
        self.vectorstore = vs
        return vs

    def build_retriever(
        self,
        mode: str = "mmr",
        k: int = 6,
        **kwargs
    ):
        """
        Creates a retriever from QdrantVectorStore.
        mode: "mmr" or "similarity"
        k: number of chunks for the context
        **kwargs: e.g. lambda_mult, fetch_k, filter (Qdrant Filter)
        """
        log_debug(f"[RAG-QUERY] Building retriever (mode={mode}, k={k})")
        
        vs = self._build_vectorstore()
        if mode == "mmr":
            search_kwargs = {"k": k, "fetch_k": max(10, 3 * k), "lambda_mult": kwargs.pop("lambda_mult", 0.5)}
        else:
            search_kwargs = {"k": k}
        search_kwargs.update(kwargs)
        
        return vs.as_retriever(search_type=mode, search_kwargs=search_kwargs)

    def _build_prompt_and_chain(self, retriever) -> Any:
        """Builds a simple RAG chain: retriever → prompt → LLM → text.
        Supports multiple LLM providers (OpenAI, Anthropic, Google).
        """
        log_debug(f"[RAG-QUERY] Building prompt and chain with model: {self.CHAT_MODEL}")
        
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "You are a helpful assistant. Answer concisely using ONLY the provided context. "
                 "If the answer is not in the context, say you don't know. "
                 "Use short citations like [1], [2] that correspond to the provided chunks."),
                ("human", "Question: {question}\n\nContext:\n{context}")
            ]
        )
        
        # Select appropriate LLM based on model type
        if 'gpt' in self.CHAT_MODEL.lower():
            log_debug("[RAG-QUERY] Using ChatOpenAI for GPT model")
            llm = ChatOpenAI(model=self.CHAT_MODEL, temperature=0)
        elif 'claude' in self.CHAT_MODEL.lower():
            log_debug("[RAG-QUERY] Using ChatAnthropic for Claude model")
            llm = ChatAnthropic(model=self.CHAT_MODEL, temperature=0, api_key=config.ANTHROPIC_API_KEY)
        elif 'gemini' in self.CHAT_MODEL.lower():
            log_debug("[RAG-QUERY] Using ChatGoogleGenerativeAI for Gemini model")
            # Gemini models in LangChain don't need the 'models/' prefix
            model_name = self.CHAT_MODEL.replace('models/', '')
            llm = ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=config.GOOGLE_API_KEY)
        else:
            # Default to OpenAI if model type not recognized
            log_warning(f"[RAG-QUERY] Unknown model type: {self.CHAT_MODEL}, defaulting to ChatOpenAI")
            llm = ChatOpenAI(model=self.CHAT_MODEL, temperature=0)
        
        chain = (
            {
                "context": retriever | self._format_docs,
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )
        return chain

    def rag_answer(
        self,
        question: str,
        mode: str = "mmr",
        k: int = 6,
        topk_sources: int = 4
    ) -> Tuple[str, List[str]]:
        """Returns (answer, sources_list).
        """
        log_info(f"[RAG-QUERY] Processing question: {question}")
        log_debug(f"[RAG-QUERY] Parameters: mode={mode}, k={k}, topk_sources={topk_sources}")
        
        try:
            vs = self._build_vectorstore()
            retriever = self.build_retriever(mode=mode, k=k)
            chain = self._build_prompt_and_chain(retriever)
            
            log_debug("[RAG-QUERY] Retrieving relevant documents")
            start_time = os.times().elapsed
            
            answer = chain.invoke(question)
            
            elapsed = os.times().elapsed - start_time
            log_info(f"[RAG-QUERY] Answer generated in {elapsed:.2f}s")
            
            topk_with_scores = vs.similarity_search_with_score(question, k=max(topk_sources, k))
            sources = self._pretty_sources(topk_with_scores, topk=topk_sources)
            
            log_info(f"[RAG-QUERY] Retrieved {len(sources)} sources")
            log_debug(f"[RAG-QUERY] Sources: {sources}")
            
            return answer, sources
            
        except Exception as e:
            log_error(f"[RAG-ERROR] Failed to generate answer: {str(e)}")
            return f"Error: {str(e)}", []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the current collection"""
        log_debug("[RAG] Getting collection statistics")
        
        try:
            client = self.get_qdrant_client()
            
            # Check if collection exists
            if not client.collection_exists(self.COLLECTION_NAME):
                log_warning(f"[RAG] Collection {self.COLLECTION_NAME} does not exist")
                return {
                    "exists": False,
                    "collection_name": self.COLLECTION_NAME
                }
            
            # Get collection info
            collection_info = client.get_collection(self.COLLECTION_NAME)
            
            stats = {
                "exists": True,
                "collection_name": self.COLLECTION_NAME,
                "vectors_count": collection_info.vectors_count,
                "points_count": collection_info.points_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "config": {
                    "size": collection_info.config.params.vectors.size,
                    "distance": collection_info.config.params.vectors.distance
                }
            }
            
            log_info(f"[RAG] Collection stats: {stats['vectors_count']} vectors")
            return stats
            
        except Exception as e:
            log_error(f"[RAG-ERROR] Failed to get collection stats: {str(e)}")
            return {
                "exists": False,
                "error": str(e)
            }
    
    def test_connection(self) -> bool:
        """Test if Qdrant connection is working"""
        log_info("[RAG] Testing Qdrant connection")
        
        try:
            client = self.get_qdrant_client()
            collections = client.get_collections()
            log_info(f"[RAG] Connection successful. Found {len(collections.collections)} collections")
            return True
        except Exception as e:
            log_error(f"[RAG-ERROR] Connection failed: {str(e)}")
            return False


# ============= STREAMLIT INTEGRATION HELPER FUNCTIONS =============

def init_rag_system(chat_model=None) -> ArkadiuszRAGSystem:
    """Initialize RAG system for Streamlit
    
    Args:
        chat_model: Optional model name to use for chat. If not provided, uses config default.
    """
    # Check if we need to reinitialize due to model change
    if 'rag_system' in st.session_state and st.session_state.rag_system is not None:
        current_model = getattr(st.session_state.rag_system, 'CHAT_MODEL', None)
        if chat_model and current_model != chat_model:
            log_info(f"[RAG] Model changed from {current_model} to {chat_model}, reinitializing")
            st.session_state.rag_system = None
    
    if 'rag_system' not in st.session_state or st.session_state.rag_system is None:
        log_info("[RAG] Creating new RAG system instance")
        try:
            rag_system = ArkadiuszRAGSystem(chat_model=chat_model)
            st.session_state.rag_system = rag_system
            log_info(f"[RAG] RAG system instance created successfully with model: {rag_system.CHAT_MODEL}")
            return rag_system
        except Exception as e:
            log_error(f"[RAG] Failed to create RAG system instance: {str(e)}")
            st.session_state.rag_system = None
            raise
    else:
        log_debug("[RAG] Returning existing RAG system from session state")
        return st.session_state.rag_system

def get_rag_system() -> Optional[ArkadiuszRAGSystem]:
    """Get existing RAG system from session state"""
    return st.session_state.get('rag_system', None)