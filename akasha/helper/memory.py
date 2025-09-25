import json
import logging
from pathlib import Path
from typing import List, Tuple, Union
from datetime import datetime
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from akasha.helper.run_llm import call_model
from akasha.helper import handle_embeddings, handle_model_type, handle_model
from langchain_chroma import Chroma
from chromadb.config import Settings
from akasha.helper import get_mac_address, separate_name
from akasha.helper.handle_objects import handle_embeddings_and_name
from akasha.utils.db.db_structure import (
    get_storage_directory,
    HNSW_THRESHOLD,
    FILE_LAST_CHANGE_FILE_NAME,
    dbs,
)
from akasha.utils.db.create_db import create_directory_db
from akasha.utils.search.retrievers.base import get_retrivers
from akasha.utils.search.search_doc import retri_docs
from akasha.utils.db.load_db import load_directory_db
from akasha.utils.prompts.gen_prompt import (
    default_extract_memory_prompt,
    default_categorize_memory_prompt,
)
from akasha.utils.base import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBED,
    DEFAULT_MODEL,
)


class MemoryManager:
    """
    Manages the creation, storage, and retrieval of long-term semantic memory.

    This manager extracts salient information from conversations, categorizes it,
    and stores it in human-readable Markdown files. It also provides a method
    to search the corresponding vector store, which is assumed to be kept in
    sync with the Markdown files by an external process.
    """

    def __init__(
        self,
        memory_name: str,
        model: Union[str, BaseLanguageModel] = DEFAULT_MODEL,
        embeddings: Union[str, Embeddings] = DEFAULT_EMBED,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        verbose: bool = False,
        memory_dirname: str = "docs",
    ):
        """
        Initializes the MemoryManager.

        Args:
            model_obj (BaseLanguageModel): The language model object for processing.
            embeddings (Embeddings): The embedding model for the vector store.
            chunk_size (int): Size of text chunks for embedding.
            memory_name (str): Directory to store memory Markdown files.
            db_path (str): Path to the ChromaDB for searching memories.
            verbose (bool): Whether to print detailed logs.
        """
        self.model_obj = handle_model(model, verbose, 1.0, 2048)
        self.model = handle_model_type(model)
        self.embeddings_obj = handle_embeddings(embeddings, verbose)
        self.embeddings = handle_model_type(embeddings)
        self.chunk_size = chunk_size
        self.memory_name = memory_name
        self.mem_dir_path = Path(memory_dirname) / memory_name
        self.verbose = verbose

        if not self.mem_dir_path.exists():
            self.mem_dir_path.mkdir(parents=True, exist_ok=True)
            hello_file = self.mem_dir_path / "hello memory.md"
            hello_file.write_text("# Hello Memory\n\nThis is a new memory file.")

        # create & load directory db
        suc, mis = create_directory_db(
            directory_path=self.mem_dir_path,
            embeddings=self.embeddings_obj,
            chunk_size=self.chunk_size,
        )
        self.db = load_directory_db(
            directory_path=self.mem_dir_path,
            embeddings=self.embeddings_obj,
            chunk_size=self.chunk_size,
        )

        embeddings, embeddings_name = handle_embeddings_and_name(
            self.embeddings, False, ""
        )
        embed_type, embed_name = separate_name(embeddings_name)

        self.storage_dir = get_storage_directory(
            self.mem_dir_path, self.chunk_size, embed_type, embed_name
        )
        client_settings = Settings(
            is_persistent=True,
            persist_directory=self.storage_dir,
            anonymized_telemetry=False,
        )
        self.chroma = Chroma(
            persist_directory=self.storage_dir,
            embedding_function=embeddings,
            client_settings=client_settings,
            collection_metadata={"hnsw:sync_threshold": HNSW_THRESHOLD},
        )

    def _extract_salient_info(
        self, user_prompt: str, ai_response: str, language: str = "ch"
    ) -> str:
        """Uses an LLM to extract key information from a conversation turn."""
        if self.verbose:
            print("\n[Memory] Extracting salient information...")

        conversation_context = f"User asks: {user_prompt}\nAI responds: {ai_response}"
        extraction_prompt = default_extract_memory_prompt(language)

        extracted_memory = call_model(
            self.model_obj,
            "System: " + extraction_prompt + "\n\nHuman: " + conversation_context,
            verbose=self.verbose,
        )

        if "none" in extracted_memory.lower() or "無" in extracted_memory:
            if self.verbose:
                print("[Memory] No salient information found.")
            return ""

        if self.verbose:
            print(f"[Memory] Extracted: {extracted_memory}")
        return extracted_memory

    def _categorize_memory(self, memory_text: str, language: str = "ch") -> str:
        """Uses an LLM to determine a suitable topic for the memory."""
        if self.verbose:
            print(f"\n[Memory] Categorizing memory: '{memory_text[:50]}...'")

        categorization_prompt = default_categorize_memory_prompt(language)

        category = call_model(
            self.model_obj,
            "System: " + categorization_prompt + "\n\nHuman: " + memory_text,
            verbose=self.verbose,
        ).strip()

        # Sanitize category to be a valid filename
        sanitized_category = "".join(
            c for c in category if c.isalnum() or c in (" ", "_")
        ).rstrip()

        if self.verbose:
            print(f"[Memory] Categorized as: {sanitized_category}")
        return sanitized_category if sanitized_category else "uncategorized"

    def add_memory(self, user_prompt: str, ai_response: str, language: str = "ch"):
        """
        The main pipeline to process a conversation turn and save it to memory.
        """
        # 1. Extract important information
        extracted_info = self._extract_salient_info(user_prompt, ai_response, language)
        if not extracted_info:
            return

        # 2. Categorize the information
        category = self._categorize_memory(extracted_info, language)

        # 3. Save to a Markdown file
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        memory_entry = (
            f"- **Create Time: ** {timestamp}\n"
            f"- **Memory: ** {extracted_info}\n"
            f"- **Source Prompt: ** {user_prompt}\n\n---\n\n"
        )

        file_path = Path(self.mem_dir_path) / f"{category}.md"

        try:
            with file_path.open("a", encoding="utf-8") as f:
                f.write(memory_entry)
            if self.verbose:
                print(f"[Memory] Saved memory to '{file_path}'")
        except IOError as e:
            print(f"[Memory] Error saving memory to file: {e}")

        # 4. Update the vector store
        formatted_date = datetime.now().strftime("%Y-%m-%d-%H_%M_%S_%f")
        self.chroma.add_texts(
            [extracted_info],
            [{"source": str(file_path), "page": 0}],
            ids=[formatted_date + "_" + get_mac_address()],
        )
        self.db.merge(dbs(self.chroma))

        # Update db json file edit date
        last_m_time = file_path.stat().st_mtime
        file_name = file_path.name
        json_file_path = Path(self.storage_dir) / FILE_LAST_CHANGE_FILE_NAME

        try:
            with open(json_file_path, "r", encoding="utf-8") as json_file:
                file_last_changed = json.load(json_file)
        except Exception as e:
            logging.warning(f"Error reading last edit time JSON file: {e}")
            print(f"Error reading last edit time JSON file: {e}")
            return

        file_last_changed[file_name] = last_m_time
        try:
            with open(json_file_path, "w", encoding="utf-8") as json_file:
                json.dump(file_last_changed, json_file, ensure_ascii=False, indent=4)
        except Exception as e:
            logging.warning(f"Error writing last edit time JSON file: {e}")
            print(f"Error writing last edit time JSON file: {e}")
        return

    def search_memory(self, query: str, top_k: int = 3) -> List[str]:
        """
        Searches the vector store for memories relevant to the query.
        """
        if not self.db:
            print("[Memory] Memory database not available for search.")
            return []

        retrivers_list = get_retrivers(
            self.db,
            self.embeddings_obj,
            0.0,
            "faiss",
            "",
        )

        searched_docs = retri_docs(
            retrivers_list,
            query,
            "faiss",
            top_k,
        )

        results = [doc.page_content for doc in searched_docs]

        if self.verbose:
            print(f"[Memory] Found {len(results)} relevant memories.")
        return results

    def show_memory(self, num: int = 100) -> List[str]:
        # , self.db.get_ids(), self.db.get_metadatas()

        return self.db.get_docs()[:num]
