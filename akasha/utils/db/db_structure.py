from langchain.docstore.document import Document
from typing import Union
from pathlib import Path
import re


class dbs:
    def __init__(self, chrdb=[]):
        self.ids = []
        self.embeds = []
        self.metadatas = []
        self.docs = []
        self.vis = set()
        if isinstance(chrdb, list):
            pass

        else:
            data = chrdb.get(include=["embeddings", "metadatas", "documents"])
            if "ids" in data:
                self.ids = list(data["ids"])
                self.vis = set(self.ids)
            if "embeddings" in data and data["embeddings"] is not None:
                # Use tolist() if it's a numpy array, otherwise use list()
                self.embeds = getattr(data["embeddings"], "tolist", lambda: list(data["embeddings"]))()
            else:
                self.embeds = [[] for _ in range(len(self.ids))]
            if "metadatas" in data and data["metadatas"] is not None:
                self.metadatas = list(data["metadatas"])
            else:
                self.metadatas = [{} for _ in range(len(self.ids))]
            if "documents" in data and data["documents"] is not None:
                self.docs = list(data["documents"])
            else:
                self.docs = ["" for _ in range(len(self.ids))]

    def merge(self, db: "dbs"):
        for i in range(len(db.ids)):
            if db.ids[i] not in self.vis:
                self.ids.append(db.ids[i])
                self.embeds.append(db.embeds[i])
                self.metadatas.append(db.metadatas[i])
                self.docs.append(db.docs[i])
                self.vis.add(db.ids[i])
        # self.ids.extend(db.ids)
        # self.embeds.extend(db.embeds)
        # self.metadatas.extend(db.metadatas)
        # self.docs.extend(db.docs)

    def add_chromadb(self, chrdb):
        data = chrdb.get(include=["embeddings", "metadatas", "documents"])
        if "ids" in data:
            self.ids.extend(list(data["ids"]))

        if "embeddings" in data and data["embeddings"] is not None:
            self.embeds.extend(getattr(data["embeddings"], "tolist", lambda: list(data["embeddings"]))())
        else:
            self.embeds.extend([[] for _ in range(len(data["ids"]))])
        if "metadatas" in data and data["metadatas"] is not None:
            self.metadatas.extend(list(data["metadatas"]))
        else:
            self.metadatas.extend([{} for _ in range(len(data["ids"]))])
        if "documents" in data and data["documents"] is not None:
            self.docs.extend(list(data["documents"]))
        else:
            self.docs.extend(["" for _ in range(len(data["ids"]))])

    def get_Documents(self):
        return [
            Document(page_content=self.docs[i], metadata=self.metadatas[i])
            for i in range(len(self.docs))
        ]

    def get_docs(self):
        return self.docs

    def get_ids(self):
        return self.ids

    def get_metadatas(self):
        return self.metadatas

    def get_embeds(self):
        return self.embeds


NO_PARENT_DIR_NAME = "NoPaReNtDiR"
FILE_LAST_CHANGE_FILE_NAME = "file_last_changed.json"
TEXT_EXTENSIONS = ["pdf", "md", "docx", "txt", "csv", "pptx"]
ALREADY_BUILT = "already_built"
NOT_BUILT = "not_built"
OLD_BUILT = "old_built"
HNSW_THRESHOLD = 400000


def get_storage_directory(
    dir_path: Union[Path, str],
    chunk_size: int,
    embed_type: str,
    embed_name: str,
) -> str:
    if isinstance(dir_path, str):
        if is_url(dir_path):
            sanitized_dir_path = _sanitize_path_string(dir_path)
            storage_directory = (
                "chromadb/"
                + sanitized_dir_path
                + "_"
                + embed_type
                + "_"
                + embed_name.replace("/", "-")
                + "_"
                + str(chunk_size)
            )
            return storage_directory
        else:
            dir_path = Path(dir_path)

    if dir_path != Path("."):
        db_dir = "-".join(
            [part.replace(" ", "").replace("_", "") for part in dir_path.parts if part]
        )
    else:
        db_dir = NO_PARENT_DIR_NAME

    storage_directory = (
        "chromadb/"
        + db_dir
        + "_"
        + embed_type
        + "_"
        + embed_name.replace("/", "-")
        + "_"
        + str(chunk_size)
    )

    return storage_directory


def _sanitize_path_string(path: str, max_len: int = 30) -> str:
    # Remove special characters and limit length to 30
    sanitized = re.sub(r"[^a-zA-Z0-9]", "", path)[:max_len]
    return sanitized


def is_url(pattern: str) -> bool:
    return bool(re.match(r"^(http|https)://", pattern))
