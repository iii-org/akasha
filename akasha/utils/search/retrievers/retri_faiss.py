from typing import Any, List, Optional, Tuple
from pydantic import Field
import numpy as np
from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever, Document
from akasha.utils.db.db_structure import dbs
from langchain_community.retrievers import KNNRetriever
import faiss


class myFAISSRetriever(BaseRetriever):
    embeddings: Embeddings = Field(default=None)
    """Embeddings model to use."""
    index: faiss.IndexFlatL2 = Field(default=None)
    """Index of embeddings."""
    texts: List[str] = Field(default=None)
    """List of texts to index."""
    metadata: List[dict]
    k: int = 10
    """Number of results to return."""
    relevancy_threshold: Optional[float] = None

    @classmethod
    def from_db(
        cls,
        db: dbs,
        embeddings: Embeddings,
        k: int = 10,
        relevancy_threshold: float = 0.0,
        **kwargs: Any,
    ) -> KNNRetriever:
        emb = np.array(db.get_embeds()).astype("float32")
        dimention = emb.shape[1]
        index = faiss.IndexFlatL2(dimention)
        index.add(emb)  # add vectors to the index
        texts = db.get_docs()
        metadata = db.get_metadatas()
        return cls(
            embeddings=embeddings,
            index=index,
            texts=texts,
            metadata=metadata,
            k=k,
            relevancy_threshold=relevancy_threshold,
            **kwargs,
        )

    def _gs(self, query: str) -> Tuple[List[Document], List[float]]:
        """implement FAISS search to find relevant documents

        Args:
            **query (str)**: query string that used to find relevant documents\n

        Returns:
            List[Document]: relevant documents
        """
        query_embeds = np.array([self.embeddings.embed_query(query)]).astype("float32")
        # use index to search
        distances, indexes = self.index.search(
            query_embeds, min(self.k, len(self.texts))
        )

        top_k_scores = 1 / (1 + distances[0])
        top_k_results = [
            Document(page_content=self.texts[row], metadata=self.metadata[row])
            for row in indexes[0]
            if (
                self.relevancy_threshold is None
                or top_k_scores[row] >= self.relevancy_threshold
            )
        ]
        return top_k_results, top_k_scores

    def get_relevant_documents_and_scores(
        self,
        query: str,
    ) -> Tuple[List[Document], List[float]]:
        return self._gs(query)

    def _aget_relevant_documents(self, query: str) -> List[Document]:
        """implement FAISS search to find relevant documents

        Args:
            **query (str)**: query string that used to find relevant documents\n

        Returns:
            List[Document]: relevant documents
        """
        return self._gs(query)[0]

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self._gs(query)[0]
