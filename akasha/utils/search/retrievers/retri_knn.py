from typing import Any, List, Optional, Tuple
from pydantic import Field
import numpy as np
from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever, Document
from akasha.utils.db.db_structure import dbs
from langchain_community.retrievers import KNNRetriever


class myKNNRetriever(BaseRetriever):
    embeddings: Embeddings = Field(default=None)
    """Embeddings model to use."""
    index: Any = Field(default=None)
    """Index of embeddings."""
    texts: List[str] = Field(default=None)
    """List of texts to index."""
    metadata: List[dict]
    k: int = 3
    """Number of results to return."""
    relevancy_threshold: Optional[float] = None

    @classmethod
    def from_db(
        cls,
        db: dbs,
        embeddings: Embeddings,
        k: int = 3,
        relevancy_threshold: float = 0.0,
        **kwargs: Any,
    ) -> KNNRetriever:
        # db_data = _get_all_docs(db)
        index = np.array(db.get_embeds())
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
        """implement k-means search to find relevant documents

        Args:
            **query (str)**: query string that used to find relevant documents\n

        Returns:
            List[Document]: relevant documents
        """
        query_embeds = np.array(self.embeddings.embed_query(query))
        # calc L2 norm
        index_embeds = self.index / np.sqrt((self.index**2).sum(1, keepdims=True))
        query_embeds = query_embeds / np.sqrt((query_embeds**2).sum())

        similarities = index_embeds.dot(query_embeds)
        sorted_ix = np.argsort(-similarities)

        denominator = np.max(similarities) - np.min(similarities) + 1e-6
        normalized_similarities = (similarities - np.min(similarities)) / denominator
        # print([normalized_similarities[row]
        #        for row in sorted_ix[0:self.k]])  # stats

        top_k_scores = [normalized_similarities[row] for row in sorted_ix[0 : self.k]]
        top_k_results = [
            Document(page_content=self.texts[row], metadata=self.metadata[row])
            for row in sorted_ix[0 : self.k]
            if (
                self.relevancy_threshold is None
                or normalized_similarities[row] >= self.relevancy_threshold
            )
        ]
        return top_k_results, top_k_scores

    def get_relevant_documents_and_scores(
        self,
        query: str,
    ) -> Tuple[List[Document], List[float]]:
        return self._gs(query)

    def _aget_relevant_documents(self, query: str) -> List[Document]:
        """implement k-means search to find relevant documents

        Args:
            **query (str)**: query string that used to find relevant documents\n

        Returns:
            List[Document]: relevant documents
        """
        return self._gs(query)[0]

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self._gs(query)[0]
