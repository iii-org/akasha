from typing import Any, List, Optional, Tuple
from pydantic import Field
import numpy as np

from langchain.schema import BaseRetriever, Document

import jieba
from rank_bm25 import BM25Okapi


class myBM25Retriever(BaseRetriever):
    bm25: BM25Okapi = Field(default=None)
    """BM25 class to use."""
    texts: List[str] = Field(default=None)
    """List of texts to index."""
    metadata: List[dict] = Field(default=None)
    docs: List[Document] = Field(default=None)
    k: int = 3
    """Number of results to return."""
    relevancy_threshold: Optional[float] = None

    @classmethod
    def from_documents(
        cls,
        docs: List[Document],
        k: int = 3,
        relevancy_threshold: float = 0.0,
        **kwargs: Any,
    ) -> BaseRetriever:
        tokenize_corpus = [list(jieba.cut(doc.page_content)) for doc in docs]
        bm25 = BM25Okapi(tokenize_corpus)
        return cls(
            bm25=bm25,
            texts=[doc.page_content for doc in docs],
            metadata=[doc.metadata for doc in docs],
            k=k,
            docs=docs,
            relevancy_threshold=relevancy_threshold,
            **kwargs,
        )

    def _gs(self, query: str) -> Tuple[List[Document], List[float]]:
        """implement bm25 to find relevant documents

        Args:
            **query (str)**: query string that used to find relevant documents\n

        Returns:
            List[Document]: relevant documents
        """

        tokenize_query = list(jieba.cut(query))
        docs_scores = self.bm25.get_scores(tokenize_query)
        top_k_idx = np.argsort(docs_scores)[::-1][: self.k]
        top_k_results = [self.docs[i] for i in top_k_idx]
        top_k_scores = [docs_scores[i] for i in top_k_idx]
        return top_k_results, top_k_scores

    def get_relevant_documents_and_scores(
        self,
        query: str,
    ) -> Tuple[List[Document], List[float]]:
        return self._gs(query)

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._gs(query)[0]

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self._gs(query)[0]
