from typing import Any, List, Optional, Tuple
from pydantic import Field
import numpy as np
from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever, Document
from akasha.utils.db.db_structure import dbs
from langchain_community.utils.math import cosine_similarity


class myMMRRetriever(BaseRetriever):
    embeddings: Embeddings = Field(default=None)
    """Embeddings model to use."""
    index: Any = Field(default=None)
    """Index of embeddings."""
    texts: List[str] = Field(default=None)
    """List of texts to index."""
    metadata: List[dict] = Field(default=None)
    k: int = 3
    """Number of results to return."""
    relevancy_threshold: Optional[float] = None
    log: dict = {}
    lambda_mult: float = 0.5

    @classmethod
    def from_db(
        cls,
        db: dbs,
        embeddings: Embeddings,
        k: int = 3,
        relevancy_threshold: float = 0.0,
        log: dict = {},
        lambda_mult: float = 0.5,
    ):
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
            log=log,
            lambda_mult=lambda_mult,
        )

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self._gs(query)[0]

    def _gs(self, query: str) -> Tuple[List[Document], List[float]]:
        """implement using custom function to find relevant documents, the custom function func should
        have four input.
            1. a np.array of embedding vectors of query query_embeds np.array)
            2. a np.array of np.array contain embedding vectors of query and documents docs_embeds (np.array)
            3. the number of topK return documents k (int)
            4. relevant threshold from 0.0 ~ 1.0 threshold (float)
        And the function func should return a list of index which length is equals to k, represent
        the index of documents that are most relevant to the input query.

        Args:
            **query (str)**: query string that used to find relevant documents\n

        Returns:
            List[Document]: relevant documents
        """

        top_k_results = []
        query_embeds = np.array(self.embeddings.embed_query(query))
        docs_embeds = self.index

        if min(self.k, len(docs_embeds)) <= 0:
            return [], []
        if query_embeds.ndim == 1:
            query_embeds = np.expand_dims(query_embeds, axis=0)
        similarity_to_query = cosine_similarity(query_embeds, docs_embeds)[0]
        most_similar = int(np.argmax(similarity_to_query))
        mmr_scores = [similarity_to_query[most_similar]]
        relevant_docs_idx = [most_similar]
        selected = np.array([docs_embeds[most_similar]])
        while len(relevant_docs_idx) < min(self.k, len(docs_embeds)):
            best_score = -np.inf
            idx_to_add = -1
            similarity_to_selected = cosine_similarity(docs_embeds, selected)
            for i, query_score in enumerate(similarity_to_query):
                if i in relevant_docs_idx:
                    continue
                redundant_score = max(similarity_to_selected[i])
                equation_score = (
                    self.lambda_mult * query_score
                    - (1 - self.lambda_mult) * redundant_score
                )
                if equation_score > best_score:
                    best_score = equation_score
                    idx_to_add = i
            relevant_docs_idx.append(idx_to_add)
            mmr_scores.append(best_score)
            selected = np.append(selected, [docs_embeds[idx_to_add]], axis=0)

        ### from index rebuild the documents ###
        for idx in relevant_docs_idx[: self.k]:
            top_k_results.append(
                Document(page_content=self.texts[idx], metadata=self.metadata[idx])
            )

        return top_k_results, mmr_scores[: self.k]

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._gs(query)[0]

    def get_relevant_documents_and_scores(
        self,
        query: str,
    ) -> Tuple[List[Document], List[float]]:
        return self._gs(query)
