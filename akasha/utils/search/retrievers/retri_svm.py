from langchain_community.retrievers import SVMRetriever
from typing import Any, List, Optional, Tuple
from pydantic import Field
import numpy as np
from langchain.embeddings.base import Embeddings
from langchain.schema import BaseRetriever, Document
from akasha.utils.db.db_structure import dbs


class mySVMRetriever(BaseRetriever):
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
    ) -> SVMRetriever:
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
        """implement svm to find relevant documents

        Args:
            **query (str)**: query string that used to find relevant documents\n

        Returns:
            List[Document]: relevant documents
        """
        try:
            from sklearn import svm
        except ImportError:
            raise ImportError(
                "Could not import scikit-learn, please install with `pip install "
                "scikit-learn`."
            )

        query_embeds = np.array(self.embeddings.embed_query(query))
        x = np.concatenate([query_embeds[None, ...], self.index])
        y = np.zeros(x.shape[0])
        y[0] = 1

        clf = svm.LinearSVC(
            class_weight="balanced",
            verbose=False,
            max_iter=50000,
            tol=1e-4,
            C=0.1,
            dual=True,
        )
        clf.fit(x, y)

        similarities = clf.decision_function(x)
        sorted_ix = np.argsort(-similarities)

        # svm.LinearSVC in scikit-learn is non-deterministic.
        # if a text is the same as a query, there is no guarantee
        # the query will be in the first index.
        # this performs a simple swap, this works because anything
        # left of the 0 should be equivalent.
        zero_index = np.where(sorted_ix == 0)[0][0]
        if zero_index != 0:
            sorted_ix[0], sorted_ix[zero_index] = sorted_ix[zero_index], sorted_ix[0]

        denominator = np.max(similarities) - np.min(similarities) + 1e-6
        normalized_similarities = (similarities - np.min(similarities)) / denominator

        top_k_results = []
        top_k_scores = []
        for row in sorted_ix[1 : self.k + 1]:
            # print(normalized_similarities[row])  # stats
            if (
                self.relevancy_threshold is None
                or normalized_similarities[row] >= self.relevancy_threshold
            ):
                top_k_results.append(
                    Document(
                        page_content=self.texts[row - 1],
                        metadata=self.metadata[row - 1],
                    )
                )
                top_k_scores.append(normalized_similarities[row])
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
