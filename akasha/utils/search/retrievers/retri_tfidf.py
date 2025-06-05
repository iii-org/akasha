from typing import Any, List, Optional, Tuple, Dict, Iterable
from langchain_community.retrievers import TFIDFRetriever
from langchain.schema import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
import jieba


class myTFIDFRetriever(TFIDFRetriever):
    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        documents: Optional[List[Document]] = None,
        tfidf_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TFIDFRetriever:
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError:
            raise ImportError(
                "Could not import scikit-learn, please install with `pip install "
                "scikit-learn`."
            )

        tfidf_params = tfidf_params or {}
        vectorizer = TfidfVectorizer(**tfidf_params)
        tfidf_array = vectorizer.fit_transform(texts)
        return cls(
            vectorizer=vectorizer, docs=documents, tfidf_array=tfidf_array, **kwargs
        )

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        *,
        tfidf_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TFIDFRetriever:
        texts, metadatas = zip(
            *(
                (" ".join(list(jieba.cut(d.page_content))), d.metadata)
                for d in documents
            )
        )
        return cls.from_texts(
            texts=texts, tfidf_params=tfidf_params, documents=documents, **kwargs
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        return self._gs(query)[0]

    def _gs(
        self,
        query: str,
    ) -> Tuple[List[Document], List[float]]:
        from sklearn.metrics.pairwise import cosine_similarity

        query_vec = self.vectorizer.transform(
            [query]
        )  # Ip -- (n_docs,x), Op -- (n_docs,n_Feats)
        results = cosine_similarity(self.tfidf_array, query_vec).reshape(
            (-1,)
        )  # Op -- (n_docs,1) -- Cosine Sim with each doc
        idxs = results.argsort()[-self.k :][::-1]
        return_docs = [self.docs[i] for i in idxs]
        return_values = [results[i] for i in idxs]

        return return_docs, return_values

    def get_relevant_documents_and_scores(
        self,
        query: str,
    ) -> Tuple[List[Document], List[float]]:
        return self._gs(query)

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._gs(query)[0]
