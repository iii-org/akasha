from typing import Any, List, Optional, Tuple
from pydantic import Field
from langchain.schema import BaseRetriever, Document


class myRerankRetriever(BaseRetriever):
    model_name: str = "BAAI/bge-reranker-base"
    """rerank model to use."""
    texts: List[str] = Field(default=None)
    """List of texts to index."""
    metadata: List[dict] = Field(default=None)
    docs: List[Document] = Field(default=None)
    k: int = 20
    """Number of results to return."""
    relevancy_threshold: Optional[float] = None

    @classmethod
    def from_documents(
        cls,
        docs: List[Document],
        k: int = 20,
        relevancy_threshold: float = 0.0,
        model_name: str = "BAAI/bge-reranker-base",
        **kwargs: Any,
    ) -> BaseRetriever:
        return cls(
            model_name=model_name,
            texts=[doc.page_content for doc in docs],
            metadata=[doc.metadata for doc in docs],
            k=k,
            docs=docs,
            relevancy_threshold=relevancy_threshold,
            **kwargs,
        )

    def _gs(self, query: str) -> Tuple[List[Document], List[float]]:
        """implement rerank to find relevant documents

        Args:
            **query (str)**: query string that used to find relevant documents\n

        Returns:
            List[Document]: relevant documents
        """

        top_k_results, top_k_scores = rerank(
            query, self.docs, self.relevancy_threshold, self.model_name
        )

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


def rerank(query: str, docs: list, threshold: float, model_name: str):
    import torch
    import gc
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()

    k, score_list = 0, []
    while k < len(docs):
        pairs = [[query, doc.page_content] for doc in docs[k : k + 10]]
        with torch.no_grad():
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(device)
            scores = (
                model(**inputs, return_dict=True)
                .logits.view(
                    -1,
                )
                .float()
            )
        if k == 0:
            score_list = scores
        else:
            score_list = torch.cat([score_list, scores], dim=0)
        k += 10

    # Get the sorted indices in descending order
    sorted_indices = torch.argsort(score_list, descending=True)

    # Convert the indices to a Python list
    sorted_indices_list = sorted_indices.tolist()

    # Get the documents in the order of their scores, if lower than threshold, break
    documents = []
    for i in sorted_indices_list:
        if score_list[i] < threshold:
            break
        documents.append(docs[i])

    del model, inputs, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return documents, score_list[: len(documents)]


def rerank_reduce(query, docs, topK):
    import torch
    import gc
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_name = "BAAI/bge-reranker-large"  # BAAI/bge-reranker-base
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    # topK //= 2
    k, score_list = 0, []
    while k < len(docs):
        pairs = [[query, doc.page_content] for doc in docs[k : k + 10]]
        with torch.no_grad():
            inputs = tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(device)
            scores = (
                model(**inputs, return_dict=True)
                .logits.view(
                    -1,
                )
                .float()
            )
        if k == 0:
            score_list = scores
        else:
            score_list = torch.cat([score_list, scores], dim=0)
        k += 10

    # Get the sorted indices in descending order
    sorted_indices = torch.argsort(score_list, descending=True)

    # Convert the indices to a Python list
    sorted_indices_list = sorted_indices.tolist()

    # Get the documents in the order of their scores, if lower than threshold, break
    documents = []
    for i in sorted_indices_list:
        if i > topK:
            break
        documents.append(docs[i])

    del model, inputs, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return documents
