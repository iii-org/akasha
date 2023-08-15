from typing import Dict, List, Any, Optional
from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModel
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
import torch
import warnings,os
from akasha.models.llama2 import Llama2


class chatGLM(LLM):
    max_token : int = 4096
    temperature: float = 0.1
    top_p: float = 0.95
    history: list = []
    tokenizer: Any
    model: Any

    def __init__(self, model_name:str):
        if model_name == "":
            model_name = "THUDM/chatglm2-6b"
        
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code = True, device='cuda')

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        self.model = self.model.eval()
        response, history = self.model.chat(self.tokenizer, prompt, history=[])
        return response






def get_hf_model(model_name):

    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        if hf_token is None:
            pipe = pipeline("text-generation", model=model_name,model_kwargs={"temperature":0}, device_map="auto")
            model = HuggingFacePipeline(pipeline=pipe)
        else:
            try:
                pipe = pipeline("text-generation", model=model_name, use_auth_token=hf_token,\
                max_new_tokens = 512, model_kwargs={"temperature":0,}, device_map="auto", batch_size = 1, torch_dtype=torch.float16)
                model = HuggingFacePipeline(pipeline=pipe)
            except:
                
                model = Llama2(model_name_or_path=model_name, bit4=True, max_token=4096)
                #model = HuggingFaceHub(
            #    repo_id = model_name, model_kwargs={"temperature": 0.1})
        
    return model
