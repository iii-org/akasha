from typing import Dict, List, Any, Optional
from langchain.llms.base import LLM
from transformers import AutoTokenizer, AutoModel
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
import torch
import warnings,os
from akasha.models.llama2 import Llama2, TaiwanLLaMaGPTQ


class chatGLM(LLM):
    max_token : int = 4096
    temperature: float = 0.01
    top_p: float = 0.95
    history: list = []
    tokenizer: Any
    model: Any

    def __init__(self, model_name:str,temperature:float = 0.01):
        """define chatglm model and the tokenizer

        Args:
            **model_name (str)**: chatglm model name\n
        """
        if model_name == "":
            model_name = "THUDM/chatglm2-6b"
        
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code = True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code = True, device='cuda')
        self.temperature = temperature
        if self.temperature == 0.0:
            self.temperature = 0.01
    @property
    def _llm_type(self) -> str:
        """return llm type

        Returns:
            str: llm type
        """
        return "ChatGLM"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """run llm and get the response

        Args:
            **prompt (str)**: user prompt
            **stop (Optional[List[str]], optional)**: not use. Defaults to None.\n

        Returns:
            str: llm response
        """
        self.model = self.model.eval()
        response, history = self.model.chat(self.tokenizer, prompt, history=[])
        return response






def get_hf_model(model_name, temperature:float=0.0):
    """try different methods to define huggingface model, first use pipline and then use llama2.

    Args:
        model_name (str): huggingface model name\n

    Returns:
        _type_: llm model
    """
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        try:
            if hf_token is None:
                pipe = pipeline("text-generation", model=model_name,model_kwargs={"temperature":temperature,"repetition_penalty":1.2}, device_map="auto",\
                max_new_tokens = 512,batch_size = 1,torch_dtype=torch.float16)
                model = HuggingFacePipeline(pipeline=pipe)
            else:
            
                pipe = pipeline("text-generation", model=model_name, use_auth_token=hf_token,\
                max_new_tokens = 512, model_kwargs={"temperature":temperature,"repetition_penalty":1.2}, device_map="auto", batch_size = 1, torch_dtype=torch.float16)
                model = HuggingFacePipeline(pipeline=pipe)
        except:
                
            if model_name.lower().find("taiwan-llama")!=-1:
            
                model = TaiwanLLaMaGPTQ(model_name_or_path=model_name, temperature=temperature)
            else:
                
                model = Llama2(model_name_or_path=model_name,temperature=temperature, bit4=True, max_token=4096)
        
    return model
