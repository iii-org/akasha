import time
from akasha.utils.atman import atman
from akasha.utils.prompts.gen_prompt import (
    default_self_ask_prompt,
    OutputSchema,
    JSON_formatter,
    format_sys_prompt,
)
from akasha.utils.prompts.format import language_dict
from akasha.helper.run_llm import call_model
from akasha.helper.base import extract_json


def self_ask_f(self: atman, start_time: float, timestamp: str) -> str:
    """implement the self ask rag function, first get the follow up questions by user prompt,
    then answer all follow up questions, then use the follow up information to answer the user prompt.

    Args:
        self (atman): _description_
        start_time (float): _description_
        timestamp (str): _description_

    Returns:
        _type_: _description_
    """
    final_prompt = self.prompt

    self_ask_prompt = default_self_ask_prompt() + "Question: " + final_prompt

    formatter = [
        OutputSchema(
            name="need",
            description="the 0 or 1 value that the question need follow up questions to answer or not.",
            type="bool",
        ),
        OutputSchema(
            name="follow_up",
            description="if need follow up questions, list of string of the follow up questions, else return empty list.",
            type="list",
        ),
    ]

    ## format sys prompt ##
    self_ask_sys_prompt = ""
    if "chinese" in language_dict[self.language]:
        self_ask_sys_prompt = "用中文回答 "
    self_ask_sys_prompt += JSON_formatter(formatter)
    prod_sys_prompt = format_sys_prompt(
        self_ask_sys_prompt, self_ask_prompt, self.prompt_format_type, self.model
    )

    ret = call_model(
        self.model_obj,
        prod_sys_prompt,
        self.verbose,
    )
    parse_json = extract_json(ret)

    if parse_json is None or int(parse_json["need"]) == 0:
        self.follow_up = []
        return self(self.db, self.prompt)

    else:
        self.follow_up = parse_json["follow_up"]
    ### start to get response ###

    inter_q = []
    inter_a = []
    tot_prompt_len, tot_prompt_tokens = self.prompt_length, self.prompt_tokens
    tot_doc_len, tot_doc_tokens = 0, 0

    for each_fol_up in self.follow_up:
        each_fol_ans = self(self.db, each_fol_up)
        inter_q.append(each_fol_up)
        inter_a.append(each_fol_ans)
        tot_prompt_len += self.prompt_length
        tot_prompt_tokens += self.prompt_tokens
        tot_doc_len += self.doc_length
        tot_doc_tokens += self.doc_tokens

    inter_info = get_inter_info(inter_q, inter_a)

    text_input = format_sys_prompt(
        self.system_prompt,
        inter_info + "User Question: " + final_prompt,
        self.prompt_format_type,
        self.model,
    )

    self.prompt_length = tot_prompt_len
    self.prompt_tokens = tot_prompt_tokens
    self.doc_length = tot_doc_len
    self.doc_tokens = tot_doc_tokens
    self.prompt = final_prompt

    end_time = time.time()
    if self.stream:
        return self._display_stream(
            text_input,
        )

    self.response = call_model(
        self.model_obj,
        text_input,
        self.verbose,
    )

    self._add_result_log(timestamp, end_time - start_time)

    self._upload_logs(end_time - start_time, self.doc_length, self.doc_tokens)

    return self.response


def get_inter_info(inter_q: list, inter_a: list) -> str:
    """format the follow up question and answer to a string

    Args:
        inter_q (_type_): _description_
        inter_a (_type_): _description_

    Returns:
        str: _description_
    """
    inter_info = []
    for i in range(len(inter_q)):
        inter_info.append(f"{inter_q[i]}  {inter_a[i]}")
    return (
        "Use follow information to answer User Question\n"
        + "\n----------------------\n".join(inter_info)
    )
