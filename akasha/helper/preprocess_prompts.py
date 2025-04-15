from typing import Union
from akasha.utils.prompts.gen_prompt import (
    format_sys_prompt,
    format_history_prompt,
    decide_auto_prompt_format_type,
)
from typing import Tuple
from .token_counter import myTokenizer


def merge_history_and_prompt(
    history_messages: list,
    system_prompt: str,
    prompt: str,
    prompt_format_type: str = "auto",
    user_tag: str = "user",
    assistant_tag: str = "assistant",
    model: str = "remote:xxx",
) -> Union[str, list]:
    """merge system prompt, history messages, and prompt based on the prompt format type, if history_messages is empty, return the prompt
    if prompt_format_type is start with "chat_", it will become list of dictionary, otherwise it will become string

    Args:
        history_messages (list): _description_
        system_prompt (str): _description_
        prompt (str): _description_
        prompt_format_type (str, optional): _description_. Defaults to "gpt".
        user_tag (str, optional): _description_. Defaults to "user".
        assistant_tag (str, optional): _description_. Defaults to "assistant".
        model (_type_, optional): _description_. Defaults to "remote:xxx".

    Returns:
        Union[str, list]: _description_
    """
    ### decide prompt format type if auto###
    if prompt_format_type == "auto":
        prompt_format_type = decide_auto_prompt_format_type(model)

    ### if history_messages is empty, return the prompt ###
    if history_messages == [] or history_messages is None or history_messages == "":
        return format_sys_prompt(system_prompt, prompt, prompt_format_type)

    if "chat_" in prompt_format_type and prompt_format_type != "chat_gemma":
        if prompt_format_type == "chat_gemini":
            assistant_tag = "model"

        text_input = format_sys_prompt(system_prompt, "", prompt_format_type)

        prod_prompt = format_sys_prompt("", prompt, prompt_format_type)

        prod_history = format_history_prompt(
            history_messages, prompt_format_type, user_tag, assistant_tag
        )

        text_input.extend(prod_history)

        text_input.extend(prod_prompt)

        return text_input

    else:
        history_str = ""

        for i in range(len(history_messages)):
            if i % 2 == 0:
                history_str += user_tag + ": " + history_messages[i] + "\n"

            else:
                history_str += assistant_tag + ": " + history_messages[i] + "\n"

        history_str += "\n\n"
        return format_sys_prompt(
            system_prompt, history_str + prompt, prompt_format_type
        )


def retri_history_messages(
    messages: list,
    pairs: int = 10,
    max_input_tokens: int = 1500,
    model_name: str = "openai:gpt-3.5-turbo",
    role1: str = "User",
    role2: str = "Assistant",
) -> Tuple[str, int]:
    """from messages dict list, get pairs of user question and assistant response from most recent and not exceed max_input_tokens and pairs, and return the text with total length

    Args:
        messages (list): history messages list, each index is a dict with keys: role("user", "assistant"), content(content of message)
        pairs (int, optional): the maximum number of messages. Defaults to 10.
        max_input_tokens (int, optional): the maximum number of messages tokens. Defaults to 1500.
        model_name (str, optional): model name. Defaults to "openai:gpt-3.5-turbo".
        role1 (str, optional): role1 name. Defaults to "User".
        role2 (str, optional): role2 name. Defaults to "Assistant".

    Returns:
        Tuple[str, int]: return the text with total length.
    """
    cur_len = 0
    count = 0
    ret = []
    splitter = "\n----------------\n"

    for i in range(len(messages) - 1, -1, -2):
        if count >= pairs:
            break
        if (messages[i]["role"] != role2) or (messages[i - 1]["role"] != role1):
            i += 1
            continue
        texta = f"{role2}: " + messages[i]["content"].replace("\n", "") + "\n"
        textq = f"{role1}: " + messages[i - 1]["content"].replace(
            "\n", ""
        )  # {(i+1)//2}.
        len_texta = myTokenizer.compute_tokens(texta, model_name)
        len_textq = myTokenizer.compute_tokens(textq, model_name)

        if cur_len + len_texta > max_input_tokens:
            break
        cur_len += len_texta
        ret.append(texta)

        if cur_len + len_textq > max_input_tokens:
            break
        cur_len += len_textq
        ret.append(textq)

        count += 1

    if count == 0:
        return "", 0

    ret.reverse()
    ret_str = splitter + "".join(ret) + splitter

    return ret_str, cur_len
