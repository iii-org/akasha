from typing import Union
from akasha.utils.prompts.gen_prompt import format_sys_prompt, format_history_prompt, decide_auto_prompt_format_type


def merge_history_and_prompt(history_messages: list,
                             system_prompt: str,
                             prompt: str,
                             prompt_format_type: str = "auto",
                             user_tag: str = "user",
                             assistant_tag: str = "assistant",
                             model: str = "remote:xxx") -> Union[str, list]:
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
    if history_messages == [] or history_messages == None or history_messages == "":
        return format_sys_prompt(system_prompt, prompt, prompt_format_type)

    if "chat_" in prompt_format_type and prompt_format_type != "chat_gemma":

        if prompt_format_type == "chat_gemini":
            assistant_tag = "model"

        text_input = format_sys_prompt(system_prompt, "", prompt_format_type)

        prod_prompt = format_sys_prompt("", prompt, prompt_format_type)

        prod_history = format_history_prompt(history_messages,
                                             prompt_format_type, user_tag,
                                             assistant_tag)

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
        return format_sys_prompt(system_prompt, history_str + prompt,
                                 prompt_format_type)
