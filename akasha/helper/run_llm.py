from typing import Union, List, Generator
from pydantic import BaseModel
from langchain_core.messages.ai import AIMessage
from langchain_core.messages import AIMessageChunk
from langchain_core.language_models.chat_models import BaseChatModel
import traceback
import logging
from langchain_core.language_models.base import BaseLanguageModel
from akasha.helper.handle_objects import handle_model_and_name
from akasha.helper.base import sim_to_trad, extract_json, extract_multiple_json
from akasha.utils.prompts.gen_prompt import format_sys_prompt, default_translate_prompt


def call_model(
    model: BaseLanguageModel,
    input_text: Union[str, list],
    verbose: bool = True,
    keep_logs: bool = False,
) -> str:
    """call llm model and return the response

    Args:
        model (BaseLanguageModel): llm model
        input_text (str): the input_text that send to llm model
        verbose (bool, optional): whether to print the response. Defaults to True.

    Returns:
        str: llm response
    """

    ### for openai, change system prompt and prompt into system meg and human meg ###

    response = ""

    model, model_name = handle_model_and_name(model)

    try:
        try:
            model_type = model._llm_type
        except Exception:
            model_type = "unknown"

        response = None
        log_enabled = verbose or keep_logs
        max_retries = 20
        attempt = 0
        def normalize_response(value):
            if isinstance(value, (AIMessage, AIMessageChunk)):
                value = content_to_text(value.content)
            elif isinstance(value, dict):
                value = str(value)
            elif isinstance(value, list):
                value = content_to_text(value)
            return value

        def is_empty_response(value):
            if value is None:
                return True
            if isinstance(value, str):
                return value.strip() == ""
            return not bool(value)

        while attempt < max_retries and is_empty_response(response):
            if isinstance(model, BaseChatModel):
                response = model.invoke(normalize_chat_input(input_text))
            elif "remote" in model_type:
                response = model._call(input_text, verbose=verbose)
            else:
                response = model._call(input_text, verbose=verbose)

            response = normalize_response(response)
            if is_empty_response(response):
                if log_enabled:
                    logging.warning(
                        "LLM response is empty. Retrying call_model. response_type=%s",
                        type(response).__name__,
                    )
                attempt += 1

        if is_empty_response(response):
            error_message = "LLM response is empty after max retries; giving up."
            if log_enabled:
                logging.error(error_message)
            raise Exception(error_message)

    except Exception as e:
        trace_text = traceback.format_exc()
        logging.error(
            trace_text
            + "\n\nText generation encountered an error.\
            Please check your model setting.\n\n"
        )
        raise e

    if isinstance(response, str):
        response = sim_to_trad(response)

    return response


def call_batch_model(
    model: BaseLanguageModel,
    input_text: list,
    verbose: bool = False,
    keep_logs: bool = False,
) -> List[str]:
    """call llm model in batch and return the response

    Args:
        model (BaseLanguageModel): llm model
        input_text (list):  the input_text that send to llm model
        verbose (bool, optional): whether to print the response. Defaults to False.

    Returns:
        str: llm response
    """

    ### check the input prompt and system prompt ###
    if isinstance(input_text, str):
        input_text = [input_text]

    response = ""
    responses = []
    model, model_name = handle_model_and_name(model)

    try:
        log_enabled = verbose or keep_logs
        max_retries = 3
        attempt = 0
        while attempt < max_retries and (response is None or response == "" or "".join(responses) == ""):
            batch_input = (
                [normalize_chat_input(item) for item in input_text]
                if isinstance(model, BaseChatModel)
                else input_text
            )
            response = model.batch(batch_input)
            responses = []
            for res in response:
                if isinstance(res, (AIMessage, AIMessageChunk)):
                    res = content_to_text(res.content)
                if isinstance(res, dict):
                    res = str(res)
                if isinstance(res, list):
                    res = content_to_text(res)
                responses.append(res)

            if response is None or response == "" or "".join(responses) == "":
                if log_enabled:
                    logging.warning("LLM response is empty. Retrying batch call.")
                attempt += 1

        if response is None or response == "" or "".join(responses) == "":
            raise Exception("LLM response is empty.")

    except Exception as e:
        trace_text = traceback.format_exc()
        logging.error(
            trace_text
            + "\n\nText generation encountered an error.\
            Please check your model setting.\n\n"
        )
        raise e

    # if print_flag:
    #     print("llm response:", "\n\n" + response)

    return responses


def call_stream_model(
    model: BaseLanguageModel,
    input_text: Union[str, list],
    verbose: bool = True,
    keep_logs: bool = False,
) -> Generator[str, None, None]:
    """call llm model and yield the response

    Args:
        model (BaseLanguageModel): llm model
        input_text (str): the input_text that send to llm model
        verbose (bool, optional): whether to print the response. Defaults to True.

    Returns:
        str: llm response
    """

    ### for openai, change system prompt and prompt into system meg and human meg ###

    response = None
    model, model_name = handle_model_and_name(model)
    try:
        log_enabled = verbose or keep_logs
        max_retries = 3
        attempt = 0
        while attempt < max_retries:
            texts = ""
            try:
                response = model.stream(normalize_chat_input(input_text))
            except Exception:
                response = model._call(input_text, verbose=verbose)

            for r in response:
                if isinstance(r, (AIMessage, AIMessageChunk)):
                    r = content_to_text(r.content)
                elif isinstance(r, list):
                    r = content_to_text(r)
                elif not isinstance(r, str):
                    r = str(r)
                texts += r
                yield sim_to_trad(r)

            if texts != "":
                break

            if log_enabled:
                logging.warning("LLM response is empty. Retrying stream call.")
            attempt += 1

        if texts == "":
            yield "ERROR! LLM response is empty.\n\n"

    except Exception as e:
        trace_text = traceback.format_exc()
        logging.error(
            trace_text
            + "\n\nText generation encountered an error.\
            Please check your model setting.\n\n"
        )
        # A provider exception is not a valid stream chunk.  Yielding it
        # makes callers fail later with unrelated type errors while
        # concatenating the response.  Preserve the original failure.
        raise


def content_to_text(content) -> str:
    """Extract visible text from LangChain content blocks.

    Reasoning/thinking blocks are intentionally excluded from the public answer
    path. They remain available on the original AIMessage for logging.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            if isinstance(block, dict):
                if block.get("type") in {"reasoning", "thinking"}:
                    continue
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    parts.append(text)
                elif text is not None:
                    parts.append(str(text))
                continue
            parts.append(str(block))
        return "".join(parts)
    return str(content)


def content_to_thinking(content, additional_kwargs=None) -> str:
    """Extract reasoning/thinking text from LangChain content blocks."""
    parts = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") not in {"reasoning", "thinking"}:
                continue
            value = block.get("reasoning") or block.get("thinking") or block.get("text")
            if value:
                parts.append(str(value))
    if isinstance(additional_kwargs, dict):
        for key in ("reasoning_content", "thinking", "reasoning"):
            value = additional_kwargs.get(key)
            if value:
                parts.append(value if isinstance(value, str) else str(value))
    return "".join(parts)


def call_stream_events(
    model: BaseLanguageModel,
    input_text: Union[str, list],
    include_thinking: bool = False,
    verbose: bool = True,
    keep_logs: bool = False,
) -> Generator[dict, None, None]:
    """Yield normalized answer/thinking events from a ChatModel stream."""
    if not isinstance(model, BaseChatModel):
        raise ValueError("thinking streaming requires a LangChain ChatModel.")

    for chunk in model.stream(normalize_chat_input(input_text)):
        content = getattr(chunk, "content", "")
        thinking = content_to_thinking(
            content, getattr(chunk, "additional_kwargs", None)
        )
        answer = content_to_text(content)
        if include_thinking and thinking:
            yield {"type": "thinking", "data": thinking}
        if answer:
            if verbose:
                print(answer, end="", flush=True)
            yield {"type": "answer", "data": sim_to_trad(answer)}


def normalize_chat_input(input_text):
    """Convert legacy prompt dictionaries to LangChain chat messages."""
    if not isinstance(input_text, list):
        return input_text
    normalized = []
    for message in input_text:
        if not isinstance(message, dict):
            normalized.append(message)
            continue
        role = message.get("role", "user")
        if role == "model":
            role = "system" if not normalized else "assistant"
        content = message.get("content", message.get("parts", ""))
        if isinstance(content, list):
            content = "".join(
                item if isinstance(item, str) else str(item)
                for item in content
            )
        normalized.append({"role": role, "content": content})
    return normalized


def call_image_model(
    model: BaseLanguageModel,
    input_text: Union[str, list],
    verbose: bool = True,
    keep_logs: bool = False,
) -> str:
    """
    Calls a multimodal model with image input and returns its text response.

    Args:
        model (BaseLanguageModel): The multimodal model to use.
        input_text (Union[str, list]): The input prompt(s) for the model.
        verbose (bool, optional): If True, enables verbose output. Defaults to True.
        keep_logs (bool, optional): If True, keeps logs even if verbose is False. Defaults to False.

    Returns:
        str: The response from the image generation model.
    """
    response = ""
    print_flag = True
    model, model_name = handle_model_and_name(model)
    try:
        model_type = model._llm_type
        log_enabled = verbose or keep_logs

        if isinstance(model, BaseChatModel) or (
            ("openai" in model_type)
            or ("remote" in model_type)
            or ("gemini" in model_type)
        ):
            print_flag = False
            response = model.invoke(input_text)

        else:
            response = model.call_image(input_text, verbose=verbose)

        if isinstance(response, AIMessage):
            response = response.content
            if isinstance(response, dict):
                response = response.__str__()
            if isinstance(response, list):
                response = "\n".join(response)

        max_retries = 3
        attempt = 0
        while attempt < max_retries and (response is None or response == ""):
            if log_enabled:
                logging.warning("LLM response is empty. Retrying image call.")
            if isinstance(model, BaseChatModel) or (
                ("openai" in model_type)
                or ("remote" in model_type)
                or ("gemini" in model_type)
            ):
                response = model.invoke(input_text)
            else:
                response = model.call_image(input_text, verbose=verbose)

            if isinstance(response, AIMessage):
                response = response.content
                if isinstance(response, dict):
                    response = response.__str__()
                if isinstance(response, list):
                    response = "\n".join(response)

            if response is None or response == "":
                print_flag = False
                attempt += 1

        if response is None or response == "":
            print_flag = False
            raise Exception("LLM response is empty.")

    except Exception as e:
        trace_text = traceback.format_exc()
        logging.error(
            trace_text
            + "\n\nText generation encountered an error.\
            Please check your model setting.\n\n"
        )
        raise e

    response = sim_to_trad(response)

    if print_flag:
        print("llm response:", "\n\n" + response)
    return response


def check_relevant_answer(
    model_obj: BaseLanguageModel,
    batch_responses: List[str],
    question: str,
    prompt_format_type: str = "auto",
) -> List[str]:
    """ask LLM that each of the retrieved answers list is relevant to the question or not"""
    from akasha.utils.prompts.gen_prompt import (
        default_answer_grader_prompt,
        format_sys_prompt,
    )

    results = []
    txts = []
    sys_prompt = default_answer_grader_prompt()
    model_obj, model_name = handle_model_and_name(model_obj)
    for idx in range(len(batch_responses)):
        prod_prompt = f"Retrieved answer: \n\n {batch_responses[idx]} \n\n User question: {question}"
        text_input = format_sys_prompt(
            sys_prompt, prod_prompt, prompt_format_type, model_name
        )
        txts.append(text_input)

    response_list = call_batch_model(model_obj, txts)
    for idx, response in enumerate(response_list):
        if "yes" in response.lower():
            results.append(batch_responses[idx])

    return results


def call_translator(
    model_obj: BaseLanguageModel,
    texts: str,
    prompt_format_type: str = "auto",
    language: str = "zh",
    verbose: bool = True,
) -> str:
    """translate texts to target language

    Args:
        model_obj (BaseLanguageModel): LLM that used to translate
        texts (str): texts that need to be translated
        prompt_format_type (str, optional): system prompt format. Defaults to "auto".
        language (str, optional): target language. Defaults to "zh".

    Returns:
        str: translated texts
    """
    model_obj, model_name = handle_model_and_name(model_obj, verbose=verbose)
    sys_prompt = default_translate_prompt(language)
    prod_prompt = format_sys_prompt(sys_prompt, texts, prompt_format_type, model_name)

    response = call_model(model_obj, prod_prompt, verbose=verbose)

    return response


def call_JSON_formatter(
    model_obj: BaseLanguageModel,
    texts: str,
    keys: Union[str, list, BaseModel] = "",
    prompt_format_type: str = "auto",
    verbose: bool = True,
) -> Union[dict, List[dict], None]:
    """use LLM to transfer texts into JSON format

    Args:
        model_obj (BaseLanguageModel): LLM that used to transfer
        texts (str): texts that need to be transferred
        keys (Union[str, list], optional): keys name of output dictionary. Defaults to "".
        prompt_format_type (str, optional): system prompt format. Defaults to "auto".

    Returns:
        Union[dict, None]: return the JSON part of the string, if not found return None
    """

    ### RESPONSE FORMAT FAILED OR NOT OPENAI/GEMINI MODEL ###
    if keys == "":
        sys_prompt = "Format the following TEXTS into a single JSON instance that conforms to the JSON schema."
    elif isinstance(keys, str):
        keys = [keys]

    model_obj, model_name = handle_model_and_name(model_obj)
    model_name = model_name.lower()

    if keys != "":
        if not isinstance(keys, list):
            keys_list = basemodel_keys_list(keys)
        else:
            keys_list = keys

        sys_prompt = f"Format the following TEXTS into a single JSON instance that conforms to the JSON schema which includes: {', '.join(keys_list)}\n\n"

    prod_prompt = format_sys_prompt(
        sys_prompt, "TEXTS: " + texts, prompt_format_type, model_name
    )

    ## try use response format ##
    if keys != "" and ("openai" in model_name or "gemini" in model_name):
        try:
            if issubclass(keys, BaseModel):
                json_base_model = keys
            else:
                json_base_model = keys_to_basemodel_class(keys)

            if "openai" in model_name:
                response_format = {"type": "json_object"}

                # can not use it yet #
                # else:
                #    response_format = json_base_model
                response = model_obj.invoke(
                    prod_prompt,
                    response_format=response_format,
                    verbose=verbose,
                )
                return extract_json(response)

            else:
                response = model_obj.invoke(
                    prod_prompt,
                    response_format=json_base_model,
                    verbose=verbose,
                )
                return extract_multiple_json(response)
        except Exception as e:
            print("Error in using JSON response format:", e)

    response = call_model(model_obj, prod_prompt, verbose=verbose)
    return extract_json(response)


def keys_to_basemodel_class(
    keys: list[str], class_name: str = "JSONFormatModel"
) -> type:
    """
    Dynamically create a Pydantic BaseModel class with all fields as str.

    Args:
        keys (list[str]): List of field names.
        class_name (str): Name of the generated class.

    Returns:
        type: A new BaseModel subclass.
    """
    return type(class_name, (BaseModel,), {"__annotations__": {k: str for k in keys}})


def basemodel_keys_list(model: type) -> list[str]:
    """
    Return all field names of a Pydantic BaseModel class as a list of strings.

    Args:
        model (type): A Pydantic BaseModel class.

    Returns:
        list[str]: List of field names.
    """
    return list(model.__annotations__.keys())
