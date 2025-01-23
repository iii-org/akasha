from pathlib import Path
from typing import Callable, Union, Tuple, List, Generator
from langchain_core.messages.ai import AIMessage
import traceback, logging
from langchain_core.language_models.base import BaseLanguageModel
from akasha.helper.handle_objects import handle_model_and_name
from akasha.helper.base import sim_to_trad


def call_model(
    model: BaseLanguageModel,
    input_text: Union[str, list],
) -> str:
    """call llm model and return the response

    Args:
        model (BaseLanguageModel): llm model
        input_text (str): the input_text that send to llm model
        prompt_type (str): the type of prompt, default "gpt"

    Returns:
        str: llm response
    """

    ### for openai, change system prompt and prompt into system meg and human meg ###

    response = ""
    print_flag = True

    model, model_name = handle_model_and_name(model)

    try:
        try:
            model_type = model._llm_type
        except:
            print_flag = False
            model_type = "unknown"

        if ("openai" in model_type):
            print_flag = False
            response = model.invoke(input_text)

        elif "remote" in model_type:
            print_flag = False
            response = model._call(input_text)
        else:
            try:
                response = model._call(input_text)
            except:
                response = model._generate(input_text)

        if isinstance(response, AIMessage):
            response = response.content
            if isinstance(response, dict):
                response = response.__str__()
            if isinstance(response, list):
                response = '\n'.join(response)

        if ("huggingface" in model_type) or ("llama cpp" in model_type) or (
                "gemini" in model_type) or ("anthropic" in model_type):
            print_flag = False

        if response is None or response == "":
            print_flag = False
            raise Exception("LLM response is empty.")

    except Exception as e:
        trace_text = traceback.format_exc()
        logging.error(trace_text + "\n\nText generation encountered an error.\
            Please check your model setting.\n\n")
        raise e

    if print_flag:
        print("llm response:", "\n\n" + response)

    if isinstance(response, str):
        response = sim_to_trad(response)

    return response


def call_batch_model(
    model: BaseLanguageModel,
    input_text: list,
) -> List[str]:
    """call llm model in batch and return the response 

    Args:
        model (BaseLanguageModel): llm model
        input_text: list

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

        response = model.batch(input_text)
        for res in response:
            if isinstance(res, AIMessage):
                res = res.content
            if isinstance(res, dict):
                res = res.__str__()
            if isinstance(res, list):
                res = '\n'.join(res)
            responses.append(res)

        if response is None or response == "" or ''.join(responses) == "":
            print_flag = False
            raise Exception("LLM response is empty.")

    except Exception as e:
        trace_text = traceback.format_exc()
        logging.error(trace_text + "\n\nText generation encountered an error.\
            Please check your model setting.\n\n")
        raise e

    # if print_flag:
    #     print("llm response:", "\n\n" + response)

    return responses


def call_stream_model(
    model: BaseLanguageModel,
    input_text: Union[str, list],
) -> Generator[str, None, None]:
    """call llm model and yield the response

    Args:
        model (BaseLanguageModel): llm model
        input_text (str): the input_text that send to llm model
        prompt_type (str): the type of prompt, default "gpt"

    Returns:
        str: llm response
    """

    ### for openai, change system prompt and prompt into system meg and human meg ###

    response = None
    texts = ""
    model, model_name = handle_model_and_name(model)
    try:

        try:
            response = model.stream(input_text)
        except:
            response = model._call(input_text)

        for r in response:
            if isinstance(r, AIMessage):
                r = r.content
                if isinstance(r, dict):
                    r = r.__str__()
                if isinstance(r, list):
                    r = '\n'.join(r)
            texts += r
            yield sim_to_trad(r)

        if texts == "":
            yield "ERROR! LLM response is empty.\n\n"

    except Exception as e:
        trace_text = traceback.format_exc()
        logging.error(trace_text + "\n\nText generation encountered an error.\
            Please check your model setting.\n\n")
        yield e


def call_image_model(
    model: BaseLanguageModel,
    input_text: Union[str, list],
) -> str:

    response = ""
    print_flag = True
    model, model_name = handle_model_and_name(model)
    try:

        model_type = model._llm_type

        if ("openai" in model_type) or ("remote" in model_type):
            print_flag = False
            response = model.invoke(input_text)

        else:

            try:
                response = model.call_image(input_text)
            except:
                response = model._generate(input_text)

        if isinstance(response, AIMessage):
            response = response.content
            if isinstance(response, dict):
                response = response.__str__()
            if isinstance(response, list):
                response = '\n'.join(response)

        if response is None or response == "":
            print_flag = False
            raise Exception("LLM response is empty.")

    except Exception as e:
        trace_text = traceback.format_exc()
        logging.error(trace_text + "\n\nText generation encountered an error.\
            Please check your model setting.\n\n")
        raise e

    response = sim_to_trad(response)

    if print_flag:
        print("llm response:", "\n\n" + response)
    return response


def check_relevant_answer(model_obj: BaseLanguageModel,
                          batch_responses: List[str],
                          question: str,
                          prompt_format_type: str = "auto") -> List[str]:
    """ask LLM that each of the retrieved answers list is relevant to the question or not"""
    from akasha.utils.prompts.gen_prompt import default_answer_grader_prompt, format_sys_prompt
    results = []
    txts = []
    sys_prompt = default_answer_grader_prompt()
    model_obj, model_name = handle_model_and_name(model_obj)
    for idx in range(len(batch_responses)):
        prod_prompt = f"Retrieved answer: \n\n {batch_responses[idx]} \n\n User question: {question}"
        text_input = format_sys_prompt(sys_prompt, prod_prompt,
                                       prompt_format_type, model_name)
        txts.append(text_input)

    response_list = call_batch_model(model_obj, txts)
    for idx, response in enumerate(response_list):
        if 'yes' in response.lower():
            results.append(batch_responses[idx])

    return results
