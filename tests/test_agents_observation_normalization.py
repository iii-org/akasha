import asyncio
import importlib

import pytest

agents_module = importlib.import_module("akasha.agent.agents")


class _DummyModel:
    def get_num_tokens(self, _text):
        return 0


def _fake_basic_llm_init(
    self,
    model="mock:model",
    max_input_tokens=4096,
    max_output_tokens=512,
    temperature=0.0,
    language="ch",
    record_exp="",
    system_prompt="",
    keep_logs=False,
    verbose=False,
    env_file="",
):
    self.verbose = verbose
    self.language = language
    self.record_exp = record_exp
    self.system_prompt = system_prompt
    self.temperature = temperature
    self.keep_logs = keep_logs
    self.max_output_tokens = max_output_tokens
    self.max_input_tokens = max_input_tokens
    self.env_file = env_file
    self.timestamp_list = []
    self.logs = {}
    self.model_obj = _DummyModel()
    self.model = model


@pytest.mark.parametrize(
    "tool_output, expected_observation_text",
    [
        ([1, "x"], '[1, "x"]'),
        ({"k": "v"}, '{"k": "v"}'),
        ((1, "x"), '[1, "x"]'),
    ],
)
def test_run_agent_retri_observation_accepts_non_string_outputs(
    monkeypatch, tool_output, expected_observation_text
):
    monkeypatch.setattr(agents_module.basic_llm, "__init__", _fake_basic_llm_init)
    monkeypatch.setattr(agents_module, "retri_history_messages", lambda *args, **kwargs: ("", 0))
    monkeypatch.setattr(agents_module, "get_doc_length", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        agents_module,
        "format_sys_prompt",
        lambda _system_prompt, prompt, *_args, **_kwargs: prompt,
    )

    call_inputs = []

    def fake_call_model(_model_obj, text_input, _verbose, keep_logs=False):
        call_inputs.append(text_input)
        if len(call_inputs) == 1:
            return "TOOL_CALL"
        if len(call_inputs) == 2:
            return "OBS_SUMMARY"
        if len(call_inputs) == 3:
            return "FINAL_CALL"
        raise AssertionError("Unexpected call_model invocation count")

    def fake_extract_json(response):
        if response == "TOOL_CALL":
            return {"thought": "use-tool", "action": "dummy_tool", "action_input": {}}
        if response == "FINAL_CALL":
            return {
                "thought": "done",
                "action": "final answer",
                "action_input": "ok",
            }
        raise AssertionError(f"Unexpected response for extract_json: {response}")

    monkeypatch.setattr(agents_module, "call_model", fake_call_model)
    monkeypatch.setattr(agents_module, "extract_json", fake_extract_json)

    agent = agents_module.agents(
        tools=[],
        model="mock:model",
        keep_logs=False,
        retri_observation=True,
        max_round=3,
    )
    agent.tools = {"dummy_tool": object()}

    result = asyncio.run(
        agent._run_agent(
            question="q",
            tool_runner=lambda _tool, _tool_input: tool_output,
            messages=[],
        )
    )

    assert result == "ok"
    assert len(call_inputs) >= 2
    assert f"\n\nObservation: {expected_observation_text}" in call_inputs[1]
