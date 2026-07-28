from akasha.tools.ask import ask


def _ask_for_display(**overrides):
    instance = ask.__new__(ask)
    values = {
        "verbose": True,
        "keep_logs": False,
        "model": "gemini:gemini-2.5-flash",
        "temperature": 0.0,
        "thinking": True,
        "thinking_budget": None,
        "thinking_budget_level": None,
        "effective_thinking_budget": None,
        "prompt_format_type": "auto",
        "max_input_tokens": 1024,
        "prompt_tokens": 0,
        "prompt_length": 0,
        "doc_tokens": 0,
        "doc_length": 0,
    }
    values.update(overrides)
    instance.__dict__.update(values)
    return instance


def test_verbose_reports_gemini_api_default_without_thinking_budget(capsys):
    _ask_for_display()._display_info()

    output = capsys.readouterr().out
    assert "Info: Gemini thinking_budget is not set" in output
    assert "\033[33m" in output


def test_verbose_does_not_warn_when_gemini_budget_is_set(capsys):
    _ask_for_display(
        thinking_budget=8192,
        effective_thinking_budget=8192,
    )._display_info()

    assert "Warning: Gemini thinking is enabled" not in capsys.readouterr().out


def test_verbose_does_not_warn_when_thinking_is_disabled(capsys):
    _ask_for_display(thinking=False)._display_info()

