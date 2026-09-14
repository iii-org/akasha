"""Public MCP result normalization contract tests."""

from akasha.agent.mcp import normalize_mcp_result


def test_normalize_mcp_result_decodes_one_json_text_block():
    assert normalize_mcp_result(
        [{"type": "text", "text": '{"sum": 42}'}]
    ) == {"sum": 42}


def test_normalize_mcp_result_keeps_one_plain_text_block_as_text():
    assert normalize_mcp_result(
        [{"type": "text", "text": "MCP_STREAMABLE_HTTP_OK"}]
    ) == "MCP_STREAMABLE_HTTP_OK"


def test_normalize_mcp_result_preserves_multiple_content_blocks():
    blocks = [
        {"type": "text", "text": "first"},
        {"type": "resource_link", "uri": "file:///report.json"},
    ]
    assert normalize_mcp_result(blocks) == blocks
