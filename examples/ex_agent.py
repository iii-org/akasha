"""Compatibility entry point for the MCP agent example.

Run ``examples/ex_mcp.py`` directly for the complete current example.
"""

try:
    from .ex_mcp import main
except ImportError:  # Running this file directly from the repository root.
    from ex_mcp import main


if __name__ == "__main__":
    main()
