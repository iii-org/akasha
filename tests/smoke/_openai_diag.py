from pathlib import Path
from urllib.parse import urlparse

from dotenv import dotenv_values

from akasha.utils.models.chat import build_chat_model

env = dotenv_values(Path("tests/.env"))
for key in (
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_BASE_URL",
):
    value = env.get(key)
    if key.endswith("BASE_URL") and value:
        parsed = urlparse(value)
        print(f"{key}: configured host={parsed.netloc!r} path={parsed.path!r}")
    else:
        print(f"{key}: {'configured' if value else 'missing'}")

model = build_chat_model("openai", "gpt-5.4", env, max_output_tokens=32)
print(f"model_type: {type(model).__name__}")
print(f"model_name: {getattr(model, 'model_name', None)!r}")
client = getattr(model, "client", None)
base_url = getattr(client, "base_url", None)
print(f"client_base_url: {str(base_url)!r}")
