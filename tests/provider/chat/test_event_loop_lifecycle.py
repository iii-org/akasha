"""Exercise real keep-alive sockets across short-lived asyncio loops."""
import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from akasha.utils.models.chat import build_chat_model

pytestmark = pytest.mark.integration


@pytest.fixture(autouse=True)
def selector_event_loop():
    # Match Linux CI rather than Windows Proactor transport cleanup.
    previous = asyncio.get_event_loop_policy()
    if hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    try:
        yield
    finally:
        asyncio.set_event_loop_policy(previous)


@pytest.fixture
def completion_server():
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers['Content-Length']))
            body = json.dumps({'id': 'local', 'object': 'chat.completion', 'created': 1,
                'model': 'gpt-4o-mini', 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': '42'}}],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.wfile.flush()

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/v1'
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


@pytest.mark.parametrize('facade', ['model', 'agent'])
@pytest.mark.parametrize('reuse_model', [True, False], ids=['same-model', 'new-model'])
@pytest.mark.parametrize('provider', ['openai', 'azure'])
def test_model_async_connections_do_not_outlive_event_loop(
    completion_server, reuse_model, provider, facade
):
    if provider == 'azure':
        env = {'AZURE_OPENAI_API_KEY': 'local-test', 'AZURE_OPENAI_BASE_URL': completion_server}
    else:
        env = {'OPENAI_API_KEY': 'local-test', 'OPENAI_BASE_URL': completion_server}
    model = build_chat_model(provider, 'gpt-4o-mini', env)
    import akasha
    agent = akasha.agents(model=model, stream=False, env_file='')
    for _ in range(3):
        if not reuse_model:
            model = build_chat_model(provider, 'gpt-4o-mini', env)
            agent = akasha.agents(model=model, stream=False, env_file='')
        # Retries can mask reuse of a socket owned by a closed loop.
        model.root_async_client.max_retries = 0
        if facade == 'agent':
            assert agent('What is 20 + 22?') == '42'
        else:
            assert asyncio.run(model.ainvoke('What is 20 + 22?')).content == '42'

