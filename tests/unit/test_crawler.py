import pytest

from akasha.helper import crawler

pytestmark = pytest.mark.unit


class _Response:
    def __init__(self, content=b"", headers=None, should_raise=False):
        self.content = content
        self.headers = headers or {}
        self._should_raise = should_raise

    def raise_for_status(self):
        if self._should_raise:
            raise crawler.requests.RequestException("boom")


def test_get_text_from_url_extracts_title_and_visible_text(monkeypatch):
    html = b"""
    <html>
      <head><title>Demo</title><meta name='x' content='y'></head>
      <body>
        <nav>ignore</nav>
        <h1>Hello</h1>
        <p>World</p>
        <a href='/'>hidden link</a>
        <footer>ignore footer</footer>
      </body>
    </html>
    """
    monkeypatch.setattr(crawler.requests, "get", lambda *args, **kwargs: _Response(content=html))

    title, text = crawler.get_text_from_url("https://example.com")

    assert title == "Demo"
    assert "Hello World" in text
    assert "ignore" not in text
    assert "hidden link" not in text


def test_get_text_from_url_handles_request_exceptions(monkeypatch):
    monkeypatch.setattr(
        crawler.requests,
        "get",
        lambda *args, **kwargs: _Response(should_raise=True),
    )

    assert crawler.get_text_from_url("https://example.com") == ("", "")


def test_get_webpage_last_modified_handles_present_and_missing_headers(monkeypatch):
    monkeypatch.setattr(
        crawler.requests,
        "head",
        lambda *args, **kwargs: _Response(headers={"Last-Modified": "Mon, 01 Jan 2024 00:00:00 GMT"}),
    )

    last_modified, timestamp = crawler.get_webpage_last_modified("https://example.com")

    assert last_modified.year == 2024
    assert isinstance(timestamp, float)

    monkeypatch.setattr(crawler.requests, "head", lambda *args, **kwargs: _Response(headers={}))

    assert crawler.get_webpage_last_modified("https://example.com") is None
