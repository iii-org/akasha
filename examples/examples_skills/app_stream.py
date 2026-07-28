# --coding: utf-8--

from pathlib import Path
import sys

import akasha


BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    agent = akasha.agents(
        model="gemini:gemini-2.5-flash",
        skills=[str(BASE_DIR / "hello-skill")],
        env_file=str(BASE_DIR / ".env"),
        stream=True,
        thinking=True,
        temperature=0.0,
        verbose=True,
        keep_logs=True,
    )
    response = agent(
        "Use the hello skill to greet Alice. Follow the skill's bundled "
        "script instructions and return its output."
    )

    # The package consumes and displays stream events when verbose=True.
    # This loop only keeps the generator flowing; a FastAPI/SSE adapter can
    # instead forward each yielded event to the frontend.
    for _event in response:
        pass
    return None

if __name__ == "__main__":
    main()
