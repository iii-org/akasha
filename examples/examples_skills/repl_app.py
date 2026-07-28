from pathlib import Path

import akasha


BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    prompt = (
        "Create values=[2, 4, 6, 8] and total=sum(values) in one Python "
        "execution. In a separate execution, reuse those existing variables "
        "to calculate and report the average."
    )
    agent = akasha.agents(
        model="gemini:gemini-2.5-flash",
        skills=[str(BASE_DIR / "python-repl-skill")],
        env_file=str(BASE_DIR / ".env"),
        temperature=0.0,
        verbose=True,
        stream=True,
        keep_logs=False,
    )
    response = agent(prompt)
    print(response)
    # breakpoint()
    
    for event in response:
        event_type = event.get("type")
        if event_type == "thinking":
            print(f"[thinking] {event.get('data', '')}", flush=True)
        elif event_type == "tool":
            tool = event.get("data", {})
            name = tool.get("name", "unknown")
            content = tool.get("content", "")
            print(f"\n[tool:{name}] {content}", flush=True)
        elif event_type == "answer":
            chunk = event.get("data", "")
            print(chunk, end="", flush=True)


if __name__ == "__main__":
    main()
