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
    breakpoint()


if __name__ == "__main__":
    main()
