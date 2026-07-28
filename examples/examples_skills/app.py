from pathlib import Path

import akasha


BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    agent = akasha.agents(
        model="gemini:gemini-2.5-flash",
        skills=[str(BASE_DIR / "hello-skill")],
        env_file=str(BASE_DIR / ".env"),
        temperature=0.0,
        keep_logs=False,
    )
    response = agent(
        "Use the hello skill to greet Alice. Follow the skill's bundled "
        "script instructions and return its output."
    )
    print(response)


if __name__ == "__main__":
    main()
