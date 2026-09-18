"""Store and retrieve a conversation with a persistent memory collection."""
from pathlib import Path
import sys

# Support both python path/to/example.py and python -m examples.<module>.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples._common import DATA, configure, parser, print_response, workspace

def main(argv=None):
    cli = parser(__doc__)
    cli.add_argument("--memory-name", default="demo-memory")
    args = configure(cli, argv)
    import akasha
    # Keep a stable dedicated directory so memory survives subsequent runs.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import os
    previous = Path.cwd()
    memory_root = args.output_dir / "memory"
    memory_root.mkdir(exist_ok=True)
    try:
        os.chdir(memory_root)
        memory = akasha.MemoryManager(memory_name=args.memory_name, model=args.model,
                                      embeddings=args.embeddings, env_file=args.env_file,
                                      memory_dirname="documents")
        memory.add_memory("My name is Alice.", "Hello, Alice!")
        prompt = "What is my name?"
        history = memory.search_memory(prompt, top_k=3)
        client = akasha.ask(model=args.model, env_file=args.env_file)
        print(client(prompt=prompt, history_messages=history))
    finally:
        os.chdir(previous)
    print(f"Memory: {memory_root}")


if __name__ == "__main__":
    main()
