"""Ask from local reference text; optionally stream or inspect an image."""
from pathlib import Path
import sys

# Support both python path/to/example.py and python -m examples.<module>.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples._common import DATA, configure, parser, print_response, workspace

def main(argv=None):
    cli = parser(__doc__)
    cli.add_argument("--stream", action="store_true")
    cli.add_argument("--image", help="Optional local image path or image URL")
    args = configure(cli, argv)
    if args.image and not args.image.startswith(("http://", "https://")):
        args.image = str(Path(args.image).expanduser().resolve())
    import akasha
    with workspace(args, "ask"):
        client = akasha.ask(model=args.model, env_file=args.env_file, keep_logs=True)
        if args.image:
            response = client.vision(prompt="Describe this image.", image_path=args.image)
        else:
            response = client("What is Industry 4.0?", info=str(DATA), stream=args.stream)
        print_response(response)
        client.save_logs("ask.json")


if __name__ == "__main__":
    main()
