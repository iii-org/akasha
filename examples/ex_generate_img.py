"""Generate an image; optionally edit it with the same configured image model."""
from pathlib import Path
import sys

# Support both python path/to/example.py and python -m examples.<module>.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples._common import DATA, configure, parser, print_response, workspace

def main(argv=None):
    cli = parser(__doc__)
    cli.add_argument("--image-model", help="Defaults to AKASHA_IMAGE_MODEL or openai:gpt-image-1")
    cli.add_argument("--edit", action="store_true", help="Make a second image API call to edit the result")
    args = configure(cli, argv)
    import os
    import akasha
    image_model = args.image_model or os.getenv("AKASHA_IMAGE_MODEL", "openai:gpt-image-1")
    with workspace(args, "images") as output:
        generated = akasha.gen_image(prompt="A friendly white rabbit sitting in a sunny garden.",
                                     model=image_model, save_path=str(output / "rabbit.png"),
                                     env_file=args.env_file)
        if not generated or not Path(generated).is_file():
            raise RuntimeError("The image provider did not produce an image.")
        print(generated)
        if args.edit:
            edited = akasha.edit_image(prompt="Add a small blue butterfly beside the rabbit.",
                                       images=generated, model=image_model,
                                       save_path=str(output / "rabbit-edited.png"), env_file=args.env_file)
            if not edited or not Path(edited).is_file():
                raise RuntimeError("The image provider did not produce an edited image.")
            print(edited)


if __name__ == "__main__":
    main()
