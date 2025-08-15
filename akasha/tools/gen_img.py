from akasha.helper.handle_objects import handle_client
from pathlib import Path
from typing import Union
from PIL import Image

IMAGE_DEFAULT_MODEL = "openai:gpt-image-1"


def gen_image(
    prompt: str,
    save_path: str = "./image.png",
    model: str = IMAGE_DEFAULT_MODEL,
    size: str = "auto",
    quality: str = "auto",
    verbose: bool = False,
    env_file: str = "",
):
    """the image generate image call Model to generate image based on the user prompt.\n

    Args:
        prompt (str): user prompt
        save_path (str, optional): image save path, can be .png, .jpeg, .webp. Defaults to "./image.png".
        model (str, optional): model name. Defaults to "openai:gpt-image-1".
        size (str, optional): image size(256x256, 1024x1024...). Defaults to "auto".
        quality (str, optional): image quality. Defaults to "auto".
        verbose (bool, optional): whether to print the response. Defaults to False.
        env_file (str, optional): env file path. Defaults to "".
    """

    model_obj = handle_client(model, env_file)

    print(
        f"Generating, may take some time...\n\nmodel: {model}, save_path: {save_path}"
    )
    ret = model_obj.generate(
        prompt=prompt,
        save_path=save_path,
        size=size,
        quality=quality,
        moderation="auto",
        background="auto",
        verbose=verbose,
    )

    if verbose and ret:
        try:
            img = Image.open(ret)
            img.show()
        except Exception as e:
            print(f"Failed to open image: {e}")

    return ret


def edit_image(
    prompt: str,
    images: Union[list[str], str, Path],
    save_path: str = "./image.png",
    model: str = IMAGE_DEFAULT_MODEL,
    size: str = "auto",
    quality: str = "auto",
    verbose: bool = False,
    env_file: str = "",
):
    """the image generate image call Model to generate image based on the user prompt.\n

    Args:
        prompt (str): user prompt
        images (list[str]): the list of image path(string) you want to use as source.
        save_path (str, optional): image save path, can be .png, .jpeg, .webp. Defaults to "./image.png".
        model (str, optional): model name. Defaults to "openai:gpt-image-1".
        size (str, optional): image size(256x256, 1024x1024...). Defaults to "auto".
        quality (str, optional): image quality. Defaults to "auto".
        verbose (bool, optional): whether to print the response. Defaults to False.
        env_file (str, optional): env file path. Defaults to "".
    """

    model_obj = handle_client(model, env_file)

    ### check if images is a list
    if not isinstance(images, list):
        images = [images]

    print(
        f"Generating, may take some time...\n\nmodel: {model}, save_path: {save_path}"
    )

    ret = model_obj.edit(
        prompt=prompt,
        images=images,
        save_path=save_path,
        size=size,
        quality=quality,
        moderation="auto",
        background="auto",
        verbose=verbose,
    )

    if verbose and ret:
        try:
            img = Image.open(ret)
            img.show()
        except Exception as e:
            print(f"Failed to open image: {e}")

    return ret
