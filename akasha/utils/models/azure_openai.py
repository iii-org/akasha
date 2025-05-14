from openai import OpenAI, AzureOpenAI
import base64
from pathlib import Path
import openai as ai


class AzureOpenAIClient:
    def __init__(
        self,
        api_key: str,
        model_name: str,
        api_type: str = "openai",
        api_base: str = "",
        api_version: str = "2023-05-15",
        **kwargs,
    ):
        self.model_name = model_name
        if api_type == "openai":
            self.client = OpenAI(
                api_key=api_key,
            )
        else:
            self.client = AzureOpenAI(
                azure_endpoint=api_base,
                azure_deployment=model_name,
                api_key=api_key,
                api_version=api_version,
            )
            print(self.client.base_url)

    def generate(
        self,
        prompt: str,
        save_path: str = "./image.png",
        size: str = "auto",
        quality: str = "auto",
        moderation: str = "auto",
        background: str = "auto",
    ) -> Path:
        """generate image based on the user prompt.\n
        Args:
            prompt (str): _description_
            size (str, optional): _description_. Defaults to "auto".
            quality (str, optional): _description_. Defaults to "auto".
            output_format (str, optional): _description_. Defaults to "png".
            moderation (str, optional): _description_. Defaults to "auto".
            background (str, optional): _description_. Defaults to "auto".
        """

        # Convert save_path to Path object
        path = Path(save_path)
        allowed_exts = {"png", "jpeg", "webp"}

        if path.is_dir():
            # If it's a directory, append default filename
            path = path / "image.png"
            output_format = "png"
        else:
            ext = path.suffix.lower().lstrip(".")
            if ext not in allowed_exts:
                # Invalid or missing extension → default to .png
                path = path.with_suffix(".png")
                output_format = "png"
            else:
                output_format = ext

        # Generate image
        result = self.client.images.generate(
            model=self.model_name,
            prompt=prompt,
            size=size,
            quality=quality,
            output_format=output_format,
        )

        ## decode the image

        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)

        ## save the image
        with open(path, "wb") as f:
            f.write(image_bytes)

        print(f"Image saved to {path}")

        return path

    def edit(
        self,
        prompt: str,
        images: str,
        save_path: str = "./image.png",
        size: str = "auto",
        quality: str = "auto",
        moderation: str = "auto",
        background: str = "auto",
    ) -> Path:
        """generate image based on the user prompt.\n
        Args:
            prompt (str): _description_
            size (str, optional): _description_. Defaults to "auto".
            quality (str, optional): _description_. Defaults to "auto".
            output_format (str, optional): _description_. Defaults to "png".
            moderation (str, optional): _description_. Defaults to "auto".
            background (str, optional): _description_. Defaults to "auto".
        """
        ai._azure._deployments_endpoints.add("/images/edits")  # hack fix openai bug
        ## process the image list
        images_source = []
        for image in images:
            if isinstance(image, str):
                # use Path to check if the image is exist
                image_path = Path(image)
                if not image_path.exists():
                    raise ValueError(f"Image {image} does not exist.")
                images_source.append(open(image, "rb"))
            elif isinstance(image, Path):
                if not image.exists():
                    raise ValueError(f"Image {image} does not exist.")
                images_source.append(open(image, "rb"))
            else:
                raise ValueError(f"Image {image} is not a valid path or file object.")

        # Convert save_path to Path object
        path = Path(save_path)
        allowed_exts = {"png", "jpeg", "webp"}

        if path.is_dir():
            # If it's a directory, append default filename
            path = path / "image.png"

        else:
            ext = path.suffix.lower().lstrip(".")
            if ext not in allowed_exts:
                # Invalid or missing extension → default to .png
                path = path.with_suffix(".png")

        # Generate image
        result = self.client.images.edit(
            image=images_source,
            model=self.model_name,
            prompt=prompt,
            size=size,
        )

        ## decode the image

        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)

        ## save the image
        with open(path, "wb") as f:
            f.write(image_bytes)

        print(f"Image saved to {path}")

        return path
