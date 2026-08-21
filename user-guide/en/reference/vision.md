# Vision, image generation, and image editing

akasha exposes three different image workflows. Choose the API based on whether the model should return text or an image.

## Understand an image with `vision`

Use `vision()` when you want to ask a question about an existing image. The result is text.

```python
import akasha

qa = akasha.ask(model="gemini:gemini-2.5-flash")
answer = qa.vision(
    prompt="What information appears on this business card?",
    image_path="business-card.png",
)
print(answer)
```

`gemini:gemini-2.5-flash` accepts image and text input and returns a text answer. The image is not modified.

## Generate an image with `gen_image`

Use `gen_image()` when there is no source image and you want to create a new image from a prompt.

```python
import akasha

output_path = akasha.gen_image(
    prompt="A small red bicycle in a watercolor style",
    model="openai:gpt-image-1",
    save_path="bicycle.png",
)
print(output_path)
```

The default image model is `openai:gpt-image-1`. Provider and model availability depends on the configured account and API.

## Edit an image with `edit_image`

Use `edit_image()` when you want to remove, add, or change content in an existing image.

```python
import akasha

output_path = akasha.edit_image(
    prompt="Remove the bicycle and add a green potted plant on the right",
    images="bicycle.png",
    model="openai:gpt-image-1",
    save_path="edited.png",
)
print(output_path)
```

Multiple reference images are also supported:

```python
output_path = akasha.edit_image(
    prompt="Combine the subject from the first image with the background from the second",
    images=["subject.png", "background.png"],
    save_path="combined.png",
)
```

The image-generation and image-editing APIs save the returned image to `save_path` and return that path.
