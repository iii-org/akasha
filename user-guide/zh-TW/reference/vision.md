# 圖片理解、圖片生成與圖片編輯

akasha 提供三種不同的圖片流程。請依照模型要回傳文字或圖片，選擇適合的 API。

## 使用 `vision` 理解圖片

當你想詢問既有圖片的內容時，使用 `vision()`。回傳結果是文字。

```python
import akasha

qa = akasha.ask(model="gemini:gemini-2.5-flash")
answer = qa.vision(
    prompt="這張名片上有哪些資訊？",
    image_path="business-card.png",
)
print(answer)
```

`gemini:gemini-2.5-flash` 可以接收圖片與文字，並回傳文字答案；它不會修改圖片。

## 使用 `gen_image` 生成圖片

當沒有來源圖片，而是想根據文字描述建立新圖片時，使用 `gen_image()`。

```python
import akasha

output_path = akasha.gen_image(
    prompt="一台紅色小腳踏車，水彩畫風格",
    model="openai:gpt-image-1",
    save_path="bicycle.png",
)
print(output_path)
```

預設圖片模型是 `openai:gpt-image-1`。實際可用的模型取決於所設定的 provider 帳號與 API。

## 使用 `edit_image` 編輯圖片

當你想移除、增加或修改既有圖片中的內容時，使用 `edit_image()`。

```python
import akasha

output_path = akasha.edit_image(
    prompt="移除腳踏車，並在右側增加一盆綠色盆栽",
    images="bicycle.png",
    model="openai:gpt-image-1",
    save_path="edited.png",
)
print(output_path)
```

也可以提供多張參考圖片：

```python
output_path = akasha.edit_image(
    prompt="使用第一張圖片的主體，搭配第二張圖片的背景",
    images=["subject.png", "background.png"],
    save_path="combined.png",
)
```

圖片生成與編輯 API 會將結果儲存到 `save_path`，並回傳該路徑。
