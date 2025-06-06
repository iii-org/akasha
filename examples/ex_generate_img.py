import akasha

### currently only support openai/azure openai  and gemini (model="gemini:gemini-2.0-flash-preview-image-generation") ###
# please noted that your openai account may need verification to use protected model like "gpt-image-1"
# also, if you are using azure openai, please make sure your AZURE_API_VERSION is set to "2025-04-01-preview" in your .env file
###  generate image with prompt, you can select "high", "medium", "low" quality or size like "256x256", "512x512", "1024x1024"
save_path = akasha.gen_image(
    prompt="一隻可愛的絨毛娃娃，是北海道的長尾山雀，坐在白雪的樹枝上唱歌",
    model="gemini:gemini-2.0-flash-preview-image-generation",
    save_path="長尾山雀.png",
    env_file=".env3",
)


### edit the source image with the prompt, can based on a list of image or a single image
save_path = akasha.edit_image(
    model="openai:gpt-image-1",
    prompt="增加一隻可愛的鯊魚娃娃在旁邊",
    images="長尾山雀.png",
    save_path="鯊鯊.png",
    env_file=".env3",
)
