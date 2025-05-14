import akasha

### currently only support openai and azure openai ###
# please noted that your openai account may need verification to use protected model like "gpt-image-1"

###  generate image with prompt, you can select "high", "medium", "low" quality or size like "256x256", "512x512", "1024x1024"
save_path = akasha.gen_image(
    "一隻可愛的絨毛娃娃，是北海道的長尾山雀，坐在白雪的樹枝上唱歌",
    save_path="長尾山雀.png",
    env_file=".env3",
)


### edit the source image with the prompt, can based on a list of image or a single image
save_path = akasha.edit_image(
    "增加一隻可愛的IKEA鯊魚娃娃在旁邊",
    images="長尾山雀.png",
    save_path="鯊鯊.png",
    env_file=".env3",
)
