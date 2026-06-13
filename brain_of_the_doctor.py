import os
import base64
from io import BytesIO
from PIL import Image
from groq import Groq
from config import load_env_file, require_env

load_env_file()

# Step 2: Convert image to required format
def encode_image(image_path, max_size=(1024, 1024), quality=75):
    image = Image.open(image_path).convert("RGB")
    image.thumbnail(max_size)

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# Step 3: Setup Multimodal LLM
def analyze_image_with_query(query, model, encoded_image=None):
    client = Groq(api_key=require_env("GROQ_API_KEY"))
    content = [
        {
            "type": "text",
            "text": query
        }
    ]

    if encoded_image:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_image}",
                },
            }
        )

    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]

    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model
    )

    return chat_completion.choices[0].message.content

# Example usage:
# image_path = "acne.jpg"
# encoded_img = encode_image(image_path)
# result = analyze_image_with_query("Is there something wrong with my face?", "llama-3.2-90b-vision-preview", encoded_img)
# print(result)
