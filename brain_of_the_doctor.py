# Uncomment these lines if you're not using pipenv and want to load environment variables from a .env file
# from dotenv import load_dotenv
# load_dotenv()

# Step 1: Setup GROQ API key
import os
import base64
from groq import Groq

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Step 2: Convert image to required format
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Step 3: Setup Multimodal LLM
def analyze_image_with_query(query, model, encoded_image):
    client = Groq(api_key=GROQ_API_KEY)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": query
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                },
            ],
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
