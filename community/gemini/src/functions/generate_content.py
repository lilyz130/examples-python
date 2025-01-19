from restack_ai.function import function, log
from pydantic import BaseModel
from google import genai
from google.genai import types
import base64

import os

class FunctionInputParams(BaseModel):
    user_content: str

@function.defn()
async def gemini_generate_content(input: FunctionInputParams) -> str:
    try:
        log.info("gemini_generate_content function started", input=input)
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        image_file = open("/Users/qiulin/Downloads/IMG_0539.jpeg", "rb")
        encoded_string = base64.b64encode(image_file.read())
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=[input.user_content, encoded_string]
        )
        log.info("gemini_generate_content function completed", response=response.text)
        return response.text
    except Exception as e:
        log.error("gemini_generate_content function failed", error=e)
        raise e
