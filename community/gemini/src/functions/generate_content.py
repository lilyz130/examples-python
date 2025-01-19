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
        image_file_path = "/Users/qiulin/Downloads/IMG_0539.jpeg"
        image_file_path = input.user_content
        file_upload = client.files.upload(path=image_file_path)
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
        contents=["This image is encoded by base64 model. Now you are a dermatologist. Show me the analysis on the hair, \
        forehead, eyelids, nose, mouth and cheek. Do you have any recommendations on the product to improve face condition?", 
                      types.Content(role="user", parts=[
                        types.Part.from_uri(
                            file_uri=file_upload.uri,
                            mime_type=file_upload.mime_type
                        ),
                    ]
                ),]
        )
        log.info("gemini_generate_content function completed", response=response.text)
        return response.text
    except Exception as e:
        log.error("gemini_generate_content function failed", error=e)
        raise e
