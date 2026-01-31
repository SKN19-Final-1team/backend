import json
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
from app.core.prompt import FEEDBACK_SYSTEM_PROMPT

load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

async def generate_feedback(script, model_name="gpt-4.1-mini"):
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": FEEDBACK_SYSTEM_PROMPT},
                {"role": "user", "content": f"상담 스크립트:\n{script}"}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        result_json = json.loads(content)

        return result_json

    except Exception as e:
        return f"{model_name} 호출 중 오류 발생 : {e}"