from openai import AsyncOpenAI
from dotenv import load_dotenv
import json
import os
from app.core.prompt import SUMMARIZE_SYSTEM_PROMPT

load_dotenv()

# client = AsyncOpenAI(
#     base_url=os.getenv("RUNPOD_URL"),
#     api_key=os.getenv("API_KEY")
# )

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

async def get_summarize(script):
    try:
        response = await client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
                {"role": "user", "content": f"상담 전문:\n{script}"}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )

        # 답변 반환
        return json.loads(response.choices[0].message.content)

    except Exception as e:
        return {"error": f"런팟 서버 통신 중 오류 발생: {str(e)}"}