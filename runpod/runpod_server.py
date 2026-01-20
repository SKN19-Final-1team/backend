# RunPod에서 실행할 파일
from fastapi import FastAPI, Request
from pydantic import BaseModel
from summarizer import SLLMSummarizer
import uvicorn

app = FastAPI()
summarizer = SLLMSummarizer()

class ScriptRequest(BaseModel):
    script: str

@app.post("/generate")
async def generate(data: ScriptRequest):
    result = summarizer.summarize(data.script)
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)