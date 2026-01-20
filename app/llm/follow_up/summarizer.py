import requests

class SLLMSummarizer:
    def __init__(self):
        # RunPod 서버 실행 후 나오는 Proxy URL 입력
        # 끝에 /generate
        self.remote_url = "https://your-runpod-id-8000.proxy.runpod.net/generate"

    def summarize(self, script):
        try:
            # RunPod 서버로 요청 전달
            response = requests.post(
                self.remote_url,
                json={"script": script},
                timeout=120
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Remote Server Error: {response.status_code}", "raw": response.text}
        except Exception as e:
            return {"error": f"Connection Error: {str(e)}"}