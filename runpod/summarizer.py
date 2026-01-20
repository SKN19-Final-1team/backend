import re
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class SLLMSummarizer:
    def __init__(self, model_name="kakaocorp/kanana-1.5-8b-instruct-2505"):
        # 모델 및 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("모델 로딩 완료.")

    # 프롬프트
    def _make_prompt(self, script):
        return f"""
        상담 스크립트를 바탕으로 아래 JSON 형식에 맞춰 응답하세요

        ### 제약 사항
        1. 출력은 JSON 데이터만 허용합니다
        2. JSON 외의 서론, 결론, 마크다운 기호(```), ```json\n은 절대 포함하지 마세요
        3. ▲는 포함하지 마세요

        ### 상담 스크립트
        {script}

        ### 출력 형식 (JSON)
        {{
            "title": "상담의 핵심 주제를 나타내는 간결한 제목",
            "status": "'진행중', '완료' 중 택일",
            "inquiry": "고객이 문의한 핵심 내용을 1줄로 요약",
            "process": ["상담 과정 요약", "1단계", "2단계"],
            "result": "상담 결과 요약",
            "next_step": "상담 종료 후 상담원이 추가로 할 일 (없으면 '')",
            "transfer_dep": "이관이 필요한 경우 부서명 (없으면 '')",
            "transfer_note": "이관 부서에 전달할 내용 (없으면 '')"
        }}
        """

    def _clean_json(self, text):
        try:
            # 마크다운 태그 및 주석 제거
            text = re.sub(r'```json|```', '', text)
            text = re.sub(r'//.*', '', text)
            
            # 중괄호 { } 사이의 내용만 추출
            match = re.search(r'(\{.*\})', text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            return {"error": "JSON 형식을 찾을 수 없음", "raw": text}
        
        except Exception as e:
            return {"error": f"JSON 파싱 실패: {e}", "raw": text}

    def summarize(self, script):
        try:
            prompt = self._make_prompt(script)
            messages = [{"role": "system", "content": prompt}]
            
            # 템플릿 적용 및 인코딩
            inputs = self.tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True, 
                return_tensors="pt"
            ).to(self.model.device)

            outputs = self.model.generate(
                inputs,
                max_new_tokens=1024,
                repetition_penalty=1.2,
                do_sample=False,
                temperature=0.0,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )

            full_content = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
            
            return self._clean_json(full_content)

        except Exception as e:
            return {"error": f"수행 중 오류 발생: {e}"}