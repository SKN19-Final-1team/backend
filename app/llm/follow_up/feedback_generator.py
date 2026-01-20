import json
import openai
def generate_feedback_openai(prompt, model_name="gpt-4.1-mini"):
    try:
        response = openai.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        result_json = json.loads(content)

        return result_json

    except Exception as e:
        return f"{model_name} 호출 중 오류 발생 : {e}"