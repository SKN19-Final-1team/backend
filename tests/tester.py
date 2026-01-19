import pandas as pd
from prompts import get_system_prompt, PERSONA_PROFILE, TEST_SCENARIOS
from evaluator import ModelEvaluator

MODELS = [
    {
        "name": "Gemma-2-2B-It",
        "path": "./models/Gemma-2-2b-it-Q4_K_M.gguf"
    },
    {
        "name": "Llama-3.2-3B-Instruct",
        "path": "./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    },
    {
        "name": "Qwen2.5-3B-Instruct",
        "path": "./models/Qwen2.5-3B-Instruct-Q4_K_M.gguf"
    },
    {
        "name": "EXAONE-3.5-2.4B-Instruct",
        "path": "./models/EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf"
    }
]

def main():
    results = []
    system_prompt = get_system_prompt(PERSONA_PROFILE)

    for model in MODELS:

        evaluator = ModelEvaluator(model['path'], model['name'])
        evaluator.load_model()
        
        if not evaluator.llm:
            continue

        # 시나리오별 테스트
        print(f"▶ {model['name']} 테스트")
        for scenario in TEST_SCENARIOS:
            result = evaluator.generate_and_measure(system_prompt, scenario)
            if result:
                results.append(result)
                print(f"   [Query] {scenario[:30]}...")
                print(f"   [Resp]  {result['response'][:30]}...")
                print(f"   [Perf]  TTFT: {result['ttft_sec']}s | TPS: {result['tps']}")
        
        # 모델 언로드 (다음 모델을 위해 메모리 비우기)
        evaluator.unload_model()

    # 4. 결과 저장 및 출력
    df = pd.DataFrame(results)
    
    # 가독성을 위한 컬럼 정렬
    df = df[['model', 'ttft_sec', 'tps', 'input', 'response']]
    
    print("\n📊 [Final Comparison Report]")
    print(df.groupby('model')[['ttft_sec', 'tps']].mean()) # 모델별 평균 성능
    
    # CSV 저장
    df.to_csv("sllm_persona_test_result.csv", index=False, encoding='utf-8-sig')
    print("\n💾 Results saved to 'sllm_persona_test_result.csv'")

if __name__ == "__main__":
    main()