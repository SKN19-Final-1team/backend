"""
화자분리 대화 교정 테스트
scripts_diarized.json 처리
sllm_refiner.py의 refine_diarized_batch 함수 테스트
"""
import sys
import json
import time
sys.path.insert(0, 'c:/SKN19/backend')

import io
import contextlib

RESULT_FILE = './diarized_result.txt'
INPUT_FILE = './scripts_diarized.json'

print("=" * 60)
print("화자분리 대화 교정 테스트 (Refactored Module)")
print("=" * 60)

print("\n모듈 로딩 중...")
with contextlib.redirect_stdout(io.StringIO()):
    with contextlib.redirect_stderr(io.StringIO()):
        from app.llm.delivery.sllm_refiner import refine_diarized_batch, MODEL_NAME

print(f"모델: {MODEL_NAME}")

# 데이터 로드
print("데이터 로딩...")
with open(INPUT_FILE, 'r', encoding='utf-8') as f:
    content = f.read().rstrip('.')
    data = json.loads(content)

print(f"총 {len(data)}개 케이스")

# 워밍업
print("모델 워밍업 중...")
warmup_start = time.time()
test_utt = [{"speaker": "agent", "message": "테스트"}]
with contextlib.redirect_stdout(io.StringIO()):
    with contextlib.redirect_stderr(io.StringIO()):
        _ = refine_diarized_batch(test_utt)
warmup_time = time.time() - warmup_start
print(f"워밍업 완료 ({warmup_time:.1f}초)")

# 결과 저장
with open(RESULT_FILE, 'w', encoding='utf-8') as f:
    def log(msg):
        f.write(msg + '\n')
        print(msg)
    
    log("=" * 80)
    log("화자분리 대화 교정 결과")
    log(f"함수: refiner_diarized_batch (sllm_refiner.py)")
    log(f"모델: {MODEL_NAME}")
    log("=" * 80)
    
    total_utterances = 0
    total_corrected = 0
    
    for case_name, utterances in data.items():
        log(f"\n{'='*80}")
        log(f"[{case_name.upper()}] - {len(utterances)}개 발화")
        log("=" * 80)
        
        case_start = time.time()
        
        # 교정 (correction_map + sLLM 배치)
        # sllm_refiner 내부에서 correction_map도 자동 적용됨
        refined_results = refine_diarized_batch(utterances)
        
        case_time = time.time() - case_start
        
        # 결과 출력
        for i, (orig, refined) in enumerate(zip(utterances, refined_results), 1):
            speaker_kr = "상담원" if orig["speaker"] == "agent" else "고객"
            original_text = orig["message"]
            final_text = refined["message"]
            
            total_utterances += 1
            if final_text != original_text:
                total_corrected += 1
            
            log(f"\n[{i}] {speaker_kr}")
            log(f"  원본: {original_text}")
            
            if final_text != original_text:
                log(f"  최종: {final_text}")
            else:
                log(f"  (변경 없음)")
        
        log(f"\n[처리시간] {case_time:.2f}초")
    
    log("\n" + "=" * 80)
    log("요약")
    log("=" * 80)
    log(f"총 발화: {total_utterances}개")
    log(f"교정된 발화: {total_corrected}개 ({total_corrected/total_utterances*100:.1f}%)")

print(f"\n결과 저장: {RESULT_FILE}")
