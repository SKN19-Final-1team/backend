"""
연결형 대화 교정 테스트
scripts_for_test.txt 처리 (화자 구분 없이 한 줄로 연결된 형태)
"""
import sys
import time
sys.path.insert(0, 'c:/SKN19/backend')

import io
import contextlib
import re
import difflib

RESULT_FILE = './analysis_result.txt'
INPUT_FILE = './scripts_for_test.txt'

print("=" * 60)
print("연결형 대화 교정 테스트")
print("=" * 60)

print("\n모듈 로딩 중...")
with contextlib.redirect_stdout(io.StringIO()):
    with contextlib.redirect_stderr(io.StringIO()):
        from app.llm.delivery.deliverer import pipeline
        from app.llm.delivery.sllm_refiner import MODEL_NAME

print(f"모델: {MODEL_NAME}")

def parse_test_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    cases = []
    case_blocks = re.split(r'\[case \d+\]', content)[1:]
    for i, block in enumerate(case_blocks, 1):
        parts = re.split(r'\[script\]|\[stt\]', block)
        if len(parts) >= 3:
            cases.append({
                'case_id': i,
                'script': parts[1].strip(),
                'stt': parts[2].strip()
            })
    return cases

def calc_sim(t1, t2):
    return difflib.SequenceMatcher(None, ' '.join(t1.split()), ' '.join(t2.split())).ratio()

print("테스트 파일 로딩...")
cases = parse_test_file(INPUT_FILE)
print(f"총 {len(cases)}개 케이스")

print("모델 워밍업 중...")
warmup_start = time.time()
with contextlib.redirect_stdout(io.StringIO()):
    with contextlib.redirect_stderr(io.StringIO()):
        warmup_result = pipeline("테스트 문장입니다.", use_sllm=True)
warmup_time = time.time() - warmup_start

from app.utils.runpod_connector import get_runpod_status
status = get_runpod_status()

if warmup_result.get('raw_output'):
    print(f"RunPod 연결 성공 (워밍업 {warmup_time:.1f}초)")
else:
    print(f"RunPod 연결 실패 또는 sLLM 미응답")

with open(RESULT_FILE, 'w', encoding='utf-8') as f:
    def log(msg):
        f.write(msg + '\n')
        print(msg)
    
    log("=" * 80)
    log("연결형 대화 교정 결과")
    log(f"프롬프트: refinement_prompt_with_context (화자 구분 없음)")
    log(f"모델: {MODEL_NAME}")
    log("=" * 80)
    
    total_before = 0
    total_after = 0
    
    for c in cases:
        log(f"\n{'='*80}")
        log(f"[Case {c['case_id']}]")
        log("=" * 80)
        
        start_time = time.time()
        result = pipeline(c['stt'], use_sllm=True)
        elapsed = time.time() - start_time
        
        step1 = result['step1_corrected']
        final = result['refined']
        raw = result.get('raw_output', '')
        
        before_sim = calc_sim(c['stt'], c['script'])
        after_sim = calc_sim(final, c['script'])
        total_before += before_sim
        total_after += after_sim
        
        log(f"\n[원본 스크립트]")
        log(c['script'][:300] + ('...' if len(c['script']) > 300 else ''))
        
        log(f"\n[STT 입력]")
        log(c['stt'][:300] + ('...' if len(c['stt']) > 300 else ''))
        
        log(f"\n[correction_map 적용]")
        log(step1[:300] + ('...' if len(step1) > 300 else ''))
        
        log(f"\n[sLLM 최종 결과]")
        log(final[:300] + ('...' if len(final) > 300 else ''))
        
        log(f"\n[유사도] {before_sim:.2%} -> {after_sim:.2%} (개선: {after_sim - before_sim:+.2%})")
        log(f"[처리시간] {elapsed:.2f}초")
    
    log("\n" + "=" * 80)
    log("요약")
    log("=" * 80)
    avg_before = total_before / len(cases)
    avg_after = total_after / len(cases)
    log(f"평균 유사도: {avg_before:.2%} -> {avg_after:.2%} (개선: {avg_after - avg_before:+.2%})")

print(f"\n결과 저장: {RESULT_FILE}")
