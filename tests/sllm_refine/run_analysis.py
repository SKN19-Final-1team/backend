"""
STT 교정 결과 분석 (correction_map + sLLM)
- 모델 로드 후부터 시간 측정
- 교정된 텍스트 전체 출력
"""
import sys
import time
sys.path.insert(0, 'c:/SKN19/backend')

import io
import contextlib

RESULT_FILE = './analysis_result.txt'

class TeeOutput:
    def __init__(self, file, stream):
        self.file = file
        self.stream = stream
    def write(self, data):
        self.file.write(data)
        self.stream.write(data)
    def flush(self):
        self.file.flush()
        self.stream.flush()

print("모듈 로딩 중...")
with contextlib.redirect_stdout(io.StringIO()):
    with contextlib.redirect_stderr(io.StringIO()):
        from app.llm.delivery.deliverer import pipeline
        from app.llm.delivery.morphology_analyzer import apply_text_corrections

import re
import difflib

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
cases = parse_test_file('./scripts_for_test.txt')

print("모델 워밍업 중...")
warmup_result = None
with contextlib.redirect_stdout(io.StringIO()):
    with contextlib.redirect_stderr(io.StringIO()):
        warmup_result = pipeline("테스트 문장입니다.", use_sllm=True)

# RunPod 연결 상태 체크
from app.utils.runpod_connector import get_runpod_status
status = get_runpod_status()
if status['configured'] and warmup_result and warmup_result.get('refined'):
    print(f"RunPod 연결 성공: {status['api_url']}")
else:
    print(f"RunPod 연결 실패: {status['api_url']} (configured: {status['configured']})")
print("워밍업 완료!")

with open(RESULT_FILE, 'w', encoding='utf-8') as f:
    tee = TeeOutput(f, sys.stdout)
    
    def log(msg):
        tee.write(msg + '\n')
    
    log("=" * 100)
    log("STT 교정 결과 분석 (correction_map + sLLM)")
    log("=" * 100)
    
    results = []
    
    for c in cases:
        log(f"\n{'='*100}")
        log(f"[Case {c['case_id']}]")
        log("=" * 100)
        
        stt = c['stt']
        script = c['script']
        
        start_time = time.time()
        result = pipeline(stt, use_sllm=True)
        elapsed = time.time() - start_time
        
        step1 = result['step1_corrected']
        final = result['refined']
        
        before_sim = calc_sim(stt, script)
        step1_sim = calc_sim(step1, script)
        final_sim = calc_sim(final, script)
        
        results.append({
            'case_id': c['case_id'],
            'before': before_sim,
            'step1': step1_sim,
            'final': final_sim,
            'time': elapsed
        })
        
        log(f"\n[원본 스크립트]")
        log(script)
        
        log(f"\n[STT 전사 (Whisper)]")
        log(stt)
        
        log(f"\n[Step1: correction_map 교정]")
        log(step1)
        
        log(f"\n[최종: sLLM 교정]")
        log(final)
        
        log(f"\n[유사도] 원본: {before_sim:.2%} -> Step1: {step1_sim:.2%} -> 최종: {final_sim:.2%} (개선: {final_sim - before_sim:+.2%})")
        log(f"[처리 시간] {elapsed:.2f}초")
    
    # 결과 요약
    log("\n" + "=" * 100)
    log("결과 요약")
    log("=" * 100)
    
    log(f"\n{'Case':^6} | {'원본 STT':^10} | {'Step1':^10} | {'sLLM':^10} | {'개선율':^10} | {'시간':^8}")
    log("-" * 70)
    
    total_before = 0
    total_final = 0
    total_time = 0
    
    for r in results:
        improvement = r['final'] - r['before']
        log(f"{r['case_id']:^6} | {r['before']:^10.2%} | {r['step1']:^10.2%} | {r['final']:^10.2%} | {improvement:^+10.2%} | {r['time']:^7.2f}s")
        total_before += r['before']
        total_final += r['final']
        total_time += r['time']
    
    log("-" * 70)
    avg_before = total_before / len(results)
    avg_final = total_final / len(results)
    avg_improvement = avg_final - avg_before
    log(f"{'평균':^6} | {avg_before:^10.2%} | {'-':^10} | {avg_final:^10.2%} | {avg_improvement:^+10.2%} | {total_time:^7.2f}s")
    
    log("\n" + "=" * 100)
    log("테스트 완료")
    log("=" * 100)

print(f"\n결과가 {RESULT_FILE}에 저장되었습니다.")
