"""
STT 교정 결과 분석 테스트

scripts_for_test.txt의 4개 케이스에 대해:
1. [stt] 텍스트를 교정 파이프라인에 통과
2. [script] 원본과 비교
3. 결과 분석

사용법:
    C:\\Users\\bsjun\\anaconda3\\envs\\final_env\\python.exe tests/sllm_refine/test_real_stt.py
"""

import sys
import re
import difflib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.llm.delivery.deliverer import pipeline
from app.llm.delivery.morphology_analyzer import apply_text_corrections


def parse_test_file(filepath: str) -> list:
    """테스트 파일 파싱"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    cases = []
    case_blocks = re.split(r'\[case \d+\]', content)[1:]  # 첫 번째 빈 요소 제외
    
    for i, block in enumerate(case_blocks, 1):
        # [script]와 [stt] 분리
        parts = re.split(r'\[script\]|\[stt\]', block)
        if len(parts) >= 3:
            script = parts[1].strip()
            stt = parts[2].strip()
            cases.append({
                "case_id": i,
                "script": script,
                "stt": stt
            })
    
    return cases


def calculate_similarity(text1: str, text2: str) -> float:
    """두 텍스트의 유사도 계산"""
    # 공백 정규화
    t1 = ' '.join(text1.split())
    t2 = ' '.join(text2.split())
    
    matcher = difflib.SequenceMatcher(None, t1, t2)
    return matcher.ratio()


def find_corrections(original: str, corrected: str) -> list:
    """교정된 부분 찾기"""
    corrections = []
    
    # 단어 단위 비교
    orig_words = original.split()
    corr_words = corrected.split()
    
    matcher = difflib.SequenceMatcher(None, orig_words, corr_words)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == 'replace':
            orig_part = ' '.join(orig_words[i1:i2])
            corr_part = ' '.join(corr_words[j1:j2])
            corrections.append(f"{orig_part} -> {corr_part}")
        elif op == 'delete':
            orig_part = ' '.join(orig_words[i1:i2])
            corrections.append(f"{orig_part} -> (삭제)")
        elif op == 'insert':
            corr_part = ' '.join(corr_words[j1:j2])
            corrections.append(f"(추가) -> {corr_part}")
    
    return corrections


def test_correction_map_only():
    """correction_map만 사용한 교정 테스트"""
    print("=" * 80)
    print("correction_map 단독 교정 테스트")
    print("=" * 80)
    
    filepath = Path(__file__).parent / "scripts_for_test.txt"
    cases = parse_test_file(str(filepath))
    
    results = []
    
    for case in cases:
        print(f"\n{'='*80}")
        print(f"[Case {case['case_id']}]")
        print("=" * 80)
        
        stt_text = case['stt']
        script_text = case['script']
        
        # correction_map 적용
        corrected = apply_text_corrections(stt_text)
        
        # 유사도 계산
        before_sim = calculate_similarity(stt_text, script_text)
        after_sim = calculate_similarity(corrected, script_text)
        improvement = after_sim - before_sim
        
        # 교정된 부분 찾기
        corrections = find_corrections(stt_text, corrected)
        
        print(f"\n[원본 스크립트 (일부)]")
        print(script_text[:200] + "..." if len(script_text) > 200 else script_text)
        
        print(f"\n[STT 전사 (일부)]")
        print(stt_text[:200] + "..." if len(stt_text) > 200 else stt_text)
        
        print(f"\n[교정 결과 (일부)]")
        print(corrected[:200] + "..." if len(corrected) > 200 else corrected)
        
        print(f"\n[교정된 항목] ({len(corrections)}개)")
        for c in corrections[:10]:  # 최대 10개만 표시
            print(f"  - {c}")
        if len(corrections) > 10:
            print(f"  ... 외 {len(corrections) - 10}개")
        
        print(f"\n[유사도]")
        print(f"  교정 전: {before_sim:.2%}")
        print(f"  교정 후: {after_sim:.2%}")
        print(f"  개선율:  {improvement:+.2%}")
        
        results.append({
            "case_id": case['case_id'],
            "before_sim": before_sim,
            "after_sim": after_sim,
            "improvement": improvement,
            "corrections_count": len(corrections)
        })
    
    # 전체 요약
    print("\n" + "=" * 80)
    print("전체 결과 요약")
    print("=" * 80)
    
    print(f"\n{'Case':^6} | {'교정 전':^10} | {'교정 후':^10} | {'개선율':^10} | {'교정 수':^8}")
    print("-" * 60)
    
    total_before = 0
    total_after = 0
    
    for r in results:
        print(f"{r['case_id']:^6} | {r['before_sim']:^10.2%} | {r['after_sim']:^10.2%} | {r['improvement']:^+10.2%} | {r['corrections_count']:^8}")
        total_before += r['before_sim']
        total_after += r['after_sim']
    
    avg_before = total_before / len(results)
    avg_after = total_after / len(results)
    avg_improvement = avg_after - avg_before
    
    print("-" * 60)
    print(f"{'평균':^6} | {avg_before:^10.2%} | {avg_after:^10.2%} | {avg_improvement:^+10.2%} |")
    
    return results


def test_full_pipeline():
    """전체 파이프라인 (correction_map + sLLM) 테스트"""
    print("\n" + "=" * 80)
    print("전체 파이프라인 교정 테스트 (correction_map + sLLM)")
    print("=" * 80)
    
    filepath = Path(__file__).parent / "scripts_for_test.txt"
    cases = parse_test_file(str(filepath))
    
    results = []
    
    for case in cases:
        print(f"\n{'='*80}")
        print(f"[Case {case['case_id']}]")
        print("=" * 80)
        
        stt_text = case['stt']
        script_text = case['script']
        
        # 전체 파이프라인 적용
        result = pipeline(stt_text, use_sllm=True)
        corrected = result['refined']
        step1_corrected = result['step1_corrected']
        
        # 유사도 계산
        before_sim = calculate_similarity(stt_text, script_text)
        step1_sim = calculate_similarity(step1_corrected, script_text)
        after_sim = calculate_similarity(corrected, script_text)
        
        print(f"\n[원본 스크립트 (일부)]")
        print(script_text[:200] + "...")
        
        print(f"\n[STT 전사 (일부)]")
        print(stt_text[:200] + "...")
        
        print(f"\n[Step1: correction_map 결과 (일부)]")
        print(step1_corrected[:200] + "...")
        
        print(f"\n[최종: sLLM 결과 (일부)]")
        print(corrected[:200] + "...")
        
        print(f"\n[유사도]")
        print(f"  원본 STT:        {before_sim:.2%}")
        print(f"  correction_map:  {step1_sim:.2%} ({step1_sim - before_sim:+.2%})")
        print(f"  sLLM 최종:       {after_sim:.2%} ({after_sim - before_sim:+.2%})")
        
        results.append({
            "case_id": case['case_id'],
            "before_sim": before_sim,
            "step1_sim": step1_sim,
            "after_sim": after_sim,
        })
    
    return results


def main():
    print("\n" + "=" * 80)
    print("STT 교정 결과 분석 테스트")
    print("=" * 80)
    
    # sLLM 포함 여부
    include_sllm = input("\nsLLM 포함 테스트? (y/n, 시간 소요): ").strip().lower() == 'y'
    
    if include_sllm:
        test_full_pipeline()
    else:
        test_correction_map_only()
    
    print("\n" + "=" * 80)
    print("테스트 완료")
    print("=" * 80)


if __name__ == "__main__":
    main()
