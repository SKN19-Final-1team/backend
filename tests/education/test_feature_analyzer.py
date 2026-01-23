"""
Feature Analyzer 테스트
"""
import json
import sys
import os

from app.llm.education.feature_analyzer import analyze_consultation, format_analysis_for_db


def test_with_sample_consultation():
    """샘플 상담 데이터로 feature analyzer 테스트"""
    
    # consultation.json 로드
    with open("tests/education/consultation.json", "r", encoding="utf-8") as f:
        consultation = json.load(f)
    
    print("=" * 60)
    print("Feature Analyzer 테스트")
    print("=" * 60)
    print(f"\n상담 ID: {consultation['id']}")
    print(f"카테고리: {consultation['category']}")
    print(f"내용 미리보기: {consultation['content'][:150]}...")
    
    print("\n분석 시작...")
    analysis = analyze_consultation(consultation["content"])
    
    print("\n[분석 결과]")
    print(json.dumps(analysis, indent=2, ensure_ascii=False))
    
    print("\n[DB 저장 형식]")
    db_format = format_analysis_for_db(analysis)
    print(json.dumps(db_format, indent=2, ensure_ascii=False))
    
    print("\n✅ 테스트 완료")


if __name__ == "__main__":
    test_with_sample_consultation()
