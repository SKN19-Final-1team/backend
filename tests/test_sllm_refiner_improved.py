"""
Unit tests for enhanced sllm_refiner module

Tests:
1. Context-aware prompt generation
2. JSON extraction from various formats
3. Refinement result parsing
4. Validation logic
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm.delivery.sllm_refiner import (
    refinement_prompt_with_context,
    extract_json_from_text,
    parse_refinement_result,
    validate_refinement,
    refinement_payload_with_context
)


def test_context_aware_prompt():
    """Test that context is properly injected into prompt"""
    print("\n=== Test 1: Context-aware Prompt Generation ===")
    
    text = "나라사람카드 바우처"
    known_entities = ["나라사랑카드", "신세계상품권"]
    morphology_hints = [("나라사랑카드", "NNP"), ("바우처", "NNG")]
    
    prompt = refinement_prompt_with_context(text, known_entities, morphology_hints)
    
    # Check that known entities are in prompt
    assert "나라사랑카드" in prompt, "Known entity not in prompt"
    assert "신세계상품권" in prompt, "Known entity not in prompt"
    
    # Check that STT hallucination examples are in prompt
    assert "연예비" in prompt, "STT hallucination example missing"
    assert "연회비" in prompt, "STT hallucination correction missing"
    
    print("✅ PASS: Context properly injected into prompt")
    print(f"   - Known entities: {known_entities}")
    print(f"   - Morphology hints: {len(morphology_hints)} items")
    print(f"   - Prompt length: {len(prompt)} chars")


def test_json_extraction():
    """Test JSON extraction from various formats"""
    print("\n=== Test 2: JSON Extraction ===")
    
    test_cases = [
        # Case 1: Clean JSON
        (
            '{"original": "test", "refined": "테스트"}',
            True,
            "Clean JSON"
        ),
        # Case 2: Markdown code block
        (
            '```json\n{"original": "test", "refined": "테스트"}\n```',
            True,
            "Markdown code block"
        ),
        # Case 3: JSON with surrounding text
        (
            'Here is the result: {"original": "test", "refined": "테스트"} Done.',
            True,
            "JSON with surrounding text"
        ),
        # Case 4: Nested JSON
        (
            '{"original": "test", "refined": "테스트", "corrections": [{"from": "a", "to": "b"}]}',
            True,
            "Nested JSON"
        ),
        # Case 5: Invalid (no JSON)
        (
            'This is just plain text without JSON',
            False,
            "No JSON"
        ),
    ]
    
    for text, should_succeed, description in test_cases:
        result = extract_json_from_text(text)
        
        if should_succeed:
            assert result is not None, f"Failed to extract JSON: {description}"
            assert "{" in result and "}" in result, f"Invalid JSON format: {description}"
            print(f"✅ PASS: {description}")
        else:
            assert result is None, f"Should not extract JSON: {description}"
            print(f"✅ PASS: {description} (correctly returned None)")


def test_parse_refinement_result():
    """Test parsing of LLM output"""
    print("\n=== Test 3: Refinement Result Parsing ===")
    
    # Case 1: Valid JSON with all fields
    llm_output_1 = """{
        "original": "연예비 납부",
        "refined": "연회비 납부",
        "corrections": [
            {"from": "연예비", "to": "연회비", "reason": "STT hallucination"}
        ],
        "confidence": 0.95
    }"""
    
    result_1 = parse_refinement_result(llm_output_1, "연예비 납부")
    assert result_1["text"] == "연회비 납부", "Incorrect refined text"
    assert len(result_1["corrections"]) == 1, "Corrections not parsed"
    assert result_1["confidence"] == 0.95, "Confidence not parsed"
    print("✅ PASS: Valid JSON with all fields")
    
    # Case 2: Minimal JSON (only refined field)
    llm_output_2 = '{"refined": "교정된 텍스트"}'
    result_2 = parse_refinement_result(llm_output_2, "원본 텍스트")
    assert result_2["text"] == "교정된 텍스트", "Failed to parse minimal JSON"
    print("✅ PASS: Minimal JSON")
    
    # Case 3: Invalid JSON (fallback to original)
    llm_output_3 = "This is not JSON"
    result_3 = parse_refinement_result(llm_output_3, "원본 텍스트")
    assert result_3["text"] == "원본 텍스트", "Should fallback to original"
    print("✅ PASS: Invalid JSON fallback")
    
    # Case 4: Empty output (fallback to original)
    result_4 = parse_refinement_result("", "원본 텍스트")
    assert result_4["text"] == "원본 텍스트", "Should fallback to original on empty"
    print("✅ PASS: Empty output fallback")


def test_validation():
    """Test refinement validation logic"""
    print("\n=== Test 4: Refinement Validation ===")
    
    # Case 1: Valid refinement - entity preserved
    original = "나라사랑카드 바우처"
    refined = "나라사랑카드 바우처 신청"
    known_entities = ["나라사랑카드"]
    
    validated, warnings = validate_refinement(original, refined, known_entities)
    assert validated == refined, "Should accept valid refinement"
    assert len(warnings) == 0, f"Should have no warnings, got: {warnings}"
    print("✅ PASS: Valid refinement accepted")
    
    # Case 2: Entity was in original but removed in refined - should revert
    original_2 = "나라사랑카드 바우처"
    refined_2 = "카드 바우처"  # "나라사랑카드" removed
    known_entities_2 = ["나라사랑카드"]
    
    validated_2, warnings_2 = validate_refinement(original_2, refined_2, known_entities_2)
    assert validated_2 == original_2, f"Should revert to original, got: {validated_2}"
    assert len(warnings_2) > 0, "Should have warnings"
    print(f"✅ PASS: Missing entity detected - {warnings_2[0]}")
    
    # Case 3: Empty refinement - should revert
    validated_3, warnings_3 = validate_refinement(original, "", known_entities)
    assert validated_3 == original, "Should revert on empty refinement"
    assert len(warnings_3) > 0, "Should warn about empty result"
    print("✅ PASS: Empty refinement detected and reverted")
    
    # Case 4: Overly long refinement - warning only
    refined_long = original * 5
    validated_4, warnings_4 = validate_refinement(original, refined_long, known_entities)
    assert len(warnings_4) > 0, "Should warn about length"
    print("✅ PASS: Length warning generated")


def test_payload_generation():
    """Test payload generation with context"""
    print("\n=== Test 5: Payload Generation ===")
    
    text = "연예비 납부"
    known_entities = ["나라사랑카드"]
    morphology_hints = [("연회비", "NNG"), ("납부", "NNG")]
    
    payload = refinement_payload_with_context(
        text,
        known_entities=known_entities,
        morphology_hints=morphology_hints
    )
    
    # Check payload structure
    assert "model" in payload, "Missing model field"
    assert "messages" in payload, "Missing messages field"
    assert len(payload["messages"]) == 2, "Should have system and user messages"
    
    # Check that context is in system message
    system_content = payload["messages"][0]["content"]
    assert "나라사랑카드" in system_content, "Known entity not in system prompt"
    
    print("✅ PASS: Payload properly generated with context")
    print(f"   - Model: {payload['model']}")
    print(f"   - Temperature: {payload['temperature']}")
    print(f"   - Max tokens: {payload['max_tokens']}")


def run_all_tests():
    """Run all unit tests"""
    print("=" * 60)
    print("Running sllm_refiner Unit Tests")
    print("=" * 60)
    
    try:
        test_context_aware_prompt()
        test_json_extraction()
        test_parse_refinement_result()
        test_validation()
        test_payload_generation()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
