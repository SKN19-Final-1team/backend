from typing import Dict
from app.utils.runpod_connector import call_runpod, get_runpod_status
from app.llm.delivery.sllm_refiner import (
    get_model_name as get_refiner_model,
    refinement_payload,
    parse_refinement_result
)
from app.llm.delivery.sllm_makser import (
    get_model_name as get_masker_model,
    masking_payload,
    parse_masking_result
)

def refine_text(text: str) -> Dict[str, any]:
    """
    텍스트를 교정합니다.
    """
    if not text or not text.strip():
        return {"text": text, "keywords": []}
        
    payload = refinement_payload(text)
    llm_output = call_runpod(payload)
    return parse_refinement_result(llm_output, text)

def mask_text(text: str) -> Dict[str, any]:
    """
    텍스트에서 개인정보를 마스킹합니다.
    """
    if not text or not text.strip():
        return {"text": text, "info": []}
        
    payload = masking_payload(text)
    llm_output = call_runpod(payload)
    return parse_masking_result(llm_output, text)

def deliver(text: str) -> Dict[str, any]:
    """
    텍스트 교정 -> 마스킹 순차 처리
    """
    # 1. 텍스트 교정
    refine_result = refine_text(text)
    refined_text = refine_result["text"]
    
    # 2. 마스킹
    mask_result = mask_text(refined_text)
    masked_text = mask_result["text"]
    
    return {
        "original": text,
        "refined": refined_text,
        "masked": masked_text,
        "keywords": refine_result.get("keywords", []),
        "detected_info": mask_result.get("info", [])
    }
