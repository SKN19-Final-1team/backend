"""
sLLM Text Refinement Module

This module loads a local small Language Model (sLLM) to refine and correct
STT (Speech-to-Text) transcription errors before the text is passed to the router.

The module is completely standalone and does not modify any existing logic.
"""

import os
from typing import Optional
from llama_cpp import Llama

# Global model instance (singleton pattern)
_sllm_model: Optional[Llama] = None
_model_loaded = False
_model_load_failed = False


def _load_sllm_model() -> Optional[Llama]:
    """
    Load the sLLM model from the path specified in environment variables.
    
    Returns:
        Llama model instance or None if loading fails
    """
    global _sllm_model, _model_loaded, _model_load_failed
    
    if _model_loaded:
        return _sllm_model
    
    if _model_load_failed:
        return None
    
    try:
        model_path = os.getenv("SLLM_MODEL_PATH")
        if not model_path:
            print("[sLLM] SLLM_MODEL_PATH not set in environment variables")
            _model_load_failed = True
            return None
        
        if not os.path.exists(model_path):
            print(f"[sLLM] Model file not found: {model_path}")
            _model_load_failed = True
            return None
        
        print(f"[sLLM] Loading model from: {model_path}")
        
        # Load model with optimized settings for text correction
        _sllm_model = Llama(
            model_path=model_path,
            n_ctx=512,  # Context window (sufficient for short STT corrections)
            n_threads=4,  # CPU threads
            n_gpu_layers=0,  # Set to -1 for full GPU offload if available
            verbose=False,
        )
        
        _model_loaded = True
        print("[sLLM] Model loaded successfully")
        return _sllm_model
        
    except Exception as e:
        print(f"[sLLM] Failed to load model: {e}")
        _model_load_failed = True
        return None


def refine_stt_text(text: str) -> str:
    """
    Refine STT text using local sLLM model to correct transcription errors.
    
    Args:
        text: Original STT transcription text
        
    Returns:
        Refined/corrected text, or original text if refinement fails
    """
    # Check if sLLM is enabled
    if os.getenv("SLLM_ENABLED", "false").lower() != "true":
        return text
    
    # Return original text if empty
    if not text or not text.strip():
        return text
    
    # Load model (singleton)
    model = _load_sllm_model()
    if model is None:
        return text
    
    try:
        # Create prompt for text correction
        prompt = f"""다음은 음성인식(STT)으로 변환된 텍스트입니다. 오타나 잘못 인식된 부분을 수정하여 정확한 한국어 문장으로 교정해주세요. 카드 관련 용어나 금융 용어는 정확하게 유지해주세요.

원본: {text}

교정된 텍스트:"""

        # Generate correction
        response = model(
            prompt,
            max_tokens=128,
            temperature=0.1,  # Low temperature for more deterministic output
            top_p=0.9,
            stop=["원본:", "\n\n"],
            echo=False,
        )
        
        # Extract refined text
        refined = response["choices"][0]["text"].strip()
        
        # Validate output
        if refined and len(refined) > 0 and len(refined) < len(text) * 3:
            print(f"[sLLM] Original: {text}")
            print(f"[sLLM] Refined:  {refined}")
            return refined
        else:
            print(f"[sLLM] Invalid refinement output, using original text")
            return text
            
    except Exception as e:
        print(f"[sLLM] Error during text refinement: {e}")
        return text


def is_sllm_enabled() -> bool:
    """
    Check if sLLM text refinement is enabled.
    
    Returns:
        True if enabled, False otherwise
    """
    return os.getenv("SLLM_ENABLED", "false").lower() == "true"


def get_model_status() -> dict:
    """
    Get current model loading status.
    
    Returns:
        Dictionary with model status information
    """
    return {
        "enabled": is_sllm_enabled(),
        "model_loaded": _model_loaded,
        "model_load_failed": _model_load_failed,
        "model_path": os.getenv("SLLM_MODEL_PATH"),
    }
