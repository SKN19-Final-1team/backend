import os
import sys
import uuid
import torch
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# 1. Patch torch.load (Must be before importing TTS)
_original_load = torch.load

def _safe_load(*args, **kwargs):
    # weights_only 옵션이 명시되지 않았다면 False로 강제 설정
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)

torch.load = _safe_load

print("[Test] torch.load patched for XTTS compatibility.")

# 2. Initialize TTS
from TTS.api import TTS

print("[Test] Initializing TTS model (this may take a while)...")
try:
    # gpu=False as per test.py reference, or check cuda availability
    use_gpu = torch.cuda.is_available()
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=use_gpu)
    print(f"[Test] TTS model initialized successfully (GPU: {use_gpu})")
except Exception as e:
    print(f"[Test] Failed to initialize TTS: {e}")
    sys.exit(1)

# Add project root to path to allow imports
sys.path.append(os.getcwd())

from app.llm.education import persona_generator
from app.llm.education import tts_speaker

def run_simulation():
    print("=" * 70)
    print("[Standalone Test] Education Simulation with TTS")
    print("=" * 70)

    # 1. Start Simulation Setup
    print("\n[Step 1] Initializing Session")
    category = "도난/분실 신청/해제"
    difficulty = "beginner"
    
    # Generate Persona (Manual, as per test_conversation.py)
    customer_profile = {
        "name": "김영희",
        "age_group": "40대",
        "personality_tags": ["emotional", "expressive"],  # As per user's update
        "communication_style": {
            "tone": "neutral",
            "speed": "moderate"
        },
        "llm_guidance": "친절하고 명확하게 안내해주세요."
    }
    
    # Force difficult/emotional instructions as per user's prompt update
    # Note: user updated prompt template, so we just pass the profile.
    system_prompt = persona_generator.create_system_prompt(customer_profile, difficulty)
    session_id = str(uuid.uuid4())
    
    # Initialize Session
    session = tts_speaker.initialize_conversation(session_id, system_prompt, customer_profile)
    
    print(f"✅ Session Created: {session_id}")
    print(f"Customer: {customer_profile.get('name')}")
    print(f"Context: {category} ({difficulty})")

    # 2. Conversation Loop
    print("\n[Step 2] Start Conversation")
    print("-" * 70)
    
    turn = 1
    # Check for speaker wav
    speaker_wav = "0001.wav"
    if not os.path.exists(speaker_wav):
        print(f"[Warning] '{speaker_wav}' not found. Looking for alternatives...")
        if os.path.exists("output.wav"):
            speaker_wav = "output.wav"
            print(f"[Info] Using 'output.wav' as speaker reference.")
        else:
             print("[Warning] No .wav file found for speaker cloning. TTS might default or fail.")
             speaker_wav = None

    while True:
        try:
            agent_msg = input(f"\nTurn {turn} Agent (You): ").strip()
        except EOFError:
            break
            
        if agent_msg.lower() in ["종료", "quit", "exit"]:
            print("Ending conversation.")
            break
            
        if not agent_msg:
            continue
            
        # Process Message
        # Note: process_agent_input internally calls tts_engine.generate_speech.
        # Since we patched torch.load globally, the internal call might ALSO work now.
        # But we will explicitely generate TTS using our local object as requested.
        
        response_data = tts_speaker.process_agent_input(session_id, agent_msg)
        
        customer_text = response_data['customer_response']
        print(f"  Customer: {customer_text}")
        print(f"  (Turn: {response_data['turn_number']})")
        
        # Explicit TTS Generation
        output_filename = f"output_turn_{turn}.wav"
        print(f"  [TTS] Generating audio for turn {turn}...")
        
        try:
            # Preparing arguments
            tts_args = {
                "text": customer_text,
                "file_path": output_filename,
                "language": "ko"
            }
            if speaker_wav:
                 tts_args["speaker_wav"] = speaker_wav
            
            tts.tts_to_file(**tts_args)
            print(f"  ✅ Audio saved to: {output_filename}")
            
            # Attempt to play (Windows specific)
            try:
                import winsound
                winsound.PlaySound(output_filename, winsound.SND_FILENAME | winsound.SND_ASYNC)
            except:
                pass
                
        except Exception as e:
            print(f"  ❌ TTS Generation failed: {e}")

        turn += 1

    # 3. End Session
    print("\n[Step 3] Session Ended")
    summary = tts_speaker.end_conversation(session_id)
    print(f"Total Turns: {summary['turn_count']}")
    
if __name__ == "__main__":
    run_simulation()
