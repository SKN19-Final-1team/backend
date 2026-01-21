def evaluate_call(emotions):
    """
    emotions: {"early": "...", "mid": "...", "late": "..."} 형태의 딕셔너리
    """
    emotion_score = 0
    
    # 감정 전환 평가 내부 함수
    def get_step_score(before, after):
        if before == after:
            return 3
        
        scores = {
            ("부정", "중립"): 5,
            ("부정", "긍정"): 10,
            ("중립", "긍정"): 10,
            ("중립", "부정"): 0,
            ("긍정", "부정"): 0,
            ("긍정", "중립"): 5
        }
        return scores.get((before, after), 0)

    score_step_1 = get_step_score(emotions.get('early'), emotions.get('mid')) # 초반 -> 중반
    score_step_2 = get_step_score(emotions.get('mid'), emotions.get('late'))  # 중반 -> 후반
    
    emotion_score = score_step_1 + score_step_2
    
    return {
        "emotion_score": emotion_score
    }