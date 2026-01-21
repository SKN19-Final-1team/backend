def evaluate_call(work_time, emotions):
    work_score = 0
    emotion_score = 0
    
    # 후처리 시간 평가
    time_indicator = 90
    
    if work_time <= time_indicator:
        work_score = 20
    elif work_time >= 90:
        if work_time < 120:
            work_score = 15
        elif work_time < 150:
            work_score = 10
        elif work_time < 200:
            work_score = 5
        else:
            work_score = 0
    
    # 감정 전환 평가
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

    # 단계별 점수 계산
    score_step_1 = get_step_score(emotions[0], emotions[1]) # 초반 -> 중반
    score_step_2 = get_step_score(emotions[1], emotions[2]) # 중반 -> 후반
    
    emotion_score = score_step_1 + score_step_2
    
    return {
        "work_score": work_score, 
        "emotion_score": emotion_score
        }