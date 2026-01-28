import pandas as pd
import random
import os
from dotenv import load_dotenv

load_dotenv()

from openai import OpenAI

# OpenAI API 클라이언트 초기화
# 환경변수에서 API 키를 가져오거나 직접 설정
api_key = os.getenv("OPENAI_API_KEY")  # 또는 직접 입력: "your-api-key-here"
client = OpenAI(api_key=api_key)

# 1. 데이터 로드
df = pd.read_csv("hana.csv")

emotions = ["불만", "성급함", "사투리", ""]
weights = [0.2, 0.2, 0.1, 0.5]

# 전체 행의 10%만 감정 기반 각색 대상으로 선정
total_rows = len(df)
sample_size = int(total_rows * 0.1)  # 10%

# 랜덤하게 10% 행 선택
sample_indices = random.sample(range(total_rows), sample_size)

def assign_emotion(row):
    # 선택된 10%에만 감정 배정, 나머지는 "일반"
    if row.name in sample_indices:
        return random.choices(emotions, weights=weights)[0]
    else:
        return "일반"

df['emotion'] = df.apply(assign_emotion, axis=1)

print(f"전체 행 수: {total_rows:,}")
print(f"감정 각색 대상: {sample_size:,} (10%)")
print(f"일반 처리: {total_rows - sample_size:,} (90%)")
print(f"\n감정 분포:")
print(df['emotion'].value_counts())

# 2. LLM에게 보낼 프롬프트 생성 함수
def make_rewrite_prompt(counselor_utterance_text, customer_utterance_text, emotion):
    if emotion == "일반":
        return None  # 일반은 변환 안 함
    
    prompt = f"""
    당신은 '드라마 대사 작가'입니다. 아래 대화에서 [고객]의 대사를 '{emotion}' 감정이 느껴지도록 실감 나게 각색해 주세요.
    
    [상황]
    상담원: {counselor_utterance_text}
    고객(원문): {customer_utterance_text}
    
    [요청사항]
    1. 의미는 훼손하지 말 것.
    2. '{emotion}'의 특징(반말, 사투리, 감탄사 등)을 극대화할 것.
    3. 오직 [각색된 대사]만 출력할 것.
    """
    return prompt

# 3. OpenAI GPT API를 호출하여 텍스트 변환
def rewrite_with_gpt(counselor_utterance_text, customer_utterance_text, emotion, model="gpt-4.1"):
    """
    OpenAI GPT 모델을 사용하여 고객 대사를 감정에 맞게 각색합니다.
    
    Args:
        counselor_utterance_text: 상담원 대사
        customer_utterance_text: 고객 원문 대사
        emotion: 적용할 감정 ("불만/분노", "성급함", "사투리(노인)", "일반")
        model: 사용할 GPT 모델 (기본값: "gpt-4o-mini")
    
    Returns:
        각색된 고객 대사 또는 원문 (일반인 경우)
    """
    if emotion == "일반":
        return customer_utterance_text  # 일반은 변환 안 함
    
    prompt = make_rewrite_prompt(counselor_utterance_text, customer_utterance_text, emotion)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "당신은 드라마 대사 작가입니다. 주어진 대사를 지정된 감정에 맞게 각색하세요."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        rewritten_text = response.choices[0].message.content.strip()
        return rewritten_text
    
    except Exception as e:
        print(f"API 호출 오류: {e}")
        return customer_utterance_text  # 오류 발생 시 원문 반환

# 4. 데이터프레임에 각색된 대사 추가 (체크포인트 포함)
def process_dataframe(df, model="gpt-4o-mini", checkpoint_interval=50, output_file="hana_rewritten.csv"):
    """
    데이터프레임의 모든 행에 대해 GPT를 사용하여 대사를 각색합니다.
    
    Args:
        df: 처리할 데이터프레임
        model: 사용할 GPT 모델
        checkpoint_interval: 체크포인트 저장 간격 (기본값: 50)
        output_file: 출력 파일명
    
    Returns:
        각색된 대사가 추가된 데이터프레임
    """
    # 결과를 저장할 컬럼 초기화 (컬럼이 없는 경우에만)
    if 'customer_utterance_rewritten' not in df.columns:
        df['customer_utterance_rewritten'] = ""
    if 'rewrite_emotion' not in df.columns:
        df['rewrite_emotion'] = ""
    
    total_rows = len(df)
    count = 0
    processed_count = 0
    skipped_count = 0
    
    for idx, row in df.iterrows():
        count += 1
        
        # 조건 1: 고객 발화가 4글자 미만이면 건너뛰기
        if len(str(row['customer_utterance'])) < 4:
            df.at[idx, 'customer_utterance_rewritten'] = row['customer_utterance']
            df.at[idx, 'rewrite_emotion'] = "건너뜀(짧음)"
            skipped_count += 1
            continue
        
        # 조건 2: 이미 각색된 내용이 있으면 건너뛰기
        if pd.notna(row.get('customer_utterance_rewritten')) and str(row.get('customer_utterance_rewritten')).strip() != "":
            skipped_count += 1
            continue
        
        # GPT로 각색
        rewritten = rewrite_with_gpt(
            row['counselor_utterance'], 
            row['customer_utterance'], 
            row['emotion'], 
            model
        )
        
        # 결과 저장
        df.at[idx, 'customer_utterance_rewritten'] = rewritten
        df.at[idx, 'rewrite_emotion'] = row['emotion']
        processed_count += 1
        
        # 체크포인트: 50개마다 출력 및 저장
        if count % checkpoint_interval == 0:
            print(f"진행 상황: {count}/{total_rows} 완료 ({count/total_rows*100:.1f}%) | 처리: {processed_count}, 건너뜀: {skipped_count}")
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"  → 체크포인트 저장 완료: {output_file}")
    
    # 최종 저장
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n최종 완료: {count}/{total_rows} ({count/total_rows*100:.1f}%)")
    
    return df

# 예시 실행
if __name__ == "__main__":
    print("\n전체 데이터 처리 중...")
    df = process_dataframe(df, model="gpt-4o-mini")
    print("처리 완료! 결과가 'hana_rewritten.csv'에 저장되었습니다.")