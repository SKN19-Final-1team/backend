import pandas as pd
import random
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

load_dotenv()

from openai import OpenAI

# OpenAI API 클라이언트 초기화
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# 1. 데이터 로드
df = pd.read_csv("hana.csv")

# 감정 리스트 및 가중치
emotions = ["불만", "성급함", "사투리", "수다"]
weights = [0.3, 0.3, 0.2, 0.2]

# 4글자 이상인 발화만 필터링
valid_indices = df[df['customer_utterance'].apply(lambda x: len(str(x)) >= 4)].index.tolist()

# 4글자 이상인 발화 중에서 정확히 9,688개 선택
total_rows = len(df)
valid_rows = len(valid_indices)
sample_size = min(9688, valid_rows)
sample_indices = set(random.sample(valid_indices, sample_size))

print(f"=" * 60)
print(f"전체 행 수: {total_rows:,}")
print(f"4글자 이상 발화: {valid_rows:,}")
print(f"감정 각색 대상: {sample_size:,}개 선별")
print(f"=" * 60)

# 2. LLM 프롬프트 생성 함수
def make_rewrite_prompt(counselor_utterance, customer_utterance, emotion):
    prompt = f"""
당신은 '드라마 대사 작가'입니다. 아래 대화에서 [고객]의 대사를 '{emotion}' 감정이 느껴지도록 실감 나게 각색해 주세요.

[상황]
상담원: {counselor_utterance}
고객(원문): {customer_utterance}

[요청사항]
1. 의미는 훼손하지 말 것.
2. '{emotion}'의 특징(반말, 사투리, 감탄사 등)을 극대화할 것.
3. 오직 [각색된 대사]만 출력할 것.
"""
    return prompt

# 3. OpenAI GPT API 호출 (단일 행 처리)
def process_single_row(idx, row, emotion, model="gpt-4o-mini"):
    """단일 행을 GPT로 처리"""
    prompt = make_rewrite_prompt(row['counselor_utterance'], row['customer_utterance'], emotion)
    
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
        rewritten = response.choices[0].message.content.strip()
        
        return {
            'idx': idx,
            'counselor_utterance': row['counselor_utterance'],
            'customer_utterance': row['customer_utterance'],
            'emotion': emotion,
            'customer_utterance_rewritten': rewritten
        }
    except Exception as e:
        print(f"\nAPI 호출 오류 (인덱스 {idx}): {e}")
        return {
            'idx': idx,
            'counselor_utterance': row['counselor_utterance'],
            'customer_utterance': row['customer_utterance'],
            'emotion': emotion,
            'customer_utterance_rewritten': row['customer_utterance']
        }

# 4. 선별된 9,688개 행 병렬 처리
def process_selected_rows_parallel(df, selected_indices, model="gpt-4o-mini", max_workers=20):
    """선별된 행을 병렬로 GPT 처리"""
    
    indices_list = sorted(list(selected_indices))
    total = len(indices_list)
    
    print(f"\n[1단계] 선별된 {total:,}개 행 GPT 병렬 처리 시작 (동시 작업: {max_workers}개)")
    print("=" * 60)
    
    # 각 행에 대한 작업 준비
    tasks = []
    for idx in indices_list:
        row = df.loc[idx]
        emotion = random.choices(emotions, weights=weights)[0]
        tasks.append((idx, row, emotion))
    
    # 병렬 처리
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 모든 작업 제출
        futures = {executor.submit(process_single_row, idx, row, emotion, model): idx 
                   for idx, row, emotion in tasks}
        
        # 진행 상황 표시와 함께 결과 수집
        with tqdm(total=total, desc="GPT 처리 중", unit="행") as pbar:
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                pbar.update(1)
    
    print(f"완료: {total}/{total} (100.0%)")
    print("=" * 60)
    
    # 결과를 DataFrame으로 변환
    results_df = pd.DataFrame(results)
    # idx 순서로 정렬
    results_df = results_df.sort_values('idx').reset_index(drop=True)
    # idx 컬럼 제거
    results_df = results_df.drop('idx', axis=1)
    
    return results_df

# 5. 나머지 행 일괄 처리
def process_remaining_rows(df, selected_indices):
    """선별되지 않은 나머지 행은 '일반'으로 일괄 처리"""
    
    print(f"\n[2단계] 나머지 행 일괄 처리 시작")
    print("=" * 60)
    
    results = {
        'counselor_utterance': [],
        'customer_utterance': [],
        'emotion': [],
        'customer_utterance_rewritten': []
    }
    
    remaining_count = 0
    for idx in df.index:
        if idx not in selected_indices:
            row = df.loc[idx]
            results['counselor_utterance'].append(row['counselor_utterance'])
            results['customer_utterance'].append(row['customer_utterance'])
            results['emotion'].append('일반')
            results['customer_utterance_rewritten'].append(row['customer_utterance'])
            remaining_count += 1
    
    print(f"일괄 처리 완료: {remaining_count:,}개 행 ('일반' 감정, 원문 복사)")
    print("=" * 60)
    
    return pd.DataFrame(results)

# 실행
if __name__ == "__main__":
    import time
    start_time = time.time()
    
    # 1단계: 선별된 9,688개 행 병렬 GPT 처리
    selected_df = process_selected_rows_parallel(df, sample_indices, model="gpt-4o-mini", max_workers=20)
    
    # 2단계: 나머지 행 일괄 처리
    remaining_df = process_remaining_rows(df, sample_indices)
    
    # 3단계: 결합 및 저장
    print(f"\n[3단계] 결과 결합 및 저장")
    print("=" * 60)
    
    final_df = pd.concat([selected_df, remaining_df], ignore_index=True)
    
    # 4개 컬럼만 선택하여 저장
    output_df = final_df[['counselor_utterance', 'customer_utterance', 'emotion', 'customer_utterance_rewritten']]
    output_df.to_csv("hana_rewritten.csv", index=False, encoding='utf-8-sig')
    
    elapsed_time = time.time() - start_time
    
    print(f"최종 결과:")
    print(f"  - 전체 행 수: {len(output_df):,}")
    print(f"  - 컬럼: {list(output_df.columns)}")
    print(f"  - 처리 시간: {elapsed_time:.1f}초 ({elapsed_time/60:.1f}분)")
    print(f"\n감정 분포:")
    print(output_df['emotion'].value_counts())
    print("=" * 60)
    print("\n✅ 처리 완료! 결과가 'hana_rewritten.csv'에 저장되었습니다.")