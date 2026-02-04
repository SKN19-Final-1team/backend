"""
card_products 테이블 데이터 품질 보완 스크립트

기능:
1. annual_fee_domestic/global NULL 값 보완 (metadata.full_content에서 추출)
2. brand NULL 값 보완 (카드명/full_content에서 추론)
3. main_benefits 잘림 확인 및 보완
"""

import re
from typing import Dict, Any, Optional, Tuple, List

from psycopg2.extensions import connection as psycopg2_connection
from psycopg2.extras import RealDictCursor

from . import connect_db


# ==============================================================================
# 연회비 추출 패턴
# ==============================================================================

ANNUAL_FEE_PATTERNS = [
    # 국내전용 패턴
    (r'국내전용[^\d]*?(\d+)[만천백]?\s*원?', 'domestic'),
    (r'Local[^\d]*?(\d+)[만천백]?\s*원?', 'domestic'),
    # 해외겸용 패턴
    (r'해외겸용[^\d]*?(\d+)[만천백]?\s*원?', 'global'),
    (r'VISA[^\d]*?(\d+)[만천백]?\s*원?', 'global'),
    (r'Mastercard[^\d]*?(\d+)[만천백]?\s*원?', 'global'),
    # 총연회비 패턴
    (r'총연회비[^\d]*?(\d+)[만천백]?\s*원?', 'domestic'),
    # 테이블 형식 패턴
    (r'\|\s*국내전용[^|]*\|\s*(\d+)[만천백]?\s*원?\s*\|', 'domestic'),
    (r'\|\s*해외겸용[^|]*\|\s*(\d+)[만천백]?\s*원?\s*\|', 'global'),
]


def parse_korean_number(text: str, match_value: str) -> Optional[int]:
    """한글 숫자 표현을 정수로 변환

    예: "2만 7천원" -> 27000, "15000" -> 15000
    """
    if not match_value:
        return None

    try:
        # 순수 숫자인 경우
        if match_value.isdigit():
            value = int(match_value)
            # 문맥에서 만/천 단위 확인
            context_start = max(0, text.find(match_value) - 10)
            context_end = text.find(match_value) + len(match_value) + 5
            context = text[context_start:context_end]

            if '만' in context and value < 100:
                return value * 10000
            elif '천' in context and value < 100:
                return value * 1000
            return value
        return None
    except (ValueError, AttributeError):
        return None


def extract_annual_fee_from_content(full_content: str) -> Tuple[Optional[int], Optional[int]]:
    """full_content에서 연회비 정보 추출

    Returns:
        (domestic_fee, global_fee): 국내전용, 해외겸용 연회비
    """
    if not full_content:
        return None, None

    domestic_fee = None
    global_fee = None

    for pattern, fee_type in ANNUAL_FEE_PATTERNS:
        matches = re.finditer(pattern, full_content, re.IGNORECASE)
        for match in matches:
            value = parse_korean_number(full_content, match.group(1))
            if value and value > 0:
                if fee_type == 'domestic' and domestic_fee is None:
                    domestic_fee = value
                elif fee_type == 'global' and global_fee is None:
                    global_fee = value

    return domestic_fee, global_fee


# ==============================================================================
# 브랜드 추론 패턴
# ==============================================================================

BRAND_PATTERNS = [
    (r'\bVISA\b', 'visa'),
    (r'\bMastercard\b', 'mastercard'),
    (r'\bMaster\b', 'mastercard'),
    (r'\bAMEX\b', 'amex'),
    (r'\bAmerican Express\b', 'amex'),
    (r'\bUnionPay\b', 'unionpay'),
    (r'\b은련\b', 'unionpay'),
    (r'\bLocal\b', 'local'),
    (r'국내전용', 'local'),
]


def infer_brand_from_content(name: str, full_content: str) -> Optional[str]:
    """카드명과 full_content에서 브랜드 추론

    Returns:
        브랜드 문자열 (visa, mastercard, amex, unionpay, local) 또는 None
    """
    # 카드명에서 먼저 검색
    for pattern, brand in BRAND_PATTERNS:
        if re.search(pattern, name, re.IGNORECASE):
            return brand

    # full_content에서 검색
    if full_content:
        for pattern, brand in BRAND_PATTERNS:
            if re.search(pattern, full_content, re.IGNORECASE):
                return brand

    # 기본값: local (국내전용)
    return 'local'


# ==============================================================================
# main_benefits 보완
# ==============================================================================

def check_main_benefits_truncation(main_benefits: str, full_content: str) -> bool:
    """main_benefits가 잘려있는지 확인

    Returns:
        True if truncated, False otherwise
    """
    if not main_benefits or not full_content:
        return False

    # main_benefits가 full_content보다 현저히 짧고, 문장이 완성되지 않은 경우
    if len(main_benefits) < len(full_content) * 0.3:
        # 문장이 완료되지 않은 경우 (마지막이 "다음과 같은 사" 등으로 끝나는 경우)
        incomplete_endings = ['다음과 같은 ', '에 따라 ', '을 위해 ', '경우에는 ', '있습니다. -']
        for ending in incomplete_endings:
            if main_benefits.rstrip().endswith(ending.rstrip()):
                return True

    return False


# ==============================================================================
# 메인 수정 함수
# ==============================================================================

def fix_card_products_data(conn: psycopg2_connection, dry_run: bool = True) -> Dict[str, Any]:
    """card_products 테이블 데이터 품질 보완

    Args:
        conn: PostgreSQL 연결
        dry_run: True인 경우 실제 수정하지 않고 결과만 반환

    Returns:
        수정 결과 통계
    """
    print("\n" + "=" * 60)
    print("[card_products 데이터 품질 보완]")
    print("=" * 60)
    print(f"모드: {'DRY RUN (실제 수정 없음)' if dry_run else '실제 수정'}")

    cursor = conn.cursor(cursor_factory=RealDictCursor)

    # 현재 데이터 조회
    cursor.execute("""
        SELECT id, name, brand, annual_fee_domestic, annual_fee_global,
               main_benefits, metadata
        FROM card_products
    """)
    rows = cursor.fetchall()

    stats = {
        'total': len(rows),
        'brand_null_fixed': 0,
        'annual_fee_domestic_fixed': 0,
        'annual_fee_global_fixed': 0,
        'main_benefits_truncated': 0,
        'updates': []
    }

    for row in rows:
        card_id = row['id']
        name = row['name'] or ''
        current_brand = row['brand']
        current_domestic = row['annual_fee_domestic']
        current_global = row['annual_fee_global']
        main_benefits = row['main_benefits'] or ''
        metadata = row['metadata'] or {}
        full_content = metadata.get('full_content', '')

        updates = {}

        # 1. brand NULL 보완
        if current_brand is None:
            new_brand = infer_brand_from_content(name, full_content)
            if new_brand:
                updates['brand'] = new_brand
                stats['brand_null_fixed'] += 1
                print(f"  [BRAND] {card_id}: NULL -> '{new_brand}'")

        # 2. annual_fee_domestic NULL 보완
        if current_domestic is None:
            extracted_domestic, _ = extract_annual_fee_from_content(full_content)
            if extracted_domestic:
                updates['annual_fee_domestic'] = extracted_domestic
                stats['annual_fee_domestic_fixed'] += 1
                print(f"  [FEE_DOM] {card_id}: NULL -> {extracted_domestic}")

        # 3. annual_fee_global NULL 보완
        if current_global is None:
            _, extracted_global = extract_annual_fee_from_content(full_content)
            if extracted_global:
                updates['annual_fee_global'] = extracted_global
                stats['annual_fee_global_fixed'] += 1
                print(f"  [FEE_GLB] {card_id}: NULL -> {extracted_global}")

        # 4. main_benefits 잘림 확인
        if check_main_benefits_truncation(main_benefits, full_content):
            stats['main_benefits_truncated'] += 1
            print(f"  [TRUNCATED] {card_id}: main_benefits가 잘려있음 (길이: {len(main_benefits)})")

        # 업데이트 실행
        if updates and not dry_run:
            set_clause = ", ".join([f"{k} = %s" for k in updates.keys()])
            values = list(updates.values()) + [card_id]
            cursor.execute(f"""
                UPDATE card_products
                SET {set_clause}, updated_at = NOW()
                WHERE id = %s
            """, values)
            stats['updates'].append({'id': card_id, 'updates': updates})

    if not dry_run:
        conn.commit()

    cursor.close()

    # 결과 출력
    print("\n" + "-" * 40)
    print("[결과 요약]")
    print(f"  총 카드 수: {stats['total']}")
    print(f"  brand NULL 보완: {stats['brand_null_fixed']}건")
    print(f"  annual_fee_domestic NULL 보완: {stats['annual_fee_domestic_fixed']}건")
    print(f"  annual_fee_global NULL 보완: {stats['annual_fee_global_fixed']}건")
    print(f"  main_benefits 잘림 감지: {stats['main_benefits_truncated']}건")
    if not dry_run:
        print(f"  실제 업데이트: {len(stats['updates'])}건")
    print("-" * 40)

    return stats


def get_card_products_quality_report(conn: psycopg2_connection) -> Dict[str, Any]:
    """card_products 테이블 품질 리포트 생성

    Returns:
        품질 리포트 딕셔너리
    """
    cursor = conn.cursor(cursor_factory=RealDictCursor)

    report = {}

    # 전체 개수
    cursor.execute("SELECT COUNT(*) as cnt FROM card_products")
    report['total_count'] = cursor.fetchone()['cnt']

    # NULL 값 통계
    cursor.execute("""
        SELECT
            COUNT(*) FILTER (WHERE brand IS NULL) as brand_null,
            COUNT(*) FILTER (WHERE annual_fee_domestic IS NULL) as fee_domestic_null,
            COUNT(*) FILTER (WHERE annual_fee_global IS NULL) as fee_global_null,
            COUNT(*) FILTER (WHERE embedding IS NULL) as embedding_null,
            COUNT(*) FILTER (WHERE main_benefits IS NULL OR main_benefits = '') as benefits_empty
        FROM card_products
    """)
    null_stats = cursor.fetchone()
    report['null_stats'] = dict(null_stats)

    # 브랜드별 분포
    cursor.execute("""
        SELECT brand, COUNT(*) as cnt
        FROM card_products
        GROUP BY brand
        ORDER BY cnt DESC
    """)
    report['brand_distribution'] = [dict(r) for r in cursor.fetchall()]

    # 카드 타입별 분포
    cursor.execute("""
        SELECT card_type, COUNT(*) as cnt
        FROM card_products
        GROUP BY card_type
        ORDER BY cnt DESC
    """)
    report['card_type_distribution'] = [dict(r) for r in cursor.fetchall()]

    cursor.close()

    return report


def print_quality_report(conn: psycopg2_connection):
    """품질 리포트 출력"""
    report = get_card_products_quality_report(conn)

    print("\n" + "=" * 60)
    print("[card_products 테이블 품질 리포트]")
    print("=" * 60)

    print(f"\n총 카드 수: {report['total_count']}")

    print("\n[NULL 값 통계]")
    for key, value in report['null_stats'].items():
        print(f"  {key}: {value}건")

    print("\n[브랜드별 분포]")
    for item in report['brand_distribution']:
        brand = item['brand'] or 'NULL'
        print(f"  {brand}: {item['cnt']}건")

    print("\n[카드 타입별 분포]")
    for item in report['card_type_distribution']:
        print(f"  {item['card_type']}: {item['cnt']}건")

    print("=" * 60)


# ==============================================================================
# CLI 엔트리포인트
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='card_products 데이터 품질 보완')
    parser.add_argument('--dry-run', action='store_true', default=True,
                        help='실제 수정 없이 결과만 확인 (기본값: True)')
    parser.add_argument('--apply', action='store_true',
                        help='실제 데이터 수정 적용')
    parser.add_argument('--report', action='store_true',
                        help='품질 리포트만 출력')

    args = parser.parse_args()

    conn = connect_db()

    try:
        if args.report:
            print_quality_report(conn)
        else:
            dry_run = not args.apply
            fix_card_products_data(conn, dry_run=dry_run)
    finally:
        conn.close()
