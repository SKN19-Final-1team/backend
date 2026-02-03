"""
DB 스크립트 설정 파일

모든 데이터를 backend_dev/app/db/data/ 하나로 통합 관리
Windows/Mac 호환성 보장 (Path 사용)
"""

from pathlib import Path

# ==============================================================================
# 기본 경로 설정
# ==============================================================================

# config.py 위치: backend_dev/app/db/scripts/config.py
# backend_dev/app/db/ 까지: parents[1]
DB_DIR = Path(__file__).resolve().parents[1]  # backend_dev/app/db

# 데이터 디렉토리 (모든 데이터 통합)
DATA_DIR = DB_DIR / "data"  # backend_dev/app/db/data

# 프로젝트 루트 디렉토리 (필요시 사용)
PROJECT_ROOT = Path(__file__).resolve().parents[4]  # call-act

# ==============================================================================
# 테디카드 데이터 경로
# ==============================================================================

TEDDY_DATA_DIR = DATA_DIR / "teddycard"  # backend_dev/app/db/data/teddycard

# 호환성을 위한 별칭 (기존 코드 지원)
TEDDY_DATA_DIR_PROD = TEDDY_DATA_DIR
TEDDY_DATA_DIR_DEV = TEDDY_DATA_DIR

# 키워드 사전 경로
KEYWORDS_DICT_DIR_PROD = TEDDY_DATA_DIR
KEYWORDS_DICT_DIR_DEV = TEDDY_DATA_DIR

# 카드 상품 데이터
CARD_PRODUCTS_FILE = TEDDY_DATA_DIR / "teddycard_card_products_with_embeddings.json"

# 서비스 가이드 데이터
SERVICE_GUIDES_FILE = TEDDY_DATA_DIR / "teddycard_service_guides_with_embeddings.json"

# 공지사항 데이터
NOTICES_FILE = TEDDY_DATA_DIR / "teddycard_notices_with_embeddings.json"

# 키워드 사전 데이터
KEYWORDS_DICT_FILE = TEDDY_DATA_DIR / "keywords_dict_v2_with_patterns.json"

# ==============================================================================
# 하나카드 데이터 경로
# ==============================================================================

HANA_DATA_DIR = DATA_DIR / "hana"  # backend_dev/app/db/data/hana
HANA_RDB_METADATA_FILE = HANA_DATA_DIR / "hana_rdb_metadata.json"
HANA_VECTORDB_FILE = HANA_DATA_DIR / "hana_vectordb_with_embeddings.json"

# ==============================================================================
# 기타 데이터 경로
# ==============================================================================

# 상담사/고객 데이터 (db/data/ 디렉토리 루트)
EMPLOYEES_DATA_FILE = DATA_DIR / "employeesData.json"
CUSTOMERS_DATA_FILE = DATA_DIR / "customersData.json"
