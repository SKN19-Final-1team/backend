"""
DB 스크립트 설정 파일

개발 환경(_dev)과 프로덕션 환경 간 경로 전환을 쉽게 하기 위한 설정
Windows/Mac 호환성 보장 (Path 사용)
"""

import os
from pathlib import Path
from typing import Literal

# 환경 타입: 'dev' 또는 'prod'
# dev: data-preprocessing_dev 사용
# prod: data-preprocessing 사용
# 기본값: 'prod' (팀 레포는 프로덕션 환경)
ENV_TYPE: Literal['dev', 'prod'] = os.getenv('DB_SCRIPTS_ENV', 'prod').lower()

# 프로젝트 루트 디렉토리
# config.py 위치: backend/app/db/scripts/config.py
# call-act 루트까지: parents[4]
PROJECT_ROOT = Path(__file__).resolve().parents[4]  # call-act

# 테디카드 데이터 경로
if ENV_TYPE == 'dev':
    DATA_DIR = PROJECT_ROOT / "data-preprocessing_dev"
    TEDDY_DATA_DIR_PROD = PROJECT_ROOT / "data-preprocessing" / "data" / "teddycard"
    TEDDY_DATA_DIR_DEV = DATA_DIR / "data" / "teddycard"
    KEYWORDS_DICT_DIR_PROD = PROJECT_ROOT / "data-preprocessing" / "data" / "teddycard"
    KEYWORDS_DICT_DIR_DEV = DATA_DIR / "data" / "teddycard"  # 팀 레포와 동일한 구조
else:
    DATA_DIR = PROJECT_ROOT / "data-preprocessing"
    TEDDY_DATA_DIR_PROD = DATA_DIR / "data" / "teddycard"
    TEDDY_DATA_DIR_DEV = None
    KEYWORDS_DICT_DIR_PROD = DATA_DIR / "data" / "teddycard"
    KEYWORDS_DICT_DIR_DEV = None

# 하나카드 데이터 경로 (항상 프로덕션 경로)
HANA_DATA_DIR = PROJECT_ROOT / "data-preprocessing" / "data" / "hana"
HANA_RDB_METADATA_FILE = HANA_DATA_DIR / "hana_rdb_metadata.json"
HANA_VECTORDB_FILE = HANA_DATA_DIR / "hana_vectordb_with_embeddings.json"

# 상담사 데이터 경로 (db/data/ 디렉토리)
# config.py 위치: backend/app/db/scripts/config.py
# db/data/ 까지: parents[1] / "data"
DB_DIR = Path(__file__).resolve().parents[1]  # backend/app/db
EMPLOYEES_DATA_FILE = DB_DIR / "data" / "employeesData.json"
