import os
from pathlib import Path
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv(Path(__file__).parent.parent.parent / '.env', override=False)

# 상수
BASE_DIR = Path(__file__).parent.parent.parent.parent.parent

# 환경 변수
# DB
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", "5432")) if os.getenv("DB_PORT") else None
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

BATCH_SIZE = int(os.getenv("DB_LOAD_BATCH_SIZE", "100"))
COMMIT_INTERVAL = int(os.getenv("DB_COMMIT_INTERVAL", "500"))

# REDIS
DIALOGUE_REDIS_URL = os.getenv("DIALOGUE_REDIS_URL")