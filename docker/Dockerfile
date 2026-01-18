# ==================================================
# CALL:ACT PostgreSQL + pgvector Docker 이미지
# ==================================================
# Description: PostgreSQL 17 with pgvector extension and initial schema setup
# Author: CALL:ACT Team
# Date: 2026-01-13
# ==================================================

# 베이스 이미지: pgvector/pgvector:pg17
FROM pgvector/pgvector:pg17

# 메타데이터
LABEL maintainer="CALL:ACT Team"
LABEL description="PostgreSQL 17 with pgvector extension for CALL:ACT project"
LABEL version="1.0"

# 작업 디렉토리 설정
WORKDIR /docker-entrypoint-initdb.d

# SQL 스크립트 파일 복사 (실행 순서: 숫자 순서대로)
# PostgreSQL은 /docker-entrypoint-initdb.d/ 디렉토리의 파일을 알파벳 순서로 실행
# 주의: 빌드 컨텍스트는 backend/ 디렉토리이므로 docker build는 backend/에서 실행해야 함
COPY app/db/scripts/db_setup.sql ./01_db_setup.sql
COPY app/db/scripts/02_setup_tedicard_tables.sql ./02_setup_tedicard_tables.sql
COPY app/db/scripts/03_setup_keyword_dictionary.sql ./03_setup_keyword_dictionary.sql

# 파일 권한 설정 (실행 가능하도록)
RUN chmod 644 ./01_db_setup.sql ./02_setup_tedicard_tables.sql ./03_setup_keyword_dictionary.sql

# 기본 포트 노출
EXPOSE 5432

# PostgreSQL 기본 명령어 사용 (베이스 이미지에서 상속)
# 컨테이너 시작 시 /docker-entrypoint-initdb.d/의 SQL 파일들이 자동 실행됨
