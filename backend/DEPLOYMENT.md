# CALL:ACT 배포 가이드

## 📋 배포 환경 정보

- **EC2 IP**: `43.201.107.173`
- **API 문서**: http://43.201.107.173/docs
- **Health Check**: http://43.201.107.173/api/v1/health
- **DB**: RDS PostgreSQL 17 (callact-db)
- **Redis**: EC2 내부 (추후 ElastiCache 전환 예정)

---

## 🚀 빠른 배포

### 1단계: EC2 접속

```bash
# SSH 키가 있는 디렉토리에서 실행
ssh -i callact-key.pem ubuntu@43.201.107.173
```

**SSH 키 파일이 없다면** 팀장에게 `callact-key.pem` 파일을 요청하세요.

### 2단계: 배포 실행

```bash
cd ~/backend
./deploy.sh
```

### 3단계: 확인

배포 스크립트가 자동으로:
- 최신 코드 다운로드 (`git pull`)
- 기존 서비스 중지
- Docker 이미지 빌드
- 서비스 시작
- Health check 확인

**끝!** 약 2-3분 후 배포 완료됩니다.

브라우저에서 http://43.201.107.173/docs 열어서 확인하세요!

---

## 📝 유틸리티 스크립트

### 배포
```bash
./deploy.sh
```
- 최신 코드 다운로드 (`git pull`)
- 기존 서비스 중지
- Docker 이미지 빌드
- 서비스 시작
- Health check 확인

### 로그 확인
```bash
./logs.sh           # 모든 서비스 로그
./logs.sh app       # FastAPI 앱 로그만
./logs.sh nginx     # Nginx 로그만
./logs.sh redis     # Redis 로그만
```

### 상태 확인
```bash
./status.sh
```
- 컨테이너 상태
- Health check
- Readiness check (DB + Redis)

### 서비스 중지
```bash
./stop.sh
```

---

## 🔧 수동 명령어 (고급 사용자용)

### Docker Compose 직접 실행
```bash
cd ~/backend

# 빌드 및 시작
docker compose -f docker-compose.prod.yml up --build -d

# 중지
docker compose -f docker-compose.prod.yml down

# 로그
docker compose -f docker-compose.prod.yml logs -f

# 상태
docker compose -f docker-compose.prod.yml ps
```

### 최신 코드 받기
```bash
cd ~/backend
git pull origin main
```

### 특정 컨테이너만 재시작
```bash
docker compose -f docker-compose.prod.yml restart app
docker compose -f docker-compose.prod.yml restart nginx
```

---

## 🌐 접속 정보

### 외부 접속 (인터넷에서)
- **API 문서**: http://43.201.107.173/docs
- **ReDoc 문서**: http://43.201.107.173/redoc
- **Health Check**: http://43.201.107.173/api/v1/health
- **WebSocket**: ws://43.201.107.173/api/v1/ws/call

### EC2 내부에서
- **API 문서**: http://localhost/docs
- **Health Check**: http://localhost/api/v1/health

---

## 🔧 배포 스크립트 설명

배포 스크립트는 EC2의 `~/backend/` 디렉토리에 있습니다.

### deploy.sh - 배포 실행
```bash
./deploy.sh
```
자동으로 최신 코드를 받아서 Docker 이미지를 빌드하고 서비스를 재시작합니다.

### logs.sh - 로그 확인
```bash
./logs.sh           # 모든 서비스 로그
./logs.sh app       # FastAPI 앱 로그만
./logs.sh nginx     # Nginx 로그만
```

### status.sh - 상태 확인
```bash
./status.sh
```
컨테이너 상태와 Health check를 확인합니다.

### stop.sh - 서비스 중지
```bash
./stop.sh
```

---

## ❗ 트러블슈팅

### 서비스가 시작되지 않을 때
```bash
# 로그 확인
./logs.sh app

# 컨테이너 상태 확인
docker compose -f docker-compose.prod.yml ps

# 모든 컨테이너 재시작
docker compose -f docker-compose.prod.yml down
docker compose -f docker-compose.prod.yml up --build -d
```

### DB 연결 실패
```bash
# .env.prod 파일 확인
cat .env.prod

# RDS 엔드포인트가 올바른지 확인
# DB_HOST, DB_PASSWORD 등 확인
```

### Health check 실패
```bash
# 앱 로그 확인
./logs.sh app

# 직접 health check
curl http://localhost/api/v1/health
```

---

## 🔐 보안 주의사항

- `.env.prod` 파일은 **절대 Git에 커밋하지 마세요**
- EC2 키 파일(`callact-key.pem`)은 안전한 곳에 보관
- RDS 보안그룹은 EC2에서만 접근 가능하도록 설정

---

## 📞 문제 발생 시

1. `./logs.sh`로 로그 확인
2. `./status.sh`로 상태 확인
3. 위 방법으로 해결 안 되면 팀원에게 연락

---

**배포 성공!** 🎉
