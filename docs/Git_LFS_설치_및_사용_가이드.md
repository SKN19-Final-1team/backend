# Git LFS 설치 및 사용 가이드

**작성일**: 2026-01-13  
**작성자**: CALL:ACT Team  
**버전**: v1.0

---

## 개요

이 프로젝트에서는 큰 파일(특히 JSON 데이터 파일)을 Git LFS(Large File Storage)로 관리합니다. 팀원들이 프로젝트를 clone하거나 pull할 때 Git LFS가 제대로 작동하도록 설정하는 방법을 안내합니다.

---

## 1. Git LFS란?

Git LFS는 Git의 확장 기능으로, 큰 파일을 효율적으로 관리할 수 있게 해줍니다.

**장점**:
- 큰 파일을 별도 저장소에 저장하여 Git 레포지토리 크기 감소
- 빠른 clone 및 pull 속도
- 대용량 파일 관리 가능

**이 프로젝트에서 사용하는 이유**:
- 전처리된 JSON 데이터 파일이 크기 때문에 (수 MB ~ 수십 MB)
- 일반 Git으로 관리하면 레포지토리가 너무 커짐
- 팀원들이 clone하는 시간이 오래 걸림

---

## 2. Git LFS 설치

### 2.1 Windows

**방법 1: Git for Windows 설치 시 포함**
- Git for Windows를 설치하면 Git LFS가 함께 설치됩니다
- 확인: `git lfs version` 명령어로 확인

**방법 2: 수동 설치**
1. [Git LFS 공식 사이트](https://git-lfs.github.com/)에서 다운로드
2. 설치 프로그램 실행
3. 설치 확인:
   ```bash
   git lfs version
   ```

### 2.2 macOS

```bash
# Homebrew 사용
brew install git-lfs

# 설치 확인
git lfs version
```

### 2.3 Linux

```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# 설치 확인
git lfs version
```

---

## 3. Git LFS 초기화

### 3.1 전역 초기화 (권장)

프로젝트를 처음 clone하기 전에 한 번만 실행:

```bash
git lfs install
```

**설명**:
- 이 명령어는 전역 Git 설정에 LFS 후크를 설치합니다
- 한 번만 실행하면 모든 Git 레포지토리에서 LFS가 작동합니다

### 3.2 프로젝트별 초기화

특정 프로젝트에서만 LFS를 사용하는 경우:

```bash
cd call-act
git lfs install --local
```

---

## 4. 프로젝트 Clone 및 Pull

### 4.1 처음 Clone하는 경우

각 팀 레포지토리를 개별적으로 clone합니다:

```bash
# 1. Git LFS 설치 확인
git lfs version

# 2. Git LFS 초기화 (처음 한 번만)
git lfs install

# 3. 각 팀 레포지토리 Clone
# Backend 레포
git clone https://github.com/SKN19-Final-1team/backend.git
cd backend
git lfs pull

# Data-preprocessing 레포 (큰 데이터 파일 포함)
cd ..
git clone https://github.com/SKN19-Final-1team/data-preprocessing.git
cd data-preprocessing
git lfs pull

# Frontend 레포
cd ..
git clone https://github.com/SKN19-Final-1team/frontend.git
cd frontend
git lfs pull
```

**중요**: 
- 각 레포지토리를 개별적으로 clone합니다
- `git lfs pull`을 실행하여 LFS 파일을 다운로드합니다
- `data-preprocessing` 레포에 큰 JSON 파일들이 LFS로 관리됩니다

### 4.2 기존 프로젝트에서 Pull하는 경우

```bash
# 1. Git LFS 설치 확인
git lfs version

# 2. Git LFS 초기화 (아직 안 했다면)
git lfs install

# 3. 각 레포지토리에서 최신 변경사항 가져오기
cd backend
git pull origin main
git lfs pull

cd ../data-preprocessing
git pull origin main
git lfs pull

cd ../frontend
git pull origin main
git lfs pull
```

---

## 5. 현재 프로젝트의 LFS 설정

### 5.1 .gitattributes 파일

`data-preprocessing/.gitattributes` 파일에 다음 설정이 있습니다:

```
*.json filter=lfs diff=lfs merge=lfs -text
```

**의미**:
- 모든 `.json` 파일을 Git LFS로 추적
- `filter=lfs`: LFS 필터 사용
- `diff=lfs`: diff 시 LFS 사용
- `merge=lfs`: merge 시 LFS 사용
- `-text`: 텍스트 파일이 아님을 명시

### 5.2 LFS로 추적되는 파일 확인

```bash
cd data-preprocessing
git lfs ls-files
```

**출력 예시**:
```
abc123def456 * data/teddycard/teddycard_service_guides_with_embeddings.json
def456ghi789 * data/teddycard/teddycard_card_products_with_embeddings.json
...
```

---

## 6. 문제 해결

### 6.1 "git lfs: command not found" 오류

**원인**: Git LFS가 설치되지 않음

**해결**:
1. Git LFS 설치 (위의 "2. Git LFS 설치" 참고)
2. 설치 후 터미널 재시작
3. `git lfs version`으로 확인

### 6.2 "This repository is configured for Git LFS but 'git-lfs' was not found" 오류

**원인**: Git LFS가 설치되지 않았거나 PATH에 없음

**해결**:
```bash
# Git LFS 설치 확인
git lfs version

# 설치되지 않았다면 설치
# Windows: Git for Windows 재설치 또는 Git LFS 수동 설치
# macOS: brew install git-lfs
# Linux: sudo apt-get install git-lfs
```

### 6.3 LFS 파일이 다운로드되지 않는 경우

**증상**: 파일이 있지만 내용이 비어있거나 포인터만 보임

**해결**:
```bash
# 1. Git LFS 초기화 확인
git lfs install

# 2. LFS 파일 수동 다운로드
git lfs pull

# 3. 특정 파일만 다운로드
git lfs pull --include="data/teddycard/*.json"

# 4. 서브모듈의 LFS 파일 다운로드
cd data-preprocessing
git lfs pull
```

### 6.4 "batch response: This repository is over its data quota" 오류

**원인**: Git LFS 저장소의 데이터 할당량 초과

**해결**:
1. 불필요한 큰 파일 제거
2. Git LFS 저장소 관리자에게 할당량 증가 요청
3. 또는 로컬에서만 필요한 파일은 `.gitignore`에 추가

### 6.5 LFS 파일이 보이지 않는 경우

**증상**: 레포를 clone했지만 LFS 파일이 비어있거나 포인터만 보임

**해결**:
```bash
# 1. 해당 레포로 이동
cd data-preprocessing

# 2. LFS 파일 다운로드
git lfs pull

# 3. 특정 파일만 다운로드
git lfs pull --include="data/teddycard/*.json"
```

### 6.6 LFS 파일이 너무 느리게 다운로드되는 경우

**해결**:
```bash
# 1. 병렬 다운로드 설정
git config lfs.concurrenttransfers 8

# 2. 또는 더 많은 동시 전송
git config lfs.concurrenttransfers 16
```

---

## 7. 팀원 체크리스트

새로운 팀원이 프로젝트를 시작할 때:

- [ ] Git LFS 설치 확인 (`git lfs version`)
- [ ] Git LFS 초기화 (`git lfs install`)
- [ ] 각 팀 레포지토리 Clone (backend, data-preprocessing, frontend)
- [ ] 각 레포에서 LFS 파일 다운로드 (`git lfs pull`)
- [ ] LFS 파일 확인 (`git lfs ls-files`)

---

## 8. 주의사항

### 8.1 LFS 파일 수정 시

LFS로 추적되는 파일을 수정하면:
1. 파일이 자동으로 LFS에 업로드됩니다
2. 커밋 시 일반 파일처럼 `git add` 및 `git commit`을 사용합니다
3. 별도의 LFS 명령어는 필요 없습니다

### 8.2 LFS 파일 크기 제한

GitHub의 Git LFS 무료 할당량:
- 저장소당 1GB 저장 공간
- 월 1GB 대역폭

**권장사항**:
- 불필요한 큰 파일은 제거
- 압축 가능한 파일은 압축 후 커밋
- 정말 필요한 파일만 LFS로 추적

### 8.3 .gitattributes 파일 관리

`.gitattributes` 파일은 반드시 커밋해야 합니다:
- 이 파일이 없으면 LFS가 작동하지 않습니다
- 팀원 모두가 동일한 설정을 사용해야 합니다

---

## 9. 유용한 명령어 모음

```bash
# Git LFS 버전 확인
git lfs version

# Git LFS 초기화
git lfs install

# LFS로 추적되는 파일 목록
git lfs ls-files

# LFS 파일 다운로드
git lfs pull

# 특정 패턴의 파일만 다운로드
git lfs pull --include="*.json"

# LFS 파일 상태 확인
git lfs status

# LFS 파일 정보 확인
git lfs pointer --file=path/to/file.json

# 각 레포에서 LFS 파일 다운로드
cd backend && git lfs pull
cd ../data-preprocessing && git lfs pull
cd ../frontend && git lfs pull

# LFS 설정 확인
git lfs env
```

---

## 10. 참고 자료

- [Git LFS 공식 문서](https://git-lfs.github.com/)
- [Git LFS GitHub](https://github.com/git-lfs/git-lfs)

---

## 11. 문제 발생 시

위의 문제 해결 방법으로 해결되지 않으면:

1. **팀 채널에 문의**: Git LFS 관련 문제 공유
2. **로그 확인**: `git lfs logs` 명령어로 상세 로그 확인
3. **재설치**: Git LFS 재설치 시도

---

**마지막 업데이트**: 2026-01-13
