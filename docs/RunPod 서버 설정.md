# RunPod 서버 설정

진행 상태: 완료
담당자: Windy
기간: 2026/01/20
우선순위: 중간

### 설명

런팟에 sllm 모델만 올려놓고 로컬에서 통신하여 작업하기 위한 작업😘

### Todo

- [x]  자동화 스크립트 작성
    - 파이썬 프로젝트를 위해 필요한 사전작업을 수행
- [x]  스크립트 동작 및 외부 연결 테스트

### 수행한 작업

- **런팟 세팅 자동화 스크립트**
    
    터미널에서 실행되는 쉘 스크립트(.sh)로 작성
    
    https://gist.github.com/WindyAle/8b4c61a61c74e6f1401677ac49e1d4a0
    
    1. 파이썬 코딩을 위한 필수 패키지 설치
        - `zstd`, `curl`, `hugingface-hub`
    2. 모델 다운로드
        - 지정된 리포지토리 url에서 4bit 양자화된 gguf 파일을 찾아 다운로드
    3. llama-cpp-python 설치
        - CUDA 지원 설정 선택
    4. llama-cpp 서버 실행
        - host 0.0.0.0 → 공인 IP
        - port 8000 → 서버 분할 시 1씩 증가
        - api key → 공인 IP에 대한 보호 장치
- **런팟 템플릿 사용 및 자동화 적용**
    1. Runpod > My Templates > New Templates
        - Pod 열 때마다 자동으로 적용할 설정
        
    2. 템플릿 설정
        - 컨테이너 네임 → 자신이 알아볼 명칭 자유롭게
        - 컨테이너 이미지명은 실제 도커에 공식적으로 올라가있는 이미지명을 써야 함
            - 가장 보편적으로 쓰이는 것 적용
        - Start Command에 입력
            
            ```bash
            bash -c "apt-get update && apt-get install -y curl && curl -fsSL https://gist.githubusercontent.com/WindyAle/8b4c61a61c74e6f1401677ac49e1d4a0/raw/499d1c5467625a3e3d3350060880719a4e261741/start.sh | bash"
            ```
        
    3. 포트 설정
        - 사용할 포트번호와 그 번호를 식별할 라벨 입력
        - 한 서버에 모델 여러 개를 놓는다면 각 모델에 대해 포트를 할당해주는게 좋음
            - 한 포트에 다 넣으면 부하가 걸릴 가능성
        - 허깅페이스 토큰 입력
        
    4. 템플릿으로 Pod 열기
        - GPU 선택 후 나오는 아래 부분에서 Change Template을 눌러 자신이 지정한 템플릿 선택
        
    5. 서버 실행
        - Pod > Logs
        - 모든 동작이 끝나고 서버가 정상적으로 열리면 다음과 같이 출력됨
        - Address 뒤에 나오는 IP가 외부에서 접속가능한 통로가 됨
        
    6. 포트 확인
        - Pod > Connect
        - 아래 TCP 포트 확인
            - 8000번 포트에 연결되어있는 5자리 숫자 (사진에서는 40127)
        
    7. 로컬에서 연결