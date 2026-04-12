# Local Deployment Command

로컬 개발 환경 전체 스택 실행

## 백엔드 + 프론트엔드 동시 실행

### Terminal 1: Backend
```bash
cd backend
source venv/Scripts/activate  # Windows
# source venv/bin/activate    # Linux/Mac
python app.py
```

### Terminal 2: Frontend
```bash
cd frontend
npm run dev
```

## 접속 URL
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000
- **Health Check**: http://localhost:5000/api/health

## 환경 변수 확인
```bash
# Backend (.env)
OPENAI_API_KEY=sk-...  # Optional: for AI interpretation

# Frontend (.env.local)
NEXT_PUBLIC_API_URL=http://localhost:5000
```

## 빠른 검증
```bash
# API 상태 확인
curl http://localhost:5000/api/health

# 분석 테스트 (비디오 업로드)
curl -X POST -F "video=@test_video.mp4" http://localhost:5000/api/analysis/start
```

## 문제 해결
- Port 5000 사용 중: `lsof -i :5000` 또는 `netstat -ano | findstr :5000`
- CORS 오류: Backend app.py의 CORS 설정 확인
- 모듈 없음: `pip install -r requirements.txt`
