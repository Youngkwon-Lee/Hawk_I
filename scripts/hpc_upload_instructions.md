# HPC Leg Agility Data Upload Instructions

## Step 1: HPC 코드 업데이트 (SSH로 접속)

```bash
ssh gun3856@10.246.246.111

# HPC 내부에서 실행
cd ~/hawkeye
git pull origin main

# 기존 pkl 파일 삭제
rm -f leg_agility_train.pkl leg_agility_test.pkl
rm -f data/leg_agility_train.pkl data/leg_agility_test.pkl

# 폴더 구조 확인
ls -la scripts/extract_leg_agility*

# HPC에서 나오기
exit
```

## Step 2: 로컬에서 파일 업로드 (Git Bash에서 실행)

**IMPORTANT**: 아래 명령어는 **로컬 Windows Git Bash**에서 실행!

```bash
# Git Bash 열기 (HPC가 아님!)
cd /c/Users/YK/tulip/Hawkeye

# 파일 업로드 (train - 11MB, 약 1분 소요)
scp data/leg_agility_train_v2.pkl gun3856@10.246.246.111:~/hawkeye/data/

# 파일 업로드 (test - 2.4MB, 약 10초 소요)
scp data/leg_agility_test_v2.pkl gun3856@10.246.246.111:~/hawkeye/data/
```

## Step 3: HPC에서 확인

```bash
ssh gun3856@10.246.246.111

cd ~/hawkeye/data
ls -lh leg_agility*v2.pkl

# 예상 출력:
# -rw-r--r-- 1 gun3856 users 11M  leg_agility_train_v2.pkl
# -rw-r--r-- 1 gun3856 users 2.4M leg_agility_test_v2.pkl

# 검증 (Python으로 확인)
conda activate hawkeye
python -c "import pickle; data=pickle.load(open('data/leg_agility_train_v2.pkl','rb')); print('Train:', data['X'].shape, 'Side:', len(data['side']))"
```

## Troubleshooting

### 오류: "Could not resolve hostname c:"
→ HPC 내부에서 scp를 실행했기 때문. **로컬 Git Bash**에서 실행 필요!

### 오류: "Host key verification failed"
→ 처음 접속 시 `yes` 입력 필요

### 업로드 속도 느림
→ 정상입니다. 11MB 파일은 약 1분 소요

## Quick Commands (로컬 Git Bash에서 한 번에)

```bash
# 1줄 명령어 (Git Bash에서 실행)
scp /c/Users/YK/tulip/Hawkeye/data/leg_agility_train_v2.pkl gun3856@10.246.246.111:~/hawkeye/data/ && \
scp /c/Users/YK/tulip/Hawkeye/data/leg_agility_test_v2.pkl gun3856@10.246.246.111:~/hawkeye/data/
```
