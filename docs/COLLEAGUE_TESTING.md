# ParkiCheck → Hawk I 동료 테스트 절차

이 절차는 실제 동료 계정으로 다음 흐름을 확인하기 위한 운영 가이드입니다.

`ParkiCheck 수행·저장 → Hawk I 분석 → 임상 History 타임라인 확인`

서비스 역할 키를 브라우저나 동료에게 전달하지 않습니다. 환자 테스터는 가입 후 운영자의 명시적 승인을 받아 Hawk I 연구 조직에 연결됩니다. 의료진 권한은 이 절차로 부여하지 않습니다.

## 공개 테스트 주소

- ParkiCheck: <https://finger-tap-fx.vercel.app/>
- Hawk I 새 검사: <https://hawkeye-labeling-tool.vercel.app/test>
- Hawk I 기록: <https://hawkeye-labeling-tool.vercel.app/history>

Hawk I의 로컬 C0B 모델을 사용하는 동안에는 집 데스크탑의 모델 서버와 WSL 백엔드가 켜져 있어야 합니다.

## 1. 환자 테스터 준비

1. 동료가 ParkiCheck에서 본인 이메일로 가입합니다.
2. 이메일 인증이 요구되면 인증을 완료합니다.
3. ParkiCheck에 한 번 로그인해 초기 계정 생성을 완료합니다.
4. 아직 검사를 시작하지 않고 운영자에게 가입 이메일을 전달합니다.

로그인 전에는 `persons` 계정이 없어 승인할 수 없습니다.

## 2. 운영자 승인

운영자 컴퓨터에서 승인된 비밀 환경을 불러온 뒤 먼저 dry-run을 실행합니다. 이메일 입력은 화면에 표시되지 않으며 셸 기록에도 남지 않습니다.

```bash
cd /path/to/Hawk_I
set -a
source /path/to/approved/physio_app.production.env
set +a

uv run python backend/scripts/manage_colleague_tester.py approve \
  --organization-id 13d483d9-b785-41d4-8eb8-fc0b55d48c86
```

출력이 `mode=dry_run result=would_create`이면 대상과 승인자를 다시 확인한 후 실제 적용합니다.

```bash
uv run python backend/scripts/manage_colleague_tester.py approve \
  --organization-id 13d483d9-b785-41d4-8eb8-fc0b55d48c86 \
  --apply
```

이미 비활성화된 임상 연결은 자동으로 되살리지 않습니다. 상태를 검토한 경우에만 `--reactivate --apply`를 사용합니다.

## 3. 동료 테스트 시나리오

1. 동료가 ParkiCheck에서 로그아웃 후 다시 로그인하거나 페이지를 새로 엽니다.
2. 손가락 태핑 검사를 수행하고 결과 저장을 확인합니다.
3. Hawk I 분석 연결에 동의한 경우 분석 완료 상태를 확인합니다.
4. 승인된 의료진 계정으로 Hawk I History를 엽니다.
5. 해당 환자를 선택하고 새 ParkiCheck 검사, 분석 점수, 검사 시각이 같은 기록인지 확인합니다.
6. 복약 정보를 입력했다면 검사 전후 맥락과 함께 표시되는지 확인합니다.

테스트 중 실제 환자 개인정보나 임상 영상을 사용하지 않습니다. 동료 본인 데이터 또는 명시적으로 준비한 테스트 데이터만 사용합니다.

## 4. 테스트 종료 후 연결 비활성화

먼저 dry-run으로 대상을 확인하고 `--apply`를 추가합니다.

```bash
uv run python backend/scripts/manage_colleague_tester.py deactivate \
  --organization-id 13d483d9-b785-41d4-8eb8-fc0b55d48c86

uv run python backend/scripts/manage_colleague_tester.py deactivate \
  --organization-id 13d483d9-b785-41d4-8eb8-fc0b55d48c86 \
  --apply
```

이 작업은 계정이나 검사 기록을 삭제하지 않고 조직 연결 상태만 `inactive`로 바꿉니다.
