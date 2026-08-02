# 학습 타깃 설계 — 라벨링한 것을 어떻게 쓸 것인가

작성: 2026-08-03 · 결정 필요: 6차 정기 회의

## 질문

동성이 만들고 있는 gait primitive 라벨을 학습에 어떻게 쓸 것인가. **UPDRS 점수만 학습**할 것인가, 아니면 **primitive까지 학습**할 것인가.

## 결론 먼저

**UPDRS 점수만 학습하는 것은 권하지 않습니다.** primitive를 함께 학습하는 multi-task를 권합니다. 근거는 세 가지입니다.

## 1. UPDRS 점수는 타깃으로서 품질이 낮습니다

| 근거 | 내용 |
| --- | --- |
| Williams 2023 (J Parkinsons Dis) | MDS-UPDRS 항목 3.4 채점의 **전문가 21명 간 ICC 0.53**. 더 심각한 건 **정상 대조군 영상의 53%가 0보다 큰 점수를 받았다**는 것 |
| Czech 2024 (Commun Med, n=82+50) | 12개월 종단 변화에서 **MDS-UPDRS 항목 점수는 유의한 변화를 잡지 못했고**(pronation-supination Cohen's d = 0.00), 같은 기간 디지털 측정은 잡았습니다(d = 0.45) |

즉 UPDRS 점수만 타깃으로 쓰면 **노이즈가 큰 라벨을 배우고, 종단 변화에 둔감한 출력을 내게 됩니다.** 우리가 만들려는 게 "시간에 따른 변화를 보는 임상 화면"이라면 이건 정면으로 어긋납니다.

## 2. 우리 primitive 축은 이미 외부 검증을 받았습니다

Ehsan 2026 (npj Parkinson's Disease, PPP n=74)이 finger tapping을 **hypokinesia / bradykinesia / sequence effect / hesitation-halts** 4축으로 분해했고, varimax PCA가 이 구조를 재현했으며, 이 표현이 기존 최고 성능보다 UPDRS 예측을 잘했습니다.

우리 `finger_tapping_primitive_v0.1`의 축(진폭 / 속도 / 감소 / 주저·정지)이 여기에 사실상 일치합니다. **우리가 만들고 있는 라벨이 근거 있는 중간 표현이라는 외부 확증**입니다.

## 3. 지금은 사슬이 끊겨 있습니다

```
라벨링툴(primitive 정의·사람 라벨) → 학습 데이터 → [끊김] → Hawk I 출력(점수+문장) → 임상 화면
```

Hawk I 코드에는 `primitive` 개념이 아예 없습니다. 모델이 점수만 내면 임상 화면도 점수만 보여줄 수밖에 없고, 조사에서 확인한 1순위 원칙("점수가 아니라 관찰을 앞에")을 지킬 수 없습니다.

**모델이 primitive를 출력하면 사슬이 이어지고, 사람 라벨과 모델 판정을 같은 축에서 비교할 수 있게 됩니다.**

## 권고 설계

### 타깃 구조

| 타깃 | 형태 | 손실 |
| --- | --- | --- |
| primitive observability | 3-class (observed / unobservable / uncertain) | CE |
| primitive severity | 0–3 서열, **observed일 때만** | ordinal (CORAL 등) |
| quality gate | 2-class (pass / hold) | CE |
| UPDRS anchor | 0–4 서열, **별도 헤드** | ordinal |

**핵심 제약**: severity를 합산해 UPDRS를 만들지 않습니다. 별도 헤드로 둡니다. 온톨로지가 이미 금지한 사항이고, 합산은 임상적으로 근거가 없습니다.

### 데이터 현실에 맞춘 단계

우리는 **라벨 두 종류가 비대칭**하게 있습니다.

- PD4T 원본 UPDRS 라벨: 많음 (전체 데이터셋)
- 우리 primitive 라벨: 적음 (동성이 만드는 중)

따라서:

1. **UPDRS 라벨 전체로 먼저 학습** (표현 학습 + C0 baseline 확보)
2. **primitive 라벨로 multi-task 미세조정** — primitive를 auxiliary task로 두고 손실 가중치를 조정
3. 두 단계 결과를 비교하면 **"구조화된 중간 표현이 점수 예측을 돕는가"**라는 검증 가능한 주장이 나옵니다

이 구조는 회의록의 C0~C3 설계와도 맞습니다. 다만 **C1/C3의 "reasoning"을 자유서술이 아니라 primitive로 정의하기를 권합니다.** 자유서술은 사람이 읽어야 채점되지만, primitive는 라벨과 직접 비교해 정확도·F1을 낼 수 있어 **논문에서 기여를 수치로 주장할 수 있습니다.**

### abstention을 학습에 포함

온톨로지가 이미 "quality gate hold면 severity를 null로 강제"하고 있습니다. 모델도 **"판정 불가"를 출력할 수 있어야** 합니다. 낮은 신뢰도일 때 점수를 억누르되 관찰은 유지하는 방식이 문헌이 지지하는 패턴이고, 우리 화면도 이미 그렇게 만들어 뒀습니다.

## 먼저 측정해야 할 것 (이게 안 되면 위 설계가 무의미합니다)

**primitive 라벨의 라벨러 간 신뢰도를 아직 모릅니다.** 파일럿에 2인 독립 리뷰가 이미 설계돼 있으니, 첫 배치가 끝나면 다음을 계산해야 합니다.

- primitive별 라벨러 간 일치도 (κ 또는 ICC)
- 이 값이 **UPDRS 점수의 ICC 0.53보다 높은가** — 높지 않으면 primitive를 타깃으로 쓸 근거가 약해집니다
- observability 판정의 일치도 (특히 `uncertain` 사용 빈도)

primitive가 UPDRS보다 일치도가 높다면, 그 자체가 논문에서 주장할 수 있는 결과입니다.

## 회의에서 정할 것

1. C1/C3의 reasoning을 **primitive 구조로 정의**할 것인가 (권고: 예)
2. gait 라벨링에 **UPDRS 앵커 점수를 함께 기록**할 것인가 — 현재 온톨로지는 `score_anchor`로 분리해 두었고, 두 타깃을 모두 쓰려면 필요합니다
3. 첫 배치 후 **라벨러 간 신뢰도 측정**을 누가 언제 할 것인가
