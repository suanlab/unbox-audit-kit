# 채점 기준 상세 — 예시 케이스 (Korean)

각 점수의 의미를 12개 예시로 보여드립니다. 본 작업의 모든 가정은 영문이므로 예시도 영문으로 제시합니다.

---

## 점수 5: 명백히 같은 가정 (단어 차이만)

### 예시 1 (transformer)
- **target**: recurrence is necessary for sequence modeling
- **extracted**: Recurrent neural networks are necessary for modeling sequential data.
- **score**: 5
- **이유**: "RNN = recurrent network", "sequential data ⊂ sequence modeling" — 명제 동일

### 예시 2 (vit)
- **target**: convolutional layers are necessary for vision
- **target_aliases**: vision models require convolutional layers
- **extracted**: Vision models require convolutional layers for effective image processing.
- **score**: 5
- **이유**: aliases와 거의 글자 단위 일치

---

## 점수 4: 같은 가정 (논리적 동치 / 명백한 패러프레이즈)

### 예시 3 (vit)
- **target**: convolutional layers are necessary for vision
- **target_aliases**: image understanding needs convolutional inductive biases
- **extracted**: Image understanding requires convolutional inductive biases.
- **score**: 4
- **이유**: 표현 다르나 핵심 명제 동일 (alias와 일치)

### 예시 4 (icl)
- **target**: fine-tuning is necessary for task adaptation
- **extracted**: Adapting models to downstream tasks requires updating model parameters.
- **score**: 4
- **이유**: "updating model parameters" ≈ "fine-tuning"

### 예시 5 (diffusion)
- **target**: adversarial training is necessary for high-quality generation
- **extracted**: GAN-style discriminators are required for producing realistic images.
- **score**: 4
- **이유**: GAN ≈ adversarial, realistic ≈ high-quality

---

## 점수 3: 부분 일치 / 모호

### 예시 6 (transformer)
- **target**: recurrence is necessary for sequence modeling
- **extracted**: Attention mechanisms are essential for handling long-range dependencies in sequences.
- **score**: 3
- **이유**: 시퀀스 모델링 영역은 같지만 "recurrence 필수성"은 명시 안 됨. 어텐션이 시퀀스에 필요하다는 별개 명제.

### 예시 7 (icl)
- **target**: fine-tuning is necessary for task adaptation
- **extracted**: Pre-training large models on diverse data is essential for downstream task generalization.
- **score**: 3
- **이유**: 사전학습 강조는 맞으나 fine-tuning 필수성에 대한 명제 아님

### 예시 8 (vit)
- **target**: convolutional layers are necessary for vision
- **extracted**: Hierarchical feature extraction is essential for image recognition.
- **score**: 3
- **이유**: 컨볼루션이 위계적 특징 추출의 한 방법이긴 하지만, 명제가 다름 (방법 vs 추상)

---

## 점수 2: 관련 영역이지만 다른 명제

### 예시 9 (transformer)
- **target**: recurrence is necessary for sequence modeling
- **extracted**: End-to-end learning is essential for sequence-to-sequence modeling.
- **score**: 2
- **이유**: 시퀀스 모델링이라는 공통 영역, 그러나 "recurrence" vs "end-to-end" — 다른 차원의 명제

### 예시 10 (diffusion)
- **target**: adversarial training is necessary for high-quality generation
- **extracted**: Large-scale datasets are required for training generative models.
- **score**: 2
- **이유**: 생성 모델링이라는 영역은 같으나 데이터 규모 vs adversarial 메소드 — 무관

---

## 점수 1: 명백히 다른 가정

### 예시 11 (diffusion)
- **target**: adversarial training is necessary for high-quality generation
- **extracted**: Word error rate is the primary evaluation metric for speech recognition.
- **score**: 1
- **이유**: 영역도 무관, 명제도 무관

### 예시 12 (vit)
- **target**: convolutional layers are necessary for vision
- **extracted**: Recurrent neural networks are essential for modeling sequential data.
- **score**: 1
- **이유**: vision vs sequence — 영역 자체가 다름

---

## 까다로운 케이스 가이드

| 상황 | 권장 점수 |
|------|----------|
| 추출이 정답의 **상위 일반화** | 2~3 (너무 모호) |
| 추출이 정답의 **하위 특수화** | 3~4 (구체화는 OK) |
| 추출이 정답을 **간접 함의** | 3 |
| 추출이 정답의 **반대 명제** | 1 (논리적 무관함) |
| 추출이 정답과 **단어 일부만 겹침** | 2 |

## 모호 케이스 → note에 기록

확신 점수가 어렵다면:
- `score`에 가운데값 (3) 입력
- `note`에 무엇이 모호한지 한 줄
- 예: `"sequence modeling vs sequence-to-sequence: 같은 영역인지 모호"`
