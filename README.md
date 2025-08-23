# 그룹화된 신경망 (Grouped Neural Network) for 청소년 건강 설문 데이터

해석 가능한 그룹화된 신경망을 통한 청소년 정신건강 위험도 예측 시스템

## 🎯 프로젝트 개요

이 프로젝트는 청소년 건강 설문 데이터를 분석하여 정신건강 위험도를 예측하는 해석 가능한 그룹화된 신경망을 구축합니다. 서로 다른 특성 그룹(사회경제적 요인, 신체활동, 식생활, 음주, 흡연 등)을 각각 처리하여 동적인 입력 크기를 지원하고, 그룹별 중요도 분석을 통해 해석성을 제공합니다.

## 🌟 주요 특징

- **🔗 그룹별 처리**: 서로 다른 특성 그룹을 독립적으로 처리하는 서브넷 구조
- **📏 동적 입력 크기**: 각 그룹마다 다른 수의 특성을 동적으로 처리
- **🧠 누락 처리**: 누락된 그룹에 대한 학습 가능한 null 임베딩
- **🔍 해석성**: Permutation Importance 및 SHAP을 통한 그룹별 중요도 분석
- **📊 시각화**: 훈련 과정 및 중요도 분석 결과 시각화
- **⚡ 효율성**: PyTorch 기반 효율적인 구현

## 📁 프로젝트 구조

```
med.exp7_25/
├── README.md                    # 프로젝트 가이드
├── models.ipynb                 # 전체 파이프라인 노트북
├── adolescent_health_data.csv   # 데이터 파일
├── analyze_columns.py           # 컬럼 분석 스크립트
│
├── src/                         # 소스 코드
│   ├── config/
│   │   └── groups.yaml          # 변수 그룹 정의
│   ├── data/
│   │   └── dataset.py           # 데이터 로딩 및 전처리
│   ├── model/
│   │   └── grouped_nn.py        # 그룹화된 신경망 모델
│   ├── train.py                 # 훈련 스크립트
│   └── interpret/
│       └── importance.py        # 해석성 분석
│
├── tests/
│   └── smoke_test.py            # 기본 기능 테스트
│
└── outputs/                     # 결과 저장 디렉토리
    ├── models/                  # 훈련된 모델
    ├── plots/                   # 시각화 결과
    └── results/                 # 분석 결과
```

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 필요한 라이브러리 설치
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn PyYAML

# 선택적: SHAP 설치 (해석성 분석용)
pip install shap
```

### 2. 데이터 준비

`adolescent_health_data.csv` 파일이 프로젝트 루트에 있는지 확인하세요.

### 3. Jupyter 노트북 실행 (권장)

```bash
jupyter notebook models.ipynb
```

노트북을 순서대로 실행하면 전체 파이프라인을 체험할 수 있습니다:
1. 데이터 탐색 및 그룹화
2. 모델 생성 및 구조 확인
3. 훈련 및 성능 모니터링
4. 해석성 분석
5. 결과 시각화

### 4. 명령행 인터페이스 사용

```bash
# 기본 훈련
python src/train.py --csv_path adolescent_health_data.csv --groups_yaml src/config/groups.yaml

# 커스텀 설정으로 훈련
python src/train.py \
    --csv_path adolescent_health_data.csv \
    --groups_yaml src/config/groups.yaml \
    --hidden 128 \
    --group_embedding_dim 64 \
    --lr 0.0005 \
    --batch 128 \
    --epochs 100 \
    --output_dir my_results
```

### 5. 기본 테스트 실행

```bash
# 모든 모듈이 정상 작동하는지 확인
python tests/smoke_test.py
```

## 📊 데이터 구조

### 변수 그룹

| 그룹 | 설명 | 변수 수 | 예시 변수 |
|------|------|---------|-----------|
| S | 사회경제적 요인 | 5 | S_SI, S_FAGE, S_CONT |
| PA | 신체활동 | 7 | PA_SWD_S, PA_VIG_D, PA_MSC |
| F | 식생활 | 7 | F_BR, F_FRUIT, F_FASTFOOD |
| AC | 음주 관련 | 27 | AC_LT, AC_FAGE, AC_DRUNK |
| TC | 흡연 관련 | 42 | TC_LT, TC_FAGE, TC_AMNT |
| I | 부상 관련 | 13 | I_SB_FR, I_SCH, I_HM_MOT |
| INT | 인터넷 사용 | 4 | INT_SPWD, INT_SPWK |
| O | 기타 건강행동 | 8 | O_BR_FQ, O_SLNT |
| HW | 손씻기 | 5 | HW_SPML_S, HW_EDU |
| E | 환경적 요인 | 22 | E_SES, E_FM_F_1, E_EDU_F |

### 타겟 변수

- **M_STR**: 스트레스 수준 (1-5)
- **이진 분류**: 4 이상 = 높은 스트레스(1), 미만 = 낮은 스트레스(0)

## 🧠 모델 아키텍처

### GroupedNN 구조

```
입력 그룹들 → 그룹별 서브넷 → 임베딩 결합 → 최종 MLP → 예측
     ↓              ↓              ↓           ↓        ↓
  [S: 5차원]    [MLP 64→32]    [연결: 320차원]  [128→64→1] [확률]
  [PA: 7차원]   [MLP 64→32]         ↓
  [F: 7차원]    [MLP 64→32]    [Dropout 0.1]
     ...           ...        [ReLU 활성화]
```

### 주요 구성요소

1. **그룹별 서브넷**: 각 그룹을 위한 독립적인 MLP
2. **Null 임베딩**: 누락된 그룹을 위한 학습 가능한 임베딩
3. **임베딩 결합**: 모든 그룹 임베딩을 연결
4. **최종 분류기**: 이진 분류를 위한 MLP

## 📈 성능 메트릭

- **AUROC**: ROC 곡선 아래 면적
- **AUPRC**: Precision-Recall 곡선 아래 면적
- **F1 Score**: 정밀도와 재현율의 조화평균
- **정확도, 정밀도, 재현율, 특이도**

## 🔍 해석성 분석

### Permutation Importance

각 그룹의 특성들을 무작위로 섞어서 성능 감소를 측정하여 중요도를 계산합니다.

```python
from src.interpret.importance import ImportanceAnalyzer

analyzer = ImportanceAnalyzer(model=trained_model)
results = analyzer.analyze_full_importance(val_loader)
```

### 결과 예시

```
그룹별 중요도 순위:
1. AC (음주 관련): 0.0245 ± 0.0031 (상대적: 3.1%)
2. TC (흡연 관련): 0.0198 ± 0.0028 (상대적: 2.5%)
3. M (정신건강): 0.0156 ± 0.0025 (상대적: 2.0%)
```

## 🎛️ 설정 옵션

### 모델 설정

```yaml
# src/config/groups.yaml 에서 그룹 정의 수정 가능
groups:
  - name: "S"
    description: "사회경제적 요인"
    features: ["S_SI", "S_FAGE", ...]
```

### 훈련 파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `--lr` | 0.001 | 학습률 |
| `--batch` | 32 | 배치 크기 |
| `--epochs` | 100 | 최대 에포크 |
| `--hidden` | 64 | 은닉층 차원 |
| `--group_embedding_dim` | 32 | 그룹 임베딩 차원 |
| `--dropout` | 0.1 | 드롭아웃 비율 |
| `--patience` | 15 | 조기 종료 인내심 |

## 📊 출력 파일

### 훈련 후 생성되는 파일들

```
outputs/
├── best_grouped_nn.pth          # 최고 성능 모델
├── grouped_nn_history.json      # 훈련 히스토리
├── grouped_nn_info.json         # 모델 정보
├── grouped_nn_training.png      # 훈련 과정 그래프
│
results/
├── importance_results.json      # 중요도 분석 결과
├── group_importance.png         # 중요도 막대 그래프
└── importance_summary.txt       # 요약 보고서
```

## 🔧 고급 사용법

### 1. 커스텀 그룹 정의

`src/config/groups.yaml`을 수정하여 새로운 변수 그룹을 정의할 수 있습니다.

### 2. 모델 저장/로드

```python
# 모델 저장
model.save_model('my_model.pth')

# 모델 로드
loaded_model = GroupedNN.load_model('my_model.pth', 'src/config/groups.yaml')
```

### 3. 예측

```python
# 확률 예측
probs = model.predict_proba(x_groups)

# 그룹별 기여도 계산
contributions = model.get_group_contributions(x_groups)
```

### 4. SHAP 분석 활성화

```python
analyzer = ImportanceAnalyzer(model=model, use_shap=True)
results = analyzer.analyze_full_importance(data_loader)
```

## 🧪 테스트

### Smoke Test 실행

```bash
python tests/smoke_test.py
```

이 테스트는 다음을 검증합니다:
- 합성 데이터로 전체 파이프라인 작동
- 모든 모듈의 정상 동작
- 2 에포크 훈련 가능성
- 모델 저장/로드 기능

## 🚨 문제 해결

### 일반적인 문제들

1. **메모리 부족**
   ```bash
   # 배치 크기 줄이기
   python src/train.py --batch 16
   ```

2. **CUDA 오류**
   ```bash
   # CPU 모드 강제 사용
   export CUDA_VISIBLE_DEVICES=""
   ```

3. **한글 폰트 문제**
   ```python
   # matplotlib에서 다른 폰트 사용
   plt.rcParams['font.family'] = 'DejaVu Sans'
   ```

4. **SHAP 설치 오류**
   ```bash
   # SHAP 없이 실행 (Permutation Importance만 사용)
   # 코드에서 use_shap=False로 설정
   ```

## 📚 참고 자료

### 논문 및 이론

- **Permutation Importance**: Breiman, L. (2001). Random forests.
- **SHAP**: Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions.
- **Group-wise Neural Networks**: 다양한 특성 그룹을 처리하는 아키텍처 연구

### 관련 라이브러리

- [PyTorch](https://pytorch.org/): 딥러닝 프레임워크
- [scikit-learn](https://scikit-learn.org/): 머신러닝 유틸리티
- [SHAP](https://github.com/slundberg/shap): 모델 해석성 도구
- [pandas](https://pandas.pydata.org/): 데이터 조작
- [matplotlib](https://matplotlib.org/): 시각화

## 🤝 기여하기

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 📧 연락처

프로젝트에 대한 질문이나 제안이 있으시면 Issues를 통해 연락해 주세요.

---

**🎉 이제 청소년 건강 데이터로 해석 가능한 머신러닝 모델을 구축할 준비가 되었습니다!**
