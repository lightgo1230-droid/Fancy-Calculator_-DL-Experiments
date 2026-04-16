# 애니메이션 영화 ML 프로젝트 - 연속 작업 컨텍스트
작성일: 2026-04-16

## 데이터셋
- 경로: C:\Users\USER\OneDrive\Desktop\animation_movies_enriched_1878_2029.csv
- 규모: 25,390개 영화 / 44 컬럼 / 1878~2029년

## 환경
- Python: C:\Users\USER\anaconda3\python.exe
- PyTorch: 2.10.0+cpu
- transformers: 5.5.4
- torch_geometric: 2.7.0

## 파일 구성
| 파일 | 설명 |
|---|---|
| 01_nlp_genre_classification.py | DistilBERT 장르 분류 + 영화 추천 (전체 데이터) |
| 02_tabular_rating_prediction.py | Entity Embedding MLP 평점 예측 (전체 데이터) |
| 03_graph_network_analysis.py | HeteroGNN 노드분류 + 링크예측 (전체 데이터) |
| eval_01_nlp_fast.py | NLP 정확도 측정 (8,000샘플, 빠른 버전) |
| eval_02_tabular.py | 정형 데이터 정확도 측정 |
| eval_03_graph.py | 그래프 정확도 측정 |
| create_report.py | Word 보고서 생성 |

## 측정된 정확도 결과
### 프로젝트 1 - NLP (DistilBERT)
- F1-micro:       69.41%
- F1-weighted:    57.38%
- Jaccard:        67.43%
- Exact Match:    43.69%
- Hamming Loss:   0.1218

### 프로젝트 2 - 정형 데이터 (Entity Embedding MLP)
- [회귀] MAE:     1.0393점
- [회귀] RMSE:    1.5096점
- [회귀] R²:      11.14%
- [분류] Accuracy: 71.56%
- [분류] ROC-AUC:  75.34%

### 프로젝트 3 - 그래프 (HeteroGNN)
- Node Classification Accuracy:  78.80%
- Node Classification F1-macro:  82.31%
- Link Prediction Accuracy:      78.12%
- Link Prediction ROC-AUC:       86.95%

## 다음 작업 아이디어
- 세 모델 앙상블 (NLP 임베딩 + 정형 피처 + GNN 임베딩)
- GPU 환경에서 전체 데이터 재학습
- 클래스 불균형 보정 (pos_weight, Focal Loss)
- 시계열 트렌드 분석 (Temporal GNN)
- 실시간 영화 추천 API 서버 구축
