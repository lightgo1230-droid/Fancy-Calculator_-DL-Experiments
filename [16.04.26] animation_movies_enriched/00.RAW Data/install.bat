@echo off
echo ======================================================
echo  애니메이션 ML 프로젝트 패키지 설치
echo ======================================================

echo [1/4] 공통 패키지 설치...
pip install matplotlib tqdm

echo [2/4] 프로젝트 1 - HuggingFace Transformers 설치...
pip install transformers tokenizers sentencepiece

echo [3/4] 프로젝트 3 - PyTorch Geometric 설치...
pip install torch_geometric

echo [4/4] 네트워크 시각화 - networkx 설치...
pip install networkx

echo.
echo 설치 완료! 아래 순서로 실행하세요:
echo   python 01_nlp_genre_classification.py
echo   python 02_tabular_rating_prediction.py
echo   python 03_graph_network_analysis.py
pause
