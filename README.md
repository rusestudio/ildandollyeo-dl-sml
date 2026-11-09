# Outline
현재 브랜치(ts_token)는 토큰 분할까지의 데이터 파이프라인을 다룬다.

# 
- From scratch
    - fineweb-edu sample 10BT만 사용
    - 토크나이저와 모델은 직접 설계, 학습 - 사전학습된 외부 모델과 데이터 사용 금지

# Dataset
- Fineweb-edu sample10BT
    - 정제, 중복 제거, 교육성 필터링 완료된 데이터
    - 토크나이저 학습, 토큰화와 샤딩, 데이터 로딩
    - 선택 사항으로 품질 필터(QC) 진행


