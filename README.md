# AI_Championship
고신대복음병원_보행이상환자사전체크 AI개발_에이젠글로벌
# 개발 배경
* 고령층의 낙상 위험  
 65세 이상 고령인구의 20%는 낙상을 경험한 적이 있으며, 고령층 안전사고의 47.4%를 차지할 정도로 일상생활에서 빈번하게 일어난다. 또한, 입원환자의 안전사고의 절반을 차지하는 것도 낙상으로 발생 비율이 높다.  
 낙상으로 인한 피해는 골절, 타박상, 열상 등의 물리적 손상 뿐만 아니라 심리적으로 위축감과 의존성을 증가시키는 정신적 피해를 유발한다. 심할 경우 합병증으로 조기사망에 이르게 할 수 있으며, 실제 노인 사망원인의 2위로 나타났다.  
 따라서 낙상 사고를 사전에 예방하기 위한 노력이 필요하다.  
 
* 낙상의 위험요인  
 낙상의 원인으로는 지면 상태(미끄러움, 급한 경사)가 영향을 주거나 장애물에 대한 부주의로 발생할 수도 있다. 하지만 어지러움과 같은 건강 상의 문제로 발생하는 경우도 약 18%로 높게 나타난다. 어지러움 혹은 평형기능의 장애를 가져오는 질병이 원인이 되기도 한다. 이와 같이 특정 질병들이 보행 이상 및 불균형을 유발하고, 낙상 위험으로 이어지기도 한다.
 이에 따라 보행 이상 관련 질병을 여부에 따른 보행 데이터를 학습함으로써, 보행 데이터를 기반으로 보행 이상 관련 질병의 보유 유무 혹은 그 가능성을 예측할 수 있다.

# 개발 모델 설명
 환자 데이터와 대조군 데이터에서 공통된 칼럼만 사용하여 "보행 이상 관련 질병 보유 여부"를 예측하는 모델을 개발하였다.  
 분류 모형이므로 예측 성능을 평가하는 여러 가지 지표가 있지만, 그 중 정밀도(precision)와 재현율(recall)의 조화평균인 f1-score를 스코어보드의 평가지표로 선정하였다. 실제 1을 얼마나 잘 예측하는지에 대한 재현율과 1로 예측한 것 중 얼마나 맞았나에 대한 정밀도는 trade-off의 관계를 가지므로, 둘의 조화평균인 f1-score를 평가지표로 삼았다.  
 모델링은 자사의 머신러닝 자동화 엔진 'ABACUS(아바커스)'를 활용하였다. 데이터 전처리, 변수 선택, 모델 성능 검증 방법, 알고리즘을 클릭을 통해 선택하고 학습시키면, 알고리즘별 훈련/검증/테스트 셋에 대한 분류 성능을 여러가지 지표에 대해 보여주고, 중요 변수에 대한 정보를 제공한다. 최종적으로 성능이 가장 좋게 나온 알고리즘을 선택하면 개별 샘플에 대한 예측과 특정 데이터 셋에 대한 예측 결과를 얻을 수 있다.
 

# 데이터 전처리 및 모델링

## 코드 설명
* SplitData.ipynb  
스코어보드 정답파일 및 테스트데이터 생성을 위한 train-test split code
보행 관련 질병 유무(Diagnosis_YN)에 대한 칼럼 추가 (진단명이 있으면 1, 없으면 0), 공통된 컬럼만 사용
* 
