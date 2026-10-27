# week02

# 혼공머 : Ch 3 회귀 알고리즘과 모델 규제

## 3.1 k-최근접 이웃 회귀

### *분류 vs 회귀

- 분류 (Classification)
    - 여러 종류를 구분(정해진 클래스 예측)
        
        ex) 도미(1)와 빙어(0) 구분
        
- 회귀 (Regression)
    - 두 변수의 상관관계를 이용해 임의의 숫자 예측
        
        ex) 농어의 무게(target) 예측
        
    
    → 이진 분류와 달리 타겟을 0,1의 값으로 준비하지 X
    
    → 데이터 중 하나가 타겟값이 되는 경우가 많다.
    

### 3.1.1 k-최근접 이웃 회귀

: 가까운 이웃 k개를 찾음 → 샘플 k개 타겟값의 평균을 예측값으로 삼음

### 3.1.2 데이터 준비

```python
# 농어 무게 예측 알고리즘

import numpy as np
import matplotlib.pyplot as plt

# 데이터 준비 : 농어 길이 및 무게

perch_length = np.array(
    [8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0,
     21.0, 21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5,
     22.5, 22.7, 23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5,
     27.3, 27.5, 27.5, 27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0,
     36.5, 36.0, 37.0, 37.0, 39.0, 39.0, 39.0, 40.0, 40.0, 40.0,
     40.0, 42.0, 43.0, 43.0, 43.5, 44.0]
     )

perch_weight = np.array(
    [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0,
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0,
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0,
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0,
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0,
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0,
     1000.0, 1000.0]
     ) 
     
# 농어 '길이 - 무게' 관계 그래프 그리기
plt.scatter(perch_length, perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

*무게(예측 대상) : 타겟 데이터

- 그래프를 통해 알 수 있는 점 : 길이가 증가함에 따라 무게가 증가함을 보여줌

```python
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42)
    
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)
```

- `np.reshape` : 배열의 형태 변경
    - 새로운 모양을 나타내는 튜플 : `(2, 3)` = 2*3 배열
    - `a = np.arange(6)`이면 `b = a.reshape((2,4))` 불가능
    - `-1`을 사용하여 특정 차원을 자동으로 계산 가능, 즉 배열의 길이를 유지하면서 가능한 형태로 맞춤
        - `a = np.arange(12)`, `b = a.reshape((3, -1))` → 3x4 array

### 3.1.3 결정 계수 (R^2)

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

# 1. 모델 생성
knr = KNeighborsRegressor()

# 2. 모델 학습
knr.fit(train_input, train_target)

# 3. 모델 평가
knr.score(test_input, test_target)

# 4. 실전 예측
test_prediction = knr.predict(test_input)

# 5. 테스트 데이터에 대한 평균 절댓값 오차 계산 (모델의 성능 판단 방법 중 하나)
mae = mean_absolute_error(test_target, test_prediction)
```

- 결정 계수 (R^2)
    
    = 1 - (타겟 - 예측)^2의 합 / (타겟 - 평균)^2의 합
    
    - 예측값과 샘플들의 평균값이 비슷 → R^2 = 0에 가까워짐 (bad)
    - 예측값과 타겟값이 비슷 → R^2 = 1에 가까워짐 (good)
    
    → 모델의 성능 판단
    

### 3.1.4 과대적합 vs 과소적합

```python
print(knr.score(train_input, train_target))
print(knr.score(test_input, test_target))
```

- 출력 결과
    - `0.9698823289099254`  (훈련 세트)
    - `0.992809406101064`  (테스트 세트)
    
    → 훈련 세트보다 테스트 세트의 점수가 더 높음
    
    ⇒ “회귀 모델이 훈련 세트를 적절히 학습하지 못했다!” (**과소적합 Underfitting**)
    
    ⇒ 반대) “회귀 모델이 훈련 세트에만 강하고 실전에서는 제 능력 발휘 못한다! ” (**과대적합 Overfitting**)
    

```python
# k-최근접 이웃 모델의 이웃 개수 줄이기 (k=5 -> k=3)
knr.n_neighbors = 3

# 모델 재훈련
knr.fit(train_input, train_target)

print(knr.score(train_input, train_target))
print(knr.score(test_input, test_target))
```

- 출력 결과
    - `0.9804899950518966`  (훈련 세트)
    - `0.9746459963987609`  (테스트 세트)
    
    → 과소적합 상태에서 k(이웃의 개수)를 줄여서 성능 개선
    
- k(이웃의 개수)에 따른 모델의 성능 변화
    - 늘리기 : 과소적합 발생 가능 (= 너무 단순한 모델)
    - 줄이기 : 과대적합 발생 가능 (= 너무 복잡한 모델)
    
    → 최적의 값 k는 문제마다 다르다.
    

---

## 3.2 선형 회귀

### 3.2.1 k-최근접 이웃의 한계

```python
import numpy as np
from sklearn.model_selection import train_test_split

# 데이터 준비
perch_length = np.array(
    [8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0,
     21.0, 21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5,
     22.5, 22.7, 23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5,
     27.3, 27.5, 27.5, 27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0,
     36.5, 36.0, 37.0, 37.0, 39.0, 39.0, 39.0, 40.0, 40.0, 40.0,
     40.0, 42.0, 43.0, 43.0, 43.5, 44.0]
     )
perch_weight = np.array(
    [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0,
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0,
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0,
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0,
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0,
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0,
     1000.0, 1000.0]
     )
         
# 훈련 세트와 테스트 세트 준비
train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42)

train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)
```

```python
from sklearn.neighbors import KNeighborsRegressor

# 1. k-최근접 이웃 회귀 모델 생성
knr = KNeighborsRegressor(n_neighbors=3)

# 2. 모델 학습
knr.fit(train_input, train_target)

# 3. 길이 50cm 농어의 무게 예측
print(knr.predict([[50]]))
```

- 출력 결과
    - `[1033.33333333]`  → 1.033kg
    
    → 실제값 1.5kg과 다른 예측 결과 발생!
    
    → 원인을 알고 문제를 해결하자 ⬇️
    

```python
import matplotlib.pyplot as plt

# 50cm 농어의 이웃 구하기
distances, indexes = knr.kneighbors([[50]])

# 훈련 세트의 산점도 그리기
plt.scatter(train_input, train_target)

# 훈련 세트 중에서 이웃 샘플만 다시 그리기
plt.scatter(train_input[indexes], train_target[indexes], marker='D')

# 50cm 농어 데이터 표시
plt.scatter(50, 1033, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

- 출력 결과

    → 잘못 예측함

- 원인
    - k-최근접 이웃 모델이 가장 가까운 위치에 있는 샘플 데이터를 참고해서 예측값이 제한됨
    - 훈련 세트 내 이웃한 샘플로 예측하므로 범위 밖에 있는 값을 예측하기 어려움
    
    ⇒ 문제 해결 방법 : **선형 회귀** 알고리즘
    

### 3.2.2 선형 회귀 (Linear Regression)

- 1차원 데이터(특성 1개) → 직선의 방정식(일차식)으로 나타냄
    - y = ax+b
        
        → a(기울기)와 b(절편)을 알아내자.
        

```python
from sklearn.linear_model import LinearRegression

# 1. 선형 회귀 모델 생성
lr = LinearRegression()

# 2. 모델 학습
lr.fit(train_input, train_target)

# 3. 훈련 세트 산점도 및 직선 그리기
plt.scatter(train_input, train_target)

# 1차 방정식 그래프
plt.plot([15, 50], [15*lr.coef_+lr.intercept_, 50*lr.coef_+lr.intercept_])
print(lr.coef_, lr.intercept_)

# 50cm 농어 데이터 표시
plt.scatter(50, 1241.8, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```

- 출력 결과
    - `[39.01714496] -709.0186449535477`
        
        → `lr.coef_` : 기울기(가중치) / `lr.intercept_`  : 절편(편향)
        
    - 일차함수 그래프의 절편이 음수값을 가짐 (실제 존재하지 않음)
    
    
    

```python
# 50cm 농어에 대한 예측
print(lr.predict([[50]]))
```

- 출력 결과
    - `[1241.83860323]`
    
    → k-최근접 이웃 회귀 모델보다 추세를 잘 따라감
    
    → 그러나 과소적합 발생 가능!
    

```python
print(lr.score(train_input, train_target))
print(lr.score(test_input, test_target))
```

- 출력 결과
    - `0.939846333997604`  (훈련 세트)
    - `0.8247503123313558`  (테스트 세트)
    
    → 모델의 성능을 더 높일 수 있는 방법을 알아보자 ⬇️
    

### 3.2.3 다항 회귀 (Polynomial Regression)

- 모델이 다항식을 학습하여 예측하는 선형 회귀 (선형 회귀에서 특성 추가)
    
    예) 이차함수 꼴로 데이터 추세를 나타내보자.
    
    - y = ax^2 + bx + c
        
        → a, b, c를 알아내자.
        

```python
# 이차식을 사용하기 위해 2차원 배열 생성
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))

# 1. 다항 회귀 모델 생성
lr = LinearRegression()

# 2. 모델 학습
lr.fit(train_poly, train_target)

# 3. 50cm 농어 무게 예측
print(lr.predict([[50**2, 50]]))
print(lr.coef_, lr.intercept_)
```

- 출력 결과
    - `[1573.98423528]`  (농어 무게) : 선형 회귀보다 더 예측 잘함
    - `[1.01433211 -21.55792498] 116.0502107827827`  → 절편 > 0
    
    → 선형 회귀가 갖는 문제점(1. 모델 성능 2. 음수 절편값)을 모두 해결 가능
    

```python
# 그래프

point = np.arange(15, 50)

# 훈련 세트의 산점도 그리기
plt.scatter(train_input, train_target)

# 2차 방정식 그래프 그리기
plt.plot(point, 1.01*point**2 - 21.6*point + 116.05)

# 50cm 농어 데이터 표시
plt.scatter([50], [1574], marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```    

---

## 3.3 특성 공학과 규제

### 3.3.1 다중 회귀 (Multiple Regression)

- 특성이 여러 개인 회귀
    - 다중 회귀 : 2개 이상의 서로 다른 독립 변수로 하나의 종속 변수 예측하는 회귀 분석
    - 다항 회귀 : 하나의 독립 변수 차수를 높여 곡선 형태의 비선형 관계 표현하는 회귀 분석
- 특성 공학 : 특성을 추가/변경/조합

### 3.3.2 데이터 준비

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('https://bit.ly/perch_csv_data')
perch_full = df.to_numpy()

perch_weight = np.array(
    [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0,
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0,
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0,
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0,
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0,
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0,
     1000.0, 1000.0]
     )
     
train_input, test_input, train_target, test_target = train_test_split(
perch_full, perch_weight, random_state=42)
```

### 3.3.3 사이킷런의 변환기

```python
from sklearn.preprocessing import PolynomialFeatures

# 1. 다중 회귀 모델 생성
poly = PolynomialFeatures() # 매개변수 degree(기본값 : 2)

# 2. 모델 학습 (입력 : 특성이 2개인 샘플) 
poly.fit([[2, 3]])
print(poly.transform([[2, 3]]))
```

- 출력 결과
    - `[[1. 2. 3. 4. 6. 9.]]`
        - `1` : 절편(bias)
        - `2 , 3` : 원래 특성
        - `4` : 2^2
        - `6` : 2*3
        - `9` : 3^2
    
    → degree=2 (2제곱까지)여서 위와 같이 변환됨
    
    → 특성 하나만 제곱하는 다항회귀와 달리, 특성 간 상호작용까지 반영 (ex. 특성끼리 곱함)
    
- `PolynomialFeatures` : 변환기(Transformer)
    - 기존 특성들을 조합해서 새 특성 만듦
- `LinearRegression`, `knn` : 추정기(Estimator)
    - 특성 변환 이외 모델링
    - ex. `fit`, `predict`, `score`

```python
poly = PolynomialFeatures(include_bias=False)
poly.fit([[2, 3]])
print(poly.transform([[2, 3]]))
```

- 출력 결과
    - `[[2. 3. 4. 6. 9.]]`
        
        → 1(절편 bias)이 사라짐
        
        → `include_bias=False` : 선형 회귀에서 절편을 따로 계산하기 때문에 굳이 필요 없음
        

```python
poly = PolynomialFeatures(include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
print(train_poly.shape)  
```

- 출력 결과
    - `(42, 9)` : 샘플 42개 / 특성 9개
    
    → 변환 후) 원래 특성 3 + 제곱 항 3 + 서로 곱한 항 3 = 9
    

```python
poly.get_feature_names_out()
```

- 출력 결과
    - `array(['x0', 'x1', 'x2', 'x0^2', 'x0 x1', 'x0 x2', 'x1^2', 'x1 x2', 'x2^2'])`
        
        → 9개 열(특성)의 이름을 알려줌
        

```python
test_poly = poly.transform(test_input)
```

- 훈련 세트 → `fit` / 테스트 세트 → `transform`
    - 훈련 세트와 똑같은 조합으로 변환해야 모델이 같은 형태의 입력을 받을 수 있음!

### 3.3.4 다중 회귀 모델 훈련하기

```python
from sklearn.linear_model import LinearRegression

# 1. 다중 회귀 모델 학습 (degree=2, 특성 9개)
lr = LinearRegression()
lr.fit(train_poly, train_target)

# 2. PolynomialFeatures 재설정 (degree=5)
# 목적 : 특성 더 많이 만들기
poly = PolynomialFeatures(degree=5, include_bias=False)
poly.fit(train_input)

# 3. 훈련 세트 및 테스트 세트를 같은 규칙으로 변환
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)

# 4. 늘어난 특성들을 이용해 다중 회귀 모델 재학습
# 같은 객체 재사용 -> 새 데이터로 다시 학습
lr.fit(train_poly, train_target)
```

- 더 많은 특성을 만들어서 다시 다중 회귀 모델을 학습시킬 수 있다.
- 훈련 세트의 개수(42개) < 특성 개수(55개) : 과대적합 발생
    - 극도로 과대적합한 모델도 이를 해결할 수 있는 방법이 존재 ⬇️

### 3.3.5 규제 (Regularization)

- 모델이 훈련 데이터에 과하게 맞춰지지 않도록 계수(가중치 = 기울기) 크기를 일부러 억제하는 방법
- 더 일반화된 모델을 생성하는 방법

- 규제 이전에 표준화 필요
    - 모든 특성을 같은 기준으로 맞춰야 정상적으로 규제 가능

```python
# 표준화

from sklearn.preprocessing import StandardScaler

# 1. 표준화 객체 생성
ss = StandardScaler()

# 2. 훈련 세트 기준 각 특성의 평균/표준편차 계산
ss.fit(train_poly)

# 3. 표준화 : (값 - 평균) / 표준편차
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)
```

- 계수가 크다 = 특성 하나에 모델이 과하게 민감하게 반응한다.
    
    → 계수가 커지는 것 자체에 제한을 줘서 모델을 단순하게 만들자!
    
    → 그러기 위해 규제 강도 **alpha** 사용
    
    → 특성은 그대로 but 각 특성이 결과에 미치는 영향력만 감소
    

### 1) 릿지(Ridge) 회귀

- 계수를 제곱한 값을 제한
    
    → 계수를 전체적으로 작게 만듦
    

```python
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

# 1. 릿지 모델 학습 (기본 alpha=1)
ridge = Ridge()
ridge.fit(train_scaled, train_target)

# 2. alpha별 점수를 저장할 리스트
train_score = []
test_score = []

# 3. alpha를 바꿔가며 반복 학습 -> 적절한 alpha 찾기
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100] # 후보값이 10배씩 커짐
for alpha in alpha_list:
    # alpha를 바꿔 릿지 모델 생성
    ridge = Ridge(alpha=alpha)
    
    # 릿지 모델 학습
    ridge.fit(train_scaled, train_target)
    
    # 훈련 점수와 테스트 점수 저장
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))

# 4. alpha에 따른 점수 변화 그래프 그리기  
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()
```    

```python
# 테스트 점수가 가장 높고, 훈련 점수와 차이가 가장 작은 alpha 선택
ridge = Ridge(alpha=0.1)
ridge.fit(train_scaled, train_target)
```

- 가장 적절한 alpha 값 : 그래프에서 두 선이 가장 가깝고, 테스트 점수가 높은 지점
    - 이처럼 alpha는 학습하는 값이 아닌 임의로 정해주는 값이다.

### 2) 라쏘(Lasso) 회귀

- 계수의 절댓값을 제한
    
    → 일부 계수를 아예 0으로 만들 수 있음
    

```python
from sklearn.linear_model import Lasso

# 1. 라쏘 모델 학습 (기본 alpha=1)
lasso = Lasso()
lasso.fit(train_scaled, train_target)

# 2. alpha별 점수를 저장할 리스트
train_score = []
test_score = []

# 3. alpha를 바꿔가며 반복 학습 -> 적절한 alpha 찾기
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    lasso = Lasso(alpha=alpha, max_iter=10000)   # 반복 횟수를 늘려서 생성
    lasso.fit(train_scaled, train_target)
    train_score.append(lasso.score(train_scaled, train_target))
    test_score.append(lasso.score(test_scaled, test_target))
 
# 4. alpha에 따른 점수 변화 그래프 그리기  
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()
```

```python
# 그래프에서 두 선이 가장 가까운 alpha 선택
lasso = Lasso(alpha=10)
lasso.fit(train_scaled, train_target)
```

- 릿지 모델과 라쏘 모델의 alpha 값이 다르므로 모델마다 최적값을 따로 찾아봐야 한다.

- 릿지와의 차이점
    - `max_iter=10000`  : 계수를 조금씩 고쳐 가며 반복 계산으로 찾을 때, 최대 반복 횟수
    - 계수가 정확히 0인 특성의 개수 세기
        
        ```python
        print(np.sum(lasso.coef_ == 0))
        ```
        
        → 전체 특성 중 모델이 사용하지 않는(중요하지 않은) 특성의 개수