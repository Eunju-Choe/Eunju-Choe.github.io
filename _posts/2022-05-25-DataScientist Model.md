---
title: 데이터 과학자 전직 여부 예측 - 분류 모델링
author: Eunju Choe
category: Project
tags: [Analytics, Python, HR, Machine Learning, Classification]
img: ":20220525.png"
date: 2022-05-25 23:31:00 +0900
---
<p align="center"><img src="https://eunju-choe.github.io/assets/img/posts/20220525.png" width='50%'></p>

#### 데이터 과학자의 전직 여부 예측 모델링 과정

이전 게시글에서는 두 차례에 걸쳐 데이터 과학자의 전직 여부를 예측하기 전 EDA와 전처리 과정을 진행하였습니다. 오늘은 전처리한 데이터를 바탕으로 예측 모델링을 실시해보도록 하겠습니다. EDA와 전처리 과정이 궁금하신 분들은 이전 게시글을 참고해주세요.
- [데이터 과학자 전직 여부 예측 - 시각화와 전처리 (1)][postlink1]
- [데이터 과학자 전직 여부 예측 - 시각화와 전처리 (2)][postlink2]

[postlink1]: https://eunju-choe.github.io/posts/2022-05-20-DataScientist%20EDA%201
[postlink2]: https://eunju-choe.github.io/posts/2022-05-22-DataScientist%20EDA%202

# 1. 데이터 불러오기 & train test split

먼저 전처리한 데이터를 불러와서 train 데이터와 test 데이터를 분리하였습니다.


```python
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('aug_train_processed.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city_development_index</th>
      <th>relevent_experience</th>
      <th>education_level</th>
      <th>experience</th>
      <th>company_size</th>
      <th>last_new_job</th>
      <th>training_hours</th>
      <th>target</th>
      <th>gender_M</th>
      <th>gender_F</th>
      <th>Fulltime</th>
      <th>Parttime</th>
      <th>is_STEM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.920</td>
      <td>1</td>
      <td>2.0</td>
      <td>21.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>3.610918</td>
      <td>1.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.776</td>
      <td>0</td>
      <td>2.0</td>
      <td>15.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>3.871201</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.624</td>
      <td>0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>4.430817</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.789</td>
      <td>0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.970292</td>
      <td>1.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.767</td>
      <td>1</td>
      <td>3.0</td>
      <td>21.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>2.197225</td>
      <td>0.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.model_selection import train_test_split
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)
```

# 2. 데이터 스케일링

변수가 각자 다른 단위를 가지고 있거나 편차가 심할 때에는 정규화나 표준화 과정이 필요합니다. 표준화와 정규화는 값의 스케일을 일정한 수준으로 변환시켜주는 과정으로 특정 머신러닝 모델의 학습 효율을 증가시키기 위해 자주 사용됩니다. 그럼 표준화와 정규화에 대해 각각 자세히 알아보도록 하겠습니다.

표준화는 평균을 기준으로 각 관측치의 값이 얼마나 떨어져있는지를 나타낼 때 사용합니다. zero-mean으로부터 얼마나 떨어져있는지를 나타내기 때문에 Z-score라고 표현합니다. 표준화는 서로 다른 변수 간 값의 크기를 직관적으로 비교할 수 있다는 장점이 있습니다.

정규화는 데이터의 범위를 0~1 사이로 변환하여 데이터의 분포를 조정하는 방법입니다. 전체 데이터에서 값의 위치를 파악하기 유용합니다. 정규화는 가장 큰 값이 1, 가장 작은 값이 0으로 직관적으로 표현되는 것이 장점이지만, 특정값이 평균으로부터 얼마나 떨어져있는지 알기 어렵다는 단점이 있습니다.

하지만 표준화와 정규화는 이상치에 민감하다는 단점이 존재합니다. 이를 보완한 스케일링 기법인 Robust Scaler도 자주 사용됩니다. Robust Scaler는 데이터의 중앙값(Q2)을 0으로 잡고, IQR(Q3-Q1)이 1이 되도록하는 스케일링 기법입니다. 이상치의 영향력을 최소화하여 일반적으로 표준화와 정규화보다 성능이 우수한 것으로 알려져 있습니다. 이번 실습에서는 Robust Scaler를 한 번 사용해보도록 하겠습니다.


```python
from sklearn.preprocessing import RobustScaler

sc = RobustScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)
```

스케일링을 할 때에는 train 데이터에서 fit_transform을 적용하고 test 데이터에서는 transfrom을 적용해야한다는 점입니다. test 데이터에서도 fit_transform을 적용하는 경우 train과 test에 서로 다른 스케일러가 적용되어 하나의 값이 train과 test에서 각각 다른 값을 가질 수 있기 때문입니다.

# 3. 오버 샘플링

해당 데이터의 경우에는 범주 불균형이 존재하여, 오버 샘플링이나 언더 샘플링을 통해 범주의 비율을 맞춰주는 과정이 필요했습니다.

언더 샘플링은 소수 범주의 크기 만큼 다수 범주의 데이터를 감소시키고, 오버 샘플링은 다수 범주의 크기만큼 소수 범주의 데이터를 증폭시키는 것입니다. 언더샘플링은 데이터가 유실된다는 단점이 있고, 오버샘플링은 데이터가 중복된다는 단점이 있습니다. 이번 실습에서는 ROS를 사용해보려고 합니다.

```python
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler()
X_train_r, y_train_r = ros.fit_resample(X_train_sc, y_train)
```

오버 샘플링을 적용하는 과정에서 주의해야할 점은 train 데이터에만 샘플링을 적용해야한다는 것입니다. test 데이터에서도 샘플링을 진행하거나, 샘플링 후에 train_test_split을 진행하는 경우에는 성과 평가 결과를 신뢰할 수 없기 때문입니다.

# 4. 머신러닝 모델링과 성과평가

머신러닝 모델링 과정에 Decision Tree, Logistic Regression을 사용해보려고합니다. GridSearchCV를 통해 최적의 하이퍼 파라미터를 찾고, 이를 통해 test 데이터를 예측하는 방식으로 진행할 것입니다.

성과 평가 지표는 정확도와 더불어 범주 불균형에서 주로 사용하는 f1-score를 함께 이용하여 비교해보도록 하겠습니다.

## 4.1. Decision Tree


```python
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
              'ccp_alpha': [0.1, 0.01, 0.001],
              'max_depth' : [5, 6, 7, 8, 9],
              'criterion' :['gini', 'entropy']
             }
tree = DecisionTreeClassifier(random_state=1)
clf = GridSearchCV(estimator=tree,
                   param_grid=param_grid,
                   cv=5, scoring='f1')

clf.fit(X_train_r, y_train_r)
pred = clf.predict(X_test_sc)

print(f'Best Parameters : {clf.best_params_}')
print('\n########################################\n')
print(f'Accuracy : {accuracy_score(y_test, pred):.3f}')
print(f'F1 Score : {f1_score(y_test, pred):.3f}')
```

    Best Parameters : {'ccp_alpha': 0.001, 'criterion': 'entropy', 'max_depth': 6, 'max_features': 'auto'}
    
    ########################################
    
    Accuracy : 0.722
    F1 Score : 0.510


## 4.2. Logistic Regression


```python
from sklearn.linear_model import LogisticRegression

param_grid = {"C":[0.01, 0.1, 1, 5, 10],
              "penalty":["l1","l2"]
             }
logit = LogisticRegression(random_state=1)
clf = GridSearchCV(estimator=logit,
                   param_grid=param_grid,
                   cv=5, scoring='f1')

clf.fit(X_train_r, y_train_r)
pred = clf.predict(X_test_sc)

print(f'Best Parameters : {clf.best_params_}')
print('\n########################################\n')
print(f'Accuracy : {accuracy_score(y_test, pred):.3f}')
print(f'F1 Score : {f1_score(y_test, pred):.3f}')
```

    Best Parameters : {'C': 5, 'penalty': 'l2'}
    
    ########################################
    
    Accuracy : 0.715
    F1 Score : 0.527


모델별로 성과를 비교해보면 아래 표와 같습니다.

지표|Tree|Logistic
:--:|--|--
정확도|0.722|0.715
F1|0.510|0.527

두 모델은 큰 차이 없이 accuracy 0.72와 f1-score 0.51 정도의 성능이 나오는 것을 확인할 수 있었습니다. 이번 프로젝트는 이러한 성과에 만족하기로 하고 마무리하려고 합니다.

이번 프로젝트에서는 처음으로 HR 데이터를 다뤄봤으며 데이터 과학자를 주제로 하여 재미있게 프로젝트를 진행할 수 있었습니다. 다음 프로젝트에서는 이번에 배운 것들을 바탕으로 조금 더 발전한 모습으로 돌아오도록 하겠습니다.

오늘도 긴 글 읽어주셔서 감사합니다. :)
