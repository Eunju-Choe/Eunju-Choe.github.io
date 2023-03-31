---
title: 동아리 자체 공모전 기획 및 운영
author: Eunju Choe
category: Python
tags: [Python, IBA, Machie Learning]
img: ":20230329.png"
date: 2023-03-28 22:50:00 +0900
---
![png](https://eunju-choe.github.io/assets/img/posts/20230329.png){: width='50%'}{: .center}

2022년 5월 동아리에서는 신규 기수를 대상으로 [분류 모델 경진대회](https://cafe.naver.com/pnuiba/115)를 개최하였습니다. 동아리 회장으로서 대회를 기획하고 운영하였으며, 오늘은 대회 운영에 사용했던 코드를 정리하려고 합니다.\
\
DACON에서 진행하는 대회와 동일한 방식으로 경진대회를 진행하였으며, 이를 위해 데이터 셋을 만들고 성과를 평가하기 위해 아래와 같이 코드를 작성하였습니다.

# 1. train/test 데이터 셋 만들기

대회에서 사용한 데이터는 UCI의 [adult](https://archive-beta.ics.uci.edu/dataset/2/adult) 데이터로, 인구 데이터를 바탕으로 소득이 5만 달러 이하인지 초과인지를 예측하는 데이터입니다. 먼저 UCI에서 주어진 데이터를 대회에서 사용할 train과 test 데이터로 만들었습니다.


```python
import pandas as pd
import urllib

# 데이터 불러오기
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
headers = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
        'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'target']

df = pd.read_csv(urllib.request.urlopen(url), header=None, names=headers)
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
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>39</td>
      <td>State-gov</td>
      <td>77516</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Never-married</td>
      <td>Adm-clerical</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>2174</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>1</th>
      <td>50</td>
      <td>Self-emp-not-inc</td>
      <td>83311</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Exec-managerial</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>13</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>215646</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Divorced</td>
      <td>Handlers-cleaners</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>3</th>
      <td>53</td>
      <td>Private</td>
      <td>234721</td>
      <td>11th</td>
      <td>7</td>
      <td>Married-civ-spouse</td>
      <td>Handlers-cleaners</td>
      <td>Husband</td>
      <td>Black</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&lt;=50K</td>
    </tr>
    <tr>
      <th>4</th>
      <td>28</td>
      <td>Private</td>
      <td>338409</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>Cuba</td>
      <td>&lt;=50K</td>
    </tr>
  </tbody>
</table>
</div>



대회 과정에서 원본 데이터를 찾아 target을 확인하지 못하도록, 데이터를 섞고 index를 재설정하였습니다. 그리고 문자형으로 되어있는 target을 편의상 숫자형으로 변환하였습니다.


```python
# 데이터를 랜덤하게 정렬
df = df.sample(frac=1, random_state=1)
# index 재설정
df = df.reset_index(drop=True)

# target을 숫자형으로 변환
df['target'] = df['target'].apply(lambda x : 0 if x==' <=50K' else 1)

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
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>62</td>
      <td>Self-emp-not-inc</td>
      <td>26911</td>
      <td>7th-8th</td>
      <td>4</td>
      <td>Widowed</td>
      <td>Other-service</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>66</td>
      <td>United-States</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>18</td>
      <td>Private</td>
      <td>208103</td>
      <td>11th</td>
      <td>7</td>
      <td>Never-married</td>
      <td>Other-service</td>
      <td>Other-relative</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>25</td>
      <td>United-States</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>25</td>
      <td>Private</td>
      <td>102476</td>
      <td>Bachelors</td>
      <td>13</td>
      <td>Never-married</td>
      <td>Farming-fishing</td>
      <td>Own-child</td>
      <td>White</td>
      <td>Male</td>
      <td>27828</td>
      <td>0</td>
      <td>50</td>
      <td>United-States</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33</td>
      <td>Private</td>
      <td>511517</td>
      <td>HS-grad</td>
      <td>9</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Husband</td>
      <td>White</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>36</td>
      <td>Private</td>
      <td>292570</td>
      <td>11th</td>
      <td>7</td>
      <td>Never-married</td>
      <td>Machine-op-inspct</td>
      <td>Unmarried</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



다음으로 train_test_split을 활용하여 train 데이터셋과 test 데이터셋을 만들었습니다. train 데이터는 X_train과 y_train을 merge 하여 train data로 만들었습니다. X_test는 test 데이터로 제공하고, y_test는 answer로 저장하여 운영진인 저만 가지고 있었습니다.


```python
from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=.25, random_state=1)

train_data = X_train.merge(y_train, left_index=True, right_index=True)
train_data.to_csv('train.csv')
X_test.to_csv('test.csv')
y_test.to_csv('answer.csv')
```

대회가 진행될 때 참가자는 중간 성과를 확인할 수 있었습니다. 이때, test 데이터 전체를 사용한 성과를 알려주지 않고 전체의 50%를 사용한 성과를 안내하였습니다. 50%로 성과를 안내하는 이유는 테스트 데이터에 과적합이 되지 않도록 하기 위함입니다. 이에 필요한 답안지를 y_test의 50%를 랜덤추출하여 만들었습니다.


```python
answer_50 = y_test.sample(frac=.5, random_state=1)
answer_50.to_csv('answer_50.csv')
```

# 2. 성과 평가를 위한 함수 만들기

데이터셋을 구성한 후, 성과 평가를 위한 함수를 작성하였습니다. 대회 진행 중에 사용할 중간 평가와 최종 평가를 하나의 함수로 구현하였습니다.\
\
대회 참가자에게는 몇 가지 규칙을 안내하였습니다.
- 컬럼명은 반드시 pred로 설정
- index를 포함하여 저장
- test 데이터 삭제 금지

이러한 규칙이 지켜졌는지 확인하기 위해 먼저 데이터에 문제가 있는지 확인하였으며, 오류가 없는 데이터를 대상으로 성과를 출력하는 코드를 작성하였습니다.

대회에 사용한 원본 데이터가 범주 불균형 문제가 있었기 때문에, accuracy가 아닌 f1-score를 사용하였습니다.


```python
from sklearn.metrics import f1_score

def IBA_contest_result(df_pred):
    while True:
        # 제출 답안의 오류 검증
        if len(df_pred) != 8141:    # test 데이터의 길이 == 8141
            print('# 오류 : 제출 답안의 길이가 일치하지 않음')
            break
            
        if df_pred.isna().any().any():
            print('# 오류 : 제출 답안에 결측치 존재')
            break
            
        if 'pred' not in(df_pred.columns):
            print('# 오류 : 제출 답안에 pred 컬럼이 없음')
            break
            
        mode = input('1 - 중간 성과 /// 2 - 최종 성과 ==> ')
        if mode == '1':
            mode = '중간 평가'
            answer = pd.read_csv('answer_50.csv', index_col=0)
        
        elif mode =='2':
            mode = '최종 성과'
            answer = pd.read_csv('answer.csv', index_col=0)
            
        else:
            print('# 오류 : mode 입력값 오류')
            break
        
        result = answer.merge(right=df_pred, how='left', left_index=True, right_index=True)
        
        if result.isna().any().any():
            print('# 오류 => 예측 파일의 index 오류')
            break
            
        f1 = f1_score(result['target'], result['pred'])
        print(f'\n##### {mode} #####\n')
        print(f'f1_score ==> {f1:.4f}')
        
        break
```

함수를 사용하는 방법은 제출 받은 파일을 불러와서 함수에 넣어주기만 하면 됩니다.


```python
# 제출 답안 불러오기
fpath = 'example.csv'
df_pred = pd.read_csv(fpath, index_col=0)

# 성과 평가
IBA_contest_result(df_pred)
```

    1 - 중간 성과 /// 2 - 최종 성과 ==> 1
    
    ##### 중간 평가 #####
    
    f1_score ==> 0.3230


이러한 프로그램을 사용하여 대회를 성공적으로 마칠 수 있었습니다. 참가자 입장이 아닌 운영자의 입장에서 대회를 바라볼 수 있는 좋은 경험이었습니다.\
대회를 진행하기 위해 다른 방법이 아닌 스스로 프로그램을 만들어 대회를 기획한 것이 힘들어도 유익한 경험이었던 것 같습니다.\
혹시 소모임이나 동아리 내 자체 공모전을 기획하시는 분들께 제 글이 도움이 되었길 바라며, 오늘은 여기서 마무리하도록 하겠습니다. 읽어주셔서 감사합니다.
