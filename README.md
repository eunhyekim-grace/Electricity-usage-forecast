# Electricity-usage-forecast

2023년 8월 중반 부터 말까지 참여했던 한국 에너지 공단이 주최하고 Dacon이 주관한 '2023 전력사용량 예측 AI 경진대회' 공모전 코드 파일 정리 및 기록입니다. 

## 대회 개요 및 설명

### 배경
안정적이고 효율적인 에너지 공급을 위해서는 전력 사용량에 대한 정확한 예측이 필요합니다.
따라서 한국에너지공단에서는 전력 사용량 예측 시뮬레이션을 통한 효율적인 인공지능 알고리즘 발굴을 목표로 본 대회를 개최합니다.

### 설명
건물 정보와 시공간 정보를 활용하여 특정 시점의 전력 사용량을 예측하는 AI 모델을 개발해주세요.

### 데이터
* train.csv - 100개 건물들의 2022년 06월 01일부터 2022년 08월 24일까지의 데이터
  일시별 기온, 강수량, 풍속, 습도, 일조, 일사 정보
  전력사용량(kWh) 포함
* building_info.csv - 100개 건물 정보
  건물 번호, 건물 유형, 연면적, 냉방 면적, 태양광 용량, ESS 저장 용량, PCS 용량
* test.csv - 100개 건물들의 2022년 08월 25일부터 2022년 08월 31일까지의 데이터
  일시별 기온, 강수량, 풍속, 습도의 예보 정보
* submission.csv - 제출을 위한 양식, 100개 건물들의 2022년 08월 25일부터 2022년 08월 31일까지의 전력사용량(kWh)을 예측

## 데이터 전처리

### 변수 추가
+ 시계열 데이터 시간 관련 변수 추가
  월, 일, 시간, 요일, 공휴일
```
eda_df['datetime'] = pd.to_datetime(eda_df['datetime'])
eda_df['hour'] = eda_df['datetime'].dt.hour
eda_df['weekday'] = eda_df['datetime'].dt.weekday #요일
eda_df['date'] = eda_df['datetime'].dt.date
eda_df['day'] = eda_df['datetime'].dt.day
eda_df['month'] = eda_df['datetime'].dt.month
eda_df['weekend'] = eda_df['weekday'].isin([5,6]).astype(int)
```
+ 건물별 요일별 시간별 반전량 평균 열 추가
```
power_mean = pd.pivot_table(train, values = 'power', index = ['num', 'hour', 'day'], aggfunc = np.mean).reset_index()
train['day_hour_mean'] = train.progress_apply(lambda x : power_mean.loc[(power_mean.num == x['num']) & (power_mean.hour == x['hour']) & (power_mean.day == x['day']) ,'power'].values[0], axis = 1)
## 건물별 요일별 시간별 발전량 표준편차 넣어주기
power_hour_std = pd.pivot_table(train, values = 'power', index = ['num', 'hour', 'day'], aggfunc = np.std).reset_index()
train['day_hour_std'] = train.progress_apply(lambda x : power_hour_std.loc[(power_hour_std.num == x['num']) & (power_hour_std.hour == x['hour']) & (power_hour_std.day == x['day']) ,'power'].values[0], axis = 1)
train.head()
```


+ 전력 사용량은 기온의 변화에 따라 즉각적으로 변하지 않음
  기온 대신 냉방도일의 개념을 추가하여 변수 생성
```
def CDH(xs):
    ys = []
    for i in range(len(xs)):
        if i < 11:
            ys.append(np.sum(xs[:(i+1)]-24))
        else:
            ys.append(np.sum(xs[(i-11):(i+1)]-24))
    return np.array(ys)

cdhs = np.array([])
for num in range(1,101,1):
    temp = train[train['num'] == num]
    cdh = CDH(temp['temp'].values)
    cdhs = np.concatenate([cdhs, cdh])
train['CDH'] = cdhs
```

### 결측치 및 이상치 처리
+ 일조, 일사는 test 데이터에는 없으므로 제거
+ 강수량 결측치는 0으로 처리
+ 풍속, 습도 결측치는 보간법으로 대체
```
eda_df.precipitation.fillna(0.0, inplace = True)
eda_df.windspeed = eda_df.windspeed.interpolate(method = 'polynomial', order = 3)
eda_df.humidity = eda_df.humidity.interpolate(method = 'polynomial', order = 3)
```
+ 이상치 처리
  건물별 그래프 확인하여 34번 - 월요일, 56번 건물 - 수요일 이상치 확인
  NUll 값으로 대체한 후 pandas의 interpolation(보간법) 활용하여 이상치 채움

### test 셋에도 같은 과정 적용으로 변수 통일



## 모델링

### xgboost, lgbm, catboost
+ 트리 기반 모델로 회귀와 분류에 모두 사용됨
+ 트리 기반 모델이라 변수 scale 의 영향을 덜 받아 따로 scaling 을 안해줘도 됨
https://docs.aws.amazon.com/ko_kr/sagemaker/latest/dg/catboost-hyperparameters.html  
파라미터 설정 참고한 링크

```
grid = {'iterations': [500],
        'learning_rate' : [0.01, 0.03, 0.05, 0.07],
        'depth': [4,6,8],
        'l2_leaf_reg': [2,4,6,8],
        'subsample': [0.8],
        'colsample_bylevel': [0.8],
        'random_strength' : [1,3,5,7]
         }
```
gridsearchCV 를 이용해 최적의 파라미터 설정

### 손실함수
$MAE = \frac{1}{n}\Sigma_{i=1}^n| \hat{Y_i}-Y_i|$ <br/>
$RSME = (\frac{1}{n}\Sigma_{i=1}^n (\hat{Y_i}-Y_i)^2)^{0.5}$  

과소적합보다 과대 적합이 더 좋은 평가지표인 SMAPE 이므로 RMSE 보다 MAE 가 더 좋은 손실함수로 판단





## 결과 및 리뷰
+ xgboost 와 lgbm 으로 시작하였고 제일 결과가 좋은 모델은 catboost 였음
+ Pycaret 으로 가장 효과가 좋은 모델을 찾았을 때도 catboost, xgboost, lgbm 순임
+ 데이터 전처리를 마치고 바로 모델을 돌려보는 것보다는 Pycaret 을 돌려보고 성능 좋은 모델을 선별하여 모델링하는게 좋을 것으로 판단됨.
