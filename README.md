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

## EDA, Exploartory Data Analysis

### 이상치 처리
건물별 전력 사용량을 seaborn의 heatmap을 사용해서 행은 24hr로 열은 7일로 가지고 있는 그래프를 그려 이상치를 찾아 시계열 데이터의 특성을 고려해 pandas의 보간법으로 값을 변경함.


### 건물 유형 별, 평균 전력량 사용별 clustering
요일 / 시간 별 전력 중앙 값에 대해 scaling하고 건물 유형별, 평균 전력량 사용별 clustering을 진행해 line graph를 그렸을 때 건물 유형 기타를 제외하고는 대체로 비슷한 양상을 보였음. 그러나 보다 정확한 예측을 위해 각 건물 별로 학습을 따로 진행함.

### 태양열
건물 정보 csv에서 태양광 발전판을 소유하고 있는 회사들에 대해 전일 일조량 평균 값과 당일 전력 사용량 평균값 사이의 관계를 x축 일조량 값, y축을 전력 사용량 그래프로 시각화함. 그래프를 보면 태양광 에너지 저축량이 올라감에 따라 전력 사용량이 줄어드는 반비례 관계에 있는 건물들을 확인할 수 있음. 


### ESS+ PCS capacity
태양열과 동일한 방법으로 실시했으나 반비례 관계에 있는 건물이 존재하지 않아 사용하지 않음.


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
