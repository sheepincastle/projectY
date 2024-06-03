import keras
import pandas
import numpy

'''
keras를 이용해 딥러닝 모델 제작
numpy를 이용해 배열을 numpy의 형태로 변환해 딥러닝에 이용할 수 있게 함
'''

#앞에서 저장한 pickle데이터 불러옴
eye_data = pandas.read_pickle('eye_data.pkl')
#딥러닝에 입력값과 정답으로 입력할 데이터를 담을 배열 생성
datax = []
datay = []
#데이터 담음
for i, rows in eye_data.iterrows():
    datax.append([rows['distance1'], rows['distance2'], rows['distance3'], rows['distance4'], rows['distance5'], rows['distance6']])
datay = eye_data['value'].values

#keras로 딥러닝 학습모델 제작
#hidden layer는 총 5개로 각각 64, 64, 128, 128, 64개의 노드를 가짐

#활성화 함수로 사용된 relu 함수는 음수는 0으로 양수는 그대로 반환하는 함수이다.
#1. 간단한 형태이기에 계산이 빠르다
#2. 입력값이 매우 크거나 작아 기울기가 0에 가까워지는 그래디언트 소실 문제가 없다
#3. 음수는 모두 0을 출력하고 양수만 변경하지 않고 출력하게 되는데 이를 희소 활성화라고 한다. 이렇게 하면 계산 비용이 줄고, 과적합을 줄이는 데도 도움이 된다.
#위와 같은 장점으로 인해 relu 함수를 활성화 함수로 사용한다.

#결과값을 0~1 사이 실수로 만들어 눈을 감았는지 떴는지의 확률을 표현해야 함
#sigmoid 함수를 이용해 결과값을 0~1 사이 실수로 변환
model = keras.models.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])

#모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'] )
#모델에 입력값과 정답을 입력하고 1000번 반복하여 최적화를 수행한다.
model.fit(numpy.array(datax), numpy.array(datay), epochs=1000)

#모델을 keras 파일 형태로 저장한다
model.save('predict_model.keras')
