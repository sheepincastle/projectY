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

'''
딥 러닝은 인간의 두뇌에서 영감을 얻은 방식으로 데이터를 처리하도록 컴퓨터를 가르치는 인공 지능(AI) 방식이다.
딥 러닝 알고리즘은 인간의 뇌를 모델로 한 신경망이다.
예를 들어 인간의 뇌 안에는 함께 작동하여 정보를 배우고 처리하는 수백만 개의 신경 세포가 상호 연결되어 있다.
마찬가지로 딥 러닝 신경망 또는 인공 신경망도 컴퓨터 내부에서 함께 작동하는 여러 계층의 인공 신경 세포로 구성된다.
인공 신경 세포를 노드라고 하며 이 노드는 수학적 계산을 사용하여 데이터를 처리하는 소프트웨어 모듈이다.
인공 신경망은 이 노드를 사용하여 복잡한 문제를 해결하는 딥 러닝 알고리즘이다.

인공 신경망은 입력층에서 데이터를 입력받는다.
또한 은닉 계층에서 서로 연결되어 정보를 계산하고 올바른 결과값을 유도한다.
출력 계층은 출력할 노드로 구성되어 있다.
이번 프로젝트에서는 눈을 감은 것과 뜬것 2가지를 출력해야 하기 때문에 한 노드에서 0~1 사이 실수로 표현한다.

딥러닝 학습시 최대한 틀리지 않는 방향으로 학습해야 하는데 얼마나 틀리는지 알게 하는 함수가 손실함수(loss function)이다.
이 손실함수의 최솟값을 찾는 것을 학습 목표로 한다.
이를 수행하는 최적화 알고리즘이 optimizer이다.

optimizer는 학습속도를 빠르고 안정적이게 하는 것을 목표로 한다.
Adam은 Momentum과 RMSProp를 섞은 알고리즘이다.
즉, 진행하던 속도에 관성을 주고 최근 경로의 곡면의 변화량에 따른 적응적 학습률을 갖은 알고리즘이다.
매우 넓은 범위의 아키덱처를 가진 서로 다른 신경망에서 잘 작동한다는 것이 증명되어 일반적 알고리즘에 현재 가장 많이 사용되고 있다.
아담의 강점은 bounded step size 이다.
Momentum 방식과 유사하게 지금까지 계산해온 기울기의 지수 평균을 저장하며 RMSProp과 유사하게 기울기의 제곱값에 지수평균을 저장한다.
Adam 에서는 기울기 값과 기울기의 제곱값의 지수이동편균을 활용하여 step 변화량을 조절한다.
또한, 초기 몇번의 update 에서 0으로 편향되어 출발 지점에서 멀리 떨어진 곳으로 이동하는,
초기 경로의 편향 문제가 있는 RMSProp의 단점을 보정하는 매커니즘이 반영되어있다.

평가 지표(Metrics)는 모델의 성능을 숫자로 표현하는 것을 말한다.
각종 평가 지표들은 수식에 따라 다양한 특성을 지니고 있어서 올바른 지표를 선택하는 것이 굉장히 중요하다.
Accuracy는 정확도를 나타내는 값으로 혼동 행렬로 따지면 모든 케이스 중 TP, TN의 비율을 구한 것입니다. 쉽게 말하면 얼마나 실제와 동일하게 예측했는 지를 측정합니다. Accuracy 값은 비율이기에 0과 1 사이의 값을 가지게 되며, 1에 가까울수록 좋은 성능을 보인다고 할 수 있습니다. 다만, Accuracy의 경우에는 치명적인 문제가 있습니다.

예를 들어, 1000개 중 불량이 5개 나오는 공장에서 불량인지 판단하는 모델을 만들었다고 가정해본다.
평가 지표를 Accuracy로 적용한다면, 단순히 모든 제품을 불량이 아니라고 해도 1000개 중에 5개 빼고는 다 맞는 결과가 나오기 때문에 0.995의 높은 결과를 얻게 된다.
이처럼 class의 불균형이 심한 데이터에서는 활용하는 것이 적절하지 않다.
'''
#모델 컴파일
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'] )
#모델에 입력값과 정답을 입력하고 1000번 반복하여 최적화를 수행한다.
model.fit(numpy.array(datax), numpy.array(datay), epochs=1000)

#모델을 keras 파일 형태로 저장한다
model.save('predict_model.keras')