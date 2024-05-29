import tensorflow as tf
import keras
import pandas
import numpy

eye_data = pandas.read_pickle('eye_data.pkl')
datax = []
datay = []
for i, rows in eye_data.iterrows():
    datax.append([rows['distance1'], rows['distance2'], rows['distance3'], rows['distance4'], rows['distance5'], rows['distance6']])

datay = eye_data['value'].values

#keras로 딥러닝 학습모델 제작
model = keras.models.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])

#model compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'] )
model.fit(numpy.array(datax), numpy.array(datay), epochs=1000)

#모델 저장
model.save('predict_model.keras')