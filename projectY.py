import dlib
import cv2
import numpy as np
import math
import keras
import time
import pyfirmata
#time을 이용해 시간 측정
#pyfirmata 이용해 아두이노 조작
#아두이노 포트, 부저 위치

#dlib에 있는 얼굴 인식 AI 모델 가져옴
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('dlib data/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib data/dlib_face_recognition_resnet_model_v1.dat')

#url을 통해 스트리밍되는 esp32 cam의 영상 받음
url = "https://<ESP32_CAM_IP>/stream"
cap = cv2.VideoCapture(url)
#아두이노 연결
board = pyfirmata.Arduino('COM1')

#저장한 keras 형식의 딥러닝 모델 불러옴
model = keras.models.load_model('predict_model.keras')

is_open = True#눈이 열렸는가
    
start = 0.0
end = 0.0

#눈을 찾아 좌표를 배열로 만드는 함수, 이미지를 입력으로 받음
def find_eyes(img):
    #받은 이미지를 얼굴 인식 AI 모델에 넣어 얼굴을 모두 찾음
    dets = detector(img, 1)
    left_landmarks = []
    right_landmarks = []
    result = []
    
    if len(dets) == 0:
        #얼굴을 찾지 못했을 때 반환할 값: 0
        return result
    
    #튜플의 형태로 dets 변환, dets의 개수만큼 반복
    #k는 사람 얼굴마다 주어지는 번호
    #d는 얼굴을 담은 박스
    for k, d in enumerate(dets):
        #사람의 얼굴을 나타내는 68개의 랜드마크(점) 저장
        shape = sp(img, d)
        
        #눈 랜드마크의 좌표 저장
        for i in range(6):
            x = shape.part(i+36).x
            y = shape.part(i+36).y
            left_landmarks.append((x, y))
        for i in range(6):
            x = shape.part(i+42).x
            y = shape.part(i+42).y
            right_landmarks.append((x, y))
    
    #눈의 0, 3번째 좌표, 1, 5번째 좌표, 2, 4번째 좌표 사이 거리 계산
    distance1_right = math.sqrt((right_landmarks[0][0]-right_landmarks[3][0])**2+(right_landmarks[0][1]-left_landmarks[3][1])**2)
    distance2_right = math.sqrt((right_landmarks[1][0]-right_landmarks[5][0])**2+(right_landmarks[1][1]-left_landmarks[5][1])**2)
    distance3_right = math.sqrt((right_landmarks[2][0]-right_landmarks[4][0])**2+(right_landmarks[2][1]-left_landmarks[4][1])**2)
    distance1_left = math.sqrt((left_landmarks[0][0]-left_landmarks[3][0])**2+(left_landmarks[0][1]-left_landmarks[3][1])**2)
    distance2_left = math.sqrt((left_landmarks[1][0]-left_landmarks[5][0])**2+(left_landmarks[1][1]-left_landmarks[5][1])**2)
    distance3_left = math.sqrt((left_landmarks[2][0]-left_landmarks[4][0])**2+(left_landmarks[2][1]-left_landmarks[4][1])**2)
    
    #결과값을 리스트로 정리
    result = [distance1_right, distance2_right, distance3_right, distance1_left, distance2_left, distance3_left]
    
    #리스트 반환
    return result

while True:
    #url의 영상을 읽어서 이미지로 저장
    #성공했다면 ret에 true 저장, 실패했다면 ret에 false 저장
    #frame에 이미지 저장
    ret, frame = cap.read()
    #실패하면 반복문 탈출
    if not ret:
        print("Failed to grab frame")#이미지 획득에 실패했습니다 라는뜻
        break
    
    #find_eyes 함수를 통해 딥러닝 완료된 모델에 넣을 데이터 수집, datas에 저장
    datas = find_eyes(frame)
    inputs = np.array(datas)
    #입력값을 예측 모델에 넣고 예측값 저장
    prediction = model.predict(inputs)
    #예측값은 0~1 사이 실수이기 때문에 반올림
    prediction = round(prediction)
    
    #시간을 재서 3초 이상 눈을 감은 상태가 유지되면 아두이노에 신호를 전송해 부저를 울리게 한다
    if is_open == True:
        if prediction == 0:
            start = time.time()
    elif is_open == False:
        end = time.time()
    
    is_open = bool(prediction)
    
    if (end-start) > 3:#3초 이상 눈 감았을때
        board.digital[2].write(1)#부저 활성화
        time.sleep(1)
        board.digital[2].write(0)#부저 비활성화
        time.sleep(1)
    
    #q키를 입력하면 반복문 탈출
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break