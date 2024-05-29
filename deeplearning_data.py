import dlib
import cv2
import glob
import math
import pandas

'''
dlib을 이용해 얼굴 이미지 분석
opencv(=cv2)를 이용해 이미지를 엶
glob을 이용해 대량의 이미지 및 자료 한번에 엶
math를 이용해 사칙연산 이외의 수학 연산 수행
pandas를 이용해 행렬 생성 및 관리
'''

#dlib에 있는 얼굴 인식 AI 모델 가져옴
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('dlib data/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib data/dlib_face_recognition_resnet_model_v1.dat')

#눈을 찾아 좌표를 배열로 만드는 함수, 이미지를 입력으로 받음
def find_eyes(img):
    #받은 이미지를 얼굴 인식 AI 모델에 넣어 얼굴을 모두 찾음
    dets = detector(img, 1)
    left_landmarks = []
    right_landmarks = []
    result = []
    
    if len(dets) == 0:
        #얼굴을 찾지 못했을 때 빈 리스트를 반환하여 저장할 값에서 배제
        return result
    
    #dets의 개수만큼 반복문 반복
    #k는 사람 얼굴마다 주어지는 번호
    #d는 얼굴을 담은 박스
    for k, d in enumerate(dets):
        #사람의 얼굴을 나타내는 68개의 랜드마크(점) 저장
        shape = sp(img, d)
        
        #눈을 나타내는 랜드마크의 좌표 저장
        for i in range(6):#왼쪽 눈
            x = shape.part(i+36).x
            y = shape.part(i+36).y
            left_landmarks.append((x, y))
        for i in range(6):#오른쪽 눈
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

#이미지 불러오기
#0은 닫은 눈, 1은 열린 눈
img_list_0 = glob.glob('material data/ClosedFace/*.jpg')#닫은 눈 이미지들
img_list_1 = glob.glob('material data/OpenFace/*.jpg')#열린 눈 이미지들
print('img0: ', len(img_list_0))
print('img1: ', len(img_list_1))

#학습할 데이터와 정답을 저장할 행렬 생성
eye_data = pandas.DataFrame(columns=['distance1', 'distance2', 'distance3', 'distance4', 'distance5', 'distance6', 'value'], index=[], dtype=float)

#학습할 데이터 및 정답 저장
#find_eyes함수의 결과로 나온 리스트의 항목이 7 미만이라면, 즉 값의 일부가 없다면 continue를 통해 데이터를 저장하지 않고 다음 사진으로 넘어감
for i in range(len(img_list_0)):
    a = find_eyes(cv2.imread(img_list_0[i]))
    a.append(0)
    if len(a) < 7:
        continue
    eye_data.loc[i] = a
for i in range(len(img_list_1)):
    a = find_eyes(cv2.imread(img_list_1[i]))
    a.append(1)
    if len(a) < 7:
        continue
    eye_data.loc[i + len(eye_data)] = a
    
print(eye_data)

#값을 저장한 행렬을 pickle의 형태로 데이터 저장
eye_data.to_pickle('eye_data.pkl')
    
print('fin')