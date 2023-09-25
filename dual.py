import mediapipe as mp
import cv2
import numpy as np

# 설정
CONFIG = {
    "max_num_hands": 2,
    "knn_checkpoint": './model/knn.xml' 
}

# 제스처 id_class 
rps_gesture = {0: 'rock', 5: 'paper', 9: 'scissors'}

mp_hands = mp.solutions.hands             # 미디어파이프에서 제공하는 손 인식 모델을 사용하기 위한 모듈
mp_drawing = mp.solutions.drawing_utils   # 인식결과를 시각적으로 표기하기 위한 도구

# 손 인식 모델 초기화
hands = mp_hands.Hands(
    max_num_hands=CONFIG["max_num_hands"],
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# train knn model load
knn = cv2.ml.KNearest_load(CONFIG['knn_checkpoint'])

# 웹캠에서 비디오캡쳐 객체 생성
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read() # 웹캠에서 한 프레임씩 이미지를 읽어서 성공여부와 이미지 저장

    if not ret: # ret이 False인 경우(프레임을 제대로 읽어오지 못한 경우)
        continue # 다음 루프로 넘어감
    
    img = cv2.flip(img, 1) # 이미지를 수평방향으로 뒤집음
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 이미지 색상채널 변경

    result = hands.process(img) # 이미지에서 손익식 모델로 손 인식 수행

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # 이미지 색상채널 원래대로 변경

    if result.multi_hand_landmarks is not None: # 만약 손 인식이 성공했다면
        # 가위바위보 결과와 손의 좌표를 저장할 리스트
        rps_result = []

        for res in result.multi_hand_landmarks: # 리스트에 인식한 손의 특징점 x,y,z 좌표 저장되어있음저장되어있음
            joint = np.zeros((21, 3))
            for j, lm in enumerate(res.landmark): # 객체있는 손 특징점 20개를 차례로 순회하면서 
                joint[j] = [lm.x, lm.y, lm.z] # 각 joint마다 x, y, z 좌표 저장
            
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:]
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:]
            v = v2 - v1 # 해당하는 index의 joint 좌표값의 차로 관절벡터 구하기 [20, 3]
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis] # 관절벡터 정규화

            # 각도계산
            angle = np.arccos(np.einsum('nt,nt->n',  # 내적해서 행방향으로 더한 후에 arccos 연산으로 각도를 계산한다.
                                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], # 정규화된 관절벡터에서 해당하는 index 열을 뽑아서 [15,3] 행렬 두 개 만들고
                                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))
            angle = np.degrees(angle) # 라디안 > 도

            # 모델에 들어갈 입력값
            data = np.array([angle], dtype=np.float32)

            # 예측
            ret, results, neighbours, dist = knn.findNearest(data, 3)
                # ret: 입력데이터에 대한 예측결과(클래스 레이블)
                # results: 입력 데이터에 대한 예측 결과를 포함하는 넘파이 배열, (n_samples, 1) 형태로 n_samples는 입력데이터의 샘플 수를 나타냄
                # neighbours: 입력 데이터 샘플에 대한 Nearest neighbour 클래스레이블, (n_samples, k)
                # dist: 입력 데이터 샘플과 가장 가까운 이웃들과의 거리

            # 예측결과에서 클래스 레이블 추출
            idx = int(results[0][0]) # results: array([[9.]], dtype=float32)

            # 예측결과의 클래스레이블이 제스쳐레이블에 있으면 img에 클래스레이블 텍스트 넣음
            if idx in rps_gesture.keys():
                org = (int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0]))
                cv2.putText(img, text=rps_gesture[idx].upper(), org=(org[0], org[1] + 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

                rps_result.append({
                    'rps': rps_gesture[idx], # 제스처 결과를 저장 
                    'org': org # 두개의 손의 위치 
                })
            
            # imgd에 손의 특징점과 연결선 그리기
            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

            # 승자 판별
            if len(rps_result) >= 2:
                winner = None
                text = ''

                if rps_result[0]['rps'] == 'rock': # 첫번째 사람이 주먹을 냈을 떼
                    if rps_result[1]['rps'] == 'rock': 
                        text = 'Tie'
                    elif rps_result[1]['rps'] == 'paper': 
                        text = 'Paper Win'
                        winner = 1
                    elif rps_result[1]['rps'] == 'scissors': 
                        text = 'Rock Win'
                        winner = 0
                
                elif rps_result[0]['rps'] == 'paper': # 첫번째 사람이 보자기를 냈을 떼
                    if rps_result[1]['rps'] == 'rock': 
                        text = 'Paper Win'
                        winner = 0
                    elif rps_result[1]['rps'] == 'paper': 
                        text = 'Tie'
                    elif rps_result[1]['rps'] == 'scissors': 
                        text = 'Scissors Win'
                        winner = 1
                
                elif rps_result[0]['rps'] == 'scissors': # 첫번째 사람이 가위를 냈을 떼
                    if rps_result[1]['rps'] == 'rock': 
                        text = 'Rock Win'
                        winner = 1
                    elif rps_result[1]['rps'] == 'paper': 
                        text = 'Scissors Win'
                        winner = 0
                    elif rps_result[1]['rps'] == 'scissors': 
                        text = 'Tie'
                
                if winner is not None: # 승자가 결정됐다면(비기지 않았다면)
                    # 승자 표시
                    cv2.putText(img, text='Winner', org=(rps_result[winner]['org'][0], rps_result[winner]['org'][1] + 70), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 255, 0), thickness=3)
                
                # 결과멘트 표시
                cv2.putText(img, text=text, org=(int(img.shape[1] / 2), 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=3)


        # dual_rps 창에 img 표시
        cv2.imshow("dual_rps", img)

        # ESC 누르면 창 닫기
        if cv2.waitKey(10) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
cv2.waitKey(1)
