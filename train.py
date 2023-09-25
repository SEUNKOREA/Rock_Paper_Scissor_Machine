import pandas as pd
import numpy as np
import cv2

# 각도와 제스쳐 라벨정보가 포함된 csv 파일 읽기
data_df = pd.read_csv('./data/gesture_train.csv')
angle = np.array(data_df.iloc[:, :-1]).astype(np.float32) # 각도 추출
label = np.array(data_df.iloc[:,-1]).astype(np.float32) # 라벨 추출

# KNN 모델객체 생성후 훈련 
knn = cv2.ml.KNearest_create() 
knn.train(angle, cv2.ml.ROW_SAMPLE, label) # cv2.ml.ROW_SAMPLE: 데이터가 행 단위로 구성되어있음

# 훈련된 모델 저장
knn.save('./model/knn.xml')