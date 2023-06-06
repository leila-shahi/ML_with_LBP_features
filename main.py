import numpy as np
import cv2
import glob
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage.feature import local_binary_pattern

import my_MV_funcs

x_train,y_train,x_test,y_test=my_MV_funcs.my_train_test_split2('files/file****.jpg',2162,1838,4000)
print(x_train.shape)
print(x_train.shape[0])

x_train_sbys=np.array([cv2.resize(img,(32,32)) for img in x_train])
x_test_sbys=np.array([cv2.resize(img,(32,32)) for img in x_test])

x_train_sbys_flat=np.array([np.reshape(img,(img.shape[0]*img.shape[1])) for img in x_train_sbys])
x_test_sbys_flat=np.array([np.reshape(img,(img.shape[0]*img.shape[1])) for img in x_test_sbys])

xlbp1_train=my_MV_funcs.lbp_matrix(x_train_sbys,2)
xlbp1_test=my_MV_funcs.lbp_matrix(x_test_sbys,2)

xlbp1_train_flat=np.array([np.reshape(img,(img.shape[0]*img.shape[1])) for img in xlbp1_train])
xlbp1_test_flat=np.array([np.reshape(img,(img.shape[0]*img.shape[1])) for img in xlbp1_test])

model_svm=svm.SVC(kernel='rbf')
model_svm.fit(xlbp1_train_flat,y_train)
print(xlbp1_train_flat.shape)
print('acuracy on lbp2')
print(model_svm.score(xlbp1_test_flat,y_test))

scalor=StandardScaler()
xlbp1_train_flat1=scalor.fit_transform(xlbp1_train_flat)
xlbp1_test_flat1=scalor.transform(xlbp1_test_flat)

model_svm=svm.SVC(kernel='rbf')
model_svm.fit(xlbp1_train_flat1,y_train)
print(xlbp1_train_flat1.shape)
print('acuracy on lbp2+norm')
print(model_svm.score(xlbp1_test_flat1,y_test))


dr=PCA(n_components=0.9)
xlbp1_train_flat1_1=dr.fit_transform(xlbp1_train_flat1)
xlbp1_test_flat1_1=dr.transform(xlbp1_test_flat1)

model_svm=svm.SVC(kernel='rbf')
model_svm.fit(xlbp1_train_flat1_1,y_train)
print(xlbp1_train_flat1_1.shape)
print('acuracy on lbp2+norm+pca')
print(model_svm.score(xlbp1_test_flat1_1,y_test))

cap = cv2.VideoCapture(0)
facedetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while 1:
    ret, frame = cap.read()
    if ret:
        #frame = cv2.resize(frame, (640, 480))
        g_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetector.detectMultiScale(g_frame, 1.1, 4)
        # print(faces)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            frame1 = g_frame[y:y + h, x:x + w]
            img = frame1
            image = cv2.resize(img, (32, 32))
            lbp_matrix = local_binary_pattern(image, 16, 2)
            lbp_matrix_flat = np.reshape(lbp_matrix, (lbp_matrix.shape[0] * lbp_matrix.shape[1]))
            # print(lbp_matrix_flat.shape)
            lbp_matrix_flat1 = lbp_matrix_flat.reshape(1, -1)
            lbp_matrix_flat2 = scalor.transform(lbp_matrix_flat1)
            lbp_matrix_flat3 = dr.transform(lbp_matrix_flat2)
            # print(lbp_matrix_flat3.shape)

            p = model_svm.predict(lbp_matrix_flat3)
            # print(p)
            if p == 1:
                cv2.putText(frame, "smile", (x, y + h), 0, 1, 255)
            else:
                cv2.putText(frame, "non_smile", (x, y + h), 0, 1, 255)

        cv2.imshow('frame', frame)
        c = cv2.waitKey(1)
        if c == 27:
            break

cap.release()
cv2.destroyAllWindows()