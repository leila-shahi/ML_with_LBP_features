import numpy as np 
import cv2
import glob
def my_train_test_split(adress_file,n_class1,n_class2,n_samples):
    files = glob.glob(adress_file, 
                   recursive = True)
    files1=[]
    files2=[]
    for i in range(n_class1):
        files1.append(files[i]) 
    for i in range(n_class1,n_samples):
        files2.append(files[i]) 
    np.random.seed(1)
    shuffle1=np.random.permutation(n_class1)
    shuffle2=np.random.permutation(n_class2)
    train_x=[]
    train_y=[]
    n_train=1280
    n_test=557
    for i in range(n_train):
        smile=files1[shuffle1[i]]
        img1=cv2.imread(smile,0)
        train_x.append(img1)
        train_y.append(1)
    for i in range(n_train):    
        non_smile=files2[shuffle2[i]]
        img2=cv2.imread(non_smile,0)
        train_x.append(img2)
        train_y.append(0)
    x_train=np.array(train_x)
    y_train=np.array(train_y)
    
    
    test_x=[]
    test_y=[]
    for i in range( n_train,n_train+n_test):
        smile=files1[shuffle1[i]]
        img1=cv2.imread(smile,0)
        test_x.append(img1)
        test_y.append(1)
    for i in range( n_train,n_train+n_test):
        non_smile=files2[shuffle2[i]]
        img2=cv2.imread(non_smile,0)
        test_x.append(img2)
        test_y.append(0)
    x_test=np.array(test_x)
    y_test=np.array(test_y)
    
    return(x_train,y_train,x_test,y_test)
