def data_face_deector(files):
        
    facedetector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    f=[]
    for i in range(len(files)):
        img1=cv2.imread(files[i],0)
        faces=facedetector.detectMultiScale(img1,1.1,4)
        face=np.array(faces)
        if face.shape==(1,4):
            for (x,y,w,h) in faces:
                img=img1[y:y+h,x:x+w]
                f.append(img)   
    f1=np.array(f)
    return(f1)
#####
import numpy as np 
import cv2
import glob
def my_train_test_split2(adress_file,n_class1,n_class2,n_samples):
    files = glob.glob(adress_file, 
                   recursive = True)
    files1=[]
    files2=[]
    for i in range(n_class1):
        files1.append(files[i]) 
    for i in range(n_class1,n_samples):
        files2.append(files[i])
        
    f1=data_face_deector(files1)
    f2=data_face_deector(files2)

    
    np.random.seed(1)
    shuffle1=np.random.permutation(f1.shape[0])
    shuffle2=np.random.permutation(f2.shape[0])
    train_x=[]
    train_y=[]
    n_train=1220
    n_test=521
    for i in range(n_train):
        smile=f1[shuffle1[i]]
        img1=smile
        #img1=cv2.resize(img1,(170,180))
        train_x.append(img1)
        train_y.append(1)    
        non_smile=f2[shuffle2[i]]
        img2=non_smile
        #img2=cv2.resize(img2,(170,180))
        train_x.append(img2)
        train_y.append(0)
    x_train=np.array(train_x)
    y_train=np.array(train_y)
    test_x=[]
    test_y=[]
    for i in range( n_train,n_train+n_test):
        smile=f1[shuffle1[i]]
        img1=smile
        #img1=cv2.resize(img1,(170,180))
        test_x.append(img1)
        test_y.append(1)
        non_smile=f2[shuffle2[i]]
        img2=non_smile
        #img2=cv2.resize(img2,(170,180))
        test_x.append(img2)
        test_y.append(0)
    x_test=np.array(test_x)
    y_test=np.array(test_y)
    
    return(x_train,y_train,x_test,y_test)
#####
def lbp_feature_extraction(x,r):
    from skimage.feature import local_binary_pattern
    radius = r
    n_points = 8 * radius
    f=[]
    for i in range(x.shape[0]):
        image=x[i]
        lbp = local_binary_pattern(image, n_points, radius)
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
        f.append(hist)
    f1=np.array(f)
    return(f1)
#####
def haar_feature_extraction(x):
    
    import pywt
    flh=[]
    fhl=[]

    for i in range(x.shape[0]):
        ll,(lh,hl,hh)=pywt.dwt2(x[i],'haar')
        flh.append(lh)
        fhl.append(hl)
    fhl=np.array(fhl)
    flh=np.array(flh)
    return(flh,fhl)
#######
def lbp2_feature_extraction(x,r,method):
    from skimage.feature import local_binary_pattern
    radius = r
    n_points = 8 * radius
    f=[]
    for i in range(x.shape[0]):
        image=x[i]
        lbp = local_binary_pattern(image, n_points, radius,method)
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
        f.append(hist)
    f1=np.array(f)
    return(f1)
#####
def lbp_matrix(x,r):
    from skimage.feature import local_binary_pattern
    radius = r
    n_points = 8 * radius
    f=[]
    for i in range(x.shape[0]):
        image=x[i]
        lbp = local_binary_pattern(image, n_points, radius)
        f.append(lbp)
    f1=np.array(f)
    return(f1)
#####
def lbp2_matrix(x,r,method):
    from skimage.feature import local_binary_pattern
    radius = r
    n_points = 8 * radius
    f=[]
    for i in range(x.shape[0]):
        image=x[i]
        lbp = local_binary_pattern(image, n_points, radius, method)
        f.append(lbp)
    f1=np.array(f)
    return(f1)
