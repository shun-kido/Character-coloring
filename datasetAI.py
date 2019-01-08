import numpy as np
import glob
import h5py
import cv2
from keras.preprocessing.image import load_img, img_to_array
import os
import sys
from statistics import mean
import os.path

def re_size(path):
    name = path.lstrip(datapath)
    #print(name)
    file = cv2.imread(path)
    #学習時のサイズを入力
    file = cv2.resize(file, (128,128), interpolation = cv2.INTER_AREA)
    cv2.imwrite(inpath+"/org/"+name+".jpg", file)

#線画抽出
def make_contour_image(path):
    neiborhood24 = np.array([[1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1]],
                             np.uint8)  
    
    #線画を抽出
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    dilated = cv2.dilate(gray, neiborhood24, iterations=1)
    diff = cv2.absdiff(dilated, gray)
    contour = 255 - diff
    #name = path.lstrip(inpath+"/org/")
    #cv2.imwrite(inpath+"/mask/"+name+".jpg", contour)
    return contour

def make_hint(true, mask):
    #ヒント用のRGB情報を取得
    hint = cv2.imread(true)
    masked = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
    height = hint.shape[0]
    width = hint.shape[1]

    num = np.random.randint(0,4)  #ヒントの数（０～３）
    #print(num)
    for l in range(num):
        hint_col = []
        t = np.random.randint(0 ,3)
        if t == 0:
            deep = 0
            wide = 1
        elif t == 1:
            deep = 1
            wide = 0
        else:
            deep = 1
            wide = 1

        x = (np.random.randint(0, width-16))
        y = (np.random.randint(0, height-16))

        for i in range(15):
            if deep == 1:
                for j in range(3):
                    [b, g, r] = hint[y+(i*deep), x+(i*wide)+j]
                    hint_col.append([b, g, r])
            else:
                for j in range(3):
                    [b, g, r] = hint[y+(i*deep)+j, x+(i*wide)]
                    hint_col.append([b, g, r])
        
        #ヒントを与える
        m = 0
        for i in range(15):
            if deep == 1:
                for j in range(3):
                    masked[y+(i*deep), x+(i*wide)+j] = hint_col[m][0], hint_col[m][1], hint_col[m][2]
                    m += 1
            else:
                for j in range(3):
                    masked[y+(i*deep)+j, x+(i*wide)] = hint_col[m][0], hint_col[m][1], hint_col[m][2]
                    m += 1
    name = true.lstrip(inpath+"/org/")
    cv2.imwrite(inpath+"/mask/"+name+".jpg", masked)
                    

def detect(filename, cascade_file = "./lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)
        
    cascade = cv2.CascadeClassifier(cascade_file)
    gray = cv2.cvtColor(filename, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 2,
                                     minSize = (16, 16))
    return faces


inpath = './input2'    #出力先
outpath = './output2'    #h5py保存先
datapath = './safebooru'    #読み込み先


orgs = []
masks = []

print('original img')
j_file = glob.glob(datapath+"/*.jpg")
p_file = glob.glob(datapath+"/*.png")
files = j_file + p_file
    
#print(files)

#彩度と顔認証で画像厳選
for i, file in enumerate(files):
    filer = cv2.imread(file)
    
    #顔認証 
    faces = detect(filer)
    if faces == ():
        #cv2.imwrite('./out/'+str(i)+'.jpg', filer)
        del files[i]
        print("del:{}, None".format(str(i),))
        continue
    
    #彩度
    hsv = cv2.cvtColor(filer, cv2.COLOR_BGR2HSV)
    
    h = hsv.shape[0]
    w = hsv.shape[1] // 2    
    hsv_s = []
    for j in range(h):
        s = hsv[j, w, 1]
        hsv_s.append(s)
        
    ave_s = mean(hsv_s)
    
    #彩度<20はデータセットから除外
    if ave_s < 18:
        #cv2.imwrite('./out/'+str(i)+'.jpg', filer)
        del files[i]
        print("del:{},{}".format(file, ave_s))
    
    
for file in files:
    re_file = re_size(file)
    
true_files = glob.glob(inpath+'/org/*.jpg')
for file in true_files:
    mask_file = make_contour_image(file)
    mask_file = make_hint(file, mask_file)
    
mask_files = glob.glob(inpath+'/mask/*.jpg')

for imgfile in true_files:
    try:
        #print(imgfile)
        img = load_img(imgfile)
        imgarray = img_to_array(img)
        #orgs.append(imgarray)
        masks.append(imgarray)
    except:
        continue

print('mask img')
for imgfile in mask_files:
    try:
        #print(imgfile)
        img = load_img(imgfile)
        imgarray = img_to_array(img)
        #masks.append(imgarray)
        orgs.append(imgarray)
    except:
        continue

perm = np.random.permutation(len(orgs))
orgs = np.array(orgs)[perm]
masks = np.array(masks)[perm]
threshold = len(orgs)//10*9
imgs = orgs[:threshold]
gimgs = masks[:threshold]
vimgs = orgs[threshold:]
vgimgs = masks[threshold:]
'''
print('shapes')
print('org imgs  : ', imgs.shape)
print('mask imgs : ', gimgs.shape)
print('test org  : ', vimgs.shape)
print('test tset : ', vgimgs.shape)
'''
outh5 = h5py.File(outpath+'/datasetimages.hdf5', 'w')
outh5.create_dataset('train_data_raw', data=imgs)
outh5.create_dataset('train_data_gen', data=gimgs)
outh5.create_dataset('val_data_raw', data=vimgs)
outh5.create_dataset('val_data_gen', data=vgimgs)
outh5.flush()
outh5.close()
