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
    cv2.imwrite(inpath+"/org/"+name, file)

#線画抽出
def make_contour_image(path, s):

    neiborhood24 = np.array([[1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1],
                             [1, 1, 1, 1, 1]],
                             np.uint8)


    neiborhood8 = np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]],
                            np.uint8)

    neiborhood4 = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]],
                            np.uint8)

    #線画を抽出
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    dilated = cv2.dilate(gray, neiborhood4, iterations=1)
    diff = cv2.absdiff(dilated, gray)
    contour = 255 - diff
    #print(contour.shape)

    if s < 60:
        bl = 235
    elif s < 85:
        bl = 220
    else:
        bl = 210
    for i, x in enumerate(contour):
        for j, y in enumerate(x):
            if y <= bl:
                contour[i][j] = 140
            else:
                contour[i][j] = 255

    ''' ///ヒントなしで保存する場合///
    name = path.lstrip(inpath+"/org/")
    cv2.imwrite(inpath+"/mask/"+name+".jpg", contour)
    '''
    return contour

def make_hint(true, mask):
    #ヒント用のRGB情報を取得
    hint = cv2.imread(true)
    masked = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    height = hint.shape[0]
    width = hint.shape[1]

    num = np.random.randint(5, 15)#ヒントの数（0～10）
    num = 0
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

        x = (np.random.randint(0, width-21))
        y = (np.random.randint(0, height-21))

        for i in range(20):
            if deep == 1:
                for j in range(2):
                    [b, g, r] = hint[y+(i*deep), x+(i*wide)+j]
                    hint_col.append([b, g, r])
            else:
                for j in range(2):
                    [b, g, r] = hint[y+(i*deep)+j, x+(i*wide)]
                    hint_col.append([b, g, r])

        #ヒントを与える
        m = 0
        for i in range(20):
            if deep == 1:
                for j in range(2):
                    masked[y+(i*deep), x+(i*wide)+j] = hint_col[m][0], hint_col[m][1], hint_col[m][2]
                    m += 1
            else:
                for j in range(2):
                    masked[y+(i*deep)+j, x+(i*wide)] = hint_col[m][0], hint_col[m][1], hint_col[m][2]
                    m += 1
    name = true.lstrip(inpath+"/org/")
    cv2.imwrite(inpath+"/mask/"+name, masked)


def detect(filename, cascade_file = "./lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)

    try:
        gray = cv2.cvtColor(filename, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 2,
                                     minSize = (16, 16))

    except:
        faces = None

    return faces


inpath = './dataset/input'    #出力先
outpath = './dataset/output'    #h5py保存先
datapath = './dataset/safebooru'    #読み込み先


orgs = []
masks = []

print('original img')
j_file = glob.glob(datapath+"/*.jpg")
#p_file = glob.glob(datapath+"/*.png")

files = j_file

#彩度と顔認証で画像厳選
del_num = []
all_s = []
for i, file in enumerate(files):
    #print(file)
    filer = cv2.imread(file)
    name = file.lstrip(datapath)

    #顔認証
    faces = detect(filer)
    if faces == ():
        #cv2.imwrite('./out/'+name+'.jpg', filer)
        del_num.append(i)
        print("del:{}, None".format(file))
        continue

    #彩度取得
    hsv = cv2.cvtColor(filer, cv2.COLOR_BGR2HSV)

    h = hsv.shape[0]
    w = hsv.shape[1]//2
    hsv_s = []

    try:
        for k in range(5):
            for j in range(h):
                s = hsv[j, w+5, 1]
                hsv_s.append(s)

        ave_s = mean(hsv_s)

    except:
        ave_s = 0

    #彩度<18はデータセットから除外
    if ave_s < 18:
        cv2.imwrite('./out/'+name+'.jpg', filer)
        del_num.append(i)
        print("del:{},{}".format(file, ave_s))
        continue

    all_s.append(ave_s)
    print("{},{}".format(file, ave_s))


for i in del_num:
    files[i] = 'N'

for i in range(len(del_num)):
    files.remove('N')

for file in files:
    re_file = re_size(file)

true_files = glob.glob(inpath+'/org/*.jpg')
for i, file in enumerate(true_files):
    mask_file = make_contour_image(file, all_s[i])
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

for imgfile in mask_files:
    try:
        #print(imgfile)
        img = load_img(imgfile)
        imgarray = img_to_array(img)
        #masks.append(imgarray)
        orgs.append(imgarray)
    except:
        continue

#orgs = orgs[:28000]
#masks = masks[:28000]
perm = np.random.permutation(len(orgs))
orgs = np.array(orgs)[perm]
masks = np.array(masks)[perm]
threshold = len(orgs)//10*9
imgs = orgs[:threshold]
gimgs = masks[:threshold]
vimgs = orgs[threshold:]
vgimgs = masks[threshold:]

outh5 = h5py.File(outpath+'/datasetimages.hdf5', 'w')
outh5.create_dataset('train_data_raw', data=imgs)
outh5.create_dataset('train_data_gen', data=gimgs)
outh5.create_dataset('val_data_raw', data=vimgs)
outh5.create_dataset('val_data_gen', data=vgimgs)
outh5.flush()
outh5.close()
