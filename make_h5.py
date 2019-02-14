import numpy as np
import glob
import h5py
import cv2
from keras.preprocessing.image import load_img, img_to_array
import os
import sys
import os.path

inpath = './dataset/input'    #出力先
outpath = './dataset/output'

true_files = glob.glob(inpath+'/org/*.jpg')
mask_files = glob.glob(inpath+'/mask/*.jpg')

orgs = []
masks = []
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
#mask = mask[:28000]
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
