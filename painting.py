import glob
import numpy as np
import cv2
from PIL import Image
import matplotlib.pylab as plt
import h5py
#import keras.backend as K
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array, array_to_img, save_img

def normalization(X):
    return X / 127.5 - 1

def rgb(X):
    return (X + 1) * 127.5

def to3d(X):
    if X.shape[-1]==3: return X
    b = X.transpose(3,1,2,0)
    c = np.array([b[0],b[0],b[0]])
    return c.transpose(3,1,2,0)

def inverse_normalization(X):
    return (X + 1.) / 2.

def coloring(path, height, width):
    outpath = './uploads/'
    
    generator_model = load_model('param.h5')
    '''
    img = glob.glob('./uploads/*.jpg')
    
    for img_file in img:
        img_col = load_img(img_file)
        height, width = img_col.size
        imgarray = load_img(img_file, target_size=(128,128))
    '''
    imgarray = img_to_array(path)
    imgarray = normalization(imgarray)
    imgarray = np.expand_dims(imgarray, axis=0)

    color = generator_model.predict(imgarray) 
    #color = inverse_normalization(color)
    colored = to3d(color[:5])
    #print(colored.shape)
    colored = np.concatenate(colored, axis=1)
    #print(colored)
    colored = rgb(colored)

    #print(colored)

    pil_img_f = Image.fromarray(np.uint8(colored))
    pil_img_f =pil_img_f.resize((width, height),)
    pil_img_f.save(outpath+'colored.png',)
    
    return outpath+'colored.png'
    
if __name__== '__main__':
    img = glob.glob('./out/*.jpg')
    
    for img_file in img:
        img_col = load_img(img_file)
        height, width = img_col.size
        imgarray = load_img(img_file, target_size=(128,128))
    img = coloring(imgarray, height, width)
'''
plt.imshow(colored)
plt.axis('off')
plt.savefig("colored.png")
plt.clf()
plt.close()
'''