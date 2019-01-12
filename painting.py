import glob
import numpy as np
import cv2
from PIL import Image
import matplotlib.pylab as plt
import h5py
#import keras.backend as K
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array, array_to_img, save_img
import tensorflow as tf
import test

generator_model = load_model('param.h5')
graph = tf.get_default_graph()

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

def coloring(path, height, width, name):
    outpath = './raisr/test/'
    
   #generator_model = load_model('param.h5')
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
    
    global graph
    with graph.as_default():
        color = generator_model.predict(imgarray)
        
    #color = inverse_normalization(color)
    color = to3d(color[:5])
    #print(colored.shape)
    color = np.concatenate(color, axis=1)
    #print(colored)
    colored = rgb(color)

    #print(colored)
    
    pil_img_f = Image.fromarray(np.uint8(colored))
    
    if height >= 256 or width >= 256:
        pil_img_f =pil_img_f.resize((round(height/2), round(width/2)),)
        pil_img_f.save(outpath+'color_'+name+'.jpg',)
        re_colored = test.re_colored(outpath+'color_'+name+'.jpg')
    else:
        pil_img_f =pil_img_f.resize((height, width),)
        pil_img_f.save('upload/color_'+name+'_result.jpg',)
        
        img_url = 'upload/color_'+name+'_result.jpg'
    #pil_img_f.save(outpath+'color_'+name+'.jpg',)
    #print(outpath+'color_'+name+'.jpg')
    #re_colored = test.re_colored(outpath+'color_'+name+'.jpg')
    
    #return img_url
    
if __name__== '__main__':
    img = glob.glob('./upload/*.jpg')
    
    for img_file in img:
        img_col = load_img(img_file)
        width, height = img_col.size
        #print(width, height)
        imgarray = load_img(img_file, target_size=(128,128))
    img = coloring(imgarray, height, width, "a")
'''
plt.imshow(img)
plt.axis('off')
plt.savefig("colored.png")
plt.clf()
plt.close()
'''