import os
import argparse
import glob
import numpy as np
import cv2
import h5py
import time

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt

from keras.models import model_from_json
import keras.backend as K
from keras.utils import generic_utils
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import load_img, img_to_array, array_to_img, save_img

import modelAI

datasetpath = './output2/datasetimages.hdf5'
modelpath = './param.h5'
patch_size = 32  #分割
batch_size = 5
epoch = 200

def normalization(X):
    return X / 127.5 - 1

def load_data(datasetpath):
    with h5py.File(datasetpath, "r") as hf:
        X_full_train = hf["train_data_raw"][:].astype(np.float32)
        X_full_train = normalization(X_full_train)
        X_sketch_train = hf["train_data_gen"][:].astype(np.float32)
        X_sketch_train = normalization(X_sketch_train)
        X_full_val = hf["val_data_raw"][:].astype(np.float32)
        X_full_val = normalization(X_full_val)
        X_sketch_val = hf["val_data_gen"][:].astype(np.float32)
        X_sketch_val = normalization(X_sketch_val)
        return X_full_train, X_sketch_train, X_full_val, X_sketch_val 

def l1_loss(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true), axis=-1)

def inverse_normalization(X):
    return (X + 1.) / 2.

def to3d(X):
    if X.shape[-1]==3: return X
    b = X.transpose(3,1,2,0)
    c = np.array([b[0],b[0],b[0]])
    return c.transpose(3,1,2,0)

#ーーーーーーーーーーーーーー結果表示ーーーーーーーーーーーーー
def plot_generated_batch(X_proc, X_raw, generator_model, batch_size, suffix):
    X_gen = generator_model.predict(X_raw)
    X_raw = inverse_normalization(X_raw)
    X_proc = inverse_normalization(X_proc)
    X_gen = inverse_normalization(X_gen)

    Xs = to3d(X_raw[:5])
    Xg = to3d(X_gen[:5])
    Xr = to3d(X_proc[:5])
    Xs = np.concatenate(Xs, axis=1)
    Xg = np.concatenate(Xg, axis=1)
    Xr = np.concatenate(Xr, axis=1)
    XX = np.concatenate((Xs,Xg,Xr), axis=0)

    plt.imshow(XX)
    plt.axis('off')
    plt.savefig("current_batch_"+suffix+".png")
    plt.clf()
    plt.close()
    

def extract_patches(X, patch_size):
    list_X = []
    list_row_idx = [(i*patch_size, (i+1)*patch_size) for i in range(X.shape[1] // patch_size)]
    list_col_idx = [(i*patch_size, (i+1)*patch_size) for i in range(X.shape[2] // patch_size)]
    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            list_X.append(X[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])
    return list_X

def get_disc_batch(procImage, rawImage, generator_model, batch_counter, patch_size):
    if batch_counter % 2 == 0:
        # produce an output
        X_disc = generator_model.predict(rawImage)
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
        y_disc[:, 0] = 1
    else:
        X_disc = procImage
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)

    X_disc = extract_patches(X_disc, patch_size)
    return X_disc, y_disc

#ーーーーーーーーーーーー学習ーーーーーーーーーーーーーーーーーー
def train():
    # load data
    rawImage, procImage, rawImage_val, procImage_val = load_data(datasetpath)

    img_shape = rawImage.shape[-3:]
    patch_num = (img_shape[0] // patch_size) * (img_shape[1] // patch_size)
    disc_img_shape = (patch_size, patch_size, procImage.shape[-1])

    # train
    opt_dcgan = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    opt_discriminator = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # load generator model
    generator_model = modelAI.load_generator(img_shape, disc_img_shape)
    # load discriminator model
    discriminator_model = modelAI.load_DCGAN_discriminator(img_shape, disc_img_shape, patch_num)

    generator_model.compile(loss='mae', optimizer=opt_discriminator)
    discriminator_model.trainable = False

    DCGAN_model = modelAI.load_DCGAN(generator_model, discriminator_model, img_shape, patch_size)

    loss = [l1_loss, 'binary_crossentropy']
    loss_weights = [1E1, 1]
    DCGAN_model.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)

    discriminator_model.trainable = True
    discriminator_model.compile(loss='binary_crossentropy', optimizer=opt_discriminator)

    # start training
    print('start training')
    for e in range(epoch):

        starttime = time.time()
        perm = np.random.permutation(rawImage.shape[0])
        X_procImage = procImage[perm]
        X_rawImage  = rawImage[perm]
        X_procImageIter = [X_procImage[i:i+batch_size] for i in range(0, rawImage.shape[0], batch_size)]
        X_rawImageIter  = [X_rawImage[i:i+batch_size] for i in range(0, rawImage.shape[0], batch_size)]
        b_it = 0
        progbar = generic_utils.Progbar(len(X_procImageIter)*batch_size)
        for (X_proc_batch, X_raw_batch) in zip(X_procImageIter, X_rawImageIter):
            b_it += 1
            X_disc, y_disc = get_disc_batch(X_proc_batch, X_raw_batch, generator_model, b_it, patch_size)
            raw_disc, _ = get_disc_batch(X_raw_batch, X_raw_batch, generator_model, 1, patch_size)
            x_disc = X_disc + raw_disc
            # update the discriminator
            disc_loss = discriminator_model.train_on_batch(x_disc, y_disc)

            # create a batch to feed the generator model
            idx = np.random.choice(procImage.shape[0], batch_size)
            X_gen_target, X_gen = procImage[idx], rawImage[idx]
            y_gen = np.zeros((X_gen.shape[0], 2), dtype=np.uint8)
            y_gen[:, 1] = 1

            # Freeze the discriminator
            discriminator_model.trainable = False
            gen_loss = DCGAN_model.train_on_batch(X_gen, [X_gen_target, y_gen])
            # Unfreeze the discriminator
            discriminator_model.trainable = True

            progbar.add(batch_size, values=[
                ("D logloss", disc_loss),
                ("G tot", gen_loss[0]),
                ("G L1", gen_loss[1]),
                ("G logloss", gen_loss[2])
            ])

            # save images for visualization
            if b_it % (procImage.shape[0]//batch_size//2) == 0:
                plot_generated_batch(X_proc_batch, X_raw_batch, generator_model, batch_size, "training")
                idx = np.random.choice(procImage_val.shape[0], batch_size)
                #print(idx)
                X_gen_target, X_gen = procImage_val[idx], rawImage_val[idx]
                #print(X_gen.shape)
                plot_generated_batch(X_gen_target, X_gen, generator_model, batch_size, "validation")

        print("")
        print('Epoch %s/%s, Time: %s' % (e + 1, epoch, time.time() - starttime))
         
    generator_model.save(modelpath)

#pre_generator = train()
if __name__ == "__main__":
    train()


'''
rawImage, procImage, rawImage_val, procImage_val = load_data(datasetpath)
idx = [0]
X_gen_target, X_gen = procImage_val[idx], rawImage_val[idx]
print(X_gen)
plot_generated_batch(X_gen_target, X_gen, pre_generator, batch_size, "validation")
'''

'''
#print(pre_generator)
img = glob.glob('./out/*.jpg')
for img_file in img:
    img_col = load_img(img_file)
    imgarray = img_to_array(img_col)

imgarray_4 = normalization(imgarray_4)
imgarray_4 = np.expand_dims(imgarray, axis=0)
color = pre_generator.predict(imgarray_4) 
colored = inverse_normalization(color)
colored = to3d(colored[:5])
colored = np.concatenate(colored, axis=1)

plt.imshow(colored)
plt.axis('off')
plt.savefig("colored.png")
plt.clf()
plt.close()
'''