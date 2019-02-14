import os
import io
import time
import numpy as np
import random
import string
import cv2
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from werkzeug import secure_filename
import painting
import glob
import base64
from keras.preprocessing.image import load_img, img_to_array, array_to_img, save_img

def re_cv(img):
    height, width = img.shape[0], img.shape[1]
    hw = height/width
    if height >= 128 or width >= 128:
        height = round(128/hw)
        width = 128
        img = cv2.resize(img, (height,width), interpolation = cv2.INTER_AREA)
    return img

def re_pillow(img):
    height, width = img.height, img.width
    hw = height/width
    if height >= 128 or width >= 128:
        height = round(128/hw)
        width = 128
        img = img.resize((height,width), Image.LANCZOS)
    return img

def make_contour_image(path):
    neiborhood4 = np.array([[0, 1, 0],
                            [1, 1, 1],
                            [0, 1, 0]],
                            np.uint8)
    #線画を抽出
    #gray = cv2.cvtColor(path, cv2.IMREAD_GRAYSCALE)
    dilated = cv2.dilate(path, neiborhood4, iterations=1)
    diff = cv2.absdiff(dilated, path)
    contour = 255 - diff
    #print(contour.shape)

    bl = 235
    for i, x in enumerate(contour):
        for j, y in enumerate(x):
            if y <= bl:
                contour[i][j] = 140
            else:
                contour[i][j] = 255

    return contour

app = Flask(__name__)

UPLOAD_FOLDER = './upload/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'PNG', 'JPG'])
IMAGE_WIDTH = 640
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)

def random_string(length, seq=string.digits + string.ascii_lowercase):
    sr = random.SystemRandom()
    return ''.join([sr.choice(seq) for i in range(length)])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('main_index.html')

@app.route('/send', methods=['POST'])
def send():
    if request.method == 'POST':
        num = np.random.randint(10,25)
        name = random_string(num)    #ランダムネーム
        #name = "abcd"
        filename = UPLOAD_FOLDER + name + '.png'
        color_img_url = UPLOAD_FOLDER+"color_"+name+"_result.png"

        org_file = base64.b64decode(request.form["org"].split(",")[1])
        output = open(filename, "wb")
        output.write(org_file)
        output.close()
        #print(org.shape)
        org = Image.open(filename)
        filter = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        org = re_pillow(org)

        img_file = base64.b64decode(request.form["image"].split(",")[1])
        output = open(filename, 'wb')
        output.write(img_file)
        output.close()

        hint = Image.open(filename)
        hint = re_pillow(hint)

        filter = re_cv(filter)
        filter = make_contour_image(filter)
        filter = Image.fromarray(np.uint8(filter)).convert("RGB")

        #filter.save(UPLOAD_FOLDER+"aaaa.jpg")

        filter.paste(hint, (0,0), hint)
        org.paste(hint, (0,0), hint)
        height, width = org.height, org.width
        filter.save(color_img_url)
        #filter.save(UPLOAD_FOLDER+"aaaa.jpg")  #入力画像保存
        org.save(filename)

        imgarray = load_img(color_img_url, target_size=(128,128))
        # なにがしかの加工
        painting.coloring(imgarray, width, height, name)
        #color_img_url = os.path.join(app.config['UPLOAD_FOLDER'], 'color_'+filename)
        color_img_url = UPLOAD_FOLDER+"color_"+name+"_result.png"
        del imgarray

        return render_template('main_index.html', raw_img_url=filename, color_img_url=color_img_url)

    else:
        return redirect(url_for('main_index'))

@app.route('/upload/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.debug = True
    app.run()
