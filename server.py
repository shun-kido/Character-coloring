import os
import io
import time
import numpy as np
import random
import string
import cv2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from werkzeug import secure_filename
import painting
import glob
import base64
from keras.preprocessing.image import load_img, img_to_array, array_to_img, save_img

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
    return render_template('asad.html')

@app.route('/send', methods=['POST'])
def send():
    if request.method == 'POST':
        img_file = base64.b64decode(request.form["image"])
        num = np.random.randint(10,25)
        name = random_string(num)
        filename = UPLOAD_FOLDER + name + '.png'
        output = open(filename, 'wb')
        output.write(img_file)
        output.close()
        '''
        # 変なファイル弾き
        if img_file and allowed_file(img_file.filename):
            filename = secure_filename(img_file.filename)
            filename = filename.replace('.jpg', '').replace('.png', '')
        '''

        #else:
            #return ''' <p>許可されていない拡張子です</p> '''
        '''
        # BytesIOで読み込んでOpenCVで扱える型にする
        f = img_file.stream.read()
        bin_data = io.BytesIO(f)
        file_bytes = np.asarray(bytearray(bin_data.read()), dtype=np.uint8)
        '''
        #img = cv2.imdecode(img_file, cv2.IMREAD_COLOR)
        img = cv2.imread(filename)
        # 最大サイズは(512,512)
        height, width = img.shape[0], img.shape[1]
        hw = height/width
        if height >= 256 or width >= 256:
            height = round(256/hw)
            width = 256
            img = cv2.resize(img, (height,width), interpolation = cv2.INTER_AREA)

        # サイズだけ変えたものも保存する
        #raw_img_url = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(filename, img)

        imgarray = load_img(filename, target_size=(128,128))

        # なにがしかの加工
        painting.coloring(imgarray, height, width, name)
        #color_img_url = os.path.join(app.config['UPLOAD_FOLDER'], 'color_'+filename)
        color_img_url = UPLOAD_FOLDER+"color_"+name+"_result.jpg"
        del imgarray

        return render_template('asad.html', raw_img_url=filename, color_img_url=color_img_url)

    else:
        return redirect(url_for('asad'))

@app.route('/upload/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.debug = True
    app.run()
