import os
import io
import time
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from werkzeug import secure_filename
import painting
import glob
from keras.preprocessing.image import load_img, img_to_array, array_to_img, save_img

app = Flask(__name__)

UPLOAD_FOLDER = './upload'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'PNG', 'JPG'])
IMAGE_WIDTH = 640
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send', methods=['GET', 'POST'])
def send():
    if request.method == 'POST':
        img_file = request.files['img_file']

        # 変なファイル弾き
        if img_file and allowed_file(img_file.filename):
            filename = secure_filename(img_file.filename)
            filename = filename.replace('.jpg', '').replace('.png', '')
        else:
            return ''' <p>許可されていない拡張子です</p> '''

        # BytesIOで読み込んでOpenCVで扱える型にする
        f = img_file.stream.read()
        bin_data = io.BytesIO(f)
        file_bytes = np.asarray(bytearray(bin_data.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # 最大サイズは(512,512)
        height, width = img.shape[0], img.shape[1]
        hw = height/width
        if height >= 512 or width >= 512:
            height = round(512/hw)
            width = 512
            img = cv2.resize(img, (height,width), interpolation = cv2.INTER_AREA)
        
        # サイズだけ変えたものも保存する
        raw_img_url = os.path.join(app.config['UPLOAD_FOLDER'], 'raw_'+filename)
        cv2.imwrite(raw_img_url+'.jpg', img)
        
        imgarray = load_img(raw_img_url+'.jpg', target_size=(256,256))

        # なにがしかの加工
        painting.coloring(imgarray, height, width, filename)
        color_img_url = os.path.join(app.config['UPLOAD_FOLDER'], 'color_'+filename)
        
        del imgarray
        
        return render_template('index.html', raw_img_url=raw_img_url+'.jpg', color_img_url=color_img_url+'_result.jpg')

    else:
        return redirect(url_for('index'))

@app.route('/upload/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.debug = True
    app.run()