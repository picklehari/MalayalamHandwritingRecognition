import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm as cm
import urllib
from scipy import ndimage
import csv
import sys
from fastai import *
from fastai.vision import *
from fastai.metrics import accuracy
import png2svg

UPLOAD_FOLDER = './uploads'
OUTPUT_FOLDER = './output'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def set_tag():

    alphabet = "്  ാ  ി  ീ  ു  ൂ  െ ൃ  െ  ൌ  ം അ ആ ഇ ഉ ഋ എ ഏ ഒ ക ഖ ഗ ഘ ങ ച ഛ ജ ഝ ഞ ട ഠ ഢ ഡ ണ ത ഫ ദ ധ ന പ ഫ ബ ഭ മ യ ര റ ല ള ഴ വ ശ ഷ സ ഹ ൺ ൻ ർ ൽ ൾ ക്ക ക്ഷ ങ്ക ങ്ങ ച്ച ഞ്ച ഞ്ഞ ട്ട ണ്ട ണ്ണ ത്ത ദ്ധ ന്ത ന്ദ ന്ന പ്പ മ്പ മ്മ യ്യ ല്ല ള്ള  ്യ   ്ര  ്വ"
    alphabet = alphabet.split(" ")
    alphabets = []
    for x in alphabet:
        if x != '':
            alphabets.append(x)
    return alphabets


def predict_alphabets(image_path):
    print(image_path, file=sys.stdout)
    defaults.device = torch.device("cpu")
    learn = load_learner(path='.', file="MalHand_18.pkl")
    # image_path = Path(image_path)

    img = open_image(image_path)
    # alphabets = set_tag()
    alphabets = ['്', 'ാ', 'ി', 'ീ', 'ു', 'ൂ', 'െ', 'ൃ', 'െ', 'ൌ', 'ം', 'അ', 'ആ', 'ഇ', 'ഉ', 'ഋ', 'എ', 'ഏ', 'ഒ', 'ക', 'ഖ', 'ഗ', 'ഘ', 'ങ', 'ച', 'ഛ', 'ജ', 'ഝ', 'ഞ', 'ട', 'ഠ', 'ഢ', 'ഡ', 'ണ', 'ത', 'ഫ', 'ദ', 'ധ', 'ന', 'പ', 'ഫ', 'ബ', 'ഭ', 'മ', 'യ', 'ര',
                 'റ', 'ല', 'ള', 'ഴ', 'വ', 'ശ', 'ഷ', 'സ', 'ഹ', 'ൺ', 'ൻ', 'ർ', 'ൽ', 'ൾ', 'ക്ക', 'ക്ഷ', 'ങ്ക', 'ങ്ങ', 'ച്ച', 'ഞ്ച', 'ഞ്ഞ', 'ട്ട', 'ണ്ട', 'ണ്ണ', 'ത്ത', 'ദ്ധ', 'ന്ത', 'ന്ദ', 'ന്ന', 'പ്പ', 'മ്പ', 'മ്മ', 'യ്യ', 'ല്ല', 'ള്ള', 'വ്വ', '്യ', '്ര', '്വ']
    pred_class, pred_idx, outputs = learn.predict(img)
    index = int(str(pred_class)) - 1
    print(alphabets[index], file=sys.stdout)
    return alphabets[index]


def clipping_image(new):
    colsums = np.sum(new, axis=0)
    linessum = np.sum(new, axis=1)
    colsums2 = np.nonzero(0-colsums)
    linessum2 = np.nonzero(0-linessum)

    xx = linessum2[0][0]
    yy = linessum2[0][linessum2[0].shape[0]-1]
    ww = colsums2[0][0]
    hh = colsums2[0][colsums2[0].shape[0]-1]

    imgcrop = new[xx:yy, ww:hh]

    return imgcrop


def padding_resizing_image(img):
    img = cv2.copyMakeBorder(img, 2, 2, 0, 0, cv2.BORDER_CONSTANT)
    try:
        img = cv2.resize(np.uint8(img), (32, 32))
    except:
        return img
    finally:
        return img


def segmentation(filename):
    img = cv2.imread(os.path.join(
        UPLOAD_FOLDER, filename), 0)
    
    width = 1280
    height = int(img.shape[0] * (1280 / img.shape[1]))
    img = cv2.resize(img, (width, height))
    
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    num_labels, labels_im = cv2.connectedComponents(thresh)

    for i in range(1, num_labels):
        new, nr_objects = ndimage.label(labels_im == i)
        dst = os.path.join(OUTPUT_FOLDER, str(i)+".png")
        new = clipping_image(new)
        new = padding_resizing_image(new)
        try:
            plt.imsave(dst, new, cmap=cm.gray)
            print(png2svg.png_to_svg(dst))
        except:
            print("some error", file=sys.stdout)
        finally:
            print("some error", file=sys.stdout)
        print("completed", file=sys.stdout)
    return "1"


@app.route('/')
def index():
    return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(filename, file=sys.stdout)
            x = segmentation(filename)
    if x == "1":
        alpha = {}
        for f in os.listdir(OUTPUT_FOLDER):
            alpha[str(f)]=str(predict_alphabets(os.path.join(
                OUTPUT_FOLDER, f)))
        print(alpha)
    return "completed successfully"


if __name__ == "__main__":
    app.run(debug=True)
