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
OUTPUT_FOLDER_PNG = './output/png'
OUTPUT_FOLDER_SVG = './output/svg'
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
    learn = load_learner(path='.', file="resnet_18_non-pre_trained.pkl")
    # image_path = Path(image_path)

    img = open_image(image_path)
    # alphabets = set_tag()
    alphabets = ['്', 'ാ', 'ി', 'ീ', 'ു', 'ൂ', 'െ', 'ൃ', 'െ', 'ൌ', 'ം', 'അ', 'ആ', 'ഇ', 'ഉ', 'ഋ', 'എ', 'ഏ', 'ഒ', 'ക', 'ഖ', 'ഗ', 'ഘ', 'ങ', 'ച', 'ഛ', 'ജ', 'ഝ', 'ഞ', 'ട', 'ഠ', 'ഡ', 'ഢ', 'ണ', 'ത', 'ഫ', 'ദ', 'ധ', 'ന', 'പ', 'ഫ', 'ബ', 'ഭ', 'മ', 'യ', 'ര',
                 'റ', 'ല', 'ള', 'ഴ', 'വ', 'ശ', 'ഷ', 'സ', 'ഹ', 'ൺ', 'ൻ', 'ർ', 'ൽ', 'ൾ', 'ക്ക', 'ക്ഷ', 'ങ്ക', 'ങ്ങ', 'ച്ച', 'ഞ്ച', 'ഞ്ഞ', 'ട്ട', 'ണ്ട', 'ണ്ണ', 'ത്ത', 'ദ്ധ', 'ന്ത', 'ന്ദ', 'ന്ന', 'പ്പ', 'മ്പ', 'മ്മ', 'യ്യ', 'ല്ല', 'ള്ള', 'വ്വ', '്യ', '്ര', '്വ']
    pred_class, pred_idx, outputs = learn.predict(img)
    index = int(str(pred_class)) - 1
    print(alphabets[index], file=sys.stdout)
    return alphabets[index]

def _edge_detect(im):
    return np.max(np.array([_sobel_detect(im[:,:, 0]),_sobel_detect(im[:,:, 1]),_sobel_detect(im[:,:, 2])]), axis=0)


def _sobel_detect(channel): # sobel edge detection
    sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
    sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
    sobel = np.hypot(sobelX, sobelY)
    sobel[sobel > 255] = 255
    return np.uint8(sobel)

def sort_words(boxes):
    for i in range(1, len(boxes)):
        key = boxes[i]
        j = i - 1
        while(j >= 0 and key[2] < boxes[j][2]):
            boxes[j+1] = boxes[j]
            j -= 1
        boxes[j+1] = key
    return boxes

def sort_boxes(boxes):
    lines = []
    new_lines = []
    tmp_box = boxes[0]
    lines.append(tmp_box)
    for box in boxes[1:]:
        if((box[0] + (box[1] - box[0])/2) < tmp_box[1]):
            lines.append(box)
            tmp_box = box
        else:
            new_lines.append(sort_words(lines))
            lines = []
            tmp_box = box
            lines.append(box)
    new_lines.append(sort_words(lines))
    return(new_lines)

def sort_labels(label_boxes):
    for i in range(1, len(label_boxes)):
        key = label_boxes[i]
        j = i - 1
        while(j >= 0 and key[1][2] < label_boxes[j][1][2]):
            label_boxes[j+1] = label_boxes[j]
            j -= 1
        label_boxes[j+1] = key
    return label_boxes

def scale_boxes(new_img, old_img, coords):
    coords[0] = int((coords[0]) * new_img.shape[0] / old_img.shape[0]); # new top left
    coords[1] = int((coords[1]) * new_img.shape[1] / old_img.shape[1] ); # new bottom left
    coords[2] = int((coords[2] + 1) * new_img.shape[0] / old_img.shape[0]) - 1; # new top right
    coords[3] = int((coords[3] + 1) * new_img.shape[1] / old_img.shape[1] ) - 1; # new bottom right
    return coords

def clipping_image(new):
    colsums = np.sum(new, axis=0)
    linessum = np.sum(new, axis=1)
    colsums2 = np.nonzero(0-colsums)
    linessum2 = np.nonzero(0-linessum)

    x = linessum2[0][0] # top left
    xh = linessum2[0][linessum2[0].shape[0]-1] # bottom left
    y = colsums2[0][0] # top right
    yw = colsums2[0][colsums2[0].shape[0]-1] # bottom right

    imgcrop = new[x:xh, y:yw] # crop the image

    return imgcrop, [x, xh, y, yw]


def padding_resizing_image(img):
    img = cv2.copyMakeBorder(img, 2, 2, 0, 0, cv2.BORDER_CONSTANT) # add 2px padding to image
    try:
        img = cv2.resize(np.uint8(img), (32, 32)) # resize the image to 32*32
    except:
        return img
    finally:
        return img

def segmentation(img, sequence, origimg=None, wordNo=None, filename=None):
    if(sequence == "word"): # resize to find the words
        width = 640
        height = int(img.shape[0] * (width / img.shape[1]))
        sigma = 18
    elif(sequence == "character"): # resize to find the characters
        width = img.shape[1] # 1280
        height = img.shape[0] # int(img.shape[0] * (width / img.shape[1]))
        sigma = 0

    img = cv2.resize(img, (width, height))
    blurred = cv2.GaussianBlur(img, (5, 5), sigma) # apply gaussian blur

    if(sequence == "word"):
        blurred = _edge_detect(blurred) # edge detect in blurred image (words)
        ret, img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # Otsu's thresholding with Binary
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((15,15), np.uint8)) # Morphological processing - Black&White

    elif(sequence == "character"):
        ret, img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) # Otsu's thresholding with Binary Inverted

    num_labels, labels_im = cv2.connectedComponents(img) # find the connected components

    if(sequence == "word"):
        boxes = [] # for storing the coordinates of the bounding boxes
        for i in range(1, num_labels):
            new, nr_objects = ndimage.label(labels_im == i) # select the images with label
            new, new_coord = clipping_image(new) # clipping the image to the edges
            if(not(new.shape[0] < 10 or new.shape[1] < 10)):
                boxes.append(new_coord)

    if(sequence == "character"):
        boxes = []
        label_box = []
        for i in range(1, num_labels):
            new, nr_objects = ndimage.label(labels_im == i) # select the images with label
            new, new_coord = clipping_image(new) # clipping the image to the edges
            if(not(new.shape[0] < 10 or new.shape[1] < 10)):
                label_box.append([i, new_coord])
        label_box = sort_labels(label_box) # sort the words
        chNo = 0 
        for box in label_box:
            ch_img, nr_objects = ndimage.label(labels_im == box[0])
            ch_img, new_coord = clipping_image(ch_img)
            cropped_image = padding_resizing_image(ch_img)
            try:
                dst = os.path.join(OUTPUT_FOLDER_PNG, filename, str(wordNo)+"_"+str(chNo)+".png")
                plt.imsave(dst, cropped_image, cmap=cm.gray)
            except:
                pass
            finally:
                pass
            chNo += 1
    return img, boxes

def img_to_seg(img, filename):
    img = cv2.resize(img, (1280, int(img.shape[0] * (1280 / img.shape[1])))) # change image width to 1280px
    kernel = np.ones((5,5),np.uint8) # kernel for opening
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) # opening image to remove minor noise
    new_img, boxes = segmentation(img, "word") # line segmentation
    boxes.sort()
    boxes = sort_boxes(boxes)
    scaled_boxes = [] # coordinates values scaled to img
    for box in boxes:
        for box in box:
            box = scale_boxes(img, new_img, box)
            scaled_boxes.append(box)
    wordNo = 0
    for scaled_box in scaled_boxes:
        img_gray = cv2.cvtColor(img[scaled_box[0]:scaled_box[1], scaled_box[2]:scaled_box[3]], cv2.COLOR_BGR2GRAY)
        img_new, _ = segmentation(img_gray, "character", None, wordNo, filename.split(".")[0])
        wordNo += 1
    return "1", wordNo

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
            os.mkdir(os.path.join(OUTPUT_FOLDER_PNG, filename.split(".")[0]))
            os.mkdir(os.path.join(OUTPUT_FOLDER_SVG, filename.split(".")[0]))
            img = cv2.cvtColor(cv2.imread(os.path.join(UPLOAD_FOLDER, filename)), cv2.COLOR_BGR2RGB) # input image
            x, wordNo = img_to_seg(img, filename)

    if x == "1":
        alpha = {} # dictionary with predicted alphabet and image name
        for i in range(wordNo):
            alpha[str(i)] = {}
        for f in os.listdir(os.path.join(OUTPUT_FOLDER_PNG, filename.split(".")[0])):
            alpha[str(f.split("_")[0])][str(f.split("_")[1].split(".")[0])] = str(predict_alphabets(os.path.join(
                OUTPUT_FOLDER_PNG, filename.split(".")[0], f)))
            subprocess.call(["convert", os.path.join(OUTPUT_FOLDER_PNG, filename.split(".")[0], f), "-negate", os.path.join(OUTPUT_FOLDER_SVG, filename.split(".")[0], f.split(".")[0]+".svg")])
        print(alpha)
        for word in range(0, len(alpha)):
            string = ""
            for ch in range(len(alpha[str(word)])):
                string += alpha[str(word)][str(ch)]
            print(string)
    return "completed successfully"


if __name__ == "__main__":
    app.run(debug=True)
