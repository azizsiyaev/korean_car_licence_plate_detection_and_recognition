import csv
import os, time
import argparse
import json
import cv2
from utils.utils import get_yolo_boxes
from keras.models import load_model
import numpy as np
import itertools
from Model import get_Model
from parameter import letters
from keras import backend as K
K.set_learning_phase(0)


#-START---- Dictionary ------
Region = {"A": "서울", "B": "경기", "C": "인천", "D": "강원", "E": "충남", "F": "대전",
          "G": "충북", "H": "부산", "I": "울산", "J": "대구", "K": "경북", "L": "경남",
          "M": "전남", "N": "광주", "O": "전북", "P": "제주"}

Hangul = {"dk": "아", "dj": "어", "dh": "오", "dn": "우", "qk": "바", "qj": "버", "qh": "보", "qn": "부",
          "ek": "다", "ej": "더", "eh": "도", "en": "두", "rk": "가", "rj": "거", "rh": "고", "rn": "구",
          "wk": "자", "wj": "저", "wh": "조", "wn": "주", "ak": "마", "aj": "머", "ah": "모", "an": "무",
          "sk": "나", "sj": "너", "sh": "노", "sn": "누", "fk": "라", "fj": "러", "fh": "로", "fn": "루",
          "tk": "사", "tj": "서", "th": "소", "tn": "수", "gj": "허", "qo": "배", "gk": "하", "gh": "호"}
#-END---- Dictionary ------



def decode_label(out):
    out_best = list(np.argmax(out[0, 2:], axis=1))
    out_best = [k for k, g in itertools.groupby(out_best)]
    outstr = ''
    for i in out_best:
        if i < len(letters):
            outstr += letters[i]
    return outstr


def label_to_hangul(label, img):
    if len(label) == 8:
        try:
            hangul = Hangul[label[2:4]]
            return label[:2] + hangul + label[4:]
        except:
            print("Could not translate " + label)
            return label
    elif len(label) == 9:
        try:
            region = Region[label[0]]
            hangul = Hangul[label[3:5]]
            return region + label[1:3] + hangul + label[5:]
        except:
            print("Could not translate " + label)
            return '10나2569'
    else:
        return '10나2569'
        
def get_label(img, model):
    pred_text = None
    img_pred = img.astype(np.float32)
    img_pred = cv2.resize(img_pred, (128, 64))
    img_pred = (img_pred / 255.0) * 2.0 - 1.0
    img_pred = img_pred.T
    img_pred = np.expand_dims(img_pred, axis=-1)
    img_pred = np.expand_dims(img_pred, axis=0)
    net_out_value = model.predict(img_pred)
    pred_text = decode_label(net_out_value)
    if pred_text[0] == 'Z':
        pred_text = pred_text[1:]
    pred_text = label_to_hangul(pred_text, img)    
    return pred_text

def write_result(output_path, dataset_name, filename, label, box): 
    with open(output_path + dataset_name + '.csv', 'a', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow([filename, label, box.xmin, box.ymin, box.xmax, box.ymax])

def _main_(args):
    
    # ----- Initialization -----------
    input_path   = args.input
    output_path  = args.output
    dataset_name = args.dataset
    trial_version = args.trial_version

    net_h, net_w = 416, 416
    #obj_thresh, nms_thresh = 0.7, 0.7
    obj_thresh, nms_thresh = 0.7, 0.7

    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
    
    if trial_version!="WITH_TRAIN" and trial_version!="WITH_TEST":
        print("Wrong Trial version parameter")
        return
    
    #-START---- Model Loading------
    if dataset_name == "parking":
        try:    
            infer_model = load_model('W/' + trial_version + '/parking_detection_' + trial_version + '.h5')
            anchors = [9,7, 10,4, 16,7, 17,15, 24,10, 30,14, 57,41, 71,26, 99,62]
            print("Parking detection weights loaded")
        except:
            print("Could not load parking detection weights")
            return
    elif dataset_name == "cctv":
        try:    
            infer_model = load_model('W/'  + trial_version + '/cctv_detection_' + trial_version + '.h5')
            #anchors = [4,2, 8,7, 9,4, 13,5, 14,11, 19,16, 19,7, 24,10, 29,13]
            anchors = [8,4, 9,8, 13,5, 15,13, 17,7, 20,17, 22,8, 26,10, 29,14]
             
            print("CCTV detection weights loaded")
        except:
            print("Could not load CCTV detection weights")
            return
    else:
        print("Please specify dataset")
        return
        
        
    model = get_Model(training=False)
    if dataset_name == "parking":
        try:
            model.load_weights('W/'  + trial_version + '/parking_recognition_' + trial_version + '.hdf5')
            print("Parking recognition weights loaded")
        except:
            raise Exception("No parking recognition weight file!")
            return
    elif dataset_name == "cctv":
        try:
            model.load_weights('W/' + trial_version + '/cctv_recognition_' + trial_version + '.hdf5')
            print("CCTV recognition weights loaded")
        except:
            raise Exception("No cctv recognition weight file!")
            return
            
    #-END---- Model Loading------
 
    image_paths = []
    
    for fold in os.listdir(input_path):
        for fname in os.listdir(os.path.join(input_path, fold)):
            if fname.endswith('.jpg') or fname.endswith('.png'):
                image_paths.append(os.path.join(input_path, fold, fname))
    image_paths = sorted(image_paths)
    
    
    
    print("Number of images to test: " + str(len(image_paths)))
    # ----- Initialization -----------
    
    # -START----- Main loop -------------
    total_time = 0
    for i, image_path in enumerate(image_paths):
        #if i == 10:
        #    break
        print('[{}] / [{}]'.format(i + 1, len(image_paths)))
        image = cv2.imread(image_path)
        tic = time.time()
        boxes = get_yolo_boxes(infer_model, [image], net_h, net_w, anchors, obj_thresh, nms_thresh)[0]
        
        if len(boxes)!=False:
            box = boxes[0]
        else:
            print("Box not found! " + image_path)
            continue
        if box.ymin < 0 or box.ymax < 0 or box.xmin < 0 or box.xmax < 0:
            print("Negative box found. Cannot proceed! " + image_path)
            continue

        label = get_label(image[box.ymin:box.ymax, box.xmin:box.xmax, 0], model)
        toc = time.time()
        total_time += toc - tic
        write_result(output_path, dataset_name, image_path, label, box)
    # -END----- Main loop -------------
    
       
    print('Avg. PT: {} ms.'.format(total_time / len(image_paths) * 1000))


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='License Plate Detection and Recognition')
    argparser.add_argument('-i', '--input', help='path to an images directory')    
    argparser.add_argument('-o', '--output', help='path to output directory')
    argparser.add_argument('-d', '--dataset', help='path to output directory')
    argparser.add_argument('-t', '--trial_version', help='testing with only trained (WITH_TRAIN) or with trained & test sets (WITH_TEST)')
    args = argparser.parse_args()
    _main_(args)
