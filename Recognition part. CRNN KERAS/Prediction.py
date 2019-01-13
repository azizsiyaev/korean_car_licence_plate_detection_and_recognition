import cv2
import itertools, os, time
import numpy as np
from Model import get_Model
from parameter import letters
import argparse
from keras import backend as K
K.set_learning_phase(0)

Region = {"A": "서울 ", "B": "경기 ", "C": "인천 ", "D": "강원 ", "E": "충남 ", "F": "대전 ",
          "G": "충북 ", "H": "부산 ", "I": "울산 ", "J": "대구 ", "K": "경북 ", "L": "경남 ",
          "M": "전남 ", "N": "광주 ", "O": "전북 ", "P": "제주 "}

#Hangul = {"ah": "모",  "aj": "머", "ak": "마", "al":"미", "an": "무", "ck":"차", "dh": "오",
#          "dj": "어", "dk": "아", "dl":"이", "dn": "우", "eh": "도", "ej": "더", "ek": "다",
#          "el":"디", "en": "두", "fh": "로", "fj": "러", "fk": "라", "fn": "루", "gj": "허",
#          "gl":"히", "gn":"후",  "qh": "보", "qj": "버", "qk": "바", "qn": "부", "qo":"배", 
#          "rh": "고", "rj": "거", "rk": "가", "rn": "구", "sh": "노", "sj": "너","sk": "나", 
#          "sl":"니", "sm":"느", "sn": "누", "th": "소", "tj": "서", "tk": "사", "tl":"시", 
#           "tn": "수", "vk":"파", "wh": "조", "wj": "저", "wk": "자", "wl":"지", "wn": "주", 
#          "xn":"투", "zj":"커",                "gk":"하", "fl":"리", "gh":"호", "em":"드", "vj":"경", "vn":"국" }

Hangul = {"dk": "아", "dj": "어", "dh": "오", "dn": "우", "qk": "바", "qj": "버", "qh": "보", "qn": "부",
          "ek": "다", "ej": "더", "eh": "도", "en": "두", "rk": "가", "rj": "거", "rh": "고", "rn": "구",
          "wk": "자", "wj": "저", "wh": "조", "wn": "주", "ak": "마", "aj": "머", "ah": "모", "an": "무",
          "sk": "나", "sj": "너", "sh": "노", "sn": "누", "fk": "라", "fj": "러", "fh": "로", "fn": "루",
          "tk": "사", "tj": "서", "th": "소", "tn": "수", "gj": "허", "qo": "배", "gk": "하", "gh": "호"}


def decode_label(out):
    # out : (1, 32, 42)
    out_best = list(np.argmax(out[0, 2:], axis=1))  # get max index -> len = 32
    out_best = [k for k, g in itertools.groupby(out_best)]  # remove overlap value
    #print(out_best)
    outstr = ''
    for i in out_best:
        if i < len(letters):
            #print("->" + letters[i])
            outstr += letters[i]
    return outstr


def label_to_hangul(label):  # eng -> hangul
    region = label[0]
    two_num = label[1:3]
    hangul = label[3:5]
    four_num = label[5:]

    try:
        region = Region[region] if region != 'Z' else ''
    except:
        pass
    try:
        hangul = Hangul[hangul]
    except:
        pass
    return region + two_num + hangul + four_num

parser = argparse.ArgumentParser()
parser.add_argument("-w", "--weight", help="weight file directory",
                    type=str, default="working_model.hdf5")
parser.add_argument("-t", "--test_img", help="Test image directory",
                    type=str, default="./DB/parking_test/")
args = parser.parse_args()

# Get CRNN model
model = get_Model(training=False)

try:
    model.load_weights(args.weight)
    print("...Previous weight data...")
except:
    raise Exception("No weight file!")


test_dir =args.test_img
test_imgs = os.listdir(args.test_img)
total = 0
acc = 0
letter_total = 0
letter_acc = 0
start = time.time()
for test_img in test_imgs:
    if test_img[-3:]!= 'jpg':
        continue
    #print(test_img)
    img = cv2.imread(test_dir + test_img, cv2.IMREAD_GRAYSCALE)

    img_pred = img.astype(np.float32)
    img_pred = cv2.resize(img_pred, (128, 64))
    img_pred = (img_pred / 255.0) * 2.0 - 1.0
    img_pred = img_pred.T
    img_pred = np.expand_dims(img_pred, axis=-1)
    img_pred = np.expand_dims(img_pred, axis=0)
    
    #print(img_pred.shape)
    # specify batch size = 1 for evaluation
    net_out_value = model.predict(img_pred)
    #print(net_out_value)
    pred_texts = decode_label(net_out_value)
    
    if pred_texts[0] == 'Z':
        pred_texts = pred_texts[1:]

    for i in range(min(len(pred_texts), len(test_img[0:-9]))):
        #ti = test_img[i][:-9]
        if pred_texts[i] == test_img[:-9][i]:
            letter_acc += 1
    letter_total += max(len(pred_texts), len(test_img[0:-9]))
    
    
    if pred_texts == test_img[0:-9]:
        acc += 1
    total += 1
    print("Predicted text: " + pred_texts)
    print("True text: " + test_img[:-9])
    #print('Predicted: %s  /  True: %s' % (label_to_hangul(pred_texts), label_to_hangul(test_img[0:-4])))
    
    # cv2.rectangle(img, (0,0), (150, 30), (0,0,0), -1)
    # cv2.putText(img, pred_texts, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),2)

    #cv2.imshow("q", img)
    #if cv2.waitKey(0) == 27:
    #   break
    #cv2.destroyAllWindows()

end = time.time()
total_time = (end - start)
print("Time : ",total_time / total)
print("ACC : ", acc / total)
print("letter ACC : ", letter_acc / letter_total)
