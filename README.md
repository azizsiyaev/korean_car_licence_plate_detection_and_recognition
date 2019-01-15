# Korean car licence plate detection and recognition
Korean cars licence plate detection and recognition using Keras.

Licence plate detection and Recognition is made by using Keras framework. 
You may find 3 folders that consist of separate implementation of 
* Detection part using Yolov3, 
* Recognition part implemented with CRNN 
* Program that combines both techniques

In addition, you can find synthetic Data generator that creates Plates, immitated CCTV and Parking images

I used ideas from other github accounts, combined them to solve Licence Plate Detection and Recognition Problem.
* YoloV3 (https://github.com/experiencor/keras-yolo3)
* CRNN (https://github.com/qjadud1994/CRNN-Keras) - Thank you for your help! 


## Main Logical Flow
1. YoloV3 finds 2 points that indicate plate location in the picture 
2. We pass an array of cropped plate to Recognition model
3. CRNN finds label to the given image

![alt text](https://github.com/azizsiyaev/korean_car_licence_plate_detection_and_recognition/blob/master/Readme%20pics/model.png)

I have attached YoloV3 and CRNN papers to this repository. I suggest you to read them before you start


## Data Generator
CRNN requires a lot of training data. For that reason I made a plate generator and using it created 500k plate images. 
Train my model on synthetic data first and applied fine-tunning techniques with real data.


**Generated recognition data**
![alt text](https://github.com/azizsiyaev/korean_car_licence_plate_detection_and_recognition/blob/master/Readme%20pics/generated%20plates.png)


**Generated parking data** 
![alt text](https://github.com/azizsiyaev/korean_car_licence_plate_detection_and_recognition/blob/master/Readme%20pics/generated%20parking%20cars.png)


**Generated cctv data** 
![alt text](https://github.com/azizsiyaev/korean_car_licence_plate_detection_and_recognition/blob/master/Readme%20pics/generated%20cctv%20cars.png)

## Training Detection Part

Detection of Parking and CCTV images are different tasks, since you are dealing with different scales. I trained them separately. 

1. Prepare data. Create images folder and folder with annotations (PASCAL format)
2. Compute anchors.
`python gen_anchors.py -c config.json`
3. Write data path, anchors to config.json file
4. Train
`python train.py -c config.json`
5. Evaluate
`python evaluate.py -c config.json`


## Train Recognition Part

CRNN requires a lot of training data. For that reason I made a plate generator and using it created 500k plate images. 
Train my model on synthetic data first and applied fine-tunning techniques with real data. 
So generaly speaking, I made pretrained model and on top of that trained with real images.

![alt text](https://github.com/azizsiyaev/korean_car_licence_plate_detection_and_recognition/blob/master/Readme%20pics/fine-tunning%20.png)

CCTV and Parking data was trained separately. 


|       File         |Description                                       |
|--------------------|--------------------------------------------------|
|Model .py           |Network using CNN (VGG) + Bidirectional LSTM      |
|Model_GRU. py       |Network using CNN (VGG) + Bidirectional GRU       |
|Image_Generator. py |Image batch generator for training                |
|parameter. py       |Parameters used in CRNN                           |
|training. py        |CRNN training                                     |
|Prediction. py      |CRNN prediction                                   |

## Conclusion and Suggestions

Provieed techniques work fine. But here are some things to note:

1. YoloV3 doesn't like small objects. So, CCTV detection accuracy was relatively bad in comparison with Parking. My suggestion is while working with CCTV, **detect car first and than run plate detection**
2. Models don't really like synthetic data. So **don't overfit your model with generated data, but remember about real test data.** 


Hope this guide will be helpful. Have fun!
