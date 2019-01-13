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
* CRNN (https://github.com/qjadud1994/CRNN-Keras)


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



