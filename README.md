# Facial-Emotion-Gender-and-Age-Prediction-
Humans have the innate ability to distinguish faces, facial emotions, gender and to estimate age. Today because of the availability of GPU’s, hard disk and technologies we can make computers do a lot of work that humans do. This opens tons of applications. Face, emotion, gender and age recognition are often helpful in improving the security, criminal identification and in several facial applications like Snapchat and Instagram. This project aims at identifying the gender, age and emotion of a person given in the image. This work is done as part of Deep Learning course.
The dataset for emotion is taken from FER challenge Kaggle and is used for training emotion. The dataset for used to train for gender and age detection is same and is UTK face dataset. The predict file is used to predict the gender age and emotion of a given emotion by combining the 3 outputs. The model folders inside code folder contains various models generated after training for emotion, gender and age. Notebooks emotion, gender_age are used for training the model and to save them for emotion, age and gender respectively. The Project write up contains the overall information about the project. 
To predict an image, run the predict file from command line using:
python predict.py --image s3.jpg (image location) --model_emotion Emotion_model(name of model) --model_age Age_model (name of model) --model_gender Gender_model (name of model)

Sample: predict.py --image C:/Users/nihal/Desktop/DL_2/Sample-Images/niha.jpg --model_emotion Emotion_model --model_age Age_model --model_gender Gender_model
Download  UTK Face (Aligned & Cropped Faces) dataset from : https://susanqq.github.io/UTKFace/
Download the FER challenge dataset from:
https://www.kaggle.com/ashishpatel26/facial-expression-recognitionferchallenge
