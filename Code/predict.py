import numpy as np
import cv2
import argparse
import tensorflow.keras


def convert_to_array(img):
    img = cv2.imread(img)
    resized_image = np.resize(img,(1,48, 48, 1))
    return resized_image

def convert_to_array2(img):
    img = cv2.imread(img)
    resized_image = np.resize(img,(1,48, 48, 3))
    return resized_image



def get_emotion(label):
    if label == 0:
        return "anger"
    if label == 1:
        return "disgust"
    if label == 2:
        return "fear"
    if label == 3:
        return "happiness"
    if label == 4:
        return "sadness"
    if label == 5:
        return "surprise"
    if label == 6:
        return "neutral"

def get_age(label):
    if label == 0:
        return "Age < 18"
    if label == 1:
        return "Age (18-35)"
    if label == 2:
        return "Age (35-50)"
    if label == 3:
        return "Age (50-65)"
    if label == 4:
        return "Age > 65"
     
def get_gender(label):
    if label == 0:
        return "Male"
    if label == 1:
        return "Female"

def predict_emotion(file, model):
    print("Predicting Emotion.................................")
    ar = convert_to_array(file)
    ar = ar.astype("float") / 255.
    score = model.predict(ar, verbose=1)
    label_index = np.argmax(score)
    emotion = get_emotion(label_index)
    print("The predicted emotion is a " + emotion)

def predict_age(file, model):
    print("Predicting Age.................................")
    ar = convert_to_array2(file)
    ar = ar.astype("float") / 255.
    score = model.predict(ar, verbose=1)
    label_index = np.argmax(score)
    age = get_age(label_index)
    print("The predicted age is " + age)

def predict_gender(file, model):
    print("Predicting Gender.................................")
    ar = convert_to_array2(file)
    ar = ar.astype("float") / 255.
    score = model.predict(ar, verbose=1)
    label_index = np.argmax(score)
    gender = get_gender(label_index)
    print("The predicted gender is " + gender)
    

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to input image for prediction")
ap.add_argument("-m1", "--model_emotion", required=True, help="Path to input trained model")
ap.add_argument("-m2", "--model_age", required=True, help="Path to input trained model")
ap.add_argument("-m3", "--model_gender", required=True, help="Path to input trained model")
args = vars(ap.parse_args())

model_emotion = tensorflow.keras.models.load_model(args['model_emotion'])
predict_emotion(args['image'], model_emotion)
model_age = tensorflow.keras.models.load_model(args['model_age'])
predict_age(args['image'], model_age)
model_gender = tensorflow.keras.models.load_model(args['model_gender'])
predict_gender(args['image'], model_gender)