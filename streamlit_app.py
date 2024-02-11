from skimage.feature import hog
import streamlit as st
import numpy as np
import cv2
from pickle_ml import load_model
from PIL import Image
import pickle
import matplotlib.pyplot as plt

#Load the same parameters HOG from the training
with open('hog_parameters.para','rb') as f:
    parameters=pickle.load(f)

def HOG(x):
    return hog(x,pixels_per_cell=parameters['pixels_per_cell'],
               cells_per_block=parameters['cells_per_block'],
               orientations=parameters['orientations'])

def extract_HOG(X):
    features = HOG(X)
    return features

def identify(x):
    model = load_model('model.p')
    classes = model.named_steps.classifier.classes_
    x = cv2.resize(x,parameters['image_size'])
    features = extract_HOG(x)
    label = model.predict([features])
    prob = model.predict_proba([features])
    #rounded_prob = round(prob.max(),3)
    label[0] = label[0]#+'-p:' +str(rounded_prob)
    class_probability = (prob, model.named_steps.classifier.classes_)
    return label, class_probability

def detect_and_recognize(image): 
    # Load the cascade classifier
    cascade_path="haarcascade_frontalface_alt.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),)
    n_faces = len(faces)
    st.text(f'{len(faces)} Faces detected ')
    if n_faces>0:
        plt.figure(figsize=(10,8))
        fig,ax = plt.subplots(n_faces,2)
        index =0
        
        for (x, y, w, h) in faces:
            face_roi = image[y : y + h, x : x + w]
            name,class_probability = identify(face_roi)
            draw_bounding_boxes(image=image,text=name[0],x=x,y=y,w=w,h=h)
            if n_faces==1:
                ax[0].imshow(face_roi)
                ax[1].barh(width=class_probability[0][0], y=class_probability[1])
                #st.text(str(class_probability[0][0]))
            else:
                ax[index,0].imshow(face_roi)
                ax[index,1].barh(width=class_probability[0][0], y=class_probability[1],alpha=0.9)
                index = index+1
        plt.subplots_adjust(wspace=0.7, hspace=0.1)
    else:
        fig = None
    return image, fig

def draw_bounding_boxes(image, text, x, y, w, h):
    cv2.rectangle(image, (x, y), (x + w, y + h), color=(200, 200, 10), thickness=1)
    font_scale = min(w/100, h /60)
    cv2.putText(image, text, (x+5, y-3), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (20, 203, 35), 1)


def main():    
    st.header('Football Player - Face Recognition')
    uploaded_file = st.file_uploader("Choose a file to upload here:", type=["jpg",'jpeg','jpg'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        image = np.array(image)
        image,fig = detect_and_recognize(image)
        col1,col2 = st.columns(2)

        if fig:
            with col1:
                st.header("Input Image : ")
                st.image(image)
                
            with col2:
                st.header('Class Probabilities: ')
                st.pyplot(fig)


if __name__=='__main__':
    main()

    