import streamlit as st
import pandas as pd
import os
from PIL import Image
import numpy as np
import json
import tensorflow

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
import efficientnet
from tensorflow.keras.models import load_model
import efficientnet.tfkeras
from efficientnet.tfkeras import EfficientNetB6
#import keras.backend.tensorflow_backend as tb

st.write("""
# Welcome to Image Classifier Prototype

""")

st.write("You can explore functionality using our prepared images OR upload your own image below")

@st.cache(allow_output_mutation=True)
def load_config(config_path: str):

    with open(config_path, 'r') as fr:
        config = json.load(fr)

    return config

@st.cache(allow_output_mutation=True)
def load_nn(weight_path: str):

    model = load_model(weight_path)

    return model

config = load_config('config/config.json')
model_weight_path = config['model_weight_path']
class_indices = config['class_indices']
class_info = {v:k for k,v in class_indices.items()}

def file_selector(folder_path='./images'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select Image from our database:', filenames)
    return os.path.join(folder_path, selected_filename)

filename = file_selector()
st.write('Chosen Image `%s`' % filename)

def preprocessing_image(image_pil_array: 'PIL.Image'):

    image_pil_array = image_pil_array.convert('RGB')
    image_pil_array = image_pil_array.resize((299,299))
    x = image.img_to_array(image_pil_array)

    x = np.expand_dims(x, axis=0)
    test_datagen = ImageDataGenerator(rescale=1./255)

    return test_datagen.flow(x)

if filename:
    img = Image.open(filename)
    st.image(img, caption="Your Image", use_column_width=True)
    model = load_nn(model_weight_path)
    x = preprocessing_image(img)
    label = model.predict_generator(x)
    predict_rank = np.argsort(np.ravel(label))[::-1]
    st.write('The image is %s with %.5f%% probability' % (class_info[predict_rank[0]], (label[0][predict_rank[0]]*100) ))
    st.write('Class Probability')
    df = pd.DataFrame(label, columns=class_info.items(), index=['predict_proba'])
    st.write(df)

st.set_option('deprecation.showfileUploaderEncoding', False)

uploaded_file = st.file_uploader('Upload your own Image:')

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Your Image", use_column_width=True)
    model = load_nn(model_weight_path)
    x = preprocessing_image(img)
    label = model.predict_generator(x)
    predict_rank = np.argsort(np.ravel(label))[::-1]
    st.write('The image is %s with %.5f%% probability' % (class_info[predict_rank[0]], (label[0][predict_rank[0]]*100) ))
    st.write('Class Probability')
    df = pd.DataFrame(label, columns=class_info.items(), index=['predict_proba'])
    st.write(df)