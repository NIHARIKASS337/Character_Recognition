import numpy as np
import streamlit as st
from skimage.io import imread, imshow
from skimage.transform import resize
import pickle


model1 = pickle.load(open('Models/KCR.pkl', 'rb'))  # LogisticRegression
numbers = {'17': 'ಕ್', '18': 'ಕ', '19': 'ಕಾ', '20': 'ಕಿ',
           '21': 'ಕೀ', '22': 'ಕು', '23': 'ಕೂ', '24': 'ಕೃ', '25': 'ಕೆ', '26': 'ಕೇ', '27': 'ಕೈ', '28': 'ಕೊ', '29': 'ಕೋ', '30': 'ಕೌ',
           '31': 'ಕಂ', '32': 'ಕಃ'}


def tranform_image(img):
    feature = []
    tranform_img = resize(img, (150, 150, 3))
    flatten_img = tranform_img.flatten()
    feature.append(flatten_img)
    return np.array(feature)


# --------------UI Beginning------------
st.title('Kannada Characters Recoginition')

# Image Picker
try:
    st.title('Pick a Image')
    img_name = st.file_uploader('Select Image')
    try:
        st.image(f'test_img/{img_name.name}')
        img_ = imread(f'test_img/{img_name.name}')
    except:
        st.image(f'test_img/test_img/{img_name.name}')
        img_ = imread(f'test_img/test_img/{img_name.name}')
    # Predict Button
    if st.button('Predict Class'):
        try:
            # LR
            res1 = (model1.predict(tranform_image(img_)))[0]
            res1 = str(res1)
            st.header(
                f'Character is {numbers[res1]}')

        except:
            st.info('Please try again.')
except:
    pass
