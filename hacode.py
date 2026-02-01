import streamlit as st 
import tensorflow as tf
from PIL import Image
import numpy as np
st.set_page_config(page_title="Hand Alphabet Clasifier" ,page_icon= "🔤")
st.title("Hand Alphabet Clasifier 🤞🏻✌🏻🔤")

model = tf.keras.models.load_model('hamodel2.h5')


upload_image = st.file_uploader("select image" , type=['jpg' , 'jpeg' ,'png'])
if upload_image is not None :
    image= Image.open(upload_image)
    st.image(image , caption = "image uploader" )


    if image.mode != "RGB":
     image = image.convert("RGB")
    
    image = image.resize((64,64))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image,axis = 0)

    pred = model.predict(image)
    class_index = np.argmax(pred)
    class_list = ["A","B","Blank","C","D","E","F","G","H","I","J","K","L","M","N","O",
              "P","Q","R","S","T","U","V","W","X","Y","Z"]    
    btn = st.button("Print Result")
    if btn :
        st.success(class_list[class_index])

