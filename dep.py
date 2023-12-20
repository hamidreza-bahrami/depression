import streamlit as st
import pandas as pd
import numpy as np
import pickle 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def load_model():
    with open('saved.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

model = data['model']
x = data['x']

def show_page():
    st.write("<h1 style='text-align: center; color: blue;'>مدل تشخیص افسردگی اساسی</h1>", unsafe_allow_html=True)
    st.write("<h2 style='text-align: center; color: gray;'>علائم خود را وارد کنید</h2>", unsafe_allow_html=True)
    st.write("<h4 style='text-align: center; color: gray;'>True = بله , False = خیر</h4>", unsafe_allow_html=True)

    sadness = (True , False)
    sadness = st.selectbox('آیا در طول روز احساس اندوه می کنید؟', sadness)

    emptiness = (True , False)
    emptiness = st.selectbox('آیا احساس پوچی و ناامیدی می کنید؟', emptiness)

    allday1	 = (True , False)
    allday1	 = st.selectbox('درصورت مثبت بودن پاسخ قبلی، تمام روز این احساس را دارید؟', allday1)

    joyless = (True , False)
    joyless = st.selectbox('علاقه و لذت خود را در فعالیت ها از دست داده اید؟', joyless)

    allday2 = (True , False)
    allday2 = st.selectbox('در صورت مثبت بودن پاسخ قبلی، آیا در تمام روز این احساس را دارید؟', allday2)

    Irritable = (True , False)
    Irritable = st.selectbox('آیا اخیرا از نظر خلقی به راحتی تحریک می شوید؟', Irritable)

    weight = (True , False)
    weight = st.selectbox('آیا کاهش وزن قابل توجه داشته اید؟ کاهش 5 درصد از وزن بدن در هر ماه', weight)

    insomnia = (True , False)
    insomnia = st.selectbox('آیا اخیرا دچار بی خوابی یا پرخوابی شده اید؟', insomnia)

    worriness = (True , False)
    worriness = st.selectbox('آیا اخیرا دچار سراسیمگی یا نگرانی توام با رفتار عصبی شده اید؟', worriness)

    slowness = (True , False)
    slowness = st.selectbox('آیا دچار کندی فکری و عملی شده اید؟', slowness)

    allday3 = (True , False)
    allday3 = st.selectbox('در صورت مثبت بودن پاسخ دو سوال اخیر، آیا این علائم هر روز رخ داده است؟', allday3)

    worthless = (True , False)
    worthless = st.selectbox('آیا احساس بی ارزشی می کنید؟', worthless)

    guilty = (True , False)
    guilty = st.selectbox('آیا احساس گناه می کنید؟', guilty)

    focus = (True , False)
    focus = st.selectbox('آیا اخیرا توانایی تمرکز و تفکر شما کاهش یافته است؟', focus)

    allday4 = (True , False)
    allday4 = st.selectbox('آیا این کاهش تمرکز، هر روز اتفاق افتاده است؟', allday4)

    killing = (True , False)
    killing = st.selectbox('آیا افکار خودکشی دارید؟', killing)

    drugs = (True , False)
    drugs = st.selectbox('آیا مواد مخدر یا محرک مصرف می کنید؟', drugs)

    disturbed = (True , False)
    disturbed = st.selectbox('آیا موارد ذکر شده باعث اختلال عملکرد شغلی، اجتماعی و ... شما شده است؟', disturbed)

    weeks = st.slider('بازه زمانی علائم خود را از نظر تعداد هفته مشخص کنید ', 1.0, 100.0, 2.0)

    age = st.slider('بازه سنی خود را مشخص کنید', 18.0, 50.0, 20.0)

    button = st.button('معاینه و تشخیص')
    if button:
        x = np.array([[sadness, emptiness, allday1, joyless, allday2, Irritable, weight, insomnia, worriness,
                       slowness, allday3, worthless, guilty, focus, allday4, killing, drugs, disturbed, weeks, age]])

        y_prediction = model.predict(x)
        if y_prediction == True:
            st.write("<h4 style='text-align: center; color: gray;'>بر اساس داده های وارد شده، شما به افسردگی اساسی مبتلا هستید</h4>", unsafe_allow_html=True)
            st.write("<h5 style='text-align: center; color: gray;'>برای درمان به روانشناس مراجعه کنید</h5>", unsafe_allow_html=True)
        elif y_prediction == False:
            st.write("<h4 style='text-align: center; color: gray;'>بر اساس داده های وارد شده، شما در سلامتی کامل هستید</h4>", unsafe_allow_html=True)

show_page()
