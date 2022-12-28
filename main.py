import streamlit as st
import pickle
import numpy as np

pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Laptop Price Predictor")

company = st.selectbox('Brand Name', df['Company'].unique())
typename = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weights')
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS Display', ['No', 'Yes'])
screen_size = st.number_input('Screen Size')
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768',
                                                '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600',
                                                '2560x1440', '2304x1440'])
cpu = st.selectbox('CPU Brand', df['CPU Brand'].unique())
hdd = st.selectbox('HDD', ['No', 'Yes'])
ssd = st.selectbox('SSD', ['No', 'Yes'])
gpu = st.selectbox('GPU', df['Gpu Brand'].unique())
os = st.selectbox('OS', df['OS'].unique())
x_res = int(resolution.split("x")[0])
y_res = int(resolution.split("x")[1])

if st.button('Predict Price'):
    ppi = ((x_res ** 2 + y_res ** 2) ** 0.5) / screen_size
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0
    hdd = 1 if hdd == 'Yes' else 0
    ssd = 1 if ssd == 'Yes' else 0
    query = np.array([company, typename, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
    print(query)
    query = query.reshape(1, 12)
    print(query)
    st.title(np.exp(pipe.predict(query)[0]))
