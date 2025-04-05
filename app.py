import os
import math
import pandas as pd
import requests
from pathlib import Path
from PIL import Image, ImageDraw
import torch
import joblib
from sklearn.preprocessing import StandardScaler
import streamlit as st
import urllib.parse
import base64
import io
import sys

# إصلاح مشاكل OpenCV
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

try:
    import cv2
    from ultralytics import YOLO
except ImportError as e:
    st.error(f"خطأ في تحميل المكتبات: {str(e)}")
    st.stop()

# -------------------------
# إعدادات التطبيق
# -------------------------
st.set_page_config(
    page_title="نظام اكتشاف الفاقد الكهربائي الزراعي",
    layout="wide",
    page_icon="🌾"
)

# إعدادات API
API_KEY = "AIzaSyAY7NJrBjS42s6upa9z_qgNLVXESuu366Q"
ZOOM = 16
IMG_SIZE = 640
MAP_TYPE = "satellite"

# مسارات الملفات
IMG_DIR = "images"
DETECTED_DIR = "detected_images"
MODEL_PATH = "best.pt"
ML_MODEL_PATH = "isolation_forest_model.joblib"

# عتبات الاستهلاك
capacity_thresholds = {
    20: 6000, 50: 15000, 70: 21000, 100: 30000, 150: 45000,
    200: 60000, 300: 90000, 400: 120000, 500: 150000
}

# إنشاء المجلدات
Path(IMG_DIR).mkdir(exist_ok=True)
Path(DETECTED_DIR).mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)

# -------------------------
# دوال المساعدة
# -------------------------
def setup_ui():
    """تكوين واجهة المستخدم"""
    st.markdown("""
    <style>
    .main {direction: rtl; text-align: right;}
    .header {background-color: #2c3e50; color: white; padding: 15px; border-radius: 10px;}
    .card {border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); padding: 12px; margin-bottom: 15px;}
    </style>
    """, unsafe_allow_html=True)

def download_image(lat, lon, meter_id):
    """تحميل صورة من خرائط جوجل"""
    img_path = os.path.join(IMG_DIR, f"{meter_id}.png")
    if os.path.exists(img_path):
        return img_path
        
    params = {
        "center": f"{lat},{lon}",
        "zoom": ZOOM,
        "size": f"{IMG_SIZE}x{IMG_SIZE}",
        "maptype": MAP_TYPE,
        "key": API_KEY
    }
    
    try:
        response = requests.get("https://maps.googleapis.com/maps/api/staticmap", params=params, timeout=20)
        if response.status_code == 200:
            with open(img_path, 'wb') as f:
                f.write(response.content)
            return img_path
    except Exception as e:
        st.error(f"خطأ في تحميل الصورة: {str(e)}")
        return None

def detect_fields(img_path, meter_id, info, model):
    """كشف الحقول الزراعية باستخدام YOLO"""
    try:
        results = model(img_path)
        boxes = results[0].boxes
        if len(boxes) > 0:
            box = boxes[0].xyxy[0].tolist()
            conf = boxes[0].conf.item()
            
            img = Image.open(img_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            draw.rectangle(box, outline="green", width=3)
            
            # حساب المساحة
            scale = 156543.03392 * abs(math.cos(math.radians(info['y']))) / (2 ** ZOOM)
            width_m = abs(box[2] - box[0]) * scale
            height_m = abs(box[3] - box[1]) * scale
            area = width_m * height_m
            
            draw.text((10, 10), f"ID: {meter_id}\nArea: {int(area)} m²", fill="yellow")
            
            output_path = os.path.join(DETECTED_DIR, f"{meter_id}.png")
            img.save(output_path)
            
            return conf * 100, output_path, int(area)
    except Exception as e:
        st.error(f"خطأ في معالجة الصورة: {str(e)}")
    return None, None, None

# -------------------------
# الواجهة الرئيسية
# -------------------------
setup_ui()
st.markdown("<div class='header'><h1>🌾 نظام اكتشاف الفاقد الكهربائي الزراعي</h1></div>", unsafe_allow_html=True)

# إدارة الحالة
if 'results' not in st.session_state:
    st.session_state.update({
        'results': [],
        'df': None,
        'model_yolo': None,
        'model_ml': None,
        'analysis_done': False
    })

# تحميل البيانات
with st.expander("📁 تحميل البيانات", expanded=True):
    uploaded_file = st.file_uploader("رفع ملف البيانات (Excel)", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = df.columns.str.strip()
        df["cont"] = df["الاشتراك"].astype(str).str.strip()
        st.session_state.df = df
    except Exception as e:
        st.error(f"خطأ في قراءة الملف: {str(e)}")

# تحميل النماذج
if st.session_state.df is not None and st.session_state.model_yolo is None:
    with st.spinner('جاري تحميل النماذج...'):
        try:
            st.session_state.model_yolo = YOLO(MODEL_PATH)
            st.session_state.model_ml = joblib.load(ML_MODEL_PATH)
        except Exception as e:
            st.error(f"خطأ في تحميل النماذج: {str(e)}")

# التحليل والعرض
if st.session_state.df is not None and st.session_state.model_yolo is not None:
    if st.button("▶️ بدء التحليل"):
        with st.spinner('جاري تحليل البيانات...'):
            results = []
            for _, row in st.session_state.df.iterrows():
                meter_id = str(row["cont"])
                img_path = download_image(row['y'], row['x'], meter_id)
                
                if img_path:
                    conf, detected_path, area = detect_fields(img_path, meter_id, row, st.session_state.model_yolo)
                    if conf and detected_path:
                        results.append({
                            "رقم العداد": meter_id,
                            "المساحة": f"{area:,} م²",
                            "الثقة": f"{conf:.2f}%",
                            "الصورة": detected_path
                        })
            
            st.session_state.results = results
            st.session_state.analysis_done = True
            st.rerun()

    if st.session_state.analysis_done:
        st.subheader("النتائج")
        for result in st.session_state.results:
            with st.expander(f"العداد: {result['رقم العداد']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.image(result["الصورة"], use_column_width=True)
                with col2:
                    st.metric("المساحة", result["المساحة"])
                    st.metric("ثقة الكشف", result["الثقة"])
