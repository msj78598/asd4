import os
import pandas as pd
import torch
import requests
import streamlit as st
from PIL import Image
import io

# إعدادات API لقوقل ماب
API_KEY = "AIzaSyAY7NJrBjS42s6upa9z_qgNLVXESuu366Q"  # مفتاح API الخاص بك
ZOOM = 16
IMG_SIZE = 640
MAP_TYPE = "satellite"

import torch

# تحميل نموذج YOLOv5 المدرب المحلي
def load_model():
    try:
        model = torch.load(MODEL_PATH)  # تحميل النموذج المدرب
        model.eval()  # وضع النموذج في وضع التقييم
        print("✅ تم تحميل النموذج بنجاح!")
        return model
    except Exception as e:
        print(f"❌ خطأ في تحميل النموذج: {e}")

# تحميل النموذج
model = load_model()



# تحميل الملف من المستخدم
uploaded_file = st.file_uploader("رفع ملف الإحداثيات (Excel)", type=["xlsx"])

# تحميل الصور من قوقل ماب بناءً على الإحداثيات
def download_image(lat, lon):
    img_path = os.path.join("images", f"{lat}_{lon}.png")
    if os.path.exists(img_path):
        return img_path
        
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    params = {
        "center": f"{lat},{lon}",
        "zoom": ZOOM,
        "size": f"{IMG_SIZE}x{IMG_SIZE}",
        "maptype": MAP_TYPE,
        "key": API_KEY
    }
    try:
        response = requests.get(base_url, params=params, timeout=20)
        if response.status_code == 200:
            with open(img_path, 'wb') as f:
                f.write(response.content)
            return img_path
    except Exception as e:
        st.error(f"خطأ في تحميل الصورة: {e}")
        return None

# معالجة البيانات وقراءة الملف
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip()
    st.write("تم تحميل الملف بنجاح!")
    st.dataframe(df)  # عرض البيانات

    # معالجة كل صف من البيانات
    for index, row in df.iterrows():
        lat = row['y']  # عمود خط العرض
        lon = row['x']  # عمود خط الطول
        meter_id = row['اسم']  # عمود اسم العداد

        # تحميل الصورة بناءً على الإحداثيات
        img_path = download_image(lat, lon)
        if img_path:
            # قراءة الصورة
            image = Image.open(img_path)
            st.image(image, caption=f"الصورة للموقع: {lat}, {lon}", use_column_width=True)

            # تحويل الصورة إلى بيانات tensor لـ YOLOv5
            img_bytes = io.BytesIO()
            image.save(img_bytes, format='PNG')
            img_bytes.seek(0)

            # استخدم النموذج للكشف عن الأجسام
            results = model(img_bytes)

            # عرض النتائج
            st.header(f"النتائج للموقع: {lat}, {lon}")
            st.write("تم الكشف عن الأجسام في الصورة.")
            st.image(results.render()[0])  # عرض الصورة بعد التعرف على الحقول

            # إظهار معلومات عن الأجسام المكتشفة
            results_df = results.pandas().xyxy[0]
            st.write(results_df)  # عرض البيانات في جدول

            # إضافة رابط لتخزين الصورة الناتجة
            output_image_path = "detected_field_image.jpg"
            results.save(output_image_path)

            st.download_button(
                label="تحميل الصورة الناتجة",
                data=open(output_image_path, "rb"),
                file_name=output_image_path,
                mime="image/jpeg"
            )
