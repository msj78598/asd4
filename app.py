import os
import torch
import requests
import streamlit as st
from PIL import Image, ImageDraw
import io

# إعدادات واجهة المستخدم Streamlit
st.set_page_config(page_title="اكتشاف الحقول الزراعية باستخدام YOLOv5", layout="wide")
st.title("نظام اكتشاف الحقول الزراعية")

# إعدادات API لقوقل ماب
API_KEY = "AIzaSyAY7NJrBjS42s6upa9z_qgNLVXESuu366Q"  # ضع هنا مفتاح API الخاص بك
ZOOM = 16
IMG_SIZE = 640
MAP_TYPE = "satellite"

# مسار النموذج المدرب YOLOv5
MODEL_PATH = "best.pt"  # ضع مسار النموذج المدرب لديك


# بدلاً من استخدام torch.hub.load لتحميل النموذج من GitHub
def load_model():
    model = torch.load(MODEL_PATH)  # تحميل النموذج المدرب المحلي
    model.eval()  # تحويل النموذج إلى وضع التقييم
    return model


# رفع الصورة من قبل المستخدم
st.sidebar.header("اختيار موقع على الخريطة")
lat = st.sidebar.number_input("خط العرض", value=21.4225)
lon = st.sidebar.number_input("خط الطول", value=39.8262)

# تحميل الصورة من قوقل ماب بناءً على الإحداثيات
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

# جلب الصورة بناءً على الإحداثيات
img_path = download_image(lat, lon)
if img_path:
    # قراءة الصورة
    image = Image.open(img_path)
    st.image(image, caption="الصورة المدخلة من خرائط قوقل", use_column_width=True)

    # تحويل الصورة إلى بيانات tensor لـ YOLOv5
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    # استخدم النموذج للكشف عن الأجسام
    results = model(img_bytes)

    # عرض النتائج
    st.header("النتائج:")
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
