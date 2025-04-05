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

# -------------------------
# إعدادات عامة
# -------------------------
st.set_page_config(
    page_title="نظام اكتشاف حالات الفاقد الكهربائي المحتملة للفئة الزراعية",
    layout="wide",
    page_icon="🌾"
)

# إعدادات API للقمر الصناعي
API_KEY = "AIzaSyAY7NJrBjS42s6upa9z_qgNLVXESuu366Q"
ZOOM = 16
IMG_SIZE = 640
MAP_TYPE = "satellite"

# إعدادات المجلدات والمسارات
IMG_DIR = "images"
DETECTED_DIR = "DETECTED_IMAGES"
MODEL_PATH = "best.pt"
ML_MODEL_PATH = "isolation_forest_model.joblib"
OUTPUT_EXCEL = "detected_low_usage.xlsx"

# تعريف الحدود القصوى لاستهلاك الطاقة
capacity_thresholds = {
    20: 6000, 50: 15000, 70: 21000, 100: 30000, 150: 45000,
    200: 60000, 300: 90000, 400: 120000, 500: 150000
}

# إنشاء المجلدات اللازمة
Path(IMG_DIR).mkdir(parents=True, exist_ok=True)
Path(DETECTED_DIR).mkdir(parents=True, exist_ok=True)
Path("output").mkdir(parents=True, exist_ok=True)

# -------------------------
# تحسينات واجهة المستخدم
# -------------------------
def setup_ui():
    st.markdown("""
    <style>
    /* تنسيقات الطباعة */
    @media print {
        .no-print {
            display: none !important;
        }
        .print-only {
            display: block !important;
        }
        body {
            direction: rtl;
            font-family: Arial, sans-serif;
        }
        .card {
            page-break-inside: avoid;
            margin-bottom: 15px;
            border: 1px solid #ddd;
            padding: 10px;
        }
    }
    
    /* تنسيقات عامة */
    :root {
        --high-color: #ff0000;
        --medium-color: #ffa500;
        --low-color: #008000;
    }
    .main {
        direction: rtl;
        text-align: right;
        font-family: 'Arial', sans-serif;
    }
    .header {
        background-color: #2c3e50;
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    .cards-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr); /* تحديد 3 أعمدة ثابتة في كل صف */
        gap: 15px;
        margin-top: 20px;
    }
    @media (max-width: 768px) {
        .cards-container {
            grid-template-columns: 1fr;
        }
    }
    .card {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        padding: 12px;
        border-left: 4px solid;
        background-color: #f9f9f9;
    }
    .priority-high {
        border-color: var(--high-color);
        background-color: #ffebee;
    }
    .priority-medium {
        border-color: var(--medium-color);
        background-color: #fff3e0;
    }
    .priority-low {
        border-color: var(--low-color);
        background-color: #e8f5e9;
    }
    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }
    .priority-badge {
        padding: 3px 10px;
        border-radius: 15px;
        font-weight: bold;
        font-size: 10pt;
        color: white;
    }
    .high-badge {
        background-color: var(--high-color);
    }
    .medium-badge {
        background-color: var(--medium-color);
    }
    .low-badge {
        background-color: var(--low-color);
    }
    .card-content {
        display: flex;
        gap: 10px;
        align-items: flex-start;
    }
    .card-image-container {
    width: 200px;  /* زيادة العرض بشكل معتدل */
    flex-shrink: 0;
}

.card-image {
    width: 100%;
    height: 150px;  /* زيادة الارتفاع بشكل معتدل */
    object-fit: cover;
    border-radius: 4px;
    border: 1px solid #ddd;
}


    .card-details {
        flex: 1;
        min-width: 0;
    }
    .detail-row {
        margin-bottom: 4px;
        display: flex;
        font-size: 10pt;
    }
    .detail-label {
        font-weight: bold;
        min-width: 80px;
        color: #555;
    }
    .detail-value {
        flex: 1;
        text-align: left;
    }
    .card-actions {
        margin-top: 8px;
        display: flex;
        gap: 8px;
    }
    .action-btn {
        padding: 4px 8px;
        border-radius: 4px;
        text-decoration: none;
        font-size: 10pt;
        white-space: nowrap;
    }
    .whatsapp-btn {
        background: #25D366;
        color: white;
    }
    .map-btn {
        background: #4285F4;
        color: white;
    }
    .buttons-container {
        display: flex;
        gap: 10px;
        margin-bottom: 15px;
    }
    .print-btn {
        background: #4CAF50;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        cursor: pointer;
        font-size: 12pt;
    }
    .download-btn-top {
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------------
# الدوال المساعدة
# -------------------------
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except:
        return ""

def download_image(lat, lon, meter_id):
    img_path = os.path.join(IMG_DIR, f"{meter_id}.png")
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

def pixel_to_area(lat, box):
    scale = 156543.03392 * abs(math.cos(math.radians(lat))) / (2 ** ZOOM)
    width_m = abs(box[2] - box[0]) * scale
    height_m = abs(box[3] - box[1]) * scale
    return width_m * height_m

def detect_field(img_path, meter_id, info, model):
    try:
        results = model(img_path)
        df_result = results.pandas().xyxy[0]
        fields = df_result[df_result["name"] == "field"]
        
        if not fields.empty:
            confidence = round(fields["confidence"].max() * 100, 2)
            if confidence >= 85:
                image = Image.open(img_path).convert("RGB")
                draw = ImageDraw.Draw(image)
                largest_field = fields.iloc[0]
                box = [largest_field["xmin"], largest_field["ymin"], 
                      largest_field["xmax"], largest_field["ymax"]]
                draw.rectangle(box, outline="green", width=3)
                area = pixel_to_area(info['y'], box)
                draw.text((10, 10), f"ID: {meter_id}\nArea: {int(area)} m²", fill="yellow")
                
                os.makedirs(DETECTED_DIR, exist_ok=True)
                image_path = os.path.join(DETECTED_DIR, f"{meter_id}.png")
                image.save(image_path)
                
                return confidence, image_path, int(area)
    except Exception as e:
        st.error(f"خطأ في معالجة الصورة: {e}")
    return None, None, None

def predict_loss(info, model_ml):
    X = [[info["Breaker Capacity"], info["الكمية"]]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return model_ml.predict(X_scaled)[0]

def determine_priority(has_field, anomaly, consumption_check, high_priority_condition):
    if high_priority_condition:
        return "أولوية عالية جدًا"
    elif has_field and anomaly == 1 and consumption_check:
        return "قصوى"
    elif has_field and (anomaly == 1 or consumption_check):
        return "متوسطة"
    elif has_field:
        return "منخفضة"
    return "طبيعية"

def generate_whatsapp_share_link(meter_id, confidence, area, location_link, quantity, capacity, office_number, priority):
    message = (
        f"⚡ تقرير حالة عداد زراعي\n\n"
        f"🔢 رقم العداد: {meter_id}\n"
        f"🏢 المكتب: {office_number}\n"
        f"🚨 الأولوية: {priority}\n"
        f"📊 ثقة الكشف: {confidence}%\n"
        f"🔳 المساحة: {area:,} م²\n"
        f"💡 الاستهلاك: {quantity:,} ك.و.س\n"
        f"⚡ سعة القاطع: {capacity:,} أمبير\n"
        f"📍 الموقع: {location_link}"
    )
    return f"https://wa.me/?text={urllib.parse.quote(message)}"

def generate_google_maps_link(lat, lon):
    return f"https://www.google.com/maps?q={lat},{lon}"

# -------------------------
# واجهة Streamlit
# -------------------------
setup_ui()

# عنوان الصفحة
st.markdown("""
<div class="header">
    <h1 style="margin:0;">🌾 نظام اكتشاف حالات الفاقد الكهربائي المحتملة للفئة الزراعية</h1>
</div>
""", unsafe_allow_html=True)

# زر تحميل النتائج في أعلى الصفحة
if 'results' in st.session_state and st.session_state.results:
    # إنشاء ملف Excel للتحميل
    df_results = pd.DataFrame(st.session_state.results)
    df_results = df_results.sort_values(by=["x", "y"], ascending=[True, True])
    df_results["رابط الموقع"] = df_results.apply(lambda row: generate_google_maps_link(row["x"], row["y"]), axis=1)
    df_results = df_results.drop(columns=["x", "y"])
    
    file_path = "output/detected_low_usage_sorted.xlsx"
    df_results.to_excel(file_path, index=False, engine='openpyxl')
    
    # عرض زر التحميل في أعلى الصفحة
    with open(file_path, "rb") as f:
        st.download_button(
            label="📥 تحميل النتائج (Excel)",
            data=f,
            file_name="نتائج_الفحص.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="تحميل النتائج كملف Excel مع رابط الموقع مرتب حسب الإحداثيات",
            key="top_download_button"
        )
