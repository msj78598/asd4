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
DETECTED_DIR = "DETECTED_FIELDS/FIELDS/farms"
MODEL_PATH = "yolov5/farms_project/field_detector/weights/best.pt"
ML_MODEL_PATH = "isolation_forest_model.joblib"
OUTPUT_EXCEL = "output/detected_low_usage.xlsx"

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

# قسم تحميل الملفات
with st.expander("📁 تحميل البيانات", expanded=True):
    uploaded_file = st.file_uploader(
        "رفع ملف البيانات (Excel)",
        type=["xlsx"],
        help="يرجى رفع ملف Excel يحتوي على بيانات العملاء",
        key="data_uploader"
    )

# تهيئة حالة الجلسة
if 'results' not in st.session_state:
    st.session_state.results = []
    st.session_state.df = None
    st.session_state.model_yolo = None
    st.session_state.model_ml = None
    st.session_state.analysis_done = False
    st.session_state.file_uploaded = False

if uploaded_file:
    st.session_state.file_uploaded = True
    # تحميل البيانات مع تصحيح ترميز الأعمدة
    try:
        df = pd.read_excel(uploaded_file)
        df.columns = df.columns.str.strip()
        df["cont"] = df["الاشتراك"].astype(str).str.strip()
        df["المكتب"] = df["المكتب"].astype(str)
        df["الكمية"] = pd.to_numeric(df["الكمية"], errors="coerce")
        st.session_state.df = df
    except Exception as e:
        st.error(f"خطأ في قراءة الملف: {e}")
        st.stop()

    # تحميل النماذج
    if st.session_state.model_yolo is None or st.session_state.model_ml is None:
        with st.spinner('جاري تحميل النماذج...'):
            try:
                model_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)
                model_ml = joblib.load(ML_MODEL_PATH)
                st.session_state.model_yolo = model_yolo
                st.session_state.model_ml = model_ml
                st.success("✅ تم تحميل النماذج بنجاح")
            except Exception as e:
                st.error(f"خطأ في تحميل النماذج: {e}")
                st.stop()

# -------------------------
# الشريط الجانبي
# -------------------------
if st.session_state.file_uploaded:
    st.sidebar.markdown("### 🛠️ أدوات التحكم")
    
    # زر تحليل البيانات
    if st.sidebar.button("▶️ بدء تحليل البيانات", key="analyze_btn", help="انقر لبدء عملية تحليل البيانات"):
        with st.spinner('جاري تحليل البيانات...'):
            results = []
            gallery = set()
            progress_bar = st.sidebar.progress(0)
            total_rows = len(st.session_state.df)

            for idx, row in st.session_state.df.iterrows():
                progress = (idx + 1) / total_rows
                progress_bar.progress(progress)

                meter_id = str(row["cont"])
                lat, lon = row['y'], row['x']
                office_number = row["المكتب"]
                img_path = download_image(lat, lon, meter_id)

                if img_path:
                    conf, img_detected, area = detect_field(img_path, meter_id, row, st.session_state.model_yolo)

                    if conf is not None and img_detected is not None and (conf, img_detected) not in gallery:
                        gallery.add((conf, img_detected))

                        anomaly = predict_loss(row, st.session_state.model_ml)
                        capacity_limit = capacity_thresholds.get(row['Breaker Capacity'], 0)
                        consumption_check = row['الكمية'] < 0.5 * capacity_limit
                        high_priority_condition = (conf >= 85 and row['الكمية'] == 0) or (conf >= 85 and row['Breaker Capacity'] < 200)
                        priority = determine_priority(conf >= 85, anomaly, consumption_check, high_priority_condition)

                        result_row = {
                            "رقم العداد": meter_id,
                            "المكتب": office_number,
                            "الأولوية": priority,
                            "ثقة الكشف": f"{conf}%",
                            "المساحة": f"{area:,} م²",
                            "الاستهلاك": f"{row['الكمية']:,} ك.و.س",
                            "سعة القاطع": f"{row['Breaker Capacity']:,} أمبير",
                            "img_path": img_detected,
                            "رابط الموقع": generate_google_maps_link(lat, lon),
                            "x": row['x'],
                            "y": row['y']
                        }
                        results.append(result_row)

            progress_bar.empty()
            st.session_state.results = results
            st.session_state.analysis_done = True
            st.rerun()

    # إحصائيات سريعة
    if st.session_state.analysis_done:
        st.sidebar.markdown("### 📊 إحصائيات سريعة")
        results = st.session_state.results
        high_priority = len([r for r in results if r["الأولوية"] in ["أولوية عالية جدًا", "قصوى"]])
        medium_priority = len([r for r in results if r["الأولوية"] == "متوسطة"])
        low_priority = len([r for r in results if r["الأولوية"] == "منخفضة"])

        st.sidebar.metric("🔴 حالات عالية الخطورة", high_priority)
        st.sidebar.metric("🟠 حالات متوسطة الخطورة", medium_priority)
        st.sidebar.metric("🟢 حالات منخفضة الخطورة", low_priority)

# -------------------------
# تبويبات العرض الرئيسية
# -------------------------
if st.session_state.file_uploaded:
    tab1, tab2 = st.tabs(["🎯 النتائج", "📊 البيانات الخام"])

    with tab1:
        if st.session_state.analysis_done:
            st.subheader("النتائج المباشرة")
            results_container = st.container()

            with results_container:
                st.markdown('<div class="cards-container">', unsafe_allow_html=True)
                for result in st.session_state.results:
                    priority = result["الأولوية"]
                    priority_class = {
                        "أولوية عالية جدًا": "high",
                        "قصوى": "high",
                        "متوسطة": "medium",
                        "منخفضة": "low"
                    }.get(priority, "")

                    img_base64 = get_base64_image(result["img_path"])
                    
                    st.markdown(f"""
                    <div class="card priority-{priority_class}">
                        <div class="card-header">
                            <h4 style="margin:0;">العداد: {result['رقم العداد']}</h4>
                            <span class="priority-badge {priority_class}-badge">{priority}</span>
                        </div>
                        <div class="card-content">
                            <div class="card-image-container">
                                <img class="card-image" src="data:image/png;base64,{img_base64}" alt="صورة الحقل">
                            </div>
                            <div class="card-details">
                                <div class="detail-row">
                                    <span class="detail-label">المكتب:</span>
                                    <span class="detail-value">{result['المكتب']}</span>
                                </div>
                                <div class="detail-row">
                                    <span class="detail-label">الثقة:</span>
                                    <span class="detail-value">{result['ثقة الكشف']}</span>
                                </div>
                                <div class="detail-row">
                                    <span class="detail-label">المساحة:</span>
                                    <span class="detail-value">{result['المساحة']}</span>
                                </div>
                                <div class="detail-row">
                                    <span class="detail-label">الاستهلاك:</span>
                                    <span class="detail-value">{result['الاستهلاك']}</span>
                                </div>
                                <div class="detail-row">
                                    <span class="detail-label">سعة القاطع:</span>
                                    <span class="detail-value">{result['سعة القاطع']}</span>
                                </div>
                            </div>
                        </div>
                        <div class="card-actions">
                            <a href="{generate_whatsapp_share_link(
                                result['رقم العداد'],
                                float(result['ثقة الكشف'].replace('%', '')),
                                int(result['المساحة'].replace(' م²', '').replace(',', '')),
                                result['رابط الموقع'],
                                float(result['الاستهلاك'].replace(' ك.و.س', '').replace(',', '')),
                                float(result['سعة القاطع'].replace(' أمبير', '').replace(',', '')),
                                result['المكتب'],
                                result['الأولوية']
                            )}" class="action-btn whatsapp-btn" target="_blank">واتساب</a>
                            <a href="{result['رابط الموقع']}" class="action-btn map-btn" target="_blank">خريطة</a>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # زر تحميل إضافي في أسفل الصفحة
            st.markdown("---")
            st.markdown("### خيارات التصدير")
            with open(file_path, "rb") as f:
                st.download_button(
                    label="📥 تحميل النتائج (Excel)",
                    data=f,
                    file_name="نتائج_الفحص.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    help="تحميل النتائج كملف Excel مع رابط الموقع مرتب حسب الإحداثيات",
                    key="bottom_download_button"
                )
        else:
            st.info("⏳ يرجى النقر على زر 'بدء تحليل البيانات' في الشريط الجانبي لرؤية النتائج")

    with tab2:
        if st.session_state.df is not None:
            st.subheader("البيانات الخام")
            st.dataframe(st.session_state.df)
