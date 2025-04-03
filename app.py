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
ML_MODEL_PATH = "C:/Users/Sec/Documents/DEEP/isolation_forest_model.joblib"
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
        @page {
            size: A4 portrait;
            margin: 1cm;
        }
        body {
            zoom: 85%;
        }
        .card {
            page-break-inside: avoid;
            margin-bottom: 0.5cm;
            padding: 10px;
            min-height: auto;
        }
        .no-print {
            display: none !important;
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
    .card {
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        padding: 15px;
        margin-bottom: 15px;
        border-left: 4px solid;
        background-color: #f9f9f9;
        width: 100%;
        min-height: auto;
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
        margin-bottom: 10px;
    }
    .priority-badge {
        padding: 4px 12px;
        border-radius: 15px;
        font-weight: bold;
        font-size: 12pt;
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
    }
    .card-image-container {
        width: 40%;
    }
    .card-image {
        width: 100%;
        max-height: 6cm;
        object-fit: contain;
        border-radius: 6px;
        border: 1px solid #ddd;
    }
    .card-details {
        width: 60%;
    }
    .detail-row {
        margin-bottom: 8px;
        display: flex;
        font-size: 12pt;
    }
    .detail-label {
        font-weight: bold;
        min-width: 120px;
        color: #555;
    }
    .detail-value {
        flex: 1;
    }
    .card-actions {
        margin-top: 15px;
        display: flex;
        gap: 10px;
    }
    .action-btn {
        padding: 6px 15px;
        border-radius: 4px;
        text-decoration: none;
        font-weight: bold;
        font-size: 12pt;
    }
    .whatsapp-btn {
        background: #25D366;
        color: white;
    }
    .map-btn {
        background: #4285F4;
        color: white;
    }
    .print-btn {
        background: #4CAF50;
        color: white;
        border: none;
        padding: 8px 20px;
        border-radius: 5px;
        cursor: pointer;
        font-size: 14px;
        margin-left: 10px;
    }
    .buttons-container {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 15px;
    }
    .compact-card {
        margin-bottom: 10px;
    }
    </style>

    <script>
    function printReport() {
        window.print();
    }
    </script>
    """, unsafe_allow_html=True)

# -------------------------
# الدوال المساعدة
# -------------------------
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

# تابع باقي الكود بما فيه التحليل، الطباعة والتصدير بدون تغيير. 


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

# شريط العنوان
st.markdown("""
<div class="header">
    <h1 style="margin:0;">🌾 نظام اكتشاف حالات الفاقد الكهربائي المحتملة للفئة الزراعية</h1>
</div>
""", unsafe_allow_html=True)

# زر الطباعة
st.markdown("""
<div class="buttons-container">
    <button onclick="printReport()" class="print-btn">🖨️ طباعة التقرير</button>
</div>
""", unsafe_allow_html=True)

# قسم تحميل الملفات
with st.expander("📁 تحميل البيانات", expanded=True):
    col1, col2 = st.columns(2)
    
    with col1:
        template_file = "C:/Users/Sec/Documents/DEEP/fram.xlsx"
        st.download_button(
            "📥 تحميل نموذج البيانات",
            open(template_file, "rb"),
            file_name="نموذج_البيانات.xlsx",
            help="قم بتحميل نموذج البيانات لملئه بالمعلومات المطلوبة"
        )
    
    with col2:
        uploaded_file = st.file_uploader(
            "رفع ملف البيانات (Excel)",
            type=["xlsx"],
            help="يرجى رفع ملف Excel يحتوي على بيانات العملاء"
        )

if uploaded_file:
    # تحميل البيانات
    df = pd.read_excel(uploaded_file)
    df["cont"] = df["الاشتراك"].astype(str).str.strip()
    df["المكتب"] = df["المكتب"].astype(str)
    df["الكمية"] = pd.to_numeric(df["الكمية"], errors="coerce")
    
    # تحميل النماذج
    with st.spinner('جاري تحميل النماذج...'):
        model_yolo = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)
        model_ml = joblib.load(ML_MODEL_PATH)
    
    st.success("✅ تم تحميل النماذج بنجاح")
    
    # تبويبات للعرض
    tab1, tab2 = st.tabs(["🎯 النتائج", "📊 البيانات الخام"])
    
    with tab1:
        st.subheader("النتائج المباشرة")
        results_container = st.container()
        
        # معالجة البيانات
        results = []
        gallery = set()
        
        # شريط التقدم
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, row in df.iterrows():
            # تحديث شريط التقدم
            progress = (idx + 1) / len(df)
            progress_bar.progress(progress)
            status_text.text(f"جاري معالجة السجل {idx + 1} من {len(df)}...")
            
            meter_id = str(row["cont"])
            lat, lon = row['y'], row['x']
            office_number = row["المكتب"]
            img_path = download_image(lat, lon, meter_id)
            
            if img_path:
                # تحليل الصورة
                conf, img_detected, area = detect_field(img_path, meter_id, row, model_yolo)
                
                # عرض النتائج فقط للحالات التي تم اكتشاف حقول فيها
                if conf is not None and img_detected is not None and (conf, img_detected) not in gallery:
                    gallery.add((conf, img_detected))
                    
                    # باقي التحليلات
                    anomaly = predict_loss(row, model_ml)
                    capacity_limit = capacity_thresholds.get(row['Breaker Capacity'], 0)
                    consumption_check = row['الكمية'] < 0.5 * capacity_limit
                    high_priority_condition = (conf >= 85 and row['الكمية'] == 0) or (conf >= 85 and row['Breaker Capacity'] < 200)
                    priority = determine_priority(conf >= 85, anomaly, consumption_check, high_priority_condition)

                    # حفظ النتائج
                    result_row = row.copy()
                    result_row["نسبة_الثقة"] = conf
                    result_row["الأولوية"] = priority
                    result_row["المساحة"] = area
                    results.append(result_row)
                    
                    # إنشاء الروابط
                    location_link = generate_google_maps_link(lat, lon)
                    whatsapp_link = generate_whatsapp_share_link(
                        meter_id, conf, area, location_link, 
                        row['الكمية'], row['Breaker Capacity'], 
                        office_number, priority
                    )
                    
                    # تحديد فئة الأولوية
                    priority_class = {
                        "أولوية عالية جدًا": "high",
                        "قصوى": "high",
                        "متوسطة": "medium",
                        "منخفضة": "low"
                    }.get(priority, "")
                    
                    # قراءة الصورة كبايتات
                    try:
                        with open(img_detected, "rb") as f:
                            img_bytes = f.read()
                        img_base64 = base64.b64encode(img_bytes).decode()
                    except Exception as e:
                        st.error(f"خطأ في قراءة الصورة: {e}")
                        img_base64 = ""
                    
                    # عرض البطاقة
                    with results_container:
                        st.markdown(f"""
                        <div class="card priority-{priority_class} compact-card">
                            <div class="card-header">
                                <h3 style="margin:0;font-size:14pt;">العداد: {meter_id}</h3>
                                <span class="priority-badge {priority_class}-badge">{priority}</span>
                            </div>
                            <div class="card-content">
                                <div class="card-image-container">
                                    <img class="card-image" src="data:image/png;base64,{img_base64}" alt="صورة الحقل">
                                </div>
                                <div class="card-details">
                                    <div class="detail-row">
                                        <span class="detail-label">المكتب:</span>
                                        <span class="detail-value">{office_number}</span>
                                    </div>
                                    <div class="detail-row">
                                        <span class="detail-label">ثقة الكشف:</span>
                                        <span class="detail-value">{conf}%</span>
                                    </div>
                                    <div class="detail-row">
                                        <span class="detail-label">المساحة:</span>
                                        <span class="detail-value">{area:,} م²</span>
                                    </div>
                                    <div class="detail-row">
                                        <span class="detail-label">الاستهلاك:</span>
                                        <span class="detail-value">{row['الكمية']:,} ك.و.س</span>
                                    </div>
                                    <div class="detail-row">
                                        <span class="detail-label">سعة القاطع:</span>
                                        <span class="detail-value">{row['Breaker Capacity']:,} أمبير</span>
                                    </div>
                                </div>
                            </div>
                            <div class="card-actions">
                                <a href="{whatsapp_link}" class="action-btn whatsapp-btn" target="_blank">مشاركة عبر واتساب</a>
                                <a href="{location_link}" class="action-btn map-btn" target="_blank">عرض على الخريطة</a>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
    
    with tab2:
        st.subheader("البيانات الخام")
        st.dataframe(df)
    
    # زر تحميل النتائج في الشريط الجانبي
    if results:
        st.sidebar.download_button(
            "📥 تحميل النتائج (Excel)",
            data=pd.DataFrame(results).to_csv(index=False).encode('utf-8'),
            file_name="نتائج_الفحص.csv",
            mime="text/csv"
        )
        
        # إحصائيات سريعة في الشريط الجانبي
        st.sidebar.markdown("### 📊 إحصائيات سريعة")
        high_priority = len([r for r in results if r["الأولوية"] in ["أولوية عالية جدًا", "قصوى"]])
        medium_priority = len([r for r in results if r["الأولوية"] == "متوسطة"])
        low_priority = len([r for r in results if r["الأولوية"] == "منخفضة"])
        
        st.sidebar.metric("🔴 حالات عالية الخطورة", high_priority)
        st.sidebar.metric("🟠 حالات متوسطة الخطورة", medium_priority)
        st.sidebar.metric("🟢 حالات منخفضة الخطورة", low_priority)
    else:
        st.warning("⚠️ لم يتم العثور على أي نتائج للعرض")