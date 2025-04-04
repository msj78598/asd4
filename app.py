import os

# مسار المجلد الذي يحتوي على الملفات
folder_path = "C:/Users/Sec/Documents/DEEP"

# قائمة لتخزين معلومات الملفات
files_info = []

# استعراض الملفات وحساب الحجم
for root, dirs, files in os.walk(folder_path):
    for file in files:
        file_path = os.path.join(root, file)
        file_size = os.path.getsize(file_path)  # الحجم بالبايت
        file_info = {
            "file_name": file,
            "file_path": file_path,
            "file_size": file_size / (1024 * 1024)  # التحويل إلى ميغابايت
        }
        files_info.append(file_info)

# عرض الملفات مع حجمها
for file_info in files_info:
    print(f"اسم الملف: {file_info['file_name']}, الحجم: {file_info['file_size']:.2f} MB")

# تصنيف الملفات حسب الحجم (على سبيل المثال، تحديد الملفات الكبيرة جدًا)
large_files = [file for file in files_info if file['file_size'] > 50]  # افترض أن الملفات الكبيرة هي التي حجمها أكثر من 50 MB
print("\nالملفات الكبيرة:")
for file in large_files:
    print(f"{file['file_name']} - {file['file_size']:.2f} MB")
