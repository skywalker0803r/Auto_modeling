from google.colab import auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import google.auth
import sys
import os

# 設定你的 Google Drive 資料夾 ID
FOLDER_ID = '1awJWTKhJpiuwZtvzgKahpAF9dckb4HRh'

def upload_file_to_drive(filepath):
    if not os.path.exists(filepath):
        print(f"❌ 檔案不存在：{filepath}")
        return

    # 授權
    auth.authenticate_user()
    creds, _ = google.auth.default()
    drive_service = build('drive', 'v3', credentials=creds)

    # 上傳檔案
    file_metadata = {
        'name': os.path.basename(filepath),
        'parents': [FOLDER_ID]  # 指定目標資料夾
    }
    media = MediaFileUpload(filepath, mimetype='text/x-python')  # 根據檔案類型可自行調整
    uploaded_file = drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id, webViewLink'
    ).execute()

    print(f"✅ 上傳成功：{filepath}")
    print(f"🔗 檔案連結：{uploaded_file['webViewLink']}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("❗用法：!python upload_to_drive.py <你的檔案名稱.py>")
    else:
        upload_file_to_drive(sys.argv[1])
