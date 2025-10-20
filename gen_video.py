import os
import random
import json
from datetime import datetime
from gradio_client import Client as Client_gradio
from gradio_client import handle_file
from PIL import Image
import requests
import shutil
import io


# === Step 1: 建立日期資料夾 ===
today = datetime.now().strftime("%Y_%m_%d")
folder_path = os.path.join("videos", today)
os.makedirs(folder_path, exist_ok=True)


# === Step 2: 現行圖片 API ===
json_path = os.path.join("videos", 'current_data.json')
if os.path.exists(json_path):
    with open(json_path, "r") as f:
        last_data = json.load(f)
    response = requests.get(last_data['metadata']['nextPage'])
    data = response.json()
else:
    response = requests.get('https://civitai.com/api/v1/images?limit=1&nsfw=true')
    data = response.json()
with open(json_path, "w") as f:
    json.dump(data, f, indent=2)


# === Step 3: 調用 FLUX Space 模型產圖 ===
client = Client_gradio("zerogpu-aoti/wan2-2-fp8da-aoti-faster")

result = client.predict(
        input_image=handle_file(data['items'][0]['url']),
        prompt="",
        steps=6,
        negative_prompt="色调艳丽, 过曝, 静态, 细节模糊不清, 字幕, 风格, 作品, 画作, 画面, 静止, 整体发灰, 最差质量, 低质量, JPEG压缩残留, 丑陋的, 残缺的, 多余的手指, 画得不好的手部, 画得不好的脸部, 畸形的, 毁容的, 形态畸形的肢体, 手指融合, 静止不动的画面, 杂乱的背景, 三条腿, 背景人很多, 倒着走",
        duration_seconds=5,
        guidance_scale=1,
        guidance_scale_2=1,
        seed=42,
        randomize_seed=True,
        api_name="/generate_video"
)


# === Step 4: 建立日期資料夾與檔名 ===
existing_files = [f for f in os.listdir(folder_path) if f.endswith(".mp4")]
video_index = len(existing_files) + 1

base_filename = f"{today}_{video_index:02}"
output_path = os.path.join(folder_path, f"{base_filename}.mp4")
shutil.copy(result[0]['video'], output_path)
thumb_path = os.path.join(folder_path, f"{base_filename}_thumb.webp")


# === Step 5: 讀取原始 .webp 並儲存原圖與縮圖 ===
response = requests.get(data['items'][0]['url'])
image_data = io.BytesIO(response.content)
with Image.open(image_data) as img:
    # 儲存原圖
    width = img.width
    height = img.height

    # 建立縮圖
    thumbnail_width = 400
    ratio = thumbnail_width / img.width
    new_size = (thumbnail_width, int(img.height * ratio))

    thumb = img.convert("RGB").resize(new_size, Image.LANCZOS)
    thumb.save(thumb_path, "WEBP", quality=80)





# === Step 6: 更新 data.json ===
json_path = os.path.join(folder_path, "data.json")
timestamp = datetime.utcnow().isoformat() + "Z"

new_entry = {
    "filename": output_path,
    "thumb_path": thumb_path,
    "timestamp": timestamp
}

if os.path.exists(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
else:
    data = {"date": today, "videos": []}

data["videos"].append(new_entry)

with open(json_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"📄 data.json 已更新：{json_path}")



# === Step 7: 更新 README.md 每行最多顯示 10 張圖片 ===
readme_path = os.path.join(folder_path, "README.md")
image_files = sorted([f for f in os.listdir(folder_path) if ((f.endswith("_thumb.webp")))])

readme_lines = ["# Generated Images", ""]
row = []

for i, image_file in enumerate(image_files, 1):
    row.append(f'<img src="{image_file}" width="100"/>')
    if i % 9 == 0:
        readme_lines.append(" ".join(row))
        row = []

if row:
    readme_lines.append(" ".join(row))

with open(readme_path, "w") as f:
    f.write("\n\n".join(readme_lines))

print(f"📄 README.md 已更新：{readme_path}")
