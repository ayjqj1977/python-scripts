import requests
import os

API_KEY = 'MUAbNciiYRHo7uMbi8BQI7LhA3a8ZgLFqUiBL5KXzoo'  # 免费注册获取
keyword = "sexy beauty"  # 关键词：性感美女
save_dir = "./性感美女"
os.makedirs(save_dir, exist_ok=True)

# 调用API获取图片链接
url = f"https://api.unsplash.com/search/photos?query={keyword}&per_page=20&client_id={API_KEY}"
response = requests.get(url).json()

# 批量下载
for idx, photo in enumerate(response["results"], 1):
    img_url = photo["urls"]["regular"]  # 图片链接
    img_data = requests.get(img_url).content
    with open(f"{save_dir}/{idx}.jpg", "wb") as f:
        f.write(img_data)
    print(f"下载完成：{idx}.jpg")