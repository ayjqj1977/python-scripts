import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import os

# 仅保留基础请求头，和第一次爬取的配置一致
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36",
    "Referer": "https://book.douban.com/"
}

# 简单的文本清洗，只去换行和多余空格
def clean_text(text):
    if not text:
        return "未知"
    return text.replace("\n", "").replace("  ", "").strip()

# 单页爬取：仅修复选择器错误，和第一次逻辑一致
def crawl_single_page(page_num):
    start = (page_num - 1) * 25
    url = f"https://book.douban.com/top250?start={start}"
    # 基础延时：1-2秒，和第一次爬取一致
    time.sleep(random.uniform(1, 2))
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.encoding = "utf-8"
        if response.status_code != 200:
            print(f"第{page_num}页请求失败，状态码：{response.status_code}")
            return []
    except Exception as e:
        print(f"第{page_num}页请求异常：{str(e)[:30]}")
        return []
    
    soup = BeautifulSoup(response.text, "html.parser")
    book_data = []
    # 修复：豆瓣TOP250的书籍项是tr.item，这是第一次爬取可能忽略的关键
    items = soup.select("tr.item")
    
    for item in items:
        try:
            title = clean_text(item.select_one("div.pl2 a").get_text())
            rating = clean_text(item.select_one("span.rating_nums").get_text())
            author = clean_text(item.select_one("p.pl").get_text()).split("/")[0]
            intro = clean_text(item.select_one("span.inq").get_text()) if item.select_one("span.inq") else "无简介"
            book_data.append({"书名": title, "评分": rating, "作者": author, "简介": intro})
        except:
            continue
    return book_data

# 主逻辑：仅保留基础的多页爬取和CSV保存（避免Excel库依赖）
def crawl_douban_top250(total_pages=10):
    all_books = []
    csv_path = "豆瓣TOP250_初代版.csv"
    # 删除旧文件
    if os.path.exists(csv_path):
        os.remove(csv_path)
    
    for page in range(1, total_pages + 1):
        print(f"开始爬取第{page}页...")
        page_data = crawl_single_page(page)
        if page_data:
            all_books.extend(page_data)
            print(f"第{page}页成功，获取{len(page_data)}本")
        else:
            print(f"第{page}页无数据")
    
    # 保存为CSV，完全规避Excel的库依赖和编码问题
    if all_books:
        pd.DataFrame(all_books).to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"\n爬取完成！共{len(all_books)}本，保存至{csv_path}")
    else:
        print("未爬取到任何数据")

if __name__ == "__main__":
    # 爬取10页，和第一次的配置一致
    crawl_douban_top250(total_pages=10)