import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os

# 配置请求头，模拟浏览器
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Referer": "https://www.shuge.org"
}

# 书格网直接检索《资治通鉴》
def get_ztj_from_shuge():
    # 创建保存目录
    save_dir = "./资治通鉴_书格资源"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 书格搜索链接
    search_url = "https://www.shuge.org/search?q=资治通鉴"
    response = requests.get(search_url, headers=HEADERS, timeout=15)
    response.encoding = "utf-8"
    soup = BeautifulSoup(response.text, "html.parser")
    
    # 提取书籍链接
    book_items = soup.select(".book-list .book-item")
    if not book_items:
        print("书格网未找到《资治通鉴》相关资源，可手动访问 https://www.shuge.org 搜索")
        return
    
    print(f"在书格网找到{len(book_items)}个《资治通鉴》相关资源：")
    for idx, item in enumerate(book_items, 1):
        book_title = item.select_one(".book-title a").get_text(strip=True)
        book_url = urljoin(search_url, item.select_one(".book-title a")["href"])
        print(f"{idx}. {book_title}：{book_url}")
        
        # 访问书籍详情页，提取下载链接
        book_response = requests.get(book_url, headers=HEADERS, timeout=15)
        book_soup = BeautifulSoup(book_response.text, "html.parser")
        download_btn = book_soup.select_one(".book-download a")
        if download_btn:
            download_url = urljoin(book_url, download_btn["href"])
            # 保存下载链接到文本
            with open(f"{save_dir}/下载链接.txt", "a", encoding="utf-8") as f:
                f.write(f"{book_title}：{download_url}\n")
            print(f"   下载链接已保存：{book_title}")
        else:
            print(f"   未找到{book_title}的下载链接，需手动访问书籍页获取")

if __name__ == "__main__":
    print("开始从书格网检索《资治通鉴》资源...")
    get_ztj_from_shuge()
    print("检索完成，下载链接已保存至【资治通鉴_书格资源/下载链接.txt】，请手动复制链接到浏览器下载")