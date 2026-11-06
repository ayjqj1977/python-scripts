import os
import requests
from urllib.parse import urlparse
import time


def download_file(url, filename=None):
    """
    从指定URL下载文件
    """
    try:
        print(f"正在连接到: {url}")
        
        # 发送HTTP GET请求，流式下载
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()  # 检查请求是否成功
        
        # 如果没有指定文件名，从URL中提取
        if not filename:
            parsed_url = urlparse(response.url)
            filename = os.path.basename(parsed_url.path)
            if not filename:
                filename = "downloaded_file.pdf"
        
        # 获取文件大小
        total_size = int(response.headers.get('content-length', 0))
        
        # 开始下载文件
        print(f"开始下载文件: {filename}")
        if total_size > 0:
            print(f"文件大小: {total_size / (1024*1024):.2f} MB")
        
        with open(filename, 'wb') as file:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:  # 过滤掉保持连接的空块
                    file.write(chunk)
                    downloaded += len(chunk)
                    
                    # 显示进度
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\r下载进度: {percent:.1f}%", end='', flush=True)
        
        print(f"\n文件已成功下载: {filename}")
        return filename
        
    except requests.exceptions.RequestException as e:
        print(f"下载失败: {e}")
        return None
    except Exception as e:
        print(f"发生错误: {e}")
        return None


def download_from_pan_link(pan_url, password=None):
    """
    从网盘链接下载文件（适用于百度网盘、夸克网盘等）
    注意：对于需要登录或提取码的网盘链接，可能需要手动操作
    """
    print("注意：对于需要提取码的网盘链接，你需要先手动访问链接并输入提取码")
    print("然后将实际的PDF文件链接粘贴到程序中")
    print(f"网盘链接: {pan_url}")
    
    if password:
        print(f"提取码: {password}")
    
    print("请手动访问链接获取实际的PDF下载地址，然后运行本程序并使用该地址")
    

def search_local_resources():
    """
    搜索本地可能存在的相关资源
    """
    print("正在搜索本地可能存在的相关资源...")
    current_dir = os.getcwd()
    pdf_files = []
    
    for root, dirs, files in os.walk(current_dir):
        for file in files:
            if file.lower().endswith('.pdf') and '资治通鉴' in file:
                pdf_files.append(os.path.join(root, file))
    
    if pdf_files:
        print("在当前目录及子目录中找到以下相关PDF文件：")
        for pdf in pdf_files:
            size = os.path.getsize(pdf)
            print(f"  - {pdf} ({size / (1024*1024):.2f} MB)")
    else:
        print("在当前目录中未找到相关的资治通鉴PDF文件")


def main():
    print("=" * 60)
    print("资治通鉴 PDF下载工具")
    print("=" * 60)
    
    while True:
        print("\n请选择操作：")
        print("1. 从URL下载PDF文件")
        print("2. 从网盘链接下载（需手动获取下载地址）")
        print("3. 搜索本地相关资源")
        print("4. 退出")
        
        choice = input("\n请输入选项 (1-4): ").strip()
        
        if choice == "1":
            url = input("\n请输入PDF文件的直接下载链接: ").strip()
            if url:
                filename = input("请输入保存的文件名 (直接回车使用默认名称): ").strip()
                if not filename:
                    filename = None
                download_file(url, filename)
            else:
                print("URL不能为空")
        
        elif choice == "2":
            pan_url = input("\n请输入网盘链接 (如百度网盘、夸克网盘等): ").strip()
            if pan_url:
                password = input("请输入提取码 (没有则直接回车): ").strip()
                if not password:
                    password = None
                download_from_pan_link(pan_url, password)
            else:
                print("链接不能为空")
        
        elif choice == "3":
            search_local_resources()
        
        elif choice == "4":
            print("感谢使用，再见！")
            break
        
        else:
            print("无效选项，请重新选择")


if __name__ == "__main__":
    # 程序启动时显示一些搜索到的可能资源
    print("\n提示：根据搜索结果，以下是一些可能的《资治通鉴》PDF资源：")
    print("- 百度网盘链接: pan.baidu.com/s/1yuNfenTHO_0jowHcsTEhdQ?pwd=rpoh (提取码: rpoh)")
    print("- 夸克网盘链接: pan.quark.cn/s/8d82b33e4e15")
    print("- 中华书局版: 可能需要购买或通过合法途径获取")
    print("\n注意：由于版权原因，建议通过合法途径获取电子书资源")
    
    main()