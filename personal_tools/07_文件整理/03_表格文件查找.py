import os
import fnmatch
from pathlib import Path


def find_spreadsheet_files(root_dir):
    """
    在指定目录及其子目录中查找所有表格文件
    """
    # 定义常见的表格文件扩展名
    spreadsheet_extensions = [
        '*.csv',    # Comma Separated Values
        '*.tsv',    # Tab Separated Values
        '*.xls',    # Excel 97-2003
        '*.xlsx',   # Excel 2007+
        '*.xlsm',   # Excel with Macros
        '*.xlsb',   # Excel Binary
        '*.ods',    # OpenDocument Spreadsheet
        '*.gnumeric', # Gnumeric Spreadsheet
        '*.numbers'  # Apple Numbers
    ]
    
    found_files = []
    
    # 遍历目录树
    for root, dirs, files in os.walk(root_dir):
        # 检查当前目录下的所有文件
        for file in files:
            # 检查文件扩展名是否匹配表格文件格式
            for ext in spreadsheet_extensions:
                if fnmatch.fnmatch(file.lower(), ext):
                    full_path = os.path.join(root, file)
                    file_size = os.path.getsize(full_path)
                    found_files.append((full_path, file_size))
                    break  # 找到匹配后跳出循环，避免重复添加
    
    return found_files


def format_file_size(size_bytes):
    """
    将字节大小转换为更易读的格式
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"


def main():
    print("正在搜索电脑上的表格文件...")
    print("=" * 80)
    
    # 获取用户主目录
    home_dir = Path.home()
    print(f"搜索路径: {home_dir}")
    
    # 搜索表格文件
    spreadsheet_files = find_spreadsheet_files(home_dir)
    
    if spreadsheet_files:
        print(f"找到 {len(spreadsheet_files)} 个表格文件:")
        print("-" * 80)
        
        # 按文件大小排序（从大到小）
        spreadsheet_files.sort(key=lambda x: x[1], reverse=True)
        
        for file_path, file_size in spreadsheet_files:
            readable_size = format_file_size(file_size)
            print(f"{readable_size:>10} | {file_path}")
    else:
        print("未找到任何表格文件")
    
    print("=" * 80)
    print("搜索完成")


def search_specific_path(path):
    """
    搜索指定路径下的表格文件
    """
    print(f"正在搜索路径: {path}")
    print("=" * 80)
    
    if not os.path.exists(path):
        print(f"路径不存在: {path}")
        return
    
    spreadsheet_files = find_spreadsheet_files(path)
    
    if spreadsheet_files:
        print(f"找到 {len(spreadsheet_files)} 个表格文件:")
        print("-" * 80)
        
        # 按文件大小排序（从大到小）
        spreadsheet_files.sort(key=lambda x: x[1], reverse=True)
        
        for file_path, file_size in spreadsheet_files:
            readable_size = format_file_size(file_size)
            print(f"{readable_size:>10} | {file_path}")
    else:
        print("未找到任何表格文件")
    
    print("=" * 80)
    print("搜索完成")


if __name__ == "__main__":
    # 询问用户要搜索哪个路径
    print("表格文件搜索工具")
    print("1. 搜索整个用户目录 (推荐)")
    print("2. 搜索指定路径")
    
    choice = input("请选择搜索方式 (1 或 2): ").strip()
    
    if choice == "1":
        main()
    elif choice == "2":
        path = input("请输入要搜索的路径: ").strip()
        if path:
            search_specific_path(path)
        else:
            print("路径不能为空")
    else:
        print("无效选择，使用默认搜索方式")
        main()