import json
import time

bookmarks_path = r"C:\Users\30892\AppData\Local\Microsoft\Edge\User Data\Default\Bookmarks"

with open(bookmarks_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 重新计算并删除指定内容
delete_urls = ['fmhy.net', 'famous.usatour.com.cn']
delete_names = ['免费素材图片', '免抠图片网', '摄图网']

# 收藏夹栏 - 设计素材只保留这几个
keep_design_bar = ['588ku.com', 'nipic.com', '699pic.com', 'pixabay.com', 'pexels.com', 'zcool.com.cn', 'iconfont.cn']

# 处理收藏夹栏
bar = data['roots']['bookmark_bar']['children']
filtered_bar = []

for child in bar:
    if child['type'] != 'url':
        filtered_bar.append(child)
        continue
    
    url = child.get('url', '')
    name = child.get('name', '')
    
    # URL删除列表
    for d in delete_urls:
        if d in url:
            print(f"删除(URL): {name}")
            continue
    
    # 名称删除列表
    skip = False
    for d in delete_names:
        if d in name:
            print(f"删除(名): {name}")
            skip = True
            break
    if skip:
        continue
    
    filtered_bar.append(child)

data['roots']['bookmark_bar']['children'] = filtered_bar

# 处理移动收藏夹 - 精简设计素材类
synced = data['roots']['synced']['children']
filtered_synced = []

for child in synced:
    if child['type'] != 'url':
        filtered_synced.append(child)
        continue
    
    url = child.get('url', '')
    name = child.get('name', '')
    
    # URL删除
    for d in delete_urls:
        if d in url:
            print(f"删除同步: {name}")
            continue
    
    # 设计素材类精简
    is_design = any(k in url.lower() or k in name.lower() for k in 
        ['png', '素材', 'pic.', 'pexels', 'pixabay', 'zcool', '站酷', 'iconfont', '昵图', '摄图', '千库', 'vcg', '免抠', 'hippter', '堆友', '简单ai'])
    
    if is_design:
        keep = any(k in url.lower() for k in keep_design_bar)
        if not keep:
            print(f"精简设计: {name}")
            continue
    
    filtered_synced.append(child)

data['roots']['synced']['children'] = filtered_synced

# 写入
with open(bookmarks_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"\n收藏夹栏: {len(bar)} -> {len(filtered_bar)}")
print(f"移动收藏夹: {len(synced)} -> {len(filtered_synced)}")
print(f"总计: {len(bar) + len(synced)} -> {len(filtered_bar) + len(filtered_synced)}")
print("\n修改完成！")
