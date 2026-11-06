import json

# 修正文件名
file_path = "符号向量字库.json"

try:
    data = json.load(open(file_path, "r", encoding="utf-8"))
    chars = data.get("characters", [])
    
    # 检测逻辑：允许同字符不同分类，只检测完全重复的（char和category都一样）
    seen = set()
    duplicates = []
    
    for item in chars:
        # 用元组组合作为Key（字符+分类），允许同字符不同分类存在
        key = (item["char"], item["category"])
        if key in seen:
            duplicates.append(item["char"])
        seen.add(key)
    
    if duplicates:
        print(f"发现真正的重复字符（同一字符同一分类）：{duplicates}")
    else:
        print("✅ 检测通过：无重复（已兼容同字符不同分类情况）")

except FileNotFoundError:
    print("❌ 错误：找不到文件，请检查路径是否正确")
except Exception as e:
    print(f"❌ 错误：{e}")