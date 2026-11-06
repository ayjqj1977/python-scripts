import json

def verify_character_json(file_path):
    """
    验证字符向量JSON文件的完整性与正确性
    """
    # 1. 加载文件
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print("✅ JSON加载成功：语法无错误")
    except Exception as e:
        print(f"❌ JSON加载失败：{e}")
        return

    # 2. 验证meta信息
    meta = data.get("meta", {})
    required_meta = ["vector_dimensions", "total_count"]
    for key in required_meta:
        if key not in meta:
            print(f"⚠️ meta缺失字段：{key}")

    # 3. 验证字符列表
    chars = data.get("characters", [])
    actual_count = len(chars)
    meta_count = meta.get("total_count", 0)

    print(f"\n📊 元数据声明总数：{meta_count}")
    print(f"📊 实际字符数量：{actual_count}")

    if meta_count != actual_count:
        print(f"❌ 数量不匹配：元数据({meta_count}) ≠ 实际({actual_count})")
    else:
        print("✅ 数量匹配")

    # 4. 检查重复字符
    char_set = set()
    duplicate_chars = []
    for idx, item in enumerate(chars):
        c = item.get("char")
        if c in char_set:
            duplicate_chars.append((idx, c))
        else:
            char_set.add(c)

    if duplicate_chars:
        print(f"\n❌ 发现重复字符：")
        for idx, c in duplicate_chars:
            print(f"   行{idx+1}：'{c}'")
    else:
        print("\n✅ 无重复字符")

    # 5. 检查向量维度
    dim_error = []
    for idx, item in enumerate(chars):
        vec = item.get("vector", [])
        if len(vec) != 18:
            dim_error.append(f"行{idx+1}：{len(vec)}维（应18维）")

    if dim_error:
        print(f"\n❌ 向量维度错误：")
        for e in dim_error[:5]:  # 最多显示前5个
            print(f"   {e}")
    else:
        print("\n✅ 所有向量维度均为18维")

    # 6. 随机抽查3个字符（验证中文标点是否正常）
    print("\n🎯 随机验证（中文标点）：")
    cn_punct = [item for item in chars if item.get("category") == "punctuation_cn"][:3]
    for item in cn_punct:
        print(f"   字符:'{item['char']}' → 向量:{item['vector'][-2:]}")

if __name__ == "__main__":
    # 替换成你的文件名
    verify_character_json("符号向量字库.json")