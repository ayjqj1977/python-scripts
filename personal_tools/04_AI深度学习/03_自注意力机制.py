# 模拟：自注意力层
def self_attention_layer(input_tokens):
    print(f"自注意力处理输入：{input_tokens}")
    return [token + "_vec" for token in input_tokens]

# 修正：将列表转换为元组（可哈希）作为字典键
def predict_next_word(processed_vecs):
    # 把列表转成元组，作为字典的键
    key = tuple(processed_vecs)
    next_word_map = {
        ("今天_vec", "天气_vec"): "很",
        ("今天_vec", "天气_vec", "很_vec"): "好",
        ("今天_vec", "天气_vec", "很_vec", "好_vec"): "，",
        ("今天_vec", "天气_vec", "很_vec", "好_vec", "，_vec"): "适合",
        ("今天_vec", "天气_vec", "很_vec", "好_vec", "，_vec", "适合_vec"): "出游"
    }
    return next_word_map.get(key, "。")

# 自回归生成主流程
input_tokens = ["今天", "天气"]
max_length = 6

print("初始输入：", input_tokens)
for _ in range(max_length):
    processed_vecs = self_attention_layer(input_tokens)
    next_word = predict_next_word(processed_vecs)
    input_tokens.append(next_word)
    print("生成后输入：", input_tokens)

print("\n最终生成文本：", "".join(input_tokens))
