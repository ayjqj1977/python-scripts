import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# 配置Hugging Face国内镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

def load_model(model_path="uer/gpt2-chinese-poem"):
    """加载模型并配置关键参数，消除警告"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # 手动设置pad_token，消除pad_token_id警告
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model = model.to("cpu")
    # 关闭模型的梯度计算，提升生成速度
    model.eval()
    return model, tokenizer

def smart_generate_text(
    model, tokenizer, 
    start_text="对酒当歌", 
    max_length=80,
    style="poem",  # 支持poem(诗词)/prose(散文)/couplet(对联)
    temperature=0.6,  # 越小越严谨，越大越有创意
    top_p=0.9,       # 核采样，控制生成的多样性
    repetition_penalty=1.2  # 抑制重复内容
):
    """智能文本生成函数，支持风格定制和参数调优"""
    # 根据风格调整生成配置
    gen_config = GenerationConfig(
        max_length=max_length,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        do_sample=True,
        num_return_sequences=1,
    )
    
    # 风格化前缀（引导模型生成对应风格）
    style_prefix = {
        "poem": "诗词：",
        "prose": "散文：",
        "couplet": "对联："
    }.get(style, "诗词：")
    input_text = style_prefix + start_text
    
    # 编码输入文本
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cpu")
    
    # 生成文本
    with torch.no_grad():  # 禁用梯度计算，提升速度
        output = model.generate(input_ids, generation_config=gen_config)
    
    # 解码并清洗结果（去除风格前缀）
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    generated_text = generated_text.replace(style_prefix, "").strip()
    
    # 中文文本后处理：去除多余标点，补全诗词句尾
    generated_text = polish_chinese_text(generated_text)
    return generated_text

def polish_chinese_text(text):
    """中文文本后处理，优化格式和韵律"""
    # 去除连续重复的标点
    import re
    text = re.sub(r'([，。！？；：])\1+', r'\1', text)
    # 确保句尾以诗词常用标点结尾
    end_puncts = ["。", "！", "？", "兮", "哉"]
    if text and text[-1] not in end_puncts:
        text += "。"
    return text

# 主程序
if __name__ == "__main__":
    model, tokenizer = load_model()
    # 生成诗词风格文本
    poem_text = smart_generate_text(
        model, tokenizer,
        start_text="对酒当歌",
        style="poem",
        max_length=100,
        temperature=0.5  # 诗词风格更严谨
    )
    print("智能生成的诗词：")
    print(poem_text)
    
    # 生成散文风格文本（对比效果）
    prose_text = smart_generate_text(
        model, tokenizer,
        start_text="对酒当歌",
        style="prose",
        max_length=150,
        temperature=0.7  # 散文更有创意
    )
    print("\n智能生成的散文：")
    print(prose_text)