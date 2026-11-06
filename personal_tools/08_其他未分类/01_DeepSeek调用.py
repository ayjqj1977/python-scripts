from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# 替换为你D盘的模型文件夹路径
model_path = "D:/AI_Models/MiniCPM-V-2_6-int4"

# 初始化图文问答管道
pipe = pipeline(
    task=Tasks.multimodal_question_answering,
    model=model_path,
    model_revision="master"
)

# 测试1：图像描述（替换为你的图片路径，如本地一张风景图）
image_path = "test.jpg"  # 可换成你电脑里的任意图片路径
result1 = pipe({"image": image_path, "question": "这张图片里有什么？"})
print("图像描述结果：", result1["text"])

# 测试2：纯文本问答
result2 = pipe({"question": "介绍一下你自己"})
print("文本问答结果：", result2["text"])