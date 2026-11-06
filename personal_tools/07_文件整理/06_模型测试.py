# 用绝对路径指定Python环境的库加载（可选，防止环境混乱）
import sys
sys.executable = "D:/Python312/python.exe"

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

# 你的模型路径（注意是AI_Model，和截图里一致）
model_path = "D:/AI_Model/MiniCPM-V-2_6-int4"

# 初始化多模态管道，自动处理自定义代码
pipe = pipeline(
    task=Tasks.multimodal_question_answering,
    model=model_path,
    trust_remote_code=True,
    device="cpu"
)

# 测试纯文本问答
try:
    result = pipe({"question": "你能做什么？"})
    print("模型回答：", result["text"])
except Exception as e:
    print("运行提示：", str(e)[:300])
    print("如果还报错，直接用我帮你处理AI需求就好，别再折腾了！")