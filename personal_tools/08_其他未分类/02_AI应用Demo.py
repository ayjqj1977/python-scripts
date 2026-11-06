# 导入修正版：使用更稳定、通用的方式加载文档
from langchain_core.documents import Document
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
import os

# 1. 初始化你的本地模型
llm = Ollama(base_url="http://localhost:11434", model="deepseek-r1:1.5b")  # 请确保模型名正确

# 2. 加载并处理知识文档（更健壮的方法）
def load_document(file_path):
    """加载文本、PDF或Word文档"""
    text = ""
    _, ext = os.path.splitext(file_path)

    try:
        if ext.lower() == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        elif ext.lower() == '.pdf':
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            text = "".join([page.extract_text() or '' for page in reader.pages])
        elif ext.lower() in ['.docx', '.doc']:
            from docx import Document as DocxDocument
            doc = DocxDocument(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
        else:
            print(f"暂不支持 {ext} 格式文件")
            return []
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return []

    # 将文本包装成LangChain的Document对象
    return [Document(page_content=text, metadata={"source": file_path})]

# 指定你的文档路径
documents = load_document("./my_doc.txt")  # 确保文件存在，或换成你的实际文件名

if not documents:
    print("未加载到任何文档，请检查文件路径和格式。")
    exit()

# 3. 分割文档
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = text_splitter.split_documents(documents)
print(f"文档已分割成 {len(split_docs)} 个片段")

# 4. 创建向量知识库
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")  # 推荐的中文嵌入模型
vectorstore = Chroma.from_documents(documents=split_docs, embedding=embeddings, persist_directory="./my_kb_db")
retriever = vectorstore.as_retriever()
print("向量知识库创建成功！")

# 5. 创建一个问答链
qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# 6. 提问示例
question = "我的文档中提到了哪些关键点？"
answer = qa_chain.run(question)
print(f"\n问题：{question}\n答案：{answer}")