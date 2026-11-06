from flask import Flask, request, jsonify
from flask_cors import CORS  # 跨域配置
import os

app = Flask(__name__)
CORS(app)  # 开启跨域，解决通信限制

# 待办文件路径
TODO_FILE = "todo.txt"

# 初始化文件（如果不存在就创建）
if not os.path.exists(TODO_FILE):
    with open(TODO_FILE, "w", encoding="utf-8") as f:
        pass

# 获取所有待办
@app.route('/api/todos', methods=['GET'])
def get_todos():
    with open(TODO_FILE, "r", encoding="utf-8") as f:
        todos = [line.strip() for line in f if line.strip()]
    return jsonify({"todos": todos})

# 添加待办
@app.route('/api/todos', methods=['POST'])
def add_todo():
    data = request.get_json()
    todo = data.get("todo", "").strip()
    if todo:
        with open(TODO_FILE, "a", encoding="utf-8") as f:
            f.write(todo + "\n")
        return jsonify({"success": True})
    return jsonify({"success": False})

# 删除待办
@app.route('/api/todos/<int:index>', methods=['DELETE'])
def delete_todo(index):
    with open(TODO_FILE, "r", encoding="utf-8") as f:
        todos = [line.strip() for line in f if line.strip()]
    if 0 <= index < len(todos):
        todos.pop(index)
        with open(TODO_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(todos))
        return jsonify({"success": True})
    return jsonify({"success": False})

if __name__ == '__main__':
    app.run(debug=True)