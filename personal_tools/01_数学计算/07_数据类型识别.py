import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置中文支持的字体（如SimHei）
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
# 定义与运算
a = 10
b = -5
print(a + b)  # 加法：5
print(a * b)  # 乘法：-50
print(a // b)  # 整除：-2（向下取整）
print(a % b)   # 取模：0
print(a ** 2)  # 幂运算：100

# 进制转换
print(bin(10))  # 二进制：0b1010
print(hex(10))  # 十六进制：0xa
print(int("1010", 2))  # 二进制转十进制：10
# 定义与运算
x = 3.14
y = 2.5
print(x + y)  # 5.64
print(x * y)  # 7.85
print(round(x, 1))  # 保留1位小数：3.1
print(float("2.718"))  # 字符串转浮点数：2.718

# 注意：浮点数精度问题
print(0.1 + 0.2)  # 0.30000000000000004
print(round(0.1 + 0.2, 1))  # 解决：0.3
# 定义与运算
c1 = 5 + 7j
c2 = -2 + 8j
print(c1 + c2)  # 加法：(3+15j)
print(c1 * c2)  # 乘法：(-66+26j)
print(c1.conjugate())  # 共轭复数：(5-7j)

# 提取实部/虚部
print(c1.real)  # 实部：5.0
print(c1.imag)  # 虚部：7.0
print(abs(c1))  # 模长（根号(5²+7²)）：8.602325267042627
# 定义
s1 = "Hello"
s2 = 'Python'
s3 = """多行
字符串"""

# 拼接与重复
print(s1 + " " + s2)  # 拼接：Hello Python
print(s1 * 2)         # 重复：HelloHello

# 索引与切片
print(s1[0])          # 第1个字符：H
print(s1[-1])         # 最后1个字符：o
print(s1[1:4])        # 切片（左闭右开）：ell

# 常用方法
print(s1.lower())     # 小写：hello
print(s2.upper())     # 大写：PYTHON
print("  test  ".strip())  # 去除首尾空格：test
print(s1.replace("e", "a"))  # 替换：Hallo
print(s2.split("y"))  # 分割：['P', 'thon']
print(".".join([s1, s2]))  # 连接：Hello.Python
# 定义
flag1 = True
flag2 = False

# 逻辑运算
print(flag1 and flag2)  # 与：False
print(flag1 or flag2)   # 或：True
print(not flag1)        # 非：False

# 类型转换（0/空值为False，非0/非空为True）
print(bool(0))     # False
print(bool(10))    # True
print(bool(""))    # False
print(bool("abc")) # True
print(bool(None))  # False
# 定义（仅None一个值）
res = None
print(res is None)  # 判断是否为None：True

# 常用场景：函数默认返回值、初始化变量
def func():
    pass  # 无返回值时默认返回None
print(func())  # None
# 定义
lst = [1, "a", True, 3.14]

# 增删改查
lst.append(5)          # 末尾添加：[1, 'a', True, 3.14, 5]
lst.insert(1, "b")     # 指定位置插入：[1, 'b', 'a', True, 3.14, 5]
lst.remove("a")        # 删除指定元素：[1, 'b', True, 3.14, 5]
print(lst.pop(2))      # 弹出指定索引元素：True
lst[0] = 100           # 修改元素：[100, 'b', 3.14, 5]

# 切片与排序
print(lst[1:3])        # 切片：['b', 3.14]
num_lst = [3, 1, 4, 2]
num_lst.sort()         # 排序：[1, 2, 3, 4]
print(num_lst[::-1])   # 反转：[4, 3, 2, 1]

# 列表推导式
square_lst = [x**2 for x in range(5)]
print(square_lst)      # [0, 1, 4, 9, 16]
# 定义（单元素需加逗号）
t1 = (1, 2, 3)
t2 = ("a",)  # 单元素元组
t3 = 4, 5, 6  # 省略括号也可

# 查（无增删改方法）
print(t1[0])        # 索引：1
print(t1[1:3])      # 切片：(2, 3)
print(t1.count(2))  # 统计元素出现次数：1
print(t1.index(3))  # 查找元素索引：2

# 不可变性（以下代码会报错）
# t1[0] = 100  # TypeError: 'tuple' object does not support item assignment
# 定义
dict1 = {"name": "Alice", "age": 18, "gender": "female"}
dict2 = dict(name="Bob", age=20)  # 另一种定义方式

# 增删改查
dict1["score"] = 90  # 添加键值对：{'name': 'Alice', 'age': 18, 'gender': 'female', 'score': 90}
dict1["age"] = 19    # 修改值：{'name': 'Alice', 'age': 19, 'gender': 'female', 'score': 90}
print(dict1.pop("gender"))  # 弹出键值对：female
del dict1["score"]   # 删除键值对

print(dict1.get("name"))  # 安全获取值：Alice
print(dict1.keys())       # 获取所有键：dict_keys(['name', 'age'])
print(dict1.values())     # 获取所有值：dict_values(['Alice', 19])
print(dict1.items())      # 获取所有键值对：dict_items([('name', 'Alice'), ('age', 19)])

# 字典推导式
num_dict = {x: x*2 for x in range(3)}
print(num_dict)  # {0: 0, 1: 2, 2: 4}
# 定义（不可用列表/字典作为元素）
s = {1, 2, 3, 3}  # 自动去重：{1, 2, 3}
s2 = set([4, 5, 6])  # 列表转集合：{4, 5, 6}

# 增删
s.add(4)           # 添加元素：{1, 2, 3, 4}
s.discard(2)       # 删除元素（不存在不报错）：{1, 3, 4}
s.remove(3)        # 删除元素（不存在报错）：{1, 4}

# 集合运算
s1 = {1, 2, 3}
s2 = {3, 4, 5}
print(s1 & s2)     # 交集：{3}
print(s1 | s2)     # 并集：{1, 2, 3, 4, 5}
print(s1 - s2)     # 差集：{1, 2}
print(s1 ^ s2)     # 对称差集（互不包含）：{1, 2, 4, 5}
# 定义
b1 = b"hello"  # 字符串前加b
b2 = bytes("中文", encoding="utf-8")  # 字符串转字节

# 转换
print(b1.decode("utf-8"))  # 字节转字符串：hello
print(b2.decode("utf-8"))  # 字节转字符串：中文
# 定义函数（特殊类型，可赋值给变量）
def add(a, b):
    return a + b

func_var = add  # 函数赋值给变量
print(func_var(2, 3))  # 调用：5
print(type(add))  # 类型：<class 'function'>
