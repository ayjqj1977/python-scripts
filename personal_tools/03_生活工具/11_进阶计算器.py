def calculator():
    print("=== 控制台计算器 ===")
    print("支持的运算：+（加）、-（减）、*（乘）、/（除）")
    print("输入 'exit' 退出程序")
    print("-" * 30)
    
    while True:
        try:
            # 获取第一个数字
            num1_input = input("请输入第一个数字：")
            if num1_input.lower() == 'exit':
                print("感谢使用计算器，再见！")
                break
            num1 = float(num1_input)
            
            # 获取运算符
            operator = input("请输入运算符（+、-、*、/）：")
            if operator.lower() == 'exit':
                print("感谢使用计算器，再见！")
                break
            if operator not in ['+', '-', '*', '/']:
                print("错误：请输入有效的运算符（+、-、*、/）")
                continue
            
            # 获取第二个数字
            num2_input = input("请输入第二个数字：")
            if num2_input.lower() == 'exit':
                print("感谢使用计算器，再见！")
                break
            num2 = float(num2_input)
            
            # 执行计算
            if operator == '+':
                result = num1 + num2
                print(f"计算结果：{num1} + {num2} = {result}")
            elif operator == '-':
                result = num1 - num2
                print(f"计算结果：{num1} - {num2} = {result}")
            elif operator == '*':
                result = num1 * num2
                print(f"计算结果：{num1} * {num2} = {result}")
            elif operator == '/':
                if num2 == 0:
                    print("错误：除数不能为零！")
                    continue
                result = num1 / num2
                print(f"计算结果：{num1} / {num2} = {result}")
            print("-" * 30)
            
        except ValueError:
            print("错误：请输入有效的数字！")
            print("-" * 30)
        except Exception as e:
            print(f"发生意外错误：{e}")
            print("-" * 30)

if __name__ == "__main__":
    calculator()