# 导入随机数模块，用于生成随机目标数字（核心依赖）
import random

def number_guessing_game():
    # 打印游戏欢迎语，明确规则
    print("🎮 数字猜谜游戏开始！")
    print("我已经想好了一个1-100之间的整数，快来猜猜看吧～")
    
    # 生成1-100之间的随机整数作为目标数字（randint包含首尾值）
    target_number = random.randint(1, 100)
    
    # 无限循环，直到猜对才退出（游戏核心逻辑）
    while True:
        # 获取用户输入，并用try-except捕获非数字输入错误
        try:
            # 转换输入为整数（确保比较逻辑有效）
            guess = int(input("\n请输入你猜的数字（1-100）："))
            
            # 条件判断：对比猜测值与目标值，给出对应提示
            if guess < target_number:
                print("Too low! 再猜一次～")  # 保持原游戏提示风格
            elif guess > target_number:
                print("Too high! 再猜一次～")
            else:
                # 猜对时打印祝贺语，跳出循环结束本轮游戏
                print(f"🎉 That's it! 你猜对啦，目标数字就是{target_number}！")
                break
        
        # 处理用户输入非数字的情况（提升游戏容错性）
        except ValueError:
            print("❌ 输入错误！请输入1-100之间的整数哦～")

# 调用函数启动游戏
number_guessing_game()

# 询问是否再次游玩（完善游戏闭环）
while True:
    play_again = input("\n还要再玩一次吗？（yes/no）：").lower()
    if play_again == "yes":
        number_guessing_game()  # 重新启动游戏
    elif play_again == "no":
        print("👋 Thanks for playing! 下次再见～")
        break
    else:
        print("❌ 输入错误！请输入yes或no～")
