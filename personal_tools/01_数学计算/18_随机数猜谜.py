import random
随机数=random.randint(1,1000)
while True:
    用户输入=input('请猜1-1000之间的数字(输入q退出):')
    if 用户输入.lower()=="q":
        print(f'退出游戏，正确数字是{随机数}')
        break
    try:
        a=int(用户输入)
        if a>1000 or a<1:
            print('输入错误请重输！')
            continue
        if a<随机数:
            print('小了哈')
        elif a>随机数:
            print('大了哈')
        else:
            print('你真棒！！！')
            break
    except ValueError:
        print('输入错误')

