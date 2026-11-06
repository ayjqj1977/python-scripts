while True:
    b=input('请输入任意数字（输入q退出）:')
    if b.lower()=="q":
        print('程序已退出')
        break
    try:
        a=int(b)
        if a>0:
            print(f"这是大于零的{a}")
        elif a<0:
            print(f"这是小于零的负{-a}")
        else:
            print("这是鸭蛋哈哈")
    except ValueError:
        print('输入错误')




