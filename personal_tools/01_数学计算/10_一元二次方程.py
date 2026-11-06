while True:
    try:
        a=float(input('请输入二次项系数a '))
        b=float(input('请输入一次项系数b '))
        c=float(input('请输入常数项c '))
        d=b*b-4*a*c
        if a==0:
            print('a不能为零，请重新输入！')
            continue
        if d<0:
            print('此方程没有实数解')
            choice=input('是否继续计算？输入y继续，其他键退出：')
            continue
        if choice.lower()!='y':
            break
        else:
            x1=(-b+d**0.5)/(2*a)
            x2=(-b-d**0.5)/(2*a)
        print(f"此方程解x1= {x1}")
        print(f"此方程解x2= {x2}")
        
    except ValueError:
        print('输入的不是有效数字，请重新输入！‘')
        


