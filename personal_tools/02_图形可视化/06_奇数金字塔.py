for i in range(1,19,2):
    if i<=9:
        print(' '*((9-i)//2)+i*str(i))
    else:
        print(' '*((i-8)//2)+(18-i)*str(18-i))
        
    