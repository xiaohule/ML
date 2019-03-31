def fibo_func(n):
    list=[0,1]
    while list[-1]<n:
        list.append(list[len(list)-2]+list[len(list)-1])
    return list
