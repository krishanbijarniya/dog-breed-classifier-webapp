def calculate():
    a=10
    b=2

    for i in range(0, 1):
        a= -(i+b)
        b=a-b
    return b

result = calculate()
print(result)