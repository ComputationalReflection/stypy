
# a: int
a = 5
# b: int
b = 6
# x: int
x = (1 if (a > b) else (-1))
# y: int
y = (1 if (a > b) else ((-1) if (a < b) else 0))
# z: int \/ str
z = (1 if (a > b) else 'foo')
print x
print y
print z