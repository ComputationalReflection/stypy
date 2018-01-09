# http://stackoverflow.com/questions/394809/does-python-have-a-ternary-conditional-operator

a = 5
b = 6

x = 1 if a > b else -1
y = 1 if a > b else -1 if a < b else 0

z = 1 if a > b else "foo"

print x
print y
print z
