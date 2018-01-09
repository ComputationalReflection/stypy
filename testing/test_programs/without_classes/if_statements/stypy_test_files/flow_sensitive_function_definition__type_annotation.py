
# a: int
a = 3
# condition: bool
condition = (a > 0)

if condition:
    # f: function
    f = (lambda x: x)
else:
    # f: function
    f = (lambda x, y: (x + y))

# y: TypeError; x: int
f(1)
# y: TypeError; x: TypeError
f()