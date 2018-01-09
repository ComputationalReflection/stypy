def f(x, y, z, *arguments, **kwarguments):
    pass

def f2(x, y, z, *arguments, **kwarguments):
    pass

def f3 (x=5, y=6, z=4, *args, **kwargs):
    pass


f(2, 3, 4, 5, 6, 7)
f2(1, 2, 8, 6, 4, r=23)

f3(z="1", x=4, y=True, r=11, s="12")
