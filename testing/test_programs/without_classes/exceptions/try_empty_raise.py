
def fun():
    return Exception()

try:
    a = 3
    raise fun()
except KeyError as k:
    raise
except Exception as k2:
    a = k2

z = None