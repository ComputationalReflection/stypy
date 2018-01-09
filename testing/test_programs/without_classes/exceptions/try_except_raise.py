
def fun():
    return Exception()

try:
    a = 3
    raise fun()
except KeyError as k:
    a = "3"
except Exception as k2:
    a = k2

z = None