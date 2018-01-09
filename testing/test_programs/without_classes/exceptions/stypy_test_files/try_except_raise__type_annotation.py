
# fun: function
# fun() -> Exception 

def fun():
    return Exception()

try:
    # a: int
    a = 3
    raise fun()
except KeyError as k:
    # a: str
    a = '3'
except Exception as k2:
    # a: Exception
    a = k2
# z: NoneType
z = None