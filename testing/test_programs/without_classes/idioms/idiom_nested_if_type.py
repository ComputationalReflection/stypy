theInt = 3
theStr = "hi"
theBool = True
theComplex = complex(1, 2)
if True:
    union = 3
else:
    union = "hi"

def idiom(a):
    """
    if type(a) is int: return 3
    if type(a) is str: return "hi"
    return True
    """
    if type(a) is int:
        result = 3
    else:
        if type(a) is str:
            result = "hi"
        else:
            result = True
    return result

bigUnion = 3 if True else "a" if False else True
intOrBool = int() if True else False
intStrComplex = int() if True else str() if False else complex()

r = idiom(theInt)
r2 = idiom(theStr)
r3 = idiom(union)
r4 = idiom(theBool)
r5 = idiom(theComplex)
r6 = idiom(intOrBool)
r7 = idiom(bigUnion)
r8 = idiom(intStrComplex)
