# coding=utf-8
from numpy import add

class C:
    pass

print add

# Type error
r = add(C, C)