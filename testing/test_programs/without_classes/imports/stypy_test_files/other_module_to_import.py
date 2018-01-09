from sub.submodule import *

global_a2 = 1


def f_parent2():
    local_a = 2

    return local_a

var1 = submodule_var
var2 = submodule_func()