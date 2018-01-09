import secondary_module_mixed
import math

global_a = 1


def f_parent():
    local_a = 2

    return local_a

my_func = secondary_module_mixed.secondary_function
my_func2 = math.tan
my_func3 = secondary_module_mixed.time.time