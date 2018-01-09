import secondary_module

global_a = 1


def f_parent():
    local_a = 2

    return local_a

my_func = secondary_module.secondary_function
