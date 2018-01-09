import other_module_to_import

def f():
    return 3

r1 = f()

r2 = other_module_to_import.global_a
r3 = other_module_to_import.f_parent()
r4 = other_module_to_import.my_func
r5 = other_module_to_import.secondary_module
r6 = r5.number