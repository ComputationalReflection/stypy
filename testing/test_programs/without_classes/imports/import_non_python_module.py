import module_to_import

def f():
    return 3

z = f()

x = module_to_import.global_a
y = module_to_import.f_parent()
