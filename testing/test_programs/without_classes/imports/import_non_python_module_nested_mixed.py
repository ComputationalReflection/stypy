import other_module_to_import_mixed

def f():
    return 3

r1 = f()

r2 = other_module_to_import_mixed.global_a
r3 = other_module_to_import_mixed.f_parent()
r4 = other_module_to_import_mixed.my_func
r5 = other_module_to_import_mixed.secondary_module_mixed
r6 = r5.number


r7 = other_module_to_import_mixed.my_func2
r7b = r7(2)
r8 = other_module_to_import_mixed.my_func3
r8b = r8()
r9 = other_module_to_import_mixed.secondary_module_mixed
r10 = r9.time.clock()
r11 = other_module_to_import_mixed.secondary_module_mixed.time
r12 = other_module_to_import_mixed.secondary_module_mixed.clock_func
