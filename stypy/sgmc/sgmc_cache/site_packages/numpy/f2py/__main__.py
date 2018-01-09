
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # See http://cens.ioc.ee/projects/f2py2e/
2: from __future__ import division, print_function
3: 
4: import os
5: import sys
6: for mode in ["g3-numpy", "2e-numeric", "2e-numarray", "2e-numpy"]:
7:     try:
8:         i = sys.argv.index("--" + mode)
9:         del sys.argv[i]
10:         break
11:     except ValueError:
12:         pass
13: os.environ["NO_SCIPY_IMPORT"] = "f2py"
14: if mode == "g3-numpy":
15:     sys.stderr.write("G3 f2py support is not implemented, yet.\\n")
16:     sys.exit(1)
17: elif mode == "2e-numeric":
18:     from f2py2e import main
19: elif mode == "2e-numarray":
20:     sys.argv.append("-DNUMARRAY")
21:     from f2py2e import main
22: elif mode == "2e-numpy":
23:     from numpy.f2py import main
24: else:
25:     sys.stderr.write("Unknown mode: " + repr(mode) + "\\n")
26:     sys.exit(1)
27: main()
28: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import os' statement (line 4)
import os

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import sys' statement (line 5)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'sys', sys, module_type_store)



# Obtaining an instance of the builtin type 'list' (line 6)
list_99900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)
str_99901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 13), 'str', 'g3-numpy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 12), list_99900, str_99901)
# Adding element type (line 6)
str_99902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 25), 'str', '2e-numeric')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 12), list_99900, str_99902)
# Adding element type (line 6)
str_99903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 39), 'str', '2e-numarray')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 12), list_99900, str_99903)
# Adding element type (line 6)
str_99904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 54), 'str', '2e-numpy')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 12), list_99900, str_99904)

# Testing the type of a for loop iterable (line 6)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 6, 0), list_99900)
# Getting the type of the for loop variable (line 6)
for_loop_var_99905 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 6, 0), list_99900)
# Assigning a type to the variable 'mode' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'mode', for_loop_var_99905)
# SSA begins for a for statement (line 6)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')


# SSA begins for try-except statement (line 7)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')

# Assigning a Call to a Name (line 8):

# Call to index(...): (line 8)
# Processing the call arguments (line 8)
str_99909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 27), 'str', '--')
# Getting the type of 'mode' (line 8)
mode_99910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 34), 'mode', False)
# Applying the binary operator '+' (line 8)
result_add_99911 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 27), '+', str_99909, mode_99910)

# Processing the call keyword arguments (line 8)
kwargs_99912 = {}
# Getting the type of 'sys' (line 8)
sys_99906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 12), 'sys', False)
# Obtaining the member 'argv' of a type (line 8)
argv_99907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 12), sys_99906, 'argv')
# Obtaining the member 'index' of a type (line 8)
index_99908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 12), argv_99907, 'index')
# Calling index(args, kwargs) (line 8)
index_call_result_99913 = invoke(stypy.reporting.localization.Localization(__file__, 8, 12), index_99908, *[result_add_99911], **kwargs_99912)

# Assigning a type to the variable 'i' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'i', index_call_result_99913)
# Deleting a member
# Getting the type of 'sys' (line 9)
sys_99914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 12), 'sys')
# Obtaining the member 'argv' of a type (line 9)
argv_99915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 12), sys_99914, 'argv')

# Obtaining the type of the subscript
# Getting the type of 'i' (line 9)
i_99916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 21), 'i')
# Getting the type of 'sys' (line 9)
sys_99917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 12), 'sys')
# Obtaining the member 'argv' of a type (line 9)
argv_99918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 12), sys_99917, 'argv')
# Obtaining the member '__getitem__' of a type (line 9)
getitem___99919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 12), argv_99918, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 9)
subscript_call_result_99920 = invoke(stypy.reporting.localization.Localization(__file__, 9, 12), getitem___99919, i_99916)

del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 8), argv_99915, subscript_call_result_99920)
# SSA branch for the except part of a try statement (line 7)
# SSA branch for the except 'ValueError' branch of a try statement (line 7)
module_type_store.open_ssa_branch('except')
pass
# SSA join for try-except statement (line 7)
module_type_store = module_type_store.join_ssa_context()

# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()


# Assigning a Str to a Subscript (line 13):
str_99921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 32), 'str', 'f2py')
# Getting the type of 'os' (line 13)
os_99922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'os')
# Obtaining the member 'environ' of a type (line 13)
environ_99923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 0), os_99922, 'environ')
str_99924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 11), 'str', 'NO_SCIPY_IMPORT')
# Storing an element on a container (line 13)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 0), environ_99923, (str_99924, str_99921))


# Getting the type of 'mode' (line 14)
mode_99925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 3), 'mode')
str_99926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 11), 'str', 'g3-numpy')
# Applying the binary operator '==' (line 14)
result_eq_99927 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 3), '==', mode_99925, str_99926)

# Testing the type of an if condition (line 14)
if_condition_99928 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 0), result_eq_99927)
# Assigning a type to the variable 'if_condition_99928' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'if_condition_99928', if_condition_99928)
# SSA begins for if statement (line 14)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to write(...): (line 15)
# Processing the call arguments (line 15)
str_99932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 21), 'str', 'G3 f2py support is not implemented, yet.\\n')
# Processing the call keyword arguments (line 15)
kwargs_99933 = {}
# Getting the type of 'sys' (line 15)
sys_99929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'sys', False)
# Obtaining the member 'stderr' of a type (line 15)
stderr_99930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 4), sys_99929, 'stderr')
# Obtaining the member 'write' of a type (line 15)
write_99931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 4), stderr_99930, 'write')
# Calling write(args, kwargs) (line 15)
write_call_result_99934 = invoke(stypy.reporting.localization.Localization(__file__, 15, 4), write_99931, *[str_99932], **kwargs_99933)


# Call to exit(...): (line 16)
# Processing the call arguments (line 16)
int_99937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 13), 'int')
# Processing the call keyword arguments (line 16)
kwargs_99938 = {}
# Getting the type of 'sys' (line 16)
sys_99935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'sys', False)
# Obtaining the member 'exit' of a type (line 16)
exit_99936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 4), sys_99935, 'exit')
# Calling exit(args, kwargs) (line 16)
exit_call_result_99939 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), exit_99936, *[int_99937], **kwargs_99938)

# SSA branch for the else part of an if statement (line 14)
module_type_store.open_ssa_branch('else')


# Getting the type of 'mode' (line 17)
mode_99940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 5), 'mode')
str_99941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 13), 'str', '2e-numeric')
# Applying the binary operator '==' (line 17)
result_eq_99942 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 5), '==', mode_99940, str_99941)

# Testing the type of an if condition (line 17)
if_condition_99943 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 17, 5), result_eq_99942)
# Assigning a type to the variable 'if_condition_99943' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 5), 'if_condition_99943', if_condition_99943)
# SSA begins for if statement (line 17)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 4))

# 'from f2py2e import main' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_99944 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 4), 'f2py2e')

if (type(import_99944) is not StypyTypeError):

    if (import_99944 != 'pyd_module'):
        __import__(import_99944)
        sys_modules_99945 = sys.modules[import_99944]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 4), 'f2py2e', sys_modules_99945.module_type_store, module_type_store, ['main'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 4), __file__, sys_modules_99945, sys_modules_99945.module_type_store, module_type_store)
    else:
        from f2py2e import main

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 4), 'f2py2e', None, module_type_store, ['main'], [main])

else:
    # Assigning a type to the variable 'f2py2e' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'f2py2e', import_99944)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')

# SSA branch for the else part of an if statement (line 17)
module_type_store.open_ssa_branch('else')


# Getting the type of 'mode' (line 19)
mode_99946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 5), 'mode')
str_99947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 13), 'str', '2e-numarray')
# Applying the binary operator '==' (line 19)
result_eq_99948 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 5), '==', mode_99946, str_99947)

# Testing the type of an if condition (line 19)
if_condition_99949 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 19, 5), result_eq_99948)
# Assigning a type to the variable 'if_condition_99949' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 5), 'if_condition_99949', if_condition_99949)
# SSA begins for if statement (line 19)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to append(...): (line 20)
# Processing the call arguments (line 20)
str_99953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 20), 'str', '-DNUMARRAY')
# Processing the call keyword arguments (line 20)
kwargs_99954 = {}
# Getting the type of 'sys' (line 20)
sys_99950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'sys', False)
# Obtaining the member 'argv' of a type (line 20)
argv_99951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 4), sys_99950, 'argv')
# Obtaining the member 'append' of a type (line 20)
append_99952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 4), argv_99951, 'append')
# Calling append(args, kwargs) (line 20)
append_call_result_99955 = invoke(stypy.reporting.localization.Localization(__file__, 20, 4), append_99952, *[str_99953], **kwargs_99954)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 4))

# 'from f2py2e import main' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_99956 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 4), 'f2py2e')

if (type(import_99956) is not StypyTypeError):

    if (import_99956 != 'pyd_module'):
        __import__(import_99956)
        sys_modules_99957 = sys.modules[import_99956]
        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 4), 'f2py2e', sys_modules_99957.module_type_store, module_type_store, ['main'])
        nest_module(stypy.reporting.localization.Localization(__file__, 21, 4), __file__, sys_modules_99957, sys_modules_99957.module_type_store, module_type_store)
    else:
        from f2py2e import main

        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 4), 'f2py2e', None, module_type_store, ['main'], [main])

else:
    # Assigning a type to the variable 'f2py2e' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'f2py2e', import_99956)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')

# SSA branch for the else part of an if statement (line 19)
module_type_store.open_ssa_branch('else')


# Getting the type of 'mode' (line 22)
mode_99958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 5), 'mode')
str_99959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 13), 'str', '2e-numpy')
# Applying the binary operator '==' (line 22)
result_eq_99960 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 5), '==', mode_99958, str_99959)

# Testing the type of an if condition (line 22)
if_condition_99961 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 22, 5), result_eq_99960)
# Assigning a type to the variable 'if_condition_99961' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 5), 'if_condition_99961', if_condition_99961)
# SSA begins for if statement (line 22)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 4))

# 'from numpy.f2py import main' statement (line 23)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_99962 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 23, 4), 'numpy.f2py')

if (type(import_99962) is not StypyTypeError):

    if (import_99962 != 'pyd_module'):
        __import__(import_99962)
        sys_modules_99963 = sys.modules[import_99962]
        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 4), 'numpy.f2py', sys_modules_99963.module_type_store, module_type_store, ['main'])
        nest_module(stypy.reporting.localization.Localization(__file__, 23, 4), __file__, sys_modules_99963, sys_modules_99963.module_type_store, module_type_store)
    else:
        from numpy.f2py import main

        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 4), 'numpy.f2py', None, module_type_store, ['main'], [main])

else:
    # Assigning a type to the variable 'numpy.f2py' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'numpy.f2py', import_99962)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')

# SSA branch for the else part of an if statement (line 22)
module_type_store.open_ssa_branch('else')

# Call to write(...): (line 25)
# Processing the call arguments (line 25)
str_99967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 21), 'str', 'Unknown mode: ')

# Call to repr(...): (line 25)
# Processing the call arguments (line 25)
# Getting the type of 'mode' (line 25)
mode_99969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 45), 'mode', False)
# Processing the call keyword arguments (line 25)
kwargs_99970 = {}
# Getting the type of 'repr' (line 25)
repr_99968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 40), 'repr', False)
# Calling repr(args, kwargs) (line 25)
repr_call_result_99971 = invoke(stypy.reporting.localization.Localization(__file__, 25, 40), repr_99968, *[mode_99969], **kwargs_99970)

# Applying the binary operator '+' (line 25)
result_add_99972 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 21), '+', str_99967, repr_call_result_99971)

str_99973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 53), 'str', '\\n')
# Applying the binary operator '+' (line 25)
result_add_99974 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 51), '+', result_add_99972, str_99973)

# Processing the call keyword arguments (line 25)
kwargs_99975 = {}
# Getting the type of 'sys' (line 25)
sys_99964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'sys', False)
# Obtaining the member 'stderr' of a type (line 25)
stderr_99965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 4), sys_99964, 'stderr')
# Obtaining the member 'write' of a type (line 25)
write_99966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 4), stderr_99965, 'write')
# Calling write(args, kwargs) (line 25)
write_call_result_99976 = invoke(stypy.reporting.localization.Localization(__file__, 25, 4), write_99966, *[result_add_99974], **kwargs_99975)


# Call to exit(...): (line 26)
# Processing the call arguments (line 26)
int_99979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 13), 'int')
# Processing the call keyword arguments (line 26)
kwargs_99980 = {}
# Getting the type of 'sys' (line 26)
sys_99977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'sys', False)
# Obtaining the member 'exit' of a type (line 26)
exit_99978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 4), sys_99977, 'exit')
# Calling exit(args, kwargs) (line 26)
exit_call_result_99981 = invoke(stypy.reporting.localization.Localization(__file__, 26, 4), exit_99978, *[int_99979], **kwargs_99980)

# SSA join for if statement (line 22)
module_type_store = module_type_store.join_ssa_context()

# SSA join for if statement (line 19)
module_type_store = module_type_store.join_ssa_context()

# SSA join for if statement (line 17)
module_type_store = module_type_store.join_ssa_context()

# SSA join for if statement (line 14)
module_type_store = module_type_store.join_ssa_context()


# Call to main(...): (line 27)
# Processing the call keyword arguments (line 27)
kwargs_99983 = {}
# Getting the type of 'main' (line 27)
main_99982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'main', False)
# Calling main(args, kwargs) (line 27)
main_call_result_99984 = invoke(stypy.reporting.localization.Localization(__file__, 27, 0), main_99982, *[], **kwargs_99983)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
