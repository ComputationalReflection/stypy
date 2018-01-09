
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # https://docs.scipy.org/doc/numpy/reference/routines.math.html
2: 
3: import numpy as np
4: 
5: x = 2.1
6: 
7: # Hyperbolic functions
8: r1 = np.sinh(x)  # Hyperbolic sine, element-wise.
9: r2 = np.cosh(x)  # Hyperbolic cosine, element-wise.
10: r3 = np.tanh(x)  # Compute hyperbolic tangent element-wise.
11: r4 = np.arcsinh(x)  # Inverse hyperbolic sine element-wise.
12: r5 = np.arccosh(x)  # Inverse hyperbolic cosine, element-wise.
13: r6 = np.arctanh(x)  # Inverse hyperbolic tangent element-wise.
14: 
15: x = [0.1, 0.2, 0.3]
16: 
17: r7 = np.sinh(x)  # Hyperbolic sine, element-wise.
18: r8 = np.cosh(x)  # Hyperbolic cosine, element-wise.
19: r9 = np.tanh(x)  # Compute hyperbolic tangent element-wise.
20: r10 = np.arcsinh(x)  # Inverse hyperbolic sine element-wise.
21: r11 = np.arccosh(x)  # Inverse hyperbolic cosine, element-wise.
22: r12 = np.arctanh(x)  # Inverse hyperbolic tangent element-wise.
23: 
24: # l = globals().copy()
25: # for v in l:
26: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')
import_388 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_388) is not StypyTypeError):

    if (import_388 != 'pyd_module'):
        __import__(import_388)
        sys_modules_389 = sys.modules[import_388]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_389.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_388)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')


# Assigning a Num to a Name (line 5):
float_390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 4), 'float')
# Assigning a type to the variable 'x' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'x', float_390)

# Assigning a Call to a Name (line 8):

# Call to sinh(...): (line 8)
# Processing the call arguments (line 8)
# Getting the type of 'x' (line 8)
x_393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 13), 'x', False)
# Processing the call keyword arguments (line 8)
kwargs_394 = {}
# Getting the type of 'np' (line 8)
np_391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 5), 'np', False)
# Obtaining the member 'sinh' of a type (line 8)
sinh_392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 5), np_391, 'sinh')
# Calling sinh(args, kwargs) (line 8)
sinh_call_result_395 = invoke(stypy.reporting.localization.Localization(__file__, 8, 5), sinh_392, *[x_393], **kwargs_394)

# Assigning a type to the variable 'r1' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'r1', sinh_call_result_395)

# Assigning a Call to a Name (line 9):

# Call to cosh(...): (line 9)
# Processing the call arguments (line 9)
# Getting the type of 'x' (line 9)
x_398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 13), 'x', False)
# Processing the call keyword arguments (line 9)
kwargs_399 = {}
# Getting the type of 'np' (line 9)
np_396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 5), 'np', False)
# Obtaining the member 'cosh' of a type (line 9)
cosh_397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 5), np_396, 'cosh')
# Calling cosh(args, kwargs) (line 9)
cosh_call_result_400 = invoke(stypy.reporting.localization.Localization(__file__, 9, 5), cosh_397, *[x_398], **kwargs_399)

# Assigning a type to the variable 'r2' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'r2', cosh_call_result_400)

# Assigning a Call to a Name (line 10):

# Call to tanh(...): (line 10)
# Processing the call arguments (line 10)
# Getting the type of 'x' (line 10)
x_403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 13), 'x', False)
# Processing the call keyword arguments (line 10)
kwargs_404 = {}
# Getting the type of 'np' (line 10)
np_401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 5), 'np', False)
# Obtaining the member 'tanh' of a type (line 10)
tanh_402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 5), np_401, 'tanh')
# Calling tanh(args, kwargs) (line 10)
tanh_call_result_405 = invoke(stypy.reporting.localization.Localization(__file__, 10, 5), tanh_402, *[x_403], **kwargs_404)

# Assigning a type to the variable 'r3' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'r3', tanh_call_result_405)

# Assigning a Call to a Name (line 11):

# Call to arcsinh(...): (line 11)
# Processing the call arguments (line 11)
# Getting the type of 'x' (line 11)
x_408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 16), 'x', False)
# Processing the call keyword arguments (line 11)
kwargs_409 = {}
# Getting the type of 'np' (line 11)
np_406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 5), 'np', False)
# Obtaining the member 'arcsinh' of a type (line 11)
arcsinh_407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 5), np_406, 'arcsinh')
# Calling arcsinh(args, kwargs) (line 11)
arcsinh_call_result_410 = invoke(stypy.reporting.localization.Localization(__file__, 11, 5), arcsinh_407, *[x_408], **kwargs_409)

# Assigning a type to the variable 'r4' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'r4', arcsinh_call_result_410)

# Assigning a Call to a Name (line 12):

# Call to arccosh(...): (line 12)
# Processing the call arguments (line 12)
# Getting the type of 'x' (line 12)
x_413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 16), 'x', False)
# Processing the call keyword arguments (line 12)
kwargs_414 = {}
# Getting the type of 'np' (line 12)
np_411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 5), 'np', False)
# Obtaining the member 'arccosh' of a type (line 12)
arccosh_412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 5), np_411, 'arccosh')
# Calling arccosh(args, kwargs) (line 12)
arccosh_call_result_415 = invoke(stypy.reporting.localization.Localization(__file__, 12, 5), arccosh_412, *[x_413], **kwargs_414)

# Assigning a type to the variable 'r5' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'r5', arccosh_call_result_415)

# Assigning a Call to a Name (line 13):

# Call to arctanh(...): (line 13)
# Processing the call arguments (line 13)
# Getting the type of 'x' (line 13)
x_418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 16), 'x', False)
# Processing the call keyword arguments (line 13)
kwargs_419 = {}
# Getting the type of 'np' (line 13)
np_416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 5), 'np', False)
# Obtaining the member 'arctanh' of a type (line 13)
arctanh_417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 5), np_416, 'arctanh')
# Calling arctanh(args, kwargs) (line 13)
arctanh_call_result_420 = invoke(stypy.reporting.localization.Localization(__file__, 13, 5), arctanh_417, *[x_418], **kwargs_419)

# Assigning a type to the variable 'r6' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'r6', arctanh_call_result_420)

# Assigning a List to a Name (line 15):

# Obtaining an instance of the builtin type 'list' (line 15)
list_421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 15)
# Adding element type (line 15)
float_422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 5), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 4), list_421, float_422)
# Adding element type (line 15)
float_423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 10), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 4), list_421, float_423)
# Adding element type (line 15)
float_424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 4), list_421, float_424)

# Assigning a type to the variable 'x' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'x', list_421)

# Assigning a Call to a Name (line 17):

# Call to sinh(...): (line 17)
# Processing the call arguments (line 17)
# Getting the type of 'x' (line 17)
x_427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 13), 'x', False)
# Processing the call keyword arguments (line 17)
kwargs_428 = {}
# Getting the type of 'np' (line 17)
np_425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 5), 'np', False)
# Obtaining the member 'sinh' of a type (line 17)
sinh_426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 5), np_425, 'sinh')
# Calling sinh(args, kwargs) (line 17)
sinh_call_result_429 = invoke(stypy.reporting.localization.Localization(__file__, 17, 5), sinh_426, *[x_427], **kwargs_428)

# Assigning a type to the variable 'r7' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'r7', sinh_call_result_429)

# Assigning a Call to a Name (line 18):

# Call to cosh(...): (line 18)
# Processing the call arguments (line 18)
# Getting the type of 'x' (line 18)
x_432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 13), 'x', False)
# Processing the call keyword arguments (line 18)
kwargs_433 = {}
# Getting the type of 'np' (line 18)
np_430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 5), 'np', False)
# Obtaining the member 'cosh' of a type (line 18)
cosh_431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 5), np_430, 'cosh')
# Calling cosh(args, kwargs) (line 18)
cosh_call_result_434 = invoke(stypy.reporting.localization.Localization(__file__, 18, 5), cosh_431, *[x_432], **kwargs_433)

# Assigning a type to the variable 'r8' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'r8', cosh_call_result_434)

# Assigning a Call to a Name (line 19):

# Call to tanh(...): (line 19)
# Processing the call arguments (line 19)
# Getting the type of 'x' (line 19)
x_437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 13), 'x', False)
# Processing the call keyword arguments (line 19)
kwargs_438 = {}
# Getting the type of 'np' (line 19)
np_435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 5), 'np', False)
# Obtaining the member 'tanh' of a type (line 19)
tanh_436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 5), np_435, 'tanh')
# Calling tanh(args, kwargs) (line 19)
tanh_call_result_439 = invoke(stypy.reporting.localization.Localization(__file__, 19, 5), tanh_436, *[x_437], **kwargs_438)

# Assigning a type to the variable 'r9' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'r9', tanh_call_result_439)

# Assigning a Call to a Name (line 20):

# Call to arcsinh(...): (line 20)
# Processing the call arguments (line 20)
# Getting the type of 'x' (line 20)
x_442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 17), 'x', False)
# Processing the call keyword arguments (line 20)
kwargs_443 = {}
# Getting the type of 'np' (line 20)
np_440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 6), 'np', False)
# Obtaining the member 'arcsinh' of a type (line 20)
arcsinh_441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 6), np_440, 'arcsinh')
# Calling arcsinh(args, kwargs) (line 20)
arcsinh_call_result_444 = invoke(stypy.reporting.localization.Localization(__file__, 20, 6), arcsinh_441, *[x_442], **kwargs_443)

# Assigning a type to the variable 'r10' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'r10', arcsinh_call_result_444)

# Assigning a Call to a Name (line 21):

# Call to arccosh(...): (line 21)
# Processing the call arguments (line 21)
# Getting the type of 'x' (line 21)
x_447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 17), 'x', False)
# Processing the call keyword arguments (line 21)
kwargs_448 = {}
# Getting the type of 'np' (line 21)
np_445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 6), 'np', False)
# Obtaining the member 'arccosh' of a type (line 21)
arccosh_446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 6), np_445, 'arccosh')
# Calling arccosh(args, kwargs) (line 21)
arccosh_call_result_449 = invoke(stypy.reporting.localization.Localization(__file__, 21, 6), arccosh_446, *[x_447], **kwargs_448)

# Assigning a type to the variable 'r11' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'r11', arccosh_call_result_449)

# Assigning a Call to a Name (line 22):

# Call to arctanh(...): (line 22)
# Processing the call arguments (line 22)
# Getting the type of 'x' (line 22)
x_452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 17), 'x', False)
# Processing the call keyword arguments (line 22)
kwargs_453 = {}
# Getting the type of 'np' (line 22)
np_450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 6), 'np', False)
# Obtaining the member 'arctanh' of a type (line 22)
arctanh_451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 6), np_450, 'arctanh')
# Calling arctanh(args, kwargs) (line 22)
arctanh_call_result_454 = invoke(stypy.reporting.localization.Localization(__file__, 22, 6), arctanh_451, *[x_452], **kwargs_453)

# Assigning a type to the variable 'r12' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'r12', arctanh_call_result_454)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
