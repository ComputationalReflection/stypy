
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # https://docs.scipy.org/doc/numpy/reference/routines.math.html
2: 
3: 
4: import numpy as np
5: 
6: x = 2.1
7: x1 = 2.2
8: x2 = 2.3
9: 
10: # Floating point routines
11: r1 = np.signbit(x)  # Returns element-wise True where signbit is set (less than zero).
12: r2 = np.copysign(x1, x2)  # Change the sign of x1 to that of x2, element-wise.
13: r3 = np.frexp(x)  # Decompose the elements of x into mantissa and twos exponent.
14: #r4 = np.ldexp(x1, x2)  # Returns x1 * 2**x2, element-wise.
15: 
16: x = [2.1, 2.6]
17: x1 = [2.2, 2.7]
18: x2 = [2.3, 2.8]
19: 
20: r5 = np.signbit(x)  # Returns element-wise True where signbit is set (less than zero).
21: r6 = np.copysign(x1, x2)  # Change the sign of x1 to that of x2, element-wise.
22: r7 = np.frexp(x)  # Decompose the elements of x into mantissa and twos exponent.
23: #r8 = np.ldexp(x1, x2)  # Returns x1 * 2**x2, element-wise.
24: 
25: # l = globals().copy()
26: # for v in l:
27: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
28: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import numpy' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')
import_342 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy')

if (type(import_342) is not StypyTypeError):

    if (import_342 != 'pyd_module'):
        __import__(import_342)
        sys_modules_343 = sys.modules[import_342]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', sys_modules_343.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', import_342)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')


# Assigning a Num to a Name (line 6):
float_344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 4), 'float')
# Assigning a type to the variable 'x' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'x', float_344)

# Assigning a Num to a Name (line 7):
float_345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 5), 'float')
# Assigning a type to the variable 'x1' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'x1', float_345)

# Assigning a Num to a Name (line 8):
float_346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 5), 'float')
# Assigning a type to the variable 'x2' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'x2', float_346)

# Assigning a Call to a Name (line 11):

# Call to signbit(...): (line 11)
# Processing the call arguments (line 11)
# Getting the type of 'x' (line 11)
x_349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 16), 'x', False)
# Processing the call keyword arguments (line 11)
kwargs_350 = {}
# Getting the type of 'np' (line 11)
np_347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 5), 'np', False)
# Obtaining the member 'signbit' of a type (line 11)
signbit_348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 5), np_347, 'signbit')
# Calling signbit(args, kwargs) (line 11)
signbit_call_result_351 = invoke(stypy.reporting.localization.Localization(__file__, 11, 5), signbit_348, *[x_349], **kwargs_350)

# Assigning a type to the variable 'r1' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'r1', signbit_call_result_351)

# Assigning a Call to a Name (line 12):

# Call to copysign(...): (line 12)
# Processing the call arguments (line 12)
# Getting the type of 'x1' (line 12)
x1_354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 17), 'x1', False)
# Getting the type of 'x2' (line 12)
x2_355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 21), 'x2', False)
# Processing the call keyword arguments (line 12)
kwargs_356 = {}
# Getting the type of 'np' (line 12)
np_352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 5), 'np', False)
# Obtaining the member 'copysign' of a type (line 12)
copysign_353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 5), np_352, 'copysign')
# Calling copysign(args, kwargs) (line 12)
copysign_call_result_357 = invoke(stypy.reporting.localization.Localization(__file__, 12, 5), copysign_353, *[x1_354, x2_355], **kwargs_356)

# Assigning a type to the variable 'r2' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'r2', copysign_call_result_357)

# Assigning a Call to a Name (line 13):

# Call to frexp(...): (line 13)
# Processing the call arguments (line 13)
# Getting the type of 'x' (line 13)
x_360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 14), 'x', False)
# Processing the call keyword arguments (line 13)
kwargs_361 = {}
# Getting the type of 'np' (line 13)
np_358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 5), 'np', False)
# Obtaining the member 'frexp' of a type (line 13)
frexp_359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 5), np_358, 'frexp')
# Calling frexp(args, kwargs) (line 13)
frexp_call_result_362 = invoke(stypy.reporting.localization.Localization(__file__, 13, 5), frexp_359, *[x_360], **kwargs_361)

# Assigning a type to the variable 'r3' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'r3', frexp_call_result_362)

# Assigning a List to a Name (line 16):

# Obtaining an instance of the builtin type 'list' (line 16)
list_363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 16)
# Adding element type (line 16)
float_364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 5), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 4), list_363, float_364)
# Adding element type (line 16)
float_365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 10), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 4), list_363, float_365)

# Assigning a type to the variable 'x' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'x', list_363)

# Assigning a List to a Name (line 17):

# Obtaining an instance of the builtin type 'list' (line 17)
list_366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 17)
# Adding element type (line 17)
float_367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 6), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 5), list_366, float_367)
# Adding element type (line 17)
float_368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 11), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 5), list_366, float_368)

# Assigning a type to the variable 'x1' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'x1', list_366)

# Assigning a List to a Name (line 18):

# Obtaining an instance of the builtin type 'list' (line 18)
list_369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 5), 'list')
# Adding type elements to the builtin type 'list' instance (line 18)
# Adding element type (line 18)
float_370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 6), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 5), list_369, float_370)
# Adding element type (line 18)
float_371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 11), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 5), list_369, float_371)

# Assigning a type to the variable 'x2' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'x2', list_369)

# Assigning a Call to a Name (line 20):

# Call to signbit(...): (line 20)
# Processing the call arguments (line 20)
# Getting the type of 'x' (line 20)
x_374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'x', False)
# Processing the call keyword arguments (line 20)
kwargs_375 = {}
# Getting the type of 'np' (line 20)
np_372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 5), 'np', False)
# Obtaining the member 'signbit' of a type (line 20)
signbit_373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 5), np_372, 'signbit')
# Calling signbit(args, kwargs) (line 20)
signbit_call_result_376 = invoke(stypy.reporting.localization.Localization(__file__, 20, 5), signbit_373, *[x_374], **kwargs_375)

# Assigning a type to the variable 'r5' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'r5', signbit_call_result_376)

# Assigning a Call to a Name (line 21):

# Call to copysign(...): (line 21)
# Processing the call arguments (line 21)
# Getting the type of 'x1' (line 21)
x1_379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 17), 'x1', False)
# Getting the type of 'x2' (line 21)
x2_380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 21), 'x2', False)
# Processing the call keyword arguments (line 21)
kwargs_381 = {}
# Getting the type of 'np' (line 21)
np_377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 5), 'np', False)
# Obtaining the member 'copysign' of a type (line 21)
copysign_378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 5), np_377, 'copysign')
# Calling copysign(args, kwargs) (line 21)
copysign_call_result_382 = invoke(stypy.reporting.localization.Localization(__file__, 21, 5), copysign_378, *[x1_379, x2_380], **kwargs_381)

# Assigning a type to the variable 'r6' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'r6', copysign_call_result_382)

# Assigning a Call to a Name (line 22):

# Call to frexp(...): (line 22)
# Processing the call arguments (line 22)
# Getting the type of 'x' (line 22)
x_385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 14), 'x', False)
# Processing the call keyword arguments (line 22)
kwargs_386 = {}
# Getting the type of 'np' (line 22)
np_383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 5), 'np', False)
# Obtaining the member 'frexp' of a type (line 22)
frexp_384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 5), np_383, 'frexp')
# Calling frexp(args, kwargs) (line 22)
frexp_call_result_387 = invoke(stypy.reporting.localization.Localization(__file__, 22, 5), frexp_384, *[x_385], **kwargs_386)

# Assigning a type to the variable 'r7' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'r7', frexp_call_result_387)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
