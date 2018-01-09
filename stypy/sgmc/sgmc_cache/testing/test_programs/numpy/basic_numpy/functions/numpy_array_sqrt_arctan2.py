
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: Z = np.random.random((10, 2))
6: X, Y = Z[:, 0], Z[:, 1]
7: R = np.sqrt(X ** 2 + Y ** 2)
8: T = np.arctan2(Y, X)
9: 
10: # l = globals().copy()
11: # for v in l:
12: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
13: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_311 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_311) is not StypyTypeError):

    if (import_311 != 'pyd_module'):
        __import__(import_311)
        sys_modules_312 = sys.modules[import_311]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_312.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_311)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Assigning a Call to a Name (line 5):

# Call to random(...): (line 5)
# Processing the call arguments (line 5)

# Obtaining an instance of the builtin type 'tuple' (line 5)
tuple_316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 22), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 5)
# Adding element type (line 5)
int_317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 22), tuple_316, int_317)
# Adding element type (line 5)
int_318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 22), tuple_316, int_318)

# Processing the call keyword arguments (line 5)
kwargs_319 = {}
# Getting the type of 'np' (line 5)
np_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'random' of a type (line 5)
random_314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_313, 'random')
# Obtaining the member 'random' of a type (line 5)
random_315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), random_314, 'random')
# Calling random(args, kwargs) (line 5)
random_call_result_320 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), random_315, *[tuple_316], **kwargs_319)

# Assigning a type to the variable 'Z' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'Z', random_call_result_320)

# Assigning a Tuple to a Tuple (line 6):

# Assigning a Subscript to a Name (line 6):

# Obtaining the type of the subscript
slice_321 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 6, 7), None, None, None)
int_322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 12), 'int')
# Getting the type of 'Z' (line 6)
Z_323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 7), 'Z')
# Obtaining the member '__getitem__' of a type (line 6)
getitem___324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 7), Z_323, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 6)
subscript_call_result_325 = invoke(stypy.reporting.localization.Localization(__file__, 6, 7), getitem___324, (slice_321, int_322))

# Assigning a type to the variable 'tuple_assignment_309' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'tuple_assignment_309', subscript_call_result_325)

# Assigning a Subscript to a Name (line 6):

# Obtaining the type of the subscript
slice_326 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 6, 16), None, None, None)
int_327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 21), 'int')
# Getting the type of 'Z' (line 6)
Z_328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 16), 'Z')
# Obtaining the member '__getitem__' of a type (line 6)
getitem___329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 16), Z_328, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 6)
subscript_call_result_330 = invoke(stypy.reporting.localization.Localization(__file__, 6, 16), getitem___329, (slice_326, int_327))

# Assigning a type to the variable 'tuple_assignment_310' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'tuple_assignment_310', subscript_call_result_330)

# Assigning a Name to a Name (line 6):
# Getting the type of 'tuple_assignment_309' (line 6)
tuple_assignment_309_331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'tuple_assignment_309')
# Assigning a type to the variable 'X' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'X', tuple_assignment_309_331)

# Assigning a Name to a Name (line 6):
# Getting the type of 'tuple_assignment_310' (line 6)
tuple_assignment_310_332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'tuple_assignment_310')
# Assigning a type to the variable 'Y' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 3), 'Y', tuple_assignment_310_332)

# Assigning a Call to a Name (line 7):

# Assigning a Call to a Name (line 7):

# Call to sqrt(...): (line 7)
# Processing the call arguments (line 7)
# Getting the type of 'X' (line 7)
X_335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 12), 'X', False)
int_336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 17), 'int')
# Applying the binary operator '**' (line 7)
result_pow_337 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 12), '**', X_335, int_336)

# Getting the type of 'Y' (line 7)
Y_338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 21), 'Y', False)
int_339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 26), 'int')
# Applying the binary operator '**' (line 7)
result_pow_340 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 21), '**', Y_338, int_339)

# Applying the binary operator '+' (line 7)
result_add_341 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 12), '+', result_pow_337, result_pow_340)

# Processing the call keyword arguments (line 7)
kwargs_342 = {}
# Getting the type of 'np' (line 7)
np_333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'np', False)
# Obtaining the member 'sqrt' of a type (line 7)
sqrt_334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), np_333, 'sqrt')
# Calling sqrt(args, kwargs) (line 7)
sqrt_call_result_343 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), sqrt_334, *[result_add_341], **kwargs_342)

# Assigning a type to the variable 'R' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'R', sqrt_call_result_343)

# Assigning a Call to a Name (line 8):

# Assigning a Call to a Name (line 8):

# Call to arctan2(...): (line 8)
# Processing the call arguments (line 8)
# Getting the type of 'Y' (line 8)
Y_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 15), 'Y', False)
# Getting the type of 'X' (line 8)
X_347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 18), 'X', False)
# Processing the call keyword arguments (line 8)
kwargs_348 = {}
# Getting the type of 'np' (line 8)
np_344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'np', False)
# Obtaining the member 'arctan2' of a type (line 8)
arctan2_345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 4), np_344, 'arctan2')
# Calling arctan2(args, kwargs) (line 8)
arctan2_call_result_349 = invoke(stypy.reporting.localization.Localization(__file__, 8, 4), arctan2_345, *[Y_346, X_347], **kwargs_348)

# Assigning a type to the variable 'T' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'T', arctan2_call_result_349)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
