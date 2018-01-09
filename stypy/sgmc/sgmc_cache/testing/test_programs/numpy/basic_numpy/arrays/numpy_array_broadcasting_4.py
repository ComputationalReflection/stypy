
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://cs231n.github.io/python-numpy-tutorial/
2: 
3: import numpy as np
4: 
5: # Compute outer product of vectors
6: v = np.array([1, 2, 3])  # v has shape (3,)
7: w = np.array([4, 5])  # w has shape (2,)
8: # To compute an outer product, we first reshape v to be a column
9: # vector of shape (3, 1); we can then broadcast it against w to yield
10: # an output of shape (3, 2), which is the outer product of v and w:
11: # [[ 4  5]
12: #  [ 8 10]
13: #  [12 15]]
14: r = np.reshape(v, (3, 1)) * w
15: 
16: # Add a vector to each row of a matrix
17: x = np.array([[1, 2, 3], [4, 5, 6]])
18: # x has shape (2, 3) and v has shape (3,) so they broadcast to (2, 3),
19: # giving the following matrix:
20: # [[2 4 6]
21: #  [5 7 9]]
22: r2 = x + v
23: 
24: # Add a vector to each column of a matrix
25: # x has shape (2, 3) and w has shape (2,).
26: # If we transpose x then it has shape (3, 2) and can be broadcast
27: # against w to yield a result of shape (3, 2); transposing this result
28: # yields the final result of shape (2, 3) which is the matrix x with
29: # the vector w added to each column. Gives the following matrix:
30: # [[ 5  6  7]
31: #  [ 9 10 11]]
32: r3 = (x.T + w).T
33: # Another solution is to reshape w to be a row vector of shape (2, 1);
34: # we can then broadcast it directly against x to produce the same
35: # output.
36: r4 = x + np.reshape(w, (2, 1))
37: 
38: # Multiply a matrix by a constant:
39: # x has shape (2, 3). Numpy treats scalars as arrays of shape ();
40: # these can be broadcast together to shape (2, 3), producing the
41: # following array:
42: # [[ 2  4  6]
43: #  [ 8 10 12]]
44: r5 = x * 2
45: 
46: # l = globals().copy()
47: # for v in l:
48: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
49: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')
import_303 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_303) is not StypyTypeError):

    if (import_303 != 'pyd_module'):
        __import__(import_303)
        sys_modules_304 = sys.modules[import_303]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_304.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_303)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')


# Assigning a Call to a Name (line 6):

# Call to array(...): (line 6)
# Processing the call arguments (line 6)

# Obtaining an instance of the builtin type 'list' (line 6)
list_307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)
int_308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 13), list_307, int_308)
# Adding element type (line 6)
int_309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 13), list_307, int_309)
# Adding element type (line 6)
int_310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 13), list_307, int_310)

# Processing the call keyword arguments (line 6)
kwargs_311 = {}
# Getting the type of 'np' (line 6)
np_305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'np', False)
# Obtaining the member 'array' of a type (line 6)
array_306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 4), np_305, 'array')
# Calling array(args, kwargs) (line 6)
array_call_result_312 = invoke(stypy.reporting.localization.Localization(__file__, 6, 4), array_306, *[list_307], **kwargs_311)

# Assigning a type to the variable 'v' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'v', array_call_result_312)

# Assigning a Call to a Name (line 7):

# Call to array(...): (line 7)
# Processing the call arguments (line 7)

# Obtaining an instance of the builtin type 'list' (line 7)
list_315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
int_316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 13), list_315, int_316)
# Adding element type (line 7)
int_317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 13), list_315, int_317)

# Processing the call keyword arguments (line 7)
kwargs_318 = {}
# Getting the type of 'np' (line 7)
np_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'np', False)
# Obtaining the member 'array' of a type (line 7)
array_314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), np_313, 'array')
# Calling array(args, kwargs) (line 7)
array_call_result_319 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), array_314, *[list_315], **kwargs_318)

# Assigning a type to the variable 'w' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'w', array_call_result_319)

# Assigning a BinOp to a Name (line 14):

# Call to reshape(...): (line 14)
# Processing the call arguments (line 14)
# Getting the type of 'v' (line 14)
v_322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 15), 'v', False)

# Obtaining an instance of the builtin type 'tuple' (line 14)
tuple_323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 19), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 14)
# Adding element type (line 14)
int_324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 19), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 19), tuple_323, int_324)
# Adding element type (line 14)
int_325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 19), tuple_323, int_325)

# Processing the call keyword arguments (line 14)
kwargs_326 = {}
# Getting the type of 'np' (line 14)
np_320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'np', False)
# Obtaining the member 'reshape' of a type (line 14)
reshape_321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 4), np_320, 'reshape')
# Calling reshape(args, kwargs) (line 14)
reshape_call_result_327 = invoke(stypy.reporting.localization.Localization(__file__, 14, 4), reshape_321, *[v_322, tuple_323], **kwargs_326)

# Getting the type of 'w' (line 14)
w_328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 28), 'w')
# Applying the binary operator '*' (line 14)
result_mul_329 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 4), '*', reshape_call_result_327, w_328)

# Assigning a type to the variable 'r' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'r', result_mul_329)

# Assigning a Call to a Name (line 17):

# Call to array(...): (line 17)
# Processing the call arguments (line 17)

# Obtaining an instance of the builtin type 'list' (line 17)
list_332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 17)
# Adding element type (line 17)

# Obtaining an instance of the builtin type 'list' (line 17)
list_333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 17)
# Adding element type (line 17)
int_334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 14), list_333, int_334)
# Adding element type (line 17)
int_335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 14), list_333, int_335)
# Adding element type (line 17)
int_336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 14), list_333, int_336)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 13), list_332, list_333)
# Adding element type (line 17)

# Obtaining an instance of the builtin type 'list' (line 17)
list_337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 17)
# Adding element type (line 17)
int_338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 25), list_337, int_338)
# Adding element type (line 17)
int_339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 25), list_337, int_339)
# Adding element type (line 17)
int_340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 25), list_337, int_340)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 13), list_332, list_337)

# Processing the call keyword arguments (line 17)
kwargs_341 = {}
# Getting the type of 'np' (line 17)
np_330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'np', False)
# Obtaining the member 'array' of a type (line 17)
array_331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 4), np_330, 'array')
# Calling array(args, kwargs) (line 17)
array_call_result_342 = invoke(stypy.reporting.localization.Localization(__file__, 17, 4), array_331, *[list_332], **kwargs_341)

# Assigning a type to the variable 'x' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'x', array_call_result_342)

# Assigning a BinOp to a Name (line 22):
# Getting the type of 'x' (line 22)
x_343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 5), 'x')
# Getting the type of 'v' (line 22)
v_344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 9), 'v')
# Applying the binary operator '+' (line 22)
result_add_345 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 5), '+', x_343, v_344)

# Assigning a type to the variable 'r2' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'r2', result_add_345)

# Assigning a Attribute to a Name (line 32):
# Getting the type of 'x' (line 32)
x_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 6), 'x')
# Obtaining the member 'T' of a type (line 32)
T_347 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 6), x_346, 'T')
# Getting the type of 'w' (line 32)
w_348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'w')
# Applying the binary operator '+' (line 32)
result_add_349 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 6), '+', T_347, w_348)

# Obtaining the member 'T' of a type (line 32)
T_350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 6), result_add_349, 'T')
# Assigning a type to the variable 'r3' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'r3', T_350)

# Assigning a BinOp to a Name (line 36):
# Getting the type of 'x' (line 36)
x_351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 5), 'x')

# Call to reshape(...): (line 36)
# Processing the call arguments (line 36)
# Getting the type of 'w' (line 36)
w_354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 20), 'w', False)

# Obtaining an instance of the builtin type 'tuple' (line 36)
tuple_355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 24), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 36)
# Adding element type (line 36)
int_356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 24), tuple_355, int_356)
# Adding element type (line 36)
int_357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 24), tuple_355, int_357)

# Processing the call keyword arguments (line 36)
kwargs_358 = {}
# Getting the type of 'np' (line 36)
np_352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 9), 'np', False)
# Obtaining the member 'reshape' of a type (line 36)
reshape_353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 9), np_352, 'reshape')
# Calling reshape(args, kwargs) (line 36)
reshape_call_result_359 = invoke(stypy.reporting.localization.Localization(__file__, 36, 9), reshape_353, *[w_354, tuple_355], **kwargs_358)

# Applying the binary operator '+' (line 36)
result_add_360 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 5), '+', x_351, reshape_call_result_359)

# Assigning a type to the variable 'r4' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'r4', result_add_360)

# Assigning a BinOp to a Name (line 44):
# Getting the type of 'x' (line 44)
x_361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 5), 'x')
int_362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 9), 'int')
# Applying the binary operator '*' (line 44)
result_mul_363 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 5), '*', x_361, int_362)

# Assigning a type to the variable 'r5' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'r5', result_mul_363)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
