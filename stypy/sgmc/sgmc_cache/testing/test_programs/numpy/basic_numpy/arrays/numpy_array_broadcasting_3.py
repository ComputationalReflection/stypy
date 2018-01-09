
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://cs231n.github.io/python-numpy-tutorial/
2: 
3: import numpy as np
4: 
5: # We will add the vector v to each row of the matrix x,
6: # storing the result in the matrix y
7: x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
8: v = np.array([1, 0, 1])
9: y = x + v  # Add v to each row of x using broadcasting
10: 
11: # l = globals().copy()
12: # for v in l:
13: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
14: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')
import_269 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_269) is not StypyTypeError):

    if (import_269 != 'pyd_module'):
        __import__(import_269)
        sys_modules_270 = sys.modules[import_269]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_270.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_269)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')


# Assigning a Call to a Name (line 7):

# Call to array(...): (line 7)
# Processing the call arguments (line 7)

# Obtaining an instance of the builtin type 'list' (line 7)
list_273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)

# Obtaining an instance of the builtin type 'list' (line 7)
list_274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
int_275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 14), list_274, int_275)
# Adding element type (line 7)
int_276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 14), list_274, int_276)
# Adding element type (line 7)
int_277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 14), list_274, int_277)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 13), list_273, list_274)
# Adding element type (line 7)

# Obtaining an instance of the builtin type 'list' (line 7)
list_278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
int_279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 25), list_278, int_279)
# Adding element type (line 7)
int_280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 25), list_278, int_280)
# Adding element type (line 7)
int_281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 25), list_278, int_281)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 13), list_273, list_278)
# Adding element type (line 7)

# Obtaining an instance of the builtin type 'list' (line 7)
list_282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 36), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
int_283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 37), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 36), list_282, int_283)
# Adding element type (line 7)
int_284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 40), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 36), list_282, int_284)
# Adding element type (line 7)
int_285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 43), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 36), list_282, int_285)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 13), list_273, list_282)
# Adding element type (line 7)

# Obtaining an instance of the builtin type 'list' (line 7)
list_286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 47), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
int_287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 48), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 47), list_286, int_287)
# Adding element type (line 7)
int_288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 52), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 47), list_286, int_288)
# Adding element type (line 7)
int_289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 56), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 47), list_286, int_289)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 13), list_273, list_286)

# Processing the call keyword arguments (line 7)
kwargs_290 = {}
# Getting the type of 'np' (line 7)
np_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'np', False)
# Obtaining the member 'array' of a type (line 7)
array_272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), np_271, 'array')
# Calling array(args, kwargs) (line 7)
array_call_result_291 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), array_272, *[list_273], **kwargs_290)

# Assigning a type to the variable 'x' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'x', array_call_result_291)

# Assigning a Call to a Name (line 8):

# Call to array(...): (line 8)
# Processing the call arguments (line 8)

# Obtaining an instance of the builtin type 'list' (line 8)
list_294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 8)
# Adding element type (line 8)
int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 13), list_294, int_295)
# Adding element type (line 8)
int_296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 13), list_294, int_296)
# Adding element type (line 8)
int_297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 13), list_294, int_297)

# Processing the call keyword arguments (line 8)
kwargs_298 = {}
# Getting the type of 'np' (line 8)
np_292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'np', False)
# Obtaining the member 'array' of a type (line 8)
array_293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 4), np_292, 'array')
# Calling array(args, kwargs) (line 8)
array_call_result_299 = invoke(stypy.reporting.localization.Localization(__file__, 8, 4), array_293, *[list_294], **kwargs_298)

# Assigning a type to the variable 'v' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'v', array_call_result_299)

# Assigning a BinOp to a Name (line 9):
# Getting the type of 'x' (line 9)
x_300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'x')
# Getting the type of 'v' (line 9)
v_301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'v')
# Applying the binary operator '+' (line 9)
result_add_302 = python_operator(stypy.reporting.localization.Localization(__file__, 9, 4), '+', x_300, v_301)

# Assigning a type to the variable 'y' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'y', result_add_302)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
