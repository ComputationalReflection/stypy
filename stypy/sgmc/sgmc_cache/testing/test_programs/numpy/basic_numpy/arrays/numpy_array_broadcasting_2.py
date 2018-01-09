
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
9: vv = np.tile(v, (4, 1))  # Stack 4 copies of v on top of each other
10: y = x + vv  # Add x and vv elementwise
11: 
12: # l = globals().copy()
13: # for v in l:
14: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
15: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')
import_227 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_227) is not StypyTypeError):

    if (import_227 != 'pyd_module'):
        __import__(import_227)
        sys_modules_228 = sys.modules[import_227]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_228.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_227)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')


# Assigning a Call to a Name (line 7):

# Call to array(...): (line 7)
# Processing the call arguments (line 7)

# Obtaining an instance of the builtin type 'list' (line 7)
list_231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)

# Obtaining an instance of the builtin type 'list' (line 7)
list_232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
int_233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 14), list_232, int_233)
# Adding element type (line 7)
int_234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 14), list_232, int_234)
# Adding element type (line 7)
int_235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 14), list_232, int_235)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 13), list_231, list_232)
# Adding element type (line 7)

# Obtaining an instance of the builtin type 'list' (line 7)
list_236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
int_237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 25), list_236, int_237)
# Adding element type (line 7)
int_238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 25), list_236, int_238)
# Adding element type (line 7)
int_239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 25), list_236, int_239)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 13), list_231, list_236)
# Adding element type (line 7)

# Obtaining an instance of the builtin type 'list' (line 7)
list_240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 36), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
int_241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 37), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 36), list_240, int_241)
# Adding element type (line 7)
int_242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 40), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 36), list_240, int_242)
# Adding element type (line 7)
int_243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 43), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 36), list_240, int_243)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 13), list_231, list_240)
# Adding element type (line 7)

# Obtaining an instance of the builtin type 'list' (line 7)
list_244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 47), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
int_245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 48), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 47), list_244, int_245)
# Adding element type (line 7)
int_246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 52), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 47), list_244, int_246)
# Adding element type (line 7)
int_247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 56), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 47), list_244, int_247)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 13), list_231, list_244)

# Processing the call keyword arguments (line 7)
kwargs_248 = {}
# Getting the type of 'np' (line 7)
np_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'np', False)
# Obtaining the member 'array' of a type (line 7)
array_230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), np_229, 'array')
# Calling array(args, kwargs) (line 7)
array_call_result_249 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), array_230, *[list_231], **kwargs_248)

# Assigning a type to the variable 'x' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'x', array_call_result_249)

# Assigning a Call to a Name (line 8):

# Call to array(...): (line 8)
# Processing the call arguments (line 8)

# Obtaining an instance of the builtin type 'list' (line 8)
list_252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 8)
# Adding element type (line 8)
int_253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 13), list_252, int_253)
# Adding element type (line 8)
int_254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 13), list_252, int_254)
# Adding element type (line 8)
int_255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 13), list_252, int_255)

# Processing the call keyword arguments (line 8)
kwargs_256 = {}
# Getting the type of 'np' (line 8)
np_250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'np', False)
# Obtaining the member 'array' of a type (line 8)
array_251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 4), np_250, 'array')
# Calling array(args, kwargs) (line 8)
array_call_result_257 = invoke(stypy.reporting.localization.Localization(__file__, 8, 4), array_251, *[list_252], **kwargs_256)

# Assigning a type to the variable 'v' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'v', array_call_result_257)

# Assigning a Call to a Name (line 9):

# Call to tile(...): (line 9)
# Processing the call arguments (line 9)
# Getting the type of 'v' (line 9)
v_260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 13), 'v', False)

# Obtaining an instance of the builtin type 'tuple' (line 9)
tuple_261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 9)
# Adding element type (line 9)
int_262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 17), tuple_261, int_262)
# Adding element type (line 9)
int_263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 17), tuple_261, int_263)

# Processing the call keyword arguments (line 9)
kwargs_264 = {}
# Getting the type of 'np' (line 9)
np_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 5), 'np', False)
# Obtaining the member 'tile' of a type (line 9)
tile_259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 5), np_258, 'tile')
# Calling tile(args, kwargs) (line 9)
tile_call_result_265 = invoke(stypy.reporting.localization.Localization(__file__, 9, 5), tile_259, *[v_260, tuple_261], **kwargs_264)

# Assigning a type to the variable 'vv' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'vv', tile_call_result_265)

# Assigning a BinOp to a Name (line 10):
# Getting the type of 'x' (line 10)
x_266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'x')
# Getting the type of 'vv' (line 10)
vv_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'vv')
# Applying the binary operator '+' (line 10)
result_add_268 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 4), '+', x_266, vv_267)

# Assigning a type to the variable 'y' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'y', result_add_268)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
