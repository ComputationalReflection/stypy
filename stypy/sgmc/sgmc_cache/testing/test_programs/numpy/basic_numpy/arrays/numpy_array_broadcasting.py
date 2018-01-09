
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
9: y = np.empty_like(x)  # Create an empty matrix with the same shape as x
10: 
11: # Add the vector v to each row of the matrix x with an explicit loop
12: for i in range(4):
13:     y[i, :] = x[i, :] + v
14: 
15: # Now y is the following
16: # [[ 2  2  4]
17: #  [ 5  5  7]
18: #  [ 8  8 10]
19: #  [11 11 13]]
20: r = y
21: 
22: # l = globals().copy()
23: # for v in l:
24: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
25: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')
import_175 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_175) is not StypyTypeError):

    if (import_175 != 'pyd_module'):
        __import__(import_175)
        sys_modules_176 = sys.modules[import_175]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_176.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_175)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')


# Assigning a Call to a Name (line 7):

# Call to array(...): (line 7)
# Processing the call arguments (line 7)

# Obtaining an instance of the builtin type 'list' (line 7)
list_179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)

# Obtaining an instance of the builtin type 'list' (line 7)
list_180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
int_181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 14), list_180, int_181)
# Adding element type (line 7)
int_182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 14), list_180, int_182)
# Adding element type (line 7)
int_183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 14), list_180, int_183)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 13), list_179, list_180)
# Adding element type (line 7)

# Obtaining an instance of the builtin type 'list' (line 7)
list_184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
int_185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 25), list_184, int_185)
# Adding element type (line 7)
int_186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 25), list_184, int_186)
# Adding element type (line 7)
int_187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 25), list_184, int_187)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 13), list_179, list_184)
# Adding element type (line 7)

# Obtaining an instance of the builtin type 'list' (line 7)
list_188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 36), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
int_189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 37), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 36), list_188, int_189)
# Adding element type (line 7)
int_190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 40), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 36), list_188, int_190)
# Adding element type (line 7)
int_191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 43), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 36), list_188, int_191)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 13), list_179, list_188)
# Adding element type (line 7)

# Obtaining an instance of the builtin type 'list' (line 7)
list_192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 47), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
int_193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 48), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 47), list_192, int_193)
# Adding element type (line 7)
int_194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 52), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 47), list_192, int_194)
# Adding element type (line 7)
int_195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 56), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 47), list_192, int_195)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 13), list_179, list_192)

# Processing the call keyword arguments (line 7)
kwargs_196 = {}
# Getting the type of 'np' (line 7)
np_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'np', False)
# Obtaining the member 'array' of a type (line 7)
array_178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), np_177, 'array')
# Calling array(args, kwargs) (line 7)
array_call_result_197 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), array_178, *[list_179], **kwargs_196)

# Assigning a type to the variable 'x' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'x', array_call_result_197)

# Assigning a Call to a Name (line 8):

# Call to array(...): (line 8)
# Processing the call arguments (line 8)

# Obtaining an instance of the builtin type 'list' (line 8)
list_200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 8)
# Adding element type (line 8)
int_201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 14), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 13), list_200, int_201)
# Adding element type (line 8)
int_202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 13), list_200, int_202)
# Adding element type (line 8)
int_203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 13), list_200, int_203)

# Processing the call keyword arguments (line 8)
kwargs_204 = {}
# Getting the type of 'np' (line 8)
np_198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'np', False)
# Obtaining the member 'array' of a type (line 8)
array_199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 4), np_198, 'array')
# Calling array(args, kwargs) (line 8)
array_call_result_205 = invoke(stypy.reporting.localization.Localization(__file__, 8, 4), array_199, *[list_200], **kwargs_204)

# Assigning a type to the variable 'v' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'v', array_call_result_205)

# Assigning a Call to a Name (line 9):

# Call to empty_like(...): (line 9)
# Processing the call arguments (line 9)
# Getting the type of 'x' (line 9)
x_208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 18), 'x', False)
# Processing the call keyword arguments (line 9)
kwargs_209 = {}
# Getting the type of 'np' (line 9)
np_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'np', False)
# Obtaining the member 'empty_like' of a type (line 9)
empty_like_207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 4), np_206, 'empty_like')
# Calling empty_like(args, kwargs) (line 9)
empty_like_call_result_210 = invoke(stypy.reporting.localization.Localization(__file__, 9, 4), empty_like_207, *[x_208], **kwargs_209)

# Assigning a type to the variable 'y' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'y', empty_like_call_result_210)


# Call to range(...): (line 12)
# Processing the call arguments (line 12)
int_212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 15), 'int')
# Processing the call keyword arguments (line 12)
kwargs_213 = {}
# Getting the type of 'range' (line 12)
range_211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'range', False)
# Calling range(args, kwargs) (line 12)
range_call_result_214 = invoke(stypy.reporting.localization.Localization(__file__, 12, 9), range_211, *[int_212], **kwargs_213)

# Testing the type of a for loop iterable (line 12)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 12, 0), range_call_result_214)
# Getting the type of the for loop variable (line 12)
for_loop_var_215 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 12, 0), range_call_result_214)
# Assigning a type to the variable 'i' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'i', for_loop_var_215)
# SSA begins for a for statement (line 12)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')

# Assigning a BinOp to a Subscript (line 13):

# Obtaining the type of the subscript
# Getting the type of 'i' (line 13)
i_216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 16), 'i')
slice_217 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 13, 14), None, None, None)
# Getting the type of 'x' (line 13)
x_218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 14), 'x')
# Obtaining the member '__getitem__' of a type (line 13)
getitem___219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 14), x_218, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 13)
subscript_call_result_220 = invoke(stypy.reporting.localization.Localization(__file__, 13, 14), getitem___219, (i_216, slice_217))

# Getting the type of 'v' (line 13)
v_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 24), 'v')
# Applying the binary operator '+' (line 13)
result_add_222 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 14), '+', subscript_call_result_220, v_221)

# Getting the type of 'y' (line 13)
y_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'y')
# Getting the type of 'i' (line 13)
i_224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 6), 'i')
slice_225 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 13, 4), None, None, None)
# Storing an element on a container (line 13)
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 4), y_223, ((i_224, slice_225), result_add_222))
# SSA join for a for statement
module_type_store = module_type_store.join_ssa_context()


# Assigning a Name to a Name (line 20):
# Getting the type of 'y' (line 20)
y_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'y')
# Assigning a type to the variable 'r' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'r', y_226)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
