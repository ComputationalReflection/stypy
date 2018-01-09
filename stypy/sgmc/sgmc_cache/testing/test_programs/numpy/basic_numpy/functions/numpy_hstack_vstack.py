
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://telliott99.blogspot.com.es/2010/01/heres-question-on-so-about-how-to-make.html
2: 
3: import numpy as np
4: 
5: u = 15
6: b = np.zeros(u ** 2)
7: b.shape = (u, u)
8: w = b + 0x99
9: 
10: width = 20  # squares across of a single type
11: row1 = np.hstack([w, b] * width)
12: row2 = np.hstack([b, w] * width)
13: board = np.vstack([row1, row2] * width)
14: r = board.shape
15: 
16: # l = globals().copy()
17: # for v in l:
18: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
19: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_1192 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_1192) is not StypyTypeError):

    if (import_1192 != 'pyd_module'):
        __import__(import_1192)
        sys_modules_1193 = sys.modules[import_1192]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_1193.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_1192)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Num to a Name (line 5):
int_1194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 4), 'int')
# Assigning a type to the variable 'u' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'u', int_1194)

# Assigning a Call to a Name (line 6):

# Call to zeros(...): (line 6)
# Processing the call arguments (line 6)
# Getting the type of 'u' (line 6)
u_1197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 13), 'u', False)
int_1198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 18), 'int')
# Applying the binary operator '**' (line 6)
result_pow_1199 = python_operator(stypy.reporting.localization.Localization(__file__, 6, 13), '**', u_1197, int_1198)

# Processing the call keyword arguments (line 6)
kwargs_1200 = {}
# Getting the type of 'np' (line 6)
np_1195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'np', False)
# Obtaining the member 'zeros' of a type (line 6)
zeros_1196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 4), np_1195, 'zeros')
# Calling zeros(args, kwargs) (line 6)
zeros_call_result_1201 = invoke(stypy.reporting.localization.Localization(__file__, 6, 4), zeros_1196, *[result_pow_1199], **kwargs_1200)

# Assigning a type to the variable 'b' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'b', zeros_call_result_1201)

# Assigning a Tuple to a Attribute (line 7):

# Obtaining an instance of the builtin type 'tuple' (line 7)
tuple_1202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 11), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7)
# Adding element type (line 7)
# Getting the type of 'u' (line 7)
u_1203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 11), 'u')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 11), tuple_1202, u_1203)
# Adding element type (line 7)
# Getting the type of 'u' (line 7)
u_1204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 14), 'u')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 11), tuple_1202, u_1204)

# Getting the type of 'b' (line 7)
b_1205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'b')
# Setting the type of the member 'shape' of a type (line 7)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 0), b_1205, 'shape', tuple_1202)

# Assigning a BinOp to a Name (line 8):
# Getting the type of 'b' (line 8)
b_1206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'b')
int_1207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 8), 'int')
# Applying the binary operator '+' (line 8)
result_add_1208 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 4), '+', b_1206, int_1207)

# Assigning a type to the variable 'w' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'w', result_add_1208)

# Assigning a Num to a Name (line 10):
int_1209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 8), 'int')
# Assigning a type to the variable 'width' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'width', int_1209)

# Assigning a Call to a Name (line 11):

# Call to hstack(...): (line 11)
# Processing the call arguments (line 11)

# Obtaining an instance of the builtin type 'list' (line 11)
list_1212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 17), 'list')
# Adding type elements to the builtin type 'list' instance (line 11)
# Adding element type (line 11)
# Getting the type of 'w' (line 11)
w_1213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 18), 'w', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 17), list_1212, w_1213)
# Adding element type (line 11)
# Getting the type of 'b' (line 11)
b_1214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 21), 'b', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 17), list_1212, b_1214)

# Getting the type of 'width' (line 11)
width_1215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 26), 'width', False)
# Applying the binary operator '*' (line 11)
result_mul_1216 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 17), '*', list_1212, width_1215)

# Processing the call keyword arguments (line 11)
kwargs_1217 = {}
# Getting the type of 'np' (line 11)
np_1210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 7), 'np', False)
# Obtaining the member 'hstack' of a type (line 11)
hstack_1211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 7), np_1210, 'hstack')
# Calling hstack(args, kwargs) (line 11)
hstack_call_result_1218 = invoke(stypy.reporting.localization.Localization(__file__, 11, 7), hstack_1211, *[result_mul_1216], **kwargs_1217)

# Assigning a type to the variable 'row1' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'row1', hstack_call_result_1218)

# Assigning a Call to a Name (line 12):

# Call to hstack(...): (line 12)
# Processing the call arguments (line 12)

# Obtaining an instance of the builtin type 'list' (line 12)
list_1221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 17), 'list')
# Adding type elements to the builtin type 'list' instance (line 12)
# Adding element type (line 12)
# Getting the type of 'b' (line 12)
b_1222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 18), 'b', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 17), list_1221, b_1222)
# Adding element type (line 12)
# Getting the type of 'w' (line 12)
w_1223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 21), 'w', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 17), list_1221, w_1223)

# Getting the type of 'width' (line 12)
width_1224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 26), 'width', False)
# Applying the binary operator '*' (line 12)
result_mul_1225 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 17), '*', list_1221, width_1224)

# Processing the call keyword arguments (line 12)
kwargs_1226 = {}
# Getting the type of 'np' (line 12)
np_1219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 7), 'np', False)
# Obtaining the member 'hstack' of a type (line 12)
hstack_1220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 7), np_1219, 'hstack')
# Calling hstack(args, kwargs) (line 12)
hstack_call_result_1227 = invoke(stypy.reporting.localization.Localization(__file__, 12, 7), hstack_1220, *[result_mul_1225], **kwargs_1226)

# Assigning a type to the variable 'row2' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'row2', hstack_call_result_1227)

# Assigning a Call to a Name (line 13):

# Call to vstack(...): (line 13)
# Processing the call arguments (line 13)

# Obtaining an instance of the builtin type 'list' (line 13)
list_1230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 18), 'list')
# Adding type elements to the builtin type 'list' instance (line 13)
# Adding element type (line 13)
# Getting the type of 'row1' (line 13)
row1_1231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 19), 'row1', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 18), list_1230, row1_1231)
# Adding element type (line 13)
# Getting the type of 'row2' (line 13)
row2_1232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 25), 'row2', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 18), list_1230, row2_1232)

# Getting the type of 'width' (line 13)
width_1233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 33), 'width', False)
# Applying the binary operator '*' (line 13)
result_mul_1234 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 18), '*', list_1230, width_1233)

# Processing the call keyword arguments (line 13)
kwargs_1235 = {}
# Getting the type of 'np' (line 13)
np_1228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'np', False)
# Obtaining the member 'vstack' of a type (line 13)
vstack_1229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 8), np_1228, 'vstack')
# Calling vstack(args, kwargs) (line 13)
vstack_call_result_1236 = invoke(stypy.reporting.localization.Localization(__file__, 13, 8), vstack_1229, *[result_mul_1234], **kwargs_1235)

# Assigning a type to the variable 'board' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'board', vstack_call_result_1236)

# Assigning a Attribute to a Name (line 14):
# Getting the type of 'board' (line 14)
board_1237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'board')
# Obtaining the member 'shape' of a type (line 14)
shape_1238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 4), board_1237, 'shape')
# Assigning a type to the variable 'r' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'r', shape_1238)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
