
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # https://docs.scipy.org/doc/numpy/reference/routines.math.html
2: 
3: 
4: import numpy as np
5: 
6: x = 2.1
7: 
8: # Other special functions
9: r1 = np.i0(x)  # Modified Bessel function of the first kind, order 0.
10: r2 = np.sinc(x)  # Return the sinc function.
11: 
12: x = [2.1, 2.2, 2.3]
13: 
14: r3 = np.i0(x)  # Modified Bessel function of the first kind, order 0.
15: r4 = np.sinc(x)  # Return the sinc function.
16: 
17: # l = globals().copy()
18: # for v in l:
19: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
20: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import numpy' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')
import_1 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy')

if (type(import_1) is not StypyTypeError):

    if (import_1 != 'pyd_module'):
        __import__(import_1)
        sys_modules_2 = sys.modules[import_1]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', sys_modules_2.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', import_1)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')


# Assigning a Num to a Name (line 6):
float_3 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 4), 'float')
# Assigning a type to the variable 'x' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'x', float_3)

# Assigning a Call to a Name (line 9):

# Call to i0(...): (line 9)
# Processing the call arguments (line 9)
# Getting the type of 'x' (line 9)
x_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 11), 'x', False)
# Processing the call keyword arguments (line 9)
kwargs_7 = {}
# Getting the type of 'np' (line 9)
np_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 5), 'np', False)
# Obtaining the member 'i0' of a type (line 9)
i0_5 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 5), np_4, 'i0')
# Calling i0(args, kwargs) (line 9)
i0_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 9, 5), i0_5, *[x_6], **kwargs_7)

# Assigning a type to the variable 'r1' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'r1', i0_call_result_8)

# Assigning a Call to a Name (line 10):

# Call to sinc(...): (line 10)
# Processing the call arguments (line 10)
# Getting the type of 'x' (line 10)
x_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 13), 'x', False)
# Processing the call keyword arguments (line 10)
kwargs_12 = {}
# Getting the type of 'np' (line 10)
np_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 5), 'np', False)
# Obtaining the member 'sinc' of a type (line 10)
sinc_10 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 5), np_9, 'sinc')
# Calling sinc(args, kwargs) (line 10)
sinc_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 10, 5), sinc_10, *[x_11], **kwargs_12)

# Assigning a type to the variable 'r2' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'r2', sinc_call_result_13)

# Assigning a List to a Name (line 12):

# Obtaining an instance of the builtin type 'list' (line 12)
list_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 4), 'list')
# Adding type elements to the builtin type 'list' instance (line 12)
# Adding element type (line 12)
float_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 5), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 4), list_14, float_15)
# Adding element type (line 12)
float_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 10), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 4), list_14, float_16)
# Adding element type (line 12)
float_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 15), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 4), list_14, float_17)

# Assigning a type to the variable 'x' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'x', list_14)

# Assigning a Call to a Name (line 14):

# Call to i0(...): (line 14)
# Processing the call arguments (line 14)
# Getting the type of 'x' (line 14)
x_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 11), 'x', False)
# Processing the call keyword arguments (line 14)
kwargs_21 = {}
# Getting the type of 'np' (line 14)
np_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 5), 'np', False)
# Obtaining the member 'i0' of a type (line 14)
i0_19 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 5), np_18, 'i0')
# Calling i0(args, kwargs) (line 14)
i0_call_result_22 = invoke(stypy.reporting.localization.Localization(__file__, 14, 5), i0_19, *[x_20], **kwargs_21)

# Assigning a type to the variable 'r3' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'r3', i0_call_result_22)

# Assigning a Call to a Name (line 15):

# Call to sinc(...): (line 15)
# Processing the call arguments (line 15)
# Getting the type of 'x' (line 15)
x_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 13), 'x', False)
# Processing the call keyword arguments (line 15)
kwargs_26 = {}
# Getting the type of 'np' (line 15)
np_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 5), 'np', False)
# Obtaining the member 'sinc' of a type (line 15)
sinc_24 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 5), np_23, 'sinc')
# Calling sinc(args, kwargs) (line 15)
sinc_call_result_27 = invoke(stypy.reporting.localization.Localization(__file__, 15, 5), sinc_24, *[x_25], **kwargs_26)

# Assigning a type to the variable 'r4' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'r4', sinc_call_result_27)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
