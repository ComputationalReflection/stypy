
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: Z = np.arange(100)
6: v = np.random.uniform(0,100)
7: index = (np.abs(Z-v)).argmin()
8: e = Z[index]
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
import_1 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_1) is not StypyTypeError):

    if (import_1 != 'pyd_module'):
        __import__(import_1)
        sys_modules_2 = sys.modules[import_1]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_2.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_1)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Call to arange(...): (line 5)
# Processing the call arguments (line 5)
int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 14), 'int')
# Processing the call keyword arguments (line 5)
kwargs_6 = {}
# Getting the type of 'np' (line 5)
np_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'arange' of a type (line 5)
arange_4 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_3, 'arange')
# Calling arange(args, kwargs) (line 5)
arange_call_result_7 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), arange_4, *[int_5], **kwargs_6)

# Assigning a type to the variable 'Z' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'Z', arange_call_result_7)

# Assigning a Call to a Name (line 6):

# Call to uniform(...): (line 6)
# Processing the call arguments (line 6)
int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 22), 'int')
int_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 24), 'int')
# Processing the call keyword arguments (line 6)
kwargs_13 = {}
# Getting the type of 'np' (line 6)
np_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'np', False)
# Obtaining the member 'random' of a type (line 6)
random_9 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 4), np_8, 'random')
# Obtaining the member 'uniform' of a type (line 6)
uniform_10 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 4), random_9, 'uniform')
# Calling uniform(args, kwargs) (line 6)
uniform_call_result_14 = invoke(stypy.reporting.localization.Localization(__file__, 6, 4), uniform_10, *[int_11, int_12], **kwargs_13)

# Assigning a type to the variable 'v' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'v', uniform_call_result_14)

# Assigning a Call to a Name (line 7):

# Call to argmin(...): (line 7)
# Processing the call keyword arguments (line 7)
kwargs_23 = {}

# Call to abs(...): (line 7)
# Processing the call arguments (line 7)
# Getting the type of 'Z' (line 7)
Z_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 16), 'Z', False)
# Getting the type of 'v' (line 7)
v_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 18), 'v', False)
# Applying the binary operator '-' (line 7)
result_sub_19 = python_operator(stypy.reporting.localization.Localization(__file__, 7, 16), '-', Z_17, v_18)

# Processing the call keyword arguments (line 7)
kwargs_20 = {}
# Getting the type of 'np' (line 7)
np_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 9), 'np', False)
# Obtaining the member 'abs' of a type (line 7)
abs_16 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 9), np_15, 'abs')
# Calling abs(args, kwargs) (line 7)
abs_call_result_21 = invoke(stypy.reporting.localization.Localization(__file__, 7, 9), abs_16, *[result_sub_19], **kwargs_20)

# Obtaining the member 'argmin' of a type (line 7)
argmin_22 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 9), abs_call_result_21, 'argmin')
# Calling argmin(args, kwargs) (line 7)
argmin_call_result_24 = invoke(stypy.reporting.localization.Localization(__file__, 7, 9), argmin_22, *[], **kwargs_23)

# Assigning a type to the variable 'index' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'index', argmin_call_result_24)

# Assigning a Subscript to a Name (line 8):

# Obtaining the type of the subscript
# Getting the type of 'index' (line 8)
index_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 6), 'index')
# Getting the type of 'Z' (line 8)
Z_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'Z')
# Obtaining the member '__getitem__' of a type (line 8)
getitem___27 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 4), Z_26, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 8)
subscript_call_result_28 = invoke(stypy.reporting.localization.Localization(__file__, 8, 4), getitem___27, index_25)

# Assigning a type to the variable 'e' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'e', subscript_call_result_28)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
