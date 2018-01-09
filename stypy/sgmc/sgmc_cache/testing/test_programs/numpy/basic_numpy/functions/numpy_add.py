
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: Z = np.ones((16, 16))
6: k = 4
7: 
8: S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
9:                     np.arange(0, Z.shape[1], k), axis=1)
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
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_29 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_29) is not StypyTypeError):

    if (import_29 != 'pyd_module'):
        __import__(import_29)
        sys_modules_30 = sys.modules[import_29]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_30.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_29)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Call to ones(...): (line 5)
# Processing the call arguments (line 5)

# Obtaining an instance of the builtin type 'tuple' (line 5)
tuple_33 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 5)
# Adding element type (line 5)
int_34 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), tuple_33, int_34)
# Adding element type (line 5)
int_35 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), tuple_33, int_35)

# Processing the call keyword arguments (line 5)
kwargs_36 = {}
# Getting the type of 'np' (line 5)
np_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'ones' of a type (line 5)
ones_32 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_31, 'ones')
# Calling ones(args, kwargs) (line 5)
ones_call_result_37 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), ones_32, *[tuple_33], **kwargs_36)

# Assigning a type to the variable 'Z' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'Z', ones_call_result_37)

# Assigning a Num to a Name (line 6):
int_38 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 4), 'int')
# Assigning a type to the variable 'k' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'k', int_38)

# Assigning a Call to a Name (line 8):

# Call to reduceat(...): (line 8)
# Processing the call arguments (line 8)

# Call to reduceat(...): (line 8)
# Processing the call arguments (line 8)
# Getting the type of 'Z' (line 8)
Z_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 36), 'Z', False)

# Call to arange(...): (line 8)
# Processing the call arguments (line 8)
int_48 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 49), 'int')

# Obtaining the type of the subscript
int_49 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 60), 'int')
# Getting the type of 'Z' (line 8)
Z_50 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 52), 'Z', False)
# Obtaining the member 'shape' of a type (line 8)
shape_51 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 52), Z_50, 'shape')
# Obtaining the member '__getitem__' of a type (line 8)
getitem___52 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 52), shape_51, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 8)
subscript_call_result_53 = invoke(stypy.reporting.localization.Localization(__file__, 8, 52), getitem___52, int_49)

# Getting the type of 'k' (line 8)
k_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 64), 'k', False)
# Processing the call keyword arguments (line 8)
kwargs_55 = {}
# Getting the type of 'np' (line 8)
np_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 39), 'np', False)
# Obtaining the member 'arange' of a type (line 8)
arange_47 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 39), np_46, 'arange')
# Calling arange(args, kwargs) (line 8)
arange_call_result_56 = invoke(stypy.reporting.localization.Localization(__file__, 8, 39), arange_47, *[int_48, subscript_call_result_53, k_54], **kwargs_55)

# Processing the call keyword arguments (line 8)
int_57 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 73), 'int')
keyword_58 = int_57
kwargs_59 = {'axis': keyword_58}
# Getting the type of 'np' (line 8)
np_42 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 20), 'np', False)
# Obtaining the member 'add' of a type (line 8)
add_43 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 20), np_42, 'add')
# Obtaining the member 'reduceat' of a type (line 8)
reduceat_44 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 20), add_43, 'reduceat')
# Calling reduceat(args, kwargs) (line 8)
reduceat_call_result_60 = invoke(stypy.reporting.localization.Localization(__file__, 8, 20), reduceat_44, *[Z_45, arange_call_result_56], **kwargs_59)


# Call to arange(...): (line 9)
# Processing the call arguments (line 9)
int_63 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 30), 'int')

# Obtaining the type of the subscript
int_64 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 41), 'int')
# Getting the type of 'Z' (line 9)
Z_65 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 33), 'Z', False)
# Obtaining the member 'shape' of a type (line 9)
shape_66 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 33), Z_65, 'shape')
# Obtaining the member '__getitem__' of a type (line 9)
getitem___67 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 33), shape_66, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 9)
subscript_call_result_68 = invoke(stypy.reporting.localization.Localization(__file__, 9, 33), getitem___67, int_64)

# Getting the type of 'k' (line 9)
k_69 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 45), 'k', False)
# Processing the call keyword arguments (line 9)
kwargs_70 = {}
# Getting the type of 'np' (line 9)
np_61 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 20), 'np', False)
# Obtaining the member 'arange' of a type (line 9)
arange_62 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 20), np_61, 'arange')
# Calling arange(args, kwargs) (line 9)
arange_call_result_71 = invoke(stypy.reporting.localization.Localization(__file__, 9, 20), arange_62, *[int_63, subscript_call_result_68, k_69], **kwargs_70)

# Processing the call keyword arguments (line 8)
int_72 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 54), 'int')
keyword_73 = int_72
kwargs_74 = {'axis': keyword_73}
# Getting the type of 'np' (line 8)
np_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'np', False)
# Obtaining the member 'add' of a type (line 8)
add_40 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 4), np_39, 'add')
# Obtaining the member 'reduceat' of a type (line 8)
reduceat_41 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 4), add_40, 'reduceat')
# Calling reduceat(args, kwargs) (line 8)
reduceat_call_result_75 = invoke(stypy.reporting.localization.Localization(__file__, 8, 4), reduceat_41, *[reduceat_call_result_60, arange_call_result_71], **kwargs_74)

# Assigning a type to the variable 'S' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'S', reduceat_call_result_75)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
