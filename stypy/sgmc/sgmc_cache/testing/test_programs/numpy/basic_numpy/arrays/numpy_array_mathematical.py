
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://cs231n.github.io/python-numpy-tutorial/
2: 
3: import numpy as np
4: 
5: x = np.array([[1, 2], [3, 4]])
6: 
7: r = np.sum(x)  # Compute sum of all elements; prints "10"
8: r2 = np.sum(x, axis=0)  # Compute sum of each column; prints "[4 6]"
9: r3 = np.sum(x, axis=1)  # Compute sum of each row; prints "[3 7]"
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
import_789 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_789) is not StypyTypeError):

    if (import_789 != 'pyd_module'):
        __import__(import_789)
        sys_modules_790 = sys.modules[import_789]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_790.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_789)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')


# Assigning a Call to a Name (line 5):

# Call to array(...): (line 5)
# Processing the call arguments (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)
int_795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 14), list_794, int_795)
# Adding element type (line 5)
int_796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 14), list_794, int_796)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_793, list_794)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)
int_798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 22), list_797, int_798)
# Adding element type (line 5)
int_799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 22), list_797, int_799)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_793, list_797)

# Processing the call keyword arguments (line 5)
kwargs_800 = {}
# Getting the type of 'np' (line 5)
np_791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'array' of a type (line 5)
array_792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_791, 'array')
# Calling array(args, kwargs) (line 5)
array_call_result_801 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), array_792, *[list_793], **kwargs_800)

# Assigning a type to the variable 'x' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'x', array_call_result_801)

# Assigning a Call to a Name (line 7):

# Call to sum(...): (line 7)
# Processing the call arguments (line 7)
# Getting the type of 'x' (line 7)
x_804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 11), 'x', False)
# Processing the call keyword arguments (line 7)
kwargs_805 = {}
# Getting the type of 'np' (line 7)
np_802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'np', False)
# Obtaining the member 'sum' of a type (line 7)
sum_803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), np_802, 'sum')
# Calling sum(args, kwargs) (line 7)
sum_call_result_806 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), sum_803, *[x_804], **kwargs_805)

# Assigning a type to the variable 'r' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'r', sum_call_result_806)

# Assigning a Call to a Name (line 8):

# Call to sum(...): (line 8)
# Processing the call arguments (line 8)
# Getting the type of 'x' (line 8)
x_809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 12), 'x', False)
# Processing the call keyword arguments (line 8)
int_810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 20), 'int')
keyword_811 = int_810
kwargs_812 = {'axis': keyword_811}
# Getting the type of 'np' (line 8)
np_807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 5), 'np', False)
# Obtaining the member 'sum' of a type (line 8)
sum_808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 5), np_807, 'sum')
# Calling sum(args, kwargs) (line 8)
sum_call_result_813 = invoke(stypy.reporting.localization.Localization(__file__, 8, 5), sum_808, *[x_809], **kwargs_812)

# Assigning a type to the variable 'r2' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'r2', sum_call_result_813)

# Assigning a Call to a Name (line 9):

# Call to sum(...): (line 9)
# Processing the call arguments (line 9)
# Getting the type of 'x' (line 9)
x_816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 12), 'x', False)
# Processing the call keyword arguments (line 9)
int_817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 20), 'int')
keyword_818 = int_817
kwargs_819 = {'axis': keyword_818}
# Getting the type of 'np' (line 9)
np_814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 5), 'np', False)
# Obtaining the member 'sum' of a type (line 9)
sum_815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 5), np_814, 'sum')
# Calling sum(args, kwargs) (line 9)
sum_call_result_820 = invoke(stypy.reporting.localization.Localization(__file__, 9, 5), sum_815, *[x_816], **kwargs_819)

# Assigning a type to the variable 'r3' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'r3', sum_call_result_820)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
