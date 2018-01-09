
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.scipy-lectures.org/intro/numpy/numpy.html
2: import numpy as np
3: 
4: a = np.array([[1, 2, 3], [4, 5, 6]])
5: r = a.ravel()
6: 
7: r2 = a.T
8: 
9: r3 = a.T.ravel()
10: #
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

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import numpy' statement (line 2)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')
import_924 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy')

if (type(import_924) is not StypyTypeError):

    if (import_924 != 'pyd_module'):
        __import__(import_924)
        sys_modules_925 = sys.modules[import_924]
        import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'np', sys_modules_925.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy', import_924)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/arrays/')


# Assigning a Call to a Name (line 4):

# Call to array(...): (line 4)
# Processing the call arguments (line 4)

# Obtaining an instance of the builtin type 'list' (line 4)
list_928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 4)
# Adding element type (line 4)

# Obtaining an instance of the builtin type 'list' (line 4)
list_929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 4)
# Adding element type (line 4)
int_930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 14), list_929, int_930)
# Adding element type (line 4)
int_931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 14), list_929, int_931)
# Adding element type (line 4)
int_932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 14), list_929, int_932)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 13), list_928, list_929)
# Adding element type (line 4)

# Obtaining an instance of the builtin type 'list' (line 4)
list_933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 4)
# Adding element type (line 4)
int_934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 25), list_933, int_934)
# Adding element type (line 4)
int_935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 25), list_933, int_935)
# Adding element type (line 4)
int_936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 25), list_933, int_936)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 13), list_928, list_933)

# Processing the call keyword arguments (line 4)
kwargs_937 = {}
# Getting the type of 'np' (line 4)
np_926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'np', False)
# Obtaining the member 'array' of a type (line 4)
array_927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 4), np_926, 'array')
# Calling array(args, kwargs) (line 4)
array_call_result_938 = invoke(stypy.reporting.localization.Localization(__file__, 4, 4), array_927, *[list_928], **kwargs_937)

# Assigning a type to the variable 'a' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'a', array_call_result_938)

# Assigning a Call to a Name (line 5):

# Call to ravel(...): (line 5)
# Processing the call keyword arguments (line 5)
kwargs_941 = {}
# Getting the type of 'a' (line 5)
a_939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'a', False)
# Obtaining the member 'ravel' of a type (line 5)
ravel_940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), a_939, 'ravel')
# Calling ravel(args, kwargs) (line 5)
ravel_call_result_942 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), ravel_940, *[], **kwargs_941)

# Assigning a type to the variable 'r' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'r', ravel_call_result_942)

# Assigning a Attribute to a Name (line 7):
# Getting the type of 'a' (line 7)
a_943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 5), 'a')
# Obtaining the member 'T' of a type (line 7)
T_944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 5), a_943, 'T')
# Assigning a type to the variable 'r2' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'r2', T_944)

# Assigning a Call to a Name (line 9):

# Call to ravel(...): (line 9)
# Processing the call keyword arguments (line 9)
kwargs_948 = {}
# Getting the type of 'a' (line 9)
a_945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 5), 'a', False)
# Obtaining the member 'T' of a type (line 9)
T_946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 5), a_945, 'T')
# Obtaining the member 'ravel' of a type (line 9)
ravel_947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 5), T_946, 'ravel')
# Calling ravel(args, kwargs) (line 9)
ravel_call_result_949 = invoke(stypy.reporting.localization.Localization(__file__, 9, 5), ravel_947, *[], **kwargs_948)

# Assigning a type to the variable 'r3' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'r3', ravel_call_result_949)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
