
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: # http://www.scipy-lectures.org/intro/numpy/numpy.html
3: 
4: import numpy as np
5: 
6: r = np.all([True, True, False])
7: r2 = np.any([True, True, False])
8: 
9: # l = globals().copy()
10: # for v in l:
11: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
12: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import numpy' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_76 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy')

if (type(import_76) is not StypyTypeError):

    if (import_76 != 'pyd_module'):
        __import__(import_76)
        sys_modules_77 = sys.modules[import_76]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', sys_modules_77.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', import_76)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 6):

# Call to all(...): (line 6)
# Processing the call arguments (line 6)

# Obtaining an instance of the builtin type 'list' (line 6)
list_80 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 11), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)
# Getting the type of 'True' (line 6)
True_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 12), 'True', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 11), list_80, True_81)
# Adding element type (line 6)
# Getting the type of 'True' (line 6)
True_82 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 18), 'True', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 11), list_80, True_82)
# Adding element type (line 6)
# Getting the type of 'False' (line 6)
False_83 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 24), 'False', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 11), list_80, False_83)

# Processing the call keyword arguments (line 6)
kwargs_84 = {}
# Getting the type of 'np' (line 6)
np_78 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'np', False)
# Obtaining the member 'all' of a type (line 6)
all_79 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 4), np_78, 'all')
# Calling all(args, kwargs) (line 6)
all_call_result_85 = invoke(stypy.reporting.localization.Localization(__file__, 6, 4), all_79, *[list_80], **kwargs_84)

# Assigning a type to the variable 'r' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'r', all_call_result_85)

# Assigning a Call to a Name (line 7):

# Call to any(...): (line 7)
# Processing the call arguments (line 7)

# Obtaining an instance of the builtin type 'list' (line 7)
list_88 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
# Getting the type of 'True' (line 7)
True_89 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 13), 'True', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 12), list_88, True_89)
# Adding element type (line 7)
# Getting the type of 'True' (line 7)
True_90 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 19), 'True', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 12), list_88, True_90)
# Adding element type (line 7)
# Getting the type of 'False' (line 7)
False_91 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 25), 'False', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 12), list_88, False_91)

# Processing the call keyword arguments (line 7)
kwargs_92 = {}
# Getting the type of 'np' (line 7)
np_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 5), 'np', False)
# Obtaining the member 'any' of a type (line 7)
any_87 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 5), np_86, 'any')
# Calling any(args, kwargs) (line 7)
any_call_result_93 = invoke(stypy.reporting.localization.Localization(__file__, 7, 5), any_87, *[list_88], **kwargs_92)

# Assigning a type to the variable 'r2' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'r2', any_call_result_93)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
