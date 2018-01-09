
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.python-course.eu/numpy.php
2: 
3: import numpy as np
4: 
5: a = np.arange(1, 10)
6: 
7: # compare to range:
8: x = range(1, 10)
9: 
10: # some more arange examples:
11: x2 = np.arange(10.4)
12: 
13: x3 = np.arange(0.5, 10.4, 0.8)
14: 
15: x4 = np.arange(0.5, 10.4, 0.8, int)
16: 
17: # l = globals().copy()
18: # for v in l:
19: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_94 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_94) is not StypyTypeError):

    if (import_94 != 'pyd_module'):
        __import__(import_94)
        sys_modules_95 = sys.modules[import_94]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_95.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_94)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Call to arange(...): (line 5)
# Processing the call arguments (line 5)
int_98 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 14), 'int')
int_99 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 17), 'int')
# Processing the call keyword arguments (line 5)
kwargs_100 = {}
# Getting the type of 'np' (line 5)
np_96 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'arange' of a type (line 5)
arange_97 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_96, 'arange')
# Calling arange(args, kwargs) (line 5)
arange_call_result_101 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), arange_97, *[int_98, int_99], **kwargs_100)

# Assigning a type to the variable 'a' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'a', arange_call_result_101)

# Assigning a Call to a Name (line 8):

# Call to range(...): (line 8)
# Processing the call arguments (line 8)
int_103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 10), 'int')
int_104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 13), 'int')
# Processing the call keyword arguments (line 8)
kwargs_105 = {}
# Getting the type of 'range' (line 8)
range_102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'range', False)
# Calling range(args, kwargs) (line 8)
range_call_result_106 = invoke(stypy.reporting.localization.Localization(__file__, 8, 4), range_102, *[int_103, int_104], **kwargs_105)

# Assigning a type to the variable 'x' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'x', range_call_result_106)

# Assigning a Call to a Name (line 11):

# Call to arange(...): (line 11)
# Processing the call arguments (line 11)
float_109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 15), 'float')
# Processing the call keyword arguments (line 11)
kwargs_110 = {}
# Getting the type of 'np' (line 11)
np_107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 5), 'np', False)
# Obtaining the member 'arange' of a type (line 11)
arange_108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 5), np_107, 'arange')
# Calling arange(args, kwargs) (line 11)
arange_call_result_111 = invoke(stypy.reporting.localization.Localization(__file__, 11, 5), arange_108, *[float_109], **kwargs_110)

# Assigning a type to the variable 'x2' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'x2', arange_call_result_111)

# Assigning a Call to a Name (line 13):

# Call to arange(...): (line 13)
# Processing the call arguments (line 13)
float_114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'float')
float_115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 20), 'float')
float_116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 26), 'float')
# Processing the call keyword arguments (line 13)
kwargs_117 = {}
# Getting the type of 'np' (line 13)
np_112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 5), 'np', False)
# Obtaining the member 'arange' of a type (line 13)
arange_113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 5), np_112, 'arange')
# Calling arange(args, kwargs) (line 13)
arange_call_result_118 = invoke(stypy.reporting.localization.Localization(__file__, 13, 5), arange_113, *[float_114, float_115, float_116], **kwargs_117)

# Assigning a type to the variable 'x3' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'x3', arange_call_result_118)

# Assigning a Call to a Name (line 15):

# Call to arange(...): (line 15)
# Processing the call arguments (line 15)
float_121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'float')
float_122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 20), 'float')
float_123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 26), 'float')
# Getting the type of 'int' (line 15)
int_124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 31), 'int', False)
# Processing the call keyword arguments (line 15)
kwargs_125 = {}
# Getting the type of 'np' (line 15)
np_119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 5), 'np', False)
# Obtaining the member 'arange' of a type (line 15)
arange_120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 5), np_119, 'arange')
# Calling arange(args, kwargs) (line 15)
arange_call_result_126 = invoke(stypy.reporting.localization.Localization(__file__, 15, 5), arange_120, *[float_121, float_122, float_123, int_124], **kwargs_125)

# Assigning a type to the variable 'x4' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'x4', arange_call_result_126)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
