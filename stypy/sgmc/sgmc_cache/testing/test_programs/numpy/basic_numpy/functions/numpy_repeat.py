
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: C = np.bincount([1, 1, 2, 3, 4, 4, 6])
6: A = np.repeat(np.arange(len(C)), C)
7: #
8: # l = globals().copy()
9: # for v in l:
10: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
11: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_2221 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_2221) is not StypyTypeError):

    if (import_2221 != 'pyd_module'):
        __import__(import_2221)
        sys_modules_2222 = sys.modules[import_2221]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_2222.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_2221)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Call to bincount(...): (line 5)
# Processing the call arguments (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_2225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)
int_2226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 16), list_2225, int_2226)
# Adding element type (line 5)
int_2227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 16), list_2225, int_2227)
# Adding element type (line 5)
int_2228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 16), list_2225, int_2228)
# Adding element type (line 5)
int_2229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 16), list_2225, int_2229)
# Adding element type (line 5)
int_2230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 16), list_2225, int_2230)
# Adding element type (line 5)
int_2231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 16), list_2225, int_2231)
# Adding element type (line 5)
int_2232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 35), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 16), list_2225, int_2232)

# Processing the call keyword arguments (line 5)
kwargs_2233 = {}
# Getting the type of 'np' (line 5)
np_2223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'bincount' of a type (line 5)
bincount_2224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_2223, 'bincount')
# Calling bincount(args, kwargs) (line 5)
bincount_call_result_2234 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), bincount_2224, *[list_2225], **kwargs_2233)

# Assigning a type to the variable 'C' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'C', bincount_call_result_2234)

# Assigning a Call to a Name (line 6):

# Call to repeat(...): (line 6)
# Processing the call arguments (line 6)

# Call to arange(...): (line 6)
# Processing the call arguments (line 6)

# Call to len(...): (line 6)
# Processing the call arguments (line 6)
# Getting the type of 'C' (line 6)
C_2240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 28), 'C', False)
# Processing the call keyword arguments (line 6)
kwargs_2241 = {}
# Getting the type of 'len' (line 6)
len_2239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 24), 'len', False)
# Calling len(args, kwargs) (line 6)
len_call_result_2242 = invoke(stypy.reporting.localization.Localization(__file__, 6, 24), len_2239, *[C_2240], **kwargs_2241)

# Processing the call keyword arguments (line 6)
kwargs_2243 = {}
# Getting the type of 'np' (line 6)
np_2237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 14), 'np', False)
# Obtaining the member 'arange' of a type (line 6)
arange_2238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 14), np_2237, 'arange')
# Calling arange(args, kwargs) (line 6)
arange_call_result_2244 = invoke(stypy.reporting.localization.Localization(__file__, 6, 14), arange_2238, *[len_call_result_2242], **kwargs_2243)

# Getting the type of 'C' (line 6)
C_2245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 33), 'C', False)
# Processing the call keyword arguments (line 6)
kwargs_2246 = {}
# Getting the type of 'np' (line 6)
np_2235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'np', False)
# Obtaining the member 'repeat' of a type (line 6)
repeat_2236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 4), np_2235, 'repeat')
# Calling repeat(args, kwargs) (line 6)
repeat_call_result_2247 = invoke(stypy.reporting.localization.Localization(__file__, 6, 4), repeat_2236, *[arange_call_result_2244, C_2245], **kwargs_2246)

# Assigning a type to the variable 'A' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'A', repeat_call_result_2247)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
