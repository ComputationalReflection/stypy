
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.scipy-lectures.org/intro/numpy/numpy.html
2: import numpy as np
3: 
4: a = np.triu(np.ones((3, 3)), 1)  # see help(np.triu)
5: 
6: r = a.T
7: 
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

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import numpy' statement (line 2)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_2537 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy')

if (type(import_2537) is not StypyTypeError):

    if (import_2537 != 'pyd_module'):
        __import__(import_2537)
        sys_modules_2538 = sys.modules[import_2537]
        import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'np', sys_modules_2538.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy', import_2537)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 4):

# Call to triu(...): (line 4)
# Processing the call arguments (line 4)

# Call to ones(...): (line 4)
# Processing the call arguments (line 4)

# Obtaining an instance of the builtin type 'tuple' (line 4)
tuple_2543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 4)
# Adding element type (line 4)
int_2544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 21), tuple_2543, int_2544)
# Adding element type (line 4)
int_2545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 21), tuple_2543, int_2545)

# Processing the call keyword arguments (line 4)
kwargs_2546 = {}
# Getting the type of 'np' (line 4)
np_2541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 12), 'np', False)
# Obtaining the member 'ones' of a type (line 4)
ones_2542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 12), np_2541, 'ones')
# Calling ones(args, kwargs) (line 4)
ones_call_result_2547 = invoke(stypy.reporting.localization.Localization(__file__, 4, 12), ones_2542, *[tuple_2543], **kwargs_2546)

int_2548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 29), 'int')
# Processing the call keyword arguments (line 4)
kwargs_2549 = {}
# Getting the type of 'np' (line 4)
np_2539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'np', False)
# Obtaining the member 'triu' of a type (line 4)
triu_2540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 4), np_2539, 'triu')
# Calling triu(args, kwargs) (line 4)
triu_call_result_2550 = invoke(stypy.reporting.localization.Localization(__file__, 4, 4), triu_2540, *[ones_call_result_2547, int_2548], **kwargs_2549)

# Assigning a type to the variable 'a' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'a', triu_call_result_2550)

# Assigning a Attribute to a Name (line 6):
# Getting the type of 'a' (line 6)
a_2551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'a')
# Obtaining the member 'T' of a type (line 6)
T_2552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 4), a_2551, 'T')
# Assigning a type to the variable 'r' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'r', T_2552)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
