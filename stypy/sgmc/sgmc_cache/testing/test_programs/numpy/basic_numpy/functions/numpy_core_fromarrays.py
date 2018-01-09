
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: Z = np.array([("Hello", 2.5, 3),
6:               ("World", 3.6, 2)])
7: R = np.core.records.fromarrays(Z.T,
8:                                names='col1, col2, col3',
9:                                formats='S8, f8, i8')
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
import_586 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_586) is not StypyTypeError):

    if (import_586 != 'pyd_module'):
        __import__(import_586)
        sys_modules_587 = sys.modules[import_586]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_587.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_586)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Call to array(...): (line 5)
# Processing the call arguments (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 13), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'tuple' (line 5)
tuple_591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 5)
# Adding element type (line 5)
str_592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 15), 'str', 'Hello')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 15), tuple_591, str_592)
# Adding element type (line 5)
float_593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 15), tuple_591, float_593)
# Adding element type (line 5)
int_594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 15), tuple_591, int_594)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_590, tuple_591)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'tuple' (line 6)
tuple_595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6)
# Adding element type (line 6)
str_596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 15), 'str', 'World')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 15), tuple_595, str_596)
# Adding element type (line 6)
float_597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 24), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 15), tuple_595, float_597)
# Adding element type (line 6)
int_598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 15), tuple_595, int_598)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), list_590, tuple_595)

# Processing the call keyword arguments (line 5)
kwargs_599 = {}
# Getting the type of 'np' (line 5)
np_588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'array' of a type (line 5)
array_589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_588, 'array')
# Calling array(args, kwargs) (line 5)
array_call_result_600 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), array_589, *[list_590], **kwargs_599)

# Assigning a type to the variable 'Z' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'Z', array_call_result_600)

# Assigning a Call to a Name (line 7):

# Call to fromarrays(...): (line 7)
# Processing the call arguments (line 7)
# Getting the type of 'Z' (line 7)
Z_605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 31), 'Z', False)
# Obtaining the member 'T' of a type (line 7)
T_606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 31), Z_605, 'T')
# Processing the call keyword arguments (line 7)
str_607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 37), 'str', 'col1, col2, col3')
keyword_608 = str_607
str_609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 39), 'str', 'S8, f8, i8')
keyword_610 = str_609
kwargs_611 = {'names': keyword_608, 'formats': keyword_610}
# Getting the type of 'np' (line 7)
np_601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'np', False)
# Obtaining the member 'core' of a type (line 7)
core_602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), np_601, 'core')
# Obtaining the member 'records' of a type (line 7)
records_603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), core_602, 'records')
# Obtaining the member 'fromarrays' of a type (line 7)
fromarrays_604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), records_603, 'fromarrays')
# Calling fromarrays(args, kwargs) (line 7)
fromarrays_call_result_612 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), fromarrays_604, *[T_606], **kwargs_611)

# Assigning a type to the variable 'R' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'R', fromarrays_call_result_612)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
