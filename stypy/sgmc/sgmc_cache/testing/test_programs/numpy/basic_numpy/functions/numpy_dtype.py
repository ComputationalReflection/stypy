
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: color = np.dtype([("r", np.ubyte, 1),
6:                   ("g", np.ubyte, 1),
7:                   ("b", np.ubyte, 1),
8:                   ("a", np.ubyte, 1)])
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
import_674 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_674) is not StypyTypeError):

    if (import_674 != 'pyd_module'):
        __import__(import_674)
        sys_modules_675 = sys.modules[import_674]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_675.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_674)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Call to dtype(...): (line 5)
# Processing the call arguments (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 17), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'tuple' (line 5)
tuple_679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 19), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 5)
# Adding element type (line 5)
str_680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 19), 'str', 'r')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 19), tuple_679, str_680)
# Adding element type (line 5)
# Getting the type of 'np' (line 5)
np_681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 24), 'np', False)
# Obtaining the member 'ubyte' of a type (line 5)
ubyte_682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 24), np_681, 'ubyte')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 19), tuple_679, ubyte_682)
# Adding element type (line 5)
int_683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 34), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 19), tuple_679, int_683)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 17), list_678, tuple_679)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'tuple' (line 6)
tuple_684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 19), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 6)
# Adding element type (line 6)
str_685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 19), 'str', 'g')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 19), tuple_684, str_685)
# Adding element type (line 6)
# Getting the type of 'np' (line 6)
np_686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 24), 'np', False)
# Obtaining the member 'ubyte' of a type (line 6)
ubyte_687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 24), np_686, 'ubyte')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 19), tuple_684, ubyte_687)
# Adding element type (line 6)
int_688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 34), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 19), tuple_684, int_688)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 17), list_678, tuple_684)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'tuple' (line 7)
tuple_689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 19), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7)
# Adding element type (line 7)
str_690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 19), 'str', 'b')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 19), tuple_689, str_690)
# Adding element type (line 7)
# Getting the type of 'np' (line 7)
np_691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 24), 'np', False)
# Obtaining the member 'ubyte' of a type (line 7)
ubyte_692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 24), np_691, 'ubyte')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 19), tuple_689, ubyte_692)
# Adding element type (line 7)
int_693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 34), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 19), tuple_689, int_693)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 17), list_678, tuple_689)
# Adding element type (line 5)

# Obtaining an instance of the builtin type 'tuple' (line 8)
tuple_694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 19), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 8)
# Adding element type (line 8)
str_695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 19), 'str', 'a')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 19), tuple_694, str_695)
# Adding element type (line 8)
# Getting the type of 'np' (line 8)
np_696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 24), 'np', False)
# Obtaining the member 'ubyte' of a type (line 8)
ubyte_697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 24), np_696, 'ubyte')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 19), tuple_694, ubyte_697)
# Adding element type (line 8)
int_698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 34), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 19), tuple_694, int_698)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 17), list_678, tuple_694)

# Processing the call keyword arguments (line 5)
kwargs_699 = {}
# Getting the type of 'np' (line 5)
np_676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 8), 'np', False)
# Obtaining the member 'dtype' of a type (line 5)
dtype_677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 8), np_676, 'dtype')
# Calling dtype(args, kwargs) (line 5)
dtype_call_result_700 = invoke(stypy.reporting.localization.Localization(__file__, 5, 8), dtype_677, *[list_678], **kwargs_699)

# Assigning a type to the variable 'color' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'color', dtype_call_result_700)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
