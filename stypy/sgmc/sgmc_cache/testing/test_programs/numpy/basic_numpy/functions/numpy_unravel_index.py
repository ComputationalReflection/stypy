
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: r = (np.unravel_index(100, (6, 7, 8)))
6: 
7: # l = globals().copy()
8: # for v in l:
9: #     print ("'" + v + "'" + ": instance_of_class_name(\"" + type(l[v]).__name__ + "\"),")
10: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_2684 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_2684) is not StypyTypeError):

    if (import_2684 != 'pyd_module'):
        __import__(import_2684)
        sys_modules_2685 = sys.modules[import_2684]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_2685.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_2684)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Call to unravel_index(...): (line 5)
# Processing the call arguments (line 5)
int_2688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 22), 'int')

# Obtaining an instance of the builtin type 'tuple' (line 5)
tuple_2689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 28), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 5)
# Adding element type (line 5)
int_2690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 28), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 28), tuple_2689, int_2690)
# Adding element type (line 5)
int_2691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 31), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 28), tuple_2689, int_2691)
# Adding element type (line 5)
int_2692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 34), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 28), tuple_2689, int_2692)

# Processing the call keyword arguments (line 5)
kwargs_2693 = {}
# Getting the type of 'np' (line 5)
np_2686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 5), 'np', False)
# Obtaining the member 'unravel_index' of a type (line 5)
unravel_index_2687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 5), np_2686, 'unravel_index')
# Calling unravel_index(args, kwargs) (line 5)
unravel_index_call_result_2694 = invoke(stypy.reporting.localization.Localization(__file__, 5, 5), unravel_index_2687, *[int_2688, tuple_2689], **kwargs_2693)

# Assigning a type to the variable 'r' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'r', unravel_index_call_result_2694)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
