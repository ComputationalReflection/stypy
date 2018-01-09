
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.labri.fr/perso/nrougier/teaching/numpy.100/
2: 
3: import numpy as np
4: 
5: nz = np.nonzero([1, 2, 0, 0, 4, 0])
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
import_1887 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_1887) is not StypyTypeError):

    if (import_1887 != 'pyd_module'):
        __import__(import_1887)
        sys_modules_1888 = sys.modules[import_1887]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_1888.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_1887)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Call to nonzero(...): (line 5)
# Processing the call arguments (line 5)

# Obtaining an instance of the builtin type 'list' (line 5)
list_1891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)
int_1892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 16), list_1891, int_1892)
# Adding element type (line 5)
int_1893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 16), list_1891, int_1893)
# Adding element type (line 5)
int_1894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 16), list_1891, int_1894)
# Adding element type (line 5)
int_1895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 16), list_1891, int_1895)
# Adding element type (line 5)
int_1896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 16), list_1891, int_1896)
# Adding element type (line 5)
int_1897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 16), list_1891, int_1897)

# Processing the call keyword arguments (line 5)
kwargs_1898 = {}
# Getting the type of 'np' (line 5)
np_1889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 5), 'np', False)
# Obtaining the member 'nonzero' of a type (line 5)
nonzero_1890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 5), np_1889, 'nonzero')
# Calling nonzero(args, kwargs) (line 5)
nonzero_call_result_1899 = invoke(stypy.reporting.localization.Localization(__file__, 5, 5), nonzero_1890, *[list_1891], **kwargs_1898)

# Assigning a type to the variable 'nz' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'nz', nonzero_call_result_1899)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
