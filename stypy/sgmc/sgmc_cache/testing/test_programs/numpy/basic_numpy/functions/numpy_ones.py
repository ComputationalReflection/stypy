
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://www.python-course.eu/numpy.php
2: 
3: import numpy as np
4: 
5: E = np.ones((2, 3))
6: 
7: F = np.ones((3, 4), dtype=int)
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

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import numpy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')
import_1981 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy')

if (type(import_1981) is not StypyTypeError):

    if (import_1981 != 'pyd_module'):
        __import__(import_1981)
        sys_modules_1982 = sys.modules[import_1981]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', sys_modules_1982.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy', import_1981)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/functions/')


# Assigning a Call to a Name (line 5):

# Call to ones(...): (line 5)
# Processing the call arguments (line 5)

# Obtaining an instance of the builtin type 'tuple' (line 5)
tuple_1985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 5)
# Adding element type (line 5)
int_1986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), tuple_1985, int_1986)
# Adding element type (line 5)
int_1987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 13), tuple_1985, int_1987)

# Processing the call keyword arguments (line 5)
kwargs_1988 = {}
# Getting the type of 'np' (line 5)
np_1983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'np', False)
# Obtaining the member 'ones' of a type (line 5)
ones_1984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 4), np_1983, 'ones')
# Calling ones(args, kwargs) (line 5)
ones_call_result_1989 = invoke(stypy.reporting.localization.Localization(__file__, 5, 4), ones_1984, *[tuple_1985], **kwargs_1988)

# Assigning a type to the variable 'E' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'E', ones_call_result_1989)

# Assigning a Call to a Name (line 7):

# Call to ones(...): (line 7)
# Processing the call arguments (line 7)

# Obtaining an instance of the builtin type 'tuple' (line 7)
tuple_1992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 7)
# Adding element type (line 7)
int_1993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 13), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 13), tuple_1992, int_1993)
# Adding element type (line 7)
int_1994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 16), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 13), tuple_1992, int_1994)

# Processing the call keyword arguments (line 7)
# Getting the type of 'int' (line 7)
int_1995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 26), 'int', False)
keyword_1996 = int_1995
kwargs_1997 = {'dtype': keyword_1996}
# Getting the type of 'np' (line 7)
np_1990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'np', False)
# Obtaining the member 'ones' of a type (line 7)
ones_1991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), np_1990, 'ones')
# Calling ones(args, kwargs) (line 7)
ones_call_result_1998 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), ones_1991, *[tuple_1992], **kwargs_1997)

# Assigning a type to the variable 'F' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'F', ones_call_result_1998)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
