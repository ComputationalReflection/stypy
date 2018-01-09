
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from numpy.core import multiarray
2: 
3: 
4: x = multiarray.__doc__
5: print x

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from numpy.core import multiarray' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')
import_209098 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy.core')

if (type(import_209098) is not StypyTypeError):

    if (import_209098 != 'pyd_module'):
        __import__(import_209098)
        sys_modules_209099 = sys.modules[import_209098]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy.core', sys_modules_209099.module_type_store, module_type_store, ['multiarray'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_209099, sys_modules_209099.module_type_store, module_type_store)
    else:
        from numpy.core import multiarray

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy.core', None, module_type_store, ['multiarray'], [multiarray])

else:
    # Assigning a type to the variable 'numpy.core' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy.core', import_209098)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')


# Assigning a Attribute to a Name (line 4):
# Getting the type of 'multiarray' (line 4)
multiarray_209100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'multiarray')
# Obtaining the member '__doc__' of a type (line 4)
doc___209101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 4), multiarray_209100, '__doc__')
# Assigning a type to the variable 'x' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'x', doc___209101)
# Getting the type of 'x' (line 5)
x_209102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 6), 'x')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
