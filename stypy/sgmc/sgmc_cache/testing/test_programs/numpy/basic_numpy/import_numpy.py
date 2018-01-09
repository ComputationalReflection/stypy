
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import numpy
2: 
3: 
4: x = numpy.float32
5: print x

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import numpy' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')
import_209059 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy')

if (type(import_209059) is not StypyTypeError):

    if (import_209059 != 'pyd_module'):
        __import__(import_209059)
        sys_modules_209060 = sys.modules[import_209059]
        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy', sys_modules_209060.module_type_store, module_type_store)
    else:
        import numpy

        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy', import_209059)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')


# Assigning a Attribute to a Name (line 4):
# Getting the type of 'numpy' (line 4)
numpy_209061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'numpy')
# Obtaining the member 'float32' of a type (line 4)
float32_209062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 4), numpy_209061, 'float32')
# Assigning a type to the variable 'x' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'x', float32_209062)
# Getting the type of 'x' (line 5)
x_209063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 6), 'x')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
