
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import numpy.core.umath
2: 
3: 
4: x = numpy.core.umath.sin
5: print x
6: 
7: y = numpy.core.umath.sin(5)
8: print y

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import numpy.core.umath' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')
import_209103 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy.core.umath')

if (type(import_209103) is not StypyTypeError):

    if (import_209103 != 'pyd_module'):
        __import__(import_209103)
        sys_modules_209104 = sys.modules[import_209103]
        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy.core.umath', sys_modules_209104.module_type_store, module_type_store)
    else:
        import numpy.core.umath

        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy.core.umath', numpy.core.umath, module_type_store)

else:
    # Assigning a type to the variable 'numpy.core.umath' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy.core.umath', import_209103)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')


# Assigning a Attribute to a Name (line 4):
# Getting the type of 'numpy' (line 4)
numpy_209105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'numpy')
# Obtaining the member 'core' of a type (line 4)
core_209106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 4), numpy_209105, 'core')
# Obtaining the member 'umath' of a type (line 4)
umath_209107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 4), core_209106, 'umath')
# Obtaining the member 'sin' of a type (line 4)
sin_209108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 4), umath_209107, 'sin')
# Assigning a type to the variable 'x' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'x', sin_209108)
# Getting the type of 'x' (line 5)
x_209109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 6), 'x')

# Assigning a Call to a Name (line 7):

# Call to sin(...): (line 7)
# Processing the call arguments (line 7)
int_209114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 25), 'int')
# Processing the call keyword arguments (line 7)
kwargs_209115 = {}
# Getting the type of 'numpy' (line 7)
numpy_209110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy', False)
# Obtaining the member 'core' of a type (line 7)
core_209111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), numpy_209110, 'core')
# Obtaining the member 'umath' of a type (line 7)
umath_209112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), core_209111, 'umath')
# Obtaining the member 'sin' of a type (line 7)
sin_209113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), umath_209112, 'sin')
# Calling sin(args, kwargs) (line 7)
sin_call_result_209116 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), sin_209113, *[int_209114], **kwargs_209115)

# Assigning a type to the variable 'y' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'y', sin_call_result_209116)
# Getting the type of 'y' (line 8)
y_209117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 6), 'y')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
