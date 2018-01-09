
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: import numpy.lib.type_check
3: 
4: r = numpy.lib.type_check.iscomplex([1+1j, 1+0j, 4.5, 3, 2, 2j])
5: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import numpy.lib.type_check' statement (line 2)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')
import_209079 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy.lib.type_check')

if (type(import_209079) is not StypyTypeError):

    if (import_209079 != 'pyd_module'):
        __import__(import_209079)
        sys_modules_209080 = sys.modules[import_209079]
        import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy.lib.type_check', sys_modules_209080.module_type_store, module_type_store)
    else:
        import numpy.lib.type_check

        import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy.lib.type_check', numpy.lib.type_check, module_type_store)

else:
    # Assigning a type to the variable 'numpy.lib.type_check' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'numpy.lib.type_check', import_209079)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing//test_programs/numpy/basic_numpy/')


# Assigning a Call to a Name (line 4):

# Call to iscomplex(...): (line 4)
# Processing the call arguments (line 4)

# Obtaining an instance of the builtin type 'list' (line 4)
list_209085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 35), 'list')
# Adding type elements to the builtin type 'list' instance (line 4)
# Adding element type (line 4)
int_209086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 36), 'int')
complex_209087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 38), 'complex')
# Applying the binary operator '+' (line 4)
result_add_209088 = python_operator(stypy.reporting.localization.Localization(__file__, 4, 36), '+', int_209086, complex_209087)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 35), list_209085, result_add_209088)
# Adding element type (line 4)
int_209089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 42), 'int')
complex_209090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 44), 'complex')
# Applying the binary operator '+' (line 4)
result_add_209091 = python_operator(stypy.reporting.localization.Localization(__file__, 4, 42), '+', int_209089, complex_209090)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 35), list_209085, result_add_209091)
# Adding element type (line 4)
float_209092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 48), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 35), list_209085, float_209092)
# Adding element type (line 4)
int_209093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 53), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 35), list_209085, int_209093)
# Adding element type (line 4)
int_209094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 56), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 35), list_209085, int_209094)
# Adding element type (line 4)
complex_209095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 59), 'complex')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 4, 35), list_209085, complex_209095)

# Processing the call keyword arguments (line 4)
kwargs_209096 = {}
# Getting the type of 'numpy' (line 4)
numpy_209081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'numpy', False)
# Obtaining the member 'lib' of a type (line 4)
lib_209082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 4), numpy_209081, 'lib')
# Obtaining the member 'type_check' of a type (line 4)
type_check_209083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 4), lib_209082, 'type_check')
# Obtaining the member 'iscomplex' of a type (line 4)
iscomplex_209084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 4), type_check_209083, 'iscomplex')
# Calling iscomplex(args, kwargs) (line 4)
iscomplex_call_result_209097 = invoke(stypy.reporting.localization.Localization(__file__, 4, 4), iscomplex_209084, *[list_209085], **kwargs_209096)

# Assigning a type to the variable 'r' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'r', iscomplex_call_result_209097)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
