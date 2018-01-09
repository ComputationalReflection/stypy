
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import other_module_to_import
2: 
3: def f():
4:     return 3
5: 
6: r1 = f()
7: 
8: r2 = other_module_to_import.global_a
9: r3 = other_module_to_import.f_parent()
10: r4 = other_module_to_import.my_func
11: r5 = other_module_to_import.secondary_module
12: r6 = r5.number

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import other_module_to_import' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/without_classes/imports/')
import_5027 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'other_module_to_import')

if (type(import_5027) is not StypyTypeError):

    if (import_5027 != 'pyd_module'):
        __import__(import_5027)
        sys_modules_5028 = sys.modules[import_5027]
        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'other_module_to_import', sys_modules_5028.module_type_store, module_type_store)
    else:
        import other_module_to_import

        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'other_module_to_import', other_module_to_import, module_type_store)

else:
    # Assigning a type to the variable 'other_module_to_import' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'other_module_to_import', import_5027)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/without_classes/imports/')


@norecursion
def f(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'f'
    module_type_store = module_type_store.open_function_context('f', 3, 0, False)
    
    # Passed parameters checking function
    f.stypy_localization = localization
    f.stypy_type_of_self = None
    f.stypy_type_store = module_type_store
    f.stypy_function_name = 'f'
    f.stypy_param_names_list = []
    f.stypy_varargs_param_name = None
    f.stypy_kwargs_param_name = None
    f.stypy_call_defaults = defaults
    f.stypy_call_varargs = varargs
    f.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'f', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'f', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'f(...)' code ##################

    int_5029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 11), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'stypy_return_type', int_5029)
    
    # ################# End of 'f(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f' in the type store
    # Getting the type of 'stypy_return_type' (line 3)
    stypy_return_type_5030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5030)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f'
    return stypy_return_type_5030

# Assigning a type to the variable 'f' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'f', f)

# Assigning a Call to a Name (line 6):

# Call to f(...): (line 6)
# Processing the call keyword arguments (line 6)
kwargs_5032 = {}
# Getting the type of 'f' (line 6)
f_5031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 5), 'f', False)
# Calling f(args, kwargs) (line 6)
f_call_result_5033 = invoke(stypy.reporting.localization.Localization(__file__, 6, 5), f_5031, *[], **kwargs_5032)

# Assigning a type to the variable 'r1' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'r1', f_call_result_5033)

# Assigning a Attribute to a Name (line 8):
# Getting the type of 'other_module_to_import' (line 8)
other_module_to_import_5034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 5), 'other_module_to_import')
# Obtaining the member 'global_a' of a type (line 8)
global_a_5035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 5), other_module_to_import_5034, 'global_a')
# Assigning a type to the variable 'r2' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'r2', global_a_5035)

# Assigning a Call to a Name (line 9):

# Call to f_parent(...): (line 9)
# Processing the call keyword arguments (line 9)
kwargs_5038 = {}
# Getting the type of 'other_module_to_import' (line 9)
other_module_to_import_5036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 5), 'other_module_to_import', False)
# Obtaining the member 'f_parent' of a type (line 9)
f_parent_5037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 5), other_module_to_import_5036, 'f_parent')
# Calling f_parent(args, kwargs) (line 9)
f_parent_call_result_5039 = invoke(stypy.reporting.localization.Localization(__file__, 9, 5), f_parent_5037, *[], **kwargs_5038)

# Assigning a type to the variable 'r3' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'r3', f_parent_call_result_5039)

# Assigning a Attribute to a Name (line 10):
# Getting the type of 'other_module_to_import' (line 10)
other_module_to_import_5040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 5), 'other_module_to_import')
# Obtaining the member 'my_func' of a type (line 10)
my_func_5041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 5), other_module_to_import_5040, 'my_func')
# Assigning a type to the variable 'r4' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'r4', my_func_5041)

# Assigning a Attribute to a Name (line 11):
# Getting the type of 'other_module_to_import' (line 11)
other_module_to_import_5042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 5), 'other_module_to_import')
# Obtaining the member 'secondary_module' of a type (line 11)
secondary_module_5043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 5), other_module_to_import_5042, 'secondary_module')
# Assigning a type to the variable 'r5' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'r5', secondary_module_5043)

# Assigning a Attribute to a Name (line 12):
# Getting the type of 'r5' (line 12)
r5_5044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 5), 'r5')
# Obtaining the member 'number' of a type (line 12)
number_5045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 5), r5_5044, 'number')
# Assigning a type to the variable 'r6' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'r6', number_5045)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
