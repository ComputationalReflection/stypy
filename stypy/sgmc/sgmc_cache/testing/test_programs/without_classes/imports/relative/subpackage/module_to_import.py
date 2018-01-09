
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: from . import import_all_non_python_module_relative
3: 
4: global_a = 1
5: 
6: 
7: def f_parent():
8:     local_a = 2
9: 
10:     return import_all_non_python_module_relative.x
11: 
12: x = import_all_non_python_module_relative.x
13: y = import_all_non_python_module_relative.y
14: 
15: 
16: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'from testing.test_programs.without_classes.imports.relative.subpackage import import_all_non_python_module_relative' statement (line 2)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/without_classes/imports/relative/subpackage/')
import_5205 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'testing.test_programs.without_classes.imports.relative.subpackage')

if (type(import_5205) is not StypyTypeError):

    if (import_5205 != 'pyd_module'):
        __import__(import_5205)
        sys_modules_5206 = sys.modules[import_5205]
        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'testing.test_programs.without_classes.imports.relative.subpackage', sys_modules_5206.module_type_store, module_type_store, ['import_all_non_python_module_relative'])
        nest_module(stypy.reporting.localization.Localization(__file__, 2, 0), __file__, sys_modules_5206, sys_modules_5206.module_type_store, module_type_store)
    else:
        from testing.test_programs.without_classes.imports.relative.subpackage import import_all_non_python_module_relative

        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'testing.test_programs.without_classes.imports.relative.subpackage', None, module_type_store, ['import_all_non_python_module_relative'], [import_all_non_python_module_relative])

else:
    # Assigning a type to the variable 'testing.test_programs.without_classes.imports.relative.subpackage' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'testing.test_programs.without_classes.imports.relative.subpackage', import_5205)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/without_classes/imports/relative/subpackage/')


# Assigning a Num to a Name (line 4):
int_5207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 11), 'int')
# Assigning a type to the variable 'global_a' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'global_a', int_5207)

@norecursion
def f_parent(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'f_parent'
    module_type_store = module_type_store.open_function_context('f_parent', 7, 0, False)
    
    # Passed parameters checking function
    f_parent.stypy_localization = localization
    f_parent.stypy_type_of_self = None
    f_parent.stypy_type_store = module_type_store
    f_parent.stypy_function_name = 'f_parent'
    f_parent.stypy_param_names_list = []
    f_parent.stypy_varargs_param_name = None
    f_parent.stypy_kwargs_param_name = None
    f_parent.stypy_call_defaults = defaults
    f_parent.stypy_call_varargs = varargs
    f_parent.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'f_parent', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'f_parent', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'f_parent(...)' code ##################

    
    # Assigning a Num to a Name (line 8):
    int_5208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 14), 'int')
    # Assigning a type to the variable 'local_a' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'local_a', int_5208)
    # Getting the type of 'import_all_non_python_module_relative' (line 10)
    import_all_non_python_module_relative_5209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 11), 'import_all_non_python_module_relative')
    # Obtaining the member 'x' of a type (line 10)
    x_5210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 11), import_all_non_python_module_relative_5209, 'x')
    # Assigning a type to the variable 'stypy_return_type' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'stypy_return_type', x_5210)
    
    # ################# End of 'f_parent(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'f_parent' in the type store
    # Getting the type of 'stypy_return_type' (line 7)
    stypy_return_type_5211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5211)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'f_parent'
    return stypy_return_type_5211

# Assigning a type to the variable 'f_parent' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'f_parent', f_parent)

# Assigning a Attribute to a Name (line 12):
# Getting the type of 'import_all_non_python_module_relative' (line 12)
import_all_non_python_module_relative_5212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'import_all_non_python_module_relative')
# Obtaining the member 'x' of a type (line 12)
x_5213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 4), import_all_non_python_module_relative_5212, 'x')
# Assigning a type to the variable 'x' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'x', x_5213)

# Assigning a Attribute to a Name (line 13):
# Getting the type of 'import_all_non_python_module_relative' (line 13)
import_all_non_python_module_relative_5214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'import_all_non_python_module_relative')
# Obtaining the member 'y' of a type (line 13)
y_5215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 4), import_all_non_python_module_relative_5214, 'y')
# Assigning a type to the variable 'y' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'y', y_5215)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
