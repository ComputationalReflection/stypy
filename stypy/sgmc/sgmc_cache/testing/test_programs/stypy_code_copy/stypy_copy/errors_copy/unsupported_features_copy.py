
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from type_error_copy import TypeError
2: 
3: 
4: def create_unsupported_python_feature_message(localization, feature, description):
5:     '''
6:     Helper function to create a TypeError to indicate the usage of an stypy unsupported feature
7:     :param localization: Caller information
8:     :param feature: Used feature name
9:     :param description: Description of why this feature is unsupported
10:     :return: A TypeError with a custom message
11:     '''
12:     unsupported_error = TypeError(localization, "Unsupported feature '{0}': '{1}'".format(feature, description))
13:     TypeError.usage_of_unsupported_feature = True
14:     return unsupported_error
15: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from type_error_copy import TypeError' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/errors_copy/')
import_4178 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'type_error_copy')

if (type(import_4178) is not StypyTypeError):

    if (import_4178 != 'pyd_module'):
        __import__(import_4178)
        sys_modules_4179 = sys.modules[import_4178]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'type_error_copy', sys_modules_4179.module_type_store, module_type_store, ['TypeError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_4179, sys_modules_4179.module_type_store, module_type_store)
    else:
        from type_error_copy import TypeError

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'type_error_copy', None, module_type_store, ['TypeError'], [TypeError])

else:
    # Assigning a type to the variable 'type_error_copy' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'type_error_copy', import_4178)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/errors_copy/')


@norecursion
def create_unsupported_python_feature_message(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_unsupported_python_feature_message'
    module_type_store = module_type_store.open_function_context('create_unsupported_python_feature_message', 4, 0, False)
    
    # Passed parameters checking function
    create_unsupported_python_feature_message.stypy_localization = localization
    create_unsupported_python_feature_message.stypy_type_of_self = None
    create_unsupported_python_feature_message.stypy_type_store = module_type_store
    create_unsupported_python_feature_message.stypy_function_name = 'create_unsupported_python_feature_message'
    create_unsupported_python_feature_message.stypy_param_names_list = ['localization', 'feature', 'description']
    create_unsupported_python_feature_message.stypy_varargs_param_name = None
    create_unsupported_python_feature_message.stypy_kwargs_param_name = None
    create_unsupported_python_feature_message.stypy_call_defaults = defaults
    create_unsupported_python_feature_message.stypy_call_varargs = varargs
    create_unsupported_python_feature_message.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_unsupported_python_feature_message', ['localization', 'feature', 'description'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_unsupported_python_feature_message', localization, ['localization', 'feature', 'description'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_unsupported_python_feature_message(...)' code ##################

    str_4180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, (-1)), 'str', '\n    Helper function to create a TypeError to indicate the usage of an stypy unsupported feature\n    :param localization: Caller information\n    :param feature: Used feature name\n    :param description: Description of why this feature is unsupported\n    :return: A TypeError with a custom message\n    ')
    
    # Assigning a Call to a Name (line 12):
    
    # Call to TypeError(...): (line 12)
    # Processing the call arguments (line 12)
    # Getting the type of 'localization' (line 12)
    localization_4182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 34), 'localization', False)
    
    # Call to format(...): (line 12)
    # Processing the call arguments (line 12)
    # Getting the type of 'feature' (line 12)
    feature_4185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 90), 'feature', False)
    # Getting the type of 'description' (line 12)
    description_4186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 99), 'description', False)
    # Processing the call keyword arguments (line 12)
    kwargs_4187 = {}
    str_4183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 48), 'str', "Unsupported feature '{0}': '{1}'")
    # Obtaining the member 'format' of a type (line 12)
    format_4184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 48), str_4183, 'format')
    # Calling format(args, kwargs) (line 12)
    format_call_result_4188 = invoke(stypy.reporting.localization.Localization(__file__, 12, 48), format_4184, *[feature_4185, description_4186], **kwargs_4187)
    
    # Processing the call keyword arguments (line 12)
    kwargs_4189 = {}
    # Getting the type of 'TypeError' (line 12)
    TypeError_4181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 24), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 12)
    TypeError_call_result_4190 = invoke(stypy.reporting.localization.Localization(__file__, 12, 24), TypeError_4181, *[localization_4182, format_call_result_4188], **kwargs_4189)
    
    # Assigning a type to the variable 'unsupported_error' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'unsupported_error', TypeError_call_result_4190)
    
    # Assigning a Name to a Attribute (line 13):
    # Getting the type of 'True' (line 13)
    True_4191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 45), 'True')
    # Getting the type of 'TypeError' (line 13)
    TypeError_4192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'TypeError')
    # Setting the type of the member 'usage_of_unsupported_feature' of a type (line 13)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 4), TypeError_4192, 'usage_of_unsupported_feature', True_4191)
    # Getting the type of 'unsupported_error' (line 14)
    unsupported_error_4193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 11), 'unsupported_error')
    # Assigning a type to the variable 'stypy_return_type' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'stypy_return_type', unsupported_error_4193)
    
    # ################# End of 'create_unsupported_python_feature_message(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_unsupported_python_feature_message' in the type store
    # Getting the type of 'stypy_return_type' (line 4)
    stypy_return_type_4194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4194)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_unsupported_python_feature_message'
    return stypy_return_type_4194

# Assigning a type to the variable 'create_unsupported_python_feature_message' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'create_unsupported_python_feature_message', create_unsupported_python_feature_message)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
