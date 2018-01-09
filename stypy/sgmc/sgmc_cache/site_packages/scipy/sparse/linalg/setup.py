
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: 
4: def configuration(parent_package='',top_path=None):
5:     from numpy.distutils.misc_util import Configuration
6: 
7:     config = Configuration('linalg',parent_package,top_path)
8: 
9:     config.add_subpackage(('isolve'))
10:     config.add_subpackage(('dsolve'))
11:     config.add_subpackage(('eigen'))
12: 
13:     config.add_data_dir('tests')
14: 
15:     return config
16: 
17: if __name__ == '__main__':
18:     from numpy.distutils.core import setup
19:     setup(**configuration(top_path='').todict())
20: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


@norecursion
def configuration(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_388531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 33), 'str', '')
    # Getting the type of 'None' (line 4)
    None_388532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 45), 'None')
    defaults = [str_388531, None_388532]
    # Create a new context for function 'configuration'
    module_type_store = module_type_store.open_function_context('configuration', 4, 0, False)
    
    # Passed parameters checking function
    configuration.stypy_localization = localization
    configuration.stypy_type_of_self = None
    configuration.stypy_type_store = module_type_store
    configuration.stypy_function_name = 'configuration'
    configuration.stypy_param_names_list = ['parent_package', 'top_path']
    configuration.stypy_varargs_param_name = None
    configuration.stypy_kwargs_param_name = None
    configuration.stypy_call_defaults = defaults
    configuration.stypy_call_varargs = varargs
    configuration.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'configuration', ['parent_package', 'top_path'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'configuration', localization, ['parent_package', 'top_path'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'configuration(...)' code ##################

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 4))
    
    # 'from numpy.distutils.misc_util import Configuration' statement (line 5)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/')
    import_388533 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util')

    if (type(import_388533) is not StypyTypeError):

        if (import_388533 != 'pyd_module'):
            __import__(import_388533)
            sys_modules_388534 = sys.modules[import_388533]
            import_from_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util', sys_modules_388534.module_type_store, module_type_store, ['Configuration'])
            nest_module(stypy.reporting.localization.Localization(__file__, 5, 4), __file__, sys_modules_388534, sys_modules_388534.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration

            import_from_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 5)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util', import_388533)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/')
    
    
    # Assigning a Call to a Name (line 7):
    
    # Call to Configuration(...): (line 7)
    # Processing the call arguments (line 7)
    str_388536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 27), 'str', 'linalg')
    # Getting the type of 'parent_package' (line 7)
    parent_package_388537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 36), 'parent_package', False)
    # Getting the type of 'top_path' (line 7)
    top_path_388538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 51), 'top_path', False)
    # Processing the call keyword arguments (line 7)
    kwargs_388539 = {}
    # Getting the type of 'Configuration' (line 7)
    Configuration_388535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 7)
    Configuration_call_result_388540 = invoke(stypy.reporting.localization.Localization(__file__, 7, 13), Configuration_388535, *[str_388536, parent_package_388537, top_path_388538], **kwargs_388539)
    
    # Assigning a type to the variable 'config' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'config', Configuration_call_result_388540)
    
    # Call to add_subpackage(...): (line 9)
    # Processing the call arguments (line 9)
    str_388543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 27), 'str', 'isolve')
    # Processing the call keyword arguments (line 9)
    kwargs_388544 = {}
    # Getting the type of 'config' (line 9)
    config_388541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 9)
    add_subpackage_388542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 4), config_388541, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 9)
    add_subpackage_call_result_388545 = invoke(stypy.reporting.localization.Localization(__file__, 9, 4), add_subpackage_388542, *[str_388543], **kwargs_388544)
    
    
    # Call to add_subpackage(...): (line 10)
    # Processing the call arguments (line 10)
    str_388548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 27), 'str', 'dsolve')
    # Processing the call keyword arguments (line 10)
    kwargs_388549 = {}
    # Getting the type of 'config' (line 10)
    config_388546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 10)
    add_subpackage_388547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 4), config_388546, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 10)
    add_subpackage_call_result_388550 = invoke(stypy.reporting.localization.Localization(__file__, 10, 4), add_subpackage_388547, *[str_388548], **kwargs_388549)
    
    
    # Call to add_subpackage(...): (line 11)
    # Processing the call arguments (line 11)
    str_388553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 27), 'str', 'eigen')
    # Processing the call keyword arguments (line 11)
    kwargs_388554 = {}
    # Getting the type of 'config' (line 11)
    config_388551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 11)
    add_subpackage_388552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 4), config_388551, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 11)
    add_subpackage_call_result_388555 = invoke(stypy.reporting.localization.Localization(__file__, 11, 4), add_subpackage_388552, *[str_388553], **kwargs_388554)
    
    
    # Call to add_data_dir(...): (line 13)
    # Processing the call arguments (line 13)
    str_388558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 24), 'str', 'tests')
    # Processing the call keyword arguments (line 13)
    kwargs_388559 = {}
    # Getting the type of 'config' (line 13)
    config_388556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'config', False)
    # Obtaining the member 'add_data_dir' of a type (line 13)
    add_data_dir_388557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 4), config_388556, 'add_data_dir')
    # Calling add_data_dir(args, kwargs) (line 13)
    add_data_dir_call_result_388560 = invoke(stypy.reporting.localization.Localization(__file__, 13, 4), add_data_dir_388557, *[str_388558], **kwargs_388559)
    
    # Getting the type of 'config' (line 15)
    config_388561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'stypy_return_type', config_388561)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 4)
    stypy_return_type_388562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_388562)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_388562

# Assigning a type to the variable 'configuration' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 18)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/')
    import_388563 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 4), 'numpy.distutils.core')

    if (type(import_388563) is not StypyTypeError):

        if (import_388563 != 'pyd_module'):
            __import__(import_388563)
            sys_modules_388564 = sys.modules[import_388563]
            import_from_module(stypy.reporting.localization.Localization(__file__, 18, 4), 'numpy.distutils.core', sys_modules_388564.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 18, 4), __file__, sys_modules_388564, sys_modules_388564.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 18, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'numpy.distutils.core', import_388563)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/')
    
    
    # Call to setup(...): (line 19)
    # Processing the call keyword arguments (line 19)
    
    # Call to todict(...): (line 19)
    # Processing the call keyword arguments (line 19)
    kwargs_388572 = {}
    
    # Call to configuration(...): (line 19)
    # Processing the call keyword arguments (line 19)
    str_388567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 35), 'str', '')
    keyword_388568 = str_388567
    kwargs_388569 = {'top_path': keyword_388568}
    # Getting the type of 'configuration' (line 19)
    configuration_388566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'configuration', False)
    # Calling configuration(args, kwargs) (line 19)
    configuration_call_result_388570 = invoke(stypy.reporting.localization.Localization(__file__, 19, 12), configuration_388566, *[], **kwargs_388569)
    
    # Obtaining the member 'todict' of a type (line 19)
    todict_388571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 12), configuration_call_result_388570, 'todict')
    # Calling todict(args, kwargs) (line 19)
    todict_call_result_388573 = invoke(stypy.reporting.localization.Localization(__file__, 19, 12), todict_388571, *[], **kwargs_388572)
    
    kwargs_388574 = {'todict_call_result_388573': todict_call_result_388573}
    # Getting the type of 'setup' (line 19)
    setup_388565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 19)
    setup_call_result_388575 = invoke(stypy.reporting.localization.Localization(__file__, 19, 4), setup_388565, *[], **kwargs_388574)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
