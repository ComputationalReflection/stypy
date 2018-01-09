
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: 
4: def configuration(parent_package='io',top_path=None):
5:     from numpy.distutils.misc_util import Configuration
6:     config = Configuration('arff', parent_package, top_path)
7:     config.add_data_dir('tests')
8:     return config
9: 
10: if __name__ == '__main__':
11:     from numpy.distutils.core import setup
12:     setup(**configuration(top_path='').todict())
13: 

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
    str_129599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 33), 'str', 'io')
    # Getting the type of 'None' (line 4)
    None_129600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 47), 'None')
    defaults = [str_129599, None_129600]
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
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/arff/')
    import_129601 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util')

    if (type(import_129601) is not StypyTypeError):

        if (import_129601 != 'pyd_module'):
            __import__(import_129601)
            sys_modules_129602 = sys.modules[import_129601]
            import_from_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util', sys_modules_129602.module_type_store, module_type_store, ['Configuration'])
            nest_module(stypy.reporting.localization.Localization(__file__, 5, 4), __file__, sys_modules_129602, sys_modules_129602.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration

            import_from_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 5)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util', import_129601)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/arff/')
    
    
    # Assigning a Call to a Name (line 6):
    
    # Call to Configuration(...): (line 6)
    # Processing the call arguments (line 6)
    str_129604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 27), 'str', 'arff')
    # Getting the type of 'parent_package' (line 6)
    parent_package_129605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 35), 'parent_package', False)
    # Getting the type of 'top_path' (line 6)
    top_path_129606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 51), 'top_path', False)
    # Processing the call keyword arguments (line 6)
    kwargs_129607 = {}
    # Getting the type of 'Configuration' (line 6)
    Configuration_129603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 6)
    Configuration_call_result_129608 = invoke(stypy.reporting.localization.Localization(__file__, 6, 13), Configuration_129603, *[str_129604, parent_package_129605, top_path_129606], **kwargs_129607)
    
    # Assigning a type to the variable 'config' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'config', Configuration_call_result_129608)
    
    # Call to add_data_dir(...): (line 7)
    # Processing the call arguments (line 7)
    str_129611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 24), 'str', 'tests')
    # Processing the call keyword arguments (line 7)
    kwargs_129612 = {}
    # Getting the type of 'config' (line 7)
    config_129609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'config', False)
    # Obtaining the member 'add_data_dir' of a type (line 7)
    add_data_dir_129610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), config_129609, 'add_data_dir')
    # Calling add_data_dir(args, kwargs) (line 7)
    add_data_dir_call_result_129613 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), add_data_dir_129610, *[str_129611], **kwargs_129612)
    
    # Getting the type of 'config' (line 8)
    config_129614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'stypy_return_type', config_129614)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 4)
    stypy_return_type_129615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_129615)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_129615

# Assigning a type to the variable 'configuration' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 11)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/arff/')
    import_129616 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'numpy.distutils.core')

    if (type(import_129616) is not StypyTypeError):

        if (import_129616 != 'pyd_module'):
            __import__(import_129616)
            sys_modules_129617 = sys.modules[import_129616]
            import_from_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'numpy.distutils.core', sys_modules_129617.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 11, 4), __file__, sys_modules_129617, sys_modules_129617.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'numpy.distutils.core', import_129616)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/arff/')
    
    
    # Call to setup(...): (line 12)
    # Processing the call keyword arguments (line 12)
    
    # Call to todict(...): (line 12)
    # Processing the call keyword arguments (line 12)
    kwargs_129625 = {}
    
    # Call to configuration(...): (line 12)
    # Processing the call keyword arguments (line 12)
    str_129620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 35), 'str', '')
    keyword_129621 = str_129620
    kwargs_129622 = {'top_path': keyword_129621}
    # Getting the type of 'configuration' (line 12)
    configuration_129619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'configuration', False)
    # Calling configuration(args, kwargs) (line 12)
    configuration_call_result_129623 = invoke(stypy.reporting.localization.Localization(__file__, 12, 12), configuration_129619, *[], **kwargs_129622)
    
    # Obtaining the member 'todict' of a type (line 12)
    todict_129624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 12), configuration_call_result_129623, 'todict')
    # Calling todict(args, kwargs) (line 12)
    todict_call_result_129626 = invoke(stypy.reporting.localization.Localization(__file__, 12, 12), todict_129624, *[], **kwargs_129625)
    
    kwargs_129627 = {'todict_call_result_129626': todict_call_result_129626}
    # Getting the type of 'setup' (line 12)
    setup_129618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 12)
    setup_call_result_129628 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), setup_129618, *[], **kwargs_129627)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
