
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: 
4: def configuration(parent_package='',top_path=None):
5:     from numpy.distutils.misc_util import Configuration
6:     config = Configuration('misc',parent_package,top_path)
7:     config.add_data_files('*.dat')
8:     config.add_data_dir('tests')
9:     return config
10: 
11: if __name__ == '__main__':
12:     from numpy.distutils.core import setup
13:     setup(**configuration(top_path='').todict())
14: 

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
    str_115189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 33), 'str', '')
    # Getting the type of 'None' (line 4)
    None_115190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 45), 'None')
    defaults = [str_115189, None_115190]
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
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/misc/')
    import_115191 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util')

    if (type(import_115191) is not StypyTypeError):

        if (import_115191 != 'pyd_module'):
            __import__(import_115191)
            sys_modules_115192 = sys.modules[import_115191]
            import_from_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util', sys_modules_115192.module_type_store, module_type_store, ['Configuration'])
            nest_module(stypy.reporting.localization.Localization(__file__, 5, 4), __file__, sys_modules_115192, sys_modules_115192.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration

            import_from_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 5)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util', import_115191)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/misc/')
    
    
    # Assigning a Call to a Name (line 6):
    
    # Call to Configuration(...): (line 6)
    # Processing the call arguments (line 6)
    str_115194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 27), 'str', 'misc')
    # Getting the type of 'parent_package' (line 6)
    parent_package_115195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 34), 'parent_package', False)
    # Getting the type of 'top_path' (line 6)
    top_path_115196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 49), 'top_path', False)
    # Processing the call keyword arguments (line 6)
    kwargs_115197 = {}
    # Getting the type of 'Configuration' (line 6)
    Configuration_115193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 6)
    Configuration_call_result_115198 = invoke(stypy.reporting.localization.Localization(__file__, 6, 13), Configuration_115193, *[str_115194, parent_package_115195, top_path_115196], **kwargs_115197)
    
    # Assigning a type to the variable 'config' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'config', Configuration_call_result_115198)
    
    # Call to add_data_files(...): (line 7)
    # Processing the call arguments (line 7)
    str_115201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 26), 'str', '*.dat')
    # Processing the call keyword arguments (line 7)
    kwargs_115202 = {}
    # Getting the type of 'config' (line 7)
    config_115199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'config', False)
    # Obtaining the member 'add_data_files' of a type (line 7)
    add_data_files_115200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), config_115199, 'add_data_files')
    # Calling add_data_files(args, kwargs) (line 7)
    add_data_files_call_result_115203 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), add_data_files_115200, *[str_115201], **kwargs_115202)
    
    
    # Call to add_data_dir(...): (line 8)
    # Processing the call arguments (line 8)
    str_115206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 24), 'str', 'tests')
    # Processing the call keyword arguments (line 8)
    kwargs_115207 = {}
    # Getting the type of 'config' (line 8)
    config_115204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'config', False)
    # Obtaining the member 'add_data_dir' of a type (line 8)
    add_data_dir_115205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 4), config_115204, 'add_data_dir')
    # Calling add_data_dir(args, kwargs) (line 8)
    add_data_dir_call_result_115208 = invoke(stypy.reporting.localization.Localization(__file__, 8, 4), add_data_dir_115205, *[str_115206], **kwargs_115207)
    
    # Getting the type of 'config' (line 9)
    config_115209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'stypy_return_type', config_115209)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 4)
    stypy_return_type_115210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_115210)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_115210

# Assigning a type to the variable 'configuration' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 12)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/misc/')
    import_115211 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 4), 'numpy.distutils.core')

    if (type(import_115211) is not StypyTypeError):

        if (import_115211 != 'pyd_module'):
            __import__(import_115211)
            sys_modules_115212 = sys.modules[import_115211]
            import_from_module(stypy.reporting.localization.Localization(__file__, 12, 4), 'numpy.distutils.core', sys_modules_115212.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 12, 4), __file__, sys_modules_115212, sys_modules_115212.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 12, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'numpy.distutils.core', import_115211)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/misc/')
    
    
    # Call to setup(...): (line 13)
    # Processing the call keyword arguments (line 13)
    
    # Call to todict(...): (line 13)
    # Processing the call keyword arguments (line 13)
    kwargs_115220 = {}
    
    # Call to configuration(...): (line 13)
    # Processing the call keyword arguments (line 13)
    str_115215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 35), 'str', '')
    keyword_115216 = str_115215
    kwargs_115217 = {'top_path': keyword_115216}
    # Getting the type of 'configuration' (line 13)
    configuration_115214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 12), 'configuration', False)
    # Calling configuration(args, kwargs) (line 13)
    configuration_call_result_115218 = invoke(stypy.reporting.localization.Localization(__file__, 13, 12), configuration_115214, *[], **kwargs_115217)
    
    # Obtaining the member 'todict' of a type (line 13)
    todict_115219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 12), configuration_call_result_115218, 'todict')
    # Calling todict(args, kwargs) (line 13)
    todict_call_result_115221 = invoke(stypy.reporting.localization.Localization(__file__, 13, 12), todict_115219, *[], **kwargs_115220)
    
    kwargs_115222 = {'todict_call_result_115221': todict_call_result_115221}
    # Getting the type of 'setup' (line 13)
    setup_115213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 13)
    setup_call_result_115223 = invoke(stypy.reporting.localization.Localization(__file__, 13, 4), setup_115213, *[], **kwargs_115222)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
