
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #!/usr/bin/env python
2: from __future__ import division, print_function
3: 
4: def configuration(parent_package='',top_path=None):
5:     from numpy.distutils.misc_util import Configuration
6:     config = Configuration('distutils', parent_package, top_path)
7:     config.add_subpackage('command')
8:     config.add_subpackage('fcompiler')
9:     config.add_data_dir('tests')
10:     config.add_data_files('site.cfg')
11:     config.add_data_files('mingw/gfortran_vs2003_hack.c')
12:     config.make_config_py()
13:     return config
14: 
15: if __name__ == '__main__':
16:     from numpy.distutils.core      import setup
17:     setup(configuration=configuration)
18: 

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
    str_45286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 33), 'str', '')
    # Getting the type of 'None' (line 4)
    None_45287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 45), 'None')
    defaults = [str_45286, None_45287]
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
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
    import_45288 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util')

    if (type(import_45288) is not StypyTypeError):

        if (import_45288 != 'pyd_module'):
            __import__(import_45288)
            sys_modules_45289 = sys.modules[import_45288]
            import_from_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util', sys_modules_45289.module_type_store, module_type_store, ['Configuration'])
            nest_module(stypy.reporting.localization.Localization(__file__, 5, 4), __file__, sys_modules_45289, sys_modules_45289.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration

            import_from_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 5)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util', import_45288)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')
    
    
    # Assigning a Call to a Name (line 6):
    
    # Call to Configuration(...): (line 6)
    # Processing the call arguments (line 6)
    str_45291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 27), 'str', 'distutils')
    # Getting the type of 'parent_package' (line 6)
    parent_package_45292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 40), 'parent_package', False)
    # Getting the type of 'top_path' (line 6)
    top_path_45293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 56), 'top_path', False)
    # Processing the call keyword arguments (line 6)
    kwargs_45294 = {}
    # Getting the type of 'Configuration' (line 6)
    Configuration_45290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 6)
    Configuration_call_result_45295 = invoke(stypy.reporting.localization.Localization(__file__, 6, 13), Configuration_45290, *[str_45291, parent_package_45292, top_path_45293], **kwargs_45294)
    
    # Assigning a type to the variable 'config' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'config', Configuration_call_result_45295)
    
    # Call to add_subpackage(...): (line 7)
    # Processing the call arguments (line 7)
    str_45298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 26), 'str', 'command')
    # Processing the call keyword arguments (line 7)
    kwargs_45299 = {}
    # Getting the type of 'config' (line 7)
    config_45296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 7)
    add_subpackage_45297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), config_45296, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 7)
    add_subpackage_call_result_45300 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), add_subpackage_45297, *[str_45298], **kwargs_45299)
    
    
    # Call to add_subpackage(...): (line 8)
    # Processing the call arguments (line 8)
    str_45303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 26), 'str', 'fcompiler')
    # Processing the call keyword arguments (line 8)
    kwargs_45304 = {}
    # Getting the type of 'config' (line 8)
    config_45301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 8)
    add_subpackage_45302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 4), config_45301, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 8)
    add_subpackage_call_result_45305 = invoke(stypy.reporting.localization.Localization(__file__, 8, 4), add_subpackage_45302, *[str_45303], **kwargs_45304)
    
    
    # Call to add_data_dir(...): (line 9)
    # Processing the call arguments (line 9)
    str_45308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 24), 'str', 'tests')
    # Processing the call keyword arguments (line 9)
    kwargs_45309 = {}
    # Getting the type of 'config' (line 9)
    config_45306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'config', False)
    # Obtaining the member 'add_data_dir' of a type (line 9)
    add_data_dir_45307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 4), config_45306, 'add_data_dir')
    # Calling add_data_dir(args, kwargs) (line 9)
    add_data_dir_call_result_45310 = invoke(stypy.reporting.localization.Localization(__file__, 9, 4), add_data_dir_45307, *[str_45308], **kwargs_45309)
    
    
    # Call to add_data_files(...): (line 10)
    # Processing the call arguments (line 10)
    str_45313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 26), 'str', 'site.cfg')
    # Processing the call keyword arguments (line 10)
    kwargs_45314 = {}
    # Getting the type of 'config' (line 10)
    config_45311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'config', False)
    # Obtaining the member 'add_data_files' of a type (line 10)
    add_data_files_45312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 4), config_45311, 'add_data_files')
    # Calling add_data_files(args, kwargs) (line 10)
    add_data_files_call_result_45315 = invoke(stypy.reporting.localization.Localization(__file__, 10, 4), add_data_files_45312, *[str_45313], **kwargs_45314)
    
    
    # Call to add_data_files(...): (line 11)
    # Processing the call arguments (line 11)
    str_45318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 26), 'str', 'mingw/gfortran_vs2003_hack.c')
    # Processing the call keyword arguments (line 11)
    kwargs_45319 = {}
    # Getting the type of 'config' (line 11)
    config_45316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'config', False)
    # Obtaining the member 'add_data_files' of a type (line 11)
    add_data_files_45317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 4), config_45316, 'add_data_files')
    # Calling add_data_files(args, kwargs) (line 11)
    add_data_files_call_result_45320 = invoke(stypy.reporting.localization.Localization(__file__, 11, 4), add_data_files_45317, *[str_45318], **kwargs_45319)
    
    
    # Call to make_config_py(...): (line 12)
    # Processing the call keyword arguments (line 12)
    kwargs_45323 = {}
    # Getting the type of 'config' (line 12)
    config_45321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'config', False)
    # Obtaining the member 'make_config_py' of a type (line 12)
    make_config_py_45322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 4), config_45321, 'make_config_py')
    # Calling make_config_py(args, kwargs) (line 12)
    make_config_py_call_result_45324 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), make_config_py_45322, *[], **kwargs_45323)
    
    # Getting the type of 'config' (line 13)
    config_45325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'stypy_return_type', config_45325)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 4)
    stypy_return_type_45326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_45326)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_45326

# Assigning a type to the variable 'configuration' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 16)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
    import_45327 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 4), 'numpy.distutils.core')

    if (type(import_45327) is not StypyTypeError):

        if (import_45327 != 'pyd_module'):
            __import__(import_45327)
            sys_modules_45328 = sys.modules[import_45327]
            import_from_module(stypy.reporting.localization.Localization(__file__, 16, 4), 'numpy.distutils.core', sys_modules_45328.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 16, 4), __file__, sys_modules_45328, sys_modules_45328.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 16, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'numpy.distutils.core', import_45327)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')
    
    
    # Call to setup(...): (line 17)
    # Processing the call keyword arguments (line 17)
    # Getting the type of 'configuration' (line 17)
    configuration_45330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 24), 'configuration', False)
    keyword_45331 = configuration_45330
    kwargs_45332 = {'configuration': keyword_45331}
    # Getting the type of 'setup' (line 17)
    setup_45329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 17)
    setup_call_result_45333 = invoke(stypy.reporting.localization.Localization(__file__, 17, 4), setup_45329, *[], **kwargs_45332)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
