
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #!/usr/bin/env python
2: from __future__ import division, print_function
3: 
4: 
5: def configuration(parent_package='',top_path=None):
6:     from numpy.distutils.misc_util import Configuration
7:     config = Configuration('numpy', parent_package, top_path)
8: 
9:     config.add_subpackage('compat')
10:     config.add_subpackage('core')
11:     config.add_subpackage('distutils')
12:     config.add_subpackage('doc')
13:     config.add_subpackage('f2py')
14:     config.add_subpackage('fft')
15:     config.add_subpackage('lib')
16:     config.add_subpackage('linalg')
17:     config.add_subpackage('ma')
18:     config.add_subpackage('matrixlib')
19:     config.add_subpackage('polynomial')
20:     config.add_subpackage('random')
21:     config.add_subpackage('testing')
22:     config.add_data_dir('doc')
23:     config.add_data_dir('tests')
24:     config.make_config_py() # installs __config__.py
25:     return config
26: 
27: if __name__ == '__main__':
28:     print('This is the wrong setup.py file to run')
29: 

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
    str_24293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 33), 'str', '')
    # Getting the type of 'None' (line 5)
    None_24294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 45), 'None')
    defaults = [str_24293, None_24294]
    # Create a new context for function 'configuration'
    module_type_store = module_type_store.open_function_context('configuration', 5, 0, False)
    
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

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 4))
    
    # 'from numpy.distutils.misc_util import Configuration' statement (line 6)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/')
    import_24295 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 4), 'numpy.distutils.misc_util')

    if (type(import_24295) is not StypyTypeError):

        if (import_24295 != 'pyd_module'):
            __import__(import_24295)
            sys_modules_24296 = sys.modules[import_24295]
            import_from_module(stypy.reporting.localization.Localization(__file__, 6, 4), 'numpy.distutils.misc_util', sys_modules_24296.module_type_store, module_type_store, ['Configuration'])
            nest_module(stypy.reporting.localization.Localization(__file__, 6, 4), __file__, sys_modules_24296, sys_modules_24296.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration

            import_from_module(stypy.reporting.localization.Localization(__file__, 6, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 6)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'numpy.distutils.misc_util', import_24295)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/')
    
    
    # Assigning a Call to a Name (line 7):
    
    # Call to Configuration(...): (line 7)
    # Processing the call arguments (line 7)
    str_24298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 27), 'str', 'numpy')
    # Getting the type of 'parent_package' (line 7)
    parent_package_24299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 36), 'parent_package', False)
    # Getting the type of 'top_path' (line 7)
    top_path_24300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 52), 'top_path', False)
    # Processing the call keyword arguments (line 7)
    kwargs_24301 = {}
    # Getting the type of 'Configuration' (line 7)
    Configuration_24297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 7)
    Configuration_call_result_24302 = invoke(stypy.reporting.localization.Localization(__file__, 7, 13), Configuration_24297, *[str_24298, parent_package_24299, top_path_24300], **kwargs_24301)
    
    # Assigning a type to the variable 'config' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'config', Configuration_call_result_24302)
    
    # Call to add_subpackage(...): (line 9)
    # Processing the call arguments (line 9)
    str_24305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 26), 'str', 'compat')
    # Processing the call keyword arguments (line 9)
    kwargs_24306 = {}
    # Getting the type of 'config' (line 9)
    config_24303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 9)
    add_subpackage_24304 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 4), config_24303, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 9)
    add_subpackage_call_result_24307 = invoke(stypy.reporting.localization.Localization(__file__, 9, 4), add_subpackage_24304, *[str_24305], **kwargs_24306)
    
    
    # Call to add_subpackage(...): (line 10)
    # Processing the call arguments (line 10)
    str_24310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 26), 'str', 'core')
    # Processing the call keyword arguments (line 10)
    kwargs_24311 = {}
    # Getting the type of 'config' (line 10)
    config_24308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 10)
    add_subpackage_24309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 4), config_24308, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 10)
    add_subpackage_call_result_24312 = invoke(stypy.reporting.localization.Localization(__file__, 10, 4), add_subpackage_24309, *[str_24310], **kwargs_24311)
    
    
    # Call to add_subpackage(...): (line 11)
    # Processing the call arguments (line 11)
    str_24315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 26), 'str', 'distutils')
    # Processing the call keyword arguments (line 11)
    kwargs_24316 = {}
    # Getting the type of 'config' (line 11)
    config_24313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 11)
    add_subpackage_24314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 4), config_24313, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 11)
    add_subpackage_call_result_24317 = invoke(stypy.reporting.localization.Localization(__file__, 11, 4), add_subpackage_24314, *[str_24315], **kwargs_24316)
    
    
    # Call to add_subpackage(...): (line 12)
    # Processing the call arguments (line 12)
    str_24320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 26), 'str', 'doc')
    # Processing the call keyword arguments (line 12)
    kwargs_24321 = {}
    # Getting the type of 'config' (line 12)
    config_24318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 12)
    add_subpackage_24319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 4), config_24318, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 12)
    add_subpackage_call_result_24322 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), add_subpackage_24319, *[str_24320], **kwargs_24321)
    
    
    # Call to add_subpackage(...): (line 13)
    # Processing the call arguments (line 13)
    str_24325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 26), 'str', 'f2py')
    # Processing the call keyword arguments (line 13)
    kwargs_24326 = {}
    # Getting the type of 'config' (line 13)
    config_24323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 13)
    add_subpackage_24324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 4), config_24323, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 13)
    add_subpackage_call_result_24327 = invoke(stypy.reporting.localization.Localization(__file__, 13, 4), add_subpackage_24324, *[str_24325], **kwargs_24326)
    
    
    # Call to add_subpackage(...): (line 14)
    # Processing the call arguments (line 14)
    str_24330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 26), 'str', 'fft')
    # Processing the call keyword arguments (line 14)
    kwargs_24331 = {}
    # Getting the type of 'config' (line 14)
    config_24328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 14)
    add_subpackage_24329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 4), config_24328, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 14)
    add_subpackage_call_result_24332 = invoke(stypy.reporting.localization.Localization(__file__, 14, 4), add_subpackage_24329, *[str_24330], **kwargs_24331)
    
    
    # Call to add_subpackage(...): (line 15)
    # Processing the call arguments (line 15)
    str_24335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 26), 'str', 'lib')
    # Processing the call keyword arguments (line 15)
    kwargs_24336 = {}
    # Getting the type of 'config' (line 15)
    config_24333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 15)
    add_subpackage_24334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 4), config_24333, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 15)
    add_subpackage_call_result_24337 = invoke(stypy.reporting.localization.Localization(__file__, 15, 4), add_subpackage_24334, *[str_24335], **kwargs_24336)
    
    
    # Call to add_subpackage(...): (line 16)
    # Processing the call arguments (line 16)
    str_24340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 26), 'str', 'linalg')
    # Processing the call keyword arguments (line 16)
    kwargs_24341 = {}
    # Getting the type of 'config' (line 16)
    config_24338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 16)
    add_subpackage_24339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 4), config_24338, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 16)
    add_subpackage_call_result_24342 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), add_subpackage_24339, *[str_24340], **kwargs_24341)
    
    
    # Call to add_subpackage(...): (line 17)
    # Processing the call arguments (line 17)
    str_24345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 26), 'str', 'ma')
    # Processing the call keyword arguments (line 17)
    kwargs_24346 = {}
    # Getting the type of 'config' (line 17)
    config_24343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 17)
    add_subpackage_24344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 4), config_24343, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 17)
    add_subpackage_call_result_24347 = invoke(stypy.reporting.localization.Localization(__file__, 17, 4), add_subpackage_24344, *[str_24345], **kwargs_24346)
    
    
    # Call to add_subpackage(...): (line 18)
    # Processing the call arguments (line 18)
    str_24350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 26), 'str', 'matrixlib')
    # Processing the call keyword arguments (line 18)
    kwargs_24351 = {}
    # Getting the type of 'config' (line 18)
    config_24348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 18)
    add_subpackage_24349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 4), config_24348, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 18)
    add_subpackage_call_result_24352 = invoke(stypy.reporting.localization.Localization(__file__, 18, 4), add_subpackage_24349, *[str_24350], **kwargs_24351)
    
    
    # Call to add_subpackage(...): (line 19)
    # Processing the call arguments (line 19)
    str_24355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 26), 'str', 'polynomial')
    # Processing the call keyword arguments (line 19)
    kwargs_24356 = {}
    # Getting the type of 'config' (line 19)
    config_24353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 19)
    add_subpackage_24354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 4), config_24353, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 19)
    add_subpackage_call_result_24357 = invoke(stypy.reporting.localization.Localization(__file__, 19, 4), add_subpackage_24354, *[str_24355], **kwargs_24356)
    
    
    # Call to add_subpackage(...): (line 20)
    # Processing the call arguments (line 20)
    str_24360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 26), 'str', 'random')
    # Processing the call keyword arguments (line 20)
    kwargs_24361 = {}
    # Getting the type of 'config' (line 20)
    config_24358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 20)
    add_subpackage_24359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 4), config_24358, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 20)
    add_subpackage_call_result_24362 = invoke(stypy.reporting.localization.Localization(__file__, 20, 4), add_subpackage_24359, *[str_24360], **kwargs_24361)
    
    
    # Call to add_subpackage(...): (line 21)
    # Processing the call arguments (line 21)
    str_24365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 26), 'str', 'testing')
    # Processing the call keyword arguments (line 21)
    kwargs_24366 = {}
    # Getting the type of 'config' (line 21)
    config_24363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 21)
    add_subpackage_24364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 4), config_24363, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 21)
    add_subpackage_call_result_24367 = invoke(stypy.reporting.localization.Localization(__file__, 21, 4), add_subpackage_24364, *[str_24365], **kwargs_24366)
    
    
    # Call to add_data_dir(...): (line 22)
    # Processing the call arguments (line 22)
    str_24370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 24), 'str', 'doc')
    # Processing the call keyword arguments (line 22)
    kwargs_24371 = {}
    # Getting the type of 'config' (line 22)
    config_24368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'config', False)
    # Obtaining the member 'add_data_dir' of a type (line 22)
    add_data_dir_24369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 4), config_24368, 'add_data_dir')
    # Calling add_data_dir(args, kwargs) (line 22)
    add_data_dir_call_result_24372 = invoke(stypy.reporting.localization.Localization(__file__, 22, 4), add_data_dir_24369, *[str_24370], **kwargs_24371)
    
    
    # Call to add_data_dir(...): (line 23)
    # Processing the call arguments (line 23)
    str_24375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 24), 'str', 'tests')
    # Processing the call keyword arguments (line 23)
    kwargs_24376 = {}
    # Getting the type of 'config' (line 23)
    config_24373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'config', False)
    # Obtaining the member 'add_data_dir' of a type (line 23)
    add_data_dir_24374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 4), config_24373, 'add_data_dir')
    # Calling add_data_dir(args, kwargs) (line 23)
    add_data_dir_call_result_24377 = invoke(stypy.reporting.localization.Localization(__file__, 23, 4), add_data_dir_24374, *[str_24375], **kwargs_24376)
    
    
    # Call to make_config_py(...): (line 24)
    # Processing the call keyword arguments (line 24)
    kwargs_24380 = {}
    # Getting the type of 'config' (line 24)
    config_24378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'config', False)
    # Obtaining the member 'make_config_py' of a type (line 24)
    make_config_py_24379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 4), config_24378, 'make_config_py')
    # Calling make_config_py(args, kwargs) (line 24)
    make_config_py_call_result_24381 = invoke(stypy.reporting.localization.Localization(__file__, 24, 4), make_config_py_24379, *[], **kwargs_24380)
    
    # Getting the type of 'config' (line 25)
    config_24382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type', config_24382)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 5)
    stypy_return_type_24383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24383)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_24383

# Assigning a type to the variable 'configuration' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    
    # Call to print(...): (line 28)
    # Processing the call arguments (line 28)
    str_24385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 10), 'str', 'This is the wrong setup.py file to run')
    # Processing the call keyword arguments (line 28)
    kwargs_24386 = {}
    # Getting the type of 'print' (line 28)
    print_24384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'print', False)
    # Calling print(args, kwargs) (line 28)
    print_call_result_24387 = invoke(stypy.reporting.localization.Localization(__file__, 28, 4), print_24384, *[str_24385], **kwargs_24386)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
