
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function
2: 
3: 
4: def configuration(parent_package='',top_path=None):
5:     from numpy.distutils.misc_util import Configuration
6:     config = Configuration('fft', parent_package, top_path)
7: 
8:     config.add_data_dir('tests')
9: 
10:     # Configure fftpack_lite
11:     config.add_extension('fftpack_lite',
12:                          sources=['fftpack_litemodule.c', 'fftpack.c']
13:                          )
14: 
15:     return config
16: 
17: if __name__ == '__main__':
18:     from numpy.distutils.core import setup
19:     setup(configuration=configuration)
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
    str_101084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 33), 'str', '')
    # Getting the type of 'None' (line 4)
    None_101085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 45), 'None')
    defaults = [str_101084, None_101085]
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
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/fft/')
    import_101086 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util')

    if (type(import_101086) is not StypyTypeError):

        if (import_101086 != 'pyd_module'):
            __import__(import_101086)
            sys_modules_101087 = sys.modules[import_101086]
            import_from_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util', sys_modules_101087.module_type_store, module_type_store, ['Configuration'])
            nest_module(stypy.reporting.localization.Localization(__file__, 5, 4), __file__, sys_modules_101087, sys_modules_101087.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration

            import_from_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 5)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util', import_101086)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/fft/')
    
    
    # Assigning a Call to a Name (line 6):
    
    # Call to Configuration(...): (line 6)
    # Processing the call arguments (line 6)
    str_101089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 27), 'str', 'fft')
    # Getting the type of 'parent_package' (line 6)
    parent_package_101090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 34), 'parent_package', False)
    # Getting the type of 'top_path' (line 6)
    top_path_101091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 50), 'top_path', False)
    # Processing the call keyword arguments (line 6)
    kwargs_101092 = {}
    # Getting the type of 'Configuration' (line 6)
    Configuration_101088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 6)
    Configuration_call_result_101093 = invoke(stypy.reporting.localization.Localization(__file__, 6, 13), Configuration_101088, *[str_101089, parent_package_101090, top_path_101091], **kwargs_101092)
    
    # Assigning a type to the variable 'config' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'config', Configuration_call_result_101093)
    
    # Call to add_data_dir(...): (line 8)
    # Processing the call arguments (line 8)
    str_101096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 24), 'str', 'tests')
    # Processing the call keyword arguments (line 8)
    kwargs_101097 = {}
    # Getting the type of 'config' (line 8)
    config_101094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'config', False)
    # Obtaining the member 'add_data_dir' of a type (line 8)
    add_data_dir_101095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 4), config_101094, 'add_data_dir')
    # Calling add_data_dir(args, kwargs) (line 8)
    add_data_dir_call_result_101098 = invoke(stypy.reporting.localization.Localization(__file__, 8, 4), add_data_dir_101095, *[str_101096], **kwargs_101097)
    
    
    # Call to add_extension(...): (line 11)
    # Processing the call arguments (line 11)
    str_101101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 25), 'str', 'fftpack_lite')
    # Processing the call keyword arguments (line 11)
    
    # Obtaining an instance of the builtin type 'list' (line 12)
    list_101102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 12)
    # Adding element type (line 12)
    str_101103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 34), 'str', 'fftpack_litemodule.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 33), list_101102, str_101103)
    # Adding element type (line 12)
    str_101104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 58), 'str', 'fftpack.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 33), list_101102, str_101104)
    
    keyword_101105 = list_101102
    kwargs_101106 = {'sources': keyword_101105}
    # Getting the type of 'config' (line 11)
    config_101099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 11)
    add_extension_101100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 4), config_101099, 'add_extension')
    # Calling add_extension(args, kwargs) (line 11)
    add_extension_call_result_101107 = invoke(stypy.reporting.localization.Localization(__file__, 11, 4), add_extension_101100, *[str_101101], **kwargs_101106)
    
    # Getting the type of 'config' (line 15)
    config_101108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'stypy_return_type', config_101108)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 4)
    stypy_return_type_101109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_101109)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_101109

# Assigning a type to the variable 'configuration' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 18)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/fft/')
    import_101110 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 4), 'numpy.distutils.core')

    if (type(import_101110) is not StypyTypeError):

        if (import_101110 != 'pyd_module'):
            __import__(import_101110)
            sys_modules_101111 = sys.modules[import_101110]
            import_from_module(stypy.reporting.localization.Localization(__file__, 18, 4), 'numpy.distutils.core', sys_modules_101111.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 18, 4), __file__, sys_modules_101111, sys_modules_101111.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 18, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'numpy.distutils.core', import_101110)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/fft/')
    
    
    # Call to setup(...): (line 19)
    # Processing the call keyword arguments (line 19)
    # Getting the type of 'configuration' (line 19)
    configuration_101113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 24), 'configuration', False)
    keyword_101114 = configuration_101113
    kwargs_101115 = {'configuration': keyword_101114}
    # Getting the type of 'setup' (line 19)
    setup_101112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 19)
    setup_call_result_101116 = invoke(stypy.reporting.localization.Localization(__file__, 19, 4), setup_101112, *[], **kwargs_101115)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
