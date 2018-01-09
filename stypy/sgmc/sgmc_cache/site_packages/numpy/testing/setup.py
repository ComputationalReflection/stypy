
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #!/usr/bin/env python
2: from __future__ import division, print_function
3: 
4: 
5: def configuration(parent_package='',top_path=None):
6:     from numpy.distutils.misc_util import Configuration
7:     config = Configuration('testing', parent_package, top_path)
8: 
9:     config.add_data_dir('tests')
10:     return config
11: 
12: if __name__ == '__main__':
13:     from numpy.distutils.core import setup
14:     setup(maintainer="NumPy Developers",
15:           maintainer_email="numpy-dev@numpy.org",
16:           description="NumPy test module",
17:           url="http://www.numpy.org",
18:           license="NumPy License (BSD Style)",
19:           configuration=configuration,
20:           )
21: 

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
    str_182789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 33), 'str', '')
    # Getting the type of 'None' (line 5)
    None_182790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 45), 'None')
    defaults = [str_182789, None_182790]
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
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/testing/')
    import_182791 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 4), 'numpy.distutils.misc_util')

    if (type(import_182791) is not StypyTypeError):

        if (import_182791 != 'pyd_module'):
            __import__(import_182791)
            sys_modules_182792 = sys.modules[import_182791]
            import_from_module(stypy.reporting.localization.Localization(__file__, 6, 4), 'numpy.distutils.misc_util', sys_modules_182792.module_type_store, module_type_store, ['Configuration'])
            nest_module(stypy.reporting.localization.Localization(__file__, 6, 4), __file__, sys_modules_182792, sys_modules_182792.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration

            import_from_module(stypy.reporting.localization.Localization(__file__, 6, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 6)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'numpy.distutils.misc_util', import_182791)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/testing/')
    
    
    # Assigning a Call to a Name (line 7):
    
    # Call to Configuration(...): (line 7)
    # Processing the call arguments (line 7)
    str_182794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 27), 'str', 'testing')
    # Getting the type of 'parent_package' (line 7)
    parent_package_182795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 38), 'parent_package', False)
    # Getting the type of 'top_path' (line 7)
    top_path_182796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 54), 'top_path', False)
    # Processing the call keyword arguments (line 7)
    kwargs_182797 = {}
    # Getting the type of 'Configuration' (line 7)
    Configuration_182793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 7)
    Configuration_call_result_182798 = invoke(stypy.reporting.localization.Localization(__file__, 7, 13), Configuration_182793, *[str_182794, parent_package_182795, top_path_182796], **kwargs_182797)
    
    # Assigning a type to the variable 'config' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'config', Configuration_call_result_182798)
    
    # Call to add_data_dir(...): (line 9)
    # Processing the call arguments (line 9)
    str_182801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 24), 'str', 'tests')
    # Processing the call keyword arguments (line 9)
    kwargs_182802 = {}
    # Getting the type of 'config' (line 9)
    config_182799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'config', False)
    # Obtaining the member 'add_data_dir' of a type (line 9)
    add_data_dir_182800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 4), config_182799, 'add_data_dir')
    # Calling add_data_dir(args, kwargs) (line 9)
    add_data_dir_call_result_182803 = invoke(stypy.reporting.localization.Localization(__file__, 9, 4), add_data_dir_182800, *[str_182801], **kwargs_182802)
    
    # Getting the type of 'config' (line 10)
    config_182804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'stypy_return_type', config_182804)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 5)
    stypy_return_type_182805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_182805)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_182805

# Assigning a type to the variable 'configuration' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 13)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/testing/')
    import_182806 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 4), 'numpy.distutils.core')

    if (type(import_182806) is not StypyTypeError):

        if (import_182806 != 'pyd_module'):
            __import__(import_182806)
            sys_modules_182807 = sys.modules[import_182806]
            import_from_module(stypy.reporting.localization.Localization(__file__, 13, 4), 'numpy.distutils.core', sys_modules_182807.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 13, 4), __file__, sys_modules_182807, sys_modules_182807.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 13, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'numpy.distutils.core', import_182806)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/testing/')
    
    
    # Call to setup(...): (line 14)
    # Processing the call keyword arguments (line 14)
    str_182809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 21), 'str', 'NumPy Developers')
    keyword_182810 = str_182809
    str_182811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 27), 'str', 'numpy-dev@numpy.org')
    keyword_182812 = str_182811
    str_182813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 22), 'str', 'NumPy test module')
    keyword_182814 = str_182813
    str_182815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 14), 'str', 'http://www.numpy.org')
    keyword_182816 = str_182815
    str_182817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 18), 'str', 'NumPy License (BSD Style)')
    keyword_182818 = str_182817
    # Getting the type of 'configuration' (line 19)
    configuration_182819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 24), 'configuration', False)
    keyword_182820 = configuration_182819
    kwargs_182821 = {'maintainer': keyword_182810, 'description': keyword_182814, 'license': keyword_182818, 'url': keyword_182816, 'maintainer_email': keyword_182812, 'configuration': keyword_182820}
    # Getting the type of 'setup' (line 14)
    setup_182808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 14)
    setup_call_result_182822 = invoke(stypy.reporting.localization.Localization(__file__, 14, 4), setup_182808, *[], **kwargs_182821)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
