
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #!/usr/bin/env python
2: from __future__ import division, print_function
3: 
4: import os
5: 
6: def configuration(parent_package='', top_path=None):
7:     from numpy.distutils.misc_util import Configuration
8:     config = Configuration('matrixlib', parent_package, top_path)
9:     config.add_data_dir('tests')
10:     return config
11: 
12: if __name__ == "__main__":
13:     from numpy.distutils.core import setup
14:     config = configuration(top_path='').todict()
15:     setup(**config)
16: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import os' statement (line 4)
import os

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'os', os, module_type_store)


@norecursion
def configuration(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_161827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 33), 'str', '')
    # Getting the type of 'None' (line 6)
    None_161828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 46), 'None')
    defaults = [str_161827, None_161828]
    # Create a new context for function 'configuration'
    module_type_store = module_type_store.open_function_context('configuration', 6, 0, False)
    
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

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 4))
    
    # 'from numpy.distutils.misc_util import Configuration' statement (line 7)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/matrixlib/')
    import_161829 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy.distutils.misc_util')

    if (type(import_161829) is not StypyTypeError):

        if (import_161829 != 'pyd_module'):
            __import__(import_161829)
            sys_modules_161830 = sys.modules[import_161829]
            import_from_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy.distutils.misc_util', sys_modules_161830.module_type_store, module_type_store, ['Configuration'])
            nest_module(stypy.reporting.localization.Localization(__file__, 7, 4), __file__, sys_modules_161830, sys_modules_161830.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration

            import_from_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy.distutils.misc_util', import_161829)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/matrixlib/')
    
    
    # Assigning a Call to a Name (line 8):
    
    # Call to Configuration(...): (line 8)
    # Processing the call arguments (line 8)
    str_161832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 27), 'str', 'matrixlib')
    # Getting the type of 'parent_package' (line 8)
    parent_package_161833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 40), 'parent_package', False)
    # Getting the type of 'top_path' (line 8)
    top_path_161834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 56), 'top_path', False)
    # Processing the call keyword arguments (line 8)
    kwargs_161835 = {}
    # Getting the type of 'Configuration' (line 8)
    Configuration_161831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 8)
    Configuration_call_result_161836 = invoke(stypy.reporting.localization.Localization(__file__, 8, 13), Configuration_161831, *[str_161832, parent_package_161833, top_path_161834], **kwargs_161835)
    
    # Assigning a type to the variable 'config' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'config', Configuration_call_result_161836)
    
    # Call to add_data_dir(...): (line 9)
    # Processing the call arguments (line 9)
    str_161839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 24), 'str', 'tests')
    # Processing the call keyword arguments (line 9)
    kwargs_161840 = {}
    # Getting the type of 'config' (line 9)
    config_161837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'config', False)
    # Obtaining the member 'add_data_dir' of a type (line 9)
    add_data_dir_161838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 4), config_161837, 'add_data_dir')
    # Calling add_data_dir(args, kwargs) (line 9)
    add_data_dir_call_result_161841 = invoke(stypy.reporting.localization.Localization(__file__, 9, 4), add_data_dir_161838, *[str_161839], **kwargs_161840)
    
    # Getting the type of 'config' (line 10)
    config_161842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'stypy_return_type', config_161842)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 6)
    stypy_return_type_161843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_161843)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_161843

# Assigning a type to the variable 'configuration' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 13)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/matrixlib/')
    import_161844 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 4), 'numpy.distutils.core')

    if (type(import_161844) is not StypyTypeError):

        if (import_161844 != 'pyd_module'):
            __import__(import_161844)
            sys_modules_161845 = sys.modules[import_161844]
            import_from_module(stypy.reporting.localization.Localization(__file__, 13, 4), 'numpy.distutils.core', sys_modules_161845.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 13, 4), __file__, sys_modules_161845, sys_modules_161845.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 13, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'numpy.distutils.core', import_161844)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/matrixlib/')
    
    
    # Assigning a Call to a Name (line 14):
    
    # Call to todict(...): (line 14)
    # Processing the call keyword arguments (line 14)
    kwargs_161852 = {}
    
    # Call to configuration(...): (line 14)
    # Processing the call keyword arguments (line 14)
    str_161847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 36), 'str', '')
    keyword_161848 = str_161847
    kwargs_161849 = {'top_path': keyword_161848}
    # Getting the type of 'configuration' (line 14)
    configuration_161846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 13), 'configuration', False)
    # Calling configuration(args, kwargs) (line 14)
    configuration_call_result_161850 = invoke(stypy.reporting.localization.Localization(__file__, 14, 13), configuration_161846, *[], **kwargs_161849)
    
    # Obtaining the member 'todict' of a type (line 14)
    todict_161851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 13), configuration_call_result_161850, 'todict')
    # Calling todict(args, kwargs) (line 14)
    todict_call_result_161853 = invoke(stypy.reporting.localization.Localization(__file__, 14, 13), todict_161851, *[], **kwargs_161852)
    
    # Assigning a type to the variable 'config' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'config', todict_call_result_161853)
    
    # Call to setup(...): (line 15)
    # Processing the call keyword arguments (line 15)
    # Getting the type of 'config' (line 15)
    config_161855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), 'config', False)
    kwargs_161856 = {'config_161855': config_161855}
    # Getting the type of 'setup' (line 15)
    setup_161854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 15)
    setup_call_result_161857 = invoke(stypy.reporting.localization.Localization(__file__, 15, 4), setup_161854, *[], **kwargs_161856)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
