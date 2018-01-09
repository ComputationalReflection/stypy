
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: 
4: def configuration(parent_package='', top_path=None):
5:     from numpy.distutils.misc_util import Configuration
6:     config = Configuration('_lsq', parent_package, top_path)
7:     config.add_extension('givens_elimination',
8:                          sources=['givens_elimination.c'])
9:     return config
10: 
11: 
12: if __name__ == '__main__':
13:     from numpy.distutils.core import setup
14:     setup(**configuration(top_path='').todict())
15: 

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
    str_252552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 33), 'str', '')
    # Getting the type of 'None' (line 4)
    None_252553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 46), 'None')
    defaults = [str_252552, None_252553]
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
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
    import_252554 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util')

    if (type(import_252554) is not StypyTypeError):

        if (import_252554 != 'pyd_module'):
            __import__(import_252554)
            sys_modules_252555 = sys.modules[import_252554]
            import_from_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util', sys_modules_252555.module_type_store, module_type_store, ['Configuration'])
            nest_module(stypy.reporting.localization.Localization(__file__, 5, 4), __file__, sys_modules_252555, sys_modules_252555.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration

            import_from_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 5)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util', import_252554)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
    
    
    # Assigning a Call to a Name (line 6):
    
    # Call to Configuration(...): (line 6)
    # Processing the call arguments (line 6)
    str_252557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 27), 'str', '_lsq')
    # Getting the type of 'parent_package' (line 6)
    parent_package_252558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 35), 'parent_package', False)
    # Getting the type of 'top_path' (line 6)
    top_path_252559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 51), 'top_path', False)
    # Processing the call keyword arguments (line 6)
    kwargs_252560 = {}
    # Getting the type of 'Configuration' (line 6)
    Configuration_252556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 6)
    Configuration_call_result_252561 = invoke(stypy.reporting.localization.Localization(__file__, 6, 13), Configuration_252556, *[str_252557, parent_package_252558, top_path_252559], **kwargs_252560)
    
    # Assigning a type to the variable 'config' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'config', Configuration_call_result_252561)
    
    # Call to add_extension(...): (line 7)
    # Processing the call arguments (line 7)
    str_252564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 25), 'str', 'givens_elimination')
    # Processing the call keyword arguments (line 7)
    
    # Obtaining an instance of the builtin type 'list' (line 8)
    list_252565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 8)
    # Adding element type (line 8)
    str_252566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 34), 'str', 'givens_elimination.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 33), list_252565, str_252566)
    
    keyword_252567 = list_252565
    kwargs_252568 = {'sources': keyword_252567}
    # Getting the type of 'config' (line 7)
    config_252562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 7)
    add_extension_252563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 4), config_252562, 'add_extension')
    # Calling add_extension(args, kwargs) (line 7)
    add_extension_call_result_252569 = invoke(stypy.reporting.localization.Localization(__file__, 7, 4), add_extension_252563, *[str_252564], **kwargs_252568)
    
    # Getting the type of 'config' (line 9)
    config_252570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'stypy_return_type', config_252570)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 4)
    stypy_return_type_252571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_252571)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_252571

# Assigning a type to the variable 'configuration' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 13)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
    import_252572 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 4), 'numpy.distutils.core')

    if (type(import_252572) is not StypyTypeError):

        if (import_252572 != 'pyd_module'):
            __import__(import_252572)
            sys_modules_252573 = sys.modules[import_252572]
            import_from_module(stypy.reporting.localization.Localization(__file__, 13, 4), 'numpy.distutils.core', sys_modules_252573.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 13, 4), __file__, sys_modules_252573, sys_modules_252573.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 13, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'numpy.distutils.core', import_252572)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_lsq/')
    
    
    # Call to setup(...): (line 14)
    # Processing the call keyword arguments (line 14)
    
    # Call to todict(...): (line 14)
    # Processing the call keyword arguments (line 14)
    kwargs_252581 = {}
    
    # Call to configuration(...): (line 14)
    # Processing the call keyword arguments (line 14)
    str_252576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 35), 'str', '')
    keyword_252577 = str_252576
    kwargs_252578 = {'top_path': keyword_252577}
    # Getting the type of 'configuration' (line 14)
    configuration_252575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 12), 'configuration', False)
    # Calling configuration(args, kwargs) (line 14)
    configuration_call_result_252579 = invoke(stypy.reporting.localization.Localization(__file__, 14, 12), configuration_252575, *[], **kwargs_252578)
    
    # Obtaining the member 'todict' of a type (line 14)
    todict_252580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 12), configuration_call_result_252579, 'todict')
    # Calling todict(args, kwargs) (line 14)
    todict_call_result_252582 = invoke(stypy.reporting.localization.Localization(__file__, 14, 12), todict_252580, *[], **kwargs_252581)
    
    kwargs_252583 = {'todict_call_result_252582': todict_call_result_252582}
    # Getting the type of 'setup' (line 14)
    setup_252574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 14)
    setup_call_result_252584 = invoke(stypy.reporting.localization.Localization(__file__, 14, 4), setup_252574, *[], **kwargs_252583)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
