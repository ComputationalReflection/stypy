
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: 
4: def configuration(parent_name='special', top_path=None):
5:     from numpy.distutils.misc_util import Configuration
6:     config = Configuration('_precompute', parent_name, top_path)
7:     return config
8: 
9: 
10: if __name__ == '__main__':
11:     from numpy.distutils.core import setup
12:     setup(**configuration().todict())
13: 
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
    str_564685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 30), 'str', 'special')
    # Getting the type of 'None' (line 4)
    None_564686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 50), 'None')
    defaults = [str_564685, None_564686]
    # Create a new context for function 'configuration'
    module_type_store = module_type_store.open_function_context('configuration', 4, 0, False)
    
    # Passed parameters checking function
    configuration.stypy_localization = localization
    configuration.stypy_type_of_self = None
    configuration.stypy_type_store = module_type_store
    configuration.stypy_function_name = 'configuration'
    configuration.stypy_param_names_list = ['parent_name', 'top_path']
    configuration.stypy_varargs_param_name = None
    configuration.stypy_kwargs_param_name = None
    configuration.stypy_call_defaults = defaults
    configuration.stypy_call_varargs = varargs
    configuration.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'configuration', ['parent_name', 'top_path'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'configuration', localization, ['parent_name', 'top_path'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'configuration(...)' code ##################

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 4))
    
    # 'from numpy.distutils.misc_util import Configuration' statement (line 5)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/_precompute/')
    import_564687 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util')

    if (type(import_564687) is not StypyTypeError):

        if (import_564687 != 'pyd_module'):
            __import__(import_564687)
            sys_modules_564688 = sys.modules[import_564687]
            import_from_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util', sys_modules_564688.module_type_store, module_type_store, ['Configuration'])
            nest_module(stypy.reporting.localization.Localization(__file__, 5, 4), __file__, sys_modules_564688, sys_modules_564688.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration

            import_from_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 5)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.misc_util', import_564687)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/_precompute/')
    
    
    # Assigning a Call to a Name (line 6):
    
    # Call to Configuration(...): (line 6)
    # Processing the call arguments (line 6)
    str_564690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 27), 'str', '_precompute')
    # Getting the type of 'parent_name' (line 6)
    parent_name_564691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 42), 'parent_name', False)
    # Getting the type of 'top_path' (line 6)
    top_path_564692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 55), 'top_path', False)
    # Processing the call keyword arguments (line 6)
    kwargs_564693 = {}
    # Getting the type of 'Configuration' (line 6)
    Configuration_564689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 6)
    Configuration_call_result_564694 = invoke(stypy.reporting.localization.Localization(__file__, 6, 13), Configuration_564689, *[str_564690, parent_name_564691, top_path_564692], **kwargs_564693)
    
    # Assigning a type to the variable 'config' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'config', Configuration_call_result_564694)
    # Getting the type of 'config' (line 7)
    config_564695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'stypy_return_type', config_564695)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 4)
    stypy_return_type_564696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_564696)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_564696

# Assigning a type to the variable 'configuration' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 11)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/_precompute/')
    import_564697 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'numpy.distutils.core')

    if (type(import_564697) is not StypyTypeError):

        if (import_564697 != 'pyd_module'):
            __import__(import_564697)
            sys_modules_564698 = sys.modules[import_564697]
            import_from_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'numpy.distutils.core', sys_modules_564698.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 11, 4), __file__, sys_modules_564698, sys_modules_564698.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'numpy.distutils.core', import_564697)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/_precompute/')
    
    
    # Call to setup(...): (line 12)
    # Processing the call keyword arguments (line 12)
    
    # Call to todict(...): (line 12)
    # Processing the call keyword arguments (line 12)
    kwargs_564704 = {}
    
    # Call to configuration(...): (line 12)
    # Processing the call keyword arguments (line 12)
    kwargs_564701 = {}
    # Getting the type of 'configuration' (line 12)
    configuration_564700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'configuration', False)
    # Calling configuration(args, kwargs) (line 12)
    configuration_call_result_564702 = invoke(stypy.reporting.localization.Localization(__file__, 12, 12), configuration_564700, *[], **kwargs_564701)
    
    # Obtaining the member 'todict' of a type (line 12)
    todict_564703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 12), configuration_call_result_564702, 'todict')
    # Calling todict(args, kwargs) (line 12)
    todict_call_result_564705 = invoke(stypy.reporting.localization.Localization(__file__, 12, 12), todict_564703, *[], **kwargs_564704)
    
    kwargs_564706 = {'todict_call_result_564705': todict_call_result_564705}
    # Getting the type of 'setup' (line 12)
    setup_564699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 12)
    setup_call_result_564707 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), setup_564699, *[], **kwargs_564706)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
