
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: def configuration(parent_package='', top_path=None):
4:     from numpy import get_include
5:     from numpy.distutils.system_info import get_info, NotFoundError
6:     from numpy.distutils.misc_util import Configuration
7: 
8:     lapack_opt = get_info('lapack_opt')
9: 
10:     if not lapack_opt:
11:         raise NotFoundError('no lapack/blas resources found')
12: 
13:     config = Configuration('_trlib', parent_package, top_path)
14:     config.add_extension('_trlib',
15:                          sources=['_trlib.c', 'trlib_krylov.c',
16:                                   'trlib_eigen_inverse.c', 'trlib_leftmost.c',
17:                                   'trlib_quadratic_zero.c', 'trlib_tri_factor.c'],
18:                          include_dirs=[get_include(), 'trlib'],
19:                          extra_info=lapack_opt,
20:                          )
21:     return config
22: 
23: 
24: if __name__ == '__main__':
25:     from numpy.distutils.core import setup
26:     setup(**configuration(top_path='').todict())
27: 

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
    str_255718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 33), 'str', '')
    # Getting the type of 'None' (line 3)
    None_255719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 46), 'None')
    defaults = [str_255718, None_255719]
    # Create a new context for function 'configuration'
    module_type_store = module_type_store.open_function_context('configuration', 3, 0, False)
    
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

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 4))
    
    # 'from numpy import get_include' statement (line 4)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_trlib/')
    import_255720 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 4), 'numpy')

    if (type(import_255720) is not StypyTypeError):

        if (import_255720 != 'pyd_module'):
            __import__(import_255720)
            sys_modules_255721 = sys.modules[import_255720]
            import_from_module(stypy.reporting.localization.Localization(__file__, 4, 4), 'numpy', sys_modules_255721.module_type_store, module_type_store, ['get_include'])
            nest_module(stypy.reporting.localization.Localization(__file__, 4, 4), __file__, sys_modules_255721, sys_modules_255721.module_type_store, module_type_store)
        else:
            from numpy import get_include

            import_from_module(stypy.reporting.localization.Localization(__file__, 4, 4), 'numpy', None, module_type_store, ['get_include'], [get_include])

    else:
        # Assigning a type to the variable 'numpy' (line 4)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'numpy', import_255720)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_trlib/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 4))
    
    # 'from numpy.distutils.system_info import get_info, NotFoundError' statement (line 5)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_trlib/')
    import_255722 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.system_info')

    if (type(import_255722) is not StypyTypeError):

        if (import_255722 != 'pyd_module'):
            __import__(import_255722)
            sys_modules_255723 = sys.modules[import_255722]
            import_from_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.system_info', sys_modules_255723.module_type_store, module_type_store, ['get_info', 'NotFoundError'])
            nest_module(stypy.reporting.localization.Localization(__file__, 5, 4), __file__, sys_modules_255723, sys_modules_255723.module_type_store, module_type_store)
        else:
            from numpy.distutils.system_info import get_info, NotFoundError

            import_from_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.system_info', None, module_type_store, ['get_info', 'NotFoundError'], [get_info, NotFoundError])

    else:
        # Assigning a type to the variable 'numpy.distutils.system_info' (line 5)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'numpy.distutils.system_info', import_255722)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_trlib/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 4))
    
    # 'from numpy.distutils.misc_util import Configuration' statement (line 6)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_trlib/')
    import_255724 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 4), 'numpy.distutils.misc_util')

    if (type(import_255724) is not StypyTypeError):

        if (import_255724 != 'pyd_module'):
            __import__(import_255724)
            sys_modules_255725 = sys.modules[import_255724]
            import_from_module(stypy.reporting.localization.Localization(__file__, 6, 4), 'numpy.distutils.misc_util', sys_modules_255725.module_type_store, module_type_store, ['Configuration'])
            nest_module(stypy.reporting.localization.Localization(__file__, 6, 4), __file__, sys_modules_255725, sys_modules_255725.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration

            import_from_module(stypy.reporting.localization.Localization(__file__, 6, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 6)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'numpy.distutils.misc_util', import_255724)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_trlib/')
    
    
    # Assigning a Call to a Name (line 8):
    
    # Call to get_info(...): (line 8)
    # Processing the call arguments (line 8)
    str_255727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 26), 'str', 'lapack_opt')
    # Processing the call keyword arguments (line 8)
    kwargs_255728 = {}
    # Getting the type of 'get_info' (line 8)
    get_info_255726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 17), 'get_info', False)
    # Calling get_info(args, kwargs) (line 8)
    get_info_call_result_255729 = invoke(stypy.reporting.localization.Localization(__file__, 8, 17), get_info_255726, *[str_255727], **kwargs_255728)
    
    # Assigning a type to the variable 'lapack_opt' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'lapack_opt', get_info_call_result_255729)
    
    
    # Getting the type of 'lapack_opt' (line 10)
    lapack_opt_255730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 11), 'lapack_opt')
    # Applying the 'not' unary operator (line 10)
    result_not__255731 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 7), 'not', lapack_opt_255730)
    
    # Testing the type of an if condition (line 10)
    if_condition_255732 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 10, 4), result_not__255731)
    # Assigning a type to the variable 'if_condition_255732' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'if_condition_255732', if_condition_255732)
    # SSA begins for if statement (line 10)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to NotFoundError(...): (line 11)
    # Processing the call arguments (line 11)
    str_255734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 28), 'str', 'no lapack/blas resources found')
    # Processing the call keyword arguments (line 11)
    kwargs_255735 = {}
    # Getting the type of 'NotFoundError' (line 11)
    NotFoundError_255733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 14), 'NotFoundError', False)
    # Calling NotFoundError(args, kwargs) (line 11)
    NotFoundError_call_result_255736 = invoke(stypy.reporting.localization.Localization(__file__, 11, 14), NotFoundError_255733, *[str_255734], **kwargs_255735)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 11, 8), NotFoundError_call_result_255736, 'raise parameter', BaseException)
    # SSA join for if statement (line 10)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 13):
    
    # Call to Configuration(...): (line 13)
    # Processing the call arguments (line 13)
    str_255738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 27), 'str', '_trlib')
    # Getting the type of 'parent_package' (line 13)
    parent_package_255739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 37), 'parent_package', False)
    # Getting the type of 'top_path' (line 13)
    top_path_255740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 53), 'top_path', False)
    # Processing the call keyword arguments (line 13)
    kwargs_255741 = {}
    # Getting the type of 'Configuration' (line 13)
    Configuration_255737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 13)
    Configuration_call_result_255742 = invoke(stypy.reporting.localization.Localization(__file__, 13, 13), Configuration_255737, *[str_255738, parent_package_255739, top_path_255740], **kwargs_255741)
    
    # Assigning a type to the variable 'config' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'config', Configuration_call_result_255742)
    
    # Call to add_extension(...): (line 14)
    # Processing the call arguments (line 14)
    str_255745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 25), 'str', '_trlib')
    # Processing the call keyword arguments (line 14)
    
    # Obtaining an instance of the builtin type 'list' (line 15)
    list_255746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 15)
    # Adding element type (line 15)
    str_255747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 34), 'str', '_trlib.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 33), list_255746, str_255747)
    # Adding element type (line 15)
    str_255748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 46), 'str', 'trlib_krylov.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 33), list_255746, str_255748)
    # Adding element type (line 15)
    str_255749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 34), 'str', 'trlib_eigen_inverse.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 33), list_255746, str_255749)
    # Adding element type (line 15)
    str_255750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 59), 'str', 'trlib_leftmost.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 33), list_255746, str_255750)
    # Adding element type (line 15)
    str_255751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 34), 'str', 'trlib_quadratic_zero.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 33), list_255746, str_255751)
    # Adding element type (line 15)
    str_255752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 60), 'str', 'trlib_tri_factor.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 33), list_255746, str_255752)
    
    keyword_255753 = list_255746
    
    # Obtaining an instance of the builtin type 'list' (line 18)
    list_255754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 18)
    # Adding element type (line 18)
    
    # Call to get_include(...): (line 18)
    # Processing the call keyword arguments (line 18)
    kwargs_255756 = {}
    # Getting the type of 'get_include' (line 18)
    get_include_255755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 39), 'get_include', False)
    # Calling get_include(args, kwargs) (line 18)
    get_include_call_result_255757 = invoke(stypy.reporting.localization.Localization(__file__, 18, 39), get_include_255755, *[], **kwargs_255756)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 38), list_255754, get_include_call_result_255757)
    # Adding element type (line 18)
    str_255758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 54), 'str', 'trlib')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 38), list_255754, str_255758)
    
    keyword_255759 = list_255754
    # Getting the type of 'lapack_opt' (line 19)
    lapack_opt_255760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 36), 'lapack_opt', False)
    keyword_255761 = lapack_opt_255760
    kwargs_255762 = {'sources': keyword_255753, 'extra_info': keyword_255761, 'include_dirs': keyword_255759}
    # Getting the type of 'config' (line 14)
    config_255743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 14)
    add_extension_255744 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 4), config_255743, 'add_extension')
    # Calling add_extension(args, kwargs) (line 14)
    add_extension_call_result_255763 = invoke(stypy.reporting.localization.Localization(__file__, 14, 4), add_extension_255744, *[str_255745], **kwargs_255762)
    
    # Getting the type of 'config' (line 21)
    config_255764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'stypy_return_type', config_255764)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 3)
    stypy_return_type_255765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_255765)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_255765

# Assigning a type to the variable 'configuration' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 25)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/_trlib/')
    import_255766 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 25, 4), 'numpy.distutils.core')

    if (type(import_255766) is not StypyTypeError):

        if (import_255766 != 'pyd_module'):
            __import__(import_255766)
            sys_modules_255767 = sys.modules[import_255766]
            import_from_module(stypy.reporting.localization.Localization(__file__, 25, 4), 'numpy.distutils.core', sys_modules_255767.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 25, 4), __file__, sys_modules_255767, sys_modules_255767.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 25, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'numpy.distutils.core', import_255766)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/_trlib/')
    
    
    # Call to setup(...): (line 26)
    # Processing the call keyword arguments (line 26)
    
    # Call to todict(...): (line 26)
    # Processing the call keyword arguments (line 26)
    kwargs_255775 = {}
    
    # Call to configuration(...): (line 26)
    # Processing the call keyword arguments (line 26)
    str_255770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 35), 'str', '')
    keyword_255771 = str_255770
    kwargs_255772 = {'top_path': keyword_255771}
    # Getting the type of 'configuration' (line 26)
    configuration_255769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'configuration', False)
    # Calling configuration(args, kwargs) (line 26)
    configuration_call_result_255773 = invoke(stypy.reporting.localization.Localization(__file__, 26, 12), configuration_255769, *[], **kwargs_255772)
    
    # Obtaining the member 'todict' of a type (line 26)
    todict_255774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 12), configuration_call_result_255773, 'todict')
    # Calling todict(args, kwargs) (line 26)
    todict_call_result_255776 = invoke(stypy.reporting.localization.Localization(__file__, 26, 12), todict_255774, *[], **kwargs_255775)
    
    kwargs_255777 = {'todict_call_result_255776': todict_call_result_255776}
    # Getting the type of 'setup' (line 26)
    setup_255768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 26)
    setup_call_result_255778 = invoke(stypy.reporting.localization.Localization(__file__, 26, 4), setup_255768, *[], **kwargs_255777)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
