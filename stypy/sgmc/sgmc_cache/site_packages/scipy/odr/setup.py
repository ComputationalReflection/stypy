
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: from os.path import join
4: 
5: 
6: def configuration(parent_package='', top_path=None):
7:     import warnings
8:     from numpy.distutils.misc_util import Configuration
9:     from numpy.distutils.system_info import get_info, BlasNotFoundError
10:     config = Configuration('odr', parent_package, top_path)
11: 
12:     libodr_files = ['d_odr.f',
13:                     'd_mprec.f',
14:                     'dlunoc.f']
15: 
16:     blas_info = get_info('blas_opt')
17:     if blas_info:
18:         libodr_files.append('d_lpk.f')
19:     else:
20:         warnings.warn(BlasNotFoundError.__doc__)
21:         libodr_files.append('d_lpkbls.f')
22: 
23:     odrpack_src = [join('odrpack', x) for x in libodr_files]
24:     config.add_library('odrpack', sources=odrpack_src)
25: 
26:     sources = ['__odrpack.c']
27:     libraries = ['odrpack'] + blas_info.pop('libraries', [])
28:     include_dirs = ['.'] + blas_info.pop('include_dirs', [])
29:     config.add_extension('__odrpack',
30:         sources=sources,
31:         libraries=libraries,
32:         include_dirs=include_dirs,
33:         depends=(['odrpack.h'] + odrpack_src),
34:         **blas_info
35:     )
36: 
37:     config.add_data_dir('tests')
38:     return config
39: 
40: if __name__ == '__main__':
41:     from numpy.distutils.core import setup
42:     setup(**configuration(top_path='').todict())
43: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from os.path import join' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/odr/')
import_165640 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path')

if (type(import_165640) is not StypyTypeError):

    if (import_165640 != 'pyd_module'):
        __import__(import_165640)
        sys_modules_165641 = sys.modules[import_165640]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path', sys_modules_165641.module_type_store, module_type_store, ['join'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_165641, sys_modules_165641.module_type_store, module_type_store)
    else:
        from os.path import join

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path', None, module_type_store, ['join'], [join])

else:
    # Assigning a type to the variable 'os.path' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path', import_165640)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/odr/')


@norecursion
def configuration(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_165642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 33), 'str', '')
    # Getting the type of 'None' (line 6)
    None_165643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 46), 'None')
    defaults = [str_165642, None_165643]
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
    
    # 'import warnings' statement (line 7)
    import warnings

    import_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'warnings', warnings, module_type_store)
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 4))
    
    # 'from numpy.distutils.misc_util import Configuration' statement (line 8)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/odr/')
    import_165644 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'numpy.distutils.misc_util')

    if (type(import_165644) is not StypyTypeError):

        if (import_165644 != 'pyd_module'):
            __import__(import_165644)
            sys_modules_165645 = sys.modules[import_165644]
            import_from_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'numpy.distutils.misc_util', sys_modules_165645.module_type_store, module_type_store, ['Configuration'])
            nest_module(stypy.reporting.localization.Localization(__file__, 8, 4), __file__, sys_modules_165645, sys_modules_165645.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration

            import_from_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'numpy.distutils.misc_util', import_165644)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/odr/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 4))
    
    # 'from numpy.distutils.system_info import get_info, BlasNotFoundError' statement (line 9)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/odr/')
    import_165646 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'numpy.distutils.system_info')

    if (type(import_165646) is not StypyTypeError):

        if (import_165646 != 'pyd_module'):
            __import__(import_165646)
            sys_modules_165647 = sys.modules[import_165646]
            import_from_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'numpy.distutils.system_info', sys_modules_165647.module_type_store, module_type_store, ['get_info', 'BlasNotFoundError'])
            nest_module(stypy.reporting.localization.Localization(__file__, 9, 4), __file__, sys_modules_165647, sys_modules_165647.module_type_store, module_type_store)
        else:
            from numpy.distutils.system_info import get_info, BlasNotFoundError

            import_from_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'numpy.distutils.system_info', None, module_type_store, ['get_info', 'BlasNotFoundError'], [get_info, BlasNotFoundError])

    else:
        # Assigning a type to the variable 'numpy.distutils.system_info' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'numpy.distutils.system_info', import_165646)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/odr/')
    
    
    # Assigning a Call to a Name (line 10):
    
    # Call to Configuration(...): (line 10)
    # Processing the call arguments (line 10)
    str_165649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 27), 'str', 'odr')
    # Getting the type of 'parent_package' (line 10)
    parent_package_165650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 34), 'parent_package', False)
    # Getting the type of 'top_path' (line 10)
    top_path_165651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 50), 'top_path', False)
    # Processing the call keyword arguments (line 10)
    kwargs_165652 = {}
    # Getting the type of 'Configuration' (line 10)
    Configuration_165648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 10)
    Configuration_call_result_165653 = invoke(stypy.reporting.localization.Localization(__file__, 10, 13), Configuration_165648, *[str_165649, parent_package_165650, top_path_165651], **kwargs_165652)
    
    # Assigning a type to the variable 'config' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'config', Configuration_call_result_165653)
    
    # Assigning a List to a Name (line 12):
    
    # Obtaining an instance of the builtin type 'list' (line 12)
    list_165654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 12)
    # Adding element type (line 12)
    str_165655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 20), 'str', 'd_odr.f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 19), list_165654, str_165655)
    # Adding element type (line 12)
    str_165656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 20), 'str', 'd_mprec.f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 19), list_165654, str_165656)
    # Adding element type (line 12)
    str_165657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 20), 'str', 'dlunoc.f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 19), list_165654, str_165657)
    
    # Assigning a type to the variable 'libodr_files' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'libodr_files', list_165654)
    
    # Assigning a Call to a Name (line 16):
    
    # Call to get_info(...): (line 16)
    # Processing the call arguments (line 16)
    str_165659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 25), 'str', 'blas_opt')
    # Processing the call keyword arguments (line 16)
    kwargs_165660 = {}
    # Getting the type of 'get_info' (line 16)
    get_info_165658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 16), 'get_info', False)
    # Calling get_info(args, kwargs) (line 16)
    get_info_call_result_165661 = invoke(stypy.reporting.localization.Localization(__file__, 16, 16), get_info_165658, *[str_165659], **kwargs_165660)
    
    # Assigning a type to the variable 'blas_info' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'blas_info', get_info_call_result_165661)
    
    # Getting the type of 'blas_info' (line 17)
    blas_info_165662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 7), 'blas_info')
    # Testing the type of an if condition (line 17)
    if_condition_165663 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 17, 4), blas_info_165662)
    # Assigning a type to the variable 'if_condition_165663' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'if_condition_165663', if_condition_165663)
    # SSA begins for if statement (line 17)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 18)
    # Processing the call arguments (line 18)
    str_165666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 28), 'str', 'd_lpk.f')
    # Processing the call keyword arguments (line 18)
    kwargs_165667 = {}
    # Getting the type of 'libodr_files' (line 18)
    libodr_files_165664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'libodr_files', False)
    # Obtaining the member 'append' of a type (line 18)
    append_165665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 8), libodr_files_165664, 'append')
    # Calling append(args, kwargs) (line 18)
    append_call_result_165668 = invoke(stypy.reporting.localization.Localization(__file__, 18, 8), append_165665, *[str_165666], **kwargs_165667)
    
    # SSA branch for the else part of an if statement (line 17)
    module_type_store.open_ssa_branch('else')
    
    # Call to warn(...): (line 20)
    # Processing the call arguments (line 20)
    # Getting the type of 'BlasNotFoundError' (line 20)
    BlasNotFoundError_165671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 22), 'BlasNotFoundError', False)
    # Obtaining the member '__doc__' of a type (line 20)
    doc___165672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 22), BlasNotFoundError_165671, '__doc__')
    # Processing the call keyword arguments (line 20)
    kwargs_165673 = {}
    # Getting the type of 'warnings' (line 20)
    warnings_165669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 20)
    warn_165670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), warnings_165669, 'warn')
    # Calling warn(args, kwargs) (line 20)
    warn_call_result_165674 = invoke(stypy.reporting.localization.Localization(__file__, 20, 8), warn_165670, *[doc___165672], **kwargs_165673)
    
    
    # Call to append(...): (line 21)
    # Processing the call arguments (line 21)
    str_165677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 28), 'str', 'd_lpkbls.f')
    # Processing the call keyword arguments (line 21)
    kwargs_165678 = {}
    # Getting the type of 'libodr_files' (line 21)
    libodr_files_165675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'libodr_files', False)
    # Obtaining the member 'append' of a type (line 21)
    append_165676 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 8), libodr_files_165675, 'append')
    # Calling append(args, kwargs) (line 21)
    append_call_result_165679 = invoke(stypy.reporting.localization.Localization(__file__, 21, 8), append_165676, *[str_165677], **kwargs_165678)
    
    # SSA join for if statement (line 17)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a ListComp to a Name (line 23):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'libodr_files' (line 23)
    libodr_files_165685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 47), 'libodr_files')
    comprehension_165686 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 19), libodr_files_165685)
    # Assigning a type to the variable 'x' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 19), 'x', comprehension_165686)
    
    # Call to join(...): (line 23)
    # Processing the call arguments (line 23)
    str_165681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 24), 'str', 'odrpack')
    # Getting the type of 'x' (line 23)
    x_165682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 35), 'x', False)
    # Processing the call keyword arguments (line 23)
    kwargs_165683 = {}
    # Getting the type of 'join' (line 23)
    join_165680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 19), 'join', False)
    # Calling join(args, kwargs) (line 23)
    join_call_result_165684 = invoke(stypy.reporting.localization.Localization(__file__, 23, 19), join_165680, *[str_165681, x_165682], **kwargs_165683)
    
    list_165687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 19), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 19), list_165687, join_call_result_165684)
    # Assigning a type to the variable 'odrpack_src' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'odrpack_src', list_165687)
    
    # Call to add_library(...): (line 24)
    # Processing the call arguments (line 24)
    str_165690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 23), 'str', 'odrpack')
    # Processing the call keyword arguments (line 24)
    # Getting the type of 'odrpack_src' (line 24)
    odrpack_src_165691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 42), 'odrpack_src', False)
    keyword_165692 = odrpack_src_165691
    kwargs_165693 = {'sources': keyword_165692}
    # Getting the type of 'config' (line 24)
    config_165688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'config', False)
    # Obtaining the member 'add_library' of a type (line 24)
    add_library_165689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 4), config_165688, 'add_library')
    # Calling add_library(args, kwargs) (line 24)
    add_library_call_result_165694 = invoke(stypy.reporting.localization.Localization(__file__, 24, 4), add_library_165689, *[str_165690], **kwargs_165693)
    
    
    # Assigning a List to a Name (line 26):
    
    # Obtaining an instance of the builtin type 'list' (line 26)
    list_165695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 26)
    # Adding element type (line 26)
    str_165696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 15), 'str', '__odrpack.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 14), list_165695, str_165696)
    
    # Assigning a type to the variable 'sources' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'sources', list_165695)
    
    # Assigning a BinOp to a Name (line 27):
    
    # Obtaining an instance of the builtin type 'list' (line 27)
    list_165697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 27)
    # Adding element type (line 27)
    str_165698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 17), 'str', 'odrpack')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 16), list_165697, str_165698)
    
    
    # Call to pop(...): (line 27)
    # Processing the call arguments (line 27)
    str_165701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 44), 'str', 'libraries')
    
    # Obtaining an instance of the builtin type 'list' (line 27)
    list_165702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 57), 'list')
    # Adding type elements to the builtin type 'list' instance (line 27)
    
    # Processing the call keyword arguments (line 27)
    kwargs_165703 = {}
    # Getting the type of 'blas_info' (line 27)
    blas_info_165699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 30), 'blas_info', False)
    # Obtaining the member 'pop' of a type (line 27)
    pop_165700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 30), blas_info_165699, 'pop')
    # Calling pop(args, kwargs) (line 27)
    pop_call_result_165704 = invoke(stypy.reporting.localization.Localization(__file__, 27, 30), pop_165700, *[str_165701, list_165702], **kwargs_165703)
    
    # Applying the binary operator '+' (line 27)
    result_add_165705 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 16), '+', list_165697, pop_call_result_165704)
    
    # Assigning a type to the variable 'libraries' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'libraries', result_add_165705)
    
    # Assigning a BinOp to a Name (line 28):
    
    # Obtaining an instance of the builtin type 'list' (line 28)
    list_165706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 28)
    # Adding element type (line 28)
    str_165707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 20), 'str', '.')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 19), list_165706, str_165707)
    
    
    # Call to pop(...): (line 28)
    # Processing the call arguments (line 28)
    str_165710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 41), 'str', 'include_dirs')
    
    # Obtaining an instance of the builtin type 'list' (line 28)
    list_165711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 57), 'list')
    # Adding type elements to the builtin type 'list' instance (line 28)
    
    # Processing the call keyword arguments (line 28)
    kwargs_165712 = {}
    # Getting the type of 'blas_info' (line 28)
    blas_info_165708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 27), 'blas_info', False)
    # Obtaining the member 'pop' of a type (line 28)
    pop_165709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 27), blas_info_165708, 'pop')
    # Calling pop(args, kwargs) (line 28)
    pop_call_result_165713 = invoke(stypy.reporting.localization.Localization(__file__, 28, 27), pop_165709, *[str_165710, list_165711], **kwargs_165712)
    
    # Applying the binary operator '+' (line 28)
    result_add_165714 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 19), '+', list_165706, pop_call_result_165713)
    
    # Assigning a type to the variable 'include_dirs' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'include_dirs', result_add_165714)
    
    # Call to add_extension(...): (line 29)
    # Processing the call arguments (line 29)
    str_165717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 25), 'str', '__odrpack')
    # Processing the call keyword arguments (line 29)
    # Getting the type of 'sources' (line 30)
    sources_165718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 16), 'sources', False)
    keyword_165719 = sources_165718
    # Getting the type of 'libraries' (line 31)
    libraries_165720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 18), 'libraries', False)
    keyword_165721 = libraries_165720
    # Getting the type of 'include_dirs' (line 32)
    include_dirs_165722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 21), 'include_dirs', False)
    keyword_165723 = include_dirs_165722
    
    # Obtaining an instance of the builtin type 'list' (line 33)
    list_165724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 33)
    # Adding element type (line 33)
    str_165725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 18), 'str', 'odrpack.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 17), list_165724, str_165725)
    
    # Getting the type of 'odrpack_src' (line 33)
    odrpack_src_165726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 33), 'odrpack_src', False)
    # Applying the binary operator '+' (line 33)
    result_add_165727 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 17), '+', list_165724, odrpack_src_165726)
    
    keyword_165728 = result_add_165727
    # Getting the type of 'blas_info' (line 34)
    blas_info_165729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 10), 'blas_info', False)
    kwargs_165730 = {'libraries': keyword_165721, 'sources': keyword_165719, 'depends': keyword_165728, 'blas_info_165729': blas_info_165729, 'include_dirs': keyword_165723}
    # Getting the type of 'config' (line 29)
    config_165715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 29)
    add_extension_165716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 4), config_165715, 'add_extension')
    # Calling add_extension(args, kwargs) (line 29)
    add_extension_call_result_165731 = invoke(stypy.reporting.localization.Localization(__file__, 29, 4), add_extension_165716, *[str_165717], **kwargs_165730)
    
    
    # Call to add_data_dir(...): (line 37)
    # Processing the call arguments (line 37)
    str_165734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 24), 'str', 'tests')
    # Processing the call keyword arguments (line 37)
    kwargs_165735 = {}
    # Getting the type of 'config' (line 37)
    config_165732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'config', False)
    # Obtaining the member 'add_data_dir' of a type (line 37)
    add_data_dir_165733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 4), config_165732, 'add_data_dir')
    # Calling add_data_dir(args, kwargs) (line 37)
    add_data_dir_call_result_165736 = invoke(stypy.reporting.localization.Localization(__file__, 37, 4), add_data_dir_165733, *[str_165734], **kwargs_165735)
    
    # Getting the type of 'config' (line 38)
    config_165737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type', config_165737)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 6)
    stypy_return_type_165738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_165738)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_165738

# Assigning a type to the variable 'configuration' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 41, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 41)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/odr/')
    import_165739 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 41, 4), 'numpy.distutils.core')

    if (type(import_165739) is not StypyTypeError):

        if (import_165739 != 'pyd_module'):
            __import__(import_165739)
            sys_modules_165740 = sys.modules[import_165739]
            import_from_module(stypy.reporting.localization.Localization(__file__, 41, 4), 'numpy.distutils.core', sys_modules_165740.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 41, 4), __file__, sys_modules_165740, sys_modules_165740.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 41, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'numpy.distutils.core', import_165739)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/odr/')
    
    
    # Call to setup(...): (line 42)
    # Processing the call keyword arguments (line 42)
    
    # Call to todict(...): (line 42)
    # Processing the call keyword arguments (line 42)
    kwargs_165748 = {}
    
    # Call to configuration(...): (line 42)
    # Processing the call keyword arguments (line 42)
    str_165743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 35), 'str', '')
    keyword_165744 = str_165743
    kwargs_165745 = {'top_path': keyword_165744}
    # Getting the type of 'configuration' (line 42)
    configuration_165742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'configuration', False)
    # Calling configuration(args, kwargs) (line 42)
    configuration_call_result_165746 = invoke(stypy.reporting.localization.Localization(__file__, 42, 12), configuration_165742, *[], **kwargs_165745)
    
    # Obtaining the member 'todict' of a type (line 42)
    todict_165747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 12), configuration_call_result_165746, 'todict')
    # Calling todict(args, kwargs) (line 42)
    todict_call_result_165749 = invoke(stypy.reporting.localization.Localization(__file__, 42, 12), todict_165747, *[], **kwargs_165748)
    
    kwargs_165750 = {'todict_call_result_165749': todict_call_result_165749}
    # Getting the type of 'setup' (line 42)
    setup_165741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 42)
    setup_call_result_165751 = invoke(stypy.reporting.localization.Localization(__file__, 42, 4), setup_165741, *[], **kwargs_165750)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
