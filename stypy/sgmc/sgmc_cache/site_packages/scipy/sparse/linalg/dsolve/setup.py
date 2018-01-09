
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: from os.path import join, dirname
4: import sys
5: import os
6: import glob
7: 
8: 
9: def configuration(parent_package='',top_path=None):
10:     from numpy.distutils.misc_util import Configuration
11:     from numpy.distutils.system_info import get_info
12:     from scipy._build_utils import get_sgemv_fix
13:     from scipy._build_utils import numpy_nodepr_api
14: 
15:     config = Configuration('dsolve',parent_package,top_path)
16:     config.add_data_dir('tests')
17: 
18:     lapack_opt = get_info('lapack_opt',notfound_action=2)
19:     if sys.platform == 'win32':
20:         superlu_defs = [('NO_TIMER',1)]
21:     else:
22:         superlu_defs = []
23:     superlu_defs.append(('USE_VENDOR_BLAS',1))
24: 
25:     superlu_src = join(dirname(__file__), 'SuperLU', 'SRC')
26: 
27:     sources = list(glob.glob(join(superlu_src, '*.c')))
28:     headers = list(glob.glob(join(superlu_src, '*.h')))
29: 
30:     config.add_library('superlu_src',
31:                        sources=sources,
32:                        macros=superlu_defs,
33:                        include_dirs=[superlu_src],
34:                        )
35: 
36:     # Extension
37:     ext_sources = ['_superlumodule.c',
38:                    '_superlu_utils.c',
39:                    '_superluobject.c']
40:     ext_sources += get_sgemv_fix(lapack_opt)
41: 
42:     config.add_extension('_superlu',
43:                          sources=ext_sources,
44:                          libraries=['superlu_src'],
45:                          depends=(sources + headers),
46:                          extra_info=lapack_opt,
47:                          **numpy_nodepr_api
48:                          )
49: 
50:     # Add license files
51:     config.add_data_files('SuperLU/License.txt')
52: 
53:     return config
54: 
55: if __name__ == '__main__':
56:     from numpy.distutils.core import setup
57:     setup(**configuration(top_path='').todict())
58: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from os.path import join, dirname' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/')
import_392705 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path')

if (type(import_392705) is not StypyTypeError):

    if (import_392705 != 'pyd_module'):
        __import__(import_392705)
        sys_modules_392706 = sys.modules[import_392705]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path', sys_modules_392706.module_type_store, module_type_store, ['join', 'dirname'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_392706, sys_modules_392706.module_type_store, module_type_store)
    else:
        from os.path import join, dirname

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path', None, module_type_store, ['join', 'dirname'], [join, dirname])

else:
    # Assigning a type to the variable 'os.path' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path', import_392705)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import sys' statement (line 4)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import os' statement (line 5)
import os

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import glob' statement (line 6)
import glob

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'glob', glob, module_type_store)


@norecursion
def configuration(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_392707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 33), 'str', '')
    # Getting the type of 'None' (line 9)
    None_392708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 45), 'None')
    defaults = [str_392707, None_392708]
    # Create a new context for function 'configuration'
    module_type_store = module_type_store.open_function_context('configuration', 9, 0, False)
    
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

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 4))
    
    # 'from numpy.distutils.misc_util import Configuration' statement (line 10)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/')
    import_392709 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'numpy.distutils.misc_util')

    if (type(import_392709) is not StypyTypeError):

        if (import_392709 != 'pyd_module'):
            __import__(import_392709)
            sys_modules_392710 = sys.modules[import_392709]
            import_from_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'numpy.distutils.misc_util', sys_modules_392710.module_type_store, module_type_store, ['Configuration'])
            nest_module(stypy.reporting.localization.Localization(__file__, 10, 4), __file__, sys_modules_392710, sys_modules_392710.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration

            import_from_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'numpy.distutils.misc_util', import_392709)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 4))
    
    # 'from numpy.distutils.system_info import get_info' statement (line 11)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/')
    import_392711 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'numpy.distutils.system_info')

    if (type(import_392711) is not StypyTypeError):

        if (import_392711 != 'pyd_module'):
            __import__(import_392711)
            sys_modules_392712 = sys.modules[import_392711]
            import_from_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'numpy.distutils.system_info', sys_modules_392712.module_type_store, module_type_store, ['get_info'])
            nest_module(stypy.reporting.localization.Localization(__file__, 11, 4), __file__, sys_modules_392712, sys_modules_392712.module_type_store, module_type_store)
        else:
            from numpy.distutils.system_info import get_info

            import_from_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'numpy.distutils.system_info', None, module_type_store, ['get_info'], [get_info])

    else:
        # Assigning a type to the variable 'numpy.distutils.system_info' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'numpy.distutils.system_info', import_392711)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 4))
    
    # 'from scipy._build_utils import get_sgemv_fix' statement (line 12)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/')
    import_392713 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 4), 'scipy._build_utils')

    if (type(import_392713) is not StypyTypeError):

        if (import_392713 != 'pyd_module'):
            __import__(import_392713)
            sys_modules_392714 = sys.modules[import_392713]
            import_from_module(stypy.reporting.localization.Localization(__file__, 12, 4), 'scipy._build_utils', sys_modules_392714.module_type_store, module_type_store, ['get_sgemv_fix'])
            nest_module(stypy.reporting.localization.Localization(__file__, 12, 4), __file__, sys_modules_392714, sys_modules_392714.module_type_store, module_type_store)
        else:
            from scipy._build_utils import get_sgemv_fix

            import_from_module(stypy.reporting.localization.Localization(__file__, 12, 4), 'scipy._build_utils', None, module_type_store, ['get_sgemv_fix'], [get_sgemv_fix])

    else:
        # Assigning a type to the variable 'scipy._build_utils' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'scipy._build_utils', import_392713)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 4))
    
    # 'from scipy._build_utils import numpy_nodepr_api' statement (line 13)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/')
    import_392715 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 4), 'scipy._build_utils')

    if (type(import_392715) is not StypyTypeError):

        if (import_392715 != 'pyd_module'):
            __import__(import_392715)
            sys_modules_392716 = sys.modules[import_392715]
            import_from_module(stypy.reporting.localization.Localization(__file__, 13, 4), 'scipy._build_utils', sys_modules_392716.module_type_store, module_type_store, ['numpy_nodepr_api'])
            nest_module(stypy.reporting.localization.Localization(__file__, 13, 4), __file__, sys_modules_392716, sys_modules_392716.module_type_store, module_type_store)
        else:
            from scipy._build_utils import numpy_nodepr_api

            import_from_module(stypy.reporting.localization.Localization(__file__, 13, 4), 'scipy._build_utils', None, module_type_store, ['numpy_nodepr_api'], [numpy_nodepr_api])

    else:
        # Assigning a type to the variable 'scipy._build_utils' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'scipy._build_utils', import_392715)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/')
    
    
    # Assigning a Call to a Name (line 15):
    
    # Call to Configuration(...): (line 15)
    # Processing the call arguments (line 15)
    str_392718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 27), 'str', 'dsolve')
    # Getting the type of 'parent_package' (line 15)
    parent_package_392719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 36), 'parent_package', False)
    # Getting the type of 'top_path' (line 15)
    top_path_392720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 51), 'top_path', False)
    # Processing the call keyword arguments (line 15)
    kwargs_392721 = {}
    # Getting the type of 'Configuration' (line 15)
    Configuration_392717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 15)
    Configuration_call_result_392722 = invoke(stypy.reporting.localization.Localization(__file__, 15, 13), Configuration_392717, *[str_392718, parent_package_392719, top_path_392720], **kwargs_392721)
    
    # Assigning a type to the variable 'config' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'config', Configuration_call_result_392722)
    
    # Call to add_data_dir(...): (line 16)
    # Processing the call arguments (line 16)
    str_392725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 24), 'str', 'tests')
    # Processing the call keyword arguments (line 16)
    kwargs_392726 = {}
    # Getting the type of 'config' (line 16)
    config_392723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'config', False)
    # Obtaining the member 'add_data_dir' of a type (line 16)
    add_data_dir_392724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 4), config_392723, 'add_data_dir')
    # Calling add_data_dir(args, kwargs) (line 16)
    add_data_dir_call_result_392727 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), add_data_dir_392724, *[str_392725], **kwargs_392726)
    
    
    # Assigning a Call to a Name (line 18):
    
    # Call to get_info(...): (line 18)
    # Processing the call arguments (line 18)
    str_392729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 26), 'str', 'lapack_opt')
    # Processing the call keyword arguments (line 18)
    int_392730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 55), 'int')
    keyword_392731 = int_392730
    kwargs_392732 = {'notfound_action': keyword_392731}
    # Getting the type of 'get_info' (line 18)
    get_info_392728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 17), 'get_info', False)
    # Calling get_info(args, kwargs) (line 18)
    get_info_call_result_392733 = invoke(stypy.reporting.localization.Localization(__file__, 18, 17), get_info_392728, *[str_392729], **kwargs_392732)
    
    # Assigning a type to the variable 'lapack_opt' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'lapack_opt', get_info_call_result_392733)
    
    
    # Getting the type of 'sys' (line 19)
    sys_392734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 7), 'sys')
    # Obtaining the member 'platform' of a type (line 19)
    platform_392735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 7), sys_392734, 'platform')
    str_392736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 23), 'str', 'win32')
    # Applying the binary operator '==' (line 19)
    result_eq_392737 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 7), '==', platform_392735, str_392736)
    
    # Testing the type of an if condition (line 19)
    if_condition_392738 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 19, 4), result_eq_392737)
    # Assigning a type to the variable 'if_condition_392738' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'if_condition_392738', if_condition_392738)
    # SSA begins for if statement (line 19)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 20):
    
    # Obtaining an instance of the builtin type 'list' (line 20)
    list_392739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 20)
    # Adding element type (line 20)
    
    # Obtaining an instance of the builtin type 'tuple' (line 20)
    tuple_392740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 20)
    # Adding element type (line 20)
    str_392741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 25), 'str', 'NO_TIMER')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 25), tuple_392740, str_392741)
    # Adding element type (line 20)
    int_392742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 36), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 25), tuple_392740, int_392742)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 23), list_392739, tuple_392740)
    
    # Assigning a type to the variable 'superlu_defs' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'superlu_defs', list_392739)
    # SSA branch for the else part of an if statement (line 19)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a List to a Name (line 22):
    
    # Obtaining an instance of the builtin type 'list' (line 22)
    list_392743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 22)
    
    # Assigning a type to the variable 'superlu_defs' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'superlu_defs', list_392743)
    # SSA join for if statement (line 19)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 23)
    # Processing the call arguments (line 23)
    
    # Obtaining an instance of the builtin type 'tuple' (line 23)
    tuple_392746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 23)
    # Adding element type (line 23)
    str_392747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 25), 'str', 'USE_VENDOR_BLAS')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 25), tuple_392746, str_392747)
    # Adding element type (line 23)
    int_392748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 25), tuple_392746, int_392748)
    
    # Processing the call keyword arguments (line 23)
    kwargs_392749 = {}
    # Getting the type of 'superlu_defs' (line 23)
    superlu_defs_392744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'superlu_defs', False)
    # Obtaining the member 'append' of a type (line 23)
    append_392745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 4), superlu_defs_392744, 'append')
    # Calling append(args, kwargs) (line 23)
    append_call_result_392750 = invoke(stypy.reporting.localization.Localization(__file__, 23, 4), append_392745, *[tuple_392746], **kwargs_392749)
    
    
    # Assigning a Call to a Name (line 25):
    
    # Call to join(...): (line 25)
    # Processing the call arguments (line 25)
    
    # Call to dirname(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of '__file__' (line 25)
    file___392753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 31), '__file__', False)
    # Processing the call keyword arguments (line 25)
    kwargs_392754 = {}
    # Getting the type of 'dirname' (line 25)
    dirname_392752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 23), 'dirname', False)
    # Calling dirname(args, kwargs) (line 25)
    dirname_call_result_392755 = invoke(stypy.reporting.localization.Localization(__file__, 25, 23), dirname_392752, *[file___392753], **kwargs_392754)
    
    str_392756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 42), 'str', 'SuperLU')
    str_392757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 53), 'str', 'SRC')
    # Processing the call keyword arguments (line 25)
    kwargs_392758 = {}
    # Getting the type of 'join' (line 25)
    join_392751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 18), 'join', False)
    # Calling join(args, kwargs) (line 25)
    join_call_result_392759 = invoke(stypy.reporting.localization.Localization(__file__, 25, 18), join_392751, *[dirname_call_result_392755, str_392756, str_392757], **kwargs_392758)
    
    # Assigning a type to the variable 'superlu_src' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'superlu_src', join_call_result_392759)
    
    # Assigning a Call to a Name (line 27):
    
    # Call to list(...): (line 27)
    # Processing the call arguments (line 27)
    
    # Call to glob(...): (line 27)
    # Processing the call arguments (line 27)
    
    # Call to join(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'superlu_src' (line 27)
    superlu_src_392764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 34), 'superlu_src', False)
    str_392765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 47), 'str', '*.c')
    # Processing the call keyword arguments (line 27)
    kwargs_392766 = {}
    # Getting the type of 'join' (line 27)
    join_392763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 29), 'join', False)
    # Calling join(args, kwargs) (line 27)
    join_call_result_392767 = invoke(stypy.reporting.localization.Localization(__file__, 27, 29), join_392763, *[superlu_src_392764, str_392765], **kwargs_392766)
    
    # Processing the call keyword arguments (line 27)
    kwargs_392768 = {}
    # Getting the type of 'glob' (line 27)
    glob_392761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 19), 'glob', False)
    # Obtaining the member 'glob' of a type (line 27)
    glob_392762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 19), glob_392761, 'glob')
    # Calling glob(args, kwargs) (line 27)
    glob_call_result_392769 = invoke(stypy.reporting.localization.Localization(__file__, 27, 19), glob_392762, *[join_call_result_392767], **kwargs_392768)
    
    # Processing the call keyword arguments (line 27)
    kwargs_392770 = {}
    # Getting the type of 'list' (line 27)
    list_392760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 14), 'list', False)
    # Calling list(args, kwargs) (line 27)
    list_call_result_392771 = invoke(stypy.reporting.localization.Localization(__file__, 27, 14), list_392760, *[glob_call_result_392769], **kwargs_392770)
    
    # Assigning a type to the variable 'sources' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'sources', list_call_result_392771)
    
    # Assigning a Call to a Name (line 28):
    
    # Call to list(...): (line 28)
    # Processing the call arguments (line 28)
    
    # Call to glob(...): (line 28)
    # Processing the call arguments (line 28)
    
    # Call to join(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'superlu_src' (line 28)
    superlu_src_392776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 34), 'superlu_src', False)
    str_392777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 47), 'str', '*.h')
    # Processing the call keyword arguments (line 28)
    kwargs_392778 = {}
    # Getting the type of 'join' (line 28)
    join_392775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 29), 'join', False)
    # Calling join(args, kwargs) (line 28)
    join_call_result_392779 = invoke(stypy.reporting.localization.Localization(__file__, 28, 29), join_392775, *[superlu_src_392776, str_392777], **kwargs_392778)
    
    # Processing the call keyword arguments (line 28)
    kwargs_392780 = {}
    # Getting the type of 'glob' (line 28)
    glob_392773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 19), 'glob', False)
    # Obtaining the member 'glob' of a type (line 28)
    glob_392774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 19), glob_392773, 'glob')
    # Calling glob(args, kwargs) (line 28)
    glob_call_result_392781 = invoke(stypy.reporting.localization.Localization(__file__, 28, 19), glob_392774, *[join_call_result_392779], **kwargs_392780)
    
    # Processing the call keyword arguments (line 28)
    kwargs_392782 = {}
    # Getting the type of 'list' (line 28)
    list_392772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 14), 'list', False)
    # Calling list(args, kwargs) (line 28)
    list_call_result_392783 = invoke(stypy.reporting.localization.Localization(__file__, 28, 14), list_392772, *[glob_call_result_392781], **kwargs_392782)
    
    # Assigning a type to the variable 'headers' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'headers', list_call_result_392783)
    
    # Call to add_library(...): (line 30)
    # Processing the call arguments (line 30)
    str_392786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 23), 'str', 'superlu_src')
    # Processing the call keyword arguments (line 30)
    # Getting the type of 'sources' (line 31)
    sources_392787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 31), 'sources', False)
    keyword_392788 = sources_392787
    # Getting the type of 'superlu_defs' (line 32)
    superlu_defs_392789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 30), 'superlu_defs', False)
    keyword_392790 = superlu_defs_392789
    
    # Obtaining an instance of the builtin type 'list' (line 33)
    list_392791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 36), 'list')
    # Adding type elements to the builtin type 'list' instance (line 33)
    # Adding element type (line 33)
    # Getting the type of 'superlu_src' (line 33)
    superlu_src_392792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 37), 'superlu_src', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 36), list_392791, superlu_src_392792)
    
    keyword_392793 = list_392791
    kwargs_392794 = {'sources': keyword_392788, 'include_dirs': keyword_392793, 'macros': keyword_392790}
    # Getting the type of 'config' (line 30)
    config_392784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'config', False)
    # Obtaining the member 'add_library' of a type (line 30)
    add_library_392785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 4), config_392784, 'add_library')
    # Calling add_library(args, kwargs) (line 30)
    add_library_call_result_392795 = invoke(stypy.reporting.localization.Localization(__file__, 30, 4), add_library_392785, *[str_392786], **kwargs_392794)
    
    
    # Assigning a List to a Name (line 37):
    
    # Obtaining an instance of the builtin type 'list' (line 37)
    list_392796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 37)
    # Adding element type (line 37)
    str_392797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 19), 'str', '_superlumodule.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 18), list_392796, str_392797)
    # Adding element type (line 37)
    str_392798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 19), 'str', '_superlu_utils.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 18), list_392796, str_392798)
    # Adding element type (line 37)
    str_392799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 19), 'str', '_superluobject.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 18), list_392796, str_392799)
    
    # Assigning a type to the variable 'ext_sources' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'ext_sources', list_392796)
    
    # Getting the type of 'ext_sources' (line 40)
    ext_sources_392800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'ext_sources')
    
    # Call to get_sgemv_fix(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'lapack_opt' (line 40)
    lapack_opt_392802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 33), 'lapack_opt', False)
    # Processing the call keyword arguments (line 40)
    kwargs_392803 = {}
    # Getting the type of 'get_sgemv_fix' (line 40)
    get_sgemv_fix_392801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 19), 'get_sgemv_fix', False)
    # Calling get_sgemv_fix(args, kwargs) (line 40)
    get_sgemv_fix_call_result_392804 = invoke(stypy.reporting.localization.Localization(__file__, 40, 19), get_sgemv_fix_392801, *[lapack_opt_392802], **kwargs_392803)
    
    # Applying the binary operator '+=' (line 40)
    result_iadd_392805 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 4), '+=', ext_sources_392800, get_sgemv_fix_call_result_392804)
    # Assigning a type to the variable 'ext_sources' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'ext_sources', result_iadd_392805)
    
    
    # Call to add_extension(...): (line 42)
    # Processing the call arguments (line 42)
    str_392808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 25), 'str', '_superlu')
    # Processing the call keyword arguments (line 42)
    # Getting the type of 'ext_sources' (line 43)
    ext_sources_392809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 33), 'ext_sources', False)
    keyword_392810 = ext_sources_392809
    
    # Obtaining an instance of the builtin type 'list' (line 44)
    list_392811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 44)
    # Adding element type (line 44)
    str_392812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 36), 'str', 'superlu_src')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 35), list_392811, str_392812)
    
    keyword_392813 = list_392811
    # Getting the type of 'sources' (line 45)
    sources_392814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 34), 'sources', False)
    # Getting the type of 'headers' (line 45)
    headers_392815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 44), 'headers', False)
    # Applying the binary operator '+' (line 45)
    result_add_392816 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 34), '+', sources_392814, headers_392815)
    
    keyword_392817 = result_add_392816
    # Getting the type of 'lapack_opt' (line 46)
    lapack_opt_392818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 36), 'lapack_opt', False)
    keyword_392819 = lapack_opt_392818
    # Getting the type of 'numpy_nodepr_api' (line 47)
    numpy_nodepr_api_392820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 27), 'numpy_nodepr_api', False)
    kwargs_392821 = {'libraries': keyword_392813, 'sources': keyword_392810, 'depends': keyword_392817, 'numpy_nodepr_api_392820': numpy_nodepr_api_392820, 'extra_info': keyword_392819}
    # Getting the type of 'config' (line 42)
    config_392806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 42)
    add_extension_392807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 4), config_392806, 'add_extension')
    # Calling add_extension(args, kwargs) (line 42)
    add_extension_call_result_392822 = invoke(stypy.reporting.localization.Localization(__file__, 42, 4), add_extension_392807, *[str_392808], **kwargs_392821)
    
    
    # Call to add_data_files(...): (line 51)
    # Processing the call arguments (line 51)
    str_392825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 26), 'str', 'SuperLU/License.txt')
    # Processing the call keyword arguments (line 51)
    kwargs_392826 = {}
    # Getting the type of 'config' (line 51)
    config_392823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'config', False)
    # Obtaining the member 'add_data_files' of a type (line 51)
    add_data_files_392824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 4), config_392823, 'add_data_files')
    # Calling add_data_files(args, kwargs) (line 51)
    add_data_files_call_result_392827 = invoke(stypy.reporting.localization.Localization(__file__, 51, 4), add_data_files_392824, *[str_392825], **kwargs_392826)
    
    # Getting the type of 'config' (line 53)
    config_392828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'stypy_return_type', config_392828)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_392829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_392829)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_392829

# Assigning a type to the variable 'configuration' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 56, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 56)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/')
    import_392830 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 56, 4), 'numpy.distutils.core')

    if (type(import_392830) is not StypyTypeError):

        if (import_392830 != 'pyd_module'):
            __import__(import_392830)
            sys_modules_392831 = sys.modules[import_392830]
            import_from_module(stypy.reporting.localization.Localization(__file__, 56, 4), 'numpy.distutils.core', sys_modules_392831.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 56, 4), __file__, sys_modules_392831, sys_modules_392831.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 56, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'numpy.distutils.core', import_392830)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/dsolve/')
    
    
    # Call to setup(...): (line 57)
    # Processing the call keyword arguments (line 57)
    
    # Call to todict(...): (line 57)
    # Processing the call keyword arguments (line 57)
    kwargs_392839 = {}
    
    # Call to configuration(...): (line 57)
    # Processing the call keyword arguments (line 57)
    str_392834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 35), 'str', '')
    keyword_392835 = str_392834
    kwargs_392836 = {'top_path': keyword_392835}
    # Getting the type of 'configuration' (line 57)
    configuration_392833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'configuration', False)
    # Calling configuration(args, kwargs) (line 57)
    configuration_call_result_392837 = invoke(stypy.reporting.localization.Localization(__file__, 57, 12), configuration_392833, *[], **kwargs_392836)
    
    # Obtaining the member 'todict' of a type (line 57)
    todict_392838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 12), configuration_call_result_392837, 'todict')
    # Calling todict(args, kwargs) (line 57)
    todict_call_result_392840 = invoke(stypy.reporting.localization.Localization(__file__, 57, 12), todict_392838, *[], **kwargs_392839)
    
    kwargs_392841 = {'todict_call_result_392840': todict_call_result_392840}
    # Getting the type of 'setup' (line 57)
    setup_392832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 57)
    setup_call_result_392842 = invoke(stypy.reporting.localization.Localization(__file__, 57, 4), setup_392832, *[], **kwargs_392841)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
