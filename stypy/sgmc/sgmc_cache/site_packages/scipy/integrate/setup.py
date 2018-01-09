
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import os
4: from os.path import join
5: 
6: from scipy._build_utils import numpy_nodepr_api
7: 
8: 
9: def configuration(parent_package='',top_path=None):
10:     from numpy.distutils.misc_util import Configuration
11:     from numpy.distutils.system_info import get_info
12:     config = Configuration('integrate', parent_package, top_path)
13: 
14:     # Get a local copy of lapack_opt_info
15:     lapack_opt = dict(get_info('lapack_opt',notfound_action=2))
16:     # Pop off the libraries list so it can be combined with
17:     # additional required libraries
18:     lapack_libs = lapack_opt.pop('libraries', [])
19: 
20:     mach_src = [join('mach','*.f')]
21:     quadpack_src = [join('quadpack', '*.f')]
22:     lsoda_src = [join('odepack', fn) for fn in [
23:         'blkdta000.f', 'bnorm.f', 'cfode.f',
24:         'ewset.f', 'fnorm.f', 'intdy.f',
25:         'lsoda.f', 'prja.f', 'solsy.f', 'srcma.f',
26:         'stoda.f', 'vmnorm.f', 'xerrwv.f', 'xsetf.f',
27:         'xsetun.f']]
28:     vode_src = [join('odepack', 'vode.f'), join('odepack', 'zvode.f')]
29:     dop_src = [join('dop','*.f')]
30:     quadpack_test_src = [join('tests','_test_multivariate.c')]
31:     odeint_banded_test_src = [join('tests', 'banded5x5.f')]
32: 
33:     config.add_library('mach', sources=mach_src,
34:                        config_fc={'noopt':(__file__,1)})
35:     config.add_library('quadpack', sources=quadpack_src)
36:     config.add_library('lsoda', sources=lsoda_src)
37:     config.add_library('vode', sources=vode_src)
38:     config.add_library('dop', sources=dop_src)
39: 
40:     # Extensions
41:     # quadpack:
42:     include_dirs = [join(os.path.dirname(__file__), '..', '_lib', 'src')]
43:     if 'include_dirs' in lapack_opt:
44:         lapack_opt = dict(lapack_opt)
45:         include_dirs.extend(lapack_opt.pop('include_dirs'))
46: 
47:     config.add_extension('_quadpack',
48:                          sources=['_quadpackmodule.c'],
49:                          libraries=['quadpack', 'mach'] + lapack_libs,
50:                          depends=(['__quadpack.h']
51:                                   + quadpack_src + mach_src),
52:                          include_dirs=include_dirs,
53:                          **lapack_opt)
54: 
55:     # odepack/lsoda-odeint
56:     odepack_opts = lapack_opt.copy()
57:     odepack_opts.update(numpy_nodepr_api)
58:     config.add_extension('_odepack',
59:                          sources=['_odepackmodule.c'],
60:                          libraries=['lsoda', 'mach'] + lapack_libs,
61:                          depends=(lsoda_src + mach_src),
62:                          **odepack_opts)
63: 
64:     # vode
65:     config.add_extension('vode',
66:                          sources=['vode.pyf'],
67:                          libraries=['vode'] + lapack_libs,
68:                          depends=vode_src,
69:                          **lapack_opt)
70: 
71:     # lsoda
72:     config.add_extension('lsoda',
73:                          sources=['lsoda.pyf'],
74:                          libraries=['lsoda', 'mach'] + lapack_libs,
75:                          depends=(lsoda_src + mach_src),
76:                          **lapack_opt)
77: 
78:     # dop
79:     config.add_extension('_dop',
80:                          sources=['dop.pyf'],
81:                          libraries=['dop'],
82:                          depends=dop_src)
83: 
84:     config.add_extension('_test_multivariate',
85:                          sources=quadpack_test_src)
86: 
87:     # Fortran+f2py extension module for testing odeint.
88:     config.add_extension('_test_odeint_banded',
89:                          sources=odeint_banded_test_src,
90:                          libraries=['lsoda', 'mach'] + lapack_libs,
91:                          depends=(lsoda_src + mach_src),
92:                          **lapack_opt)
93: 
94:     config.add_subpackage('_ivp')
95: 
96:     config.add_data_dir('tests')
97:     return config
98: 
99: 
100: if __name__ == '__main__':
101:     from numpy.distutils.core import setup
102:     setup(**configuration(top_path='').todict())
103: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from os.path import join' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/')
import_32066 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'os.path')

if (type(import_32066) is not StypyTypeError):

    if (import_32066 != 'pyd_module'):
        __import__(import_32066)
        sys_modules_32067 = sys.modules[import_32066]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'os.path', sys_modules_32067.module_type_store, module_type_store, ['join'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_32067, sys_modules_32067.module_type_store, module_type_store)
    else:
        from os.path import join

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'os.path', None, module_type_store, ['join'], [join])

else:
    # Assigning a type to the variable 'os.path' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'os.path', import_32066)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy._build_utils import numpy_nodepr_api' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/')
import_32068 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy._build_utils')

if (type(import_32068) is not StypyTypeError):

    if (import_32068 != 'pyd_module'):
        __import__(import_32068)
        sys_modules_32069 = sys.modules[import_32068]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy._build_utils', sys_modules_32069.module_type_store, module_type_store, ['numpy_nodepr_api'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_32069, sys_modules_32069.module_type_store, module_type_store)
    else:
        from scipy._build_utils import numpy_nodepr_api

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy._build_utils', None, module_type_store, ['numpy_nodepr_api'], [numpy_nodepr_api])

else:
    # Assigning a type to the variable 'scipy._build_utils' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy._build_utils', import_32068)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/')


@norecursion
def configuration(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_32070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 33), 'str', '')
    # Getting the type of 'None' (line 9)
    None_32071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 45), 'None')
    defaults = [str_32070, None_32071]
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
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/')
    import_32072 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'numpy.distutils.misc_util')

    if (type(import_32072) is not StypyTypeError):

        if (import_32072 != 'pyd_module'):
            __import__(import_32072)
            sys_modules_32073 = sys.modules[import_32072]
            import_from_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'numpy.distutils.misc_util', sys_modules_32073.module_type_store, module_type_store, ['Configuration'])
            nest_module(stypy.reporting.localization.Localization(__file__, 10, 4), __file__, sys_modules_32073, sys_modules_32073.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration

            import_from_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'numpy.distutils.misc_util', import_32072)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 4))
    
    # 'from numpy.distutils.system_info import get_info' statement (line 11)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/')
    import_32074 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'numpy.distutils.system_info')

    if (type(import_32074) is not StypyTypeError):

        if (import_32074 != 'pyd_module'):
            __import__(import_32074)
            sys_modules_32075 = sys.modules[import_32074]
            import_from_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'numpy.distutils.system_info', sys_modules_32075.module_type_store, module_type_store, ['get_info'])
            nest_module(stypy.reporting.localization.Localization(__file__, 11, 4), __file__, sys_modules_32075, sys_modules_32075.module_type_store, module_type_store)
        else:
            from numpy.distutils.system_info import get_info

            import_from_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'numpy.distutils.system_info', None, module_type_store, ['get_info'], [get_info])

    else:
        # Assigning a type to the variable 'numpy.distutils.system_info' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'numpy.distutils.system_info', import_32074)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/')
    
    
    # Assigning a Call to a Name (line 12):
    
    # Call to Configuration(...): (line 12)
    # Processing the call arguments (line 12)
    str_32077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 27), 'str', 'integrate')
    # Getting the type of 'parent_package' (line 12)
    parent_package_32078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 40), 'parent_package', False)
    # Getting the type of 'top_path' (line 12)
    top_path_32079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 56), 'top_path', False)
    # Processing the call keyword arguments (line 12)
    kwargs_32080 = {}
    # Getting the type of 'Configuration' (line 12)
    Configuration_32076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 12)
    Configuration_call_result_32081 = invoke(stypy.reporting.localization.Localization(__file__, 12, 13), Configuration_32076, *[str_32077, parent_package_32078, top_path_32079], **kwargs_32080)
    
    # Assigning a type to the variable 'config' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'config', Configuration_call_result_32081)
    
    # Assigning a Call to a Name (line 15):
    
    # Call to dict(...): (line 15)
    # Processing the call arguments (line 15)
    
    # Call to get_info(...): (line 15)
    # Processing the call arguments (line 15)
    str_32084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 31), 'str', 'lapack_opt')
    # Processing the call keyword arguments (line 15)
    int_32085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 60), 'int')
    keyword_32086 = int_32085
    kwargs_32087 = {'notfound_action': keyword_32086}
    # Getting the type of 'get_info' (line 15)
    get_info_32083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 22), 'get_info', False)
    # Calling get_info(args, kwargs) (line 15)
    get_info_call_result_32088 = invoke(stypy.reporting.localization.Localization(__file__, 15, 22), get_info_32083, *[str_32084], **kwargs_32087)
    
    # Processing the call keyword arguments (line 15)
    kwargs_32089 = {}
    # Getting the type of 'dict' (line 15)
    dict_32082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 17), 'dict', False)
    # Calling dict(args, kwargs) (line 15)
    dict_call_result_32090 = invoke(stypy.reporting.localization.Localization(__file__, 15, 17), dict_32082, *[get_info_call_result_32088], **kwargs_32089)
    
    # Assigning a type to the variable 'lapack_opt' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'lapack_opt', dict_call_result_32090)
    
    # Assigning a Call to a Name (line 18):
    
    # Call to pop(...): (line 18)
    # Processing the call arguments (line 18)
    str_32093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 33), 'str', 'libraries')
    
    # Obtaining an instance of the builtin type 'list' (line 18)
    list_32094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 46), 'list')
    # Adding type elements to the builtin type 'list' instance (line 18)
    
    # Processing the call keyword arguments (line 18)
    kwargs_32095 = {}
    # Getting the type of 'lapack_opt' (line 18)
    lapack_opt_32091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 18), 'lapack_opt', False)
    # Obtaining the member 'pop' of a type (line 18)
    pop_32092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 18), lapack_opt_32091, 'pop')
    # Calling pop(args, kwargs) (line 18)
    pop_call_result_32096 = invoke(stypy.reporting.localization.Localization(__file__, 18, 18), pop_32092, *[str_32093, list_32094], **kwargs_32095)
    
    # Assigning a type to the variable 'lapack_libs' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'lapack_libs', pop_call_result_32096)
    
    # Assigning a List to a Name (line 20):
    
    # Obtaining an instance of the builtin type 'list' (line 20)
    list_32097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 20)
    # Adding element type (line 20)
    
    # Call to join(...): (line 20)
    # Processing the call arguments (line 20)
    str_32099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 21), 'str', 'mach')
    str_32100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 28), 'str', '*.f')
    # Processing the call keyword arguments (line 20)
    kwargs_32101 = {}
    # Getting the type of 'join' (line 20)
    join_32098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'join', False)
    # Calling join(args, kwargs) (line 20)
    join_call_result_32102 = invoke(stypy.reporting.localization.Localization(__file__, 20, 16), join_32098, *[str_32099, str_32100], **kwargs_32101)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 15), list_32097, join_call_result_32102)
    
    # Assigning a type to the variable 'mach_src' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'mach_src', list_32097)
    
    # Assigning a List to a Name (line 21):
    
    # Obtaining an instance of the builtin type 'list' (line 21)
    list_32103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 21)
    # Adding element type (line 21)
    
    # Call to join(...): (line 21)
    # Processing the call arguments (line 21)
    str_32105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 25), 'str', 'quadpack')
    str_32106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 37), 'str', '*.f')
    # Processing the call keyword arguments (line 21)
    kwargs_32107 = {}
    # Getting the type of 'join' (line 21)
    join_32104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 20), 'join', False)
    # Calling join(args, kwargs) (line 21)
    join_call_result_32108 = invoke(stypy.reporting.localization.Localization(__file__, 21, 20), join_32104, *[str_32105, str_32106], **kwargs_32107)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 19), list_32103, join_call_result_32108)
    
    # Assigning a type to the variable 'quadpack_src' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'quadpack_src', list_32103)
    
    # Assigning a ListComp to a Name (line 22):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining an instance of the builtin type 'list' (line 22)
    list_32114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 47), 'list')
    # Adding type elements to the builtin type 'list' instance (line 22)
    # Adding element type (line 22)
    str_32115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 8), 'str', 'blkdta000.f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 47), list_32114, str_32115)
    # Adding element type (line 22)
    str_32116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 23), 'str', 'bnorm.f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 47), list_32114, str_32116)
    # Adding element type (line 22)
    str_32117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 34), 'str', 'cfode.f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 47), list_32114, str_32117)
    # Adding element type (line 22)
    str_32118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 8), 'str', 'ewset.f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 47), list_32114, str_32118)
    # Adding element type (line 22)
    str_32119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 19), 'str', 'fnorm.f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 47), list_32114, str_32119)
    # Adding element type (line 22)
    str_32120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 30), 'str', 'intdy.f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 47), list_32114, str_32120)
    # Adding element type (line 22)
    str_32121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 8), 'str', 'lsoda.f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 47), list_32114, str_32121)
    # Adding element type (line 22)
    str_32122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 19), 'str', 'prja.f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 47), list_32114, str_32122)
    # Adding element type (line 22)
    str_32123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 29), 'str', 'solsy.f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 47), list_32114, str_32123)
    # Adding element type (line 22)
    str_32124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 40), 'str', 'srcma.f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 47), list_32114, str_32124)
    # Adding element type (line 22)
    str_32125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 8), 'str', 'stoda.f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 47), list_32114, str_32125)
    # Adding element type (line 22)
    str_32126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 19), 'str', 'vmnorm.f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 47), list_32114, str_32126)
    # Adding element type (line 22)
    str_32127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 31), 'str', 'xerrwv.f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 47), list_32114, str_32127)
    # Adding element type (line 22)
    str_32128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 43), 'str', 'xsetf.f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 47), list_32114, str_32128)
    # Adding element type (line 22)
    str_32129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 8), 'str', 'xsetun.f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 47), list_32114, str_32129)
    
    comprehension_32130 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 17), list_32114)
    # Assigning a type to the variable 'fn' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 17), 'fn', comprehension_32130)
    
    # Call to join(...): (line 22)
    # Processing the call arguments (line 22)
    str_32110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 22), 'str', 'odepack')
    # Getting the type of 'fn' (line 22)
    fn_32111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 33), 'fn', False)
    # Processing the call keyword arguments (line 22)
    kwargs_32112 = {}
    # Getting the type of 'join' (line 22)
    join_32109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 17), 'join', False)
    # Calling join(args, kwargs) (line 22)
    join_call_result_32113 = invoke(stypy.reporting.localization.Localization(__file__, 22, 17), join_32109, *[str_32110, fn_32111], **kwargs_32112)
    
    list_32131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 17), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 17), list_32131, join_call_result_32113)
    # Assigning a type to the variable 'lsoda_src' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'lsoda_src', list_32131)
    
    # Assigning a List to a Name (line 28):
    
    # Obtaining an instance of the builtin type 'list' (line 28)
    list_32132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 28)
    # Adding element type (line 28)
    
    # Call to join(...): (line 28)
    # Processing the call arguments (line 28)
    str_32134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 21), 'str', 'odepack')
    str_32135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 32), 'str', 'vode.f')
    # Processing the call keyword arguments (line 28)
    kwargs_32136 = {}
    # Getting the type of 'join' (line 28)
    join_32133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 16), 'join', False)
    # Calling join(args, kwargs) (line 28)
    join_call_result_32137 = invoke(stypy.reporting.localization.Localization(__file__, 28, 16), join_32133, *[str_32134, str_32135], **kwargs_32136)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 15), list_32132, join_call_result_32137)
    # Adding element type (line 28)
    
    # Call to join(...): (line 28)
    # Processing the call arguments (line 28)
    str_32139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 48), 'str', 'odepack')
    str_32140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 59), 'str', 'zvode.f')
    # Processing the call keyword arguments (line 28)
    kwargs_32141 = {}
    # Getting the type of 'join' (line 28)
    join_32138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 43), 'join', False)
    # Calling join(args, kwargs) (line 28)
    join_call_result_32142 = invoke(stypy.reporting.localization.Localization(__file__, 28, 43), join_32138, *[str_32139, str_32140], **kwargs_32141)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 15), list_32132, join_call_result_32142)
    
    # Assigning a type to the variable 'vode_src' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'vode_src', list_32132)
    
    # Assigning a List to a Name (line 29):
    
    # Obtaining an instance of the builtin type 'list' (line 29)
    list_32143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 29)
    # Adding element type (line 29)
    
    # Call to join(...): (line 29)
    # Processing the call arguments (line 29)
    str_32145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 20), 'str', 'dop')
    str_32146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 26), 'str', '*.f')
    # Processing the call keyword arguments (line 29)
    kwargs_32147 = {}
    # Getting the type of 'join' (line 29)
    join_32144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 15), 'join', False)
    # Calling join(args, kwargs) (line 29)
    join_call_result_32148 = invoke(stypy.reporting.localization.Localization(__file__, 29, 15), join_32144, *[str_32145, str_32146], **kwargs_32147)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 14), list_32143, join_call_result_32148)
    
    # Assigning a type to the variable 'dop_src' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'dop_src', list_32143)
    
    # Assigning a List to a Name (line 30):
    
    # Obtaining an instance of the builtin type 'list' (line 30)
    list_32149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 30)
    # Adding element type (line 30)
    
    # Call to join(...): (line 30)
    # Processing the call arguments (line 30)
    str_32151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 30), 'str', 'tests')
    str_32152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 38), 'str', '_test_multivariate.c')
    # Processing the call keyword arguments (line 30)
    kwargs_32153 = {}
    # Getting the type of 'join' (line 30)
    join_32150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 25), 'join', False)
    # Calling join(args, kwargs) (line 30)
    join_call_result_32154 = invoke(stypy.reporting.localization.Localization(__file__, 30, 25), join_32150, *[str_32151, str_32152], **kwargs_32153)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 24), list_32149, join_call_result_32154)
    
    # Assigning a type to the variable 'quadpack_test_src' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'quadpack_test_src', list_32149)
    
    # Assigning a List to a Name (line 31):
    
    # Obtaining an instance of the builtin type 'list' (line 31)
    list_32155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 29), 'list')
    # Adding type elements to the builtin type 'list' instance (line 31)
    # Adding element type (line 31)
    
    # Call to join(...): (line 31)
    # Processing the call arguments (line 31)
    str_32157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 35), 'str', 'tests')
    str_32158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 44), 'str', 'banded5x5.f')
    # Processing the call keyword arguments (line 31)
    kwargs_32159 = {}
    # Getting the type of 'join' (line 31)
    join_32156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 30), 'join', False)
    # Calling join(args, kwargs) (line 31)
    join_call_result_32160 = invoke(stypy.reporting.localization.Localization(__file__, 31, 30), join_32156, *[str_32157, str_32158], **kwargs_32159)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 29), list_32155, join_call_result_32160)
    
    # Assigning a type to the variable 'odeint_banded_test_src' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'odeint_banded_test_src', list_32155)
    
    # Call to add_library(...): (line 33)
    # Processing the call arguments (line 33)
    str_32163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 23), 'str', 'mach')
    # Processing the call keyword arguments (line 33)
    # Getting the type of 'mach_src' (line 33)
    mach_src_32164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 39), 'mach_src', False)
    keyword_32165 = mach_src_32164
    
    # Obtaining an instance of the builtin type 'dict' (line 34)
    dict_32166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 33), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 34)
    # Adding element type (key, value) (line 34)
    str_32167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 34), 'str', 'noopt')
    
    # Obtaining an instance of the builtin type 'tuple' (line 34)
    tuple_32168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 34)
    # Adding element type (line 34)
    # Getting the type of '__file__' (line 34)
    file___32169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 43), '__file__', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 43), tuple_32168, file___32169)
    # Adding element type (line 34)
    int_32170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 52), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 43), tuple_32168, int_32170)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 33), dict_32166, (str_32167, tuple_32168))
    
    keyword_32171 = dict_32166
    kwargs_32172 = {'sources': keyword_32165, 'config_fc': keyword_32171}
    # Getting the type of 'config' (line 33)
    config_32161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'config', False)
    # Obtaining the member 'add_library' of a type (line 33)
    add_library_32162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 4), config_32161, 'add_library')
    # Calling add_library(args, kwargs) (line 33)
    add_library_call_result_32173 = invoke(stypy.reporting.localization.Localization(__file__, 33, 4), add_library_32162, *[str_32163], **kwargs_32172)
    
    
    # Call to add_library(...): (line 35)
    # Processing the call arguments (line 35)
    str_32176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 23), 'str', 'quadpack')
    # Processing the call keyword arguments (line 35)
    # Getting the type of 'quadpack_src' (line 35)
    quadpack_src_32177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 43), 'quadpack_src', False)
    keyword_32178 = quadpack_src_32177
    kwargs_32179 = {'sources': keyword_32178}
    # Getting the type of 'config' (line 35)
    config_32174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'config', False)
    # Obtaining the member 'add_library' of a type (line 35)
    add_library_32175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 4), config_32174, 'add_library')
    # Calling add_library(args, kwargs) (line 35)
    add_library_call_result_32180 = invoke(stypy.reporting.localization.Localization(__file__, 35, 4), add_library_32175, *[str_32176], **kwargs_32179)
    
    
    # Call to add_library(...): (line 36)
    # Processing the call arguments (line 36)
    str_32183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 23), 'str', 'lsoda')
    # Processing the call keyword arguments (line 36)
    # Getting the type of 'lsoda_src' (line 36)
    lsoda_src_32184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 40), 'lsoda_src', False)
    keyword_32185 = lsoda_src_32184
    kwargs_32186 = {'sources': keyword_32185}
    # Getting the type of 'config' (line 36)
    config_32181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'config', False)
    # Obtaining the member 'add_library' of a type (line 36)
    add_library_32182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 4), config_32181, 'add_library')
    # Calling add_library(args, kwargs) (line 36)
    add_library_call_result_32187 = invoke(stypy.reporting.localization.Localization(__file__, 36, 4), add_library_32182, *[str_32183], **kwargs_32186)
    
    
    # Call to add_library(...): (line 37)
    # Processing the call arguments (line 37)
    str_32190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 23), 'str', 'vode')
    # Processing the call keyword arguments (line 37)
    # Getting the type of 'vode_src' (line 37)
    vode_src_32191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 39), 'vode_src', False)
    keyword_32192 = vode_src_32191
    kwargs_32193 = {'sources': keyword_32192}
    # Getting the type of 'config' (line 37)
    config_32188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'config', False)
    # Obtaining the member 'add_library' of a type (line 37)
    add_library_32189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 4), config_32188, 'add_library')
    # Calling add_library(args, kwargs) (line 37)
    add_library_call_result_32194 = invoke(stypy.reporting.localization.Localization(__file__, 37, 4), add_library_32189, *[str_32190], **kwargs_32193)
    
    
    # Call to add_library(...): (line 38)
    # Processing the call arguments (line 38)
    str_32197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 23), 'str', 'dop')
    # Processing the call keyword arguments (line 38)
    # Getting the type of 'dop_src' (line 38)
    dop_src_32198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 38), 'dop_src', False)
    keyword_32199 = dop_src_32198
    kwargs_32200 = {'sources': keyword_32199}
    # Getting the type of 'config' (line 38)
    config_32195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'config', False)
    # Obtaining the member 'add_library' of a type (line 38)
    add_library_32196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 4), config_32195, 'add_library')
    # Calling add_library(args, kwargs) (line 38)
    add_library_call_result_32201 = invoke(stypy.reporting.localization.Localization(__file__, 38, 4), add_library_32196, *[str_32197], **kwargs_32200)
    
    
    # Assigning a List to a Name (line 42):
    
    # Obtaining an instance of the builtin type 'list' (line 42)
    list_32202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 42)
    # Adding element type (line 42)
    
    # Call to join(...): (line 42)
    # Processing the call arguments (line 42)
    
    # Call to dirname(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of '__file__' (line 42)
    file___32207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 41), '__file__', False)
    # Processing the call keyword arguments (line 42)
    kwargs_32208 = {}
    # Getting the type of 'os' (line 42)
    os_32204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 25), 'os', False)
    # Obtaining the member 'path' of a type (line 42)
    path_32205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 25), os_32204, 'path')
    # Obtaining the member 'dirname' of a type (line 42)
    dirname_32206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 25), path_32205, 'dirname')
    # Calling dirname(args, kwargs) (line 42)
    dirname_call_result_32209 = invoke(stypy.reporting.localization.Localization(__file__, 42, 25), dirname_32206, *[file___32207], **kwargs_32208)
    
    str_32210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 52), 'str', '..')
    str_32211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 58), 'str', '_lib')
    str_32212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 66), 'str', 'src')
    # Processing the call keyword arguments (line 42)
    kwargs_32213 = {}
    # Getting the type of 'join' (line 42)
    join_32203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 20), 'join', False)
    # Calling join(args, kwargs) (line 42)
    join_call_result_32214 = invoke(stypy.reporting.localization.Localization(__file__, 42, 20), join_32203, *[dirname_call_result_32209, str_32210, str_32211, str_32212], **kwargs_32213)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 19), list_32202, join_call_result_32214)
    
    # Assigning a type to the variable 'include_dirs' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'include_dirs', list_32202)
    
    
    str_32215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 7), 'str', 'include_dirs')
    # Getting the type of 'lapack_opt' (line 43)
    lapack_opt_32216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 25), 'lapack_opt')
    # Applying the binary operator 'in' (line 43)
    result_contains_32217 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 7), 'in', str_32215, lapack_opt_32216)
    
    # Testing the type of an if condition (line 43)
    if_condition_32218 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 43, 4), result_contains_32217)
    # Assigning a type to the variable 'if_condition_32218' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'if_condition_32218', if_condition_32218)
    # SSA begins for if statement (line 43)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 44):
    
    # Call to dict(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'lapack_opt' (line 44)
    lapack_opt_32220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 26), 'lapack_opt', False)
    # Processing the call keyword arguments (line 44)
    kwargs_32221 = {}
    # Getting the type of 'dict' (line 44)
    dict_32219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 21), 'dict', False)
    # Calling dict(args, kwargs) (line 44)
    dict_call_result_32222 = invoke(stypy.reporting.localization.Localization(__file__, 44, 21), dict_32219, *[lapack_opt_32220], **kwargs_32221)
    
    # Assigning a type to the variable 'lapack_opt' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'lapack_opt', dict_call_result_32222)
    
    # Call to extend(...): (line 45)
    # Processing the call arguments (line 45)
    
    # Call to pop(...): (line 45)
    # Processing the call arguments (line 45)
    str_32227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 43), 'str', 'include_dirs')
    # Processing the call keyword arguments (line 45)
    kwargs_32228 = {}
    # Getting the type of 'lapack_opt' (line 45)
    lapack_opt_32225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 28), 'lapack_opt', False)
    # Obtaining the member 'pop' of a type (line 45)
    pop_32226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 28), lapack_opt_32225, 'pop')
    # Calling pop(args, kwargs) (line 45)
    pop_call_result_32229 = invoke(stypy.reporting.localization.Localization(__file__, 45, 28), pop_32226, *[str_32227], **kwargs_32228)
    
    # Processing the call keyword arguments (line 45)
    kwargs_32230 = {}
    # Getting the type of 'include_dirs' (line 45)
    include_dirs_32223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'include_dirs', False)
    # Obtaining the member 'extend' of a type (line 45)
    extend_32224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), include_dirs_32223, 'extend')
    # Calling extend(args, kwargs) (line 45)
    extend_call_result_32231 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), extend_32224, *[pop_call_result_32229], **kwargs_32230)
    
    # SSA join for if statement (line 43)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to add_extension(...): (line 47)
    # Processing the call arguments (line 47)
    str_32234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 25), 'str', '_quadpack')
    # Processing the call keyword arguments (line 47)
    
    # Obtaining an instance of the builtin type 'list' (line 48)
    list_32235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 48)
    # Adding element type (line 48)
    str_32236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 34), 'str', '_quadpackmodule.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 33), list_32235, str_32236)
    
    keyword_32237 = list_32235
    
    # Obtaining an instance of the builtin type 'list' (line 49)
    list_32238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 49)
    # Adding element type (line 49)
    str_32239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 36), 'str', 'quadpack')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 35), list_32238, str_32239)
    # Adding element type (line 49)
    str_32240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 48), 'str', 'mach')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 35), list_32238, str_32240)
    
    # Getting the type of 'lapack_libs' (line 49)
    lapack_libs_32241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 58), 'lapack_libs', False)
    # Applying the binary operator '+' (line 49)
    result_add_32242 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 35), '+', list_32238, lapack_libs_32241)
    
    keyword_32243 = result_add_32242
    
    # Obtaining an instance of the builtin type 'list' (line 50)
    list_32244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 50)
    # Adding element type (line 50)
    str_32245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 35), 'str', '__quadpack.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 34), list_32244, str_32245)
    
    # Getting the type of 'quadpack_src' (line 51)
    quadpack_src_32246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 36), 'quadpack_src', False)
    # Applying the binary operator '+' (line 50)
    result_add_32247 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 34), '+', list_32244, quadpack_src_32246)
    
    # Getting the type of 'mach_src' (line 51)
    mach_src_32248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 51), 'mach_src', False)
    # Applying the binary operator '+' (line 51)
    result_add_32249 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 49), '+', result_add_32247, mach_src_32248)
    
    keyword_32250 = result_add_32249
    # Getting the type of 'include_dirs' (line 52)
    include_dirs_32251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 38), 'include_dirs', False)
    keyword_32252 = include_dirs_32251
    # Getting the type of 'lapack_opt' (line 53)
    lapack_opt_32253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 27), 'lapack_opt', False)
    kwargs_32254 = {'libraries': keyword_32243, 'sources': keyword_32237, 'depends': keyword_32250, 'lapack_opt_32253': lapack_opt_32253, 'include_dirs': keyword_32252}
    # Getting the type of 'config' (line 47)
    config_32232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 47)
    add_extension_32233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 4), config_32232, 'add_extension')
    # Calling add_extension(args, kwargs) (line 47)
    add_extension_call_result_32255 = invoke(stypy.reporting.localization.Localization(__file__, 47, 4), add_extension_32233, *[str_32234], **kwargs_32254)
    
    
    # Assigning a Call to a Name (line 56):
    
    # Call to copy(...): (line 56)
    # Processing the call keyword arguments (line 56)
    kwargs_32258 = {}
    # Getting the type of 'lapack_opt' (line 56)
    lapack_opt_32256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 19), 'lapack_opt', False)
    # Obtaining the member 'copy' of a type (line 56)
    copy_32257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 19), lapack_opt_32256, 'copy')
    # Calling copy(args, kwargs) (line 56)
    copy_call_result_32259 = invoke(stypy.reporting.localization.Localization(__file__, 56, 19), copy_32257, *[], **kwargs_32258)
    
    # Assigning a type to the variable 'odepack_opts' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'odepack_opts', copy_call_result_32259)
    
    # Call to update(...): (line 57)
    # Processing the call arguments (line 57)
    # Getting the type of 'numpy_nodepr_api' (line 57)
    numpy_nodepr_api_32262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 24), 'numpy_nodepr_api', False)
    # Processing the call keyword arguments (line 57)
    kwargs_32263 = {}
    # Getting the type of 'odepack_opts' (line 57)
    odepack_opts_32260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'odepack_opts', False)
    # Obtaining the member 'update' of a type (line 57)
    update_32261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 4), odepack_opts_32260, 'update')
    # Calling update(args, kwargs) (line 57)
    update_call_result_32264 = invoke(stypy.reporting.localization.Localization(__file__, 57, 4), update_32261, *[numpy_nodepr_api_32262], **kwargs_32263)
    
    
    # Call to add_extension(...): (line 58)
    # Processing the call arguments (line 58)
    str_32267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 25), 'str', '_odepack')
    # Processing the call keyword arguments (line 58)
    
    # Obtaining an instance of the builtin type 'list' (line 59)
    list_32268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 59)
    # Adding element type (line 59)
    str_32269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 34), 'str', '_odepackmodule.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 33), list_32268, str_32269)
    
    keyword_32270 = list_32268
    
    # Obtaining an instance of the builtin type 'list' (line 60)
    list_32271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 60)
    # Adding element type (line 60)
    str_32272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 36), 'str', 'lsoda')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 35), list_32271, str_32272)
    # Adding element type (line 60)
    str_32273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 45), 'str', 'mach')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 35), list_32271, str_32273)
    
    # Getting the type of 'lapack_libs' (line 60)
    lapack_libs_32274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 55), 'lapack_libs', False)
    # Applying the binary operator '+' (line 60)
    result_add_32275 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 35), '+', list_32271, lapack_libs_32274)
    
    keyword_32276 = result_add_32275
    # Getting the type of 'lsoda_src' (line 61)
    lsoda_src_32277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 34), 'lsoda_src', False)
    # Getting the type of 'mach_src' (line 61)
    mach_src_32278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 46), 'mach_src', False)
    # Applying the binary operator '+' (line 61)
    result_add_32279 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 34), '+', lsoda_src_32277, mach_src_32278)
    
    keyword_32280 = result_add_32279
    # Getting the type of 'odepack_opts' (line 62)
    odepack_opts_32281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 27), 'odepack_opts', False)
    kwargs_32282 = {'libraries': keyword_32276, 'sources': keyword_32270, 'depends': keyword_32280, 'odepack_opts_32281': odepack_opts_32281}
    # Getting the type of 'config' (line 58)
    config_32265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 58)
    add_extension_32266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 4), config_32265, 'add_extension')
    # Calling add_extension(args, kwargs) (line 58)
    add_extension_call_result_32283 = invoke(stypy.reporting.localization.Localization(__file__, 58, 4), add_extension_32266, *[str_32267], **kwargs_32282)
    
    
    # Call to add_extension(...): (line 65)
    # Processing the call arguments (line 65)
    str_32286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 25), 'str', 'vode')
    # Processing the call keyword arguments (line 65)
    
    # Obtaining an instance of the builtin type 'list' (line 66)
    list_32287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 66)
    # Adding element type (line 66)
    str_32288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 34), 'str', 'vode.pyf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 33), list_32287, str_32288)
    
    keyword_32289 = list_32287
    
    # Obtaining an instance of the builtin type 'list' (line 67)
    list_32290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 67)
    # Adding element type (line 67)
    str_32291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 36), 'str', 'vode')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 35), list_32290, str_32291)
    
    # Getting the type of 'lapack_libs' (line 67)
    lapack_libs_32292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 46), 'lapack_libs', False)
    # Applying the binary operator '+' (line 67)
    result_add_32293 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 35), '+', list_32290, lapack_libs_32292)
    
    keyword_32294 = result_add_32293
    # Getting the type of 'vode_src' (line 68)
    vode_src_32295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 33), 'vode_src', False)
    keyword_32296 = vode_src_32295
    # Getting the type of 'lapack_opt' (line 69)
    lapack_opt_32297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 27), 'lapack_opt', False)
    kwargs_32298 = {'libraries': keyword_32294, 'sources': keyword_32289, 'depends': keyword_32296, 'lapack_opt_32297': lapack_opt_32297}
    # Getting the type of 'config' (line 65)
    config_32284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 65)
    add_extension_32285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 4), config_32284, 'add_extension')
    # Calling add_extension(args, kwargs) (line 65)
    add_extension_call_result_32299 = invoke(stypy.reporting.localization.Localization(__file__, 65, 4), add_extension_32285, *[str_32286], **kwargs_32298)
    
    
    # Call to add_extension(...): (line 72)
    # Processing the call arguments (line 72)
    str_32302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 25), 'str', 'lsoda')
    # Processing the call keyword arguments (line 72)
    
    # Obtaining an instance of the builtin type 'list' (line 73)
    list_32303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 73)
    # Adding element type (line 73)
    str_32304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 34), 'str', 'lsoda.pyf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 33), list_32303, str_32304)
    
    keyword_32305 = list_32303
    
    # Obtaining an instance of the builtin type 'list' (line 74)
    list_32306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 74)
    # Adding element type (line 74)
    str_32307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 36), 'str', 'lsoda')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 35), list_32306, str_32307)
    # Adding element type (line 74)
    str_32308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 45), 'str', 'mach')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 35), list_32306, str_32308)
    
    # Getting the type of 'lapack_libs' (line 74)
    lapack_libs_32309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 55), 'lapack_libs', False)
    # Applying the binary operator '+' (line 74)
    result_add_32310 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 35), '+', list_32306, lapack_libs_32309)
    
    keyword_32311 = result_add_32310
    # Getting the type of 'lsoda_src' (line 75)
    lsoda_src_32312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 34), 'lsoda_src', False)
    # Getting the type of 'mach_src' (line 75)
    mach_src_32313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 46), 'mach_src', False)
    # Applying the binary operator '+' (line 75)
    result_add_32314 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 34), '+', lsoda_src_32312, mach_src_32313)
    
    keyword_32315 = result_add_32314
    # Getting the type of 'lapack_opt' (line 76)
    lapack_opt_32316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 27), 'lapack_opt', False)
    kwargs_32317 = {'libraries': keyword_32311, 'sources': keyword_32305, 'depends': keyword_32315, 'lapack_opt_32316': lapack_opt_32316}
    # Getting the type of 'config' (line 72)
    config_32300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 72)
    add_extension_32301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 4), config_32300, 'add_extension')
    # Calling add_extension(args, kwargs) (line 72)
    add_extension_call_result_32318 = invoke(stypy.reporting.localization.Localization(__file__, 72, 4), add_extension_32301, *[str_32302], **kwargs_32317)
    
    
    # Call to add_extension(...): (line 79)
    # Processing the call arguments (line 79)
    str_32321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 25), 'str', '_dop')
    # Processing the call keyword arguments (line 79)
    
    # Obtaining an instance of the builtin type 'list' (line 80)
    list_32322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 80)
    # Adding element type (line 80)
    str_32323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 34), 'str', 'dop.pyf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 33), list_32322, str_32323)
    
    keyword_32324 = list_32322
    
    # Obtaining an instance of the builtin type 'list' (line 81)
    list_32325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 81)
    # Adding element type (line 81)
    str_32326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 36), 'str', 'dop')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 35), list_32325, str_32326)
    
    keyword_32327 = list_32325
    # Getting the type of 'dop_src' (line 82)
    dop_src_32328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 33), 'dop_src', False)
    keyword_32329 = dop_src_32328
    kwargs_32330 = {'libraries': keyword_32327, 'sources': keyword_32324, 'depends': keyword_32329}
    # Getting the type of 'config' (line 79)
    config_32319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 79)
    add_extension_32320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 4), config_32319, 'add_extension')
    # Calling add_extension(args, kwargs) (line 79)
    add_extension_call_result_32331 = invoke(stypy.reporting.localization.Localization(__file__, 79, 4), add_extension_32320, *[str_32321], **kwargs_32330)
    
    
    # Call to add_extension(...): (line 84)
    # Processing the call arguments (line 84)
    str_32334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 25), 'str', '_test_multivariate')
    # Processing the call keyword arguments (line 84)
    # Getting the type of 'quadpack_test_src' (line 85)
    quadpack_test_src_32335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 33), 'quadpack_test_src', False)
    keyword_32336 = quadpack_test_src_32335
    kwargs_32337 = {'sources': keyword_32336}
    # Getting the type of 'config' (line 84)
    config_32332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 84)
    add_extension_32333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 4), config_32332, 'add_extension')
    # Calling add_extension(args, kwargs) (line 84)
    add_extension_call_result_32338 = invoke(stypy.reporting.localization.Localization(__file__, 84, 4), add_extension_32333, *[str_32334], **kwargs_32337)
    
    
    # Call to add_extension(...): (line 88)
    # Processing the call arguments (line 88)
    str_32341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 25), 'str', '_test_odeint_banded')
    # Processing the call keyword arguments (line 88)
    # Getting the type of 'odeint_banded_test_src' (line 89)
    odeint_banded_test_src_32342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 33), 'odeint_banded_test_src', False)
    keyword_32343 = odeint_banded_test_src_32342
    
    # Obtaining an instance of the builtin type 'list' (line 90)
    list_32344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 90)
    # Adding element type (line 90)
    str_32345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 36), 'str', 'lsoda')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 35), list_32344, str_32345)
    # Adding element type (line 90)
    str_32346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 45), 'str', 'mach')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 35), list_32344, str_32346)
    
    # Getting the type of 'lapack_libs' (line 90)
    lapack_libs_32347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 55), 'lapack_libs', False)
    # Applying the binary operator '+' (line 90)
    result_add_32348 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 35), '+', list_32344, lapack_libs_32347)
    
    keyword_32349 = result_add_32348
    # Getting the type of 'lsoda_src' (line 91)
    lsoda_src_32350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 34), 'lsoda_src', False)
    # Getting the type of 'mach_src' (line 91)
    mach_src_32351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 46), 'mach_src', False)
    # Applying the binary operator '+' (line 91)
    result_add_32352 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 34), '+', lsoda_src_32350, mach_src_32351)
    
    keyword_32353 = result_add_32352
    # Getting the type of 'lapack_opt' (line 92)
    lapack_opt_32354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 27), 'lapack_opt', False)
    kwargs_32355 = {'libraries': keyword_32349, 'sources': keyword_32343, 'depends': keyword_32353, 'lapack_opt_32354': lapack_opt_32354}
    # Getting the type of 'config' (line 88)
    config_32339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 88)
    add_extension_32340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 4), config_32339, 'add_extension')
    # Calling add_extension(args, kwargs) (line 88)
    add_extension_call_result_32356 = invoke(stypy.reporting.localization.Localization(__file__, 88, 4), add_extension_32340, *[str_32341], **kwargs_32355)
    
    
    # Call to add_subpackage(...): (line 94)
    # Processing the call arguments (line 94)
    str_32359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 26), 'str', '_ivp')
    # Processing the call keyword arguments (line 94)
    kwargs_32360 = {}
    # Getting the type of 'config' (line 94)
    config_32357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 94)
    add_subpackage_32358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 4), config_32357, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 94)
    add_subpackage_call_result_32361 = invoke(stypy.reporting.localization.Localization(__file__, 94, 4), add_subpackage_32358, *[str_32359], **kwargs_32360)
    
    
    # Call to add_data_dir(...): (line 96)
    # Processing the call arguments (line 96)
    str_32364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 24), 'str', 'tests')
    # Processing the call keyword arguments (line 96)
    kwargs_32365 = {}
    # Getting the type of 'config' (line 96)
    config_32362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'config', False)
    # Obtaining the member 'add_data_dir' of a type (line 96)
    add_data_dir_32363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 4), config_32362, 'add_data_dir')
    # Calling add_data_dir(args, kwargs) (line 96)
    add_data_dir_call_result_32366 = invoke(stypy.reporting.localization.Localization(__file__, 96, 4), add_data_dir_32363, *[str_32364], **kwargs_32365)
    
    # Getting the type of 'config' (line 97)
    config_32367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'stypy_return_type', config_32367)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_32368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_32368)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_32368

# Assigning a type to the variable 'configuration' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 101, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 101)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/integrate/')
    import_32369 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 101, 4), 'numpy.distutils.core')

    if (type(import_32369) is not StypyTypeError):

        if (import_32369 != 'pyd_module'):
            __import__(import_32369)
            sys_modules_32370 = sys.modules[import_32369]
            import_from_module(stypy.reporting.localization.Localization(__file__, 101, 4), 'numpy.distutils.core', sys_modules_32370.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 101, 4), __file__, sys_modules_32370, sys_modules_32370.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 101, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'numpy.distutils.core', import_32369)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/integrate/')
    
    
    # Call to setup(...): (line 102)
    # Processing the call keyword arguments (line 102)
    
    # Call to todict(...): (line 102)
    # Processing the call keyword arguments (line 102)
    kwargs_32378 = {}
    
    # Call to configuration(...): (line 102)
    # Processing the call keyword arguments (line 102)
    str_32373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 35), 'str', '')
    keyword_32374 = str_32373
    kwargs_32375 = {'top_path': keyword_32374}
    # Getting the type of 'configuration' (line 102)
    configuration_32372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'configuration', False)
    # Calling configuration(args, kwargs) (line 102)
    configuration_call_result_32376 = invoke(stypy.reporting.localization.Localization(__file__, 102, 12), configuration_32372, *[], **kwargs_32375)
    
    # Obtaining the member 'todict' of a type (line 102)
    todict_32377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 12), configuration_call_result_32376, 'todict')
    # Calling todict(args, kwargs) (line 102)
    todict_call_result_32379 = invoke(stypy.reporting.localization.Localization(__file__, 102, 12), todict_32377, *[], **kwargs_32378)
    
    kwargs_32380 = {'todict_call_result_32379': todict_call_result_32379}
    # Getting the type of 'setup' (line 102)
    setup_32371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 102)
    setup_call_result_32381 = invoke(stypy.reporting.localization.Localization(__file__, 102, 4), setup_32371, *[], **kwargs_32380)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
