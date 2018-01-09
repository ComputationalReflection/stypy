
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import os
4: import sys
5: from os.path import join
6: from distutils.sysconfig import get_python_inc
7: import numpy
8: from numpy.distutils.misc_util import get_numpy_include_dirs
9: 
10: try:
11:     from numpy.distutils.misc_util import get_info
12: except ImportError:
13:     raise ValueError("numpy >= 1.4 is required (detected %s from %s)" %
14:                      (numpy.__version__, numpy.__file__))
15: 
16: 
17: def configuration(parent_package='',top_path=None):
18:     from numpy.distutils.misc_util import Configuration
19:     from numpy.distutils.system_info import get_info as get_system_info
20: 
21:     config = Configuration('special', parent_package, top_path)
22: 
23:     define_macros = []
24:     if sys.platform == 'win32':
25:         # define_macros.append(('NOINFINITIES',None))
26:         # define_macros.append(('NONANS',None))
27:         define_macros.append(('_USE_MATH_DEFINES',None))
28: 
29:     curdir = os.path.abspath(os.path.dirname(__file__))
30:     inc_dirs = [get_python_inc(), os.path.join(curdir, "c_misc")]
31:     if inc_dirs[0] != get_python_inc(plat_specific=1):
32:         inc_dirs.append(get_python_inc(plat_specific=1))
33:     inc_dirs.insert(0, get_numpy_include_dirs())
34: 
35:     # C libraries
36:     c_misc_src = [join('c_misc','*.c')]
37:     c_misc_hdr = [join('c_misc','*.h')]
38:     cephes_src = [join('cephes','*.c')]
39:     cephes_hdr = [join('cephes', '*.h')]
40:     config.add_library('sc_c_misc',sources=c_misc_src,
41:                        include_dirs=[curdir] + inc_dirs,
42:                        depends=(cephes_hdr + cephes_src
43:                                 + c_misc_hdr + cephes_hdr
44:                                 + ['*.h']),
45:                        macros=define_macros)
46:     config.add_library('sc_cephes',sources=cephes_src,
47:                        include_dirs=[curdir] + inc_dirs,
48:                        depends=(cephes_hdr + ['*.h']),
49:                        macros=define_macros)
50: 
51:     # Fortran/C++ libraries
52:     mach_src = [join('mach','*.f')]
53:     amos_src = [join('amos','*.f')]
54:     cdf_src = [join('cdflib','*.f')]
55:     specfun_src = [join('specfun','*.f')]
56:     config.add_library('sc_mach',sources=mach_src,
57:                        config_fc={'noopt':(__file__,1)})
58:     config.add_library('sc_amos',sources=amos_src)
59:     config.add_library('sc_cdf',sources=cdf_src)
60:     config.add_library('sc_specfun',sources=specfun_src)
61: 
62:     # Extension specfun
63:     config.add_extension('specfun',
64:                          sources=['specfun.pyf'],
65:                          f2py_options=['--no-wrap-functions'],
66:                          depends=specfun_src,
67:                          define_macros=[],
68:                          libraries=['sc_specfun'])
69: 
70:     # Extension _ufuncs
71:     headers = ['*.h', join('c_misc', '*.h'), join('cephes', '*.h')]
72:     ufuncs_src = ['_ufuncs.c', 'sf_error.c', '_logit.c.src',
73:                   "amos_wrappers.c", "cdf_wrappers.c", "specfun_wrappers.c"]
74:     ufuncs_dep = (headers + ufuncs_src + amos_src + c_misc_src + cephes_src
75:                   + mach_src + cdf_src + specfun_src)
76:     cfg = dict(get_system_info('lapack_opt'))
77:     cfg.setdefault('include_dirs', []).extend([curdir] + inc_dirs + [numpy.get_include()])
78:     cfg.setdefault('libraries', []).extend(['sc_amos','sc_c_misc','sc_cephes','sc_mach',
79:                                             'sc_cdf', 'sc_specfun'])
80:     cfg.setdefault('define_macros', []).extend(define_macros)
81:     config.add_extension('_ufuncs',
82:                          depends=ufuncs_dep,
83:                          sources=ufuncs_src,
84:                          extra_info=get_info("npymath"),
85:                          **cfg)
86: 
87:     # Extension _ufuncs_cxx
88:     ufuncs_cxx_src = ['_ufuncs_cxx.cxx', 'sf_error.c',
89:                       '_faddeeva.cxx', 'Faddeeva.cc',
90:                       '_wright.cxx', 'wright.cc']
91:     ufuncs_cxx_dep = (headers + ufuncs_cxx_src + cephes_src
92:                       + ['*.hh'])
93:     config.add_extension('_ufuncs_cxx',
94:                          sources=ufuncs_cxx_src,
95:                          depends=ufuncs_cxx_dep,
96:                          include_dirs=[curdir],
97:                          define_macros=define_macros,
98:                          extra_info=get_info("npymath"))
99: 
100:     cfg = dict(get_system_info('lapack_opt'))
101:     config.add_extension('_ellip_harm_2',
102:                          sources=['_ellip_harm_2.c', 'sf_error.c',],
103:                          **cfg
104:                          )
105: 
106:     # Cython API
107:     config.add_data_files('cython_special.pxd')
108:     
109:     cython_special_src = ['cython_special.c', 'sf_error.c', '_logit.c.src',
110:                           "amos_wrappers.c", "cdf_wrappers.c", "specfun_wrappers.c"]
111:     cython_special_dep = (headers + ufuncs_src + ufuncs_cxx_src + amos_src
112:                           + c_misc_src + cephes_src + mach_src + cdf_src
113:                           + specfun_src)
114:     cfg = dict(get_system_info('lapack_opt'))
115:     cfg.setdefault('include_dirs', []).extend([curdir] + inc_dirs + [numpy.get_include()])
116:     cfg.setdefault('libraries', []).extend(['sc_amos','sc_c_misc','sc_cephes','sc_mach',
117:                                             'sc_cdf', 'sc_specfun'])
118:     cfg.setdefault('define_macros', []).extend(define_macros)
119:     config.add_extension('cython_special',
120:                          depends=cython_special_dep,
121:                          sources=cython_special_src,
122:                          extra_info=get_info("npymath"),
123:                          **cfg)
124: 
125:     # combinatorics
126:     config.add_extension('_comb',
127:                          sources=['_comb.c'])
128: 
129:     # testing for _round.h
130:     config.add_extension('_test_round',
131:                          sources=['_test_round.c'],
132:                          depends=['_round.h', 'c_misc/double2.h'],
133:                          include_dirs=[numpy.get_include()],
134:                          extra_info=get_info('npymath'))
135: 
136:     config.add_data_files('tests/*.py')
137:     config.add_data_files('tests/data/README')
138:     config.add_data_files('tests/data/*.npz')
139: 
140:     config.add_subpackage('_precompute')
141: 
142:     return config
143: 
144: 
145: if __name__ == '__main__':
146:     from numpy.distutils.core import setup
147:     setup(**configuration(top_path='').todict())
148: 

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

# 'import sys' statement (line 4)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from os.path import join' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_503149 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'os.path')

if (type(import_503149) is not StypyTypeError):

    if (import_503149 != 'pyd_module'):
        __import__(import_503149)
        sys_modules_503150 = sys.modules[import_503149]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'os.path', sys_modules_503150.module_type_store, module_type_store, ['join'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_503150, sys_modules_503150.module_type_store, module_type_store)
    else:
        from os.path import join

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'os.path', None, module_type_store, ['join'], [join])

else:
    # Assigning a type to the variable 'os.path' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'os.path', import_503149)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from distutils.sysconfig import get_python_inc' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_503151 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.sysconfig')

if (type(import_503151) is not StypyTypeError):

    if (import_503151 != 'pyd_module'):
        __import__(import_503151)
        sys_modules_503152 = sys.modules[import_503151]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.sysconfig', sys_modules_503152.module_type_store, module_type_store, ['get_python_inc'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_503152, sys_modules_503152.module_type_store, module_type_store)
    else:
        from distutils.sysconfig import get_python_inc

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.sysconfig', None, module_type_store, ['get_python_inc'], [get_python_inc])

else:
    # Assigning a type to the variable 'distutils.sysconfig' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.sysconfig', import_503151)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import numpy' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_503153 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy')

if (type(import_503153) is not StypyTypeError):

    if (import_503153 != 'pyd_module'):
        __import__(import_503153)
        sys_modules_503154 = sys.modules[import_503153]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', sys_modules_503154.module_type_store, module_type_store)
    else:
        import numpy

        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', import_503153)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from numpy.distutils.misc_util import get_numpy_include_dirs' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_503155 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.distutils.misc_util')

if (type(import_503155) is not StypyTypeError):

    if (import_503155 != 'pyd_module'):
        __import__(import_503155)
        sys_modules_503156 = sys.modules[import_503155]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.distutils.misc_util', sys_modules_503156.module_type_store, module_type_store, ['get_numpy_include_dirs'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_503156, sys_modules_503156.module_type_store, module_type_store)
    else:
        from numpy.distutils.misc_util import get_numpy_include_dirs

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.distutils.misc_util', None, module_type_store, ['get_numpy_include_dirs'], [get_numpy_include_dirs])

else:
    # Assigning a type to the variable 'numpy.distutils.misc_util' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.distutils.misc_util', import_503155)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')



# SSA begins for try-except statement (line 10)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 4))

# 'from numpy.distutils.misc_util import get_info' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
import_503157 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'numpy.distutils.misc_util')

if (type(import_503157) is not StypyTypeError):

    if (import_503157 != 'pyd_module'):
        __import__(import_503157)
        sys_modules_503158 = sys.modules[import_503157]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'numpy.distutils.misc_util', sys_modules_503158.module_type_store, module_type_store, ['get_info'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 4), __file__, sys_modules_503158, sys_modules_503158.module_type_store, module_type_store)
    else:
        from numpy.distutils.misc_util import get_info

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'numpy.distutils.misc_util', None, module_type_store, ['get_info'], [get_info])

else:
    # Assigning a type to the variable 'numpy.distutils.misc_util' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'numpy.distutils.misc_util', import_503157)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')

# SSA branch for the except part of a try statement (line 10)
# SSA branch for the except 'ImportError' branch of a try statement (line 10)
module_type_store.open_ssa_branch('except')

# Call to ValueError(...): (line 13)
# Processing the call arguments (line 13)
str_503160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 21), 'str', 'numpy >= 1.4 is required (detected %s from %s)')

# Obtaining an instance of the builtin type 'tuple' (line 14)
tuple_503161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 22), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 14)
# Adding element type (line 14)
# Getting the type of 'numpy' (line 14)
numpy_503162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 22), 'numpy', False)
# Obtaining the member '__version__' of a type (line 14)
version___503163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 22), numpy_503162, '__version__')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 22), tuple_503161, version___503163)
# Adding element type (line 14)
# Getting the type of 'numpy' (line 14)
numpy_503164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 41), 'numpy', False)
# Obtaining the member '__file__' of a type (line 14)
file___503165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 41), numpy_503164, '__file__')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 22), tuple_503161, file___503165)

# Applying the binary operator '%' (line 13)
result_mod_503166 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 21), '%', str_503160, tuple_503161)

# Processing the call keyword arguments (line 13)
kwargs_503167 = {}
# Getting the type of 'ValueError' (line 13)
ValueError_503159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 10), 'ValueError', False)
# Calling ValueError(args, kwargs) (line 13)
ValueError_call_result_503168 = invoke(stypy.reporting.localization.Localization(__file__, 13, 10), ValueError_503159, *[result_mod_503166], **kwargs_503167)

ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 13, 4), ValueError_call_result_503168, 'raise parameter', BaseException)
# SSA join for try-except statement (line 10)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def configuration(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_503169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 33), 'str', '')
    # Getting the type of 'None' (line 17)
    None_503170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 45), 'None')
    defaults = [str_503169, None_503170]
    # Create a new context for function 'configuration'
    module_type_store = module_type_store.open_function_context('configuration', 17, 0, False)
    
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

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 4))
    
    # 'from numpy.distutils.misc_util import Configuration' statement (line 18)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
    import_503171 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 4), 'numpy.distutils.misc_util')

    if (type(import_503171) is not StypyTypeError):

        if (import_503171 != 'pyd_module'):
            __import__(import_503171)
            sys_modules_503172 = sys.modules[import_503171]
            import_from_module(stypy.reporting.localization.Localization(__file__, 18, 4), 'numpy.distutils.misc_util', sys_modules_503172.module_type_store, module_type_store, ['Configuration'])
            nest_module(stypy.reporting.localization.Localization(__file__, 18, 4), __file__, sys_modules_503172, sys_modules_503172.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration

            import_from_module(stypy.reporting.localization.Localization(__file__, 18, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'numpy.distutils.misc_util', import_503171)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 4))
    
    # 'from numpy.distutils.system_info import get_system_info' statement (line 19)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
    import_503173 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 4), 'numpy.distutils.system_info')

    if (type(import_503173) is not StypyTypeError):

        if (import_503173 != 'pyd_module'):
            __import__(import_503173)
            sys_modules_503174 = sys.modules[import_503173]
            import_from_module(stypy.reporting.localization.Localization(__file__, 19, 4), 'numpy.distutils.system_info', sys_modules_503174.module_type_store, module_type_store, ['get_info'])
            nest_module(stypy.reporting.localization.Localization(__file__, 19, 4), __file__, sys_modules_503174, sys_modules_503174.module_type_store, module_type_store)
        else:
            from numpy.distutils.system_info import get_info as get_system_info

            import_from_module(stypy.reporting.localization.Localization(__file__, 19, 4), 'numpy.distutils.system_info', None, module_type_store, ['get_info'], [get_system_info])

    else:
        # Assigning a type to the variable 'numpy.distutils.system_info' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'numpy.distutils.system_info', import_503173)

    # Adding an alias
    module_type_store.add_alias('get_system_info', 'get_info')
    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')
    
    
    # Assigning a Call to a Name (line 21):
    
    # Call to Configuration(...): (line 21)
    # Processing the call arguments (line 21)
    str_503176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 27), 'str', 'special')
    # Getting the type of 'parent_package' (line 21)
    parent_package_503177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 38), 'parent_package', False)
    # Getting the type of 'top_path' (line 21)
    top_path_503178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 54), 'top_path', False)
    # Processing the call keyword arguments (line 21)
    kwargs_503179 = {}
    # Getting the type of 'Configuration' (line 21)
    Configuration_503175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 21)
    Configuration_call_result_503180 = invoke(stypy.reporting.localization.Localization(__file__, 21, 13), Configuration_503175, *[str_503176, parent_package_503177, top_path_503178], **kwargs_503179)
    
    # Assigning a type to the variable 'config' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'config', Configuration_call_result_503180)
    
    # Assigning a List to a Name (line 23):
    
    # Obtaining an instance of the builtin type 'list' (line 23)
    list_503181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 23)
    
    # Assigning a type to the variable 'define_macros' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'define_macros', list_503181)
    
    
    # Getting the type of 'sys' (line 24)
    sys_503182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 7), 'sys')
    # Obtaining the member 'platform' of a type (line 24)
    platform_503183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 7), sys_503182, 'platform')
    str_503184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 23), 'str', 'win32')
    # Applying the binary operator '==' (line 24)
    result_eq_503185 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 7), '==', platform_503183, str_503184)
    
    # Testing the type of an if condition (line 24)
    if_condition_503186 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 24, 4), result_eq_503185)
    # Assigning a type to the variable 'if_condition_503186' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'if_condition_503186', if_condition_503186)
    # SSA begins for if statement (line 24)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 27)
    # Processing the call arguments (line 27)
    
    # Obtaining an instance of the builtin type 'tuple' (line 27)
    tuple_503189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 27)
    # Adding element type (line 27)
    str_503190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 30), 'str', '_USE_MATH_DEFINES')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 30), tuple_503189, str_503190)
    # Adding element type (line 27)
    # Getting the type of 'None' (line 27)
    None_503191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 50), 'None', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 30), tuple_503189, None_503191)
    
    # Processing the call keyword arguments (line 27)
    kwargs_503192 = {}
    # Getting the type of 'define_macros' (line 27)
    define_macros_503187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'define_macros', False)
    # Obtaining the member 'append' of a type (line 27)
    append_503188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), define_macros_503187, 'append')
    # Calling append(args, kwargs) (line 27)
    append_call_result_503193 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), append_503188, *[tuple_503189], **kwargs_503192)
    
    # SSA join for if statement (line 24)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 29):
    
    # Call to abspath(...): (line 29)
    # Processing the call arguments (line 29)
    
    # Call to dirname(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of '__file__' (line 29)
    file___503200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 45), '__file__', False)
    # Processing the call keyword arguments (line 29)
    kwargs_503201 = {}
    # Getting the type of 'os' (line 29)
    os_503197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 29), 'os', False)
    # Obtaining the member 'path' of a type (line 29)
    path_503198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 29), os_503197, 'path')
    # Obtaining the member 'dirname' of a type (line 29)
    dirname_503199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 29), path_503198, 'dirname')
    # Calling dirname(args, kwargs) (line 29)
    dirname_call_result_503202 = invoke(stypy.reporting.localization.Localization(__file__, 29, 29), dirname_503199, *[file___503200], **kwargs_503201)
    
    # Processing the call keyword arguments (line 29)
    kwargs_503203 = {}
    # Getting the type of 'os' (line 29)
    os_503194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 13), 'os', False)
    # Obtaining the member 'path' of a type (line 29)
    path_503195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 13), os_503194, 'path')
    # Obtaining the member 'abspath' of a type (line 29)
    abspath_503196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 13), path_503195, 'abspath')
    # Calling abspath(args, kwargs) (line 29)
    abspath_call_result_503204 = invoke(stypy.reporting.localization.Localization(__file__, 29, 13), abspath_503196, *[dirname_call_result_503202], **kwargs_503203)
    
    # Assigning a type to the variable 'curdir' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'curdir', abspath_call_result_503204)
    
    # Assigning a List to a Name (line 30):
    
    # Obtaining an instance of the builtin type 'list' (line 30)
    list_503205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 30)
    # Adding element type (line 30)
    
    # Call to get_python_inc(...): (line 30)
    # Processing the call keyword arguments (line 30)
    kwargs_503207 = {}
    # Getting the type of 'get_python_inc' (line 30)
    get_python_inc_503206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 16), 'get_python_inc', False)
    # Calling get_python_inc(args, kwargs) (line 30)
    get_python_inc_call_result_503208 = invoke(stypy.reporting.localization.Localization(__file__, 30, 16), get_python_inc_503206, *[], **kwargs_503207)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 15), list_503205, get_python_inc_call_result_503208)
    # Adding element type (line 30)
    
    # Call to join(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'curdir' (line 30)
    curdir_503212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 47), 'curdir', False)
    str_503213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 55), 'str', 'c_misc')
    # Processing the call keyword arguments (line 30)
    kwargs_503214 = {}
    # Getting the type of 'os' (line 30)
    os_503209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 34), 'os', False)
    # Obtaining the member 'path' of a type (line 30)
    path_503210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 34), os_503209, 'path')
    # Obtaining the member 'join' of a type (line 30)
    join_503211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 34), path_503210, 'join')
    # Calling join(args, kwargs) (line 30)
    join_call_result_503215 = invoke(stypy.reporting.localization.Localization(__file__, 30, 34), join_503211, *[curdir_503212, str_503213], **kwargs_503214)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 15), list_503205, join_call_result_503215)
    
    # Assigning a type to the variable 'inc_dirs' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'inc_dirs', list_503205)
    
    
    
    # Obtaining the type of the subscript
    int_503216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 16), 'int')
    # Getting the type of 'inc_dirs' (line 31)
    inc_dirs_503217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 7), 'inc_dirs')
    # Obtaining the member '__getitem__' of a type (line 31)
    getitem___503218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 7), inc_dirs_503217, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 31)
    subscript_call_result_503219 = invoke(stypy.reporting.localization.Localization(__file__, 31, 7), getitem___503218, int_503216)
    
    
    # Call to get_python_inc(...): (line 31)
    # Processing the call keyword arguments (line 31)
    int_503221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 51), 'int')
    keyword_503222 = int_503221
    kwargs_503223 = {'plat_specific': keyword_503222}
    # Getting the type of 'get_python_inc' (line 31)
    get_python_inc_503220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 22), 'get_python_inc', False)
    # Calling get_python_inc(args, kwargs) (line 31)
    get_python_inc_call_result_503224 = invoke(stypy.reporting.localization.Localization(__file__, 31, 22), get_python_inc_503220, *[], **kwargs_503223)
    
    # Applying the binary operator '!=' (line 31)
    result_ne_503225 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 7), '!=', subscript_call_result_503219, get_python_inc_call_result_503224)
    
    # Testing the type of an if condition (line 31)
    if_condition_503226 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 31, 4), result_ne_503225)
    # Assigning a type to the variable 'if_condition_503226' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'if_condition_503226', if_condition_503226)
    # SSA begins for if statement (line 31)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 32)
    # Processing the call arguments (line 32)
    
    # Call to get_python_inc(...): (line 32)
    # Processing the call keyword arguments (line 32)
    int_503230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 53), 'int')
    keyword_503231 = int_503230
    kwargs_503232 = {'plat_specific': keyword_503231}
    # Getting the type of 'get_python_inc' (line 32)
    get_python_inc_503229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 24), 'get_python_inc', False)
    # Calling get_python_inc(args, kwargs) (line 32)
    get_python_inc_call_result_503233 = invoke(stypy.reporting.localization.Localization(__file__, 32, 24), get_python_inc_503229, *[], **kwargs_503232)
    
    # Processing the call keyword arguments (line 32)
    kwargs_503234 = {}
    # Getting the type of 'inc_dirs' (line 32)
    inc_dirs_503227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'inc_dirs', False)
    # Obtaining the member 'append' of a type (line 32)
    append_503228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 8), inc_dirs_503227, 'append')
    # Calling append(args, kwargs) (line 32)
    append_call_result_503235 = invoke(stypy.reporting.localization.Localization(__file__, 32, 8), append_503228, *[get_python_inc_call_result_503233], **kwargs_503234)
    
    # SSA join for if statement (line 31)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to insert(...): (line 33)
    # Processing the call arguments (line 33)
    int_503238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 20), 'int')
    
    # Call to get_numpy_include_dirs(...): (line 33)
    # Processing the call keyword arguments (line 33)
    kwargs_503240 = {}
    # Getting the type of 'get_numpy_include_dirs' (line 33)
    get_numpy_include_dirs_503239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 23), 'get_numpy_include_dirs', False)
    # Calling get_numpy_include_dirs(args, kwargs) (line 33)
    get_numpy_include_dirs_call_result_503241 = invoke(stypy.reporting.localization.Localization(__file__, 33, 23), get_numpy_include_dirs_503239, *[], **kwargs_503240)
    
    # Processing the call keyword arguments (line 33)
    kwargs_503242 = {}
    # Getting the type of 'inc_dirs' (line 33)
    inc_dirs_503236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'inc_dirs', False)
    # Obtaining the member 'insert' of a type (line 33)
    insert_503237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 4), inc_dirs_503236, 'insert')
    # Calling insert(args, kwargs) (line 33)
    insert_call_result_503243 = invoke(stypy.reporting.localization.Localization(__file__, 33, 4), insert_503237, *[int_503238, get_numpy_include_dirs_call_result_503241], **kwargs_503242)
    
    
    # Assigning a List to a Name (line 36):
    
    # Obtaining an instance of the builtin type 'list' (line 36)
    list_503244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 36)
    # Adding element type (line 36)
    
    # Call to join(...): (line 36)
    # Processing the call arguments (line 36)
    str_503246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 23), 'str', 'c_misc')
    str_503247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 32), 'str', '*.c')
    # Processing the call keyword arguments (line 36)
    kwargs_503248 = {}
    # Getting the type of 'join' (line 36)
    join_503245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 18), 'join', False)
    # Calling join(args, kwargs) (line 36)
    join_call_result_503249 = invoke(stypy.reporting.localization.Localization(__file__, 36, 18), join_503245, *[str_503246, str_503247], **kwargs_503248)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 17), list_503244, join_call_result_503249)
    
    # Assigning a type to the variable 'c_misc_src' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'c_misc_src', list_503244)
    
    # Assigning a List to a Name (line 37):
    
    # Obtaining an instance of the builtin type 'list' (line 37)
    list_503250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 37)
    # Adding element type (line 37)
    
    # Call to join(...): (line 37)
    # Processing the call arguments (line 37)
    str_503252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 23), 'str', 'c_misc')
    str_503253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 32), 'str', '*.h')
    # Processing the call keyword arguments (line 37)
    kwargs_503254 = {}
    # Getting the type of 'join' (line 37)
    join_503251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 18), 'join', False)
    # Calling join(args, kwargs) (line 37)
    join_call_result_503255 = invoke(stypy.reporting.localization.Localization(__file__, 37, 18), join_503251, *[str_503252, str_503253], **kwargs_503254)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 17), list_503250, join_call_result_503255)
    
    # Assigning a type to the variable 'c_misc_hdr' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'c_misc_hdr', list_503250)
    
    # Assigning a List to a Name (line 38):
    
    # Obtaining an instance of the builtin type 'list' (line 38)
    list_503256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 38)
    # Adding element type (line 38)
    
    # Call to join(...): (line 38)
    # Processing the call arguments (line 38)
    str_503258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 23), 'str', 'cephes')
    str_503259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 32), 'str', '*.c')
    # Processing the call keyword arguments (line 38)
    kwargs_503260 = {}
    # Getting the type of 'join' (line 38)
    join_503257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 18), 'join', False)
    # Calling join(args, kwargs) (line 38)
    join_call_result_503261 = invoke(stypy.reporting.localization.Localization(__file__, 38, 18), join_503257, *[str_503258, str_503259], **kwargs_503260)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 17), list_503256, join_call_result_503261)
    
    # Assigning a type to the variable 'cephes_src' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'cephes_src', list_503256)
    
    # Assigning a List to a Name (line 39):
    
    # Obtaining an instance of the builtin type 'list' (line 39)
    list_503262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 39)
    # Adding element type (line 39)
    
    # Call to join(...): (line 39)
    # Processing the call arguments (line 39)
    str_503264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 23), 'str', 'cephes')
    str_503265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 33), 'str', '*.h')
    # Processing the call keyword arguments (line 39)
    kwargs_503266 = {}
    # Getting the type of 'join' (line 39)
    join_503263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 18), 'join', False)
    # Calling join(args, kwargs) (line 39)
    join_call_result_503267 = invoke(stypy.reporting.localization.Localization(__file__, 39, 18), join_503263, *[str_503264, str_503265], **kwargs_503266)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 17), list_503262, join_call_result_503267)
    
    # Assigning a type to the variable 'cephes_hdr' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'cephes_hdr', list_503262)
    
    # Call to add_library(...): (line 40)
    # Processing the call arguments (line 40)
    str_503270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 23), 'str', 'sc_c_misc')
    # Processing the call keyword arguments (line 40)
    # Getting the type of 'c_misc_src' (line 40)
    c_misc_src_503271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 43), 'c_misc_src', False)
    keyword_503272 = c_misc_src_503271
    
    # Obtaining an instance of the builtin type 'list' (line 41)
    list_503273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 36), 'list')
    # Adding type elements to the builtin type 'list' instance (line 41)
    # Adding element type (line 41)
    # Getting the type of 'curdir' (line 41)
    curdir_503274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 37), 'curdir', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 36), list_503273, curdir_503274)
    
    # Getting the type of 'inc_dirs' (line 41)
    inc_dirs_503275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 47), 'inc_dirs', False)
    # Applying the binary operator '+' (line 41)
    result_add_503276 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 36), '+', list_503273, inc_dirs_503275)
    
    keyword_503277 = result_add_503276
    # Getting the type of 'cephes_hdr' (line 42)
    cephes_hdr_503278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 32), 'cephes_hdr', False)
    # Getting the type of 'cephes_src' (line 42)
    cephes_src_503279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 45), 'cephes_src', False)
    # Applying the binary operator '+' (line 42)
    result_add_503280 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 32), '+', cephes_hdr_503278, cephes_src_503279)
    
    # Getting the type of 'c_misc_hdr' (line 43)
    c_misc_hdr_503281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 34), 'c_misc_hdr', False)
    # Applying the binary operator '+' (line 43)
    result_add_503282 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 32), '+', result_add_503280, c_misc_hdr_503281)
    
    # Getting the type of 'cephes_hdr' (line 43)
    cephes_hdr_503283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 47), 'cephes_hdr', False)
    # Applying the binary operator '+' (line 43)
    result_add_503284 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 45), '+', result_add_503282, cephes_hdr_503283)
    
    
    # Obtaining an instance of the builtin type 'list' (line 44)
    list_503285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 44)
    # Adding element type (line 44)
    str_503286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 35), 'str', '*.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 34), list_503285, str_503286)
    
    # Applying the binary operator '+' (line 44)
    result_add_503287 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 32), '+', result_add_503284, list_503285)
    
    keyword_503288 = result_add_503287
    # Getting the type of 'define_macros' (line 45)
    define_macros_503289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 30), 'define_macros', False)
    keyword_503290 = define_macros_503289
    kwargs_503291 = {'sources': keyword_503272, 'depends': keyword_503288, 'macros': keyword_503290, 'include_dirs': keyword_503277}
    # Getting the type of 'config' (line 40)
    config_503268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'config', False)
    # Obtaining the member 'add_library' of a type (line 40)
    add_library_503269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 4), config_503268, 'add_library')
    # Calling add_library(args, kwargs) (line 40)
    add_library_call_result_503292 = invoke(stypy.reporting.localization.Localization(__file__, 40, 4), add_library_503269, *[str_503270], **kwargs_503291)
    
    
    # Call to add_library(...): (line 46)
    # Processing the call arguments (line 46)
    str_503295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 23), 'str', 'sc_cephes')
    # Processing the call keyword arguments (line 46)
    # Getting the type of 'cephes_src' (line 46)
    cephes_src_503296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 43), 'cephes_src', False)
    keyword_503297 = cephes_src_503296
    
    # Obtaining an instance of the builtin type 'list' (line 47)
    list_503298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 36), 'list')
    # Adding type elements to the builtin type 'list' instance (line 47)
    # Adding element type (line 47)
    # Getting the type of 'curdir' (line 47)
    curdir_503299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 37), 'curdir', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 36), list_503298, curdir_503299)
    
    # Getting the type of 'inc_dirs' (line 47)
    inc_dirs_503300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 47), 'inc_dirs', False)
    # Applying the binary operator '+' (line 47)
    result_add_503301 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 36), '+', list_503298, inc_dirs_503300)
    
    keyword_503302 = result_add_503301
    # Getting the type of 'cephes_hdr' (line 48)
    cephes_hdr_503303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 32), 'cephes_hdr', False)
    
    # Obtaining an instance of the builtin type 'list' (line 48)
    list_503304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 45), 'list')
    # Adding type elements to the builtin type 'list' instance (line 48)
    # Adding element type (line 48)
    str_503305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 46), 'str', '*.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 45), list_503304, str_503305)
    
    # Applying the binary operator '+' (line 48)
    result_add_503306 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 32), '+', cephes_hdr_503303, list_503304)
    
    keyword_503307 = result_add_503306
    # Getting the type of 'define_macros' (line 49)
    define_macros_503308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 30), 'define_macros', False)
    keyword_503309 = define_macros_503308
    kwargs_503310 = {'sources': keyword_503297, 'depends': keyword_503307, 'macros': keyword_503309, 'include_dirs': keyword_503302}
    # Getting the type of 'config' (line 46)
    config_503293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'config', False)
    # Obtaining the member 'add_library' of a type (line 46)
    add_library_503294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 4), config_503293, 'add_library')
    # Calling add_library(args, kwargs) (line 46)
    add_library_call_result_503311 = invoke(stypy.reporting.localization.Localization(__file__, 46, 4), add_library_503294, *[str_503295], **kwargs_503310)
    
    
    # Assigning a List to a Name (line 52):
    
    # Obtaining an instance of the builtin type 'list' (line 52)
    list_503312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 52)
    # Adding element type (line 52)
    
    # Call to join(...): (line 52)
    # Processing the call arguments (line 52)
    str_503314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 21), 'str', 'mach')
    str_503315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 28), 'str', '*.f')
    # Processing the call keyword arguments (line 52)
    kwargs_503316 = {}
    # Getting the type of 'join' (line 52)
    join_503313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'join', False)
    # Calling join(args, kwargs) (line 52)
    join_call_result_503317 = invoke(stypy.reporting.localization.Localization(__file__, 52, 16), join_503313, *[str_503314, str_503315], **kwargs_503316)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 15), list_503312, join_call_result_503317)
    
    # Assigning a type to the variable 'mach_src' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'mach_src', list_503312)
    
    # Assigning a List to a Name (line 53):
    
    # Obtaining an instance of the builtin type 'list' (line 53)
    list_503318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 53)
    # Adding element type (line 53)
    
    # Call to join(...): (line 53)
    # Processing the call arguments (line 53)
    str_503320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 21), 'str', 'amos')
    str_503321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 28), 'str', '*.f')
    # Processing the call keyword arguments (line 53)
    kwargs_503322 = {}
    # Getting the type of 'join' (line 53)
    join_503319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 16), 'join', False)
    # Calling join(args, kwargs) (line 53)
    join_call_result_503323 = invoke(stypy.reporting.localization.Localization(__file__, 53, 16), join_503319, *[str_503320, str_503321], **kwargs_503322)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 15), list_503318, join_call_result_503323)
    
    # Assigning a type to the variable 'amos_src' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'amos_src', list_503318)
    
    # Assigning a List to a Name (line 54):
    
    # Obtaining an instance of the builtin type 'list' (line 54)
    list_503324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 54)
    # Adding element type (line 54)
    
    # Call to join(...): (line 54)
    # Processing the call arguments (line 54)
    str_503326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 20), 'str', 'cdflib')
    str_503327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 29), 'str', '*.f')
    # Processing the call keyword arguments (line 54)
    kwargs_503328 = {}
    # Getting the type of 'join' (line 54)
    join_503325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 15), 'join', False)
    # Calling join(args, kwargs) (line 54)
    join_call_result_503329 = invoke(stypy.reporting.localization.Localization(__file__, 54, 15), join_503325, *[str_503326, str_503327], **kwargs_503328)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 14), list_503324, join_call_result_503329)
    
    # Assigning a type to the variable 'cdf_src' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'cdf_src', list_503324)
    
    # Assigning a List to a Name (line 55):
    
    # Obtaining an instance of the builtin type 'list' (line 55)
    list_503330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 55)
    # Adding element type (line 55)
    
    # Call to join(...): (line 55)
    # Processing the call arguments (line 55)
    str_503332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 24), 'str', 'specfun')
    str_503333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 34), 'str', '*.f')
    # Processing the call keyword arguments (line 55)
    kwargs_503334 = {}
    # Getting the type of 'join' (line 55)
    join_503331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 19), 'join', False)
    # Calling join(args, kwargs) (line 55)
    join_call_result_503335 = invoke(stypy.reporting.localization.Localization(__file__, 55, 19), join_503331, *[str_503332, str_503333], **kwargs_503334)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 18), list_503330, join_call_result_503335)
    
    # Assigning a type to the variable 'specfun_src' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'specfun_src', list_503330)
    
    # Call to add_library(...): (line 56)
    # Processing the call arguments (line 56)
    str_503338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 23), 'str', 'sc_mach')
    # Processing the call keyword arguments (line 56)
    # Getting the type of 'mach_src' (line 56)
    mach_src_503339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 41), 'mach_src', False)
    keyword_503340 = mach_src_503339
    
    # Obtaining an instance of the builtin type 'dict' (line 57)
    dict_503341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 33), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 57)
    # Adding element type (key, value) (line 57)
    str_503342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 34), 'str', 'noopt')
    
    # Obtaining an instance of the builtin type 'tuple' (line 57)
    tuple_503343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 57)
    # Adding element type (line 57)
    # Getting the type of '__file__' (line 57)
    file___503344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 43), '__file__', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 43), tuple_503343, file___503344)
    # Adding element type (line 57)
    int_503345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 52), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 43), tuple_503343, int_503345)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 33), dict_503341, (str_503342, tuple_503343))
    
    keyword_503346 = dict_503341
    kwargs_503347 = {'sources': keyword_503340, 'config_fc': keyword_503346}
    # Getting the type of 'config' (line 56)
    config_503336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'config', False)
    # Obtaining the member 'add_library' of a type (line 56)
    add_library_503337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 4), config_503336, 'add_library')
    # Calling add_library(args, kwargs) (line 56)
    add_library_call_result_503348 = invoke(stypy.reporting.localization.Localization(__file__, 56, 4), add_library_503337, *[str_503338], **kwargs_503347)
    
    
    # Call to add_library(...): (line 58)
    # Processing the call arguments (line 58)
    str_503351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 23), 'str', 'sc_amos')
    # Processing the call keyword arguments (line 58)
    # Getting the type of 'amos_src' (line 58)
    amos_src_503352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 41), 'amos_src', False)
    keyword_503353 = amos_src_503352
    kwargs_503354 = {'sources': keyword_503353}
    # Getting the type of 'config' (line 58)
    config_503349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'config', False)
    # Obtaining the member 'add_library' of a type (line 58)
    add_library_503350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 4), config_503349, 'add_library')
    # Calling add_library(args, kwargs) (line 58)
    add_library_call_result_503355 = invoke(stypy.reporting.localization.Localization(__file__, 58, 4), add_library_503350, *[str_503351], **kwargs_503354)
    
    
    # Call to add_library(...): (line 59)
    # Processing the call arguments (line 59)
    str_503358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 23), 'str', 'sc_cdf')
    # Processing the call keyword arguments (line 59)
    # Getting the type of 'cdf_src' (line 59)
    cdf_src_503359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 40), 'cdf_src', False)
    keyword_503360 = cdf_src_503359
    kwargs_503361 = {'sources': keyword_503360}
    # Getting the type of 'config' (line 59)
    config_503356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'config', False)
    # Obtaining the member 'add_library' of a type (line 59)
    add_library_503357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 4), config_503356, 'add_library')
    # Calling add_library(args, kwargs) (line 59)
    add_library_call_result_503362 = invoke(stypy.reporting.localization.Localization(__file__, 59, 4), add_library_503357, *[str_503358], **kwargs_503361)
    
    
    # Call to add_library(...): (line 60)
    # Processing the call arguments (line 60)
    str_503365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 23), 'str', 'sc_specfun')
    # Processing the call keyword arguments (line 60)
    # Getting the type of 'specfun_src' (line 60)
    specfun_src_503366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 44), 'specfun_src', False)
    keyword_503367 = specfun_src_503366
    kwargs_503368 = {'sources': keyword_503367}
    # Getting the type of 'config' (line 60)
    config_503363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'config', False)
    # Obtaining the member 'add_library' of a type (line 60)
    add_library_503364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 4), config_503363, 'add_library')
    # Calling add_library(args, kwargs) (line 60)
    add_library_call_result_503369 = invoke(stypy.reporting.localization.Localization(__file__, 60, 4), add_library_503364, *[str_503365], **kwargs_503368)
    
    
    # Call to add_extension(...): (line 63)
    # Processing the call arguments (line 63)
    str_503372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 25), 'str', 'specfun')
    # Processing the call keyword arguments (line 63)
    
    # Obtaining an instance of the builtin type 'list' (line 64)
    list_503373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 64)
    # Adding element type (line 64)
    str_503374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 34), 'str', 'specfun.pyf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 33), list_503373, str_503374)
    
    keyword_503375 = list_503373
    
    # Obtaining an instance of the builtin type 'list' (line 65)
    list_503376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 65)
    # Adding element type (line 65)
    str_503377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 39), 'str', '--no-wrap-functions')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 38), list_503376, str_503377)
    
    keyword_503378 = list_503376
    # Getting the type of 'specfun_src' (line 66)
    specfun_src_503379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 33), 'specfun_src', False)
    keyword_503380 = specfun_src_503379
    
    # Obtaining an instance of the builtin type 'list' (line 67)
    list_503381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 39), 'list')
    # Adding type elements to the builtin type 'list' instance (line 67)
    
    keyword_503382 = list_503381
    
    # Obtaining an instance of the builtin type 'list' (line 68)
    list_503383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 68)
    # Adding element type (line 68)
    str_503384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 36), 'str', 'sc_specfun')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 35), list_503383, str_503384)
    
    keyword_503385 = list_503383
    kwargs_503386 = {'libraries': keyword_503385, 'sources': keyword_503375, 'f2py_options': keyword_503378, 'depends': keyword_503380, 'define_macros': keyword_503382}
    # Getting the type of 'config' (line 63)
    config_503370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 63)
    add_extension_503371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 4), config_503370, 'add_extension')
    # Calling add_extension(args, kwargs) (line 63)
    add_extension_call_result_503387 = invoke(stypy.reporting.localization.Localization(__file__, 63, 4), add_extension_503371, *[str_503372], **kwargs_503386)
    
    
    # Assigning a List to a Name (line 71):
    
    # Obtaining an instance of the builtin type 'list' (line 71)
    list_503388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 71)
    # Adding element type (line 71)
    str_503389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 15), 'str', '*.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 14), list_503388, str_503389)
    # Adding element type (line 71)
    
    # Call to join(...): (line 71)
    # Processing the call arguments (line 71)
    str_503391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 27), 'str', 'c_misc')
    str_503392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 37), 'str', '*.h')
    # Processing the call keyword arguments (line 71)
    kwargs_503393 = {}
    # Getting the type of 'join' (line 71)
    join_503390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 22), 'join', False)
    # Calling join(args, kwargs) (line 71)
    join_call_result_503394 = invoke(stypy.reporting.localization.Localization(__file__, 71, 22), join_503390, *[str_503391, str_503392], **kwargs_503393)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 14), list_503388, join_call_result_503394)
    # Adding element type (line 71)
    
    # Call to join(...): (line 71)
    # Processing the call arguments (line 71)
    str_503396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 50), 'str', 'cephes')
    str_503397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 60), 'str', '*.h')
    # Processing the call keyword arguments (line 71)
    kwargs_503398 = {}
    # Getting the type of 'join' (line 71)
    join_503395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 45), 'join', False)
    # Calling join(args, kwargs) (line 71)
    join_call_result_503399 = invoke(stypy.reporting.localization.Localization(__file__, 71, 45), join_503395, *[str_503396, str_503397], **kwargs_503398)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 14), list_503388, join_call_result_503399)
    
    # Assigning a type to the variable 'headers' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'headers', list_503388)
    
    # Assigning a List to a Name (line 72):
    
    # Obtaining an instance of the builtin type 'list' (line 72)
    list_503400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 72)
    # Adding element type (line 72)
    str_503401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 18), 'str', '_ufuncs.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 17), list_503400, str_503401)
    # Adding element type (line 72)
    str_503402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 31), 'str', 'sf_error.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 17), list_503400, str_503402)
    # Adding element type (line 72)
    str_503403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 45), 'str', '_logit.c.src')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 17), list_503400, str_503403)
    # Adding element type (line 72)
    str_503404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 18), 'str', 'amos_wrappers.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 17), list_503400, str_503404)
    # Adding element type (line 72)
    str_503405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 37), 'str', 'cdf_wrappers.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 17), list_503400, str_503405)
    # Adding element type (line 72)
    str_503406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 55), 'str', 'specfun_wrappers.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 17), list_503400, str_503406)
    
    # Assigning a type to the variable 'ufuncs_src' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'ufuncs_src', list_503400)
    
    # Assigning a BinOp to a Name (line 74):
    # Getting the type of 'headers' (line 74)
    headers_503407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 18), 'headers')
    # Getting the type of 'ufuncs_src' (line 74)
    ufuncs_src_503408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 28), 'ufuncs_src')
    # Applying the binary operator '+' (line 74)
    result_add_503409 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 18), '+', headers_503407, ufuncs_src_503408)
    
    # Getting the type of 'amos_src' (line 74)
    amos_src_503410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 41), 'amos_src')
    # Applying the binary operator '+' (line 74)
    result_add_503411 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 39), '+', result_add_503409, amos_src_503410)
    
    # Getting the type of 'c_misc_src' (line 74)
    c_misc_src_503412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 52), 'c_misc_src')
    # Applying the binary operator '+' (line 74)
    result_add_503413 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 50), '+', result_add_503411, c_misc_src_503412)
    
    # Getting the type of 'cephes_src' (line 74)
    cephes_src_503414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 65), 'cephes_src')
    # Applying the binary operator '+' (line 74)
    result_add_503415 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 63), '+', result_add_503413, cephes_src_503414)
    
    # Getting the type of 'mach_src' (line 75)
    mach_src_503416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 20), 'mach_src')
    # Applying the binary operator '+' (line 75)
    result_add_503417 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 18), '+', result_add_503415, mach_src_503416)
    
    # Getting the type of 'cdf_src' (line 75)
    cdf_src_503418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 31), 'cdf_src')
    # Applying the binary operator '+' (line 75)
    result_add_503419 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 29), '+', result_add_503417, cdf_src_503418)
    
    # Getting the type of 'specfun_src' (line 75)
    specfun_src_503420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 41), 'specfun_src')
    # Applying the binary operator '+' (line 75)
    result_add_503421 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 39), '+', result_add_503419, specfun_src_503420)
    
    # Assigning a type to the variable 'ufuncs_dep' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'ufuncs_dep', result_add_503421)
    
    # Assigning a Call to a Name (line 76):
    
    # Call to dict(...): (line 76)
    # Processing the call arguments (line 76)
    
    # Call to get_system_info(...): (line 76)
    # Processing the call arguments (line 76)
    str_503424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 31), 'str', 'lapack_opt')
    # Processing the call keyword arguments (line 76)
    kwargs_503425 = {}
    # Getting the type of 'get_system_info' (line 76)
    get_system_info_503423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 15), 'get_system_info', False)
    # Calling get_system_info(args, kwargs) (line 76)
    get_system_info_call_result_503426 = invoke(stypy.reporting.localization.Localization(__file__, 76, 15), get_system_info_503423, *[str_503424], **kwargs_503425)
    
    # Processing the call keyword arguments (line 76)
    kwargs_503427 = {}
    # Getting the type of 'dict' (line 76)
    dict_503422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 10), 'dict', False)
    # Calling dict(args, kwargs) (line 76)
    dict_call_result_503428 = invoke(stypy.reporting.localization.Localization(__file__, 76, 10), dict_503422, *[get_system_info_call_result_503426], **kwargs_503427)
    
    # Assigning a type to the variable 'cfg' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'cfg', dict_call_result_503428)
    
    # Call to extend(...): (line 77)
    # Processing the call arguments (line 77)
    
    # Obtaining an instance of the builtin type 'list' (line 77)
    list_503436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 46), 'list')
    # Adding type elements to the builtin type 'list' instance (line 77)
    # Adding element type (line 77)
    # Getting the type of 'curdir' (line 77)
    curdir_503437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 47), 'curdir', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 46), list_503436, curdir_503437)
    
    # Getting the type of 'inc_dirs' (line 77)
    inc_dirs_503438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 57), 'inc_dirs', False)
    # Applying the binary operator '+' (line 77)
    result_add_503439 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 46), '+', list_503436, inc_dirs_503438)
    
    
    # Obtaining an instance of the builtin type 'list' (line 77)
    list_503440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 68), 'list')
    # Adding type elements to the builtin type 'list' instance (line 77)
    # Adding element type (line 77)
    
    # Call to get_include(...): (line 77)
    # Processing the call keyword arguments (line 77)
    kwargs_503443 = {}
    # Getting the type of 'numpy' (line 77)
    numpy_503441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 69), 'numpy', False)
    # Obtaining the member 'get_include' of a type (line 77)
    get_include_503442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 69), numpy_503441, 'get_include')
    # Calling get_include(args, kwargs) (line 77)
    get_include_call_result_503444 = invoke(stypy.reporting.localization.Localization(__file__, 77, 69), get_include_503442, *[], **kwargs_503443)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 68), list_503440, get_include_call_result_503444)
    
    # Applying the binary operator '+' (line 77)
    result_add_503445 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 66), '+', result_add_503439, list_503440)
    
    # Processing the call keyword arguments (line 77)
    kwargs_503446 = {}
    
    # Call to setdefault(...): (line 77)
    # Processing the call arguments (line 77)
    str_503431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 19), 'str', 'include_dirs')
    
    # Obtaining an instance of the builtin type 'list' (line 77)
    list_503432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 77)
    
    # Processing the call keyword arguments (line 77)
    kwargs_503433 = {}
    # Getting the type of 'cfg' (line 77)
    cfg_503429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'cfg', False)
    # Obtaining the member 'setdefault' of a type (line 77)
    setdefault_503430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 4), cfg_503429, 'setdefault')
    # Calling setdefault(args, kwargs) (line 77)
    setdefault_call_result_503434 = invoke(stypy.reporting.localization.Localization(__file__, 77, 4), setdefault_503430, *[str_503431, list_503432], **kwargs_503433)
    
    # Obtaining the member 'extend' of a type (line 77)
    extend_503435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 4), setdefault_call_result_503434, 'extend')
    # Calling extend(args, kwargs) (line 77)
    extend_call_result_503447 = invoke(stypy.reporting.localization.Localization(__file__, 77, 4), extend_503435, *[result_add_503445], **kwargs_503446)
    
    
    # Call to extend(...): (line 78)
    # Processing the call arguments (line 78)
    
    # Obtaining an instance of the builtin type 'list' (line 78)
    list_503455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 43), 'list')
    # Adding type elements to the builtin type 'list' instance (line 78)
    # Adding element type (line 78)
    str_503456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 44), 'str', 'sc_amos')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 43), list_503455, str_503456)
    # Adding element type (line 78)
    str_503457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 54), 'str', 'sc_c_misc')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 43), list_503455, str_503457)
    # Adding element type (line 78)
    str_503458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 66), 'str', 'sc_cephes')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 43), list_503455, str_503458)
    # Adding element type (line 78)
    str_503459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 78), 'str', 'sc_mach')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 43), list_503455, str_503459)
    # Adding element type (line 78)
    str_503460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 44), 'str', 'sc_cdf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 43), list_503455, str_503460)
    # Adding element type (line 78)
    str_503461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 54), 'str', 'sc_specfun')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 43), list_503455, str_503461)
    
    # Processing the call keyword arguments (line 78)
    kwargs_503462 = {}
    
    # Call to setdefault(...): (line 78)
    # Processing the call arguments (line 78)
    str_503450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 19), 'str', 'libraries')
    
    # Obtaining an instance of the builtin type 'list' (line 78)
    list_503451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 32), 'list')
    # Adding type elements to the builtin type 'list' instance (line 78)
    
    # Processing the call keyword arguments (line 78)
    kwargs_503452 = {}
    # Getting the type of 'cfg' (line 78)
    cfg_503448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'cfg', False)
    # Obtaining the member 'setdefault' of a type (line 78)
    setdefault_503449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 4), cfg_503448, 'setdefault')
    # Calling setdefault(args, kwargs) (line 78)
    setdefault_call_result_503453 = invoke(stypy.reporting.localization.Localization(__file__, 78, 4), setdefault_503449, *[str_503450, list_503451], **kwargs_503452)
    
    # Obtaining the member 'extend' of a type (line 78)
    extend_503454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 4), setdefault_call_result_503453, 'extend')
    # Calling extend(args, kwargs) (line 78)
    extend_call_result_503463 = invoke(stypy.reporting.localization.Localization(__file__, 78, 4), extend_503454, *[list_503455], **kwargs_503462)
    
    
    # Call to extend(...): (line 80)
    # Processing the call arguments (line 80)
    # Getting the type of 'define_macros' (line 80)
    define_macros_503471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 47), 'define_macros', False)
    # Processing the call keyword arguments (line 80)
    kwargs_503472 = {}
    
    # Call to setdefault(...): (line 80)
    # Processing the call arguments (line 80)
    str_503466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 19), 'str', 'define_macros')
    
    # Obtaining an instance of the builtin type 'list' (line 80)
    list_503467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 36), 'list')
    # Adding type elements to the builtin type 'list' instance (line 80)
    
    # Processing the call keyword arguments (line 80)
    kwargs_503468 = {}
    # Getting the type of 'cfg' (line 80)
    cfg_503464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'cfg', False)
    # Obtaining the member 'setdefault' of a type (line 80)
    setdefault_503465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 4), cfg_503464, 'setdefault')
    # Calling setdefault(args, kwargs) (line 80)
    setdefault_call_result_503469 = invoke(stypy.reporting.localization.Localization(__file__, 80, 4), setdefault_503465, *[str_503466, list_503467], **kwargs_503468)
    
    # Obtaining the member 'extend' of a type (line 80)
    extend_503470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 4), setdefault_call_result_503469, 'extend')
    # Calling extend(args, kwargs) (line 80)
    extend_call_result_503473 = invoke(stypy.reporting.localization.Localization(__file__, 80, 4), extend_503470, *[define_macros_503471], **kwargs_503472)
    
    
    # Call to add_extension(...): (line 81)
    # Processing the call arguments (line 81)
    str_503476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 25), 'str', '_ufuncs')
    # Processing the call keyword arguments (line 81)
    # Getting the type of 'ufuncs_dep' (line 82)
    ufuncs_dep_503477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 33), 'ufuncs_dep', False)
    keyword_503478 = ufuncs_dep_503477
    # Getting the type of 'ufuncs_src' (line 83)
    ufuncs_src_503479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 33), 'ufuncs_src', False)
    keyword_503480 = ufuncs_src_503479
    
    # Call to get_info(...): (line 84)
    # Processing the call arguments (line 84)
    str_503482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 45), 'str', 'npymath')
    # Processing the call keyword arguments (line 84)
    kwargs_503483 = {}
    # Getting the type of 'get_info' (line 84)
    get_info_503481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 36), 'get_info', False)
    # Calling get_info(args, kwargs) (line 84)
    get_info_call_result_503484 = invoke(stypy.reporting.localization.Localization(__file__, 84, 36), get_info_503481, *[str_503482], **kwargs_503483)
    
    keyword_503485 = get_info_call_result_503484
    # Getting the type of 'cfg' (line 85)
    cfg_503486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 27), 'cfg', False)
    kwargs_503487 = {'sources': keyword_503480, 'depends': keyword_503478, 'cfg_503486': cfg_503486, 'extra_info': keyword_503485}
    # Getting the type of 'config' (line 81)
    config_503474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 81)
    add_extension_503475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 4), config_503474, 'add_extension')
    # Calling add_extension(args, kwargs) (line 81)
    add_extension_call_result_503488 = invoke(stypy.reporting.localization.Localization(__file__, 81, 4), add_extension_503475, *[str_503476], **kwargs_503487)
    
    
    # Assigning a List to a Name (line 88):
    
    # Obtaining an instance of the builtin type 'list' (line 88)
    list_503489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 88)
    # Adding element type (line 88)
    str_503490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 22), 'str', '_ufuncs_cxx.cxx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 21), list_503489, str_503490)
    # Adding element type (line 88)
    str_503491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 41), 'str', 'sf_error.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 21), list_503489, str_503491)
    # Adding element type (line 88)
    str_503492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 22), 'str', '_faddeeva.cxx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 21), list_503489, str_503492)
    # Adding element type (line 88)
    str_503493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 39), 'str', 'Faddeeva.cc')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 21), list_503489, str_503493)
    # Adding element type (line 88)
    str_503494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 22), 'str', '_wright.cxx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 21), list_503489, str_503494)
    # Adding element type (line 88)
    str_503495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 37), 'str', 'wright.cc')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 21), list_503489, str_503495)
    
    # Assigning a type to the variable 'ufuncs_cxx_src' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'ufuncs_cxx_src', list_503489)
    
    # Assigning a BinOp to a Name (line 91):
    # Getting the type of 'headers' (line 91)
    headers_503496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 22), 'headers')
    # Getting the type of 'ufuncs_cxx_src' (line 91)
    ufuncs_cxx_src_503497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 32), 'ufuncs_cxx_src')
    # Applying the binary operator '+' (line 91)
    result_add_503498 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 22), '+', headers_503496, ufuncs_cxx_src_503497)
    
    # Getting the type of 'cephes_src' (line 91)
    cephes_src_503499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 49), 'cephes_src')
    # Applying the binary operator '+' (line 91)
    result_add_503500 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 47), '+', result_add_503498, cephes_src_503499)
    
    
    # Obtaining an instance of the builtin type 'list' (line 92)
    list_503501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 92)
    # Adding element type (line 92)
    str_503502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 25), 'str', '*.hh')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 24), list_503501, str_503502)
    
    # Applying the binary operator '+' (line 92)
    result_add_503503 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 22), '+', result_add_503500, list_503501)
    
    # Assigning a type to the variable 'ufuncs_cxx_dep' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'ufuncs_cxx_dep', result_add_503503)
    
    # Call to add_extension(...): (line 93)
    # Processing the call arguments (line 93)
    str_503506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 25), 'str', '_ufuncs_cxx')
    # Processing the call keyword arguments (line 93)
    # Getting the type of 'ufuncs_cxx_src' (line 94)
    ufuncs_cxx_src_503507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 33), 'ufuncs_cxx_src', False)
    keyword_503508 = ufuncs_cxx_src_503507
    # Getting the type of 'ufuncs_cxx_dep' (line 95)
    ufuncs_cxx_dep_503509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 33), 'ufuncs_cxx_dep', False)
    keyword_503510 = ufuncs_cxx_dep_503509
    
    # Obtaining an instance of the builtin type 'list' (line 96)
    list_503511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 96)
    # Adding element type (line 96)
    # Getting the type of 'curdir' (line 96)
    curdir_503512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 39), 'curdir', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 38), list_503511, curdir_503512)
    
    keyword_503513 = list_503511
    # Getting the type of 'define_macros' (line 97)
    define_macros_503514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 39), 'define_macros', False)
    keyword_503515 = define_macros_503514
    
    # Call to get_info(...): (line 98)
    # Processing the call arguments (line 98)
    str_503517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 45), 'str', 'npymath')
    # Processing the call keyword arguments (line 98)
    kwargs_503518 = {}
    # Getting the type of 'get_info' (line 98)
    get_info_503516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 36), 'get_info', False)
    # Calling get_info(args, kwargs) (line 98)
    get_info_call_result_503519 = invoke(stypy.reporting.localization.Localization(__file__, 98, 36), get_info_503516, *[str_503517], **kwargs_503518)
    
    keyword_503520 = get_info_call_result_503519
    kwargs_503521 = {'sources': keyword_503508, 'depends': keyword_503510, 'extra_info': keyword_503520, 'define_macros': keyword_503515, 'include_dirs': keyword_503513}
    # Getting the type of 'config' (line 93)
    config_503504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 93)
    add_extension_503505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 4), config_503504, 'add_extension')
    # Calling add_extension(args, kwargs) (line 93)
    add_extension_call_result_503522 = invoke(stypy.reporting.localization.Localization(__file__, 93, 4), add_extension_503505, *[str_503506], **kwargs_503521)
    
    
    # Assigning a Call to a Name (line 100):
    
    # Call to dict(...): (line 100)
    # Processing the call arguments (line 100)
    
    # Call to get_system_info(...): (line 100)
    # Processing the call arguments (line 100)
    str_503525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 31), 'str', 'lapack_opt')
    # Processing the call keyword arguments (line 100)
    kwargs_503526 = {}
    # Getting the type of 'get_system_info' (line 100)
    get_system_info_503524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 15), 'get_system_info', False)
    # Calling get_system_info(args, kwargs) (line 100)
    get_system_info_call_result_503527 = invoke(stypy.reporting.localization.Localization(__file__, 100, 15), get_system_info_503524, *[str_503525], **kwargs_503526)
    
    # Processing the call keyword arguments (line 100)
    kwargs_503528 = {}
    # Getting the type of 'dict' (line 100)
    dict_503523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 10), 'dict', False)
    # Calling dict(args, kwargs) (line 100)
    dict_call_result_503529 = invoke(stypy.reporting.localization.Localization(__file__, 100, 10), dict_503523, *[get_system_info_call_result_503527], **kwargs_503528)
    
    # Assigning a type to the variable 'cfg' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'cfg', dict_call_result_503529)
    
    # Call to add_extension(...): (line 101)
    # Processing the call arguments (line 101)
    str_503532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 25), 'str', '_ellip_harm_2')
    # Processing the call keyword arguments (line 101)
    
    # Obtaining an instance of the builtin type 'list' (line 102)
    list_503533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 102)
    # Adding element type (line 102)
    str_503534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 34), 'str', '_ellip_harm_2.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 33), list_503533, str_503534)
    # Adding element type (line 102)
    str_503535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 53), 'str', 'sf_error.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 33), list_503533, str_503535)
    
    keyword_503536 = list_503533
    # Getting the type of 'cfg' (line 103)
    cfg_503537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 27), 'cfg', False)
    kwargs_503538 = {'sources': keyword_503536, 'cfg_503537': cfg_503537}
    # Getting the type of 'config' (line 101)
    config_503530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 101)
    add_extension_503531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 4), config_503530, 'add_extension')
    # Calling add_extension(args, kwargs) (line 101)
    add_extension_call_result_503539 = invoke(stypy.reporting.localization.Localization(__file__, 101, 4), add_extension_503531, *[str_503532], **kwargs_503538)
    
    
    # Call to add_data_files(...): (line 107)
    # Processing the call arguments (line 107)
    str_503542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 26), 'str', 'cython_special.pxd')
    # Processing the call keyword arguments (line 107)
    kwargs_503543 = {}
    # Getting the type of 'config' (line 107)
    config_503540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'config', False)
    # Obtaining the member 'add_data_files' of a type (line 107)
    add_data_files_503541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 4), config_503540, 'add_data_files')
    # Calling add_data_files(args, kwargs) (line 107)
    add_data_files_call_result_503544 = invoke(stypy.reporting.localization.Localization(__file__, 107, 4), add_data_files_503541, *[str_503542], **kwargs_503543)
    
    
    # Assigning a List to a Name (line 109):
    
    # Obtaining an instance of the builtin type 'list' (line 109)
    list_503545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 25), 'list')
    # Adding type elements to the builtin type 'list' instance (line 109)
    # Adding element type (line 109)
    str_503546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 26), 'str', 'cython_special.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 25), list_503545, str_503546)
    # Adding element type (line 109)
    str_503547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 46), 'str', 'sf_error.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 25), list_503545, str_503547)
    # Adding element type (line 109)
    str_503548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 60), 'str', '_logit.c.src')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 25), list_503545, str_503548)
    # Adding element type (line 109)
    str_503549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 26), 'str', 'amos_wrappers.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 25), list_503545, str_503549)
    # Adding element type (line 109)
    str_503550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 45), 'str', 'cdf_wrappers.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 25), list_503545, str_503550)
    # Adding element type (line 109)
    str_503551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 63), 'str', 'specfun_wrappers.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 25), list_503545, str_503551)
    
    # Assigning a type to the variable 'cython_special_src' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'cython_special_src', list_503545)
    
    # Assigning a BinOp to a Name (line 111):
    # Getting the type of 'headers' (line 111)
    headers_503552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 26), 'headers')
    # Getting the type of 'ufuncs_src' (line 111)
    ufuncs_src_503553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 36), 'ufuncs_src')
    # Applying the binary operator '+' (line 111)
    result_add_503554 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 26), '+', headers_503552, ufuncs_src_503553)
    
    # Getting the type of 'ufuncs_cxx_src' (line 111)
    ufuncs_cxx_src_503555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 49), 'ufuncs_cxx_src')
    # Applying the binary operator '+' (line 111)
    result_add_503556 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 47), '+', result_add_503554, ufuncs_cxx_src_503555)
    
    # Getting the type of 'amos_src' (line 111)
    amos_src_503557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 66), 'amos_src')
    # Applying the binary operator '+' (line 111)
    result_add_503558 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 64), '+', result_add_503556, amos_src_503557)
    
    # Getting the type of 'c_misc_src' (line 112)
    c_misc_src_503559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 28), 'c_misc_src')
    # Applying the binary operator '+' (line 112)
    result_add_503560 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 26), '+', result_add_503558, c_misc_src_503559)
    
    # Getting the type of 'cephes_src' (line 112)
    cephes_src_503561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 41), 'cephes_src')
    # Applying the binary operator '+' (line 112)
    result_add_503562 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 39), '+', result_add_503560, cephes_src_503561)
    
    # Getting the type of 'mach_src' (line 112)
    mach_src_503563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 54), 'mach_src')
    # Applying the binary operator '+' (line 112)
    result_add_503564 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 52), '+', result_add_503562, mach_src_503563)
    
    # Getting the type of 'cdf_src' (line 112)
    cdf_src_503565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 65), 'cdf_src')
    # Applying the binary operator '+' (line 112)
    result_add_503566 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 63), '+', result_add_503564, cdf_src_503565)
    
    # Getting the type of 'specfun_src' (line 113)
    specfun_src_503567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 28), 'specfun_src')
    # Applying the binary operator '+' (line 113)
    result_add_503568 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 26), '+', result_add_503566, specfun_src_503567)
    
    # Assigning a type to the variable 'cython_special_dep' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'cython_special_dep', result_add_503568)
    
    # Assigning a Call to a Name (line 114):
    
    # Call to dict(...): (line 114)
    # Processing the call arguments (line 114)
    
    # Call to get_system_info(...): (line 114)
    # Processing the call arguments (line 114)
    str_503571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 31), 'str', 'lapack_opt')
    # Processing the call keyword arguments (line 114)
    kwargs_503572 = {}
    # Getting the type of 'get_system_info' (line 114)
    get_system_info_503570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 15), 'get_system_info', False)
    # Calling get_system_info(args, kwargs) (line 114)
    get_system_info_call_result_503573 = invoke(stypy.reporting.localization.Localization(__file__, 114, 15), get_system_info_503570, *[str_503571], **kwargs_503572)
    
    # Processing the call keyword arguments (line 114)
    kwargs_503574 = {}
    # Getting the type of 'dict' (line 114)
    dict_503569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 10), 'dict', False)
    # Calling dict(args, kwargs) (line 114)
    dict_call_result_503575 = invoke(stypy.reporting.localization.Localization(__file__, 114, 10), dict_503569, *[get_system_info_call_result_503573], **kwargs_503574)
    
    # Assigning a type to the variable 'cfg' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'cfg', dict_call_result_503575)
    
    # Call to extend(...): (line 115)
    # Processing the call arguments (line 115)
    
    # Obtaining an instance of the builtin type 'list' (line 115)
    list_503583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 46), 'list')
    # Adding type elements to the builtin type 'list' instance (line 115)
    # Adding element type (line 115)
    # Getting the type of 'curdir' (line 115)
    curdir_503584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 47), 'curdir', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 46), list_503583, curdir_503584)
    
    # Getting the type of 'inc_dirs' (line 115)
    inc_dirs_503585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 57), 'inc_dirs', False)
    # Applying the binary operator '+' (line 115)
    result_add_503586 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 46), '+', list_503583, inc_dirs_503585)
    
    
    # Obtaining an instance of the builtin type 'list' (line 115)
    list_503587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 68), 'list')
    # Adding type elements to the builtin type 'list' instance (line 115)
    # Adding element type (line 115)
    
    # Call to get_include(...): (line 115)
    # Processing the call keyword arguments (line 115)
    kwargs_503590 = {}
    # Getting the type of 'numpy' (line 115)
    numpy_503588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 69), 'numpy', False)
    # Obtaining the member 'get_include' of a type (line 115)
    get_include_503589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 69), numpy_503588, 'get_include')
    # Calling get_include(args, kwargs) (line 115)
    get_include_call_result_503591 = invoke(stypy.reporting.localization.Localization(__file__, 115, 69), get_include_503589, *[], **kwargs_503590)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 68), list_503587, get_include_call_result_503591)
    
    # Applying the binary operator '+' (line 115)
    result_add_503592 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 66), '+', result_add_503586, list_503587)
    
    # Processing the call keyword arguments (line 115)
    kwargs_503593 = {}
    
    # Call to setdefault(...): (line 115)
    # Processing the call arguments (line 115)
    str_503578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 19), 'str', 'include_dirs')
    
    # Obtaining an instance of the builtin type 'list' (line 115)
    list_503579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 115)
    
    # Processing the call keyword arguments (line 115)
    kwargs_503580 = {}
    # Getting the type of 'cfg' (line 115)
    cfg_503576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'cfg', False)
    # Obtaining the member 'setdefault' of a type (line 115)
    setdefault_503577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 4), cfg_503576, 'setdefault')
    # Calling setdefault(args, kwargs) (line 115)
    setdefault_call_result_503581 = invoke(stypy.reporting.localization.Localization(__file__, 115, 4), setdefault_503577, *[str_503578, list_503579], **kwargs_503580)
    
    # Obtaining the member 'extend' of a type (line 115)
    extend_503582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 4), setdefault_call_result_503581, 'extend')
    # Calling extend(args, kwargs) (line 115)
    extend_call_result_503594 = invoke(stypy.reporting.localization.Localization(__file__, 115, 4), extend_503582, *[result_add_503592], **kwargs_503593)
    
    
    # Call to extend(...): (line 116)
    # Processing the call arguments (line 116)
    
    # Obtaining an instance of the builtin type 'list' (line 116)
    list_503602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 43), 'list')
    # Adding type elements to the builtin type 'list' instance (line 116)
    # Adding element type (line 116)
    str_503603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 44), 'str', 'sc_amos')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 43), list_503602, str_503603)
    # Adding element type (line 116)
    str_503604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 54), 'str', 'sc_c_misc')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 43), list_503602, str_503604)
    # Adding element type (line 116)
    str_503605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 66), 'str', 'sc_cephes')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 43), list_503602, str_503605)
    # Adding element type (line 116)
    str_503606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 78), 'str', 'sc_mach')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 43), list_503602, str_503606)
    # Adding element type (line 116)
    str_503607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 44), 'str', 'sc_cdf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 43), list_503602, str_503607)
    # Adding element type (line 116)
    str_503608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 54), 'str', 'sc_specfun')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 43), list_503602, str_503608)
    
    # Processing the call keyword arguments (line 116)
    kwargs_503609 = {}
    
    # Call to setdefault(...): (line 116)
    # Processing the call arguments (line 116)
    str_503597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 19), 'str', 'libraries')
    
    # Obtaining an instance of the builtin type 'list' (line 116)
    list_503598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 32), 'list')
    # Adding type elements to the builtin type 'list' instance (line 116)
    
    # Processing the call keyword arguments (line 116)
    kwargs_503599 = {}
    # Getting the type of 'cfg' (line 116)
    cfg_503595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'cfg', False)
    # Obtaining the member 'setdefault' of a type (line 116)
    setdefault_503596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 4), cfg_503595, 'setdefault')
    # Calling setdefault(args, kwargs) (line 116)
    setdefault_call_result_503600 = invoke(stypy.reporting.localization.Localization(__file__, 116, 4), setdefault_503596, *[str_503597, list_503598], **kwargs_503599)
    
    # Obtaining the member 'extend' of a type (line 116)
    extend_503601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 4), setdefault_call_result_503600, 'extend')
    # Calling extend(args, kwargs) (line 116)
    extend_call_result_503610 = invoke(stypy.reporting.localization.Localization(__file__, 116, 4), extend_503601, *[list_503602], **kwargs_503609)
    
    
    # Call to extend(...): (line 118)
    # Processing the call arguments (line 118)
    # Getting the type of 'define_macros' (line 118)
    define_macros_503618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 47), 'define_macros', False)
    # Processing the call keyword arguments (line 118)
    kwargs_503619 = {}
    
    # Call to setdefault(...): (line 118)
    # Processing the call arguments (line 118)
    str_503613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 19), 'str', 'define_macros')
    
    # Obtaining an instance of the builtin type 'list' (line 118)
    list_503614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 36), 'list')
    # Adding type elements to the builtin type 'list' instance (line 118)
    
    # Processing the call keyword arguments (line 118)
    kwargs_503615 = {}
    # Getting the type of 'cfg' (line 118)
    cfg_503611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'cfg', False)
    # Obtaining the member 'setdefault' of a type (line 118)
    setdefault_503612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 4), cfg_503611, 'setdefault')
    # Calling setdefault(args, kwargs) (line 118)
    setdefault_call_result_503616 = invoke(stypy.reporting.localization.Localization(__file__, 118, 4), setdefault_503612, *[str_503613, list_503614], **kwargs_503615)
    
    # Obtaining the member 'extend' of a type (line 118)
    extend_503617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 4), setdefault_call_result_503616, 'extend')
    # Calling extend(args, kwargs) (line 118)
    extend_call_result_503620 = invoke(stypy.reporting.localization.Localization(__file__, 118, 4), extend_503617, *[define_macros_503618], **kwargs_503619)
    
    
    # Call to add_extension(...): (line 119)
    # Processing the call arguments (line 119)
    str_503623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 25), 'str', 'cython_special')
    # Processing the call keyword arguments (line 119)
    # Getting the type of 'cython_special_dep' (line 120)
    cython_special_dep_503624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 33), 'cython_special_dep', False)
    keyword_503625 = cython_special_dep_503624
    # Getting the type of 'cython_special_src' (line 121)
    cython_special_src_503626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 33), 'cython_special_src', False)
    keyword_503627 = cython_special_src_503626
    
    # Call to get_info(...): (line 122)
    # Processing the call arguments (line 122)
    str_503629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 45), 'str', 'npymath')
    # Processing the call keyword arguments (line 122)
    kwargs_503630 = {}
    # Getting the type of 'get_info' (line 122)
    get_info_503628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 36), 'get_info', False)
    # Calling get_info(args, kwargs) (line 122)
    get_info_call_result_503631 = invoke(stypy.reporting.localization.Localization(__file__, 122, 36), get_info_503628, *[str_503629], **kwargs_503630)
    
    keyword_503632 = get_info_call_result_503631
    # Getting the type of 'cfg' (line 123)
    cfg_503633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 27), 'cfg', False)
    kwargs_503634 = {'sources': keyword_503627, 'depends': keyword_503625, 'cfg_503633': cfg_503633, 'extra_info': keyword_503632}
    # Getting the type of 'config' (line 119)
    config_503621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 119)
    add_extension_503622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 4), config_503621, 'add_extension')
    # Calling add_extension(args, kwargs) (line 119)
    add_extension_call_result_503635 = invoke(stypy.reporting.localization.Localization(__file__, 119, 4), add_extension_503622, *[str_503623], **kwargs_503634)
    
    
    # Call to add_extension(...): (line 126)
    # Processing the call arguments (line 126)
    str_503638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 25), 'str', '_comb')
    # Processing the call keyword arguments (line 126)
    
    # Obtaining an instance of the builtin type 'list' (line 127)
    list_503639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 127)
    # Adding element type (line 127)
    str_503640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 34), 'str', '_comb.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 33), list_503639, str_503640)
    
    keyword_503641 = list_503639
    kwargs_503642 = {'sources': keyword_503641}
    # Getting the type of 'config' (line 126)
    config_503636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 126)
    add_extension_503637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 4), config_503636, 'add_extension')
    # Calling add_extension(args, kwargs) (line 126)
    add_extension_call_result_503643 = invoke(stypy.reporting.localization.Localization(__file__, 126, 4), add_extension_503637, *[str_503638], **kwargs_503642)
    
    
    # Call to add_extension(...): (line 130)
    # Processing the call arguments (line 130)
    str_503646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 25), 'str', '_test_round')
    # Processing the call keyword arguments (line 130)
    
    # Obtaining an instance of the builtin type 'list' (line 131)
    list_503647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 131)
    # Adding element type (line 131)
    str_503648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 34), 'str', '_test_round.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 131, 33), list_503647, str_503648)
    
    keyword_503649 = list_503647
    
    # Obtaining an instance of the builtin type 'list' (line 132)
    list_503650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 132)
    # Adding element type (line 132)
    str_503651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 34), 'str', '_round.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 33), list_503650, str_503651)
    # Adding element type (line 132)
    str_503652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 46), 'str', 'c_misc/double2.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 33), list_503650, str_503652)
    
    keyword_503653 = list_503650
    
    # Obtaining an instance of the builtin type 'list' (line 133)
    list_503654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 133)
    # Adding element type (line 133)
    
    # Call to get_include(...): (line 133)
    # Processing the call keyword arguments (line 133)
    kwargs_503657 = {}
    # Getting the type of 'numpy' (line 133)
    numpy_503655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 39), 'numpy', False)
    # Obtaining the member 'get_include' of a type (line 133)
    get_include_503656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 39), numpy_503655, 'get_include')
    # Calling get_include(args, kwargs) (line 133)
    get_include_call_result_503658 = invoke(stypy.reporting.localization.Localization(__file__, 133, 39), get_include_503656, *[], **kwargs_503657)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 38), list_503654, get_include_call_result_503658)
    
    keyword_503659 = list_503654
    
    # Call to get_info(...): (line 134)
    # Processing the call arguments (line 134)
    str_503661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 45), 'str', 'npymath')
    # Processing the call keyword arguments (line 134)
    kwargs_503662 = {}
    # Getting the type of 'get_info' (line 134)
    get_info_503660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 36), 'get_info', False)
    # Calling get_info(args, kwargs) (line 134)
    get_info_call_result_503663 = invoke(stypy.reporting.localization.Localization(__file__, 134, 36), get_info_503660, *[str_503661], **kwargs_503662)
    
    keyword_503664 = get_info_call_result_503663
    kwargs_503665 = {'sources': keyword_503649, 'depends': keyword_503653, 'extra_info': keyword_503664, 'include_dirs': keyword_503659}
    # Getting the type of 'config' (line 130)
    config_503644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 130)
    add_extension_503645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 4), config_503644, 'add_extension')
    # Calling add_extension(args, kwargs) (line 130)
    add_extension_call_result_503666 = invoke(stypy.reporting.localization.Localization(__file__, 130, 4), add_extension_503645, *[str_503646], **kwargs_503665)
    
    
    # Call to add_data_files(...): (line 136)
    # Processing the call arguments (line 136)
    str_503669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 26), 'str', 'tests/*.py')
    # Processing the call keyword arguments (line 136)
    kwargs_503670 = {}
    # Getting the type of 'config' (line 136)
    config_503667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'config', False)
    # Obtaining the member 'add_data_files' of a type (line 136)
    add_data_files_503668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 4), config_503667, 'add_data_files')
    # Calling add_data_files(args, kwargs) (line 136)
    add_data_files_call_result_503671 = invoke(stypy.reporting.localization.Localization(__file__, 136, 4), add_data_files_503668, *[str_503669], **kwargs_503670)
    
    
    # Call to add_data_files(...): (line 137)
    # Processing the call arguments (line 137)
    str_503674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 26), 'str', 'tests/data/README')
    # Processing the call keyword arguments (line 137)
    kwargs_503675 = {}
    # Getting the type of 'config' (line 137)
    config_503672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'config', False)
    # Obtaining the member 'add_data_files' of a type (line 137)
    add_data_files_503673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 4), config_503672, 'add_data_files')
    # Calling add_data_files(args, kwargs) (line 137)
    add_data_files_call_result_503676 = invoke(stypy.reporting.localization.Localization(__file__, 137, 4), add_data_files_503673, *[str_503674], **kwargs_503675)
    
    
    # Call to add_data_files(...): (line 138)
    # Processing the call arguments (line 138)
    str_503679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 26), 'str', 'tests/data/*.npz')
    # Processing the call keyword arguments (line 138)
    kwargs_503680 = {}
    # Getting the type of 'config' (line 138)
    config_503677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'config', False)
    # Obtaining the member 'add_data_files' of a type (line 138)
    add_data_files_503678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 4), config_503677, 'add_data_files')
    # Calling add_data_files(args, kwargs) (line 138)
    add_data_files_call_result_503681 = invoke(stypy.reporting.localization.Localization(__file__, 138, 4), add_data_files_503678, *[str_503679], **kwargs_503680)
    
    
    # Call to add_subpackage(...): (line 140)
    # Processing the call arguments (line 140)
    str_503684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 26), 'str', '_precompute')
    # Processing the call keyword arguments (line 140)
    kwargs_503685 = {}
    # Getting the type of 'config' (line 140)
    config_503682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'config', False)
    # Obtaining the member 'add_subpackage' of a type (line 140)
    add_subpackage_503683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 4), config_503682, 'add_subpackage')
    # Calling add_subpackage(args, kwargs) (line 140)
    add_subpackage_call_result_503686 = invoke(stypy.reporting.localization.Localization(__file__, 140, 4), add_subpackage_503683, *[str_503684], **kwargs_503685)
    
    # Getting the type of 'config' (line 142)
    config_503687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'stypy_return_type', config_503687)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 17)
    stypy_return_type_503688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_503688)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_503688

# Assigning a type to the variable 'configuration' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 146, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 146)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/')
    import_503689 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 146, 4), 'numpy.distutils.core')

    if (type(import_503689) is not StypyTypeError):

        if (import_503689 != 'pyd_module'):
            __import__(import_503689)
            sys_modules_503690 = sys.modules[import_503689]
            import_from_module(stypy.reporting.localization.Localization(__file__, 146, 4), 'numpy.distutils.core', sys_modules_503690.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 146, 4), __file__, sys_modules_503690, sys_modules_503690.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 146, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'numpy.distutils.core', import_503689)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/')
    
    
    # Call to setup(...): (line 147)
    # Processing the call keyword arguments (line 147)
    
    # Call to todict(...): (line 147)
    # Processing the call keyword arguments (line 147)
    kwargs_503698 = {}
    
    # Call to configuration(...): (line 147)
    # Processing the call keyword arguments (line 147)
    str_503693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 35), 'str', '')
    keyword_503694 = str_503693
    kwargs_503695 = {'top_path': keyword_503694}
    # Getting the type of 'configuration' (line 147)
    configuration_503692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'configuration', False)
    # Calling configuration(args, kwargs) (line 147)
    configuration_call_result_503696 = invoke(stypy.reporting.localization.Localization(__file__, 147, 12), configuration_503692, *[], **kwargs_503695)
    
    # Obtaining the member 'todict' of a type (line 147)
    todict_503697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 12), configuration_call_result_503696, 'todict')
    # Calling todict(args, kwargs) (line 147)
    todict_call_result_503699 = invoke(stypy.reporting.localization.Localization(__file__, 147, 12), todict_503697, *[], **kwargs_503698)
    
    kwargs_503700 = {'todict_call_result_503699': todict_call_result_503699}
    # Getting the type of 'setup' (line 147)
    setup_503691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 147)
    setup_call_result_503701 = invoke(stypy.reporting.localization.Localization(__file__, 147, 4), setup_503691, *[], **kwargs_503700)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
