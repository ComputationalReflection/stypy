
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import os
4: from os.path import join
5: 
6: 
7: def configuration(parent_package='', top_path=None):
8:     from distutils.sysconfig import get_python_inc
9:     from numpy.distutils.system_info import get_info, NotFoundError, numpy_info
10:     from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs
11:     from scipy._build_utils import (get_sgemv_fix, get_g77_abi_wrappers,
12:                                     split_fortran_files)
13: 
14:     config = Configuration('linalg', parent_package, top_path)
15: 
16:     lapack_opt = get_info('lapack_opt')
17: 
18:     if not lapack_opt:
19:         raise NotFoundError('no lapack/blas resources found')
20: 
21:     atlas_version = ([v[3:-3] for k, v in lapack_opt.get('define_macros', [])
22:                       if k == 'ATLAS_INFO']+[None])[0]
23:     if atlas_version:
24:         print(('ATLAS version: %s' % atlas_version))
25: 
26:     # fblas:
27:     sources = ['fblas.pyf.src']
28:     sources += get_g77_abi_wrappers(lapack_opt)
29:     sources += get_sgemv_fix(lapack_opt)
30: 
31:     config.add_extension('_fblas',
32:                          sources=sources,
33:                          depends=['fblas_l?.pyf.src'],
34:                          extra_info=lapack_opt
35:                          )
36: 
37:     # flapack:
38:     sources = ['flapack.pyf.src']
39:     sources += get_g77_abi_wrappers(lapack_opt)
40:     dep_pfx = join('src', 'lapack_deprecations')
41:     deprecated_lapack_routines = [join(dep_pfx, c + 'gegv.f') for c in 'cdsz']
42:     sources += deprecated_lapack_routines
43: 
44:     config.add_extension('_flapack',
45:                          sources=sources,
46:                          depends=['flapack_user.pyf.src'],
47:                          extra_info=lapack_opt
48:                          )
49: 
50:     if atlas_version is not None:
51:         # cblas:
52:         config.add_extension('_cblas',
53:                              sources=['cblas.pyf.src'],
54:                              depends=['cblas.pyf.src', 'cblas_l1.pyf.src'],
55:                              extra_info=lapack_opt
56:                              )
57: 
58:         # clapack:
59:         config.add_extension('_clapack',
60:                              sources=['clapack.pyf.src'],
61:                              depends=['clapack.pyf.src'],
62:                              extra_info=lapack_opt
63:                              )
64: 
65:     # _flinalg:
66:     config.add_extension('_flinalg',
67:                          sources=[join('src', 'det.f'), join('src', 'lu.f')],
68:                          extra_info=lapack_opt
69:                          )
70: 
71:     # _interpolative:
72:     routines_to_split = [
73:         'dfftb1',
74:         'dfftf1',
75:         'dffti1',
76:         'dsint1',
77:         'dzfft1',
78:         'id_srand',
79:         'idd_copyints',
80:         'idd_id2svd0',
81:         'idd_pairsamps',
82:         'idd_permute',
83:         'idd_permuter',
84:         'idd_random_transf0',
85:         'idd_random_transf0_inv',
86:         'idd_random_transf_init0',
87:         'idd_subselect',
88:         'iddp_asvd0',
89:         'iddp_rsvd0',
90:         'iddr_asvd0',
91:         'iddr_rsvd0',
92:         'idz_estrank0',
93:         'idz_id2svd0',
94:         'idz_permute',
95:         'idz_permuter',
96:         'idz_random_transf0_inv',
97:         'idz_random_transf_init0',
98:         'idz_random_transf_init00',
99:         'idz_realcomp',
100:         'idz_realcomplex',
101:         'idz_reco',
102:         'idz_subselect',
103:         'idzp_aid0',
104:         'idzp_aid1',
105:         'idzp_asvd0',
106:         'idzp_rsvd0',
107:         'idzr_asvd0',
108:         'idzr_reco',
109:         'idzr_rsvd0',
110:         'zfftb1',
111:         'zfftf1',
112:         'zffti1',
113:     ]
114:     print('Splitting linalg.interpolative Fortran source files')
115:     dirname = os.path.split(os.path.abspath(__file__))[0]
116:     fnames = split_fortran_files(join(dirname, 'src', 'id_dist', 'src'),
117:                                  routines_to_split)
118:     fnames = [join('src', 'id_dist', 'src', f) for f in fnames]
119:     config.add_extension('_interpolative', fnames + ["interpolative.pyf"],
120:                          extra_info=lapack_opt
121:                          )
122: 
123:     # _solve_toeplitz:
124:     config.add_extension('_solve_toeplitz',
125:                          sources=[('_solve_toeplitz.c')],
126:                          include_dirs=[get_numpy_include_dirs()])
127: 
128:     config.add_data_dir('tests')
129: 
130:     # Cython BLAS/LAPACK
131:     config.add_data_files('cython_blas.pxd')
132:     config.add_data_files('cython_lapack.pxd')
133: 
134:     sources = ['_blas_subroutine_wrappers.f', '_lapack_subroutine_wrappers.f']
135:     sources += get_g77_abi_wrappers(lapack_opt)
136:     sources += get_sgemv_fix(lapack_opt)
137:     includes = numpy_info().get_include_dirs() + [get_python_inc()]
138:     config.add_library('fwrappers', sources=sources, include_dirs=includes)
139: 
140:     config.add_extension('cython_blas',
141:                          sources=['cython_blas.c'],
142:                          depends=['cython_blas.pyx', 'cython_blas.pxd',
143:                                   'fortran_defs.h', '_blas_subroutines.h'],
144:                          include_dirs=['.'],
145:                          libraries=['fwrappers'],
146:                          extra_info=lapack_opt)
147: 
148:     config.add_extension('cython_lapack',
149:                          sources=['cython_lapack.c'],
150:                          depends=['cython_lapack.pyx', 'cython_lapack.pxd',
151:                                   'fortran_defs.h', '_lapack_subroutines.h'],
152:                          include_dirs=['.'],
153:                          libraries=['fwrappers'],
154:                          extra_info=lapack_opt)
155: 
156:     config.add_extension('_decomp_update',
157:                          sources=['_decomp_update.c'])
158: 
159:     # Add any license files
160:     config.add_data_files('src/id_dist/doc/doc.tex')
161:     config.add_data_files('src/lapack_deprecations/LICENSE')
162: 
163:     return config
164: 
165: 
166: if __name__ == '__main__':
167:     from numpy.distutils.core import setup
168:     from linalg_version import linalg_version
169: 
170:     setup(version=linalg_version,
171:           **configuration(top_path='').todict())
172: 

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
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
import_23621 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'os.path')

if (type(import_23621) is not StypyTypeError):

    if (import_23621 != 'pyd_module'):
        __import__(import_23621)
        sys_modules_23622 = sys.modules[import_23621]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'os.path', sys_modules_23622.module_type_store, module_type_store, ['join'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_23622, sys_modules_23622.module_type_store, module_type_store)
    else:
        from os.path import join

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'os.path', None, module_type_store, ['join'], [join])

else:
    # Assigning a type to the variable 'os.path' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'os.path', import_23621)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')


@norecursion
def configuration(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_23623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 33), 'str', '')
    # Getting the type of 'None' (line 7)
    None_23624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 46), 'None')
    defaults = [str_23623, None_23624]
    # Create a new context for function 'configuration'
    module_type_store = module_type_store.open_function_context('configuration', 7, 0, False)
    
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

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 4))
    
    # 'from distutils.sysconfig import get_python_inc' statement (line 8)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
    import_23625 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'distutils.sysconfig')

    if (type(import_23625) is not StypyTypeError):

        if (import_23625 != 'pyd_module'):
            __import__(import_23625)
            sys_modules_23626 = sys.modules[import_23625]
            import_from_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'distutils.sysconfig', sys_modules_23626.module_type_store, module_type_store, ['get_python_inc'])
            nest_module(stypy.reporting.localization.Localization(__file__, 8, 4), __file__, sys_modules_23626, sys_modules_23626.module_type_store, module_type_store)
        else:
            from distutils.sysconfig import get_python_inc

            import_from_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'distutils.sysconfig', None, module_type_store, ['get_python_inc'], [get_python_inc])

    else:
        # Assigning a type to the variable 'distutils.sysconfig' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'distutils.sysconfig', import_23625)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 4))
    
    # 'from numpy.distutils.system_info import get_info, NotFoundError, numpy_info' statement (line 9)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
    import_23627 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'numpy.distutils.system_info')

    if (type(import_23627) is not StypyTypeError):

        if (import_23627 != 'pyd_module'):
            __import__(import_23627)
            sys_modules_23628 = sys.modules[import_23627]
            import_from_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'numpy.distutils.system_info', sys_modules_23628.module_type_store, module_type_store, ['get_info', 'NotFoundError', 'numpy_info'])
            nest_module(stypy.reporting.localization.Localization(__file__, 9, 4), __file__, sys_modules_23628, sys_modules_23628.module_type_store, module_type_store)
        else:
            from numpy.distutils.system_info import get_info, NotFoundError, numpy_info

            import_from_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'numpy.distutils.system_info', None, module_type_store, ['get_info', 'NotFoundError', 'numpy_info'], [get_info, NotFoundError, numpy_info])

    else:
        # Assigning a type to the variable 'numpy.distutils.system_info' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'numpy.distutils.system_info', import_23627)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 4))
    
    # 'from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs' statement (line 10)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
    import_23629 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'numpy.distutils.misc_util')

    if (type(import_23629) is not StypyTypeError):

        if (import_23629 != 'pyd_module'):
            __import__(import_23629)
            sys_modules_23630 = sys.modules[import_23629]
            import_from_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'numpy.distutils.misc_util', sys_modules_23630.module_type_store, module_type_store, ['Configuration', 'get_numpy_include_dirs'])
            nest_module(stypy.reporting.localization.Localization(__file__, 10, 4), __file__, sys_modules_23630, sys_modules_23630.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration, get_numpy_include_dirs

            import_from_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration', 'get_numpy_include_dirs'], [Configuration, get_numpy_include_dirs])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'numpy.distutils.misc_util', import_23629)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 4))
    
    # 'from scipy._build_utils import get_sgemv_fix, get_g77_abi_wrappers, split_fortran_files' statement (line 11)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
    import_23631 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'scipy._build_utils')

    if (type(import_23631) is not StypyTypeError):

        if (import_23631 != 'pyd_module'):
            __import__(import_23631)
            sys_modules_23632 = sys.modules[import_23631]
            import_from_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'scipy._build_utils', sys_modules_23632.module_type_store, module_type_store, ['get_sgemv_fix', 'get_g77_abi_wrappers', 'split_fortran_files'])
            nest_module(stypy.reporting.localization.Localization(__file__, 11, 4), __file__, sys_modules_23632, sys_modules_23632.module_type_store, module_type_store)
        else:
            from scipy._build_utils import get_sgemv_fix, get_g77_abi_wrappers, split_fortran_files

            import_from_module(stypy.reporting.localization.Localization(__file__, 11, 4), 'scipy._build_utils', None, module_type_store, ['get_sgemv_fix', 'get_g77_abi_wrappers', 'split_fortran_files'], [get_sgemv_fix, get_g77_abi_wrappers, split_fortran_files])

    else:
        # Assigning a type to the variable 'scipy._build_utils' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'scipy._build_utils', import_23631)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')
    
    
    # Assigning a Call to a Name (line 14):
    
    # Call to Configuration(...): (line 14)
    # Processing the call arguments (line 14)
    str_23634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 27), 'str', 'linalg')
    # Getting the type of 'parent_package' (line 14)
    parent_package_23635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 37), 'parent_package', False)
    # Getting the type of 'top_path' (line 14)
    top_path_23636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 53), 'top_path', False)
    # Processing the call keyword arguments (line 14)
    kwargs_23637 = {}
    # Getting the type of 'Configuration' (line 14)
    Configuration_23633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 14)
    Configuration_call_result_23638 = invoke(stypy.reporting.localization.Localization(__file__, 14, 13), Configuration_23633, *[str_23634, parent_package_23635, top_path_23636], **kwargs_23637)
    
    # Assigning a type to the variable 'config' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'config', Configuration_call_result_23638)
    
    # Assigning a Call to a Name (line 16):
    
    # Call to get_info(...): (line 16)
    # Processing the call arguments (line 16)
    str_23640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 26), 'str', 'lapack_opt')
    # Processing the call keyword arguments (line 16)
    kwargs_23641 = {}
    # Getting the type of 'get_info' (line 16)
    get_info_23639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 17), 'get_info', False)
    # Calling get_info(args, kwargs) (line 16)
    get_info_call_result_23642 = invoke(stypy.reporting.localization.Localization(__file__, 16, 17), get_info_23639, *[str_23640], **kwargs_23641)
    
    # Assigning a type to the variable 'lapack_opt' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'lapack_opt', get_info_call_result_23642)
    
    
    # Getting the type of 'lapack_opt' (line 18)
    lapack_opt_23643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 11), 'lapack_opt')
    # Applying the 'not' unary operator (line 18)
    result_not__23644 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 7), 'not', lapack_opt_23643)
    
    # Testing the type of an if condition (line 18)
    if_condition_23645 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 18, 4), result_not__23644)
    # Assigning a type to the variable 'if_condition_23645' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'if_condition_23645', if_condition_23645)
    # SSA begins for if statement (line 18)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to NotFoundError(...): (line 19)
    # Processing the call arguments (line 19)
    str_23647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 28), 'str', 'no lapack/blas resources found')
    # Processing the call keyword arguments (line 19)
    kwargs_23648 = {}
    # Getting the type of 'NotFoundError' (line 19)
    NotFoundError_23646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 14), 'NotFoundError', False)
    # Calling NotFoundError(args, kwargs) (line 19)
    NotFoundError_call_result_23649 = invoke(stypy.reporting.localization.Localization(__file__, 19, 14), NotFoundError_23646, *[str_23647], **kwargs_23648)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 19, 8), NotFoundError_call_result_23649, 'raise parameter', BaseException)
    # SSA join for if statement (line 18)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 21):
    
    # Obtaining the type of the subscript
    int_23650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 52), 'int')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to get(...): (line 21)
    # Processing the call arguments (line 21)
    str_23662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 57), 'str', 'define_macros')
    
    # Obtaining an instance of the builtin type 'list' (line 21)
    list_23663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 74), 'list')
    # Adding type elements to the builtin type 'list' instance (line 21)
    
    # Processing the call keyword arguments (line 21)
    kwargs_23664 = {}
    # Getting the type of 'lapack_opt' (line 21)
    lapack_opt_23660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 42), 'lapack_opt', False)
    # Obtaining the member 'get' of a type (line 21)
    get_23661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 42), lapack_opt_23660, 'get')
    # Calling get(args, kwargs) (line 21)
    get_call_result_23665 = invoke(stypy.reporting.localization.Localization(__file__, 21, 42), get_23661, *[str_23662, list_23663], **kwargs_23664)
    
    comprehension_23666 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 22), get_call_result_23665)
    # Assigning a type to the variable 'k' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 22), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 22), comprehension_23666))
    # Assigning a type to the variable 'v' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 22), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 22), comprehension_23666))
    
    # Getting the type of 'k' (line 22)
    k_23657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 25), 'k')
    str_23658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 30), 'str', 'ATLAS_INFO')
    # Applying the binary operator '==' (line 22)
    result_eq_23659 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 25), '==', k_23657, str_23658)
    
    
    # Obtaining the type of the subscript
    int_23651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 24), 'int')
    int_23652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 26), 'int')
    slice_23653 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 21, 22), int_23651, int_23652, None)
    # Getting the type of 'v' (line 21)
    v_23654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 22), 'v')
    # Obtaining the member '__getitem__' of a type (line 21)
    getitem___23655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 22), v_23654, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 21)
    subscript_call_result_23656 = invoke(stypy.reporting.localization.Localization(__file__, 21, 22), getitem___23655, slice_23653)
    
    list_23667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 22), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 22), list_23667, subscript_call_result_23656)
    
    # Obtaining an instance of the builtin type 'list' (line 22)
    list_23668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 44), 'list')
    # Adding type elements to the builtin type 'list' instance (line 22)
    # Adding element type (line 22)
    # Getting the type of 'None' (line 22)
    None_23669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 45), 'None')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 44), list_23668, None_23669)
    
    # Applying the binary operator '+' (line 21)
    result_add_23670 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 21), '+', list_23667, list_23668)
    
    # Obtaining the member '__getitem__' of a type (line 21)
    getitem___23671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 21), result_add_23670, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 21)
    subscript_call_result_23672 = invoke(stypy.reporting.localization.Localization(__file__, 21, 21), getitem___23671, int_23650)
    
    # Assigning a type to the variable 'atlas_version' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'atlas_version', subscript_call_result_23672)
    
    # Getting the type of 'atlas_version' (line 23)
    atlas_version_23673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 7), 'atlas_version')
    # Testing the type of an if condition (line 23)
    if_condition_23674 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 23, 4), atlas_version_23673)
    # Assigning a type to the variable 'if_condition_23674' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'if_condition_23674', if_condition_23674)
    # SSA begins for if statement (line 23)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 24)
    # Processing the call arguments (line 24)
    str_23676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 15), 'str', 'ATLAS version: %s')
    # Getting the type of 'atlas_version' (line 24)
    atlas_version_23677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 37), 'atlas_version', False)
    # Applying the binary operator '%' (line 24)
    result_mod_23678 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 15), '%', str_23676, atlas_version_23677)
    
    # Processing the call keyword arguments (line 24)
    kwargs_23679 = {}
    # Getting the type of 'print' (line 24)
    print_23675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'print', False)
    # Calling print(args, kwargs) (line 24)
    print_call_result_23680 = invoke(stypy.reporting.localization.Localization(__file__, 24, 8), print_23675, *[result_mod_23678], **kwargs_23679)
    
    # SSA join for if statement (line 23)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 27):
    
    # Obtaining an instance of the builtin type 'list' (line 27)
    list_23681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 27)
    # Adding element type (line 27)
    str_23682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 15), 'str', 'fblas.pyf.src')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 14), list_23681, str_23682)
    
    # Assigning a type to the variable 'sources' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'sources', list_23681)
    
    # Getting the type of 'sources' (line 28)
    sources_23683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'sources')
    
    # Call to get_g77_abi_wrappers(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'lapack_opt' (line 28)
    lapack_opt_23685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 36), 'lapack_opt', False)
    # Processing the call keyword arguments (line 28)
    kwargs_23686 = {}
    # Getting the type of 'get_g77_abi_wrappers' (line 28)
    get_g77_abi_wrappers_23684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 15), 'get_g77_abi_wrappers', False)
    # Calling get_g77_abi_wrappers(args, kwargs) (line 28)
    get_g77_abi_wrappers_call_result_23687 = invoke(stypy.reporting.localization.Localization(__file__, 28, 15), get_g77_abi_wrappers_23684, *[lapack_opt_23685], **kwargs_23686)
    
    # Applying the binary operator '+=' (line 28)
    result_iadd_23688 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 4), '+=', sources_23683, get_g77_abi_wrappers_call_result_23687)
    # Assigning a type to the variable 'sources' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'sources', result_iadd_23688)
    
    
    # Getting the type of 'sources' (line 29)
    sources_23689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'sources')
    
    # Call to get_sgemv_fix(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of 'lapack_opt' (line 29)
    lapack_opt_23691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 29), 'lapack_opt', False)
    # Processing the call keyword arguments (line 29)
    kwargs_23692 = {}
    # Getting the type of 'get_sgemv_fix' (line 29)
    get_sgemv_fix_23690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 15), 'get_sgemv_fix', False)
    # Calling get_sgemv_fix(args, kwargs) (line 29)
    get_sgemv_fix_call_result_23693 = invoke(stypy.reporting.localization.Localization(__file__, 29, 15), get_sgemv_fix_23690, *[lapack_opt_23691], **kwargs_23692)
    
    # Applying the binary operator '+=' (line 29)
    result_iadd_23694 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 4), '+=', sources_23689, get_sgemv_fix_call_result_23693)
    # Assigning a type to the variable 'sources' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'sources', result_iadd_23694)
    
    
    # Call to add_extension(...): (line 31)
    # Processing the call arguments (line 31)
    str_23697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 25), 'str', '_fblas')
    # Processing the call keyword arguments (line 31)
    # Getting the type of 'sources' (line 32)
    sources_23698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 33), 'sources', False)
    keyword_23699 = sources_23698
    
    # Obtaining an instance of the builtin type 'list' (line 33)
    list_23700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 33)
    # Adding element type (line 33)
    str_23701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 34), 'str', 'fblas_l?.pyf.src')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 33), list_23700, str_23701)
    
    keyword_23702 = list_23700
    # Getting the type of 'lapack_opt' (line 34)
    lapack_opt_23703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 36), 'lapack_opt', False)
    keyword_23704 = lapack_opt_23703
    kwargs_23705 = {'sources': keyword_23699, 'depends': keyword_23702, 'extra_info': keyword_23704}
    # Getting the type of 'config' (line 31)
    config_23695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 31)
    add_extension_23696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 4), config_23695, 'add_extension')
    # Calling add_extension(args, kwargs) (line 31)
    add_extension_call_result_23706 = invoke(stypy.reporting.localization.Localization(__file__, 31, 4), add_extension_23696, *[str_23697], **kwargs_23705)
    
    
    # Assigning a List to a Name (line 38):
    
    # Obtaining an instance of the builtin type 'list' (line 38)
    list_23707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 38)
    # Adding element type (line 38)
    str_23708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 15), 'str', 'flapack.pyf.src')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 14), list_23707, str_23708)
    
    # Assigning a type to the variable 'sources' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'sources', list_23707)
    
    # Getting the type of 'sources' (line 39)
    sources_23709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'sources')
    
    # Call to get_g77_abi_wrappers(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'lapack_opt' (line 39)
    lapack_opt_23711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 36), 'lapack_opt', False)
    # Processing the call keyword arguments (line 39)
    kwargs_23712 = {}
    # Getting the type of 'get_g77_abi_wrappers' (line 39)
    get_g77_abi_wrappers_23710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 15), 'get_g77_abi_wrappers', False)
    # Calling get_g77_abi_wrappers(args, kwargs) (line 39)
    get_g77_abi_wrappers_call_result_23713 = invoke(stypy.reporting.localization.Localization(__file__, 39, 15), get_g77_abi_wrappers_23710, *[lapack_opt_23711], **kwargs_23712)
    
    # Applying the binary operator '+=' (line 39)
    result_iadd_23714 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 4), '+=', sources_23709, get_g77_abi_wrappers_call_result_23713)
    # Assigning a type to the variable 'sources' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'sources', result_iadd_23714)
    
    
    # Assigning a Call to a Name (line 40):
    
    # Call to join(...): (line 40)
    # Processing the call arguments (line 40)
    str_23716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 19), 'str', 'src')
    str_23717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 26), 'str', 'lapack_deprecations')
    # Processing the call keyword arguments (line 40)
    kwargs_23718 = {}
    # Getting the type of 'join' (line 40)
    join_23715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 14), 'join', False)
    # Calling join(args, kwargs) (line 40)
    join_call_result_23719 = invoke(stypy.reporting.localization.Localization(__file__, 40, 14), join_23715, *[str_23716, str_23717], **kwargs_23718)
    
    # Assigning a type to the variable 'dep_pfx' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'dep_pfx', join_call_result_23719)
    
    # Assigning a ListComp to a Name (line 41):
    # Calculating list comprehension
    # Calculating comprehension expression
    str_23727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 71), 'str', 'cdsz')
    comprehension_23728 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 34), str_23727)
    # Assigning a type to the variable 'c' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 34), 'c', comprehension_23728)
    
    # Call to join(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'dep_pfx' (line 41)
    dep_pfx_23721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 39), 'dep_pfx', False)
    # Getting the type of 'c' (line 41)
    c_23722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 48), 'c', False)
    str_23723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 52), 'str', 'gegv.f')
    # Applying the binary operator '+' (line 41)
    result_add_23724 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 48), '+', c_23722, str_23723)
    
    # Processing the call keyword arguments (line 41)
    kwargs_23725 = {}
    # Getting the type of 'join' (line 41)
    join_23720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 34), 'join', False)
    # Calling join(args, kwargs) (line 41)
    join_call_result_23726 = invoke(stypy.reporting.localization.Localization(__file__, 41, 34), join_23720, *[dep_pfx_23721, result_add_23724], **kwargs_23725)
    
    list_23729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 34), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 34), list_23729, join_call_result_23726)
    # Assigning a type to the variable 'deprecated_lapack_routines' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'deprecated_lapack_routines', list_23729)
    
    # Getting the type of 'sources' (line 42)
    sources_23730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'sources')
    # Getting the type of 'deprecated_lapack_routines' (line 42)
    deprecated_lapack_routines_23731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 15), 'deprecated_lapack_routines')
    # Applying the binary operator '+=' (line 42)
    result_iadd_23732 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 4), '+=', sources_23730, deprecated_lapack_routines_23731)
    # Assigning a type to the variable 'sources' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'sources', result_iadd_23732)
    
    
    # Call to add_extension(...): (line 44)
    # Processing the call arguments (line 44)
    str_23735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 25), 'str', '_flapack')
    # Processing the call keyword arguments (line 44)
    # Getting the type of 'sources' (line 45)
    sources_23736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 33), 'sources', False)
    keyword_23737 = sources_23736
    
    # Obtaining an instance of the builtin type 'list' (line 46)
    list_23738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 46)
    # Adding element type (line 46)
    str_23739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 34), 'str', 'flapack_user.pyf.src')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 33), list_23738, str_23739)
    
    keyword_23740 = list_23738
    # Getting the type of 'lapack_opt' (line 47)
    lapack_opt_23741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 36), 'lapack_opt', False)
    keyword_23742 = lapack_opt_23741
    kwargs_23743 = {'sources': keyword_23737, 'depends': keyword_23740, 'extra_info': keyword_23742}
    # Getting the type of 'config' (line 44)
    config_23733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 44)
    add_extension_23734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 4), config_23733, 'add_extension')
    # Calling add_extension(args, kwargs) (line 44)
    add_extension_call_result_23744 = invoke(stypy.reporting.localization.Localization(__file__, 44, 4), add_extension_23734, *[str_23735], **kwargs_23743)
    
    
    # Type idiom detected: calculating its left and rigth part (line 50)
    # Getting the type of 'atlas_version' (line 50)
    atlas_version_23745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'atlas_version')
    # Getting the type of 'None' (line 50)
    None_23746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 28), 'None')
    
    (may_be_23747, more_types_in_union_23748) = may_not_be_none(atlas_version_23745, None_23746)

    if may_be_23747:

        if more_types_in_union_23748:
            # Runtime conditional SSA (line 50)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to add_extension(...): (line 52)
        # Processing the call arguments (line 52)
        str_23751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 29), 'str', '_cblas')
        # Processing the call keyword arguments (line 52)
        
        # Obtaining an instance of the builtin type 'list' (line 53)
        list_23752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 53)
        # Adding element type (line 53)
        str_23753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 38), 'str', 'cblas.pyf.src')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 37), list_23752, str_23753)
        
        keyword_23754 = list_23752
        
        # Obtaining an instance of the builtin type 'list' (line 54)
        list_23755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 54)
        # Adding element type (line 54)
        str_23756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 38), 'str', 'cblas.pyf.src')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 37), list_23755, str_23756)
        # Adding element type (line 54)
        str_23757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 55), 'str', 'cblas_l1.pyf.src')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 37), list_23755, str_23757)
        
        keyword_23758 = list_23755
        # Getting the type of 'lapack_opt' (line 55)
        lapack_opt_23759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 40), 'lapack_opt', False)
        keyword_23760 = lapack_opt_23759
        kwargs_23761 = {'sources': keyword_23754, 'depends': keyword_23758, 'extra_info': keyword_23760}
        # Getting the type of 'config' (line 52)
        config_23749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'config', False)
        # Obtaining the member 'add_extension' of a type (line 52)
        add_extension_23750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), config_23749, 'add_extension')
        # Calling add_extension(args, kwargs) (line 52)
        add_extension_call_result_23762 = invoke(stypy.reporting.localization.Localization(__file__, 52, 8), add_extension_23750, *[str_23751], **kwargs_23761)
        
        
        # Call to add_extension(...): (line 59)
        # Processing the call arguments (line 59)
        str_23765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 29), 'str', '_clapack')
        # Processing the call keyword arguments (line 59)
        
        # Obtaining an instance of the builtin type 'list' (line 60)
        list_23766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 60)
        # Adding element type (line 60)
        str_23767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 38), 'str', 'clapack.pyf.src')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 37), list_23766, str_23767)
        
        keyword_23768 = list_23766
        
        # Obtaining an instance of the builtin type 'list' (line 61)
        list_23769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 61)
        # Adding element type (line 61)
        str_23770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 38), 'str', 'clapack.pyf.src')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 37), list_23769, str_23770)
        
        keyword_23771 = list_23769
        # Getting the type of 'lapack_opt' (line 62)
        lapack_opt_23772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 40), 'lapack_opt', False)
        keyword_23773 = lapack_opt_23772
        kwargs_23774 = {'sources': keyword_23768, 'depends': keyword_23771, 'extra_info': keyword_23773}
        # Getting the type of 'config' (line 59)
        config_23763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'config', False)
        # Obtaining the member 'add_extension' of a type (line 59)
        add_extension_23764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), config_23763, 'add_extension')
        # Calling add_extension(args, kwargs) (line 59)
        add_extension_call_result_23775 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), add_extension_23764, *[str_23765], **kwargs_23774)
        

        if more_types_in_union_23748:
            # SSA join for if statement (line 50)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to add_extension(...): (line 66)
    # Processing the call arguments (line 66)
    str_23778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 25), 'str', '_flinalg')
    # Processing the call keyword arguments (line 66)
    
    # Obtaining an instance of the builtin type 'list' (line 67)
    list_23779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 67)
    # Adding element type (line 67)
    
    # Call to join(...): (line 67)
    # Processing the call arguments (line 67)
    str_23781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 39), 'str', 'src')
    str_23782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 46), 'str', 'det.f')
    # Processing the call keyword arguments (line 67)
    kwargs_23783 = {}
    # Getting the type of 'join' (line 67)
    join_23780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 34), 'join', False)
    # Calling join(args, kwargs) (line 67)
    join_call_result_23784 = invoke(stypy.reporting.localization.Localization(__file__, 67, 34), join_23780, *[str_23781, str_23782], **kwargs_23783)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 33), list_23779, join_call_result_23784)
    # Adding element type (line 67)
    
    # Call to join(...): (line 67)
    # Processing the call arguments (line 67)
    str_23786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 61), 'str', 'src')
    str_23787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 68), 'str', 'lu.f')
    # Processing the call keyword arguments (line 67)
    kwargs_23788 = {}
    # Getting the type of 'join' (line 67)
    join_23785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 56), 'join', False)
    # Calling join(args, kwargs) (line 67)
    join_call_result_23789 = invoke(stypy.reporting.localization.Localization(__file__, 67, 56), join_23785, *[str_23786, str_23787], **kwargs_23788)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 33), list_23779, join_call_result_23789)
    
    keyword_23790 = list_23779
    # Getting the type of 'lapack_opt' (line 68)
    lapack_opt_23791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 36), 'lapack_opt', False)
    keyword_23792 = lapack_opt_23791
    kwargs_23793 = {'sources': keyword_23790, 'extra_info': keyword_23792}
    # Getting the type of 'config' (line 66)
    config_23776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 66)
    add_extension_23777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 4), config_23776, 'add_extension')
    # Calling add_extension(args, kwargs) (line 66)
    add_extension_call_result_23794 = invoke(stypy.reporting.localization.Localization(__file__, 66, 4), add_extension_23777, *[str_23778], **kwargs_23793)
    
    
    # Assigning a List to a Name (line 72):
    
    # Obtaining an instance of the builtin type 'list' (line 72)
    list_23795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 72)
    # Adding element type (line 72)
    str_23796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 8), 'str', 'dfftb1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23796)
    # Adding element type (line 72)
    str_23797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 8), 'str', 'dfftf1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23797)
    # Adding element type (line 72)
    str_23798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 8), 'str', 'dffti1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23798)
    # Adding element type (line 72)
    str_23799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 8), 'str', 'dsint1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23799)
    # Adding element type (line 72)
    str_23800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 8), 'str', 'dzfft1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23800)
    # Adding element type (line 72)
    str_23801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 8), 'str', 'id_srand')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23801)
    # Adding element type (line 72)
    str_23802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 8), 'str', 'idd_copyints')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23802)
    # Adding element type (line 72)
    str_23803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 8), 'str', 'idd_id2svd0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23803)
    # Adding element type (line 72)
    str_23804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 8), 'str', 'idd_pairsamps')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23804)
    # Adding element type (line 72)
    str_23805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 8), 'str', 'idd_permute')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23805)
    # Adding element type (line 72)
    str_23806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 8), 'str', 'idd_permuter')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23806)
    # Adding element type (line 72)
    str_23807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 8), 'str', 'idd_random_transf0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23807)
    # Adding element type (line 72)
    str_23808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 8), 'str', 'idd_random_transf0_inv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23808)
    # Adding element type (line 72)
    str_23809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 8), 'str', 'idd_random_transf_init0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23809)
    # Adding element type (line 72)
    str_23810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 8), 'str', 'idd_subselect')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23810)
    # Adding element type (line 72)
    str_23811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 8), 'str', 'iddp_asvd0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23811)
    # Adding element type (line 72)
    str_23812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 8), 'str', 'iddp_rsvd0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23812)
    # Adding element type (line 72)
    str_23813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 8), 'str', 'iddr_asvd0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23813)
    # Adding element type (line 72)
    str_23814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 8), 'str', 'iddr_rsvd0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23814)
    # Adding element type (line 72)
    str_23815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 8), 'str', 'idz_estrank0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23815)
    # Adding element type (line 72)
    str_23816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 8), 'str', 'idz_id2svd0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23816)
    # Adding element type (line 72)
    str_23817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 8), 'str', 'idz_permute')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23817)
    # Adding element type (line 72)
    str_23818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 8), 'str', 'idz_permuter')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23818)
    # Adding element type (line 72)
    str_23819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 8), 'str', 'idz_random_transf0_inv')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23819)
    # Adding element type (line 72)
    str_23820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 8), 'str', 'idz_random_transf_init0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23820)
    # Adding element type (line 72)
    str_23821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 8), 'str', 'idz_random_transf_init00')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23821)
    # Adding element type (line 72)
    str_23822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 8), 'str', 'idz_realcomp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23822)
    # Adding element type (line 72)
    str_23823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 8), 'str', 'idz_realcomplex')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23823)
    # Adding element type (line 72)
    str_23824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 8), 'str', 'idz_reco')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23824)
    # Adding element type (line 72)
    str_23825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 8), 'str', 'idz_subselect')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23825)
    # Adding element type (line 72)
    str_23826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 8), 'str', 'idzp_aid0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23826)
    # Adding element type (line 72)
    str_23827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 8), 'str', 'idzp_aid1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23827)
    # Adding element type (line 72)
    str_23828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 8), 'str', 'idzp_asvd0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23828)
    # Adding element type (line 72)
    str_23829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 8), 'str', 'idzp_rsvd0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23829)
    # Adding element type (line 72)
    str_23830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 8), 'str', 'idzr_asvd0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23830)
    # Adding element type (line 72)
    str_23831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 8), 'str', 'idzr_reco')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23831)
    # Adding element type (line 72)
    str_23832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 8), 'str', 'idzr_rsvd0')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23832)
    # Adding element type (line 72)
    str_23833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 8), 'str', 'zfftb1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23833)
    # Adding element type (line 72)
    str_23834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 8), 'str', 'zfftf1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23834)
    # Adding element type (line 72)
    str_23835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 8), 'str', 'zffti1')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 24), list_23795, str_23835)
    
    # Assigning a type to the variable 'routines_to_split' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'routines_to_split', list_23795)
    
    # Call to print(...): (line 114)
    # Processing the call arguments (line 114)
    str_23837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 10), 'str', 'Splitting linalg.interpolative Fortran source files')
    # Processing the call keyword arguments (line 114)
    kwargs_23838 = {}
    # Getting the type of 'print' (line 114)
    print_23836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'print', False)
    # Calling print(args, kwargs) (line 114)
    print_call_result_23839 = invoke(stypy.reporting.localization.Localization(__file__, 114, 4), print_23836, *[str_23837], **kwargs_23838)
    
    
    # Assigning a Subscript to a Name (line 115):
    
    # Obtaining the type of the subscript
    int_23840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 55), 'int')
    
    # Call to split(...): (line 115)
    # Processing the call arguments (line 115)
    
    # Call to abspath(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of '__file__' (line 115)
    file___23847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 44), '__file__', False)
    # Processing the call keyword arguments (line 115)
    kwargs_23848 = {}
    # Getting the type of 'os' (line 115)
    os_23844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 28), 'os', False)
    # Obtaining the member 'path' of a type (line 115)
    path_23845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 28), os_23844, 'path')
    # Obtaining the member 'abspath' of a type (line 115)
    abspath_23846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 28), path_23845, 'abspath')
    # Calling abspath(args, kwargs) (line 115)
    abspath_call_result_23849 = invoke(stypy.reporting.localization.Localization(__file__, 115, 28), abspath_23846, *[file___23847], **kwargs_23848)
    
    # Processing the call keyword arguments (line 115)
    kwargs_23850 = {}
    # Getting the type of 'os' (line 115)
    os_23841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 14), 'os', False)
    # Obtaining the member 'path' of a type (line 115)
    path_23842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 14), os_23841, 'path')
    # Obtaining the member 'split' of a type (line 115)
    split_23843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 14), path_23842, 'split')
    # Calling split(args, kwargs) (line 115)
    split_call_result_23851 = invoke(stypy.reporting.localization.Localization(__file__, 115, 14), split_23843, *[abspath_call_result_23849], **kwargs_23850)
    
    # Obtaining the member '__getitem__' of a type (line 115)
    getitem___23852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 14), split_call_result_23851, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 115)
    subscript_call_result_23853 = invoke(stypy.reporting.localization.Localization(__file__, 115, 14), getitem___23852, int_23840)
    
    # Assigning a type to the variable 'dirname' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'dirname', subscript_call_result_23853)
    
    # Assigning a Call to a Name (line 116):
    
    # Call to split_fortran_files(...): (line 116)
    # Processing the call arguments (line 116)
    
    # Call to join(...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 'dirname' (line 116)
    dirname_23856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 38), 'dirname', False)
    str_23857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 47), 'str', 'src')
    str_23858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 54), 'str', 'id_dist')
    str_23859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 65), 'str', 'src')
    # Processing the call keyword arguments (line 116)
    kwargs_23860 = {}
    # Getting the type of 'join' (line 116)
    join_23855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 33), 'join', False)
    # Calling join(args, kwargs) (line 116)
    join_call_result_23861 = invoke(stypy.reporting.localization.Localization(__file__, 116, 33), join_23855, *[dirname_23856, str_23857, str_23858, str_23859], **kwargs_23860)
    
    # Getting the type of 'routines_to_split' (line 117)
    routines_to_split_23862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 33), 'routines_to_split', False)
    # Processing the call keyword arguments (line 116)
    kwargs_23863 = {}
    # Getting the type of 'split_fortran_files' (line 116)
    split_fortran_files_23854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 13), 'split_fortran_files', False)
    # Calling split_fortran_files(args, kwargs) (line 116)
    split_fortran_files_call_result_23864 = invoke(stypy.reporting.localization.Localization(__file__, 116, 13), split_fortran_files_23854, *[join_call_result_23861, routines_to_split_23862], **kwargs_23863)
    
    # Assigning a type to the variable 'fnames' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'fnames', split_fortran_files_call_result_23864)
    
    # Assigning a ListComp to a Name (line 118):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'fnames' (line 118)
    fnames_23872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 56), 'fnames')
    comprehension_23873 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 14), fnames_23872)
    # Assigning a type to the variable 'f' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 14), 'f', comprehension_23873)
    
    # Call to join(...): (line 118)
    # Processing the call arguments (line 118)
    str_23866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 19), 'str', 'src')
    str_23867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 26), 'str', 'id_dist')
    str_23868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 37), 'str', 'src')
    # Getting the type of 'f' (line 118)
    f_23869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 44), 'f', False)
    # Processing the call keyword arguments (line 118)
    kwargs_23870 = {}
    # Getting the type of 'join' (line 118)
    join_23865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 14), 'join', False)
    # Calling join(args, kwargs) (line 118)
    join_call_result_23871 = invoke(stypy.reporting.localization.Localization(__file__, 118, 14), join_23865, *[str_23866, str_23867, str_23868, f_23869], **kwargs_23870)
    
    list_23874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 14), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 14), list_23874, join_call_result_23871)
    # Assigning a type to the variable 'fnames' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'fnames', list_23874)
    
    # Call to add_extension(...): (line 119)
    # Processing the call arguments (line 119)
    str_23877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 25), 'str', '_interpolative')
    # Getting the type of 'fnames' (line 119)
    fnames_23878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 43), 'fnames', False)
    
    # Obtaining an instance of the builtin type 'list' (line 119)
    list_23879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 52), 'list')
    # Adding type elements to the builtin type 'list' instance (line 119)
    # Adding element type (line 119)
    str_23880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 53), 'str', 'interpolative.pyf')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 52), list_23879, str_23880)
    
    # Applying the binary operator '+' (line 119)
    result_add_23881 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 43), '+', fnames_23878, list_23879)
    
    # Processing the call keyword arguments (line 119)
    # Getting the type of 'lapack_opt' (line 120)
    lapack_opt_23882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 36), 'lapack_opt', False)
    keyword_23883 = lapack_opt_23882
    kwargs_23884 = {'extra_info': keyword_23883}
    # Getting the type of 'config' (line 119)
    config_23875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 119)
    add_extension_23876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 4), config_23875, 'add_extension')
    # Calling add_extension(args, kwargs) (line 119)
    add_extension_call_result_23885 = invoke(stypy.reporting.localization.Localization(__file__, 119, 4), add_extension_23876, *[str_23877, result_add_23881], **kwargs_23884)
    
    
    # Call to add_extension(...): (line 124)
    # Processing the call arguments (line 124)
    str_23888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 25), 'str', '_solve_toeplitz')
    # Processing the call keyword arguments (line 124)
    
    # Obtaining an instance of the builtin type 'list' (line 125)
    list_23889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 125)
    # Adding element type (line 125)
    str_23890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 35), 'str', '_solve_toeplitz.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 125, 33), list_23889, str_23890)
    
    keyword_23891 = list_23889
    
    # Obtaining an instance of the builtin type 'list' (line 126)
    list_23892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 126)
    # Adding element type (line 126)
    
    # Call to get_numpy_include_dirs(...): (line 126)
    # Processing the call keyword arguments (line 126)
    kwargs_23894 = {}
    # Getting the type of 'get_numpy_include_dirs' (line 126)
    get_numpy_include_dirs_23893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 39), 'get_numpy_include_dirs', False)
    # Calling get_numpy_include_dirs(args, kwargs) (line 126)
    get_numpy_include_dirs_call_result_23895 = invoke(stypy.reporting.localization.Localization(__file__, 126, 39), get_numpy_include_dirs_23893, *[], **kwargs_23894)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 38), list_23892, get_numpy_include_dirs_call_result_23895)
    
    keyword_23896 = list_23892
    kwargs_23897 = {'sources': keyword_23891, 'include_dirs': keyword_23896}
    # Getting the type of 'config' (line 124)
    config_23886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 124)
    add_extension_23887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 4), config_23886, 'add_extension')
    # Calling add_extension(args, kwargs) (line 124)
    add_extension_call_result_23898 = invoke(stypy.reporting.localization.Localization(__file__, 124, 4), add_extension_23887, *[str_23888], **kwargs_23897)
    
    
    # Call to add_data_dir(...): (line 128)
    # Processing the call arguments (line 128)
    str_23901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 24), 'str', 'tests')
    # Processing the call keyword arguments (line 128)
    kwargs_23902 = {}
    # Getting the type of 'config' (line 128)
    config_23899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'config', False)
    # Obtaining the member 'add_data_dir' of a type (line 128)
    add_data_dir_23900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 4), config_23899, 'add_data_dir')
    # Calling add_data_dir(args, kwargs) (line 128)
    add_data_dir_call_result_23903 = invoke(stypy.reporting.localization.Localization(__file__, 128, 4), add_data_dir_23900, *[str_23901], **kwargs_23902)
    
    
    # Call to add_data_files(...): (line 131)
    # Processing the call arguments (line 131)
    str_23906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 26), 'str', 'cython_blas.pxd')
    # Processing the call keyword arguments (line 131)
    kwargs_23907 = {}
    # Getting the type of 'config' (line 131)
    config_23904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'config', False)
    # Obtaining the member 'add_data_files' of a type (line 131)
    add_data_files_23905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 4), config_23904, 'add_data_files')
    # Calling add_data_files(args, kwargs) (line 131)
    add_data_files_call_result_23908 = invoke(stypy.reporting.localization.Localization(__file__, 131, 4), add_data_files_23905, *[str_23906], **kwargs_23907)
    
    
    # Call to add_data_files(...): (line 132)
    # Processing the call arguments (line 132)
    str_23911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 26), 'str', 'cython_lapack.pxd')
    # Processing the call keyword arguments (line 132)
    kwargs_23912 = {}
    # Getting the type of 'config' (line 132)
    config_23909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'config', False)
    # Obtaining the member 'add_data_files' of a type (line 132)
    add_data_files_23910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 4), config_23909, 'add_data_files')
    # Calling add_data_files(args, kwargs) (line 132)
    add_data_files_call_result_23913 = invoke(stypy.reporting.localization.Localization(__file__, 132, 4), add_data_files_23910, *[str_23911], **kwargs_23912)
    
    
    # Assigning a List to a Name (line 134):
    
    # Obtaining an instance of the builtin type 'list' (line 134)
    list_23914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 134)
    # Adding element type (line 134)
    str_23915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 15), 'str', '_blas_subroutine_wrappers.f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 14), list_23914, str_23915)
    # Adding element type (line 134)
    str_23916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 46), 'str', '_lapack_subroutine_wrappers.f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 14), list_23914, str_23916)
    
    # Assigning a type to the variable 'sources' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'sources', list_23914)
    
    # Getting the type of 'sources' (line 135)
    sources_23917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'sources')
    
    # Call to get_g77_abi_wrappers(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'lapack_opt' (line 135)
    lapack_opt_23919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 36), 'lapack_opt', False)
    # Processing the call keyword arguments (line 135)
    kwargs_23920 = {}
    # Getting the type of 'get_g77_abi_wrappers' (line 135)
    get_g77_abi_wrappers_23918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 15), 'get_g77_abi_wrappers', False)
    # Calling get_g77_abi_wrappers(args, kwargs) (line 135)
    get_g77_abi_wrappers_call_result_23921 = invoke(stypy.reporting.localization.Localization(__file__, 135, 15), get_g77_abi_wrappers_23918, *[lapack_opt_23919], **kwargs_23920)
    
    # Applying the binary operator '+=' (line 135)
    result_iadd_23922 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 4), '+=', sources_23917, get_g77_abi_wrappers_call_result_23921)
    # Assigning a type to the variable 'sources' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'sources', result_iadd_23922)
    
    
    # Getting the type of 'sources' (line 136)
    sources_23923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'sources')
    
    # Call to get_sgemv_fix(...): (line 136)
    # Processing the call arguments (line 136)
    # Getting the type of 'lapack_opt' (line 136)
    lapack_opt_23925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 29), 'lapack_opt', False)
    # Processing the call keyword arguments (line 136)
    kwargs_23926 = {}
    # Getting the type of 'get_sgemv_fix' (line 136)
    get_sgemv_fix_23924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 15), 'get_sgemv_fix', False)
    # Calling get_sgemv_fix(args, kwargs) (line 136)
    get_sgemv_fix_call_result_23927 = invoke(stypy.reporting.localization.Localization(__file__, 136, 15), get_sgemv_fix_23924, *[lapack_opt_23925], **kwargs_23926)
    
    # Applying the binary operator '+=' (line 136)
    result_iadd_23928 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 4), '+=', sources_23923, get_sgemv_fix_call_result_23927)
    # Assigning a type to the variable 'sources' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'sources', result_iadd_23928)
    
    
    # Assigning a BinOp to a Name (line 137):
    
    # Call to get_include_dirs(...): (line 137)
    # Processing the call keyword arguments (line 137)
    kwargs_23933 = {}
    
    # Call to numpy_info(...): (line 137)
    # Processing the call keyword arguments (line 137)
    kwargs_23930 = {}
    # Getting the type of 'numpy_info' (line 137)
    numpy_info_23929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 15), 'numpy_info', False)
    # Calling numpy_info(args, kwargs) (line 137)
    numpy_info_call_result_23931 = invoke(stypy.reporting.localization.Localization(__file__, 137, 15), numpy_info_23929, *[], **kwargs_23930)
    
    # Obtaining the member 'get_include_dirs' of a type (line 137)
    get_include_dirs_23932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 15), numpy_info_call_result_23931, 'get_include_dirs')
    # Calling get_include_dirs(args, kwargs) (line 137)
    get_include_dirs_call_result_23934 = invoke(stypy.reporting.localization.Localization(__file__, 137, 15), get_include_dirs_23932, *[], **kwargs_23933)
    
    
    # Obtaining an instance of the builtin type 'list' (line 137)
    list_23935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 49), 'list')
    # Adding type elements to the builtin type 'list' instance (line 137)
    # Adding element type (line 137)
    
    # Call to get_python_inc(...): (line 137)
    # Processing the call keyword arguments (line 137)
    kwargs_23937 = {}
    # Getting the type of 'get_python_inc' (line 137)
    get_python_inc_23936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 50), 'get_python_inc', False)
    # Calling get_python_inc(args, kwargs) (line 137)
    get_python_inc_call_result_23938 = invoke(stypy.reporting.localization.Localization(__file__, 137, 50), get_python_inc_23936, *[], **kwargs_23937)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 49), list_23935, get_python_inc_call_result_23938)
    
    # Applying the binary operator '+' (line 137)
    result_add_23939 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 15), '+', get_include_dirs_call_result_23934, list_23935)
    
    # Assigning a type to the variable 'includes' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'includes', result_add_23939)
    
    # Call to add_library(...): (line 138)
    # Processing the call arguments (line 138)
    str_23942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 23), 'str', 'fwrappers')
    # Processing the call keyword arguments (line 138)
    # Getting the type of 'sources' (line 138)
    sources_23943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 44), 'sources', False)
    keyword_23944 = sources_23943
    # Getting the type of 'includes' (line 138)
    includes_23945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 66), 'includes', False)
    keyword_23946 = includes_23945
    kwargs_23947 = {'sources': keyword_23944, 'include_dirs': keyword_23946}
    # Getting the type of 'config' (line 138)
    config_23940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'config', False)
    # Obtaining the member 'add_library' of a type (line 138)
    add_library_23941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 4), config_23940, 'add_library')
    # Calling add_library(args, kwargs) (line 138)
    add_library_call_result_23948 = invoke(stypy.reporting.localization.Localization(__file__, 138, 4), add_library_23941, *[str_23942], **kwargs_23947)
    
    
    # Call to add_extension(...): (line 140)
    # Processing the call arguments (line 140)
    str_23951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 25), 'str', 'cython_blas')
    # Processing the call keyword arguments (line 140)
    
    # Obtaining an instance of the builtin type 'list' (line 141)
    list_23952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 141)
    # Adding element type (line 141)
    str_23953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 34), 'str', 'cython_blas.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 141, 33), list_23952, str_23953)
    
    keyword_23954 = list_23952
    
    # Obtaining an instance of the builtin type 'list' (line 142)
    list_23955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 142)
    # Adding element type (line 142)
    str_23956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 34), 'str', 'cython_blas.pyx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 33), list_23955, str_23956)
    # Adding element type (line 142)
    str_23957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 53), 'str', 'cython_blas.pxd')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 33), list_23955, str_23957)
    # Adding element type (line 142)
    str_23958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 34), 'str', 'fortran_defs.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 33), list_23955, str_23958)
    # Adding element type (line 142)
    str_23959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 52), 'str', '_blas_subroutines.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 142, 33), list_23955, str_23959)
    
    keyword_23960 = list_23955
    
    # Obtaining an instance of the builtin type 'list' (line 144)
    list_23961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 144)
    # Adding element type (line 144)
    str_23962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 39), 'str', '.')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 144, 38), list_23961, str_23962)
    
    keyword_23963 = list_23961
    
    # Obtaining an instance of the builtin type 'list' (line 145)
    list_23964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 145)
    # Adding element type (line 145)
    str_23965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 36), 'str', 'fwrappers')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 145, 35), list_23964, str_23965)
    
    keyword_23966 = list_23964
    # Getting the type of 'lapack_opt' (line 146)
    lapack_opt_23967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 36), 'lapack_opt', False)
    keyword_23968 = lapack_opt_23967
    kwargs_23969 = {'libraries': keyword_23966, 'sources': keyword_23954, 'depends': keyword_23960, 'extra_info': keyword_23968, 'include_dirs': keyword_23963}
    # Getting the type of 'config' (line 140)
    config_23949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 140)
    add_extension_23950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 4), config_23949, 'add_extension')
    # Calling add_extension(args, kwargs) (line 140)
    add_extension_call_result_23970 = invoke(stypy.reporting.localization.Localization(__file__, 140, 4), add_extension_23950, *[str_23951], **kwargs_23969)
    
    
    # Call to add_extension(...): (line 148)
    # Processing the call arguments (line 148)
    str_23973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 25), 'str', 'cython_lapack')
    # Processing the call keyword arguments (line 148)
    
    # Obtaining an instance of the builtin type 'list' (line 149)
    list_23974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 149)
    # Adding element type (line 149)
    str_23975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 34), 'str', 'cython_lapack.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 33), list_23974, str_23975)
    
    keyword_23976 = list_23974
    
    # Obtaining an instance of the builtin type 'list' (line 150)
    list_23977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 150)
    # Adding element type (line 150)
    str_23978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 34), 'str', 'cython_lapack.pyx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 33), list_23977, str_23978)
    # Adding element type (line 150)
    str_23979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 55), 'str', 'cython_lapack.pxd')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 33), list_23977, str_23979)
    # Adding element type (line 150)
    str_23980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 34), 'str', 'fortran_defs.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 33), list_23977, str_23980)
    # Adding element type (line 150)
    str_23981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 52), 'str', '_lapack_subroutines.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 33), list_23977, str_23981)
    
    keyword_23982 = list_23977
    
    # Obtaining an instance of the builtin type 'list' (line 152)
    list_23983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 152)
    # Adding element type (line 152)
    str_23984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 39), 'str', '.')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 38), list_23983, str_23984)
    
    keyword_23985 = list_23983
    
    # Obtaining an instance of the builtin type 'list' (line 153)
    list_23986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 153)
    # Adding element type (line 153)
    str_23987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 36), 'str', 'fwrappers')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 35), list_23986, str_23987)
    
    keyword_23988 = list_23986
    # Getting the type of 'lapack_opt' (line 154)
    lapack_opt_23989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 36), 'lapack_opt', False)
    keyword_23990 = lapack_opt_23989
    kwargs_23991 = {'libraries': keyword_23988, 'sources': keyword_23976, 'depends': keyword_23982, 'extra_info': keyword_23990, 'include_dirs': keyword_23985}
    # Getting the type of 'config' (line 148)
    config_23971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 148)
    add_extension_23972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 4), config_23971, 'add_extension')
    # Calling add_extension(args, kwargs) (line 148)
    add_extension_call_result_23992 = invoke(stypy.reporting.localization.Localization(__file__, 148, 4), add_extension_23972, *[str_23973], **kwargs_23991)
    
    
    # Call to add_extension(...): (line 156)
    # Processing the call arguments (line 156)
    str_23995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 25), 'str', '_decomp_update')
    # Processing the call keyword arguments (line 156)
    
    # Obtaining an instance of the builtin type 'list' (line 157)
    list_23996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 157)
    # Adding element type (line 157)
    str_23997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 34), 'str', '_decomp_update.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 33), list_23996, str_23997)
    
    keyword_23998 = list_23996
    kwargs_23999 = {'sources': keyword_23998}
    # Getting the type of 'config' (line 156)
    config_23993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 156)
    add_extension_23994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 4), config_23993, 'add_extension')
    # Calling add_extension(args, kwargs) (line 156)
    add_extension_call_result_24000 = invoke(stypy.reporting.localization.Localization(__file__, 156, 4), add_extension_23994, *[str_23995], **kwargs_23999)
    
    
    # Call to add_data_files(...): (line 160)
    # Processing the call arguments (line 160)
    str_24003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 26), 'str', 'src/id_dist/doc/doc.tex')
    # Processing the call keyword arguments (line 160)
    kwargs_24004 = {}
    # Getting the type of 'config' (line 160)
    config_24001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'config', False)
    # Obtaining the member 'add_data_files' of a type (line 160)
    add_data_files_24002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 4), config_24001, 'add_data_files')
    # Calling add_data_files(args, kwargs) (line 160)
    add_data_files_call_result_24005 = invoke(stypy.reporting.localization.Localization(__file__, 160, 4), add_data_files_24002, *[str_24003], **kwargs_24004)
    
    
    # Call to add_data_files(...): (line 161)
    # Processing the call arguments (line 161)
    str_24008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 26), 'str', 'src/lapack_deprecations/LICENSE')
    # Processing the call keyword arguments (line 161)
    kwargs_24009 = {}
    # Getting the type of 'config' (line 161)
    config_24006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'config', False)
    # Obtaining the member 'add_data_files' of a type (line 161)
    add_data_files_24007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 4), config_24006, 'add_data_files')
    # Calling add_data_files(args, kwargs) (line 161)
    add_data_files_call_result_24010 = invoke(stypy.reporting.localization.Localization(__file__, 161, 4), add_data_files_24007, *[str_24008], **kwargs_24009)
    
    # Getting the type of 'config' (line 163)
    config_24011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'stypy_return_type', config_24011)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 7)
    stypy_return_type_24012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24012)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_24012

# Assigning a type to the variable 'configuration' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 167, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 167)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
    import_24013 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 167, 4), 'numpy.distutils.core')

    if (type(import_24013) is not StypyTypeError):

        if (import_24013 != 'pyd_module'):
            __import__(import_24013)
            sys_modules_24014 = sys.modules[import_24013]
            import_from_module(stypy.reporting.localization.Localization(__file__, 167, 4), 'numpy.distutils.core', sys_modules_24014.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 167, 4), __file__, sys_modules_24014, sys_modules_24014.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 167, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'numpy.distutils.core', import_24013)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 168, 4))
    
    # 'from linalg_version import linalg_version' statement (line 168)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/')
    import_24015 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 168, 4), 'linalg_version')

    if (type(import_24015) is not StypyTypeError):

        if (import_24015 != 'pyd_module'):
            __import__(import_24015)
            sys_modules_24016 = sys.modules[import_24015]
            import_from_module(stypy.reporting.localization.Localization(__file__, 168, 4), 'linalg_version', sys_modules_24016.module_type_store, module_type_store, ['linalg_version'])
            nest_module(stypy.reporting.localization.Localization(__file__, 168, 4), __file__, sys_modules_24016, sys_modules_24016.module_type_store, module_type_store)
        else:
            from linalg_version import linalg_version

            import_from_module(stypy.reporting.localization.Localization(__file__, 168, 4), 'linalg_version', None, module_type_store, ['linalg_version'], [linalg_version])

    else:
        # Assigning a type to the variable 'linalg_version' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'linalg_version', import_24015)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/')
    
    
    # Call to setup(...): (line 170)
    # Processing the call keyword arguments (line 170)
    # Getting the type of 'linalg_version' (line 170)
    linalg_version_24018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 18), 'linalg_version', False)
    keyword_24019 = linalg_version_24018
    
    # Call to todict(...): (line 171)
    # Processing the call keyword arguments (line 171)
    kwargs_24026 = {}
    
    # Call to configuration(...): (line 171)
    # Processing the call keyword arguments (line 171)
    str_24021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 35), 'str', '')
    keyword_24022 = str_24021
    kwargs_24023 = {'top_path': keyword_24022}
    # Getting the type of 'configuration' (line 171)
    configuration_24020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'configuration', False)
    # Calling configuration(args, kwargs) (line 171)
    configuration_call_result_24024 = invoke(stypy.reporting.localization.Localization(__file__, 171, 12), configuration_24020, *[], **kwargs_24023)
    
    # Obtaining the member 'todict' of a type (line 171)
    todict_24025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 12), configuration_call_result_24024, 'todict')
    # Calling todict(args, kwargs) (line 171)
    todict_call_result_24027 = invoke(stypy.reporting.localization.Localization(__file__, 171, 12), todict_24025, *[], **kwargs_24026)
    
    kwargs_24028 = {'todict_call_result_24027': todict_call_result_24027, 'version': keyword_24019}
    # Getting the type of 'setup' (line 170)
    setup_24017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 170)
    setup_call_result_24029 = invoke(stypy.reporting.localization.Localization(__file__, 170, 4), setup_24017, *[], **kwargs_24028)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
