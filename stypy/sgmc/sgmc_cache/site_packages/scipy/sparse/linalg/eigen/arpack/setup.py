
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: from os.path import join
4: 
5: 
6: def configuration(parent_package='',top_path=None):
7:     from numpy.distutils.system_info import get_info, NotFoundError
8:     from numpy.distutils.misc_util import Configuration
9:     from scipy._build_utils import get_g77_abi_wrappers, get_sgemv_fix
10: 
11:     lapack_opt = get_info('lapack_opt')
12: 
13:     if not lapack_opt:
14:         raise NotFoundError('no lapack/blas resources found')
15: 
16:     config = Configuration('arpack', parent_package, top_path)
17: 
18:     arpack_sources = [join('ARPACK','SRC', '*.f')]
19:     arpack_sources.extend([join('ARPACK','UTIL', '*.f')])
20: 
21:     arpack_sources += get_g77_abi_wrappers(lapack_opt)
22: 
23:     config.add_library('arpack_scipy', sources=arpack_sources,
24:                        include_dirs=[join('ARPACK', 'SRC')])
25: 
26:     ext_sources = ['arpack.pyf.src']
27:     ext_sources += get_sgemv_fix(lapack_opt)
28:     config.add_extension('_arpack',
29:                          sources=ext_sources,
30:                          libraries=['arpack_scipy'],
31:                          extra_info=lapack_opt,
32:                          depends=arpack_sources,
33:                          )
34: 
35:     config.add_data_dir('tests')
36: 
37:     # Add license files
38:     config.add_data_files('ARPACK/COPYING')
39: 
40:     return config
41: 
42: if __name__ == '__main__':
43:     from numpy.distutils.core import setup
44:     setup(**configuration(top_path='').todict())
45: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from os.path import join' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/arpack/')
import_401192 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path')

if (type(import_401192) is not StypyTypeError):

    if (import_401192 != 'pyd_module'):
        __import__(import_401192)
        sys_modules_401193 = sys.modules[import_401192]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path', sys_modules_401193.module_type_store, module_type_store, ['join'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_401193, sys_modules_401193.module_type_store, module_type_store)
    else:
        from os.path import join

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path', None, module_type_store, ['join'], [join])

else:
    # Assigning a type to the variable 'os.path' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path', import_401192)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/arpack/')


@norecursion
def configuration(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_401194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 33), 'str', '')
    # Getting the type of 'None' (line 6)
    None_401195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 45), 'None')
    defaults = [str_401194, None_401195]
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
    
    # 'from numpy.distutils.system_info import get_info, NotFoundError' statement (line 7)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/arpack/')
    import_401196 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy.distutils.system_info')

    if (type(import_401196) is not StypyTypeError):

        if (import_401196 != 'pyd_module'):
            __import__(import_401196)
            sys_modules_401197 = sys.modules[import_401196]
            import_from_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy.distutils.system_info', sys_modules_401197.module_type_store, module_type_store, ['get_info', 'NotFoundError'])
            nest_module(stypy.reporting.localization.Localization(__file__, 7, 4), __file__, sys_modules_401197, sys_modules_401197.module_type_store, module_type_store)
        else:
            from numpy.distutils.system_info import get_info, NotFoundError

            import_from_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy.distutils.system_info', None, module_type_store, ['get_info', 'NotFoundError'], [get_info, NotFoundError])

    else:
        # Assigning a type to the variable 'numpy.distutils.system_info' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy.distutils.system_info', import_401196)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/arpack/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 4))
    
    # 'from numpy.distutils.misc_util import Configuration' statement (line 8)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/arpack/')
    import_401198 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'numpy.distutils.misc_util')

    if (type(import_401198) is not StypyTypeError):

        if (import_401198 != 'pyd_module'):
            __import__(import_401198)
            sys_modules_401199 = sys.modules[import_401198]
            import_from_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'numpy.distutils.misc_util', sys_modules_401199.module_type_store, module_type_store, ['Configuration'])
            nest_module(stypy.reporting.localization.Localization(__file__, 8, 4), __file__, sys_modules_401199, sys_modules_401199.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration

            import_from_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'numpy.distutils.misc_util', import_401198)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/arpack/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 4))
    
    # 'from scipy._build_utils import get_g77_abi_wrappers, get_sgemv_fix' statement (line 9)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/arpack/')
    import_401200 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'scipy._build_utils')

    if (type(import_401200) is not StypyTypeError):

        if (import_401200 != 'pyd_module'):
            __import__(import_401200)
            sys_modules_401201 = sys.modules[import_401200]
            import_from_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'scipy._build_utils', sys_modules_401201.module_type_store, module_type_store, ['get_g77_abi_wrappers', 'get_sgemv_fix'])
            nest_module(stypy.reporting.localization.Localization(__file__, 9, 4), __file__, sys_modules_401201, sys_modules_401201.module_type_store, module_type_store)
        else:
            from scipy._build_utils import get_g77_abi_wrappers, get_sgemv_fix

            import_from_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'scipy._build_utils', None, module_type_store, ['get_g77_abi_wrappers', 'get_sgemv_fix'], [get_g77_abi_wrappers, get_sgemv_fix])

    else:
        # Assigning a type to the variable 'scipy._build_utils' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'scipy._build_utils', import_401200)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/arpack/')
    
    
    # Assigning a Call to a Name (line 11):
    
    # Call to get_info(...): (line 11)
    # Processing the call arguments (line 11)
    str_401203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 26), 'str', 'lapack_opt')
    # Processing the call keyword arguments (line 11)
    kwargs_401204 = {}
    # Getting the type of 'get_info' (line 11)
    get_info_401202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 17), 'get_info', False)
    # Calling get_info(args, kwargs) (line 11)
    get_info_call_result_401205 = invoke(stypy.reporting.localization.Localization(__file__, 11, 17), get_info_401202, *[str_401203], **kwargs_401204)
    
    # Assigning a type to the variable 'lapack_opt' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'lapack_opt', get_info_call_result_401205)
    
    
    # Getting the type of 'lapack_opt' (line 13)
    lapack_opt_401206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 11), 'lapack_opt')
    # Applying the 'not' unary operator (line 13)
    result_not__401207 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 7), 'not', lapack_opt_401206)
    
    # Testing the type of an if condition (line 13)
    if_condition_401208 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 13, 4), result_not__401207)
    # Assigning a type to the variable 'if_condition_401208' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'if_condition_401208', if_condition_401208)
    # SSA begins for if statement (line 13)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to NotFoundError(...): (line 14)
    # Processing the call arguments (line 14)
    str_401210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 28), 'str', 'no lapack/blas resources found')
    # Processing the call keyword arguments (line 14)
    kwargs_401211 = {}
    # Getting the type of 'NotFoundError' (line 14)
    NotFoundError_401209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 14), 'NotFoundError', False)
    # Calling NotFoundError(args, kwargs) (line 14)
    NotFoundError_call_result_401212 = invoke(stypy.reporting.localization.Localization(__file__, 14, 14), NotFoundError_401209, *[str_401210], **kwargs_401211)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 14, 8), NotFoundError_call_result_401212, 'raise parameter', BaseException)
    # SSA join for if statement (line 13)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 16):
    
    # Call to Configuration(...): (line 16)
    # Processing the call arguments (line 16)
    str_401214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 27), 'str', 'arpack')
    # Getting the type of 'parent_package' (line 16)
    parent_package_401215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 37), 'parent_package', False)
    # Getting the type of 'top_path' (line 16)
    top_path_401216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 53), 'top_path', False)
    # Processing the call keyword arguments (line 16)
    kwargs_401217 = {}
    # Getting the type of 'Configuration' (line 16)
    Configuration_401213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 16)
    Configuration_call_result_401218 = invoke(stypy.reporting.localization.Localization(__file__, 16, 13), Configuration_401213, *[str_401214, parent_package_401215, top_path_401216], **kwargs_401217)
    
    # Assigning a type to the variable 'config' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'config', Configuration_call_result_401218)
    
    # Assigning a List to a Name (line 18):
    
    # Obtaining an instance of the builtin type 'list' (line 18)
    list_401219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 18)
    # Adding element type (line 18)
    
    # Call to join(...): (line 18)
    # Processing the call arguments (line 18)
    str_401221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 27), 'str', 'ARPACK')
    str_401222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 36), 'str', 'SRC')
    str_401223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 43), 'str', '*.f')
    # Processing the call keyword arguments (line 18)
    kwargs_401224 = {}
    # Getting the type of 'join' (line 18)
    join_401220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 22), 'join', False)
    # Calling join(args, kwargs) (line 18)
    join_call_result_401225 = invoke(stypy.reporting.localization.Localization(__file__, 18, 22), join_401220, *[str_401221, str_401222, str_401223], **kwargs_401224)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 21), list_401219, join_call_result_401225)
    
    # Assigning a type to the variable 'arpack_sources' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'arpack_sources', list_401219)
    
    # Call to extend(...): (line 19)
    # Processing the call arguments (line 19)
    
    # Obtaining an instance of the builtin type 'list' (line 19)
    list_401228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 26), 'list')
    # Adding type elements to the builtin type 'list' instance (line 19)
    # Adding element type (line 19)
    
    # Call to join(...): (line 19)
    # Processing the call arguments (line 19)
    str_401230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 32), 'str', 'ARPACK')
    str_401231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 41), 'str', 'UTIL')
    str_401232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 49), 'str', '*.f')
    # Processing the call keyword arguments (line 19)
    kwargs_401233 = {}
    # Getting the type of 'join' (line 19)
    join_401229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 27), 'join', False)
    # Calling join(args, kwargs) (line 19)
    join_call_result_401234 = invoke(stypy.reporting.localization.Localization(__file__, 19, 27), join_401229, *[str_401230, str_401231, str_401232], **kwargs_401233)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 26), list_401228, join_call_result_401234)
    
    # Processing the call keyword arguments (line 19)
    kwargs_401235 = {}
    # Getting the type of 'arpack_sources' (line 19)
    arpack_sources_401226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'arpack_sources', False)
    # Obtaining the member 'extend' of a type (line 19)
    extend_401227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 4), arpack_sources_401226, 'extend')
    # Calling extend(args, kwargs) (line 19)
    extend_call_result_401236 = invoke(stypy.reporting.localization.Localization(__file__, 19, 4), extend_401227, *[list_401228], **kwargs_401235)
    
    
    # Getting the type of 'arpack_sources' (line 21)
    arpack_sources_401237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'arpack_sources')
    
    # Call to get_g77_abi_wrappers(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 'lapack_opt' (line 21)
    lapack_opt_401239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 43), 'lapack_opt', False)
    # Processing the call keyword arguments (line 21)
    kwargs_401240 = {}
    # Getting the type of 'get_g77_abi_wrappers' (line 21)
    get_g77_abi_wrappers_401238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 22), 'get_g77_abi_wrappers', False)
    # Calling get_g77_abi_wrappers(args, kwargs) (line 21)
    get_g77_abi_wrappers_call_result_401241 = invoke(stypy.reporting.localization.Localization(__file__, 21, 22), get_g77_abi_wrappers_401238, *[lapack_opt_401239], **kwargs_401240)
    
    # Applying the binary operator '+=' (line 21)
    result_iadd_401242 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 4), '+=', arpack_sources_401237, get_g77_abi_wrappers_call_result_401241)
    # Assigning a type to the variable 'arpack_sources' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'arpack_sources', result_iadd_401242)
    
    
    # Call to add_library(...): (line 23)
    # Processing the call arguments (line 23)
    str_401245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 23), 'str', 'arpack_scipy')
    # Processing the call keyword arguments (line 23)
    # Getting the type of 'arpack_sources' (line 23)
    arpack_sources_401246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 47), 'arpack_sources', False)
    keyword_401247 = arpack_sources_401246
    
    # Obtaining an instance of the builtin type 'list' (line 24)
    list_401248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 36), 'list')
    # Adding type elements to the builtin type 'list' instance (line 24)
    # Adding element type (line 24)
    
    # Call to join(...): (line 24)
    # Processing the call arguments (line 24)
    str_401250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 42), 'str', 'ARPACK')
    str_401251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 52), 'str', 'SRC')
    # Processing the call keyword arguments (line 24)
    kwargs_401252 = {}
    # Getting the type of 'join' (line 24)
    join_401249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 37), 'join', False)
    # Calling join(args, kwargs) (line 24)
    join_call_result_401253 = invoke(stypy.reporting.localization.Localization(__file__, 24, 37), join_401249, *[str_401250, str_401251], **kwargs_401252)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 36), list_401248, join_call_result_401253)
    
    keyword_401254 = list_401248
    kwargs_401255 = {'sources': keyword_401247, 'include_dirs': keyword_401254}
    # Getting the type of 'config' (line 23)
    config_401243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'config', False)
    # Obtaining the member 'add_library' of a type (line 23)
    add_library_401244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 4), config_401243, 'add_library')
    # Calling add_library(args, kwargs) (line 23)
    add_library_call_result_401256 = invoke(stypy.reporting.localization.Localization(__file__, 23, 4), add_library_401244, *[str_401245], **kwargs_401255)
    
    
    # Assigning a List to a Name (line 26):
    
    # Obtaining an instance of the builtin type 'list' (line 26)
    list_401257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 26)
    # Adding element type (line 26)
    str_401258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 19), 'str', 'arpack.pyf.src')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 18), list_401257, str_401258)
    
    # Assigning a type to the variable 'ext_sources' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'ext_sources', list_401257)
    
    # Getting the type of 'ext_sources' (line 27)
    ext_sources_401259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'ext_sources')
    
    # Call to get_sgemv_fix(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'lapack_opt' (line 27)
    lapack_opt_401261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 33), 'lapack_opt', False)
    # Processing the call keyword arguments (line 27)
    kwargs_401262 = {}
    # Getting the type of 'get_sgemv_fix' (line 27)
    get_sgemv_fix_401260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 19), 'get_sgemv_fix', False)
    # Calling get_sgemv_fix(args, kwargs) (line 27)
    get_sgemv_fix_call_result_401263 = invoke(stypy.reporting.localization.Localization(__file__, 27, 19), get_sgemv_fix_401260, *[lapack_opt_401261], **kwargs_401262)
    
    # Applying the binary operator '+=' (line 27)
    result_iadd_401264 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 4), '+=', ext_sources_401259, get_sgemv_fix_call_result_401263)
    # Assigning a type to the variable 'ext_sources' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'ext_sources', result_iadd_401264)
    
    
    # Call to add_extension(...): (line 28)
    # Processing the call arguments (line 28)
    str_401267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 25), 'str', '_arpack')
    # Processing the call keyword arguments (line 28)
    # Getting the type of 'ext_sources' (line 29)
    ext_sources_401268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 33), 'ext_sources', False)
    keyword_401269 = ext_sources_401268
    
    # Obtaining an instance of the builtin type 'list' (line 30)
    list_401270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 35), 'list')
    # Adding type elements to the builtin type 'list' instance (line 30)
    # Adding element type (line 30)
    str_401271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 36), 'str', 'arpack_scipy')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 35), list_401270, str_401271)
    
    keyword_401272 = list_401270
    # Getting the type of 'lapack_opt' (line 31)
    lapack_opt_401273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 36), 'lapack_opt', False)
    keyword_401274 = lapack_opt_401273
    # Getting the type of 'arpack_sources' (line 32)
    arpack_sources_401275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 33), 'arpack_sources', False)
    keyword_401276 = arpack_sources_401275
    kwargs_401277 = {'libraries': keyword_401272, 'sources': keyword_401269, 'depends': keyword_401276, 'extra_info': keyword_401274}
    # Getting the type of 'config' (line 28)
    config_401265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 28)
    add_extension_401266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 4), config_401265, 'add_extension')
    # Calling add_extension(args, kwargs) (line 28)
    add_extension_call_result_401278 = invoke(stypy.reporting.localization.Localization(__file__, 28, 4), add_extension_401266, *[str_401267], **kwargs_401277)
    
    
    # Call to add_data_dir(...): (line 35)
    # Processing the call arguments (line 35)
    str_401281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 24), 'str', 'tests')
    # Processing the call keyword arguments (line 35)
    kwargs_401282 = {}
    # Getting the type of 'config' (line 35)
    config_401279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'config', False)
    # Obtaining the member 'add_data_dir' of a type (line 35)
    add_data_dir_401280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 4), config_401279, 'add_data_dir')
    # Calling add_data_dir(args, kwargs) (line 35)
    add_data_dir_call_result_401283 = invoke(stypy.reporting.localization.Localization(__file__, 35, 4), add_data_dir_401280, *[str_401281], **kwargs_401282)
    
    
    # Call to add_data_files(...): (line 38)
    # Processing the call arguments (line 38)
    str_401286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 26), 'str', 'ARPACK/COPYING')
    # Processing the call keyword arguments (line 38)
    kwargs_401287 = {}
    # Getting the type of 'config' (line 38)
    config_401284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'config', False)
    # Obtaining the member 'add_data_files' of a type (line 38)
    add_data_files_401285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 4), config_401284, 'add_data_files')
    # Calling add_data_files(args, kwargs) (line 38)
    add_data_files_call_result_401288 = invoke(stypy.reporting.localization.Localization(__file__, 38, 4), add_data_files_401285, *[str_401286], **kwargs_401287)
    
    # Getting the type of 'config' (line 40)
    config_401289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'stypy_return_type', config_401289)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 6)
    stypy_return_type_401290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_401290)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_401290

# Assigning a type to the variable 'configuration' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 43, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 43)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/arpack/')
    import_401291 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 43, 4), 'numpy.distutils.core')

    if (type(import_401291) is not StypyTypeError):

        if (import_401291 != 'pyd_module'):
            __import__(import_401291)
            sys_modules_401292 = sys.modules[import_401291]
            import_from_module(stypy.reporting.localization.Localization(__file__, 43, 4), 'numpy.distutils.core', sys_modules_401292.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 43, 4), __file__, sys_modules_401292, sys_modules_401292.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 43, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'numpy.distutils.core', import_401291)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/eigen/arpack/')
    
    
    # Call to setup(...): (line 44)
    # Processing the call keyword arguments (line 44)
    
    # Call to todict(...): (line 44)
    # Processing the call keyword arguments (line 44)
    kwargs_401300 = {}
    
    # Call to configuration(...): (line 44)
    # Processing the call keyword arguments (line 44)
    str_401295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 35), 'str', '')
    keyword_401296 = str_401295
    kwargs_401297 = {'top_path': keyword_401296}
    # Getting the type of 'configuration' (line 44)
    configuration_401294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'configuration', False)
    # Calling configuration(args, kwargs) (line 44)
    configuration_call_result_401298 = invoke(stypy.reporting.localization.Localization(__file__, 44, 12), configuration_401294, *[], **kwargs_401297)
    
    # Obtaining the member 'todict' of a type (line 44)
    todict_401299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 12), configuration_call_result_401298, 'todict')
    # Calling todict(args, kwargs) (line 44)
    todict_call_result_401301 = invoke(stypy.reporting.localization.Localization(__file__, 44, 12), todict_401299, *[], **kwargs_401300)
    
    kwargs_401302 = {'todict_call_result_401301': todict_call_result_401301}
    # Getting the type of 'setup' (line 44)
    setup_401293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 44)
    setup_call_result_401303 = invoke(stypy.reporting.localization.Localization(__file__, 44, 4), setup_401293, *[], **kwargs_401302)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
