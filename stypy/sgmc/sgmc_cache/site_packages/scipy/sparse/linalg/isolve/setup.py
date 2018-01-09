
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
9:     from scipy._build_utils import get_g77_abi_wrappers
10: 
11:     config = Configuration('isolve',parent_package,top_path)
12: 
13:     lapack_opt = get_info('lapack_opt')
14: 
15:     if not lapack_opt:
16:         raise NotFoundError('no lapack/blas resources found')
17: 
18:     # iterative methods
19:     methods = ['BiCGREVCOM.f.src',
20:                'BiCGSTABREVCOM.f.src',
21:                'CGREVCOM.f.src',
22:                'CGSREVCOM.f.src',
23: #               'ChebyREVCOM.f.src',
24:                'GMRESREVCOM.f.src',
25: #               'JacobiREVCOM.f.src',
26:                'QMRREVCOM.f.src',
27: #               'SORREVCOM.f.src'
28:                ]
29: 
30:     Util = ['STOPTEST2.f.src','getbreak.f.src']
31:     sources = Util + methods + ['_iterative.pyf.src']
32:     sources = [join('iterative', x) for x in sources]
33:     sources += get_g77_abi_wrappers(lapack_opt)
34: 
35:     config.add_extension('_iterative',
36:                          sources=sources,
37:                          extra_info=lapack_opt)
38: 
39:     config.add_data_dir('tests')
40: 
41:     return config
42: 
43: 
44: if __name__ == '__main__':
45:     from numpy.distutils.core import setup
46: 
47:     setup(**configuration(top_path='').todict())
48: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from os.path import join' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
import_414217 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path')

if (type(import_414217) is not StypyTypeError):

    if (import_414217 != 'pyd_module'):
        __import__(import_414217)
        sys_modules_414218 = sys.modules[import_414217]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path', sys_modules_414218.module_type_store, module_type_store, ['join'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_414218, sys_modules_414218.module_type_store, module_type_store)
    else:
        from os.path import join

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path', None, module_type_store, ['join'], [join])

else:
    # Assigning a type to the variable 'os.path' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'os.path', import_414217)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')


@norecursion
def configuration(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_414219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 33), 'str', '')
    # Getting the type of 'None' (line 6)
    None_414220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 45), 'None')
    defaults = [str_414219, None_414220]
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
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
    import_414221 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy.distutils.system_info')

    if (type(import_414221) is not StypyTypeError):

        if (import_414221 != 'pyd_module'):
            __import__(import_414221)
            sys_modules_414222 = sys.modules[import_414221]
            import_from_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy.distutils.system_info', sys_modules_414222.module_type_store, module_type_store, ['get_info', 'NotFoundError'])
            nest_module(stypy.reporting.localization.Localization(__file__, 7, 4), __file__, sys_modules_414222, sys_modules_414222.module_type_store, module_type_store)
        else:
            from numpy.distutils.system_info import get_info, NotFoundError

            import_from_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy.distutils.system_info', None, module_type_store, ['get_info', 'NotFoundError'], [get_info, NotFoundError])

    else:
        # Assigning a type to the variable 'numpy.distutils.system_info' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy.distutils.system_info', import_414221)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 4))
    
    # 'from numpy.distutils.misc_util import Configuration' statement (line 8)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
    import_414223 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'numpy.distutils.misc_util')

    if (type(import_414223) is not StypyTypeError):

        if (import_414223 != 'pyd_module'):
            __import__(import_414223)
            sys_modules_414224 = sys.modules[import_414223]
            import_from_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'numpy.distutils.misc_util', sys_modules_414224.module_type_store, module_type_store, ['Configuration'])
            nest_module(stypy.reporting.localization.Localization(__file__, 8, 4), __file__, sys_modules_414224, sys_modules_414224.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration

            import_from_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'numpy.distutils.misc_util', import_414223)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 4))
    
    # 'from scipy._build_utils import get_g77_abi_wrappers' statement (line 9)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
    import_414225 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'scipy._build_utils')

    if (type(import_414225) is not StypyTypeError):

        if (import_414225 != 'pyd_module'):
            __import__(import_414225)
            sys_modules_414226 = sys.modules[import_414225]
            import_from_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'scipy._build_utils', sys_modules_414226.module_type_store, module_type_store, ['get_g77_abi_wrappers'])
            nest_module(stypy.reporting.localization.Localization(__file__, 9, 4), __file__, sys_modules_414226, sys_modules_414226.module_type_store, module_type_store)
        else:
            from scipy._build_utils import get_g77_abi_wrappers

            import_from_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'scipy._build_utils', None, module_type_store, ['get_g77_abi_wrappers'], [get_g77_abi_wrappers])

    else:
        # Assigning a type to the variable 'scipy._build_utils' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'scipy._build_utils', import_414225)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
    
    
    # Assigning a Call to a Name (line 11):
    
    # Call to Configuration(...): (line 11)
    # Processing the call arguments (line 11)
    str_414228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 27), 'str', 'isolve')
    # Getting the type of 'parent_package' (line 11)
    parent_package_414229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 36), 'parent_package', False)
    # Getting the type of 'top_path' (line 11)
    top_path_414230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 51), 'top_path', False)
    # Processing the call keyword arguments (line 11)
    kwargs_414231 = {}
    # Getting the type of 'Configuration' (line 11)
    Configuration_414227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 11)
    Configuration_call_result_414232 = invoke(stypy.reporting.localization.Localization(__file__, 11, 13), Configuration_414227, *[str_414228, parent_package_414229, top_path_414230], **kwargs_414231)
    
    # Assigning a type to the variable 'config' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'config', Configuration_call_result_414232)
    
    # Assigning a Call to a Name (line 13):
    
    # Call to get_info(...): (line 13)
    # Processing the call arguments (line 13)
    str_414234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 26), 'str', 'lapack_opt')
    # Processing the call keyword arguments (line 13)
    kwargs_414235 = {}
    # Getting the type of 'get_info' (line 13)
    get_info_414233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 17), 'get_info', False)
    # Calling get_info(args, kwargs) (line 13)
    get_info_call_result_414236 = invoke(stypy.reporting.localization.Localization(__file__, 13, 17), get_info_414233, *[str_414234], **kwargs_414235)
    
    # Assigning a type to the variable 'lapack_opt' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'lapack_opt', get_info_call_result_414236)
    
    
    # Getting the type of 'lapack_opt' (line 15)
    lapack_opt_414237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 11), 'lapack_opt')
    # Applying the 'not' unary operator (line 15)
    result_not__414238 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 7), 'not', lapack_opt_414237)
    
    # Testing the type of an if condition (line 15)
    if_condition_414239 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 15, 4), result_not__414238)
    # Assigning a type to the variable 'if_condition_414239' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'if_condition_414239', if_condition_414239)
    # SSA begins for if statement (line 15)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to NotFoundError(...): (line 16)
    # Processing the call arguments (line 16)
    str_414241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 28), 'str', 'no lapack/blas resources found')
    # Processing the call keyword arguments (line 16)
    kwargs_414242 = {}
    # Getting the type of 'NotFoundError' (line 16)
    NotFoundError_414240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 14), 'NotFoundError', False)
    # Calling NotFoundError(args, kwargs) (line 16)
    NotFoundError_call_result_414243 = invoke(stypy.reporting.localization.Localization(__file__, 16, 14), NotFoundError_414240, *[str_414241], **kwargs_414242)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 16, 8), NotFoundError_call_result_414243, 'raise parameter', BaseException)
    # SSA join for if statement (line 15)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 19):
    
    # Obtaining an instance of the builtin type 'list' (line 19)
    list_414244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 19)
    # Adding element type (line 19)
    str_414245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 15), 'str', 'BiCGREVCOM.f.src')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 14), list_414244, str_414245)
    # Adding element type (line 19)
    str_414246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 15), 'str', 'BiCGSTABREVCOM.f.src')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 14), list_414244, str_414246)
    # Adding element type (line 19)
    str_414247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 15), 'str', 'CGREVCOM.f.src')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 14), list_414244, str_414247)
    # Adding element type (line 19)
    str_414248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 15), 'str', 'CGSREVCOM.f.src')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 14), list_414244, str_414248)
    # Adding element type (line 19)
    str_414249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 15), 'str', 'GMRESREVCOM.f.src')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 14), list_414244, str_414249)
    # Adding element type (line 19)
    str_414250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 15), 'str', 'QMRREVCOM.f.src')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 14), list_414244, str_414250)
    
    # Assigning a type to the variable 'methods' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'methods', list_414244)
    
    # Assigning a List to a Name (line 30):
    
    # Obtaining an instance of the builtin type 'list' (line 30)
    list_414251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 30)
    # Adding element type (line 30)
    str_414252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 12), 'str', 'STOPTEST2.f.src')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 11), list_414251, str_414252)
    # Adding element type (line 30)
    str_414253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 30), 'str', 'getbreak.f.src')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 11), list_414251, str_414253)
    
    # Assigning a type to the variable 'Util' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'Util', list_414251)
    
    # Assigning a BinOp to a Name (line 31):
    # Getting the type of 'Util' (line 31)
    Util_414254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 14), 'Util')
    # Getting the type of 'methods' (line 31)
    methods_414255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 21), 'methods')
    # Applying the binary operator '+' (line 31)
    result_add_414256 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 14), '+', Util_414254, methods_414255)
    
    
    # Obtaining an instance of the builtin type 'list' (line 31)
    list_414257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 31), 'list')
    # Adding type elements to the builtin type 'list' instance (line 31)
    # Adding element type (line 31)
    str_414258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 32), 'str', '_iterative.pyf.src')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 31), list_414257, str_414258)
    
    # Applying the binary operator '+' (line 31)
    result_add_414259 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 29), '+', result_add_414256, list_414257)
    
    # Assigning a type to the variable 'sources' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'sources', result_add_414259)
    
    # Assigning a ListComp to a Name (line 32):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'sources' (line 32)
    sources_414265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 45), 'sources')
    comprehension_414266 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 15), sources_414265)
    # Assigning a type to the variable 'x' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 15), 'x', comprehension_414266)
    
    # Call to join(...): (line 32)
    # Processing the call arguments (line 32)
    str_414261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 20), 'str', 'iterative')
    # Getting the type of 'x' (line 32)
    x_414262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 33), 'x', False)
    # Processing the call keyword arguments (line 32)
    kwargs_414263 = {}
    # Getting the type of 'join' (line 32)
    join_414260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 15), 'join', False)
    # Calling join(args, kwargs) (line 32)
    join_call_result_414264 = invoke(stypy.reporting.localization.Localization(__file__, 32, 15), join_414260, *[str_414261, x_414262], **kwargs_414263)
    
    list_414267 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 15), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 15), list_414267, join_call_result_414264)
    # Assigning a type to the variable 'sources' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'sources', list_414267)
    
    # Getting the type of 'sources' (line 33)
    sources_414268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'sources')
    
    # Call to get_g77_abi_wrappers(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'lapack_opt' (line 33)
    lapack_opt_414270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 36), 'lapack_opt', False)
    # Processing the call keyword arguments (line 33)
    kwargs_414271 = {}
    # Getting the type of 'get_g77_abi_wrappers' (line 33)
    get_g77_abi_wrappers_414269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 15), 'get_g77_abi_wrappers', False)
    # Calling get_g77_abi_wrappers(args, kwargs) (line 33)
    get_g77_abi_wrappers_call_result_414272 = invoke(stypy.reporting.localization.Localization(__file__, 33, 15), get_g77_abi_wrappers_414269, *[lapack_opt_414270], **kwargs_414271)
    
    # Applying the binary operator '+=' (line 33)
    result_iadd_414273 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 4), '+=', sources_414268, get_g77_abi_wrappers_call_result_414272)
    # Assigning a type to the variable 'sources' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'sources', result_iadd_414273)
    
    
    # Call to add_extension(...): (line 35)
    # Processing the call arguments (line 35)
    str_414276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 25), 'str', '_iterative')
    # Processing the call keyword arguments (line 35)
    # Getting the type of 'sources' (line 36)
    sources_414277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 33), 'sources', False)
    keyword_414278 = sources_414277
    # Getting the type of 'lapack_opt' (line 37)
    lapack_opt_414279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 36), 'lapack_opt', False)
    keyword_414280 = lapack_opt_414279
    kwargs_414281 = {'sources': keyword_414278, 'extra_info': keyword_414280}
    # Getting the type of 'config' (line 35)
    config_414274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 35)
    add_extension_414275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 4), config_414274, 'add_extension')
    # Calling add_extension(args, kwargs) (line 35)
    add_extension_call_result_414282 = invoke(stypy.reporting.localization.Localization(__file__, 35, 4), add_extension_414275, *[str_414276], **kwargs_414281)
    
    
    # Call to add_data_dir(...): (line 39)
    # Processing the call arguments (line 39)
    str_414285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 24), 'str', 'tests')
    # Processing the call keyword arguments (line 39)
    kwargs_414286 = {}
    # Getting the type of 'config' (line 39)
    config_414283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'config', False)
    # Obtaining the member 'add_data_dir' of a type (line 39)
    add_data_dir_414284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 4), config_414283, 'add_data_dir')
    # Calling add_data_dir(args, kwargs) (line 39)
    add_data_dir_call_result_414287 = invoke(stypy.reporting.localization.Localization(__file__, 39, 4), add_data_dir_414284, *[str_414285], **kwargs_414286)
    
    # Getting the type of 'config' (line 41)
    config_414288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type', config_414288)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 6)
    stypy_return_type_414289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_414289)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_414289

# Assigning a type to the variable 'configuration' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 45, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 45)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
    import_414290 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 45, 4), 'numpy.distutils.core')

    if (type(import_414290) is not StypyTypeError):

        if (import_414290 != 'pyd_module'):
            __import__(import_414290)
            sys_modules_414291 = sys.modules[import_414290]
            import_from_module(stypy.reporting.localization.Localization(__file__, 45, 4), 'numpy.distutils.core', sys_modules_414291.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 45, 4), __file__, sys_modules_414291, sys_modules_414291.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 45, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'numpy.distutils.core', import_414290)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/sparse/linalg/isolve/')
    
    
    # Call to setup(...): (line 47)
    # Processing the call keyword arguments (line 47)
    
    # Call to todict(...): (line 47)
    # Processing the call keyword arguments (line 47)
    kwargs_414299 = {}
    
    # Call to configuration(...): (line 47)
    # Processing the call keyword arguments (line 47)
    str_414294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 35), 'str', '')
    keyword_414295 = str_414294
    kwargs_414296 = {'top_path': keyword_414295}
    # Getting the type of 'configuration' (line 47)
    configuration_414293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'configuration', False)
    # Calling configuration(args, kwargs) (line 47)
    configuration_call_result_414297 = invoke(stypy.reporting.localization.Localization(__file__, 47, 12), configuration_414293, *[], **kwargs_414296)
    
    # Obtaining the member 'todict' of a type (line 47)
    todict_414298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 12), configuration_call_result_414297, 'todict')
    # Calling todict(args, kwargs) (line 47)
    todict_call_result_414300 = invoke(stypy.reporting.localization.Localization(__file__, 47, 12), todict_414298, *[], **kwargs_414299)
    
    kwargs_414301 = {'todict_call_result_414300': todict_call_result_414300}
    # Getting the type of 'setup' (line 47)
    setup_414292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 47)
    setup_call_result_414302 = invoke(stypy.reporting.localization.Localization(__file__, 47, 4), setup_414292, *[], **kwargs_414301)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
