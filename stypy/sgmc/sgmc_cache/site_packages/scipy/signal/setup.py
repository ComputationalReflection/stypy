
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: from scipy._build_utils import numpy_nodepr_api
4: 
5: 
6: def configuration(parent_package='', top_path=None):
7:     from numpy.distutils.misc_util import Configuration
8: 
9:     config = Configuration('signal', parent_package, top_path)
10: 
11:     config.add_data_dir('tests')
12: 
13:     config.add_extension('sigtools',
14:                          sources=['sigtoolsmodule.c', 'firfilter.c',
15:                                   'medianfilter.c', 'lfilter.c.src',
16:                                   'correlate_nd.c.src'],
17:                          depends=['sigtools.h'],
18:                          include_dirs=['.'],
19:                          **numpy_nodepr_api)
20: 
21:     config.add_extension('_spectral', sources=['_spectral.c'])
22:     config.add_extension('_max_len_seq_inner', sources=['_max_len_seq_inner.c'])
23:     config.add_extension('_upfirdn_apply', sources=['_upfirdn_apply.c'])
24:     spline_src = ['splinemodule.c', 'S_bspline_util.c', 'D_bspline_util.c',
25:                   'C_bspline_util.c', 'Z_bspline_util.c', 'bspline_util.c']
26:     config.add_extension('spline', sources=spline_src, **numpy_nodepr_api)
27: 
28:     return config
29: 
30: 
31: if __name__ == '__main__':
32:     from numpy.distutils.core import setup
33:     setup(**configuration(top_path='').todict())
34: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from scipy._build_utils import numpy_nodepr_api' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
import_274216 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy._build_utils')

if (type(import_274216) is not StypyTypeError):

    if (import_274216 != 'pyd_module'):
        __import__(import_274216)
        sys_modules_274217 = sys.modules[import_274216]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy._build_utils', sys_modules_274217.module_type_store, module_type_store, ['numpy_nodepr_api'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_274217, sys_modules_274217.module_type_store, module_type_store)
    else:
        from scipy._build_utils import numpy_nodepr_api

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy._build_utils', None, module_type_store, ['numpy_nodepr_api'], [numpy_nodepr_api])

else:
    # Assigning a type to the variable 'scipy._build_utils' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy._build_utils', import_274216)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')


@norecursion
def configuration(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_274218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 33), 'str', '')
    # Getting the type of 'None' (line 6)
    None_274219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 46), 'None')
    defaults = [str_274218, None_274219]
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
    
    # 'from numpy.distutils.misc_util import Configuration' statement (line 7)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
    import_274220 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy.distutils.misc_util')

    if (type(import_274220) is not StypyTypeError):

        if (import_274220 != 'pyd_module'):
            __import__(import_274220)
            sys_modules_274221 = sys.modules[import_274220]
            import_from_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy.distutils.misc_util', sys_modules_274221.module_type_store, module_type_store, ['Configuration'])
            nest_module(stypy.reporting.localization.Localization(__file__, 7, 4), __file__, sys_modules_274221, sys_modules_274221.module_type_store, module_type_store)
        else:
            from numpy.distutils.misc_util import Configuration

            import_from_module(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

    else:
        # Assigning a type to the variable 'numpy.distutils.misc_util' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'numpy.distutils.misc_util', import_274220)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')
    
    
    # Assigning a Call to a Name (line 9):
    
    # Call to Configuration(...): (line 9)
    # Processing the call arguments (line 9)
    str_274223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 27), 'str', 'signal')
    # Getting the type of 'parent_package' (line 9)
    parent_package_274224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 37), 'parent_package', False)
    # Getting the type of 'top_path' (line 9)
    top_path_274225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 53), 'top_path', False)
    # Processing the call keyword arguments (line 9)
    kwargs_274226 = {}
    # Getting the type of 'Configuration' (line 9)
    Configuration_274222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 9)
    Configuration_call_result_274227 = invoke(stypy.reporting.localization.Localization(__file__, 9, 13), Configuration_274222, *[str_274223, parent_package_274224, top_path_274225], **kwargs_274226)
    
    # Assigning a type to the variable 'config' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'config', Configuration_call_result_274227)
    
    # Call to add_data_dir(...): (line 11)
    # Processing the call arguments (line 11)
    str_274230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 24), 'str', 'tests')
    # Processing the call keyword arguments (line 11)
    kwargs_274231 = {}
    # Getting the type of 'config' (line 11)
    config_274228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'config', False)
    # Obtaining the member 'add_data_dir' of a type (line 11)
    add_data_dir_274229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 4), config_274228, 'add_data_dir')
    # Calling add_data_dir(args, kwargs) (line 11)
    add_data_dir_call_result_274232 = invoke(stypy.reporting.localization.Localization(__file__, 11, 4), add_data_dir_274229, *[str_274230], **kwargs_274231)
    
    
    # Call to add_extension(...): (line 13)
    # Processing the call arguments (line 13)
    str_274235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 25), 'str', 'sigtools')
    # Processing the call keyword arguments (line 13)
    
    # Obtaining an instance of the builtin type 'list' (line 14)
    list_274236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 14)
    # Adding element type (line 14)
    str_274237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 34), 'str', 'sigtoolsmodule.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 33), list_274236, str_274237)
    # Adding element type (line 14)
    str_274238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 54), 'str', 'firfilter.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 33), list_274236, str_274238)
    # Adding element type (line 14)
    str_274239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 34), 'str', 'medianfilter.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 33), list_274236, str_274239)
    # Adding element type (line 14)
    str_274240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 52), 'str', 'lfilter.c.src')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 33), list_274236, str_274240)
    # Adding element type (line 14)
    str_274241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 34), 'str', 'correlate_nd.c.src')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 33), list_274236, str_274241)
    
    keyword_274242 = list_274236
    
    # Obtaining an instance of the builtin type 'list' (line 17)
    list_274243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 17)
    # Adding element type (line 17)
    str_274244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 34), 'str', 'sigtools.h')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 33), list_274243, str_274244)
    
    keyword_274245 = list_274243
    
    # Obtaining an instance of the builtin type 'list' (line 18)
    list_274246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 18)
    # Adding element type (line 18)
    str_274247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 39), 'str', '.')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 38), list_274246, str_274247)
    
    keyword_274248 = list_274246
    # Getting the type of 'numpy_nodepr_api' (line 19)
    numpy_nodepr_api_274249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 27), 'numpy_nodepr_api', False)
    kwargs_274250 = {'sources': keyword_274242, 'depends': keyword_274245, 'numpy_nodepr_api_274249': numpy_nodepr_api_274249, 'include_dirs': keyword_274248}
    # Getting the type of 'config' (line 13)
    config_274233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 13)
    add_extension_274234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 4), config_274233, 'add_extension')
    # Calling add_extension(args, kwargs) (line 13)
    add_extension_call_result_274251 = invoke(stypy.reporting.localization.Localization(__file__, 13, 4), add_extension_274234, *[str_274235], **kwargs_274250)
    
    
    # Call to add_extension(...): (line 21)
    # Processing the call arguments (line 21)
    str_274254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 25), 'str', '_spectral')
    # Processing the call keyword arguments (line 21)
    
    # Obtaining an instance of the builtin type 'list' (line 21)
    list_274255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 46), 'list')
    # Adding type elements to the builtin type 'list' instance (line 21)
    # Adding element type (line 21)
    str_274256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 47), 'str', '_spectral.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 46), list_274255, str_274256)
    
    keyword_274257 = list_274255
    kwargs_274258 = {'sources': keyword_274257}
    # Getting the type of 'config' (line 21)
    config_274252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 21)
    add_extension_274253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 4), config_274252, 'add_extension')
    # Calling add_extension(args, kwargs) (line 21)
    add_extension_call_result_274259 = invoke(stypy.reporting.localization.Localization(__file__, 21, 4), add_extension_274253, *[str_274254], **kwargs_274258)
    
    
    # Call to add_extension(...): (line 22)
    # Processing the call arguments (line 22)
    str_274262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 25), 'str', '_max_len_seq_inner')
    # Processing the call keyword arguments (line 22)
    
    # Obtaining an instance of the builtin type 'list' (line 22)
    list_274263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 55), 'list')
    # Adding type elements to the builtin type 'list' instance (line 22)
    # Adding element type (line 22)
    str_274264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 56), 'str', '_max_len_seq_inner.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 55), list_274263, str_274264)
    
    keyword_274265 = list_274263
    kwargs_274266 = {'sources': keyword_274265}
    # Getting the type of 'config' (line 22)
    config_274260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 22)
    add_extension_274261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 4), config_274260, 'add_extension')
    # Calling add_extension(args, kwargs) (line 22)
    add_extension_call_result_274267 = invoke(stypy.reporting.localization.Localization(__file__, 22, 4), add_extension_274261, *[str_274262], **kwargs_274266)
    
    
    # Call to add_extension(...): (line 23)
    # Processing the call arguments (line 23)
    str_274270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 25), 'str', '_upfirdn_apply')
    # Processing the call keyword arguments (line 23)
    
    # Obtaining an instance of the builtin type 'list' (line 23)
    list_274271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 51), 'list')
    # Adding type elements to the builtin type 'list' instance (line 23)
    # Adding element type (line 23)
    str_274272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 52), 'str', '_upfirdn_apply.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 51), list_274271, str_274272)
    
    keyword_274273 = list_274271
    kwargs_274274 = {'sources': keyword_274273}
    # Getting the type of 'config' (line 23)
    config_274268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 23)
    add_extension_274269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 4), config_274268, 'add_extension')
    # Calling add_extension(args, kwargs) (line 23)
    add_extension_call_result_274275 = invoke(stypy.reporting.localization.Localization(__file__, 23, 4), add_extension_274269, *[str_274270], **kwargs_274274)
    
    
    # Assigning a List to a Name (line 24):
    
    # Obtaining an instance of the builtin type 'list' (line 24)
    list_274276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 24)
    # Adding element type (line 24)
    str_274277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 18), 'str', 'splinemodule.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 17), list_274276, str_274277)
    # Adding element type (line 24)
    str_274278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 36), 'str', 'S_bspline_util.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 17), list_274276, str_274278)
    # Adding element type (line 24)
    str_274279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 56), 'str', 'D_bspline_util.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 17), list_274276, str_274279)
    # Adding element type (line 24)
    str_274280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 18), 'str', 'C_bspline_util.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 17), list_274276, str_274280)
    # Adding element type (line 24)
    str_274281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 38), 'str', 'Z_bspline_util.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 17), list_274276, str_274281)
    # Adding element type (line 24)
    str_274282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 58), 'str', 'bspline_util.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 17), list_274276, str_274282)
    
    # Assigning a type to the variable 'spline_src' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'spline_src', list_274276)
    
    # Call to add_extension(...): (line 26)
    # Processing the call arguments (line 26)
    str_274285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 25), 'str', 'spline')
    # Processing the call keyword arguments (line 26)
    # Getting the type of 'spline_src' (line 26)
    spline_src_274286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 43), 'spline_src', False)
    keyword_274287 = spline_src_274286
    # Getting the type of 'numpy_nodepr_api' (line 26)
    numpy_nodepr_api_274288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 57), 'numpy_nodepr_api', False)
    kwargs_274289 = {'sources': keyword_274287, 'numpy_nodepr_api_274288': numpy_nodepr_api_274288}
    # Getting the type of 'config' (line 26)
    config_274283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'config', False)
    # Obtaining the member 'add_extension' of a type (line 26)
    add_extension_274284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 4), config_274283, 'add_extension')
    # Calling add_extension(args, kwargs) (line 26)
    add_extension_call_result_274290 = invoke(stypy.reporting.localization.Localization(__file__, 26, 4), add_extension_274284, *[str_274285], **kwargs_274289)
    
    # Getting the type of 'config' (line 28)
    config_274291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'stypy_return_type', config_274291)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 6)
    stypy_return_type_274292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_274292)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_274292

# Assigning a type to the variable 'configuration' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 32, 4))
    
    # 'from numpy.distutils.core import setup' statement (line 32)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/signal/')
    import_274293 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 32, 4), 'numpy.distutils.core')

    if (type(import_274293) is not StypyTypeError):

        if (import_274293 != 'pyd_module'):
            __import__(import_274293)
            sys_modules_274294 = sys.modules[import_274293]
            import_from_module(stypy.reporting.localization.Localization(__file__, 32, 4), 'numpy.distutils.core', sys_modules_274294.module_type_store, module_type_store, ['setup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 32, 4), __file__, sys_modules_274294, sys_modules_274294.module_type_store, module_type_store)
        else:
            from numpy.distutils.core import setup

            import_from_module(stypy.reporting.localization.Localization(__file__, 32, 4), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

    else:
        # Assigning a type to the variable 'numpy.distutils.core' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'numpy.distutils.core', import_274293)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/signal/')
    
    
    # Call to setup(...): (line 33)
    # Processing the call keyword arguments (line 33)
    
    # Call to todict(...): (line 33)
    # Processing the call keyword arguments (line 33)
    kwargs_274302 = {}
    
    # Call to configuration(...): (line 33)
    # Processing the call keyword arguments (line 33)
    str_274297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 35), 'str', '')
    keyword_274298 = str_274297
    kwargs_274299 = {'top_path': keyword_274298}
    # Getting the type of 'configuration' (line 33)
    configuration_274296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'configuration', False)
    # Calling configuration(args, kwargs) (line 33)
    configuration_call_result_274300 = invoke(stypy.reporting.localization.Localization(__file__, 33, 12), configuration_274296, *[], **kwargs_274299)
    
    # Obtaining the member 'todict' of a type (line 33)
    todict_274301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 12), configuration_call_result_274300, 'todict')
    # Calling todict(args, kwargs) (line 33)
    todict_call_result_274303 = invoke(stypy.reporting.localization.Localization(__file__, 33, 12), todict_274301, *[], **kwargs_274302)
    
    kwargs_274304 = {'todict_call_result_274303': todict_call_result_274303}
    # Getting the type of 'setup' (line 33)
    setup_274295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 33)
    setup_call_result_274305 = invoke(stypy.reporting.localization.Localization(__file__, 33, 4), setup_274295, *[], **kwargs_274304)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
