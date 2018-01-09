
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import (absolute_import, division, print_function,
2:                         unicode_literals)
3: 
4: import pytest
5: 
6: import matplotlib
7: 
8: 
9: def pytest_configure(config):
10:     matplotlib.use('agg')
11:     matplotlib._called_from_pytest = True
12:     matplotlib._init_tests()
13: 
14: 
15: def pytest_unconfigure(config):
16:     matplotlib._called_from_pytest = False
17: 
18: 
19: @pytest.fixture(autouse=True)
20: def mpl_test_settings(request):
21:     from matplotlib.testing.decorators import _do_cleanup
22: 
23:     original_units_registry = matplotlib.units.registry.copy()
24:     original_settings = matplotlib.rcParams.copy()
25: 
26:     backend = None
27:     backend_marker = request.keywords.get('backend')
28:     if backend_marker is not None:
29:         assert len(backend_marker.args) == 1, \
30:             "Marker 'backend' must specify 1 backend."
31:         backend = backend_marker.args[0]
32:         prev_backend = matplotlib.get_backend()
33: 
34:     style = '_classic_test'  # Default of cleanup and image_comparison too.
35:     style_marker = request.keywords.get('style')
36:     if style_marker is not None:
37:         assert len(style_marker.args) == 1, \
38:             "Marker 'style' must specify 1 style."
39:         style = style_marker.args[0]
40: 
41:     matplotlib.testing.setup()
42:     if backend is not None:
43:         # This import must come after setup() so it doesn't load the default
44:         # backend prematurely.
45:         import matplotlib.pyplot as plt
46:         plt.switch_backend(backend)
47:     matplotlib.style.use(style)
48:     try:
49:         yield
50:     finally:
51:         if backend is not None:
52:             plt.switch_backend(prev_backend)
53:         _do_cleanup(original_units_registry,
54:                     original_settings)
55: 
56: 
57: @pytest.fixture
58: def mpl_image_comparison_parameters(request, extension):
59:     # This fixture is applied automatically by the image_comparison decorator.
60:     #
61:     # The sole purpose of this fixture is to provide an indirect method of
62:     # obtaining parameters *without* modifying the decorated function
63:     # signature. In this way, the function signature can stay the same and
64:     # pytest won't get confused.
65:     # We annotate the decorated function with any parameters captured by this
66:     # fixture so that they can be used by the wrapper in image_comparison.
67:     baseline_images = request.keywords['baseline_images'].args[0]
68:     if baseline_images is None:
69:         # Allow baseline image list to be produced on the fly based on current
70:         # parametrization.
71:         baseline_images = request.getfixturevalue('baseline_images')
72: 
73:     func = request.function
74:     func.__wrapped__.parameters = (baseline_images, extension)
75:     try:
76:         yield
77:     finally:
78:         delattr(func.__wrapped__, 'parameters')
79: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import pytest' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
import_289891 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'pytest')

if (type(import_289891) is not StypyTypeError):

    if (import_289891 != 'pyd_module'):
        __import__(import_289891)
        sys_modules_289892 = sys.modules[import_289891]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'pytest', sys_modules_289892.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'pytest', import_289891)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import matplotlib' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
import_289893 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib')

if (type(import_289893) is not StypyTypeError):

    if (import_289893 != 'pyd_module'):
        __import__(import_289893)
        sys_modules_289894 = sys.modules[import_289893]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib', sys_modules_289894.module_type_store, module_type_store)
    else:
        import matplotlib

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib', matplotlib, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'matplotlib', import_289893)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')


@norecursion
def pytest_configure(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'pytest_configure'
    module_type_store = module_type_store.open_function_context('pytest_configure', 9, 0, False)
    
    # Passed parameters checking function
    pytest_configure.stypy_localization = localization
    pytest_configure.stypy_type_of_self = None
    pytest_configure.stypy_type_store = module_type_store
    pytest_configure.stypy_function_name = 'pytest_configure'
    pytest_configure.stypy_param_names_list = ['config']
    pytest_configure.stypy_varargs_param_name = None
    pytest_configure.stypy_kwargs_param_name = None
    pytest_configure.stypy_call_defaults = defaults
    pytest_configure.stypy_call_varargs = varargs
    pytest_configure.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'pytest_configure', ['config'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'pytest_configure', localization, ['config'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'pytest_configure(...)' code ##################

    
    # Call to use(...): (line 10)
    # Processing the call arguments (line 10)
    unicode_289897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 19), 'unicode', u'agg')
    # Processing the call keyword arguments (line 10)
    kwargs_289898 = {}
    # Getting the type of 'matplotlib' (line 10)
    matplotlib_289895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'matplotlib', False)
    # Obtaining the member 'use' of a type (line 10)
    use_289896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 4), matplotlib_289895, 'use')
    # Calling use(args, kwargs) (line 10)
    use_call_result_289899 = invoke(stypy.reporting.localization.Localization(__file__, 10, 4), use_289896, *[unicode_289897], **kwargs_289898)
    
    
    # Assigning a Name to a Attribute (line 11):
    # Getting the type of 'True' (line 11)
    True_289900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 37), 'True')
    # Getting the type of 'matplotlib' (line 11)
    matplotlib_289901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'matplotlib')
    # Setting the type of the member '_called_from_pytest' of a type (line 11)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 4), matplotlib_289901, '_called_from_pytest', True_289900)
    
    # Call to _init_tests(...): (line 12)
    # Processing the call keyword arguments (line 12)
    kwargs_289904 = {}
    # Getting the type of 'matplotlib' (line 12)
    matplotlib_289902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'matplotlib', False)
    # Obtaining the member '_init_tests' of a type (line 12)
    _init_tests_289903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 4), matplotlib_289902, '_init_tests')
    # Calling _init_tests(args, kwargs) (line 12)
    _init_tests_call_result_289905 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), _init_tests_289903, *[], **kwargs_289904)
    
    
    # ################# End of 'pytest_configure(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'pytest_configure' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_289906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_289906)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'pytest_configure'
    return stypy_return_type_289906

# Assigning a type to the variable 'pytest_configure' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest_configure', pytest_configure)

@norecursion
def pytest_unconfigure(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'pytest_unconfigure'
    module_type_store = module_type_store.open_function_context('pytest_unconfigure', 15, 0, False)
    
    # Passed parameters checking function
    pytest_unconfigure.stypy_localization = localization
    pytest_unconfigure.stypy_type_of_self = None
    pytest_unconfigure.stypy_type_store = module_type_store
    pytest_unconfigure.stypy_function_name = 'pytest_unconfigure'
    pytest_unconfigure.stypy_param_names_list = ['config']
    pytest_unconfigure.stypy_varargs_param_name = None
    pytest_unconfigure.stypy_kwargs_param_name = None
    pytest_unconfigure.stypy_call_defaults = defaults
    pytest_unconfigure.stypy_call_varargs = varargs
    pytest_unconfigure.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'pytest_unconfigure', ['config'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'pytest_unconfigure', localization, ['config'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'pytest_unconfigure(...)' code ##################

    
    # Assigning a Name to a Attribute (line 16):
    # Getting the type of 'False' (line 16)
    False_289907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 37), 'False')
    # Getting the type of 'matplotlib' (line 16)
    matplotlib_289908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'matplotlib')
    # Setting the type of the member '_called_from_pytest' of a type (line 16)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 4), matplotlib_289908, '_called_from_pytest', False_289907)
    
    # ################# End of 'pytest_unconfigure(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'pytest_unconfigure' in the type store
    # Getting the type of 'stypy_return_type' (line 15)
    stypy_return_type_289909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_289909)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'pytest_unconfigure'
    return stypy_return_type_289909

# Assigning a type to the variable 'pytest_unconfigure' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'pytest_unconfigure', pytest_unconfigure)

@norecursion
def mpl_test_settings(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'mpl_test_settings'
    module_type_store = module_type_store.open_function_context('mpl_test_settings', 19, 0, False)
    
    # Passed parameters checking function
    mpl_test_settings.stypy_localization = localization
    mpl_test_settings.stypy_type_of_self = None
    mpl_test_settings.stypy_type_store = module_type_store
    mpl_test_settings.stypy_function_name = 'mpl_test_settings'
    mpl_test_settings.stypy_param_names_list = ['request']
    mpl_test_settings.stypy_varargs_param_name = None
    mpl_test_settings.stypy_kwargs_param_name = None
    mpl_test_settings.stypy_call_defaults = defaults
    mpl_test_settings.stypy_call_varargs = varargs
    mpl_test_settings.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'mpl_test_settings', ['request'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'mpl_test_settings', localization, ['request'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'mpl_test_settings(...)' code ##################

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 4))
    
    # 'from matplotlib.testing.decorators import _do_cleanup' statement (line 21)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
    import_289910 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 4), 'matplotlib.testing.decorators')

    if (type(import_289910) is not StypyTypeError):

        if (import_289910 != 'pyd_module'):
            __import__(import_289910)
            sys_modules_289911 = sys.modules[import_289910]
            import_from_module(stypy.reporting.localization.Localization(__file__, 21, 4), 'matplotlib.testing.decorators', sys_modules_289911.module_type_store, module_type_store, ['_do_cleanup'])
            nest_module(stypy.reporting.localization.Localization(__file__, 21, 4), __file__, sys_modules_289911, sys_modules_289911.module_type_store, module_type_store)
        else:
            from matplotlib.testing.decorators import _do_cleanup

            import_from_module(stypy.reporting.localization.Localization(__file__, 21, 4), 'matplotlib.testing.decorators', None, module_type_store, ['_do_cleanup'], [_do_cleanup])

    else:
        # Assigning a type to the variable 'matplotlib.testing.decorators' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'matplotlib.testing.decorators', import_289910)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')
    
    
    # Assigning a Call to a Name (line 23):
    
    # Call to copy(...): (line 23)
    # Processing the call keyword arguments (line 23)
    kwargs_289916 = {}
    # Getting the type of 'matplotlib' (line 23)
    matplotlib_289912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 30), 'matplotlib', False)
    # Obtaining the member 'units' of a type (line 23)
    units_289913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 30), matplotlib_289912, 'units')
    # Obtaining the member 'registry' of a type (line 23)
    registry_289914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 30), units_289913, 'registry')
    # Obtaining the member 'copy' of a type (line 23)
    copy_289915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 30), registry_289914, 'copy')
    # Calling copy(args, kwargs) (line 23)
    copy_call_result_289917 = invoke(stypy.reporting.localization.Localization(__file__, 23, 30), copy_289915, *[], **kwargs_289916)
    
    # Assigning a type to the variable 'original_units_registry' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'original_units_registry', copy_call_result_289917)
    
    # Assigning a Call to a Name (line 24):
    
    # Call to copy(...): (line 24)
    # Processing the call keyword arguments (line 24)
    kwargs_289921 = {}
    # Getting the type of 'matplotlib' (line 24)
    matplotlib_289918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 24), 'matplotlib', False)
    # Obtaining the member 'rcParams' of a type (line 24)
    rcParams_289919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 24), matplotlib_289918, 'rcParams')
    # Obtaining the member 'copy' of a type (line 24)
    copy_289920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 24), rcParams_289919, 'copy')
    # Calling copy(args, kwargs) (line 24)
    copy_call_result_289922 = invoke(stypy.reporting.localization.Localization(__file__, 24, 24), copy_289920, *[], **kwargs_289921)
    
    # Assigning a type to the variable 'original_settings' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'original_settings', copy_call_result_289922)
    
    # Assigning a Name to a Name (line 26):
    # Getting the type of 'None' (line 26)
    None_289923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 14), 'None')
    # Assigning a type to the variable 'backend' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'backend', None_289923)
    
    # Assigning a Call to a Name (line 27):
    
    # Call to get(...): (line 27)
    # Processing the call arguments (line 27)
    unicode_289927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 42), 'unicode', u'backend')
    # Processing the call keyword arguments (line 27)
    kwargs_289928 = {}
    # Getting the type of 'request' (line 27)
    request_289924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 21), 'request', False)
    # Obtaining the member 'keywords' of a type (line 27)
    keywords_289925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 21), request_289924, 'keywords')
    # Obtaining the member 'get' of a type (line 27)
    get_289926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 21), keywords_289925, 'get')
    # Calling get(args, kwargs) (line 27)
    get_call_result_289929 = invoke(stypy.reporting.localization.Localization(__file__, 27, 21), get_289926, *[unicode_289927], **kwargs_289928)
    
    # Assigning a type to the variable 'backend_marker' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'backend_marker', get_call_result_289929)
    
    # Type idiom detected: calculating its left and rigth part (line 28)
    # Getting the type of 'backend_marker' (line 28)
    backend_marker_289930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'backend_marker')
    # Getting the type of 'None' (line 28)
    None_289931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 29), 'None')
    
    (may_be_289932, more_types_in_union_289933) = may_not_be_none(backend_marker_289930, None_289931)

    if may_be_289932:

        if more_types_in_union_289933:
            # Runtime conditional SSA (line 28)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Evaluating assert statement condition
        
        
        # Call to len(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'backend_marker' (line 29)
        backend_marker_289935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 19), 'backend_marker', False)
        # Obtaining the member 'args' of a type (line 29)
        args_289936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 19), backend_marker_289935, 'args')
        # Processing the call keyword arguments (line 29)
        kwargs_289937 = {}
        # Getting the type of 'len' (line 29)
        len_289934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 15), 'len', False)
        # Calling len(args, kwargs) (line 29)
        len_call_result_289938 = invoke(stypy.reporting.localization.Localization(__file__, 29, 15), len_289934, *[args_289936], **kwargs_289937)
        
        int_289939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 43), 'int')
        # Applying the binary operator '==' (line 29)
        result_eq_289940 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 15), '==', len_call_result_289938, int_289939)
        
        
        # Assigning a Subscript to a Name (line 31):
        
        # Obtaining the type of the subscript
        int_289941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 38), 'int')
        # Getting the type of 'backend_marker' (line 31)
        backend_marker_289942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 18), 'backend_marker')
        # Obtaining the member 'args' of a type (line 31)
        args_289943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 18), backend_marker_289942, 'args')
        # Obtaining the member '__getitem__' of a type (line 31)
        getitem___289944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 18), args_289943, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 31)
        subscript_call_result_289945 = invoke(stypy.reporting.localization.Localization(__file__, 31, 18), getitem___289944, int_289941)
        
        # Assigning a type to the variable 'backend' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'backend', subscript_call_result_289945)
        
        # Assigning a Call to a Name (line 32):
        
        # Call to get_backend(...): (line 32)
        # Processing the call keyword arguments (line 32)
        kwargs_289948 = {}
        # Getting the type of 'matplotlib' (line 32)
        matplotlib_289946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 23), 'matplotlib', False)
        # Obtaining the member 'get_backend' of a type (line 32)
        get_backend_289947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 23), matplotlib_289946, 'get_backend')
        # Calling get_backend(args, kwargs) (line 32)
        get_backend_call_result_289949 = invoke(stypy.reporting.localization.Localization(__file__, 32, 23), get_backend_289947, *[], **kwargs_289948)
        
        # Assigning a type to the variable 'prev_backend' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'prev_backend', get_backend_call_result_289949)

        if more_types_in_union_289933:
            # SSA join for if statement (line 28)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Str to a Name (line 34):
    unicode_289950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 12), 'unicode', u'_classic_test')
    # Assigning a type to the variable 'style' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'style', unicode_289950)
    
    # Assigning a Call to a Name (line 35):
    
    # Call to get(...): (line 35)
    # Processing the call arguments (line 35)
    unicode_289954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 40), 'unicode', u'style')
    # Processing the call keyword arguments (line 35)
    kwargs_289955 = {}
    # Getting the type of 'request' (line 35)
    request_289951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 19), 'request', False)
    # Obtaining the member 'keywords' of a type (line 35)
    keywords_289952 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 19), request_289951, 'keywords')
    # Obtaining the member 'get' of a type (line 35)
    get_289953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 19), keywords_289952, 'get')
    # Calling get(args, kwargs) (line 35)
    get_call_result_289956 = invoke(stypy.reporting.localization.Localization(__file__, 35, 19), get_289953, *[unicode_289954], **kwargs_289955)
    
    # Assigning a type to the variable 'style_marker' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'style_marker', get_call_result_289956)
    
    # Type idiom detected: calculating its left and rigth part (line 36)
    # Getting the type of 'style_marker' (line 36)
    style_marker_289957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'style_marker')
    # Getting the type of 'None' (line 36)
    None_289958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 27), 'None')
    
    (may_be_289959, more_types_in_union_289960) = may_not_be_none(style_marker_289957, None_289958)

    if may_be_289959:

        if more_types_in_union_289960:
            # Runtime conditional SSA (line 36)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Evaluating assert statement condition
        
        
        # Call to len(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'style_marker' (line 37)
        style_marker_289962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 19), 'style_marker', False)
        # Obtaining the member 'args' of a type (line 37)
        args_289963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 19), style_marker_289962, 'args')
        # Processing the call keyword arguments (line 37)
        kwargs_289964 = {}
        # Getting the type of 'len' (line 37)
        len_289961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 15), 'len', False)
        # Calling len(args, kwargs) (line 37)
        len_call_result_289965 = invoke(stypy.reporting.localization.Localization(__file__, 37, 15), len_289961, *[args_289963], **kwargs_289964)
        
        int_289966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 41), 'int')
        # Applying the binary operator '==' (line 37)
        result_eq_289967 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 15), '==', len_call_result_289965, int_289966)
        
        
        # Assigning a Subscript to a Name (line 39):
        
        # Obtaining the type of the subscript
        int_289968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 34), 'int')
        # Getting the type of 'style_marker' (line 39)
        style_marker_289969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 16), 'style_marker')
        # Obtaining the member 'args' of a type (line 39)
        args_289970 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 16), style_marker_289969, 'args')
        # Obtaining the member '__getitem__' of a type (line 39)
        getitem___289971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 16), args_289970, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 39)
        subscript_call_result_289972 = invoke(stypy.reporting.localization.Localization(__file__, 39, 16), getitem___289971, int_289968)
        
        # Assigning a type to the variable 'style' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'style', subscript_call_result_289972)

        if more_types_in_union_289960:
            # SSA join for if statement (line 36)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to setup(...): (line 41)
    # Processing the call keyword arguments (line 41)
    kwargs_289976 = {}
    # Getting the type of 'matplotlib' (line 41)
    matplotlib_289973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'matplotlib', False)
    # Obtaining the member 'testing' of a type (line 41)
    testing_289974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 4), matplotlib_289973, 'testing')
    # Obtaining the member 'setup' of a type (line 41)
    setup_289975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 4), testing_289974, 'setup')
    # Calling setup(args, kwargs) (line 41)
    setup_call_result_289977 = invoke(stypy.reporting.localization.Localization(__file__, 41, 4), setup_289975, *[], **kwargs_289976)
    
    
    # Type idiom detected: calculating its left and rigth part (line 42)
    # Getting the type of 'backend' (line 42)
    backend_289978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'backend')
    # Getting the type of 'None' (line 42)
    None_289979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 22), 'None')
    
    (may_be_289980, more_types_in_union_289981) = may_not_be_none(backend_289978, None_289979)

    if may_be_289980:

        if more_types_in_union_289981:
            # Runtime conditional SSA (line 42)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 45, 8))
        
        # 'import matplotlib.pyplot' statement (line 45)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
        import_289982 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 45, 8), 'matplotlib.pyplot')

        if (type(import_289982) is not StypyTypeError):

            if (import_289982 != 'pyd_module'):
                __import__(import_289982)
                sys_modules_289983 = sys.modules[import_289982]
                import_module(stypy.reporting.localization.Localization(__file__, 45, 8), 'plt', sys_modules_289983.module_type_store, module_type_store)
            else:
                import matplotlib.pyplot as plt

                import_module(stypy.reporting.localization.Localization(__file__, 45, 8), 'plt', matplotlib.pyplot, module_type_store)

        else:
            # Assigning a type to the variable 'matplotlib.pyplot' (line 45)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'matplotlib.pyplot', import_289982)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')
        
        
        # Call to switch_backend(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'backend' (line 46)
        backend_289986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 27), 'backend', False)
        # Processing the call keyword arguments (line 46)
        kwargs_289987 = {}
        # Getting the type of 'plt' (line 46)
        plt_289984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'plt', False)
        # Obtaining the member 'switch_backend' of a type (line 46)
        switch_backend_289985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), plt_289984, 'switch_backend')
        # Calling switch_backend(args, kwargs) (line 46)
        switch_backend_call_result_289988 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), switch_backend_289985, *[backend_289986], **kwargs_289987)
        

        if more_types_in_union_289981:
            # SSA join for if statement (line 42)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to use(...): (line 47)
    # Processing the call arguments (line 47)
    # Getting the type of 'style' (line 47)
    style_289992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 25), 'style', False)
    # Processing the call keyword arguments (line 47)
    kwargs_289993 = {}
    # Getting the type of 'matplotlib' (line 47)
    matplotlib_289989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'matplotlib', False)
    # Obtaining the member 'style' of a type (line 47)
    style_289990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 4), matplotlib_289989, 'style')
    # Obtaining the member 'use' of a type (line 47)
    use_289991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 4), style_289990, 'use')
    # Calling use(args, kwargs) (line 47)
    use_call_result_289994 = invoke(stypy.reporting.localization.Localization(__file__, 47, 4), use_289991, *[style_289992], **kwargs_289993)
    
    
    # Try-finally block (line 48)
    # Creating a generator
    GeneratorType_289995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 8), 'GeneratorType')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 8), GeneratorType_289995, None)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'stypy_return_type', GeneratorType_289995)
    
    # finally branch of the try-finally block (line 48)
    
    # Type idiom detected: calculating its left and rigth part (line 51)
    # Getting the type of 'backend' (line 51)
    backend_289996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'backend')
    # Getting the type of 'None' (line 51)
    None_289997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 26), 'None')
    
    (may_be_289998, more_types_in_union_289999) = may_not_be_none(backend_289996, None_289997)

    if may_be_289998:

        if more_types_in_union_289999:
            # Runtime conditional SSA (line 51)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to switch_backend(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'prev_backend' (line 52)
        prev_backend_290002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 31), 'prev_backend', False)
        # Processing the call keyword arguments (line 52)
        kwargs_290003 = {}
        # Getting the type of 'plt' (line 52)
        plt_290000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'plt', False)
        # Obtaining the member 'switch_backend' of a type (line 52)
        switch_backend_290001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 12), plt_290000, 'switch_backend')
        # Calling switch_backend(args, kwargs) (line 52)
        switch_backend_call_result_290004 = invoke(stypy.reporting.localization.Localization(__file__, 52, 12), switch_backend_290001, *[prev_backend_290002], **kwargs_290003)
        

        if more_types_in_union_289999:
            # SSA join for if statement (line 51)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to _do_cleanup(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'original_units_registry' (line 53)
    original_units_registry_290006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 20), 'original_units_registry', False)
    # Getting the type of 'original_settings' (line 54)
    original_settings_290007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 20), 'original_settings', False)
    # Processing the call keyword arguments (line 53)
    kwargs_290008 = {}
    # Getting the type of '_do_cleanup' (line 53)
    _do_cleanup_290005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), '_do_cleanup', False)
    # Calling _do_cleanup(args, kwargs) (line 53)
    _do_cleanup_call_result_290009 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), _do_cleanup_290005, *[original_units_registry_290006, original_settings_290007], **kwargs_290008)
    
    
    
    # ################# End of 'mpl_test_settings(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'mpl_test_settings' in the type store
    # Getting the type of 'stypy_return_type' (line 19)
    stypy_return_type_290010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_290010)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'mpl_test_settings'
    return stypy_return_type_290010

# Assigning a type to the variable 'mpl_test_settings' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'mpl_test_settings', mpl_test_settings)

@norecursion
def mpl_image_comparison_parameters(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'mpl_image_comparison_parameters'
    module_type_store = module_type_store.open_function_context('mpl_image_comparison_parameters', 57, 0, False)
    
    # Passed parameters checking function
    mpl_image_comparison_parameters.stypy_localization = localization
    mpl_image_comparison_parameters.stypy_type_of_self = None
    mpl_image_comparison_parameters.stypy_type_store = module_type_store
    mpl_image_comparison_parameters.stypy_function_name = 'mpl_image_comparison_parameters'
    mpl_image_comparison_parameters.stypy_param_names_list = ['request', 'extension']
    mpl_image_comparison_parameters.stypy_varargs_param_name = None
    mpl_image_comparison_parameters.stypy_kwargs_param_name = None
    mpl_image_comparison_parameters.stypy_call_defaults = defaults
    mpl_image_comparison_parameters.stypy_call_varargs = varargs
    mpl_image_comparison_parameters.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'mpl_image_comparison_parameters', ['request', 'extension'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'mpl_image_comparison_parameters', localization, ['request', 'extension'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'mpl_image_comparison_parameters(...)' code ##################

    
    # Assigning a Subscript to a Name (line 67):
    
    # Obtaining the type of the subscript
    int_290011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 63), 'int')
    
    # Obtaining the type of the subscript
    unicode_290012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 39), 'unicode', u'baseline_images')
    # Getting the type of 'request' (line 67)
    request_290013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 22), 'request')
    # Obtaining the member 'keywords' of a type (line 67)
    keywords_290014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 22), request_290013, 'keywords')
    # Obtaining the member '__getitem__' of a type (line 67)
    getitem___290015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 22), keywords_290014, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 67)
    subscript_call_result_290016 = invoke(stypy.reporting.localization.Localization(__file__, 67, 22), getitem___290015, unicode_290012)
    
    # Obtaining the member 'args' of a type (line 67)
    args_290017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 22), subscript_call_result_290016, 'args')
    # Obtaining the member '__getitem__' of a type (line 67)
    getitem___290018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 22), args_290017, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 67)
    subscript_call_result_290019 = invoke(stypy.reporting.localization.Localization(__file__, 67, 22), getitem___290018, int_290011)
    
    # Assigning a type to the variable 'baseline_images' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'baseline_images', subscript_call_result_290019)
    
    # Type idiom detected: calculating its left and rigth part (line 68)
    # Getting the type of 'baseline_images' (line 68)
    baseline_images_290020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 7), 'baseline_images')
    # Getting the type of 'None' (line 68)
    None_290021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 26), 'None')
    
    (may_be_290022, more_types_in_union_290023) = may_be_none(baseline_images_290020, None_290021)

    if may_be_290022:

        if more_types_in_union_290023:
            # Runtime conditional SSA (line 68)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 71):
        
        # Call to getfixturevalue(...): (line 71)
        # Processing the call arguments (line 71)
        unicode_290026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 50), 'unicode', u'baseline_images')
        # Processing the call keyword arguments (line 71)
        kwargs_290027 = {}
        # Getting the type of 'request' (line 71)
        request_290024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 26), 'request', False)
        # Obtaining the member 'getfixturevalue' of a type (line 71)
        getfixturevalue_290025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 26), request_290024, 'getfixturevalue')
        # Calling getfixturevalue(args, kwargs) (line 71)
        getfixturevalue_call_result_290028 = invoke(stypy.reporting.localization.Localization(__file__, 71, 26), getfixturevalue_290025, *[unicode_290026], **kwargs_290027)
        
        # Assigning a type to the variable 'baseline_images' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'baseline_images', getfixturevalue_call_result_290028)

        if more_types_in_union_290023:
            # SSA join for if statement (line 68)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Attribute to a Name (line 73):
    # Getting the type of 'request' (line 73)
    request_290029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 11), 'request')
    # Obtaining the member 'function' of a type (line 73)
    function_290030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 11), request_290029, 'function')
    # Assigning a type to the variable 'func' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'func', function_290030)
    
    # Assigning a Tuple to a Attribute (line 74):
    
    # Obtaining an instance of the builtin type 'tuple' (line 74)
    tuple_290031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 74)
    # Adding element type (line 74)
    # Getting the type of 'baseline_images' (line 74)
    baseline_images_290032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 35), 'baseline_images')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 35), tuple_290031, baseline_images_290032)
    # Adding element type (line 74)
    # Getting the type of 'extension' (line 74)
    extension_290033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 52), 'extension')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 35), tuple_290031, extension_290033)
    
    # Getting the type of 'func' (line 74)
    func_290034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'func')
    # Obtaining the member '__wrapped__' of a type (line 74)
    wrapped___290035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 4), func_290034, '__wrapped__')
    # Setting the type of the member 'parameters' of a type (line 74)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 4), wrapped___290035, 'parameters', tuple_290031)
    
    # Try-finally block (line 75)
    # Creating a generator
    GeneratorType_290036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 8), 'GeneratorType')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 8), GeneratorType_290036, None)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'stypy_return_type', GeneratorType_290036)
    
    # finally branch of the try-finally block (line 75)
    
    # Call to delattr(...): (line 78)
    # Processing the call arguments (line 78)
    # Getting the type of 'func' (line 78)
    func_290038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'func', False)
    # Obtaining the member '__wrapped__' of a type (line 78)
    wrapped___290039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 16), func_290038, '__wrapped__')
    unicode_290040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 34), 'unicode', u'parameters')
    # Processing the call keyword arguments (line 78)
    kwargs_290041 = {}
    # Getting the type of 'delattr' (line 78)
    delattr_290037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'delattr', False)
    # Calling delattr(args, kwargs) (line 78)
    delattr_call_result_290042 = invoke(stypy.reporting.localization.Localization(__file__, 78, 8), delattr_290037, *[wrapped___290039, unicode_290040], **kwargs_290041)
    
    
    
    # ################# End of 'mpl_image_comparison_parameters(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'mpl_image_comparison_parameters' in the type store
    # Getting the type of 'stypy_return_type' (line 57)
    stypy_return_type_290043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_290043)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'mpl_image_comparison_parameters'
    return stypy_return_type_290043

# Assigning a type to the variable 'mpl_image_comparison_parameters' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'mpl_image_comparison_parameters', mpl_image_comparison_parameters)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
