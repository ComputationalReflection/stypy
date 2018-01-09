
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import (absolute_import, division, print_function,
2:                         unicode_literals)
3: 
4: import sys
5: 
6: 
7: def get_extra_test_plugins():
8:     from .plugins.performgc import PerformGC
9:     from .plugins.knownfailure import KnownFailure
10:     from nose.plugins import attrib
11: 
12:     return [PerformGC, KnownFailure, attrib.Plugin]
13: 
14: 
15: def get_env():
16:     env = {'NOSE_COVER_PACKAGE': ['matplotlib', 'mpl_toolkits'],
17:            'NOSE_COVER_HTML': 1,
18:            'NOSE_COVER_NO_PRINT': 1}
19:     return env
20: 
21: 
22: def check_deps():
23:     try:
24:         import nose
25:         try:
26:             from unittest import mock
27:         except ImportError:
28:             import mock
29:     except ImportError:
30:         print("matplotlib.test requires nose and mock to run.")
31:         raise
32: 
33: 
34: def test(verbosity=None, coverage=False, switch_backend_warn=True,
35:          recursionlimit=0, **kwargs):
36:     from ... import default_test_modules, get_backend, use
37: 
38:     old_backend = get_backend()
39:     old_recursionlimit = sys.getrecursionlimit()
40:     try:
41:         use('agg')
42:         if recursionlimit:
43:             sys.setrecursionlimit(recursionlimit)
44:         import nose
45:         from nose.plugins import multiprocess
46: 
47:         # Nose doesn't automatically instantiate all of the plugins in the
48:         # child processes, so we have to provide the multiprocess plugin with
49:         # a list.
50:         extra_plugins = get_extra_test_plugins()
51:         multiprocess._instantiate_plugins = extra_plugins
52: 
53:         env = get_env()
54:         if coverage:
55:             env['NOSE_WITH_COVERAGE'] = 1
56: 
57:         if verbosity is not None:
58:             env['NOSE_VERBOSE'] = verbosity
59: 
60:         success = nose.run(
61:             addplugins=[plugin() for plugin in extra_plugins],
62:             env=env,
63:             defaultTest=default_test_modules,
64:             **kwargs
65:         )
66:     finally:
67:         if old_backend.lower() != 'agg':
68:             use(old_backend, warn=switch_backend_warn)
69:         if recursionlimit:
70:             sys.setrecursionlimit(old_recursionlimit)
71: 
72:     return success
73: 
74: 
75: def knownfail(msg):
76:     from .exceptions import KnownFailureTest
77:     # Keep the next ultra-long comment so it shows in console.
78:     raise KnownFailureTest(msg)  # An error here when running nose means that you don't have the matplotlib.testing.nose.plugins:KnownFailure plugin in use.  # noqa
79: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import sys' statement (line 4)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'sys', sys, module_type_store)


@norecursion
def get_extra_test_plugins(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_extra_test_plugins'
    module_type_store = module_type_store.open_function_context('get_extra_test_plugins', 7, 0, False)
    
    # Passed parameters checking function
    get_extra_test_plugins.stypy_localization = localization
    get_extra_test_plugins.stypy_type_of_self = None
    get_extra_test_plugins.stypy_type_store = module_type_store
    get_extra_test_plugins.stypy_function_name = 'get_extra_test_plugins'
    get_extra_test_plugins.stypy_param_names_list = []
    get_extra_test_plugins.stypy_varargs_param_name = None
    get_extra_test_plugins.stypy_kwargs_param_name = None
    get_extra_test_plugins.stypy_call_defaults = defaults
    get_extra_test_plugins.stypy_call_varargs = varargs
    get_extra_test_plugins.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_extra_test_plugins', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_extra_test_plugins', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_extra_test_plugins(...)' code ##################

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 4))
    
    # 'from matplotlib.testing._nose.plugins.performgc import PerformGC' statement (line 8)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/_nose/')
    import_294130 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'matplotlib.testing._nose.plugins.performgc')

    if (type(import_294130) is not StypyTypeError):

        if (import_294130 != 'pyd_module'):
            __import__(import_294130)
            sys_modules_294131 = sys.modules[import_294130]
            import_from_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'matplotlib.testing._nose.plugins.performgc', sys_modules_294131.module_type_store, module_type_store, ['PerformGC'])
            nest_module(stypy.reporting.localization.Localization(__file__, 8, 4), __file__, sys_modules_294131, sys_modules_294131.module_type_store, module_type_store)
        else:
            from matplotlib.testing._nose.plugins.performgc import PerformGC

            import_from_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'matplotlib.testing._nose.plugins.performgc', None, module_type_store, ['PerformGC'], [PerformGC])

    else:
        # Assigning a type to the variable 'matplotlib.testing._nose.plugins.performgc' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'matplotlib.testing._nose.plugins.performgc', import_294130)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/_nose/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 4))
    
    # 'from matplotlib.testing._nose.plugins.knownfailure import KnownFailure' statement (line 9)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/_nose/')
    import_294132 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'matplotlib.testing._nose.plugins.knownfailure')

    if (type(import_294132) is not StypyTypeError):

        if (import_294132 != 'pyd_module'):
            __import__(import_294132)
            sys_modules_294133 = sys.modules[import_294132]
            import_from_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'matplotlib.testing._nose.plugins.knownfailure', sys_modules_294133.module_type_store, module_type_store, ['KnownFailure'])
            nest_module(stypy.reporting.localization.Localization(__file__, 9, 4), __file__, sys_modules_294133, sys_modules_294133.module_type_store, module_type_store)
        else:
            from matplotlib.testing._nose.plugins.knownfailure import KnownFailure

            import_from_module(stypy.reporting.localization.Localization(__file__, 9, 4), 'matplotlib.testing._nose.plugins.knownfailure', None, module_type_store, ['KnownFailure'], [KnownFailure])

    else:
        # Assigning a type to the variable 'matplotlib.testing._nose.plugins.knownfailure' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'matplotlib.testing._nose.plugins.knownfailure', import_294132)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/_nose/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 4))
    
    # 'from nose.plugins import attrib' statement (line 10)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/_nose/')
    import_294134 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'nose.plugins')

    if (type(import_294134) is not StypyTypeError):

        if (import_294134 != 'pyd_module'):
            __import__(import_294134)
            sys_modules_294135 = sys.modules[import_294134]
            import_from_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'nose.plugins', sys_modules_294135.module_type_store, module_type_store, ['attrib'])
            nest_module(stypy.reporting.localization.Localization(__file__, 10, 4), __file__, sys_modules_294135, sys_modules_294135.module_type_store, module_type_store)
        else:
            from nose.plugins import attrib

            import_from_module(stypy.reporting.localization.Localization(__file__, 10, 4), 'nose.plugins', None, module_type_store, ['attrib'], [attrib])

    else:
        # Assigning a type to the variable 'nose.plugins' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'nose.plugins', import_294134)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/_nose/')
    
    
    # Obtaining an instance of the builtin type 'list' (line 12)
    list_294136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 12)
    # Adding element type (line 12)
    # Getting the type of 'PerformGC' (line 12)
    PerformGC_294137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'PerformGC')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 11), list_294136, PerformGC_294137)
    # Adding element type (line 12)
    # Getting the type of 'KnownFailure' (line 12)
    KnownFailure_294138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 23), 'KnownFailure')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 11), list_294136, KnownFailure_294138)
    # Adding element type (line 12)
    # Getting the type of 'attrib' (line 12)
    attrib_294139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 37), 'attrib')
    # Obtaining the member 'Plugin' of a type (line 12)
    Plugin_294140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 37), attrib_294139, 'Plugin')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 11), list_294136, Plugin_294140)
    
    # Assigning a type to the variable 'stypy_return_type' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'stypy_return_type', list_294136)
    
    # ################# End of 'get_extra_test_plugins(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_extra_test_plugins' in the type store
    # Getting the type of 'stypy_return_type' (line 7)
    stypy_return_type_294141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_294141)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_extra_test_plugins'
    return stypy_return_type_294141

# Assigning a type to the variable 'get_extra_test_plugins' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'get_extra_test_plugins', get_extra_test_plugins)

@norecursion
def get_env(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_env'
    module_type_store = module_type_store.open_function_context('get_env', 15, 0, False)
    
    # Passed parameters checking function
    get_env.stypy_localization = localization
    get_env.stypy_type_of_self = None
    get_env.stypy_type_store = module_type_store
    get_env.stypy_function_name = 'get_env'
    get_env.stypy_param_names_list = []
    get_env.stypy_varargs_param_name = None
    get_env.stypy_kwargs_param_name = None
    get_env.stypy_call_defaults = defaults
    get_env.stypy_call_varargs = varargs
    get_env.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_env', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_env', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_env(...)' code ##################

    
    # Assigning a Dict to a Name (line 16):
    
    # Obtaining an instance of the builtin type 'dict' (line 16)
    dict_294142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 10), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 16)
    # Adding element type (key, value) (line 16)
    unicode_294143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 11), 'unicode', u'NOSE_COVER_PACKAGE')
    
    # Obtaining an instance of the builtin type 'list' (line 16)
    list_294144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 16)
    # Adding element type (line 16)
    unicode_294145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 34), 'unicode', u'matplotlib')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 33), list_294144, unicode_294145)
    # Adding element type (line 16)
    unicode_294146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 48), 'unicode', u'mpl_toolkits')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 33), list_294144, unicode_294146)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), dict_294142, (unicode_294143, list_294144))
    # Adding element type (key, value) (line 16)
    unicode_294147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 11), 'unicode', u'NOSE_COVER_HTML')
    int_294148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 30), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), dict_294142, (unicode_294147, int_294148))
    # Adding element type (key, value) (line 16)
    unicode_294149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 11), 'unicode', u'NOSE_COVER_NO_PRINT')
    int_294150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 34), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 10), dict_294142, (unicode_294149, int_294150))
    
    # Assigning a type to the variable 'env' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'env', dict_294142)
    # Getting the type of 'env' (line 19)
    env_294151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 11), 'env')
    # Assigning a type to the variable 'stypy_return_type' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'stypy_return_type', env_294151)
    
    # ################# End of 'get_env(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_env' in the type store
    # Getting the type of 'stypy_return_type' (line 15)
    stypy_return_type_294152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_294152)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_env'
    return stypy_return_type_294152

# Assigning a type to the variable 'get_env' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'get_env', get_env)

@norecursion
def check_deps(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_deps'
    module_type_store = module_type_store.open_function_context('check_deps', 22, 0, False)
    
    # Passed parameters checking function
    check_deps.stypy_localization = localization
    check_deps.stypy_type_of_self = None
    check_deps.stypy_type_store = module_type_store
    check_deps.stypy_function_name = 'check_deps'
    check_deps.stypy_param_names_list = []
    check_deps.stypy_varargs_param_name = None
    check_deps.stypy_kwargs_param_name = None
    check_deps.stypy_call_defaults = defaults
    check_deps.stypy_call_varargs = varargs
    check_deps.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_deps', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_deps', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_deps(...)' code ##################

    
    
    # SSA begins for try-except statement (line 23)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 8))
    
    # 'import nose' statement (line 24)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/_nose/')
    import_294153 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 24, 8), 'nose')

    if (type(import_294153) is not StypyTypeError):

        if (import_294153 != 'pyd_module'):
            __import__(import_294153)
            sys_modules_294154 = sys.modules[import_294153]
            import_module(stypy.reporting.localization.Localization(__file__, 24, 8), 'nose', sys_modules_294154.module_type_store, module_type_store)
        else:
            import nose

            import_module(stypy.reporting.localization.Localization(__file__, 24, 8), 'nose', nose, module_type_store)

    else:
        # Assigning a type to the variable 'nose' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'nose', import_294153)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/_nose/')
    
    
    
    # SSA begins for try-except statement (line 25)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 12))
    
    # 'from unittest import mock' statement (line 26)
    try:
        from unittest import mock

    except:
        mock = UndefinedType
    import_from_module(stypy.reporting.localization.Localization(__file__, 26, 12), 'unittest', None, module_type_store, ['mock'], [mock])
    
    # SSA branch for the except part of a try statement (line 25)
    # SSA branch for the except 'ImportError' branch of a try statement (line 25)
    module_type_store.open_ssa_branch('except')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 28, 12))
    
    # 'import mock' statement (line 28)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/_nose/')
    import_294155 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 28, 12), 'mock')

    if (type(import_294155) is not StypyTypeError):

        if (import_294155 != 'pyd_module'):
            __import__(import_294155)
            sys_modules_294156 = sys.modules[import_294155]
            import_module(stypy.reporting.localization.Localization(__file__, 28, 12), 'mock', sys_modules_294156.module_type_store, module_type_store)
        else:
            import mock

            import_module(stypy.reporting.localization.Localization(__file__, 28, 12), 'mock', mock, module_type_store)

    else:
        # Assigning a type to the variable 'mock' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'mock', import_294155)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/_nose/')
    
    # SSA join for try-except statement (line 25)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the except part of a try statement (line 23)
    # SSA branch for the except 'ImportError' branch of a try statement (line 23)
    module_type_store.open_ssa_branch('except')
    
    # Call to print(...): (line 30)
    # Processing the call arguments (line 30)
    unicode_294158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 14), 'unicode', u'matplotlib.test requires nose and mock to run.')
    # Processing the call keyword arguments (line 30)
    kwargs_294159 = {}
    # Getting the type of 'print' (line 30)
    print_294157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'print', False)
    # Calling print(args, kwargs) (line 30)
    print_call_result_294160 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), print_294157, *[unicode_294158], **kwargs_294159)
    
    # SSA join for try-except statement (line 23)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'check_deps(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_deps' in the type store
    # Getting the type of 'stypy_return_type' (line 22)
    stypy_return_type_294161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_294161)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_deps'
    return stypy_return_type_294161

# Assigning a type to the variable 'check_deps' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'check_deps', check_deps)

@norecursion
def test(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 34)
    None_294162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 19), 'None')
    # Getting the type of 'False' (line 34)
    False_294163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 34), 'False')
    # Getting the type of 'True' (line 34)
    True_294164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 61), 'True')
    int_294165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 24), 'int')
    defaults = [None_294162, False_294163, True_294164, int_294165]
    # Create a new context for function 'test'
    module_type_store = module_type_store.open_function_context('test', 34, 0, False)
    
    # Passed parameters checking function
    test.stypy_localization = localization
    test.stypy_type_of_self = None
    test.stypy_type_store = module_type_store
    test.stypy_function_name = 'test'
    test.stypy_param_names_list = ['verbosity', 'coverage', 'switch_backend_warn', 'recursionlimit']
    test.stypy_varargs_param_name = None
    test.stypy_kwargs_param_name = 'kwargs'
    test.stypy_call_defaults = defaults
    test.stypy_call_varargs = varargs
    test.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test', ['verbosity', 'coverage', 'switch_backend_warn', 'recursionlimit'], None, 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test', localization, ['verbosity', 'coverage', 'switch_backend_warn', 'recursionlimit'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test(...)' code ##################

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 36, 4))
    
    # 'from matplotlib import default_test_modules, get_backend, use' statement (line 36)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/_nose/')
    import_294166 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 36, 4), 'matplotlib')

    if (type(import_294166) is not StypyTypeError):

        if (import_294166 != 'pyd_module'):
            __import__(import_294166)
            sys_modules_294167 = sys.modules[import_294166]
            import_from_module(stypy.reporting.localization.Localization(__file__, 36, 4), 'matplotlib', sys_modules_294167.module_type_store, module_type_store, ['default_test_modules', 'get_backend', 'use'])
            nest_module(stypy.reporting.localization.Localization(__file__, 36, 4), __file__, sys_modules_294167, sys_modules_294167.module_type_store, module_type_store)
        else:
            from matplotlib import default_test_modules, get_backend, use

            import_from_module(stypy.reporting.localization.Localization(__file__, 36, 4), 'matplotlib', None, module_type_store, ['default_test_modules', 'get_backend', 'use'], [default_test_modules, get_backend, use])

    else:
        # Assigning a type to the variable 'matplotlib' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'matplotlib', import_294166)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/_nose/')
    
    
    # Assigning a Call to a Name (line 38):
    
    # Call to get_backend(...): (line 38)
    # Processing the call keyword arguments (line 38)
    kwargs_294169 = {}
    # Getting the type of 'get_backend' (line 38)
    get_backend_294168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 18), 'get_backend', False)
    # Calling get_backend(args, kwargs) (line 38)
    get_backend_call_result_294170 = invoke(stypy.reporting.localization.Localization(__file__, 38, 18), get_backend_294168, *[], **kwargs_294169)
    
    # Assigning a type to the variable 'old_backend' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'old_backend', get_backend_call_result_294170)
    
    # Assigning a Call to a Name (line 39):
    
    # Call to getrecursionlimit(...): (line 39)
    # Processing the call keyword arguments (line 39)
    kwargs_294173 = {}
    # Getting the type of 'sys' (line 39)
    sys_294171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 25), 'sys', False)
    # Obtaining the member 'getrecursionlimit' of a type (line 39)
    getrecursionlimit_294172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 25), sys_294171, 'getrecursionlimit')
    # Calling getrecursionlimit(args, kwargs) (line 39)
    getrecursionlimit_call_result_294174 = invoke(stypy.reporting.localization.Localization(__file__, 39, 25), getrecursionlimit_294172, *[], **kwargs_294173)
    
    # Assigning a type to the variable 'old_recursionlimit' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'old_recursionlimit', getrecursionlimit_call_result_294174)
    
    # Try-finally block (line 40)
    
    # Call to use(...): (line 41)
    # Processing the call arguments (line 41)
    unicode_294176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 12), 'unicode', u'agg')
    # Processing the call keyword arguments (line 41)
    kwargs_294177 = {}
    # Getting the type of 'use' (line 41)
    use_294175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'use', False)
    # Calling use(args, kwargs) (line 41)
    use_call_result_294178 = invoke(stypy.reporting.localization.Localization(__file__, 41, 8), use_294175, *[unicode_294176], **kwargs_294177)
    
    
    # Getting the type of 'recursionlimit' (line 42)
    recursionlimit_294179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 11), 'recursionlimit')
    # Testing the type of an if condition (line 42)
    if_condition_294180 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 8), recursionlimit_294179)
    # Assigning a type to the variable 'if_condition_294180' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'if_condition_294180', if_condition_294180)
    # SSA begins for if statement (line 42)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to setrecursionlimit(...): (line 43)
    # Processing the call arguments (line 43)
    # Getting the type of 'recursionlimit' (line 43)
    recursionlimit_294183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 34), 'recursionlimit', False)
    # Processing the call keyword arguments (line 43)
    kwargs_294184 = {}
    # Getting the type of 'sys' (line 43)
    sys_294181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'sys', False)
    # Obtaining the member 'setrecursionlimit' of a type (line 43)
    setrecursionlimit_294182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 12), sys_294181, 'setrecursionlimit')
    # Calling setrecursionlimit(args, kwargs) (line 43)
    setrecursionlimit_call_result_294185 = invoke(stypy.reporting.localization.Localization(__file__, 43, 12), setrecursionlimit_294182, *[recursionlimit_294183], **kwargs_294184)
    
    # SSA join for if statement (line 42)
    module_type_store = module_type_store.join_ssa_context()
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 44, 8))
    
    # 'import nose' statement (line 44)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/_nose/')
    import_294186 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 44, 8), 'nose')

    if (type(import_294186) is not StypyTypeError):

        if (import_294186 != 'pyd_module'):
            __import__(import_294186)
            sys_modules_294187 = sys.modules[import_294186]
            import_module(stypy.reporting.localization.Localization(__file__, 44, 8), 'nose', sys_modules_294187.module_type_store, module_type_store)
        else:
            import nose

            import_module(stypy.reporting.localization.Localization(__file__, 44, 8), 'nose', nose, module_type_store)

    else:
        # Assigning a type to the variable 'nose' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'nose', import_294186)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/_nose/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 45, 8))
    
    # 'from nose.plugins import multiprocess' statement (line 45)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/_nose/')
    import_294188 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 45, 8), 'nose.plugins')

    if (type(import_294188) is not StypyTypeError):

        if (import_294188 != 'pyd_module'):
            __import__(import_294188)
            sys_modules_294189 = sys.modules[import_294188]
            import_from_module(stypy.reporting.localization.Localization(__file__, 45, 8), 'nose.plugins', sys_modules_294189.module_type_store, module_type_store, ['multiprocess'])
            nest_module(stypy.reporting.localization.Localization(__file__, 45, 8), __file__, sys_modules_294189, sys_modules_294189.module_type_store, module_type_store)
        else:
            from nose.plugins import multiprocess

            import_from_module(stypy.reporting.localization.Localization(__file__, 45, 8), 'nose.plugins', None, module_type_store, ['multiprocess'], [multiprocess])

    else:
        # Assigning a type to the variable 'nose.plugins' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'nose.plugins', import_294188)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/_nose/')
    
    
    # Assigning a Call to a Name (line 50):
    
    # Call to get_extra_test_plugins(...): (line 50)
    # Processing the call keyword arguments (line 50)
    kwargs_294191 = {}
    # Getting the type of 'get_extra_test_plugins' (line 50)
    get_extra_test_plugins_294190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 24), 'get_extra_test_plugins', False)
    # Calling get_extra_test_plugins(args, kwargs) (line 50)
    get_extra_test_plugins_call_result_294192 = invoke(stypy.reporting.localization.Localization(__file__, 50, 24), get_extra_test_plugins_294190, *[], **kwargs_294191)
    
    # Assigning a type to the variable 'extra_plugins' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'extra_plugins', get_extra_test_plugins_call_result_294192)
    
    # Assigning a Name to a Attribute (line 51):
    # Getting the type of 'extra_plugins' (line 51)
    extra_plugins_294193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 44), 'extra_plugins')
    # Getting the type of 'multiprocess' (line 51)
    multiprocess_294194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'multiprocess')
    # Setting the type of the member '_instantiate_plugins' of a type (line 51)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), multiprocess_294194, '_instantiate_plugins', extra_plugins_294193)
    
    # Assigning a Call to a Name (line 53):
    
    # Call to get_env(...): (line 53)
    # Processing the call keyword arguments (line 53)
    kwargs_294196 = {}
    # Getting the type of 'get_env' (line 53)
    get_env_294195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 14), 'get_env', False)
    # Calling get_env(args, kwargs) (line 53)
    get_env_call_result_294197 = invoke(stypy.reporting.localization.Localization(__file__, 53, 14), get_env_294195, *[], **kwargs_294196)
    
    # Assigning a type to the variable 'env' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'env', get_env_call_result_294197)
    
    # Getting the type of 'coverage' (line 54)
    coverage_294198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 11), 'coverage')
    # Testing the type of an if condition (line 54)
    if_condition_294199 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 54, 8), coverage_294198)
    # Assigning a type to the variable 'if_condition_294199' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'if_condition_294199', if_condition_294199)
    # SSA begins for if statement (line 54)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Subscript (line 55):
    int_294200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 40), 'int')
    # Getting the type of 'env' (line 55)
    env_294201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'env')
    unicode_294202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 16), 'unicode', u'NOSE_WITH_COVERAGE')
    # Storing an element on a container (line 55)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 12), env_294201, (unicode_294202, int_294200))
    # SSA join for if statement (line 54)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 57)
    # Getting the type of 'verbosity' (line 57)
    verbosity_294203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'verbosity')
    # Getting the type of 'None' (line 57)
    None_294204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 28), 'None')
    
    (may_be_294205, more_types_in_union_294206) = may_not_be_none(verbosity_294203, None_294204)

    if may_be_294205:

        if more_types_in_union_294206:
            # Runtime conditional SSA (line 57)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Subscript (line 58):
        # Getting the type of 'verbosity' (line 58)
        verbosity_294207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 34), 'verbosity')
        # Getting the type of 'env' (line 58)
        env_294208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'env')
        unicode_294209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 16), 'unicode', u'NOSE_VERBOSE')
        # Storing an element on a container (line 58)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 12), env_294208, (unicode_294209, verbosity_294207))

        if more_types_in_union_294206:
            # SSA join for if statement (line 57)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 60):
    
    # Call to run(...): (line 60)
    # Processing the call keyword arguments (line 60)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'extra_plugins' (line 61)
    extra_plugins_294215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 47), 'extra_plugins', False)
    comprehension_294216 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 24), extra_plugins_294215)
    # Assigning a type to the variable 'plugin' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 24), 'plugin', comprehension_294216)
    
    # Call to plugin(...): (line 61)
    # Processing the call keyword arguments (line 61)
    kwargs_294213 = {}
    # Getting the type of 'plugin' (line 61)
    plugin_294212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 24), 'plugin', False)
    # Calling plugin(args, kwargs) (line 61)
    plugin_call_result_294214 = invoke(stypy.reporting.localization.Localization(__file__, 61, 24), plugin_294212, *[], **kwargs_294213)
    
    list_294217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 24), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 24), list_294217, plugin_call_result_294214)
    keyword_294218 = list_294217
    # Getting the type of 'env' (line 62)
    env_294219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 16), 'env', False)
    keyword_294220 = env_294219
    # Getting the type of 'default_test_modules' (line 63)
    default_test_modules_294221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 24), 'default_test_modules', False)
    keyword_294222 = default_test_modules_294221
    # Getting the type of 'kwargs' (line 64)
    kwargs_294223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 14), 'kwargs', False)
    kwargs_294224 = {'addplugins': keyword_294218, 'defaultTest': keyword_294222, 'env': keyword_294220, 'kwargs_294223': kwargs_294223}
    # Getting the type of 'nose' (line 60)
    nose_294210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 18), 'nose', False)
    # Obtaining the member 'run' of a type (line 60)
    run_294211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 18), nose_294210, 'run')
    # Calling run(args, kwargs) (line 60)
    run_call_result_294225 = invoke(stypy.reporting.localization.Localization(__file__, 60, 18), run_294211, *[], **kwargs_294224)
    
    # Assigning a type to the variable 'success' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'success', run_call_result_294225)
    
    # finally branch of the try-finally block (line 40)
    
    
    
    # Call to lower(...): (line 67)
    # Processing the call keyword arguments (line 67)
    kwargs_294228 = {}
    # Getting the type of 'old_backend' (line 67)
    old_backend_294226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 11), 'old_backend', False)
    # Obtaining the member 'lower' of a type (line 67)
    lower_294227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 11), old_backend_294226, 'lower')
    # Calling lower(args, kwargs) (line 67)
    lower_call_result_294229 = invoke(stypy.reporting.localization.Localization(__file__, 67, 11), lower_294227, *[], **kwargs_294228)
    
    unicode_294230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 34), 'unicode', u'agg')
    # Applying the binary operator '!=' (line 67)
    result_ne_294231 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 11), '!=', lower_call_result_294229, unicode_294230)
    
    # Testing the type of an if condition (line 67)
    if_condition_294232 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 8), result_ne_294231)
    # Assigning a type to the variable 'if_condition_294232' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'if_condition_294232', if_condition_294232)
    # SSA begins for if statement (line 67)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to use(...): (line 68)
    # Processing the call arguments (line 68)
    # Getting the type of 'old_backend' (line 68)
    old_backend_294234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'old_backend', False)
    # Processing the call keyword arguments (line 68)
    # Getting the type of 'switch_backend_warn' (line 68)
    switch_backend_warn_294235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 34), 'switch_backend_warn', False)
    keyword_294236 = switch_backend_warn_294235
    kwargs_294237 = {'warn': keyword_294236}
    # Getting the type of 'use' (line 68)
    use_294233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'use', False)
    # Calling use(args, kwargs) (line 68)
    use_call_result_294238 = invoke(stypy.reporting.localization.Localization(__file__, 68, 12), use_294233, *[old_backend_294234], **kwargs_294237)
    
    # SSA join for if statement (line 67)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'recursionlimit' (line 69)
    recursionlimit_294239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 11), 'recursionlimit')
    # Testing the type of an if condition (line 69)
    if_condition_294240 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 8), recursionlimit_294239)
    # Assigning a type to the variable 'if_condition_294240' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'if_condition_294240', if_condition_294240)
    # SSA begins for if statement (line 69)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to setrecursionlimit(...): (line 70)
    # Processing the call arguments (line 70)
    # Getting the type of 'old_recursionlimit' (line 70)
    old_recursionlimit_294243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 34), 'old_recursionlimit', False)
    # Processing the call keyword arguments (line 70)
    kwargs_294244 = {}
    # Getting the type of 'sys' (line 70)
    sys_294241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'sys', False)
    # Obtaining the member 'setrecursionlimit' of a type (line 70)
    setrecursionlimit_294242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 12), sys_294241, 'setrecursionlimit')
    # Calling setrecursionlimit(args, kwargs) (line 70)
    setrecursionlimit_call_result_294245 = invoke(stypy.reporting.localization.Localization(__file__, 70, 12), setrecursionlimit_294242, *[old_recursionlimit_294243], **kwargs_294244)
    
    # SSA join for if statement (line 69)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'success' (line 72)
    success_294246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 11), 'success')
    # Assigning a type to the variable 'stypy_return_type' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'stypy_return_type', success_294246)
    
    # ################# End of 'test(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test' in the type store
    # Getting the type of 'stypy_return_type' (line 34)
    stypy_return_type_294247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_294247)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test'
    return stypy_return_type_294247

# Assigning a type to the variable 'test' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'test', test)

@norecursion
def knownfail(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'knownfail'
    module_type_store = module_type_store.open_function_context('knownfail', 75, 0, False)
    
    # Passed parameters checking function
    knownfail.stypy_localization = localization
    knownfail.stypy_type_of_self = None
    knownfail.stypy_type_store = module_type_store
    knownfail.stypy_function_name = 'knownfail'
    knownfail.stypy_param_names_list = ['msg']
    knownfail.stypy_varargs_param_name = None
    knownfail.stypy_kwargs_param_name = None
    knownfail.stypy_call_defaults = defaults
    knownfail.stypy_call_varargs = varargs
    knownfail.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'knownfail', ['msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'knownfail', localization, ['msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'knownfail(...)' code ##################

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 76, 4))
    
    # 'from matplotlib.testing._nose.exceptions import KnownFailureTest' statement (line 76)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/_nose/')
    import_294248 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 76, 4), 'matplotlib.testing._nose.exceptions')

    if (type(import_294248) is not StypyTypeError):

        if (import_294248 != 'pyd_module'):
            __import__(import_294248)
            sys_modules_294249 = sys.modules[import_294248]
            import_from_module(stypy.reporting.localization.Localization(__file__, 76, 4), 'matplotlib.testing._nose.exceptions', sys_modules_294249.module_type_store, module_type_store, ['KnownFailureTest'])
            nest_module(stypy.reporting.localization.Localization(__file__, 76, 4), __file__, sys_modules_294249, sys_modules_294249.module_type_store, module_type_store)
        else:
            from matplotlib.testing._nose.exceptions import KnownFailureTest

            import_from_module(stypy.reporting.localization.Localization(__file__, 76, 4), 'matplotlib.testing._nose.exceptions', None, module_type_store, ['KnownFailureTest'], [KnownFailureTest])

    else:
        # Assigning a type to the variable 'matplotlib.testing._nose.exceptions' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'matplotlib.testing._nose.exceptions', import_294248)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/_nose/')
    
    
    # Call to KnownFailureTest(...): (line 78)
    # Processing the call arguments (line 78)
    # Getting the type of 'msg' (line 78)
    msg_294251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 27), 'msg', False)
    # Processing the call keyword arguments (line 78)
    kwargs_294252 = {}
    # Getting the type of 'KnownFailureTest' (line 78)
    KnownFailureTest_294250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 10), 'KnownFailureTest', False)
    # Calling KnownFailureTest(args, kwargs) (line 78)
    KnownFailureTest_call_result_294253 = invoke(stypy.reporting.localization.Localization(__file__, 78, 10), KnownFailureTest_294250, *[msg_294251], **kwargs_294252)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 78, 4), KnownFailureTest_call_result_294253, 'raise parameter', BaseException)
    
    # ################# End of 'knownfail(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'knownfail' in the type store
    # Getting the type of 'stypy_return_type' (line 75)
    stypy_return_type_294254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_294254)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'knownfail'
    return stypy_return_type_294254

# Assigning a type to the variable 'knownfail' (line 75)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'knownfail', knownfail)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
