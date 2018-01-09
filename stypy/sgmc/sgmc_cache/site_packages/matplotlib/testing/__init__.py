
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import (absolute_import, division, print_function,
2:                         unicode_literals)
3: 
4: import inspect
5: import warnings
6: from contextlib import contextmanager
7: 
8: import matplotlib
9: from matplotlib.cbook import iterable
10: from matplotlib import rcParams, rcdefaults, use
11: 
12: 
13: def _is_list_like(obj):
14:     '''Returns whether the obj is iterable and not a string'''
15:     return not isinstance(obj, six.string_types) and iterable(obj)
16: 
17: 
18: def is_called_from_pytest():
19:     '''Returns whether the call was done from pytest'''
20:     return getattr(matplotlib, '_called_from_pytest', False)
21: 
22: 
23: # stolen from pytest
24: def _getrawcode(obj, trycall=True):
25:     '''Return code object for given function.'''
26:     try:
27:         return obj.__code__
28:     except AttributeError:
29:         obj = getattr(obj, 'im_func', obj)
30:         obj = getattr(obj, 'func_code', obj)
31:         obj = getattr(obj, 'f_code', obj)
32:         obj = getattr(obj, '__code__', obj)
33:         if trycall and not hasattr(obj, 'co_firstlineno'):
34:             if hasattr(obj, '__call__') and not inspect.isclass(obj):
35:                 x = getrawcode(obj.__call__, trycall=False)
36:                 if hasattr(x, 'co_firstlineno'):
37:                     return x
38:         return obj
39: 
40: 
41: def _copy_metadata(src_func, tgt_func):
42:     '''Replicates metadata of the function. Returns target function.'''
43:     tgt_func.__dict__.update(src_func.__dict__)
44:     tgt_func.__doc__ = src_func.__doc__
45:     tgt_func.__module__ = src_func.__module__
46:     tgt_func.__name__ = src_func.__name__
47:     if hasattr(src_func, '__qualname__'):
48:         tgt_func.__qualname__ = src_func.__qualname__
49:     if not hasattr(tgt_func, 'compat_co_firstlineno'):
50:         tgt_func.compat_co_firstlineno = _getrawcode(src_func).co_firstlineno
51:     return tgt_func
52: 
53: 
54: # stolen from pandas
55: @contextmanager
56: def assert_produces_warning(expected_warning=Warning, filter_level="always",
57:                             clear=None):
58:     '''
59:     Context manager for running code that expects to raise (or not raise)
60:     warnings.  Checks that code raises the expected warning and only the
61:     expected warning. Pass ``False`` or ``None`` to check that it does *not*
62:     raise a warning. Defaults to ``exception.Warning``, baseclass of all
63:     Warnings. (basically a wrapper around ``warnings.catch_warnings``).
64: 
65:     >>> import warnings
66:     >>> with assert_produces_warning():
67:     ...     warnings.warn(UserWarning())
68:     ...
69:     >>> with assert_produces_warning(False):
70:     ...     warnings.warn(RuntimeWarning())
71:     ...
72:     Traceback (most recent call last):
73:         ...
74:     AssertionError: Caused unexpected warning(s): ['RuntimeWarning'].
75:     >>> with assert_produces_warning(UserWarning):
76:     ...     warnings.warn(RuntimeWarning())
77:     Traceback (most recent call last):
78:         ...
79:     AssertionError: Did not see expected warning of class 'UserWarning'.
80: 
81:     ..warn:: This is *not* thread-safe.
82:     '''
83:     with warnings.catch_warnings(record=True) as w:
84: 
85:         if clear is not None:
86:             # make sure that we are clearning these warnings
87:             # if they have happened before
88:             # to guarantee that we will catch them
89:             if not _is_list_like(clear):
90:                 clear = [clear]
91:             for m in clear:
92:                 getattr(m, "__warningregistry__", {}).clear()
93: 
94:         saw_warning = False
95:         warnings.simplefilter(filter_level)
96:         yield w
97:         extra_warnings = []
98:         for actual_warning in w:
99:             if (expected_warning and issubclass(actual_warning.category,
100:                                                 expected_warning)):
101:                 saw_warning = True
102:             else:
103:                 extra_warnings.append(actual_warning.category.__name__)
104:         if expected_warning:
105:             assert saw_warning, ("Did not see expected warning of class %r."
106:                                  % expected_warning.__name__)
107:         assert not extra_warnings, ("Caused unexpected warning(s): %r."
108:                                     % extra_warnings)
109: 
110: 
111: def set_font_settings_for_testing():
112:     rcParams['font.family'] = 'DejaVu Sans'
113:     rcParams['text.hinting'] = False
114:     rcParams['text.hinting_factor'] = 8
115: 
116: 
117: def set_reproducibility_for_testing():
118:     rcParams['svg.hashsalt'] = 'matplotlib'
119: 
120: 
121: def setup():
122:     # The baseline images are created in this locale, so we should use
123:     # it during all of the tests.
124:     import locale
125:     import warnings
126:     from matplotlib.backends import backend_agg, backend_pdf, backend_svg
127: 
128:     try:
129:         locale.setlocale(locale.LC_ALL, str('en_US.UTF-8'))
130:     except locale.Error:
131:         try:
132:             locale.setlocale(locale.LC_ALL, str('English_United States.1252'))
133:         except locale.Error:
134:             warnings.warn(
135:                 "Could not set locale to English/United States. "
136:                 "Some date-related tests may fail")
137: 
138:     use('Agg', warn=False)  # use Agg backend for these tests
139: 
140:     # These settings *must* be hardcoded for running the comparison
141:     # tests and are not necessarily the default values as specified in
142:     # rcsetup.py
143:     rcdefaults()  # Start with all defaults
144: 
145:     set_font_settings_for_testing()
146:     set_reproducibility_for_testing()
147: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import inspect' statement (line 4)
import inspect

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'inspect', inspect, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import warnings' statement (line 5)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from contextlib import contextmanager' statement (line 6)
try:
    from contextlib import contextmanager

except:
    contextmanager = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'contextlib', None, module_type_store, ['contextmanager'], [contextmanager])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import matplotlib' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
import_291978 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'matplotlib')

if (type(import_291978) is not StypyTypeError):

    if (import_291978 != 'pyd_module'):
        __import__(import_291978)
        sys_modules_291979 = sys.modules[import_291978]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'matplotlib', sys_modules_291979.module_type_store, module_type_store)
    else:
        import matplotlib

        import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'matplotlib', matplotlib, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'matplotlib', import_291978)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from matplotlib.cbook import iterable' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
import_291980 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.cbook')

if (type(import_291980) is not StypyTypeError):

    if (import_291980 != 'pyd_module'):
        __import__(import_291980)
        sys_modules_291981 = sys.modules[import_291980]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.cbook', sys_modules_291981.module_type_store, module_type_store, ['iterable'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_291981, sys_modules_291981.module_type_store, module_type_store)
    else:
        from matplotlib.cbook import iterable

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.cbook', None, module_type_store, ['iterable'], [iterable])

else:
    # Assigning a type to the variable 'matplotlib.cbook' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.cbook', import_291980)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from matplotlib import rcParams, rcdefaults, use' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
import_291982 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib')

if (type(import_291982) is not StypyTypeError):

    if (import_291982 != 'pyd_module'):
        __import__(import_291982)
        sys_modules_291983 = sys.modules[import_291982]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib', sys_modules_291983.module_type_store, module_type_store, ['rcParams', 'rcdefaults', 'use'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_291983, sys_modules_291983.module_type_store, module_type_store)
    else:
        from matplotlib import rcParams, rcdefaults, use

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib', None, module_type_store, ['rcParams', 'rcdefaults', 'use'], [rcParams, rcdefaults, use])

else:
    # Assigning a type to the variable 'matplotlib' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'matplotlib', import_291982)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')


@norecursion
def _is_list_like(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_is_list_like'
    module_type_store = module_type_store.open_function_context('_is_list_like', 13, 0, False)
    
    # Passed parameters checking function
    _is_list_like.stypy_localization = localization
    _is_list_like.stypy_type_of_self = None
    _is_list_like.stypy_type_store = module_type_store
    _is_list_like.stypy_function_name = '_is_list_like'
    _is_list_like.stypy_param_names_list = ['obj']
    _is_list_like.stypy_varargs_param_name = None
    _is_list_like.stypy_kwargs_param_name = None
    _is_list_like.stypy_call_defaults = defaults
    _is_list_like.stypy_call_varargs = varargs
    _is_list_like.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_is_list_like', ['obj'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_is_list_like', localization, ['obj'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_is_list_like(...)' code ##################

    unicode_291984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 4), 'unicode', u'Returns whether the obj is iterable and not a string')
    
    # Evaluating a boolean operation
    
    
    # Call to isinstance(...): (line 15)
    # Processing the call arguments (line 15)
    # Getting the type of 'obj' (line 15)
    obj_291986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 26), 'obj', False)
    # Getting the type of 'six' (line 15)
    six_291987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 31), 'six', False)
    # Obtaining the member 'string_types' of a type (line 15)
    string_types_291988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 31), six_291987, 'string_types')
    # Processing the call keyword arguments (line 15)
    kwargs_291989 = {}
    # Getting the type of 'isinstance' (line 15)
    isinstance_291985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 15), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 15)
    isinstance_call_result_291990 = invoke(stypy.reporting.localization.Localization(__file__, 15, 15), isinstance_291985, *[obj_291986, string_types_291988], **kwargs_291989)
    
    # Applying the 'not' unary operator (line 15)
    result_not__291991 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 11), 'not', isinstance_call_result_291990)
    
    
    # Call to iterable(...): (line 15)
    # Processing the call arguments (line 15)
    # Getting the type of 'obj' (line 15)
    obj_291993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 62), 'obj', False)
    # Processing the call keyword arguments (line 15)
    kwargs_291994 = {}
    # Getting the type of 'iterable' (line 15)
    iterable_291992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 53), 'iterable', False)
    # Calling iterable(args, kwargs) (line 15)
    iterable_call_result_291995 = invoke(stypy.reporting.localization.Localization(__file__, 15, 53), iterable_291992, *[obj_291993], **kwargs_291994)
    
    # Applying the binary operator 'and' (line 15)
    result_and_keyword_291996 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 11), 'and', result_not__291991, iterable_call_result_291995)
    
    # Assigning a type to the variable 'stypy_return_type' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'stypy_return_type', result_and_keyword_291996)
    
    # ################# End of '_is_list_like(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_is_list_like' in the type store
    # Getting the type of 'stypy_return_type' (line 13)
    stypy_return_type_291997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_291997)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_is_list_like'
    return stypy_return_type_291997

# Assigning a type to the variable '_is_list_like' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), '_is_list_like', _is_list_like)

@norecursion
def is_called_from_pytest(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'is_called_from_pytest'
    module_type_store = module_type_store.open_function_context('is_called_from_pytest', 18, 0, False)
    
    # Passed parameters checking function
    is_called_from_pytest.stypy_localization = localization
    is_called_from_pytest.stypy_type_of_self = None
    is_called_from_pytest.stypy_type_store = module_type_store
    is_called_from_pytest.stypy_function_name = 'is_called_from_pytest'
    is_called_from_pytest.stypy_param_names_list = []
    is_called_from_pytest.stypy_varargs_param_name = None
    is_called_from_pytest.stypy_kwargs_param_name = None
    is_called_from_pytest.stypy_call_defaults = defaults
    is_called_from_pytest.stypy_call_varargs = varargs
    is_called_from_pytest.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'is_called_from_pytest', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'is_called_from_pytest', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'is_called_from_pytest(...)' code ##################

    unicode_291998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 4), 'unicode', u'Returns whether the call was done from pytest')
    
    # Call to getattr(...): (line 20)
    # Processing the call arguments (line 20)
    # Getting the type of 'matplotlib' (line 20)
    matplotlib_292000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 19), 'matplotlib', False)
    unicode_292001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 31), 'unicode', u'_called_from_pytest')
    # Getting the type of 'False' (line 20)
    False_292002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 54), 'False', False)
    # Processing the call keyword arguments (line 20)
    kwargs_292003 = {}
    # Getting the type of 'getattr' (line 20)
    getattr_291999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 11), 'getattr', False)
    # Calling getattr(args, kwargs) (line 20)
    getattr_call_result_292004 = invoke(stypy.reporting.localization.Localization(__file__, 20, 11), getattr_291999, *[matplotlib_292000, unicode_292001, False_292002], **kwargs_292003)
    
    # Assigning a type to the variable 'stypy_return_type' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'stypy_return_type', getattr_call_result_292004)
    
    # ################# End of 'is_called_from_pytest(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_called_from_pytest' in the type store
    # Getting the type of 'stypy_return_type' (line 18)
    stypy_return_type_292005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_292005)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_called_from_pytest'
    return stypy_return_type_292005

# Assigning a type to the variable 'is_called_from_pytest' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'is_called_from_pytest', is_called_from_pytest)

@norecursion
def _getrawcode(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 24)
    True_292006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 29), 'True')
    defaults = [True_292006]
    # Create a new context for function '_getrawcode'
    module_type_store = module_type_store.open_function_context('_getrawcode', 24, 0, False)
    
    # Passed parameters checking function
    _getrawcode.stypy_localization = localization
    _getrawcode.stypy_type_of_self = None
    _getrawcode.stypy_type_store = module_type_store
    _getrawcode.stypy_function_name = '_getrawcode'
    _getrawcode.stypy_param_names_list = ['obj', 'trycall']
    _getrawcode.stypy_varargs_param_name = None
    _getrawcode.stypy_kwargs_param_name = None
    _getrawcode.stypy_call_defaults = defaults
    _getrawcode.stypy_call_varargs = varargs
    _getrawcode.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_getrawcode', ['obj', 'trycall'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_getrawcode', localization, ['obj', 'trycall'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_getrawcode(...)' code ##################

    unicode_292007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 4), 'unicode', u'Return code object for given function.')
    
    
    # SSA begins for try-except statement (line 26)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    # Getting the type of 'obj' (line 27)
    obj_292008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 15), 'obj')
    # Obtaining the member '__code__' of a type (line 27)
    code___292009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 15), obj_292008, '__code__')
    # Assigning a type to the variable 'stypy_return_type' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'stypy_return_type', code___292009)
    # SSA branch for the except part of a try statement (line 26)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 26)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Call to a Name (line 29):
    
    # Call to getattr(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of 'obj' (line 29)
    obj_292011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 22), 'obj', False)
    unicode_292012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 27), 'unicode', u'im_func')
    # Getting the type of 'obj' (line 29)
    obj_292013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 38), 'obj', False)
    # Processing the call keyword arguments (line 29)
    kwargs_292014 = {}
    # Getting the type of 'getattr' (line 29)
    getattr_292010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 14), 'getattr', False)
    # Calling getattr(args, kwargs) (line 29)
    getattr_call_result_292015 = invoke(stypy.reporting.localization.Localization(__file__, 29, 14), getattr_292010, *[obj_292011, unicode_292012, obj_292013], **kwargs_292014)
    
    # Assigning a type to the variable 'obj' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'obj', getattr_call_result_292015)
    
    # Assigning a Call to a Name (line 30):
    
    # Call to getattr(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'obj' (line 30)
    obj_292017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 22), 'obj', False)
    unicode_292018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 27), 'unicode', u'func_code')
    # Getting the type of 'obj' (line 30)
    obj_292019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 40), 'obj', False)
    # Processing the call keyword arguments (line 30)
    kwargs_292020 = {}
    # Getting the type of 'getattr' (line 30)
    getattr_292016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 14), 'getattr', False)
    # Calling getattr(args, kwargs) (line 30)
    getattr_call_result_292021 = invoke(stypy.reporting.localization.Localization(__file__, 30, 14), getattr_292016, *[obj_292017, unicode_292018, obj_292019], **kwargs_292020)
    
    # Assigning a type to the variable 'obj' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'obj', getattr_call_result_292021)
    
    # Assigning a Call to a Name (line 31):
    
    # Call to getattr(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'obj' (line 31)
    obj_292023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 22), 'obj', False)
    unicode_292024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 27), 'unicode', u'f_code')
    # Getting the type of 'obj' (line 31)
    obj_292025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 37), 'obj', False)
    # Processing the call keyword arguments (line 31)
    kwargs_292026 = {}
    # Getting the type of 'getattr' (line 31)
    getattr_292022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 14), 'getattr', False)
    # Calling getattr(args, kwargs) (line 31)
    getattr_call_result_292027 = invoke(stypy.reporting.localization.Localization(__file__, 31, 14), getattr_292022, *[obj_292023, unicode_292024, obj_292025], **kwargs_292026)
    
    # Assigning a type to the variable 'obj' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'obj', getattr_call_result_292027)
    
    # Assigning a Call to a Name (line 32):
    
    # Call to getattr(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'obj' (line 32)
    obj_292029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 22), 'obj', False)
    unicode_292030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 27), 'unicode', u'__code__')
    # Getting the type of 'obj' (line 32)
    obj_292031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 39), 'obj', False)
    # Processing the call keyword arguments (line 32)
    kwargs_292032 = {}
    # Getting the type of 'getattr' (line 32)
    getattr_292028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 14), 'getattr', False)
    # Calling getattr(args, kwargs) (line 32)
    getattr_call_result_292033 = invoke(stypy.reporting.localization.Localization(__file__, 32, 14), getattr_292028, *[obj_292029, unicode_292030, obj_292031], **kwargs_292032)
    
    # Assigning a type to the variable 'obj' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'obj', getattr_call_result_292033)
    
    
    # Evaluating a boolean operation
    # Getting the type of 'trycall' (line 33)
    trycall_292034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 11), 'trycall')
    
    
    # Call to hasattr(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'obj' (line 33)
    obj_292036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 35), 'obj', False)
    unicode_292037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 40), 'unicode', u'co_firstlineno')
    # Processing the call keyword arguments (line 33)
    kwargs_292038 = {}
    # Getting the type of 'hasattr' (line 33)
    hasattr_292035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 27), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 33)
    hasattr_call_result_292039 = invoke(stypy.reporting.localization.Localization(__file__, 33, 27), hasattr_292035, *[obj_292036, unicode_292037], **kwargs_292038)
    
    # Applying the 'not' unary operator (line 33)
    result_not__292040 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 23), 'not', hasattr_call_result_292039)
    
    # Applying the binary operator 'and' (line 33)
    result_and_keyword_292041 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 11), 'and', trycall_292034, result_not__292040)
    
    # Testing the type of an if condition (line 33)
    if_condition_292042 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 33, 8), result_and_keyword_292041)
    # Assigning a type to the variable 'if_condition_292042' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'if_condition_292042', if_condition_292042)
    # SSA begins for if statement (line 33)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Evaluating a boolean operation
    
    # Call to hasattr(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'obj' (line 34)
    obj_292044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 23), 'obj', False)
    unicode_292045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 28), 'unicode', u'__call__')
    # Processing the call keyword arguments (line 34)
    kwargs_292046 = {}
    # Getting the type of 'hasattr' (line 34)
    hasattr_292043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 15), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 34)
    hasattr_call_result_292047 = invoke(stypy.reporting.localization.Localization(__file__, 34, 15), hasattr_292043, *[obj_292044, unicode_292045], **kwargs_292046)
    
    
    
    # Call to isclass(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'obj' (line 34)
    obj_292050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 64), 'obj', False)
    # Processing the call keyword arguments (line 34)
    kwargs_292051 = {}
    # Getting the type of 'inspect' (line 34)
    inspect_292048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 48), 'inspect', False)
    # Obtaining the member 'isclass' of a type (line 34)
    isclass_292049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 48), inspect_292048, 'isclass')
    # Calling isclass(args, kwargs) (line 34)
    isclass_call_result_292052 = invoke(stypy.reporting.localization.Localization(__file__, 34, 48), isclass_292049, *[obj_292050], **kwargs_292051)
    
    # Applying the 'not' unary operator (line 34)
    result_not__292053 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 44), 'not', isclass_call_result_292052)
    
    # Applying the binary operator 'and' (line 34)
    result_and_keyword_292054 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 15), 'and', hasattr_call_result_292047, result_not__292053)
    
    # Testing the type of an if condition (line 34)
    if_condition_292055 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 34, 12), result_and_keyword_292054)
    # Assigning a type to the variable 'if_condition_292055' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'if_condition_292055', if_condition_292055)
    # SSA begins for if statement (line 34)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 35):
    
    # Call to getrawcode(...): (line 35)
    # Processing the call arguments (line 35)
    # Getting the type of 'obj' (line 35)
    obj_292057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 31), 'obj', False)
    # Obtaining the member '__call__' of a type (line 35)
    call___292058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 31), obj_292057, '__call__')
    # Processing the call keyword arguments (line 35)
    # Getting the type of 'False' (line 35)
    False_292059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 53), 'False', False)
    keyword_292060 = False_292059
    kwargs_292061 = {'trycall': keyword_292060}
    # Getting the type of 'getrawcode' (line 35)
    getrawcode_292056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 20), 'getrawcode', False)
    # Calling getrawcode(args, kwargs) (line 35)
    getrawcode_call_result_292062 = invoke(stypy.reporting.localization.Localization(__file__, 35, 20), getrawcode_292056, *[call___292058], **kwargs_292061)
    
    # Assigning a type to the variable 'x' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 16), 'x', getrawcode_call_result_292062)
    
    # Type idiom detected: calculating its left and rigth part (line 36)
    unicode_292063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 30), 'unicode', u'co_firstlineno')
    # Getting the type of 'x' (line 36)
    x_292064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 27), 'x')
    
    (may_be_292065, more_types_in_union_292066) = may_provide_member(unicode_292063, x_292064)

    if may_be_292065:

        if more_types_in_union_292066:
            # Runtime conditional SSA (line 36)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'x' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'x', remove_not_member_provider_from_union(x_292064, u'co_firstlineno'))
        # Getting the type of 'x' (line 37)
        x_292067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 27), 'x')
        # Assigning a type to the variable 'stypy_return_type' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 20), 'stypy_return_type', x_292067)

        if more_types_in_union_292066:
            # SSA join for if statement (line 36)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 34)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 33)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'obj' (line 38)
    obj_292068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 15), 'obj')
    # Assigning a type to the variable 'stypy_return_type' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'stypy_return_type', obj_292068)
    # SSA join for try-except statement (line 26)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_getrawcode(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_getrawcode' in the type store
    # Getting the type of 'stypy_return_type' (line 24)
    stypy_return_type_292069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_292069)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_getrawcode'
    return stypy_return_type_292069

# Assigning a type to the variable '_getrawcode' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), '_getrawcode', _getrawcode)

@norecursion
def _copy_metadata(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_copy_metadata'
    module_type_store = module_type_store.open_function_context('_copy_metadata', 41, 0, False)
    
    # Passed parameters checking function
    _copy_metadata.stypy_localization = localization
    _copy_metadata.stypy_type_of_self = None
    _copy_metadata.stypy_type_store = module_type_store
    _copy_metadata.stypy_function_name = '_copy_metadata'
    _copy_metadata.stypy_param_names_list = ['src_func', 'tgt_func']
    _copy_metadata.stypy_varargs_param_name = None
    _copy_metadata.stypy_kwargs_param_name = None
    _copy_metadata.stypy_call_defaults = defaults
    _copy_metadata.stypy_call_varargs = varargs
    _copy_metadata.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_copy_metadata', ['src_func', 'tgt_func'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_copy_metadata', localization, ['src_func', 'tgt_func'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_copy_metadata(...)' code ##################

    unicode_292070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 4), 'unicode', u'Replicates metadata of the function. Returns target function.')
    
    # Call to update(...): (line 43)
    # Processing the call arguments (line 43)
    # Getting the type of 'src_func' (line 43)
    src_func_292074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 29), 'src_func', False)
    # Obtaining the member '__dict__' of a type (line 43)
    dict___292075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 29), src_func_292074, '__dict__')
    # Processing the call keyword arguments (line 43)
    kwargs_292076 = {}
    # Getting the type of 'tgt_func' (line 43)
    tgt_func_292071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'tgt_func', False)
    # Obtaining the member '__dict__' of a type (line 43)
    dict___292072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 4), tgt_func_292071, '__dict__')
    # Obtaining the member 'update' of a type (line 43)
    update_292073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 4), dict___292072, 'update')
    # Calling update(args, kwargs) (line 43)
    update_call_result_292077 = invoke(stypy.reporting.localization.Localization(__file__, 43, 4), update_292073, *[dict___292075], **kwargs_292076)
    
    
    # Assigning a Attribute to a Attribute (line 44):
    # Getting the type of 'src_func' (line 44)
    src_func_292078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 23), 'src_func')
    # Obtaining the member '__doc__' of a type (line 44)
    doc___292079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 23), src_func_292078, '__doc__')
    # Getting the type of 'tgt_func' (line 44)
    tgt_func_292080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'tgt_func')
    # Setting the type of the member '__doc__' of a type (line 44)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 4), tgt_func_292080, '__doc__', doc___292079)
    
    # Assigning a Attribute to a Attribute (line 45):
    # Getting the type of 'src_func' (line 45)
    src_func_292081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 26), 'src_func')
    # Obtaining the member '__module__' of a type (line 45)
    module___292082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 26), src_func_292081, '__module__')
    # Getting the type of 'tgt_func' (line 45)
    tgt_func_292083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'tgt_func')
    # Setting the type of the member '__module__' of a type (line 45)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 4), tgt_func_292083, '__module__', module___292082)
    
    # Assigning a Attribute to a Attribute (line 46):
    # Getting the type of 'src_func' (line 46)
    src_func_292084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 24), 'src_func')
    # Obtaining the member '__name__' of a type (line 46)
    name___292085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 24), src_func_292084, '__name__')
    # Getting the type of 'tgt_func' (line 46)
    tgt_func_292086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'tgt_func')
    # Setting the type of the member '__name__' of a type (line 46)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 4), tgt_func_292086, '__name__', name___292085)
    
    # Type idiom detected: calculating its left and rigth part (line 47)
    unicode_292087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 25), 'unicode', u'__qualname__')
    # Getting the type of 'src_func' (line 47)
    src_func_292088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 15), 'src_func')
    
    (may_be_292089, more_types_in_union_292090) = may_provide_member(unicode_292087, src_func_292088)

    if may_be_292089:

        if more_types_in_union_292090:
            # Runtime conditional SSA (line 47)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'src_func' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'src_func', remove_not_member_provider_from_union(src_func_292088, u'__qualname__'))
        
        # Assigning a Attribute to a Attribute (line 48):
        # Getting the type of 'src_func' (line 48)
        src_func_292091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 32), 'src_func')
        # Obtaining the member '__qualname__' of a type (line 48)
        qualname___292092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 32), src_func_292091, '__qualname__')
        # Getting the type of 'tgt_func' (line 48)
        tgt_func_292093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'tgt_func')
        # Setting the type of the member '__qualname__' of a type (line 48)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), tgt_func_292093, '__qualname__', qualname___292092)

        if more_types_in_union_292090:
            # SSA join for if statement (line 47)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 49)
    unicode_292094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 29), 'unicode', u'compat_co_firstlineno')
    # Getting the type of 'tgt_func' (line 49)
    tgt_func_292095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 19), 'tgt_func')
    
    (may_be_292096, more_types_in_union_292097) = may_not_provide_member(unicode_292094, tgt_func_292095)

    if may_be_292096:

        if more_types_in_union_292097:
            # Runtime conditional SSA (line 49)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'tgt_func' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'tgt_func', remove_member_provider_from_union(tgt_func_292095, u'compat_co_firstlineno'))
        
        # Assigning a Attribute to a Attribute (line 50):
        
        # Call to _getrawcode(...): (line 50)
        # Processing the call arguments (line 50)
        # Getting the type of 'src_func' (line 50)
        src_func_292099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 53), 'src_func', False)
        # Processing the call keyword arguments (line 50)
        kwargs_292100 = {}
        # Getting the type of '_getrawcode' (line 50)
        _getrawcode_292098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 41), '_getrawcode', False)
        # Calling _getrawcode(args, kwargs) (line 50)
        _getrawcode_call_result_292101 = invoke(stypy.reporting.localization.Localization(__file__, 50, 41), _getrawcode_292098, *[src_func_292099], **kwargs_292100)
        
        # Obtaining the member 'co_firstlineno' of a type (line 50)
        co_firstlineno_292102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 41), _getrawcode_call_result_292101, 'co_firstlineno')
        # Getting the type of 'tgt_func' (line 50)
        tgt_func_292103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'tgt_func')
        # Setting the type of the member 'compat_co_firstlineno' of a type (line 50)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 8), tgt_func_292103, 'compat_co_firstlineno', co_firstlineno_292102)

        if more_types_in_union_292097:
            # SSA join for if statement (line 49)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'tgt_func' (line 51)
    tgt_func_292104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 11), 'tgt_func')
    # Assigning a type to the variable 'stypy_return_type' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'stypy_return_type', tgt_func_292104)
    
    # ################# End of '_copy_metadata(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_copy_metadata' in the type store
    # Getting the type of 'stypy_return_type' (line 41)
    stypy_return_type_292105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_292105)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_copy_metadata'
    return stypy_return_type_292105

# Assigning a type to the variable '_copy_metadata' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), '_copy_metadata', _copy_metadata)

@norecursion
def assert_produces_warning(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'Warning' (line 56)
    Warning_292106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 45), 'Warning')
    unicode_292107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 67), 'unicode', u'always')
    # Getting the type of 'None' (line 57)
    None_292108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 34), 'None')
    defaults = [Warning_292106, unicode_292107, None_292108]
    # Create a new context for function 'assert_produces_warning'
    module_type_store = module_type_store.open_function_context('assert_produces_warning', 55, 0, False)
    
    # Passed parameters checking function
    assert_produces_warning.stypy_localization = localization
    assert_produces_warning.stypy_type_of_self = None
    assert_produces_warning.stypy_type_store = module_type_store
    assert_produces_warning.stypy_function_name = 'assert_produces_warning'
    assert_produces_warning.stypy_param_names_list = ['expected_warning', 'filter_level', 'clear']
    assert_produces_warning.stypy_varargs_param_name = None
    assert_produces_warning.stypy_kwargs_param_name = None
    assert_produces_warning.stypy_call_defaults = defaults
    assert_produces_warning.stypy_call_varargs = varargs
    assert_produces_warning.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'assert_produces_warning', ['expected_warning', 'filter_level', 'clear'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'assert_produces_warning', localization, ['expected_warning', 'filter_level', 'clear'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'assert_produces_warning(...)' code ##################

    unicode_292109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, (-1)), 'unicode', u"\n    Context manager for running code that expects to raise (or not raise)\n    warnings.  Checks that code raises the expected warning and only the\n    expected warning. Pass ``False`` or ``None`` to check that it does *not*\n    raise a warning. Defaults to ``exception.Warning``, baseclass of all\n    Warnings. (basically a wrapper around ``warnings.catch_warnings``).\n\n    >>> import warnings\n    >>> with assert_produces_warning():\n    ...     warnings.warn(UserWarning())\n    ...\n    >>> with assert_produces_warning(False):\n    ...     warnings.warn(RuntimeWarning())\n    ...\n    Traceback (most recent call last):\n        ...\n    AssertionError: Caused unexpected warning(s): ['RuntimeWarning'].\n    >>> with assert_produces_warning(UserWarning):\n    ...     warnings.warn(RuntimeWarning())\n    Traceback (most recent call last):\n        ...\n    AssertionError: Did not see expected warning of class 'UserWarning'.\n\n    ..warn:: This is *not* thread-safe.\n    ")
    
    # Call to catch_warnings(...): (line 83)
    # Processing the call keyword arguments (line 83)
    # Getting the type of 'True' (line 83)
    True_292112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 40), 'True', False)
    keyword_292113 = True_292112
    kwargs_292114 = {'record': keyword_292113}
    # Getting the type of 'warnings' (line 83)
    warnings_292110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 9), 'warnings', False)
    # Obtaining the member 'catch_warnings' of a type (line 83)
    catch_warnings_292111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 9), warnings_292110, 'catch_warnings')
    # Calling catch_warnings(args, kwargs) (line 83)
    catch_warnings_call_result_292115 = invoke(stypy.reporting.localization.Localization(__file__, 83, 9), catch_warnings_292111, *[], **kwargs_292114)
    
    with_292116 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 83, 9), catch_warnings_call_result_292115, 'with parameter', '__enter__', '__exit__')

    if with_292116:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 83)
        enter___292117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 9), catch_warnings_call_result_292115, '__enter__')
        with_enter_292118 = invoke(stypy.reporting.localization.Localization(__file__, 83, 9), enter___292117)
        # Assigning a type to the variable 'w' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 9), 'w', with_enter_292118)
        
        # Type idiom detected: calculating its left and rigth part (line 85)
        # Getting the type of 'clear' (line 85)
        clear_292119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'clear')
        # Getting the type of 'None' (line 85)
        None_292120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 24), 'None')
        
        (may_be_292121, more_types_in_union_292122) = may_not_be_none(clear_292119, None_292120)

        if may_be_292121:

            if more_types_in_union_292122:
                # Runtime conditional SSA (line 85)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            
            # Call to _is_list_like(...): (line 89)
            # Processing the call arguments (line 89)
            # Getting the type of 'clear' (line 89)
            clear_292124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 33), 'clear', False)
            # Processing the call keyword arguments (line 89)
            kwargs_292125 = {}
            # Getting the type of '_is_list_like' (line 89)
            _is_list_like_292123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 19), '_is_list_like', False)
            # Calling _is_list_like(args, kwargs) (line 89)
            _is_list_like_call_result_292126 = invoke(stypy.reporting.localization.Localization(__file__, 89, 19), _is_list_like_292123, *[clear_292124], **kwargs_292125)
            
            # Applying the 'not' unary operator (line 89)
            result_not__292127 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 15), 'not', _is_list_like_call_result_292126)
            
            # Testing the type of an if condition (line 89)
            if_condition_292128 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 12), result_not__292127)
            # Assigning a type to the variable 'if_condition_292128' (line 89)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'if_condition_292128', if_condition_292128)
            # SSA begins for if statement (line 89)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a List to a Name (line 90):
            
            # Obtaining an instance of the builtin type 'list' (line 90)
            list_292129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 24), 'list')
            # Adding type elements to the builtin type 'list' instance (line 90)
            # Adding element type (line 90)
            # Getting the type of 'clear' (line 90)
            clear_292130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 25), 'clear')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 24), list_292129, clear_292130)
            
            # Assigning a type to the variable 'clear' (line 90)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 16), 'clear', list_292129)
            # SSA join for if statement (line 89)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Getting the type of 'clear' (line 91)
            clear_292131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 21), 'clear')
            # Testing the type of a for loop iterable (line 91)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 91, 12), clear_292131)
            # Getting the type of the for loop variable (line 91)
            for_loop_var_292132 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 91, 12), clear_292131)
            # Assigning a type to the variable 'm' (line 91)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'm', for_loop_var_292132)
            # SSA begins for a for statement (line 91)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to clear(...): (line 92)
            # Processing the call keyword arguments (line 92)
            kwargs_292140 = {}
            
            # Call to getattr(...): (line 92)
            # Processing the call arguments (line 92)
            # Getting the type of 'm' (line 92)
            m_292134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 24), 'm', False)
            unicode_292135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 27), 'unicode', u'__warningregistry__')
            
            # Obtaining an instance of the builtin type 'dict' (line 92)
            dict_292136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 50), 'dict')
            # Adding type elements to the builtin type 'dict' instance (line 92)
            
            # Processing the call keyword arguments (line 92)
            kwargs_292137 = {}
            # Getting the type of 'getattr' (line 92)
            getattr_292133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 16), 'getattr', False)
            # Calling getattr(args, kwargs) (line 92)
            getattr_call_result_292138 = invoke(stypy.reporting.localization.Localization(__file__, 92, 16), getattr_292133, *[m_292134, unicode_292135, dict_292136], **kwargs_292137)
            
            # Obtaining the member 'clear' of a type (line 92)
            clear_292139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 16), getattr_call_result_292138, 'clear')
            # Calling clear(args, kwargs) (line 92)
            clear_call_result_292141 = invoke(stypy.reporting.localization.Localization(__file__, 92, 16), clear_292139, *[], **kwargs_292140)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_292122:
                # SSA join for if statement (line 85)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Name (line 94):
        # Getting the type of 'False' (line 94)
        False_292142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 22), 'False')
        # Assigning a type to the variable 'saw_warning' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'saw_warning', False_292142)
        
        # Call to simplefilter(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'filter_level' (line 95)
        filter_level_292145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 30), 'filter_level', False)
        # Processing the call keyword arguments (line 95)
        kwargs_292146 = {}
        # Getting the type of 'warnings' (line 95)
        warnings_292143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'warnings', False)
        # Obtaining the member 'simplefilter' of a type (line 95)
        simplefilter_292144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), warnings_292143, 'simplefilter')
        # Calling simplefilter(args, kwargs) (line 95)
        simplefilter_call_result_292147 = invoke(stypy.reporting.localization.Localization(__file__, 95, 8), simplefilter_292144, *[filter_level_292145], **kwargs_292146)
        
        # Creating a generator
        # Getting the type of 'w' (line 96)
        w_292148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 14), 'w')
        GeneratorType_292149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 8), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 8), GeneratorType_292149, w_292148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'stypy_return_type', GeneratorType_292149)
        
        # Assigning a List to a Name (line 97):
        
        # Obtaining an instance of the builtin type 'list' (line 97)
        list_292150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 25), 'list')
        # Adding type elements to the builtin type 'list' instance (line 97)
        
        # Assigning a type to the variable 'extra_warnings' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'extra_warnings', list_292150)
        
        # Getting the type of 'w' (line 98)
        w_292151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 30), 'w')
        # Testing the type of a for loop iterable (line 98)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 98, 8), w_292151)
        # Getting the type of the for loop variable (line 98)
        for_loop_var_292152 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 98, 8), w_292151)
        # Assigning a type to the variable 'actual_warning' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'actual_warning', for_loop_var_292152)
        # SSA begins for a for statement (line 98)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        # Getting the type of 'expected_warning' (line 99)
        expected_warning_292153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'expected_warning')
        
        # Call to issubclass(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'actual_warning' (line 99)
        actual_warning_292155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 48), 'actual_warning', False)
        # Obtaining the member 'category' of a type (line 99)
        category_292156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 48), actual_warning_292155, 'category')
        # Getting the type of 'expected_warning' (line 100)
        expected_warning_292157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 48), 'expected_warning', False)
        # Processing the call keyword arguments (line 99)
        kwargs_292158 = {}
        # Getting the type of 'issubclass' (line 99)
        issubclass_292154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 37), 'issubclass', False)
        # Calling issubclass(args, kwargs) (line 99)
        issubclass_call_result_292159 = invoke(stypy.reporting.localization.Localization(__file__, 99, 37), issubclass_292154, *[category_292156, expected_warning_292157], **kwargs_292158)
        
        # Applying the binary operator 'and' (line 99)
        result_and_keyword_292160 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 16), 'and', expected_warning_292153, issubclass_call_result_292159)
        
        # Testing the type of an if condition (line 99)
        if_condition_292161 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 99, 12), result_and_keyword_292160)
        # Assigning a type to the variable 'if_condition_292161' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'if_condition_292161', if_condition_292161)
        # SSA begins for if statement (line 99)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 101):
        # Getting the type of 'True' (line 101)
        True_292162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 30), 'True')
        # Assigning a type to the variable 'saw_warning' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 16), 'saw_warning', True_292162)
        # SSA branch for the else part of an if statement (line 99)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'actual_warning' (line 103)
        actual_warning_292165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 38), 'actual_warning', False)
        # Obtaining the member 'category' of a type (line 103)
        category_292166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 38), actual_warning_292165, 'category')
        # Obtaining the member '__name__' of a type (line 103)
        name___292167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 38), category_292166, '__name__')
        # Processing the call keyword arguments (line 103)
        kwargs_292168 = {}
        # Getting the type of 'extra_warnings' (line 103)
        extra_warnings_292163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 16), 'extra_warnings', False)
        # Obtaining the member 'append' of a type (line 103)
        append_292164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 16), extra_warnings_292163, 'append')
        # Calling append(args, kwargs) (line 103)
        append_call_result_292169 = invoke(stypy.reporting.localization.Localization(__file__, 103, 16), append_292164, *[name___292167], **kwargs_292168)
        
        # SSA join for if statement (line 99)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'expected_warning' (line 104)
        expected_warning_292170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 11), 'expected_warning')
        # Testing the type of an if condition (line 104)
        if_condition_292171 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 104, 8), expected_warning_292170)
        # Assigning a type to the variable 'if_condition_292171' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'if_condition_292171', if_condition_292171)
        # SSA begins for if statement (line 104)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Evaluating assert statement condition
        # Getting the type of 'saw_warning' (line 105)
        saw_warning_292172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 19), 'saw_warning')
        # SSA join for if statement (line 104)
        module_type_store = module_type_store.join_ssa_context()
        
        # Evaluating assert statement condition
        
        # Getting the type of 'extra_warnings' (line 107)
        extra_warnings_292173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 19), 'extra_warnings')
        # Applying the 'not' unary operator (line 107)
        result_not__292174 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 15), 'not', extra_warnings_292173)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 83)
        exit___292175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 9), catch_warnings_call_result_292115, '__exit__')
        with_exit_292176 = invoke(stypy.reporting.localization.Localization(__file__, 83, 9), exit___292175, None, None, None)

    
    # ################# End of 'assert_produces_warning(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'assert_produces_warning' in the type store
    # Getting the type of 'stypy_return_type' (line 55)
    stypy_return_type_292177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_292177)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'assert_produces_warning'
    return stypy_return_type_292177

# Assigning a type to the variable 'assert_produces_warning' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'assert_produces_warning', assert_produces_warning)

@norecursion
def set_font_settings_for_testing(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'set_font_settings_for_testing'
    module_type_store = module_type_store.open_function_context('set_font_settings_for_testing', 111, 0, False)
    
    # Passed parameters checking function
    set_font_settings_for_testing.stypy_localization = localization
    set_font_settings_for_testing.stypy_type_of_self = None
    set_font_settings_for_testing.stypy_type_store = module_type_store
    set_font_settings_for_testing.stypy_function_name = 'set_font_settings_for_testing'
    set_font_settings_for_testing.stypy_param_names_list = []
    set_font_settings_for_testing.stypy_varargs_param_name = None
    set_font_settings_for_testing.stypy_kwargs_param_name = None
    set_font_settings_for_testing.stypy_call_defaults = defaults
    set_font_settings_for_testing.stypy_call_varargs = varargs
    set_font_settings_for_testing.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'set_font_settings_for_testing', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'set_font_settings_for_testing', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'set_font_settings_for_testing(...)' code ##################

    
    # Assigning a Str to a Subscript (line 112):
    unicode_292178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 30), 'unicode', u'DejaVu Sans')
    # Getting the type of 'rcParams' (line 112)
    rcParams_292179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'rcParams')
    unicode_292180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 13), 'unicode', u'font.family')
    # Storing an element on a container (line 112)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 4), rcParams_292179, (unicode_292180, unicode_292178))
    
    # Assigning a Name to a Subscript (line 113):
    # Getting the type of 'False' (line 113)
    False_292181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 31), 'False')
    # Getting the type of 'rcParams' (line 113)
    rcParams_292182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'rcParams')
    unicode_292183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 13), 'unicode', u'text.hinting')
    # Storing an element on a container (line 113)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 4), rcParams_292182, (unicode_292183, False_292181))
    
    # Assigning a Num to a Subscript (line 114):
    int_292184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 38), 'int')
    # Getting the type of 'rcParams' (line 114)
    rcParams_292185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'rcParams')
    unicode_292186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 13), 'unicode', u'text.hinting_factor')
    # Storing an element on a container (line 114)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 4), rcParams_292185, (unicode_292186, int_292184))
    
    # ################# End of 'set_font_settings_for_testing(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'set_font_settings_for_testing' in the type store
    # Getting the type of 'stypy_return_type' (line 111)
    stypy_return_type_292187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_292187)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'set_font_settings_for_testing'
    return stypy_return_type_292187

# Assigning a type to the variable 'set_font_settings_for_testing' (line 111)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 0), 'set_font_settings_for_testing', set_font_settings_for_testing)

@norecursion
def set_reproducibility_for_testing(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'set_reproducibility_for_testing'
    module_type_store = module_type_store.open_function_context('set_reproducibility_for_testing', 117, 0, False)
    
    # Passed parameters checking function
    set_reproducibility_for_testing.stypy_localization = localization
    set_reproducibility_for_testing.stypy_type_of_self = None
    set_reproducibility_for_testing.stypy_type_store = module_type_store
    set_reproducibility_for_testing.stypy_function_name = 'set_reproducibility_for_testing'
    set_reproducibility_for_testing.stypy_param_names_list = []
    set_reproducibility_for_testing.stypy_varargs_param_name = None
    set_reproducibility_for_testing.stypy_kwargs_param_name = None
    set_reproducibility_for_testing.stypy_call_defaults = defaults
    set_reproducibility_for_testing.stypy_call_varargs = varargs
    set_reproducibility_for_testing.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'set_reproducibility_for_testing', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'set_reproducibility_for_testing', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'set_reproducibility_for_testing(...)' code ##################

    
    # Assigning a Str to a Subscript (line 118):
    unicode_292188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 31), 'unicode', u'matplotlib')
    # Getting the type of 'rcParams' (line 118)
    rcParams_292189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'rcParams')
    unicode_292190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 13), 'unicode', u'svg.hashsalt')
    # Storing an element on a container (line 118)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 4), rcParams_292189, (unicode_292190, unicode_292188))
    
    # ################# End of 'set_reproducibility_for_testing(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'set_reproducibility_for_testing' in the type store
    # Getting the type of 'stypy_return_type' (line 117)
    stypy_return_type_292191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_292191)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'set_reproducibility_for_testing'
    return stypy_return_type_292191

# Assigning a type to the variable 'set_reproducibility_for_testing' (line 117)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 0), 'set_reproducibility_for_testing', set_reproducibility_for_testing)

@norecursion
def setup(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'setup'
    module_type_store = module_type_store.open_function_context('setup', 121, 0, False)
    
    # Passed parameters checking function
    setup.stypy_localization = localization
    setup.stypy_type_of_self = None
    setup.stypy_type_store = module_type_store
    setup.stypy_function_name = 'setup'
    setup.stypy_param_names_list = []
    setup.stypy_varargs_param_name = None
    setup.stypy_kwargs_param_name = None
    setup.stypy_call_defaults = defaults
    setup.stypy_call_varargs = varargs
    setup.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'setup', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'setup', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'setup(...)' code ##################

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 124, 4))
    
    # 'import locale' statement (line 124)
    import locale

    import_module(stypy.reporting.localization.Localization(__file__, 124, 4), 'locale', locale, module_type_store)
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 125, 4))
    
    # 'import warnings' statement (line 125)
    import warnings

    import_module(stypy.reporting.localization.Localization(__file__, 125, 4), 'warnings', warnings, module_type_store)
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 126, 4))
    
    # 'from matplotlib.backends import backend_agg, backend_pdf, backend_svg' statement (line 126)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/')
    import_292192 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 126, 4), 'matplotlib.backends')

    if (type(import_292192) is not StypyTypeError):

        if (import_292192 != 'pyd_module'):
            __import__(import_292192)
            sys_modules_292193 = sys.modules[import_292192]
            import_from_module(stypy.reporting.localization.Localization(__file__, 126, 4), 'matplotlib.backends', sys_modules_292193.module_type_store, module_type_store, ['backend_agg', 'backend_pdf', 'backend_svg'])
            nest_module(stypy.reporting.localization.Localization(__file__, 126, 4), __file__, sys_modules_292193, sys_modules_292193.module_type_store, module_type_store)
        else:
            from matplotlib.backends import backend_agg, backend_pdf, backend_svg

            import_from_module(stypy.reporting.localization.Localization(__file__, 126, 4), 'matplotlib.backends', None, module_type_store, ['backend_agg', 'backend_pdf', 'backend_svg'], [backend_agg, backend_pdf, backend_svg])

    else:
        # Assigning a type to the variable 'matplotlib.backends' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'matplotlib.backends', import_292192)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/')
    
    
    
    # SSA begins for try-except statement (line 128)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to setlocale(...): (line 129)
    # Processing the call arguments (line 129)
    # Getting the type of 'locale' (line 129)
    locale_292196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 25), 'locale', False)
    # Obtaining the member 'LC_ALL' of a type (line 129)
    LC_ALL_292197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 25), locale_292196, 'LC_ALL')
    
    # Call to str(...): (line 129)
    # Processing the call arguments (line 129)
    unicode_292199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 44), 'unicode', u'en_US.UTF-8')
    # Processing the call keyword arguments (line 129)
    kwargs_292200 = {}
    # Getting the type of 'str' (line 129)
    str_292198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 40), 'str', False)
    # Calling str(args, kwargs) (line 129)
    str_call_result_292201 = invoke(stypy.reporting.localization.Localization(__file__, 129, 40), str_292198, *[unicode_292199], **kwargs_292200)
    
    # Processing the call keyword arguments (line 129)
    kwargs_292202 = {}
    # Getting the type of 'locale' (line 129)
    locale_292194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'locale', False)
    # Obtaining the member 'setlocale' of a type (line 129)
    setlocale_292195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 8), locale_292194, 'setlocale')
    # Calling setlocale(args, kwargs) (line 129)
    setlocale_call_result_292203 = invoke(stypy.reporting.localization.Localization(__file__, 129, 8), setlocale_292195, *[LC_ALL_292197, str_call_result_292201], **kwargs_292202)
    
    # SSA branch for the except part of a try statement (line 128)
    # SSA branch for the except 'Attribute' branch of a try statement (line 128)
    module_type_store.open_ssa_branch('except')
    
    
    # SSA begins for try-except statement (line 131)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to setlocale(...): (line 132)
    # Processing the call arguments (line 132)
    # Getting the type of 'locale' (line 132)
    locale_292206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 29), 'locale', False)
    # Obtaining the member 'LC_ALL' of a type (line 132)
    LC_ALL_292207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 29), locale_292206, 'LC_ALL')
    
    # Call to str(...): (line 132)
    # Processing the call arguments (line 132)
    unicode_292209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 48), 'unicode', u'English_United States.1252')
    # Processing the call keyword arguments (line 132)
    kwargs_292210 = {}
    # Getting the type of 'str' (line 132)
    str_292208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 44), 'str', False)
    # Calling str(args, kwargs) (line 132)
    str_call_result_292211 = invoke(stypy.reporting.localization.Localization(__file__, 132, 44), str_292208, *[unicode_292209], **kwargs_292210)
    
    # Processing the call keyword arguments (line 132)
    kwargs_292212 = {}
    # Getting the type of 'locale' (line 132)
    locale_292204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'locale', False)
    # Obtaining the member 'setlocale' of a type (line 132)
    setlocale_292205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 12), locale_292204, 'setlocale')
    # Calling setlocale(args, kwargs) (line 132)
    setlocale_call_result_292213 = invoke(stypy.reporting.localization.Localization(__file__, 132, 12), setlocale_292205, *[LC_ALL_292207, str_call_result_292211], **kwargs_292212)
    
    # SSA branch for the except part of a try statement (line 131)
    # SSA branch for the except 'Attribute' branch of a try statement (line 131)
    module_type_store.open_ssa_branch('except')
    
    # Call to warn(...): (line 134)
    # Processing the call arguments (line 134)
    unicode_292216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 16), 'unicode', u'Could not set locale to English/United States. Some date-related tests may fail')
    # Processing the call keyword arguments (line 134)
    kwargs_292217 = {}
    # Getting the type of 'warnings' (line 134)
    warnings_292214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'warnings', False)
    # Obtaining the member 'warn' of a type (line 134)
    warn_292215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 12), warnings_292214, 'warn')
    # Calling warn(args, kwargs) (line 134)
    warn_call_result_292218 = invoke(stypy.reporting.localization.Localization(__file__, 134, 12), warn_292215, *[unicode_292216], **kwargs_292217)
    
    # SSA join for try-except statement (line 131)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for try-except statement (line 128)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to use(...): (line 138)
    # Processing the call arguments (line 138)
    unicode_292220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 8), 'unicode', u'Agg')
    # Processing the call keyword arguments (line 138)
    # Getting the type of 'False' (line 138)
    False_292221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 20), 'False', False)
    keyword_292222 = False_292221
    kwargs_292223 = {'warn': keyword_292222}
    # Getting the type of 'use' (line 138)
    use_292219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'use', False)
    # Calling use(args, kwargs) (line 138)
    use_call_result_292224 = invoke(stypy.reporting.localization.Localization(__file__, 138, 4), use_292219, *[unicode_292220], **kwargs_292223)
    
    
    # Call to rcdefaults(...): (line 143)
    # Processing the call keyword arguments (line 143)
    kwargs_292226 = {}
    # Getting the type of 'rcdefaults' (line 143)
    rcdefaults_292225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'rcdefaults', False)
    # Calling rcdefaults(args, kwargs) (line 143)
    rcdefaults_call_result_292227 = invoke(stypy.reporting.localization.Localization(__file__, 143, 4), rcdefaults_292225, *[], **kwargs_292226)
    
    
    # Call to set_font_settings_for_testing(...): (line 145)
    # Processing the call keyword arguments (line 145)
    kwargs_292229 = {}
    # Getting the type of 'set_font_settings_for_testing' (line 145)
    set_font_settings_for_testing_292228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'set_font_settings_for_testing', False)
    # Calling set_font_settings_for_testing(args, kwargs) (line 145)
    set_font_settings_for_testing_call_result_292230 = invoke(stypy.reporting.localization.Localization(__file__, 145, 4), set_font_settings_for_testing_292228, *[], **kwargs_292229)
    
    
    # Call to set_reproducibility_for_testing(...): (line 146)
    # Processing the call keyword arguments (line 146)
    kwargs_292232 = {}
    # Getting the type of 'set_reproducibility_for_testing' (line 146)
    set_reproducibility_for_testing_292231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'set_reproducibility_for_testing', False)
    # Calling set_reproducibility_for_testing(args, kwargs) (line 146)
    set_reproducibility_for_testing_call_result_292233 = invoke(stypy.reporting.localization.Localization(__file__, 146, 4), set_reproducibility_for_testing_292231, *[], **kwargs_292232)
    
    
    # ################# End of 'setup(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'setup' in the type store
    # Getting the type of 'stypy_return_type' (line 121)
    stypy_return_type_292234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_292234)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'setup'
    return stypy_return_type_292234

# Assigning a type to the variable 'setup' (line 121)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 0), 'setup', setup)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
