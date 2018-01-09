
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import warnings
4: 
5: from numpy.testing import assert_, assert_equal
6: from scipy._lib._numpy_compat import suppress_warnings
7: import pytest
8: from pytest import raises as assert_raises
9: 
10: import scipy.special as sc
11: from scipy.special._ufuncs import _sf_error_test_function
12: 
13: _sf_error_code_map = {
14:     # skip 'ok'
15:     'singular': 1,
16:     'underflow': 2,
17:     'overflow': 3,
18:     'slow': 4,
19:     'loss': 5,
20:     'no_result': 6,
21:     'domain': 7,
22:     'arg': 8,
23:     'other': 9
24: }
25: 
26: _sf_error_actions = [
27:     'ignore',
28:     'warn',
29:     'raise'
30: ]
31: 
32: 
33: def _check_action(fun, args, action):
34:     if action == 'warn':
35:         with pytest.warns(sc.SpecialFunctionWarning):
36:             fun(*args)
37:     elif action == 'raise':
38:         with assert_raises(sc.SpecialFunctionError):
39:             fun(*args)
40:     else:
41:         # action == 'ignore', make sure there are no warnings/exceptions
42:         with warnings.catch_warnings():
43:             warnings.simplefilter("error")
44:             fun(*args)
45: 
46: 
47: def test_geterr():
48:     err = sc.geterr()
49:     for key, value in err.items():
50:         assert_(key in _sf_error_code_map.keys())
51:         assert_(value in _sf_error_actions)
52: 
53: 
54: def test_seterr():
55:     entry_err = sc.geterr()
56:     try:
57:         for category in _sf_error_code_map.keys():
58:             for action in _sf_error_actions:
59:                 geterr_olderr = sc.geterr()
60:                 seterr_olderr = sc.seterr(**{category: action})
61:                 assert_(geterr_olderr == seterr_olderr)
62:                 newerr = sc.geterr()
63:                 assert_(newerr[category] == action)
64:                 geterr_olderr.pop(category)
65:                 newerr.pop(category)
66:                 assert_(geterr_olderr == newerr)
67:                 _check_action(_sf_error_test_function,
68:                               (_sf_error_code_map[category],),
69:                                action)
70:     finally:
71:         sc.seterr(**entry_err)
72: 
73: 
74: def test_errstate_pyx_basic():
75:     olderr = sc.geterr()
76:     with sc.errstate(singular='raise'):
77:         with assert_raises(sc.SpecialFunctionError):
78:             sc.loggamma(0)
79:     assert_equal(olderr, sc.geterr())
80: 
81: 
82: def test_errstate_c_basic():
83:     olderr = sc.geterr()
84:     with sc.errstate(domain='raise'):
85:         with assert_raises(sc.SpecialFunctionError):
86:             sc.spence(-1)
87:     assert_equal(olderr, sc.geterr())
88: 
89: 
90: def test_errstate_cpp_basic():
91:     olderr = sc.geterr()
92:     with sc.errstate(underflow='raise'):
93:         with assert_raises(sc.SpecialFunctionError):
94:             sc.wrightomega(-1000)
95:     assert_equal(olderr, sc.geterr())
96: 
97: 
98: def test_errstate():
99:     for category in _sf_error_code_map.keys():
100:         for action in _sf_error_actions:
101:             olderr = sc.geterr()
102:             with sc.errstate(**{category: action}):
103:                 _check_action(_sf_error_test_function,
104:                               (_sf_error_code_map[category],),
105:                               action)
106:             assert_equal(olderr, sc.geterr())
107: 
108: 
109: def test_errstate_all_but_one():
110:     olderr = sc.geterr()
111:     with sc.errstate(all='raise', singular='ignore'):
112:         sc.gammaln(0)
113:         with assert_raises(sc.SpecialFunctionError):
114:             sc.spence(-1.0)
115:     assert_equal(olderr, sc.geterr())
116: 
117: 
118: def test_errprint():
119:     with suppress_warnings() as sup:
120:         sup.filter(DeprecationWarning, "`errprint` is deprecated!")
121:         flag = sc.errprint(True)
122: 
123:     try:
124:         assert_(isinstance(flag, bool))
125:         with pytest.warns(sc.SpecialFunctionWarning):
126:             sc.loggamma(0)
127:     finally:
128:         with suppress_warnings() as sup:
129:             sup.filter(DeprecationWarning, "`errprint` is deprecated!")
130:             sc.errprint(flag)
131: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import warnings' statement (line 3)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from numpy.testing import assert_, assert_equal' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_559830 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing')

if (type(import_559830) is not StypyTypeError):

    if (import_559830 != 'pyd_module'):
        __import__(import_559830)
        sys_modules_559831 = sys.modules[import_559830]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', sys_modules_559831.module_type_store, module_type_store, ['assert_', 'assert_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_559831, sys_modules_559831.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_, assert_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', None, module_type_store, ['assert_', 'assert_equal'], [assert_, assert_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', import_559830)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy._lib._numpy_compat import suppress_warnings' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_559832 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy._lib._numpy_compat')

if (type(import_559832) is not StypyTypeError):

    if (import_559832 != 'pyd_module'):
        __import__(import_559832)
        sys_modules_559833 = sys.modules[import_559832]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy._lib._numpy_compat', sys_modules_559833.module_type_store, module_type_store, ['suppress_warnings'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_559833, sys_modules_559833.module_type_store, module_type_store)
    else:
        from scipy._lib._numpy_compat import suppress_warnings

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy._lib._numpy_compat', None, module_type_store, ['suppress_warnings'], [suppress_warnings])

else:
    # Assigning a type to the variable 'scipy._lib._numpy_compat' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy._lib._numpy_compat', import_559832)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import pytest' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_559834 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'pytest')

if (type(import_559834) is not StypyTypeError):

    if (import_559834 != 'pyd_module'):
        __import__(import_559834)
        sys_modules_559835 = sys.modules[import_559834]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'pytest', sys_modules_559835.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'pytest', import_559834)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from pytest import assert_raises' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_559836 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest')

if (type(import_559836) is not StypyTypeError):

    if (import_559836 != 'pyd_module'):
        __import__(import_559836)
        sys_modules_559837 = sys.modules[import_559836]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', sys_modules_559837.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_559837, sys_modules_559837.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', import_559836)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import scipy.special' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_559838 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.special')

if (type(import_559838) is not StypyTypeError):

    if (import_559838 != 'pyd_module'):
        __import__(import_559838)
        sys_modules_559839 = sys.modules[import_559838]
        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'sc', sys_modules_559839.module_type_store, module_type_store)
    else:
        import scipy.special as sc

        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'sc', scipy.special, module_type_store)

else:
    # Assigning a type to the variable 'scipy.special' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.special', import_559838)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.special._ufuncs import _sf_error_test_function' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/special/tests/')
import_559840 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.special._ufuncs')

if (type(import_559840) is not StypyTypeError):

    if (import_559840 != 'pyd_module'):
        __import__(import_559840)
        sys_modules_559841 = sys.modules[import_559840]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.special._ufuncs', sys_modules_559841.module_type_store, module_type_store, ['_sf_error_test_function'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_559841, sys_modules_559841.module_type_store, module_type_store)
    else:
        from scipy.special._ufuncs import _sf_error_test_function

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.special._ufuncs', None, module_type_store, ['_sf_error_test_function'], [_sf_error_test_function])

else:
    # Assigning a type to the variable 'scipy.special._ufuncs' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.special._ufuncs', import_559840)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/special/tests/')


# Assigning a Dict to a Name (line 13):

# Obtaining an instance of the builtin type 'dict' (line 13)
dict_559842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 21), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 13)
# Adding element type (key, value) (line 13)
str_559843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 4), 'str', 'singular')
int_559844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 16), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 21), dict_559842, (str_559843, int_559844))
# Adding element type (key, value) (line 13)
str_559845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 4), 'str', 'underflow')
int_559846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 17), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 21), dict_559842, (str_559845, int_559846))
# Adding element type (key, value) (line 13)
str_559847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 4), 'str', 'overflow')
int_559848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 16), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 21), dict_559842, (str_559847, int_559848))
# Adding element type (key, value) (line 13)
str_559849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 4), 'str', 'slow')
int_559850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 12), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 21), dict_559842, (str_559849, int_559850))
# Adding element type (key, value) (line 13)
str_559851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 4), 'str', 'loss')
int_559852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 12), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 21), dict_559842, (str_559851, int_559852))
# Adding element type (key, value) (line 13)
str_559853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 4), 'str', 'no_result')
int_559854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 17), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 21), dict_559842, (str_559853, int_559854))
# Adding element type (key, value) (line 13)
str_559855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 4), 'str', 'domain')
int_559856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 14), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 21), dict_559842, (str_559855, int_559856))
# Adding element type (key, value) (line 13)
str_559857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 4), 'str', 'arg')
int_559858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 11), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 21), dict_559842, (str_559857, int_559858))
# Adding element type (key, value) (line 13)
str_559859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 4), 'str', 'other')
int_559860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 13), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 21), dict_559842, (str_559859, int_559860))

# Assigning a type to the variable '_sf_error_code_map' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), '_sf_error_code_map', dict_559842)

# Assigning a List to a Name (line 26):

# Obtaining an instance of the builtin type 'list' (line 26)
list_559861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 20), 'list')
# Adding type elements to the builtin type 'list' instance (line 26)
# Adding element type (line 26)
str_559862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 4), 'str', 'ignore')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 20), list_559861, str_559862)
# Adding element type (line 26)
str_559863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 4), 'str', 'warn')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 20), list_559861, str_559863)
# Adding element type (line 26)
str_559864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 4), 'str', 'raise')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 20), list_559861, str_559864)

# Assigning a type to the variable '_sf_error_actions' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), '_sf_error_actions', list_559861)

@norecursion
def _check_action(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_check_action'
    module_type_store = module_type_store.open_function_context('_check_action', 33, 0, False)
    
    # Passed parameters checking function
    _check_action.stypy_localization = localization
    _check_action.stypy_type_of_self = None
    _check_action.stypy_type_store = module_type_store
    _check_action.stypy_function_name = '_check_action'
    _check_action.stypy_param_names_list = ['fun', 'args', 'action']
    _check_action.stypy_varargs_param_name = None
    _check_action.stypy_kwargs_param_name = None
    _check_action.stypy_call_defaults = defaults
    _check_action.stypy_call_varargs = varargs
    _check_action.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_check_action', ['fun', 'args', 'action'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_check_action', localization, ['fun', 'args', 'action'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_check_action(...)' code ##################

    
    
    # Getting the type of 'action' (line 34)
    action_559865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 7), 'action')
    str_559866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 17), 'str', 'warn')
    # Applying the binary operator '==' (line 34)
    result_eq_559867 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 7), '==', action_559865, str_559866)
    
    # Testing the type of an if condition (line 34)
    if_condition_559868 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 34, 4), result_eq_559867)
    # Assigning a type to the variable 'if_condition_559868' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'if_condition_559868', if_condition_559868)
    # SSA begins for if statement (line 34)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warns(...): (line 35)
    # Processing the call arguments (line 35)
    # Getting the type of 'sc' (line 35)
    sc_559871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 26), 'sc', False)
    # Obtaining the member 'SpecialFunctionWarning' of a type (line 35)
    SpecialFunctionWarning_559872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 26), sc_559871, 'SpecialFunctionWarning')
    # Processing the call keyword arguments (line 35)
    kwargs_559873 = {}
    # Getting the type of 'pytest' (line 35)
    pytest_559869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 13), 'pytest', False)
    # Obtaining the member 'warns' of a type (line 35)
    warns_559870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 13), pytest_559869, 'warns')
    # Calling warns(args, kwargs) (line 35)
    warns_call_result_559874 = invoke(stypy.reporting.localization.Localization(__file__, 35, 13), warns_559870, *[SpecialFunctionWarning_559872], **kwargs_559873)
    
    with_559875 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 35, 13), warns_call_result_559874, 'with parameter', '__enter__', '__exit__')

    if with_559875:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 35)
        enter___559876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 13), warns_call_result_559874, '__enter__')
        with_enter_559877 = invoke(stypy.reporting.localization.Localization(__file__, 35, 13), enter___559876)
        
        # Call to fun(...): (line 36)
        # Getting the type of 'args' (line 36)
        args_559879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 17), 'args', False)
        # Processing the call keyword arguments (line 36)
        kwargs_559880 = {}
        # Getting the type of 'fun' (line 36)
        fun_559878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'fun', False)
        # Calling fun(args, kwargs) (line 36)
        fun_call_result_559881 = invoke(stypy.reporting.localization.Localization(__file__, 36, 12), fun_559878, *[args_559879], **kwargs_559880)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 35)
        exit___559882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 13), warns_call_result_559874, '__exit__')
        with_exit_559883 = invoke(stypy.reporting.localization.Localization(__file__, 35, 13), exit___559882, None, None, None)

    # SSA branch for the else part of an if statement (line 34)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'action' (line 37)
    action_559884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 9), 'action')
    str_559885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 19), 'str', 'raise')
    # Applying the binary operator '==' (line 37)
    result_eq_559886 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 9), '==', action_559884, str_559885)
    
    # Testing the type of an if condition (line 37)
    if_condition_559887 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 37, 9), result_eq_559886)
    # Assigning a type to the variable 'if_condition_559887' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 9), 'if_condition_559887', if_condition_559887)
    # SSA begins for if statement (line 37)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to assert_raises(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'sc' (line 38)
    sc_559889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 27), 'sc', False)
    # Obtaining the member 'SpecialFunctionError' of a type (line 38)
    SpecialFunctionError_559890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 27), sc_559889, 'SpecialFunctionError')
    # Processing the call keyword arguments (line 38)
    kwargs_559891 = {}
    # Getting the type of 'assert_raises' (line 38)
    assert_raises_559888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 13), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 38)
    assert_raises_call_result_559892 = invoke(stypy.reporting.localization.Localization(__file__, 38, 13), assert_raises_559888, *[SpecialFunctionError_559890], **kwargs_559891)
    
    with_559893 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 38, 13), assert_raises_call_result_559892, 'with parameter', '__enter__', '__exit__')

    if with_559893:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 38)
        enter___559894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 13), assert_raises_call_result_559892, '__enter__')
        with_enter_559895 = invoke(stypy.reporting.localization.Localization(__file__, 38, 13), enter___559894)
        
        # Call to fun(...): (line 39)
        # Getting the type of 'args' (line 39)
        args_559897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 17), 'args', False)
        # Processing the call keyword arguments (line 39)
        kwargs_559898 = {}
        # Getting the type of 'fun' (line 39)
        fun_559896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'fun', False)
        # Calling fun(args, kwargs) (line 39)
        fun_call_result_559899 = invoke(stypy.reporting.localization.Localization(__file__, 39, 12), fun_559896, *[args_559897], **kwargs_559898)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 38)
        exit___559900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 13), assert_raises_call_result_559892, '__exit__')
        with_exit_559901 = invoke(stypy.reporting.localization.Localization(__file__, 38, 13), exit___559900, None, None, None)

    # SSA branch for the else part of an if statement (line 37)
    module_type_store.open_ssa_branch('else')
    
    # Call to catch_warnings(...): (line 42)
    # Processing the call keyword arguments (line 42)
    kwargs_559904 = {}
    # Getting the type of 'warnings' (line 42)
    warnings_559902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 13), 'warnings', False)
    # Obtaining the member 'catch_warnings' of a type (line 42)
    catch_warnings_559903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 13), warnings_559902, 'catch_warnings')
    # Calling catch_warnings(args, kwargs) (line 42)
    catch_warnings_call_result_559905 = invoke(stypy.reporting.localization.Localization(__file__, 42, 13), catch_warnings_559903, *[], **kwargs_559904)
    
    with_559906 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 42, 13), catch_warnings_call_result_559905, 'with parameter', '__enter__', '__exit__')

    if with_559906:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 42)
        enter___559907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 13), catch_warnings_call_result_559905, '__enter__')
        with_enter_559908 = invoke(stypy.reporting.localization.Localization(__file__, 42, 13), enter___559907)
        
        # Call to simplefilter(...): (line 43)
        # Processing the call arguments (line 43)
        str_559911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 34), 'str', 'error')
        # Processing the call keyword arguments (line 43)
        kwargs_559912 = {}
        # Getting the type of 'warnings' (line 43)
        warnings_559909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'warnings', False)
        # Obtaining the member 'simplefilter' of a type (line 43)
        simplefilter_559910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 12), warnings_559909, 'simplefilter')
        # Calling simplefilter(args, kwargs) (line 43)
        simplefilter_call_result_559913 = invoke(stypy.reporting.localization.Localization(__file__, 43, 12), simplefilter_559910, *[str_559911], **kwargs_559912)
        
        
        # Call to fun(...): (line 44)
        # Getting the type of 'args' (line 44)
        args_559915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 17), 'args', False)
        # Processing the call keyword arguments (line 44)
        kwargs_559916 = {}
        # Getting the type of 'fun' (line 44)
        fun_559914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'fun', False)
        # Calling fun(args, kwargs) (line 44)
        fun_call_result_559917 = invoke(stypy.reporting.localization.Localization(__file__, 44, 12), fun_559914, *[args_559915], **kwargs_559916)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 42)
        exit___559918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 13), catch_warnings_call_result_559905, '__exit__')
        with_exit_559919 = invoke(stypy.reporting.localization.Localization(__file__, 42, 13), exit___559918, None, None, None)

    # SSA join for if statement (line 37)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 34)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_check_action(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_check_action' in the type store
    # Getting the type of 'stypy_return_type' (line 33)
    stypy_return_type_559920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_559920)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_check_action'
    return stypy_return_type_559920

# Assigning a type to the variable '_check_action' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), '_check_action', _check_action)

@norecursion
def test_geterr(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_geterr'
    module_type_store = module_type_store.open_function_context('test_geterr', 47, 0, False)
    
    # Passed parameters checking function
    test_geterr.stypy_localization = localization
    test_geterr.stypy_type_of_self = None
    test_geterr.stypy_type_store = module_type_store
    test_geterr.stypy_function_name = 'test_geterr'
    test_geterr.stypy_param_names_list = []
    test_geterr.stypy_varargs_param_name = None
    test_geterr.stypy_kwargs_param_name = None
    test_geterr.stypy_call_defaults = defaults
    test_geterr.stypy_call_varargs = varargs
    test_geterr.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_geterr', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_geterr', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_geterr(...)' code ##################

    
    # Assigning a Call to a Name (line 48):
    
    # Call to geterr(...): (line 48)
    # Processing the call keyword arguments (line 48)
    kwargs_559923 = {}
    # Getting the type of 'sc' (line 48)
    sc_559921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 10), 'sc', False)
    # Obtaining the member 'geterr' of a type (line 48)
    geterr_559922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 10), sc_559921, 'geterr')
    # Calling geterr(args, kwargs) (line 48)
    geterr_call_result_559924 = invoke(stypy.reporting.localization.Localization(__file__, 48, 10), geterr_559922, *[], **kwargs_559923)
    
    # Assigning a type to the variable 'err' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'err', geterr_call_result_559924)
    
    
    # Call to items(...): (line 49)
    # Processing the call keyword arguments (line 49)
    kwargs_559927 = {}
    # Getting the type of 'err' (line 49)
    err_559925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 22), 'err', False)
    # Obtaining the member 'items' of a type (line 49)
    items_559926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 22), err_559925, 'items')
    # Calling items(args, kwargs) (line 49)
    items_call_result_559928 = invoke(stypy.reporting.localization.Localization(__file__, 49, 22), items_559926, *[], **kwargs_559927)
    
    # Testing the type of a for loop iterable (line 49)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 49, 4), items_call_result_559928)
    # Getting the type of the for loop variable (line 49)
    for_loop_var_559929 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 49, 4), items_call_result_559928)
    # Assigning a type to the variable 'key' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'key', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 4), for_loop_var_559929))
    # Assigning a type to the variable 'value' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 4), for_loop_var_559929))
    # SSA begins for a for statement (line 49)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to assert_(...): (line 50)
    # Processing the call arguments (line 50)
    
    # Getting the type of 'key' (line 50)
    key_559931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 16), 'key', False)
    
    # Call to keys(...): (line 50)
    # Processing the call keyword arguments (line 50)
    kwargs_559934 = {}
    # Getting the type of '_sf_error_code_map' (line 50)
    _sf_error_code_map_559932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 23), '_sf_error_code_map', False)
    # Obtaining the member 'keys' of a type (line 50)
    keys_559933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 23), _sf_error_code_map_559932, 'keys')
    # Calling keys(args, kwargs) (line 50)
    keys_call_result_559935 = invoke(stypy.reporting.localization.Localization(__file__, 50, 23), keys_559933, *[], **kwargs_559934)
    
    # Applying the binary operator 'in' (line 50)
    result_contains_559936 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 16), 'in', key_559931, keys_call_result_559935)
    
    # Processing the call keyword arguments (line 50)
    kwargs_559937 = {}
    # Getting the type of 'assert_' (line 50)
    assert__559930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 50)
    assert__call_result_559938 = invoke(stypy.reporting.localization.Localization(__file__, 50, 8), assert__559930, *[result_contains_559936], **kwargs_559937)
    
    
    # Call to assert_(...): (line 51)
    # Processing the call arguments (line 51)
    
    # Getting the type of 'value' (line 51)
    value_559940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'value', False)
    # Getting the type of '_sf_error_actions' (line 51)
    _sf_error_actions_559941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 25), '_sf_error_actions', False)
    # Applying the binary operator 'in' (line 51)
    result_contains_559942 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 16), 'in', value_559940, _sf_error_actions_559941)
    
    # Processing the call keyword arguments (line 51)
    kwargs_559943 = {}
    # Getting the type of 'assert_' (line 51)
    assert__559939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 51)
    assert__call_result_559944 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), assert__559939, *[result_contains_559942], **kwargs_559943)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_geterr(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_geterr' in the type store
    # Getting the type of 'stypy_return_type' (line 47)
    stypy_return_type_559945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_559945)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_geterr'
    return stypy_return_type_559945

# Assigning a type to the variable 'test_geterr' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'test_geterr', test_geterr)

@norecursion
def test_seterr(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_seterr'
    module_type_store = module_type_store.open_function_context('test_seterr', 54, 0, False)
    
    # Passed parameters checking function
    test_seterr.stypy_localization = localization
    test_seterr.stypy_type_of_self = None
    test_seterr.stypy_type_store = module_type_store
    test_seterr.stypy_function_name = 'test_seterr'
    test_seterr.stypy_param_names_list = []
    test_seterr.stypy_varargs_param_name = None
    test_seterr.stypy_kwargs_param_name = None
    test_seterr.stypy_call_defaults = defaults
    test_seterr.stypy_call_varargs = varargs
    test_seterr.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_seterr', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_seterr', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_seterr(...)' code ##################

    
    # Assigning a Call to a Name (line 55):
    
    # Call to geterr(...): (line 55)
    # Processing the call keyword arguments (line 55)
    kwargs_559948 = {}
    # Getting the type of 'sc' (line 55)
    sc_559946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), 'sc', False)
    # Obtaining the member 'geterr' of a type (line 55)
    geterr_559947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 16), sc_559946, 'geterr')
    # Calling geterr(args, kwargs) (line 55)
    geterr_call_result_559949 = invoke(stypy.reporting.localization.Localization(__file__, 55, 16), geterr_559947, *[], **kwargs_559948)
    
    # Assigning a type to the variable 'entry_err' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'entry_err', geterr_call_result_559949)
    
    # Try-finally block (line 56)
    
    
    # Call to keys(...): (line 57)
    # Processing the call keyword arguments (line 57)
    kwargs_559952 = {}
    # Getting the type of '_sf_error_code_map' (line 57)
    _sf_error_code_map_559950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 24), '_sf_error_code_map', False)
    # Obtaining the member 'keys' of a type (line 57)
    keys_559951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 24), _sf_error_code_map_559950, 'keys')
    # Calling keys(args, kwargs) (line 57)
    keys_call_result_559953 = invoke(stypy.reporting.localization.Localization(__file__, 57, 24), keys_559951, *[], **kwargs_559952)
    
    # Testing the type of a for loop iterable (line 57)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 57, 8), keys_call_result_559953)
    # Getting the type of the for loop variable (line 57)
    for_loop_var_559954 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 57, 8), keys_call_result_559953)
    # Assigning a type to the variable 'category' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'category', for_loop_var_559954)
    # SSA begins for a for statement (line 57)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of '_sf_error_actions' (line 58)
    _sf_error_actions_559955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 26), '_sf_error_actions')
    # Testing the type of a for loop iterable (line 58)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 58, 12), _sf_error_actions_559955)
    # Getting the type of the for loop variable (line 58)
    for_loop_var_559956 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 58, 12), _sf_error_actions_559955)
    # Assigning a type to the variable 'action' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'action', for_loop_var_559956)
    # SSA begins for a for statement (line 58)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 59):
    
    # Call to geterr(...): (line 59)
    # Processing the call keyword arguments (line 59)
    kwargs_559959 = {}
    # Getting the type of 'sc' (line 59)
    sc_559957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 32), 'sc', False)
    # Obtaining the member 'geterr' of a type (line 59)
    geterr_559958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 32), sc_559957, 'geterr')
    # Calling geterr(args, kwargs) (line 59)
    geterr_call_result_559960 = invoke(stypy.reporting.localization.Localization(__file__, 59, 32), geterr_559958, *[], **kwargs_559959)
    
    # Assigning a type to the variable 'geterr_olderr' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'geterr_olderr', geterr_call_result_559960)
    
    # Assigning a Call to a Name (line 60):
    
    # Call to seterr(...): (line 60)
    # Processing the call keyword arguments (line 60)
    
    # Obtaining an instance of the builtin type 'dict' (line 60)
    dict_559963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 44), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 60)
    # Adding element type (key, value) (line 60)
    # Getting the type of 'category' (line 60)
    category_559964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 45), 'category', False)
    # Getting the type of 'action' (line 60)
    action_559965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 55), 'action', False)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 44), dict_559963, (category_559964, action_559965))
    
    kwargs_559966 = {'dict_559963': dict_559963}
    # Getting the type of 'sc' (line 60)
    sc_559961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 32), 'sc', False)
    # Obtaining the member 'seterr' of a type (line 60)
    seterr_559962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 32), sc_559961, 'seterr')
    # Calling seterr(args, kwargs) (line 60)
    seterr_call_result_559967 = invoke(stypy.reporting.localization.Localization(__file__, 60, 32), seterr_559962, *[], **kwargs_559966)
    
    # Assigning a type to the variable 'seterr_olderr' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'seterr_olderr', seterr_call_result_559967)
    
    # Call to assert_(...): (line 61)
    # Processing the call arguments (line 61)
    
    # Getting the type of 'geterr_olderr' (line 61)
    geterr_olderr_559969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 24), 'geterr_olderr', False)
    # Getting the type of 'seterr_olderr' (line 61)
    seterr_olderr_559970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 41), 'seterr_olderr', False)
    # Applying the binary operator '==' (line 61)
    result_eq_559971 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 24), '==', geterr_olderr_559969, seterr_olderr_559970)
    
    # Processing the call keyword arguments (line 61)
    kwargs_559972 = {}
    # Getting the type of 'assert_' (line 61)
    assert__559968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'assert_', False)
    # Calling assert_(args, kwargs) (line 61)
    assert__call_result_559973 = invoke(stypy.reporting.localization.Localization(__file__, 61, 16), assert__559968, *[result_eq_559971], **kwargs_559972)
    
    
    # Assigning a Call to a Name (line 62):
    
    # Call to geterr(...): (line 62)
    # Processing the call keyword arguments (line 62)
    kwargs_559976 = {}
    # Getting the type of 'sc' (line 62)
    sc_559974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 25), 'sc', False)
    # Obtaining the member 'geterr' of a type (line 62)
    geterr_559975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 25), sc_559974, 'geterr')
    # Calling geterr(args, kwargs) (line 62)
    geterr_call_result_559977 = invoke(stypy.reporting.localization.Localization(__file__, 62, 25), geterr_559975, *[], **kwargs_559976)
    
    # Assigning a type to the variable 'newerr' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 16), 'newerr', geterr_call_result_559977)
    
    # Call to assert_(...): (line 63)
    # Processing the call arguments (line 63)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'category' (line 63)
    category_559979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 31), 'category', False)
    # Getting the type of 'newerr' (line 63)
    newerr_559980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 24), 'newerr', False)
    # Obtaining the member '__getitem__' of a type (line 63)
    getitem___559981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 24), newerr_559980, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 63)
    subscript_call_result_559982 = invoke(stypy.reporting.localization.Localization(__file__, 63, 24), getitem___559981, category_559979)
    
    # Getting the type of 'action' (line 63)
    action_559983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 44), 'action', False)
    # Applying the binary operator '==' (line 63)
    result_eq_559984 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 24), '==', subscript_call_result_559982, action_559983)
    
    # Processing the call keyword arguments (line 63)
    kwargs_559985 = {}
    # Getting the type of 'assert_' (line 63)
    assert__559978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 16), 'assert_', False)
    # Calling assert_(args, kwargs) (line 63)
    assert__call_result_559986 = invoke(stypy.reporting.localization.Localization(__file__, 63, 16), assert__559978, *[result_eq_559984], **kwargs_559985)
    
    
    # Call to pop(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'category' (line 64)
    category_559989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 34), 'category', False)
    # Processing the call keyword arguments (line 64)
    kwargs_559990 = {}
    # Getting the type of 'geterr_olderr' (line 64)
    geterr_olderr_559987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 16), 'geterr_olderr', False)
    # Obtaining the member 'pop' of a type (line 64)
    pop_559988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 16), geterr_olderr_559987, 'pop')
    # Calling pop(args, kwargs) (line 64)
    pop_call_result_559991 = invoke(stypy.reporting.localization.Localization(__file__, 64, 16), pop_559988, *[category_559989], **kwargs_559990)
    
    
    # Call to pop(...): (line 65)
    # Processing the call arguments (line 65)
    # Getting the type of 'category' (line 65)
    category_559994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 27), 'category', False)
    # Processing the call keyword arguments (line 65)
    kwargs_559995 = {}
    # Getting the type of 'newerr' (line 65)
    newerr_559992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 16), 'newerr', False)
    # Obtaining the member 'pop' of a type (line 65)
    pop_559993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 16), newerr_559992, 'pop')
    # Calling pop(args, kwargs) (line 65)
    pop_call_result_559996 = invoke(stypy.reporting.localization.Localization(__file__, 65, 16), pop_559993, *[category_559994], **kwargs_559995)
    
    
    # Call to assert_(...): (line 66)
    # Processing the call arguments (line 66)
    
    # Getting the type of 'geterr_olderr' (line 66)
    geterr_olderr_559998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 24), 'geterr_olderr', False)
    # Getting the type of 'newerr' (line 66)
    newerr_559999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 41), 'newerr', False)
    # Applying the binary operator '==' (line 66)
    result_eq_560000 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 24), '==', geterr_olderr_559998, newerr_559999)
    
    # Processing the call keyword arguments (line 66)
    kwargs_560001 = {}
    # Getting the type of 'assert_' (line 66)
    assert__559997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 16), 'assert_', False)
    # Calling assert_(args, kwargs) (line 66)
    assert__call_result_560002 = invoke(stypy.reporting.localization.Localization(__file__, 66, 16), assert__559997, *[result_eq_560000], **kwargs_560001)
    
    
    # Call to _check_action(...): (line 67)
    # Processing the call arguments (line 67)
    # Getting the type of '_sf_error_test_function' (line 67)
    _sf_error_test_function_560004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 30), '_sf_error_test_function', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 68)
    tuple_560005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 68)
    # Adding element type (line 68)
    
    # Obtaining the type of the subscript
    # Getting the type of 'category' (line 68)
    category_560006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 50), 'category', False)
    # Getting the type of '_sf_error_code_map' (line 68)
    _sf_error_code_map_560007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 31), '_sf_error_code_map', False)
    # Obtaining the member '__getitem__' of a type (line 68)
    getitem___560008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 31), _sf_error_code_map_560007, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 68)
    subscript_call_result_560009 = invoke(stypy.reporting.localization.Localization(__file__, 68, 31), getitem___560008, category_560006)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 31), tuple_560005, subscript_call_result_560009)
    
    # Getting the type of 'action' (line 69)
    action_560010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 31), 'action', False)
    # Processing the call keyword arguments (line 67)
    kwargs_560011 = {}
    # Getting the type of '_check_action' (line 67)
    _check_action_560003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 16), '_check_action', False)
    # Calling _check_action(args, kwargs) (line 67)
    _check_action_call_result_560012 = invoke(stypy.reporting.localization.Localization(__file__, 67, 16), _check_action_560003, *[_sf_error_test_function_560004, tuple_560005, action_560010], **kwargs_560011)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # finally branch of the try-finally block (line 56)
    
    # Call to seterr(...): (line 71)
    # Processing the call keyword arguments (line 71)
    # Getting the type of 'entry_err' (line 71)
    entry_err_560015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 20), 'entry_err', False)
    kwargs_560016 = {'entry_err_560015': entry_err_560015}
    # Getting the type of 'sc' (line 71)
    sc_560013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'sc', False)
    # Obtaining the member 'seterr' of a type (line 71)
    seterr_560014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), sc_560013, 'seterr')
    # Calling seterr(args, kwargs) (line 71)
    seterr_call_result_560017 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), seterr_560014, *[], **kwargs_560016)
    
    
    
    # ################# End of 'test_seterr(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_seterr' in the type store
    # Getting the type of 'stypy_return_type' (line 54)
    stypy_return_type_560018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_560018)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_seterr'
    return stypy_return_type_560018

# Assigning a type to the variable 'test_seterr' (line 54)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'test_seterr', test_seterr)

@norecursion
def test_errstate_pyx_basic(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_errstate_pyx_basic'
    module_type_store = module_type_store.open_function_context('test_errstate_pyx_basic', 74, 0, False)
    
    # Passed parameters checking function
    test_errstate_pyx_basic.stypy_localization = localization
    test_errstate_pyx_basic.stypy_type_of_self = None
    test_errstate_pyx_basic.stypy_type_store = module_type_store
    test_errstate_pyx_basic.stypy_function_name = 'test_errstate_pyx_basic'
    test_errstate_pyx_basic.stypy_param_names_list = []
    test_errstate_pyx_basic.stypy_varargs_param_name = None
    test_errstate_pyx_basic.stypy_kwargs_param_name = None
    test_errstate_pyx_basic.stypy_call_defaults = defaults
    test_errstate_pyx_basic.stypy_call_varargs = varargs
    test_errstate_pyx_basic.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_errstate_pyx_basic', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_errstate_pyx_basic', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_errstate_pyx_basic(...)' code ##################

    
    # Assigning a Call to a Name (line 75):
    
    # Call to geterr(...): (line 75)
    # Processing the call keyword arguments (line 75)
    kwargs_560021 = {}
    # Getting the type of 'sc' (line 75)
    sc_560019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 13), 'sc', False)
    # Obtaining the member 'geterr' of a type (line 75)
    geterr_560020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 13), sc_560019, 'geterr')
    # Calling geterr(args, kwargs) (line 75)
    geterr_call_result_560022 = invoke(stypy.reporting.localization.Localization(__file__, 75, 13), geterr_560020, *[], **kwargs_560021)
    
    # Assigning a type to the variable 'olderr' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'olderr', geterr_call_result_560022)
    
    # Call to errstate(...): (line 76)
    # Processing the call keyword arguments (line 76)
    str_560025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 30), 'str', 'raise')
    keyword_560026 = str_560025
    kwargs_560027 = {'singular': keyword_560026}
    # Getting the type of 'sc' (line 76)
    sc_560023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 9), 'sc', False)
    # Obtaining the member 'errstate' of a type (line 76)
    errstate_560024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 9), sc_560023, 'errstate')
    # Calling errstate(args, kwargs) (line 76)
    errstate_call_result_560028 = invoke(stypy.reporting.localization.Localization(__file__, 76, 9), errstate_560024, *[], **kwargs_560027)
    
    with_560029 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 76, 9), errstate_call_result_560028, 'with parameter', '__enter__', '__exit__')

    if with_560029:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 76)
        enter___560030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 9), errstate_call_result_560028, '__enter__')
        with_enter_560031 = invoke(stypy.reporting.localization.Localization(__file__, 76, 9), enter___560030)
        
        # Call to assert_raises(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'sc' (line 77)
        sc_560033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 27), 'sc', False)
        # Obtaining the member 'SpecialFunctionError' of a type (line 77)
        SpecialFunctionError_560034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 27), sc_560033, 'SpecialFunctionError')
        # Processing the call keyword arguments (line 77)
        kwargs_560035 = {}
        # Getting the type of 'assert_raises' (line 77)
        assert_raises_560032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 13), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 77)
        assert_raises_call_result_560036 = invoke(stypy.reporting.localization.Localization(__file__, 77, 13), assert_raises_560032, *[SpecialFunctionError_560034], **kwargs_560035)
        
        with_560037 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 77, 13), assert_raises_call_result_560036, 'with parameter', '__enter__', '__exit__')

        if with_560037:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 77)
            enter___560038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 13), assert_raises_call_result_560036, '__enter__')
            with_enter_560039 = invoke(stypy.reporting.localization.Localization(__file__, 77, 13), enter___560038)
            
            # Call to loggamma(...): (line 78)
            # Processing the call arguments (line 78)
            int_560042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 24), 'int')
            # Processing the call keyword arguments (line 78)
            kwargs_560043 = {}
            # Getting the type of 'sc' (line 78)
            sc_560040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'sc', False)
            # Obtaining the member 'loggamma' of a type (line 78)
            loggamma_560041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 12), sc_560040, 'loggamma')
            # Calling loggamma(args, kwargs) (line 78)
            loggamma_call_result_560044 = invoke(stypy.reporting.localization.Localization(__file__, 78, 12), loggamma_560041, *[int_560042], **kwargs_560043)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 77)
            exit___560045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 13), assert_raises_call_result_560036, '__exit__')
            with_exit_560046 = invoke(stypy.reporting.localization.Localization(__file__, 77, 13), exit___560045, None, None, None)

        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 76)
        exit___560047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 9), errstate_call_result_560028, '__exit__')
        with_exit_560048 = invoke(stypy.reporting.localization.Localization(__file__, 76, 9), exit___560047, None, None, None)

    
    # Call to assert_equal(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'olderr' (line 79)
    olderr_560050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 17), 'olderr', False)
    
    # Call to geterr(...): (line 79)
    # Processing the call keyword arguments (line 79)
    kwargs_560053 = {}
    # Getting the type of 'sc' (line 79)
    sc_560051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 25), 'sc', False)
    # Obtaining the member 'geterr' of a type (line 79)
    geterr_560052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 25), sc_560051, 'geterr')
    # Calling geterr(args, kwargs) (line 79)
    geterr_call_result_560054 = invoke(stypy.reporting.localization.Localization(__file__, 79, 25), geterr_560052, *[], **kwargs_560053)
    
    # Processing the call keyword arguments (line 79)
    kwargs_560055 = {}
    # Getting the type of 'assert_equal' (line 79)
    assert_equal_560049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 79)
    assert_equal_call_result_560056 = invoke(stypy.reporting.localization.Localization(__file__, 79, 4), assert_equal_560049, *[olderr_560050, geterr_call_result_560054], **kwargs_560055)
    
    
    # ################# End of 'test_errstate_pyx_basic(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_errstate_pyx_basic' in the type store
    # Getting the type of 'stypy_return_type' (line 74)
    stypy_return_type_560057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_560057)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_errstate_pyx_basic'
    return stypy_return_type_560057

# Assigning a type to the variable 'test_errstate_pyx_basic' (line 74)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'test_errstate_pyx_basic', test_errstate_pyx_basic)

@norecursion
def test_errstate_c_basic(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_errstate_c_basic'
    module_type_store = module_type_store.open_function_context('test_errstate_c_basic', 82, 0, False)
    
    # Passed parameters checking function
    test_errstate_c_basic.stypy_localization = localization
    test_errstate_c_basic.stypy_type_of_self = None
    test_errstate_c_basic.stypy_type_store = module_type_store
    test_errstate_c_basic.stypy_function_name = 'test_errstate_c_basic'
    test_errstate_c_basic.stypy_param_names_list = []
    test_errstate_c_basic.stypy_varargs_param_name = None
    test_errstate_c_basic.stypy_kwargs_param_name = None
    test_errstate_c_basic.stypy_call_defaults = defaults
    test_errstate_c_basic.stypy_call_varargs = varargs
    test_errstate_c_basic.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_errstate_c_basic', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_errstate_c_basic', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_errstate_c_basic(...)' code ##################

    
    # Assigning a Call to a Name (line 83):
    
    # Call to geterr(...): (line 83)
    # Processing the call keyword arguments (line 83)
    kwargs_560060 = {}
    # Getting the type of 'sc' (line 83)
    sc_560058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 13), 'sc', False)
    # Obtaining the member 'geterr' of a type (line 83)
    geterr_560059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 13), sc_560058, 'geterr')
    # Calling geterr(args, kwargs) (line 83)
    geterr_call_result_560061 = invoke(stypy.reporting.localization.Localization(__file__, 83, 13), geterr_560059, *[], **kwargs_560060)
    
    # Assigning a type to the variable 'olderr' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'olderr', geterr_call_result_560061)
    
    # Call to errstate(...): (line 84)
    # Processing the call keyword arguments (line 84)
    str_560064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 28), 'str', 'raise')
    keyword_560065 = str_560064
    kwargs_560066 = {'domain': keyword_560065}
    # Getting the type of 'sc' (line 84)
    sc_560062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 9), 'sc', False)
    # Obtaining the member 'errstate' of a type (line 84)
    errstate_560063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 9), sc_560062, 'errstate')
    # Calling errstate(args, kwargs) (line 84)
    errstate_call_result_560067 = invoke(stypy.reporting.localization.Localization(__file__, 84, 9), errstate_560063, *[], **kwargs_560066)
    
    with_560068 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 84, 9), errstate_call_result_560067, 'with parameter', '__enter__', '__exit__')

    if with_560068:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 84)
        enter___560069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 9), errstate_call_result_560067, '__enter__')
        with_enter_560070 = invoke(stypy.reporting.localization.Localization(__file__, 84, 9), enter___560069)
        
        # Call to assert_raises(...): (line 85)
        # Processing the call arguments (line 85)
        # Getting the type of 'sc' (line 85)
        sc_560072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 27), 'sc', False)
        # Obtaining the member 'SpecialFunctionError' of a type (line 85)
        SpecialFunctionError_560073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 27), sc_560072, 'SpecialFunctionError')
        # Processing the call keyword arguments (line 85)
        kwargs_560074 = {}
        # Getting the type of 'assert_raises' (line 85)
        assert_raises_560071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 13), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 85)
        assert_raises_call_result_560075 = invoke(stypy.reporting.localization.Localization(__file__, 85, 13), assert_raises_560071, *[SpecialFunctionError_560073], **kwargs_560074)
        
        with_560076 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 85, 13), assert_raises_call_result_560075, 'with parameter', '__enter__', '__exit__')

        if with_560076:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 85)
            enter___560077 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 13), assert_raises_call_result_560075, '__enter__')
            with_enter_560078 = invoke(stypy.reporting.localization.Localization(__file__, 85, 13), enter___560077)
            
            # Call to spence(...): (line 86)
            # Processing the call arguments (line 86)
            int_560081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 22), 'int')
            # Processing the call keyword arguments (line 86)
            kwargs_560082 = {}
            # Getting the type of 'sc' (line 86)
            sc_560079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'sc', False)
            # Obtaining the member 'spence' of a type (line 86)
            spence_560080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 12), sc_560079, 'spence')
            # Calling spence(args, kwargs) (line 86)
            spence_call_result_560083 = invoke(stypy.reporting.localization.Localization(__file__, 86, 12), spence_560080, *[int_560081], **kwargs_560082)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 85)
            exit___560084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 13), assert_raises_call_result_560075, '__exit__')
            with_exit_560085 = invoke(stypy.reporting.localization.Localization(__file__, 85, 13), exit___560084, None, None, None)

        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 84)
        exit___560086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 9), errstate_call_result_560067, '__exit__')
        with_exit_560087 = invoke(stypy.reporting.localization.Localization(__file__, 84, 9), exit___560086, None, None, None)

    
    # Call to assert_equal(...): (line 87)
    # Processing the call arguments (line 87)
    # Getting the type of 'olderr' (line 87)
    olderr_560089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 17), 'olderr', False)
    
    # Call to geterr(...): (line 87)
    # Processing the call keyword arguments (line 87)
    kwargs_560092 = {}
    # Getting the type of 'sc' (line 87)
    sc_560090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 25), 'sc', False)
    # Obtaining the member 'geterr' of a type (line 87)
    geterr_560091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 25), sc_560090, 'geterr')
    # Calling geterr(args, kwargs) (line 87)
    geterr_call_result_560093 = invoke(stypy.reporting.localization.Localization(__file__, 87, 25), geterr_560091, *[], **kwargs_560092)
    
    # Processing the call keyword arguments (line 87)
    kwargs_560094 = {}
    # Getting the type of 'assert_equal' (line 87)
    assert_equal_560088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 87)
    assert_equal_call_result_560095 = invoke(stypy.reporting.localization.Localization(__file__, 87, 4), assert_equal_560088, *[olderr_560089, geterr_call_result_560093], **kwargs_560094)
    
    
    # ################# End of 'test_errstate_c_basic(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_errstate_c_basic' in the type store
    # Getting the type of 'stypy_return_type' (line 82)
    stypy_return_type_560096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_560096)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_errstate_c_basic'
    return stypy_return_type_560096

# Assigning a type to the variable 'test_errstate_c_basic' (line 82)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 0), 'test_errstate_c_basic', test_errstate_c_basic)

@norecursion
def test_errstate_cpp_basic(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_errstate_cpp_basic'
    module_type_store = module_type_store.open_function_context('test_errstate_cpp_basic', 90, 0, False)
    
    # Passed parameters checking function
    test_errstate_cpp_basic.stypy_localization = localization
    test_errstate_cpp_basic.stypy_type_of_self = None
    test_errstate_cpp_basic.stypy_type_store = module_type_store
    test_errstate_cpp_basic.stypy_function_name = 'test_errstate_cpp_basic'
    test_errstate_cpp_basic.stypy_param_names_list = []
    test_errstate_cpp_basic.stypy_varargs_param_name = None
    test_errstate_cpp_basic.stypy_kwargs_param_name = None
    test_errstate_cpp_basic.stypy_call_defaults = defaults
    test_errstate_cpp_basic.stypy_call_varargs = varargs
    test_errstate_cpp_basic.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_errstate_cpp_basic', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_errstate_cpp_basic', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_errstate_cpp_basic(...)' code ##################

    
    # Assigning a Call to a Name (line 91):
    
    # Call to geterr(...): (line 91)
    # Processing the call keyword arguments (line 91)
    kwargs_560099 = {}
    # Getting the type of 'sc' (line 91)
    sc_560097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 13), 'sc', False)
    # Obtaining the member 'geterr' of a type (line 91)
    geterr_560098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 13), sc_560097, 'geterr')
    # Calling geterr(args, kwargs) (line 91)
    geterr_call_result_560100 = invoke(stypy.reporting.localization.Localization(__file__, 91, 13), geterr_560098, *[], **kwargs_560099)
    
    # Assigning a type to the variable 'olderr' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'olderr', geterr_call_result_560100)
    
    # Call to errstate(...): (line 92)
    # Processing the call keyword arguments (line 92)
    str_560103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 31), 'str', 'raise')
    keyword_560104 = str_560103
    kwargs_560105 = {'underflow': keyword_560104}
    # Getting the type of 'sc' (line 92)
    sc_560101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 9), 'sc', False)
    # Obtaining the member 'errstate' of a type (line 92)
    errstate_560102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 9), sc_560101, 'errstate')
    # Calling errstate(args, kwargs) (line 92)
    errstate_call_result_560106 = invoke(stypy.reporting.localization.Localization(__file__, 92, 9), errstate_560102, *[], **kwargs_560105)
    
    with_560107 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 92, 9), errstate_call_result_560106, 'with parameter', '__enter__', '__exit__')

    if with_560107:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 92)
        enter___560108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 9), errstate_call_result_560106, '__enter__')
        with_enter_560109 = invoke(stypy.reporting.localization.Localization(__file__, 92, 9), enter___560108)
        
        # Call to assert_raises(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'sc' (line 93)
        sc_560111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 27), 'sc', False)
        # Obtaining the member 'SpecialFunctionError' of a type (line 93)
        SpecialFunctionError_560112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 27), sc_560111, 'SpecialFunctionError')
        # Processing the call keyword arguments (line 93)
        kwargs_560113 = {}
        # Getting the type of 'assert_raises' (line 93)
        assert_raises_560110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 13), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 93)
        assert_raises_call_result_560114 = invoke(stypy.reporting.localization.Localization(__file__, 93, 13), assert_raises_560110, *[SpecialFunctionError_560112], **kwargs_560113)
        
        with_560115 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 93, 13), assert_raises_call_result_560114, 'with parameter', '__enter__', '__exit__')

        if with_560115:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 93)
            enter___560116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 13), assert_raises_call_result_560114, '__enter__')
            with_enter_560117 = invoke(stypy.reporting.localization.Localization(__file__, 93, 13), enter___560116)
            
            # Call to wrightomega(...): (line 94)
            # Processing the call arguments (line 94)
            int_560120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 27), 'int')
            # Processing the call keyword arguments (line 94)
            kwargs_560121 = {}
            # Getting the type of 'sc' (line 94)
            sc_560118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'sc', False)
            # Obtaining the member 'wrightomega' of a type (line 94)
            wrightomega_560119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 12), sc_560118, 'wrightomega')
            # Calling wrightomega(args, kwargs) (line 94)
            wrightomega_call_result_560122 = invoke(stypy.reporting.localization.Localization(__file__, 94, 12), wrightomega_560119, *[int_560120], **kwargs_560121)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 93)
            exit___560123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 13), assert_raises_call_result_560114, '__exit__')
            with_exit_560124 = invoke(stypy.reporting.localization.Localization(__file__, 93, 13), exit___560123, None, None, None)

        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 92)
        exit___560125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 9), errstate_call_result_560106, '__exit__')
        with_exit_560126 = invoke(stypy.reporting.localization.Localization(__file__, 92, 9), exit___560125, None, None, None)

    
    # Call to assert_equal(...): (line 95)
    # Processing the call arguments (line 95)
    # Getting the type of 'olderr' (line 95)
    olderr_560128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 17), 'olderr', False)
    
    # Call to geterr(...): (line 95)
    # Processing the call keyword arguments (line 95)
    kwargs_560131 = {}
    # Getting the type of 'sc' (line 95)
    sc_560129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 25), 'sc', False)
    # Obtaining the member 'geterr' of a type (line 95)
    geterr_560130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 25), sc_560129, 'geterr')
    # Calling geterr(args, kwargs) (line 95)
    geterr_call_result_560132 = invoke(stypy.reporting.localization.Localization(__file__, 95, 25), geterr_560130, *[], **kwargs_560131)
    
    # Processing the call keyword arguments (line 95)
    kwargs_560133 = {}
    # Getting the type of 'assert_equal' (line 95)
    assert_equal_560127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 95)
    assert_equal_call_result_560134 = invoke(stypy.reporting.localization.Localization(__file__, 95, 4), assert_equal_560127, *[olderr_560128, geterr_call_result_560132], **kwargs_560133)
    
    
    # ################# End of 'test_errstate_cpp_basic(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_errstate_cpp_basic' in the type store
    # Getting the type of 'stypy_return_type' (line 90)
    stypy_return_type_560135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_560135)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_errstate_cpp_basic'
    return stypy_return_type_560135

# Assigning a type to the variable 'test_errstate_cpp_basic' (line 90)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 0), 'test_errstate_cpp_basic', test_errstate_cpp_basic)

@norecursion
def test_errstate(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_errstate'
    module_type_store = module_type_store.open_function_context('test_errstate', 98, 0, False)
    
    # Passed parameters checking function
    test_errstate.stypy_localization = localization
    test_errstate.stypy_type_of_self = None
    test_errstate.stypy_type_store = module_type_store
    test_errstate.stypy_function_name = 'test_errstate'
    test_errstate.stypy_param_names_list = []
    test_errstate.stypy_varargs_param_name = None
    test_errstate.stypy_kwargs_param_name = None
    test_errstate.stypy_call_defaults = defaults
    test_errstate.stypy_call_varargs = varargs
    test_errstate.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_errstate', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_errstate', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_errstate(...)' code ##################

    
    
    # Call to keys(...): (line 99)
    # Processing the call keyword arguments (line 99)
    kwargs_560138 = {}
    # Getting the type of '_sf_error_code_map' (line 99)
    _sf_error_code_map_560136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 20), '_sf_error_code_map', False)
    # Obtaining the member 'keys' of a type (line 99)
    keys_560137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 20), _sf_error_code_map_560136, 'keys')
    # Calling keys(args, kwargs) (line 99)
    keys_call_result_560139 = invoke(stypy.reporting.localization.Localization(__file__, 99, 20), keys_560137, *[], **kwargs_560138)
    
    # Testing the type of a for loop iterable (line 99)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 99, 4), keys_call_result_560139)
    # Getting the type of the for loop variable (line 99)
    for_loop_var_560140 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 99, 4), keys_call_result_560139)
    # Assigning a type to the variable 'category' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'category', for_loop_var_560140)
    # SSA begins for a for statement (line 99)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of '_sf_error_actions' (line 100)
    _sf_error_actions_560141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 22), '_sf_error_actions')
    # Testing the type of a for loop iterable (line 100)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 100, 8), _sf_error_actions_560141)
    # Getting the type of the for loop variable (line 100)
    for_loop_var_560142 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 100, 8), _sf_error_actions_560141)
    # Assigning a type to the variable 'action' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'action', for_loop_var_560142)
    # SSA begins for a for statement (line 100)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 101):
    
    # Call to geterr(...): (line 101)
    # Processing the call keyword arguments (line 101)
    kwargs_560145 = {}
    # Getting the type of 'sc' (line 101)
    sc_560143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 21), 'sc', False)
    # Obtaining the member 'geterr' of a type (line 101)
    geterr_560144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 21), sc_560143, 'geterr')
    # Calling geterr(args, kwargs) (line 101)
    geterr_call_result_560146 = invoke(stypy.reporting.localization.Localization(__file__, 101, 21), geterr_560144, *[], **kwargs_560145)
    
    # Assigning a type to the variable 'olderr' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'olderr', geterr_call_result_560146)
    
    # Call to errstate(...): (line 102)
    # Processing the call keyword arguments (line 102)
    
    # Obtaining an instance of the builtin type 'dict' (line 102)
    dict_560149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 31), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 102)
    # Adding element type (key, value) (line 102)
    # Getting the type of 'category' (line 102)
    category_560150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 32), 'category', False)
    # Getting the type of 'action' (line 102)
    action_560151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 42), 'action', False)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 31), dict_560149, (category_560150, action_560151))
    
    kwargs_560152 = {'dict_560149': dict_560149}
    # Getting the type of 'sc' (line 102)
    sc_560147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 17), 'sc', False)
    # Obtaining the member 'errstate' of a type (line 102)
    errstate_560148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 17), sc_560147, 'errstate')
    # Calling errstate(args, kwargs) (line 102)
    errstate_call_result_560153 = invoke(stypy.reporting.localization.Localization(__file__, 102, 17), errstate_560148, *[], **kwargs_560152)
    
    with_560154 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 102, 17), errstate_call_result_560153, 'with parameter', '__enter__', '__exit__')

    if with_560154:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 102)
        enter___560155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 17), errstate_call_result_560153, '__enter__')
        with_enter_560156 = invoke(stypy.reporting.localization.Localization(__file__, 102, 17), enter___560155)
        
        # Call to _check_action(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of '_sf_error_test_function' (line 103)
        _sf_error_test_function_560158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 30), '_sf_error_test_function', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 104)
        tuple_560159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 31), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 104)
        # Adding element type (line 104)
        
        # Obtaining the type of the subscript
        # Getting the type of 'category' (line 104)
        category_560160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 50), 'category', False)
        # Getting the type of '_sf_error_code_map' (line 104)
        _sf_error_code_map_560161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 31), '_sf_error_code_map', False)
        # Obtaining the member '__getitem__' of a type (line 104)
        getitem___560162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 31), _sf_error_code_map_560161, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 104)
        subscript_call_result_560163 = invoke(stypy.reporting.localization.Localization(__file__, 104, 31), getitem___560162, category_560160)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 31), tuple_560159, subscript_call_result_560163)
        
        # Getting the type of 'action' (line 105)
        action_560164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 30), 'action', False)
        # Processing the call keyword arguments (line 103)
        kwargs_560165 = {}
        # Getting the type of '_check_action' (line 103)
        _check_action_560157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 16), '_check_action', False)
        # Calling _check_action(args, kwargs) (line 103)
        _check_action_call_result_560166 = invoke(stypy.reporting.localization.Localization(__file__, 103, 16), _check_action_560157, *[_sf_error_test_function_560158, tuple_560159, action_560164], **kwargs_560165)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 102)
        exit___560167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 17), errstate_call_result_560153, '__exit__')
        with_exit_560168 = invoke(stypy.reporting.localization.Localization(__file__, 102, 17), exit___560167, None, None, None)

    
    # Call to assert_equal(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'olderr' (line 106)
    olderr_560170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 25), 'olderr', False)
    
    # Call to geterr(...): (line 106)
    # Processing the call keyword arguments (line 106)
    kwargs_560173 = {}
    # Getting the type of 'sc' (line 106)
    sc_560171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 33), 'sc', False)
    # Obtaining the member 'geterr' of a type (line 106)
    geterr_560172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 33), sc_560171, 'geterr')
    # Calling geterr(args, kwargs) (line 106)
    geterr_call_result_560174 = invoke(stypy.reporting.localization.Localization(__file__, 106, 33), geterr_560172, *[], **kwargs_560173)
    
    # Processing the call keyword arguments (line 106)
    kwargs_560175 = {}
    # Getting the type of 'assert_equal' (line 106)
    assert_equal_560169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 106)
    assert_equal_call_result_560176 = invoke(stypy.reporting.localization.Localization(__file__, 106, 12), assert_equal_560169, *[olderr_560170, geterr_call_result_560174], **kwargs_560175)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'test_errstate(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_errstate' in the type store
    # Getting the type of 'stypy_return_type' (line 98)
    stypy_return_type_560177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_560177)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_errstate'
    return stypy_return_type_560177

# Assigning a type to the variable 'test_errstate' (line 98)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), 'test_errstate', test_errstate)

@norecursion
def test_errstate_all_but_one(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_errstate_all_but_one'
    module_type_store = module_type_store.open_function_context('test_errstate_all_but_one', 109, 0, False)
    
    # Passed parameters checking function
    test_errstate_all_but_one.stypy_localization = localization
    test_errstate_all_but_one.stypy_type_of_self = None
    test_errstate_all_but_one.stypy_type_store = module_type_store
    test_errstate_all_but_one.stypy_function_name = 'test_errstate_all_but_one'
    test_errstate_all_but_one.stypy_param_names_list = []
    test_errstate_all_but_one.stypy_varargs_param_name = None
    test_errstate_all_but_one.stypy_kwargs_param_name = None
    test_errstate_all_but_one.stypy_call_defaults = defaults
    test_errstate_all_but_one.stypy_call_varargs = varargs
    test_errstate_all_but_one.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_errstate_all_but_one', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_errstate_all_but_one', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_errstate_all_but_one(...)' code ##################

    
    # Assigning a Call to a Name (line 110):
    
    # Call to geterr(...): (line 110)
    # Processing the call keyword arguments (line 110)
    kwargs_560180 = {}
    # Getting the type of 'sc' (line 110)
    sc_560178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 13), 'sc', False)
    # Obtaining the member 'geterr' of a type (line 110)
    geterr_560179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 13), sc_560178, 'geterr')
    # Calling geterr(args, kwargs) (line 110)
    geterr_call_result_560181 = invoke(stypy.reporting.localization.Localization(__file__, 110, 13), geterr_560179, *[], **kwargs_560180)
    
    # Assigning a type to the variable 'olderr' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'olderr', geterr_call_result_560181)
    
    # Call to errstate(...): (line 111)
    # Processing the call keyword arguments (line 111)
    str_560184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 25), 'str', 'raise')
    keyword_560185 = str_560184
    str_560186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 43), 'str', 'ignore')
    keyword_560187 = str_560186
    kwargs_560188 = {'all': keyword_560185, 'singular': keyword_560187}
    # Getting the type of 'sc' (line 111)
    sc_560182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 9), 'sc', False)
    # Obtaining the member 'errstate' of a type (line 111)
    errstate_560183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 9), sc_560182, 'errstate')
    # Calling errstate(args, kwargs) (line 111)
    errstate_call_result_560189 = invoke(stypy.reporting.localization.Localization(__file__, 111, 9), errstate_560183, *[], **kwargs_560188)
    
    with_560190 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 111, 9), errstate_call_result_560189, 'with parameter', '__enter__', '__exit__')

    if with_560190:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 111)
        enter___560191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 9), errstate_call_result_560189, '__enter__')
        with_enter_560192 = invoke(stypy.reporting.localization.Localization(__file__, 111, 9), enter___560191)
        
        # Call to gammaln(...): (line 112)
        # Processing the call arguments (line 112)
        int_560195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 19), 'int')
        # Processing the call keyword arguments (line 112)
        kwargs_560196 = {}
        # Getting the type of 'sc' (line 112)
        sc_560193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'sc', False)
        # Obtaining the member 'gammaln' of a type (line 112)
        gammaln_560194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), sc_560193, 'gammaln')
        # Calling gammaln(args, kwargs) (line 112)
        gammaln_call_result_560197 = invoke(stypy.reporting.localization.Localization(__file__, 112, 8), gammaln_560194, *[int_560195], **kwargs_560196)
        
        
        # Call to assert_raises(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'sc' (line 113)
        sc_560199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 27), 'sc', False)
        # Obtaining the member 'SpecialFunctionError' of a type (line 113)
        SpecialFunctionError_560200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 27), sc_560199, 'SpecialFunctionError')
        # Processing the call keyword arguments (line 113)
        kwargs_560201 = {}
        # Getting the type of 'assert_raises' (line 113)
        assert_raises_560198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 13), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 113)
        assert_raises_call_result_560202 = invoke(stypy.reporting.localization.Localization(__file__, 113, 13), assert_raises_560198, *[SpecialFunctionError_560200], **kwargs_560201)
        
        with_560203 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 113, 13), assert_raises_call_result_560202, 'with parameter', '__enter__', '__exit__')

        if with_560203:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 113)
            enter___560204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 13), assert_raises_call_result_560202, '__enter__')
            with_enter_560205 = invoke(stypy.reporting.localization.Localization(__file__, 113, 13), enter___560204)
            
            # Call to spence(...): (line 114)
            # Processing the call arguments (line 114)
            float_560208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 22), 'float')
            # Processing the call keyword arguments (line 114)
            kwargs_560209 = {}
            # Getting the type of 'sc' (line 114)
            sc_560206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'sc', False)
            # Obtaining the member 'spence' of a type (line 114)
            spence_560207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 12), sc_560206, 'spence')
            # Calling spence(args, kwargs) (line 114)
            spence_call_result_560210 = invoke(stypy.reporting.localization.Localization(__file__, 114, 12), spence_560207, *[float_560208], **kwargs_560209)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 113)
            exit___560211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 13), assert_raises_call_result_560202, '__exit__')
            with_exit_560212 = invoke(stypy.reporting.localization.Localization(__file__, 113, 13), exit___560211, None, None, None)

        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 111)
        exit___560213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 9), errstate_call_result_560189, '__exit__')
        with_exit_560214 = invoke(stypy.reporting.localization.Localization(__file__, 111, 9), exit___560213, None, None, None)

    
    # Call to assert_equal(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'olderr' (line 115)
    olderr_560216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 17), 'olderr', False)
    
    # Call to geterr(...): (line 115)
    # Processing the call keyword arguments (line 115)
    kwargs_560219 = {}
    # Getting the type of 'sc' (line 115)
    sc_560217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 25), 'sc', False)
    # Obtaining the member 'geterr' of a type (line 115)
    geterr_560218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 25), sc_560217, 'geterr')
    # Calling geterr(args, kwargs) (line 115)
    geterr_call_result_560220 = invoke(stypy.reporting.localization.Localization(__file__, 115, 25), geterr_560218, *[], **kwargs_560219)
    
    # Processing the call keyword arguments (line 115)
    kwargs_560221 = {}
    # Getting the type of 'assert_equal' (line 115)
    assert_equal_560215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 115)
    assert_equal_call_result_560222 = invoke(stypy.reporting.localization.Localization(__file__, 115, 4), assert_equal_560215, *[olderr_560216, geterr_call_result_560220], **kwargs_560221)
    
    
    # ################# End of 'test_errstate_all_but_one(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_errstate_all_but_one' in the type store
    # Getting the type of 'stypy_return_type' (line 109)
    stypy_return_type_560223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_560223)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_errstate_all_but_one'
    return stypy_return_type_560223

# Assigning a type to the variable 'test_errstate_all_but_one' (line 109)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), 'test_errstate_all_but_one', test_errstate_all_but_one)

@norecursion
def test_errprint(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_errprint'
    module_type_store = module_type_store.open_function_context('test_errprint', 118, 0, False)
    
    # Passed parameters checking function
    test_errprint.stypy_localization = localization
    test_errprint.stypy_type_of_self = None
    test_errprint.stypy_type_store = module_type_store
    test_errprint.stypy_function_name = 'test_errprint'
    test_errprint.stypy_param_names_list = []
    test_errprint.stypy_varargs_param_name = None
    test_errprint.stypy_kwargs_param_name = None
    test_errprint.stypy_call_defaults = defaults
    test_errprint.stypy_call_varargs = varargs
    test_errprint.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_errprint', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_errprint', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_errprint(...)' code ##################

    
    # Call to suppress_warnings(...): (line 119)
    # Processing the call keyword arguments (line 119)
    kwargs_560225 = {}
    # Getting the type of 'suppress_warnings' (line 119)
    suppress_warnings_560224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 9), 'suppress_warnings', False)
    # Calling suppress_warnings(args, kwargs) (line 119)
    suppress_warnings_call_result_560226 = invoke(stypy.reporting.localization.Localization(__file__, 119, 9), suppress_warnings_560224, *[], **kwargs_560225)
    
    with_560227 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 119, 9), suppress_warnings_call_result_560226, 'with parameter', '__enter__', '__exit__')

    if with_560227:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 119)
        enter___560228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 9), suppress_warnings_call_result_560226, '__enter__')
        with_enter_560229 = invoke(stypy.reporting.localization.Localization(__file__, 119, 9), enter___560228)
        # Assigning a type to the variable 'sup' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 9), 'sup', with_enter_560229)
        
        # Call to filter(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'DeprecationWarning' (line 120)
        DeprecationWarning_560232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 19), 'DeprecationWarning', False)
        str_560233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 39), 'str', '`errprint` is deprecated!')
        # Processing the call keyword arguments (line 120)
        kwargs_560234 = {}
        # Getting the type of 'sup' (line 120)
        sup_560230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'sup', False)
        # Obtaining the member 'filter' of a type (line 120)
        filter_560231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), sup_560230, 'filter')
        # Calling filter(args, kwargs) (line 120)
        filter_call_result_560235 = invoke(stypy.reporting.localization.Localization(__file__, 120, 8), filter_560231, *[DeprecationWarning_560232, str_560233], **kwargs_560234)
        
        
        # Assigning a Call to a Name (line 121):
        
        # Call to errprint(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'True' (line 121)
        True_560238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 27), 'True', False)
        # Processing the call keyword arguments (line 121)
        kwargs_560239 = {}
        # Getting the type of 'sc' (line 121)
        sc_560236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 15), 'sc', False)
        # Obtaining the member 'errprint' of a type (line 121)
        errprint_560237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 15), sc_560236, 'errprint')
        # Calling errprint(args, kwargs) (line 121)
        errprint_call_result_560240 = invoke(stypy.reporting.localization.Localization(__file__, 121, 15), errprint_560237, *[True_560238], **kwargs_560239)
        
        # Assigning a type to the variable 'flag' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'flag', errprint_call_result_560240)
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 119)
        exit___560241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 9), suppress_warnings_call_result_560226, '__exit__')
        with_exit_560242 = invoke(stypy.reporting.localization.Localization(__file__, 119, 9), exit___560241, None, None, None)

    
    # Try-finally block (line 123)
    
    # Call to assert_(...): (line 124)
    # Processing the call arguments (line 124)
    
    # Call to isinstance(...): (line 124)
    # Processing the call arguments (line 124)
    # Getting the type of 'flag' (line 124)
    flag_560245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 27), 'flag', False)
    # Getting the type of 'bool' (line 124)
    bool_560246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 33), 'bool', False)
    # Processing the call keyword arguments (line 124)
    kwargs_560247 = {}
    # Getting the type of 'isinstance' (line 124)
    isinstance_560244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 16), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 124)
    isinstance_call_result_560248 = invoke(stypy.reporting.localization.Localization(__file__, 124, 16), isinstance_560244, *[flag_560245, bool_560246], **kwargs_560247)
    
    # Processing the call keyword arguments (line 124)
    kwargs_560249 = {}
    # Getting the type of 'assert_' (line 124)
    assert__560243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'assert_', False)
    # Calling assert_(args, kwargs) (line 124)
    assert__call_result_560250 = invoke(stypy.reporting.localization.Localization(__file__, 124, 8), assert__560243, *[isinstance_call_result_560248], **kwargs_560249)
    
    
    # Call to warns(...): (line 125)
    # Processing the call arguments (line 125)
    # Getting the type of 'sc' (line 125)
    sc_560253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 26), 'sc', False)
    # Obtaining the member 'SpecialFunctionWarning' of a type (line 125)
    SpecialFunctionWarning_560254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 26), sc_560253, 'SpecialFunctionWarning')
    # Processing the call keyword arguments (line 125)
    kwargs_560255 = {}
    # Getting the type of 'pytest' (line 125)
    pytest_560251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 13), 'pytest', False)
    # Obtaining the member 'warns' of a type (line 125)
    warns_560252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 13), pytest_560251, 'warns')
    # Calling warns(args, kwargs) (line 125)
    warns_call_result_560256 = invoke(stypy.reporting.localization.Localization(__file__, 125, 13), warns_560252, *[SpecialFunctionWarning_560254], **kwargs_560255)
    
    with_560257 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 125, 13), warns_call_result_560256, 'with parameter', '__enter__', '__exit__')

    if with_560257:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 125)
        enter___560258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 13), warns_call_result_560256, '__enter__')
        with_enter_560259 = invoke(stypy.reporting.localization.Localization(__file__, 125, 13), enter___560258)
        
        # Call to loggamma(...): (line 126)
        # Processing the call arguments (line 126)
        int_560262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 24), 'int')
        # Processing the call keyword arguments (line 126)
        kwargs_560263 = {}
        # Getting the type of 'sc' (line 126)
        sc_560260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'sc', False)
        # Obtaining the member 'loggamma' of a type (line 126)
        loggamma_560261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 12), sc_560260, 'loggamma')
        # Calling loggamma(args, kwargs) (line 126)
        loggamma_call_result_560264 = invoke(stypy.reporting.localization.Localization(__file__, 126, 12), loggamma_560261, *[int_560262], **kwargs_560263)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 125)
        exit___560265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 13), warns_call_result_560256, '__exit__')
        with_exit_560266 = invoke(stypy.reporting.localization.Localization(__file__, 125, 13), exit___560265, None, None, None)

    
    # finally branch of the try-finally block (line 123)
    
    # Call to suppress_warnings(...): (line 128)
    # Processing the call keyword arguments (line 128)
    kwargs_560268 = {}
    # Getting the type of 'suppress_warnings' (line 128)
    suppress_warnings_560267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 13), 'suppress_warnings', False)
    # Calling suppress_warnings(args, kwargs) (line 128)
    suppress_warnings_call_result_560269 = invoke(stypy.reporting.localization.Localization(__file__, 128, 13), suppress_warnings_560267, *[], **kwargs_560268)
    
    with_560270 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 128, 13), suppress_warnings_call_result_560269, 'with parameter', '__enter__', '__exit__')

    if with_560270:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 128)
        enter___560271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 13), suppress_warnings_call_result_560269, '__enter__')
        with_enter_560272 = invoke(stypy.reporting.localization.Localization(__file__, 128, 13), enter___560271)
        # Assigning a type to the variable 'sup' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 13), 'sup', with_enter_560272)
        
        # Call to filter(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'DeprecationWarning' (line 129)
        DeprecationWarning_560275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 23), 'DeprecationWarning', False)
        str_560276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 43), 'str', '`errprint` is deprecated!')
        # Processing the call keyword arguments (line 129)
        kwargs_560277 = {}
        # Getting the type of 'sup' (line 129)
        sup_560273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'sup', False)
        # Obtaining the member 'filter' of a type (line 129)
        filter_560274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 12), sup_560273, 'filter')
        # Calling filter(args, kwargs) (line 129)
        filter_call_result_560278 = invoke(stypy.reporting.localization.Localization(__file__, 129, 12), filter_560274, *[DeprecationWarning_560275, str_560276], **kwargs_560277)
        
        
        # Call to errprint(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'flag' (line 130)
        flag_560281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 24), 'flag', False)
        # Processing the call keyword arguments (line 130)
        kwargs_560282 = {}
        # Getting the type of 'sc' (line 130)
        sc_560279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'sc', False)
        # Obtaining the member 'errprint' of a type (line 130)
        errprint_560280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 12), sc_560279, 'errprint')
        # Calling errprint(args, kwargs) (line 130)
        errprint_call_result_560283 = invoke(stypy.reporting.localization.Localization(__file__, 130, 12), errprint_560280, *[flag_560281], **kwargs_560282)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 128)
        exit___560284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 13), suppress_warnings_call_result_560269, '__exit__')
        with_exit_560285 = invoke(stypy.reporting.localization.Localization(__file__, 128, 13), exit___560284, None, None, None)

    
    
    # ################# End of 'test_errprint(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_errprint' in the type store
    # Getting the type of 'stypy_return_type' (line 118)
    stypy_return_type_560286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_560286)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_errprint'
    return stypy_return_type_560286

# Assigning a type to the variable 'test_errprint' (line 118)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 0), 'test_errprint', test_errprint)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
