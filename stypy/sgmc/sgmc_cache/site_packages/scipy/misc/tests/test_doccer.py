
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Some tests for the documenting decorator and support functions '''
2: 
3: from __future__ import division, print_function, absolute_import
4: 
5: import sys
6: import pytest
7: from numpy.testing import assert_equal
8: 
9: from scipy.misc import doccer
10: 
11: # python -OO strips docstrings
12: DOCSTRINGS_STRIPPED = sys.flags.optimize > 1
13: 
14: docstring = \
15: '''Docstring
16:     %(strtest1)s
17:         %(strtest2)s
18:      %(strtest3)s
19: '''
20: param_doc1 = \
21: '''Another test
22:    with some indent'''
23: 
24: param_doc2 = \
25: '''Another test, one line'''
26: 
27: param_doc3 = \
28: '''    Another test
29:        with some indent'''
30: 
31: doc_dict = {'strtest1':param_doc1,
32:             'strtest2':param_doc2,
33:             'strtest3':param_doc3}
34: 
35: filled_docstring = \
36: '''Docstring
37:     Another test
38:        with some indent
39:         Another test, one line
40:      Another test
41:        with some indent
42: '''
43: 
44: 
45: def test_unindent():
46:     assert_equal(doccer.unindent_string(param_doc1), param_doc1)
47:     assert_equal(doccer.unindent_string(param_doc2), param_doc2)
48:     assert_equal(doccer.unindent_string(param_doc3), param_doc1)
49: 
50: 
51: def test_unindent_dict():
52:     d2 = doccer.unindent_dict(doc_dict)
53:     assert_equal(d2['strtest1'], doc_dict['strtest1'])
54:     assert_equal(d2['strtest2'], doc_dict['strtest2'])
55:     assert_equal(d2['strtest3'], doc_dict['strtest1'])
56: 
57: 
58: def test_docformat():
59:     udd = doccer.unindent_dict(doc_dict)
60:     formatted = doccer.docformat(docstring, udd)
61:     assert_equal(formatted, filled_docstring)
62:     single_doc = 'Single line doc %(strtest1)s'
63:     formatted = doccer.docformat(single_doc, doc_dict)
64:     # Note - initial indent of format string does not
65:     # affect subsequent indent of inserted parameter
66:     assert_equal(formatted, '''Single line doc Another test
67:    with some indent''')
68: 
69: 
70: @pytest.mark.skipif(DOCSTRINGS_STRIPPED, reason="docstrings stripped")
71: def test_decorator():
72:     # with unindentation of parameters
73:     decorator = doccer.filldoc(doc_dict, True)
74: 
75:     @decorator
76:     def func():
77:         ''' Docstring
78:         %(strtest3)s
79:         '''
80:     assert_equal(func.__doc__, ''' Docstring
81:         Another test
82:            with some indent
83:         ''')
84: 
85:     # without unindentation of parameters
86:     decorator = doccer.filldoc(doc_dict, False)
87: 
88:     @decorator
89:     def func():
90:         ''' Docstring
91:         %(strtest3)s
92:         '''
93:     assert_equal(func.__doc__, ''' Docstring
94:             Another test
95:                with some indent
96:         ''')
97: 
98: 
99: @pytest.mark.skipif(DOCSTRINGS_STRIPPED, reason="docstrings stripped")
100: def test_inherit_docstring_from():
101: 
102:     class Foo(object):
103:         def func(self):
104:             '''Do something useful.'''
105:             return
106: 
107:         def func2(self):
108:             '''Something else.'''
109: 
110:     class Bar(Foo):
111:         @doccer.inherit_docstring_from(Foo)
112:         def func(self):
113:             '''%(super)sABC'''
114:             return
115: 
116:         @doccer.inherit_docstring_from(Foo)
117:         def func2(self):
118:             # No docstring.
119:             return
120: 
121:     assert_equal(Bar.func.__doc__, Foo.func.__doc__ + 'ABC')
122:     assert_equal(Bar.func2.__doc__, Foo.func2.__doc__)
123:     bar = Bar()
124:     assert_equal(bar.func.__doc__, Foo.func.__doc__ + 'ABC')
125:     assert_equal(bar.func2.__doc__, Foo.func2.__doc__)
126: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_115461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', ' Some tests for the documenting decorator and support functions ')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import sys' statement (line 5)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import pytest' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/misc/tests/')
import_115462 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'pytest')

if (type(import_115462) is not StypyTypeError):

    if (import_115462 != 'pyd_module'):
        __import__(import_115462)
        sys_modules_115463 = sys.modules[import_115462]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'pytest', sys_modules_115463.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'pytest', import_115462)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/misc/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.testing import assert_equal' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/misc/tests/')
import_115464 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing')

if (type(import_115464) is not StypyTypeError):

    if (import_115464 != 'pyd_module'):
        __import__(import_115464)
        sys_modules_115465 = sys.modules[import_115464]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', sys_modules_115465.module_type_store, module_type_store, ['assert_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_115465, sys_modules_115465.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', None, module_type_store, ['assert_equal'], [assert_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', import_115464)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/misc/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.misc import doccer' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/misc/tests/')
import_115466 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.misc')

if (type(import_115466) is not StypyTypeError):

    if (import_115466 != 'pyd_module'):
        __import__(import_115466)
        sys_modules_115467 = sys.modules[import_115466]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.misc', sys_modules_115467.module_type_store, module_type_store, ['doccer'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_115467, sys_modules_115467.module_type_store, module_type_store)
    else:
        from scipy.misc import doccer

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.misc', None, module_type_store, ['doccer'], [doccer])

else:
    # Assigning a type to the variable 'scipy.misc' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.misc', import_115466)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/misc/tests/')


# Assigning a Compare to a Name (line 12):

# Getting the type of 'sys' (line 12)
sys_115468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 22), 'sys')
# Obtaining the member 'flags' of a type (line 12)
flags_115469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 22), sys_115468, 'flags')
# Obtaining the member 'optimize' of a type (line 12)
optimize_115470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 22), flags_115469, 'optimize')
int_115471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 43), 'int')
# Applying the binary operator '>' (line 12)
result_gt_115472 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 22), '>', optimize_115470, int_115471)

# Assigning a type to the variable 'DOCSTRINGS_STRIPPED' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'DOCSTRINGS_STRIPPED', result_gt_115472)

# Assigning a Str to a Name (line 14):
str_115473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, (-1)), 'str', 'Docstring\n    %(strtest1)s\n        %(strtest2)s\n     %(strtest3)s\n')
# Assigning a type to the variable 'docstring' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'docstring', str_115473)

# Assigning a Str to a Name (line 20):
str_115474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, (-1)), 'str', 'Another test\n   with some indent')
# Assigning a type to the variable 'param_doc1' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'param_doc1', str_115474)

# Assigning a Str to a Name (line 24):
str_115475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 0), 'str', 'Another test, one line')
# Assigning a type to the variable 'param_doc2' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'param_doc2', str_115475)

# Assigning a Str to a Name (line 27):
str_115476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, (-1)), 'str', '    Another test\n       with some indent')
# Assigning a type to the variable 'param_doc3' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'param_doc3', str_115476)

# Assigning a Dict to a Name (line 31):

# Obtaining an instance of the builtin type 'dict' (line 31)
dict_115477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 11), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 31)
# Adding element type (key, value) (line 31)
str_115478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 12), 'str', 'strtest1')
# Getting the type of 'param_doc1' (line 31)
param_doc1_115479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 23), 'param_doc1')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 11), dict_115477, (str_115478, param_doc1_115479))
# Adding element type (key, value) (line 31)
str_115480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 12), 'str', 'strtest2')
# Getting the type of 'param_doc2' (line 32)
param_doc2_115481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 23), 'param_doc2')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 11), dict_115477, (str_115480, param_doc2_115481))
# Adding element type (key, value) (line 31)
str_115482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 12), 'str', 'strtest3')
# Getting the type of 'param_doc3' (line 33)
param_doc3_115483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 23), 'param_doc3')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 11), dict_115477, (str_115482, param_doc3_115483))

# Assigning a type to the variable 'doc_dict' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'doc_dict', dict_115477)

# Assigning a Str to a Name (line 35):
str_115484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, (-1)), 'str', 'Docstring\n    Another test\n       with some indent\n        Another test, one line\n     Another test\n       with some indent\n')
# Assigning a type to the variable 'filled_docstring' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'filled_docstring', str_115484)

@norecursion
def test_unindent(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_unindent'
    module_type_store = module_type_store.open_function_context('test_unindent', 45, 0, False)
    
    # Passed parameters checking function
    test_unindent.stypy_localization = localization
    test_unindent.stypy_type_of_self = None
    test_unindent.stypy_type_store = module_type_store
    test_unindent.stypy_function_name = 'test_unindent'
    test_unindent.stypy_param_names_list = []
    test_unindent.stypy_varargs_param_name = None
    test_unindent.stypy_kwargs_param_name = None
    test_unindent.stypy_call_defaults = defaults
    test_unindent.stypy_call_varargs = varargs
    test_unindent.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_unindent', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_unindent', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_unindent(...)' code ##################

    
    # Call to assert_equal(...): (line 46)
    # Processing the call arguments (line 46)
    
    # Call to unindent_string(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'param_doc1' (line 46)
    param_doc1_115488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 40), 'param_doc1', False)
    # Processing the call keyword arguments (line 46)
    kwargs_115489 = {}
    # Getting the type of 'doccer' (line 46)
    doccer_115486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 17), 'doccer', False)
    # Obtaining the member 'unindent_string' of a type (line 46)
    unindent_string_115487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 17), doccer_115486, 'unindent_string')
    # Calling unindent_string(args, kwargs) (line 46)
    unindent_string_call_result_115490 = invoke(stypy.reporting.localization.Localization(__file__, 46, 17), unindent_string_115487, *[param_doc1_115488], **kwargs_115489)
    
    # Getting the type of 'param_doc1' (line 46)
    param_doc1_115491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 53), 'param_doc1', False)
    # Processing the call keyword arguments (line 46)
    kwargs_115492 = {}
    # Getting the type of 'assert_equal' (line 46)
    assert_equal_115485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 46)
    assert_equal_call_result_115493 = invoke(stypy.reporting.localization.Localization(__file__, 46, 4), assert_equal_115485, *[unindent_string_call_result_115490, param_doc1_115491], **kwargs_115492)
    
    
    # Call to assert_equal(...): (line 47)
    # Processing the call arguments (line 47)
    
    # Call to unindent_string(...): (line 47)
    # Processing the call arguments (line 47)
    # Getting the type of 'param_doc2' (line 47)
    param_doc2_115497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 40), 'param_doc2', False)
    # Processing the call keyword arguments (line 47)
    kwargs_115498 = {}
    # Getting the type of 'doccer' (line 47)
    doccer_115495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 17), 'doccer', False)
    # Obtaining the member 'unindent_string' of a type (line 47)
    unindent_string_115496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 17), doccer_115495, 'unindent_string')
    # Calling unindent_string(args, kwargs) (line 47)
    unindent_string_call_result_115499 = invoke(stypy.reporting.localization.Localization(__file__, 47, 17), unindent_string_115496, *[param_doc2_115497], **kwargs_115498)
    
    # Getting the type of 'param_doc2' (line 47)
    param_doc2_115500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 53), 'param_doc2', False)
    # Processing the call keyword arguments (line 47)
    kwargs_115501 = {}
    # Getting the type of 'assert_equal' (line 47)
    assert_equal_115494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 47)
    assert_equal_call_result_115502 = invoke(stypy.reporting.localization.Localization(__file__, 47, 4), assert_equal_115494, *[unindent_string_call_result_115499, param_doc2_115500], **kwargs_115501)
    
    
    # Call to assert_equal(...): (line 48)
    # Processing the call arguments (line 48)
    
    # Call to unindent_string(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'param_doc3' (line 48)
    param_doc3_115506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 40), 'param_doc3', False)
    # Processing the call keyword arguments (line 48)
    kwargs_115507 = {}
    # Getting the type of 'doccer' (line 48)
    doccer_115504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 17), 'doccer', False)
    # Obtaining the member 'unindent_string' of a type (line 48)
    unindent_string_115505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 17), doccer_115504, 'unindent_string')
    # Calling unindent_string(args, kwargs) (line 48)
    unindent_string_call_result_115508 = invoke(stypy.reporting.localization.Localization(__file__, 48, 17), unindent_string_115505, *[param_doc3_115506], **kwargs_115507)
    
    # Getting the type of 'param_doc1' (line 48)
    param_doc1_115509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 53), 'param_doc1', False)
    # Processing the call keyword arguments (line 48)
    kwargs_115510 = {}
    # Getting the type of 'assert_equal' (line 48)
    assert_equal_115503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 48)
    assert_equal_call_result_115511 = invoke(stypy.reporting.localization.Localization(__file__, 48, 4), assert_equal_115503, *[unindent_string_call_result_115508, param_doc1_115509], **kwargs_115510)
    
    
    # ################# End of 'test_unindent(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_unindent' in the type store
    # Getting the type of 'stypy_return_type' (line 45)
    stypy_return_type_115512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_115512)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_unindent'
    return stypy_return_type_115512

# Assigning a type to the variable 'test_unindent' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'test_unindent', test_unindent)

@norecursion
def test_unindent_dict(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_unindent_dict'
    module_type_store = module_type_store.open_function_context('test_unindent_dict', 51, 0, False)
    
    # Passed parameters checking function
    test_unindent_dict.stypy_localization = localization
    test_unindent_dict.stypy_type_of_self = None
    test_unindent_dict.stypy_type_store = module_type_store
    test_unindent_dict.stypy_function_name = 'test_unindent_dict'
    test_unindent_dict.stypy_param_names_list = []
    test_unindent_dict.stypy_varargs_param_name = None
    test_unindent_dict.stypy_kwargs_param_name = None
    test_unindent_dict.stypy_call_defaults = defaults
    test_unindent_dict.stypy_call_varargs = varargs
    test_unindent_dict.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_unindent_dict', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_unindent_dict', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_unindent_dict(...)' code ##################

    
    # Assigning a Call to a Name (line 52):
    
    # Call to unindent_dict(...): (line 52)
    # Processing the call arguments (line 52)
    # Getting the type of 'doc_dict' (line 52)
    doc_dict_115515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 30), 'doc_dict', False)
    # Processing the call keyword arguments (line 52)
    kwargs_115516 = {}
    # Getting the type of 'doccer' (line 52)
    doccer_115513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 9), 'doccer', False)
    # Obtaining the member 'unindent_dict' of a type (line 52)
    unindent_dict_115514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 9), doccer_115513, 'unindent_dict')
    # Calling unindent_dict(args, kwargs) (line 52)
    unindent_dict_call_result_115517 = invoke(stypy.reporting.localization.Localization(__file__, 52, 9), unindent_dict_115514, *[doc_dict_115515], **kwargs_115516)
    
    # Assigning a type to the variable 'd2' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'd2', unindent_dict_call_result_115517)
    
    # Call to assert_equal(...): (line 53)
    # Processing the call arguments (line 53)
    
    # Obtaining the type of the subscript
    str_115519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 20), 'str', 'strtest1')
    # Getting the type of 'd2' (line 53)
    d2_115520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 17), 'd2', False)
    # Obtaining the member '__getitem__' of a type (line 53)
    getitem___115521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 17), d2_115520, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 53)
    subscript_call_result_115522 = invoke(stypy.reporting.localization.Localization(__file__, 53, 17), getitem___115521, str_115519)
    
    
    # Obtaining the type of the subscript
    str_115523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 42), 'str', 'strtest1')
    # Getting the type of 'doc_dict' (line 53)
    doc_dict_115524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 33), 'doc_dict', False)
    # Obtaining the member '__getitem__' of a type (line 53)
    getitem___115525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 33), doc_dict_115524, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 53)
    subscript_call_result_115526 = invoke(stypy.reporting.localization.Localization(__file__, 53, 33), getitem___115525, str_115523)
    
    # Processing the call keyword arguments (line 53)
    kwargs_115527 = {}
    # Getting the type of 'assert_equal' (line 53)
    assert_equal_115518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 53)
    assert_equal_call_result_115528 = invoke(stypy.reporting.localization.Localization(__file__, 53, 4), assert_equal_115518, *[subscript_call_result_115522, subscript_call_result_115526], **kwargs_115527)
    
    
    # Call to assert_equal(...): (line 54)
    # Processing the call arguments (line 54)
    
    # Obtaining the type of the subscript
    str_115530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 20), 'str', 'strtest2')
    # Getting the type of 'd2' (line 54)
    d2_115531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 17), 'd2', False)
    # Obtaining the member '__getitem__' of a type (line 54)
    getitem___115532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 17), d2_115531, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 54)
    subscript_call_result_115533 = invoke(stypy.reporting.localization.Localization(__file__, 54, 17), getitem___115532, str_115530)
    
    
    # Obtaining the type of the subscript
    str_115534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 42), 'str', 'strtest2')
    # Getting the type of 'doc_dict' (line 54)
    doc_dict_115535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 33), 'doc_dict', False)
    # Obtaining the member '__getitem__' of a type (line 54)
    getitem___115536 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 33), doc_dict_115535, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 54)
    subscript_call_result_115537 = invoke(stypy.reporting.localization.Localization(__file__, 54, 33), getitem___115536, str_115534)
    
    # Processing the call keyword arguments (line 54)
    kwargs_115538 = {}
    # Getting the type of 'assert_equal' (line 54)
    assert_equal_115529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 54)
    assert_equal_call_result_115539 = invoke(stypy.reporting.localization.Localization(__file__, 54, 4), assert_equal_115529, *[subscript_call_result_115533, subscript_call_result_115537], **kwargs_115538)
    
    
    # Call to assert_equal(...): (line 55)
    # Processing the call arguments (line 55)
    
    # Obtaining the type of the subscript
    str_115541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 20), 'str', 'strtest3')
    # Getting the type of 'd2' (line 55)
    d2_115542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 17), 'd2', False)
    # Obtaining the member '__getitem__' of a type (line 55)
    getitem___115543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 17), d2_115542, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 55)
    subscript_call_result_115544 = invoke(stypy.reporting.localization.Localization(__file__, 55, 17), getitem___115543, str_115541)
    
    
    # Obtaining the type of the subscript
    str_115545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 42), 'str', 'strtest1')
    # Getting the type of 'doc_dict' (line 55)
    doc_dict_115546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 33), 'doc_dict', False)
    # Obtaining the member '__getitem__' of a type (line 55)
    getitem___115547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 33), doc_dict_115546, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 55)
    subscript_call_result_115548 = invoke(stypy.reporting.localization.Localization(__file__, 55, 33), getitem___115547, str_115545)
    
    # Processing the call keyword arguments (line 55)
    kwargs_115549 = {}
    # Getting the type of 'assert_equal' (line 55)
    assert_equal_115540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 55)
    assert_equal_call_result_115550 = invoke(stypy.reporting.localization.Localization(__file__, 55, 4), assert_equal_115540, *[subscript_call_result_115544, subscript_call_result_115548], **kwargs_115549)
    
    
    # ################# End of 'test_unindent_dict(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_unindent_dict' in the type store
    # Getting the type of 'stypy_return_type' (line 51)
    stypy_return_type_115551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_115551)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_unindent_dict'
    return stypy_return_type_115551

# Assigning a type to the variable 'test_unindent_dict' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'test_unindent_dict', test_unindent_dict)

@norecursion
def test_docformat(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_docformat'
    module_type_store = module_type_store.open_function_context('test_docformat', 58, 0, False)
    
    # Passed parameters checking function
    test_docformat.stypy_localization = localization
    test_docformat.stypy_type_of_self = None
    test_docformat.stypy_type_store = module_type_store
    test_docformat.stypy_function_name = 'test_docformat'
    test_docformat.stypy_param_names_list = []
    test_docformat.stypy_varargs_param_name = None
    test_docformat.stypy_kwargs_param_name = None
    test_docformat.stypy_call_defaults = defaults
    test_docformat.stypy_call_varargs = varargs
    test_docformat.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_docformat', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_docformat', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_docformat(...)' code ##################

    
    # Assigning a Call to a Name (line 59):
    
    # Call to unindent_dict(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'doc_dict' (line 59)
    doc_dict_115554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 31), 'doc_dict', False)
    # Processing the call keyword arguments (line 59)
    kwargs_115555 = {}
    # Getting the type of 'doccer' (line 59)
    doccer_115552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 10), 'doccer', False)
    # Obtaining the member 'unindent_dict' of a type (line 59)
    unindent_dict_115553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 10), doccer_115552, 'unindent_dict')
    # Calling unindent_dict(args, kwargs) (line 59)
    unindent_dict_call_result_115556 = invoke(stypy.reporting.localization.Localization(__file__, 59, 10), unindent_dict_115553, *[doc_dict_115554], **kwargs_115555)
    
    # Assigning a type to the variable 'udd' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'udd', unindent_dict_call_result_115556)
    
    # Assigning a Call to a Name (line 60):
    
    # Call to docformat(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'docstring' (line 60)
    docstring_115559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 33), 'docstring', False)
    # Getting the type of 'udd' (line 60)
    udd_115560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 44), 'udd', False)
    # Processing the call keyword arguments (line 60)
    kwargs_115561 = {}
    # Getting the type of 'doccer' (line 60)
    doccer_115557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'doccer', False)
    # Obtaining the member 'docformat' of a type (line 60)
    docformat_115558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 16), doccer_115557, 'docformat')
    # Calling docformat(args, kwargs) (line 60)
    docformat_call_result_115562 = invoke(stypy.reporting.localization.Localization(__file__, 60, 16), docformat_115558, *[docstring_115559, udd_115560], **kwargs_115561)
    
    # Assigning a type to the variable 'formatted' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'formatted', docformat_call_result_115562)
    
    # Call to assert_equal(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'formatted' (line 61)
    formatted_115564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 17), 'formatted', False)
    # Getting the type of 'filled_docstring' (line 61)
    filled_docstring_115565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 28), 'filled_docstring', False)
    # Processing the call keyword arguments (line 61)
    kwargs_115566 = {}
    # Getting the type of 'assert_equal' (line 61)
    assert_equal_115563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 61)
    assert_equal_call_result_115567 = invoke(stypy.reporting.localization.Localization(__file__, 61, 4), assert_equal_115563, *[formatted_115564, filled_docstring_115565], **kwargs_115566)
    
    
    # Assigning a Str to a Name (line 62):
    str_115568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 17), 'str', 'Single line doc %(strtest1)s')
    # Assigning a type to the variable 'single_doc' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'single_doc', str_115568)
    
    # Assigning a Call to a Name (line 63):
    
    # Call to docformat(...): (line 63)
    # Processing the call arguments (line 63)
    # Getting the type of 'single_doc' (line 63)
    single_doc_115571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 33), 'single_doc', False)
    # Getting the type of 'doc_dict' (line 63)
    doc_dict_115572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 45), 'doc_dict', False)
    # Processing the call keyword arguments (line 63)
    kwargs_115573 = {}
    # Getting the type of 'doccer' (line 63)
    doccer_115569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 16), 'doccer', False)
    # Obtaining the member 'docformat' of a type (line 63)
    docformat_115570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 16), doccer_115569, 'docformat')
    # Calling docformat(args, kwargs) (line 63)
    docformat_call_result_115574 = invoke(stypy.reporting.localization.Localization(__file__, 63, 16), docformat_115570, *[single_doc_115571, doc_dict_115572], **kwargs_115573)
    
    # Assigning a type to the variable 'formatted' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'formatted', docformat_call_result_115574)
    
    # Call to assert_equal(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 'formatted' (line 66)
    formatted_115576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 17), 'formatted', False)
    str_115577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, (-1)), 'str', 'Single line doc Another test\n   with some indent')
    # Processing the call keyword arguments (line 66)
    kwargs_115578 = {}
    # Getting the type of 'assert_equal' (line 66)
    assert_equal_115575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 66)
    assert_equal_call_result_115579 = invoke(stypy.reporting.localization.Localization(__file__, 66, 4), assert_equal_115575, *[formatted_115576, str_115577], **kwargs_115578)
    
    
    # ################# End of 'test_docformat(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_docformat' in the type store
    # Getting the type of 'stypy_return_type' (line 58)
    stypy_return_type_115580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_115580)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_docformat'
    return stypy_return_type_115580

# Assigning a type to the variable 'test_docformat' (line 58)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'test_docformat', test_docformat)

@norecursion
def test_decorator(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_decorator'
    module_type_store = module_type_store.open_function_context('test_decorator', 70, 0, False)
    
    # Passed parameters checking function
    test_decorator.stypy_localization = localization
    test_decorator.stypy_type_of_self = None
    test_decorator.stypy_type_store = module_type_store
    test_decorator.stypy_function_name = 'test_decorator'
    test_decorator.stypy_param_names_list = []
    test_decorator.stypy_varargs_param_name = None
    test_decorator.stypy_kwargs_param_name = None
    test_decorator.stypy_call_defaults = defaults
    test_decorator.stypy_call_varargs = varargs
    test_decorator.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_decorator', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_decorator', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_decorator(...)' code ##################

    
    # Assigning a Call to a Name (line 73):
    
    # Call to filldoc(...): (line 73)
    # Processing the call arguments (line 73)
    # Getting the type of 'doc_dict' (line 73)
    doc_dict_115583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 31), 'doc_dict', False)
    # Getting the type of 'True' (line 73)
    True_115584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 41), 'True', False)
    # Processing the call keyword arguments (line 73)
    kwargs_115585 = {}
    # Getting the type of 'doccer' (line 73)
    doccer_115581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 16), 'doccer', False)
    # Obtaining the member 'filldoc' of a type (line 73)
    filldoc_115582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 16), doccer_115581, 'filldoc')
    # Calling filldoc(args, kwargs) (line 73)
    filldoc_call_result_115586 = invoke(stypy.reporting.localization.Localization(__file__, 73, 16), filldoc_115582, *[doc_dict_115583, True_115584], **kwargs_115585)
    
    # Assigning a type to the variable 'decorator' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'decorator', filldoc_call_result_115586)

    @norecursion
    def func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'func'
        module_type_store = module_type_store.open_function_context('func', 75, 4, False)
        
        # Passed parameters checking function
        func.stypy_localization = localization
        func.stypy_type_of_self = None
        func.stypy_type_store = module_type_store
        func.stypy_function_name = 'func'
        func.stypy_param_names_list = []
        func.stypy_varargs_param_name = None
        func.stypy_kwargs_param_name = None
        func.stypy_call_defaults = defaults
        func.stypy_call_varargs = varargs
        func.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'func', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'func', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'func(...)' code ##################

        str_115587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, (-1)), 'str', ' Docstring\n        %(strtest3)s\n        ')
        
        # ################# End of 'func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'func' in the type store
        # Getting the type of 'stypy_return_type' (line 75)
        stypy_return_type_115588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_115588)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'func'
        return stypy_return_type_115588

    # Assigning a type to the variable 'func' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'func', func)
    
    # Call to assert_equal(...): (line 80)
    # Processing the call arguments (line 80)
    # Getting the type of 'func' (line 80)
    func_115590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 17), 'func', False)
    # Obtaining the member '__doc__' of a type (line 80)
    doc___115591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 17), func_115590, '__doc__')
    str_115592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, (-1)), 'str', ' Docstring\n        Another test\n           with some indent\n        ')
    # Processing the call keyword arguments (line 80)
    kwargs_115593 = {}
    # Getting the type of 'assert_equal' (line 80)
    assert_equal_115589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 80)
    assert_equal_call_result_115594 = invoke(stypy.reporting.localization.Localization(__file__, 80, 4), assert_equal_115589, *[doc___115591, str_115592], **kwargs_115593)
    
    
    # Assigning a Call to a Name (line 86):
    
    # Call to filldoc(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'doc_dict' (line 86)
    doc_dict_115597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 31), 'doc_dict', False)
    # Getting the type of 'False' (line 86)
    False_115598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 41), 'False', False)
    # Processing the call keyword arguments (line 86)
    kwargs_115599 = {}
    # Getting the type of 'doccer' (line 86)
    doccer_115595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 16), 'doccer', False)
    # Obtaining the member 'filldoc' of a type (line 86)
    filldoc_115596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 16), doccer_115595, 'filldoc')
    # Calling filldoc(args, kwargs) (line 86)
    filldoc_call_result_115600 = invoke(stypy.reporting.localization.Localization(__file__, 86, 16), filldoc_115596, *[doc_dict_115597, False_115598], **kwargs_115599)
    
    # Assigning a type to the variable 'decorator' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'decorator', filldoc_call_result_115600)

    @norecursion
    def func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'func'
        module_type_store = module_type_store.open_function_context('func', 88, 4, False)
        
        # Passed parameters checking function
        func.stypy_localization = localization
        func.stypy_type_of_self = None
        func.stypy_type_store = module_type_store
        func.stypy_function_name = 'func'
        func.stypy_param_names_list = []
        func.stypy_varargs_param_name = None
        func.stypy_kwargs_param_name = None
        func.stypy_call_defaults = defaults
        func.stypy_call_varargs = varargs
        func.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'func', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'func', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'func(...)' code ##################

        str_115601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, (-1)), 'str', ' Docstring\n        %(strtest3)s\n        ')
        
        # ################# End of 'func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'func' in the type store
        # Getting the type of 'stypy_return_type' (line 88)
        stypy_return_type_115602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_115602)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'func'
        return stypy_return_type_115602

    # Assigning a type to the variable 'func' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'func', func)
    
    # Call to assert_equal(...): (line 93)
    # Processing the call arguments (line 93)
    # Getting the type of 'func' (line 93)
    func_115604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 17), 'func', False)
    # Obtaining the member '__doc__' of a type (line 93)
    doc___115605 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 17), func_115604, '__doc__')
    str_115606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, (-1)), 'str', ' Docstring\n            Another test\n               with some indent\n        ')
    # Processing the call keyword arguments (line 93)
    kwargs_115607 = {}
    # Getting the type of 'assert_equal' (line 93)
    assert_equal_115603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 93)
    assert_equal_call_result_115608 = invoke(stypy.reporting.localization.Localization(__file__, 93, 4), assert_equal_115603, *[doc___115605, str_115606], **kwargs_115607)
    
    
    # ################# End of 'test_decorator(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_decorator' in the type store
    # Getting the type of 'stypy_return_type' (line 70)
    stypy_return_type_115609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_115609)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_decorator'
    return stypy_return_type_115609

# Assigning a type to the variable 'test_decorator' (line 70)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'test_decorator', test_decorator)

@norecursion
def test_inherit_docstring_from(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_inherit_docstring_from'
    module_type_store = module_type_store.open_function_context('test_inherit_docstring_from', 99, 0, False)
    
    # Passed parameters checking function
    test_inherit_docstring_from.stypy_localization = localization
    test_inherit_docstring_from.stypy_type_of_self = None
    test_inherit_docstring_from.stypy_type_store = module_type_store
    test_inherit_docstring_from.stypy_function_name = 'test_inherit_docstring_from'
    test_inherit_docstring_from.stypy_param_names_list = []
    test_inherit_docstring_from.stypy_varargs_param_name = None
    test_inherit_docstring_from.stypy_kwargs_param_name = None
    test_inherit_docstring_from.stypy_call_defaults = defaults
    test_inherit_docstring_from.stypy_call_varargs = varargs
    test_inherit_docstring_from.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_inherit_docstring_from', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_inherit_docstring_from', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_inherit_docstring_from(...)' code ##################

    # Declaration of the 'Foo' class

    class Foo(object, ):

        @norecursion
        def func(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'func'
            module_type_store = module_type_store.open_function_context('func', 103, 8, False)
            # Assigning a type to the variable 'self' (line 104)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Foo.func.__dict__.__setitem__('stypy_localization', localization)
            Foo.func.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Foo.func.__dict__.__setitem__('stypy_type_store', module_type_store)
            Foo.func.__dict__.__setitem__('stypy_function_name', 'Foo.func')
            Foo.func.__dict__.__setitem__('stypy_param_names_list', [])
            Foo.func.__dict__.__setitem__('stypy_varargs_param_name', None)
            Foo.func.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Foo.func.__dict__.__setitem__('stypy_call_defaults', defaults)
            Foo.func.__dict__.__setitem__('stypy_call_varargs', varargs)
            Foo.func.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Foo.func.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.func', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'func', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'func(...)' code ##################

            str_115610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 12), 'str', 'Do something useful.')
            # Assigning a type to the variable 'stypy_return_type' (line 105)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'stypy_return_type', types.NoneType)
            
            # ################# End of 'func(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'func' in the type store
            # Getting the type of 'stypy_return_type' (line 103)
            stypy_return_type_115611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_115611)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'func'
            return stypy_return_type_115611


        @norecursion
        def func2(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'func2'
            module_type_store = module_type_store.open_function_context('func2', 107, 8, False)
            # Assigning a type to the variable 'self' (line 108)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Foo.func2.__dict__.__setitem__('stypy_localization', localization)
            Foo.func2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Foo.func2.__dict__.__setitem__('stypy_type_store', module_type_store)
            Foo.func2.__dict__.__setitem__('stypy_function_name', 'Foo.func2')
            Foo.func2.__dict__.__setitem__('stypy_param_names_list', [])
            Foo.func2.__dict__.__setitem__('stypy_varargs_param_name', None)
            Foo.func2.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Foo.func2.__dict__.__setitem__('stypy_call_defaults', defaults)
            Foo.func2.__dict__.__setitem__('stypy_call_varargs', varargs)
            Foo.func2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Foo.func2.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.func2', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'func2', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'func2(...)' code ##################

            str_115612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 12), 'str', 'Something else.')
            
            # ################# End of 'func2(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'func2' in the type store
            # Getting the type of 'stypy_return_type' (line 107)
            stypy_return_type_115613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_115613)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'func2'
            return stypy_return_type_115613


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 102, 4, False)
            # Assigning a type to the variable 'self' (line 103)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.__init__', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return

            # Initialize method data
            init_call_information(module_type_store, '__init__', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__init__(...)' code ##################

            pass
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()

    
    # Assigning a type to the variable 'Foo' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'Foo', Foo)
    # Declaration of the 'Bar' class
    # Getting the type of 'Foo' (line 110)
    Foo_115614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 14), 'Foo')

    class Bar(Foo_115614, ):

        @norecursion
        def func(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'func'
            module_type_store = module_type_store.open_function_context('func', 111, 8, False)
            # Assigning a type to the variable 'self' (line 112)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Bar.func.__dict__.__setitem__('stypy_localization', localization)
            Bar.func.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Bar.func.__dict__.__setitem__('stypy_type_store', module_type_store)
            Bar.func.__dict__.__setitem__('stypy_function_name', 'Bar.func')
            Bar.func.__dict__.__setitem__('stypy_param_names_list', [])
            Bar.func.__dict__.__setitem__('stypy_varargs_param_name', None)
            Bar.func.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Bar.func.__dict__.__setitem__('stypy_call_defaults', defaults)
            Bar.func.__dict__.__setitem__('stypy_call_varargs', varargs)
            Bar.func.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Bar.func.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Bar.func', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'func', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'func(...)' code ##################

            str_115615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 12), 'str', '%(super)sABC')
            # Assigning a type to the variable 'stypy_return_type' (line 114)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'stypy_return_type', types.NoneType)
            
            # ################# End of 'func(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'func' in the type store
            # Getting the type of 'stypy_return_type' (line 111)
            stypy_return_type_115616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_115616)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'func'
            return stypy_return_type_115616


        @norecursion
        def func2(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'func2'
            module_type_store = module_type_store.open_function_context('func2', 116, 8, False)
            # Assigning a type to the variable 'self' (line 117)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Bar.func2.__dict__.__setitem__('stypy_localization', localization)
            Bar.func2.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Bar.func2.__dict__.__setitem__('stypy_type_store', module_type_store)
            Bar.func2.__dict__.__setitem__('stypy_function_name', 'Bar.func2')
            Bar.func2.__dict__.__setitem__('stypy_param_names_list', [])
            Bar.func2.__dict__.__setitem__('stypy_varargs_param_name', None)
            Bar.func2.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Bar.func2.__dict__.__setitem__('stypy_call_defaults', defaults)
            Bar.func2.__dict__.__setitem__('stypy_call_varargs', varargs)
            Bar.func2.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Bar.func2.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Bar.func2', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'func2', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'func2(...)' code ##################

            # Assigning a type to the variable 'stypy_return_type' (line 119)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'stypy_return_type', types.NoneType)
            
            # ################# End of 'func2(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'func2' in the type store
            # Getting the type of 'stypy_return_type' (line 116)
            stypy_return_type_115617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_115617)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'func2'
            return stypy_return_type_115617


        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 110, 4, False)
            # Assigning a type to the variable 'self' (line 111)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Bar.__init__', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return

            # Initialize method data
            init_call_information(module_type_store, '__init__', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '__init__(...)' code ##################

            pass
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()

    
    # Assigning a type to the variable 'Bar' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'Bar', Bar)
    
    # Call to assert_equal(...): (line 121)
    # Processing the call arguments (line 121)
    # Getting the type of 'Bar' (line 121)
    Bar_115619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 17), 'Bar', False)
    # Obtaining the member 'func' of a type (line 121)
    func_115620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 17), Bar_115619, 'func')
    # Obtaining the member '__doc__' of a type (line 121)
    doc___115621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 17), func_115620, '__doc__')
    # Getting the type of 'Foo' (line 121)
    Foo_115622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 35), 'Foo', False)
    # Obtaining the member 'func' of a type (line 121)
    func_115623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 35), Foo_115622, 'func')
    # Obtaining the member '__doc__' of a type (line 121)
    doc___115624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 35), func_115623, '__doc__')
    str_115625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 54), 'str', 'ABC')
    # Applying the binary operator '+' (line 121)
    result_add_115626 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 35), '+', doc___115624, str_115625)
    
    # Processing the call keyword arguments (line 121)
    kwargs_115627 = {}
    # Getting the type of 'assert_equal' (line 121)
    assert_equal_115618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 121)
    assert_equal_call_result_115628 = invoke(stypy.reporting.localization.Localization(__file__, 121, 4), assert_equal_115618, *[doc___115621, result_add_115626], **kwargs_115627)
    
    
    # Call to assert_equal(...): (line 122)
    # Processing the call arguments (line 122)
    # Getting the type of 'Bar' (line 122)
    Bar_115630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 17), 'Bar', False)
    # Obtaining the member 'func2' of a type (line 122)
    func2_115631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 17), Bar_115630, 'func2')
    # Obtaining the member '__doc__' of a type (line 122)
    doc___115632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 17), func2_115631, '__doc__')
    # Getting the type of 'Foo' (line 122)
    Foo_115633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 36), 'Foo', False)
    # Obtaining the member 'func2' of a type (line 122)
    func2_115634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 36), Foo_115633, 'func2')
    # Obtaining the member '__doc__' of a type (line 122)
    doc___115635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 36), func2_115634, '__doc__')
    # Processing the call keyword arguments (line 122)
    kwargs_115636 = {}
    # Getting the type of 'assert_equal' (line 122)
    assert_equal_115629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 122)
    assert_equal_call_result_115637 = invoke(stypy.reporting.localization.Localization(__file__, 122, 4), assert_equal_115629, *[doc___115632, doc___115635], **kwargs_115636)
    
    
    # Assigning a Call to a Name (line 123):
    
    # Call to Bar(...): (line 123)
    # Processing the call keyword arguments (line 123)
    kwargs_115639 = {}
    # Getting the type of 'Bar' (line 123)
    Bar_115638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 10), 'Bar', False)
    # Calling Bar(args, kwargs) (line 123)
    Bar_call_result_115640 = invoke(stypy.reporting.localization.Localization(__file__, 123, 10), Bar_115638, *[], **kwargs_115639)
    
    # Assigning a type to the variable 'bar' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'bar', Bar_call_result_115640)
    
    # Call to assert_equal(...): (line 124)
    # Processing the call arguments (line 124)
    # Getting the type of 'bar' (line 124)
    bar_115642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 17), 'bar', False)
    # Obtaining the member 'func' of a type (line 124)
    func_115643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 17), bar_115642, 'func')
    # Obtaining the member '__doc__' of a type (line 124)
    doc___115644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 17), func_115643, '__doc__')
    # Getting the type of 'Foo' (line 124)
    Foo_115645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 35), 'Foo', False)
    # Obtaining the member 'func' of a type (line 124)
    func_115646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 35), Foo_115645, 'func')
    # Obtaining the member '__doc__' of a type (line 124)
    doc___115647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 35), func_115646, '__doc__')
    str_115648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 54), 'str', 'ABC')
    # Applying the binary operator '+' (line 124)
    result_add_115649 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 35), '+', doc___115647, str_115648)
    
    # Processing the call keyword arguments (line 124)
    kwargs_115650 = {}
    # Getting the type of 'assert_equal' (line 124)
    assert_equal_115641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 124)
    assert_equal_call_result_115651 = invoke(stypy.reporting.localization.Localization(__file__, 124, 4), assert_equal_115641, *[doc___115644, result_add_115649], **kwargs_115650)
    
    
    # Call to assert_equal(...): (line 125)
    # Processing the call arguments (line 125)
    # Getting the type of 'bar' (line 125)
    bar_115653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 17), 'bar', False)
    # Obtaining the member 'func2' of a type (line 125)
    func2_115654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 17), bar_115653, 'func2')
    # Obtaining the member '__doc__' of a type (line 125)
    doc___115655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 17), func2_115654, '__doc__')
    # Getting the type of 'Foo' (line 125)
    Foo_115656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 36), 'Foo', False)
    # Obtaining the member 'func2' of a type (line 125)
    func2_115657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 36), Foo_115656, 'func2')
    # Obtaining the member '__doc__' of a type (line 125)
    doc___115658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 36), func2_115657, '__doc__')
    # Processing the call keyword arguments (line 125)
    kwargs_115659 = {}
    # Getting the type of 'assert_equal' (line 125)
    assert_equal_115652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 125)
    assert_equal_call_result_115660 = invoke(stypy.reporting.localization.Localization(__file__, 125, 4), assert_equal_115652, *[doc___115655, doc___115658], **kwargs_115659)
    
    
    # ################# End of 'test_inherit_docstring_from(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_inherit_docstring_from' in the type store
    # Getting the type of 'stypy_return_type' (line 99)
    stypy_return_type_115661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_115661)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_inherit_docstring_from'
    return stypy_return_type_115661

# Assigning a type to the variable 'test_inherit_docstring_from' (line 99)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 0), 'test_inherit_docstring_from', test_inherit_docstring_from)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
