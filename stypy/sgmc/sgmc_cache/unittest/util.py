
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Various utility functions.'''
2: from collections import namedtuple, OrderedDict
3: 
4: 
5: __unittest = True
6: 
7: _MAX_LENGTH = 80
8: def safe_repr(obj, short=False):
9:     try:
10:         result = repr(obj)
11:     except Exception:
12:         result = object.__repr__(obj)
13:     if not short or len(result) < _MAX_LENGTH:
14:         return result
15:     return result[:_MAX_LENGTH] + ' [truncated]...'
16: 
17: 
18: def strclass(cls):
19:     return "%s.%s" % (cls.__module__, cls.__name__)
20: 
21: def sorted_list_difference(expected, actual):
22:     '''Finds elements in only one or the other of two, sorted input lists.
23: 
24:     Returns a two-element tuple of lists.    The first list contains those
25:     elements in the "expected" list but not in the "actual" list, and the
26:     second contains those elements in the "actual" list but not in the
27:     "expected" list.    Duplicate elements in either input list are ignored.
28:     '''
29:     i = j = 0
30:     missing = []
31:     unexpected = []
32:     while True:
33:         try:
34:             e = expected[i]
35:             a = actual[j]
36:             if e < a:
37:                 missing.append(e)
38:                 i += 1
39:                 while expected[i] == e:
40:                     i += 1
41:             elif e > a:
42:                 unexpected.append(a)
43:                 j += 1
44:                 while actual[j] == a:
45:                     j += 1
46:             else:
47:                 i += 1
48:                 try:
49:                     while expected[i] == e:
50:                         i += 1
51:                 finally:
52:                     j += 1
53:                     while actual[j] == a:
54:                         j += 1
55:         except IndexError:
56:             missing.extend(expected[i:])
57:             unexpected.extend(actual[j:])
58:             break
59:     return missing, unexpected
60: 
61: 
62: def unorderable_list_difference(expected, actual, ignore_duplicate=False):
63:     '''Same behavior as sorted_list_difference but
64:     for lists of unorderable items (like dicts).
65: 
66:     As it does a linear search per item (remove) it
67:     has O(n*n) performance.
68:     '''
69:     missing = []
70:     unexpected = []
71:     while expected:
72:         item = expected.pop()
73:         try:
74:             actual.remove(item)
75:         except ValueError:
76:             missing.append(item)
77:         if ignore_duplicate:
78:             for lst in expected, actual:
79:                 try:
80:                     while True:
81:                         lst.remove(item)
82:                 except ValueError:
83:                     pass
84:     if ignore_duplicate:
85:         while actual:
86:             item = actual.pop()
87:             unexpected.append(item)
88:             try:
89:                 while True:
90:                     actual.remove(item)
91:             except ValueError:
92:                 pass
93:         return missing, unexpected
94: 
95:     # anything left in actual is unexpected
96:     return missing, actual
97: 
98: _Mismatch = namedtuple('Mismatch', 'actual expected value')
99: 
100: def _count_diff_all_purpose(actual, expected):
101:     'Returns list of (cnt_act, cnt_exp, elem) triples where the counts differ'
102:     # elements need not be hashable
103:     s, t = list(actual), list(expected)
104:     m, n = len(s), len(t)
105:     NULL = object()
106:     result = []
107:     for i, elem in enumerate(s):
108:         if elem is NULL:
109:             continue
110:         cnt_s = cnt_t = 0
111:         for j in range(i, m):
112:             if s[j] == elem:
113:                 cnt_s += 1
114:                 s[j] = NULL
115:         for j, other_elem in enumerate(t):
116:             if other_elem == elem:
117:                 cnt_t += 1
118:                 t[j] = NULL
119:         if cnt_s != cnt_t:
120:             diff = _Mismatch(cnt_s, cnt_t, elem)
121:             result.append(diff)
122: 
123:     for i, elem in enumerate(t):
124:         if elem is NULL:
125:             continue
126:         cnt_t = 0
127:         for j in range(i, n):
128:             if t[j] == elem:
129:                 cnt_t += 1
130:                 t[j] = NULL
131:         diff = _Mismatch(0, cnt_t, elem)
132:         result.append(diff)
133:     return result
134: 
135: def _ordered_count(iterable):
136:     'Return dict of element counts, in the order they were first seen'
137:     c = OrderedDict()
138:     for elem in iterable:
139:         c[elem] = c.get(elem, 0) + 1
140:     return c
141: 
142: def _count_diff_hashable(actual, expected):
143:     'Returns list of (cnt_act, cnt_exp, elem) triples where the counts differ'
144:     # elements must be hashable
145:     s, t = _ordered_count(actual), _ordered_count(expected)
146:     result = []
147:     for elem, cnt_s in s.items():
148:         cnt_t = t.get(elem, 0)
149:         if cnt_s != cnt_t:
150:             diff = _Mismatch(cnt_s, cnt_t, elem)
151:             result.append(diff)
152:     for elem, cnt_t in t.items():
153:         if elem not in s:
154:             diff = _Mismatch(0, cnt_t, elem)
155:             result.append(diff)
156:     return result
157: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_192638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Various utility functions.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'from collections import namedtuple, OrderedDict' statement (line 2)
from collections import namedtuple, OrderedDict

import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'collections', None, module_type_store, ['namedtuple', 'OrderedDict'], [namedtuple, OrderedDict])


# Assigning a Name to a Name (line 5):

# Assigning a Name to a Name (line 5):
# Getting the type of 'True' (line 5)
True_192639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 13), 'True')
# Assigning a type to the variable '__unittest' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), '__unittest', True_192639)

# Assigning a Num to a Name (line 7):

# Assigning a Num to a Name (line 7):
int_192640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 14), 'int')
# Assigning a type to the variable '_MAX_LENGTH' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '_MAX_LENGTH', int_192640)

@norecursion
def safe_repr(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 8)
    False_192641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 25), 'False')
    defaults = [False_192641]
    # Create a new context for function 'safe_repr'
    module_type_store = module_type_store.open_function_context('safe_repr', 8, 0, False)
    
    # Passed parameters checking function
    safe_repr.stypy_localization = localization
    safe_repr.stypy_type_of_self = None
    safe_repr.stypy_type_store = module_type_store
    safe_repr.stypy_function_name = 'safe_repr'
    safe_repr.stypy_param_names_list = ['obj', 'short']
    safe_repr.stypy_varargs_param_name = None
    safe_repr.stypy_kwargs_param_name = None
    safe_repr.stypy_call_defaults = defaults
    safe_repr.stypy_call_varargs = varargs
    safe_repr.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'safe_repr', ['obj', 'short'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'safe_repr', localization, ['obj', 'short'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'safe_repr(...)' code ##################

    
    
    # SSA begins for try-except statement (line 9)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 10):
    
    # Assigning a Call to a Name (line 10):
    
    # Call to repr(...): (line 10)
    # Processing the call arguments (line 10)
    # Getting the type of 'obj' (line 10)
    obj_192643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 22), 'obj', False)
    # Processing the call keyword arguments (line 10)
    kwargs_192644 = {}
    # Getting the type of 'repr' (line 10)
    repr_192642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 17), 'repr', False)
    # Calling repr(args, kwargs) (line 10)
    repr_call_result_192645 = invoke(stypy.reporting.localization.Localization(__file__, 10, 17), repr_192642, *[obj_192643], **kwargs_192644)
    
    # Assigning a type to the variable 'result' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'result', repr_call_result_192645)
    # SSA branch for the except part of a try statement (line 9)
    # SSA branch for the except 'Exception' branch of a try statement (line 9)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Call to a Name (line 12):
    
    # Assigning a Call to a Name (line 12):
    
    # Call to __repr__(...): (line 12)
    # Processing the call arguments (line 12)
    # Getting the type of 'obj' (line 12)
    obj_192648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 33), 'obj', False)
    # Processing the call keyword arguments (line 12)
    kwargs_192649 = {}
    # Getting the type of 'object' (line 12)
    object_192646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 17), 'object', False)
    # Obtaining the member '__repr__' of a type (line 12)
    repr___192647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 17), object_192646, '__repr__')
    # Calling __repr__(args, kwargs) (line 12)
    repr___call_result_192650 = invoke(stypy.reporting.localization.Localization(__file__, 12, 17), repr___192647, *[obj_192648], **kwargs_192649)
    
    # Assigning a type to the variable 'result' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'result', repr___call_result_192650)
    # SSA join for try-except statement (line 9)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'short' (line 13)
    short_192651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 11), 'short')
    # Applying the 'not' unary operator (line 13)
    result_not__192652 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 7), 'not', short_192651)
    
    
    
    # Call to len(...): (line 13)
    # Processing the call arguments (line 13)
    # Getting the type of 'result' (line 13)
    result_192654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 24), 'result', False)
    # Processing the call keyword arguments (line 13)
    kwargs_192655 = {}
    # Getting the type of 'len' (line 13)
    len_192653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 20), 'len', False)
    # Calling len(args, kwargs) (line 13)
    len_call_result_192656 = invoke(stypy.reporting.localization.Localization(__file__, 13, 20), len_192653, *[result_192654], **kwargs_192655)
    
    # Getting the type of '_MAX_LENGTH' (line 13)
    _MAX_LENGTH_192657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 34), '_MAX_LENGTH')
    # Applying the binary operator '<' (line 13)
    result_lt_192658 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 20), '<', len_call_result_192656, _MAX_LENGTH_192657)
    
    # Applying the binary operator 'or' (line 13)
    result_or_keyword_192659 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 7), 'or', result_not__192652, result_lt_192658)
    
    # Testing the type of an if condition (line 13)
    if_condition_192660 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 13, 4), result_or_keyword_192659)
    # Assigning a type to the variable 'if_condition_192660' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'if_condition_192660', if_condition_192660)
    # SSA begins for if statement (line 13)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'result' (line 14)
    result_192661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 15), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'stypy_return_type', result_192661)
    # SSA join for if statement (line 13)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining the type of the subscript
    # Getting the type of '_MAX_LENGTH' (line 15)
    _MAX_LENGTH_192662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 19), '_MAX_LENGTH')
    slice_192663 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 15, 11), None, _MAX_LENGTH_192662, None)
    # Getting the type of 'result' (line 15)
    result_192664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 11), 'result')
    # Obtaining the member '__getitem__' of a type (line 15)
    getitem___192665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 11), result_192664, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 15)
    subscript_call_result_192666 = invoke(stypy.reporting.localization.Localization(__file__, 15, 11), getitem___192665, slice_192663)
    
    str_192667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 34), 'str', ' [truncated]...')
    # Applying the binary operator '+' (line 15)
    result_add_192668 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 11), '+', subscript_call_result_192666, str_192667)
    
    # Assigning a type to the variable 'stypy_return_type' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'stypy_return_type', result_add_192668)
    
    # ################# End of 'safe_repr(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'safe_repr' in the type store
    # Getting the type of 'stypy_return_type' (line 8)
    stypy_return_type_192669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_192669)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'safe_repr'
    return stypy_return_type_192669

# Assigning a type to the variable 'safe_repr' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'safe_repr', safe_repr)

@norecursion
def strclass(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'strclass'
    module_type_store = module_type_store.open_function_context('strclass', 18, 0, False)
    
    # Passed parameters checking function
    strclass.stypy_localization = localization
    strclass.stypy_type_of_self = None
    strclass.stypy_type_store = module_type_store
    strclass.stypy_function_name = 'strclass'
    strclass.stypy_param_names_list = ['cls']
    strclass.stypy_varargs_param_name = None
    strclass.stypy_kwargs_param_name = None
    strclass.stypy_call_defaults = defaults
    strclass.stypy_call_varargs = varargs
    strclass.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'strclass', ['cls'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'strclass', localization, ['cls'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'strclass(...)' code ##################

    str_192670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 11), 'str', '%s.%s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 19)
    tuple_192671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 19)
    # Adding element type (line 19)
    # Getting the type of 'cls' (line 19)
    cls_192672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 22), 'cls')
    # Obtaining the member '__module__' of a type (line 19)
    module___192673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 22), cls_192672, '__module__')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 22), tuple_192671, module___192673)
    # Adding element type (line 19)
    # Getting the type of 'cls' (line 19)
    cls_192674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 38), 'cls')
    # Obtaining the member '__name__' of a type (line 19)
    name___192675 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 38), cls_192674, '__name__')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 22), tuple_192671, name___192675)
    
    # Applying the binary operator '%' (line 19)
    result_mod_192676 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 11), '%', str_192670, tuple_192671)
    
    # Assigning a type to the variable 'stypy_return_type' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'stypy_return_type', result_mod_192676)
    
    # ################# End of 'strclass(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'strclass' in the type store
    # Getting the type of 'stypy_return_type' (line 18)
    stypy_return_type_192677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_192677)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'strclass'
    return stypy_return_type_192677

# Assigning a type to the variable 'strclass' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'strclass', strclass)

@norecursion
def sorted_list_difference(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'sorted_list_difference'
    module_type_store = module_type_store.open_function_context('sorted_list_difference', 21, 0, False)
    
    # Passed parameters checking function
    sorted_list_difference.stypy_localization = localization
    sorted_list_difference.stypy_type_of_self = None
    sorted_list_difference.stypy_type_store = module_type_store
    sorted_list_difference.stypy_function_name = 'sorted_list_difference'
    sorted_list_difference.stypy_param_names_list = ['expected', 'actual']
    sorted_list_difference.stypy_varargs_param_name = None
    sorted_list_difference.stypy_kwargs_param_name = None
    sorted_list_difference.stypy_call_defaults = defaults
    sorted_list_difference.stypy_call_varargs = varargs
    sorted_list_difference.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'sorted_list_difference', ['expected', 'actual'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'sorted_list_difference', localization, ['expected', 'actual'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'sorted_list_difference(...)' code ##################

    str_192678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, (-1)), 'str', 'Finds elements in only one or the other of two, sorted input lists.\n\n    Returns a two-element tuple of lists.    The first list contains those\n    elements in the "expected" list but not in the "actual" list, and the\n    second contains those elements in the "actual" list but not in the\n    "expected" list.    Duplicate elements in either input list are ignored.\n    ')
    
    # Multiple assignment of 2 elements.
    
    # Assigning a Num to a Name (line 29):
    int_192679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 12), 'int')
    # Assigning a type to the variable 'j' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'j', int_192679)
    
    # Assigning a Name to a Name (line 29):
    # Getting the type of 'j' (line 29)
    j_192680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'j')
    # Assigning a type to the variable 'i' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'i', j_192680)
    
    # Assigning a List to a Name (line 30):
    
    # Assigning a List to a Name (line 30):
    
    # Obtaining an instance of the builtin type 'list' (line 30)
    list_192681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 30)
    
    # Assigning a type to the variable 'missing' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'missing', list_192681)
    
    # Assigning a List to a Name (line 31):
    
    # Assigning a List to a Name (line 31):
    
    # Obtaining an instance of the builtin type 'list' (line 31)
    list_192682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 31)
    
    # Assigning a type to the variable 'unexpected' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'unexpected', list_192682)
    
    # Getting the type of 'True' (line 32)
    True_192683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 10), 'True')
    # Testing the type of an if condition (line 32)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 32, 4), True_192683)
    # SSA begins for while statement (line 32)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    
    # SSA begins for try-except statement (line 33)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 34):
    
    # Assigning a Subscript to a Name (line 34):
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 34)
    i_192684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 25), 'i')
    # Getting the type of 'expected' (line 34)
    expected_192685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 16), 'expected')
    # Obtaining the member '__getitem__' of a type (line 34)
    getitem___192686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 16), expected_192685, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 34)
    subscript_call_result_192687 = invoke(stypy.reporting.localization.Localization(__file__, 34, 16), getitem___192686, i_192684)
    
    # Assigning a type to the variable 'e' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'e', subscript_call_result_192687)
    
    # Assigning a Subscript to a Name (line 35):
    
    # Assigning a Subscript to a Name (line 35):
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 35)
    j_192688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 23), 'j')
    # Getting the type of 'actual' (line 35)
    actual_192689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 16), 'actual')
    # Obtaining the member '__getitem__' of a type (line 35)
    getitem___192690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 16), actual_192689, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 35)
    subscript_call_result_192691 = invoke(stypy.reporting.localization.Localization(__file__, 35, 16), getitem___192690, j_192688)
    
    # Assigning a type to the variable 'a' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'a', subscript_call_result_192691)
    
    
    # Getting the type of 'e' (line 36)
    e_192692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 15), 'e')
    # Getting the type of 'a' (line 36)
    a_192693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 19), 'a')
    # Applying the binary operator '<' (line 36)
    result_lt_192694 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 15), '<', e_192692, a_192693)
    
    # Testing the type of an if condition (line 36)
    if_condition_192695 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 36, 12), result_lt_192694)
    # Assigning a type to the variable 'if_condition_192695' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'if_condition_192695', if_condition_192695)
    # SSA begins for if statement (line 36)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'e' (line 37)
    e_192698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 31), 'e', False)
    # Processing the call keyword arguments (line 37)
    kwargs_192699 = {}
    # Getting the type of 'missing' (line 37)
    missing_192696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 16), 'missing', False)
    # Obtaining the member 'append' of a type (line 37)
    append_192697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 16), missing_192696, 'append')
    # Calling append(args, kwargs) (line 37)
    append_call_result_192700 = invoke(stypy.reporting.localization.Localization(__file__, 37, 16), append_192697, *[e_192698], **kwargs_192699)
    
    
    # Getting the type of 'i' (line 38)
    i_192701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'i')
    int_192702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 21), 'int')
    # Applying the binary operator '+=' (line 38)
    result_iadd_192703 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 16), '+=', i_192701, int_192702)
    # Assigning a type to the variable 'i' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'i', result_iadd_192703)
    
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 39)
    i_192704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 31), 'i')
    # Getting the type of 'expected' (line 39)
    expected_192705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 22), 'expected')
    # Obtaining the member '__getitem__' of a type (line 39)
    getitem___192706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 22), expected_192705, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 39)
    subscript_call_result_192707 = invoke(stypy.reporting.localization.Localization(__file__, 39, 22), getitem___192706, i_192704)
    
    # Getting the type of 'e' (line 39)
    e_192708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 37), 'e')
    # Applying the binary operator '==' (line 39)
    result_eq_192709 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 22), '==', subscript_call_result_192707, e_192708)
    
    # Testing the type of an if condition (line 39)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 39, 16), result_eq_192709)
    # SSA begins for while statement (line 39)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Getting the type of 'i' (line 40)
    i_192710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 20), 'i')
    int_192711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 25), 'int')
    # Applying the binary operator '+=' (line 40)
    result_iadd_192712 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 20), '+=', i_192710, int_192711)
    # Assigning a type to the variable 'i' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 20), 'i', result_iadd_192712)
    
    # SSA join for while statement (line 39)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 36)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'e' (line 41)
    e_192713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 17), 'e')
    # Getting the type of 'a' (line 41)
    a_192714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 21), 'a')
    # Applying the binary operator '>' (line 41)
    result_gt_192715 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 17), '>', e_192713, a_192714)
    
    # Testing the type of an if condition (line 41)
    if_condition_192716 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 17), result_gt_192715)
    # Assigning a type to the variable 'if_condition_192716' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 17), 'if_condition_192716', if_condition_192716)
    # SSA begins for if statement (line 41)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'a' (line 42)
    a_192719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 34), 'a', False)
    # Processing the call keyword arguments (line 42)
    kwargs_192720 = {}
    # Getting the type of 'unexpected' (line 42)
    unexpected_192717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 16), 'unexpected', False)
    # Obtaining the member 'append' of a type (line 42)
    append_192718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 16), unexpected_192717, 'append')
    # Calling append(args, kwargs) (line 42)
    append_call_result_192721 = invoke(stypy.reporting.localization.Localization(__file__, 42, 16), append_192718, *[a_192719], **kwargs_192720)
    
    
    # Getting the type of 'j' (line 43)
    j_192722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 16), 'j')
    int_192723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 21), 'int')
    # Applying the binary operator '+=' (line 43)
    result_iadd_192724 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 16), '+=', j_192722, int_192723)
    # Assigning a type to the variable 'j' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 16), 'j', result_iadd_192724)
    
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 44)
    j_192725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 29), 'j')
    # Getting the type of 'actual' (line 44)
    actual_192726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 22), 'actual')
    # Obtaining the member '__getitem__' of a type (line 44)
    getitem___192727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 22), actual_192726, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 44)
    subscript_call_result_192728 = invoke(stypy.reporting.localization.Localization(__file__, 44, 22), getitem___192727, j_192725)
    
    # Getting the type of 'a' (line 44)
    a_192729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 35), 'a')
    # Applying the binary operator '==' (line 44)
    result_eq_192730 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 22), '==', subscript_call_result_192728, a_192729)
    
    # Testing the type of an if condition (line 44)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 44, 16), result_eq_192730)
    # SSA begins for while statement (line 44)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Getting the type of 'j' (line 45)
    j_192731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 20), 'j')
    int_192732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 25), 'int')
    # Applying the binary operator '+=' (line 45)
    result_iadd_192733 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 20), '+=', j_192731, int_192732)
    # Assigning a type to the variable 'j' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 20), 'j', result_iadd_192733)
    
    # SSA join for while statement (line 44)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 41)
    module_type_store.open_ssa_branch('else')
    
    # Getting the type of 'i' (line 47)
    i_192734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 16), 'i')
    int_192735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 21), 'int')
    # Applying the binary operator '+=' (line 47)
    result_iadd_192736 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 16), '+=', i_192734, int_192735)
    # Assigning a type to the variable 'i' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 16), 'i', result_iadd_192736)
    
    
    # Try-finally block (line 48)
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 49)
    i_192737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 35), 'i')
    # Getting the type of 'expected' (line 49)
    expected_192738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 26), 'expected')
    # Obtaining the member '__getitem__' of a type (line 49)
    getitem___192739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 26), expected_192738, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 49)
    subscript_call_result_192740 = invoke(stypy.reporting.localization.Localization(__file__, 49, 26), getitem___192739, i_192737)
    
    # Getting the type of 'e' (line 49)
    e_192741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 41), 'e')
    # Applying the binary operator '==' (line 49)
    result_eq_192742 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 26), '==', subscript_call_result_192740, e_192741)
    
    # Testing the type of an if condition (line 49)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 20), result_eq_192742)
    # SSA begins for while statement (line 49)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Getting the type of 'i' (line 50)
    i_192743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 24), 'i')
    int_192744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 29), 'int')
    # Applying the binary operator '+=' (line 50)
    result_iadd_192745 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 24), '+=', i_192743, int_192744)
    # Assigning a type to the variable 'i' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 24), 'i', result_iadd_192745)
    
    # SSA join for while statement (line 49)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # finally branch of the try-finally block (line 48)
    
    # Getting the type of 'j' (line 52)
    j_192746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 20), 'j')
    int_192747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 25), 'int')
    # Applying the binary operator '+=' (line 52)
    result_iadd_192748 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 20), '+=', j_192746, int_192747)
    # Assigning a type to the variable 'j' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 20), 'j', result_iadd_192748)
    
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 53)
    j_192749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 33), 'j')
    # Getting the type of 'actual' (line 53)
    actual_192750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 26), 'actual')
    # Obtaining the member '__getitem__' of a type (line 53)
    getitem___192751 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 26), actual_192750, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 53)
    subscript_call_result_192752 = invoke(stypy.reporting.localization.Localization(__file__, 53, 26), getitem___192751, j_192749)
    
    # Getting the type of 'a' (line 53)
    a_192753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 39), 'a')
    # Applying the binary operator '==' (line 53)
    result_eq_192754 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 26), '==', subscript_call_result_192752, a_192753)
    
    # Testing the type of an if condition (line 53)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 53, 20), result_eq_192754)
    # SSA begins for while statement (line 53)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Getting the type of 'j' (line 54)
    j_192755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 24), 'j')
    int_192756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 29), 'int')
    # Applying the binary operator '+=' (line 54)
    result_iadd_192757 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 24), '+=', j_192755, int_192756)
    # Assigning a type to the variable 'j' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 24), 'j', result_iadd_192757)
    
    # SSA join for while statement (line 53)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # SSA join for if statement (line 41)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 36)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the except part of a try statement (line 33)
    # SSA branch for the except 'IndexError' branch of a try statement (line 33)
    module_type_store.open_ssa_branch('except')
    
    # Call to extend(...): (line 56)
    # Processing the call arguments (line 56)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 56)
    i_192760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 36), 'i', False)
    slice_192761 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 56, 27), i_192760, None, None)
    # Getting the type of 'expected' (line 56)
    expected_192762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 27), 'expected', False)
    # Obtaining the member '__getitem__' of a type (line 56)
    getitem___192763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 27), expected_192762, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 56)
    subscript_call_result_192764 = invoke(stypy.reporting.localization.Localization(__file__, 56, 27), getitem___192763, slice_192761)
    
    # Processing the call keyword arguments (line 56)
    kwargs_192765 = {}
    # Getting the type of 'missing' (line 56)
    missing_192758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'missing', False)
    # Obtaining the member 'extend' of a type (line 56)
    extend_192759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 12), missing_192758, 'extend')
    # Calling extend(args, kwargs) (line 56)
    extend_call_result_192766 = invoke(stypy.reporting.localization.Localization(__file__, 56, 12), extend_192759, *[subscript_call_result_192764], **kwargs_192765)
    
    
    # Call to extend(...): (line 57)
    # Processing the call arguments (line 57)
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 57)
    j_192769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 37), 'j', False)
    slice_192770 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 57, 30), j_192769, None, None)
    # Getting the type of 'actual' (line 57)
    actual_192771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 30), 'actual', False)
    # Obtaining the member '__getitem__' of a type (line 57)
    getitem___192772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 30), actual_192771, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 57)
    subscript_call_result_192773 = invoke(stypy.reporting.localization.Localization(__file__, 57, 30), getitem___192772, slice_192770)
    
    # Processing the call keyword arguments (line 57)
    kwargs_192774 = {}
    # Getting the type of 'unexpected' (line 57)
    unexpected_192767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'unexpected', False)
    # Obtaining the member 'extend' of a type (line 57)
    extend_192768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 12), unexpected_192767, 'extend')
    # Calling extend(args, kwargs) (line 57)
    extend_call_result_192775 = invoke(stypy.reporting.localization.Localization(__file__, 57, 12), extend_192768, *[subscript_call_result_192773], **kwargs_192774)
    
    # SSA join for try-except statement (line 33)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 32)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 59)
    tuple_192776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 59)
    # Adding element type (line 59)
    # Getting the type of 'missing' (line 59)
    missing_192777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 11), 'missing')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 11), tuple_192776, missing_192777)
    # Adding element type (line 59)
    # Getting the type of 'unexpected' (line 59)
    unexpected_192778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 20), 'unexpected')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 11), tuple_192776, unexpected_192778)
    
    # Assigning a type to the variable 'stypy_return_type' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'stypy_return_type', tuple_192776)
    
    # ################# End of 'sorted_list_difference(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sorted_list_difference' in the type store
    # Getting the type of 'stypy_return_type' (line 21)
    stypy_return_type_192779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_192779)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sorted_list_difference'
    return stypy_return_type_192779

# Assigning a type to the variable 'sorted_list_difference' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'sorted_list_difference', sorted_list_difference)

@norecursion
def unorderable_list_difference(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 62)
    False_192780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 67), 'False')
    defaults = [False_192780]
    # Create a new context for function 'unorderable_list_difference'
    module_type_store = module_type_store.open_function_context('unorderable_list_difference', 62, 0, False)
    
    # Passed parameters checking function
    unorderable_list_difference.stypy_localization = localization
    unorderable_list_difference.stypy_type_of_self = None
    unorderable_list_difference.stypy_type_store = module_type_store
    unorderable_list_difference.stypy_function_name = 'unorderable_list_difference'
    unorderable_list_difference.stypy_param_names_list = ['expected', 'actual', 'ignore_duplicate']
    unorderable_list_difference.stypy_varargs_param_name = None
    unorderable_list_difference.stypy_kwargs_param_name = None
    unorderable_list_difference.stypy_call_defaults = defaults
    unorderable_list_difference.stypy_call_varargs = varargs
    unorderable_list_difference.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'unorderable_list_difference', ['expected', 'actual', 'ignore_duplicate'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'unorderable_list_difference', localization, ['expected', 'actual', 'ignore_duplicate'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'unorderable_list_difference(...)' code ##################

    str_192781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, (-1)), 'str', 'Same behavior as sorted_list_difference but\n    for lists of unorderable items (like dicts).\n\n    As it does a linear search per item (remove) it\n    has O(n*n) performance.\n    ')
    
    # Assigning a List to a Name (line 69):
    
    # Assigning a List to a Name (line 69):
    
    # Obtaining an instance of the builtin type 'list' (line 69)
    list_192782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 69)
    
    # Assigning a type to the variable 'missing' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'missing', list_192782)
    
    # Assigning a List to a Name (line 70):
    
    # Assigning a List to a Name (line 70):
    
    # Obtaining an instance of the builtin type 'list' (line 70)
    list_192783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 70)
    
    # Assigning a type to the variable 'unexpected' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'unexpected', list_192783)
    
    # Getting the type of 'expected' (line 71)
    expected_192784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 10), 'expected')
    # Testing the type of an if condition (line 71)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 4), expected_192784)
    # SSA begins for while statement (line 71)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Name (line 72):
    
    # Assigning a Call to a Name (line 72):
    
    # Call to pop(...): (line 72)
    # Processing the call keyword arguments (line 72)
    kwargs_192787 = {}
    # Getting the type of 'expected' (line 72)
    expected_192785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 15), 'expected', False)
    # Obtaining the member 'pop' of a type (line 72)
    pop_192786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 15), expected_192785, 'pop')
    # Calling pop(args, kwargs) (line 72)
    pop_call_result_192788 = invoke(stypy.reporting.localization.Localization(__file__, 72, 15), pop_192786, *[], **kwargs_192787)
    
    # Assigning a type to the variable 'item' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'item', pop_call_result_192788)
    
    
    # SSA begins for try-except statement (line 73)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to remove(...): (line 74)
    # Processing the call arguments (line 74)
    # Getting the type of 'item' (line 74)
    item_192791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 26), 'item', False)
    # Processing the call keyword arguments (line 74)
    kwargs_192792 = {}
    # Getting the type of 'actual' (line 74)
    actual_192789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'actual', False)
    # Obtaining the member 'remove' of a type (line 74)
    remove_192790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 12), actual_192789, 'remove')
    # Calling remove(args, kwargs) (line 74)
    remove_call_result_192793 = invoke(stypy.reporting.localization.Localization(__file__, 74, 12), remove_192790, *[item_192791], **kwargs_192792)
    
    # SSA branch for the except part of a try statement (line 73)
    # SSA branch for the except 'ValueError' branch of a try statement (line 73)
    module_type_store.open_ssa_branch('except')
    
    # Call to append(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'item' (line 76)
    item_192796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 27), 'item', False)
    # Processing the call keyword arguments (line 76)
    kwargs_192797 = {}
    # Getting the type of 'missing' (line 76)
    missing_192794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'missing', False)
    # Obtaining the member 'append' of a type (line 76)
    append_192795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 12), missing_192794, 'append')
    # Calling append(args, kwargs) (line 76)
    append_call_result_192798 = invoke(stypy.reporting.localization.Localization(__file__, 76, 12), append_192795, *[item_192796], **kwargs_192797)
    
    # SSA join for try-except statement (line 73)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'ignore_duplicate' (line 77)
    ignore_duplicate_192799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 11), 'ignore_duplicate')
    # Testing the type of an if condition (line 77)
    if_condition_192800 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 77, 8), ignore_duplicate_192799)
    # Assigning a type to the variable 'if_condition_192800' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'if_condition_192800', if_condition_192800)
    # SSA begins for if statement (line 77)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 78)
    tuple_192801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 23), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 78)
    # Adding element type (line 78)
    # Getting the type of 'expected' (line 78)
    expected_192802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'expected')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 23), tuple_192801, expected_192802)
    # Adding element type (line 78)
    # Getting the type of 'actual' (line 78)
    actual_192803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 33), 'actual')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 23), tuple_192801, actual_192803)
    
    # Testing the type of a for loop iterable (line 78)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 78, 12), tuple_192801)
    # Getting the type of the for loop variable (line 78)
    for_loop_var_192804 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 78, 12), tuple_192801)
    # Assigning a type to the variable 'lst' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'lst', for_loop_var_192804)
    # SSA begins for a for statement (line 78)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # SSA begins for try-except statement (line 79)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Getting the type of 'True' (line 80)
    True_192805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 26), 'True')
    # Testing the type of an if condition (line 80)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 20), True_192805)
    # SSA begins for while statement (line 80)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Call to remove(...): (line 81)
    # Processing the call arguments (line 81)
    # Getting the type of 'item' (line 81)
    item_192808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 35), 'item', False)
    # Processing the call keyword arguments (line 81)
    kwargs_192809 = {}
    # Getting the type of 'lst' (line 81)
    lst_192806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 24), 'lst', False)
    # Obtaining the member 'remove' of a type (line 81)
    remove_192807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 24), lst_192806, 'remove')
    # Calling remove(args, kwargs) (line 81)
    remove_call_result_192810 = invoke(stypy.reporting.localization.Localization(__file__, 81, 24), remove_192807, *[item_192808], **kwargs_192809)
    
    # SSA join for while statement (line 80)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the except part of a try statement (line 79)
    # SSA branch for the except 'ValueError' branch of a try statement (line 79)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 79)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 77)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 71)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'ignore_duplicate' (line 84)
    ignore_duplicate_192811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 7), 'ignore_duplicate')
    # Testing the type of an if condition (line 84)
    if_condition_192812 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 84, 4), ignore_duplicate_192811)
    # Assigning a type to the variable 'if_condition_192812' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'if_condition_192812', if_condition_192812)
    # SSA begins for if statement (line 84)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'actual' (line 85)
    actual_192813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 14), 'actual')
    # Testing the type of an if condition (line 85)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 85, 8), actual_192813)
    # SSA begins for while statement (line 85)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Name (line 86):
    
    # Assigning a Call to a Name (line 86):
    
    # Call to pop(...): (line 86)
    # Processing the call keyword arguments (line 86)
    kwargs_192816 = {}
    # Getting the type of 'actual' (line 86)
    actual_192814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 19), 'actual', False)
    # Obtaining the member 'pop' of a type (line 86)
    pop_192815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 19), actual_192814, 'pop')
    # Calling pop(args, kwargs) (line 86)
    pop_call_result_192817 = invoke(stypy.reporting.localization.Localization(__file__, 86, 19), pop_192815, *[], **kwargs_192816)
    
    # Assigning a type to the variable 'item' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'item', pop_call_result_192817)
    
    # Call to append(...): (line 87)
    # Processing the call arguments (line 87)
    # Getting the type of 'item' (line 87)
    item_192820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 30), 'item', False)
    # Processing the call keyword arguments (line 87)
    kwargs_192821 = {}
    # Getting the type of 'unexpected' (line 87)
    unexpected_192818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'unexpected', False)
    # Obtaining the member 'append' of a type (line 87)
    append_192819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 12), unexpected_192818, 'append')
    # Calling append(args, kwargs) (line 87)
    append_call_result_192822 = invoke(stypy.reporting.localization.Localization(__file__, 87, 12), append_192819, *[item_192820], **kwargs_192821)
    
    
    
    # SSA begins for try-except statement (line 88)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Getting the type of 'True' (line 89)
    True_192823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 22), 'True')
    # Testing the type of an if condition (line 89)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 16), True_192823)
    # SSA begins for while statement (line 89)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Call to remove(...): (line 90)
    # Processing the call arguments (line 90)
    # Getting the type of 'item' (line 90)
    item_192826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 34), 'item', False)
    # Processing the call keyword arguments (line 90)
    kwargs_192827 = {}
    # Getting the type of 'actual' (line 90)
    actual_192824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 20), 'actual', False)
    # Obtaining the member 'remove' of a type (line 90)
    remove_192825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 20), actual_192824, 'remove')
    # Calling remove(args, kwargs) (line 90)
    remove_call_result_192828 = invoke(stypy.reporting.localization.Localization(__file__, 90, 20), remove_192825, *[item_192826], **kwargs_192827)
    
    # SSA join for while statement (line 89)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the except part of a try statement (line 88)
    # SSA branch for the except 'ValueError' branch of a try statement (line 88)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 88)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 85)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 93)
    tuple_192829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 93)
    # Adding element type (line 93)
    # Getting the type of 'missing' (line 93)
    missing_192830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 15), 'missing')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 15), tuple_192829, missing_192830)
    # Adding element type (line 93)
    # Getting the type of 'unexpected' (line 93)
    unexpected_192831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 24), 'unexpected')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 15), tuple_192829, unexpected_192831)
    
    # Assigning a type to the variable 'stypy_return_type' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'stypy_return_type', tuple_192829)
    # SSA join for if statement (line 84)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 96)
    tuple_192832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 96)
    # Adding element type (line 96)
    # Getting the type of 'missing' (line 96)
    missing_192833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 11), 'missing')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 11), tuple_192832, missing_192833)
    # Adding element type (line 96)
    # Getting the type of 'actual' (line 96)
    actual_192834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 20), 'actual')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 11), tuple_192832, actual_192834)
    
    # Assigning a type to the variable 'stypy_return_type' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'stypy_return_type', tuple_192832)
    
    # ################# End of 'unorderable_list_difference(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'unorderable_list_difference' in the type store
    # Getting the type of 'stypy_return_type' (line 62)
    stypy_return_type_192835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_192835)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'unorderable_list_difference'
    return stypy_return_type_192835

# Assigning a type to the variable 'unorderable_list_difference' (line 62)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 0), 'unorderable_list_difference', unorderable_list_difference)

# Assigning a Call to a Name (line 98):

# Assigning a Call to a Name (line 98):

# Call to namedtuple(...): (line 98)
# Processing the call arguments (line 98)
str_192837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 23), 'str', 'Mismatch')
str_192838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 35), 'str', 'actual expected value')
# Processing the call keyword arguments (line 98)
kwargs_192839 = {}
# Getting the type of 'namedtuple' (line 98)
namedtuple_192836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'namedtuple', False)
# Calling namedtuple(args, kwargs) (line 98)
namedtuple_call_result_192840 = invoke(stypy.reporting.localization.Localization(__file__, 98, 12), namedtuple_192836, *[str_192837, str_192838], **kwargs_192839)

# Assigning a type to the variable '_Mismatch' (line 98)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), '_Mismatch', namedtuple_call_result_192840)

@norecursion
def _count_diff_all_purpose(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_count_diff_all_purpose'
    module_type_store = module_type_store.open_function_context('_count_diff_all_purpose', 100, 0, False)
    
    # Passed parameters checking function
    _count_diff_all_purpose.stypy_localization = localization
    _count_diff_all_purpose.stypy_type_of_self = None
    _count_diff_all_purpose.stypy_type_store = module_type_store
    _count_diff_all_purpose.stypy_function_name = '_count_diff_all_purpose'
    _count_diff_all_purpose.stypy_param_names_list = ['actual', 'expected']
    _count_diff_all_purpose.stypy_varargs_param_name = None
    _count_diff_all_purpose.stypy_kwargs_param_name = None
    _count_diff_all_purpose.stypy_call_defaults = defaults
    _count_diff_all_purpose.stypy_call_varargs = varargs
    _count_diff_all_purpose.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_count_diff_all_purpose', ['actual', 'expected'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_count_diff_all_purpose', localization, ['actual', 'expected'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_count_diff_all_purpose(...)' code ##################

    str_192841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 4), 'str', 'Returns list of (cnt_act, cnt_exp, elem) triples where the counts differ')
    
    # Assigning a Tuple to a Tuple (line 103):
    
    # Assigning a Call to a Name (line 103):
    
    # Call to list(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'actual' (line 103)
    actual_192843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 16), 'actual', False)
    # Processing the call keyword arguments (line 103)
    kwargs_192844 = {}
    # Getting the type of 'list' (line 103)
    list_192842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 11), 'list', False)
    # Calling list(args, kwargs) (line 103)
    list_call_result_192845 = invoke(stypy.reporting.localization.Localization(__file__, 103, 11), list_192842, *[actual_192843], **kwargs_192844)
    
    # Assigning a type to the variable 'tuple_assignment_192632' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'tuple_assignment_192632', list_call_result_192845)
    
    # Assigning a Call to a Name (line 103):
    
    # Call to list(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'expected' (line 103)
    expected_192847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 30), 'expected', False)
    # Processing the call keyword arguments (line 103)
    kwargs_192848 = {}
    # Getting the type of 'list' (line 103)
    list_192846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 25), 'list', False)
    # Calling list(args, kwargs) (line 103)
    list_call_result_192849 = invoke(stypy.reporting.localization.Localization(__file__, 103, 25), list_192846, *[expected_192847], **kwargs_192848)
    
    # Assigning a type to the variable 'tuple_assignment_192633' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'tuple_assignment_192633', list_call_result_192849)
    
    # Assigning a Name to a Name (line 103):
    # Getting the type of 'tuple_assignment_192632' (line 103)
    tuple_assignment_192632_192850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'tuple_assignment_192632')
    # Assigning a type to the variable 's' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 's', tuple_assignment_192632_192850)
    
    # Assigning a Name to a Name (line 103):
    # Getting the type of 'tuple_assignment_192633' (line 103)
    tuple_assignment_192633_192851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'tuple_assignment_192633')
    # Assigning a type to the variable 't' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 7), 't', tuple_assignment_192633_192851)
    
    # Assigning a Tuple to a Tuple (line 104):
    
    # Assigning a Call to a Name (line 104):
    
    # Call to len(...): (line 104)
    # Processing the call arguments (line 104)
    # Getting the type of 's' (line 104)
    s_192853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 15), 's', False)
    # Processing the call keyword arguments (line 104)
    kwargs_192854 = {}
    # Getting the type of 'len' (line 104)
    len_192852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 11), 'len', False)
    # Calling len(args, kwargs) (line 104)
    len_call_result_192855 = invoke(stypy.reporting.localization.Localization(__file__, 104, 11), len_192852, *[s_192853], **kwargs_192854)
    
    # Assigning a type to the variable 'tuple_assignment_192634' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'tuple_assignment_192634', len_call_result_192855)
    
    # Assigning a Call to a Name (line 104):
    
    # Call to len(...): (line 104)
    # Processing the call arguments (line 104)
    # Getting the type of 't' (line 104)
    t_192857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 23), 't', False)
    # Processing the call keyword arguments (line 104)
    kwargs_192858 = {}
    # Getting the type of 'len' (line 104)
    len_192856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 19), 'len', False)
    # Calling len(args, kwargs) (line 104)
    len_call_result_192859 = invoke(stypy.reporting.localization.Localization(__file__, 104, 19), len_192856, *[t_192857], **kwargs_192858)
    
    # Assigning a type to the variable 'tuple_assignment_192635' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'tuple_assignment_192635', len_call_result_192859)
    
    # Assigning a Name to a Name (line 104):
    # Getting the type of 'tuple_assignment_192634' (line 104)
    tuple_assignment_192634_192860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'tuple_assignment_192634')
    # Assigning a type to the variable 'm' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'm', tuple_assignment_192634_192860)
    
    # Assigning a Name to a Name (line 104):
    # Getting the type of 'tuple_assignment_192635' (line 104)
    tuple_assignment_192635_192861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'tuple_assignment_192635')
    # Assigning a type to the variable 'n' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 7), 'n', tuple_assignment_192635_192861)
    
    # Assigning a Call to a Name (line 105):
    
    # Assigning a Call to a Name (line 105):
    
    # Call to object(...): (line 105)
    # Processing the call keyword arguments (line 105)
    kwargs_192863 = {}
    # Getting the type of 'object' (line 105)
    object_192862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 11), 'object', False)
    # Calling object(args, kwargs) (line 105)
    object_call_result_192864 = invoke(stypy.reporting.localization.Localization(__file__, 105, 11), object_192862, *[], **kwargs_192863)
    
    # Assigning a type to the variable 'NULL' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'NULL', object_call_result_192864)
    
    # Assigning a List to a Name (line 106):
    
    # Assigning a List to a Name (line 106):
    
    # Obtaining an instance of the builtin type 'list' (line 106)
    list_192865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 106)
    
    # Assigning a type to the variable 'result' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'result', list_192865)
    
    
    # Call to enumerate(...): (line 107)
    # Processing the call arguments (line 107)
    # Getting the type of 's' (line 107)
    s_192867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 29), 's', False)
    # Processing the call keyword arguments (line 107)
    kwargs_192868 = {}
    # Getting the type of 'enumerate' (line 107)
    enumerate_192866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 19), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 107)
    enumerate_call_result_192869 = invoke(stypy.reporting.localization.Localization(__file__, 107, 19), enumerate_192866, *[s_192867], **kwargs_192868)
    
    # Testing the type of a for loop iterable (line 107)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 107, 4), enumerate_call_result_192869)
    # Getting the type of the for loop variable (line 107)
    for_loop_var_192870 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 107, 4), enumerate_call_result_192869)
    # Assigning a type to the variable 'i' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 4), for_loop_var_192870))
    # Assigning a type to the variable 'elem' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'elem', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 4), for_loop_var_192870))
    # SSA begins for a for statement (line 107)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'elem' (line 108)
    elem_192871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 11), 'elem')
    # Getting the type of 'NULL' (line 108)
    NULL_192872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 19), 'NULL')
    # Applying the binary operator 'is' (line 108)
    result_is__192873 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 11), 'is', elem_192871, NULL_192872)
    
    # Testing the type of an if condition (line 108)
    if_condition_192874 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 8), result_is__192873)
    # Assigning a type to the variable 'if_condition_192874' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'if_condition_192874', if_condition_192874)
    # SSA begins for if statement (line 108)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 108)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Multiple assignment of 2 elements.
    
    # Assigning a Num to a Name (line 110):
    int_192875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 24), 'int')
    # Assigning a type to the variable 'cnt_t' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'cnt_t', int_192875)
    
    # Assigning a Name to a Name (line 110):
    # Getting the type of 'cnt_t' (line 110)
    cnt_t_192876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'cnt_t')
    # Assigning a type to the variable 'cnt_s' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'cnt_s', cnt_t_192876)
    
    
    # Call to range(...): (line 111)
    # Processing the call arguments (line 111)
    # Getting the type of 'i' (line 111)
    i_192878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 23), 'i', False)
    # Getting the type of 'm' (line 111)
    m_192879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 26), 'm', False)
    # Processing the call keyword arguments (line 111)
    kwargs_192880 = {}
    # Getting the type of 'range' (line 111)
    range_192877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 17), 'range', False)
    # Calling range(args, kwargs) (line 111)
    range_call_result_192881 = invoke(stypy.reporting.localization.Localization(__file__, 111, 17), range_192877, *[i_192878, m_192879], **kwargs_192880)
    
    # Testing the type of a for loop iterable (line 111)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 111, 8), range_call_result_192881)
    # Getting the type of the for loop variable (line 111)
    for_loop_var_192882 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 111, 8), range_call_result_192881)
    # Assigning a type to the variable 'j' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'j', for_loop_var_192882)
    # SSA begins for a for statement (line 111)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 112)
    j_192883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 17), 'j')
    # Getting the type of 's' (line 112)
    s_192884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 15), 's')
    # Obtaining the member '__getitem__' of a type (line 112)
    getitem___192885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 15), s_192884, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 112)
    subscript_call_result_192886 = invoke(stypy.reporting.localization.Localization(__file__, 112, 15), getitem___192885, j_192883)
    
    # Getting the type of 'elem' (line 112)
    elem_192887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 23), 'elem')
    # Applying the binary operator '==' (line 112)
    result_eq_192888 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 15), '==', subscript_call_result_192886, elem_192887)
    
    # Testing the type of an if condition (line 112)
    if_condition_192889 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 12), result_eq_192888)
    # Assigning a type to the variable 'if_condition_192889' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'if_condition_192889', if_condition_192889)
    # SSA begins for if statement (line 112)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'cnt_s' (line 113)
    cnt_s_192890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 16), 'cnt_s')
    int_192891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 25), 'int')
    # Applying the binary operator '+=' (line 113)
    result_iadd_192892 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 16), '+=', cnt_s_192890, int_192891)
    # Assigning a type to the variable 'cnt_s' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 16), 'cnt_s', result_iadd_192892)
    
    
    # Assigning a Name to a Subscript (line 114):
    
    # Assigning a Name to a Subscript (line 114):
    # Getting the type of 'NULL' (line 114)
    NULL_192893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 23), 'NULL')
    # Getting the type of 's' (line 114)
    s_192894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 16), 's')
    # Getting the type of 'j' (line 114)
    j_192895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 18), 'j')
    # Storing an element on a container (line 114)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 16), s_192894, (j_192895, NULL_192893))
    # SSA join for if statement (line 112)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to enumerate(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 't' (line 115)
    t_192897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 39), 't', False)
    # Processing the call keyword arguments (line 115)
    kwargs_192898 = {}
    # Getting the type of 'enumerate' (line 115)
    enumerate_192896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 29), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 115)
    enumerate_call_result_192899 = invoke(stypy.reporting.localization.Localization(__file__, 115, 29), enumerate_192896, *[t_192897], **kwargs_192898)
    
    # Testing the type of a for loop iterable (line 115)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 115, 8), enumerate_call_result_192899)
    # Getting the type of the for loop variable (line 115)
    for_loop_var_192900 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 115, 8), enumerate_call_result_192899)
    # Assigning a type to the variable 'j' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 8), for_loop_var_192900))
    # Assigning a type to the variable 'other_elem' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'other_elem', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 8), for_loop_var_192900))
    # SSA begins for a for statement (line 115)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'other_elem' (line 116)
    other_elem_192901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 15), 'other_elem')
    # Getting the type of 'elem' (line 116)
    elem_192902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 29), 'elem')
    # Applying the binary operator '==' (line 116)
    result_eq_192903 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 15), '==', other_elem_192901, elem_192902)
    
    # Testing the type of an if condition (line 116)
    if_condition_192904 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 116, 12), result_eq_192903)
    # Assigning a type to the variable 'if_condition_192904' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'if_condition_192904', if_condition_192904)
    # SSA begins for if statement (line 116)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'cnt_t' (line 117)
    cnt_t_192905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 16), 'cnt_t')
    int_192906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 25), 'int')
    # Applying the binary operator '+=' (line 117)
    result_iadd_192907 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 16), '+=', cnt_t_192905, int_192906)
    # Assigning a type to the variable 'cnt_t' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 16), 'cnt_t', result_iadd_192907)
    
    
    # Assigning a Name to a Subscript (line 118):
    
    # Assigning a Name to a Subscript (line 118):
    # Getting the type of 'NULL' (line 118)
    NULL_192908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 23), 'NULL')
    # Getting the type of 't' (line 118)
    t_192909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 16), 't')
    # Getting the type of 'j' (line 118)
    j_192910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 18), 'j')
    # Storing an element on a container (line 118)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 16), t_192909, (j_192910, NULL_192908))
    # SSA join for if statement (line 116)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'cnt_s' (line 119)
    cnt_s_192911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 11), 'cnt_s')
    # Getting the type of 'cnt_t' (line 119)
    cnt_t_192912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 20), 'cnt_t')
    # Applying the binary operator '!=' (line 119)
    result_ne_192913 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 11), '!=', cnt_s_192911, cnt_t_192912)
    
    # Testing the type of an if condition (line 119)
    if_condition_192914 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 119, 8), result_ne_192913)
    # Assigning a type to the variable 'if_condition_192914' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'if_condition_192914', if_condition_192914)
    # SSA begins for if statement (line 119)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 120):
    
    # Assigning a Call to a Name (line 120):
    
    # Call to _Mismatch(...): (line 120)
    # Processing the call arguments (line 120)
    # Getting the type of 'cnt_s' (line 120)
    cnt_s_192916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 29), 'cnt_s', False)
    # Getting the type of 'cnt_t' (line 120)
    cnt_t_192917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 36), 'cnt_t', False)
    # Getting the type of 'elem' (line 120)
    elem_192918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 43), 'elem', False)
    # Processing the call keyword arguments (line 120)
    kwargs_192919 = {}
    # Getting the type of '_Mismatch' (line 120)
    _Mismatch_192915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 19), '_Mismatch', False)
    # Calling _Mismatch(args, kwargs) (line 120)
    _Mismatch_call_result_192920 = invoke(stypy.reporting.localization.Localization(__file__, 120, 19), _Mismatch_192915, *[cnt_s_192916, cnt_t_192917, elem_192918], **kwargs_192919)
    
    # Assigning a type to the variable 'diff' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'diff', _Mismatch_call_result_192920)
    
    # Call to append(...): (line 121)
    # Processing the call arguments (line 121)
    # Getting the type of 'diff' (line 121)
    diff_192923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 26), 'diff', False)
    # Processing the call keyword arguments (line 121)
    kwargs_192924 = {}
    # Getting the type of 'result' (line 121)
    result_192921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'result', False)
    # Obtaining the member 'append' of a type (line 121)
    append_192922 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 12), result_192921, 'append')
    # Calling append(args, kwargs) (line 121)
    append_call_result_192925 = invoke(stypy.reporting.localization.Localization(__file__, 121, 12), append_192922, *[diff_192923], **kwargs_192924)
    
    # SSA join for if statement (line 119)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to enumerate(...): (line 123)
    # Processing the call arguments (line 123)
    # Getting the type of 't' (line 123)
    t_192927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 29), 't', False)
    # Processing the call keyword arguments (line 123)
    kwargs_192928 = {}
    # Getting the type of 'enumerate' (line 123)
    enumerate_192926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 19), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 123)
    enumerate_call_result_192929 = invoke(stypy.reporting.localization.Localization(__file__, 123, 19), enumerate_192926, *[t_192927], **kwargs_192928)
    
    # Testing the type of a for loop iterable (line 123)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 123, 4), enumerate_call_result_192929)
    # Getting the type of the for loop variable (line 123)
    for_loop_var_192930 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 123, 4), enumerate_call_result_192929)
    # Assigning a type to the variable 'i' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 4), for_loop_var_192930))
    # Assigning a type to the variable 'elem' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'elem', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 4), for_loop_var_192930))
    # SSA begins for a for statement (line 123)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'elem' (line 124)
    elem_192931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 11), 'elem')
    # Getting the type of 'NULL' (line 124)
    NULL_192932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 19), 'NULL')
    # Applying the binary operator 'is' (line 124)
    result_is__192933 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 11), 'is', elem_192931, NULL_192932)
    
    # Testing the type of an if condition (line 124)
    if_condition_192934 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 124, 8), result_is__192933)
    # Assigning a type to the variable 'if_condition_192934' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'if_condition_192934', if_condition_192934)
    # SSA begins for if statement (line 124)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 124)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Name (line 126):
    
    # Assigning a Num to a Name (line 126):
    int_192935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 16), 'int')
    # Assigning a type to the variable 'cnt_t' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'cnt_t', int_192935)
    
    
    # Call to range(...): (line 127)
    # Processing the call arguments (line 127)
    # Getting the type of 'i' (line 127)
    i_192937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 23), 'i', False)
    # Getting the type of 'n' (line 127)
    n_192938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 26), 'n', False)
    # Processing the call keyword arguments (line 127)
    kwargs_192939 = {}
    # Getting the type of 'range' (line 127)
    range_192936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 17), 'range', False)
    # Calling range(args, kwargs) (line 127)
    range_call_result_192940 = invoke(stypy.reporting.localization.Localization(__file__, 127, 17), range_192936, *[i_192937, n_192938], **kwargs_192939)
    
    # Testing the type of a for loop iterable (line 127)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 127, 8), range_call_result_192940)
    # Getting the type of the for loop variable (line 127)
    for_loop_var_192941 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 127, 8), range_call_result_192940)
    # Assigning a type to the variable 'j' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'j', for_loop_var_192941)
    # SSA begins for a for statement (line 127)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 128)
    j_192942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 17), 'j')
    # Getting the type of 't' (line 128)
    t_192943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 15), 't')
    # Obtaining the member '__getitem__' of a type (line 128)
    getitem___192944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 15), t_192943, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 128)
    subscript_call_result_192945 = invoke(stypy.reporting.localization.Localization(__file__, 128, 15), getitem___192944, j_192942)
    
    # Getting the type of 'elem' (line 128)
    elem_192946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 23), 'elem')
    # Applying the binary operator '==' (line 128)
    result_eq_192947 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 15), '==', subscript_call_result_192945, elem_192946)
    
    # Testing the type of an if condition (line 128)
    if_condition_192948 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 128, 12), result_eq_192947)
    # Assigning a type to the variable 'if_condition_192948' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'if_condition_192948', if_condition_192948)
    # SSA begins for if statement (line 128)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'cnt_t' (line 129)
    cnt_t_192949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'cnt_t')
    int_192950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 25), 'int')
    # Applying the binary operator '+=' (line 129)
    result_iadd_192951 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 16), '+=', cnt_t_192949, int_192950)
    # Assigning a type to the variable 'cnt_t' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'cnt_t', result_iadd_192951)
    
    
    # Assigning a Name to a Subscript (line 130):
    
    # Assigning a Name to a Subscript (line 130):
    # Getting the type of 'NULL' (line 130)
    NULL_192952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 23), 'NULL')
    # Getting the type of 't' (line 130)
    t_192953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 't')
    # Getting the type of 'j' (line 130)
    j_192954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 18), 'j')
    # Storing an element on a container (line 130)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 16), t_192953, (j_192954, NULL_192952))
    # SSA join for if statement (line 128)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 131):
    
    # Assigning a Call to a Name (line 131):
    
    # Call to _Mismatch(...): (line 131)
    # Processing the call arguments (line 131)
    int_192956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 25), 'int')
    # Getting the type of 'cnt_t' (line 131)
    cnt_t_192957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 28), 'cnt_t', False)
    # Getting the type of 'elem' (line 131)
    elem_192958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 35), 'elem', False)
    # Processing the call keyword arguments (line 131)
    kwargs_192959 = {}
    # Getting the type of '_Mismatch' (line 131)
    _Mismatch_192955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 15), '_Mismatch', False)
    # Calling _Mismatch(args, kwargs) (line 131)
    _Mismatch_call_result_192960 = invoke(stypy.reporting.localization.Localization(__file__, 131, 15), _Mismatch_192955, *[int_192956, cnt_t_192957, elem_192958], **kwargs_192959)
    
    # Assigning a type to the variable 'diff' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'diff', _Mismatch_call_result_192960)
    
    # Call to append(...): (line 132)
    # Processing the call arguments (line 132)
    # Getting the type of 'diff' (line 132)
    diff_192963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 22), 'diff', False)
    # Processing the call keyword arguments (line 132)
    kwargs_192964 = {}
    # Getting the type of 'result' (line 132)
    result_192961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'result', False)
    # Obtaining the member 'append' of a type (line 132)
    append_192962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 8), result_192961, 'append')
    # Calling append(args, kwargs) (line 132)
    append_call_result_192965 = invoke(stypy.reporting.localization.Localization(__file__, 132, 8), append_192962, *[diff_192963], **kwargs_192964)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'result' (line 133)
    result_192966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 11), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'stypy_return_type', result_192966)
    
    # ################# End of '_count_diff_all_purpose(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_count_diff_all_purpose' in the type store
    # Getting the type of 'stypy_return_type' (line 100)
    stypy_return_type_192967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_192967)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_count_diff_all_purpose'
    return stypy_return_type_192967

# Assigning a type to the variable '_count_diff_all_purpose' (line 100)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 0), '_count_diff_all_purpose', _count_diff_all_purpose)

@norecursion
def _ordered_count(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_ordered_count'
    module_type_store = module_type_store.open_function_context('_ordered_count', 135, 0, False)
    
    # Passed parameters checking function
    _ordered_count.stypy_localization = localization
    _ordered_count.stypy_type_of_self = None
    _ordered_count.stypy_type_store = module_type_store
    _ordered_count.stypy_function_name = '_ordered_count'
    _ordered_count.stypy_param_names_list = ['iterable']
    _ordered_count.stypy_varargs_param_name = None
    _ordered_count.stypy_kwargs_param_name = None
    _ordered_count.stypy_call_defaults = defaults
    _ordered_count.stypy_call_varargs = varargs
    _ordered_count.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_ordered_count', ['iterable'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_ordered_count', localization, ['iterable'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_ordered_count(...)' code ##################

    str_192968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 4), 'str', 'Return dict of element counts, in the order they were first seen')
    
    # Assigning a Call to a Name (line 137):
    
    # Assigning a Call to a Name (line 137):
    
    # Call to OrderedDict(...): (line 137)
    # Processing the call keyword arguments (line 137)
    kwargs_192970 = {}
    # Getting the type of 'OrderedDict' (line 137)
    OrderedDict_192969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'OrderedDict', False)
    # Calling OrderedDict(args, kwargs) (line 137)
    OrderedDict_call_result_192971 = invoke(stypy.reporting.localization.Localization(__file__, 137, 8), OrderedDict_192969, *[], **kwargs_192970)
    
    # Assigning a type to the variable 'c' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'c', OrderedDict_call_result_192971)
    
    # Getting the type of 'iterable' (line 138)
    iterable_192972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'iterable')
    # Testing the type of a for loop iterable (line 138)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 138, 4), iterable_192972)
    # Getting the type of the for loop variable (line 138)
    for_loop_var_192973 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 138, 4), iterable_192972)
    # Assigning a type to the variable 'elem' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'elem', for_loop_var_192973)
    # SSA begins for a for statement (line 138)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Subscript (line 139):
    
    # Assigning a BinOp to a Subscript (line 139):
    
    # Call to get(...): (line 139)
    # Processing the call arguments (line 139)
    # Getting the type of 'elem' (line 139)
    elem_192976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 24), 'elem', False)
    int_192977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 30), 'int')
    # Processing the call keyword arguments (line 139)
    kwargs_192978 = {}
    # Getting the type of 'c' (line 139)
    c_192974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 18), 'c', False)
    # Obtaining the member 'get' of a type (line 139)
    get_192975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 18), c_192974, 'get')
    # Calling get(args, kwargs) (line 139)
    get_call_result_192979 = invoke(stypy.reporting.localization.Localization(__file__, 139, 18), get_192975, *[elem_192976, int_192977], **kwargs_192978)
    
    int_192980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 35), 'int')
    # Applying the binary operator '+' (line 139)
    result_add_192981 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 18), '+', get_call_result_192979, int_192980)
    
    # Getting the type of 'c' (line 139)
    c_192982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'c')
    # Getting the type of 'elem' (line 139)
    elem_192983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 10), 'elem')
    # Storing an element on a container (line 139)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 8), c_192982, (elem_192983, result_add_192981))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'c' (line 140)
    c_192984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 11), 'c')
    # Assigning a type to the variable 'stypy_return_type' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'stypy_return_type', c_192984)
    
    # ################# End of '_ordered_count(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_ordered_count' in the type store
    # Getting the type of 'stypy_return_type' (line 135)
    stypy_return_type_192985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_192985)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_ordered_count'
    return stypy_return_type_192985

# Assigning a type to the variable '_ordered_count' (line 135)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 0), '_ordered_count', _ordered_count)

@norecursion
def _count_diff_hashable(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_count_diff_hashable'
    module_type_store = module_type_store.open_function_context('_count_diff_hashable', 142, 0, False)
    
    # Passed parameters checking function
    _count_diff_hashable.stypy_localization = localization
    _count_diff_hashable.stypy_type_of_self = None
    _count_diff_hashable.stypy_type_store = module_type_store
    _count_diff_hashable.stypy_function_name = '_count_diff_hashable'
    _count_diff_hashable.stypy_param_names_list = ['actual', 'expected']
    _count_diff_hashable.stypy_varargs_param_name = None
    _count_diff_hashable.stypy_kwargs_param_name = None
    _count_diff_hashable.stypy_call_defaults = defaults
    _count_diff_hashable.stypy_call_varargs = varargs
    _count_diff_hashable.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_count_diff_hashable', ['actual', 'expected'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_count_diff_hashable', localization, ['actual', 'expected'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_count_diff_hashable(...)' code ##################

    str_192986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 4), 'str', 'Returns list of (cnt_act, cnt_exp, elem) triples where the counts differ')
    
    # Assigning a Tuple to a Tuple (line 145):
    
    # Assigning a Call to a Name (line 145):
    
    # Call to _ordered_count(...): (line 145)
    # Processing the call arguments (line 145)
    # Getting the type of 'actual' (line 145)
    actual_192988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 26), 'actual', False)
    # Processing the call keyword arguments (line 145)
    kwargs_192989 = {}
    # Getting the type of '_ordered_count' (line 145)
    _ordered_count_192987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 11), '_ordered_count', False)
    # Calling _ordered_count(args, kwargs) (line 145)
    _ordered_count_call_result_192990 = invoke(stypy.reporting.localization.Localization(__file__, 145, 11), _ordered_count_192987, *[actual_192988], **kwargs_192989)
    
    # Assigning a type to the variable 'tuple_assignment_192636' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'tuple_assignment_192636', _ordered_count_call_result_192990)
    
    # Assigning a Call to a Name (line 145):
    
    # Call to _ordered_count(...): (line 145)
    # Processing the call arguments (line 145)
    # Getting the type of 'expected' (line 145)
    expected_192992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 50), 'expected', False)
    # Processing the call keyword arguments (line 145)
    kwargs_192993 = {}
    # Getting the type of '_ordered_count' (line 145)
    _ordered_count_192991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 35), '_ordered_count', False)
    # Calling _ordered_count(args, kwargs) (line 145)
    _ordered_count_call_result_192994 = invoke(stypy.reporting.localization.Localization(__file__, 145, 35), _ordered_count_192991, *[expected_192992], **kwargs_192993)
    
    # Assigning a type to the variable 'tuple_assignment_192637' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'tuple_assignment_192637', _ordered_count_call_result_192994)
    
    # Assigning a Name to a Name (line 145):
    # Getting the type of 'tuple_assignment_192636' (line 145)
    tuple_assignment_192636_192995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'tuple_assignment_192636')
    # Assigning a type to the variable 's' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 's', tuple_assignment_192636_192995)
    
    # Assigning a Name to a Name (line 145):
    # Getting the type of 'tuple_assignment_192637' (line 145)
    tuple_assignment_192637_192996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'tuple_assignment_192637')
    # Assigning a type to the variable 't' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 7), 't', tuple_assignment_192637_192996)
    
    # Assigning a List to a Name (line 146):
    
    # Assigning a List to a Name (line 146):
    
    # Obtaining an instance of the builtin type 'list' (line 146)
    list_192997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 146)
    
    # Assigning a type to the variable 'result' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'result', list_192997)
    
    
    # Call to items(...): (line 147)
    # Processing the call keyword arguments (line 147)
    kwargs_193000 = {}
    # Getting the type of 's' (line 147)
    s_192998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 23), 's', False)
    # Obtaining the member 'items' of a type (line 147)
    items_192999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 23), s_192998, 'items')
    # Calling items(args, kwargs) (line 147)
    items_call_result_193001 = invoke(stypy.reporting.localization.Localization(__file__, 147, 23), items_192999, *[], **kwargs_193000)
    
    # Testing the type of a for loop iterable (line 147)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 147, 4), items_call_result_193001)
    # Getting the type of the for loop variable (line 147)
    for_loop_var_193002 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 147, 4), items_call_result_193001)
    # Assigning a type to the variable 'elem' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'elem', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 4), for_loop_var_193002))
    # Assigning a type to the variable 'cnt_s' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'cnt_s', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 4), for_loop_var_193002))
    # SSA begins for a for statement (line 147)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 148):
    
    # Assigning a Call to a Name (line 148):
    
    # Call to get(...): (line 148)
    # Processing the call arguments (line 148)
    # Getting the type of 'elem' (line 148)
    elem_193005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 22), 'elem', False)
    int_193006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 28), 'int')
    # Processing the call keyword arguments (line 148)
    kwargs_193007 = {}
    # Getting the type of 't' (line 148)
    t_193003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 16), 't', False)
    # Obtaining the member 'get' of a type (line 148)
    get_193004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 16), t_193003, 'get')
    # Calling get(args, kwargs) (line 148)
    get_call_result_193008 = invoke(stypy.reporting.localization.Localization(__file__, 148, 16), get_193004, *[elem_193005, int_193006], **kwargs_193007)
    
    # Assigning a type to the variable 'cnt_t' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'cnt_t', get_call_result_193008)
    
    
    # Getting the type of 'cnt_s' (line 149)
    cnt_s_193009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 11), 'cnt_s')
    # Getting the type of 'cnt_t' (line 149)
    cnt_t_193010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 20), 'cnt_t')
    # Applying the binary operator '!=' (line 149)
    result_ne_193011 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 11), '!=', cnt_s_193009, cnt_t_193010)
    
    # Testing the type of an if condition (line 149)
    if_condition_193012 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 149, 8), result_ne_193011)
    # Assigning a type to the variable 'if_condition_193012' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'if_condition_193012', if_condition_193012)
    # SSA begins for if statement (line 149)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 150):
    
    # Assigning a Call to a Name (line 150):
    
    # Call to _Mismatch(...): (line 150)
    # Processing the call arguments (line 150)
    # Getting the type of 'cnt_s' (line 150)
    cnt_s_193014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 29), 'cnt_s', False)
    # Getting the type of 'cnt_t' (line 150)
    cnt_t_193015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 36), 'cnt_t', False)
    # Getting the type of 'elem' (line 150)
    elem_193016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 43), 'elem', False)
    # Processing the call keyword arguments (line 150)
    kwargs_193017 = {}
    # Getting the type of '_Mismatch' (line 150)
    _Mismatch_193013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 19), '_Mismatch', False)
    # Calling _Mismatch(args, kwargs) (line 150)
    _Mismatch_call_result_193018 = invoke(stypy.reporting.localization.Localization(__file__, 150, 19), _Mismatch_193013, *[cnt_s_193014, cnt_t_193015, elem_193016], **kwargs_193017)
    
    # Assigning a type to the variable 'diff' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'diff', _Mismatch_call_result_193018)
    
    # Call to append(...): (line 151)
    # Processing the call arguments (line 151)
    # Getting the type of 'diff' (line 151)
    diff_193021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 26), 'diff', False)
    # Processing the call keyword arguments (line 151)
    kwargs_193022 = {}
    # Getting the type of 'result' (line 151)
    result_193019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'result', False)
    # Obtaining the member 'append' of a type (line 151)
    append_193020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 12), result_193019, 'append')
    # Calling append(args, kwargs) (line 151)
    append_call_result_193023 = invoke(stypy.reporting.localization.Localization(__file__, 151, 12), append_193020, *[diff_193021], **kwargs_193022)
    
    # SSA join for if statement (line 149)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to items(...): (line 152)
    # Processing the call keyword arguments (line 152)
    kwargs_193026 = {}
    # Getting the type of 't' (line 152)
    t_193024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 23), 't', False)
    # Obtaining the member 'items' of a type (line 152)
    items_193025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 23), t_193024, 'items')
    # Calling items(args, kwargs) (line 152)
    items_call_result_193027 = invoke(stypy.reporting.localization.Localization(__file__, 152, 23), items_193025, *[], **kwargs_193026)
    
    # Testing the type of a for loop iterable (line 152)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 152, 4), items_call_result_193027)
    # Getting the type of the for loop variable (line 152)
    for_loop_var_193028 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 152, 4), items_call_result_193027)
    # Assigning a type to the variable 'elem' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'elem', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 4), for_loop_var_193028))
    # Assigning a type to the variable 'cnt_t' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'cnt_t', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 4), for_loop_var_193028))
    # SSA begins for a for statement (line 152)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'elem' (line 153)
    elem_193029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 11), 'elem')
    # Getting the type of 's' (line 153)
    s_193030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 23), 's')
    # Applying the binary operator 'notin' (line 153)
    result_contains_193031 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 11), 'notin', elem_193029, s_193030)
    
    # Testing the type of an if condition (line 153)
    if_condition_193032 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 8), result_contains_193031)
    # Assigning a type to the variable 'if_condition_193032' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'if_condition_193032', if_condition_193032)
    # SSA begins for if statement (line 153)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 154):
    
    # Assigning a Call to a Name (line 154):
    
    # Call to _Mismatch(...): (line 154)
    # Processing the call arguments (line 154)
    int_193034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 29), 'int')
    # Getting the type of 'cnt_t' (line 154)
    cnt_t_193035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 32), 'cnt_t', False)
    # Getting the type of 'elem' (line 154)
    elem_193036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 39), 'elem', False)
    # Processing the call keyword arguments (line 154)
    kwargs_193037 = {}
    # Getting the type of '_Mismatch' (line 154)
    _Mismatch_193033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 19), '_Mismatch', False)
    # Calling _Mismatch(args, kwargs) (line 154)
    _Mismatch_call_result_193038 = invoke(stypy.reporting.localization.Localization(__file__, 154, 19), _Mismatch_193033, *[int_193034, cnt_t_193035, elem_193036], **kwargs_193037)
    
    # Assigning a type to the variable 'diff' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'diff', _Mismatch_call_result_193038)
    
    # Call to append(...): (line 155)
    # Processing the call arguments (line 155)
    # Getting the type of 'diff' (line 155)
    diff_193041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 26), 'diff', False)
    # Processing the call keyword arguments (line 155)
    kwargs_193042 = {}
    # Getting the type of 'result' (line 155)
    result_193039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'result', False)
    # Obtaining the member 'append' of a type (line 155)
    append_193040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 12), result_193039, 'append')
    # Calling append(args, kwargs) (line 155)
    append_call_result_193043 = invoke(stypy.reporting.localization.Localization(__file__, 155, 12), append_193040, *[diff_193041], **kwargs_193042)
    
    # SSA join for if statement (line 153)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'result' (line 156)
    result_193044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 11), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 156)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'stypy_return_type', result_193044)
    
    # ################# End of '_count_diff_hashable(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_count_diff_hashable' in the type store
    # Getting the type of 'stypy_return_type' (line 142)
    stypy_return_type_193045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_193045)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_count_diff_hashable'
    return stypy_return_type_193045

# Assigning a type to the variable '_count_diff_hashable' (line 142)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 0), '_count_diff_hashable', _count_diff_hashable)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
