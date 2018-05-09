
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # '''
2: # This file contains functions that will be called from type inference code to deal with Python operator calls. It also
3: # contains the logic to convert from operator names to its symbolic representation and viceversa.
4: # '''
5: #
6: # Table to transform operator symbols to operator names (the name of the function that have to be called to implement
7: # the operator functionality in the builtin operator type inference module
8: __symbol_to_operator_table = {
9:     '+': 'add',
10:     '&': 'and_',
11:     'in': 'contains',
12:     '==': 'eq',
13:     '//': 'floordiv',
14:     '>=': 'ge',
15:     '[]': 'getitem',
16:     '>': 'gt',
17:     '+=': 'iadd',
18:     '&=': 'iand',
19:     '//=': 'ifloordiv',
20:     '<<=': 'ilshift',
21:     '%=': 'imod',
22:     '*=': 'imul',
23:     '~': 'inv',
24:     '|=': 'ior',
25:     '**=': 'ipow',
26:     '>>=': 'irshift',
27:     'is': 'is_',
28:     'is not': 'is_not',
29:     '-=': 'isub',
30:     '/=': 'itruediv',
31:     '^=': 'ixor',
32:     '<=': 'le',
33:     '<<': 'lshift',
34:     '<': 'lt',
35:     '%': 'mod',
36:     '*': 'mul',
37:     '!=': 'ne',
38:     '-': 'sub',  # beware of neg (unary)
39:     '/': 'truediv',
40:     '^': 'xor',
41:     '|': 'or_',
42:     'mult': 'mul',
43:     'and': 'and_',
44:     'not': 'not_',
45:     'or': 'or_',
46:     'div': 'div',
47:     'isnot': 'is_not',
48:     '**': 'pow',
49:     'notin': 'contains',
50: }
51: 
52: # Table to perform the opposite operation than the previous one
53: __operator_to_symbol_table = {
54:     'lte': '<=',
55:     'gte': '>=',
56:     'eq': '==',
57:     'is_': 'is',
58:     'ior': '|=',
59:     'iand': '&=',
60:     'getitem': '[]',
61:     'imod': '%=',
62:     'not_': 'not',
63:     'xor': '^',
64:     'contains': 'in',
65:     'ifloordiv': '//=',
66:     'noteq': '!=',
67:     'is_not': 'isnot',
68:     'floordiv': '//',
69:     'mod': '%',
70:     'ixor': '^=',
71:     'ilshift': '<<=',
72:     'and_': '&',
73:     'add': '+',
74:     'mul': '*',
75:     'mult': '*',
76:     'sub': '-',
77:     'itruediv': '/=',
78:     'truediv': '/',
79:     'div': 'div',  # Integer division. We cannot use / as it is the float division (truediv)
80:     'lt': '<',
81:     'irshift': '>>=',
82:     'isub': '-=',
83:     'inv': '~',
84:     'lshift': '<<',
85:     'iadd': '+=',
86:     'gt': '>',
87:     'pow': '**',
88:     'bitor': '|',
89:     'bitand': '&',
90:     'bitxor': '^',
91:     'invert': '~',
92: }
93: 
94: 
95: # ###################################### OPERATOR REPRESENTATION CONVERSION #######################################
96: 
97: def operator_name_to_symbol(operator_name):
98:     '''
99:     Transform an operator name to its symbolic representation (example: 'add' -> '+'. If no symbol is available, return
100:     the passed name
101:     :param operator_name: Operator name
102:     :return: Operator symbol
103:     '''
104:     try:
105:         return __operator_to_symbol_table[operator_name]
106:     except KeyError:
107:         return operator_name
108: 
109: 
110: def operator_symbol_to_name(operator_symbol):
111:     '''
112:     Transform an operator symbol to its function name (example: '+' -> 'add'.
113:     :param operator_symbol: Operator symbol
114:     :return: Operator name
115:     '''
116:     return __symbol_to_operator_table[operator_symbol]
117: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Dict to a Name (line 8):

# Obtaining an instance of the builtin type 'dict' (line 8)
dict_2426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 29), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 8)
# Adding element type (key, value) (line 8)
str_2427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 4), 'str', '+')
str_2428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 9), 'str', 'add')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2427, str_2428))
# Adding element type (key, value) (line 8)
str_2429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 4), 'str', '&')
str_2430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 9), 'str', 'and_')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2429, str_2430))
# Adding element type (key, value) (line 8)
str_2431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 4), 'str', 'in')
str_2432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 10), 'str', 'contains')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2431, str_2432))
# Adding element type (key, value) (line 8)
str_2433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 4), 'str', '==')
str_2434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 10), 'str', 'eq')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2433, str_2434))
# Adding element type (key, value) (line 8)
str_2435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 4), 'str', '//')
str_2436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 10), 'str', 'floordiv')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2435, str_2436))
# Adding element type (key, value) (line 8)
str_2437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 4), 'str', '>=')
str_2438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 10), 'str', 'ge')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2437, str_2438))
# Adding element type (key, value) (line 8)
str_2439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 4), 'str', '[]')
str_2440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 10), 'str', 'getitem')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2439, str_2440))
# Adding element type (key, value) (line 8)
str_2441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 4), 'str', '>')
str_2442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 9), 'str', 'gt')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2441, str_2442))
# Adding element type (key, value) (line 8)
str_2443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 4), 'str', '+=')
str_2444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 10), 'str', 'iadd')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2443, str_2444))
# Adding element type (key, value) (line 8)
str_2445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 4), 'str', '&=')
str_2446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 10), 'str', 'iand')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2445, str_2446))
# Adding element type (key, value) (line 8)
str_2447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 4), 'str', '//=')
str_2448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 11), 'str', 'ifloordiv')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2447, str_2448))
# Adding element type (key, value) (line 8)
str_2449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 4), 'str', '<<=')
str_2450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 11), 'str', 'ilshift')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2449, str_2450))
# Adding element type (key, value) (line 8)
str_2451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 4), 'str', '%=')
str_2452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 10), 'str', 'imod')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2451, str_2452))
# Adding element type (key, value) (line 8)
str_2453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 4), 'str', '*=')
str_2454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 10), 'str', 'imul')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2453, str_2454))
# Adding element type (key, value) (line 8)
str_2455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 4), 'str', '~')
str_2456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 9), 'str', 'inv')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2455, str_2456))
# Adding element type (key, value) (line 8)
str_2457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 4), 'str', '|=')
str_2458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 10), 'str', 'ior')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2457, str_2458))
# Adding element type (key, value) (line 8)
str_2459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 4), 'str', '**=')
str_2460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 11), 'str', 'ipow')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2459, str_2460))
# Adding element type (key, value) (line 8)
str_2461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 4), 'str', '>>=')
str_2462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 11), 'str', 'irshift')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2461, str_2462))
# Adding element type (key, value) (line 8)
str_2463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 4), 'str', 'is')
str_2464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 10), 'str', 'is_')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2463, str_2464))
# Adding element type (key, value) (line 8)
str_2465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 4), 'str', 'is not')
str_2466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 14), 'str', 'is_not')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2465, str_2466))
# Adding element type (key, value) (line 8)
str_2467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 4), 'str', '-=')
str_2468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 10), 'str', 'isub')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2467, str_2468))
# Adding element type (key, value) (line 8)
str_2469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 4), 'str', '/=')
str_2470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 10), 'str', 'itruediv')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2469, str_2470))
# Adding element type (key, value) (line 8)
str_2471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 4), 'str', '^=')
str_2472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 10), 'str', 'ixor')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2471, str_2472))
# Adding element type (key, value) (line 8)
str_2473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 4), 'str', '<=')
str_2474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 10), 'str', 'le')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2473, str_2474))
# Adding element type (key, value) (line 8)
str_2475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 4), 'str', '<<')
str_2476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 10), 'str', 'lshift')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2475, str_2476))
# Adding element type (key, value) (line 8)
str_2477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 4), 'str', '<')
str_2478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 9), 'str', 'lt')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2477, str_2478))
# Adding element type (key, value) (line 8)
str_2479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 4), 'str', '%')
str_2480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 9), 'str', 'mod')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2479, str_2480))
# Adding element type (key, value) (line 8)
str_2481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 4), 'str', '*')
str_2482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 9), 'str', 'mul')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2481, str_2482))
# Adding element type (key, value) (line 8)
str_2483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 4), 'str', '!=')
str_2484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 10), 'str', 'ne')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2483, str_2484))
# Adding element type (key, value) (line 8)
str_2485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 4), 'str', '-')
str_2486 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 9), 'str', 'sub')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2485, str_2486))
# Adding element type (key, value) (line 8)
str_2487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 4), 'str', '/')
str_2488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 9), 'str', 'truediv')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2487, str_2488))
# Adding element type (key, value) (line 8)
str_2489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 4), 'str', '^')
str_2490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 9), 'str', 'xor')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2489, str_2490))
# Adding element type (key, value) (line 8)
str_2491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 4), 'str', '|')
str_2492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 9), 'str', 'or_')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2491, str_2492))
# Adding element type (key, value) (line 8)
str_2493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 4), 'str', 'mult')
str_2494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 12), 'str', 'mul')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2493, str_2494))
# Adding element type (key, value) (line 8)
str_2495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 4), 'str', 'and')
str_2496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 11), 'str', 'and_')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2495, str_2496))
# Adding element type (key, value) (line 8)
str_2497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 4), 'str', 'not')
str_2498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 11), 'str', 'not_')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2497, str_2498))
# Adding element type (key, value) (line 8)
str_2499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 4), 'str', 'or')
str_2500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 10), 'str', 'or_')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2499, str_2500))
# Adding element type (key, value) (line 8)
str_2501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 4), 'str', 'div')
str_2502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 11), 'str', 'div')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2501, str_2502))
# Adding element type (key, value) (line 8)
str_2503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 4), 'str', 'isnot')
str_2504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 13), 'str', 'is_not')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2503, str_2504))
# Adding element type (key, value) (line 8)
str_2505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 4), 'str', '**')
str_2506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 10), 'str', 'pow')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2505, str_2506))
# Adding element type (key, value) (line 8)
str_2507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 4), 'str', 'notin')
str_2508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 13), 'str', 'contains')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2426, (str_2507, str_2508))

# Assigning a type to the variable '__symbol_to_operator_table' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), '__symbol_to_operator_table', dict_2426)

# Assigning a Dict to a Name (line 53):

# Obtaining an instance of the builtin type 'dict' (line 53)
dict_2509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 29), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 53)
# Adding element type (key, value) (line 53)
str_2510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 4), 'str', 'lte')
str_2511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 11), 'str', '<=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2510, str_2511))
# Adding element type (key, value) (line 53)
str_2512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 4), 'str', 'gte')
str_2513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 11), 'str', '>=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2512, str_2513))
# Adding element type (key, value) (line 53)
str_2514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 4), 'str', 'eq')
str_2515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 10), 'str', '==')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2514, str_2515))
# Adding element type (key, value) (line 53)
str_2516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 4), 'str', 'is_')
str_2517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 11), 'str', 'is')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2516, str_2517))
# Adding element type (key, value) (line 53)
str_2518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 4), 'str', 'ior')
str_2519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 11), 'str', '|=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2518, str_2519))
# Adding element type (key, value) (line 53)
str_2520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 4), 'str', 'iand')
str_2521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 12), 'str', '&=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2520, str_2521))
# Adding element type (key, value) (line 53)
str_2522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 4), 'str', 'getitem')
str_2523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 15), 'str', '[]')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2522, str_2523))
# Adding element type (key, value) (line 53)
str_2524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 4), 'str', 'imod')
str_2525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 12), 'str', '%=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2524, str_2525))
# Adding element type (key, value) (line 53)
str_2526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 4), 'str', 'not_')
str_2527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 12), 'str', 'not')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2526, str_2527))
# Adding element type (key, value) (line 53)
str_2528 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 4), 'str', 'xor')
str_2529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 11), 'str', '^')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2528, str_2529))
# Adding element type (key, value) (line 53)
str_2530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 4), 'str', 'contains')
str_2531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 16), 'str', 'in')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2530, str_2531))
# Adding element type (key, value) (line 53)
str_2532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 4), 'str', 'ifloordiv')
str_2533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 17), 'str', '//=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2532, str_2533))
# Adding element type (key, value) (line 53)
str_2534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 4), 'str', 'noteq')
str_2535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 13), 'str', '!=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2534, str_2535))
# Adding element type (key, value) (line 53)
str_2536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 4), 'str', 'is_not')
str_2537 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 14), 'str', 'isnot')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2536, str_2537))
# Adding element type (key, value) (line 53)
str_2538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 4), 'str', 'floordiv')
str_2539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 16), 'str', '//')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2538, str_2539))
# Adding element type (key, value) (line 53)
str_2540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 4), 'str', 'mod')
str_2541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 11), 'str', '%')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2540, str_2541))
# Adding element type (key, value) (line 53)
str_2542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 4), 'str', 'ixor')
str_2543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 12), 'str', '^=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2542, str_2543))
# Adding element type (key, value) (line 53)
str_2544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 4), 'str', 'ilshift')
str_2545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 15), 'str', '<<=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2544, str_2545))
# Adding element type (key, value) (line 53)
str_2546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 4), 'str', 'and_')
str_2547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 12), 'str', '&')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2546, str_2547))
# Adding element type (key, value) (line 53)
str_2548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 4), 'str', 'add')
str_2549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 11), 'str', '+')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2548, str_2549))
# Adding element type (key, value) (line 53)
str_2550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 4), 'str', 'mul')
str_2551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 11), 'str', '*')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2550, str_2551))
# Adding element type (key, value) (line 53)
str_2552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 4), 'str', 'mult')
str_2553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 12), 'str', '*')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2552, str_2553))
# Adding element type (key, value) (line 53)
str_2554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 4), 'str', 'sub')
str_2555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 11), 'str', '-')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2554, str_2555))
# Adding element type (key, value) (line 53)
str_2556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 4), 'str', 'itruediv')
str_2557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 16), 'str', '/=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2556, str_2557))
# Adding element type (key, value) (line 53)
str_2558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 4), 'str', 'truediv')
str_2559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 15), 'str', '/')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2558, str_2559))
# Adding element type (key, value) (line 53)
str_2560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 4), 'str', 'div')
str_2561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 11), 'str', 'div')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2560, str_2561))
# Adding element type (key, value) (line 53)
str_2562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 4), 'str', 'lt')
str_2563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 10), 'str', '<')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2562, str_2563))
# Adding element type (key, value) (line 53)
str_2564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 4), 'str', 'irshift')
str_2565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 15), 'str', '>>=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2564, str_2565))
# Adding element type (key, value) (line 53)
str_2566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 4), 'str', 'isub')
str_2567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 12), 'str', '-=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2566, str_2567))
# Adding element type (key, value) (line 53)
str_2568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 4), 'str', 'inv')
str_2569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 11), 'str', '~')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2568, str_2569))
# Adding element type (key, value) (line 53)
str_2570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 4), 'str', 'lshift')
str_2571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 14), 'str', '<<')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2570, str_2571))
# Adding element type (key, value) (line 53)
str_2572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 4), 'str', 'iadd')
str_2573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 12), 'str', '+=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2572, str_2573))
# Adding element type (key, value) (line 53)
str_2574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 4), 'str', 'gt')
str_2575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 10), 'str', '>')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2574, str_2575))
# Adding element type (key, value) (line 53)
str_2576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 4), 'str', 'pow')
str_2577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 11), 'str', '**')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2576, str_2577))
# Adding element type (key, value) (line 53)
str_2578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 4), 'str', 'bitor')
str_2579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 13), 'str', '|')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2578, str_2579))
# Adding element type (key, value) (line 53)
str_2580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 4), 'str', 'bitand')
str_2581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 14), 'str', '&')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2580, str_2581))
# Adding element type (key, value) (line 53)
str_2582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 4), 'str', 'bitxor')
str_2583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 14), 'str', '^')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2582, str_2583))
# Adding element type (key, value) (line 53)
str_2584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 4), 'str', 'invert')
str_2585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 14), 'str', '~')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2509, (str_2584, str_2585))

# Assigning a type to the variable '__operator_to_symbol_table' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), '__operator_to_symbol_table', dict_2509)

@norecursion
def operator_name_to_symbol(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'operator_name_to_symbol'
    module_type_store = module_type_store.open_function_context('operator_name_to_symbol', 97, 0, False)
    
    # Passed parameters checking function
    operator_name_to_symbol.stypy_localization = localization
    operator_name_to_symbol.stypy_type_of_self = None
    operator_name_to_symbol.stypy_type_store = module_type_store
    operator_name_to_symbol.stypy_function_name = 'operator_name_to_symbol'
    operator_name_to_symbol.stypy_param_names_list = ['operator_name']
    operator_name_to_symbol.stypy_varargs_param_name = None
    operator_name_to_symbol.stypy_kwargs_param_name = None
    operator_name_to_symbol.stypy_call_defaults = defaults
    operator_name_to_symbol.stypy_call_varargs = varargs
    operator_name_to_symbol.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'operator_name_to_symbol', ['operator_name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'operator_name_to_symbol', localization, ['operator_name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'operator_name_to_symbol(...)' code ##################

    str_2586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, (-1)), 'str', "\n    Transform an operator name to its symbolic representation (example: 'add' -> '+'. If no symbol is available, return\n    the passed name\n    :param operator_name: Operator name\n    :return: Operator symbol\n    ")
    
    
    # SSA begins for try-except statement (line 104)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Obtaining the type of the subscript
    # Getting the type of 'operator_name' (line 105)
    operator_name_2587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 42), 'operator_name')
    # Getting the type of '__operator_to_symbol_table' (line 105)
    operator_to_symbol_table_2588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 15), '__operator_to_symbol_table')
    # Obtaining the member '__getitem__' of a type (line 105)
    getitem___2589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 15), operator_to_symbol_table_2588, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 105)
    subscript_call_result_2590 = invoke(stypy.reporting.localization.Localization(__file__, 105, 15), getitem___2589, operator_name_2587)
    
    # Assigning a type to the variable 'stypy_return_type' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'stypy_return_type', subscript_call_result_2590)
    # SSA branch for the except part of a try statement (line 104)
    # SSA branch for the except 'KeyError' branch of a try statement (line 104)
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'operator_name' (line 107)
    operator_name_2591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 15), 'operator_name')
    # Assigning a type to the variable 'stypy_return_type' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'stypy_return_type', operator_name_2591)
    # SSA join for try-except statement (line 104)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'operator_name_to_symbol(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'operator_name_to_symbol' in the type store
    # Getting the type of 'stypy_return_type' (line 97)
    stypy_return_type_2592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2592)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'operator_name_to_symbol'
    return stypy_return_type_2592

# Assigning a type to the variable 'operator_name_to_symbol' (line 97)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 0), 'operator_name_to_symbol', operator_name_to_symbol)

@norecursion
def operator_symbol_to_name(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'operator_symbol_to_name'
    module_type_store = module_type_store.open_function_context('operator_symbol_to_name', 110, 0, False)
    
    # Passed parameters checking function
    operator_symbol_to_name.stypy_localization = localization
    operator_symbol_to_name.stypy_type_of_self = None
    operator_symbol_to_name.stypy_type_store = module_type_store
    operator_symbol_to_name.stypy_function_name = 'operator_symbol_to_name'
    operator_symbol_to_name.stypy_param_names_list = ['operator_symbol']
    operator_symbol_to_name.stypy_varargs_param_name = None
    operator_symbol_to_name.stypy_kwargs_param_name = None
    operator_symbol_to_name.stypy_call_defaults = defaults
    operator_symbol_to_name.stypy_call_varargs = varargs
    operator_symbol_to_name.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'operator_symbol_to_name', ['operator_symbol'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'operator_symbol_to_name', localization, ['operator_symbol'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'operator_symbol_to_name(...)' code ##################

    str_2593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, (-1)), 'str', "\n    Transform an operator symbol to its function name (example: '+' -> 'add'.\n    :param operator_symbol: Operator symbol\n    :return: Operator name\n    ")
    
    # Obtaining the type of the subscript
    # Getting the type of 'operator_symbol' (line 116)
    operator_symbol_2594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 38), 'operator_symbol')
    # Getting the type of '__symbol_to_operator_table' (line 116)
    symbol_to_operator_table_2595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 11), '__symbol_to_operator_table')
    # Obtaining the member '__getitem__' of a type (line 116)
    getitem___2596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 11), symbol_to_operator_table_2595, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 116)
    subscript_call_result_2597 = invoke(stypy.reporting.localization.Localization(__file__, 116, 11), getitem___2596, operator_symbol_2594)
    
    # Assigning a type to the variable 'stypy_return_type' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'stypy_return_type', subscript_call_result_2597)
    
    # ################# End of 'operator_symbol_to_name(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'operator_symbol_to_name' in the type store
    # Getting the type of 'stypy_return_type' (line 110)
    stypy_return_type_2598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2598)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'operator_symbol_to_name'
    return stypy_return_type_2598

# Assigning a type to the variable 'operator_symbol_to_name' (line 110)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 0), 'operator_symbol_to_name', operator_symbol_to_name)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
