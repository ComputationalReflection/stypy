
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
dict_2712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 29), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 8)
# Adding element type (key, value) (line 8)
str_2713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 4), 'str', '+')
str_2714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 9), 'str', 'add')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2713, str_2714))
# Adding element type (key, value) (line 8)
str_2715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 4), 'str', '&')
str_2716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 9), 'str', 'and_')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2715, str_2716))
# Adding element type (key, value) (line 8)
str_2717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 4), 'str', 'in')
str_2718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 10), 'str', 'contains')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2717, str_2718))
# Adding element type (key, value) (line 8)
str_2719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 4), 'str', '==')
str_2720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 10), 'str', 'eq')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2719, str_2720))
# Adding element type (key, value) (line 8)
str_2721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 4), 'str', '//')
str_2722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 10), 'str', 'floordiv')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2721, str_2722))
# Adding element type (key, value) (line 8)
str_2723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 4), 'str', '>=')
str_2724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 10), 'str', 'ge')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2723, str_2724))
# Adding element type (key, value) (line 8)
str_2725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 4), 'str', '[]')
str_2726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 10), 'str', 'getitem')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2725, str_2726))
# Adding element type (key, value) (line 8)
str_2727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 4), 'str', '>')
str_2728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 9), 'str', 'gt')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2727, str_2728))
# Adding element type (key, value) (line 8)
str_2729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 4), 'str', '+=')
str_2730 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 10), 'str', 'iadd')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2729, str_2730))
# Adding element type (key, value) (line 8)
str_2731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 4), 'str', '&=')
str_2732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 10), 'str', 'iand')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2731, str_2732))
# Adding element type (key, value) (line 8)
str_2733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 4), 'str', '//=')
str_2734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 11), 'str', 'ifloordiv')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2733, str_2734))
# Adding element type (key, value) (line 8)
str_2735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 4), 'str', '<<=')
str_2736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 11), 'str', 'ilshift')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2735, str_2736))
# Adding element type (key, value) (line 8)
str_2737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 4), 'str', '%=')
str_2738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 10), 'str', 'imod')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2737, str_2738))
# Adding element type (key, value) (line 8)
str_2739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 4), 'str', '*=')
str_2740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 10), 'str', 'imul')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2739, str_2740))
# Adding element type (key, value) (line 8)
str_2741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 4), 'str', '~')
str_2742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 9), 'str', 'inv')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2741, str_2742))
# Adding element type (key, value) (line 8)
str_2743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 4), 'str', '|=')
str_2744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 10), 'str', 'ior')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2743, str_2744))
# Adding element type (key, value) (line 8)
str_2745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 4), 'str', '**=')
str_2746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 11), 'str', 'ipow')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2745, str_2746))
# Adding element type (key, value) (line 8)
str_2747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 4), 'str', '>>=')
str_2748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 11), 'str', 'irshift')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2747, str_2748))
# Adding element type (key, value) (line 8)
str_2749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 4), 'str', 'is')
str_2750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 10), 'str', 'is_')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2749, str_2750))
# Adding element type (key, value) (line 8)
str_2751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 4), 'str', 'is not')
str_2752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 14), 'str', 'is_not')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2751, str_2752))
# Adding element type (key, value) (line 8)
str_2753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 4), 'str', '-=')
str_2754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 10), 'str', 'isub')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2753, str_2754))
# Adding element type (key, value) (line 8)
str_2755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 4), 'str', '/=')
str_2756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 10), 'str', 'itruediv')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2755, str_2756))
# Adding element type (key, value) (line 8)
str_2757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 4), 'str', '^=')
str_2758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 10), 'str', 'ixor')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2757, str_2758))
# Adding element type (key, value) (line 8)
str_2759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 4), 'str', '<=')
str_2760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 10), 'str', 'le')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2759, str_2760))
# Adding element type (key, value) (line 8)
str_2761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 4), 'str', '<<')
str_2762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 10), 'str', 'lshift')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2761, str_2762))
# Adding element type (key, value) (line 8)
str_2763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 4), 'str', '<')
str_2764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 9), 'str', 'lt')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2763, str_2764))
# Adding element type (key, value) (line 8)
str_2765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 4), 'str', '%')
str_2766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 9), 'str', 'mod')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2765, str_2766))
# Adding element type (key, value) (line 8)
str_2767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 4), 'str', '*')
str_2768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 9), 'str', 'mul')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2767, str_2768))
# Adding element type (key, value) (line 8)
str_2769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 4), 'str', '!=')
str_2770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 10), 'str', 'ne')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2769, str_2770))
# Adding element type (key, value) (line 8)
str_2771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 4), 'str', '-')
str_2772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 9), 'str', 'sub')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2771, str_2772))
# Adding element type (key, value) (line 8)
str_2773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 4), 'str', '/')
str_2774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 9), 'str', 'truediv')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2773, str_2774))
# Adding element type (key, value) (line 8)
str_2775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 4), 'str', '^')
str_2776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 9), 'str', 'xor')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2775, str_2776))
# Adding element type (key, value) (line 8)
str_2777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 4), 'str', '|')
str_2778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 9), 'str', 'or_')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2777, str_2778))
# Adding element type (key, value) (line 8)
str_2779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 4), 'str', 'mult')
str_2780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 12), 'str', 'mul')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2779, str_2780))
# Adding element type (key, value) (line 8)
str_2781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 4), 'str', 'and')
str_2782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 11), 'str', 'and_')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2781, str_2782))
# Adding element type (key, value) (line 8)
str_2783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 4), 'str', 'not')
str_2784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 11), 'str', 'not_')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2783, str_2784))
# Adding element type (key, value) (line 8)
str_2785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 4), 'str', 'or')
str_2786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 10), 'str', 'or_')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2785, str_2786))
# Adding element type (key, value) (line 8)
str_2787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 4), 'str', 'div')
str_2788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 11), 'str', 'div')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2787, str_2788))
# Adding element type (key, value) (line 8)
str_2789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 4), 'str', 'isnot')
str_2790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 13), 'str', 'is_not')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2789, str_2790))
# Adding element type (key, value) (line 8)
str_2791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 4), 'str', '**')
str_2792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 10), 'str', 'pow')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2791, str_2792))
# Adding element type (key, value) (line 8)
str_2793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 4), 'str', 'notin')
str_2794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 13), 'str', 'contains')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 29), dict_2712, (str_2793, str_2794))

# Assigning a type to the variable '__symbol_to_operator_table' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), '__symbol_to_operator_table', dict_2712)

# Assigning a Dict to a Name (line 53):

# Obtaining an instance of the builtin type 'dict' (line 53)
dict_2795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 29), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 53)
# Adding element type (key, value) (line 53)
str_2796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 4), 'str', 'lte')
str_2797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 11), 'str', '<=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2796, str_2797))
# Adding element type (key, value) (line 53)
str_2798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 4), 'str', 'gte')
str_2799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 11), 'str', '>=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2798, str_2799))
# Adding element type (key, value) (line 53)
str_2800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 4), 'str', 'eq')
str_2801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 10), 'str', '==')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2800, str_2801))
# Adding element type (key, value) (line 53)
str_2802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 4), 'str', 'is_')
str_2803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 11), 'str', 'is')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2802, str_2803))
# Adding element type (key, value) (line 53)
str_2804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 4), 'str', 'ior')
str_2805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 11), 'str', '|=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2804, str_2805))
# Adding element type (key, value) (line 53)
str_2806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 4), 'str', 'iand')
str_2807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 12), 'str', '&=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2806, str_2807))
# Adding element type (key, value) (line 53)
str_2808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 4), 'str', 'getitem')
str_2809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 15), 'str', '[]')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2808, str_2809))
# Adding element type (key, value) (line 53)
str_2810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 4), 'str', 'imod')
str_2811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 12), 'str', '%=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2810, str_2811))
# Adding element type (key, value) (line 53)
str_2812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 4), 'str', 'not_')
str_2813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 12), 'str', 'not')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2812, str_2813))
# Adding element type (key, value) (line 53)
str_2814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 4), 'str', 'xor')
str_2815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 11), 'str', '^')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2814, str_2815))
# Adding element type (key, value) (line 53)
str_2816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 4), 'str', 'contains')
str_2817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 16), 'str', 'in')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2816, str_2817))
# Adding element type (key, value) (line 53)
str_2818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 4), 'str', 'ifloordiv')
str_2819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 17), 'str', '//=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2818, str_2819))
# Adding element type (key, value) (line 53)
str_2820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 4), 'str', 'noteq')
str_2821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 13), 'str', '!=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2820, str_2821))
# Adding element type (key, value) (line 53)
str_2822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 4), 'str', 'is_not')
str_2823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 14), 'str', 'isnot')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2822, str_2823))
# Adding element type (key, value) (line 53)
str_2824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 4), 'str', 'floordiv')
str_2825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 16), 'str', '//')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2824, str_2825))
# Adding element type (key, value) (line 53)
str_2826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 4), 'str', 'mod')
str_2827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 11), 'str', '%')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2826, str_2827))
# Adding element type (key, value) (line 53)
str_2828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 4), 'str', 'ixor')
str_2829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 12), 'str', '^=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2828, str_2829))
# Adding element type (key, value) (line 53)
str_2830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 4), 'str', 'ilshift')
str_2831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 15), 'str', '<<=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2830, str_2831))
# Adding element type (key, value) (line 53)
str_2832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 4), 'str', 'and_')
str_2833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 12), 'str', '&')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2832, str_2833))
# Adding element type (key, value) (line 53)
str_2834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 4), 'str', 'add')
str_2835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 11), 'str', '+')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2834, str_2835))
# Adding element type (key, value) (line 53)
str_2836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 4), 'str', 'mul')
str_2837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 11), 'str', '*')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2836, str_2837))
# Adding element type (key, value) (line 53)
str_2838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 4), 'str', 'mult')
str_2839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 12), 'str', '*')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2838, str_2839))
# Adding element type (key, value) (line 53)
str_2840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 4), 'str', 'sub')
str_2841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 11), 'str', '-')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2840, str_2841))
# Adding element type (key, value) (line 53)
str_2842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 4), 'str', 'itruediv')
str_2843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 16), 'str', '/=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2842, str_2843))
# Adding element type (key, value) (line 53)
str_2844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 4), 'str', 'truediv')
str_2845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 15), 'str', '/')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2844, str_2845))
# Adding element type (key, value) (line 53)
str_2846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 4), 'str', 'div')
str_2847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 11), 'str', 'div')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2846, str_2847))
# Adding element type (key, value) (line 53)
str_2848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 4), 'str', 'lt')
str_2849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 10), 'str', '<')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2848, str_2849))
# Adding element type (key, value) (line 53)
str_2850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 4), 'str', 'irshift')
str_2851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 15), 'str', '>>=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2850, str_2851))
# Adding element type (key, value) (line 53)
str_2852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 4), 'str', 'isub')
str_2853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 12), 'str', '-=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2852, str_2853))
# Adding element type (key, value) (line 53)
str_2854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 4), 'str', 'inv')
str_2855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 11), 'str', '~')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2854, str_2855))
# Adding element type (key, value) (line 53)
str_2856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 4), 'str', 'lshift')
str_2857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 14), 'str', '<<')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2856, str_2857))
# Adding element type (key, value) (line 53)
str_2858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 4), 'str', 'iadd')
str_2859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 12), 'str', '+=')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2858, str_2859))
# Adding element type (key, value) (line 53)
str_2860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 4), 'str', 'gt')
str_2861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 10), 'str', '>')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2860, str_2861))
# Adding element type (key, value) (line 53)
str_2862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 4), 'str', 'pow')
str_2863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 11), 'str', '**')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2862, str_2863))
# Adding element type (key, value) (line 53)
str_2864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 4), 'str', 'bitor')
str_2865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 13), 'str', '|')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2864, str_2865))
# Adding element type (key, value) (line 53)
str_2866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 4), 'str', 'bitand')
str_2867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 14), 'str', '&')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2866, str_2867))
# Adding element type (key, value) (line 53)
str_2868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 4), 'str', 'bitxor')
str_2869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 14), 'str', '^')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2868, str_2869))
# Adding element type (key, value) (line 53)
str_2870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 4), 'str', 'invert')
str_2871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 14), 'str', '~')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 29), dict_2795, (str_2870, str_2871))

# Assigning a type to the variable '__operator_to_symbol_table' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), '__operator_to_symbol_table', dict_2795)

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

    str_2872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, (-1)), 'str', "\n    Transform an operator name to its symbolic representation (example: 'add' -> '+'. If no symbol is available, return\n    the passed name\n    :param operator_name: Operator name\n    :return: Operator symbol\n    ")
    
    
    # SSA begins for try-except statement (line 104)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Obtaining the type of the subscript
    # Getting the type of 'operator_name' (line 105)
    operator_name_2873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 42), 'operator_name')
    # Getting the type of '__operator_to_symbol_table' (line 105)
    operator_to_symbol_table_2874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 15), '__operator_to_symbol_table')
    # Obtaining the member '__getitem__' of a type (line 105)
    getitem___2875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 15), operator_to_symbol_table_2874, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 105)
    subscript_call_result_2876 = invoke(stypy.reporting.localization.Localization(__file__, 105, 15), getitem___2875, operator_name_2873)
    
    # Assigning a type to the variable 'stypy_return_type' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'stypy_return_type', subscript_call_result_2876)
    # SSA branch for the except part of a try statement (line 104)
    # SSA branch for the except 'KeyError' branch of a try statement (line 104)
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'operator_name' (line 107)
    operator_name_2877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 15), 'operator_name')
    # Assigning a type to the variable 'stypy_return_type' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'stypy_return_type', operator_name_2877)
    # SSA join for try-except statement (line 104)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'operator_name_to_symbol(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'operator_name_to_symbol' in the type store
    # Getting the type of 'stypy_return_type' (line 97)
    stypy_return_type_2878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2878)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'operator_name_to_symbol'
    return stypy_return_type_2878

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

    str_2879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, (-1)), 'str', "\n    Transform an operator symbol to its function name (example: '+' -> 'add'.\n    :param operator_symbol: Operator symbol\n    :return: Operator name\n    ")
    
    # Obtaining the type of the subscript
    # Getting the type of 'operator_symbol' (line 116)
    operator_symbol_2880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 38), 'operator_symbol')
    # Getting the type of '__symbol_to_operator_table' (line 116)
    symbol_to_operator_table_2881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 11), '__symbol_to_operator_table')
    # Obtaining the member '__getitem__' of a type (line 116)
    getitem___2882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 11), symbol_to_operator_table_2881, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 116)
    subscript_call_result_2883 = invoke(stypy.reporting.localization.Localization(__file__, 116, 11), getitem___2882, operator_symbol_2880)
    
    # Assigning a type to the variable 'stypy_return_type' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'stypy_return_type', subscript_call_result_2883)
    
    # ################# End of 'operator_symbol_to_name(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'operator_symbol_to_name' in the type store
    # Getting the type of 'stypy_return_type' (line 110)
    stypy_return_type_2884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2884)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'operator_symbol_to_name'
    return stypy_return_type_2884

# Assigning a type to the variable 'operator_symbol_to_name' (line 110)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 0), 'operator_symbol_to_name', operator_symbol_to_name)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
