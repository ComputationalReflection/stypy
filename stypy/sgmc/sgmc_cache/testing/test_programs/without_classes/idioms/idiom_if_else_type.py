
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: theInt = 3
2: theStr = "hi"
3: if True:
4:     union = 3
5: else:
6:     union = "hi"
7: 
8: def if_else_base1(a):
9:     b = "hi"
10:     if type(a) is int:
11:         r = a / 3
12:         r2 = a[0]
13:         b = 3
14:     else:
15:         r3 = a[0]
16:         r4 = a / 3
17:         b = "bye"
18:     r5 = a / 3
19:     r6 = b / 3
20: 
21: def if_else_base2(a):
22:     b = "hi"
23:     if type(a) is int:
24:         r = a / 3
25:         r2 = a[0]
26:         b = 3
27:     else:
28:         r3 = a[0]
29:         r4 = a / 3
30:         b = "bye"
31:     r5 = a / 3
32:     r6 = b / 3
33: 
34: def if_else_base3(a):
35:     b = "hi"
36:     if type(a) is int:
37:         r = a / 3
38:         r2 = a[0]
39:         b = 3
40:     else:
41:         r3 = a[0]
42:         r4 = a / 3
43:         b = "bye"
44:     r5 = a / 3
45:     r6 = b / 3
46: 
47: def if_else_base4(a):
48:     b = "hi"
49:     if type(a) is int:
50:         r = a / 3
51:         r2 = a[0]
52:         b = 3
53:     else:
54:         r3 = a[0]
55:         r4 = a / 3
56:         b = "bye"
57:     r5 = a / 3
58:     r6 = b / 3
59: 
60: 
61: bigUnion = int() if True else str() if False else False
62: 
63: if_else_base1(theInt)
64: if_else_base2(theStr)
65: if_else_base3(union)
66: if_else_base4(bigUnion)
67: 
68: def simple_if_else_idiom_variant(a):
69:     b = "hi"
70:     if type(a) is type(3):
71:         r = a / 3
72:         r2 = a[0]
73:         b = 3
74:     else:
75:         r3 = a[0]
76:         r4 = a / 3
77:         b = "bye"
78: 
79:     r5 = a / 3
80:     r6 = b / 3
81: 
82: simple_if_else_idiom_variant(union)
83: 
84: def simple_if_else_not_idiom(a):
85:     b = "hi"
86:     if type(a) is 3:
87:         r = a / 3
88:         r2 = a[0]
89:         b = 3
90:     else:
91:         r3 = a[0]
92:         r4 = a / 3
93:         b = "bye"
94: 
95:     r5 = a / 3
96:     r6 = b / 3
97: 
98: simple_if_else_not_idiom(union)
99: 
100: class Foo:
101:     def __init__(self):
102:         self.attr = 4
103:         self.strattr = "bar"
104: 
105: def simple_if_else_idiom_attr(a):
106:     b = "hi"
107:     if type(a.attr) is int:
108:         r = a.attr / 3
109:         r2 = a.attr[0]
110:         b = 3
111:     else:
112:         r3 = a[0]
113:         r4 = a / 3
114:         b = "bye"
115: 
116:     r5 = a.attr / 3
117:     r6 = b / 3
118: 
119: def simple_if_else_idiom_attr_b(a):
120:     b = "hi"
121:     if type(a.strattr) is int:
122:         r = a.attr / 3
123:         r2 = a.strattr[0]
124:         b = 3
125:     else:
126:         r3 = a[0]
127:         r4 = a / 3
128:         b = "bye"
129: 
130:     r3 = a.strattr / 3
131:     r4 = b / 3
132: 
133: simple_if_else_idiom_attr(Foo())
134: simple_if_else_idiom_attr_b(Foo())

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Num to a Name (line 1):
int_2775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 9), 'int')
# Assigning a type to the variable 'theInt' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'theInt', int_2775)

# Assigning a Str to a Name (line 2):
str_2776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 9), 'str', 'hi')
# Assigning a type to the variable 'theStr' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'theStr', str_2776)

# Getting the type of 'True' (line 3)
True_2777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 3), 'True')
# Testing the type of an if condition (line 3)
if_condition_2778 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 3, 0), True_2777)
# Assigning a type to the variable 'if_condition_2778' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'if_condition_2778', if_condition_2778)
# SSA begins for if statement (line 3)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Num to a Name (line 4):
int_2779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 12), 'int')
# Assigning a type to the variable 'union' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 4), 'union', int_2779)
# SSA branch for the else part of an if statement (line 3)
module_type_store.open_ssa_branch('else')

# Assigning a Str to a Name (line 6):
str_2780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 12), 'str', 'hi')
# Assigning a type to the variable 'union' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'union', str_2780)
# SSA join for if statement (line 3)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def if_else_base1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'if_else_base1'
    module_type_store = module_type_store.open_function_context('if_else_base1', 8, 0, False)
    
    # Passed parameters checking function
    if_else_base1.stypy_localization = localization
    if_else_base1.stypy_type_of_self = None
    if_else_base1.stypy_type_store = module_type_store
    if_else_base1.stypy_function_name = 'if_else_base1'
    if_else_base1.stypy_param_names_list = ['a']
    if_else_base1.stypy_varargs_param_name = None
    if_else_base1.stypy_kwargs_param_name = None
    if_else_base1.stypy_call_defaults = defaults
    if_else_base1.stypy_call_varargs = varargs
    if_else_base1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'if_else_base1', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'if_else_base1', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'if_else_base1(...)' code ##################

    
    # Assigning a Str to a Name (line 9):
    str_2781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'b', str_2781)
    
    # Type idiom detected: calculating its left and rigth part (line 10)
    # Getting the type of 'a' (line 10)
    a_2782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 12), 'a')
    # Getting the type of 'int' (line 10)
    int_2783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 18), 'int')
    
    (may_be_2784, more_types_in_union_2785) = may_be_type(a_2782, int_2783)

    if may_be_2784:

        if more_types_in_union_2785:
            # Runtime conditional SSA (line 10)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'a' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'a', int_2783())
        
        # Assigning a BinOp to a Name (line 11):
        # Getting the type of 'a' (line 11)
        a_2786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 12), 'a')
        int_2787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 16), 'int')
        # Applying the binary operator 'div' (line 11)
        result_div_2788 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 12), 'div', a_2786, int_2787)
        
        # Assigning a type to the variable 'r' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'r', result_div_2788)
        
        # Assigning a Subscript to a Name (line 12):
        
        # Obtaining the type of the subscript
        int_2789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 15), 'int')
        # Getting the type of 'a' (line 12)
        a_2790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 12)
        getitem___2791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 13), a_2790, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 12)
        subscript_call_result_2792 = invoke(stypy.reporting.localization.Localization(__file__, 12, 13), getitem___2791, int_2789)
        
        # Assigning a type to the variable 'r2' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'r2', subscript_call_result_2792)
        
        # Assigning a Num to a Name (line 13):
        int_2793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 12), 'int')
        # Assigning a type to the variable 'b' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'b', int_2793)

        if more_types_in_union_2785:
            # Runtime conditional SSA for else branch (line 10)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_2784) or more_types_in_union_2785):
        # Getting the type of 'a' (line 10)
        a_2794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'a')
        # Assigning a type to the variable 'a' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'a', remove_type_from_union(a_2794, int_2783))
        
        # Assigning a Subscript to a Name (line 15):
        
        # Obtaining the type of the subscript
        int_2795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'int')
        # Getting the type of 'a' (line 15)
        a_2796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 15)
        getitem___2797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 13), a_2796, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 15)
        subscript_call_result_2798 = invoke(stypy.reporting.localization.Localization(__file__, 15, 13), getitem___2797, int_2795)
        
        # Assigning a type to the variable 'r3' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'r3', subscript_call_result_2798)
        
        # Assigning a BinOp to a Name (line 16):
        # Getting the type of 'a' (line 16)
        a_2799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 13), 'a')
        int_2800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 17), 'int')
        # Applying the binary operator 'div' (line 16)
        result_div_2801 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 13), 'div', a_2799, int_2800)
        
        # Assigning a type to the variable 'r4' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'r4', result_div_2801)
        
        # Assigning a Str to a Name (line 17):
        str_2802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 12), 'str', 'bye')
        # Assigning a type to the variable 'b' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'b', str_2802)

        if (may_be_2784 and more_types_in_union_2785):
            # SSA join for if statement (line 10)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 18):
    # Getting the type of 'a' (line 18)
    a_2803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 9), 'a')
    int_2804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 13), 'int')
    # Applying the binary operator 'div' (line 18)
    result_div_2805 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 9), 'div', a_2803, int_2804)
    
    # Assigning a type to the variable 'r5' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'r5', result_div_2805)
    
    # Assigning a BinOp to a Name (line 19):
    # Getting the type of 'b' (line 19)
    b_2806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 9), 'b')
    int_2807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 13), 'int')
    # Applying the binary operator 'div' (line 19)
    result_div_2808 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 9), 'div', b_2806, int_2807)
    
    # Assigning a type to the variable 'r6' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'r6', result_div_2808)
    
    # ################# End of 'if_else_base1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'if_else_base1' in the type store
    # Getting the type of 'stypy_return_type' (line 8)
    stypy_return_type_2809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2809)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'if_else_base1'
    return stypy_return_type_2809

# Assigning a type to the variable 'if_else_base1' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'if_else_base1', if_else_base1)

@norecursion
def if_else_base2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'if_else_base2'
    module_type_store = module_type_store.open_function_context('if_else_base2', 21, 0, False)
    
    # Passed parameters checking function
    if_else_base2.stypy_localization = localization
    if_else_base2.stypy_type_of_self = None
    if_else_base2.stypy_type_store = module_type_store
    if_else_base2.stypy_function_name = 'if_else_base2'
    if_else_base2.stypy_param_names_list = ['a']
    if_else_base2.stypy_varargs_param_name = None
    if_else_base2.stypy_kwargs_param_name = None
    if_else_base2.stypy_call_defaults = defaults
    if_else_base2.stypy_call_varargs = varargs
    if_else_base2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'if_else_base2', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'if_else_base2', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'if_else_base2(...)' code ##################

    
    # Assigning a Str to a Name (line 22):
    str_2810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'b', str_2810)
    
    # Type idiom detected: calculating its left and rigth part (line 23)
    # Getting the type of 'a' (line 23)
    a_2811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'a')
    # Getting the type of 'int' (line 23)
    int_2812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 18), 'int')
    
    (may_be_2813, more_types_in_union_2814) = may_be_type(a_2811, int_2812)

    if may_be_2813:

        if more_types_in_union_2814:
            # Runtime conditional SSA (line 23)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'a' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'a', int_2812())
        
        # Assigning a BinOp to a Name (line 24):
        # Getting the type of 'a' (line 24)
        a_2815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'a')
        int_2816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 16), 'int')
        # Applying the binary operator 'div' (line 24)
        result_div_2817 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 12), 'div', a_2815, int_2816)
        
        # Assigning a type to the variable 'r' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'r', result_div_2817)
        
        # Assigning a Subscript to a Name (line 25):
        
        # Obtaining the type of the subscript
        int_2818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 15), 'int')
        # Getting the type of 'a' (line 25)
        a_2819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 25)
        getitem___2820 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 13), a_2819, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 25)
        subscript_call_result_2821 = invoke(stypy.reporting.localization.Localization(__file__, 25, 13), getitem___2820, int_2818)
        
        # Assigning a type to the variable 'r2' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'r2', subscript_call_result_2821)
        
        # Assigning a Num to a Name (line 26):
        int_2822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 12), 'int')
        # Assigning a type to the variable 'b' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'b', int_2822)

        if more_types_in_union_2814:
            # Runtime conditional SSA for else branch (line 23)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_2813) or more_types_in_union_2814):
        # Getting the type of 'a' (line 23)
        a_2823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'a')
        # Assigning a type to the variable 'a' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'a', remove_type_from_union(a_2823, int_2812))
        
        # Assigning a Subscript to a Name (line 28):
        
        # Obtaining the type of the subscript
        int_2824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 15), 'int')
        # Getting the type of 'a' (line 28)
        a_2825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 28)
        getitem___2826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 13), a_2825, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 28)
        subscript_call_result_2827 = invoke(stypy.reporting.localization.Localization(__file__, 28, 13), getitem___2826, int_2824)
        
        # Assigning a type to the variable 'r3' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'r3', subscript_call_result_2827)
        
        # Assigning a BinOp to a Name (line 29):
        # Getting the type of 'a' (line 29)
        a_2828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 13), 'a')
        int_2829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 17), 'int')
        # Applying the binary operator 'div' (line 29)
        result_div_2830 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 13), 'div', a_2828, int_2829)
        
        # Assigning a type to the variable 'r4' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'r4', result_div_2830)
        
        # Assigning a Str to a Name (line 30):
        str_2831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 12), 'str', 'bye')
        # Assigning a type to the variable 'b' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'b', str_2831)

        if (may_be_2813 and more_types_in_union_2814):
            # SSA join for if statement (line 23)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 31):
    # Getting the type of 'a' (line 31)
    a_2832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 9), 'a')
    int_2833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 13), 'int')
    # Applying the binary operator 'div' (line 31)
    result_div_2834 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 9), 'div', a_2832, int_2833)
    
    # Assigning a type to the variable 'r5' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'r5', result_div_2834)
    
    # Assigning a BinOp to a Name (line 32):
    # Getting the type of 'b' (line 32)
    b_2835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 9), 'b')
    int_2836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 13), 'int')
    # Applying the binary operator 'div' (line 32)
    result_div_2837 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 9), 'div', b_2835, int_2836)
    
    # Assigning a type to the variable 'r6' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'r6', result_div_2837)
    
    # ################# End of 'if_else_base2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'if_else_base2' in the type store
    # Getting the type of 'stypy_return_type' (line 21)
    stypy_return_type_2838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2838)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'if_else_base2'
    return stypy_return_type_2838

# Assigning a type to the variable 'if_else_base2' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'if_else_base2', if_else_base2)

@norecursion
def if_else_base3(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'if_else_base3'
    module_type_store = module_type_store.open_function_context('if_else_base3', 34, 0, False)
    
    # Passed parameters checking function
    if_else_base3.stypy_localization = localization
    if_else_base3.stypy_type_of_self = None
    if_else_base3.stypy_type_store = module_type_store
    if_else_base3.stypy_function_name = 'if_else_base3'
    if_else_base3.stypy_param_names_list = ['a']
    if_else_base3.stypy_varargs_param_name = None
    if_else_base3.stypy_kwargs_param_name = None
    if_else_base3.stypy_call_defaults = defaults
    if_else_base3.stypy_call_varargs = varargs
    if_else_base3.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'if_else_base3', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'if_else_base3', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'if_else_base3(...)' code ##################

    
    # Assigning a Str to a Name (line 35):
    str_2839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'b', str_2839)
    
    # Type idiom detected: calculating its left and rigth part (line 36)
    # Getting the type of 'a' (line 36)
    a_2840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'a')
    # Getting the type of 'int' (line 36)
    int_2841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 18), 'int')
    
    (may_be_2842, more_types_in_union_2843) = may_be_type(a_2840, int_2841)

    if may_be_2842:

        if more_types_in_union_2843:
            # Runtime conditional SSA (line 36)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'a' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'a', int_2841())
        
        # Assigning a BinOp to a Name (line 37):
        # Getting the type of 'a' (line 37)
        a_2844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'a')
        int_2845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 16), 'int')
        # Applying the binary operator 'div' (line 37)
        result_div_2846 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 12), 'div', a_2844, int_2845)
        
        # Assigning a type to the variable 'r' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'r', result_div_2846)
        
        # Assigning a Subscript to a Name (line 38):
        
        # Obtaining the type of the subscript
        int_2847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 15), 'int')
        # Getting the type of 'a' (line 38)
        a_2848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 38)
        getitem___2849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 13), a_2848, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 38)
        subscript_call_result_2850 = invoke(stypy.reporting.localization.Localization(__file__, 38, 13), getitem___2849, int_2847)
        
        # Assigning a type to the variable 'r2' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'r2', subscript_call_result_2850)
        
        # Assigning a Num to a Name (line 39):
        int_2851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 12), 'int')
        # Assigning a type to the variable 'b' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'b', int_2851)

        if more_types_in_union_2843:
            # Runtime conditional SSA for else branch (line 36)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_2842) or more_types_in_union_2843):
        # Getting the type of 'a' (line 36)
        a_2852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'a')
        # Assigning a type to the variable 'a' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'a', remove_type_from_union(a_2852, int_2841))
        
        # Assigning a Subscript to a Name (line 41):
        
        # Obtaining the type of the subscript
        int_2853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 15), 'int')
        # Getting the type of 'a' (line 41)
        a_2854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 41)
        getitem___2855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 13), a_2854, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 41)
        subscript_call_result_2856 = invoke(stypy.reporting.localization.Localization(__file__, 41, 13), getitem___2855, int_2853)
        
        # Assigning a type to the variable 'r3' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'r3', subscript_call_result_2856)
        
        # Assigning a BinOp to a Name (line 42):
        # Getting the type of 'a' (line 42)
        a_2857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 13), 'a')
        int_2858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 17), 'int')
        # Applying the binary operator 'div' (line 42)
        result_div_2859 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 13), 'div', a_2857, int_2858)
        
        # Assigning a type to the variable 'r4' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'r4', result_div_2859)
        
        # Assigning a Str to a Name (line 43):
        str_2860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 12), 'str', 'bye')
        # Assigning a type to the variable 'b' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'b', str_2860)

        if (may_be_2842 and more_types_in_union_2843):
            # SSA join for if statement (line 36)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 44):
    # Getting the type of 'a' (line 44)
    a_2861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 9), 'a')
    int_2862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 13), 'int')
    # Applying the binary operator 'div' (line 44)
    result_div_2863 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 9), 'div', a_2861, int_2862)
    
    # Assigning a type to the variable 'r5' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'r5', result_div_2863)
    
    # Assigning a BinOp to a Name (line 45):
    # Getting the type of 'b' (line 45)
    b_2864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 9), 'b')
    int_2865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 13), 'int')
    # Applying the binary operator 'div' (line 45)
    result_div_2866 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 9), 'div', b_2864, int_2865)
    
    # Assigning a type to the variable 'r6' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'r6', result_div_2866)
    
    # ################# End of 'if_else_base3(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'if_else_base3' in the type store
    # Getting the type of 'stypy_return_type' (line 34)
    stypy_return_type_2867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2867)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'if_else_base3'
    return stypy_return_type_2867

# Assigning a type to the variable 'if_else_base3' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'if_else_base3', if_else_base3)

@norecursion
def if_else_base4(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'if_else_base4'
    module_type_store = module_type_store.open_function_context('if_else_base4', 47, 0, False)
    
    # Passed parameters checking function
    if_else_base4.stypy_localization = localization
    if_else_base4.stypy_type_of_self = None
    if_else_base4.stypy_type_store = module_type_store
    if_else_base4.stypy_function_name = 'if_else_base4'
    if_else_base4.stypy_param_names_list = ['a']
    if_else_base4.stypy_varargs_param_name = None
    if_else_base4.stypy_kwargs_param_name = None
    if_else_base4.stypy_call_defaults = defaults
    if_else_base4.stypy_call_varargs = varargs
    if_else_base4.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'if_else_base4', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'if_else_base4', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'if_else_base4(...)' code ##################

    
    # Assigning a Str to a Name (line 48):
    str_2868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'b', str_2868)
    
    # Type idiom detected: calculating its left and rigth part (line 49)
    # Getting the type of 'a' (line 49)
    a_2869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'a')
    # Getting the type of 'int' (line 49)
    int_2870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 18), 'int')
    
    (may_be_2871, more_types_in_union_2872) = may_be_type(a_2869, int_2870)

    if may_be_2871:

        if more_types_in_union_2872:
            # Runtime conditional SSA (line 49)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'a' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'a', int_2870())
        
        # Assigning a BinOp to a Name (line 50):
        # Getting the type of 'a' (line 50)
        a_2873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'a')
        int_2874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 16), 'int')
        # Applying the binary operator 'div' (line 50)
        result_div_2875 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 12), 'div', a_2873, int_2874)
        
        # Assigning a type to the variable 'r' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'r', result_div_2875)
        
        # Assigning a Subscript to a Name (line 51):
        
        # Obtaining the type of the subscript
        int_2876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 15), 'int')
        # Getting the type of 'a' (line 51)
        a_2877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 51)
        getitem___2878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 13), a_2877, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
        subscript_call_result_2879 = invoke(stypy.reporting.localization.Localization(__file__, 51, 13), getitem___2878, int_2876)
        
        # Assigning a type to the variable 'r2' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'r2', subscript_call_result_2879)
        
        # Assigning a Num to a Name (line 52):
        int_2880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 12), 'int')
        # Assigning a type to the variable 'b' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'b', int_2880)

        if more_types_in_union_2872:
            # Runtime conditional SSA for else branch (line 49)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_2871) or more_types_in_union_2872):
        # Getting the type of 'a' (line 49)
        a_2881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'a')
        # Assigning a type to the variable 'a' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'a', remove_type_from_union(a_2881, int_2870))
        
        # Assigning a Subscript to a Name (line 54):
        
        # Obtaining the type of the subscript
        int_2882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 15), 'int')
        # Getting the type of 'a' (line 54)
        a_2883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 54)
        getitem___2884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 13), a_2883, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 54)
        subscript_call_result_2885 = invoke(stypy.reporting.localization.Localization(__file__, 54, 13), getitem___2884, int_2882)
        
        # Assigning a type to the variable 'r3' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'r3', subscript_call_result_2885)
        
        # Assigning a BinOp to a Name (line 55):
        # Getting the type of 'a' (line 55)
        a_2886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 13), 'a')
        int_2887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 17), 'int')
        # Applying the binary operator 'div' (line 55)
        result_div_2888 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 13), 'div', a_2886, int_2887)
        
        # Assigning a type to the variable 'r4' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'r4', result_div_2888)
        
        # Assigning a Str to a Name (line 56):
        str_2889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 12), 'str', 'bye')
        # Assigning a type to the variable 'b' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'b', str_2889)

        if (may_be_2871 and more_types_in_union_2872):
            # SSA join for if statement (line 49)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 57):
    # Getting the type of 'a' (line 57)
    a_2890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 9), 'a')
    int_2891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 13), 'int')
    # Applying the binary operator 'div' (line 57)
    result_div_2892 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 9), 'div', a_2890, int_2891)
    
    # Assigning a type to the variable 'r5' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'r5', result_div_2892)
    
    # Assigning a BinOp to a Name (line 58):
    # Getting the type of 'b' (line 58)
    b_2893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 9), 'b')
    int_2894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 13), 'int')
    # Applying the binary operator 'div' (line 58)
    result_div_2895 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 9), 'div', b_2893, int_2894)
    
    # Assigning a type to the variable 'r6' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'r6', result_div_2895)
    
    # ################# End of 'if_else_base4(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'if_else_base4' in the type store
    # Getting the type of 'stypy_return_type' (line 47)
    stypy_return_type_2896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2896)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'if_else_base4'
    return stypy_return_type_2896

# Assigning a type to the variable 'if_else_base4' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'if_else_base4', if_else_base4)

# Assigning a IfExp to a Name (line 61):

# Getting the type of 'True' (line 61)
True_2897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 20), 'True')
# Testing the type of an if expression (line 61)
is_suitable_condition(stypy.reporting.localization.Localization(__file__, 61, 11), True_2897)
# SSA begins for if expression (line 61)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')

# Call to int(...): (line 61)
# Processing the call keyword arguments (line 61)
kwargs_2899 = {}
# Getting the type of 'int' (line 61)
int_2898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 11), 'int', False)
# Calling int(args, kwargs) (line 61)
int_call_result_2900 = invoke(stypy.reporting.localization.Localization(__file__, 61, 11), int_2898, *[], **kwargs_2899)

# SSA branch for the else part of an if expression (line 61)
module_type_store.open_ssa_branch('if expression else')

# Getting the type of 'False' (line 61)
False_2901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 39), 'False')
# Testing the type of an if expression (line 61)
is_suitable_condition(stypy.reporting.localization.Localization(__file__, 61, 30), False_2901)
# SSA begins for if expression (line 61)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')

# Call to str(...): (line 61)
# Processing the call keyword arguments (line 61)
kwargs_2903 = {}
# Getting the type of 'str' (line 61)
str_2902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 30), 'str', False)
# Calling str(args, kwargs) (line 61)
str_call_result_2904 = invoke(stypy.reporting.localization.Localization(__file__, 61, 30), str_2902, *[], **kwargs_2903)

# SSA branch for the else part of an if expression (line 61)
module_type_store.open_ssa_branch('if expression else')
# Getting the type of 'False' (line 61)
False_2905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 50), 'False')
# SSA join for if expression (line 61)
module_type_store = module_type_store.join_ssa_context()
if_exp_2906 = union_type.UnionType.add(str_call_result_2904, False_2905)

# SSA join for if expression (line 61)
module_type_store = module_type_store.join_ssa_context()
if_exp_2907 = union_type.UnionType.add(int_call_result_2900, if_exp_2906)

# Assigning a type to the variable 'bigUnion' (line 61)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 0), 'bigUnion', if_exp_2907)

# Call to if_else_base1(...): (line 63)
# Processing the call arguments (line 63)
# Getting the type of 'theInt' (line 63)
theInt_2909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 14), 'theInt', False)
# Processing the call keyword arguments (line 63)
kwargs_2910 = {}
# Getting the type of 'if_else_base1' (line 63)
if_else_base1_2908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'if_else_base1', False)
# Calling if_else_base1(args, kwargs) (line 63)
if_else_base1_call_result_2911 = invoke(stypy.reporting.localization.Localization(__file__, 63, 0), if_else_base1_2908, *[theInt_2909], **kwargs_2910)


# Call to if_else_base2(...): (line 64)
# Processing the call arguments (line 64)
# Getting the type of 'theStr' (line 64)
theStr_2913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 14), 'theStr', False)
# Processing the call keyword arguments (line 64)
kwargs_2914 = {}
# Getting the type of 'if_else_base2' (line 64)
if_else_base2_2912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'if_else_base2', False)
# Calling if_else_base2(args, kwargs) (line 64)
if_else_base2_call_result_2915 = invoke(stypy.reporting.localization.Localization(__file__, 64, 0), if_else_base2_2912, *[theStr_2913], **kwargs_2914)


# Call to if_else_base3(...): (line 65)
# Processing the call arguments (line 65)
# Getting the type of 'union' (line 65)
union_2917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 14), 'union', False)
# Processing the call keyword arguments (line 65)
kwargs_2918 = {}
# Getting the type of 'if_else_base3' (line 65)
if_else_base3_2916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'if_else_base3', False)
# Calling if_else_base3(args, kwargs) (line 65)
if_else_base3_call_result_2919 = invoke(stypy.reporting.localization.Localization(__file__, 65, 0), if_else_base3_2916, *[union_2917], **kwargs_2918)


# Call to if_else_base4(...): (line 66)
# Processing the call arguments (line 66)
# Getting the type of 'bigUnion' (line 66)
bigUnion_2921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 14), 'bigUnion', False)
# Processing the call keyword arguments (line 66)
kwargs_2922 = {}
# Getting the type of 'if_else_base4' (line 66)
if_else_base4_2920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'if_else_base4', False)
# Calling if_else_base4(args, kwargs) (line 66)
if_else_base4_call_result_2923 = invoke(stypy.reporting.localization.Localization(__file__, 66, 0), if_else_base4_2920, *[bigUnion_2921], **kwargs_2922)


@norecursion
def simple_if_else_idiom_variant(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_else_idiom_variant'
    module_type_store = module_type_store.open_function_context('simple_if_else_idiom_variant', 68, 0, False)
    
    # Passed parameters checking function
    simple_if_else_idiom_variant.stypy_localization = localization
    simple_if_else_idiom_variant.stypy_type_of_self = None
    simple_if_else_idiom_variant.stypy_type_store = module_type_store
    simple_if_else_idiom_variant.stypy_function_name = 'simple_if_else_idiom_variant'
    simple_if_else_idiom_variant.stypy_param_names_list = ['a']
    simple_if_else_idiom_variant.stypy_varargs_param_name = None
    simple_if_else_idiom_variant.stypy_kwargs_param_name = None
    simple_if_else_idiom_variant.stypy_call_defaults = defaults
    simple_if_else_idiom_variant.stypy_call_varargs = varargs
    simple_if_else_idiom_variant.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_else_idiom_variant', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_else_idiom_variant', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_else_idiom_variant(...)' code ##################

    
    # Assigning a Str to a Name (line 69):
    str_2924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'b', str_2924)
    
    # Type idiom detected: calculating its left and rigth part (line 70)
    # Getting the type of 'a' (line 70)
    a_2925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'a')
    
    # Call to type(...): (line 70)
    # Processing the call arguments (line 70)
    int_2927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 23), 'int')
    # Processing the call keyword arguments (line 70)
    kwargs_2928 = {}
    # Getting the type of 'type' (line 70)
    type_2926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 18), 'type', False)
    # Calling type(args, kwargs) (line 70)
    type_call_result_2929 = invoke(stypy.reporting.localization.Localization(__file__, 70, 18), type_2926, *[int_2927], **kwargs_2928)
    
    
    (may_be_2930, more_types_in_union_2931) = may_be_type(a_2925, type_call_result_2929)

    if may_be_2930:

        if more_types_in_union_2931:
            # Runtime conditional SSA (line 70)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'a' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'a', type_call_result_2929())
        
        # Assigning a BinOp to a Name (line 71):
        # Getting the type of 'a' (line 71)
        a_2932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'a')
        int_2933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 16), 'int')
        # Applying the binary operator 'div' (line 71)
        result_div_2934 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 12), 'div', a_2932, int_2933)
        
        # Assigning a type to the variable 'r' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'r', result_div_2934)
        
        # Assigning a Subscript to a Name (line 72):
        
        # Obtaining the type of the subscript
        int_2935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 15), 'int')
        # Getting the type of 'a' (line 72)
        a_2936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 72)
        getitem___2937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 13), a_2936, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 72)
        subscript_call_result_2938 = invoke(stypy.reporting.localization.Localization(__file__, 72, 13), getitem___2937, int_2935)
        
        # Assigning a type to the variable 'r2' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'r2', subscript_call_result_2938)
        
        # Assigning a Num to a Name (line 73):
        int_2939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 12), 'int')
        # Assigning a type to the variable 'b' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'b', int_2939)

        if more_types_in_union_2931:
            # Runtime conditional SSA for else branch (line 70)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_2930) or more_types_in_union_2931):
        # Getting the type of 'a' (line 70)
        a_2940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'a')
        # Assigning a type to the variable 'a' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'a', remove_type_from_union(a_2940, type_call_result_2929))
        
        # Assigning a Subscript to a Name (line 75):
        
        # Obtaining the type of the subscript
        int_2941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 15), 'int')
        # Getting the type of 'a' (line 75)
        a_2942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 75)
        getitem___2943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 13), a_2942, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 75)
        subscript_call_result_2944 = invoke(stypy.reporting.localization.Localization(__file__, 75, 13), getitem___2943, int_2941)
        
        # Assigning a type to the variable 'r3' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'r3', subscript_call_result_2944)
        
        # Assigning a BinOp to a Name (line 76):
        # Getting the type of 'a' (line 76)
        a_2945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 13), 'a')
        int_2946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 17), 'int')
        # Applying the binary operator 'div' (line 76)
        result_div_2947 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 13), 'div', a_2945, int_2946)
        
        # Assigning a type to the variable 'r4' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'r4', result_div_2947)
        
        # Assigning a Str to a Name (line 77):
        str_2948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 12), 'str', 'bye')
        # Assigning a type to the variable 'b' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'b', str_2948)

        if (may_be_2930 and more_types_in_union_2931):
            # SSA join for if statement (line 70)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 79):
    # Getting the type of 'a' (line 79)
    a_2949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 9), 'a')
    int_2950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 13), 'int')
    # Applying the binary operator 'div' (line 79)
    result_div_2951 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 9), 'div', a_2949, int_2950)
    
    # Assigning a type to the variable 'r5' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'r5', result_div_2951)
    
    # Assigning a BinOp to a Name (line 80):
    # Getting the type of 'b' (line 80)
    b_2952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 9), 'b')
    int_2953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 13), 'int')
    # Applying the binary operator 'div' (line 80)
    result_div_2954 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 9), 'div', b_2952, int_2953)
    
    # Assigning a type to the variable 'r6' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'r6', result_div_2954)
    
    # ################# End of 'simple_if_else_idiom_variant(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_else_idiom_variant' in the type store
    # Getting the type of 'stypy_return_type' (line 68)
    stypy_return_type_2955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2955)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_else_idiom_variant'
    return stypy_return_type_2955

# Assigning a type to the variable 'simple_if_else_idiom_variant' (line 68)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'simple_if_else_idiom_variant', simple_if_else_idiom_variant)

# Call to simple_if_else_idiom_variant(...): (line 82)
# Processing the call arguments (line 82)
# Getting the type of 'union' (line 82)
union_2957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 29), 'union', False)
# Processing the call keyword arguments (line 82)
kwargs_2958 = {}
# Getting the type of 'simple_if_else_idiom_variant' (line 82)
simple_if_else_idiom_variant_2956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 0), 'simple_if_else_idiom_variant', False)
# Calling simple_if_else_idiom_variant(args, kwargs) (line 82)
simple_if_else_idiom_variant_call_result_2959 = invoke(stypy.reporting.localization.Localization(__file__, 82, 0), simple_if_else_idiom_variant_2956, *[union_2957], **kwargs_2958)


@norecursion
def simple_if_else_not_idiom(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_else_not_idiom'
    module_type_store = module_type_store.open_function_context('simple_if_else_not_idiom', 84, 0, False)
    
    # Passed parameters checking function
    simple_if_else_not_idiom.stypy_localization = localization
    simple_if_else_not_idiom.stypy_type_of_self = None
    simple_if_else_not_idiom.stypy_type_store = module_type_store
    simple_if_else_not_idiom.stypy_function_name = 'simple_if_else_not_idiom'
    simple_if_else_not_idiom.stypy_param_names_list = ['a']
    simple_if_else_not_idiom.stypy_varargs_param_name = None
    simple_if_else_not_idiom.stypy_kwargs_param_name = None
    simple_if_else_not_idiom.stypy_call_defaults = defaults
    simple_if_else_not_idiom.stypy_call_varargs = varargs
    simple_if_else_not_idiom.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_else_not_idiom', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_else_not_idiom', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_else_not_idiom(...)' code ##################

    
    # Assigning a Str to a Name (line 85):
    str_2960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'b', str_2960)
    
    
    
    # Call to type(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'a' (line 86)
    a_2962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'a', False)
    # Processing the call keyword arguments (line 86)
    kwargs_2963 = {}
    # Getting the type of 'type' (line 86)
    type_2961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 7), 'type', False)
    # Calling type(args, kwargs) (line 86)
    type_call_result_2964 = invoke(stypy.reporting.localization.Localization(__file__, 86, 7), type_2961, *[a_2962], **kwargs_2963)
    
    int_2965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 18), 'int')
    # Applying the binary operator 'is' (line 86)
    result_is__2966 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 7), 'is', type_call_result_2964, int_2965)
    
    # Testing the type of an if condition (line 86)
    if_condition_2967 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 86, 4), result_is__2966)
    # Assigning a type to the variable 'if_condition_2967' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'if_condition_2967', if_condition_2967)
    # SSA begins for if statement (line 86)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 87):
    # Getting the type of 'a' (line 87)
    a_2968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'a')
    int_2969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 16), 'int')
    # Applying the binary operator 'div' (line 87)
    result_div_2970 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 12), 'div', a_2968, int_2969)
    
    # Assigning a type to the variable 'r' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'r', result_div_2970)
    
    # Assigning a Subscript to a Name (line 88):
    
    # Obtaining the type of the subscript
    int_2971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 15), 'int')
    # Getting the type of 'a' (line 88)
    a_2972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 13), 'a')
    # Obtaining the member '__getitem__' of a type (line 88)
    getitem___2973 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 13), a_2972, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 88)
    subscript_call_result_2974 = invoke(stypy.reporting.localization.Localization(__file__, 88, 13), getitem___2973, int_2971)
    
    # Assigning a type to the variable 'r2' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'r2', subscript_call_result_2974)
    
    # Assigning a Num to a Name (line 89):
    int_2975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 12), 'int')
    # Assigning a type to the variable 'b' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'b', int_2975)
    # SSA branch for the else part of an if statement (line 86)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Subscript to a Name (line 91):
    
    # Obtaining the type of the subscript
    int_2976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 15), 'int')
    # Getting the type of 'a' (line 91)
    a_2977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 13), 'a')
    # Obtaining the member '__getitem__' of a type (line 91)
    getitem___2978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 13), a_2977, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 91)
    subscript_call_result_2979 = invoke(stypy.reporting.localization.Localization(__file__, 91, 13), getitem___2978, int_2976)
    
    # Assigning a type to the variable 'r3' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'r3', subscript_call_result_2979)
    
    # Assigning a BinOp to a Name (line 92):
    # Getting the type of 'a' (line 92)
    a_2980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 13), 'a')
    int_2981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 17), 'int')
    # Applying the binary operator 'div' (line 92)
    result_div_2982 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 13), 'div', a_2980, int_2981)
    
    # Assigning a type to the variable 'r4' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'r4', result_div_2982)
    
    # Assigning a Str to a Name (line 93):
    str_2983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 12), 'str', 'bye')
    # Assigning a type to the variable 'b' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'b', str_2983)
    # SSA join for if statement (line 86)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 95):
    # Getting the type of 'a' (line 95)
    a_2984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 9), 'a')
    int_2985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 13), 'int')
    # Applying the binary operator 'div' (line 95)
    result_div_2986 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 9), 'div', a_2984, int_2985)
    
    # Assigning a type to the variable 'r5' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'r5', result_div_2986)
    
    # Assigning a BinOp to a Name (line 96):
    # Getting the type of 'b' (line 96)
    b_2987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 9), 'b')
    int_2988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 13), 'int')
    # Applying the binary operator 'div' (line 96)
    result_div_2989 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 9), 'div', b_2987, int_2988)
    
    # Assigning a type to the variable 'r6' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'r6', result_div_2989)
    
    # ################# End of 'simple_if_else_not_idiom(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_else_not_idiom' in the type store
    # Getting the type of 'stypy_return_type' (line 84)
    stypy_return_type_2990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2990)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_else_not_idiom'
    return stypy_return_type_2990

# Assigning a type to the variable 'simple_if_else_not_idiom' (line 84)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'simple_if_else_not_idiom', simple_if_else_not_idiom)

# Call to simple_if_else_not_idiom(...): (line 98)
# Processing the call arguments (line 98)
# Getting the type of 'union' (line 98)
union_2992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 25), 'union', False)
# Processing the call keyword arguments (line 98)
kwargs_2993 = {}
# Getting the type of 'simple_if_else_not_idiom' (line 98)
simple_if_else_not_idiom_2991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), 'simple_if_else_not_idiom', False)
# Calling simple_if_else_not_idiom(args, kwargs) (line 98)
simple_if_else_not_idiom_call_result_2994 = invoke(stypy.reporting.localization.Localization(__file__, 98, 0), simple_if_else_not_idiom_2991, *[union_2992], **kwargs_2993)

# Declaration of the 'Foo' class

class Foo:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 101, 4, False)
        # Assigning a type to the variable 'self' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'self', type_of_self)
        
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

        
        # Assigning a Num to a Attribute (line 102):
        int_2995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 20), 'int')
        # Getting the type of 'self' (line 102)
        self_2996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'self')
        # Setting the type of the member 'attr' of a type (line 102)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 8), self_2996, 'attr', int_2995)
        
        # Assigning a Str to a Attribute (line 103):
        str_2997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 23), 'str', 'bar')
        # Getting the type of 'self' (line 103)
        self_2998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'self')
        # Setting the type of the member 'strattr' of a type (line 103)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 8), self_2998, 'strattr', str_2997)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Foo' (line 100)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 0), 'Foo', Foo)

@norecursion
def simple_if_else_idiom_attr(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_else_idiom_attr'
    module_type_store = module_type_store.open_function_context('simple_if_else_idiom_attr', 105, 0, False)
    
    # Passed parameters checking function
    simple_if_else_idiom_attr.stypy_localization = localization
    simple_if_else_idiom_attr.stypy_type_of_self = None
    simple_if_else_idiom_attr.stypy_type_store = module_type_store
    simple_if_else_idiom_attr.stypy_function_name = 'simple_if_else_idiom_attr'
    simple_if_else_idiom_attr.stypy_param_names_list = ['a']
    simple_if_else_idiom_attr.stypy_varargs_param_name = None
    simple_if_else_idiom_attr.stypy_kwargs_param_name = None
    simple_if_else_idiom_attr.stypy_call_defaults = defaults
    simple_if_else_idiom_attr.stypy_call_varargs = varargs
    simple_if_else_idiom_attr.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_else_idiom_attr', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_else_idiom_attr', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_else_idiom_attr(...)' code ##################

    
    # Assigning a Str to a Name (line 106):
    str_2999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'b', str_2999)
    
    # Type idiom detected: calculating its left and rigth part (line 107)
    # Getting the type of 'a' (line 107)
    a_3000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'a')
    # Obtaining the member 'attr' of a type (line 107)
    attr_3001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 12), a_3000, 'attr')
    # Getting the type of 'int' (line 107)
    int_3002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 23), 'int')
    
    (may_be_3003, more_types_in_union_3004) = may_be_type(attr_3001, int_3002)

    if may_be_3003:

        if more_types_in_union_3004:
            # Runtime conditional SSA (line 107)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'a' (line 107)
        a_3005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'a')
        # Setting the type of the member 'attr' of a type (line 107)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 4), a_3005, 'attr', int_3002())
        
        # Assigning a BinOp to a Name (line 108):
        # Getting the type of 'a' (line 108)
        a_3006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'a')
        # Obtaining the member 'attr' of a type (line 108)
        attr_3007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 12), a_3006, 'attr')
        int_3008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 21), 'int')
        # Applying the binary operator 'div' (line 108)
        result_div_3009 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 12), 'div', attr_3007, int_3008)
        
        # Assigning a type to the variable 'r' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'r', result_div_3009)
        
        # Assigning a Subscript to a Name (line 109):
        
        # Obtaining the type of the subscript
        int_3010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 20), 'int')
        # Getting the type of 'a' (line 109)
        a_3011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 13), 'a')
        # Obtaining the member 'attr' of a type (line 109)
        attr_3012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 13), a_3011, 'attr')
        # Obtaining the member '__getitem__' of a type (line 109)
        getitem___3013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 13), attr_3012, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 109)
        subscript_call_result_3014 = invoke(stypy.reporting.localization.Localization(__file__, 109, 13), getitem___3013, int_3010)
        
        # Assigning a type to the variable 'r2' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'r2', subscript_call_result_3014)
        
        # Assigning a Num to a Name (line 110):
        int_3015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 12), 'int')
        # Assigning a type to the variable 'b' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'b', int_3015)

        if more_types_in_union_3004:
            # Runtime conditional SSA for else branch (line 107)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_3003) or more_types_in_union_3004):
        # Getting the type of 'a' (line 107)
        a_3016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'a')
        # Obtaining the member 'attr' of a type (line 107)
        attr_3017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 4), a_3016, 'attr')
        # Setting the type of the member 'attr' of a type (line 107)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 4), a_3016, 'attr', remove_type_from_union(attr_3017, int_3002))
        
        # Assigning a Subscript to a Name (line 112):
        
        # Obtaining the type of the subscript
        int_3018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 15), 'int')
        # Getting the type of 'a' (line 112)
        a_3019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 112)
        getitem___3020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 13), a_3019, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 112)
        subscript_call_result_3021 = invoke(stypy.reporting.localization.Localization(__file__, 112, 13), getitem___3020, int_3018)
        
        # Assigning a type to the variable 'r3' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'r3', subscript_call_result_3021)
        
        # Assigning a BinOp to a Name (line 113):
        # Getting the type of 'a' (line 113)
        a_3022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 13), 'a')
        int_3023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 17), 'int')
        # Applying the binary operator 'div' (line 113)
        result_div_3024 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 13), 'div', a_3022, int_3023)
        
        # Assigning a type to the variable 'r4' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'r4', result_div_3024)
        
        # Assigning a Str to a Name (line 114):
        str_3025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 12), 'str', 'bye')
        # Assigning a type to the variable 'b' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'b', str_3025)

        if (may_be_3003 and more_types_in_union_3004):
            # SSA join for if statement (line 107)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 116):
    # Getting the type of 'a' (line 116)
    a_3026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 9), 'a')
    # Obtaining the member 'attr' of a type (line 116)
    attr_3027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 9), a_3026, 'attr')
    int_3028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 18), 'int')
    # Applying the binary operator 'div' (line 116)
    result_div_3029 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 9), 'div', attr_3027, int_3028)
    
    # Assigning a type to the variable 'r5' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'r5', result_div_3029)
    
    # Assigning a BinOp to a Name (line 117):
    # Getting the type of 'b' (line 117)
    b_3030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 9), 'b')
    int_3031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 13), 'int')
    # Applying the binary operator 'div' (line 117)
    result_div_3032 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 9), 'div', b_3030, int_3031)
    
    # Assigning a type to the variable 'r6' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'r6', result_div_3032)
    
    # ################# End of 'simple_if_else_idiom_attr(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_else_idiom_attr' in the type store
    # Getting the type of 'stypy_return_type' (line 105)
    stypy_return_type_3033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3033)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_else_idiom_attr'
    return stypy_return_type_3033

# Assigning a type to the variable 'simple_if_else_idiom_attr' (line 105)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), 'simple_if_else_idiom_attr', simple_if_else_idiom_attr)

@norecursion
def simple_if_else_idiom_attr_b(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'simple_if_else_idiom_attr_b'
    module_type_store = module_type_store.open_function_context('simple_if_else_idiom_attr_b', 119, 0, False)
    
    # Passed parameters checking function
    simple_if_else_idiom_attr_b.stypy_localization = localization
    simple_if_else_idiom_attr_b.stypy_type_of_self = None
    simple_if_else_idiom_attr_b.stypy_type_store = module_type_store
    simple_if_else_idiom_attr_b.stypy_function_name = 'simple_if_else_idiom_attr_b'
    simple_if_else_idiom_attr_b.stypy_param_names_list = ['a']
    simple_if_else_idiom_attr_b.stypy_varargs_param_name = None
    simple_if_else_idiom_attr_b.stypy_kwargs_param_name = None
    simple_if_else_idiom_attr_b.stypy_call_defaults = defaults
    simple_if_else_idiom_attr_b.stypy_call_varargs = varargs
    simple_if_else_idiom_attr_b.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'simple_if_else_idiom_attr_b', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'simple_if_else_idiom_attr_b', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'simple_if_else_idiom_attr_b(...)' code ##################

    
    # Assigning a Str to a Name (line 120):
    str_3034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 8), 'str', 'hi')
    # Assigning a type to the variable 'b' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'b', str_3034)
    
    # Type idiom detected: calculating its left and rigth part (line 121)
    # Getting the type of 'a' (line 121)
    a_3035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'a')
    # Obtaining the member 'strattr' of a type (line 121)
    strattr_3036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 12), a_3035, 'strattr')
    # Getting the type of 'int' (line 121)
    int_3037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 26), 'int')
    
    (may_be_3038, more_types_in_union_3039) = may_be_type(strattr_3036, int_3037)

    if may_be_3038:

        if more_types_in_union_3039:
            # Runtime conditional SSA (line 121)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'a' (line 121)
        a_3040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'a')
        # Setting the type of the member 'strattr' of a type (line 121)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 4), a_3040, 'strattr', int_3037())
        
        # Assigning a BinOp to a Name (line 122):
        # Getting the type of 'a' (line 122)
        a_3041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'a')
        # Obtaining the member 'attr' of a type (line 122)
        attr_3042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 12), a_3041, 'attr')
        int_3043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 21), 'int')
        # Applying the binary operator 'div' (line 122)
        result_div_3044 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 12), 'div', attr_3042, int_3043)
        
        # Assigning a type to the variable 'r' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'r', result_div_3044)
        
        # Assigning a Subscript to a Name (line 123):
        
        # Obtaining the type of the subscript
        int_3045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 23), 'int')
        # Getting the type of 'a' (line 123)
        a_3046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 13), 'a')
        # Obtaining the member 'strattr' of a type (line 123)
        strattr_3047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 13), a_3046, 'strattr')
        # Obtaining the member '__getitem__' of a type (line 123)
        getitem___3048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 13), strattr_3047, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 123)
        subscript_call_result_3049 = invoke(stypy.reporting.localization.Localization(__file__, 123, 13), getitem___3048, int_3045)
        
        # Assigning a type to the variable 'r2' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'r2', subscript_call_result_3049)
        
        # Assigning a Num to a Name (line 124):
        int_3050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 12), 'int')
        # Assigning a type to the variable 'b' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'b', int_3050)

        if more_types_in_union_3039:
            # Runtime conditional SSA for else branch (line 121)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_3038) or more_types_in_union_3039):
        # Getting the type of 'a' (line 121)
        a_3051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'a')
        # Obtaining the member 'strattr' of a type (line 121)
        strattr_3052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 4), a_3051, 'strattr')
        # Setting the type of the member 'strattr' of a type (line 121)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 4), a_3051, 'strattr', remove_type_from_union(strattr_3052, int_3037))
        
        # Assigning a Subscript to a Name (line 126):
        
        # Obtaining the type of the subscript
        int_3053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 15), 'int')
        # Getting the type of 'a' (line 126)
        a_3054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 126)
        getitem___3055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 13), a_3054, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 126)
        subscript_call_result_3056 = invoke(stypy.reporting.localization.Localization(__file__, 126, 13), getitem___3055, int_3053)
        
        # Assigning a type to the variable 'r3' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'r3', subscript_call_result_3056)
        
        # Assigning a BinOp to a Name (line 127):
        # Getting the type of 'a' (line 127)
        a_3057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 13), 'a')
        int_3058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 17), 'int')
        # Applying the binary operator 'div' (line 127)
        result_div_3059 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 13), 'div', a_3057, int_3058)
        
        # Assigning a type to the variable 'r4' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'r4', result_div_3059)
        
        # Assigning a Str to a Name (line 128):
        str_3060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 12), 'str', 'bye')
        # Assigning a type to the variable 'b' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'b', str_3060)

        if (may_be_3038 and more_types_in_union_3039):
            # SSA join for if statement (line 121)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 130):
    # Getting the type of 'a' (line 130)
    a_3061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 9), 'a')
    # Obtaining the member 'strattr' of a type (line 130)
    strattr_3062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 9), a_3061, 'strattr')
    int_3063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 21), 'int')
    # Applying the binary operator 'div' (line 130)
    result_div_3064 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 9), 'div', strattr_3062, int_3063)
    
    # Assigning a type to the variable 'r3' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'r3', result_div_3064)
    
    # Assigning a BinOp to a Name (line 131):
    # Getting the type of 'b' (line 131)
    b_3065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 9), 'b')
    int_3066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 13), 'int')
    # Applying the binary operator 'div' (line 131)
    result_div_3067 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 9), 'div', b_3065, int_3066)
    
    # Assigning a type to the variable 'r4' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'r4', result_div_3067)
    
    # ################# End of 'simple_if_else_idiom_attr_b(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'simple_if_else_idiom_attr_b' in the type store
    # Getting the type of 'stypy_return_type' (line 119)
    stypy_return_type_3068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3068)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'simple_if_else_idiom_attr_b'
    return stypy_return_type_3068

# Assigning a type to the variable 'simple_if_else_idiom_attr_b' (line 119)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 0), 'simple_if_else_idiom_attr_b', simple_if_else_idiom_attr_b)

# Call to simple_if_else_idiom_attr(...): (line 133)
# Processing the call arguments (line 133)

# Call to Foo(...): (line 133)
# Processing the call keyword arguments (line 133)
kwargs_3071 = {}
# Getting the type of 'Foo' (line 133)
Foo_3070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 26), 'Foo', False)
# Calling Foo(args, kwargs) (line 133)
Foo_call_result_3072 = invoke(stypy.reporting.localization.Localization(__file__, 133, 26), Foo_3070, *[], **kwargs_3071)

# Processing the call keyword arguments (line 133)
kwargs_3073 = {}
# Getting the type of 'simple_if_else_idiom_attr' (line 133)
simple_if_else_idiom_attr_3069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 0), 'simple_if_else_idiom_attr', False)
# Calling simple_if_else_idiom_attr(args, kwargs) (line 133)
simple_if_else_idiom_attr_call_result_3074 = invoke(stypy.reporting.localization.Localization(__file__, 133, 0), simple_if_else_idiom_attr_3069, *[Foo_call_result_3072], **kwargs_3073)


# Call to simple_if_else_idiom_attr_b(...): (line 134)
# Processing the call arguments (line 134)

# Call to Foo(...): (line 134)
# Processing the call keyword arguments (line 134)
kwargs_3077 = {}
# Getting the type of 'Foo' (line 134)
Foo_3076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 28), 'Foo', False)
# Calling Foo(args, kwargs) (line 134)
Foo_call_result_3078 = invoke(stypy.reporting.localization.Localization(__file__, 134, 28), Foo_3076, *[], **kwargs_3077)

# Processing the call keyword arguments (line 134)
kwargs_3079 = {}
# Getting the type of 'simple_if_else_idiom_attr_b' (line 134)
simple_if_else_idiom_attr_b_3075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 0), 'simple_if_else_idiom_attr_b', False)
# Calling simple_if_else_idiom_attr_b(args, kwargs) (line 134)
simple_if_else_idiom_attr_b_call_result_3080 = invoke(stypy.reporting.localization.Localization(__file__, 134, 0), simple_if_else_idiom_attr_b_3075, *[Foo_call_result_3078], **kwargs_3079)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
