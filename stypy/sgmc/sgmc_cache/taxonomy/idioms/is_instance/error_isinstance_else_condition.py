
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # -*- coding: utf-8 -*-
2: __doc__ = "If / else isinstance condition with dynamic type inspection"
3: 
4: if __name__ == '__main__':
5:     if True:
6:         a = 3
7:     else:
8:         a = "3"
9: 
10:     b = None
11:     if isinstance(a, int):
12:         r = a / 3
13:         # Type warning
14:         r2 = a[0]
15:         b = 3
16:     else:
17:         r3 = a[0]
18:         # Type warning
19:         r4 = a / 3
20:         b = "bye"
21: 
22:     # Type warning
23:     r5 = a / 3
24:     # Type warning
25:     r6 = b / 3
26: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'If / else isinstance condition with dynamic type inspection')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Getting the type of 'True' (line 5)
    True_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 7), 'True')
    # Testing the type of an if condition (line 5)
    if_condition_3 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 5, 4), True_2)
    # Assigning a type to the variable 'if_condition_3' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'if_condition_3', if_condition_3)
    # SSA begins for if statement (line 5)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 6):
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 12), 'int')
    # Assigning a type to the variable 'a' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'a', int_4)
    # SSA branch for the else part of an if statement (line 5)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 8):
    str_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 12), 'str', '3')
    # Assigning a type to the variable 'a' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'a', str_5)
    # SSA join for if statement (line 5)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 10):
    # Getting the type of 'None' (line 10)
    None_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'None')
    # Assigning a type to the variable 'b' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'b', None_6)
    
    # Type idiom detected: calculating its left and rigth part (line 11)
    # Getting the type of 'int' (line 11)
    int_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 21), 'int')
    # Getting the type of 'a' (line 11)
    a_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 18), 'a')
    
    (may_be_9, more_types_in_union_10) = may_be_subtype(int_7, a_8)

    if may_be_9:

        if more_types_in_union_10:
            # Runtime conditional SSA (line 11)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'a' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'a', remove_not_subtype_from_union(a_8, int))
        
        # Assigning a BinOp to a Name (line 12):
        # Getting the type of 'a' (line 12)
        a_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'a')
        int_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 16), 'int')
        # Applying the binary operator 'div' (line 12)
        result_div_13 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 12), 'div', a_11, int_12)
        
        # Assigning a type to the variable 'r' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'r', result_div_13)
        
        # Assigning a Subscript to a Name (line 14):
        
        # Obtaining the type of the subscript
        int_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 15), 'int')
        # Getting the type of 'a' (line 14)
        a_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 14)
        getitem___16 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 13), a_15, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 14)
        subscript_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 14, 13), getitem___16, int_14)
        
        # Assigning a type to the variable 'r2' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'r2', subscript_call_result_17)
        
        # Assigning a Num to a Name (line 15):
        int_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 12), 'int')
        # Assigning a type to the variable 'b' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'b', int_18)

        if more_types_in_union_10:
            # Runtime conditional SSA for else branch (line 11)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_9) or more_types_in_union_10):
        # Assigning a type to the variable 'a' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'a', remove_subtype_from_union(a_8, int))
        
        # Assigning a Subscript to a Name (line 17):
        
        # Obtaining the type of the subscript
        int_19 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 15), 'int')
        # Getting the type of 'a' (line 17)
        a_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 17)
        getitem___21 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 13), a_20, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 17)
        subscript_call_result_22 = invoke(stypy.reporting.localization.Localization(__file__, 17, 13), getitem___21, int_19)
        
        # Assigning a type to the variable 'r3' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'r3', subscript_call_result_22)
        
        # Assigning a BinOp to a Name (line 19):
        # Getting the type of 'a' (line 19)
        a_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 13), 'a')
        int_24 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 17), 'int')
        # Applying the binary operator 'div' (line 19)
        result_div_25 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 13), 'div', a_23, int_24)
        
        # Assigning a type to the variable 'r4' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'r4', result_div_25)
        
        # Assigning a Str to a Name (line 20):
        str_26 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 12), 'str', 'bye')
        # Assigning a type to the variable 'b' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'b', str_26)

        if (may_be_9 and more_types_in_union_10):
            # SSA join for if statement (line 11)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 23):
    # Getting the type of 'a' (line 23)
    a_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 9), 'a')
    int_28 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 13), 'int')
    # Applying the binary operator 'div' (line 23)
    result_div_29 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 9), 'div', a_27, int_28)
    
    # Assigning a type to the variable 'r5' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'r5', result_div_29)
    
    # Assigning a BinOp to a Name (line 25):
    # Getting the type of 'b' (line 25)
    b_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 9), 'b')
    int_31 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 13), 'int')
    # Applying the binary operator 'div' (line 25)
    result_div_32 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 9), 'div', b_30, int_31)
    
    # Assigning a type to the variable 'r6' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'r6', result_div_32)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
