
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "If condition with dynamic type inspection"
4: 
5: if __name__ == '__main__':
6:     if True:
7:         a = 3
8:     else:
9:         a = "3"
10: 
11:     b = None
12:     if type(a) is int:
13:         r = a / 3
14:         # Type warning
15:         r2 = a[0]
16:         b = 3
17: 
18:     # Type warning
19:     r3 = a / 3
20: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'If condition with dynamic type inspection')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Getting the type of 'True' (line 6)
    True_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 7), 'True')
    # Testing the type of an if condition (line 6)
    if_condition_3 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 6, 4), True_2)
    # Assigning a type to the variable 'if_condition_3' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'if_condition_3', if_condition_3)
    # SSA begins for if statement (line 6)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 7):
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 12), 'int')
    # Assigning a type to the variable 'a' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'a', int_4)
    # SSA branch for the else part of an if statement (line 6)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Str to a Name (line 9):
    str_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 12), 'str', '3')
    # Assigning a type to the variable 'a' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'a', str_5)
    # SSA join for if statement (line 6)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 11):
    # Getting the type of 'None' (line 11)
    None_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'None')
    # Assigning a type to the variable 'b' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'b', None_6)
    
    # Type idiom detected: calculating its left and rigth part (line 12)
    # Getting the type of 'a' (line 12)
    a_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'a')
    # Getting the type of 'int' (line 12)
    int_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 18), 'int')
    
    (may_be_9, more_types_in_union_10) = may_be_type(a_7, int_8)

    if may_be_9:

        if more_types_in_union_10:
            # Runtime conditional SSA (line 12)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'a' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'a', int_8())
        
        # Assigning a BinOp to a Name (line 13):
        # Getting the type of 'a' (line 13)
        a_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 12), 'a')
        int_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 16), 'int')
        # Applying the binary operator 'div' (line 13)
        result_div_13 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 12), 'div', a_11, int_12)
        
        # Assigning a type to the variable 'r' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'r', result_div_13)
        
        # Assigning a Subscript to a Name (line 15):
        
        # Obtaining the type of the subscript
        int_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 15), 'int')
        # Getting the type of 'a' (line 15)
        a_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 15)
        getitem___16 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 13), a_15, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 15)
        subscript_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 15, 13), getitem___16, int_14)
        
        # Assigning a type to the variable 'r2' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'r2', subscript_call_result_17)
        
        # Assigning a Num to a Name (line 16):
        int_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 12), 'int')
        # Assigning a type to the variable 'b' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'b', int_18)

        if more_types_in_union_10:
            # SSA join for if statement (line 12)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 19):
    # Getting the type of 'a' (line 19)
    a_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 9), 'a')
    int_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 13), 'int')
    # Applying the binary operator 'div' (line 19)
    result_div_21 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 9), 'div', a_19, int_20)
    
    # Assigning a type to the variable 'r3' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'r3', result_div_21)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
