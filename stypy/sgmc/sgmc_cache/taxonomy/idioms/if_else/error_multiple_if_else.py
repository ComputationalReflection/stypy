
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: import math
3: 
4: __doc__ = "Multiple if / else conditions with dynamic type checking inspections"
5: 
6: if __name__ == '__main__':
7:     if True:
8:         a = 3
9:     else:
10:         if math.pi > 3:
11:             a = "3"
12:         else:
13:             a = [1, 2]
14: 
15:     b = None
16:     if type(a) is int:
17:         r = a / 3
18:         # Type warning
19:         r2 = a[0]
20:         b = 3
21:     else:
22:         if type(a) is str:
23:             r3 = a[0]
24:             # Type warning
25:             r4 = a / 3
26:             b = "bye"
27:         else:
28:             r5 = a[0]
29:             # Type warning
30:             r6 = a / 3
31:             # Type warning
32:             wrong = r5 + "str"
33:             b = list()
34: 
35:     # Type warning
36:     r7 = a / 3
37:     # Type warning
38:     r8 = b / 3
39: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import math' statement (line 2)
import math

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'math', math, module_type_store)


# Assigning a Str to a Name (line 4):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 10), 'str', 'Multiple if / else conditions with dynamic type checking inspections')
# Assigning a type to the variable '__doc__' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    
    # Getting the type of 'True' (line 7)
    True_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 7), 'True')
    # Testing the type of an if condition (line 7)
    if_condition_3 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 7, 4), True_2)
    # Assigning a type to the variable 'if_condition_3' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'if_condition_3', if_condition_3)
    # SSA begins for if statement (line 7)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 8):
    int_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 12), 'int')
    # Assigning a type to the variable 'a' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'a', int_4)
    # SSA branch for the else part of an if statement (line 7)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'math' (line 10)
    math_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 11), 'math')
    # Obtaining the member 'pi' of a type (line 10)
    pi_6 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 11), math_5, 'pi')
    int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 21), 'int')
    # Applying the binary operator '>' (line 10)
    result_gt_8 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 11), '>', pi_6, int_7)
    
    # Testing the type of an if condition (line 10)
    if_condition_9 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 10, 8), result_gt_8)
    # Assigning a type to the variable 'if_condition_9' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'if_condition_9', if_condition_9)
    # SSA begins for if statement (line 10)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 11):
    str_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 16), 'str', '3')
    # Assigning a type to the variable 'a' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 12), 'a', str_10)
    # SSA branch for the else part of an if statement (line 10)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a List to a Name (line 13):
    
    # Obtaining an instance of the builtin type 'list' (line 13)
    list_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 13)
    # Adding element type (line 13)
    int_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 16), list_11, int_12)
    # Adding element type (line 13)
    int_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 16), list_11, int_13)
    
    # Assigning a type to the variable 'a' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 12), 'a', list_11)
    # SSA join for if statement (line 10)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 7)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 15):
    # Getting the type of 'None' (line 15)
    None_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'None')
    # Assigning a type to the variable 'b' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'b', None_14)
    
    # Type idiom detected: calculating its left and rigth part (line 16)
    # Getting the type of 'a' (line 16)
    a_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 12), 'a')
    # Getting the type of 'int' (line 16)
    int_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 18), 'int')
    
    (may_be_17, more_types_in_union_18) = may_be_type(a_15, int_16)

    if may_be_17:

        if more_types_in_union_18:
            # Runtime conditional SSA (line 16)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'a' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'a', int_16())
        
        # Assigning a BinOp to a Name (line 17):
        # Getting the type of 'a' (line 17)
        a_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'a')
        int_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 16), 'int')
        # Applying the binary operator 'div' (line 17)
        result_div_21 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 12), 'div', a_19, int_20)
        
        # Assigning a type to the variable 'r' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'r', result_div_21)
        
        # Assigning a Subscript to a Name (line 19):
        
        # Obtaining the type of the subscript
        int_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 15), 'int')
        # Getting the type of 'a' (line 19)
        a_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 13), 'a')
        # Obtaining the member '__getitem__' of a type (line 19)
        getitem___24 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 13), a_23, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 19)
        subscript_call_result_25 = invoke(stypy.reporting.localization.Localization(__file__, 19, 13), getitem___24, int_22)
        
        # Assigning a type to the variable 'r2' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'r2', subscript_call_result_25)
        
        # Assigning a Num to a Name (line 20):
        int_26 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 12), 'int')
        # Assigning a type to the variable 'b' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'b', int_26)

        if more_types_in_union_18:
            # Runtime conditional SSA for else branch (line 16)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_17) or more_types_in_union_18):
        # Getting the type of 'a' (line 16)
        a_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'a')
        # Assigning a type to the variable 'a' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'a', remove_type_from_union(a_27, int_16))
        
        # Type idiom detected: calculating its left and rigth part (line 22)
        # Getting the type of 'a' (line 22)
        a_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'a')
        # Getting the type of 'str' (line 22)
        str_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 22), 'str')
        
        (may_be_30, more_types_in_union_31) = may_be_type(a_28, str_29)

        if may_be_30:

            if more_types_in_union_31:
                # Runtime conditional SSA (line 22)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'a' (line 22)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'a', str_29())
            
            # Assigning a Subscript to a Name (line 23):
            
            # Obtaining the type of the subscript
            int_32 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 19), 'int')
            # Getting the type of 'a' (line 23)
            a_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 17), 'a')
            # Obtaining the member '__getitem__' of a type (line 23)
            getitem___34 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 17), a_33, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 23)
            subscript_call_result_35 = invoke(stypy.reporting.localization.Localization(__file__, 23, 17), getitem___34, int_32)
            
            # Assigning a type to the variable 'r3' (line 23)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'r3', subscript_call_result_35)
            
            # Assigning a BinOp to a Name (line 25):
            # Getting the type of 'a' (line 25)
            a_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 17), 'a')
            int_37 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 21), 'int')
            # Applying the binary operator 'div' (line 25)
            result_div_38 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 17), 'div', a_36, int_37)
            
            # Assigning a type to the variable 'r4' (line 25)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'r4', result_div_38)
            
            # Assigning a Str to a Name (line 26):
            str_39 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 16), 'str', 'bye')
            # Assigning a type to the variable 'b' (line 26)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'b', str_39)

            if more_types_in_union_31:
                # Runtime conditional SSA for else branch (line 22)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_30) or more_types_in_union_31):
            # Getting the type of 'a' (line 22)
            a_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'a')
            # Assigning a type to the variable 'a' (line 22)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'a', remove_type_from_union(a_40, str_29))
            
            # Assigning a Subscript to a Name (line 28):
            
            # Obtaining the type of the subscript
            int_41 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 19), 'int')
            # Getting the type of 'a' (line 28)
            a_42 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 17), 'a')
            # Obtaining the member '__getitem__' of a type (line 28)
            getitem___43 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 17), a_42, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 28)
            subscript_call_result_44 = invoke(stypy.reporting.localization.Localization(__file__, 28, 17), getitem___43, int_41)
            
            # Assigning a type to the variable 'r5' (line 28)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'r5', subscript_call_result_44)
            
            # Assigning a BinOp to a Name (line 30):
            # Getting the type of 'a' (line 30)
            a_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 17), 'a')
            int_46 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 21), 'int')
            # Applying the binary operator 'div' (line 30)
            result_div_47 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 17), 'div', a_45, int_46)
            
            # Assigning a type to the variable 'r6' (line 30)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 12), 'r6', result_div_47)
            
            # Assigning a BinOp to a Name (line 32):
            # Getting the type of 'r5' (line 32)
            r5_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 20), 'r5')
            str_49 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 25), 'str', 'str')
            # Applying the binary operator '+' (line 32)
            result_add_50 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 20), '+', r5_48, str_49)
            
            # Assigning a type to the variable 'wrong' (line 32)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'wrong', result_add_50)
            
            # Assigning a Call to a Name (line 33):
            
            # Call to list(...): (line 33)
            # Processing the call keyword arguments (line 33)
            kwargs_52 = {}
            # Getting the type of 'list' (line 33)
            list_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 16), 'list', False)
            # Calling list(args, kwargs) (line 33)
            list_call_result_53 = invoke(stypy.reporting.localization.Localization(__file__, 33, 16), list_51, *[], **kwargs_52)
            
            # Assigning a type to the variable 'b' (line 33)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'b', list_call_result_53)

            if (may_be_30 and more_types_in_union_31):
                # SSA join for if statement (line 22)
                module_type_store = module_type_store.join_ssa_context()


        

        if (may_be_17 and more_types_in_union_18):
            # SSA join for if statement (line 16)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 36):
    # Getting the type of 'a' (line 36)
    a_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 9), 'a')
    int_55 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 13), 'int')
    # Applying the binary operator 'div' (line 36)
    result_div_56 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 9), 'div', a_54, int_55)
    
    # Assigning a type to the variable 'r7' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'r7', result_div_56)
    
    # Assigning a BinOp to a Name (line 38):
    # Getting the type of 'b' (line 38)
    b_57 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 9), 'b')
    int_58 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 13), 'int')
    # Applying the binary operator 'div' (line 38)
    result_div_59 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 9), 'div', b_57, int_58)
    
    # Assigning a type to the variable 'r8' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'r8', result_div_59)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
