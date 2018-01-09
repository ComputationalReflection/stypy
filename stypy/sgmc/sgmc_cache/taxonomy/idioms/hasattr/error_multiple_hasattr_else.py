
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: import math
3: 
4: __doc__ = "Multiple if / else hasattr conditions with dynamic type checking inspections"
5: 
6: if __name__ == '__main__':
7: 
8:     class Test:
9:         attribute = None
10:         attribute2 = None
11: 
12: 
13:     test = Test()
14: 
15:     if True:
16:         test.attribute = 3
17:     else:
18:         if math.pi > 3:
19:             test.attribute = "3"
20:         else:
21:             test.attribute2 = [1, 2]
22: 
23:     if hasattr(test, 'attribute'):
24:         # Type warning
25:         r = test.attribute / 3
26:         # Type warning
27:         r2 = test.attribute[0]
28: 
29:     if hasattr(test, 'attribute2'):
30:         # Type warning
31:         r3 = test.attribute2[0]
32:         # Type error
33:         r4 = test.attribute2 / 3
34:         # Type error
35:         wrong = r3 + "str"
36:     else:
37:         # Not executed
38:         r5 = test.attribute[0]
39:         r6 = test.attribute2 / 3
40: 
41:     # Type warning
42:     r7 = test.attribute / 3
43:     # Type error
44:     r8 = test.attribute2 / 3
45: 

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
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 10), 'str', 'Multiple if / else hasattr conditions with dynamic type checking inspections')
# Assigning a type to the variable '__doc__' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Test' class

    class Test:
        pass

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 8, 4, False)
            # Assigning a type to the variable 'self' (line 9)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Test.__init__', [], None, None, defaults, varargs, kwargs)

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

    
    # Assigning a type to the variable 'Test' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'Test', Test)
    
    # Assigning a Name to a Name (line 9):
    # Getting the type of 'None' (line 9)
    None_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 20), 'None')
    # Getting the type of 'Test'
    Test_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Test')
    # Setting the type of the member 'attribute' of a type
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Test_3, 'attribute', None_2)
    
    # Assigning a Name to a Name (line 10):
    # Getting the type of 'None' (line 10)
    None_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 21), 'None')
    # Getting the type of 'Test'
    Test_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Test')
    # Setting the type of the member 'attribute2' of a type
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Test_5, 'attribute2', None_4)
    
    # Assigning a Call to a Name (line 13):
    
    # Call to Test(...): (line 13)
    # Processing the call keyword arguments (line 13)
    kwargs_7 = {}
    # Getting the type of 'Test' (line 13)
    Test_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 11), 'Test', False)
    # Calling Test(args, kwargs) (line 13)
    Test_call_result_8 = invoke(stypy.reporting.localization.Localization(__file__, 13, 11), Test_6, *[], **kwargs_7)
    
    # Assigning a type to the variable 'test' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'test', Test_call_result_8)
    
    # Getting the type of 'True' (line 15)
    True_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 7), 'True')
    # Testing the type of an if condition (line 15)
    if_condition_10 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 15, 4), True_9)
    # Assigning a type to the variable 'if_condition_10' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'if_condition_10', if_condition_10)
    # SSA begins for if statement (line 15)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Attribute (line 16):
    int_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 25), 'int')
    # Getting the type of 'test' (line 16)
    test_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'test')
    # Setting the type of the member 'attribute' of a type (line 16)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), test_12, 'attribute', int_11)
    # SSA branch for the else part of an if statement (line 15)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'math' (line 18)
    math_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 11), 'math')
    # Obtaining the member 'pi' of a type (line 18)
    pi_14 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 11), math_13, 'pi')
    int_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 21), 'int')
    # Applying the binary operator '>' (line 18)
    result_gt_16 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 11), '>', pi_14, int_15)
    
    # Testing the type of an if condition (line 18)
    if_condition_17 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 18, 8), result_gt_16)
    # Assigning a type to the variable 'if_condition_17' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'if_condition_17', if_condition_17)
    # SSA begins for if statement (line 18)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Attribute (line 19):
    str_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 29), 'str', '3')
    # Getting the type of 'test' (line 19)
    test_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'test')
    # Setting the type of the member 'attribute' of a type (line 19)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 12), test_19, 'attribute', str_18)
    # SSA branch for the else part of an if statement (line 18)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a List to a Attribute (line 21):
    
    # Obtaining an instance of the builtin type 'list' (line 21)
    list_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 21)
    # Adding element type (line 21)
    int_21 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 31), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 30), list_20, int_21)
    # Adding element type (line 21)
    int_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 30), list_20, int_22)
    
    # Getting the type of 'test' (line 21)
    test_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'test')
    # Setting the type of the member 'attribute2' of a type (line 21)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 12), test_23, 'attribute2', list_20)
    # SSA join for if statement (line 18)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 15)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 23)
    str_24 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 21), 'str', 'attribute')
    # Getting the type of 'test' (line 23)
    test_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 15), 'test')
    
    (may_be_26, more_types_in_union_27) = may_provide_member(str_24, test_25)

    if may_be_26:

        if more_types_in_union_27:
            # Runtime conditional SSA (line 23)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'test' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'test', remove_not_member_provider_from_union(test_25, 'attribute'))
        
        # Assigning a BinOp to a Name (line 25):
        # Getting the type of 'test' (line 25)
        test_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'test')
        # Obtaining the member 'attribute' of a type (line 25)
        attribute_29 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 12), test_28, 'attribute')
        int_30 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 29), 'int')
        # Applying the binary operator 'div' (line 25)
        result_div_31 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 12), 'div', attribute_29, int_30)
        
        # Assigning a type to the variable 'r' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'r', result_div_31)
        
        # Assigning a Subscript to a Name (line 27):
        
        # Obtaining the type of the subscript
        int_32 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 28), 'int')
        # Getting the type of 'test' (line 27)
        test_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 13), 'test')
        # Obtaining the member 'attribute' of a type (line 27)
        attribute_34 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 13), test_33, 'attribute')
        # Obtaining the member '__getitem__' of a type (line 27)
        getitem___35 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 13), attribute_34, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 27)
        subscript_call_result_36 = invoke(stypy.reporting.localization.Localization(__file__, 27, 13), getitem___35, int_32)
        
        # Assigning a type to the variable 'r2' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'r2', subscript_call_result_36)

        if more_types_in_union_27:
            # SSA join for if statement (line 23)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 29)
    str_37 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 21), 'str', 'attribute2')
    # Getting the type of 'test' (line 29)
    test_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 15), 'test')
    
    (may_be_39, more_types_in_union_40) = may_provide_member(str_37, test_38)

    if may_be_39:

        if more_types_in_union_40:
            # Runtime conditional SSA (line 29)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'test' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'test', remove_not_member_provider_from_union(test_38, 'attribute2'))
        
        # Assigning a Subscript to a Name (line 31):
        
        # Obtaining the type of the subscript
        int_41 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 29), 'int')
        # Getting the type of 'test' (line 31)
        test_42 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 13), 'test')
        # Obtaining the member 'attribute2' of a type (line 31)
        attribute2_43 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 13), test_42, 'attribute2')
        # Obtaining the member '__getitem__' of a type (line 31)
        getitem___44 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 13), attribute2_43, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 31)
        subscript_call_result_45 = invoke(stypy.reporting.localization.Localization(__file__, 31, 13), getitem___44, int_41)
        
        # Assigning a type to the variable 'r3' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'r3', subscript_call_result_45)
        
        # Assigning a BinOp to a Name (line 33):
        # Getting the type of 'test' (line 33)
        test_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 13), 'test')
        # Obtaining the member 'attribute2' of a type (line 33)
        attribute2_47 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 13), test_46, 'attribute2')
        int_48 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 31), 'int')
        # Applying the binary operator 'div' (line 33)
        result_div_49 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 13), 'div', attribute2_47, int_48)
        
        # Assigning a type to the variable 'r4' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'r4', result_div_49)
        
        # Assigning a BinOp to a Name (line 35):
        # Getting the type of 'r3' (line 35)
        r3_50 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 16), 'r3')
        str_51 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 21), 'str', 'str')
        # Applying the binary operator '+' (line 35)
        result_add_52 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 16), '+', r3_50, str_51)
        
        # Assigning a type to the variable 'wrong' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'wrong', result_add_52)

        if more_types_in_union_40:
            # Runtime conditional SSA for else branch (line 29)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_39) or more_types_in_union_40):
        # Assigning a type to the variable 'test' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'test', remove_member_provider_from_union(test_38, 'attribute2'))
        
        # Assigning a Subscript to a Name (line 38):
        
        # Obtaining the type of the subscript
        int_53 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 28), 'int')
        # Getting the type of 'test' (line 38)
        test_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 13), 'test')
        # Obtaining the member 'attribute' of a type (line 38)
        attribute_55 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 13), test_54, 'attribute')
        # Obtaining the member '__getitem__' of a type (line 38)
        getitem___56 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 13), attribute_55, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 38)
        subscript_call_result_57 = invoke(stypy.reporting.localization.Localization(__file__, 38, 13), getitem___56, int_53)
        
        # Assigning a type to the variable 'r5' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'r5', subscript_call_result_57)
        
        # Assigning a BinOp to a Name (line 39):
        # Getting the type of 'test' (line 39)
        test_58 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 13), 'test')
        # Obtaining the member 'attribute2' of a type (line 39)
        attribute2_59 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 13), test_58, 'attribute2')
        int_60 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 31), 'int')
        # Applying the binary operator 'div' (line 39)
        result_div_61 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 13), 'div', attribute2_59, int_60)
        
        # Assigning a type to the variable 'r6' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'r6', result_div_61)

        if (may_be_39 and more_types_in_union_40):
            # SSA join for if statement (line 29)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a BinOp to a Name (line 42):
    # Getting the type of 'test' (line 42)
    test_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 9), 'test')
    # Obtaining the member 'attribute' of a type (line 42)
    attribute_63 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 9), test_62, 'attribute')
    int_64 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 26), 'int')
    # Applying the binary operator 'div' (line 42)
    result_div_65 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 9), 'div', attribute_63, int_64)
    
    # Assigning a type to the variable 'r7' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'r7', result_div_65)
    
    # Assigning a BinOp to a Name (line 44):
    # Getting the type of 'test' (line 44)
    test_66 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 9), 'test')
    # Obtaining the member 'attribute2' of a type (line 44)
    attribute2_67 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 9), test_66, 'attribute2')
    int_68 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 27), 'int')
    # Applying the binary operator 'div' (line 44)
    result_div_69 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 9), 'div', attribute2_67, int_68)
    
    # Assigning a type to the variable 'r8' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'r8', result_div_69)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
