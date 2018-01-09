
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: import math
3: 
4: __doc__ = "Multiple if / else hasattr conditions with dynamic type checking inspections and different return statements"
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
23: 
24:     def dynamic_type_ret_func(param):
25:         if hasattr(param, 'attribute'):
26:             # Type warning
27:             return param.attribute / 3
28: 
29:         if hasattr(param, 'attribute2'):
30:             # Type warning
31:             return param.attribute2[0]
32:         else:
33:             # Not executed
34:             return param.attribute + param.attribute
35: 
36: 
37:     r7 = dynamic_type_ret_func(test) / 3
38: 

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
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 10), 'str', 'Multiple if / else hasattr conditions with dynamic type checking inspections and different return statements')
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
    

    @norecursion
    def dynamic_type_ret_func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'dynamic_type_ret_func'
        module_type_store = module_type_store.open_function_context('dynamic_type_ret_func', 24, 4, False)
        
        # Passed parameters checking function
        dynamic_type_ret_func.stypy_localization = localization
        dynamic_type_ret_func.stypy_type_of_self = None
        dynamic_type_ret_func.stypy_type_store = module_type_store
        dynamic_type_ret_func.stypy_function_name = 'dynamic_type_ret_func'
        dynamic_type_ret_func.stypy_param_names_list = ['param']
        dynamic_type_ret_func.stypy_varargs_param_name = None
        dynamic_type_ret_func.stypy_kwargs_param_name = None
        dynamic_type_ret_func.stypy_call_defaults = defaults
        dynamic_type_ret_func.stypy_call_varargs = varargs
        dynamic_type_ret_func.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'dynamic_type_ret_func', ['param'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'dynamic_type_ret_func', localization, ['param'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'dynamic_type_ret_func(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 25)
        str_24 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 26), 'str', 'attribute')
        # Getting the type of 'param' (line 25)
        param_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 19), 'param')
        
        (may_be_26, more_types_in_union_27) = may_provide_member(str_24, param_25)

        if may_be_26:

            if more_types_in_union_27:
                # Runtime conditional SSA (line 25)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'param' (line 25)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'param', remove_not_member_provider_from_union(param_25, 'attribute'))
            # Getting the type of 'param' (line 27)
            param_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 19), 'param')
            # Obtaining the member 'attribute' of a type (line 27)
            attribute_29 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 19), param_28, 'attribute')
            int_30 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 37), 'int')
            # Applying the binary operator 'div' (line 27)
            result_div_31 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 19), 'div', attribute_29, int_30)
            
            # Assigning a type to the variable 'stypy_return_type' (line 27)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'stypy_return_type', result_div_31)

            if more_types_in_union_27:
                # SSA join for if statement (line 25)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 29)
        str_32 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 26), 'str', 'attribute2')
        # Getting the type of 'param' (line 29)
        param_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 19), 'param')
        
        (may_be_34, more_types_in_union_35) = may_provide_member(str_32, param_33)

        if may_be_34:

            if more_types_in_union_35:
                # Runtime conditional SSA (line 29)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'param' (line 29)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'param', remove_not_member_provider_from_union(param_33, 'attribute2'))
            
            # Obtaining the type of the subscript
            int_36 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 36), 'int')
            # Getting the type of 'param' (line 31)
            param_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 19), 'param')
            # Obtaining the member 'attribute2' of a type (line 31)
            attribute2_38 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 19), param_37, 'attribute2')
            # Obtaining the member '__getitem__' of a type (line 31)
            getitem___39 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 19), attribute2_38, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 31)
            subscript_call_result_40 = invoke(stypy.reporting.localization.Localization(__file__, 31, 19), getitem___39, int_36)
            
            # Assigning a type to the variable 'stypy_return_type' (line 31)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'stypy_return_type', subscript_call_result_40)

            if more_types_in_union_35:
                # Runtime conditional SSA for else branch (line 29)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_34) or more_types_in_union_35):
            # Assigning a type to the variable 'param' (line 29)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'param', remove_member_provider_from_union(param_33, 'attribute2'))
            # Getting the type of 'param' (line 34)
            param_41 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 19), 'param')
            # Obtaining the member 'attribute' of a type (line 34)
            attribute_42 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 19), param_41, 'attribute')
            # Getting the type of 'param' (line 34)
            param_43 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 37), 'param')
            # Obtaining the member 'attribute' of a type (line 34)
            attribute_44 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 37), param_43, 'attribute')
            # Applying the binary operator '+' (line 34)
            result_add_45 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 19), '+', attribute_42, attribute_44)
            
            # Assigning a type to the variable 'stypy_return_type' (line 34)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'stypy_return_type', result_add_45)

            if (may_be_34 and more_types_in_union_35):
                # SSA join for if statement (line 29)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'dynamic_type_ret_func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'dynamic_type_ret_func' in the type store
        # Getting the type of 'stypy_return_type' (line 24)
        stypy_return_type_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_46)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dynamic_type_ret_func'
        return stypy_return_type_46

    # Assigning a type to the variable 'dynamic_type_ret_func' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'dynamic_type_ret_func', dynamic_type_ret_func)
    
    # Assigning a BinOp to a Name (line 37):
    
    # Call to dynamic_type_ret_func(...): (line 37)
    # Processing the call arguments (line 37)
    # Getting the type of 'test' (line 37)
    test_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 31), 'test', False)
    # Processing the call keyword arguments (line 37)
    kwargs_49 = {}
    # Getting the type of 'dynamic_type_ret_func' (line 37)
    dynamic_type_ret_func_47 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 9), 'dynamic_type_ret_func', False)
    # Calling dynamic_type_ret_func(args, kwargs) (line 37)
    dynamic_type_ret_func_call_result_50 = invoke(stypy.reporting.localization.Localization(__file__, 37, 9), dynamic_type_ret_func_47, *[test_48], **kwargs_49)
    
    int_51 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 39), 'int')
    # Applying the binary operator 'div' (line 37)
    result_div_52 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 9), 'div', dynamic_type_ret_func_call_result_50, int_51)
    
    # Assigning a type to the variable 'r7' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'r7', result_div_52)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
