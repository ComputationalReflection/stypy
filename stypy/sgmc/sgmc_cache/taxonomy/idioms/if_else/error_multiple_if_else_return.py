
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: import math
3: 
4: __doc__ = "Multiple if / else conditions with dynamic type checking inspections and different return statements"
5: 
6: if __name__ == '__main__':
7: 
8:     def dynamic_type_ret_func(param):
9:         b = None
10:         if type(param) is int:
11:             return param / 3
12:         else:
13:             if type(param) is str:
14:                 return param[0]
15:             else:
16:                 return param + param
17: 
18: 
19:     if True:
20:         a = 3
21:     else:
22:         if math.pi > 3:
23:             a = "3"
24:         else:
25:             a = [1, 2]
26: 
27:     # Type warning
28:     r7 = dynamic_type_ret_func(a) / 3
29: 

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
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 10), 'str', 'Multiple if / else conditions with dynamic type checking inspections and different return statements')
# Assigning a type to the variable '__doc__' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), '__doc__', str_1)

if (__name__ == '__main__'):

    @norecursion
    def dynamic_type_ret_func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'dynamic_type_ret_func'
        module_type_store = module_type_store.open_function_context('dynamic_type_ret_func', 8, 4, False)
        
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

        
        # Assigning a Name to a Name (line 9):
        # Getting the type of 'None' (line 9)
        None_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 12), 'None')
        # Assigning a type to the variable 'b' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'b', None_2)
        
        # Type idiom detected: calculating its left and rigth part (line 10)
        # Getting the type of 'param' (line 10)
        param_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 16), 'param')
        # Getting the type of 'int' (line 10)
        int_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 26), 'int')
        
        (may_be_5, more_types_in_union_6) = may_be_type(param_3, int_4)

        if may_be_5:

            if more_types_in_union_6:
                # Runtime conditional SSA (line 10)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'param' (line 10)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'param', int_4())
            # Getting the type of 'param' (line 11)
            param_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 19), 'param')
            int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 27), 'int')
            # Applying the binary operator 'div' (line 11)
            result_div_9 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 19), 'div', param_7, int_8)
            
            # Assigning a type to the variable 'stypy_return_type' (line 11)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 12), 'stypy_return_type', result_div_9)

            if more_types_in_union_6:
                # Runtime conditional SSA for else branch (line 10)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_5) or more_types_in_union_6):
            # Getting the type of 'param' (line 10)
            param_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'param')
            # Assigning a type to the variable 'param' (line 10)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'param', remove_type_from_union(param_10, int_4))
            
            # Type idiom detected: calculating its left and rigth part (line 13)
            # Getting the type of 'param' (line 13)
            param_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 20), 'param')
            # Getting the type of 'str' (line 13)
            str_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 30), 'str')
            
            (may_be_13, more_types_in_union_14) = may_be_type(param_11, str_12)

            if may_be_13:

                if more_types_in_union_14:
                    # Runtime conditional SSA (line 13)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'param' (line 13)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 12), 'param', str_12())
                
                # Obtaining the type of the subscript
                int_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 29), 'int')
                # Getting the type of 'param' (line 14)
                param_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 23), 'param')
                # Obtaining the member '__getitem__' of a type (line 14)
                getitem___17 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 23), param_16, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 14)
                subscript_call_result_18 = invoke(stypy.reporting.localization.Localization(__file__, 14, 23), getitem___17, int_15)
                
                # Assigning a type to the variable 'stypy_return_type' (line 14)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 16), 'stypy_return_type', subscript_call_result_18)

                if more_types_in_union_14:
                    # Runtime conditional SSA for else branch (line 13)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_13) or more_types_in_union_14):
                # Getting the type of 'param' (line 13)
                param_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 12), 'param')
                # Assigning a type to the variable 'param' (line 13)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 12), 'param', remove_type_from_union(param_19, str_12))
                # Getting the type of 'param' (line 16)
                param_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 23), 'param')
                # Getting the type of 'param' (line 16)
                param_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 31), 'param')
                # Applying the binary operator '+' (line 16)
                result_add_22 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 23), '+', param_20, param_21)
                
                # Assigning a type to the variable 'stypy_return_type' (line 16)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 16), 'stypy_return_type', result_add_22)

                if (may_be_13 and more_types_in_union_14):
                    # SSA join for if statement (line 13)
                    module_type_store = module_type_store.join_ssa_context()


            

            if (may_be_5 and more_types_in_union_6):
                # SSA join for if statement (line 10)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'dynamic_type_ret_func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'dynamic_type_ret_func' in the type store
        # Getting the type of 'stypy_return_type' (line 8)
        stypy_return_type_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_23)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'dynamic_type_ret_func'
        return stypy_return_type_23

    # Assigning a type to the variable 'dynamic_type_ret_func' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'dynamic_type_ret_func', dynamic_type_ret_func)
    
    # Getting the type of 'True' (line 19)
    True_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 7), 'True')
    # Testing the type of an if condition (line 19)
    if_condition_25 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 19, 4), True_24)
    # Assigning a type to the variable 'if_condition_25' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'if_condition_25', if_condition_25)
    # SSA begins for if statement (line 19)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 20):
    int_26 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 12), 'int')
    # Assigning a type to the variable 'a' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'a', int_26)
    # SSA branch for the else part of an if statement (line 19)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'math' (line 22)
    math_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 11), 'math')
    # Obtaining the member 'pi' of a type (line 22)
    pi_28 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 11), math_27, 'pi')
    int_29 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 21), 'int')
    # Applying the binary operator '>' (line 22)
    result_gt_30 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 11), '>', pi_28, int_29)
    
    # Testing the type of an if condition (line 22)
    if_condition_31 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 22, 8), result_gt_30)
    # Assigning a type to the variable 'if_condition_31' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'if_condition_31', if_condition_31)
    # SSA begins for if statement (line 22)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Str to a Name (line 23):
    str_32 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 16), 'str', '3')
    # Assigning a type to the variable 'a' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'a', str_32)
    # SSA branch for the else part of an if statement (line 22)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a List to a Name (line 25):
    
    # Obtaining an instance of the builtin type 'list' (line 25)
    list_33 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 25)
    # Adding element type (line 25)
    int_34 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 16), list_33, int_34)
    # Adding element type (line 25)
    int_35 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 16), list_33, int_35)
    
    # Assigning a type to the variable 'a' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'a', list_33)
    # SSA join for if statement (line 22)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 19)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 28):
    
    # Call to dynamic_type_ret_func(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'a' (line 28)
    a_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 31), 'a', False)
    # Processing the call keyword arguments (line 28)
    kwargs_38 = {}
    # Getting the type of 'dynamic_type_ret_func' (line 28)
    dynamic_type_ret_func_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 9), 'dynamic_type_ret_func', False)
    # Calling dynamic_type_ret_func(args, kwargs) (line 28)
    dynamic_type_ret_func_call_result_39 = invoke(stypy.reporting.localization.Localization(__file__, 28, 9), dynamic_type_ret_func_36, *[a_37], **kwargs_38)
    
    int_40 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 36), 'int')
    # Applying the binary operator 'div' (line 28)
    result_div_41 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 9), 'div', dynamic_type_ret_func_call_result_39, int_40)
    
    # Assigning a type to the variable 'r7' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'r7', result_div_41)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
