
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # -------------------
2: # Recursive functions
3: # ------------------
4: 
5: import traceback
6: 
7: 
8: class RecursionType:
9:     pass
10: 
11: 
12: def norecursion(f):
13:     '''
14:     Annotation that detects recursive functions, returning a RecursionType instance
15:     :param f: Function
16:     :return: RecursionType if the function is recursive or the passed function if not
17:     '''
18:     if isinstance(f, classmethod) or isinstance(f, staticmethod):
19:         func_name = f.__func__.func_name
20:     else:
21:         func_name = f.func_name
22: 
23:     def func(*args, **kwargs):
24:         if isinstance(f, classmethod) or isinstance(f, staticmethod):
25:             func_name = f.__func__.func_name
26:             fun = f#.__func__
27:         else:
28:             func_name = f.func_name
29:             fun = f
30: 
31:         if len([l[2] for l in traceback.extract_stack() if l[2] == func_name]) > 0:
32:             return RecursionType()  # RecursionType is returned when recursion is detected
33:         return fun(*args, **kwargs)
34: 
35:     func.__name__ = func_name
36: 
37:     return func
38: 
39: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import traceback' statement (line 5)
import traceback

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'traceback', traceback, module_type_store)

# Declaration of the 'RecursionType' class

class RecursionType:
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 8, 0, False)
        # Assigning a type to the variable 'self' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RecursionType.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'RecursionType' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'RecursionType', RecursionType)

@norecursion
def norecursion(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'norecursion'
    module_type_store = module_type_store.open_function_context('norecursion', 12, 0, False)
    
    # Passed parameters checking function
    norecursion.stypy_localization = localization
    norecursion.stypy_type_of_self = None
    norecursion.stypy_type_store = module_type_store
    norecursion.stypy_function_name = 'norecursion'
    norecursion.stypy_param_names_list = ['f']
    norecursion.stypy_varargs_param_name = None
    norecursion.stypy_kwargs_param_name = None
    norecursion.stypy_call_defaults = defaults
    norecursion.stypy_call_varargs = varargs
    norecursion.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'norecursion', ['f'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'norecursion', localization, ['f'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'norecursion(...)' code ##################

    str_9773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, (-1)), 'str', '\n    Annotation that detects recursive functions, returning a RecursionType instance\n    :param f: Function\n    :return: RecursionType if the function is recursive or the passed function if not\n    ')
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'f' (line 18)
    f_9775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 18), 'f', False)
    # Getting the type of 'classmethod' (line 18)
    classmethod_9776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 21), 'classmethod', False)
    # Processing the call keyword arguments (line 18)
    kwargs_9777 = {}
    # Getting the type of 'isinstance' (line 18)
    isinstance_9774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 18)
    isinstance_call_result_9778 = invoke(stypy.reporting.localization.Localization(__file__, 18, 7), isinstance_9774, *[f_9775, classmethod_9776], **kwargs_9777)
    
    
    # Call to isinstance(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'f' (line 18)
    f_9780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 48), 'f', False)
    # Getting the type of 'staticmethod' (line 18)
    staticmethod_9781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 51), 'staticmethod', False)
    # Processing the call keyword arguments (line 18)
    kwargs_9782 = {}
    # Getting the type of 'isinstance' (line 18)
    isinstance_9779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 37), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 18)
    isinstance_call_result_9783 = invoke(stypy.reporting.localization.Localization(__file__, 18, 37), isinstance_9779, *[f_9780, staticmethod_9781], **kwargs_9782)
    
    # Applying the binary operator 'or' (line 18)
    result_or_keyword_9784 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 7), 'or', isinstance_call_result_9778, isinstance_call_result_9783)
    
    # Testing if the type of an if condition is none (line 18)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 18, 4), result_or_keyword_9784):
        
        # Assigning a Attribute to a Name (line 21):
        # Getting the type of 'f' (line 21)
        f_9789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 20), 'f')
        # Obtaining the member 'func_name' of a type (line 21)
        func_name_9790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 20), f_9789, 'func_name')
        # Assigning a type to the variable 'func_name' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'func_name', func_name_9790)
    else:
        
        # Testing the type of an if condition (line 18)
        if_condition_9785 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 18, 4), result_or_keyword_9784)
        # Assigning a type to the variable 'if_condition_9785' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'if_condition_9785', if_condition_9785)
        # SSA begins for if statement (line 18)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 19):
        # Getting the type of 'f' (line 19)
        f_9786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 20), 'f')
        # Obtaining the member '__func__' of a type (line 19)
        func___9787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 20), f_9786, '__func__')
        # Obtaining the member 'func_name' of a type (line 19)
        func_name_9788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 20), func___9787, 'func_name')
        # Assigning a type to the variable 'func_name' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'func_name', func_name_9788)
        # SSA branch for the else part of an if statement (line 18)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 21):
        # Getting the type of 'f' (line 21)
        f_9789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 20), 'f')
        # Obtaining the member 'func_name' of a type (line 21)
        func_name_9790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 20), f_9789, 'func_name')
        # Assigning a type to the variable 'func_name' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'func_name', func_name_9790)
        # SSA join for if statement (line 18)
        module_type_store = module_type_store.join_ssa_context()
        


    @norecursion
    def func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'func'
        module_type_store = module_type_store.open_function_context('func', 23, 4, False)
        
        # Passed parameters checking function
        func.stypy_localization = localization
        func.stypy_type_of_self = None
        func.stypy_type_store = module_type_store
        func.stypy_function_name = 'func'
        func.stypy_param_names_list = []
        func.stypy_varargs_param_name = 'args'
        func.stypy_kwargs_param_name = 'kwargs'
        func.stypy_call_defaults = defaults
        func.stypy_call_varargs = varargs
        func.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'func', [], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'func', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'func(...)' code ##################

        
        # Evaluating a boolean operation
        
        # Call to isinstance(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'f' (line 24)
        f_9792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 22), 'f', False)
        # Getting the type of 'classmethod' (line 24)
        classmethod_9793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 25), 'classmethod', False)
        # Processing the call keyword arguments (line 24)
        kwargs_9794 = {}
        # Getting the type of 'isinstance' (line 24)
        isinstance_9791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 24)
        isinstance_call_result_9795 = invoke(stypy.reporting.localization.Localization(__file__, 24, 11), isinstance_9791, *[f_9792, classmethod_9793], **kwargs_9794)
        
        
        # Call to isinstance(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'f' (line 24)
        f_9797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 52), 'f', False)
        # Getting the type of 'staticmethod' (line 24)
        staticmethod_9798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 55), 'staticmethod', False)
        # Processing the call keyword arguments (line 24)
        kwargs_9799 = {}
        # Getting the type of 'isinstance' (line 24)
        isinstance_9796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 41), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 24)
        isinstance_call_result_9800 = invoke(stypy.reporting.localization.Localization(__file__, 24, 41), isinstance_9796, *[f_9797, staticmethod_9798], **kwargs_9799)
        
        # Applying the binary operator 'or' (line 24)
        result_or_keyword_9801 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 11), 'or', isinstance_call_result_9795, isinstance_call_result_9800)
        
        # Testing if the type of an if condition is none (line 24)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 24, 8), result_or_keyword_9801):
            
            # Assigning a Attribute to a Name (line 28):
            # Getting the type of 'f' (line 28)
            f_9807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 24), 'f')
            # Obtaining the member 'func_name' of a type (line 28)
            func_name_9808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 24), f_9807, 'func_name')
            # Assigning a type to the variable 'func_name' (line 28)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'func_name', func_name_9808)
            
            # Assigning a Name to a Name (line 29):
            # Getting the type of 'f' (line 29)
            f_9809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 18), 'f')
            # Assigning a type to the variable 'fun' (line 29)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'fun', f_9809)
        else:
            
            # Testing the type of an if condition (line 24)
            if_condition_9802 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 24, 8), result_or_keyword_9801)
            # Assigning a type to the variable 'if_condition_9802' (line 24)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'if_condition_9802', if_condition_9802)
            # SSA begins for if statement (line 24)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Name (line 25):
            # Getting the type of 'f' (line 25)
            f_9803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 24), 'f')
            # Obtaining the member '__func__' of a type (line 25)
            func___9804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 24), f_9803, '__func__')
            # Obtaining the member 'func_name' of a type (line 25)
            func_name_9805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 24), func___9804, 'func_name')
            # Assigning a type to the variable 'func_name' (line 25)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'func_name', func_name_9805)
            
            # Assigning a Name to a Name (line 26):
            # Getting the type of 'f' (line 26)
            f_9806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 18), 'f')
            # Assigning a type to the variable 'fun' (line 26)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'fun', f_9806)
            # SSA branch for the else part of an if statement (line 24)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Attribute to a Name (line 28):
            # Getting the type of 'f' (line 28)
            f_9807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 24), 'f')
            # Obtaining the member 'func_name' of a type (line 28)
            func_name_9808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 24), f_9807, 'func_name')
            # Assigning a type to the variable 'func_name' (line 28)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'func_name', func_name_9808)
            
            # Assigning a Name to a Name (line 29):
            # Getting the type of 'f' (line 29)
            f_9809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 18), 'f')
            # Assigning a type to the variable 'fun' (line 29)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'fun', f_9809)
            # SSA join for if statement (line 24)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to len(...): (line 31)
        # Processing the call arguments (line 31)
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to extract_stack(...): (line 31)
        # Processing the call keyword arguments (line 31)
        kwargs_9823 = {}
        # Getting the type of 'traceback' (line 31)
        traceback_9821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 30), 'traceback', False)
        # Obtaining the member 'extract_stack' of a type (line 31)
        extract_stack_9822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 30), traceback_9821, 'extract_stack')
        # Calling extract_stack(args, kwargs) (line 31)
        extract_stack_call_result_9824 = invoke(stypy.reporting.localization.Localization(__file__, 31, 30), extract_stack_9822, *[], **kwargs_9823)
        
        comprehension_9825 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 16), extract_stack_call_result_9824)
        # Assigning a type to the variable 'l' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 16), 'l', comprehension_9825)
        
        
        # Obtaining the type of the subscript
        int_9815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 61), 'int')
        # Getting the type of 'l' (line 31)
        l_9816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 59), 'l', False)
        # Obtaining the member '__getitem__' of a type (line 31)
        getitem___9817 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 59), l_9816, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 31)
        subscript_call_result_9818 = invoke(stypy.reporting.localization.Localization(__file__, 31, 59), getitem___9817, int_9815)
        
        # Getting the type of 'func_name' (line 31)
        func_name_9819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 67), 'func_name', False)
        # Applying the binary operator '==' (line 31)
        result_eq_9820 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 59), '==', subscript_call_result_9818, func_name_9819)
        
        
        # Obtaining the type of the subscript
        int_9811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 18), 'int')
        # Getting the type of 'l' (line 31)
        l_9812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 16), 'l', False)
        # Obtaining the member '__getitem__' of a type (line 31)
        getitem___9813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 16), l_9812, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 31)
        subscript_call_result_9814 = invoke(stypy.reporting.localization.Localization(__file__, 31, 16), getitem___9813, int_9811)
        
        list_9826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 16), list_9826, subscript_call_result_9814)
        # Processing the call keyword arguments (line 31)
        kwargs_9827 = {}
        # Getting the type of 'len' (line 31)
        len_9810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 11), 'len', False)
        # Calling len(args, kwargs) (line 31)
        len_call_result_9828 = invoke(stypy.reporting.localization.Localization(__file__, 31, 11), len_9810, *[list_9826], **kwargs_9827)
        
        int_9829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 81), 'int')
        # Applying the binary operator '>' (line 31)
        result_gt_9830 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 11), '>', len_call_result_9828, int_9829)
        
        # Testing if the type of an if condition is none (line 31)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 31, 8), result_gt_9830):
            pass
        else:
            
            # Testing the type of an if condition (line 31)
            if_condition_9831 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 31, 8), result_gt_9830)
            # Assigning a type to the variable 'if_condition_9831' (line 31)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'if_condition_9831', if_condition_9831)
            # SSA begins for if statement (line 31)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to RecursionType(...): (line 32)
            # Processing the call keyword arguments (line 32)
            kwargs_9833 = {}
            # Getting the type of 'RecursionType' (line 32)
            RecursionType_9832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 19), 'RecursionType', False)
            # Calling RecursionType(args, kwargs) (line 32)
            RecursionType_call_result_9834 = invoke(stypy.reporting.localization.Localization(__file__, 32, 19), RecursionType_9832, *[], **kwargs_9833)
            
            # Assigning a type to the variable 'stypy_return_type' (line 32)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'stypy_return_type', RecursionType_call_result_9834)
            # SSA join for if statement (line 31)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to fun(...): (line 33)
        # Getting the type of 'args' (line 33)
        args_9836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 20), 'args', False)
        # Processing the call keyword arguments (line 33)
        # Getting the type of 'kwargs' (line 33)
        kwargs_9837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 28), 'kwargs', False)
        kwargs_9838 = {'kwargs_9837': kwargs_9837}
        # Getting the type of 'fun' (line 33)
        fun_9835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 15), 'fun', False)
        # Calling fun(args, kwargs) (line 33)
        fun_call_result_9839 = invoke(stypy.reporting.localization.Localization(__file__, 33, 15), fun_9835, *[args_9836], **kwargs_9838)
        
        # Assigning a type to the variable 'stypy_return_type' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'stypy_return_type', fun_call_result_9839)
        
        # ################# End of 'func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'func' in the type store
        # Getting the type of 'stypy_return_type' (line 23)
        stypy_return_type_9840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_9840)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'func'
        return stypy_return_type_9840

    # Assigning a type to the variable 'func' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'func', func)
    
    # Assigning a Name to a Attribute (line 35):
    # Getting the type of 'func_name' (line 35)
    func_name_9841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 20), 'func_name')
    # Getting the type of 'func' (line 35)
    func_9842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'func')
    # Setting the type of the member '__name__' of a type (line 35)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 4), func_9842, '__name__', func_name_9841)
    # Getting the type of 'func' (line 37)
    func_9843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 11), 'func')
    # Assigning a type to the variable 'stypy_return_type' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type', func_9843)
    
    # ################# End of 'norecursion(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'norecursion' in the type store
    # Getting the type of 'stypy_return_type' (line 12)
    stypy_return_type_9844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_9844)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'norecursion'
    return stypy_return_type_9844

# Assigning a type to the variable 'norecursion' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'norecursion', norecursion)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
