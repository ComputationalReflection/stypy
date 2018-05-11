
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

    str_10059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, (-1)), 'str', '\n    Annotation that detects recursive functions, returning a RecursionType instance\n    :param f: Function\n    :return: RecursionType if the function is recursive or the passed function if not\n    ')
    
    # Evaluating a boolean operation
    
    # Call to isinstance(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'f' (line 18)
    f_10061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 18), 'f', False)
    # Getting the type of 'classmethod' (line 18)
    classmethod_10062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 21), 'classmethod', False)
    # Processing the call keyword arguments (line 18)
    kwargs_10063 = {}
    # Getting the type of 'isinstance' (line 18)
    isinstance_10060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 18)
    isinstance_call_result_10064 = invoke(stypy.reporting.localization.Localization(__file__, 18, 7), isinstance_10060, *[f_10061, classmethod_10062], **kwargs_10063)
    
    
    # Call to isinstance(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'f' (line 18)
    f_10066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 48), 'f', False)
    # Getting the type of 'staticmethod' (line 18)
    staticmethod_10067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 51), 'staticmethod', False)
    # Processing the call keyword arguments (line 18)
    kwargs_10068 = {}
    # Getting the type of 'isinstance' (line 18)
    isinstance_10065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 37), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 18)
    isinstance_call_result_10069 = invoke(stypy.reporting.localization.Localization(__file__, 18, 37), isinstance_10065, *[f_10066, staticmethod_10067], **kwargs_10068)
    
    # Applying the binary operator 'or' (line 18)
    result_or_keyword_10070 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 7), 'or', isinstance_call_result_10064, isinstance_call_result_10069)
    
    # Testing if the type of an if condition is none (line 18)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 18, 4), result_or_keyword_10070):
        
        # Assigning a Attribute to a Name (line 21):
        # Getting the type of 'f' (line 21)
        f_10075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 20), 'f')
        # Obtaining the member 'func_name' of a type (line 21)
        func_name_10076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 20), f_10075, 'func_name')
        # Assigning a type to the variable 'func_name' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'func_name', func_name_10076)
    else:
        
        # Testing the type of an if condition (line 18)
        if_condition_10071 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 18, 4), result_or_keyword_10070)
        # Assigning a type to the variable 'if_condition_10071' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'if_condition_10071', if_condition_10071)
        # SSA begins for if statement (line 18)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 19):
        # Getting the type of 'f' (line 19)
        f_10072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 20), 'f')
        # Obtaining the member '__func__' of a type (line 19)
        func___10073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 20), f_10072, '__func__')
        # Obtaining the member 'func_name' of a type (line 19)
        func_name_10074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 20), func___10073, 'func_name')
        # Assigning a type to the variable 'func_name' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'func_name', func_name_10074)
        # SSA branch for the else part of an if statement (line 18)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Attribute to a Name (line 21):
        # Getting the type of 'f' (line 21)
        f_10075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 20), 'f')
        # Obtaining the member 'func_name' of a type (line 21)
        func_name_10076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 20), f_10075, 'func_name')
        # Assigning a type to the variable 'func_name' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'func_name', func_name_10076)
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
        f_10078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 22), 'f', False)
        # Getting the type of 'classmethod' (line 24)
        classmethod_10079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 25), 'classmethod', False)
        # Processing the call keyword arguments (line 24)
        kwargs_10080 = {}
        # Getting the type of 'isinstance' (line 24)
        isinstance_10077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 24)
        isinstance_call_result_10081 = invoke(stypy.reporting.localization.Localization(__file__, 24, 11), isinstance_10077, *[f_10078, classmethod_10079], **kwargs_10080)
        
        
        # Call to isinstance(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'f' (line 24)
        f_10083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 52), 'f', False)
        # Getting the type of 'staticmethod' (line 24)
        staticmethod_10084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 55), 'staticmethod', False)
        # Processing the call keyword arguments (line 24)
        kwargs_10085 = {}
        # Getting the type of 'isinstance' (line 24)
        isinstance_10082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 41), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 24)
        isinstance_call_result_10086 = invoke(stypy.reporting.localization.Localization(__file__, 24, 41), isinstance_10082, *[f_10083, staticmethod_10084], **kwargs_10085)
        
        # Applying the binary operator 'or' (line 24)
        result_or_keyword_10087 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 11), 'or', isinstance_call_result_10081, isinstance_call_result_10086)
        
        # Testing if the type of an if condition is none (line 24)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 24, 8), result_or_keyword_10087):
            
            # Assigning a Attribute to a Name (line 28):
            # Getting the type of 'f' (line 28)
            f_10093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 24), 'f')
            # Obtaining the member 'func_name' of a type (line 28)
            func_name_10094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 24), f_10093, 'func_name')
            # Assigning a type to the variable 'func_name' (line 28)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'func_name', func_name_10094)
            
            # Assigning a Name to a Name (line 29):
            # Getting the type of 'f' (line 29)
            f_10095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 18), 'f')
            # Assigning a type to the variable 'fun' (line 29)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'fun', f_10095)
        else:
            
            # Testing the type of an if condition (line 24)
            if_condition_10088 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 24, 8), result_or_keyword_10087)
            # Assigning a type to the variable 'if_condition_10088' (line 24)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'if_condition_10088', if_condition_10088)
            # SSA begins for if statement (line 24)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Name (line 25):
            # Getting the type of 'f' (line 25)
            f_10089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 24), 'f')
            # Obtaining the member '__func__' of a type (line 25)
            func___10090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 24), f_10089, '__func__')
            # Obtaining the member 'func_name' of a type (line 25)
            func_name_10091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 24), func___10090, 'func_name')
            # Assigning a type to the variable 'func_name' (line 25)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'func_name', func_name_10091)
            
            # Assigning a Name to a Name (line 26):
            # Getting the type of 'f' (line 26)
            f_10092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 18), 'f')
            # Assigning a type to the variable 'fun' (line 26)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'fun', f_10092)
            # SSA branch for the else part of an if statement (line 24)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Attribute to a Name (line 28):
            # Getting the type of 'f' (line 28)
            f_10093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 24), 'f')
            # Obtaining the member 'func_name' of a type (line 28)
            func_name_10094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 24), f_10093, 'func_name')
            # Assigning a type to the variable 'func_name' (line 28)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'func_name', func_name_10094)
            
            # Assigning a Name to a Name (line 29):
            # Getting the type of 'f' (line 29)
            f_10095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 18), 'f')
            # Assigning a type to the variable 'fun' (line 29)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'fun', f_10095)
            # SSA join for if statement (line 24)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to len(...): (line 31)
        # Processing the call arguments (line 31)
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to extract_stack(...): (line 31)
        # Processing the call keyword arguments (line 31)
        kwargs_10109 = {}
        # Getting the type of 'traceback' (line 31)
        traceback_10107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 30), 'traceback', False)
        # Obtaining the member 'extract_stack' of a type (line 31)
        extract_stack_10108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 30), traceback_10107, 'extract_stack')
        # Calling extract_stack(args, kwargs) (line 31)
        extract_stack_call_result_10110 = invoke(stypy.reporting.localization.Localization(__file__, 31, 30), extract_stack_10108, *[], **kwargs_10109)
        
        comprehension_10111 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 16), extract_stack_call_result_10110)
        # Assigning a type to the variable 'l' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 16), 'l', comprehension_10111)
        
        
        # Obtaining the type of the subscript
        int_10101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 61), 'int')
        # Getting the type of 'l' (line 31)
        l_10102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 59), 'l', False)
        # Obtaining the member '__getitem__' of a type (line 31)
        getitem___10103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 59), l_10102, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 31)
        subscript_call_result_10104 = invoke(stypy.reporting.localization.Localization(__file__, 31, 59), getitem___10103, int_10101)
        
        # Getting the type of 'func_name' (line 31)
        func_name_10105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 67), 'func_name', False)
        # Applying the binary operator '==' (line 31)
        result_eq_10106 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 59), '==', subscript_call_result_10104, func_name_10105)
        
        
        # Obtaining the type of the subscript
        int_10097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 18), 'int')
        # Getting the type of 'l' (line 31)
        l_10098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 16), 'l', False)
        # Obtaining the member '__getitem__' of a type (line 31)
        getitem___10099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 16), l_10098, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 31)
        subscript_call_result_10100 = invoke(stypy.reporting.localization.Localization(__file__, 31, 16), getitem___10099, int_10097)
        
        list_10112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 16), list_10112, subscript_call_result_10100)
        # Processing the call keyword arguments (line 31)
        kwargs_10113 = {}
        # Getting the type of 'len' (line 31)
        len_10096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 11), 'len', False)
        # Calling len(args, kwargs) (line 31)
        len_call_result_10114 = invoke(stypy.reporting.localization.Localization(__file__, 31, 11), len_10096, *[list_10112], **kwargs_10113)
        
        int_10115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 81), 'int')
        # Applying the binary operator '>' (line 31)
        result_gt_10116 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 11), '>', len_call_result_10114, int_10115)
        
        # Testing if the type of an if condition is none (line 31)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 31, 8), result_gt_10116):
            pass
        else:
            
            # Testing the type of an if condition (line 31)
            if_condition_10117 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 31, 8), result_gt_10116)
            # Assigning a type to the variable 'if_condition_10117' (line 31)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'if_condition_10117', if_condition_10117)
            # SSA begins for if statement (line 31)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to RecursionType(...): (line 32)
            # Processing the call keyword arguments (line 32)
            kwargs_10119 = {}
            # Getting the type of 'RecursionType' (line 32)
            RecursionType_10118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 19), 'RecursionType', False)
            # Calling RecursionType(args, kwargs) (line 32)
            RecursionType_call_result_10120 = invoke(stypy.reporting.localization.Localization(__file__, 32, 19), RecursionType_10118, *[], **kwargs_10119)
            
            # Assigning a type to the variable 'stypy_return_type' (line 32)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'stypy_return_type', RecursionType_call_result_10120)
            # SSA join for if statement (line 31)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to fun(...): (line 33)
        # Getting the type of 'args' (line 33)
        args_10122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 20), 'args', False)
        # Processing the call keyword arguments (line 33)
        # Getting the type of 'kwargs' (line 33)
        kwargs_10123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 28), 'kwargs', False)
        kwargs_10124 = {'kwargs_10123': kwargs_10123}
        # Getting the type of 'fun' (line 33)
        fun_10121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 15), 'fun', False)
        # Calling fun(args, kwargs) (line 33)
        fun_call_result_10125 = invoke(stypy.reporting.localization.Localization(__file__, 33, 15), fun_10121, *[args_10122], **kwargs_10124)
        
        # Assigning a type to the variable 'stypy_return_type' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'stypy_return_type', fun_call_result_10125)
        
        # ################# End of 'func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'func' in the type store
        # Getting the type of 'stypy_return_type' (line 23)
        stypy_return_type_10126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_10126)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'func'
        return stypy_return_type_10126

    # Assigning a type to the variable 'func' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'func', func)
    
    # Assigning a Name to a Attribute (line 35):
    # Getting the type of 'func_name' (line 35)
    func_name_10127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 20), 'func_name')
    # Getting the type of 'func' (line 35)
    func_10128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'func')
    # Setting the type of the member '__name__' of a type (line 35)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 4), func_10128, '__name__', func_name_10127)
    # Getting the type of 'func' (line 37)
    func_10129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 11), 'func')
    # Assigning a type to the variable 'stypy_return_type' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type', func_10129)
    
    # ################# End of 'norecursion(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'norecursion' in the type store
    # Getting the type of 'stypy_return_type' (line 12)
    stypy_return_type_10130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_10130)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'norecursion'
    return stypy_return_type_10130

# Assigning a type to the variable 'norecursion' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'norecursion', norecursion)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
