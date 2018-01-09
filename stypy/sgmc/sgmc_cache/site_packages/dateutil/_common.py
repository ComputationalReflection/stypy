
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Common code used in multiple modules.
3: '''
4: 
5: 
6: class weekday(object):
7:     __slots__ = ["weekday", "n"]
8: 
9:     def __init__(self, weekday, n=None):
10:         self.weekday = weekday
11:         self.n = n
12: 
13:     def __call__(self, n):
14:         if n == self.n:
15:             return self
16:         else:
17:             return self.__class__(self.weekday, n)
18: 
19:     def __eq__(self, other):
20:         try:
21:             if self.weekday != other.weekday or self.n != other.n:
22:                 return False
23:         except AttributeError:
24:             return False
25:         return True
26: 
27:     __hash__ = None
28: 
29:     def __repr__(self):
30:         s = ("MO", "TU", "WE", "TH", "FR", "SA", "SU")[self.weekday]
31:         if not self.n:
32:             return s
33:         else:
34:             return "%s(%+d)" % (s, self.n)
35: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_320110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', '\nCommon code used in multiple modules.\n')
# Declaration of the 'weekday' class

class weekday(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 9)
        None_320111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 34), 'None')
        defaults = [None_320111]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 9, 4, False)
        # Assigning a type to the variable 'self' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'weekday.__init__', ['weekday', 'n'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['weekday', 'n'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 10):
        # Getting the type of 'weekday' (line 10)
        weekday_320112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 23), 'weekday')
        # Getting the type of 'self' (line 10)
        self_320113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'self')
        # Setting the type of the member 'weekday' of a type (line 10)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 8), self_320113, 'weekday', weekday_320112)
        
        # Assigning a Name to a Attribute (line 11):
        # Getting the type of 'n' (line 11)
        n_320114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 17), 'n')
        # Getting the type of 'self' (line 11)
        self_320115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'self')
        # Setting the type of the member 'n' of a type (line 11)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 8), self_320115, 'n', n_320114)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 13, 4, False)
        # Assigning a type to the variable 'self' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        weekday.__call__.__dict__.__setitem__('stypy_localization', localization)
        weekday.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        weekday.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        weekday.__call__.__dict__.__setitem__('stypy_function_name', 'weekday.__call__')
        weekday.__call__.__dict__.__setitem__('stypy_param_names_list', ['n'])
        weekday.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        weekday.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        weekday.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        weekday.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        weekday.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        weekday.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'weekday.__call__', ['n'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['n'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        
        # Getting the type of 'n' (line 14)
        n_320116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 11), 'n')
        # Getting the type of 'self' (line 14)
        self_320117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 16), 'self')
        # Obtaining the member 'n' of a type (line 14)
        n_320118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 16), self_320117, 'n')
        # Applying the binary operator '==' (line 14)
        result_eq_320119 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 11), '==', n_320116, n_320118)
        
        # Testing the type of an if condition (line 14)
        if_condition_320120 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 8), result_eq_320119)
        # Assigning a type to the variable 'if_condition_320120' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'if_condition_320120', if_condition_320120)
        # SSA begins for if statement (line 14)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'self' (line 15)
        self_320121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 19), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), 'stypy_return_type', self_320121)
        # SSA branch for the else part of an if statement (line 14)
        module_type_store.open_ssa_branch('else')
        
        # Call to __class__(...): (line 17)
        # Processing the call arguments (line 17)
        # Getting the type of 'self' (line 17)
        self_320124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 34), 'self', False)
        # Obtaining the member 'weekday' of a type (line 17)
        weekday_320125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 34), self_320124, 'weekday')
        # Getting the type of 'n' (line 17)
        n_320126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 48), 'n', False)
        # Processing the call keyword arguments (line 17)
        kwargs_320127 = {}
        # Getting the type of 'self' (line 17)
        self_320122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 19), 'self', False)
        # Obtaining the member '__class__' of a type (line 17)
        class___320123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 19), self_320122, '__class__')
        # Calling __class__(args, kwargs) (line 17)
        class___call_result_320128 = invoke(stypy.reporting.localization.Localization(__file__, 17, 19), class___320123, *[weekday_320125, n_320126], **kwargs_320127)
        
        # Assigning a type to the variable 'stypy_return_type' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'stypy_return_type', class___call_result_320128)
        # SSA join for if statement (line 14)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 13)
        stypy_return_type_320129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_320129)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_320129


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 19, 4, False)
        # Assigning a type to the variable 'self' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        weekday.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        weekday.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        weekday.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        weekday.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'weekday.stypy__eq__')
        weekday.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        weekday.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        weekday.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        weekday.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        weekday.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        weekday.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        weekday.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'weekday.stypy__eq__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__eq__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__eq__(...)' code ##################

        
        
        # SSA begins for try-except statement (line 20)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 21)
        self_320130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 15), 'self')
        # Obtaining the member 'weekday' of a type (line 21)
        weekday_320131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 15), self_320130, 'weekday')
        # Getting the type of 'other' (line 21)
        other_320132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 31), 'other')
        # Obtaining the member 'weekday' of a type (line 21)
        weekday_320133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 31), other_320132, 'weekday')
        # Applying the binary operator '!=' (line 21)
        result_ne_320134 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 15), '!=', weekday_320131, weekday_320133)
        
        
        # Getting the type of 'self' (line 21)
        self_320135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 48), 'self')
        # Obtaining the member 'n' of a type (line 21)
        n_320136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 48), self_320135, 'n')
        # Getting the type of 'other' (line 21)
        other_320137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 58), 'other')
        # Obtaining the member 'n' of a type (line 21)
        n_320138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 58), other_320137, 'n')
        # Applying the binary operator '!=' (line 21)
        result_ne_320139 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 48), '!=', n_320136, n_320138)
        
        # Applying the binary operator 'or' (line 21)
        result_or_keyword_320140 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 15), 'or', result_ne_320134, result_ne_320139)
        
        # Testing the type of an if condition (line 21)
        if_condition_320141 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 21, 12), result_or_keyword_320140)
        # Assigning a type to the variable 'if_condition_320141' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'if_condition_320141', if_condition_320141)
        # SSA begins for if statement (line 21)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 22)
        False_320142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 23), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 16), 'stypy_return_type', False_320142)
        # SSA join for if statement (line 21)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the except part of a try statement (line 20)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 20)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'False' (line 24)
        False_320143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 19), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'stypy_return_type', False_320143)
        # SSA join for try-except statement (line 20)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'True' (line 25)
        True_320144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'stypy_return_type', True_320144)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 19)
        stypy_return_type_320145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_320145)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_320145


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 29, 4, False)
        # Assigning a type to the variable 'self' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        weekday.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        weekday.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        weekday.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        weekday.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'weekday.stypy__repr__')
        weekday.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        weekday.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        weekday.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        weekday.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        weekday.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        weekday.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        weekday.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'weekday.stypy__repr__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__repr__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__repr__(...)' code ##################

        
        # Assigning a Subscript to a Name (line 30):
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 30)
        self_320146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 55), 'self')
        # Obtaining the member 'weekday' of a type (line 30)
        weekday_320147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 55), self_320146, 'weekday')
        
        # Obtaining an instance of the builtin type 'tuple' (line 30)
        tuple_320148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 30)
        # Adding element type (line 30)
        str_320149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 13), 'str', 'MO')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 13), tuple_320148, str_320149)
        # Adding element type (line 30)
        str_320150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 19), 'str', 'TU')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 13), tuple_320148, str_320150)
        # Adding element type (line 30)
        str_320151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 25), 'str', 'WE')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 13), tuple_320148, str_320151)
        # Adding element type (line 30)
        str_320152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 31), 'str', 'TH')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 13), tuple_320148, str_320152)
        # Adding element type (line 30)
        str_320153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 37), 'str', 'FR')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 13), tuple_320148, str_320153)
        # Adding element type (line 30)
        str_320154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 43), 'str', 'SA')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 13), tuple_320148, str_320154)
        # Adding element type (line 30)
        str_320155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 49), 'str', 'SU')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 13), tuple_320148, str_320155)
        
        # Obtaining the member '__getitem__' of a type (line 30)
        getitem___320156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 13), tuple_320148, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 30)
        subscript_call_result_320157 = invoke(stypy.reporting.localization.Localization(__file__, 30, 13), getitem___320156, weekday_320147)
        
        # Assigning a type to the variable 's' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 's', subscript_call_result_320157)
        
        
        # Getting the type of 'self' (line 31)
        self_320158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 15), 'self')
        # Obtaining the member 'n' of a type (line 31)
        n_320159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 15), self_320158, 'n')
        # Applying the 'not' unary operator (line 31)
        result_not__320160 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 11), 'not', n_320159)
        
        # Testing the type of an if condition (line 31)
        if_condition_320161 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 31, 8), result_not__320160)
        # Assigning a type to the variable 'if_condition_320161' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'if_condition_320161', if_condition_320161)
        # SSA begins for if statement (line 31)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 's' (line 32)
        s_320162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 19), 's')
        # Assigning a type to the variable 'stypy_return_type' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'stypy_return_type', s_320162)
        # SSA branch for the else part of an if statement (line 31)
        module_type_store.open_ssa_branch('else')
        str_320163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 19), 'str', '%s(%+d)')
        
        # Obtaining an instance of the builtin type 'tuple' (line 34)
        tuple_320164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 32), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 34)
        # Adding element type (line 34)
        # Getting the type of 's' (line 34)
        s_320165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 32), 's')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 32), tuple_320164, s_320165)
        # Adding element type (line 34)
        # Getting the type of 'self' (line 34)
        self_320166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 35), 'self')
        # Obtaining the member 'n' of a type (line 34)
        n_320167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 35), self_320166, 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 32), tuple_320164, n_320167)
        
        # Applying the binary operator '%' (line 34)
        result_mod_320168 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 19), '%', str_320163, tuple_320164)
        
        # Assigning a type to the variable 'stypy_return_type' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'stypy_return_type', result_mod_320168)
        # SSA join for if statement (line 31)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 29)
        stypy_return_type_320169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_320169)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_320169


# Assigning a type to the variable 'weekday' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'weekday', weekday)

# Assigning a List to a Name (line 7):

# Obtaining an instance of the builtin type 'list' (line 7)
list_320170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_320171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 17), 'str', 'weekday')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 16), list_320170, str_320171)
# Adding element type (line 7)
str_320172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 28), 'str', 'n')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 16), list_320170, str_320172)

# Getting the type of 'weekday'
weekday_320173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'weekday')
# Setting the type of the member '__slots__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), weekday_320173, '__slots__', list_320170)

# Assigning a Name to a Name (line 27):
# Getting the type of 'None' (line 27)
None_320174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 15), 'None')
# Getting the type of 'weekday'
weekday_320175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'weekday')
# Setting the type of the member '__hash__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), weekday_320175, '__hash__', None_320174)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
