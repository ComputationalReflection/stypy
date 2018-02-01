
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import code
2: 
3: ''' copyright Sean McCarthy, license GPL v2 or later '''
4: 
5: class Row:
6:     '''Class containing a guess code and answer code'''
7: 
8:     def __init__(self,guess,result):
9:         self.__guess = guess
10:         self.__result = result
11: 
12:     def setGuess(self,guess):
13:         self.__guess = guess
14: 
15:     def setResult(self,result):
16:         self.__result = result
17: 
18:     def getGuess(self):
19:         return self.__guess
20: 
21:     def getResult(self):
22:         return self.__result
23: 
24: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import code' statement (line 1)
import code

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'code', code, module_type_store)

str_737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 0), 'str', ' copyright Sean McCarthy, license GPL v2 or later ')
# Declaration of the 'Row' class

class Row:
    str_738 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 4), 'str', 'Class containing a guess code and answer code')

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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Row.__init__', ['guess', 'result'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['guess', 'result'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 9):
        # Getting the type of 'guess' (line 9)
        guess_739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 23), 'guess')
        # Getting the type of 'self' (line 9)
        self_740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'self')
        # Setting the type of the member '__guess' of a type (line 9)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 8), self_740, '__guess', guess_739)
        
        # Assigning a Name to a Attribute (line 10):
        # Getting the type of 'result' (line 10)
        result_741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 24), 'result')
        # Getting the type of 'self' (line 10)
        self_742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'self')
        # Setting the type of the member '__result' of a type (line 10)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 8), self_742, '__result', result_741)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def setGuess(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setGuess'
        module_type_store = module_type_store.open_function_context('setGuess', 12, 4, False)
        # Assigning a type to the variable 'self' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Row.setGuess.__dict__.__setitem__('stypy_localization', localization)
        Row.setGuess.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Row.setGuess.__dict__.__setitem__('stypy_type_store', module_type_store)
        Row.setGuess.__dict__.__setitem__('stypy_function_name', 'Row.setGuess')
        Row.setGuess.__dict__.__setitem__('stypy_param_names_list', ['guess'])
        Row.setGuess.__dict__.__setitem__('stypy_varargs_param_name', None)
        Row.setGuess.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Row.setGuess.__dict__.__setitem__('stypy_call_defaults', defaults)
        Row.setGuess.__dict__.__setitem__('stypy_call_varargs', varargs)
        Row.setGuess.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Row.setGuess.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Row.setGuess', ['guess'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setGuess', localization, ['guess'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setGuess(...)' code ##################

        
        # Assigning a Name to a Attribute (line 13):
        # Getting the type of 'guess' (line 13)
        guess_743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 23), 'guess')
        # Getting the type of 'self' (line 13)
        self_744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'self')
        # Setting the type of the member '__guess' of a type (line 13)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 8), self_744, '__guess', guess_743)
        
        # ################# End of 'setGuess(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setGuess' in the type store
        # Getting the type of 'stypy_return_type' (line 12)
        stypy_return_type_745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_745)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setGuess'
        return stypy_return_type_745


    @norecursion
    def setResult(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setResult'
        module_type_store = module_type_store.open_function_context('setResult', 15, 4, False)
        # Assigning a type to the variable 'self' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Row.setResult.__dict__.__setitem__('stypy_localization', localization)
        Row.setResult.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Row.setResult.__dict__.__setitem__('stypy_type_store', module_type_store)
        Row.setResult.__dict__.__setitem__('stypy_function_name', 'Row.setResult')
        Row.setResult.__dict__.__setitem__('stypy_param_names_list', ['result'])
        Row.setResult.__dict__.__setitem__('stypy_varargs_param_name', None)
        Row.setResult.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Row.setResult.__dict__.__setitem__('stypy_call_defaults', defaults)
        Row.setResult.__dict__.__setitem__('stypy_call_varargs', varargs)
        Row.setResult.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Row.setResult.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Row.setResult', ['result'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setResult', localization, ['result'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setResult(...)' code ##################

        
        # Assigning a Name to a Attribute (line 16):
        # Getting the type of 'result' (line 16)
        result_746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 24), 'result')
        # Getting the type of 'self' (line 16)
        self_747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'self')
        # Setting the type of the member '__result' of a type (line 16)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), self_747, '__result', result_746)
        
        # ################# End of 'setResult(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setResult' in the type store
        # Getting the type of 'stypy_return_type' (line 15)
        stypy_return_type_748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_748)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setResult'
        return stypy_return_type_748


    @norecursion
    def getGuess(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getGuess'
        module_type_store = module_type_store.open_function_context('getGuess', 18, 4, False)
        # Assigning a type to the variable 'self' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Row.getGuess.__dict__.__setitem__('stypy_localization', localization)
        Row.getGuess.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Row.getGuess.__dict__.__setitem__('stypy_type_store', module_type_store)
        Row.getGuess.__dict__.__setitem__('stypy_function_name', 'Row.getGuess')
        Row.getGuess.__dict__.__setitem__('stypy_param_names_list', [])
        Row.getGuess.__dict__.__setitem__('stypy_varargs_param_name', None)
        Row.getGuess.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Row.getGuess.__dict__.__setitem__('stypy_call_defaults', defaults)
        Row.getGuess.__dict__.__setitem__('stypy_call_varargs', varargs)
        Row.getGuess.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Row.getGuess.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Row.getGuess', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getGuess', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getGuess(...)' code ##################

        # Getting the type of 'self' (line 19)
        self_749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 15), 'self')
        # Obtaining the member '__guess' of a type (line 19)
        guess_750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 15), self_749, '__guess')
        # Assigning a type to the variable 'stypy_return_type' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'stypy_return_type', guess_750)
        
        # ################# End of 'getGuess(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getGuess' in the type store
        # Getting the type of 'stypy_return_type' (line 18)
        stypy_return_type_751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_751)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getGuess'
        return stypy_return_type_751


    @norecursion
    def getResult(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getResult'
        module_type_store = module_type_store.open_function_context('getResult', 21, 4, False)
        # Assigning a type to the variable 'self' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Row.getResult.__dict__.__setitem__('stypy_localization', localization)
        Row.getResult.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Row.getResult.__dict__.__setitem__('stypy_type_store', module_type_store)
        Row.getResult.__dict__.__setitem__('stypy_function_name', 'Row.getResult')
        Row.getResult.__dict__.__setitem__('stypy_param_names_list', [])
        Row.getResult.__dict__.__setitem__('stypy_varargs_param_name', None)
        Row.getResult.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Row.getResult.__dict__.__setitem__('stypy_call_defaults', defaults)
        Row.getResult.__dict__.__setitem__('stypy_call_varargs', varargs)
        Row.getResult.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Row.getResult.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Row.getResult', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getResult', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getResult(...)' code ##################

        # Getting the type of 'self' (line 22)
        self_752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 15), 'self')
        # Obtaining the member '__result' of a type (line 22)
        result_753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 15), self_752, '__result')
        # Assigning a type to the variable 'stypy_return_type' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'stypy_return_type', result_753)
        
        # ################# End of 'getResult(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getResult' in the type store
        # Getting the type of 'stypy_return_type' (line 21)
        stypy_return_type_754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_754)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getResult'
        return stypy_return_type_754


# Assigning a type to the variable 'Row' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'Row', Row)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
