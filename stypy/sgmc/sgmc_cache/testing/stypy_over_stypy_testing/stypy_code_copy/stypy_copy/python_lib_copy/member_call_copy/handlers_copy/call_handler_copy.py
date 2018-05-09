
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import abc
2: 
3: 
4: class CallHandler:
5:     '''
6:     Base abstract class for all the call handlers
7:     '''
8:     __metaclass__ = abc.ABCMeta
9: 
10:     def __init__(self):
11:         pass
12: 
13:     @abc.abstractmethod
14:     def applies_to(self, proxy_obj, callable_entity):
15:         '''
16:         This method determines if this call handler can respond to a call to this entity.
17:         '''
18:         return False
19: 
20:     @abc.abstractmethod
21:     def __call__(self, proxy_obj, localization, callable_entity, *arg_types, **kwargs_types):
22:         '''
23:         This method calls callable_entity(localization, *arg_types, **kwargs_types) with the call handler strategy
24:         modeled by its subclasses.
25:         :param proxy_obj:
26:         :param localization:
27:         :param callable_entity:
28:         :param arg_types:
29:         :param kwargs_types:
30:         :return:
31:         '''
32:         pass
33: 
34:     @staticmethod
35:     def compose_type_modifier_member_name(name):
36:         return "type_modifier_" + name
37: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import abc' statement (line 1)
import abc

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'abc', abc, module_type_store)

# Declaration of the 'CallHandler' class

class CallHandler:
    str_5669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, (-1)), 'str', '\n    Base abstract class for all the call handlers\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 10, 4, False)
        # Assigning a type to the variable 'self' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CallHandler.__init__', [], None, None, defaults, varargs, kwargs)

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


    @norecursion
    def applies_to(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'applies_to'
        module_type_store = module_type_store.open_function_context('applies_to', 13, 4, False)
        # Assigning a type to the variable 'self' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CallHandler.applies_to.__dict__.__setitem__('stypy_localization', localization)
        CallHandler.applies_to.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CallHandler.applies_to.__dict__.__setitem__('stypy_type_store', module_type_store)
        CallHandler.applies_to.__dict__.__setitem__('stypy_function_name', 'CallHandler.applies_to')
        CallHandler.applies_to.__dict__.__setitem__('stypy_param_names_list', ['proxy_obj', 'callable_entity'])
        CallHandler.applies_to.__dict__.__setitem__('stypy_varargs_param_name', None)
        CallHandler.applies_to.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CallHandler.applies_to.__dict__.__setitem__('stypy_call_defaults', defaults)
        CallHandler.applies_to.__dict__.__setitem__('stypy_call_varargs', varargs)
        CallHandler.applies_to.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CallHandler.applies_to.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CallHandler.applies_to', ['proxy_obj', 'callable_entity'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'applies_to', localization, ['proxy_obj', 'callable_entity'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'applies_to(...)' code ##################

        str_5670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, (-1)), 'str', '\n        This method determines if this call handler can respond to a call to this entity.\n        ')
        # Getting the type of 'False' (line 18)
        False_5671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'stypy_return_type', False_5671)
        
        # ################# End of 'applies_to(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'applies_to' in the type store
        # Getting the type of 'stypy_return_type' (line 13)
        stypy_return_type_5672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5672)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'applies_to'
        return stypy_return_type_5672


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 20, 4, False)
        # Assigning a type to the variable 'self' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        CallHandler.__call__.__dict__.__setitem__('stypy_localization', localization)
        CallHandler.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        CallHandler.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        CallHandler.__call__.__dict__.__setitem__('stypy_function_name', 'CallHandler.__call__')
        CallHandler.__call__.__dict__.__setitem__('stypy_param_names_list', ['proxy_obj', 'localization', 'callable_entity'])
        CallHandler.__call__.__dict__.__setitem__('stypy_varargs_param_name', 'arg_types')
        CallHandler.__call__.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs_types')
        CallHandler.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        CallHandler.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        CallHandler.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CallHandler.__call__.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CallHandler.__call__', ['proxy_obj', 'localization', 'callable_entity'], 'arg_types', 'kwargs_types', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['proxy_obj', 'localization', 'callable_entity'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        str_5673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, (-1)), 'str', '\n        This method calls callable_entity(localization, *arg_types, **kwargs_types) with the call handler strategy\n        modeled by its subclasses.\n        :param proxy_obj:\n        :param localization:\n        :param callable_entity:\n        :param arg_types:\n        :param kwargs_types:\n        :return:\n        ')
        pass
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 20)
        stypy_return_type_5674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5674)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_5674


    @staticmethod
    @norecursion
    def compose_type_modifier_member_name(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'compose_type_modifier_member_name'
        module_type_store = module_type_store.open_function_context('compose_type_modifier_member_name', 34, 4, False)
        
        # Passed parameters checking function
        CallHandler.compose_type_modifier_member_name.__dict__.__setitem__('stypy_localization', localization)
        CallHandler.compose_type_modifier_member_name.__dict__.__setitem__('stypy_type_of_self', None)
        CallHandler.compose_type_modifier_member_name.__dict__.__setitem__('stypy_type_store', module_type_store)
        CallHandler.compose_type_modifier_member_name.__dict__.__setitem__('stypy_function_name', 'compose_type_modifier_member_name')
        CallHandler.compose_type_modifier_member_name.__dict__.__setitem__('stypy_param_names_list', ['name'])
        CallHandler.compose_type_modifier_member_name.__dict__.__setitem__('stypy_varargs_param_name', None)
        CallHandler.compose_type_modifier_member_name.__dict__.__setitem__('stypy_kwargs_param_name', None)
        CallHandler.compose_type_modifier_member_name.__dict__.__setitem__('stypy_call_defaults', defaults)
        CallHandler.compose_type_modifier_member_name.__dict__.__setitem__('stypy_call_varargs', varargs)
        CallHandler.compose_type_modifier_member_name.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        CallHandler.compose_type_modifier_member_name.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, None, module_type_store, 'compose_type_modifier_member_name', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'compose_type_modifier_member_name', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'compose_type_modifier_member_name(...)' code ##################

        str_5675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 15), 'str', 'type_modifier_')
        # Getting the type of 'name' (line 36)
        name_5676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 34), 'name')
        # Applying the binary operator '+' (line 36)
        result_add_5677 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 15), '+', str_5675, name_5676)
        
        # Assigning a type to the variable 'stypy_return_type' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'stypy_return_type', result_add_5677)
        
        # ################# End of 'compose_type_modifier_member_name(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'compose_type_modifier_member_name' in the type store
        # Getting the type of 'stypy_return_type' (line 34)
        stypy_return_type_5678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5678)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'compose_type_modifier_member_name'
        return stypy_return_type_5678


# Assigning a type to the variable 'CallHandler' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'CallHandler', CallHandler)

# Assigning a Attribute to a Name (line 8):
# Getting the type of 'abc' (line 8)
abc_5679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 20), 'abc')
# Obtaining the member 'ABCMeta' of a type (line 8)
ABCMeta_5680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 20), abc_5679, 'ABCMeta')
# Getting the type of 'CallHandler'
CallHandler_5681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'CallHandler')
# Setting the type of the member '__metaclass__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), CallHandler_5681, '__metaclass__', ABCMeta_5680)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
