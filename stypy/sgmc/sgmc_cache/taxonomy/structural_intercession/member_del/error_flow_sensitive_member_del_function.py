
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: __doc__ = "Flow-sensitive member deletion inside a function"
3: 
4: if __name__ == '__main__':
5: 
6:     class Dummy:
7:         class_attribute = 0
8: 
9:         def __init__(self):
10:             self.instance_attribute = "str"
11: 
12:         def method(self):
13:             return self.instance_attribute
14: 
15: 
16:     def func():
17:         if True:
18:             delattr(Dummy, 'class_attribute')
19: 
20: 
21:     func()
22: 
23:     # Type warning
24:     print Dummy.class_attribute / 2
25: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 2):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, 10), 'str', 'Flow-sensitive member deletion inside a function')
# Assigning a type to the variable '__doc__' (line 2)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), '__doc__', str_1)

if (__name__ == '__main__'):
    # Declaration of the 'Dummy' class

    class Dummy:

        @norecursion
        def __init__(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '__init__'
            module_type_store = module_type_store.open_function_context('__init__', 9, 8, False)
            # Assigning a type to the variable 'self' (line 10)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dummy.__init__', [], None, None, defaults, varargs, kwargs)

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

            
            # Assigning a Str to a Attribute (line 10):
            str_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 38), 'str', 'str')
            # Getting the type of 'self' (line 10)
            self_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 12), 'self')
            # Setting the type of the member 'instance_attribute' of a type (line 10)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 12), self_3, 'instance_attribute', str_2)
            
            # ################# End of '__init__(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()


        @norecursion
        def method(type_of_self, localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'method'
            module_type_store = module_type_store.open_function_context('method', 12, 8, False)
            # Assigning a type to the variable 'self' (line 13)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'self', type_of_self)
            
            # Passed parameters checking function
            Dummy.method.__dict__.__setitem__('stypy_localization', localization)
            Dummy.method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
            Dummy.method.__dict__.__setitem__('stypy_type_store', module_type_store)
            Dummy.method.__dict__.__setitem__('stypy_function_name', 'Dummy.method')
            Dummy.method.__dict__.__setitem__('stypy_param_names_list', [])
            Dummy.method.__dict__.__setitem__('stypy_varargs_param_name', None)
            Dummy.method.__dict__.__setitem__('stypy_kwargs_param_name', None)
            Dummy.method.__dict__.__setitem__('stypy_call_defaults', defaults)
            Dummy.method.__dict__.__setitem__('stypy_call_varargs', varargs)
            Dummy.method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
            Dummy.method.__dict__.__setitem__('stypy_declared_arg_number', 1)
            arguments = process_argument_values(localization, type_of_self, module_type_store, 'Dummy.method', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'method', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'method(...)' code ##################

            # Getting the type of 'self' (line 13)
            self_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 19), 'self')
            # Obtaining the member 'instance_attribute' of a type (line 13)
            instance_attribute_5 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 19), self_4, 'instance_attribute')
            # Assigning a type to the variable 'stypy_return_type' (line 13)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 12), 'stypy_return_type', instance_attribute_5)
            
            # ################# End of 'method(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'method' in the type store
            # Getting the type of 'stypy_return_type' (line 12)
            stypy_return_type_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_6)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'method'
            return stypy_return_type_6

    
    # Assigning a type to the variable 'Dummy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'Dummy', Dummy)
    
    # Assigning a Num to a Name (line 7):
    int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 26), 'int')
    # Getting the type of 'Dummy'
    Dummy_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Dummy')
    # Setting the type of the member 'class_attribute' of a type
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Dummy_8, 'class_attribute', int_7)

    @norecursion
    def func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'func'
        module_type_store = module_type_store.open_function_context('func', 16, 4, False)
        
        # Passed parameters checking function
        func.stypy_localization = localization
        func.stypy_type_of_self = None
        func.stypy_type_store = module_type_store
        func.stypy_function_name = 'func'
        func.stypy_param_names_list = []
        func.stypy_varargs_param_name = None
        func.stypy_kwargs_param_name = None
        func.stypy_call_defaults = defaults
        func.stypy_call_varargs = varargs
        func.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'func', [], None, None, defaults, varargs, kwargs)

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

        
        # Getting the type of 'True' (line 17)
        True_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 11), 'True')
        # Testing the type of an if condition (line 17)
        if_condition_10 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 17, 8), True_9)
        # Assigning a type to the variable 'if_condition_10' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'if_condition_10', if_condition_10)
        # SSA begins for if statement (line 17)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to delattr(...): (line 18)
        # Processing the call arguments (line 18)
        # Getting the type of 'Dummy' (line 18)
        Dummy_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 20), 'Dummy', False)
        str_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 27), 'str', 'class_attribute')
        # Processing the call keyword arguments (line 18)
        kwargs_14 = {}
        # Getting the type of 'delattr' (line 18)
        delattr_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 12), 'delattr', False)
        # Calling delattr(args, kwargs) (line 18)
        delattr_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 18, 12), delattr_11, *[Dummy_12, str_13], **kwargs_14)
        
        # SSA join for if statement (line 17)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'func' in the type store
        # Getting the type of 'stypy_return_type' (line 16)
        stypy_return_type_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'func'
        return stypy_return_type_16

    # Assigning a type to the variable 'func' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'func', func)
    
    # Call to func(...): (line 21)
    # Processing the call keyword arguments (line 21)
    kwargs_18 = {}
    # Getting the type of 'func' (line 21)
    func_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'func', False)
    # Calling func(args, kwargs) (line 21)
    func_call_result_19 = invoke(stypy.reporting.localization.Localization(__file__, 21, 4), func_17, *[], **kwargs_18)
    
    # Getting the type of 'Dummy' (line 24)
    Dummy_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 10), 'Dummy')
    # Obtaining the member 'class_attribute' of a type (line 24)
    class_attribute_21 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 10), Dummy_20, 'class_attribute')
    int_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 34), 'int')
    # Applying the binary operator 'div' (line 24)
    result_div_23 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 10), 'div', class_attribute_21, int_22)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
