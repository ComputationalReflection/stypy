
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import math
2: 
3: 
4: class Empty:
5:     def __init__(self):
6:         pass
7: 
8: 
9: r1 = math.pow(Empty(), 3)  # Error: No __float__ method
10: 
11: 
12: class Ops:
13:     def __float__(self):
14:         return 3  # Don't return a float (even if the type promotes to float, a runtime error is reported
15: 
16: 
17: r2 = math.pow(Ops(), 3)  # Wrong __float__ (type conversion) method, not detected, runtime error
18: 
19: 
20: class WrongOps:
21:     def __float__(self):
22:         return "not a float"
23: 
24: 
25: r3 = math.pow(WrongOps(), 3)  # Runtime error, not reported
26: 
27: 
28: class EvenMoreWrongOps:
29:     def __float__(self, extra):
30:         return 3.0
31: 
32: 
33: r4 = math.pow(EvenMoreWrongOps(), 3)  # Not reported, even if the problem is parameter arity
34: 
35: 
36: class RightOps:
37:     def __float__(self):
38:         return 3.0
39: 
40: 
41: r5 = math.pow(RightOps(), 3)
42: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import math' statement (line 1)
import math

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'math', math, module_type_store)

# Declaration of the 'Empty' class

class Empty:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 5, 4, False)
        # Assigning a type to the variable 'self' (line 6)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Empty.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Empty' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'Empty', Empty)

# Assigning a Call to a Name (line 9):

# Call to pow(...): (line 9)
# Processing the call arguments (line 9)

# Call to Empty(...): (line 9)
# Processing the call keyword arguments (line 9)
kwargs_8040 = {}
# Getting the type of 'Empty' (line 9)
Empty_8039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 14), 'Empty', False)
# Calling Empty(args, kwargs) (line 9)
Empty_call_result_8041 = invoke(stypy.reporting.localization.Localization(__file__, 9, 14), Empty_8039, *[], **kwargs_8040)

int_8042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 23), 'int')
# Processing the call keyword arguments (line 9)
kwargs_8043 = {}
# Getting the type of 'math' (line 9)
math_8037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 5), 'math', False)
# Obtaining the member 'pow' of a type (line 9)
pow_8038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 5), math_8037, 'pow')
# Calling pow(args, kwargs) (line 9)
pow_call_result_8044 = invoke(stypy.reporting.localization.Localization(__file__, 9, 5), pow_8038, *[Empty_call_result_8041, int_8042], **kwargs_8043)

# Assigning a type to the variable 'r1' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'r1', pow_call_result_8044)
# Declaration of the 'Ops' class

class Ops:

    @norecursion
    def __float__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__float__'
        module_type_store = module_type_store.open_function_context('__float__', 13, 4, False)
        # Assigning a type to the variable 'self' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Ops.__float__.__dict__.__setitem__('stypy_localization', localization)
        Ops.__float__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Ops.__float__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Ops.__float__.__dict__.__setitem__('stypy_function_name', 'Ops.__float__')
        Ops.__float__.__dict__.__setitem__('stypy_param_names_list', [])
        Ops.__float__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Ops.__float__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Ops.__float__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Ops.__float__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Ops.__float__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Ops.__float__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Ops.__float__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__float__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__float__(...)' code ##################

        int_8045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 15), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'stypy_return_type', int_8045)
        
        # ################# End of '__float__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__float__' in the type store
        # Getting the type of 'stypy_return_type' (line 13)
        stypy_return_type_8046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8046)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__float__'
        return stypy_return_type_8046


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 12, 0, False)
        # Assigning a type to the variable 'self' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Ops.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Ops' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'Ops', Ops)

# Assigning a Call to a Name (line 17):

# Call to pow(...): (line 17)
# Processing the call arguments (line 17)

# Call to Ops(...): (line 17)
# Processing the call keyword arguments (line 17)
kwargs_8050 = {}
# Getting the type of 'Ops' (line 17)
Ops_8049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 14), 'Ops', False)
# Calling Ops(args, kwargs) (line 17)
Ops_call_result_8051 = invoke(stypy.reporting.localization.Localization(__file__, 17, 14), Ops_8049, *[], **kwargs_8050)

int_8052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 21), 'int')
# Processing the call keyword arguments (line 17)
kwargs_8053 = {}
# Getting the type of 'math' (line 17)
math_8047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 5), 'math', False)
# Obtaining the member 'pow' of a type (line 17)
pow_8048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 5), math_8047, 'pow')
# Calling pow(args, kwargs) (line 17)
pow_call_result_8054 = invoke(stypy.reporting.localization.Localization(__file__, 17, 5), pow_8048, *[Ops_call_result_8051, int_8052], **kwargs_8053)

# Assigning a type to the variable 'r2' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'r2', pow_call_result_8054)
# Declaration of the 'WrongOps' class

class WrongOps:

    @norecursion
    def __float__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__float__'
        module_type_store = module_type_store.open_function_context('__float__', 21, 4, False)
        # Assigning a type to the variable 'self' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        WrongOps.__float__.__dict__.__setitem__('stypy_localization', localization)
        WrongOps.__float__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        WrongOps.__float__.__dict__.__setitem__('stypy_type_store', module_type_store)
        WrongOps.__float__.__dict__.__setitem__('stypy_function_name', 'WrongOps.__float__')
        WrongOps.__float__.__dict__.__setitem__('stypy_param_names_list', [])
        WrongOps.__float__.__dict__.__setitem__('stypy_varargs_param_name', None)
        WrongOps.__float__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        WrongOps.__float__.__dict__.__setitem__('stypy_call_defaults', defaults)
        WrongOps.__float__.__dict__.__setitem__('stypy_call_varargs', varargs)
        WrongOps.__float__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        WrongOps.__float__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'WrongOps.__float__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__float__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__float__(...)' code ##################

        str_8055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 15), 'str', 'not a float')
        # Assigning a type to the variable 'stypy_return_type' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'stypy_return_type', str_8055)
        
        # ################# End of '__float__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__float__' in the type store
        # Getting the type of 'stypy_return_type' (line 21)
        stypy_return_type_8056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8056)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__float__'
        return stypy_return_type_8056


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 20, 0, False)
        # Assigning a type to the variable 'self' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'WrongOps.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'WrongOps' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'WrongOps', WrongOps)

# Assigning a Call to a Name (line 25):

# Call to pow(...): (line 25)
# Processing the call arguments (line 25)

# Call to WrongOps(...): (line 25)
# Processing the call keyword arguments (line 25)
kwargs_8060 = {}
# Getting the type of 'WrongOps' (line 25)
WrongOps_8059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 14), 'WrongOps', False)
# Calling WrongOps(args, kwargs) (line 25)
WrongOps_call_result_8061 = invoke(stypy.reporting.localization.Localization(__file__, 25, 14), WrongOps_8059, *[], **kwargs_8060)

int_8062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 26), 'int')
# Processing the call keyword arguments (line 25)
kwargs_8063 = {}
# Getting the type of 'math' (line 25)
math_8057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 5), 'math', False)
# Obtaining the member 'pow' of a type (line 25)
pow_8058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 5), math_8057, 'pow')
# Calling pow(args, kwargs) (line 25)
pow_call_result_8064 = invoke(stypy.reporting.localization.Localization(__file__, 25, 5), pow_8058, *[WrongOps_call_result_8061, int_8062], **kwargs_8063)

# Assigning a type to the variable 'r3' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'r3', pow_call_result_8064)
# Declaration of the 'EvenMoreWrongOps' class

class EvenMoreWrongOps:

    @norecursion
    def __float__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__float__'
        module_type_store = module_type_store.open_function_context('__float__', 29, 4, False)
        # Assigning a type to the variable 'self' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        EvenMoreWrongOps.__float__.__dict__.__setitem__('stypy_localization', localization)
        EvenMoreWrongOps.__float__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        EvenMoreWrongOps.__float__.__dict__.__setitem__('stypy_type_store', module_type_store)
        EvenMoreWrongOps.__float__.__dict__.__setitem__('stypy_function_name', 'EvenMoreWrongOps.__float__')
        EvenMoreWrongOps.__float__.__dict__.__setitem__('stypy_param_names_list', ['extra'])
        EvenMoreWrongOps.__float__.__dict__.__setitem__('stypy_varargs_param_name', None)
        EvenMoreWrongOps.__float__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        EvenMoreWrongOps.__float__.__dict__.__setitem__('stypy_call_defaults', defaults)
        EvenMoreWrongOps.__float__.__dict__.__setitem__('stypy_call_varargs', varargs)
        EvenMoreWrongOps.__float__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        EvenMoreWrongOps.__float__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'EvenMoreWrongOps.__float__', ['extra'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__float__', localization, ['extra'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__float__(...)' code ##################

        float_8065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 15), 'float')
        # Assigning a type to the variable 'stypy_return_type' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'stypy_return_type', float_8065)
        
        # ################# End of '__float__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__float__' in the type store
        # Getting the type of 'stypy_return_type' (line 29)
        stypy_return_type_8066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8066)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__float__'
        return stypy_return_type_8066


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 28, 0, False)
        # Assigning a type to the variable 'self' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'EvenMoreWrongOps.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'EvenMoreWrongOps' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'EvenMoreWrongOps', EvenMoreWrongOps)

# Assigning a Call to a Name (line 33):

# Call to pow(...): (line 33)
# Processing the call arguments (line 33)

# Call to EvenMoreWrongOps(...): (line 33)
# Processing the call keyword arguments (line 33)
kwargs_8070 = {}
# Getting the type of 'EvenMoreWrongOps' (line 33)
EvenMoreWrongOps_8069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 14), 'EvenMoreWrongOps', False)
# Calling EvenMoreWrongOps(args, kwargs) (line 33)
EvenMoreWrongOps_call_result_8071 = invoke(stypy.reporting.localization.Localization(__file__, 33, 14), EvenMoreWrongOps_8069, *[], **kwargs_8070)

int_8072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 34), 'int')
# Processing the call keyword arguments (line 33)
kwargs_8073 = {}
# Getting the type of 'math' (line 33)
math_8067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 5), 'math', False)
# Obtaining the member 'pow' of a type (line 33)
pow_8068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 5), math_8067, 'pow')
# Calling pow(args, kwargs) (line 33)
pow_call_result_8074 = invoke(stypy.reporting.localization.Localization(__file__, 33, 5), pow_8068, *[EvenMoreWrongOps_call_result_8071, int_8072], **kwargs_8073)

# Assigning a type to the variable 'r4' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'r4', pow_call_result_8074)
# Declaration of the 'RightOps' class

class RightOps:

    @norecursion
    def __float__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__float__'
        module_type_store = module_type_store.open_function_context('__float__', 37, 4, False)
        # Assigning a type to the variable 'self' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RightOps.__float__.__dict__.__setitem__('stypy_localization', localization)
        RightOps.__float__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RightOps.__float__.__dict__.__setitem__('stypy_type_store', module_type_store)
        RightOps.__float__.__dict__.__setitem__('stypy_function_name', 'RightOps.__float__')
        RightOps.__float__.__dict__.__setitem__('stypy_param_names_list', [])
        RightOps.__float__.__dict__.__setitem__('stypy_varargs_param_name', None)
        RightOps.__float__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RightOps.__float__.__dict__.__setitem__('stypy_call_defaults', defaults)
        RightOps.__float__.__dict__.__setitem__('stypy_call_varargs', varargs)
        RightOps.__float__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RightOps.__float__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RightOps.__float__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__float__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__float__(...)' code ##################

        float_8075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 15), 'float')
        # Assigning a type to the variable 'stypy_return_type' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'stypy_return_type', float_8075)
        
        # ################# End of '__float__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__float__' in the type store
        # Getting the type of 'stypy_return_type' (line 37)
        stypy_return_type_8076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8076)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__float__'
        return stypy_return_type_8076


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 36, 0, False)
        # Assigning a type to the variable 'self' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RightOps.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'RightOps' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'RightOps', RightOps)

# Assigning a Call to a Name (line 41):

# Call to pow(...): (line 41)
# Processing the call arguments (line 41)

# Call to RightOps(...): (line 41)
# Processing the call keyword arguments (line 41)
kwargs_8080 = {}
# Getting the type of 'RightOps' (line 41)
RightOps_8079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 14), 'RightOps', False)
# Calling RightOps(args, kwargs) (line 41)
RightOps_call_result_8081 = invoke(stypy.reporting.localization.Localization(__file__, 41, 14), RightOps_8079, *[], **kwargs_8080)

int_8082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 26), 'int')
# Processing the call keyword arguments (line 41)
kwargs_8083 = {}
# Getting the type of 'math' (line 41)
math_8077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 5), 'math', False)
# Obtaining the member 'pow' of a type (line 41)
pow_8078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 5), math_8077, 'pow')
# Calling pow(args, kwargs) (line 41)
pow_call_result_8084 = invoke(stypy.reporting.localization.Localization(__file__, 41, 5), pow_8078, *[RightOps_call_result_8081, int_8082], **kwargs_8083)

# Assigning a type to the variable 'r5' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), 'r5', pow_call_result_8084)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
