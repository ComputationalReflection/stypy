
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: 
2: 
3: import random
4: 
5: class Counter:
6:     count = 0
7:     def inc(self, value):
8:         self.count += value
9:         return self.count
10: 
11: 
12: def bitwise_or(counter, n):
13:     x = counter.count
14:     return counter.count | n
15: 
16: def flow_sensitive(obj, condition):
17:     if condition:
18:         obj.inc(1)
19:     else:
20:         obj.inc(0.5)
21:     return bitwise_or(obj, 3)
22: 
23: obj = Counter()
24: flow_sensitive(obj, random.randint(0, 1) == 0)
25: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import random' statement (line 3)
import random

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'random', random, module_type_store)

# Declaration of the 'Counter' class

class Counter:

    @norecursion
    def inc(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'inc'
        module_type_store = module_type_store.open_function_context('inc', 7, 4, False)
        # Assigning a type to the variable 'self' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Counter.inc.__dict__.__setitem__('stypy_localization', localization)
        Counter.inc.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Counter.inc.__dict__.__setitem__('stypy_type_store', module_type_store)
        Counter.inc.__dict__.__setitem__('stypy_function_name', 'Counter.inc')
        Counter.inc.__dict__.__setitem__('stypy_param_names_list', ['value'])
        Counter.inc.__dict__.__setitem__('stypy_varargs_param_name', None)
        Counter.inc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Counter.inc.__dict__.__setitem__('stypy_call_defaults', defaults)
        Counter.inc.__dict__.__setitem__('stypy_call_varargs', varargs)
        Counter.inc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Counter.inc.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Counter.inc', ['value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'inc', localization, ['value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'inc(...)' code ##################

        
        # Getting the type of 'self' (line 8)
        self_2229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'self')
        # Obtaining the member 'count' of a type (line 8)
        count_2230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 8), self_2229, 'count')
        # Getting the type of 'value' (line 8)
        value_2231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 22), 'value')
        # Applying the binary operator '+=' (line 8)
        result_iadd_2232 = python_operator(stypy.reporting.localization.Localization(__file__, 8, 8), '+=', count_2230, value_2231)
        # Getting the type of 'self' (line 8)
        self_2233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'self')
        # Setting the type of the member 'count' of a type (line 8)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 8), self_2233, 'count', result_iadd_2232)
        
        # Getting the type of 'self' (line 9)
        self_2234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 15), 'self')
        # Obtaining the member 'count' of a type (line 9)
        count_2235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 15), self_2234, 'count')
        # Assigning a type to the variable 'stypy_return_type' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'stypy_return_type', count_2235)
        
        # ################# End of 'inc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'inc' in the type store
        # Getting the type of 'stypy_return_type' (line 7)
        stypy_return_type_2236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2236)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'inc'
        return stypy_return_type_2236


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 5, 0, False)
        # Assigning a type to the variable 'self' (line 6)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Counter.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Counter' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'Counter', Counter)

# Assigning a Num to a Name (line 6):
int_2237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 12), 'int')
# Getting the type of 'Counter'
Counter_2238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Counter')
# Setting the type of the member 'count' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Counter_2238, 'count', int_2237)

@norecursion
def bitwise_or(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'bitwise_or'
    module_type_store = module_type_store.open_function_context('bitwise_or', 12, 0, False)
    
    # Passed parameters checking function
    bitwise_or.stypy_localization = localization
    bitwise_or.stypy_type_of_self = None
    bitwise_or.stypy_type_store = module_type_store
    bitwise_or.stypy_function_name = 'bitwise_or'
    bitwise_or.stypy_param_names_list = ['counter', 'n']
    bitwise_or.stypy_varargs_param_name = None
    bitwise_or.stypy_kwargs_param_name = None
    bitwise_or.stypy_call_defaults = defaults
    bitwise_or.stypy_call_varargs = varargs
    bitwise_or.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'bitwise_or', ['counter', 'n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'bitwise_or', localization, ['counter', 'n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'bitwise_or(...)' code ##################

    
    # Assigning a Attribute to a Name (line 13):
    # Getting the type of 'counter' (line 13)
    counter_2239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'counter')
    # Obtaining the member 'count' of a type (line 13)
    count_2240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 8), counter_2239, 'count')
    # Assigning a type to the variable 'x' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'x', count_2240)
    # Getting the type of 'counter' (line 14)
    counter_2241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 11), 'counter')
    # Obtaining the member 'count' of a type (line 14)
    count_2242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 11), counter_2241, 'count')
    # Getting the type of 'n' (line 14)
    n_2243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 27), 'n')
    # Applying the binary operator '|' (line 14)
    result_or__2244 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 11), '|', count_2242, n_2243)
    
    # Assigning a type to the variable 'stypy_return_type' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'stypy_return_type', result_or__2244)
    
    # ################# End of 'bitwise_or(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'bitwise_or' in the type store
    # Getting the type of 'stypy_return_type' (line 12)
    stypy_return_type_2245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2245)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'bitwise_or'
    return stypy_return_type_2245

# Assigning a type to the variable 'bitwise_or' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'bitwise_or', bitwise_or)

@norecursion
def flow_sensitive(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'flow_sensitive'
    module_type_store = module_type_store.open_function_context('flow_sensitive', 16, 0, False)
    
    # Passed parameters checking function
    flow_sensitive.stypy_localization = localization
    flow_sensitive.stypy_type_of_self = None
    flow_sensitive.stypy_type_store = module_type_store
    flow_sensitive.stypy_function_name = 'flow_sensitive'
    flow_sensitive.stypy_param_names_list = ['obj', 'condition']
    flow_sensitive.stypy_varargs_param_name = None
    flow_sensitive.stypy_kwargs_param_name = None
    flow_sensitive.stypy_call_defaults = defaults
    flow_sensitive.stypy_call_varargs = varargs
    flow_sensitive.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'flow_sensitive', ['obj', 'condition'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'flow_sensitive', localization, ['obj', 'condition'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'flow_sensitive(...)' code ##################

    
    # Getting the type of 'condition' (line 17)
    condition_2246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 7), 'condition')
    # Testing the type of an if condition (line 17)
    if_condition_2247 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 17, 4), condition_2246)
    # Assigning a type to the variable 'if_condition_2247' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'if_condition_2247', if_condition_2247)
    # SSA begins for if statement (line 17)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to inc(...): (line 18)
    # Processing the call arguments (line 18)
    int_2250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 16), 'int')
    # Processing the call keyword arguments (line 18)
    kwargs_2251 = {}
    # Getting the type of 'obj' (line 18)
    obj_2248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'obj', False)
    # Obtaining the member 'inc' of a type (line 18)
    inc_2249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 8), obj_2248, 'inc')
    # Calling inc(args, kwargs) (line 18)
    inc_call_result_2252 = invoke(stypy.reporting.localization.Localization(__file__, 18, 8), inc_2249, *[int_2250], **kwargs_2251)
    
    # SSA branch for the else part of an if statement (line 17)
    module_type_store.open_ssa_branch('else')
    
    # Call to inc(...): (line 20)
    # Processing the call arguments (line 20)
    float_2255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 16), 'float')
    # Processing the call keyword arguments (line 20)
    kwargs_2256 = {}
    # Getting the type of 'obj' (line 20)
    obj_2253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'obj', False)
    # Obtaining the member 'inc' of a type (line 20)
    inc_2254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), obj_2253, 'inc')
    # Calling inc(args, kwargs) (line 20)
    inc_call_result_2257 = invoke(stypy.reporting.localization.Localization(__file__, 20, 8), inc_2254, *[float_2255], **kwargs_2256)
    
    # SSA join for if statement (line 17)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to bitwise_or(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 'obj' (line 21)
    obj_2259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 22), 'obj', False)
    int_2260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 27), 'int')
    # Processing the call keyword arguments (line 21)
    kwargs_2261 = {}
    # Getting the type of 'bitwise_or' (line 21)
    bitwise_or_2258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 11), 'bitwise_or', False)
    # Calling bitwise_or(args, kwargs) (line 21)
    bitwise_or_call_result_2262 = invoke(stypy.reporting.localization.Localization(__file__, 21, 11), bitwise_or_2258, *[obj_2259, int_2260], **kwargs_2261)
    
    # Assigning a type to the variable 'stypy_return_type' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'stypy_return_type', bitwise_or_call_result_2262)
    
    # ################# End of 'flow_sensitive(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'flow_sensitive' in the type store
    # Getting the type of 'stypy_return_type' (line 16)
    stypy_return_type_2263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2263)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'flow_sensitive'
    return stypy_return_type_2263

# Assigning a type to the variable 'flow_sensitive' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'flow_sensitive', flow_sensitive)

# Assigning a Call to a Name (line 23):

# Call to Counter(...): (line 23)
# Processing the call keyword arguments (line 23)
kwargs_2265 = {}
# Getting the type of 'Counter' (line 23)
Counter_2264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 6), 'Counter', False)
# Calling Counter(args, kwargs) (line 23)
Counter_call_result_2266 = invoke(stypy.reporting.localization.Localization(__file__, 23, 6), Counter_2264, *[], **kwargs_2265)

# Assigning a type to the variable 'obj' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'obj', Counter_call_result_2266)

# Call to flow_sensitive(...): (line 24)
# Processing the call arguments (line 24)
# Getting the type of 'obj' (line 24)
obj_2268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 15), 'obj', False)


# Call to randint(...): (line 24)
# Processing the call arguments (line 24)
int_2271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 35), 'int')
int_2272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 38), 'int')
# Processing the call keyword arguments (line 24)
kwargs_2273 = {}
# Getting the type of 'random' (line 24)
random_2269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 20), 'random', False)
# Obtaining the member 'randint' of a type (line 24)
randint_2270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 20), random_2269, 'randint')
# Calling randint(args, kwargs) (line 24)
randint_call_result_2274 = invoke(stypy.reporting.localization.Localization(__file__, 24, 20), randint_2270, *[int_2271, int_2272], **kwargs_2273)

int_2275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 44), 'int')
# Applying the binary operator '==' (line 24)
result_eq_2276 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 20), '==', randint_call_result_2274, int_2275)

# Processing the call keyword arguments (line 24)
kwargs_2277 = {}
# Getting the type of 'flow_sensitive' (line 24)
flow_sensitive_2267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'flow_sensitive', False)
# Calling flow_sensitive(args, kwargs) (line 24)
flow_sensitive_call_result_2278 = invoke(stypy.reporting.localization.Localization(__file__, 24, 0), flow_sensitive_2267, *[obj_2268, result_eq_2276], **kwargs_2277)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
