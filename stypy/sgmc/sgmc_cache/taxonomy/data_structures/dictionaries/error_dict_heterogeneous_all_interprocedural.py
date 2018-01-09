
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # coding=utf-8
2: 
3: __doc__ = "Inferring the type of inter-procedural heterogeneous dicts"
4: 
5: if __name__ == '__main__':
6:     def create_dict():
7:         d = dict()
8: 
9:         d["one"] = 1
10:         d[2] = "two"
11:         d[3.4] = list()
12: 
13:         return d
14: 
15: 
16:     d = create_dict()
17:     it_keys = d.keys()
18:     it_values = d.values()
19:     it_items = d.items()
20: 
21:     # Type warning
22:     print it_keys[0] + 3
23:     # Type warning
24:     print it_values[0] + "str"
25:     # Type error
26:     print it_items[0] + 3
27: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


# Assigning a Str to a Name (line 3):
str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 10), 'str', 'Inferring the type of inter-procedural heterogeneous dicts')
# Assigning a type to the variable '__doc__' (line 3)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), '__doc__', str_1)

if (__name__ == '__main__'):

    @norecursion
    def create_dict(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'create_dict'
        module_type_store = module_type_store.open_function_context('create_dict', 6, 4, False)
        
        # Passed parameters checking function
        create_dict.stypy_localization = localization
        create_dict.stypy_type_of_self = None
        create_dict.stypy_type_store = module_type_store
        create_dict.stypy_function_name = 'create_dict'
        create_dict.stypy_param_names_list = []
        create_dict.stypy_varargs_param_name = None
        create_dict.stypy_kwargs_param_name = None
        create_dict.stypy_call_defaults = defaults
        create_dict.stypy_call_varargs = varargs
        create_dict.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'create_dict', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'create_dict', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'create_dict(...)' code ##################

        
        # Assigning a Call to a Name (line 7):
        
        # Call to dict(...): (line 7)
        # Processing the call keyword arguments (line 7)
        kwargs_3 = {}
        # Getting the type of 'dict' (line 7)
        dict_2 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 12), 'dict', False)
        # Calling dict(args, kwargs) (line 7)
        dict_call_result_4 = invoke(stypy.reporting.localization.Localization(__file__, 7, 12), dict_2, *[], **kwargs_3)
        
        # Assigning a type to the variable 'd' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'd', dict_call_result_4)
        
        # Assigning a Num to a Subscript (line 9):
        int_5 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 19), 'int')
        # Getting the type of 'd' (line 9)
        d_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'd')
        str_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 10), 'str', 'one')
        # Storing an element on a container (line 9)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 8), d_6, (str_7, int_5))
        
        # Assigning a Str to a Subscript (line 10):
        str_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 15), 'str', 'two')
        # Getting the type of 'd' (line 10)
        d_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'd')
        int_10 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 10), 'int')
        # Storing an element on a container (line 10)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 8), d_9, (int_10, str_8))
        
        # Assigning a Call to a Subscript (line 11):
        
        # Call to list(...): (line 11)
        # Processing the call keyword arguments (line 11)
        kwargs_12 = {}
        # Getting the type of 'list' (line 11)
        list_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 17), 'list', False)
        # Calling list(args, kwargs) (line 11)
        list_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 11, 17), list_11, *[], **kwargs_12)
        
        # Getting the type of 'd' (line 11)
        d_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'd')
        float_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 10), 'float')
        # Storing an element on a container (line 11)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 8), d_14, (float_15, list_call_result_13))
        # Getting the type of 'd' (line 13)
        d_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 15), 'd')
        # Assigning a type to the variable 'stypy_return_type' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'stypy_return_type', d_16)
        
        # ################# End of 'create_dict(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_dict' in the type store
        # Getting the type of 'stypy_return_type' (line 6)
        stypy_return_type_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_dict'
        return stypy_return_type_17

    # Assigning a type to the variable 'create_dict' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'create_dict', create_dict)
    
    # Assigning a Call to a Name (line 16):
    
    # Call to create_dict(...): (line 16)
    # Processing the call keyword arguments (line 16)
    kwargs_19 = {}
    # Getting the type of 'create_dict' (line 16)
    create_dict_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'create_dict', False)
    # Calling create_dict(args, kwargs) (line 16)
    create_dict_call_result_20 = invoke(stypy.reporting.localization.Localization(__file__, 16, 8), create_dict_18, *[], **kwargs_19)
    
    # Assigning a type to the variable 'd' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'd', create_dict_call_result_20)
    
    # Assigning a Call to a Name (line 17):
    
    # Call to keys(...): (line 17)
    # Processing the call keyword arguments (line 17)
    kwargs_23 = {}
    # Getting the type of 'd' (line 17)
    d_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 14), 'd', False)
    # Obtaining the member 'keys' of a type (line 17)
    keys_22 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 14), d_21, 'keys')
    # Calling keys(args, kwargs) (line 17)
    keys_call_result_24 = invoke(stypy.reporting.localization.Localization(__file__, 17, 14), keys_22, *[], **kwargs_23)
    
    # Assigning a type to the variable 'it_keys' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'it_keys', keys_call_result_24)
    
    # Assigning a Call to a Name (line 18):
    
    # Call to values(...): (line 18)
    # Processing the call keyword arguments (line 18)
    kwargs_27 = {}
    # Getting the type of 'd' (line 18)
    d_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 16), 'd', False)
    # Obtaining the member 'values' of a type (line 18)
    values_26 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 16), d_25, 'values')
    # Calling values(args, kwargs) (line 18)
    values_call_result_28 = invoke(stypy.reporting.localization.Localization(__file__, 18, 16), values_26, *[], **kwargs_27)
    
    # Assigning a type to the variable 'it_values' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'it_values', values_call_result_28)
    
    # Assigning a Call to a Name (line 19):
    
    # Call to items(...): (line 19)
    # Processing the call keyword arguments (line 19)
    kwargs_31 = {}
    # Getting the type of 'd' (line 19)
    d_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 15), 'd', False)
    # Obtaining the member 'items' of a type (line 19)
    items_30 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 15), d_29, 'items')
    # Calling items(args, kwargs) (line 19)
    items_call_result_32 = invoke(stypy.reporting.localization.Localization(__file__, 19, 15), items_30, *[], **kwargs_31)
    
    # Assigning a type to the variable 'it_items' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'it_items', items_call_result_32)
    
    # Obtaining the type of the subscript
    int_33 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 18), 'int')
    # Getting the type of 'it_keys' (line 22)
    it_keys_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 10), 'it_keys')
    # Obtaining the member '__getitem__' of a type (line 22)
    getitem___35 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 10), it_keys_34, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 22)
    subscript_call_result_36 = invoke(stypy.reporting.localization.Localization(__file__, 22, 10), getitem___35, int_33)
    
    int_37 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 23), 'int')
    # Applying the binary operator '+' (line 22)
    result_add_38 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 10), '+', subscript_call_result_36, int_37)
    
    
    # Obtaining the type of the subscript
    int_39 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 20), 'int')
    # Getting the type of 'it_values' (line 24)
    it_values_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 10), 'it_values')
    # Obtaining the member '__getitem__' of a type (line 24)
    getitem___41 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 10), it_values_40, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 24)
    subscript_call_result_42 = invoke(stypy.reporting.localization.Localization(__file__, 24, 10), getitem___41, int_39)
    
    str_43 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 25), 'str', 'str')
    # Applying the binary operator '+' (line 24)
    result_add_44 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 10), '+', subscript_call_result_42, str_43)
    
    
    # Obtaining the type of the subscript
    int_45 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 19), 'int')
    # Getting the type of 'it_items' (line 26)
    it_items_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 10), 'it_items')
    # Obtaining the member '__getitem__' of a type (line 26)
    getitem___47 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 10), it_items_46, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 26)
    subscript_call_result_48 = invoke(stypy.reporting.localization.Localization(__file__, 26, 10), getitem___47, int_45)
    
    int_49 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 24), 'int')
    # Applying the binary operator '+' (line 26)
    result_add_50 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 10), '+', subscript_call_result_48, int_49)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
