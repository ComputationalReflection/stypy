
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy import *
2: import types
3: 
4: '''
5: Several functions that work with known Python types, placed here to facilitate the readability of the code.
6: Altough any list of known types may be used, the functions are aimed to manipulate the
7: known_python_type_typename_samplevalues type list defined in known_python_types.py
8: 
9: Automatic type rule generation makes heavy usage of these functions to create type rules by invoking members using
10: known types
11: '''
12: 
13: 
14: def get_known_types(type_table=known_python_type_typename_samplevalues):
15:     '''
16:     Obtains a list of the known types in the known_python_type_typename_samplevalues list (used by default)
17:     '''
18:     return type_table.keys()
19: 
20: 
21: def add_known_type(type_, type_name, type_value, type_table=known_python_type_typename_samplevalues):
22:     '''
23:     Allows to add a type to the list of known types in the known_python_type_typename_samplevalues list (used by
24:     default)
25:     '''
26:     type_table[type_] = (type_name, type_value)
27: 
28: 
29: def remove_known_type(type_, type_table=known_python_type_typename_samplevalues):
30:     '''
31:     Delete a type to the list of known types in the known_python_type_typename_samplevalues list (used by default)
32:     '''
33:     del type_table[type_]
34: 
35: 
36: def is_known_type(type_, type_table=known_python_type_typename_samplevalues):
37:     '''
38:     Determines if a type or a type name is in the list of known types in the known_python_type_typename_samplevalues
39:     list (used by default)
40:     '''
41:     # Is a type name instead of a type?
42:     if isinstance(type_, str):
43:         for table_entry in type_table:
44:             if type_ == type_table[table_entry][0]:
45:                 return True
46: 
47:         return False
48:     else:
49:         return type_ in type_table
50: 
51: 
52: def get_known_types_and_values(type_table=known_python_type_typename_samplevalues):
53:     '''
54:     Obtains a list of the library known types and a sample value for each one from the
55:     known_python_type_typename_samplevalues  list (used by default)
56:     '''
57:     list_ = type_table.items()
58:     ret_list = []
59:     for elem in list_:
60:         ret_list.append((elem[0], elem[1][1]))
61: 
62:     return ret_list
63: 
64: 
65: def get_type_name(type_, type_table=known_python_type_typename_samplevalues):
66:     '''
67:     Gets the type name of the passed type as defined in the known_python_type_typename_samplevalues
68:     list (used by default)
69:     '''
70:     if type_ == types.NotImplementedType:
71:         return "types.NotImplementedType"
72: 
73:     try:
74:         return type_table[type_][0]
75:     except (KeyError, TypeError):
76:         if type_ is __builtins__:
77:             return '__builtins__'
78: 
79:         if hasattr(type_, "__name__"):
80:             if hasattr(ExtraTypeDefinitions, type_.__name__):
81:                 return "ExtraTypeDefinitions." + type_.__name__
82: 
83:             if type_.__name__ == "iterator":
84:                 return "ExtraTypeDefinitions.listiterator"
85: 
86:             return type_.__name__
87:         else:
88:             return type(type_).__name__
89: 
90: 
91: def get_type_sample_value(type_, type_table=known_python_type_typename_samplevalues):
92:     '''
93:     Gets a sample value of the passed type from the known_python_type_typename_samplevalues
94:     list (used by default)
95:     '''
96:     return type_table[type_][1]
97: 
98: 
99: def get_type_from_name(name, type_table=known_python_type_typename_samplevalues):
100:     '''
101:     Gets the type object of the passed type name from the known_python_type_typename_samplevalues
102:     list (used by default)
103:     '''
104:     if "NotImplementedType" in name:
105:         return "types.NotImplementedType"
106: 
107:     keys = type_table.keys()
108:     for key in keys:
109:         if name == type_table[key][0]:
110:             return key
111: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy import ' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/instantiation_copy/')
import_9457 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy')

if (type(import_9457) is not StypyTypeError):

    if (import_9457 != 'pyd_module'):
        __import__(import_9457)
        sys_modules_9458 = sys.modules[import_9457]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy', sys_modules_9458.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_9458, sys_modules_9458.module_type_store, module_type_store)
    else:
        from stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy', import_9457)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/instantiation_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import types' statement (line 2)
import types

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'types', types, module_type_store)

str_9459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, (-1)), 'str', '\nSeveral functions that work with known Python types, placed here to facilitate the readability of the code.\nAltough any list of known types may be used, the functions are aimed to manipulate the\nknown_python_type_typename_samplevalues type list defined in known_python_types.py\n\nAutomatic type rule generation makes heavy usage of these functions to create type rules by invoking members using\nknown types\n')

@norecursion
def get_known_types(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'known_python_type_typename_samplevalues' (line 14)
    known_python_type_typename_samplevalues_9460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 31), 'known_python_type_typename_samplevalues')
    defaults = [known_python_type_typename_samplevalues_9460]
    # Create a new context for function 'get_known_types'
    module_type_store = module_type_store.open_function_context('get_known_types', 14, 0, False)
    
    # Passed parameters checking function
    get_known_types.stypy_localization = localization
    get_known_types.stypy_type_of_self = None
    get_known_types.stypy_type_store = module_type_store
    get_known_types.stypy_function_name = 'get_known_types'
    get_known_types.stypy_param_names_list = ['type_table']
    get_known_types.stypy_varargs_param_name = None
    get_known_types.stypy_kwargs_param_name = None
    get_known_types.stypy_call_defaults = defaults
    get_known_types.stypy_call_varargs = varargs
    get_known_types.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_known_types', ['type_table'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_known_types', localization, ['type_table'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_known_types(...)' code ##################

    str_9461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, (-1)), 'str', '\n    Obtains a list of the known types in the known_python_type_typename_samplevalues list (used by default)\n    ')
    
    # Call to keys(...): (line 18)
    # Processing the call keyword arguments (line 18)
    kwargs_9464 = {}
    # Getting the type of 'type_table' (line 18)
    type_table_9462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 11), 'type_table', False)
    # Obtaining the member 'keys' of a type (line 18)
    keys_9463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 11), type_table_9462, 'keys')
    # Calling keys(args, kwargs) (line 18)
    keys_call_result_9465 = invoke(stypy.reporting.localization.Localization(__file__, 18, 11), keys_9463, *[], **kwargs_9464)
    
    # Assigning a type to the variable 'stypy_return_type' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'stypy_return_type', keys_call_result_9465)
    
    # ################# End of 'get_known_types(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_known_types' in the type store
    # Getting the type of 'stypy_return_type' (line 14)
    stypy_return_type_9466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_9466)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_known_types'
    return stypy_return_type_9466

# Assigning a type to the variable 'get_known_types' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'get_known_types', get_known_types)

@norecursion
def add_known_type(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'known_python_type_typename_samplevalues' (line 21)
    known_python_type_typename_samplevalues_9467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 60), 'known_python_type_typename_samplevalues')
    defaults = [known_python_type_typename_samplevalues_9467]
    # Create a new context for function 'add_known_type'
    module_type_store = module_type_store.open_function_context('add_known_type', 21, 0, False)
    
    # Passed parameters checking function
    add_known_type.stypy_localization = localization
    add_known_type.stypy_type_of_self = None
    add_known_type.stypy_type_store = module_type_store
    add_known_type.stypy_function_name = 'add_known_type'
    add_known_type.stypy_param_names_list = ['type_', 'type_name', 'type_value', 'type_table']
    add_known_type.stypy_varargs_param_name = None
    add_known_type.stypy_kwargs_param_name = None
    add_known_type.stypy_call_defaults = defaults
    add_known_type.stypy_call_varargs = varargs
    add_known_type.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'add_known_type', ['type_', 'type_name', 'type_value', 'type_table'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'add_known_type', localization, ['type_', 'type_name', 'type_value', 'type_table'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'add_known_type(...)' code ##################

    str_9468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, (-1)), 'str', '\n    Allows to add a type to the list of known types in the known_python_type_typename_samplevalues list (used by\n    default)\n    ')
    
    # Assigning a Tuple to a Subscript (line 26):
    
    # Obtaining an instance of the builtin type 'tuple' (line 26)
    tuple_9469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 26)
    # Adding element type (line 26)
    # Getting the type of 'type_name' (line 26)
    type_name_9470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 25), 'type_name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 25), tuple_9469, type_name_9470)
    # Adding element type (line 26)
    # Getting the type of 'type_value' (line 26)
    type_value_9471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 36), 'type_value')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 25), tuple_9469, type_value_9471)
    
    # Getting the type of 'type_table' (line 26)
    type_table_9472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'type_table')
    # Getting the type of 'type_' (line 26)
    type__9473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 15), 'type_')
    # Storing an element on a container (line 26)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 4), type_table_9472, (type__9473, tuple_9469))
    
    # ################# End of 'add_known_type(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'add_known_type' in the type store
    # Getting the type of 'stypy_return_type' (line 21)
    stypy_return_type_9474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_9474)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'add_known_type'
    return stypy_return_type_9474

# Assigning a type to the variable 'add_known_type' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'add_known_type', add_known_type)

@norecursion
def remove_known_type(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'known_python_type_typename_samplevalues' (line 29)
    known_python_type_typename_samplevalues_9475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 40), 'known_python_type_typename_samplevalues')
    defaults = [known_python_type_typename_samplevalues_9475]
    # Create a new context for function 'remove_known_type'
    module_type_store = module_type_store.open_function_context('remove_known_type', 29, 0, False)
    
    # Passed parameters checking function
    remove_known_type.stypy_localization = localization
    remove_known_type.stypy_type_of_self = None
    remove_known_type.stypy_type_store = module_type_store
    remove_known_type.stypy_function_name = 'remove_known_type'
    remove_known_type.stypy_param_names_list = ['type_', 'type_table']
    remove_known_type.stypy_varargs_param_name = None
    remove_known_type.stypy_kwargs_param_name = None
    remove_known_type.stypy_call_defaults = defaults
    remove_known_type.stypy_call_varargs = varargs
    remove_known_type.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'remove_known_type', ['type_', 'type_table'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'remove_known_type', localization, ['type_', 'type_table'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'remove_known_type(...)' code ##################

    str_9476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, (-1)), 'str', '\n    Delete a type to the list of known types in the known_python_type_typename_samplevalues list (used by default)\n    ')
    # Deleting a member
    # Getting the type of 'type_table' (line 33)
    type_table_9477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'type_table')
    
    # Obtaining the type of the subscript
    # Getting the type of 'type_' (line 33)
    type__9478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 19), 'type_')
    # Getting the type of 'type_table' (line 33)
    type_table_9479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'type_table')
    # Obtaining the member '__getitem__' of a type (line 33)
    getitem___9480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), type_table_9479, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 33)
    subscript_call_result_9481 = invoke(stypy.reporting.localization.Localization(__file__, 33, 8), getitem___9480, type__9478)
    
    del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 4), type_table_9477, subscript_call_result_9481)
    
    # ################# End of 'remove_known_type(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'remove_known_type' in the type store
    # Getting the type of 'stypy_return_type' (line 29)
    stypy_return_type_9482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_9482)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'remove_known_type'
    return stypy_return_type_9482

# Assigning a type to the variable 'remove_known_type' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'remove_known_type', remove_known_type)

@norecursion
def is_known_type(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'known_python_type_typename_samplevalues' (line 36)
    known_python_type_typename_samplevalues_9483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 36), 'known_python_type_typename_samplevalues')
    defaults = [known_python_type_typename_samplevalues_9483]
    # Create a new context for function 'is_known_type'
    module_type_store = module_type_store.open_function_context('is_known_type', 36, 0, False)
    
    # Passed parameters checking function
    is_known_type.stypy_localization = localization
    is_known_type.stypy_type_of_self = None
    is_known_type.stypy_type_store = module_type_store
    is_known_type.stypy_function_name = 'is_known_type'
    is_known_type.stypy_param_names_list = ['type_', 'type_table']
    is_known_type.stypy_varargs_param_name = None
    is_known_type.stypy_kwargs_param_name = None
    is_known_type.stypy_call_defaults = defaults
    is_known_type.stypy_call_varargs = varargs
    is_known_type.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'is_known_type', ['type_', 'type_table'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'is_known_type', localization, ['type_', 'type_table'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'is_known_type(...)' code ##################

    str_9484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, (-1)), 'str', '\n    Determines if a type or a type name is in the list of known types in the known_python_type_typename_samplevalues\n    list (used by default)\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 42)
    # Getting the type of 'str' (line 42)
    str_9485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 25), 'str')
    # Getting the type of 'type_' (line 42)
    type__9486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 18), 'type_')
    
    (may_be_9487, more_types_in_union_9488) = may_be_subtype(str_9485, type__9486)

    if may_be_9487:

        if more_types_in_union_9488:
            # Runtime conditional SSA (line 42)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'type_' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'type_', remove_not_subtype_from_union(type__9486, str))
        
        # Getting the type of 'type_table' (line 43)
        type_table_9489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 27), 'type_table')
        # Assigning a type to the variable 'type_table_9489' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'type_table_9489', type_table_9489)
        # Testing if the for loop is going to be iterated (line 43)
        # Testing the type of a for loop iterable (line 43)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 43, 8), type_table_9489)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 43, 8), type_table_9489):
            # Getting the type of the for loop variable (line 43)
            for_loop_var_9490 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 43, 8), type_table_9489)
            # Assigning a type to the variable 'table_entry' (line 43)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'table_entry', for_loop_var_9490)
            # SSA begins for a for statement (line 43)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'type_' (line 44)
            type__9491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 15), 'type_')
            
            # Obtaining the type of the subscript
            int_9492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 48), 'int')
            
            # Obtaining the type of the subscript
            # Getting the type of 'table_entry' (line 44)
            table_entry_9493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 35), 'table_entry')
            # Getting the type of 'type_table' (line 44)
            type_table_9494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 24), 'type_table')
            # Obtaining the member '__getitem__' of a type (line 44)
            getitem___9495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 24), type_table_9494, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 44)
            subscript_call_result_9496 = invoke(stypy.reporting.localization.Localization(__file__, 44, 24), getitem___9495, table_entry_9493)
            
            # Obtaining the member '__getitem__' of a type (line 44)
            getitem___9497 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 24), subscript_call_result_9496, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 44)
            subscript_call_result_9498 = invoke(stypy.reporting.localization.Localization(__file__, 44, 24), getitem___9497, int_9492)
            
            # Applying the binary operator '==' (line 44)
            result_eq_9499 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 15), '==', type__9491, subscript_call_result_9498)
            
            # Testing if the type of an if condition is none (line 44)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 44, 12), result_eq_9499):
                pass
            else:
                
                # Testing the type of an if condition (line 44)
                if_condition_9500 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 44, 12), result_eq_9499)
                # Assigning a type to the variable 'if_condition_9500' (line 44)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'if_condition_9500', if_condition_9500)
                # SSA begins for if statement (line 44)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'True' (line 45)
                True_9501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 23), 'True')
                # Assigning a type to the variable 'stypy_return_type' (line 45)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 16), 'stypy_return_type', True_9501)
                # SSA join for if statement (line 44)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'False' (line 47)
        False_9502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'stypy_return_type', False_9502)

        if more_types_in_union_9488:
            # Runtime conditional SSA for else branch (line 42)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_9487) or more_types_in_union_9488):
        # Assigning a type to the variable 'type_' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'type_', remove_subtype_from_union(type__9486, str))
        
        # Getting the type of 'type_' (line 49)
        type__9503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 15), 'type_')
        # Getting the type of 'type_table' (line 49)
        type_table_9504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 24), 'type_table')
        # Applying the binary operator 'in' (line 49)
        result_contains_9505 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 15), 'in', type__9503, type_table_9504)
        
        # Assigning a type to the variable 'stypy_return_type' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'stypy_return_type', result_contains_9505)

        if (may_be_9487 and more_types_in_union_9488):
            # SSA join for if statement (line 42)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of 'is_known_type(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_known_type' in the type store
    # Getting the type of 'stypy_return_type' (line 36)
    stypy_return_type_9506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_9506)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_known_type'
    return stypy_return_type_9506

# Assigning a type to the variable 'is_known_type' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'is_known_type', is_known_type)

@norecursion
def get_known_types_and_values(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'known_python_type_typename_samplevalues' (line 52)
    known_python_type_typename_samplevalues_9507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 42), 'known_python_type_typename_samplevalues')
    defaults = [known_python_type_typename_samplevalues_9507]
    # Create a new context for function 'get_known_types_and_values'
    module_type_store = module_type_store.open_function_context('get_known_types_and_values', 52, 0, False)
    
    # Passed parameters checking function
    get_known_types_and_values.stypy_localization = localization
    get_known_types_and_values.stypy_type_of_self = None
    get_known_types_and_values.stypy_type_store = module_type_store
    get_known_types_and_values.stypy_function_name = 'get_known_types_and_values'
    get_known_types_and_values.stypy_param_names_list = ['type_table']
    get_known_types_and_values.stypy_varargs_param_name = None
    get_known_types_and_values.stypy_kwargs_param_name = None
    get_known_types_and_values.stypy_call_defaults = defaults
    get_known_types_and_values.stypy_call_varargs = varargs
    get_known_types_and_values.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_known_types_and_values', ['type_table'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_known_types_and_values', localization, ['type_table'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_known_types_and_values(...)' code ##################

    str_9508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, (-1)), 'str', '\n    Obtains a list of the library known types and a sample value for each one from the\n    known_python_type_typename_samplevalues  list (used by default)\n    ')
    
    # Assigning a Call to a Name (line 57):
    
    # Call to items(...): (line 57)
    # Processing the call keyword arguments (line 57)
    kwargs_9511 = {}
    # Getting the type of 'type_table' (line 57)
    type_table_9509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'type_table', False)
    # Obtaining the member 'items' of a type (line 57)
    items_9510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 12), type_table_9509, 'items')
    # Calling items(args, kwargs) (line 57)
    items_call_result_9512 = invoke(stypy.reporting.localization.Localization(__file__, 57, 12), items_9510, *[], **kwargs_9511)
    
    # Assigning a type to the variable 'list_' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'list_', items_call_result_9512)
    
    # Assigning a List to a Name (line 58):
    
    # Obtaining an instance of the builtin type 'list' (line 58)
    list_9513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 58)
    
    # Assigning a type to the variable 'ret_list' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'ret_list', list_9513)
    
    # Getting the type of 'list_' (line 59)
    list__9514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'list_')
    # Assigning a type to the variable 'list__9514' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'list__9514', list__9514)
    # Testing if the for loop is going to be iterated (line 59)
    # Testing the type of a for loop iterable (line 59)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 59, 4), list__9514)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 59, 4), list__9514):
        # Getting the type of the for loop variable (line 59)
        for_loop_var_9515 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 59, 4), list__9514)
        # Assigning a type to the variable 'elem' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'elem', for_loop_var_9515)
        # SSA begins for a for statement (line 59)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 60)
        # Processing the call arguments (line 60)
        
        # Obtaining an instance of the builtin type 'tuple' (line 60)
        tuple_9518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 60)
        # Adding element type (line 60)
        
        # Obtaining the type of the subscript
        int_9519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 30), 'int')
        # Getting the type of 'elem' (line 60)
        elem_9520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 25), 'elem', False)
        # Obtaining the member '__getitem__' of a type (line 60)
        getitem___9521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 25), elem_9520, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 60)
        subscript_call_result_9522 = invoke(stypy.reporting.localization.Localization(__file__, 60, 25), getitem___9521, int_9519)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 25), tuple_9518, subscript_call_result_9522)
        # Adding element type (line 60)
        
        # Obtaining the type of the subscript
        int_9523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 42), 'int')
        
        # Obtaining the type of the subscript
        int_9524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 39), 'int')
        # Getting the type of 'elem' (line 60)
        elem_9525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 34), 'elem', False)
        # Obtaining the member '__getitem__' of a type (line 60)
        getitem___9526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 34), elem_9525, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 60)
        subscript_call_result_9527 = invoke(stypy.reporting.localization.Localization(__file__, 60, 34), getitem___9526, int_9524)
        
        # Obtaining the member '__getitem__' of a type (line 60)
        getitem___9528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 34), subscript_call_result_9527, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 60)
        subscript_call_result_9529 = invoke(stypy.reporting.localization.Localization(__file__, 60, 34), getitem___9528, int_9523)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 25), tuple_9518, subscript_call_result_9529)
        
        # Processing the call keyword arguments (line 60)
        kwargs_9530 = {}
        # Getting the type of 'ret_list' (line 60)
        ret_list_9516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'ret_list', False)
        # Obtaining the member 'append' of a type (line 60)
        append_9517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), ret_list_9516, 'append')
        # Calling append(args, kwargs) (line 60)
        append_call_result_9531 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), append_9517, *[tuple_9518], **kwargs_9530)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'ret_list' (line 62)
    ret_list_9532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 11), 'ret_list')
    # Assigning a type to the variable 'stypy_return_type' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'stypy_return_type', ret_list_9532)
    
    # ################# End of 'get_known_types_and_values(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_known_types_and_values' in the type store
    # Getting the type of 'stypy_return_type' (line 52)
    stypy_return_type_9533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_9533)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_known_types_and_values'
    return stypy_return_type_9533

# Assigning a type to the variable 'get_known_types_and_values' (line 52)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'get_known_types_and_values', get_known_types_and_values)

@norecursion
def get_type_name(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'known_python_type_typename_samplevalues' (line 65)
    known_python_type_typename_samplevalues_9534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 36), 'known_python_type_typename_samplevalues')
    defaults = [known_python_type_typename_samplevalues_9534]
    # Create a new context for function 'get_type_name'
    module_type_store = module_type_store.open_function_context('get_type_name', 65, 0, False)
    
    # Passed parameters checking function
    get_type_name.stypy_localization = localization
    get_type_name.stypy_type_of_self = None
    get_type_name.stypy_type_store = module_type_store
    get_type_name.stypy_function_name = 'get_type_name'
    get_type_name.stypy_param_names_list = ['type_', 'type_table']
    get_type_name.stypy_varargs_param_name = None
    get_type_name.stypy_kwargs_param_name = None
    get_type_name.stypy_call_defaults = defaults
    get_type_name.stypy_call_varargs = varargs
    get_type_name.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_type_name', ['type_', 'type_table'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_type_name', localization, ['type_', 'type_table'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_type_name(...)' code ##################

    str_9535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, (-1)), 'str', '\n    Gets the type name of the passed type as defined in the known_python_type_typename_samplevalues\n    list (used by default)\n    ')
    
    # Getting the type of 'type_' (line 70)
    type__9536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 7), 'type_')
    # Getting the type of 'types' (line 70)
    types_9537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'types')
    # Obtaining the member 'NotImplementedType' of a type (line 70)
    NotImplementedType_9538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 16), types_9537, 'NotImplementedType')
    # Applying the binary operator '==' (line 70)
    result_eq_9539 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 7), '==', type__9536, NotImplementedType_9538)
    
    # Testing if the type of an if condition is none (line 70)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 70, 4), result_eq_9539):
        pass
    else:
        
        # Testing the type of an if condition (line 70)
        if_condition_9540 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 70, 4), result_eq_9539)
        # Assigning a type to the variable 'if_condition_9540' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'if_condition_9540', if_condition_9540)
        # SSA begins for if statement (line 70)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_9541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 15), 'str', 'types.NotImplementedType')
        # Assigning a type to the variable 'stypy_return_type' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'stypy_return_type', str_9541)
        # SSA join for if statement (line 70)
        module_type_store = module_type_store.join_ssa_context()
        

    
    
    # SSA begins for try-except statement (line 73)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Obtaining the type of the subscript
    int_9542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 33), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'type_' (line 74)
    type__9543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 26), 'type_')
    # Getting the type of 'type_table' (line 74)
    type_table_9544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 15), 'type_table')
    # Obtaining the member '__getitem__' of a type (line 74)
    getitem___9545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 15), type_table_9544, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 74)
    subscript_call_result_9546 = invoke(stypy.reporting.localization.Localization(__file__, 74, 15), getitem___9545, type__9543)
    
    # Obtaining the member '__getitem__' of a type (line 74)
    getitem___9547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 15), subscript_call_result_9546, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 74)
    subscript_call_result_9548 = invoke(stypy.reporting.localization.Localization(__file__, 74, 15), getitem___9547, int_9542)
    
    # Assigning a type to the variable 'stypy_return_type' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'stypy_return_type', subscript_call_result_9548)
    # SSA branch for the except part of a try statement (line 73)
    # SSA branch for the except 'Tuple' branch of a try statement (line 73)
    module_type_store.open_ssa_branch('except')
    
    # Getting the type of 'type_' (line 76)
    type__9549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 11), 'type_')
    # Getting the type of '__builtins__' (line 76)
    builtins___9550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 20), '__builtins__')
    # Applying the binary operator 'is' (line 76)
    result_is__9551 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 11), 'is', type__9549, builtins___9550)
    
    # Testing if the type of an if condition is none (line 76)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 76, 8), result_is__9551):
        pass
    else:
        
        # Testing the type of an if condition (line 76)
        if_condition_9552 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 8), result_is__9551)
        # Assigning a type to the variable 'if_condition_9552' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'if_condition_9552', if_condition_9552)
        # SSA begins for if statement (line 76)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_9553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 19), 'str', '__builtins__')
        # Assigning a type to the variable 'stypy_return_type' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'stypy_return_type', str_9553)
        # SSA join for if statement (line 76)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Type idiom detected: calculating its left and rigth part (line 79)
    str_9554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 26), 'str', '__name__')
    # Getting the type of 'type_' (line 79)
    type__9555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 19), 'type_')
    
    (may_be_9556, more_types_in_union_9557) = may_provide_member(str_9554, type__9555)

    if may_be_9556:

        if more_types_in_union_9557:
            # Runtime conditional SSA (line 79)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'type_' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'type_', remove_not_member_provider_from_union(type__9555, '__name__'))
        
        # Call to hasattr(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'ExtraTypeDefinitions' (line 80)
        ExtraTypeDefinitions_9559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 23), 'ExtraTypeDefinitions', False)
        # Getting the type of 'type_' (line 80)
        type__9560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 45), 'type_', False)
        # Obtaining the member '__name__' of a type (line 80)
        name___9561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 45), type__9560, '__name__')
        # Processing the call keyword arguments (line 80)
        kwargs_9562 = {}
        # Getting the type of 'hasattr' (line 80)
        hasattr_9558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 15), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 80)
        hasattr_call_result_9563 = invoke(stypy.reporting.localization.Localization(__file__, 80, 15), hasattr_9558, *[ExtraTypeDefinitions_9559, name___9561], **kwargs_9562)
        
        # Testing if the type of an if condition is none (line 80)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 80, 12), hasattr_call_result_9563):
            pass
        else:
            
            # Testing the type of an if condition (line 80)
            if_condition_9564 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 12), hasattr_call_result_9563)
            # Assigning a type to the variable 'if_condition_9564' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'if_condition_9564', if_condition_9564)
            # SSA begins for if statement (line 80)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            str_9565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 23), 'str', 'ExtraTypeDefinitions.')
            # Getting the type of 'type_' (line 81)
            type__9566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 49), 'type_')
            # Obtaining the member '__name__' of a type (line 81)
            name___9567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 49), type__9566, '__name__')
            # Applying the binary operator '+' (line 81)
            result_add_9568 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 23), '+', str_9565, name___9567)
            
            # Assigning a type to the variable 'stypy_return_type' (line 81)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'stypy_return_type', result_add_9568)
            # SSA join for if statement (line 80)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'type_' (line 83)
        type__9569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 15), 'type_')
        # Obtaining the member '__name__' of a type (line 83)
        name___9570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 15), type__9569, '__name__')
        str_9571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 33), 'str', 'iterator')
        # Applying the binary operator '==' (line 83)
        result_eq_9572 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 15), '==', name___9570, str_9571)
        
        # Testing if the type of an if condition is none (line 83)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 83, 12), result_eq_9572):
            pass
        else:
            
            # Testing the type of an if condition (line 83)
            if_condition_9573 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 12), result_eq_9572)
            # Assigning a type to the variable 'if_condition_9573' (line 83)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'if_condition_9573', if_condition_9573)
            # SSA begins for if statement (line 83)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            str_9574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 23), 'str', 'ExtraTypeDefinitions.listiterator')
            # Assigning a type to the variable 'stypy_return_type' (line 84)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 16), 'stypy_return_type', str_9574)
            # SSA join for if statement (line 83)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'type_' (line 86)
        type__9575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 19), 'type_')
        # Obtaining the member '__name__' of a type (line 86)
        name___9576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 19), type__9575, '__name__')
        # Assigning a type to the variable 'stypy_return_type' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'stypy_return_type', name___9576)

        if more_types_in_union_9557:
            # Runtime conditional SSA for else branch (line 79)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_9556) or more_types_in_union_9557):
        # Assigning a type to the variable 'type_' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'type_', remove_member_provider_from_union(type__9555, '__name__'))
        
        # Call to type(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'type_' (line 88)
        type__9578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 24), 'type_', False)
        # Processing the call keyword arguments (line 88)
        kwargs_9579 = {}
        # Getting the type of 'type' (line 88)
        type_9577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 19), 'type', False)
        # Calling type(args, kwargs) (line 88)
        type_call_result_9580 = invoke(stypy.reporting.localization.Localization(__file__, 88, 19), type_9577, *[type__9578], **kwargs_9579)
        
        # Obtaining the member '__name__' of a type (line 88)
        name___9581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 19), type_call_result_9580, '__name__')
        # Assigning a type to the variable 'stypy_return_type' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'stypy_return_type', name___9581)

        if (may_be_9556 and more_types_in_union_9557):
            # SSA join for if statement (line 79)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for try-except statement (line 73)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'get_type_name(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_type_name' in the type store
    # Getting the type of 'stypy_return_type' (line 65)
    stypy_return_type_9582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_9582)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_type_name'
    return stypy_return_type_9582

# Assigning a type to the variable 'get_type_name' (line 65)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'get_type_name', get_type_name)

@norecursion
def get_type_sample_value(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'known_python_type_typename_samplevalues' (line 91)
    known_python_type_typename_samplevalues_9583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 44), 'known_python_type_typename_samplevalues')
    defaults = [known_python_type_typename_samplevalues_9583]
    # Create a new context for function 'get_type_sample_value'
    module_type_store = module_type_store.open_function_context('get_type_sample_value', 91, 0, False)
    
    # Passed parameters checking function
    get_type_sample_value.stypy_localization = localization
    get_type_sample_value.stypy_type_of_self = None
    get_type_sample_value.stypy_type_store = module_type_store
    get_type_sample_value.stypy_function_name = 'get_type_sample_value'
    get_type_sample_value.stypy_param_names_list = ['type_', 'type_table']
    get_type_sample_value.stypy_varargs_param_name = None
    get_type_sample_value.stypy_kwargs_param_name = None
    get_type_sample_value.stypy_call_defaults = defaults
    get_type_sample_value.stypy_call_varargs = varargs
    get_type_sample_value.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_type_sample_value', ['type_', 'type_table'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_type_sample_value', localization, ['type_', 'type_table'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_type_sample_value(...)' code ##################

    str_9584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, (-1)), 'str', '\n    Gets a sample value of the passed type from the known_python_type_typename_samplevalues\n    list (used by default)\n    ')
    
    # Obtaining the type of the subscript
    int_9585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 29), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'type_' (line 96)
    type__9586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 22), 'type_')
    # Getting the type of 'type_table' (line 96)
    type_table_9587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 11), 'type_table')
    # Obtaining the member '__getitem__' of a type (line 96)
    getitem___9588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 11), type_table_9587, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 96)
    subscript_call_result_9589 = invoke(stypy.reporting.localization.Localization(__file__, 96, 11), getitem___9588, type__9586)
    
    # Obtaining the member '__getitem__' of a type (line 96)
    getitem___9590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 11), subscript_call_result_9589, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 96)
    subscript_call_result_9591 = invoke(stypy.reporting.localization.Localization(__file__, 96, 11), getitem___9590, int_9585)
    
    # Assigning a type to the variable 'stypy_return_type' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'stypy_return_type', subscript_call_result_9591)
    
    # ################# End of 'get_type_sample_value(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_type_sample_value' in the type store
    # Getting the type of 'stypy_return_type' (line 91)
    stypy_return_type_9592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_9592)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_type_sample_value'
    return stypy_return_type_9592

# Assigning a type to the variable 'get_type_sample_value' (line 91)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'get_type_sample_value', get_type_sample_value)

@norecursion
def get_type_from_name(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'known_python_type_typename_samplevalues' (line 99)
    known_python_type_typename_samplevalues_9593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 40), 'known_python_type_typename_samplevalues')
    defaults = [known_python_type_typename_samplevalues_9593]
    # Create a new context for function 'get_type_from_name'
    module_type_store = module_type_store.open_function_context('get_type_from_name', 99, 0, False)
    
    # Passed parameters checking function
    get_type_from_name.stypy_localization = localization
    get_type_from_name.stypy_type_of_self = None
    get_type_from_name.stypy_type_store = module_type_store
    get_type_from_name.stypy_function_name = 'get_type_from_name'
    get_type_from_name.stypy_param_names_list = ['name', 'type_table']
    get_type_from_name.stypy_varargs_param_name = None
    get_type_from_name.stypy_kwargs_param_name = None
    get_type_from_name.stypy_call_defaults = defaults
    get_type_from_name.stypy_call_varargs = varargs
    get_type_from_name.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_type_from_name', ['name', 'type_table'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_type_from_name', localization, ['name', 'type_table'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_type_from_name(...)' code ##################

    str_9594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, (-1)), 'str', '\n    Gets the type object of the passed type name from the known_python_type_typename_samplevalues\n    list (used by default)\n    ')
    
    str_9595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 7), 'str', 'NotImplementedType')
    # Getting the type of 'name' (line 104)
    name_9596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 31), 'name')
    # Applying the binary operator 'in' (line 104)
    result_contains_9597 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 7), 'in', str_9595, name_9596)
    
    # Testing if the type of an if condition is none (line 104)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 104, 4), result_contains_9597):
        pass
    else:
        
        # Testing the type of an if condition (line 104)
        if_condition_9598 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 104, 4), result_contains_9597)
        # Assigning a type to the variable 'if_condition_9598' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'if_condition_9598', if_condition_9598)
        # SSA begins for if statement (line 104)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_9599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 15), 'str', 'types.NotImplementedType')
        # Assigning a type to the variable 'stypy_return_type' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'stypy_return_type', str_9599)
        # SSA join for if statement (line 104)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 107):
    
    # Call to keys(...): (line 107)
    # Processing the call keyword arguments (line 107)
    kwargs_9602 = {}
    # Getting the type of 'type_table' (line 107)
    type_table_9600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 11), 'type_table', False)
    # Obtaining the member 'keys' of a type (line 107)
    keys_9601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 11), type_table_9600, 'keys')
    # Calling keys(args, kwargs) (line 107)
    keys_call_result_9603 = invoke(stypy.reporting.localization.Localization(__file__, 107, 11), keys_9601, *[], **kwargs_9602)
    
    # Assigning a type to the variable 'keys' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'keys', keys_call_result_9603)
    
    # Getting the type of 'keys' (line 108)
    keys_9604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 15), 'keys')
    # Assigning a type to the variable 'keys_9604' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'keys_9604', keys_9604)
    # Testing if the for loop is going to be iterated (line 108)
    # Testing the type of a for loop iterable (line 108)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 108, 4), keys_9604)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 108, 4), keys_9604):
        # Getting the type of the for loop variable (line 108)
        for_loop_var_9605 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 108, 4), keys_9604)
        # Assigning a type to the variable 'key' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'key', for_loop_var_9605)
        # SSA begins for a for statement (line 108)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'name' (line 109)
        name_9606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 11), 'name')
        
        # Obtaining the type of the subscript
        int_9607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 35), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'key' (line 109)
        key_9608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 30), 'key')
        # Getting the type of 'type_table' (line 109)
        type_table_9609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'type_table')
        # Obtaining the member '__getitem__' of a type (line 109)
        getitem___9610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), type_table_9609, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 109)
        subscript_call_result_9611 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), getitem___9610, key_9608)
        
        # Obtaining the member '__getitem__' of a type (line 109)
        getitem___9612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), subscript_call_result_9611, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 109)
        subscript_call_result_9613 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), getitem___9612, int_9607)
        
        # Applying the binary operator '==' (line 109)
        result_eq_9614 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 11), '==', name_9606, subscript_call_result_9613)
        
        # Testing if the type of an if condition is none (line 109)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 109, 8), result_eq_9614):
            pass
        else:
            
            # Testing the type of an if condition (line 109)
            if_condition_9615 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 109, 8), result_eq_9614)
            # Assigning a type to the variable 'if_condition_9615' (line 109)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'if_condition_9615', if_condition_9615)
            # SSA begins for if statement (line 109)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'key' (line 110)
            key_9616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 19), 'key')
            # Assigning a type to the variable 'stypy_return_type' (line 110)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'stypy_return_type', key_9616)
            # SSA join for if statement (line 109)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'get_type_from_name(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_type_from_name' in the type store
    # Getting the type of 'stypy_return_type' (line 99)
    stypy_return_type_9617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_9617)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_type_from_name'
    return stypy_return_type_9617

# Assigning a type to the variable 'get_type_from_name' (line 99)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 0), 'get_type_from_name', get_type_from_name)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
