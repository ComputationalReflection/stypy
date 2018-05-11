
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from ....python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy import *
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

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy import ' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/instantiation_copy/')
import_9743 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy')

if (type(import_9743) is not StypyTypeError):

    if (import_9743 != 'pyd_module'):
        __import__(import_9743)
        sys_modules_9744 = sys.modules[import_9743]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy', sys_modules_9744.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_9744, sys_modules_9744.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy', import_9743)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/instantiation_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import types' statement (line 2)
import types

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'types', types, module_type_store)

str_9745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, (-1)), 'str', '\nSeveral functions that work with known Python types, placed here to facilitate the readability of the code.\nAltough any list of known types may be used, the functions are aimed to manipulate the\nknown_python_type_typename_samplevalues type list defined in known_python_types.py\n\nAutomatic type rule generation makes heavy usage of these functions to create type rules by invoking members using\nknown types\n')

@norecursion
def get_known_types(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'known_python_type_typename_samplevalues' (line 14)
    known_python_type_typename_samplevalues_9746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 31), 'known_python_type_typename_samplevalues')
    defaults = [known_python_type_typename_samplevalues_9746]
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

    str_9747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, (-1)), 'str', '\n    Obtains a list of the known types in the known_python_type_typename_samplevalues list (used by default)\n    ')
    
    # Call to keys(...): (line 18)
    # Processing the call keyword arguments (line 18)
    kwargs_9750 = {}
    # Getting the type of 'type_table' (line 18)
    type_table_9748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 11), 'type_table', False)
    # Obtaining the member 'keys' of a type (line 18)
    keys_9749 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 11), type_table_9748, 'keys')
    # Calling keys(args, kwargs) (line 18)
    keys_call_result_9751 = invoke(stypy.reporting.localization.Localization(__file__, 18, 11), keys_9749, *[], **kwargs_9750)
    
    # Assigning a type to the variable 'stypy_return_type' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'stypy_return_type', keys_call_result_9751)
    
    # ################# End of 'get_known_types(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_known_types' in the type store
    # Getting the type of 'stypy_return_type' (line 14)
    stypy_return_type_9752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_9752)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_known_types'
    return stypy_return_type_9752

# Assigning a type to the variable 'get_known_types' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'get_known_types', get_known_types)

@norecursion
def add_known_type(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'known_python_type_typename_samplevalues' (line 21)
    known_python_type_typename_samplevalues_9753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 60), 'known_python_type_typename_samplevalues')
    defaults = [known_python_type_typename_samplevalues_9753]
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

    str_9754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, (-1)), 'str', '\n    Allows to add a type to the list of known types in the known_python_type_typename_samplevalues list (used by\n    default)\n    ')
    
    # Assigning a Tuple to a Subscript (line 26):
    
    # Obtaining an instance of the builtin type 'tuple' (line 26)
    tuple_9755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 26)
    # Adding element type (line 26)
    # Getting the type of 'type_name' (line 26)
    type_name_9756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 25), 'type_name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 25), tuple_9755, type_name_9756)
    # Adding element type (line 26)
    # Getting the type of 'type_value' (line 26)
    type_value_9757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 36), 'type_value')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 25), tuple_9755, type_value_9757)
    
    # Getting the type of 'type_table' (line 26)
    type_table_9758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'type_table')
    # Getting the type of 'type_' (line 26)
    type__9759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 15), 'type_')
    # Storing an element on a container (line 26)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 4), type_table_9758, (type__9759, tuple_9755))
    
    # ################# End of 'add_known_type(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'add_known_type' in the type store
    # Getting the type of 'stypy_return_type' (line 21)
    stypy_return_type_9760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_9760)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'add_known_type'
    return stypy_return_type_9760

# Assigning a type to the variable 'add_known_type' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'add_known_type', add_known_type)

@norecursion
def remove_known_type(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'known_python_type_typename_samplevalues' (line 29)
    known_python_type_typename_samplevalues_9761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 40), 'known_python_type_typename_samplevalues')
    defaults = [known_python_type_typename_samplevalues_9761]
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

    str_9762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, (-1)), 'str', '\n    Delete a type to the list of known types in the known_python_type_typename_samplevalues list (used by default)\n    ')
    # Deleting a member
    # Getting the type of 'type_table' (line 33)
    type_table_9763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'type_table')
    
    # Obtaining the type of the subscript
    # Getting the type of 'type_' (line 33)
    type__9764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 19), 'type_')
    # Getting the type of 'type_table' (line 33)
    type_table_9765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'type_table')
    # Obtaining the member '__getitem__' of a type (line 33)
    getitem___9766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), type_table_9765, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 33)
    subscript_call_result_9767 = invoke(stypy.reporting.localization.Localization(__file__, 33, 8), getitem___9766, type__9764)
    
    del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 4), type_table_9763, subscript_call_result_9767)
    
    # ################# End of 'remove_known_type(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'remove_known_type' in the type store
    # Getting the type of 'stypy_return_type' (line 29)
    stypy_return_type_9768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_9768)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'remove_known_type'
    return stypy_return_type_9768

# Assigning a type to the variable 'remove_known_type' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'remove_known_type', remove_known_type)

@norecursion
def is_known_type(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'known_python_type_typename_samplevalues' (line 36)
    known_python_type_typename_samplevalues_9769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 36), 'known_python_type_typename_samplevalues')
    defaults = [known_python_type_typename_samplevalues_9769]
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

    str_9770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, (-1)), 'str', '\n    Determines if a type or a type name is in the list of known types in the known_python_type_typename_samplevalues\n    list (used by default)\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 42)
    # Getting the type of 'str' (line 42)
    str_9771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 25), 'str')
    # Getting the type of 'type_' (line 42)
    type__9772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 18), 'type_')
    
    (may_be_9773, more_types_in_union_9774) = may_be_subtype(str_9771, type__9772)

    if may_be_9773:

        if more_types_in_union_9774:
            # Runtime conditional SSA (line 42)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'type_' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'type_', remove_not_subtype_from_union(type__9772, str))
        
        # Getting the type of 'type_table' (line 43)
        type_table_9775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 27), 'type_table')
        # Assigning a type to the variable 'type_table_9775' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'type_table_9775', type_table_9775)
        # Testing if the for loop is going to be iterated (line 43)
        # Testing the type of a for loop iterable (line 43)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 43, 8), type_table_9775)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 43, 8), type_table_9775):
            # Getting the type of the for loop variable (line 43)
            for_loop_var_9776 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 43, 8), type_table_9775)
            # Assigning a type to the variable 'table_entry' (line 43)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'table_entry', for_loop_var_9776)
            # SSA begins for a for statement (line 43)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'type_' (line 44)
            type__9777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 15), 'type_')
            
            # Obtaining the type of the subscript
            int_9778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 48), 'int')
            
            # Obtaining the type of the subscript
            # Getting the type of 'table_entry' (line 44)
            table_entry_9779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 35), 'table_entry')
            # Getting the type of 'type_table' (line 44)
            type_table_9780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 24), 'type_table')
            # Obtaining the member '__getitem__' of a type (line 44)
            getitem___9781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 24), type_table_9780, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 44)
            subscript_call_result_9782 = invoke(stypy.reporting.localization.Localization(__file__, 44, 24), getitem___9781, table_entry_9779)
            
            # Obtaining the member '__getitem__' of a type (line 44)
            getitem___9783 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 24), subscript_call_result_9782, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 44)
            subscript_call_result_9784 = invoke(stypy.reporting.localization.Localization(__file__, 44, 24), getitem___9783, int_9778)
            
            # Applying the binary operator '==' (line 44)
            result_eq_9785 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 15), '==', type__9777, subscript_call_result_9784)
            
            # Testing if the type of an if condition is none (line 44)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 44, 12), result_eq_9785):
                pass
            else:
                
                # Testing the type of an if condition (line 44)
                if_condition_9786 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 44, 12), result_eq_9785)
                # Assigning a type to the variable 'if_condition_9786' (line 44)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'if_condition_9786', if_condition_9786)
                # SSA begins for if statement (line 44)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'True' (line 45)
                True_9787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 23), 'True')
                # Assigning a type to the variable 'stypy_return_type' (line 45)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 16), 'stypy_return_type', True_9787)
                # SSA join for if statement (line 44)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'False' (line 47)
        False_9788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'stypy_return_type', False_9788)

        if more_types_in_union_9774:
            # Runtime conditional SSA for else branch (line 42)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_9773) or more_types_in_union_9774):
        # Assigning a type to the variable 'type_' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'type_', remove_subtype_from_union(type__9772, str))
        
        # Getting the type of 'type_' (line 49)
        type__9789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 15), 'type_')
        # Getting the type of 'type_table' (line 49)
        type_table_9790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 24), 'type_table')
        # Applying the binary operator 'in' (line 49)
        result_contains_9791 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 15), 'in', type__9789, type_table_9790)
        
        # Assigning a type to the variable 'stypy_return_type' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'stypy_return_type', result_contains_9791)

        if (may_be_9773 and more_types_in_union_9774):
            # SSA join for if statement (line 42)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of 'is_known_type(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_known_type' in the type store
    # Getting the type of 'stypy_return_type' (line 36)
    stypy_return_type_9792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_9792)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_known_type'
    return stypy_return_type_9792

# Assigning a type to the variable 'is_known_type' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'is_known_type', is_known_type)

@norecursion
def get_known_types_and_values(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'known_python_type_typename_samplevalues' (line 52)
    known_python_type_typename_samplevalues_9793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 42), 'known_python_type_typename_samplevalues')
    defaults = [known_python_type_typename_samplevalues_9793]
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

    str_9794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, (-1)), 'str', '\n    Obtains a list of the library known types and a sample value for each one from the\n    known_python_type_typename_samplevalues  list (used by default)\n    ')
    
    # Assigning a Call to a Name (line 57):
    
    # Call to items(...): (line 57)
    # Processing the call keyword arguments (line 57)
    kwargs_9797 = {}
    # Getting the type of 'type_table' (line 57)
    type_table_9795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'type_table', False)
    # Obtaining the member 'items' of a type (line 57)
    items_9796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 12), type_table_9795, 'items')
    # Calling items(args, kwargs) (line 57)
    items_call_result_9798 = invoke(stypy.reporting.localization.Localization(__file__, 57, 12), items_9796, *[], **kwargs_9797)
    
    # Assigning a type to the variable 'list_' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'list_', items_call_result_9798)
    
    # Assigning a List to a Name (line 58):
    
    # Obtaining an instance of the builtin type 'list' (line 58)
    list_9799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 58)
    
    # Assigning a type to the variable 'ret_list' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'ret_list', list_9799)
    
    # Getting the type of 'list_' (line 59)
    list__9800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'list_')
    # Assigning a type to the variable 'list__9800' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'list__9800', list__9800)
    # Testing if the for loop is going to be iterated (line 59)
    # Testing the type of a for loop iterable (line 59)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 59, 4), list__9800)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 59, 4), list__9800):
        # Getting the type of the for loop variable (line 59)
        for_loop_var_9801 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 59, 4), list__9800)
        # Assigning a type to the variable 'elem' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'elem', for_loop_var_9801)
        # SSA begins for a for statement (line 59)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 60)
        # Processing the call arguments (line 60)
        
        # Obtaining an instance of the builtin type 'tuple' (line 60)
        tuple_9804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 60)
        # Adding element type (line 60)
        
        # Obtaining the type of the subscript
        int_9805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 30), 'int')
        # Getting the type of 'elem' (line 60)
        elem_9806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 25), 'elem', False)
        # Obtaining the member '__getitem__' of a type (line 60)
        getitem___9807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 25), elem_9806, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 60)
        subscript_call_result_9808 = invoke(stypy.reporting.localization.Localization(__file__, 60, 25), getitem___9807, int_9805)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 25), tuple_9804, subscript_call_result_9808)
        # Adding element type (line 60)
        
        # Obtaining the type of the subscript
        int_9809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 42), 'int')
        
        # Obtaining the type of the subscript
        int_9810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 39), 'int')
        # Getting the type of 'elem' (line 60)
        elem_9811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 34), 'elem', False)
        # Obtaining the member '__getitem__' of a type (line 60)
        getitem___9812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 34), elem_9811, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 60)
        subscript_call_result_9813 = invoke(stypy.reporting.localization.Localization(__file__, 60, 34), getitem___9812, int_9810)
        
        # Obtaining the member '__getitem__' of a type (line 60)
        getitem___9814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 34), subscript_call_result_9813, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 60)
        subscript_call_result_9815 = invoke(stypy.reporting.localization.Localization(__file__, 60, 34), getitem___9814, int_9809)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 25), tuple_9804, subscript_call_result_9815)
        
        # Processing the call keyword arguments (line 60)
        kwargs_9816 = {}
        # Getting the type of 'ret_list' (line 60)
        ret_list_9802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'ret_list', False)
        # Obtaining the member 'append' of a type (line 60)
        append_9803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), ret_list_9802, 'append')
        # Calling append(args, kwargs) (line 60)
        append_call_result_9817 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), append_9803, *[tuple_9804], **kwargs_9816)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'ret_list' (line 62)
    ret_list_9818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 11), 'ret_list')
    # Assigning a type to the variable 'stypy_return_type' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'stypy_return_type', ret_list_9818)
    
    # ################# End of 'get_known_types_and_values(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_known_types_and_values' in the type store
    # Getting the type of 'stypy_return_type' (line 52)
    stypy_return_type_9819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_9819)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_known_types_and_values'
    return stypy_return_type_9819

# Assigning a type to the variable 'get_known_types_and_values' (line 52)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'get_known_types_and_values', get_known_types_and_values)

@norecursion
def get_type_name(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'known_python_type_typename_samplevalues' (line 65)
    known_python_type_typename_samplevalues_9820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 36), 'known_python_type_typename_samplevalues')
    defaults = [known_python_type_typename_samplevalues_9820]
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

    str_9821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, (-1)), 'str', '\n    Gets the type name of the passed type as defined in the known_python_type_typename_samplevalues\n    list (used by default)\n    ')
    
    # Getting the type of 'type_' (line 70)
    type__9822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 7), 'type_')
    # Getting the type of 'types' (line 70)
    types_9823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'types')
    # Obtaining the member 'NotImplementedType' of a type (line 70)
    NotImplementedType_9824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 16), types_9823, 'NotImplementedType')
    # Applying the binary operator '==' (line 70)
    result_eq_9825 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 7), '==', type__9822, NotImplementedType_9824)
    
    # Testing if the type of an if condition is none (line 70)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 70, 4), result_eq_9825):
        pass
    else:
        
        # Testing the type of an if condition (line 70)
        if_condition_9826 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 70, 4), result_eq_9825)
        # Assigning a type to the variable 'if_condition_9826' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'if_condition_9826', if_condition_9826)
        # SSA begins for if statement (line 70)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_9827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 15), 'str', 'types.NotImplementedType')
        # Assigning a type to the variable 'stypy_return_type' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'stypy_return_type', str_9827)
        # SSA join for if statement (line 70)
        module_type_store = module_type_store.join_ssa_context()
        

    
    
    # SSA begins for try-except statement (line 73)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Obtaining the type of the subscript
    int_9828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 33), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'type_' (line 74)
    type__9829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 26), 'type_')
    # Getting the type of 'type_table' (line 74)
    type_table_9830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 15), 'type_table')
    # Obtaining the member '__getitem__' of a type (line 74)
    getitem___9831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 15), type_table_9830, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 74)
    subscript_call_result_9832 = invoke(stypy.reporting.localization.Localization(__file__, 74, 15), getitem___9831, type__9829)
    
    # Obtaining the member '__getitem__' of a type (line 74)
    getitem___9833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 15), subscript_call_result_9832, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 74)
    subscript_call_result_9834 = invoke(stypy.reporting.localization.Localization(__file__, 74, 15), getitem___9833, int_9828)
    
    # Assigning a type to the variable 'stypy_return_type' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'stypy_return_type', subscript_call_result_9834)
    # SSA branch for the except part of a try statement (line 73)
    # SSA branch for the except 'Tuple' branch of a try statement (line 73)
    module_type_store.open_ssa_branch('except')
    
    # Getting the type of 'type_' (line 76)
    type__9835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 11), 'type_')
    # Getting the type of '__builtins__' (line 76)
    builtins___9836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 20), '__builtins__')
    # Applying the binary operator 'is' (line 76)
    result_is__9837 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 11), 'is', type__9835, builtins___9836)
    
    # Testing if the type of an if condition is none (line 76)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 76, 8), result_is__9837):
        pass
    else:
        
        # Testing the type of an if condition (line 76)
        if_condition_9838 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 8), result_is__9837)
        # Assigning a type to the variable 'if_condition_9838' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'if_condition_9838', if_condition_9838)
        # SSA begins for if statement (line 76)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_9839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 19), 'str', '__builtins__')
        # Assigning a type to the variable 'stypy_return_type' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'stypy_return_type', str_9839)
        # SSA join for if statement (line 76)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Type idiom detected: calculating its left and rigth part (line 79)
    str_9840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 26), 'str', '__name__')
    # Getting the type of 'type_' (line 79)
    type__9841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 19), 'type_')
    
    (may_be_9842, more_types_in_union_9843) = may_provide_member(str_9840, type__9841)

    if may_be_9842:

        if more_types_in_union_9843:
            # Runtime conditional SSA (line 79)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'type_' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'type_', remove_not_member_provider_from_union(type__9841, '__name__'))
        
        # Call to hasattr(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'ExtraTypeDefinitions' (line 80)
        ExtraTypeDefinitions_9845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 23), 'ExtraTypeDefinitions', False)
        # Getting the type of 'type_' (line 80)
        type__9846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 45), 'type_', False)
        # Obtaining the member '__name__' of a type (line 80)
        name___9847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 45), type__9846, '__name__')
        # Processing the call keyword arguments (line 80)
        kwargs_9848 = {}
        # Getting the type of 'hasattr' (line 80)
        hasattr_9844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 15), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 80)
        hasattr_call_result_9849 = invoke(stypy.reporting.localization.Localization(__file__, 80, 15), hasattr_9844, *[ExtraTypeDefinitions_9845, name___9847], **kwargs_9848)
        
        # Testing if the type of an if condition is none (line 80)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 80, 12), hasattr_call_result_9849):
            pass
        else:
            
            # Testing the type of an if condition (line 80)
            if_condition_9850 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 12), hasattr_call_result_9849)
            # Assigning a type to the variable 'if_condition_9850' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'if_condition_9850', if_condition_9850)
            # SSA begins for if statement (line 80)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            str_9851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 23), 'str', 'ExtraTypeDefinitions.')
            # Getting the type of 'type_' (line 81)
            type__9852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 49), 'type_')
            # Obtaining the member '__name__' of a type (line 81)
            name___9853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 49), type__9852, '__name__')
            # Applying the binary operator '+' (line 81)
            result_add_9854 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 23), '+', str_9851, name___9853)
            
            # Assigning a type to the variable 'stypy_return_type' (line 81)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'stypy_return_type', result_add_9854)
            # SSA join for if statement (line 80)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'type_' (line 83)
        type__9855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 15), 'type_')
        # Obtaining the member '__name__' of a type (line 83)
        name___9856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 15), type__9855, '__name__')
        str_9857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 33), 'str', 'iterator')
        # Applying the binary operator '==' (line 83)
        result_eq_9858 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 15), '==', name___9856, str_9857)
        
        # Testing if the type of an if condition is none (line 83)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 83, 12), result_eq_9858):
            pass
        else:
            
            # Testing the type of an if condition (line 83)
            if_condition_9859 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 12), result_eq_9858)
            # Assigning a type to the variable 'if_condition_9859' (line 83)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'if_condition_9859', if_condition_9859)
            # SSA begins for if statement (line 83)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            str_9860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 23), 'str', 'ExtraTypeDefinitions.listiterator')
            # Assigning a type to the variable 'stypy_return_type' (line 84)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 16), 'stypy_return_type', str_9860)
            # SSA join for if statement (line 83)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'type_' (line 86)
        type__9861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 19), 'type_')
        # Obtaining the member '__name__' of a type (line 86)
        name___9862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 19), type__9861, '__name__')
        # Assigning a type to the variable 'stypy_return_type' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'stypy_return_type', name___9862)

        if more_types_in_union_9843:
            # Runtime conditional SSA for else branch (line 79)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_9842) or more_types_in_union_9843):
        # Assigning a type to the variable 'type_' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'type_', remove_member_provider_from_union(type__9841, '__name__'))
        
        # Call to type(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'type_' (line 88)
        type__9864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 24), 'type_', False)
        # Processing the call keyword arguments (line 88)
        kwargs_9865 = {}
        # Getting the type of 'type' (line 88)
        type_9863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 19), 'type', False)
        # Calling type(args, kwargs) (line 88)
        type_call_result_9866 = invoke(stypy.reporting.localization.Localization(__file__, 88, 19), type_9863, *[type__9864], **kwargs_9865)
        
        # Obtaining the member '__name__' of a type (line 88)
        name___9867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 19), type_call_result_9866, '__name__')
        # Assigning a type to the variable 'stypy_return_type' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'stypy_return_type', name___9867)

        if (may_be_9842 and more_types_in_union_9843):
            # SSA join for if statement (line 79)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for try-except statement (line 73)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'get_type_name(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_type_name' in the type store
    # Getting the type of 'stypy_return_type' (line 65)
    stypy_return_type_9868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_9868)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_type_name'
    return stypy_return_type_9868

# Assigning a type to the variable 'get_type_name' (line 65)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 0), 'get_type_name', get_type_name)

@norecursion
def get_type_sample_value(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'known_python_type_typename_samplevalues' (line 91)
    known_python_type_typename_samplevalues_9869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 44), 'known_python_type_typename_samplevalues')
    defaults = [known_python_type_typename_samplevalues_9869]
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

    str_9870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, (-1)), 'str', '\n    Gets a sample value of the passed type from the known_python_type_typename_samplevalues\n    list (used by default)\n    ')
    
    # Obtaining the type of the subscript
    int_9871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 29), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'type_' (line 96)
    type__9872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 22), 'type_')
    # Getting the type of 'type_table' (line 96)
    type_table_9873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 11), 'type_table')
    # Obtaining the member '__getitem__' of a type (line 96)
    getitem___9874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 11), type_table_9873, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 96)
    subscript_call_result_9875 = invoke(stypy.reporting.localization.Localization(__file__, 96, 11), getitem___9874, type__9872)
    
    # Obtaining the member '__getitem__' of a type (line 96)
    getitem___9876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 11), subscript_call_result_9875, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 96)
    subscript_call_result_9877 = invoke(stypy.reporting.localization.Localization(__file__, 96, 11), getitem___9876, int_9871)
    
    # Assigning a type to the variable 'stypy_return_type' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'stypy_return_type', subscript_call_result_9877)
    
    # ################# End of 'get_type_sample_value(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_type_sample_value' in the type store
    # Getting the type of 'stypy_return_type' (line 91)
    stypy_return_type_9878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_9878)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_type_sample_value'
    return stypy_return_type_9878

# Assigning a type to the variable 'get_type_sample_value' (line 91)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'get_type_sample_value', get_type_sample_value)

@norecursion
def get_type_from_name(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'known_python_type_typename_samplevalues' (line 99)
    known_python_type_typename_samplevalues_9879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 40), 'known_python_type_typename_samplevalues')
    defaults = [known_python_type_typename_samplevalues_9879]
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

    str_9880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, (-1)), 'str', '\n    Gets the type object of the passed type name from the known_python_type_typename_samplevalues\n    list (used by default)\n    ')
    
    str_9881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 7), 'str', 'NotImplementedType')
    # Getting the type of 'name' (line 104)
    name_9882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 31), 'name')
    # Applying the binary operator 'in' (line 104)
    result_contains_9883 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 7), 'in', str_9881, name_9882)
    
    # Testing if the type of an if condition is none (line 104)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 104, 4), result_contains_9883):
        pass
    else:
        
        # Testing the type of an if condition (line 104)
        if_condition_9884 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 104, 4), result_contains_9883)
        # Assigning a type to the variable 'if_condition_9884' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'if_condition_9884', if_condition_9884)
        # SSA begins for if statement (line 104)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_9885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 15), 'str', 'types.NotImplementedType')
        # Assigning a type to the variable 'stypy_return_type' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'stypy_return_type', str_9885)
        # SSA join for if statement (line 104)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 107):
    
    # Call to keys(...): (line 107)
    # Processing the call keyword arguments (line 107)
    kwargs_9888 = {}
    # Getting the type of 'type_table' (line 107)
    type_table_9886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 11), 'type_table', False)
    # Obtaining the member 'keys' of a type (line 107)
    keys_9887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 11), type_table_9886, 'keys')
    # Calling keys(args, kwargs) (line 107)
    keys_call_result_9889 = invoke(stypy.reporting.localization.Localization(__file__, 107, 11), keys_9887, *[], **kwargs_9888)
    
    # Assigning a type to the variable 'keys' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'keys', keys_call_result_9889)
    
    # Getting the type of 'keys' (line 108)
    keys_9890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 15), 'keys')
    # Assigning a type to the variable 'keys_9890' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'keys_9890', keys_9890)
    # Testing if the for loop is going to be iterated (line 108)
    # Testing the type of a for loop iterable (line 108)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 108, 4), keys_9890)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 108, 4), keys_9890):
        # Getting the type of the for loop variable (line 108)
        for_loop_var_9891 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 108, 4), keys_9890)
        # Assigning a type to the variable 'key' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'key', for_loop_var_9891)
        # SSA begins for a for statement (line 108)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'name' (line 109)
        name_9892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 11), 'name')
        
        # Obtaining the type of the subscript
        int_9893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 35), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'key' (line 109)
        key_9894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 30), 'key')
        # Getting the type of 'type_table' (line 109)
        type_table_9895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'type_table')
        # Obtaining the member '__getitem__' of a type (line 109)
        getitem___9896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), type_table_9895, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 109)
        subscript_call_result_9897 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), getitem___9896, key_9894)
        
        # Obtaining the member '__getitem__' of a type (line 109)
        getitem___9898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), subscript_call_result_9897, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 109)
        subscript_call_result_9899 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), getitem___9898, int_9893)
        
        # Applying the binary operator '==' (line 109)
        result_eq_9900 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 11), '==', name_9892, subscript_call_result_9899)
        
        # Testing if the type of an if condition is none (line 109)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 109, 8), result_eq_9900):
            pass
        else:
            
            # Testing the type of an if condition (line 109)
            if_condition_9901 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 109, 8), result_eq_9900)
            # Assigning a type to the variable 'if_condition_9901' (line 109)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'if_condition_9901', if_condition_9901)
            # SSA begins for if statement (line 109)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'key' (line 110)
            key_9902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 19), 'key')
            # Assigning a type to the variable 'stypy_return_type' (line 110)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'stypy_return_type', key_9902)
            # SSA join for if statement (line 109)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'get_type_from_name(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_type_from_name' in the type store
    # Getting the type of 'stypy_return_type' (line 99)
    stypy_return_type_9903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_9903)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_type_from_name'
    return stypy_return_type_9903

# Assigning a type to the variable 'get_type_from_name' (line 99)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 0), 'get_type_from_name', get_type_from_name)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
