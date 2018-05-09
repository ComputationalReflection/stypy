
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from stypy_copy import type_store_copy
2: from stypy_copy.errors_copy.type_error_copy import TypeError
3: from stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy.stypy_functions_copy import default_lambda_var_name
4: 
5: '''
6: Several functions to help printing elements with a readable style on error reports
7: '''
8: 
9: # These variables are used by stypy, therefore they should not be printed.
10: private_variable_names = ["__temp_tuple_assignment", "__temp_list_assignment",
11:                           "__temp_lambda", "__temp_call_assignment"]
12: 
13: 
14: def format_function_name(fname):
15:     '''
16:     Prints a function name, considering lambda functions.
17:     :param fname: Function name
18:     :return: Proper function name
19:     '''
20:     if default_lambda_var_name in fname:
21:         return "lambda function"
22:     return fname
23: 
24: 
25: def is_private_variable_name(var_name):
26:     '''
27:     Determines if a variable is a stypy private variable
28:     :param var_name: Variable name
29:     :return: bool
30:     '''
31:     for private_name in private_variable_names:
32:         if private_name in var_name:
33:             return True
34: 
35:     return False
36: 
37: 
38: def get_type_str(type_):
39:     '''
40:     Get the abbreviated str representation of a type for printing friendly error messages
41:     :param type_: Type
42:     :return: str
43:     '''
44:     if isinstance(type_, TypeError):
45:         return "TypeError"
46: 
47:     # Is this a type store? Then it is a non-python library module
48:     if type(type_) == type_store.typestore.TypeStore:
49:         return "External module '" + type_.program_name + "'"
50:     return str(type_)
51: 
52: 
53: def get_param_position(source_code, param_number):
54:     '''
55:     Get the offset of a parameter within a source code line that specify a method header. This is used to mark
56:     parameters with type errors when reporting them.
57: 
58:     :param source_code: Source code (method header)
59:     :param param_number: Number of parameter
60:     :return: Offset of the parameter in the source line
61:     '''
62:     try:
63:         split_str = source_code.split(',')
64:         if param_number >= len(split_str):
65:             return 0
66: 
67:         if param_number == 0:
68:             name_and_first = split_str[0].split('(')
69:             offset = len(name_and_first[0]) + 1
70: 
71:             blank_offset = 0
72:             for car in name_and_first[1]:
73:                 if car == " ":
74:                     blank_offset += 1
75:         else:
76:             offset = 0
77:             for i in range(param_number):
78:                 offset += len(split_str[i]) + 1  # The comma also counts
79: 
80:             blank_offset = 0
81:             for car in split_str[param_number]:
82:                 if car == " ":
83:                     blank_offset += 1
84: 
85:         return offset + blank_offset
86:     except:
87:         return -1
88: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from stypy_copy import type_store_copy' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/reporting_copy/')
import_14131 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy')

if (type(import_14131) is not StypyTypeError):

    if (import_14131 != 'pyd_module'):
        __import__(import_14131)
        sys_modules_14132 = sys.modules[import_14131]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy', sys_modules_14132.module_type_store, module_type_store, ['type_store_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_14132, sys_modules_14132.module_type_store, module_type_store)
    else:
        from stypy_copy import type_store_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy', None, module_type_store, ['type_store_copy'], [type_store_copy])

else:
    # Assigning a type to the variable 'stypy_copy' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy', import_14131)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/reporting_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'from stypy_copy.errors_copy.type_error_copy import TypeError' statement (line 2)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/reporting_copy/')
import_14133 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_copy.errors_copy.type_error_copy')

if (type(import_14133) is not StypyTypeError):

    if (import_14133 != 'pyd_module'):
        __import__(import_14133)
        sys_modules_14134 = sys.modules[import_14133]
        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_copy.errors_copy.type_error_copy', sys_modules_14134.module_type_store, module_type_store, ['TypeError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 2, 0), __file__, sys_modules_14134, sys_modules_14134.module_type_store, module_type_store)
    else:
        from stypy_copy.errors_copy.type_error_copy import TypeError

        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_copy.errors_copy.type_error_copy', None, module_type_store, ['TypeError'], [TypeError])

else:
    # Assigning a type to the variable 'stypy_copy.errors_copy.type_error_copy' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'stypy_copy.errors_copy.type_error_copy', import_14133)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/reporting_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy.stypy_functions_copy import default_lambda_var_name' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/reporting_copy/')
import_14135 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy.stypy_functions_copy')

if (type(import_14135) is not StypyTypeError):

    if (import_14135 != 'pyd_module'):
        __import__(import_14135)
        sys_modules_14136 = sys.modules[import_14135]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy.stypy_functions_copy', sys_modules_14136.module_type_store, module_type_store, ['default_lambda_var_name'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_14136, sys_modules_14136.module_type_store, module_type_store)
    else:
        from stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy.stypy_functions_copy import default_lambda_var_name

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy.stypy_functions_copy', None, module_type_store, ['default_lambda_var_name'], [default_lambda_var_name])

else:
    # Assigning a type to the variable 'stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy.stypy_functions_copy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy.stypy_functions_copy', import_14135)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/reporting_copy/')

str_14137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, (-1)), 'str', '\nSeveral functions to help printing elements with a readable style on error reports\n')

# Assigning a List to a Name (line 10):

# Obtaining an instance of the builtin type 'list' (line 10)
list_14138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 10)
# Adding element type (line 10)
str_14139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 26), 'str', '__temp_tuple_assignment')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 25), list_14138, str_14139)
# Adding element type (line 10)
str_14140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 53), 'str', '__temp_list_assignment')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 25), list_14138, str_14140)
# Adding element type (line 10)
str_14141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 26), 'str', '__temp_lambda')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 25), list_14138, str_14141)
# Adding element type (line 10)
str_14142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 43), 'str', '__temp_call_assignment')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 25), list_14138, str_14142)

# Assigning a type to the variable 'private_variable_names' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'private_variable_names', list_14138)

@norecursion
def format_function_name(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'format_function_name'
    module_type_store = module_type_store.open_function_context('format_function_name', 14, 0, False)
    
    # Passed parameters checking function
    format_function_name.stypy_localization = localization
    format_function_name.stypy_type_of_self = None
    format_function_name.stypy_type_store = module_type_store
    format_function_name.stypy_function_name = 'format_function_name'
    format_function_name.stypy_param_names_list = ['fname']
    format_function_name.stypy_varargs_param_name = None
    format_function_name.stypy_kwargs_param_name = None
    format_function_name.stypy_call_defaults = defaults
    format_function_name.stypy_call_varargs = varargs
    format_function_name.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'format_function_name', ['fname'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'format_function_name', localization, ['fname'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'format_function_name(...)' code ##################

    str_14143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, (-1)), 'str', '\n    Prints a function name, considering lambda functions.\n    :param fname: Function name\n    :return: Proper function name\n    ')
    
    # Getting the type of 'default_lambda_var_name' (line 20)
    default_lambda_var_name_14144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 7), 'default_lambda_var_name')
    # Getting the type of 'fname' (line 20)
    fname_14145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 34), 'fname')
    # Applying the binary operator 'in' (line 20)
    result_contains_14146 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 7), 'in', default_lambda_var_name_14144, fname_14145)
    
    # Testing if the type of an if condition is none (line 20)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 20, 4), result_contains_14146):
        pass
    else:
        
        # Testing the type of an if condition (line 20)
        if_condition_14147 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 20, 4), result_contains_14146)
        # Assigning a type to the variable 'if_condition_14147' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'if_condition_14147', if_condition_14147)
        # SSA begins for if statement (line 20)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_14148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 15), 'str', 'lambda function')
        # Assigning a type to the variable 'stypy_return_type' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'stypy_return_type', str_14148)
        # SSA join for if statement (line 20)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'fname' (line 22)
    fname_14149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 11), 'fname')
    # Assigning a type to the variable 'stypy_return_type' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'stypy_return_type', fname_14149)
    
    # ################# End of 'format_function_name(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'format_function_name' in the type store
    # Getting the type of 'stypy_return_type' (line 14)
    stypy_return_type_14150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14150)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'format_function_name'
    return stypy_return_type_14150

# Assigning a type to the variable 'format_function_name' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'format_function_name', format_function_name)

@norecursion
def is_private_variable_name(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'is_private_variable_name'
    module_type_store = module_type_store.open_function_context('is_private_variable_name', 25, 0, False)
    
    # Passed parameters checking function
    is_private_variable_name.stypy_localization = localization
    is_private_variable_name.stypy_type_of_self = None
    is_private_variable_name.stypy_type_store = module_type_store
    is_private_variable_name.stypy_function_name = 'is_private_variable_name'
    is_private_variable_name.stypy_param_names_list = ['var_name']
    is_private_variable_name.stypy_varargs_param_name = None
    is_private_variable_name.stypy_kwargs_param_name = None
    is_private_variable_name.stypy_call_defaults = defaults
    is_private_variable_name.stypy_call_varargs = varargs
    is_private_variable_name.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'is_private_variable_name', ['var_name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'is_private_variable_name', localization, ['var_name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'is_private_variable_name(...)' code ##################

    str_14151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, (-1)), 'str', '\n    Determines if a variable is a stypy private variable\n    :param var_name: Variable name\n    :return: bool\n    ')
    
    # Getting the type of 'private_variable_names' (line 31)
    private_variable_names_14152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 24), 'private_variable_names')
    # Assigning a type to the variable 'private_variable_names_14152' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'private_variable_names_14152', private_variable_names_14152)
    # Testing if the for loop is going to be iterated (line 31)
    # Testing the type of a for loop iterable (line 31)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 31, 4), private_variable_names_14152)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 31, 4), private_variable_names_14152):
        # Getting the type of the for loop variable (line 31)
        for_loop_var_14153 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 31, 4), private_variable_names_14152)
        # Assigning a type to the variable 'private_name' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'private_name', for_loop_var_14153)
        # SSA begins for a for statement (line 31)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'private_name' (line 32)
        private_name_14154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 11), 'private_name')
        # Getting the type of 'var_name' (line 32)
        var_name_14155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 27), 'var_name')
        # Applying the binary operator 'in' (line 32)
        result_contains_14156 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 11), 'in', private_name_14154, var_name_14155)
        
        # Testing if the type of an if condition is none (line 32)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 32, 8), result_contains_14156):
            pass
        else:
            
            # Testing the type of an if condition (line 32)
            if_condition_14157 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 32, 8), result_contains_14156)
            # Assigning a type to the variable 'if_condition_14157' (line 32)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'if_condition_14157', if_condition_14157)
            # SSA begins for if statement (line 32)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'True' (line 33)
            True_14158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 19), 'True')
            # Assigning a type to the variable 'stypy_return_type' (line 33)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'stypy_return_type', True_14158)
            # SSA join for if statement (line 32)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'False' (line 35)
    False_14159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 11), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type', False_14159)
    
    # ################# End of 'is_private_variable_name(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_private_variable_name' in the type store
    # Getting the type of 'stypy_return_type' (line 25)
    stypy_return_type_14160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14160)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_private_variable_name'
    return stypy_return_type_14160

# Assigning a type to the variable 'is_private_variable_name' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'is_private_variable_name', is_private_variable_name)

@norecursion
def get_type_str(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_type_str'
    module_type_store = module_type_store.open_function_context('get_type_str', 38, 0, False)
    
    # Passed parameters checking function
    get_type_str.stypy_localization = localization
    get_type_str.stypy_type_of_self = None
    get_type_str.stypy_type_store = module_type_store
    get_type_str.stypy_function_name = 'get_type_str'
    get_type_str.stypy_param_names_list = ['type_']
    get_type_str.stypy_varargs_param_name = None
    get_type_str.stypy_kwargs_param_name = None
    get_type_str.stypy_call_defaults = defaults
    get_type_str.stypy_call_varargs = varargs
    get_type_str.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_type_str', ['type_'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_type_str', localization, ['type_'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_type_str(...)' code ##################

    str_14161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, (-1)), 'str', '\n    Get the abbreviated str representation of a type for printing friendly error messages\n    :param type_: Type\n    :return: str\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 44)
    # Getting the type of 'TypeError' (line 44)
    TypeError_14162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 25), 'TypeError')
    # Getting the type of 'type_' (line 44)
    type__14163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 18), 'type_')
    
    (may_be_14164, more_types_in_union_14165) = may_be_subtype(TypeError_14162, type__14163)

    if may_be_14164:

        if more_types_in_union_14165:
            # Runtime conditional SSA (line 44)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'type_' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'type_', remove_not_subtype_from_union(type__14163, TypeError))
        str_14166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 15), 'str', 'TypeError')
        # Assigning a type to the variable 'stypy_return_type' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'stypy_return_type', str_14166)

        if more_types_in_union_14165:
            # SSA join for if statement (line 44)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Call to type(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'type_' (line 48)
    type__14168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'type_', False)
    # Processing the call keyword arguments (line 48)
    kwargs_14169 = {}
    # Getting the type of 'type' (line 48)
    type_14167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 7), 'type', False)
    # Calling type(args, kwargs) (line 48)
    type_call_result_14170 = invoke(stypy.reporting.localization.Localization(__file__, 48, 7), type_14167, *[type__14168], **kwargs_14169)
    
    # Getting the type of 'type_store' (line 48)
    type_store_14171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 22), 'type_store')
    # Obtaining the member 'typestore' of a type (line 48)
    typestore_14172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 22), type_store_14171, 'typestore')
    # Obtaining the member 'TypeStore' of a type (line 48)
    TypeStore_14173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 22), typestore_14172, 'TypeStore')
    # Applying the binary operator '==' (line 48)
    result_eq_14174 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 7), '==', type_call_result_14170, TypeStore_14173)
    
    # Testing if the type of an if condition is none (line 48)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 48, 4), result_eq_14174):
        pass
    else:
        
        # Testing the type of an if condition (line 48)
        if_condition_14175 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 48, 4), result_eq_14174)
        # Assigning a type to the variable 'if_condition_14175' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'if_condition_14175', if_condition_14175)
        # SSA begins for if statement (line 48)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_14176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 15), 'str', "External module '")
        # Getting the type of 'type_' (line 49)
        type__14177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 37), 'type_')
        # Obtaining the member 'program_name' of a type (line 49)
        program_name_14178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 37), type__14177, 'program_name')
        # Applying the binary operator '+' (line 49)
        result_add_14179 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 15), '+', str_14176, program_name_14178)
        
        str_14180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 58), 'str', "'")
        # Applying the binary operator '+' (line 49)
        result_add_14181 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 56), '+', result_add_14179, str_14180)
        
        # Assigning a type to the variable 'stypy_return_type' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'stypy_return_type', result_add_14181)
        # SSA join for if statement (line 48)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to str(...): (line 50)
    # Processing the call arguments (line 50)
    # Getting the type of 'type_' (line 50)
    type__14183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 15), 'type_', False)
    # Processing the call keyword arguments (line 50)
    kwargs_14184 = {}
    # Getting the type of 'str' (line 50)
    str_14182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 11), 'str', False)
    # Calling str(args, kwargs) (line 50)
    str_call_result_14185 = invoke(stypy.reporting.localization.Localization(__file__, 50, 11), str_14182, *[type__14183], **kwargs_14184)
    
    # Assigning a type to the variable 'stypy_return_type' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'stypy_return_type', str_call_result_14185)
    
    # ################# End of 'get_type_str(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_type_str' in the type store
    # Getting the type of 'stypy_return_type' (line 38)
    stypy_return_type_14186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14186)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_type_str'
    return stypy_return_type_14186

# Assigning a type to the variable 'get_type_str' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'get_type_str', get_type_str)

@norecursion
def get_param_position(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'get_param_position'
    module_type_store = module_type_store.open_function_context('get_param_position', 53, 0, False)
    
    # Passed parameters checking function
    get_param_position.stypy_localization = localization
    get_param_position.stypy_type_of_self = None
    get_param_position.stypy_type_store = module_type_store
    get_param_position.stypy_function_name = 'get_param_position'
    get_param_position.stypy_param_names_list = ['source_code', 'param_number']
    get_param_position.stypy_varargs_param_name = None
    get_param_position.stypy_kwargs_param_name = None
    get_param_position.stypy_call_defaults = defaults
    get_param_position.stypy_call_varargs = varargs
    get_param_position.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'get_param_position', ['source_code', 'param_number'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'get_param_position', localization, ['source_code', 'param_number'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'get_param_position(...)' code ##################

    str_14187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, (-1)), 'str', '\n    Get the offset of a parameter within a source code line that specify a method header. This is used to mark\n    parameters with type errors when reporting them.\n\n    :param source_code: Source code (method header)\n    :param param_number: Number of parameter\n    :return: Offset of the parameter in the source line\n    ')
    
    
    # SSA begins for try-except statement (line 62)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 63):
    
    # Call to split(...): (line 63)
    # Processing the call arguments (line 63)
    str_14190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 38), 'str', ',')
    # Processing the call keyword arguments (line 63)
    kwargs_14191 = {}
    # Getting the type of 'source_code' (line 63)
    source_code_14188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 20), 'source_code', False)
    # Obtaining the member 'split' of a type (line 63)
    split_14189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 20), source_code_14188, 'split')
    # Calling split(args, kwargs) (line 63)
    split_call_result_14192 = invoke(stypy.reporting.localization.Localization(__file__, 63, 20), split_14189, *[str_14190], **kwargs_14191)
    
    # Assigning a type to the variable 'split_str' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'split_str', split_call_result_14192)
    
    # Getting the type of 'param_number' (line 64)
    param_number_14193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 11), 'param_number')
    
    # Call to len(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'split_str' (line 64)
    split_str_14195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 31), 'split_str', False)
    # Processing the call keyword arguments (line 64)
    kwargs_14196 = {}
    # Getting the type of 'len' (line 64)
    len_14194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 27), 'len', False)
    # Calling len(args, kwargs) (line 64)
    len_call_result_14197 = invoke(stypy.reporting.localization.Localization(__file__, 64, 27), len_14194, *[split_str_14195], **kwargs_14196)
    
    # Applying the binary operator '>=' (line 64)
    result_ge_14198 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 11), '>=', param_number_14193, len_call_result_14197)
    
    # Testing if the type of an if condition is none (line 64)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 64, 8), result_ge_14198):
        pass
    else:
        
        # Testing the type of an if condition (line 64)
        if_condition_14199 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 64, 8), result_ge_14198)
        # Assigning a type to the variable 'if_condition_14199' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'if_condition_14199', if_condition_14199)
        # SSA begins for if statement (line 64)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        int_14200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 19), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'stypy_return_type', int_14200)
        # SSA join for if statement (line 64)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'param_number' (line 67)
    param_number_14201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 11), 'param_number')
    int_14202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 27), 'int')
    # Applying the binary operator '==' (line 67)
    result_eq_14203 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 11), '==', param_number_14201, int_14202)
    
    # Testing if the type of an if condition is none (line 67)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 67, 8), result_eq_14203):
        
        # Assigning a Num to a Name (line 76):
        int_14235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 21), 'int')
        # Assigning a type to the variable 'offset' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'offset', int_14235)
        
        
        # Call to range(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'param_number' (line 77)
        param_number_14237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 27), 'param_number', False)
        # Processing the call keyword arguments (line 77)
        kwargs_14238 = {}
        # Getting the type of 'range' (line 77)
        range_14236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 21), 'range', False)
        # Calling range(args, kwargs) (line 77)
        range_call_result_14239 = invoke(stypy.reporting.localization.Localization(__file__, 77, 21), range_14236, *[param_number_14237], **kwargs_14238)
        
        # Assigning a type to the variable 'range_call_result_14239' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'range_call_result_14239', range_call_result_14239)
        # Testing if the for loop is going to be iterated (line 77)
        # Testing the type of a for loop iterable (line 77)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 77, 12), range_call_result_14239)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 77, 12), range_call_result_14239):
            # Getting the type of the for loop variable (line 77)
            for_loop_var_14240 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 77, 12), range_call_result_14239)
            # Assigning a type to the variable 'i' (line 77)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'i', for_loop_var_14240)
            # SSA begins for a for statement (line 77)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'offset' (line 78)
            offset_14241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'offset')
            
            # Call to len(...): (line 78)
            # Processing the call arguments (line 78)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 78)
            i_14243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 40), 'i', False)
            # Getting the type of 'split_str' (line 78)
            split_str_14244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 30), 'split_str', False)
            # Obtaining the member '__getitem__' of a type (line 78)
            getitem___14245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 30), split_str_14244, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 78)
            subscript_call_result_14246 = invoke(stypy.reporting.localization.Localization(__file__, 78, 30), getitem___14245, i_14243)
            
            # Processing the call keyword arguments (line 78)
            kwargs_14247 = {}
            # Getting the type of 'len' (line 78)
            len_14242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 26), 'len', False)
            # Calling len(args, kwargs) (line 78)
            len_call_result_14248 = invoke(stypy.reporting.localization.Localization(__file__, 78, 26), len_14242, *[subscript_call_result_14246], **kwargs_14247)
            
            int_14249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 46), 'int')
            # Applying the binary operator '+' (line 78)
            result_add_14250 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 26), '+', len_call_result_14248, int_14249)
            
            # Applying the binary operator '+=' (line 78)
            result_iadd_14251 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 16), '+=', offset_14241, result_add_14250)
            # Assigning a type to the variable 'offset' (line 78)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'offset', result_iadd_14251)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Num to a Name (line 80):
        int_14252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 27), 'int')
        # Assigning a type to the variable 'blank_offset' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'blank_offset', int_14252)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'param_number' (line 81)
        param_number_14253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 33), 'param_number')
        # Getting the type of 'split_str' (line 81)
        split_str_14254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 23), 'split_str')
        # Obtaining the member '__getitem__' of a type (line 81)
        getitem___14255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 23), split_str_14254, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 81)
        subscript_call_result_14256 = invoke(stypy.reporting.localization.Localization(__file__, 81, 23), getitem___14255, param_number_14253)
        
        # Assigning a type to the variable 'subscript_call_result_14256' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'subscript_call_result_14256', subscript_call_result_14256)
        # Testing if the for loop is going to be iterated (line 81)
        # Testing the type of a for loop iterable (line 81)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 81, 12), subscript_call_result_14256)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 81, 12), subscript_call_result_14256):
            # Getting the type of the for loop variable (line 81)
            for_loop_var_14257 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 81, 12), subscript_call_result_14256)
            # Assigning a type to the variable 'car' (line 81)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'car', for_loop_var_14257)
            # SSA begins for a for statement (line 81)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'car' (line 82)
            car_14258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 19), 'car')
            str_14259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 26), 'str', ' ')
            # Applying the binary operator '==' (line 82)
            result_eq_14260 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 19), '==', car_14258, str_14259)
            
            # Testing if the type of an if condition is none (line 82)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 82, 16), result_eq_14260):
                pass
            else:
                
                # Testing the type of an if condition (line 82)
                if_condition_14261 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 16), result_eq_14260)
                # Assigning a type to the variable 'if_condition_14261' (line 82)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'if_condition_14261', if_condition_14261)
                # SSA begins for if statement (line 82)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'blank_offset' (line 83)
                blank_offset_14262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), 'blank_offset')
                int_14263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 36), 'int')
                # Applying the binary operator '+=' (line 83)
                result_iadd_14264 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 20), '+=', blank_offset_14262, int_14263)
                # Assigning a type to the variable 'blank_offset' (line 83)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), 'blank_offset', result_iadd_14264)
                
                # SSA join for if statement (line 82)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
    else:
        
        # Testing the type of an if condition (line 67)
        if_condition_14204 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 8), result_eq_14203)
        # Assigning a type to the variable 'if_condition_14204' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'if_condition_14204', if_condition_14204)
        # SSA begins for if statement (line 67)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 68):
        
        # Call to split(...): (line 68)
        # Processing the call arguments (line 68)
        str_14210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 48), 'str', '(')
        # Processing the call keyword arguments (line 68)
        kwargs_14211 = {}
        
        # Obtaining the type of the subscript
        int_14205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 39), 'int')
        # Getting the type of 'split_str' (line 68)
        split_str_14206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 29), 'split_str', False)
        # Obtaining the member '__getitem__' of a type (line 68)
        getitem___14207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 29), split_str_14206, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 68)
        subscript_call_result_14208 = invoke(stypy.reporting.localization.Localization(__file__, 68, 29), getitem___14207, int_14205)
        
        # Obtaining the member 'split' of a type (line 68)
        split_14209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 29), subscript_call_result_14208, 'split')
        # Calling split(args, kwargs) (line 68)
        split_call_result_14212 = invoke(stypy.reporting.localization.Localization(__file__, 68, 29), split_14209, *[str_14210], **kwargs_14211)
        
        # Assigning a type to the variable 'name_and_first' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'name_and_first', split_call_result_14212)
        
        # Assigning a BinOp to a Name (line 69):
        
        # Call to len(...): (line 69)
        # Processing the call arguments (line 69)
        
        # Obtaining the type of the subscript
        int_14214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 40), 'int')
        # Getting the type of 'name_and_first' (line 69)
        name_and_first_14215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 25), 'name_and_first', False)
        # Obtaining the member '__getitem__' of a type (line 69)
        getitem___14216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 25), name_and_first_14215, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 69)
        subscript_call_result_14217 = invoke(stypy.reporting.localization.Localization(__file__, 69, 25), getitem___14216, int_14214)
        
        # Processing the call keyword arguments (line 69)
        kwargs_14218 = {}
        # Getting the type of 'len' (line 69)
        len_14213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 21), 'len', False)
        # Calling len(args, kwargs) (line 69)
        len_call_result_14219 = invoke(stypy.reporting.localization.Localization(__file__, 69, 21), len_14213, *[subscript_call_result_14217], **kwargs_14218)
        
        int_14220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 46), 'int')
        # Applying the binary operator '+' (line 69)
        result_add_14221 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 21), '+', len_call_result_14219, int_14220)
        
        # Assigning a type to the variable 'offset' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'offset', result_add_14221)
        
        # Assigning a Num to a Name (line 71):
        int_14222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 27), 'int')
        # Assigning a type to the variable 'blank_offset' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'blank_offset', int_14222)
        
        
        # Obtaining the type of the subscript
        int_14223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 38), 'int')
        # Getting the type of 'name_and_first' (line 72)
        name_and_first_14224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 23), 'name_and_first')
        # Obtaining the member '__getitem__' of a type (line 72)
        getitem___14225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 23), name_and_first_14224, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 72)
        subscript_call_result_14226 = invoke(stypy.reporting.localization.Localization(__file__, 72, 23), getitem___14225, int_14223)
        
        # Assigning a type to the variable 'subscript_call_result_14226' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'subscript_call_result_14226', subscript_call_result_14226)
        # Testing if the for loop is going to be iterated (line 72)
        # Testing the type of a for loop iterable (line 72)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 72, 12), subscript_call_result_14226)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 72, 12), subscript_call_result_14226):
            # Getting the type of the for loop variable (line 72)
            for_loop_var_14227 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 72, 12), subscript_call_result_14226)
            # Assigning a type to the variable 'car' (line 72)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'car', for_loop_var_14227)
            # SSA begins for a for statement (line 72)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'car' (line 73)
            car_14228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 19), 'car')
            str_14229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 26), 'str', ' ')
            # Applying the binary operator '==' (line 73)
            result_eq_14230 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 19), '==', car_14228, str_14229)
            
            # Testing if the type of an if condition is none (line 73)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 73, 16), result_eq_14230):
                pass
            else:
                
                # Testing the type of an if condition (line 73)
                if_condition_14231 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 73, 16), result_eq_14230)
                # Assigning a type to the variable 'if_condition_14231' (line 73)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 16), 'if_condition_14231', if_condition_14231)
                # SSA begins for if statement (line 73)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'blank_offset' (line 74)
                blank_offset_14232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 20), 'blank_offset')
                int_14233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 36), 'int')
                # Applying the binary operator '+=' (line 74)
                result_iadd_14234 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 20), '+=', blank_offset_14232, int_14233)
                # Assigning a type to the variable 'blank_offset' (line 74)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 20), 'blank_offset', result_iadd_14234)
                
                # SSA join for if statement (line 73)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA branch for the else part of an if statement (line 67)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 76):
        int_14235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 21), 'int')
        # Assigning a type to the variable 'offset' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'offset', int_14235)
        
        
        # Call to range(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'param_number' (line 77)
        param_number_14237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 27), 'param_number', False)
        # Processing the call keyword arguments (line 77)
        kwargs_14238 = {}
        # Getting the type of 'range' (line 77)
        range_14236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 21), 'range', False)
        # Calling range(args, kwargs) (line 77)
        range_call_result_14239 = invoke(stypy.reporting.localization.Localization(__file__, 77, 21), range_14236, *[param_number_14237], **kwargs_14238)
        
        # Assigning a type to the variable 'range_call_result_14239' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'range_call_result_14239', range_call_result_14239)
        # Testing if the for loop is going to be iterated (line 77)
        # Testing the type of a for loop iterable (line 77)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 77, 12), range_call_result_14239)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 77, 12), range_call_result_14239):
            # Getting the type of the for loop variable (line 77)
            for_loop_var_14240 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 77, 12), range_call_result_14239)
            # Assigning a type to the variable 'i' (line 77)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'i', for_loop_var_14240)
            # SSA begins for a for statement (line 77)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'offset' (line 78)
            offset_14241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'offset')
            
            # Call to len(...): (line 78)
            # Processing the call arguments (line 78)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 78)
            i_14243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 40), 'i', False)
            # Getting the type of 'split_str' (line 78)
            split_str_14244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 30), 'split_str', False)
            # Obtaining the member '__getitem__' of a type (line 78)
            getitem___14245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 30), split_str_14244, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 78)
            subscript_call_result_14246 = invoke(stypy.reporting.localization.Localization(__file__, 78, 30), getitem___14245, i_14243)
            
            # Processing the call keyword arguments (line 78)
            kwargs_14247 = {}
            # Getting the type of 'len' (line 78)
            len_14242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 26), 'len', False)
            # Calling len(args, kwargs) (line 78)
            len_call_result_14248 = invoke(stypy.reporting.localization.Localization(__file__, 78, 26), len_14242, *[subscript_call_result_14246], **kwargs_14247)
            
            int_14249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 46), 'int')
            # Applying the binary operator '+' (line 78)
            result_add_14250 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 26), '+', len_call_result_14248, int_14249)
            
            # Applying the binary operator '+=' (line 78)
            result_iadd_14251 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 16), '+=', offset_14241, result_add_14250)
            # Assigning a type to the variable 'offset' (line 78)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'offset', result_iadd_14251)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Num to a Name (line 80):
        int_14252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 27), 'int')
        # Assigning a type to the variable 'blank_offset' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'blank_offset', int_14252)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'param_number' (line 81)
        param_number_14253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 33), 'param_number')
        # Getting the type of 'split_str' (line 81)
        split_str_14254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 23), 'split_str')
        # Obtaining the member '__getitem__' of a type (line 81)
        getitem___14255 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 23), split_str_14254, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 81)
        subscript_call_result_14256 = invoke(stypy.reporting.localization.Localization(__file__, 81, 23), getitem___14255, param_number_14253)
        
        # Assigning a type to the variable 'subscript_call_result_14256' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'subscript_call_result_14256', subscript_call_result_14256)
        # Testing if the for loop is going to be iterated (line 81)
        # Testing the type of a for loop iterable (line 81)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 81, 12), subscript_call_result_14256)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 81, 12), subscript_call_result_14256):
            # Getting the type of the for loop variable (line 81)
            for_loop_var_14257 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 81, 12), subscript_call_result_14256)
            # Assigning a type to the variable 'car' (line 81)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'car', for_loop_var_14257)
            # SSA begins for a for statement (line 81)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'car' (line 82)
            car_14258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 19), 'car')
            str_14259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 26), 'str', ' ')
            # Applying the binary operator '==' (line 82)
            result_eq_14260 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 19), '==', car_14258, str_14259)
            
            # Testing if the type of an if condition is none (line 82)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 82, 16), result_eq_14260):
                pass
            else:
                
                # Testing the type of an if condition (line 82)
                if_condition_14261 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 16), result_eq_14260)
                # Assigning a type to the variable 'if_condition_14261' (line 82)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'if_condition_14261', if_condition_14261)
                # SSA begins for if statement (line 82)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'blank_offset' (line 83)
                blank_offset_14262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), 'blank_offset')
                int_14263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 36), 'int')
                # Applying the binary operator '+=' (line 83)
                result_iadd_14264 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 20), '+=', blank_offset_14262, int_14263)
                # Assigning a type to the variable 'blank_offset' (line 83)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), 'blank_offset', result_iadd_14264)
                
                # SSA join for if statement (line 82)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for if statement (line 67)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'offset' (line 85)
    offset_14265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 15), 'offset')
    # Getting the type of 'blank_offset' (line 85)
    blank_offset_14266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 24), 'blank_offset')
    # Applying the binary operator '+' (line 85)
    result_add_14267 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 15), '+', offset_14265, blank_offset_14266)
    
    # Assigning a type to the variable 'stypy_return_type' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'stypy_return_type', result_add_14267)
    # SSA branch for the except part of a try statement (line 62)
    # SSA branch for the except '<any exception>' branch of a try statement (line 62)
    module_type_store.open_ssa_branch('except')
    int_14268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'stypy_return_type', int_14268)
    # SSA join for try-except statement (line 62)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'get_param_position(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_param_position' in the type store
    # Getting the type of 'stypy_return_type' (line 53)
    stypy_return_type_14269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_14269)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_param_position'
    return stypy_return_type_14269

# Assigning a type to the variable 'get_param_position' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'get_param_position', get_param_position)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
