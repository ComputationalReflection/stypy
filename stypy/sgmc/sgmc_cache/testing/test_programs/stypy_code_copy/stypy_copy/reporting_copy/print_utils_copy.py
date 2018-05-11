
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from ...stypy_copy import type_store_copy
2: from ...stypy_copy.errors_copy.type_error_copy import TypeError
3: from ...stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy.stypy_functions_copy import default_lambda_var_name
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
48:     if type(type_) == type_store_copy.typestore.TypeStore:
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

# 'from testing.test_programs.stypy_code_copy.stypy_copy import type_store_copy' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/reporting_copy/')
import_16361 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'testing.test_programs.stypy_code_copy.stypy_copy')

if (type(import_16361) is not StypyTypeError):

    if (import_16361 != 'pyd_module'):
        __import__(import_16361)
        sys_modules_16362 = sys.modules[import_16361]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', sys_modules_16362.module_type_store, module_type_store, ['type_store_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_16362, sys_modules_16362.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy import type_store_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', None, module_type_store, ['type_store_copy'], [type_store_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', import_16361)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/reporting_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy import TypeError' statement (line 2)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/reporting_copy/')
import_16363 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy')

if (type(import_16363) is not StypyTypeError):

    if (import_16363 != 'pyd_module'):
        __import__(import_16363)
        sys_modules_16364 = sys.modules[import_16363]
        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', sys_modules_16364.module_type_store, module_type_store, ['TypeError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 2, 0), __file__, sys_modules_16364, sys_modules_16364.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy import TypeError

        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', None, module_type_store, ['TypeError'], [TypeError])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', import_16363)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/reporting_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy.stypy_functions_copy import default_lambda_var_name' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/reporting_copy/')
import_16365 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy.stypy_functions_copy')

if (type(import_16365) is not StypyTypeError):

    if (import_16365 != 'pyd_module'):
        __import__(import_16365)
        sys_modules_16366 = sys.modules[import_16365]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy.stypy_functions_copy', sys_modules_16366.module_type_store, module_type_store, ['default_lambda_var_name'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_16366, sys_modules_16366.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy.stypy_functions_copy import default_lambda_var_name

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy.stypy_functions_copy', None, module_type_store, ['default_lambda_var_name'], [default_lambda_var_name])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy.stypy_functions_copy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy.stypy_functions_copy', import_16365)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/reporting_copy/')

str_16367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, (-1)), 'str', '\nSeveral functions to help printing elements with a readable style on error reports\n')

# Assigning a List to a Name (line 10):

# Obtaining an instance of the builtin type 'list' (line 10)
list_16368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 10)
# Adding element type (line 10)
str_16369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 26), 'str', '__temp_tuple_assignment')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 25), list_16368, str_16369)
# Adding element type (line 10)
str_16370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 53), 'str', '__temp_list_assignment')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 25), list_16368, str_16370)
# Adding element type (line 10)
str_16371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 26), 'str', '__temp_lambda')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 25), list_16368, str_16371)
# Adding element type (line 10)
str_16372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 43), 'str', '__temp_call_assignment')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 25), list_16368, str_16372)

# Assigning a type to the variable 'private_variable_names' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'private_variable_names', list_16368)

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

    str_16373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, (-1)), 'str', '\n    Prints a function name, considering lambda functions.\n    :param fname: Function name\n    :return: Proper function name\n    ')
    
    # Getting the type of 'default_lambda_var_name' (line 20)
    default_lambda_var_name_16374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 7), 'default_lambda_var_name')
    # Getting the type of 'fname' (line 20)
    fname_16375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 34), 'fname')
    # Applying the binary operator 'in' (line 20)
    result_contains_16376 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 7), 'in', default_lambda_var_name_16374, fname_16375)
    
    # Testing if the type of an if condition is none (line 20)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 20, 4), result_contains_16376):
        pass
    else:
        
        # Testing the type of an if condition (line 20)
        if_condition_16377 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 20, 4), result_contains_16376)
        # Assigning a type to the variable 'if_condition_16377' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'if_condition_16377', if_condition_16377)
        # SSA begins for if statement (line 20)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_16378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 15), 'str', 'lambda function')
        # Assigning a type to the variable 'stypy_return_type' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'stypy_return_type', str_16378)
        # SSA join for if statement (line 20)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'fname' (line 22)
    fname_16379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 11), 'fname')
    # Assigning a type to the variable 'stypy_return_type' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'stypy_return_type', fname_16379)
    
    # ################# End of 'format_function_name(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'format_function_name' in the type store
    # Getting the type of 'stypy_return_type' (line 14)
    stypy_return_type_16380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16380)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'format_function_name'
    return stypy_return_type_16380

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

    str_16381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, (-1)), 'str', '\n    Determines if a variable is a stypy private variable\n    :param var_name: Variable name\n    :return: bool\n    ')
    
    # Getting the type of 'private_variable_names' (line 31)
    private_variable_names_16382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 24), 'private_variable_names')
    # Assigning a type to the variable 'private_variable_names_16382' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'private_variable_names_16382', private_variable_names_16382)
    # Testing if the for loop is going to be iterated (line 31)
    # Testing the type of a for loop iterable (line 31)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 31, 4), private_variable_names_16382)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 31, 4), private_variable_names_16382):
        # Getting the type of the for loop variable (line 31)
        for_loop_var_16383 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 31, 4), private_variable_names_16382)
        # Assigning a type to the variable 'private_name' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'private_name', for_loop_var_16383)
        # SSA begins for a for statement (line 31)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'private_name' (line 32)
        private_name_16384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 11), 'private_name')
        # Getting the type of 'var_name' (line 32)
        var_name_16385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 27), 'var_name')
        # Applying the binary operator 'in' (line 32)
        result_contains_16386 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 11), 'in', private_name_16384, var_name_16385)
        
        # Testing if the type of an if condition is none (line 32)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 32, 8), result_contains_16386):
            pass
        else:
            
            # Testing the type of an if condition (line 32)
            if_condition_16387 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 32, 8), result_contains_16386)
            # Assigning a type to the variable 'if_condition_16387' (line 32)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'if_condition_16387', if_condition_16387)
            # SSA begins for if statement (line 32)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'True' (line 33)
            True_16388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 19), 'True')
            # Assigning a type to the variable 'stypy_return_type' (line 33)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'stypy_return_type', True_16388)
            # SSA join for if statement (line 32)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'False' (line 35)
    False_16389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 11), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type', False_16389)
    
    # ################# End of 'is_private_variable_name(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'is_private_variable_name' in the type store
    # Getting the type of 'stypy_return_type' (line 25)
    stypy_return_type_16390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16390)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'is_private_variable_name'
    return stypy_return_type_16390

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

    str_16391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, (-1)), 'str', '\n    Get the abbreviated str representation of a type for printing friendly error messages\n    :param type_: Type\n    :return: str\n    ')
    
    # Type idiom detected: calculating its left and rigth part (line 44)
    # Getting the type of 'TypeError' (line 44)
    TypeError_16392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 25), 'TypeError')
    # Getting the type of 'type_' (line 44)
    type__16393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 18), 'type_')
    
    (may_be_16394, more_types_in_union_16395) = may_be_subtype(TypeError_16392, type__16393)

    if may_be_16394:

        if more_types_in_union_16395:
            # Runtime conditional SSA (line 44)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'type_' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'type_', remove_not_subtype_from_union(type__16393, TypeError))
        str_16396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 15), 'str', 'TypeError')
        # Assigning a type to the variable 'stypy_return_type' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'stypy_return_type', str_16396)

        if more_types_in_union_16395:
            # SSA join for if statement (line 44)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Call to type(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'type_' (line 48)
    type__16398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'type_', False)
    # Processing the call keyword arguments (line 48)
    kwargs_16399 = {}
    # Getting the type of 'type' (line 48)
    type_16397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 7), 'type', False)
    # Calling type(args, kwargs) (line 48)
    type_call_result_16400 = invoke(stypy.reporting.localization.Localization(__file__, 48, 7), type_16397, *[type__16398], **kwargs_16399)
    
    # Getting the type of 'type_store_copy' (line 48)
    type_store_copy_16401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 22), 'type_store_copy')
    # Obtaining the member 'typestore' of a type (line 48)
    typestore_16402 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 22), type_store_copy_16401, 'typestore')
    # Obtaining the member 'TypeStore' of a type (line 48)
    TypeStore_16403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 22), typestore_16402, 'TypeStore')
    # Applying the binary operator '==' (line 48)
    result_eq_16404 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 7), '==', type_call_result_16400, TypeStore_16403)
    
    # Testing if the type of an if condition is none (line 48)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 48, 4), result_eq_16404):
        pass
    else:
        
        # Testing the type of an if condition (line 48)
        if_condition_16405 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 48, 4), result_eq_16404)
        # Assigning a type to the variable 'if_condition_16405' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'if_condition_16405', if_condition_16405)
        # SSA begins for if statement (line 48)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_16406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 15), 'str', "External module '")
        # Getting the type of 'type_' (line 49)
        type__16407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 37), 'type_')
        # Obtaining the member 'program_name' of a type (line 49)
        program_name_16408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 37), type__16407, 'program_name')
        # Applying the binary operator '+' (line 49)
        result_add_16409 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 15), '+', str_16406, program_name_16408)
        
        str_16410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 58), 'str', "'")
        # Applying the binary operator '+' (line 49)
        result_add_16411 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 56), '+', result_add_16409, str_16410)
        
        # Assigning a type to the variable 'stypy_return_type' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'stypy_return_type', result_add_16411)
        # SSA join for if statement (line 48)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to str(...): (line 50)
    # Processing the call arguments (line 50)
    # Getting the type of 'type_' (line 50)
    type__16413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 15), 'type_', False)
    # Processing the call keyword arguments (line 50)
    kwargs_16414 = {}
    # Getting the type of 'str' (line 50)
    str_16412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 11), 'str', False)
    # Calling str(args, kwargs) (line 50)
    str_call_result_16415 = invoke(stypy.reporting.localization.Localization(__file__, 50, 11), str_16412, *[type__16413], **kwargs_16414)
    
    # Assigning a type to the variable 'stypy_return_type' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'stypy_return_type', str_call_result_16415)
    
    # ################# End of 'get_type_str(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_type_str' in the type store
    # Getting the type of 'stypy_return_type' (line 38)
    stypy_return_type_16416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16416)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_type_str'
    return stypy_return_type_16416

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

    str_16417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, (-1)), 'str', '\n    Get the offset of a parameter within a source code line that specify a method header. This is used to mark\n    parameters with type errors when reporting them.\n\n    :param source_code: Source code (method header)\n    :param param_number: Number of parameter\n    :return: Offset of the parameter in the source line\n    ')
    
    
    # SSA begins for try-except statement (line 62)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 63):
    
    # Call to split(...): (line 63)
    # Processing the call arguments (line 63)
    str_16420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 38), 'str', ',')
    # Processing the call keyword arguments (line 63)
    kwargs_16421 = {}
    # Getting the type of 'source_code' (line 63)
    source_code_16418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 20), 'source_code', False)
    # Obtaining the member 'split' of a type (line 63)
    split_16419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 20), source_code_16418, 'split')
    # Calling split(args, kwargs) (line 63)
    split_call_result_16422 = invoke(stypy.reporting.localization.Localization(__file__, 63, 20), split_16419, *[str_16420], **kwargs_16421)
    
    # Assigning a type to the variable 'split_str' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'split_str', split_call_result_16422)
    
    # Getting the type of 'param_number' (line 64)
    param_number_16423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 11), 'param_number')
    
    # Call to len(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'split_str' (line 64)
    split_str_16425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 31), 'split_str', False)
    # Processing the call keyword arguments (line 64)
    kwargs_16426 = {}
    # Getting the type of 'len' (line 64)
    len_16424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 27), 'len', False)
    # Calling len(args, kwargs) (line 64)
    len_call_result_16427 = invoke(stypy.reporting.localization.Localization(__file__, 64, 27), len_16424, *[split_str_16425], **kwargs_16426)
    
    # Applying the binary operator '>=' (line 64)
    result_ge_16428 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 11), '>=', param_number_16423, len_call_result_16427)
    
    # Testing if the type of an if condition is none (line 64)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 64, 8), result_ge_16428):
        pass
    else:
        
        # Testing the type of an if condition (line 64)
        if_condition_16429 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 64, 8), result_ge_16428)
        # Assigning a type to the variable 'if_condition_16429' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'if_condition_16429', if_condition_16429)
        # SSA begins for if statement (line 64)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        int_16430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 19), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'stypy_return_type', int_16430)
        # SSA join for if statement (line 64)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'param_number' (line 67)
    param_number_16431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 11), 'param_number')
    int_16432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 27), 'int')
    # Applying the binary operator '==' (line 67)
    result_eq_16433 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 11), '==', param_number_16431, int_16432)
    
    # Testing if the type of an if condition is none (line 67)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 67, 8), result_eq_16433):
        
        # Assigning a Num to a Name (line 76):
        int_16465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 21), 'int')
        # Assigning a type to the variable 'offset' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'offset', int_16465)
        
        
        # Call to range(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'param_number' (line 77)
        param_number_16467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 27), 'param_number', False)
        # Processing the call keyword arguments (line 77)
        kwargs_16468 = {}
        # Getting the type of 'range' (line 77)
        range_16466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 21), 'range', False)
        # Calling range(args, kwargs) (line 77)
        range_call_result_16469 = invoke(stypy.reporting.localization.Localization(__file__, 77, 21), range_16466, *[param_number_16467], **kwargs_16468)
        
        # Assigning a type to the variable 'range_call_result_16469' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'range_call_result_16469', range_call_result_16469)
        # Testing if the for loop is going to be iterated (line 77)
        # Testing the type of a for loop iterable (line 77)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 77, 12), range_call_result_16469)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 77, 12), range_call_result_16469):
            # Getting the type of the for loop variable (line 77)
            for_loop_var_16470 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 77, 12), range_call_result_16469)
            # Assigning a type to the variable 'i' (line 77)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'i', for_loop_var_16470)
            # SSA begins for a for statement (line 77)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'offset' (line 78)
            offset_16471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'offset')
            
            # Call to len(...): (line 78)
            # Processing the call arguments (line 78)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 78)
            i_16473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 40), 'i', False)
            # Getting the type of 'split_str' (line 78)
            split_str_16474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 30), 'split_str', False)
            # Obtaining the member '__getitem__' of a type (line 78)
            getitem___16475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 30), split_str_16474, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 78)
            subscript_call_result_16476 = invoke(stypy.reporting.localization.Localization(__file__, 78, 30), getitem___16475, i_16473)
            
            # Processing the call keyword arguments (line 78)
            kwargs_16477 = {}
            # Getting the type of 'len' (line 78)
            len_16472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 26), 'len', False)
            # Calling len(args, kwargs) (line 78)
            len_call_result_16478 = invoke(stypy.reporting.localization.Localization(__file__, 78, 26), len_16472, *[subscript_call_result_16476], **kwargs_16477)
            
            int_16479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 46), 'int')
            # Applying the binary operator '+' (line 78)
            result_add_16480 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 26), '+', len_call_result_16478, int_16479)
            
            # Applying the binary operator '+=' (line 78)
            result_iadd_16481 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 16), '+=', offset_16471, result_add_16480)
            # Assigning a type to the variable 'offset' (line 78)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'offset', result_iadd_16481)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Num to a Name (line 80):
        int_16482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 27), 'int')
        # Assigning a type to the variable 'blank_offset' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'blank_offset', int_16482)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'param_number' (line 81)
        param_number_16483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 33), 'param_number')
        # Getting the type of 'split_str' (line 81)
        split_str_16484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 23), 'split_str')
        # Obtaining the member '__getitem__' of a type (line 81)
        getitem___16485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 23), split_str_16484, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 81)
        subscript_call_result_16486 = invoke(stypy.reporting.localization.Localization(__file__, 81, 23), getitem___16485, param_number_16483)
        
        # Assigning a type to the variable 'subscript_call_result_16486' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'subscript_call_result_16486', subscript_call_result_16486)
        # Testing if the for loop is going to be iterated (line 81)
        # Testing the type of a for loop iterable (line 81)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 81, 12), subscript_call_result_16486)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 81, 12), subscript_call_result_16486):
            # Getting the type of the for loop variable (line 81)
            for_loop_var_16487 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 81, 12), subscript_call_result_16486)
            # Assigning a type to the variable 'car' (line 81)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'car', for_loop_var_16487)
            # SSA begins for a for statement (line 81)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'car' (line 82)
            car_16488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 19), 'car')
            str_16489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 26), 'str', ' ')
            # Applying the binary operator '==' (line 82)
            result_eq_16490 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 19), '==', car_16488, str_16489)
            
            # Testing if the type of an if condition is none (line 82)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 82, 16), result_eq_16490):
                pass
            else:
                
                # Testing the type of an if condition (line 82)
                if_condition_16491 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 16), result_eq_16490)
                # Assigning a type to the variable 'if_condition_16491' (line 82)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'if_condition_16491', if_condition_16491)
                # SSA begins for if statement (line 82)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'blank_offset' (line 83)
                blank_offset_16492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), 'blank_offset')
                int_16493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 36), 'int')
                # Applying the binary operator '+=' (line 83)
                result_iadd_16494 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 20), '+=', blank_offset_16492, int_16493)
                # Assigning a type to the variable 'blank_offset' (line 83)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), 'blank_offset', result_iadd_16494)
                
                # SSA join for if statement (line 82)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
    else:
        
        # Testing the type of an if condition (line 67)
        if_condition_16434 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 8), result_eq_16433)
        # Assigning a type to the variable 'if_condition_16434' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'if_condition_16434', if_condition_16434)
        # SSA begins for if statement (line 67)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 68):
        
        # Call to split(...): (line 68)
        # Processing the call arguments (line 68)
        str_16440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 48), 'str', '(')
        # Processing the call keyword arguments (line 68)
        kwargs_16441 = {}
        
        # Obtaining the type of the subscript
        int_16435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 39), 'int')
        # Getting the type of 'split_str' (line 68)
        split_str_16436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 29), 'split_str', False)
        # Obtaining the member '__getitem__' of a type (line 68)
        getitem___16437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 29), split_str_16436, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 68)
        subscript_call_result_16438 = invoke(stypy.reporting.localization.Localization(__file__, 68, 29), getitem___16437, int_16435)
        
        # Obtaining the member 'split' of a type (line 68)
        split_16439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 29), subscript_call_result_16438, 'split')
        # Calling split(args, kwargs) (line 68)
        split_call_result_16442 = invoke(stypy.reporting.localization.Localization(__file__, 68, 29), split_16439, *[str_16440], **kwargs_16441)
        
        # Assigning a type to the variable 'name_and_first' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'name_and_first', split_call_result_16442)
        
        # Assigning a BinOp to a Name (line 69):
        
        # Call to len(...): (line 69)
        # Processing the call arguments (line 69)
        
        # Obtaining the type of the subscript
        int_16444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 40), 'int')
        # Getting the type of 'name_and_first' (line 69)
        name_and_first_16445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 25), 'name_and_first', False)
        # Obtaining the member '__getitem__' of a type (line 69)
        getitem___16446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 25), name_and_first_16445, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 69)
        subscript_call_result_16447 = invoke(stypy.reporting.localization.Localization(__file__, 69, 25), getitem___16446, int_16444)
        
        # Processing the call keyword arguments (line 69)
        kwargs_16448 = {}
        # Getting the type of 'len' (line 69)
        len_16443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 21), 'len', False)
        # Calling len(args, kwargs) (line 69)
        len_call_result_16449 = invoke(stypy.reporting.localization.Localization(__file__, 69, 21), len_16443, *[subscript_call_result_16447], **kwargs_16448)
        
        int_16450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 46), 'int')
        # Applying the binary operator '+' (line 69)
        result_add_16451 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 21), '+', len_call_result_16449, int_16450)
        
        # Assigning a type to the variable 'offset' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'offset', result_add_16451)
        
        # Assigning a Num to a Name (line 71):
        int_16452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 27), 'int')
        # Assigning a type to the variable 'blank_offset' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'blank_offset', int_16452)
        
        
        # Obtaining the type of the subscript
        int_16453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 38), 'int')
        # Getting the type of 'name_and_first' (line 72)
        name_and_first_16454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 23), 'name_and_first')
        # Obtaining the member '__getitem__' of a type (line 72)
        getitem___16455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 23), name_and_first_16454, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 72)
        subscript_call_result_16456 = invoke(stypy.reporting.localization.Localization(__file__, 72, 23), getitem___16455, int_16453)
        
        # Assigning a type to the variable 'subscript_call_result_16456' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'subscript_call_result_16456', subscript_call_result_16456)
        # Testing if the for loop is going to be iterated (line 72)
        # Testing the type of a for loop iterable (line 72)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 72, 12), subscript_call_result_16456)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 72, 12), subscript_call_result_16456):
            # Getting the type of the for loop variable (line 72)
            for_loop_var_16457 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 72, 12), subscript_call_result_16456)
            # Assigning a type to the variable 'car' (line 72)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'car', for_loop_var_16457)
            # SSA begins for a for statement (line 72)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'car' (line 73)
            car_16458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 19), 'car')
            str_16459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 26), 'str', ' ')
            # Applying the binary operator '==' (line 73)
            result_eq_16460 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 19), '==', car_16458, str_16459)
            
            # Testing if the type of an if condition is none (line 73)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 73, 16), result_eq_16460):
                pass
            else:
                
                # Testing the type of an if condition (line 73)
                if_condition_16461 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 73, 16), result_eq_16460)
                # Assigning a type to the variable 'if_condition_16461' (line 73)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 16), 'if_condition_16461', if_condition_16461)
                # SSA begins for if statement (line 73)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'blank_offset' (line 74)
                blank_offset_16462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 20), 'blank_offset')
                int_16463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 36), 'int')
                # Applying the binary operator '+=' (line 74)
                result_iadd_16464 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 20), '+=', blank_offset_16462, int_16463)
                # Assigning a type to the variable 'blank_offset' (line 74)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 20), 'blank_offset', result_iadd_16464)
                
                # SSA join for if statement (line 73)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA branch for the else part of an if statement (line 67)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 76):
        int_16465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 21), 'int')
        # Assigning a type to the variable 'offset' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'offset', int_16465)
        
        
        # Call to range(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'param_number' (line 77)
        param_number_16467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 27), 'param_number', False)
        # Processing the call keyword arguments (line 77)
        kwargs_16468 = {}
        # Getting the type of 'range' (line 77)
        range_16466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 21), 'range', False)
        # Calling range(args, kwargs) (line 77)
        range_call_result_16469 = invoke(stypy.reporting.localization.Localization(__file__, 77, 21), range_16466, *[param_number_16467], **kwargs_16468)
        
        # Assigning a type to the variable 'range_call_result_16469' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'range_call_result_16469', range_call_result_16469)
        # Testing if the for loop is going to be iterated (line 77)
        # Testing the type of a for loop iterable (line 77)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 77, 12), range_call_result_16469)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 77, 12), range_call_result_16469):
            # Getting the type of the for loop variable (line 77)
            for_loop_var_16470 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 77, 12), range_call_result_16469)
            # Assigning a type to the variable 'i' (line 77)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'i', for_loop_var_16470)
            # SSA begins for a for statement (line 77)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'offset' (line 78)
            offset_16471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'offset')
            
            # Call to len(...): (line 78)
            # Processing the call arguments (line 78)
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 78)
            i_16473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 40), 'i', False)
            # Getting the type of 'split_str' (line 78)
            split_str_16474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 30), 'split_str', False)
            # Obtaining the member '__getitem__' of a type (line 78)
            getitem___16475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 30), split_str_16474, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 78)
            subscript_call_result_16476 = invoke(stypy.reporting.localization.Localization(__file__, 78, 30), getitem___16475, i_16473)
            
            # Processing the call keyword arguments (line 78)
            kwargs_16477 = {}
            # Getting the type of 'len' (line 78)
            len_16472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 26), 'len', False)
            # Calling len(args, kwargs) (line 78)
            len_call_result_16478 = invoke(stypy.reporting.localization.Localization(__file__, 78, 26), len_16472, *[subscript_call_result_16476], **kwargs_16477)
            
            int_16479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 46), 'int')
            # Applying the binary operator '+' (line 78)
            result_add_16480 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 26), '+', len_call_result_16478, int_16479)
            
            # Applying the binary operator '+=' (line 78)
            result_iadd_16481 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 16), '+=', offset_16471, result_add_16480)
            # Assigning a type to the variable 'offset' (line 78)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'offset', result_iadd_16481)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Num to a Name (line 80):
        int_16482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 27), 'int')
        # Assigning a type to the variable 'blank_offset' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'blank_offset', int_16482)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'param_number' (line 81)
        param_number_16483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 33), 'param_number')
        # Getting the type of 'split_str' (line 81)
        split_str_16484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 23), 'split_str')
        # Obtaining the member '__getitem__' of a type (line 81)
        getitem___16485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 23), split_str_16484, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 81)
        subscript_call_result_16486 = invoke(stypy.reporting.localization.Localization(__file__, 81, 23), getitem___16485, param_number_16483)
        
        # Assigning a type to the variable 'subscript_call_result_16486' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'subscript_call_result_16486', subscript_call_result_16486)
        # Testing if the for loop is going to be iterated (line 81)
        # Testing the type of a for loop iterable (line 81)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 81, 12), subscript_call_result_16486)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 81, 12), subscript_call_result_16486):
            # Getting the type of the for loop variable (line 81)
            for_loop_var_16487 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 81, 12), subscript_call_result_16486)
            # Assigning a type to the variable 'car' (line 81)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'car', for_loop_var_16487)
            # SSA begins for a for statement (line 81)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'car' (line 82)
            car_16488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 19), 'car')
            str_16489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 26), 'str', ' ')
            # Applying the binary operator '==' (line 82)
            result_eq_16490 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 19), '==', car_16488, str_16489)
            
            # Testing if the type of an if condition is none (line 82)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 82, 16), result_eq_16490):
                pass
            else:
                
                # Testing the type of an if condition (line 82)
                if_condition_16491 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 16), result_eq_16490)
                # Assigning a type to the variable 'if_condition_16491' (line 82)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'if_condition_16491', if_condition_16491)
                # SSA begins for if statement (line 82)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'blank_offset' (line 83)
                blank_offset_16492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), 'blank_offset')
                int_16493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 36), 'int')
                # Applying the binary operator '+=' (line 83)
                result_iadd_16494 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 20), '+=', blank_offset_16492, int_16493)
                # Assigning a type to the variable 'blank_offset' (line 83)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), 'blank_offset', result_iadd_16494)
                
                # SSA join for if statement (line 82)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for if statement (line 67)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'offset' (line 85)
    offset_16495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 15), 'offset')
    # Getting the type of 'blank_offset' (line 85)
    blank_offset_16496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 24), 'blank_offset')
    # Applying the binary operator '+' (line 85)
    result_add_16497 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 15), '+', offset_16495, blank_offset_16496)
    
    # Assigning a type to the variable 'stypy_return_type' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'stypy_return_type', result_add_16497)
    # SSA branch for the except part of a try statement (line 62)
    # SSA branch for the except '<any exception>' branch of a try statement (line 62)
    module_type_store.open_ssa_branch('except')
    int_16498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'stypy_return_type', int_16498)
    # SSA join for try-except statement (line 62)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'get_param_position(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'get_param_position' in the type store
    # Getting the type of 'stypy_return_type' (line 53)
    stypy_return_type_16499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_16499)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'get_param_position'
    return stypy_return_type_16499

# Assigning a type to the variable 'get_param_position' (line 53)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 0), 'get_param_position', get_param_position)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
