
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import os
2: import sys
3: import inspect
4: import types
5: 
6: from stypy_copy.python_lib_copy.python_types_copy.type_copy import Type
7: from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.union_type_copy import UnionType
8: from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy import UndefinedType
9: from stypy_copy.stypy_parameters_copy import type_inference_file_directory_name, type_data_file_postfix
10: from stypy_copy.errors_copy.type_error import TypeError
11: from stypy_copy.type_store_copy import typestore_copy
12: 
13: '''
14: File with functions that are used when unit testing the generated type inference code checking the type inference code
15: type store against the type data file of the checked program
16: '''
17: 
18: 
19: def __filter_reserved_vars(types_):
20:     '''
21:     For the types_ list, eliminates the references to the TypeDataFileWriter class, to not to check this private object
22:     not part of the original program.
23:     :param types_: Type list
24:     :return:
25:     '''
26:     return filter(lambda elem: not 'TypeDataFileWriter' == elem, types_)
27: 
28: 
29: def __equal_types(expected_var, inferred_context_var):
30:     '''
31:     Helper function to check if the types of two vars can be considered equal from a unit testing point of view
32:     :param expected_var: Expected type
33:     :param inferred_context_var: Inferred type
34:     :return: bool
35:     '''
36: 
37:     # Identity equality
38:     if expected_var == inferred_context_var:
39:         return True
40: 
41:     # TypeInferenceProxy or TypeErrors
42:     if isinstance(inferred_context_var, Type):
43:         # Modules
44:         if expected_var is types.ModuleType:
45:             return inspect.ismodule(inferred_context_var.get_python_entity()) or isinstance(inferred_context_var, typestore_copy.TypeStore)
46: 
47:         if expected_var is types.ClassType:
48:             return inspect.isclass(inferred_context_var.get_python_type())
49: 
50:         if expected_var is TypeError:
51:             return isinstance(inferred_context_var, TypeError)
52: 
53:         direct_comp = inferred_context_var.get_python_type() == expected_var
54:         if not direct_comp and isinstance(expected_var, UnionType) and isinstance(inferred_context_var, UnionType):
55:             return len(expected_var.types) == len(inferred_context_var.types)
56:         return direct_comp
57: 
58:     # Functions
59:     if expected_var == types.FunctionType:
60:         return inspect.isfunction(inferred_context_var)
61: 
62:     # Builtin functions
63:     if expected_var == types.BuiltinFunctionType:
64:         return inspect.isfunction(inferred_context_var)
65: 
66:     # Undefined
67:     if isinstance(inferred_context_var, UndefinedType):
68:         return isinstance(expected_var, UndefinedType)
69: 
70:     # Classes
71:     if expected_var is types.ClassType:
72:         return inspect.isclass(inferred_context_var)
73: 
74:     # Tuples
75:     if expected_var is types.TupleType:
76:         return isinstance(inferred_context_var, tuple)
77: 
78:     # Object instances
79:     if expected_var is types.InstanceType:
80:         return type(inferred_context_var) is types.InstanceType
81:         #return inferred_context_var.get_python_type() == types.InstanceType
82: 
83:     return expected_var == inferred_context_var
84: 
85: 
86: def check_type_store(type_store, executed_file, verbose):
87:     '''
88:     This functions picks a type store of the source code of a file, calculate its associated type data file, loads
89:     it and compare variable per variable the type store type of all variables against the one declared in the type
90:     data file, printing found errors
91:     :param type_store: Type store of the program
92:     :param executed_file: File to load the attached type data file
93:     :param verbose: Verbose output? (bool)
94:     :return: 0 (No error), 1 (Type mismatch in at least one variable), 2 (no associated type data file found)
95:     '''
96:     dirname = os.path.dirname(executed_file) + "/" + type_inference_file_directory_name + "/"
97:     filename = executed_file.split("/")[-1].split(".")[0]
98:     sys.path.append(dirname)
99: 
100:     data_file = filename + type_data_file_postfix
101:     result = 0
102: 
103:     try:
104:         data = __import__(data_file)
105: 
106:         expected_types = data.test_types
107: 
108:         for context_name in expected_types:
109:             inferred_context = type_store.get_last_function_context_for(context_name)
110:             expected_vars = expected_types[context_name]
111:             for var in __filter_reserved_vars(expected_vars):
112:                 if not __equal_types(expected_vars[var], inferred_context[var]):
113:                     #if verbose:
114:                     print "Type mismatch for name '{0}' in context '{3}': {1} expected, but {2} found".format(var,
115:                                                                                                                 expected_vars[
116:                                                                                                                     var],
117:                                                                                                                 inferred_context[
118:                                                                                                                     var],
119:                                                                                                                 context_name)
120:                     result = 1  # Error: Inferred types are not the ones that we expected
121:     except Exception as exc:
122:         if verbose:
123:             print "Type checking error: " + str(exc)
124:         return 2  # Error: Data file not found or some error happened during variable testing
125:     finally:
126:         sys.path.remove(dirname)
127: 
128:     if verbose and result == 0:
129:         print "All checks OK"
130: 
131:     return result  # No Error
132: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import os' statement (line 1)
import os

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import sys' statement (line 2)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import inspect' statement (line 3)
import inspect

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'inspect', inspect, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import types' statement (line 4)
import types

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'types', types, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from stypy_copy.python_lib_copy.python_types_copy.type_copy import Type' statement (line 6)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/checking/')
import_2784 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_copy')

if (type(import_2784) is not StypyTypeError):

    if (import_2784 != 'pyd_module'):
        __import__(import_2784)
        sys_modules_2785 = sys.modules[import_2784]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_copy', sys_modules_2785.module_type_store, module_type_store, ['Type'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_2785, sys_modules_2785.module_type_store, module_type_store)
    else:
        from stypy_copy.python_lib_copy.python_types_copy.type_copy import Type

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_copy', None, module_type_store, ['Type'], [Type])

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.python_types_copy.type_copy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_copy', import_2784)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/checking/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.union_type_copy import UnionType' statement (line 7)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/checking/')
import_2786 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.union_type_copy')

if (type(import_2786) is not StypyTypeError):

    if (import_2786 != 'pyd_module'):
        __import__(import_2786)
        sys_modules_2787 = sys.modules[import_2786]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.union_type_copy', sys_modules_2787.module_type_store, module_type_store, ['UnionType'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_2787, sys_modules_2787.module_type_store, module_type_store)
    else:
        from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.union_type_copy import UnionType

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.union_type_copy', None, module_type_store, ['UnionType'], [UnionType])

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.union_type_copy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.union_type_copy', import_2786)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/checking/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy import UndefinedType' statement (line 8)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/checking/')
import_2788 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy')

if (type(import_2788) is not StypyTypeError):

    if (import_2788 != 'pyd_module'):
        __import__(import_2788)
        sys_modules_2789 = sys.modules[import_2788]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy', sys_modules_2789.module_type_store, module_type_store, ['UndefinedType'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_2789, sys_modules_2789.module_type_store, module_type_store)
    else:
        from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy import UndefinedType

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy', None, module_type_store, ['UndefinedType'], [UndefinedType])

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy', import_2788)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/checking/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from stypy_copy.stypy_parameters_copy import type_inference_file_directory_name, type_data_file_postfix' statement (line 9)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/checking/')
import_2790 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_copy.stypy_parameters_copy')

if (type(import_2790) is not StypyTypeError):

    if (import_2790 != 'pyd_module'):
        __import__(import_2790)
        sys_modules_2791 = sys.modules[import_2790]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_copy.stypy_parameters_copy', sys_modules_2791.module_type_store, module_type_store, ['type_inference_file_directory_name', 'type_data_file_postfix'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_2791, sys_modules_2791.module_type_store, module_type_store)
    else:
        from stypy_copy.stypy_parameters_copy import type_inference_file_directory_name, type_data_file_postfix

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_copy.stypy_parameters_copy', None, module_type_store, ['type_inference_file_directory_name', 'type_data_file_postfix'], [type_inference_file_directory_name, type_data_file_postfix])

else:
    # Assigning a type to the variable 'stypy_copy.stypy_parameters_copy' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_copy.stypy_parameters_copy', import_2790)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/checking/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from stypy_copy.errors_copy.type_error import TypeError' statement (line 10)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/checking/')
import_2792 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_copy.errors_copy.type_error')

if (type(import_2792) is not StypyTypeError):

    if (import_2792 != 'pyd_module'):
        __import__(import_2792)
        sys_modules_2793 = sys.modules[import_2792]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_copy.errors_copy.type_error', sys_modules_2793.module_type_store, module_type_store, ['TypeError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_2793, sys_modules_2793.module_type_store, module_type_store)
    else:
        from stypy_copy.errors_copy.type_error import TypeError

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_copy.errors_copy.type_error', None, module_type_store, ['TypeError'], [TypeError])

else:
    # Assigning a type to the variable 'stypy_copy.errors_copy.type_error' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_copy.errors_copy.type_error', import_2792)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/checking/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from stypy_copy.type_store_copy import typestore_copy' statement (line 11)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/checking/')
import_2794 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'stypy_copy.type_store_copy')

if (type(import_2794) is not StypyTypeError):

    if (import_2794 != 'pyd_module'):
        __import__(import_2794)
        sys_modules_2795 = sys.modules[import_2794]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'stypy_copy.type_store_copy', sys_modules_2795.module_type_store, module_type_store, ['typestore_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_2795, sys_modules_2795.module_type_store, module_type_store)
    else:
        from stypy_copy.type_store_copy import typestore_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'stypy_copy.type_store_copy', None, module_type_store, ['typestore_copy'], [typestore_copy])

else:
    # Assigning a type to the variable 'stypy_copy.type_store_copy' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'stypy_copy.type_store_copy', import_2794)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/code_generation_copy/type_inference_programs_copy/checking/')

str_2796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, (-1)), 'str', '\nFile with functions that are used when unit testing the generated type inference code checking the type inference code\ntype store against the type data file of the checked program\n')

@norecursion
def __filter_reserved_vars(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__filter_reserved_vars'
    module_type_store = module_type_store.open_function_context('__filter_reserved_vars', 19, 0, False)
    
    # Passed parameters checking function
    __filter_reserved_vars.stypy_localization = localization
    __filter_reserved_vars.stypy_type_of_self = None
    __filter_reserved_vars.stypy_type_store = module_type_store
    __filter_reserved_vars.stypy_function_name = '__filter_reserved_vars'
    __filter_reserved_vars.stypy_param_names_list = ['types_']
    __filter_reserved_vars.stypy_varargs_param_name = None
    __filter_reserved_vars.stypy_kwargs_param_name = None
    __filter_reserved_vars.stypy_call_defaults = defaults
    __filter_reserved_vars.stypy_call_varargs = varargs
    __filter_reserved_vars.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__filter_reserved_vars', ['types_'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__filter_reserved_vars', localization, ['types_'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__filter_reserved_vars(...)' code ##################

    str_2797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, (-1)), 'str', '\n    For the types_ list, eliminates the references to the TypeDataFileWriter class, to not to check this private object\n    not part of the original program.\n    :param types_: Type list\n    :return:\n    ')
    
    # Call to filter(...): (line 26)
    # Processing the call arguments (line 26)

    @norecursion
    def _stypy_temp_lambda_2(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_stypy_temp_lambda_2'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_2', 26, 18, True)
        # Passed parameters checking function
        _stypy_temp_lambda_2.stypy_localization = localization
        _stypy_temp_lambda_2.stypy_type_of_self = None
        _stypy_temp_lambda_2.stypy_type_store = module_type_store
        _stypy_temp_lambda_2.stypy_function_name = '_stypy_temp_lambda_2'
        _stypy_temp_lambda_2.stypy_param_names_list = ['elem']
        _stypy_temp_lambda_2.stypy_varargs_param_name = None
        _stypy_temp_lambda_2.stypy_kwargs_param_name = None
        _stypy_temp_lambda_2.stypy_call_defaults = defaults
        _stypy_temp_lambda_2.stypy_call_varargs = varargs
        _stypy_temp_lambda_2.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_2', ['elem'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_2', ['elem'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        
        str_2799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 35), 'str', 'TypeDataFileWriter')
        # Getting the type of 'elem' (line 26)
        elem_2800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 59), 'elem', False)
        # Applying the binary operator '==' (line 26)
        result_eq_2801 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 35), '==', str_2799, elem_2800)
        
        # Applying the 'not' unary operator (line 26)
        result_not__2802 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 31), 'not', result_eq_2801)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 18), 'stypy_return_type', result_not__2802)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_2' in the type store
        # Getting the type of 'stypy_return_type' (line 26)
        stypy_return_type_2803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 18), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2803)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_2'
        return stypy_return_type_2803

    # Assigning a type to the variable '_stypy_temp_lambda_2' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 18), '_stypy_temp_lambda_2', _stypy_temp_lambda_2)
    # Getting the type of '_stypy_temp_lambda_2' (line 26)
    _stypy_temp_lambda_2_2804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 18), '_stypy_temp_lambda_2')
    # Getting the type of 'types_' (line 26)
    types__2805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 65), 'types_', False)
    # Processing the call keyword arguments (line 26)
    kwargs_2806 = {}
    # Getting the type of 'filter' (line 26)
    filter_2798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 11), 'filter', False)
    # Calling filter(args, kwargs) (line 26)
    filter_call_result_2807 = invoke(stypy.reporting.localization.Localization(__file__, 26, 11), filter_2798, *[_stypy_temp_lambda_2_2804, types__2805], **kwargs_2806)
    
    # Assigning a type to the variable 'stypy_return_type' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type', filter_call_result_2807)
    
    # ################# End of '__filter_reserved_vars(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__filter_reserved_vars' in the type store
    # Getting the type of 'stypy_return_type' (line 19)
    stypy_return_type_2808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2808)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__filter_reserved_vars'
    return stypy_return_type_2808

# Assigning a type to the variable '__filter_reserved_vars' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), '__filter_reserved_vars', __filter_reserved_vars)

@norecursion
def __equal_types(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__equal_types'
    module_type_store = module_type_store.open_function_context('__equal_types', 29, 0, False)
    
    # Passed parameters checking function
    __equal_types.stypy_localization = localization
    __equal_types.stypy_type_of_self = None
    __equal_types.stypy_type_store = module_type_store
    __equal_types.stypy_function_name = '__equal_types'
    __equal_types.stypy_param_names_list = ['expected_var', 'inferred_context_var']
    __equal_types.stypy_varargs_param_name = None
    __equal_types.stypy_kwargs_param_name = None
    __equal_types.stypy_call_defaults = defaults
    __equal_types.stypy_call_varargs = varargs
    __equal_types.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '__equal_types', ['expected_var', 'inferred_context_var'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '__equal_types', localization, ['expected_var', 'inferred_context_var'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__equal_types(...)' code ##################

    str_2809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, (-1)), 'str', '\n    Helper function to check if the types of two vars can be considered equal from a unit testing point of view\n    :param expected_var: Expected type\n    :param inferred_context_var: Inferred type\n    :return: bool\n    ')
    
    # Getting the type of 'expected_var' (line 38)
    expected_var_2810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 7), 'expected_var')
    # Getting the type of 'inferred_context_var' (line 38)
    inferred_context_var_2811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 23), 'inferred_context_var')
    # Applying the binary operator '==' (line 38)
    result_eq_2812 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 7), '==', expected_var_2810, inferred_context_var_2811)
    
    # Testing if the type of an if condition is none (line 38)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 38, 4), result_eq_2812):
        pass
    else:
        
        # Testing the type of an if condition (line 38)
        if_condition_2813 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 38, 4), result_eq_2812)
        # Assigning a type to the variable 'if_condition_2813' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'if_condition_2813', if_condition_2813)
        # SSA begins for if statement (line 38)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'True' (line 39)
        True_2814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'stypy_return_type', True_2814)
        # SSA join for if statement (line 38)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to isinstance(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'inferred_context_var' (line 42)
    inferred_context_var_2816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 18), 'inferred_context_var', False)
    # Getting the type of 'Type' (line 42)
    Type_2817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 40), 'Type', False)
    # Processing the call keyword arguments (line 42)
    kwargs_2818 = {}
    # Getting the type of 'isinstance' (line 42)
    isinstance_2815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 42)
    isinstance_call_result_2819 = invoke(stypy.reporting.localization.Localization(__file__, 42, 7), isinstance_2815, *[inferred_context_var_2816, Type_2817], **kwargs_2818)
    
    # Testing if the type of an if condition is none (line 42)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 42, 4), isinstance_call_result_2819):
        pass
    else:
        
        # Testing the type of an if condition (line 42)
        if_condition_2820 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 4), isinstance_call_result_2819)
        # Assigning a type to the variable 'if_condition_2820' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'if_condition_2820', if_condition_2820)
        # SSA begins for if statement (line 42)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'expected_var' (line 44)
        expected_var_2821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 11), 'expected_var')
        # Getting the type of 'types' (line 44)
        types_2822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 27), 'types')
        # Obtaining the member 'ModuleType' of a type (line 44)
        ModuleType_2823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 27), types_2822, 'ModuleType')
        # Applying the binary operator 'is' (line 44)
        result_is__2824 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 11), 'is', expected_var_2821, ModuleType_2823)
        
        # Testing if the type of an if condition is none (line 44)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 44, 8), result_is__2824):
            pass
        else:
            
            # Testing the type of an if condition (line 44)
            if_condition_2825 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 44, 8), result_is__2824)
            # Assigning a type to the variable 'if_condition_2825' (line 44)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'if_condition_2825', if_condition_2825)
            # SSA begins for if statement (line 44)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Evaluating a boolean operation
            
            # Call to ismodule(...): (line 45)
            # Processing the call arguments (line 45)
            
            # Call to get_python_entity(...): (line 45)
            # Processing the call keyword arguments (line 45)
            kwargs_2830 = {}
            # Getting the type of 'inferred_context_var' (line 45)
            inferred_context_var_2828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 36), 'inferred_context_var', False)
            # Obtaining the member 'get_python_entity' of a type (line 45)
            get_python_entity_2829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 36), inferred_context_var_2828, 'get_python_entity')
            # Calling get_python_entity(args, kwargs) (line 45)
            get_python_entity_call_result_2831 = invoke(stypy.reporting.localization.Localization(__file__, 45, 36), get_python_entity_2829, *[], **kwargs_2830)
            
            # Processing the call keyword arguments (line 45)
            kwargs_2832 = {}
            # Getting the type of 'inspect' (line 45)
            inspect_2826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 19), 'inspect', False)
            # Obtaining the member 'ismodule' of a type (line 45)
            ismodule_2827 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 19), inspect_2826, 'ismodule')
            # Calling ismodule(args, kwargs) (line 45)
            ismodule_call_result_2833 = invoke(stypy.reporting.localization.Localization(__file__, 45, 19), ismodule_2827, *[get_python_entity_call_result_2831], **kwargs_2832)
            
            
            # Call to isinstance(...): (line 45)
            # Processing the call arguments (line 45)
            # Getting the type of 'inferred_context_var' (line 45)
            inferred_context_var_2835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 92), 'inferred_context_var', False)
            # Getting the type of 'typestore_copy' (line 45)
            typestore_copy_2836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 114), 'typestore_copy', False)
            # Obtaining the member 'TypeStore' of a type (line 45)
            TypeStore_2837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 114), typestore_copy_2836, 'TypeStore')
            # Processing the call keyword arguments (line 45)
            kwargs_2838 = {}
            # Getting the type of 'isinstance' (line 45)
            isinstance_2834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 81), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 45)
            isinstance_call_result_2839 = invoke(stypy.reporting.localization.Localization(__file__, 45, 81), isinstance_2834, *[inferred_context_var_2835, TypeStore_2837], **kwargs_2838)
            
            # Applying the binary operator 'or' (line 45)
            result_or_keyword_2840 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 19), 'or', ismodule_call_result_2833, isinstance_call_result_2839)
            
            # Assigning a type to the variable 'stypy_return_type' (line 45)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'stypy_return_type', result_or_keyword_2840)
            # SSA join for if statement (line 44)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'expected_var' (line 47)
        expected_var_2841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 11), 'expected_var')
        # Getting the type of 'types' (line 47)
        types_2842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 27), 'types')
        # Obtaining the member 'ClassType' of a type (line 47)
        ClassType_2843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 27), types_2842, 'ClassType')
        # Applying the binary operator 'is' (line 47)
        result_is__2844 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 11), 'is', expected_var_2841, ClassType_2843)
        
        # Testing if the type of an if condition is none (line 47)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 47, 8), result_is__2844):
            pass
        else:
            
            # Testing the type of an if condition (line 47)
            if_condition_2845 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 47, 8), result_is__2844)
            # Assigning a type to the variable 'if_condition_2845' (line 47)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'if_condition_2845', if_condition_2845)
            # SSA begins for if statement (line 47)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to isclass(...): (line 48)
            # Processing the call arguments (line 48)
            
            # Call to get_python_type(...): (line 48)
            # Processing the call keyword arguments (line 48)
            kwargs_2850 = {}
            # Getting the type of 'inferred_context_var' (line 48)
            inferred_context_var_2848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 35), 'inferred_context_var', False)
            # Obtaining the member 'get_python_type' of a type (line 48)
            get_python_type_2849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 35), inferred_context_var_2848, 'get_python_type')
            # Calling get_python_type(args, kwargs) (line 48)
            get_python_type_call_result_2851 = invoke(stypy.reporting.localization.Localization(__file__, 48, 35), get_python_type_2849, *[], **kwargs_2850)
            
            # Processing the call keyword arguments (line 48)
            kwargs_2852 = {}
            # Getting the type of 'inspect' (line 48)
            inspect_2846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 19), 'inspect', False)
            # Obtaining the member 'isclass' of a type (line 48)
            isclass_2847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 19), inspect_2846, 'isclass')
            # Calling isclass(args, kwargs) (line 48)
            isclass_call_result_2853 = invoke(stypy.reporting.localization.Localization(__file__, 48, 19), isclass_2847, *[get_python_type_call_result_2851], **kwargs_2852)
            
            # Assigning a type to the variable 'stypy_return_type' (line 48)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'stypy_return_type', isclass_call_result_2853)
            # SSA join for if statement (line 47)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'expected_var' (line 50)
        expected_var_2854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 11), 'expected_var')
        # Getting the type of 'TypeError' (line 50)
        TypeError_2855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 27), 'TypeError')
        # Applying the binary operator 'is' (line 50)
        result_is__2856 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 11), 'is', expected_var_2854, TypeError_2855)
        
        # Testing if the type of an if condition is none (line 50)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 50, 8), result_is__2856):
            pass
        else:
            
            # Testing the type of an if condition (line 50)
            if_condition_2857 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 8), result_is__2856)
            # Assigning a type to the variable 'if_condition_2857' (line 50)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'if_condition_2857', if_condition_2857)
            # SSA begins for if statement (line 50)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to isinstance(...): (line 51)
            # Processing the call arguments (line 51)
            # Getting the type of 'inferred_context_var' (line 51)
            inferred_context_var_2859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 30), 'inferred_context_var', False)
            # Getting the type of 'TypeError' (line 51)
            TypeError_2860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 52), 'TypeError', False)
            # Processing the call keyword arguments (line 51)
            kwargs_2861 = {}
            # Getting the type of 'isinstance' (line 51)
            isinstance_2858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 19), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 51)
            isinstance_call_result_2862 = invoke(stypy.reporting.localization.Localization(__file__, 51, 19), isinstance_2858, *[inferred_context_var_2859, TypeError_2860], **kwargs_2861)
            
            # Assigning a type to the variable 'stypy_return_type' (line 51)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'stypy_return_type', isinstance_call_result_2862)
            # SSA join for if statement (line 50)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Compare to a Name (line 53):
        
        
        # Call to get_python_type(...): (line 53)
        # Processing the call keyword arguments (line 53)
        kwargs_2865 = {}
        # Getting the type of 'inferred_context_var' (line 53)
        inferred_context_var_2863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 22), 'inferred_context_var', False)
        # Obtaining the member 'get_python_type' of a type (line 53)
        get_python_type_2864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 22), inferred_context_var_2863, 'get_python_type')
        # Calling get_python_type(args, kwargs) (line 53)
        get_python_type_call_result_2866 = invoke(stypy.reporting.localization.Localization(__file__, 53, 22), get_python_type_2864, *[], **kwargs_2865)
        
        # Getting the type of 'expected_var' (line 53)
        expected_var_2867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 64), 'expected_var')
        # Applying the binary operator '==' (line 53)
        result_eq_2868 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 22), '==', get_python_type_call_result_2866, expected_var_2867)
        
        # Assigning a type to the variable 'direct_comp' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'direct_comp', result_eq_2868)
        
        # Evaluating a boolean operation
        
        # Getting the type of 'direct_comp' (line 54)
        direct_comp_2869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 15), 'direct_comp')
        # Applying the 'not' unary operator (line 54)
        result_not__2870 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 11), 'not', direct_comp_2869)
        
        
        # Call to isinstance(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'expected_var' (line 54)
        expected_var_2872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 42), 'expected_var', False)
        # Getting the type of 'UnionType' (line 54)
        UnionType_2873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 56), 'UnionType', False)
        # Processing the call keyword arguments (line 54)
        kwargs_2874 = {}
        # Getting the type of 'isinstance' (line 54)
        isinstance_2871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 31), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 54)
        isinstance_call_result_2875 = invoke(stypy.reporting.localization.Localization(__file__, 54, 31), isinstance_2871, *[expected_var_2872, UnionType_2873], **kwargs_2874)
        
        # Applying the binary operator 'and' (line 54)
        result_and_keyword_2876 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 11), 'and', result_not__2870, isinstance_call_result_2875)
        
        # Call to isinstance(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'inferred_context_var' (line 54)
        inferred_context_var_2878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 82), 'inferred_context_var', False)
        # Getting the type of 'UnionType' (line 54)
        UnionType_2879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 104), 'UnionType', False)
        # Processing the call keyword arguments (line 54)
        kwargs_2880 = {}
        # Getting the type of 'isinstance' (line 54)
        isinstance_2877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 71), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 54)
        isinstance_call_result_2881 = invoke(stypy.reporting.localization.Localization(__file__, 54, 71), isinstance_2877, *[inferred_context_var_2878, UnionType_2879], **kwargs_2880)
        
        # Applying the binary operator 'and' (line 54)
        result_and_keyword_2882 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 11), 'and', result_and_keyword_2876, isinstance_call_result_2881)
        
        # Testing if the type of an if condition is none (line 54)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 54, 8), result_and_keyword_2882):
            pass
        else:
            
            # Testing the type of an if condition (line 54)
            if_condition_2883 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 54, 8), result_and_keyword_2882)
            # Assigning a type to the variable 'if_condition_2883' (line 54)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'if_condition_2883', if_condition_2883)
            # SSA begins for if statement (line 54)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # Call to len(...): (line 55)
            # Processing the call arguments (line 55)
            # Getting the type of 'expected_var' (line 55)
            expected_var_2885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 23), 'expected_var', False)
            # Obtaining the member 'types' of a type (line 55)
            types_2886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 23), expected_var_2885, 'types')
            # Processing the call keyword arguments (line 55)
            kwargs_2887 = {}
            # Getting the type of 'len' (line 55)
            len_2884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 19), 'len', False)
            # Calling len(args, kwargs) (line 55)
            len_call_result_2888 = invoke(stypy.reporting.localization.Localization(__file__, 55, 19), len_2884, *[types_2886], **kwargs_2887)
            
            
            # Call to len(...): (line 55)
            # Processing the call arguments (line 55)
            # Getting the type of 'inferred_context_var' (line 55)
            inferred_context_var_2890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 50), 'inferred_context_var', False)
            # Obtaining the member 'types' of a type (line 55)
            types_2891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 50), inferred_context_var_2890, 'types')
            # Processing the call keyword arguments (line 55)
            kwargs_2892 = {}
            # Getting the type of 'len' (line 55)
            len_2889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 46), 'len', False)
            # Calling len(args, kwargs) (line 55)
            len_call_result_2893 = invoke(stypy.reporting.localization.Localization(__file__, 55, 46), len_2889, *[types_2891], **kwargs_2892)
            
            # Applying the binary operator '==' (line 55)
            result_eq_2894 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 19), '==', len_call_result_2888, len_call_result_2893)
            
            # Assigning a type to the variable 'stypy_return_type' (line 55)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'stypy_return_type', result_eq_2894)
            # SSA join for if statement (line 54)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'direct_comp' (line 56)
        direct_comp_2895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 15), 'direct_comp')
        # Assigning a type to the variable 'stypy_return_type' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'stypy_return_type', direct_comp_2895)
        # SSA join for if statement (line 42)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'expected_var' (line 59)
    expected_var_2896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 7), 'expected_var')
    # Getting the type of 'types' (line 59)
    types_2897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 23), 'types')
    # Obtaining the member 'FunctionType' of a type (line 59)
    FunctionType_2898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 23), types_2897, 'FunctionType')
    # Applying the binary operator '==' (line 59)
    result_eq_2899 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 7), '==', expected_var_2896, FunctionType_2898)
    
    # Testing if the type of an if condition is none (line 59)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 59, 4), result_eq_2899):
        pass
    else:
        
        # Testing the type of an if condition (line 59)
        if_condition_2900 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 59, 4), result_eq_2899)
        # Assigning a type to the variable 'if_condition_2900' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'if_condition_2900', if_condition_2900)
        # SSA begins for if statement (line 59)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to isfunction(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'inferred_context_var' (line 60)
        inferred_context_var_2903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 34), 'inferred_context_var', False)
        # Processing the call keyword arguments (line 60)
        kwargs_2904 = {}
        # Getting the type of 'inspect' (line 60)
        inspect_2901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 15), 'inspect', False)
        # Obtaining the member 'isfunction' of a type (line 60)
        isfunction_2902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 15), inspect_2901, 'isfunction')
        # Calling isfunction(args, kwargs) (line 60)
        isfunction_call_result_2905 = invoke(stypy.reporting.localization.Localization(__file__, 60, 15), isfunction_2902, *[inferred_context_var_2903], **kwargs_2904)
        
        # Assigning a type to the variable 'stypy_return_type' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'stypy_return_type', isfunction_call_result_2905)
        # SSA join for if statement (line 59)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'expected_var' (line 63)
    expected_var_2906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 7), 'expected_var')
    # Getting the type of 'types' (line 63)
    types_2907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 23), 'types')
    # Obtaining the member 'BuiltinFunctionType' of a type (line 63)
    BuiltinFunctionType_2908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 23), types_2907, 'BuiltinFunctionType')
    # Applying the binary operator '==' (line 63)
    result_eq_2909 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 7), '==', expected_var_2906, BuiltinFunctionType_2908)
    
    # Testing if the type of an if condition is none (line 63)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 63, 4), result_eq_2909):
        pass
    else:
        
        # Testing the type of an if condition (line 63)
        if_condition_2910 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 63, 4), result_eq_2909)
        # Assigning a type to the variable 'if_condition_2910' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'if_condition_2910', if_condition_2910)
        # SSA begins for if statement (line 63)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to isfunction(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'inferred_context_var' (line 64)
        inferred_context_var_2913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 34), 'inferred_context_var', False)
        # Processing the call keyword arguments (line 64)
        kwargs_2914 = {}
        # Getting the type of 'inspect' (line 64)
        inspect_2911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 15), 'inspect', False)
        # Obtaining the member 'isfunction' of a type (line 64)
        isfunction_2912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 15), inspect_2911, 'isfunction')
        # Calling isfunction(args, kwargs) (line 64)
        isfunction_call_result_2915 = invoke(stypy.reporting.localization.Localization(__file__, 64, 15), isfunction_2912, *[inferred_context_var_2913], **kwargs_2914)
        
        # Assigning a type to the variable 'stypy_return_type' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'stypy_return_type', isfunction_call_result_2915)
        # SSA join for if statement (line 63)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to isinstance(...): (line 67)
    # Processing the call arguments (line 67)
    # Getting the type of 'inferred_context_var' (line 67)
    inferred_context_var_2917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 18), 'inferred_context_var', False)
    # Getting the type of 'UndefinedType' (line 67)
    UndefinedType_2918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 40), 'UndefinedType', False)
    # Processing the call keyword arguments (line 67)
    kwargs_2919 = {}
    # Getting the type of 'isinstance' (line 67)
    isinstance_2916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 7), 'isinstance', False)
    # Calling isinstance(args, kwargs) (line 67)
    isinstance_call_result_2920 = invoke(stypy.reporting.localization.Localization(__file__, 67, 7), isinstance_2916, *[inferred_context_var_2917, UndefinedType_2918], **kwargs_2919)
    
    # Testing if the type of an if condition is none (line 67)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 67, 4), isinstance_call_result_2920):
        pass
    else:
        
        # Testing the type of an if condition (line 67)
        if_condition_2921 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 4), isinstance_call_result_2920)
        # Assigning a type to the variable 'if_condition_2921' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'if_condition_2921', if_condition_2921)
        # SSA begins for if statement (line 67)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to isinstance(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'expected_var' (line 68)
        expected_var_2923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 26), 'expected_var', False)
        # Getting the type of 'UndefinedType' (line 68)
        UndefinedType_2924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 40), 'UndefinedType', False)
        # Processing the call keyword arguments (line 68)
        kwargs_2925 = {}
        # Getting the type of 'isinstance' (line 68)
        isinstance_2922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 68)
        isinstance_call_result_2926 = invoke(stypy.reporting.localization.Localization(__file__, 68, 15), isinstance_2922, *[expected_var_2923, UndefinedType_2924], **kwargs_2925)
        
        # Assigning a type to the variable 'stypy_return_type' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'stypy_return_type', isinstance_call_result_2926)
        # SSA join for if statement (line 67)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'expected_var' (line 71)
    expected_var_2927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 7), 'expected_var')
    # Getting the type of 'types' (line 71)
    types_2928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 23), 'types')
    # Obtaining the member 'ClassType' of a type (line 71)
    ClassType_2929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 23), types_2928, 'ClassType')
    # Applying the binary operator 'is' (line 71)
    result_is__2930 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 7), 'is', expected_var_2927, ClassType_2929)
    
    # Testing if the type of an if condition is none (line 71)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 71, 4), result_is__2930):
        pass
    else:
        
        # Testing the type of an if condition (line 71)
        if_condition_2931 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 4), result_is__2930)
        # Assigning a type to the variable 'if_condition_2931' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'if_condition_2931', if_condition_2931)
        # SSA begins for if statement (line 71)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to isclass(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'inferred_context_var' (line 72)
        inferred_context_var_2934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 31), 'inferred_context_var', False)
        # Processing the call keyword arguments (line 72)
        kwargs_2935 = {}
        # Getting the type of 'inspect' (line 72)
        inspect_2932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 15), 'inspect', False)
        # Obtaining the member 'isclass' of a type (line 72)
        isclass_2933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 15), inspect_2932, 'isclass')
        # Calling isclass(args, kwargs) (line 72)
        isclass_call_result_2936 = invoke(stypy.reporting.localization.Localization(__file__, 72, 15), isclass_2933, *[inferred_context_var_2934], **kwargs_2935)
        
        # Assigning a type to the variable 'stypy_return_type' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'stypy_return_type', isclass_call_result_2936)
        # SSA join for if statement (line 71)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'expected_var' (line 75)
    expected_var_2937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 7), 'expected_var')
    # Getting the type of 'types' (line 75)
    types_2938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 23), 'types')
    # Obtaining the member 'TupleType' of a type (line 75)
    TupleType_2939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 23), types_2938, 'TupleType')
    # Applying the binary operator 'is' (line 75)
    result_is__2940 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 7), 'is', expected_var_2937, TupleType_2939)
    
    # Testing if the type of an if condition is none (line 75)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 75, 4), result_is__2940):
        pass
    else:
        
        # Testing the type of an if condition (line 75)
        if_condition_2941 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 4), result_is__2940)
        # Assigning a type to the variable 'if_condition_2941' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'if_condition_2941', if_condition_2941)
        # SSA begins for if statement (line 75)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to isinstance(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'inferred_context_var' (line 76)
        inferred_context_var_2943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 26), 'inferred_context_var', False)
        # Getting the type of 'tuple' (line 76)
        tuple_2944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 48), 'tuple', False)
        # Processing the call keyword arguments (line 76)
        kwargs_2945 = {}
        # Getting the type of 'isinstance' (line 76)
        isinstance_2942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 76)
        isinstance_call_result_2946 = invoke(stypy.reporting.localization.Localization(__file__, 76, 15), isinstance_2942, *[inferred_context_var_2943, tuple_2944], **kwargs_2945)
        
        # Assigning a type to the variable 'stypy_return_type' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'stypy_return_type', isinstance_call_result_2946)
        # SSA join for if statement (line 75)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'expected_var' (line 79)
    expected_var_2947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 7), 'expected_var')
    # Getting the type of 'types' (line 79)
    types_2948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 23), 'types')
    # Obtaining the member 'InstanceType' of a type (line 79)
    InstanceType_2949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 23), types_2948, 'InstanceType')
    # Applying the binary operator 'is' (line 79)
    result_is__2950 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 7), 'is', expected_var_2947, InstanceType_2949)
    
    # Testing if the type of an if condition is none (line 79)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 79, 4), result_is__2950):
        pass
    else:
        
        # Testing the type of an if condition (line 79)
        if_condition_2951 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 4), result_is__2950)
        # Assigning a type to the variable 'if_condition_2951' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'if_condition_2951', if_condition_2951)
        # SSA begins for if statement (line 79)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Call to type(...): (line 80)
        # Processing the call arguments (line 80)
        # Getting the type of 'inferred_context_var' (line 80)
        inferred_context_var_2953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 20), 'inferred_context_var', False)
        # Processing the call keyword arguments (line 80)
        kwargs_2954 = {}
        # Getting the type of 'type' (line 80)
        type_2952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 15), 'type', False)
        # Calling type(args, kwargs) (line 80)
        type_call_result_2955 = invoke(stypy.reporting.localization.Localization(__file__, 80, 15), type_2952, *[inferred_context_var_2953], **kwargs_2954)
        
        # Getting the type of 'types' (line 80)
        types_2956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 45), 'types')
        # Obtaining the member 'InstanceType' of a type (line 80)
        InstanceType_2957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 45), types_2956, 'InstanceType')
        # Applying the binary operator 'is' (line 80)
        result_is__2958 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 15), 'is', type_call_result_2955, InstanceType_2957)
        
        # Assigning a type to the variable 'stypy_return_type' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'stypy_return_type', result_is__2958)
        # SSA join for if statement (line 79)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Getting the type of 'expected_var' (line 83)
    expected_var_2959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 11), 'expected_var')
    # Getting the type of 'inferred_context_var' (line 83)
    inferred_context_var_2960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 27), 'inferred_context_var')
    # Applying the binary operator '==' (line 83)
    result_eq_2961 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 11), '==', expected_var_2959, inferred_context_var_2960)
    
    # Assigning a type to the variable 'stypy_return_type' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'stypy_return_type', result_eq_2961)
    
    # ################# End of '__equal_types(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '__equal_types' in the type store
    # Getting the type of 'stypy_return_type' (line 29)
    stypy_return_type_2962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2962)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '__equal_types'
    return stypy_return_type_2962

# Assigning a type to the variable '__equal_types' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), '__equal_types', __equal_types)

@norecursion
def check_type_store(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_type_store'
    module_type_store = module_type_store.open_function_context('check_type_store', 86, 0, False)
    
    # Passed parameters checking function
    check_type_store.stypy_localization = localization
    check_type_store.stypy_type_of_self = None
    check_type_store.stypy_type_store = module_type_store
    check_type_store.stypy_function_name = 'check_type_store'
    check_type_store.stypy_param_names_list = ['type_store', 'executed_file', 'verbose']
    check_type_store.stypy_varargs_param_name = None
    check_type_store.stypy_kwargs_param_name = None
    check_type_store.stypy_call_defaults = defaults
    check_type_store.stypy_call_varargs = varargs
    check_type_store.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_type_store', ['type_store', 'executed_file', 'verbose'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_type_store', localization, ['type_store', 'executed_file', 'verbose'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_type_store(...)' code ##################

    str_2963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, (-1)), 'str', '\n    This functions picks a type store of the source code of a file, calculate its associated type data file, loads\n    it and compare variable per variable the type store type of all variables against the one declared in the type\n    data file, printing found errors\n    :param type_store: Type store of the program\n    :param executed_file: File to load the attached type data file\n    :param verbose: Verbose output? (bool)\n    :return: 0 (No error), 1 (Type mismatch in at least one variable), 2 (no associated type data file found)\n    ')
    
    # Assigning a BinOp to a Name (line 96):
    
    # Call to dirname(...): (line 96)
    # Processing the call arguments (line 96)
    # Getting the type of 'executed_file' (line 96)
    executed_file_2967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 30), 'executed_file', False)
    # Processing the call keyword arguments (line 96)
    kwargs_2968 = {}
    # Getting the type of 'os' (line 96)
    os_2964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 14), 'os', False)
    # Obtaining the member 'path' of a type (line 96)
    path_2965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 14), os_2964, 'path')
    # Obtaining the member 'dirname' of a type (line 96)
    dirname_2966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 14), path_2965, 'dirname')
    # Calling dirname(args, kwargs) (line 96)
    dirname_call_result_2969 = invoke(stypy.reporting.localization.Localization(__file__, 96, 14), dirname_2966, *[executed_file_2967], **kwargs_2968)
    
    str_2970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 47), 'str', '/')
    # Applying the binary operator '+' (line 96)
    result_add_2971 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 14), '+', dirname_call_result_2969, str_2970)
    
    # Getting the type of 'type_inference_file_directory_name' (line 96)
    type_inference_file_directory_name_2972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 53), 'type_inference_file_directory_name')
    # Applying the binary operator '+' (line 96)
    result_add_2973 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 51), '+', result_add_2971, type_inference_file_directory_name_2972)
    
    str_2974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 90), 'str', '/')
    # Applying the binary operator '+' (line 96)
    result_add_2975 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 88), '+', result_add_2973, str_2974)
    
    # Assigning a type to the variable 'dirname' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'dirname', result_add_2975)
    
    # Assigning a Subscript to a Name (line 97):
    
    # Obtaining the type of the subscript
    int_2976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 55), 'int')
    
    # Call to split(...): (line 97)
    # Processing the call arguments (line 97)
    str_2986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 50), 'str', '.')
    # Processing the call keyword arguments (line 97)
    kwargs_2987 = {}
    
    # Obtaining the type of the subscript
    int_2977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 40), 'int')
    
    # Call to split(...): (line 97)
    # Processing the call arguments (line 97)
    str_2980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 35), 'str', '/')
    # Processing the call keyword arguments (line 97)
    kwargs_2981 = {}
    # Getting the type of 'executed_file' (line 97)
    executed_file_2978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 15), 'executed_file', False)
    # Obtaining the member 'split' of a type (line 97)
    split_2979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 15), executed_file_2978, 'split')
    # Calling split(args, kwargs) (line 97)
    split_call_result_2982 = invoke(stypy.reporting.localization.Localization(__file__, 97, 15), split_2979, *[str_2980], **kwargs_2981)
    
    # Obtaining the member '__getitem__' of a type (line 97)
    getitem___2983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 15), split_call_result_2982, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 97)
    subscript_call_result_2984 = invoke(stypy.reporting.localization.Localization(__file__, 97, 15), getitem___2983, int_2977)
    
    # Obtaining the member 'split' of a type (line 97)
    split_2985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 15), subscript_call_result_2984, 'split')
    # Calling split(args, kwargs) (line 97)
    split_call_result_2988 = invoke(stypy.reporting.localization.Localization(__file__, 97, 15), split_2985, *[str_2986], **kwargs_2987)
    
    # Obtaining the member '__getitem__' of a type (line 97)
    getitem___2989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 15), split_call_result_2988, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 97)
    subscript_call_result_2990 = invoke(stypy.reporting.localization.Localization(__file__, 97, 15), getitem___2989, int_2976)
    
    # Assigning a type to the variable 'filename' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'filename', subscript_call_result_2990)
    
    # Call to append(...): (line 98)
    # Processing the call arguments (line 98)
    # Getting the type of 'dirname' (line 98)
    dirname_2994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 20), 'dirname', False)
    # Processing the call keyword arguments (line 98)
    kwargs_2995 = {}
    # Getting the type of 'sys' (line 98)
    sys_2991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'sys', False)
    # Obtaining the member 'path' of a type (line 98)
    path_2992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 4), sys_2991, 'path')
    # Obtaining the member 'append' of a type (line 98)
    append_2993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 4), path_2992, 'append')
    # Calling append(args, kwargs) (line 98)
    append_call_result_2996 = invoke(stypy.reporting.localization.Localization(__file__, 98, 4), append_2993, *[dirname_2994], **kwargs_2995)
    
    
    # Assigning a BinOp to a Name (line 100):
    # Getting the type of 'filename' (line 100)
    filename_2997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 16), 'filename')
    # Getting the type of 'type_data_file_postfix' (line 100)
    type_data_file_postfix_2998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 27), 'type_data_file_postfix')
    # Applying the binary operator '+' (line 100)
    result_add_2999 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 16), '+', filename_2997, type_data_file_postfix_2998)
    
    # Assigning a type to the variable 'data_file' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'data_file', result_add_2999)
    
    # Assigning a Num to a Name (line 101):
    int_3000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 13), 'int')
    # Assigning a type to the variable 'result' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'result', int_3000)
    
    # Try-finally block (line 103)
    
    
    # SSA begins for try-except statement (line 103)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 104):
    
    # Call to __import__(...): (line 104)
    # Processing the call arguments (line 104)
    # Getting the type of 'data_file' (line 104)
    data_file_3002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 26), 'data_file', False)
    # Processing the call keyword arguments (line 104)
    kwargs_3003 = {}
    # Getting the type of '__import__' (line 104)
    import___3001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 15), '__import__', False)
    # Calling __import__(args, kwargs) (line 104)
    import___call_result_3004 = invoke(stypy.reporting.localization.Localization(__file__, 104, 15), import___3001, *[data_file_3002], **kwargs_3003)
    
    # Assigning a type to the variable 'data' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'data', import___call_result_3004)
    
    # Assigning a Attribute to a Name (line 106):
    # Getting the type of 'data' (line 106)
    data_3005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 25), 'data')
    # Obtaining the member 'test_types' of a type (line 106)
    test_types_3006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 25), data_3005, 'test_types')
    # Assigning a type to the variable 'expected_types' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'expected_types', test_types_3006)
    
    # Getting the type of 'expected_types' (line 108)
    expected_types_3007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 28), 'expected_types')
    # Assigning a type to the variable 'expected_types_3007' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'expected_types_3007', expected_types_3007)
    # Testing if the for loop is going to be iterated (line 108)
    # Testing the type of a for loop iterable (line 108)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 108, 8), expected_types_3007)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 108, 8), expected_types_3007):
        # Getting the type of the for loop variable (line 108)
        for_loop_var_3008 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 108, 8), expected_types_3007)
        # Assigning a type to the variable 'context_name' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'context_name', for_loop_var_3008)
        # SSA begins for a for statement (line 108)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 109):
        
        # Call to get_last_function_context_for(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'context_name' (line 109)
        context_name_3011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 72), 'context_name', False)
        # Processing the call keyword arguments (line 109)
        kwargs_3012 = {}
        # Getting the type of 'type_store' (line 109)
        type_store_3009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 31), 'type_store', False)
        # Obtaining the member 'get_last_function_context_for' of a type (line 109)
        get_last_function_context_for_3010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 31), type_store_3009, 'get_last_function_context_for')
        # Calling get_last_function_context_for(args, kwargs) (line 109)
        get_last_function_context_for_call_result_3013 = invoke(stypy.reporting.localization.Localization(__file__, 109, 31), get_last_function_context_for_3010, *[context_name_3011], **kwargs_3012)
        
        # Assigning a type to the variable 'inferred_context' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'inferred_context', get_last_function_context_for_call_result_3013)
        
        # Assigning a Subscript to a Name (line 110):
        
        # Obtaining the type of the subscript
        # Getting the type of 'context_name' (line 110)
        context_name_3014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 43), 'context_name')
        # Getting the type of 'expected_types' (line 110)
        expected_types_3015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 28), 'expected_types')
        # Obtaining the member '__getitem__' of a type (line 110)
        getitem___3016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 28), expected_types_3015, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 110)
        subscript_call_result_3017 = invoke(stypy.reporting.localization.Localization(__file__, 110, 28), getitem___3016, context_name_3014)
        
        # Assigning a type to the variable 'expected_vars' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'expected_vars', subscript_call_result_3017)
        
        
        # Call to __filter_reserved_vars(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'expected_vars' (line 111)
        expected_vars_3019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 46), 'expected_vars', False)
        # Processing the call keyword arguments (line 111)
        kwargs_3020 = {}
        # Getting the type of '__filter_reserved_vars' (line 111)
        filter_reserved_vars_3018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 23), '__filter_reserved_vars', False)
        # Calling __filter_reserved_vars(args, kwargs) (line 111)
        filter_reserved_vars_call_result_3021 = invoke(stypy.reporting.localization.Localization(__file__, 111, 23), filter_reserved_vars_3018, *[expected_vars_3019], **kwargs_3020)
        
        # Assigning a type to the variable 'filter_reserved_vars_call_result_3021' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'filter_reserved_vars_call_result_3021', filter_reserved_vars_call_result_3021)
        # Testing if the for loop is going to be iterated (line 111)
        # Testing the type of a for loop iterable (line 111)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 111, 12), filter_reserved_vars_call_result_3021)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 111, 12), filter_reserved_vars_call_result_3021):
            # Getting the type of the for loop variable (line 111)
            for_loop_var_3022 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 111, 12), filter_reserved_vars_call_result_3021)
            # Assigning a type to the variable 'var' (line 111)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'var', for_loop_var_3022)
            # SSA begins for a for statement (line 111)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to __equal_types(...): (line 112)
            # Processing the call arguments (line 112)
            
            # Obtaining the type of the subscript
            # Getting the type of 'var' (line 112)
            var_3024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 51), 'var', False)
            # Getting the type of 'expected_vars' (line 112)
            expected_vars_3025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 37), 'expected_vars', False)
            # Obtaining the member '__getitem__' of a type (line 112)
            getitem___3026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 37), expected_vars_3025, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 112)
            subscript_call_result_3027 = invoke(stypy.reporting.localization.Localization(__file__, 112, 37), getitem___3026, var_3024)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'var' (line 112)
            var_3028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 74), 'var', False)
            # Getting the type of 'inferred_context' (line 112)
            inferred_context_3029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 57), 'inferred_context', False)
            # Obtaining the member '__getitem__' of a type (line 112)
            getitem___3030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 57), inferred_context_3029, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 112)
            subscript_call_result_3031 = invoke(stypy.reporting.localization.Localization(__file__, 112, 57), getitem___3030, var_3028)
            
            # Processing the call keyword arguments (line 112)
            kwargs_3032 = {}
            # Getting the type of '__equal_types' (line 112)
            equal_types_3023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 23), '__equal_types', False)
            # Calling __equal_types(args, kwargs) (line 112)
            equal_types_call_result_3033 = invoke(stypy.reporting.localization.Localization(__file__, 112, 23), equal_types_3023, *[subscript_call_result_3027, subscript_call_result_3031], **kwargs_3032)
            
            # Applying the 'not' unary operator (line 112)
            result_not__3034 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 19), 'not', equal_types_call_result_3033)
            
            # Testing if the type of an if condition is none (line 112)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 112, 16), result_not__3034):
                pass
            else:
                
                # Testing the type of an if condition (line 112)
                if_condition_3035 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 16), result_not__3034)
                # Assigning a type to the variable 'if_condition_3035' (line 112)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'if_condition_3035', if_condition_3035)
                # SSA begins for if statement (line 112)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to format(...): (line 114)
                # Processing the call arguments (line 114)
                # Getting the type of 'var' (line 114)
                var_3038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 110), 'var', False)
                
                # Obtaining the type of the subscript
                # Getting the type of 'var' (line 116)
                var_3039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 116), 'var', False)
                # Getting the type of 'expected_vars' (line 115)
                expected_vars_3040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 112), 'expected_vars', False)
                # Obtaining the member '__getitem__' of a type (line 115)
                getitem___3041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 112), expected_vars_3040, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 115)
                subscript_call_result_3042 = invoke(stypy.reporting.localization.Localization(__file__, 115, 112), getitem___3041, var_3039)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'var' (line 118)
                var_3043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 116), 'var', False)
                # Getting the type of 'inferred_context' (line 117)
                inferred_context_3044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 112), 'inferred_context', False)
                # Obtaining the member '__getitem__' of a type (line 117)
                getitem___3045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 112), inferred_context_3044, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 117)
                subscript_call_result_3046 = invoke(stypy.reporting.localization.Localization(__file__, 117, 112), getitem___3045, var_3043)
                
                # Getting the type of 'context_name' (line 119)
                context_name_3047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 112), 'context_name', False)
                # Processing the call keyword arguments (line 114)
                kwargs_3048 = {}
                str_3036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 26), 'str', "Type mismatch for name '{0}' in context '{3}': {1} expected, but {2} found")
                # Obtaining the member 'format' of a type (line 114)
                format_3037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 26), str_3036, 'format')
                # Calling format(args, kwargs) (line 114)
                format_call_result_3049 = invoke(stypy.reporting.localization.Localization(__file__, 114, 26), format_3037, *[var_3038, subscript_call_result_3042, subscript_call_result_3046, context_name_3047], **kwargs_3048)
                
                
                # Assigning a Num to a Name (line 120):
                int_3050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 29), 'int')
                # Assigning a type to the variable 'result' (line 120)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 20), 'result', int_3050)
                # SSA join for if statement (line 112)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # SSA branch for the except part of a try statement (line 103)
    # SSA branch for the except 'Exception' branch of a try statement (line 103)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'Exception' (line 121)
    Exception_3051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 11), 'Exception')
    # Assigning a type to the variable 'exc' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'exc', Exception_3051)
    # Getting the type of 'verbose' (line 122)
    verbose_3052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 11), 'verbose')
    # Testing if the type of an if condition is none (line 122)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 122, 8), verbose_3052):
        pass
    else:
        
        # Testing the type of an if condition (line 122)
        if_condition_3053 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 122, 8), verbose_3052)
        # Assigning a type to the variable 'if_condition_3053' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'if_condition_3053', if_condition_3053)
        # SSA begins for if statement (line 122)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_3054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 18), 'str', 'Type checking error: ')
        
        # Call to str(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'exc' (line 123)
        exc_3056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 48), 'exc', False)
        # Processing the call keyword arguments (line 123)
        kwargs_3057 = {}
        # Getting the type of 'str' (line 123)
        str_3055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 44), 'str', False)
        # Calling str(args, kwargs) (line 123)
        str_call_result_3058 = invoke(stypy.reporting.localization.Localization(__file__, 123, 44), str_3055, *[exc_3056], **kwargs_3057)
        
        # Applying the binary operator '+' (line 123)
        result_add_3059 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 18), '+', str_3054, str_call_result_3058)
        
        # SSA join for if statement (line 122)
        module_type_store = module_type_store.join_ssa_context()
        

    int_3060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'stypy_return_type', int_3060)
    # SSA join for try-except statement (line 103)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # finally branch of the try-finally block (line 103)
    
    # Call to remove(...): (line 126)
    # Processing the call arguments (line 126)
    # Getting the type of 'dirname' (line 126)
    dirname_3064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 24), 'dirname', False)
    # Processing the call keyword arguments (line 126)
    kwargs_3065 = {}
    # Getting the type of 'sys' (line 126)
    sys_3061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'sys', False)
    # Obtaining the member 'path' of a type (line 126)
    path_3062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 8), sys_3061, 'path')
    # Obtaining the member 'remove' of a type (line 126)
    remove_3063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 8), path_3062, 'remove')
    # Calling remove(args, kwargs) (line 126)
    remove_call_result_3066 = invoke(stypy.reporting.localization.Localization(__file__, 126, 8), remove_3063, *[dirname_3064], **kwargs_3065)
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'verbose' (line 128)
    verbose_3067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 7), 'verbose')
    
    # Getting the type of 'result' (line 128)
    result_3068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 19), 'result')
    int_3069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 29), 'int')
    # Applying the binary operator '==' (line 128)
    result_eq_3070 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 19), '==', result_3068, int_3069)
    
    # Applying the binary operator 'and' (line 128)
    result_and_keyword_3071 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 7), 'and', verbose_3067, result_eq_3070)
    
    # Testing if the type of an if condition is none (line 128)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 128, 4), result_and_keyword_3071):
        pass
    else:
        
        # Testing the type of an if condition (line 128)
        if_condition_3072 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 128, 4), result_and_keyword_3071)
        # Assigning a type to the variable 'if_condition_3072' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'if_condition_3072', if_condition_3072)
        # SSA begins for if statement (line 128)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_3073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 14), 'str', 'All checks OK')
        # SSA join for if statement (line 128)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'result' (line 131)
    result_3074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 11), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'stypy_return_type', result_3074)
    
    # ################# End of 'check_type_store(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_type_store' in the type store
    # Getting the type of 'stypy_return_type' (line 86)
    stypy_return_type_3075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_3075)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_type_store'
    return stypy_return_type_3075

# Assigning a type to the variable 'check_type_store' (line 86)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'check_type_store', check_type_store)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
