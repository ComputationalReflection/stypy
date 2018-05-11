
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from ....errors_copy.stack_trace_copy import StackTrace
2: 
3: 
4: class Localization:
5:     '''
6:     This class is used to store caller information on function calls. It comprises the following data of the caller:
7:     - Line and column of the source code that performed the call
8:     - Python source code file name
9:     - Current stack trace of calls.
10: 
11:     Localization objects are key to generate accurate errors. Therefore most of the calls that stypy does uses
12:     localization instances for this matter
13:     '''
14:     def __init__(self, file_name="[Not specified]", line=0, column=0):
15:         self.stack_trace = StackTrace.Instance()
16:         self.file_name = file_name
17:         self.line = line
18:         self.column = column
19: 
20:     def get_stack_trace(self):
21:         '''
22:         Gets the current stack trace
23:         :return:
24:         '''
25:         return self.stack_trace
26: 
27:     def set_stack_trace(self, func_name, declared_arguments, arguments):
28:         '''
29:         Modifies the stored stack trace appending a new stack trace (call begins)
30:         :param func_name:
31:         :param declared_arguments:
32:         :param arguments:
33:         :return:
34:         '''
35:         self.stack_trace.set(self.file_name, self.line, self.column, func_name, declared_arguments, arguments)
36: 
37:     def unset_stack_trace(self):
38:         '''
39:         Deletes the last set stack trace (call ends)
40:         :return:
41:         '''
42:         self.stack_trace.unset()
43: 
44:     def __eq__(self, other):
45:         '''
46:         Compares localizations using source line, column and file
47:         :param other:
48:         :return:
49:         '''
50:         return self.file_name == other.file_name and self.line == other.line and self.column == other.column
51: 
52:     def clone(self):
53:         '''
54:         Deep copy (Clone) this object
55:         :return:
56:         '''
57:         return Localization(self.file_name, self.line, self.column)
58: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.stack_trace_copy import StackTrace' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_9985 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.stack_trace_copy')

if (type(import_9985) is not StypyTypeError):

    if (import_9985 != 'pyd_module'):
        __import__(import_9985)
        sys_modules_9986 = sys.modules[import_9985]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.stack_trace_copy', sys_modules_9986.module_type_store, module_type_store, ['StackTrace'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_9986, sys_modules_9986.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.stack_trace_copy import StackTrace

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.stack_trace_copy', None, module_type_store, ['StackTrace'], [StackTrace])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.stack_trace_copy' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.stack_trace_copy', import_9985)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

# Declaration of the 'Localization' class

class Localization:
    str_9987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, (-1)), 'str', '\n    This class is used to store caller information on function calls. It comprises the following data of the caller:\n    - Line and column of the source code that performed the call\n    - Python source code file name\n    - Current stack trace of calls.\n\n    Localization objects are key to generate accurate errors. Therefore most of the calls that stypy does uses\n    localization instances for this matter\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_9988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 33), 'str', '[Not specified]')
        int_9989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 57), 'int')
        int_9990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 67), 'int')
        defaults = [str_9988, int_9989, int_9990]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 14, 4, False)
        # Assigning a type to the variable 'self' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Localization.__init__', ['file_name', 'line', 'column'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['file_name', 'line', 'column'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Attribute (line 15):
        
        # Call to Instance(...): (line 15)
        # Processing the call keyword arguments (line 15)
        kwargs_9993 = {}
        # Getting the type of 'StackTrace' (line 15)
        StackTrace_9991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 27), 'StackTrace', False)
        # Obtaining the member 'Instance' of a type (line 15)
        Instance_9992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 27), StackTrace_9991, 'Instance')
        # Calling Instance(args, kwargs) (line 15)
        Instance_call_result_9994 = invoke(stypy.reporting.localization.Localization(__file__, 15, 27), Instance_9992, *[], **kwargs_9993)
        
        # Getting the type of 'self' (line 15)
        self_9995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'self')
        # Setting the type of the member 'stack_trace' of a type (line 15)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 8), self_9995, 'stack_trace', Instance_call_result_9994)
        
        # Assigning a Name to a Attribute (line 16):
        # Getting the type of 'file_name' (line 16)
        file_name_9996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 25), 'file_name')
        # Getting the type of 'self' (line 16)
        self_9997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'self')
        # Setting the type of the member 'file_name' of a type (line 16)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), self_9997, 'file_name', file_name_9996)
        
        # Assigning a Name to a Attribute (line 17):
        # Getting the type of 'line' (line 17)
        line_9998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 20), 'line')
        # Getting the type of 'self' (line 17)
        self_9999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'self')
        # Setting the type of the member 'line' of a type (line 17)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 8), self_9999, 'line', line_9998)
        
        # Assigning a Name to a Attribute (line 18):
        # Getting the type of 'column' (line 18)
        column_10000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 22), 'column')
        # Getting the type of 'self' (line 18)
        self_10001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'self')
        # Setting the type of the member 'column' of a type (line 18)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 8), self_10001, 'column', column_10000)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def get_stack_trace(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_stack_trace'
        module_type_store = module_type_store.open_function_context('get_stack_trace', 20, 4, False)
        # Assigning a type to the variable 'self' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Localization.get_stack_trace.__dict__.__setitem__('stypy_localization', localization)
        Localization.get_stack_trace.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Localization.get_stack_trace.__dict__.__setitem__('stypy_type_store', module_type_store)
        Localization.get_stack_trace.__dict__.__setitem__('stypy_function_name', 'Localization.get_stack_trace')
        Localization.get_stack_trace.__dict__.__setitem__('stypy_param_names_list', [])
        Localization.get_stack_trace.__dict__.__setitem__('stypy_varargs_param_name', None)
        Localization.get_stack_trace.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Localization.get_stack_trace.__dict__.__setitem__('stypy_call_defaults', defaults)
        Localization.get_stack_trace.__dict__.__setitem__('stypy_call_varargs', varargs)
        Localization.get_stack_trace.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Localization.get_stack_trace.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Localization.get_stack_trace', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_stack_trace', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_stack_trace(...)' code ##################

        str_10002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, (-1)), 'str', '\n        Gets the current stack trace\n        :return:\n        ')
        # Getting the type of 'self' (line 25)
        self_10003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'self')
        # Obtaining the member 'stack_trace' of a type (line 25)
        stack_trace_10004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 15), self_10003, 'stack_trace')
        # Assigning a type to the variable 'stypy_return_type' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'stypy_return_type', stack_trace_10004)
        
        # ################# End of 'get_stack_trace(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_stack_trace' in the type store
        # Getting the type of 'stypy_return_type' (line 20)
        stypy_return_type_10005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_10005)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_stack_trace'
        return stypy_return_type_10005


    @norecursion
    def set_stack_trace(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_stack_trace'
        module_type_store = module_type_store.open_function_context('set_stack_trace', 27, 4, False)
        # Assigning a type to the variable 'self' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Localization.set_stack_trace.__dict__.__setitem__('stypy_localization', localization)
        Localization.set_stack_trace.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Localization.set_stack_trace.__dict__.__setitem__('stypy_type_store', module_type_store)
        Localization.set_stack_trace.__dict__.__setitem__('stypy_function_name', 'Localization.set_stack_trace')
        Localization.set_stack_trace.__dict__.__setitem__('stypy_param_names_list', ['func_name', 'declared_arguments', 'arguments'])
        Localization.set_stack_trace.__dict__.__setitem__('stypy_varargs_param_name', None)
        Localization.set_stack_trace.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Localization.set_stack_trace.__dict__.__setitem__('stypy_call_defaults', defaults)
        Localization.set_stack_trace.__dict__.__setitem__('stypy_call_varargs', varargs)
        Localization.set_stack_trace.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Localization.set_stack_trace.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Localization.set_stack_trace', ['func_name', 'declared_arguments', 'arguments'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_stack_trace', localization, ['func_name', 'declared_arguments', 'arguments'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_stack_trace(...)' code ##################

        str_10006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, (-1)), 'str', '\n        Modifies the stored stack trace appending a new stack trace (call begins)\n        :param func_name:\n        :param declared_arguments:\n        :param arguments:\n        :return:\n        ')
        
        # Call to set(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'self' (line 35)
        self_10010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 29), 'self', False)
        # Obtaining the member 'file_name' of a type (line 35)
        file_name_10011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 29), self_10010, 'file_name')
        # Getting the type of 'self' (line 35)
        self_10012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 45), 'self', False)
        # Obtaining the member 'line' of a type (line 35)
        line_10013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 45), self_10012, 'line')
        # Getting the type of 'self' (line 35)
        self_10014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 56), 'self', False)
        # Obtaining the member 'column' of a type (line 35)
        column_10015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 56), self_10014, 'column')
        # Getting the type of 'func_name' (line 35)
        func_name_10016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 69), 'func_name', False)
        # Getting the type of 'declared_arguments' (line 35)
        declared_arguments_10017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 80), 'declared_arguments', False)
        # Getting the type of 'arguments' (line 35)
        arguments_10018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 100), 'arguments', False)
        # Processing the call keyword arguments (line 35)
        kwargs_10019 = {}
        # Getting the type of 'self' (line 35)
        self_10007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'self', False)
        # Obtaining the member 'stack_trace' of a type (line 35)
        stack_trace_10008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), self_10007, 'stack_trace')
        # Obtaining the member 'set' of a type (line 35)
        set_10009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), stack_trace_10008, 'set')
        # Calling set(args, kwargs) (line 35)
        set_call_result_10020 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), set_10009, *[file_name_10011, line_10013, column_10015, func_name_10016, declared_arguments_10017, arguments_10018], **kwargs_10019)
        
        
        # ################# End of 'set_stack_trace(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_stack_trace' in the type store
        # Getting the type of 'stypy_return_type' (line 27)
        stypy_return_type_10021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_10021)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_stack_trace'
        return stypy_return_type_10021


    @norecursion
    def unset_stack_trace(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'unset_stack_trace'
        module_type_store = module_type_store.open_function_context('unset_stack_trace', 37, 4, False)
        # Assigning a type to the variable 'self' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Localization.unset_stack_trace.__dict__.__setitem__('stypy_localization', localization)
        Localization.unset_stack_trace.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Localization.unset_stack_trace.__dict__.__setitem__('stypy_type_store', module_type_store)
        Localization.unset_stack_trace.__dict__.__setitem__('stypy_function_name', 'Localization.unset_stack_trace')
        Localization.unset_stack_trace.__dict__.__setitem__('stypy_param_names_list', [])
        Localization.unset_stack_trace.__dict__.__setitem__('stypy_varargs_param_name', None)
        Localization.unset_stack_trace.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Localization.unset_stack_trace.__dict__.__setitem__('stypy_call_defaults', defaults)
        Localization.unset_stack_trace.__dict__.__setitem__('stypy_call_varargs', varargs)
        Localization.unset_stack_trace.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Localization.unset_stack_trace.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Localization.unset_stack_trace', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'unset_stack_trace', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'unset_stack_trace(...)' code ##################

        str_10022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, (-1)), 'str', '\n        Deletes the last set stack trace (call ends)\n        :return:\n        ')
        
        # Call to unset(...): (line 42)
        # Processing the call keyword arguments (line 42)
        kwargs_10026 = {}
        # Getting the type of 'self' (line 42)
        self_10023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'self', False)
        # Obtaining the member 'stack_trace' of a type (line 42)
        stack_trace_10024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), self_10023, 'stack_trace')
        # Obtaining the member 'unset' of a type (line 42)
        unset_10025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), stack_trace_10024, 'unset')
        # Calling unset(args, kwargs) (line 42)
        unset_call_result_10027 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), unset_10025, *[], **kwargs_10026)
        
        
        # ################# End of 'unset_stack_trace(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'unset_stack_trace' in the type store
        # Getting the type of 'stypy_return_type' (line 37)
        stypy_return_type_10028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_10028)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'unset_stack_trace'
        return stypy_return_type_10028


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 44, 4, False)
        # Assigning a type to the variable 'self' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Localization.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        Localization.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Localization.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Localization.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'Localization.stypy__eq__')
        Localization.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        Localization.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Localization.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Localization.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Localization.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Localization.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Localization.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Localization.stypy__eq__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__eq__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__eq__(...)' code ##################

        str_10029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, (-1)), 'str', '\n        Compares localizations using source line, column and file\n        :param other:\n        :return:\n        ')
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 50)
        self_10030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 15), 'self')
        # Obtaining the member 'file_name' of a type (line 50)
        file_name_10031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 15), self_10030, 'file_name')
        # Getting the type of 'other' (line 50)
        other_10032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 33), 'other')
        # Obtaining the member 'file_name' of a type (line 50)
        file_name_10033 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 33), other_10032, 'file_name')
        # Applying the binary operator '==' (line 50)
        result_eq_10034 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 15), '==', file_name_10031, file_name_10033)
        
        
        # Getting the type of 'self' (line 50)
        self_10035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 53), 'self')
        # Obtaining the member 'line' of a type (line 50)
        line_10036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 53), self_10035, 'line')
        # Getting the type of 'other' (line 50)
        other_10037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 66), 'other')
        # Obtaining the member 'line' of a type (line 50)
        line_10038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 66), other_10037, 'line')
        # Applying the binary operator '==' (line 50)
        result_eq_10039 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 53), '==', line_10036, line_10038)
        
        # Applying the binary operator 'and' (line 50)
        result_and_keyword_10040 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 15), 'and', result_eq_10034, result_eq_10039)
        
        # Getting the type of 'self' (line 50)
        self_10041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 81), 'self')
        # Obtaining the member 'column' of a type (line 50)
        column_10042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 81), self_10041, 'column')
        # Getting the type of 'other' (line 50)
        other_10043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 96), 'other')
        # Obtaining the member 'column' of a type (line 50)
        column_10044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 96), other_10043, 'column')
        # Applying the binary operator '==' (line 50)
        result_eq_10045 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 81), '==', column_10042, column_10044)
        
        # Applying the binary operator 'and' (line 50)
        result_and_keyword_10046 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 15), 'and', result_and_keyword_10040, result_eq_10045)
        
        # Assigning a type to the variable 'stypy_return_type' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'stypy_return_type', result_and_keyword_10046)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 44)
        stypy_return_type_10047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_10047)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_10047


    @norecursion
    def clone(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'clone'
        module_type_store = module_type_store.open_function_context('clone', 52, 4, False)
        # Assigning a type to the variable 'self' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Localization.clone.__dict__.__setitem__('stypy_localization', localization)
        Localization.clone.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Localization.clone.__dict__.__setitem__('stypy_type_store', module_type_store)
        Localization.clone.__dict__.__setitem__('stypy_function_name', 'Localization.clone')
        Localization.clone.__dict__.__setitem__('stypy_param_names_list', [])
        Localization.clone.__dict__.__setitem__('stypy_varargs_param_name', None)
        Localization.clone.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Localization.clone.__dict__.__setitem__('stypy_call_defaults', defaults)
        Localization.clone.__dict__.__setitem__('stypy_call_varargs', varargs)
        Localization.clone.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Localization.clone.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Localization.clone', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'clone', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'clone(...)' code ##################

        str_10048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, (-1)), 'str', '\n        Deep copy (Clone) this object\n        :return:\n        ')
        
        # Call to Localization(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'self' (line 57)
        self_10050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 28), 'self', False)
        # Obtaining the member 'file_name' of a type (line 57)
        file_name_10051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 28), self_10050, 'file_name')
        # Getting the type of 'self' (line 57)
        self_10052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 44), 'self', False)
        # Obtaining the member 'line' of a type (line 57)
        line_10053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 44), self_10052, 'line')
        # Getting the type of 'self' (line 57)
        self_10054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 55), 'self', False)
        # Obtaining the member 'column' of a type (line 57)
        column_10055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 55), self_10054, 'column')
        # Processing the call keyword arguments (line 57)
        kwargs_10056 = {}
        # Getting the type of 'Localization' (line 57)
        Localization_10049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 15), 'Localization', False)
        # Calling Localization(args, kwargs) (line 57)
        Localization_call_result_10057 = invoke(stypy.reporting.localization.Localization(__file__, 57, 15), Localization_10049, *[file_name_10051, line_10053, column_10055], **kwargs_10056)
        
        # Assigning a type to the variable 'stypy_return_type' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'stypy_return_type', Localization_call_result_10057)
        
        # ################# End of 'clone(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'clone' in the type store
        # Getting the type of 'stypy_return_type' (line 52)
        stypy_return_type_10058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_10058)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'clone'
        return stypy_return_type_10058


# Assigning a type to the variable 'Localization' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'Localization', Localization)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
