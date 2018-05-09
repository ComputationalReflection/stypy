
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from stypy_copy.errors_copy.stack_trace_copy import StackTrace
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

# 'from stypy_copy.errors_copy.stack_trace_copy import StackTrace' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')
import_9699 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy.errors_copy.stack_trace_copy')

if (type(import_9699) is not StypyTypeError):

    if (import_9699 != 'pyd_module'):
        __import__(import_9699)
        sys_modules_9700 = sys.modules[import_9699]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy.errors_copy.stack_trace_copy', sys_modules_9700.module_type_store, module_type_store, ['StackTrace'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_9700, sys_modules_9700.module_type_store, module_type_store)
    else:
        from stypy_copy.errors_copy.stack_trace_copy import StackTrace

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy.errors_copy.stack_trace_copy', None, module_type_store, ['StackTrace'], [StackTrace])

else:
    # Assigning a type to the variable 'stypy_copy.errors_copy.stack_trace_copy' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'stypy_copy.errors_copy.stack_trace_copy', import_9699)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/type_inference_copy/')

# Declaration of the 'Localization' class

class Localization:
    str_9701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, (-1)), 'str', '\n    This class is used to store caller information on function calls. It comprises the following data of the caller:\n    - Line and column of the source code that performed the call\n    - Python source code file name\n    - Current stack trace of calls.\n\n    Localization objects are key to generate accurate errors. Therefore most of the calls that stypy does uses\n    localization instances for this matter\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_9702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 33), 'str', '[Not specified]')
        int_9703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 57), 'int')
        int_9704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 67), 'int')
        defaults = [str_9702, int_9703, int_9704]
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
        kwargs_9707 = {}
        # Getting the type of 'StackTrace' (line 15)
        StackTrace_9705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 27), 'StackTrace', False)
        # Obtaining the member 'Instance' of a type (line 15)
        Instance_9706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 27), StackTrace_9705, 'Instance')
        # Calling Instance(args, kwargs) (line 15)
        Instance_call_result_9708 = invoke(stypy.reporting.localization.Localization(__file__, 15, 27), Instance_9706, *[], **kwargs_9707)
        
        # Getting the type of 'self' (line 15)
        self_9709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'self')
        # Setting the type of the member 'stack_trace' of a type (line 15)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 8), self_9709, 'stack_trace', Instance_call_result_9708)
        
        # Assigning a Name to a Attribute (line 16):
        # Getting the type of 'file_name' (line 16)
        file_name_9710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 25), 'file_name')
        # Getting the type of 'self' (line 16)
        self_9711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'self')
        # Setting the type of the member 'file_name' of a type (line 16)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), self_9711, 'file_name', file_name_9710)
        
        # Assigning a Name to a Attribute (line 17):
        # Getting the type of 'line' (line 17)
        line_9712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 20), 'line')
        # Getting the type of 'self' (line 17)
        self_9713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'self')
        # Setting the type of the member 'line' of a type (line 17)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 8), self_9713, 'line', line_9712)
        
        # Assigning a Name to a Attribute (line 18):
        # Getting the type of 'column' (line 18)
        column_9714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 22), 'column')
        # Getting the type of 'self' (line 18)
        self_9715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'self')
        # Setting the type of the member 'column' of a type (line 18)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 8), self_9715, 'column', column_9714)
        
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

        str_9716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, (-1)), 'str', '\n        Gets the current stack trace\n        :return:\n        ')
        # Getting the type of 'self' (line 25)
        self_9717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'self')
        # Obtaining the member 'stack_trace' of a type (line 25)
        stack_trace_9718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 15), self_9717, 'stack_trace')
        # Assigning a type to the variable 'stypy_return_type' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'stypy_return_type', stack_trace_9718)
        
        # ################# End of 'get_stack_trace(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_stack_trace' in the type store
        # Getting the type of 'stypy_return_type' (line 20)
        stypy_return_type_9719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_9719)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_stack_trace'
        return stypy_return_type_9719


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

        str_9720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, (-1)), 'str', '\n        Modifies the stored stack trace appending a new stack trace (call begins)\n        :param func_name:\n        :param declared_arguments:\n        :param arguments:\n        :return:\n        ')
        
        # Call to set(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'self' (line 35)
        self_9724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 29), 'self', False)
        # Obtaining the member 'file_name' of a type (line 35)
        file_name_9725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 29), self_9724, 'file_name')
        # Getting the type of 'self' (line 35)
        self_9726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 45), 'self', False)
        # Obtaining the member 'line' of a type (line 35)
        line_9727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 45), self_9726, 'line')
        # Getting the type of 'self' (line 35)
        self_9728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 56), 'self', False)
        # Obtaining the member 'column' of a type (line 35)
        column_9729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 56), self_9728, 'column')
        # Getting the type of 'func_name' (line 35)
        func_name_9730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 69), 'func_name', False)
        # Getting the type of 'declared_arguments' (line 35)
        declared_arguments_9731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 80), 'declared_arguments', False)
        # Getting the type of 'arguments' (line 35)
        arguments_9732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 100), 'arguments', False)
        # Processing the call keyword arguments (line 35)
        kwargs_9733 = {}
        # Getting the type of 'self' (line 35)
        self_9721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'self', False)
        # Obtaining the member 'stack_trace' of a type (line 35)
        stack_trace_9722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), self_9721, 'stack_trace')
        # Obtaining the member 'set' of a type (line 35)
        set_9723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), stack_trace_9722, 'set')
        # Calling set(args, kwargs) (line 35)
        set_call_result_9734 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), set_9723, *[file_name_9725, line_9727, column_9729, func_name_9730, declared_arguments_9731, arguments_9732], **kwargs_9733)
        
        
        # ################# End of 'set_stack_trace(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_stack_trace' in the type store
        # Getting the type of 'stypy_return_type' (line 27)
        stypy_return_type_9735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_9735)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_stack_trace'
        return stypy_return_type_9735


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

        str_9736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, (-1)), 'str', '\n        Deletes the last set stack trace (call ends)\n        :return:\n        ')
        
        # Call to unset(...): (line 42)
        # Processing the call keyword arguments (line 42)
        kwargs_9740 = {}
        # Getting the type of 'self' (line 42)
        self_9737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'self', False)
        # Obtaining the member 'stack_trace' of a type (line 42)
        stack_trace_9738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), self_9737, 'stack_trace')
        # Obtaining the member 'unset' of a type (line 42)
        unset_9739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), stack_trace_9738, 'unset')
        # Calling unset(args, kwargs) (line 42)
        unset_call_result_9741 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), unset_9739, *[], **kwargs_9740)
        
        
        # ################# End of 'unset_stack_trace(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'unset_stack_trace' in the type store
        # Getting the type of 'stypy_return_type' (line 37)
        stypy_return_type_9742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_9742)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'unset_stack_trace'
        return stypy_return_type_9742


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

        str_9743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, (-1)), 'str', '\n        Compares localizations using source line, column and file\n        :param other:\n        :return:\n        ')
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 50)
        self_9744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 15), 'self')
        # Obtaining the member 'file_name' of a type (line 50)
        file_name_9745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 15), self_9744, 'file_name')
        # Getting the type of 'other' (line 50)
        other_9746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 33), 'other')
        # Obtaining the member 'file_name' of a type (line 50)
        file_name_9747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 33), other_9746, 'file_name')
        # Applying the binary operator '==' (line 50)
        result_eq_9748 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 15), '==', file_name_9745, file_name_9747)
        
        
        # Getting the type of 'self' (line 50)
        self_9749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 53), 'self')
        # Obtaining the member 'line' of a type (line 50)
        line_9750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 53), self_9749, 'line')
        # Getting the type of 'other' (line 50)
        other_9751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 66), 'other')
        # Obtaining the member 'line' of a type (line 50)
        line_9752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 66), other_9751, 'line')
        # Applying the binary operator '==' (line 50)
        result_eq_9753 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 53), '==', line_9750, line_9752)
        
        # Applying the binary operator 'and' (line 50)
        result_and_keyword_9754 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 15), 'and', result_eq_9748, result_eq_9753)
        
        # Getting the type of 'self' (line 50)
        self_9755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 81), 'self')
        # Obtaining the member 'column' of a type (line 50)
        column_9756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 81), self_9755, 'column')
        # Getting the type of 'other' (line 50)
        other_9757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 96), 'other')
        # Obtaining the member 'column' of a type (line 50)
        column_9758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 96), other_9757, 'column')
        # Applying the binary operator '==' (line 50)
        result_eq_9759 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 81), '==', column_9756, column_9758)
        
        # Applying the binary operator 'and' (line 50)
        result_and_keyword_9760 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 15), 'and', result_and_keyword_9754, result_eq_9759)
        
        # Assigning a type to the variable 'stypy_return_type' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'stypy_return_type', result_and_keyword_9760)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 44)
        stypy_return_type_9761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_9761)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_9761


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

        str_9762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, (-1)), 'str', '\n        Deep copy (Clone) this object\n        :return:\n        ')
        
        # Call to Localization(...): (line 57)
        # Processing the call arguments (line 57)
        # Getting the type of 'self' (line 57)
        self_9764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 28), 'self', False)
        # Obtaining the member 'file_name' of a type (line 57)
        file_name_9765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 28), self_9764, 'file_name')
        # Getting the type of 'self' (line 57)
        self_9766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 44), 'self', False)
        # Obtaining the member 'line' of a type (line 57)
        line_9767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 44), self_9766, 'line')
        # Getting the type of 'self' (line 57)
        self_9768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 55), 'self', False)
        # Obtaining the member 'column' of a type (line 57)
        column_9769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 55), self_9768, 'column')
        # Processing the call keyword arguments (line 57)
        kwargs_9770 = {}
        # Getting the type of 'Localization' (line 57)
        Localization_9763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 15), 'Localization', False)
        # Calling Localization(args, kwargs) (line 57)
        Localization_call_result_9771 = invoke(stypy.reporting.localization.Localization(__file__, 57, 15), Localization_9763, *[file_name_9765, line_9767, column_9769], **kwargs_9770)
        
        # Assigning a type to the variable 'stypy_return_type' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'stypy_return_type', Localization_call_result_9771)
        
        # ################# End of 'clone(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'clone' in the type store
        # Getting the type of 'stypy_return_type' (line 52)
        stypy_return_type_9772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_9772)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'clone'
        return stypy_return_type_9772


# Assigning a type to the variable 'Localization' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'Localization', Localization)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
