
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import row
2: 
3: ''' copyright Sean McCarthy, license GPL v2 or later '''
4: 
5: class Board:
6:     '''Class board, a collection of rows'''
7: 
8:     def __init__(self):
9:         self.__board = []
10: 
11:     def getRow(self,rownum):
12:         return self.__board[rownum]
13: 
14:     def addRow(self, row):
15:         self.__board.append(row)
16: 
17:     def getRows(self):
18:         return self.__board
19: 
20: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import row' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/mm/')
import_22 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'row')

if (type(import_22) is not StypyTypeError):

    if (import_22 != 'pyd_module'):
        __import__(import_22)
        sys_modules_23 = sys.modules[import_22]
        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'row', sys_modules_23.module_type_store, module_type_store)
    else:
        import row

        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'row', row, module_type_store)

else:
    # Assigning a type to the variable 'row' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'row', import_22)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/mm/')

str_24 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 0), 'str', ' copyright Sean McCarthy, license GPL v2 or later ')
# Declaration of the 'Board' class

class Board:
    str_25 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 4), 'str', 'Class board, a collection of rows')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 8, 4, False)
        # Assigning a type to the variable 'self' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Board.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a List to a Attribute (line 9):
        
        # Obtaining an instance of the builtin type 'list' (line 9)
        list_26 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 9)
        
        # Getting the type of 'self' (line 9)
        self_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'self')
        # Setting the type of the member '__board' of a type (line 9)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 8), self_27, '__board', list_26)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def getRow(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getRow'
        module_type_store = module_type_store.open_function_context('getRow', 11, 4, False)
        # Assigning a type to the variable 'self' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Board.getRow.__dict__.__setitem__('stypy_localization', localization)
        Board.getRow.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Board.getRow.__dict__.__setitem__('stypy_type_store', module_type_store)
        Board.getRow.__dict__.__setitem__('stypy_function_name', 'Board.getRow')
        Board.getRow.__dict__.__setitem__('stypy_param_names_list', ['rownum'])
        Board.getRow.__dict__.__setitem__('stypy_varargs_param_name', None)
        Board.getRow.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Board.getRow.__dict__.__setitem__('stypy_call_defaults', defaults)
        Board.getRow.__dict__.__setitem__('stypy_call_varargs', varargs)
        Board.getRow.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Board.getRow.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Board.getRow', ['rownum'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getRow', localization, ['rownum'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getRow(...)' code ##################

        
        # Obtaining the type of the subscript
        # Getting the type of 'rownum' (line 12)
        rownum_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 28), 'rownum')
        # Getting the type of 'self' (line 12)
        self_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 15), 'self')
        # Obtaining the member '__board' of a type (line 12)
        board_30 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 15), self_29, '__board')
        # Obtaining the member '__getitem__' of a type (line 12)
        getitem___31 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 15), board_30, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 12)
        subscript_call_result_32 = invoke(stypy.reporting.localization.Localization(__file__, 12, 15), getitem___31, rownum_28)
        
        # Assigning a type to the variable 'stypy_return_type' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'stypy_return_type', subscript_call_result_32)
        
        # ################# End of 'getRow(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getRow' in the type store
        # Getting the type of 'stypy_return_type' (line 11)
        stypy_return_type_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_33)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getRow'
        return stypy_return_type_33


    @norecursion
    def addRow(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'addRow'
        module_type_store = module_type_store.open_function_context('addRow', 14, 4, False)
        # Assigning a type to the variable 'self' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Board.addRow.__dict__.__setitem__('stypy_localization', localization)
        Board.addRow.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Board.addRow.__dict__.__setitem__('stypy_type_store', module_type_store)
        Board.addRow.__dict__.__setitem__('stypy_function_name', 'Board.addRow')
        Board.addRow.__dict__.__setitem__('stypy_param_names_list', ['row'])
        Board.addRow.__dict__.__setitem__('stypy_varargs_param_name', None)
        Board.addRow.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Board.addRow.__dict__.__setitem__('stypy_call_defaults', defaults)
        Board.addRow.__dict__.__setitem__('stypy_call_varargs', varargs)
        Board.addRow.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Board.addRow.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Board.addRow', ['row'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'addRow', localization, ['row'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'addRow(...)' code ##################

        
        # Call to append(...): (line 15)
        # Processing the call arguments (line 15)
        # Getting the type of 'row' (line 15)
        row_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 28), 'row', False)
        # Processing the call keyword arguments (line 15)
        kwargs_38 = {}
        # Getting the type of 'self' (line 15)
        self_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'self', False)
        # Obtaining the member '__board' of a type (line 15)
        board_35 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 8), self_34, '__board')
        # Obtaining the member 'append' of a type (line 15)
        append_36 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 8), board_35, 'append')
        # Calling append(args, kwargs) (line 15)
        append_call_result_39 = invoke(stypy.reporting.localization.Localization(__file__, 15, 8), append_36, *[row_37], **kwargs_38)
        
        
        # ################# End of 'addRow(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'addRow' in the type store
        # Getting the type of 'stypy_return_type' (line 14)
        stypy_return_type_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_40)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'addRow'
        return stypy_return_type_40


    @norecursion
    def getRows(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getRows'
        module_type_store = module_type_store.open_function_context('getRows', 17, 4, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Board.getRows.__dict__.__setitem__('stypy_localization', localization)
        Board.getRows.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Board.getRows.__dict__.__setitem__('stypy_type_store', module_type_store)
        Board.getRows.__dict__.__setitem__('stypy_function_name', 'Board.getRows')
        Board.getRows.__dict__.__setitem__('stypy_param_names_list', [])
        Board.getRows.__dict__.__setitem__('stypy_varargs_param_name', None)
        Board.getRows.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Board.getRows.__dict__.__setitem__('stypy_call_defaults', defaults)
        Board.getRows.__dict__.__setitem__('stypy_call_varargs', varargs)
        Board.getRows.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Board.getRows.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Board.getRows', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getRows', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getRows(...)' code ##################

        # Getting the type of 'self' (line 18)
        self_41 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 15), 'self')
        # Obtaining the member '__board' of a type (line 18)
        board_42 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 15), self_41, '__board')
        # Assigning a type to the variable 'stypy_return_type' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'stypy_return_type', board_42)
        
        # ################# End of 'getRows(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getRows' in the type store
        # Getting the type of 'stypy_return_type' (line 17)
        stypy_return_type_43 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_43)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getRows'
        return stypy_return_type_43


# Assigning a type to the variable 'Board' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'Board', Board)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
