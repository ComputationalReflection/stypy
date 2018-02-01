
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import colour
2: 
3: ''' copyright Sean McCarthy, license GPL v2 or later '''
4: 
5: class Peg:
6:     '''Class representing a (coloured) peg on the mastermind board'''
7: 
8:     def __init__(self,colour=None):
9:         self.__colour = colour
10: 
11:     def setColour(self, colour):
12:         self.__colour = colour
13: 
14:     def getColour(self):
15:         return self.__colour
16: 
17:     def equals(self,peg):
18:         return peg.getColour() == self.__colour
19: 
20:     def display(self):
21: ##        print str(colour.getColourName(self.__colour)).rjust(6),
22:         pass
23: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import colour' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/mm/')
import_715 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'colour')

if (type(import_715) is not StypyTypeError):

    if (import_715 != 'pyd_module'):
        __import__(import_715)
        sys_modules_716 = sys.modules[import_715]
        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'colour', sys_modules_716.module_type_store, module_type_store)
    else:
        import colour

        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'colour', colour, module_type_store)

else:
    # Assigning a type to the variable 'colour' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'colour', import_715)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/benchmark_suite/shedskin/mm/')

str_717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, 0), 'str', ' copyright Sean McCarthy, license GPL v2 or later ')
# Declaration of the 'Peg' class

class Peg:
    str_718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 4), 'str', 'Class representing a (coloured) peg on the mastermind board')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 8)
        None_719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 29), 'None')
        defaults = [None_719]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 8, 4, False)
        # Assigning a type to the variable 'self' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Peg.__init__', ['colour'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['colour'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 9):
        # Getting the type of 'colour' (line 9)
        colour_720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 24), 'colour')
        # Getting the type of 'self' (line 9)
        self_721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'self')
        # Setting the type of the member '__colour' of a type (line 9)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 8), self_721, '__colour', colour_720)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def setColour(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setColour'
        module_type_store = module_type_store.open_function_context('setColour', 11, 4, False)
        # Assigning a type to the variable 'self' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Peg.setColour.__dict__.__setitem__('stypy_localization', localization)
        Peg.setColour.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Peg.setColour.__dict__.__setitem__('stypy_type_store', module_type_store)
        Peg.setColour.__dict__.__setitem__('stypy_function_name', 'Peg.setColour')
        Peg.setColour.__dict__.__setitem__('stypy_param_names_list', ['colour'])
        Peg.setColour.__dict__.__setitem__('stypy_varargs_param_name', None)
        Peg.setColour.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Peg.setColour.__dict__.__setitem__('stypy_call_defaults', defaults)
        Peg.setColour.__dict__.__setitem__('stypy_call_varargs', varargs)
        Peg.setColour.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Peg.setColour.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Peg.setColour', ['colour'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setColour', localization, ['colour'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setColour(...)' code ##################

        
        # Assigning a Name to a Attribute (line 12):
        # Getting the type of 'colour' (line 12)
        colour_722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 24), 'colour')
        # Getting the type of 'self' (line 12)
        self_723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'self')
        # Setting the type of the member '__colour' of a type (line 12)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 8), self_723, '__colour', colour_722)
        
        # ################# End of 'setColour(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setColour' in the type store
        # Getting the type of 'stypy_return_type' (line 11)
        stypy_return_type_724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_724)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setColour'
        return stypy_return_type_724


    @norecursion
    def getColour(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'getColour'
        module_type_store = module_type_store.open_function_context('getColour', 14, 4, False)
        # Assigning a type to the variable 'self' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Peg.getColour.__dict__.__setitem__('stypy_localization', localization)
        Peg.getColour.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Peg.getColour.__dict__.__setitem__('stypy_type_store', module_type_store)
        Peg.getColour.__dict__.__setitem__('stypy_function_name', 'Peg.getColour')
        Peg.getColour.__dict__.__setitem__('stypy_param_names_list', [])
        Peg.getColour.__dict__.__setitem__('stypy_varargs_param_name', None)
        Peg.getColour.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Peg.getColour.__dict__.__setitem__('stypy_call_defaults', defaults)
        Peg.getColour.__dict__.__setitem__('stypy_call_varargs', varargs)
        Peg.getColour.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Peg.getColour.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Peg.getColour', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'getColour', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'getColour(...)' code ##################

        # Getting the type of 'self' (line 15)
        self_725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 15), 'self')
        # Obtaining the member '__colour' of a type (line 15)
        colour_726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 15), self_725, '__colour')
        # Assigning a type to the variable 'stypy_return_type' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', colour_726)
        
        # ################# End of 'getColour(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'getColour' in the type store
        # Getting the type of 'stypy_return_type' (line 14)
        stypy_return_type_727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_727)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'getColour'
        return stypy_return_type_727


    @norecursion
    def equals(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'equals'
        module_type_store = module_type_store.open_function_context('equals', 17, 4, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Peg.equals.__dict__.__setitem__('stypy_localization', localization)
        Peg.equals.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Peg.equals.__dict__.__setitem__('stypy_type_store', module_type_store)
        Peg.equals.__dict__.__setitem__('stypy_function_name', 'Peg.equals')
        Peg.equals.__dict__.__setitem__('stypy_param_names_list', ['peg'])
        Peg.equals.__dict__.__setitem__('stypy_varargs_param_name', None)
        Peg.equals.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Peg.equals.__dict__.__setitem__('stypy_call_defaults', defaults)
        Peg.equals.__dict__.__setitem__('stypy_call_varargs', varargs)
        Peg.equals.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Peg.equals.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Peg.equals', ['peg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'equals', localization, ['peg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'equals(...)' code ##################

        
        
        # Call to getColour(...): (line 18)
        # Processing the call keyword arguments (line 18)
        kwargs_730 = {}
        # Getting the type of 'peg' (line 18)
        peg_728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 15), 'peg', False)
        # Obtaining the member 'getColour' of a type (line 18)
        getColour_729 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 15), peg_728, 'getColour')
        # Calling getColour(args, kwargs) (line 18)
        getColour_call_result_731 = invoke(stypy.reporting.localization.Localization(__file__, 18, 15), getColour_729, *[], **kwargs_730)
        
        # Getting the type of 'self' (line 18)
        self_732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 34), 'self')
        # Obtaining the member '__colour' of a type (line 18)
        colour_733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 34), self_732, '__colour')
        # Applying the binary operator '==' (line 18)
        result_eq_734 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 15), '==', getColour_call_result_731, colour_733)
        
        # Assigning a type to the variable 'stypy_return_type' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'stypy_return_type', result_eq_734)
        
        # ################# End of 'equals(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'equals' in the type store
        # Getting the type of 'stypy_return_type' (line 17)
        stypy_return_type_735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_735)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'equals'
        return stypy_return_type_735


    @norecursion
    def display(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'display'
        module_type_store = module_type_store.open_function_context('display', 20, 4, False)
        # Assigning a type to the variable 'self' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Peg.display.__dict__.__setitem__('stypy_localization', localization)
        Peg.display.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Peg.display.__dict__.__setitem__('stypy_type_store', module_type_store)
        Peg.display.__dict__.__setitem__('stypy_function_name', 'Peg.display')
        Peg.display.__dict__.__setitem__('stypy_param_names_list', [])
        Peg.display.__dict__.__setitem__('stypy_varargs_param_name', None)
        Peg.display.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Peg.display.__dict__.__setitem__('stypy_call_defaults', defaults)
        Peg.display.__dict__.__setitem__('stypy_call_varargs', varargs)
        Peg.display.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Peg.display.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Peg.display', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'display', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'display(...)' code ##################

        pass
        
        # ################# End of 'display(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'display' in the type store
        # Getting the type of 'stypy_return_type' (line 20)
        stypy_return_type_736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_736)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'display'
        return stypy_return_type_736


# Assigning a type to the variable 'Peg' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'Peg', Peg)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
