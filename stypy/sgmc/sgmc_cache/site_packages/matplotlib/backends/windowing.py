
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: MS Windows-specific helper for the TkAgg backend.
3: 
4: With rcParams['tk.window_focus'] default of False, it is
5: effectively disabled.
6: 
7: It uses a tiny C++ extension module to access MS Win functions.
8: '''
9: from __future__ import (absolute_import, division, print_function,
10:                         unicode_literals)
11: 
12: import six
13: 
14: from matplotlib import rcParams
15: 
16: try:
17:     if not rcParams['tk.window_focus']:
18:         raise ImportError
19:     from matplotlib._windowing import GetForegroundWindow, SetForegroundWindow
20: except ImportError:
21:     def GetForegroundWindow():
22:         return 0
23:     def SetForegroundWindow(hwnd):
24:         pass
25: 
26: class FocusManager(object):
27:     def __init__(self):
28:         self._shellWindow = GetForegroundWindow()
29: 
30:     def __del__(self):
31:         SetForegroundWindow(self._shellWindow)
32: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_269467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, (-1)), 'unicode', u"\nMS Windows-specific helper for the TkAgg backend.\n\nWith rcParams['tk.window_focus'] default of False, it is\neffectively disabled.\n\nIt uses a tiny C++ extension module to access MS Win functions.\n")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import six' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_269468 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'six')

if (type(import_269468) is not StypyTypeError):

    if (import_269468 != 'pyd_module'):
        __import__(import_269468)
        sys_modules_269469 = sys.modules[import_269468]
        import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'six', sys_modules_269469.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'six', import_269468)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from matplotlib import rcParams' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_269470 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib')

if (type(import_269470) is not StypyTypeError):

    if (import_269470 != 'pyd_module'):
        __import__(import_269470)
        sys_modules_269471 = sys.modules[import_269470]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib', sys_modules_269471.module_type_store, module_type_store, ['rcParams'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_269471, sys_modules_269471.module_type_store, module_type_store)
    else:
        from matplotlib import rcParams

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib', None, module_type_store, ['rcParams'], [rcParams])

else:
    # Assigning a type to the variable 'matplotlib' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'matplotlib', import_269470)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')



# SSA begins for try-except statement (line 16)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')



# Obtaining the type of the subscript
unicode_269472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 20), 'unicode', u'tk.window_focus')
# Getting the type of 'rcParams' (line 17)
rcParams_269473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 11), 'rcParams')
# Obtaining the member '__getitem__' of a type (line 17)
getitem___269474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 11), rcParams_269473, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 17)
subscript_call_result_269475 = invoke(stypy.reporting.localization.Localization(__file__, 17, 11), getitem___269474, unicode_269472)

# Applying the 'not' unary operator (line 17)
result_not__269476 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 7), 'not', subscript_call_result_269475)

# Testing the type of an if condition (line 17)
if_condition_269477 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 17, 4), result_not__269476)
# Assigning a type to the variable 'if_condition_269477' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'if_condition_269477', if_condition_269477)
# SSA begins for if statement (line 17)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
# Getting the type of 'ImportError' (line 18)
ImportError_269478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 14), 'ImportError')
ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 18, 8), ImportError_269478, 'raise parameter', BaseException)
# SSA join for if statement (line 17)
module_type_store = module_type_store.join_ssa_context()

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 4))

# 'from matplotlib._windowing import GetForegroundWindow, SetForegroundWindow' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/backends/')
import_269479 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 4), 'matplotlib._windowing')

if (type(import_269479) is not StypyTypeError):

    if (import_269479 != 'pyd_module'):
        __import__(import_269479)
        sys_modules_269480 = sys.modules[import_269479]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 4), 'matplotlib._windowing', sys_modules_269480.module_type_store, module_type_store, ['GetForegroundWindow', 'SetForegroundWindow'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 4), __file__, sys_modules_269480, sys_modules_269480.module_type_store, module_type_store)
    else:
        from matplotlib._windowing import GetForegroundWindow, SetForegroundWindow

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 4), 'matplotlib._windowing', None, module_type_store, ['GetForegroundWindow', 'SetForegroundWindow'], [GetForegroundWindow, SetForegroundWindow])

else:
    # Assigning a type to the variable 'matplotlib._windowing' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'matplotlib._windowing', import_269479)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/backends/')

# SSA branch for the except part of a try statement (line 16)
# SSA branch for the except 'ImportError' branch of a try statement (line 16)
module_type_store.open_ssa_branch('except')

@norecursion
def GetForegroundWindow(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'GetForegroundWindow'
    module_type_store = module_type_store.open_function_context('GetForegroundWindow', 21, 4, False)
    
    # Passed parameters checking function
    GetForegroundWindow.stypy_localization = localization
    GetForegroundWindow.stypy_type_of_self = None
    GetForegroundWindow.stypy_type_store = module_type_store
    GetForegroundWindow.stypy_function_name = 'GetForegroundWindow'
    GetForegroundWindow.stypy_param_names_list = []
    GetForegroundWindow.stypy_varargs_param_name = None
    GetForegroundWindow.stypy_kwargs_param_name = None
    GetForegroundWindow.stypy_call_defaults = defaults
    GetForegroundWindow.stypy_call_varargs = varargs
    GetForegroundWindow.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'GetForegroundWindow', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'GetForegroundWindow', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'GetForegroundWindow(...)' code ##################

    int_269481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'stypy_return_type', int_269481)
    
    # ################# End of 'GetForegroundWindow(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'GetForegroundWindow' in the type store
    # Getting the type of 'stypy_return_type' (line 21)
    stypy_return_type_269482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_269482)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'GetForegroundWindow'
    return stypy_return_type_269482

# Assigning a type to the variable 'GetForegroundWindow' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'GetForegroundWindow', GetForegroundWindow)

@norecursion
def SetForegroundWindow(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'SetForegroundWindow'
    module_type_store = module_type_store.open_function_context('SetForegroundWindow', 23, 4, False)
    
    # Passed parameters checking function
    SetForegroundWindow.stypy_localization = localization
    SetForegroundWindow.stypy_type_of_self = None
    SetForegroundWindow.stypy_type_store = module_type_store
    SetForegroundWindow.stypy_function_name = 'SetForegroundWindow'
    SetForegroundWindow.stypy_param_names_list = ['hwnd']
    SetForegroundWindow.stypy_varargs_param_name = None
    SetForegroundWindow.stypy_kwargs_param_name = None
    SetForegroundWindow.stypy_call_defaults = defaults
    SetForegroundWindow.stypy_call_varargs = varargs
    SetForegroundWindow.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'SetForegroundWindow', ['hwnd'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'SetForegroundWindow', localization, ['hwnd'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'SetForegroundWindow(...)' code ##################

    pass
    
    # ################# End of 'SetForegroundWindow(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'SetForegroundWindow' in the type store
    # Getting the type of 'stypy_return_type' (line 23)
    stypy_return_type_269483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_269483)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'SetForegroundWindow'
    return stypy_return_type_269483

# Assigning a type to the variable 'SetForegroundWindow' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'SetForegroundWindow', SetForegroundWindow)
# SSA join for try-except statement (line 16)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'FocusManager' class

class FocusManager(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 27, 4, False)
        # Assigning a type to the variable 'self' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FocusManager.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Attribute (line 28):
        
        # Call to GetForegroundWindow(...): (line 28)
        # Processing the call keyword arguments (line 28)
        kwargs_269485 = {}
        # Getting the type of 'GetForegroundWindow' (line 28)
        GetForegroundWindow_269484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 28), 'GetForegroundWindow', False)
        # Calling GetForegroundWindow(args, kwargs) (line 28)
        GetForegroundWindow_call_result_269486 = invoke(stypy.reporting.localization.Localization(__file__, 28, 28), GetForegroundWindow_269484, *[], **kwargs_269485)
        
        # Getting the type of 'self' (line 28)
        self_269487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'self')
        # Setting the type of the member '_shellWindow' of a type (line 28)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), self_269487, '_shellWindow', GetForegroundWindow_call_result_269486)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __del__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__del__'
        module_type_store = module_type_store.open_function_context('__del__', 30, 4, False)
        # Assigning a type to the variable 'self' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FocusManager.__del__.__dict__.__setitem__('stypy_localization', localization)
        FocusManager.__del__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FocusManager.__del__.__dict__.__setitem__('stypy_type_store', module_type_store)
        FocusManager.__del__.__dict__.__setitem__('stypy_function_name', 'FocusManager.__del__')
        FocusManager.__del__.__dict__.__setitem__('stypy_param_names_list', [])
        FocusManager.__del__.__dict__.__setitem__('stypy_varargs_param_name', None)
        FocusManager.__del__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FocusManager.__del__.__dict__.__setitem__('stypy_call_defaults', defaults)
        FocusManager.__del__.__dict__.__setitem__('stypy_call_varargs', varargs)
        FocusManager.__del__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FocusManager.__del__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FocusManager.__del__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__del__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__del__(...)' code ##################

        
        # Call to SetForegroundWindow(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'self' (line 31)
        self_269489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 28), 'self', False)
        # Obtaining the member '_shellWindow' of a type (line 31)
        _shellWindow_269490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 28), self_269489, '_shellWindow')
        # Processing the call keyword arguments (line 31)
        kwargs_269491 = {}
        # Getting the type of 'SetForegroundWindow' (line 31)
        SetForegroundWindow_269488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'SetForegroundWindow', False)
        # Calling SetForegroundWindow(args, kwargs) (line 31)
        SetForegroundWindow_call_result_269492 = invoke(stypy.reporting.localization.Localization(__file__, 31, 8), SetForegroundWindow_269488, *[_shellWindow_269490], **kwargs_269491)
        
        
        # ################# End of '__del__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__del__' in the type store
        # Getting the type of 'stypy_return_type' (line 30)
        stypy_return_type_269493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_269493)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__del__'
        return stypy_return_type_269493


# Assigning a type to the variable 'FocusManager' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'FocusManager', FocusManager)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
