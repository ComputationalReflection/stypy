
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #===========================================================================
2: #
3: # UnitDblFormatter
4: #
5: #===========================================================================
6: 
7: 
8: '''UnitDblFormatter module containing class UnitDblFormatter.'''
9: 
10: #===========================================================================
11: # Place all imports after here.
12: #
13: from __future__ import (absolute_import, division, print_function,
14:                         unicode_literals)
15: 
16: import six
17: 
18: import matplotlib.ticker as ticker
19: #
20: # Place all imports before here.
21: #===========================================================================
22: 
23: __all__ = [ 'UnitDblFormatter' ]
24: 
25: #===========================================================================
26: class UnitDblFormatter( ticker.ScalarFormatter ):
27:    '''The formatter for UnitDbl data types.  This allows for formatting
28:       with the unit string.
29:    '''
30:    def __init__( self, *args, **kwargs ):
31:       'The arguments are identical to matplotlib.ticker.ScalarFormatter.'
32:       ticker.ScalarFormatter.__init__( self, *args, **kwargs )
33: 
34:    def __call__( self, x, pos = None ):
35:       'Return the format for tick val x at position pos'
36:       if len(self.locs) == 0:
37:          return ''
38:       else:
39:          return str(x)
40: 
41:    def format_data_short( self, value ):
42:       "Return the value formatted in 'short' format."
43:       return str(value)
44: 
45:    def format_data( self, value ):
46:       "Return the value formatted into a string."
47:       return str(value)
48: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

unicode_293925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 0), 'unicode', u'UnitDblFormatter module containing class UnitDblFormatter.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'import six' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
import_293926 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'six')

if (type(import_293926) is not StypyTypeError):

    if (import_293926 != 'pyd_module'):
        __import__(import_293926)
        sys_modules_293927 = sys.modules[import_293926]
        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'six', sys_modules_293927.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'six', import_293926)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'import matplotlib.ticker' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')
import_293928 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'matplotlib.ticker')

if (type(import_293928) is not StypyTypeError):

    if (import_293928 != 'pyd_module'):
        __import__(import_293928)
        sys_modules_293929 = sys.modules[import_293928]
        import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'ticker', sys_modules_293929.module_type_store, module_type_store)
    else:
        import matplotlib.ticker as ticker

        import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'ticker', matplotlib.ticker, module_type_store)

else:
    # Assigning a type to the variable 'matplotlib.ticker' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'matplotlib.ticker', import_293928)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/jpl_units/')


# Assigning a List to a Name (line 23):
__all__ = [u'UnitDblFormatter']
module_type_store.set_exportable_members([u'UnitDblFormatter'])

# Obtaining an instance of the builtin type 'list' (line 23)
list_293930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 23)
# Adding element type (line 23)
unicode_293931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 12), 'unicode', u'UnitDblFormatter')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 10), list_293930, unicode_293931)

# Assigning a type to the variable '__all__' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), '__all__', list_293930)
# Declaration of the 'UnitDblFormatter' class
# Getting the type of 'ticker' (line 26)
ticker_293932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 24), 'ticker')
# Obtaining the member 'ScalarFormatter' of a type (line 26)
ScalarFormatter_293933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 24), ticker_293932, 'ScalarFormatter')

class UnitDblFormatter(ScalarFormatter_293933, ):
    unicode_293934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, (-1)), 'unicode', u'The formatter for UnitDbl data types.  This allows for formatting\n      with the unit string.\n   ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 30, 3, False)
        # Assigning a type to the variable 'self' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnitDblFormatter.__init__', [], 'args', 'kwargs', defaults, varargs, kwargs)

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

        unicode_293935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 6), 'unicode', u'The arguments are identical to matplotlib.ticker.ScalarFormatter.')
        
        # Call to __init__(...): (line 32)
        # Processing the call arguments (line 32)
        # Getting the type of 'self' (line 32)
        self_293939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 39), 'self', False)
        # Getting the type of 'args' (line 32)
        args_293940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 46), 'args', False)
        # Processing the call keyword arguments (line 32)
        # Getting the type of 'kwargs' (line 32)
        kwargs_293941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 54), 'kwargs', False)
        kwargs_293942 = {'kwargs_293941': kwargs_293941}
        # Getting the type of 'ticker' (line 32)
        ticker_293936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 6), 'ticker', False)
        # Obtaining the member 'ScalarFormatter' of a type (line 32)
        ScalarFormatter_293937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 6), ticker_293936, 'ScalarFormatter')
        # Obtaining the member '__init__' of a type (line 32)
        init___293938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 6), ScalarFormatter_293937, '__init__')
        # Calling __init__(args, kwargs) (line 32)
        init___call_result_293943 = invoke(stypy.reporting.localization.Localization(__file__, 32, 6), init___293938, *[self_293939, args_293940], **kwargs_293942)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 34)
        None_293944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 32), 'None')
        defaults = [None_293944]
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 34, 3, False)
        # Assigning a type to the variable 'self' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        UnitDblFormatter.__call__.__dict__.__setitem__('stypy_localization', localization)
        UnitDblFormatter.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnitDblFormatter.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnitDblFormatter.__call__.__dict__.__setitem__('stypy_function_name', 'UnitDblFormatter.__call__')
        UnitDblFormatter.__call__.__dict__.__setitem__('stypy_param_names_list', ['x', 'pos'])
        UnitDblFormatter.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnitDblFormatter.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnitDblFormatter.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnitDblFormatter.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnitDblFormatter.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnitDblFormatter.__call__.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnitDblFormatter.__call__', ['x', 'pos'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['x', 'pos'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        unicode_293945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 6), 'unicode', u'Return the format for tick val x at position pos')
        
        
        
        # Call to len(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'self' (line 36)
        self_293947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 13), 'self', False)
        # Obtaining the member 'locs' of a type (line 36)
        locs_293948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 13), self_293947, 'locs')
        # Processing the call keyword arguments (line 36)
        kwargs_293949 = {}
        # Getting the type of 'len' (line 36)
        len_293946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 9), 'len', False)
        # Calling len(args, kwargs) (line 36)
        len_call_result_293950 = invoke(stypy.reporting.localization.Localization(__file__, 36, 9), len_293946, *[locs_293948], **kwargs_293949)
        
        int_293951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 27), 'int')
        # Applying the binary operator '==' (line 36)
        result_eq_293952 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 9), '==', len_call_result_293950, int_293951)
        
        # Testing the type of an if condition (line 36)
        if_condition_293953 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 36, 6), result_eq_293952)
        # Assigning a type to the variable 'if_condition_293953' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 6), 'if_condition_293953', if_condition_293953)
        # SSA begins for if statement (line 36)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        unicode_293954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 16), 'unicode', u'')
        # Assigning a type to the variable 'stypy_return_type' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 9), 'stypy_return_type', unicode_293954)
        # SSA branch for the else part of an if statement (line 36)
        module_type_store.open_ssa_branch('else')
        
        # Call to str(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'x' (line 39)
        x_293956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 20), 'x', False)
        # Processing the call keyword arguments (line 39)
        kwargs_293957 = {}
        # Getting the type of 'str' (line 39)
        str_293955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 16), 'str', False)
        # Calling str(args, kwargs) (line 39)
        str_call_result_293958 = invoke(stypy.reporting.localization.Localization(__file__, 39, 16), str_293955, *[x_293956], **kwargs_293957)
        
        # Assigning a type to the variable 'stypy_return_type' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 9), 'stypy_return_type', str_call_result_293958)
        # SSA join for if statement (line 36)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 34)
        stypy_return_type_293959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_293959)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_293959


    @norecursion
    def format_data_short(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'format_data_short'
        module_type_store = module_type_store.open_function_context('format_data_short', 41, 3, False)
        # Assigning a type to the variable 'self' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        UnitDblFormatter.format_data_short.__dict__.__setitem__('stypy_localization', localization)
        UnitDblFormatter.format_data_short.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnitDblFormatter.format_data_short.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnitDblFormatter.format_data_short.__dict__.__setitem__('stypy_function_name', 'UnitDblFormatter.format_data_short')
        UnitDblFormatter.format_data_short.__dict__.__setitem__('stypy_param_names_list', ['value'])
        UnitDblFormatter.format_data_short.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnitDblFormatter.format_data_short.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnitDblFormatter.format_data_short.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnitDblFormatter.format_data_short.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnitDblFormatter.format_data_short.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnitDblFormatter.format_data_short.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnitDblFormatter.format_data_short', ['value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'format_data_short', localization, ['value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'format_data_short(...)' code ##################

        unicode_293960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 6), 'unicode', u"Return the value formatted in 'short' format.")
        
        # Call to str(...): (line 43)
        # Processing the call arguments (line 43)
        # Getting the type of 'value' (line 43)
        value_293962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 17), 'value', False)
        # Processing the call keyword arguments (line 43)
        kwargs_293963 = {}
        # Getting the type of 'str' (line 43)
        str_293961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 13), 'str', False)
        # Calling str(args, kwargs) (line 43)
        str_call_result_293964 = invoke(stypy.reporting.localization.Localization(__file__, 43, 13), str_293961, *[value_293962], **kwargs_293963)
        
        # Assigning a type to the variable 'stypy_return_type' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 6), 'stypy_return_type', str_call_result_293964)
        
        # ################# End of 'format_data_short(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'format_data_short' in the type store
        # Getting the type of 'stypy_return_type' (line 41)
        stypy_return_type_293965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_293965)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'format_data_short'
        return stypy_return_type_293965


    @norecursion
    def format_data(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'format_data'
        module_type_store = module_type_store.open_function_context('format_data', 45, 3, False)
        # Assigning a type to the variable 'self' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 3), 'self', type_of_self)
        
        # Passed parameters checking function
        UnitDblFormatter.format_data.__dict__.__setitem__('stypy_localization', localization)
        UnitDblFormatter.format_data.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UnitDblFormatter.format_data.__dict__.__setitem__('stypy_type_store', module_type_store)
        UnitDblFormatter.format_data.__dict__.__setitem__('stypy_function_name', 'UnitDblFormatter.format_data')
        UnitDblFormatter.format_data.__dict__.__setitem__('stypy_param_names_list', ['value'])
        UnitDblFormatter.format_data.__dict__.__setitem__('stypy_varargs_param_name', None)
        UnitDblFormatter.format_data.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UnitDblFormatter.format_data.__dict__.__setitem__('stypy_call_defaults', defaults)
        UnitDblFormatter.format_data.__dict__.__setitem__('stypy_call_varargs', varargs)
        UnitDblFormatter.format_data.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UnitDblFormatter.format_data.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnitDblFormatter.format_data', ['value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'format_data', localization, ['value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'format_data(...)' code ##################

        unicode_293966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 6), 'unicode', u'Return the value formatted into a string.')
        
        # Call to str(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'value' (line 47)
        value_293968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 17), 'value', False)
        # Processing the call keyword arguments (line 47)
        kwargs_293969 = {}
        # Getting the type of 'str' (line 47)
        str_293967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 13), 'str', False)
        # Calling str(args, kwargs) (line 47)
        str_call_result_293970 = invoke(stypy.reporting.localization.Localization(__file__, 47, 13), str_293967, *[value_293968], **kwargs_293969)
        
        # Assigning a type to the variable 'stypy_return_type' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 6), 'stypy_return_type', str_call_result_293970)
        
        # ################# End of 'format_data(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'format_data' in the type store
        # Getting the type of 'stypy_return_type' (line 45)
        stypy_return_type_293971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 3), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_293971)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'format_data'
        return stypy_return_type_293971


# Assigning a type to the variable 'UnitDblFormatter' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'UnitDblFormatter', UnitDblFormatter)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
