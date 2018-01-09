
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: class TestCase(object):
2:     def assertEqual(self):
3:         return None
4: 
5:     assertEquals = assertEqual
6: 
7:     def _deprecate(original_func):
8:         def deprecated_func(*args, **kwargs):
9:             return original_func(*args, **kwargs)
10: 
11:         return deprecated_func
12: 
13:     failUnlessEqual = _deprecate(assertEqual)
14: 
15: 
16: t = TestCase()
17: 
18: r = t.assertEquals
19: 
20: r2 = t.failUnlessEqual
21: 
22: print r
23: print r2
24: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

# Declaration of the 'TestCase' class

class TestCase(object, ):

    @norecursion
    def assertEqual(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'assertEqual'
        module_type_store = module_type_store.open_function_context('assertEqual', 2, 4, False)
        # Assigning a type to the variable 'self' (line 3)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestCase.assertEqual.__dict__.__setitem__('stypy_localization', localization)
        TestCase.assertEqual.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestCase.assertEqual.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase.assertEqual.__dict__.__setitem__('stypy_function_name', 'TestCase.assertEqual')
        TestCase.assertEqual.__dict__.__setitem__('stypy_param_names_list', [])
        TestCase.assertEqual.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase.assertEqual.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase.assertEqual.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase.assertEqual.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase.assertEqual.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase.assertEqual.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.assertEqual', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'assertEqual', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'assertEqual(...)' code ##################

        # Getting the type of 'None' (line 3)
        None_1088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 3, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 3)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 8), 'stypy_return_type', None_1088)
        
        # ################# End of 'assertEqual(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'assertEqual' in the type store
        # Getting the type of 'stypy_return_type' (line 2)
        stypy_return_type_1089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 2, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1089)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'assertEqual'
        return stypy_return_type_1089


    @staticmethod
    @norecursion
    def _deprecate(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_deprecate'
        module_type_store = module_type_store.open_function_context('_deprecate', 7, 4, False)
        
        # Passed parameters checking function
        TestCase._deprecate.__dict__.__setitem__('stypy_localization', localization)
        TestCase._deprecate.__dict__.__setitem__('stypy_type_of_self', None)
        TestCase._deprecate.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestCase._deprecate.__dict__.__setitem__('stypy_function_name', '_deprecate')
        TestCase._deprecate.__dict__.__setitem__('stypy_param_names_list', ['original_func'])
        TestCase._deprecate.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestCase._deprecate.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestCase._deprecate.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestCase._deprecate.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestCase._deprecate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestCase._deprecate.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, None, module_type_store, '_deprecate', ['original_func'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_deprecate', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_deprecate(...)' code ##################


        @norecursion
        def deprecated_func(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'deprecated_func'
            module_type_store = module_type_store.open_function_context('deprecated_func', 8, 8, False)
            
            # Passed parameters checking function
            deprecated_func.stypy_localization = localization
            deprecated_func.stypy_type_of_self = None
            deprecated_func.stypy_type_store = module_type_store
            deprecated_func.stypy_function_name = 'deprecated_func'
            deprecated_func.stypy_param_names_list = []
            deprecated_func.stypy_varargs_param_name = 'args'
            deprecated_func.stypy_kwargs_param_name = 'kwargs'
            deprecated_func.stypy_call_defaults = defaults
            deprecated_func.stypy_call_varargs = varargs
            deprecated_func.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'deprecated_func', [], 'args', 'kwargs', defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'deprecated_func', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'deprecated_func(...)' code ##################

            
            # Call to original_func(...): (line 9)
            # Getting the type of 'args' (line 9)
            args_1091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 34), 'args', False)
            # Processing the call keyword arguments (line 9)
            # Getting the type of 'kwargs' (line 9)
            kwargs_1092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 42), 'kwargs', False)
            kwargs_1093 = {'kwargs_1092': kwargs_1092}
            # Getting the type of 'original_func' (line 9)
            original_func_1090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 19), 'original_func', False)
            # Calling original_func(args, kwargs) (line 9)
            original_func_call_result_1094 = invoke(stypy.reporting.localization.Localization(__file__, 9, 19), original_func_1090, *[args_1091], **kwargs_1093)
            
            # Assigning a type to the variable 'stypy_return_type' (line 9)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 12), 'stypy_return_type', original_func_call_result_1094)
            
            # ################# End of 'deprecated_func(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'deprecated_func' in the type store
            # Getting the type of 'stypy_return_type' (line 8)
            stypy_return_type_1095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_1095)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'deprecated_func'
            return stypy_return_type_1095

        # Assigning a type to the variable 'deprecated_func' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 8), 'deprecated_func', deprecated_func)
        # Getting the type of 'deprecated_func' (line 11)
        deprecated_func_1096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 15), 'deprecated_func')
        # Assigning a type to the variable 'stypy_return_type' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'stypy_return_type', deprecated_func_1096)
        
        # ################# End of '_deprecate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_deprecate' in the type store
        # Getting the type of 'stypy_return_type' (line 7)
        stypy_return_type_1097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1097)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_deprecate'
        return stypy_return_type_1097


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 1, 0, False)
        # Assigning a type to the variable 'self' (line 2)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestCase.__init__', [], None, None, defaults, varargs, kwargs)

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

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'TestCase' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'TestCase', TestCase)

# Assigning a Name to a Name (line 5):
# Getting the type of 'TestCase'
TestCase_1098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase')
# Obtaining the member 'assertEqual' of a type
assertEqual_1099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_1098, 'assertEqual')
# Getting the type of 'TestCase'
TestCase_1100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase')
# Setting the type of the member 'assertEquals' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_1100, 'assertEquals', assertEqual_1099)

# Assigning a Call to a Name (line 13):

# Call to _deprecate(...): (line 13)
# Processing the call arguments (line 13)
# Getting the type of 'TestCase'
TestCase_1103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase', False)
# Obtaining the member 'assertEqual' of a type
assertEqual_1104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_1103, 'assertEqual')
# Processing the call keyword arguments (line 13)
kwargs_1105 = {}
# Getting the type of 'TestCase'
TestCase_1101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase', False)
# Obtaining the member '_deprecate' of a type
_deprecate_1102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_1101, '_deprecate')
# Calling _deprecate(args, kwargs) (line 13)
_deprecate_call_result_1106 = invoke(stypy.reporting.localization.Localization(__file__, 13, 22), _deprecate_1102, *[assertEqual_1104], **kwargs_1105)

# Getting the type of 'TestCase'
TestCase_1107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TestCase')
# Setting the type of the member 'failUnlessEqual' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TestCase_1107, 'failUnlessEqual', _deprecate_call_result_1106)

# Assigning a Call to a Name (line 16):

# Call to TestCase(...): (line 16)
# Processing the call keyword arguments (line 16)
kwargs_1109 = {}
# Getting the type of 'TestCase' (line 16)
TestCase_1108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'TestCase', False)
# Calling TestCase(args, kwargs) (line 16)
TestCase_call_result_1110 = invoke(stypy.reporting.localization.Localization(__file__, 16, 4), TestCase_1108, *[], **kwargs_1109)

# Assigning a type to the variable 't' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 't', TestCase_call_result_1110)

# Assigning a Attribute to a Name (line 18):
# Getting the type of 't' (line 18)
t_1111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 't')
# Obtaining the member 'assertEquals' of a type (line 18)
assertEquals_1112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 4), t_1111, 'assertEquals')
# Assigning a type to the variable 'r' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'r', assertEquals_1112)

# Assigning a Attribute to a Name (line 20):
# Getting the type of 't' (line 20)
t_1113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 5), 't')
# Obtaining the member 'failUnlessEqual' of a type (line 20)
failUnlessEqual_1114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 5), t_1113, 'failUnlessEqual')
# Assigning a type to the variable 'r2' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'r2', failUnlessEqual_1114)
# Getting the type of 'r' (line 22)
r_1115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 6), 'r')
# Getting the type of 'r2' (line 23)
r2_1116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 6), 'r2')

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
