
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Module defining global singleton classes.
3: 
4: This module raises a RuntimeError if an attempt to reload it is made. In that
5: way the identities of the classes defined here are fixed and will remain so
6: even if numpy itself is reloaded. In particular, a function like the following
7: will still work correctly after numpy is reloaded::
8: 
9:     def foo(arg=np._NoValue):
10:         if arg is np._NoValue:
11:             ...
12: 
13: That was not the case when the singleton classes were defined in the numpy
14: ``__init__.py`` file. See gh-7844 for a discussion of the reload problem that
15: motivated this module.
16: 
17: '''
18: from __future__ import division, absolute_import, print_function
19: 
20: 
21: __ALL__ = [
22:     'ModuleDeprecationWarning', 'VisibleDeprecationWarning', '_NoValue'
23:     ]
24: 
25: 
26: # Disallow reloading this module so as to preserve the identities of the
27: # classes defined here.
28: if '_is_loaded' in globals():
29:     raise RuntimeError('Reloading numpy._globals is not allowed')
30: _is_loaded = True
31: 
32: 
33: class ModuleDeprecationWarning(DeprecationWarning):
34:     '''Module deprecation warning.
35: 
36:     The nose tester turns ordinary Deprecation warnings into test failures.
37:     That makes it hard to deprecate whole modules, because they get
38:     imported by default. So this is a special Deprecation warning that the
39:     nose tester will let pass without making tests fail.
40: 
41:     '''
42:     pass
43: 
44: 
45: class VisibleDeprecationWarning(UserWarning):
46:     '''Visible deprecation warning.
47: 
48:     By default, python will not show deprecation warnings, so this class
49:     can be used when a very visible warning is helpful, for example because
50:     the usage is most likely a user bug.
51: 
52:     '''
53:     pass
54: 
55: 
56: class _NoValue:
57:     '''Special keyword value.
58: 
59:     This class may be used as the default value assigned to a deprecated
60:     keyword in order to check if it has been given a user defined value.
61:     '''
62:     pass
63: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_1808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, (-1)), 'str', '\nModule defining global singleton classes.\n\nThis module raises a RuntimeError if an attempt to reload it is made. In that\nway the identities of the classes defined here are fixed and will remain so\neven if numpy itself is reloaded. In particular, a function like the following\nwill still work correctly after numpy is reloaded::\n\n    def foo(arg=np._NoValue):\n        if arg is np._NoValue:\n            ...\n\nThat was not the case when the singleton classes were defined in the numpy\n``__init__.py`` file. See gh-7844 for a discussion of the reload problem that\nmotivated this module.\n\n')

# Assigning a List to a Name (line 21):

# Obtaining an instance of the builtin type 'list' (line 21)
list_1809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 21)
# Adding element type (line 21)
str_1810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 4), 'str', 'ModuleDeprecationWarning')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), list_1809, str_1810)
# Adding element type (line 21)
str_1811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 32), 'str', 'VisibleDeprecationWarning')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), list_1809, str_1811)
# Adding element type (line 21)
str_1812 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 61), 'str', '_NoValue')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 10), list_1809, str_1812)

# Assigning a type to the variable '__ALL__' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), '__ALL__', list_1809)


str_1813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 3), 'str', '_is_loaded')

# Call to globals(...): (line 28)
# Processing the call keyword arguments (line 28)
kwargs_1815 = {}
# Getting the type of 'globals' (line 28)
globals_1814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 19), 'globals', False)
# Calling globals(args, kwargs) (line 28)
globals_call_result_1816 = invoke(stypy.reporting.localization.Localization(__file__, 28, 19), globals_1814, *[], **kwargs_1815)

# Applying the binary operator 'in' (line 28)
result_contains_1817 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 3), 'in', str_1813, globals_call_result_1816)

# Testing the type of an if condition (line 28)
if_condition_1818 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 28, 0), result_contains_1817)
# Assigning a type to the variable 'if_condition_1818' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'if_condition_1818', if_condition_1818)
# SSA begins for if statement (line 28)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Call to RuntimeError(...): (line 29)
# Processing the call arguments (line 29)
str_1820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 23), 'str', 'Reloading numpy._globals is not allowed')
# Processing the call keyword arguments (line 29)
kwargs_1821 = {}
# Getting the type of 'RuntimeError' (line 29)
RuntimeError_1819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 10), 'RuntimeError', False)
# Calling RuntimeError(args, kwargs) (line 29)
RuntimeError_call_result_1822 = invoke(stypy.reporting.localization.Localization(__file__, 29, 10), RuntimeError_1819, *[str_1820], **kwargs_1821)

ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 29, 4), RuntimeError_call_result_1822, 'raise parameter', BaseException)
# SSA join for if statement (line 28)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Name to a Name (line 30):
# Getting the type of 'True' (line 30)
True_1823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 13), 'True')
# Assigning a type to the variable '_is_loaded' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), '_is_loaded', True_1823)
# Declaration of the 'ModuleDeprecationWarning' class
# Getting the type of 'DeprecationWarning' (line 33)
DeprecationWarning_1824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 31), 'DeprecationWarning')

class ModuleDeprecationWarning(DeprecationWarning_1824, ):
    str_1825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, (-1)), 'str', 'Module deprecation warning.\n\n    The nose tester turns ordinary Deprecation warnings into test failures.\n    That makes it hard to deprecate whole modules, because they get\n    imported by default. So this is a special Deprecation warning that the\n    nose tester will let pass without making tests fail.\n\n    ')
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 33, 0, False)
        # Assigning a type to the variable 'self' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ModuleDeprecationWarning.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ModuleDeprecationWarning' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'ModuleDeprecationWarning', ModuleDeprecationWarning)
# Declaration of the 'VisibleDeprecationWarning' class
# Getting the type of 'UserWarning' (line 45)
UserWarning_1826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 32), 'UserWarning')

class VisibleDeprecationWarning(UserWarning_1826, ):
    str_1827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, (-1)), 'str', 'Visible deprecation warning.\n\n    By default, python will not show deprecation warnings, so this class\n    can be used when a very visible warning is helpful, for example because\n    the usage is most likely a user bug.\n\n    ')
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 45, 0, False)
        # Assigning a type to the variable 'self' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VisibleDeprecationWarning.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'VisibleDeprecationWarning' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'VisibleDeprecationWarning', VisibleDeprecationWarning)
# Declaration of the '_NoValue' class

class _NoValue:
    str_1828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, (-1)), 'str', 'Special keyword value.\n\n    This class may be used as the default value assigned to a deprecated\n    keyword in order to check if it has been given a user defined value.\n    ')
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 56, 0, False)
        # Assigning a type to the variable 'self' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_NoValue.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable '_NoValue' (line 56)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), '_NoValue', _NoValue)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
