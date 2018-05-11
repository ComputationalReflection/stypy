
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Taken from: http://stackoverflow.com/questions/42558/python-and-the-singleton-pattern
3: '''
4: 
5: 
6: class Singleton:
7:     '''
8:     A non-thread-safe helper class to ease implementing singletons.
9:     This should be used as a decorator -- not a metaclass -- to the
10:     class that should be a singleton.
11: 
12:     The decorated class can define one `__init__` function that
13:     takes only the `self` argument. Other than that, there are
14:     no restrictions that apply to the decorated class.
15: 
16:     To get the singleton instance, use the `Instance` method. Trying
17:     to use `__call__` will result in a `TypeError` being raised.
18: 
19:     Limitations: The decorated class cannot be inherited from.
20: 
21:     '''
22: 
23:     def __init__(self, decorated):
24:         self._decorated = decorated
25: 
26:     def Instance(self):
27:         '''
28:         Returns the singleton instance. Upon its first call, it creates a
29:         new instance of the decorated class and calls its `__init__` method.
30:         On all subsequent calls, the already created instance is returned.
31: 
32:         '''
33:         try:
34:             return self._instance
35:         except AttributeError:
36:             self._instance = self._decorated()
37:             return self._instance
38: 
39:     def __call__(self):
40:         raise TypeError('Singletons must be accessed through `Instance()`.')
41: 
42:     def __instancecheck__(self, inst):
43:         return isinstance(inst, self._decorated)
44: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_3363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', '\nTaken from: http://stackoverflow.com/questions/42558/python-and-the-singleton-pattern\n')
# Declaration of the 'Singleton' class

class Singleton:
    str_3364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, (-1)), 'str', '\n    A non-thread-safe helper class to ease implementing singletons.\n    This should be used as a decorator -- not a metaclass -- to the\n    class that should be a singleton.\n\n    The decorated class can define one `__init__` function that\n    takes only the `self` argument. Other than that, there are\n    no restrictions that apply to the decorated class.\n\n    To get the singleton instance, use the `Instance` method. Trying\n    to use `__call__` will result in a `TypeError` being raised.\n\n    Limitations: The decorated class cannot be inherited from.\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 23, 4, False)
        # Assigning a type to the variable 'self' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Singleton.__init__', ['decorated'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['decorated'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 24):
        # Getting the type of 'decorated' (line 24)
        decorated_3365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 26), 'decorated')
        # Getting the type of 'self' (line 24)
        self_3366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'self')
        # Setting the type of the member '_decorated' of a type (line 24)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), self_3366, '_decorated', decorated_3365)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def Instance(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'Instance'
        module_type_store = module_type_store.open_function_context('Instance', 26, 4, False)
        # Assigning a type to the variable 'self' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Singleton.Instance.__dict__.__setitem__('stypy_localization', localization)
        Singleton.Instance.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Singleton.Instance.__dict__.__setitem__('stypy_type_store', module_type_store)
        Singleton.Instance.__dict__.__setitem__('stypy_function_name', 'Singleton.Instance')
        Singleton.Instance.__dict__.__setitem__('stypy_param_names_list', [])
        Singleton.Instance.__dict__.__setitem__('stypy_varargs_param_name', None)
        Singleton.Instance.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Singleton.Instance.__dict__.__setitem__('stypy_call_defaults', defaults)
        Singleton.Instance.__dict__.__setitem__('stypy_call_varargs', varargs)
        Singleton.Instance.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Singleton.Instance.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Singleton.Instance', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'Instance', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'Instance(...)' code ##################

        str_3367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, (-1)), 'str', '\n        Returns the singleton instance. Upon its first call, it creates a\n        new instance of the decorated class and calls its `__init__` method.\n        On all subsequent calls, the already created instance is returned.\n\n        ')
        
        
        # SSA begins for try-except statement (line 33)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        # Getting the type of 'self' (line 34)
        self_3368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 19), 'self')
        # Obtaining the member '_instance' of a type (line 34)
        _instance_3369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 19), self_3368, '_instance')
        # Assigning a type to the variable 'stypy_return_type' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'stypy_return_type', _instance_3369)
        # SSA branch for the except part of a try statement (line 33)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 33)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Call to a Attribute (line 36):
        
        # Call to _decorated(...): (line 36)
        # Processing the call keyword arguments (line 36)
        kwargs_3372 = {}
        # Getting the type of 'self' (line 36)
        self_3370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 29), 'self', False)
        # Obtaining the member '_decorated' of a type (line 36)
        _decorated_3371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 29), self_3370, '_decorated')
        # Calling _decorated(args, kwargs) (line 36)
        _decorated_call_result_3373 = invoke(stypy.reporting.localization.Localization(__file__, 36, 29), _decorated_3371, *[], **kwargs_3372)
        
        # Getting the type of 'self' (line 36)
        self_3374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'self')
        # Setting the type of the member '_instance' of a type (line 36)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 12), self_3374, '_instance', _decorated_call_result_3373)
        # Getting the type of 'self' (line 37)
        self_3375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 19), 'self')
        # Obtaining the member '_instance' of a type (line 37)
        _instance_3376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 19), self_3375, '_instance')
        # Assigning a type to the variable 'stypy_return_type' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'stypy_return_type', _instance_3376)
        # SSA join for try-except statement (line 33)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'Instance(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'Instance' in the type store
        # Getting the type of 'stypy_return_type' (line 26)
        stypy_return_type_3377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3377)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'Instance'
        return stypy_return_type_3377


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 39, 4, False)
        # Assigning a type to the variable 'self' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Singleton.__call__.__dict__.__setitem__('stypy_localization', localization)
        Singleton.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Singleton.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Singleton.__call__.__dict__.__setitem__('stypy_function_name', 'Singleton.__call__')
        Singleton.__call__.__dict__.__setitem__('stypy_param_names_list', [])
        Singleton.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Singleton.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Singleton.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Singleton.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Singleton.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Singleton.__call__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Singleton.__call__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        # Call to TypeError(...): (line 40)
        # Processing the call arguments (line 40)
        str_3379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 24), 'str', 'Singletons must be accessed through `Instance()`.')
        # Processing the call keyword arguments (line 40)
        kwargs_3380 = {}
        # Getting the type of 'TypeError' (line 40)
        TypeError_3378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 14), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 40)
        TypeError_call_result_3381 = invoke(stypy.reporting.localization.Localization(__file__, 40, 14), TypeError_3378, *[str_3379], **kwargs_3380)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 40, 8), TypeError_call_result_3381, 'raise parameter', BaseException)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 39)
        stypy_return_type_3382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3382)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_3382


    @norecursion
    def __instancecheck__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__instancecheck__'
        module_type_store = module_type_store.open_function_context('__instancecheck__', 42, 4, False)
        # Assigning a type to the variable 'self' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Singleton.__instancecheck__.__dict__.__setitem__('stypy_localization', localization)
        Singleton.__instancecheck__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Singleton.__instancecheck__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Singleton.__instancecheck__.__dict__.__setitem__('stypy_function_name', 'Singleton.__instancecheck__')
        Singleton.__instancecheck__.__dict__.__setitem__('stypy_param_names_list', ['inst'])
        Singleton.__instancecheck__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Singleton.__instancecheck__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Singleton.__instancecheck__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Singleton.__instancecheck__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Singleton.__instancecheck__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Singleton.__instancecheck__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Singleton.__instancecheck__', ['inst'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__instancecheck__', localization, ['inst'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__instancecheck__(...)' code ##################

        
        # Call to isinstance(...): (line 43)
        # Processing the call arguments (line 43)
        # Getting the type of 'inst' (line 43)
        inst_3384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 26), 'inst', False)
        # Getting the type of 'self' (line 43)
        self_3385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 32), 'self', False)
        # Obtaining the member '_decorated' of a type (line 43)
        _decorated_3386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 32), self_3385, '_decorated')
        # Processing the call keyword arguments (line 43)
        kwargs_3387 = {}
        # Getting the type of 'isinstance' (line 43)
        isinstance_3383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 43)
        isinstance_call_result_3388 = invoke(stypy.reporting.localization.Localization(__file__, 43, 15), isinstance_3383, *[inst_3384, _decorated_3386], **kwargs_3387)
        
        # Assigning a type to the variable 'stypy_return_type' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'stypy_return_type', isinstance_call_result_3388)
        
        # ################# End of '__instancecheck__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__instancecheck__' in the type store
        # Getting the type of 'stypy_return_type' (line 42)
        stypy_return_type_3389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_3389)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__instancecheck__'
        return stypy_return_type_3389


# Assigning a type to the variable 'Singleton' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'Singleton', Singleton)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
