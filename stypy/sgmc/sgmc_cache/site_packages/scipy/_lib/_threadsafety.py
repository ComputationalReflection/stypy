
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import threading
4: 
5: import scipy._lib.decorator
6: 
7: 
8: __all__ = ['ReentrancyError', 'ReentrancyLock', 'non_reentrant']
9: 
10: 
11: class ReentrancyError(RuntimeError):
12:     pass
13: 
14: 
15: class ReentrancyLock(object):
16:     '''
17:     Threading lock that raises an exception for reentrant calls.
18: 
19:     Calls from different threads are serialized, and nested calls from the
20:     same thread result to an error.
21: 
22:     The object can be used as a context manager, or to decorate functions
23:     via the decorate() method.
24: 
25:     '''
26: 
27:     def __init__(self, err_msg):
28:         self._rlock = threading.RLock()
29:         self._entered = False
30:         self._err_msg = err_msg
31: 
32:     def __enter__(self):
33:         self._rlock.acquire()
34:         if self._entered:
35:             self._rlock.release()
36:             raise ReentrancyError(self._err_msg)
37:         self._entered = True
38: 
39:     def __exit__(self, type, value, traceback):
40:         self._entered = False
41:         self._rlock.release()
42: 
43:     def decorate(self, func):
44:         def caller(func, *a, **kw):
45:             with self:
46:                 return func(*a, **kw)
47:         return scipy._lib.decorator.decorate(func, caller)
48: 
49: 
50: def non_reentrant(err_msg=None):
51:     '''
52:     Decorate a function with a threading lock and prevent reentrant calls.
53:     '''
54:     def decorator(func):
55:         msg = err_msg
56:         if msg is None:
57:             msg = "%s is not re-entrant" % func.__name__
58:         lock = ReentrancyLock(msg)
59:         return lock.decorate(func)
60:     return decorator
61: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import threading' statement (line 3)
import threading

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'threading', threading, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import scipy._lib.decorator' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/')
import_709923 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy._lib.decorator')

if (type(import_709923) is not StypyTypeError):

    if (import_709923 != 'pyd_module'):
        __import__(import_709923)
        sys_modules_709924 = sys.modules[import_709923]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy._lib.decorator', sys_modules_709924.module_type_store, module_type_store)
    else:
        import scipy._lib.decorator

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy._lib.decorator', scipy._lib.decorator, module_type_store)

else:
    # Assigning a type to the variable 'scipy._lib.decorator' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'scipy._lib.decorator', import_709923)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/')


# Assigning a List to a Name (line 8):
__all__ = ['ReentrancyError', 'ReentrancyLock', 'non_reentrant']
module_type_store.set_exportable_members(['ReentrancyError', 'ReentrancyLock', 'non_reentrant'])

# Obtaining an instance of the builtin type 'list' (line 8)
list_709925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 8)
# Adding element type (line 8)
str_709926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 11), 'str', 'ReentrancyError')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_709925, str_709926)
# Adding element type (line 8)
str_709927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 30), 'str', 'ReentrancyLock')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_709925, str_709927)
# Adding element type (line 8)
str_709928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 48), 'str', 'non_reentrant')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_709925, str_709928)

# Assigning a type to the variable '__all__' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), '__all__', list_709925)
# Declaration of the 'ReentrancyError' class
# Getting the type of 'RuntimeError' (line 11)
RuntimeError_709929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 22), 'RuntimeError')

class ReentrancyError(RuntimeError_709929, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 11, 0, False)
        # Assigning a type to the variable 'self' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ReentrancyError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ReentrancyError' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'ReentrancyError', ReentrancyError)
# Declaration of the 'ReentrancyLock' class

class ReentrancyLock(object, ):
    str_709930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, (-1)), 'str', '\n    Threading lock that raises an exception for reentrant calls.\n\n    Calls from different threads are serialized, and nested calls from the\n    same thread result to an error.\n\n    The object can be used as a context manager, or to decorate functions\n    via the decorate() method.\n\n    ')

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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ReentrancyLock.__init__', ['err_msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['err_msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Attribute (line 28):
        
        # Call to RLock(...): (line 28)
        # Processing the call keyword arguments (line 28)
        kwargs_709933 = {}
        # Getting the type of 'threading' (line 28)
        threading_709931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 22), 'threading', False)
        # Obtaining the member 'RLock' of a type (line 28)
        RLock_709932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 22), threading_709931, 'RLock')
        # Calling RLock(args, kwargs) (line 28)
        RLock_call_result_709934 = invoke(stypy.reporting.localization.Localization(__file__, 28, 22), RLock_709932, *[], **kwargs_709933)
        
        # Getting the type of 'self' (line 28)
        self_709935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'self')
        # Setting the type of the member '_rlock' of a type (line 28)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), self_709935, '_rlock', RLock_call_result_709934)
        
        # Assigning a Name to a Attribute (line 29):
        # Getting the type of 'False' (line 29)
        False_709936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 24), 'False')
        # Getting the type of 'self' (line 29)
        self_709937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'self')
        # Setting the type of the member '_entered' of a type (line 29)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), self_709937, '_entered', False_709936)
        
        # Assigning a Name to a Attribute (line 30):
        # Getting the type of 'err_msg' (line 30)
        err_msg_709938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 24), 'err_msg')
        # Getting the type of 'self' (line 30)
        self_709939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'self')
        # Setting the type of the member '_err_msg' of a type (line 30)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), self_709939, '_err_msg', err_msg_709938)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __enter__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__enter__'
        module_type_store = module_type_store.open_function_context('__enter__', 32, 4, False)
        # Assigning a type to the variable 'self' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ReentrancyLock.__enter__.__dict__.__setitem__('stypy_localization', localization)
        ReentrancyLock.__enter__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ReentrancyLock.__enter__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ReentrancyLock.__enter__.__dict__.__setitem__('stypy_function_name', 'ReentrancyLock.__enter__')
        ReentrancyLock.__enter__.__dict__.__setitem__('stypy_param_names_list', [])
        ReentrancyLock.__enter__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ReentrancyLock.__enter__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ReentrancyLock.__enter__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ReentrancyLock.__enter__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ReentrancyLock.__enter__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ReentrancyLock.__enter__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ReentrancyLock.__enter__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__enter__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__enter__(...)' code ##################

        
        # Call to acquire(...): (line 33)
        # Processing the call keyword arguments (line 33)
        kwargs_709943 = {}
        # Getting the type of 'self' (line 33)
        self_709940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'self', False)
        # Obtaining the member '_rlock' of a type (line 33)
        _rlock_709941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), self_709940, '_rlock')
        # Obtaining the member 'acquire' of a type (line 33)
        acquire_709942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), _rlock_709941, 'acquire')
        # Calling acquire(args, kwargs) (line 33)
        acquire_call_result_709944 = invoke(stypy.reporting.localization.Localization(__file__, 33, 8), acquire_709942, *[], **kwargs_709943)
        
        
        # Getting the type of 'self' (line 34)
        self_709945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 11), 'self')
        # Obtaining the member '_entered' of a type (line 34)
        _entered_709946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 11), self_709945, '_entered')
        # Testing the type of an if condition (line 34)
        if_condition_709947 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 34, 8), _entered_709946)
        # Assigning a type to the variable 'if_condition_709947' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'if_condition_709947', if_condition_709947)
        # SSA begins for if statement (line 34)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to release(...): (line 35)
        # Processing the call keyword arguments (line 35)
        kwargs_709951 = {}
        # Getting the type of 'self' (line 35)
        self_709948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'self', False)
        # Obtaining the member '_rlock' of a type (line 35)
        _rlock_709949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 12), self_709948, '_rlock')
        # Obtaining the member 'release' of a type (line 35)
        release_709950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 12), _rlock_709949, 'release')
        # Calling release(args, kwargs) (line 35)
        release_call_result_709952 = invoke(stypy.reporting.localization.Localization(__file__, 35, 12), release_709950, *[], **kwargs_709951)
        
        
        # Call to ReentrancyError(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'self' (line 36)
        self_709954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 34), 'self', False)
        # Obtaining the member '_err_msg' of a type (line 36)
        _err_msg_709955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 34), self_709954, '_err_msg')
        # Processing the call keyword arguments (line 36)
        kwargs_709956 = {}
        # Getting the type of 'ReentrancyError' (line 36)
        ReentrancyError_709953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 18), 'ReentrancyError', False)
        # Calling ReentrancyError(args, kwargs) (line 36)
        ReentrancyError_call_result_709957 = invoke(stypy.reporting.localization.Localization(__file__, 36, 18), ReentrancyError_709953, *[_err_msg_709955], **kwargs_709956)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 36, 12), ReentrancyError_call_result_709957, 'raise parameter', BaseException)
        # SSA join for if statement (line 34)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 37):
        # Getting the type of 'True' (line 37)
        True_709958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 24), 'True')
        # Getting the type of 'self' (line 37)
        self_709959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'self')
        # Setting the type of the member '_entered' of a type (line 37)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), self_709959, '_entered', True_709958)
        
        # ################# End of '__enter__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__enter__' in the type store
        # Getting the type of 'stypy_return_type' (line 32)
        stypy_return_type_709960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_709960)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__enter__'
        return stypy_return_type_709960


    @norecursion
    def __exit__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__exit__'
        module_type_store = module_type_store.open_function_context('__exit__', 39, 4, False)
        # Assigning a type to the variable 'self' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ReentrancyLock.__exit__.__dict__.__setitem__('stypy_localization', localization)
        ReentrancyLock.__exit__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ReentrancyLock.__exit__.__dict__.__setitem__('stypy_type_store', module_type_store)
        ReentrancyLock.__exit__.__dict__.__setitem__('stypy_function_name', 'ReentrancyLock.__exit__')
        ReentrancyLock.__exit__.__dict__.__setitem__('stypy_param_names_list', ['type', 'value', 'traceback'])
        ReentrancyLock.__exit__.__dict__.__setitem__('stypy_varargs_param_name', None)
        ReentrancyLock.__exit__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ReentrancyLock.__exit__.__dict__.__setitem__('stypy_call_defaults', defaults)
        ReentrancyLock.__exit__.__dict__.__setitem__('stypy_call_varargs', varargs)
        ReentrancyLock.__exit__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ReentrancyLock.__exit__.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ReentrancyLock.__exit__', ['type', 'value', 'traceback'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__exit__', localization, ['type', 'value', 'traceback'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__exit__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 40):
        # Getting the type of 'False' (line 40)
        False_709961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 24), 'False')
        # Getting the type of 'self' (line 40)
        self_709962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'self')
        # Setting the type of the member '_entered' of a type (line 40)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), self_709962, '_entered', False_709961)
        
        # Call to release(...): (line 41)
        # Processing the call keyword arguments (line 41)
        kwargs_709966 = {}
        # Getting the type of 'self' (line 41)
        self_709963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'self', False)
        # Obtaining the member '_rlock' of a type (line 41)
        _rlock_709964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), self_709963, '_rlock')
        # Obtaining the member 'release' of a type (line 41)
        release_709965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), _rlock_709964, 'release')
        # Calling release(args, kwargs) (line 41)
        release_call_result_709967 = invoke(stypy.reporting.localization.Localization(__file__, 41, 8), release_709965, *[], **kwargs_709966)
        
        
        # ################# End of '__exit__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__exit__' in the type store
        # Getting the type of 'stypy_return_type' (line 39)
        stypy_return_type_709968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_709968)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__exit__'
        return stypy_return_type_709968


    @norecursion
    def decorate(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'decorate'
        module_type_store = module_type_store.open_function_context('decorate', 43, 4, False)
        # Assigning a type to the variable 'self' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        ReentrancyLock.decorate.__dict__.__setitem__('stypy_localization', localization)
        ReentrancyLock.decorate.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        ReentrancyLock.decorate.__dict__.__setitem__('stypy_type_store', module_type_store)
        ReentrancyLock.decorate.__dict__.__setitem__('stypy_function_name', 'ReentrancyLock.decorate')
        ReentrancyLock.decorate.__dict__.__setitem__('stypy_param_names_list', ['func'])
        ReentrancyLock.decorate.__dict__.__setitem__('stypy_varargs_param_name', None)
        ReentrancyLock.decorate.__dict__.__setitem__('stypy_kwargs_param_name', None)
        ReentrancyLock.decorate.__dict__.__setitem__('stypy_call_defaults', defaults)
        ReentrancyLock.decorate.__dict__.__setitem__('stypy_call_varargs', varargs)
        ReentrancyLock.decorate.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        ReentrancyLock.decorate.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ReentrancyLock.decorate', ['func'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'decorate', localization, ['func'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'decorate(...)' code ##################


        @norecursion
        def caller(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'caller'
            module_type_store = module_type_store.open_function_context('caller', 44, 8, False)
            
            # Passed parameters checking function
            caller.stypy_localization = localization
            caller.stypy_type_of_self = None
            caller.stypy_type_store = module_type_store
            caller.stypy_function_name = 'caller'
            caller.stypy_param_names_list = ['func']
            caller.stypy_varargs_param_name = 'a'
            caller.stypy_kwargs_param_name = 'kw'
            caller.stypy_call_defaults = defaults
            caller.stypy_call_varargs = varargs
            caller.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'caller', ['func'], 'a', 'kw', defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'caller', localization, ['func'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'caller(...)' code ##################

            # Getting the type of 'self' (line 45)
            self_709969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 17), 'self')
            with_709970 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 45, 17), self_709969, 'with parameter', '__enter__', '__exit__')

            if with_709970:
                # Calling the __enter__ method to initiate a with section
                # Obtaining the member '__enter__' of a type (line 45)
                enter___709971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 17), self_709969, '__enter__')
                with_enter_709972 = invoke(stypy.reporting.localization.Localization(__file__, 45, 17), enter___709971)
                
                # Call to func(...): (line 46)
                # Getting the type of 'a' (line 46)
                a_709974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 29), 'a', False)
                # Processing the call keyword arguments (line 46)
                # Getting the type of 'kw' (line 46)
                kw_709975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 34), 'kw', False)
                kwargs_709976 = {'kw_709975': kw_709975}
                # Getting the type of 'func' (line 46)
                func_709973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 23), 'func', False)
                # Calling func(args, kwargs) (line 46)
                func_call_result_709977 = invoke(stypy.reporting.localization.Localization(__file__, 46, 23), func_709973, *[a_709974], **kwargs_709976)
                
                # Assigning a type to the variable 'stypy_return_type' (line 46)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 16), 'stypy_return_type', func_call_result_709977)
                # Calling the __exit__ method to finish a with section
                # Obtaining the member '__exit__' of a type (line 45)
                exit___709978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 17), self_709969, '__exit__')
                with_exit_709979 = invoke(stypy.reporting.localization.Localization(__file__, 45, 17), exit___709978, None, None, None)

            
            # ################# End of 'caller(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'caller' in the type store
            # Getting the type of 'stypy_return_type' (line 44)
            stypy_return_type_709980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_709980)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'caller'
            return stypy_return_type_709980

        # Assigning a type to the variable 'caller' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'caller', caller)
        
        # Call to decorate(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'func' (line 47)
        func_709985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 45), 'func', False)
        # Getting the type of 'caller' (line 47)
        caller_709986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 51), 'caller', False)
        # Processing the call keyword arguments (line 47)
        kwargs_709987 = {}
        # Getting the type of 'scipy' (line 47)
        scipy_709981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 15), 'scipy', False)
        # Obtaining the member '_lib' of a type (line 47)
        _lib_709982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 15), scipy_709981, '_lib')
        # Obtaining the member 'decorator' of a type (line 47)
        decorator_709983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 15), _lib_709982, 'decorator')
        # Obtaining the member 'decorate' of a type (line 47)
        decorate_709984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 15), decorator_709983, 'decorate')
        # Calling decorate(args, kwargs) (line 47)
        decorate_call_result_709988 = invoke(stypy.reporting.localization.Localization(__file__, 47, 15), decorate_709984, *[func_709985, caller_709986], **kwargs_709987)
        
        # Assigning a type to the variable 'stypy_return_type' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'stypy_return_type', decorate_call_result_709988)
        
        # ################# End of 'decorate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'decorate' in the type store
        # Getting the type of 'stypy_return_type' (line 43)
        stypy_return_type_709989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_709989)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'decorate'
        return stypy_return_type_709989


# Assigning a type to the variable 'ReentrancyLock' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'ReentrancyLock', ReentrancyLock)

@norecursion
def non_reentrant(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 50)
    None_709990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 26), 'None')
    defaults = [None_709990]
    # Create a new context for function 'non_reentrant'
    module_type_store = module_type_store.open_function_context('non_reentrant', 50, 0, False)
    
    # Passed parameters checking function
    non_reentrant.stypy_localization = localization
    non_reentrant.stypy_type_of_self = None
    non_reentrant.stypy_type_store = module_type_store
    non_reentrant.stypy_function_name = 'non_reentrant'
    non_reentrant.stypy_param_names_list = ['err_msg']
    non_reentrant.stypy_varargs_param_name = None
    non_reentrant.stypy_kwargs_param_name = None
    non_reentrant.stypy_call_defaults = defaults
    non_reentrant.stypy_call_varargs = varargs
    non_reentrant.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'non_reentrant', ['err_msg'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'non_reentrant', localization, ['err_msg'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'non_reentrant(...)' code ##################

    str_709991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, (-1)), 'str', '\n    Decorate a function with a threading lock and prevent reentrant calls.\n    ')

    @norecursion
    def decorator(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'decorator'
        module_type_store = module_type_store.open_function_context('decorator', 54, 4, False)
        
        # Passed parameters checking function
        decorator.stypy_localization = localization
        decorator.stypy_type_of_self = None
        decorator.stypy_type_store = module_type_store
        decorator.stypy_function_name = 'decorator'
        decorator.stypy_param_names_list = ['func']
        decorator.stypy_varargs_param_name = None
        decorator.stypy_kwargs_param_name = None
        decorator.stypy_call_defaults = defaults
        decorator.stypy_call_varargs = varargs
        decorator.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'decorator', ['func'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'decorator', localization, ['func'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'decorator(...)' code ##################

        
        # Assigning a Name to a Name (line 55):
        # Getting the type of 'err_msg' (line 55)
        err_msg_709992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 14), 'err_msg')
        # Assigning a type to the variable 'msg' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'msg', err_msg_709992)
        
        # Type idiom detected: calculating its left and rigth part (line 56)
        # Getting the type of 'msg' (line 56)
        msg_709993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 11), 'msg')
        # Getting the type of 'None' (line 56)
        None_709994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 18), 'None')
        
        (may_be_709995, more_types_in_union_709996) = may_be_none(msg_709993, None_709994)

        if may_be_709995:

            if more_types_in_union_709996:
                # Runtime conditional SSA (line 56)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Name (line 57):
            str_709997 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 18), 'str', '%s is not re-entrant')
            # Getting the type of 'func' (line 57)
            func_709998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 43), 'func')
            # Obtaining the member '__name__' of a type (line 57)
            name___709999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 43), func_709998, '__name__')
            # Applying the binary operator '%' (line 57)
            result_mod_710000 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 18), '%', str_709997, name___709999)
            
            # Assigning a type to the variable 'msg' (line 57)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'msg', result_mod_710000)

            if more_types_in_union_709996:
                # SSA join for if statement (line 56)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 58):
        
        # Call to ReentrancyLock(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'msg' (line 58)
        msg_710002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 30), 'msg', False)
        # Processing the call keyword arguments (line 58)
        kwargs_710003 = {}
        # Getting the type of 'ReentrancyLock' (line 58)
        ReentrancyLock_710001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 15), 'ReentrancyLock', False)
        # Calling ReentrancyLock(args, kwargs) (line 58)
        ReentrancyLock_call_result_710004 = invoke(stypy.reporting.localization.Localization(__file__, 58, 15), ReentrancyLock_710001, *[msg_710002], **kwargs_710003)
        
        # Assigning a type to the variable 'lock' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'lock', ReentrancyLock_call_result_710004)
        
        # Call to decorate(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'func' (line 59)
        func_710007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 29), 'func', False)
        # Processing the call keyword arguments (line 59)
        kwargs_710008 = {}
        # Getting the type of 'lock' (line 59)
        lock_710005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 15), 'lock', False)
        # Obtaining the member 'decorate' of a type (line 59)
        decorate_710006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 15), lock_710005, 'decorate')
        # Calling decorate(args, kwargs) (line 59)
        decorate_call_result_710009 = invoke(stypy.reporting.localization.Localization(__file__, 59, 15), decorate_710006, *[func_710007], **kwargs_710008)
        
        # Assigning a type to the variable 'stypy_return_type' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'stypy_return_type', decorate_call_result_710009)
        
        # ################# End of 'decorator(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'decorator' in the type store
        # Getting the type of 'stypy_return_type' (line 54)
        stypy_return_type_710010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_710010)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'decorator'
        return stypy_return_type_710010

    # Assigning a type to the variable 'decorator' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'decorator', decorator)
    # Getting the type of 'decorator' (line 60)
    decorator_710011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 11), 'decorator')
    # Assigning a type to the variable 'stypy_return_type' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'stypy_return_type', decorator_710011)
    
    # ################# End of 'non_reentrant(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'non_reentrant' in the type store
    # Getting the type of 'stypy_return_type' (line 50)
    stypy_return_type_710012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_710012)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'non_reentrant'
    return stypy_return_type_710012

# Assigning a type to the variable 'non_reentrant' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'non_reentrant', non_reentrant)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
