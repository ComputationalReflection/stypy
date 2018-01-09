
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Module for testing automatic garbage collection of objects
3: 
4: .. autosummary::
5:    :toctree: generated/
6: 
7:    set_gc_state - enable or disable garbage collection
8:    gc_state - context manager for given state of garbage collector
9:    assert_deallocated - context manager to check for circular references on object
10: 
11: '''
12: import weakref
13: import gc
14: 
15: from contextlib import contextmanager
16: 
17: __all__ = ['set_gc_state', 'gc_state', 'assert_deallocated']
18: 
19: 
20: class ReferenceError(AssertionError):
21:     pass
22: 
23: 
24: def set_gc_state(state):
25:     ''' Set status of garbage collector '''
26:     if gc.isenabled() == state:
27:         return
28:     if state:
29:         gc.enable()
30:     else:
31:         gc.disable()
32: 
33: 
34: @contextmanager
35: def gc_state(state):
36:     ''' Context manager to set state of garbage collector to `state`
37: 
38:     Parameters
39:     ----------
40:     state : bool
41:         True for gc enabled, False for disabled
42: 
43:     Examples
44:     --------
45:     >>> with gc_state(False):
46:     ...     assert not gc.isenabled()
47:     >>> with gc_state(True):
48:     ...     assert gc.isenabled()
49:     '''
50:     orig_state = gc.isenabled()
51:     set_gc_state(state)
52:     yield
53:     set_gc_state(orig_state)
54: 
55: 
56: @contextmanager
57: def assert_deallocated(func, *args, **kwargs):
58:     '''Context manager to check that object is deallocated
59: 
60:     This is useful for checking that an object can be freed directly by
61:     reference counting, without requiring gc to break reference cycles.
62:     GC is disabled inside the context manager.
63: 
64:     Parameters
65:     ----------
66:     func : callable
67:         Callable to create object to check
68:     \\*args : sequence
69:         positional arguments to `func` in order to create object to check
70:     \\*\\*kwargs : dict
71:         keyword arguments to `func` in order to create object to check
72: 
73:     Examples
74:     --------
75:     >>> class C(object): pass
76:     >>> with assert_deallocated(C) as c:
77:     ...     # do something
78:     ...     del c
79: 
80:     >>> class C(object):
81:     ...     def __init__(self):
82:     ...         self._circular = self # Make circular reference
83:     >>> with assert_deallocated(C) as c: #doctest: +IGNORE_EXCEPTION_DETAIL
84:     ...     # do something
85:     ...     del c
86:     Traceback (most recent call last):
87:         ...
88:     ReferenceError: Remaining reference(s) to object
89:     '''
90:     with gc_state(False):
91:         obj = func(*args, **kwargs)
92:         ref = weakref.ref(obj)
93:         yield obj
94:         del obj
95:         if ref() is not None:
96:             raise ReferenceError("Remaining reference(s) to object")
97: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_708563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, (-1)), 'str', '\nModule for testing automatic garbage collection of objects\n\n.. autosummary::\n   :toctree: generated/\n\n   set_gc_state - enable or disable garbage collection\n   gc_state - context manager for given state of garbage collector\n   assert_deallocated - context manager to check for circular references on object\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import weakref' statement (line 12)
import weakref

import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'weakref', weakref, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'import gc' statement (line 13)
import gc

import_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'gc', gc, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from contextlib import contextmanager' statement (line 15)
try:
    from contextlib import contextmanager

except:
    contextmanager = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'contextlib', None, module_type_store, ['contextmanager'], [contextmanager])


# Assigning a List to a Name (line 17):
__all__ = ['set_gc_state', 'gc_state', 'assert_deallocated']
module_type_store.set_exportable_members(['set_gc_state', 'gc_state', 'assert_deallocated'])

# Obtaining an instance of the builtin type 'list' (line 17)
list_708564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 17)
# Adding element type (line 17)
str_708565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 11), 'str', 'set_gc_state')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 10), list_708564, str_708565)
# Adding element type (line 17)
str_708566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 27), 'str', 'gc_state')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 10), list_708564, str_708566)
# Adding element type (line 17)
str_708567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 39), 'str', 'assert_deallocated')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 10), list_708564, str_708567)

# Assigning a type to the variable '__all__' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), '__all__', list_708564)
# Declaration of the 'ReferenceError' class
# Getting the type of 'AssertionError' (line 20)
AssertionError_708568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 21), 'AssertionError')

class ReferenceError(AssertionError_708568, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 20, 0, False)
        # Assigning a type to the variable 'self' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ReferenceError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ReferenceError' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'ReferenceError', ReferenceError)

@norecursion
def set_gc_state(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'set_gc_state'
    module_type_store = module_type_store.open_function_context('set_gc_state', 24, 0, False)
    
    # Passed parameters checking function
    set_gc_state.stypy_localization = localization
    set_gc_state.stypy_type_of_self = None
    set_gc_state.stypy_type_store = module_type_store
    set_gc_state.stypy_function_name = 'set_gc_state'
    set_gc_state.stypy_param_names_list = ['state']
    set_gc_state.stypy_varargs_param_name = None
    set_gc_state.stypy_kwargs_param_name = None
    set_gc_state.stypy_call_defaults = defaults
    set_gc_state.stypy_call_varargs = varargs
    set_gc_state.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'set_gc_state', ['state'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'set_gc_state', localization, ['state'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'set_gc_state(...)' code ##################

    str_708569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 4), 'str', ' Set status of garbage collector ')
    
    
    
    # Call to isenabled(...): (line 26)
    # Processing the call keyword arguments (line 26)
    kwargs_708572 = {}
    # Getting the type of 'gc' (line 26)
    gc_708570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 7), 'gc', False)
    # Obtaining the member 'isenabled' of a type (line 26)
    isenabled_708571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 7), gc_708570, 'isenabled')
    # Calling isenabled(args, kwargs) (line 26)
    isenabled_call_result_708573 = invoke(stypy.reporting.localization.Localization(__file__, 26, 7), isenabled_708571, *[], **kwargs_708572)
    
    # Getting the type of 'state' (line 26)
    state_708574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 25), 'state')
    # Applying the binary operator '==' (line 26)
    result_eq_708575 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 7), '==', isenabled_call_result_708573, state_708574)
    
    # Testing the type of an if condition (line 26)
    if_condition_708576 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 26, 4), result_eq_708575)
    # Assigning a type to the variable 'if_condition_708576' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'if_condition_708576', if_condition_708576)
    # SSA begins for if statement (line 26)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Assigning a type to the variable 'stypy_return_type' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 26)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'state' (line 28)
    state_708577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 7), 'state')
    # Testing the type of an if condition (line 28)
    if_condition_708578 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 28, 4), state_708577)
    # Assigning a type to the variable 'if_condition_708578' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'if_condition_708578', if_condition_708578)
    # SSA begins for if statement (line 28)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to enable(...): (line 29)
    # Processing the call keyword arguments (line 29)
    kwargs_708581 = {}
    # Getting the type of 'gc' (line 29)
    gc_708579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'gc', False)
    # Obtaining the member 'enable' of a type (line 29)
    enable_708580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), gc_708579, 'enable')
    # Calling enable(args, kwargs) (line 29)
    enable_call_result_708582 = invoke(stypy.reporting.localization.Localization(__file__, 29, 8), enable_708580, *[], **kwargs_708581)
    
    # SSA branch for the else part of an if statement (line 28)
    module_type_store.open_ssa_branch('else')
    
    # Call to disable(...): (line 31)
    # Processing the call keyword arguments (line 31)
    kwargs_708585 = {}
    # Getting the type of 'gc' (line 31)
    gc_708583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'gc', False)
    # Obtaining the member 'disable' of a type (line 31)
    disable_708584 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), gc_708583, 'disable')
    # Calling disable(args, kwargs) (line 31)
    disable_call_result_708586 = invoke(stypy.reporting.localization.Localization(__file__, 31, 8), disable_708584, *[], **kwargs_708585)
    
    # SSA join for if statement (line 28)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'set_gc_state(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'set_gc_state' in the type store
    # Getting the type of 'stypy_return_type' (line 24)
    stypy_return_type_708587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_708587)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'set_gc_state'
    return stypy_return_type_708587

# Assigning a type to the variable 'set_gc_state' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'set_gc_state', set_gc_state)

@norecursion
def gc_state(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'gc_state'
    module_type_store = module_type_store.open_function_context('gc_state', 34, 0, False)
    
    # Passed parameters checking function
    gc_state.stypy_localization = localization
    gc_state.stypy_type_of_self = None
    gc_state.stypy_type_store = module_type_store
    gc_state.stypy_function_name = 'gc_state'
    gc_state.stypy_param_names_list = ['state']
    gc_state.stypy_varargs_param_name = None
    gc_state.stypy_kwargs_param_name = None
    gc_state.stypy_call_defaults = defaults
    gc_state.stypy_call_varargs = varargs
    gc_state.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'gc_state', ['state'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'gc_state', localization, ['state'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'gc_state(...)' code ##################

    str_708588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, (-1)), 'str', ' Context manager to set state of garbage collector to `state`\n\n    Parameters\n    ----------\n    state : bool\n        True for gc enabled, False for disabled\n\n    Examples\n    --------\n    >>> with gc_state(False):\n    ...     assert not gc.isenabled()\n    >>> with gc_state(True):\n    ...     assert gc.isenabled()\n    ')
    
    # Assigning a Call to a Name (line 50):
    
    # Call to isenabled(...): (line 50)
    # Processing the call keyword arguments (line 50)
    kwargs_708591 = {}
    # Getting the type of 'gc' (line 50)
    gc_708589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 17), 'gc', False)
    # Obtaining the member 'isenabled' of a type (line 50)
    isenabled_708590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 17), gc_708589, 'isenabled')
    # Calling isenabled(args, kwargs) (line 50)
    isenabled_call_result_708592 = invoke(stypy.reporting.localization.Localization(__file__, 50, 17), isenabled_708590, *[], **kwargs_708591)
    
    # Assigning a type to the variable 'orig_state' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'orig_state', isenabled_call_result_708592)
    
    # Call to set_gc_state(...): (line 51)
    # Processing the call arguments (line 51)
    # Getting the type of 'state' (line 51)
    state_708594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 17), 'state', False)
    # Processing the call keyword arguments (line 51)
    kwargs_708595 = {}
    # Getting the type of 'set_gc_state' (line 51)
    set_gc_state_708593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'set_gc_state', False)
    # Calling set_gc_state(args, kwargs) (line 51)
    set_gc_state_call_result_708596 = invoke(stypy.reporting.localization.Localization(__file__, 51, 4), set_gc_state_708593, *[state_708594], **kwargs_708595)
    
    # Creating a generator
    GeneratorType_708597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 4), 'GeneratorType')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 4), GeneratorType_708597, None)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'stypy_return_type', GeneratorType_708597)
    
    # Call to set_gc_state(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'orig_state' (line 53)
    orig_state_708599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 17), 'orig_state', False)
    # Processing the call keyword arguments (line 53)
    kwargs_708600 = {}
    # Getting the type of 'set_gc_state' (line 53)
    set_gc_state_708598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'set_gc_state', False)
    # Calling set_gc_state(args, kwargs) (line 53)
    set_gc_state_call_result_708601 = invoke(stypy.reporting.localization.Localization(__file__, 53, 4), set_gc_state_708598, *[orig_state_708599], **kwargs_708600)
    
    
    # ################# End of 'gc_state(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'gc_state' in the type store
    # Getting the type of 'stypy_return_type' (line 34)
    stypy_return_type_708602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_708602)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'gc_state'
    return stypy_return_type_708602

# Assigning a type to the variable 'gc_state' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'gc_state', gc_state)

@norecursion
def assert_deallocated(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'assert_deallocated'
    module_type_store = module_type_store.open_function_context('assert_deallocated', 56, 0, False)
    
    # Passed parameters checking function
    assert_deallocated.stypy_localization = localization
    assert_deallocated.stypy_type_of_self = None
    assert_deallocated.stypy_type_store = module_type_store
    assert_deallocated.stypy_function_name = 'assert_deallocated'
    assert_deallocated.stypy_param_names_list = ['func']
    assert_deallocated.stypy_varargs_param_name = 'args'
    assert_deallocated.stypy_kwargs_param_name = 'kwargs'
    assert_deallocated.stypy_call_defaults = defaults
    assert_deallocated.stypy_call_varargs = varargs
    assert_deallocated.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'assert_deallocated', ['func'], 'args', 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'assert_deallocated', localization, ['func'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'assert_deallocated(...)' code ##################

    str_708603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, (-1)), 'str', 'Context manager to check that object is deallocated\n\n    This is useful for checking that an object can be freed directly by\n    reference counting, without requiring gc to break reference cycles.\n    GC is disabled inside the context manager.\n\n    Parameters\n    ----------\n    func : callable\n        Callable to create object to check\n    \\*args : sequence\n        positional arguments to `func` in order to create object to check\n    \\*\\*kwargs : dict\n        keyword arguments to `func` in order to create object to check\n\n    Examples\n    --------\n    >>> class C(object): pass\n    >>> with assert_deallocated(C) as c:\n    ...     # do something\n    ...     del c\n\n    >>> class C(object):\n    ...     def __init__(self):\n    ...         self._circular = self # Make circular reference\n    >>> with assert_deallocated(C) as c: #doctest: +IGNORE_EXCEPTION_DETAIL\n    ...     # do something\n    ...     del c\n    Traceback (most recent call last):\n        ...\n    ReferenceError: Remaining reference(s) to object\n    ')
    
    # Call to gc_state(...): (line 90)
    # Processing the call arguments (line 90)
    # Getting the type of 'False' (line 90)
    False_708605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 18), 'False', False)
    # Processing the call keyword arguments (line 90)
    kwargs_708606 = {}
    # Getting the type of 'gc_state' (line 90)
    gc_state_708604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 9), 'gc_state', False)
    # Calling gc_state(args, kwargs) (line 90)
    gc_state_call_result_708607 = invoke(stypy.reporting.localization.Localization(__file__, 90, 9), gc_state_708604, *[False_708605], **kwargs_708606)
    
    with_708608 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 90, 9), gc_state_call_result_708607, 'with parameter', '__enter__', '__exit__')

    if with_708608:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 90)
        enter___708609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 9), gc_state_call_result_708607, '__enter__')
        with_enter_708610 = invoke(stypy.reporting.localization.Localization(__file__, 90, 9), enter___708609)
        
        # Assigning a Call to a Name (line 91):
        
        # Call to func(...): (line 91)
        # Getting the type of 'args' (line 91)
        args_708612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'args', False)
        # Processing the call keyword arguments (line 91)
        # Getting the type of 'kwargs' (line 91)
        kwargs_708613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 28), 'kwargs', False)
        kwargs_708614 = {'kwargs_708613': kwargs_708613}
        # Getting the type of 'func' (line 91)
        func_708611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 14), 'func', False)
        # Calling func(args, kwargs) (line 91)
        func_call_result_708615 = invoke(stypy.reporting.localization.Localization(__file__, 91, 14), func_708611, *[args_708612], **kwargs_708614)
        
        # Assigning a type to the variable 'obj' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'obj', func_call_result_708615)
        
        # Assigning a Call to a Name (line 92):
        
        # Call to ref(...): (line 92)
        # Processing the call arguments (line 92)
        # Getting the type of 'obj' (line 92)
        obj_708618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 26), 'obj', False)
        # Processing the call keyword arguments (line 92)
        kwargs_708619 = {}
        # Getting the type of 'weakref' (line 92)
        weakref_708616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 14), 'weakref', False)
        # Obtaining the member 'ref' of a type (line 92)
        ref_708617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 14), weakref_708616, 'ref')
        # Calling ref(args, kwargs) (line 92)
        ref_call_result_708620 = invoke(stypy.reporting.localization.Localization(__file__, 92, 14), ref_708617, *[obj_708618], **kwargs_708619)
        
        # Assigning a type to the variable 'ref' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'ref', ref_call_result_708620)
        # Creating a generator
        # Getting the type of 'obj' (line 93)
        obj_708621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 14), 'obj')
        GeneratorType_708622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 8), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 8), GeneratorType_708622, obj_708621)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'stypy_return_type', GeneratorType_708622)
        # Deleting a member
        module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 94, 8), module_type_store, 'obj')
        
        
        
        # Call to ref(...): (line 95)
        # Processing the call keyword arguments (line 95)
        kwargs_708624 = {}
        # Getting the type of 'ref' (line 95)
        ref_708623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 11), 'ref', False)
        # Calling ref(args, kwargs) (line 95)
        ref_call_result_708625 = invoke(stypy.reporting.localization.Localization(__file__, 95, 11), ref_708623, *[], **kwargs_708624)
        
        # Getting the type of 'None' (line 95)
        None_708626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 24), 'None')
        # Applying the binary operator 'isnot' (line 95)
        result_is_not_708627 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 11), 'isnot', ref_call_result_708625, None_708626)
        
        # Testing the type of an if condition (line 95)
        if_condition_708628 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 95, 8), result_is_not_708627)
        # Assigning a type to the variable 'if_condition_708628' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'if_condition_708628', if_condition_708628)
        # SSA begins for if statement (line 95)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ReferenceError(...): (line 96)
        # Processing the call arguments (line 96)
        str_708630 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 33), 'str', 'Remaining reference(s) to object')
        # Processing the call keyword arguments (line 96)
        kwargs_708631 = {}
        # Getting the type of 'ReferenceError' (line 96)
        ReferenceError_708629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 18), 'ReferenceError', False)
        # Calling ReferenceError(args, kwargs) (line 96)
        ReferenceError_call_result_708632 = invoke(stypy.reporting.localization.Localization(__file__, 96, 18), ReferenceError_708629, *[str_708630], **kwargs_708631)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 96, 12), ReferenceError_call_result_708632, 'raise parameter', BaseException)
        # SSA join for if statement (line 95)
        module_type_store = module_type_store.join_ssa_context()
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 90)
        exit___708633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 9), gc_state_call_result_708607, '__exit__')
        with_exit_708634 = invoke(stypy.reporting.localization.Localization(__file__, 90, 9), exit___708633, None, None, None)

    
    # ################# End of 'assert_deallocated(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'assert_deallocated' in the type store
    # Getting the type of 'stypy_return_type' (line 56)
    stypy_return_type_708635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_708635)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'assert_deallocated'
    return stypy_return_type_708635

# Assigning a type to the variable 'assert_deallocated' (line 56)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'assert_deallocated', assert_deallocated)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
