
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import signal
2: import weakref
3: 
4: from functools import wraps
5: 
6: __unittest = True
7: 
8: 
9: class _InterruptHandler(object):
10:     def __init__(self, default_handler):
11:         self.called = False
12:         self.original_handler = default_handler
13:         if isinstance(default_handler, int):
14:             if default_handler == signal.SIG_DFL:
15:                 # Pretend it's signal.default_int_handler instead.
16:                 default_handler = signal.default_int_handler
17:             elif default_handler == signal.SIG_IGN:
18:                 # Not quite the same thing as SIG_IGN, but the closest we
19:                 # can make it: do nothing.
20:                 def default_handler(unused_signum, unused_frame):
21:                     pass
22:             else:
23:                 raise TypeError("expected SIGINT signal handler to be "
24:                                 "signal.SIG_IGN, signal.SIG_DFL, or a "
25:                                 "callable object")
26:         self.default_handler = default_handler
27: 
28:     def __call__(self, signum, frame):
29:         installed_handler = signal.getsignal(signal.SIGINT)
30:         if installed_handler is not self:
31:             # if we aren't the installed handler, then delegate immediately
32:             # to the default handler
33:             self.default_handler(signum, frame)
34: 
35:         if self.called:
36:             self.default_handler(signum, frame)
37:         self.called = True
38:         for result in _results.keys():
39:             result.stop()
40: 
41: _results = weakref.WeakKeyDictionary()
42: def registerResult(result):
43:     _results[result] = 1
44: 
45: def removeResult(result):
46:     return bool(_results.pop(result, None))
47: 
48: _interrupt_handler = None
49: def installHandler():
50:     global _interrupt_handler
51:     if _interrupt_handler is None:
52:         default_handler = signal.getsignal(signal.SIGINT)
53:         _interrupt_handler = _InterruptHandler(default_handler)
54:         signal.signal(signal.SIGINT, _interrupt_handler)
55: 
56: 
57: def removeHandler(method=None):
58:     if method is not None:
59:         @wraps(method)
60:         def inner(*args, **kwargs):
61:             initial = signal.getsignal(signal.SIGINT)
62:             removeHandler()
63:             try:
64:                 return method(*args, **kwargs)
65:             finally:
66:                 signal.signal(signal.SIGINT, initial)
67:         return inner
68: 
69:     global _interrupt_handler
70:     if _interrupt_handler is not None:
71:         signal.signal(signal.SIGINT, _interrupt_handler.original_handler)
72: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import signal' statement (line 1)
import signal

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'signal', signal, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import weakref' statement (line 2)
import weakref

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'weakref', weakref, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from functools import wraps' statement (line 4)
from functools import wraps

import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'functools', None, module_type_store, ['wraps'], [wraps])


# Assigning a Name to a Name (line 6):
# Getting the type of 'True' (line 6)
True_191846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 13), 'True')
# Assigning a type to the variable '__unittest' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), '__unittest', True_191846)
# Declaration of the '_InterruptHandler' class

class _InterruptHandler(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 10, 4, False)
        # Assigning a type to the variable 'self' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_InterruptHandler.__init__', ['default_handler'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['default_handler'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 11):
        # Getting the type of 'False' (line 11)
        False_191847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 22), 'False')
        # Getting the type of 'self' (line 11)
        self_191848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'self')
        # Setting the type of the member 'called' of a type (line 11)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 8), self_191848, 'called', False_191847)
        
        # Assigning a Name to a Attribute (line 12):
        # Getting the type of 'default_handler' (line 12)
        default_handler_191849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 32), 'default_handler')
        # Getting the type of 'self' (line 12)
        self_191850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'self')
        # Setting the type of the member 'original_handler' of a type (line 12)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 8), self_191850, 'original_handler', default_handler_191849)
        
        # Type idiom detected: calculating its left and rigth part (line 13)
        # Getting the type of 'int' (line 13)
        int_191851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 39), 'int')
        # Getting the type of 'default_handler' (line 13)
        default_handler_191852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 22), 'default_handler')
        
        (may_be_191853, more_types_in_union_191854) = may_be_subtype(int_191851, default_handler_191852)

        if may_be_191853:

            if more_types_in_union_191854:
                # Runtime conditional SSA (line 13)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'default_handler' (line 13)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'default_handler', remove_not_subtype_from_union(default_handler_191852, int))
            
            
            # Getting the type of 'default_handler' (line 14)
            default_handler_191855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 15), 'default_handler')
            # Getting the type of 'signal' (line 14)
            signal_191856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 34), 'signal')
            # Obtaining the member 'SIG_DFL' of a type (line 14)
            SIG_DFL_191857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 34), signal_191856, 'SIG_DFL')
            # Applying the binary operator '==' (line 14)
            result_eq_191858 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 15), '==', default_handler_191855, SIG_DFL_191857)
            
            # Testing the type of an if condition (line 14)
            if_condition_191859 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 12), result_eq_191858)
            # Assigning a type to the variable 'if_condition_191859' (line 14)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 12), 'if_condition_191859', if_condition_191859)
            # SSA begins for if statement (line 14)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Attribute to a Name (line 16):
            # Getting the type of 'signal' (line 16)
            signal_191860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 34), 'signal')
            # Obtaining the member 'default_int_handler' of a type (line 16)
            default_int_handler_191861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 34), signal_191860, 'default_int_handler')
            # Assigning a type to the variable 'default_handler' (line 16)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 16), 'default_handler', default_int_handler_191861)
            # SSA branch for the else part of an if statement (line 14)
            module_type_store.open_ssa_branch('else')
            
            
            # Getting the type of 'default_handler' (line 17)
            default_handler_191862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 17), 'default_handler')
            # Getting the type of 'signal' (line 17)
            signal_191863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 36), 'signal')
            # Obtaining the member 'SIG_IGN' of a type (line 17)
            SIG_IGN_191864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 36), signal_191863, 'SIG_IGN')
            # Applying the binary operator '==' (line 17)
            result_eq_191865 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 17), '==', default_handler_191862, SIG_IGN_191864)
            
            # Testing the type of an if condition (line 17)
            if_condition_191866 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 17, 17), result_eq_191865)
            # Assigning a type to the variable 'if_condition_191866' (line 17)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 17), 'if_condition_191866', if_condition_191866)
            # SSA begins for if statement (line 17)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

            @norecursion
            def default_handler(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'default_handler'
                module_type_store = module_type_store.open_function_context('default_handler', 20, 16, False)
                
                # Passed parameters checking function
                default_handler.stypy_localization = localization
                default_handler.stypy_type_of_self = None
                default_handler.stypy_type_store = module_type_store
                default_handler.stypy_function_name = 'default_handler'
                default_handler.stypy_param_names_list = ['unused_signum', 'unused_frame']
                default_handler.stypy_varargs_param_name = None
                default_handler.stypy_kwargs_param_name = None
                default_handler.stypy_call_defaults = defaults
                default_handler.stypy_call_varargs = varargs
                default_handler.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, 'default_handler', ['unused_signum', 'unused_frame'], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'default_handler', localization, ['unused_signum', 'unused_frame'], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'default_handler(...)' code ##################

                pass
                
                # ################# End of 'default_handler(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'default_handler' in the type store
                # Getting the type of 'stypy_return_type' (line 20)
                stypy_return_type_191867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_191867)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'default_handler'
                return stypy_return_type_191867

            # Assigning a type to the variable 'default_handler' (line 20)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'default_handler', default_handler)
            # SSA branch for the else part of an if statement (line 17)
            module_type_store.open_ssa_branch('else')
            
            # Call to TypeError(...): (line 23)
            # Processing the call arguments (line 23)
            str_191869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 32), 'str', 'expected SIGINT signal handler to be signal.SIG_IGN, signal.SIG_DFL, or a callable object')
            # Processing the call keyword arguments (line 23)
            kwargs_191870 = {}
            # Getting the type of 'TypeError' (line 23)
            TypeError_191868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 22), 'TypeError', False)
            # Calling TypeError(args, kwargs) (line 23)
            TypeError_call_result_191871 = invoke(stypy.reporting.localization.Localization(__file__, 23, 22), TypeError_191868, *[str_191869], **kwargs_191870)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 23, 16), TypeError_call_result_191871, 'raise parameter', BaseException)
            # SSA join for if statement (line 17)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 14)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_191854:
                # SSA join for if statement (line 13)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Name to a Attribute (line 26):
        # Getting the type of 'default_handler' (line 26)
        default_handler_191872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 31), 'default_handler')
        # Getting the type of 'self' (line 26)
        self_191873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'self')
        # Setting the type of the member 'default_handler' of a type (line 26)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), self_191873, 'default_handler', default_handler_191872)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 28, 4, False)
        # Assigning a type to the variable 'self' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _InterruptHandler.__call__.__dict__.__setitem__('stypy_localization', localization)
        _InterruptHandler.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _InterruptHandler.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        _InterruptHandler.__call__.__dict__.__setitem__('stypy_function_name', '_InterruptHandler.__call__')
        _InterruptHandler.__call__.__dict__.__setitem__('stypy_param_names_list', ['signum', 'frame'])
        _InterruptHandler.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        _InterruptHandler.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _InterruptHandler.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        _InterruptHandler.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        _InterruptHandler.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _InterruptHandler.__call__.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_InterruptHandler.__call__', ['signum', 'frame'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['signum', 'frame'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        # Assigning a Call to a Name (line 29):
        
        # Call to getsignal(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'signal' (line 29)
        signal_191876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 45), 'signal', False)
        # Obtaining the member 'SIGINT' of a type (line 29)
        SIGINT_191877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 45), signal_191876, 'SIGINT')
        # Processing the call keyword arguments (line 29)
        kwargs_191878 = {}
        # Getting the type of 'signal' (line 29)
        signal_191874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 28), 'signal', False)
        # Obtaining the member 'getsignal' of a type (line 29)
        getsignal_191875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 28), signal_191874, 'getsignal')
        # Calling getsignal(args, kwargs) (line 29)
        getsignal_call_result_191879 = invoke(stypy.reporting.localization.Localization(__file__, 29, 28), getsignal_191875, *[SIGINT_191877], **kwargs_191878)
        
        # Assigning a type to the variable 'installed_handler' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'installed_handler', getsignal_call_result_191879)
        
        
        # Getting the type of 'installed_handler' (line 30)
        installed_handler_191880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 11), 'installed_handler')
        # Getting the type of 'self' (line 30)
        self_191881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 36), 'self')
        # Applying the binary operator 'isnot' (line 30)
        result_is_not_191882 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 11), 'isnot', installed_handler_191880, self_191881)
        
        # Testing the type of an if condition (line 30)
        if_condition_191883 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 30, 8), result_is_not_191882)
        # Assigning a type to the variable 'if_condition_191883' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'if_condition_191883', if_condition_191883)
        # SSA begins for if statement (line 30)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to default_handler(...): (line 33)
        # Processing the call arguments (line 33)
        # Getting the type of 'signum' (line 33)
        signum_191886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 33), 'signum', False)
        # Getting the type of 'frame' (line 33)
        frame_191887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 41), 'frame', False)
        # Processing the call keyword arguments (line 33)
        kwargs_191888 = {}
        # Getting the type of 'self' (line 33)
        self_191884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'self', False)
        # Obtaining the member 'default_handler' of a type (line 33)
        default_handler_191885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 12), self_191884, 'default_handler')
        # Calling default_handler(args, kwargs) (line 33)
        default_handler_call_result_191889 = invoke(stypy.reporting.localization.Localization(__file__, 33, 12), default_handler_191885, *[signum_191886, frame_191887], **kwargs_191888)
        
        # SSA join for if statement (line 30)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 35)
        self_191890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 11), 'self')
        # Obtaining the member 'called' of a type (line 35)
        called_191891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 11), self_191890, 'called')
        # Testing the type of an if condition (line 35)
        if_condition_191892 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 35, 8), called_191891)
        # Assigning a type to the variable 'if_condition_191892' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'if_condition_191892', if_condition_191892)
        # SSA begins for if statement (line 35)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to default_handler(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'signum' (line 36)
        signum_191895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 33), 'signum', False)
        # Getting the type of 'frame' (line 36)
        frame_191896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 41), 'frame', False)
        # Processing the call keyword arguments (line 36)
        kwargs_191897 = {}
        # Getting the type of 'self' (line 36)
        self_191893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'self', False)
        # Obtaining the member 'default_handler' of a type (line 36)
        default_handler_191894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 12), self_191893, 'default_handler')
        # Calling default_handler(args, kwargs) (line 36)
        default_handler_call_result_191898 = invoke(stypy.reporting.localization.Localization(__file__, 36, 12), default_handler_191894, *[signum_191895, frame_191896], **kwargs_191897)
        
        # SSA join for if statement (line 35)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 37):
        # Getting the type of 'True' (line 37)
        True_191899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 22), 'True')
        # Getting the type of 'self' (line 37)
        self_191900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'self')
        # Setting the type of the member 'called' of a type (line 37)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), self_191900, 'called', True_191899)
        
        
        # Call to keys(...): (line 38)
        # Processing the call keyword arguments (line 38)
        kwargs_191903 = {}
        # Getting the type of '_results' (line 38)
        _results_191901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 22), '_results', False)
        # Obtaining the member 'keys' of a type (line 38)
        keys_191902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 22), _results_191901, 'keys')
        # Calling keys(args, kwargs) (line 38)
        keys_call_result_191904 = invoke(stypy.reporting.localization.Localization(__file__, 38, 22), keys_191902, *[], **kwargs_191903)
        
        # Testing the type of a for loop iterable (line 38)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 38, 8), keys_call_result_191904)
        # Getting the type of the for loop variable (line 38)
        for_loop_var_191905 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 38, 8), keys_call_result_191904)
        # Assigning a type to the variable 'result' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'result', for_loop_var_191905)
        # SSA begins for a for statement (line 38)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to stop(...): (line 39)
        # Processing the call keyword arguments (line 39)
        kwargs_191908 = {}
        # Getting the type of 'result' (line 39)
        result_191906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'result', False)
        # Obtaining the member 'stop' of a type (line 39)
        stop_191907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 12), result_191906, 'stop')
        # Calling stop(args, kwargs) (line 39)
        stop_call_result_191909 = invoke(stypy.reporting.localization.Localization(__file__, 39, 12), stop_191907, *[], **kwargs_191908)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 28)
        stypy_return_type_191910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_191910)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_191910


# Assigning a type to the variable '_InterruptHandler' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), '_InterruptHandler', _InterruptHandler)

# Assigning a Call to a Name (line 41):

# Call to WeakKeyDictionary(...): (line 41)
# Processing the call keyword arguments (line 41)
kwargs_191913 = {}
# Getting the type of 'weakref' (line 41)
weakref_191911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 11), 'weakref', False)
# Obtaining the member 'WeakKeyDictionary' of a type (line 41)
WeakKeyDictionary_191912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 11), weakref_191911, 'WeakKeyDictionary')
# Calling WeakKeyDictionary(args, kwargs) (line 41)
WeakKeyDictionary_call_result_191914 = invoke(stypy.reporting.localization.Localization(__file__, 41, 11), WeakKeyDictionary_191912, *[], **kwargs_191913)

# Assigning a type to the variable '_results' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), '_results', WeakKeyDictionary_call_result_191914)

@norecursion
def registerResult(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'registerResult'
    module_type_store = module_type_store.open_function_context('registerResult', 42, 0, False)
    
    # Passed parameters checking function
    registerResult.stypy_localization = localization
    registerResult.stypy_type_of_self = None
    registerResult.stypy_type_store = module_type_store
    registerResult.stypy_function_name = 'registerResult'
    registerResult.stypy_param_names_list = ['result']
    registerResult.stypy_varargs_param_name = None
    registerResult.stypy_kwargs_param_name = None
    registerResult.stypy_call_defaults = defaults
    registerResult.stypy_call_varargs = varargs
    registerResult.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'registerResult', ['result'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'registerResult', localization, ['result'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'registerResult(...)' code ##################

    
    # Assigning a Num to a Subscript (line 43):
    int_191915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 23), 'int')
    # Getting the type of '_results' (line 43)
    _results_191916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), '_results')
    # Getting the type of 'result' (line 43)
    result_191917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 13), 'result')
    # Storing an element on a container (line 43)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 4), _results_191916, (result_191917, int_191915))
    
    # ################# End of 'registerResult(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'registerResult' in the type store
    # Getting the type of 'stypy_return_type' (line 42)
    stypy_return_type_191918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_191918)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'registerResult'
    return stypy_return_type_191918

# Assigning a type to the variable 'registerResult' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'registerResult', registerResult)

@norecursion
def removeResult(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'removeResult'
    module_type_store = module_type_store.open_function_context('removeResult', 45, 0, False)
    
    # Passed parameters checking function
    removeResult.stypy_localization = localization
    removeResult.stypy_type_of_self = None
    removeResult.stypy_type_store = module_type_store
    removeResult.stypy_function_name = 'removeResult'
    removeResult.stypy_param_names_list = ['result']
    removeResult.stypy_varargs_param_name = None
    removeResult.stypy_kwargs_param_name = None
    removeResult.stypy_call_defaults = defaults
    removeResult.stypy_call_varargs = varargs
    removeResult.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'removeResult', ['result'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'removeResult', localization, ['result'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'removeResult(...)' code ##################

    
    # Call to bool(...): (line 46)
    # Processing the call arguments (line 46)
    
    # Call to pop(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'result' (line 46)
    result_191922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 29), 'result', False)
    # Getting the type of 'None' (line 46)
    None_191923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 37), 'None', False)
    # Processing the call keyword arguments (line 46)
    kwargs_191924 = {}
    # Getting the type of '_results' (line 46)
    _results_191920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 16), '_results', False)
    # Obtaining the member 'pop' of a type (line 46)
    pop_191921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 16), _results_191920, 'pop')
    # Calling pop(args, kwargs) (line 46)
    pop_call_result_191925 = invoke(stypy.reporting.localization.Localization(__file__, 46, 16), pop_191921, *[result_191922, None_191923], **kwargs_191924)
    
    # Processing the call keyword arguments (line 46)
    kwargs_191926 = {}
    # Getting the type of 'bool' (line 46)
    bool_191919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 11), 'bool', False)
    # Calling bool(args, kwargs) (line 46)
    bool_call_result_191927 = invoke(stypy.reporting.localization.Localization(__file__, 46, 11), bool_191919, *[pop_call_result_191925], **kwargs_191926)
    
    # Assigning a type to the variable 'stypy_return_type' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'stypy_return_type', bool_call_result_191927)
    
    # ################# End of 'removeResult(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'removeResult' in the type store
    # Getting the type of 'stypy_return_type' (line 45)
    stypy_return_type_191928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_191928)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'removeResult'
    return stypy_return_type_191928

# Assigning a type to the variable 'removeResult' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'removeResult', removeResult)

# Assigning a Name to a Name (line 48):
# Getting the type of 'None' (line 48)
None_191929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 21), 'None')
# Assigning a type to the variable '_interrupt_handler' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), '_interrupt_handler', None_191929)

@norecursion
def installHandler(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'installHandler'
    module_type_store = module_type_store.open_function_context('installHandler', 49, 0, False)
    
    # Passed parameters checking function
    installHandler.stypy_localization = localization
    installHandler.stypy_type_of_self = None
    installHandler.stypy_type_store = module_type_store
    installHandler.stypy_function_name = 'installHandler'
    installHandler.stypy_param_names_list = []
    installHandler.stypy_varargs_param_name = None
    installHandler.stypy_kwargs_param_name = None
    installHandler.stypy_call_defaults = defaults
    installHandler.stypy_call_varargs = varargs
    installHandler.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'installHandler', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'installHandler', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'installHandler(...)' code ##################

    # Marking variables as global (line 50)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 50, 4), '_interrupt_handler')
    
    # Type idiom detected: calculating its left and rigth part (line 51)
    # Getting the type of '_interrupt_handler' (line 51)
    _interrupt_handler_191930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 7), '_interrupt_handler')
    # Getting the type of 'None' (line 51)
    None_191931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 29), 'None')
    
    (may_be_191932, more_types_in_union_191933) = may_be_none(_interrupt_handler_191930, None_191931)

    if may_be_191932:

        if more_types_in_union_191933:
            # Runtime conditional SSA (line 51)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 52):
        
        # Call to getsignal(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'signal' (line 52)
        signal_191936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 43), 'signal', False)
        # Obtaining the member 'SIGINT' of a type (line 52)
        SIGINT_191937 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 43), signal_191936, 'SIGINT')
        # Processing the call keyword arguments (line 52)
        kwargs_191938 = {}
        # Getting the type of 'signal' (line 52)
        signal_191934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 26), 'signal', False)
        # Obtaining the member 'getsignal' of a type (line 52)
        getsignal_191935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 26), signal_191934, 'getsignal')
        # Calling getsignal(args, kwargs) (line 52)
        getsignal_call_result_191939 = invoke(stypy.reporting.localization.Localization(__file__, 52, 26), getsignal_191935, *[SIGINT_191937], **kwargs_191938)
        
        # Assigning a type to the variable 'default_handler' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'default_handler', getsignal_call_result_191939)
        
        # Assigning a Call to a Name (line 53):
        
        # Call to _InterruptHandler(...): (line 53)
        # Processing the call arguments (line 53)
        # Getting the type of 'default_handler' (line 53)
        default_handler_191941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 47), 'default_handler', False)
        # Processing the call keyword arguments (line 53)
        kwargs_191942 = {}
        # Getting the type of '_InterruptHandler' (line 53)
        _InterruptHandler_191940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 29), '_InterruptHandler', False)
        # Calling _InterruptHandler(args, kwargs) (line 53)
        _InterruptHandler_call_result_191943 = invoke(stypy.reporting.localization.Localization(__file__, 53, 29), _InterruptHandler_191940, *[default_handler_191941], **kwargs_191942)
        
        # Assigning a type to the variable '_interrupt_handler' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), '_interrupt_handler', _InterruptHandler_call_result_191943)
        
        # Call to signal(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'signal' (line 54)
        signal_191946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 22), 'signal', False)
        # Obtaining the member 'SIGINT' of a type (line 54)
        SIGINT_191947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 22), signal_191946, 'SIGINT')
        # Getting the type of '_interrupt_handler' (line 54)
        _interrupt_handler_191948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 37), '_interrupt_handler', False)
        # Processing the call keyword arguments (line 54)
        kwargs_191949 = {}
        # Getting the type of 'signal' (line 54)
        signal_191944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'signal', False)
        # Obtaining the member 'signal' of a type (line 54)
        signal_191945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), signal_191944, 'signal')
        # Calling signal(args, kwargs) (line 54)
        signal_call_result_191950 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), signal_191945, *[SIGINT_191947, _interrupt_handler_191948], **kwargs_191949)
        

        if more_types_in_union_191933:
            # SSA join for if statement (line 51)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of 'installHandler(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'installHandler' in the type store
    # Getting the type of 'stypy_return_type' (line 49)
    stypy_return_type_191951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_191951)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'installHandler'
    return stypy_return_type_191951

# Assigning a type to the variable 'installHandler' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'installHandler', installHandler)

@norecursion
def removeHandler(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 57)
    None_191952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 25), 'None')
    defaults = [None_191952]
    # Create a new context for function 'removeHandler'
    module_type_store = module_type_store.open_function_context('removeHandler', 57, 0, False)
    
    # Passed parameters checking function
    removeHandler.stypy_localization = localization
    removeHandler.stypy_type_of_self = None
    removeHandler.stypy_type_store = module_type_store
    removeHandler.stypy_function_name = 'removeHandler'
    removeHandler.stypy_param_names_list = ['method']
    removeHandler.stypy_varargs_param_name = None
    removeHandler.stypy_kwargs_param_name = None
    removeHandler.stypy_call_defaults = defaults
    removeHandler.stypy_call_varargs = varargs
    removeHandler.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'removeHandler', ['method'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'removeHandler', localization, ['method'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'removeHandler(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 58)
    # Getting the type of 'method' (line 58)
    method_191953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'method')
    # Getting the type of 'None' (line 58)
    None_191954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 21), 'None')
    
    (may_be_191955, more_types_in_union_191956) = may_not_be_none(method_191953, None_191954)

    if may_be_191955:

        if more_types_in_union_191956:
            # Runtime conditional SSA (line 58)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store


        @norecursion
        def inner(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'inner'
            module_type_store = module_type_store.open_function_context('inner', 59, 8, False)
            
            # Passed parameters checking function
            inner.stypy_localization = localization
            inner.stypy_type_of_self = None
            inner.stypy_type_store = module_type_store
            inner.stypy_function_name = 'inner'
            inner.stypy_param_names_list = []
            inner.stypy_varargs_param_name = 'args'
            inner.stypy_kwargs_param_name = 'kwargs'
            inner.stypy_call_defaults = defaults
            inner.stypy_call_varargs = varargs
            inner.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'inner', [], 'args', 'kwargs', defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'inner', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'inner(...)' code ##################

            
            # Assigning a Call to a Name (line 61):
            
            # Call to getsignal(...): (line 61)
            # Processing the call arguments (line 61)
            # Getting the type of 'signal' (line 61)
            signal_191959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 39), 'signal', False)
            # Obtaining the member 'SIGINT' of a type (line 61)
            SIGINT_191960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 39), signal_191959, 'SIGINT')
            # Processing the call keyword arguments (line 61)
            kwargs_191961 = {}
            # Getting the type of 'signal' (line 61)
            signal_191957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 22), 'signal', False)
            # Obtaining the member 'getsignal' of a type (line 61)
            getsignal_191958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 22), signal_191957, 'getsignal')
            # Calling getsignal(args, kwargs) (line 61)
            getsignal_call_result_191962 = invoke(stypy.reporting.localization.Localization(__file__, 61, 22), getsignal_191958, *[SIGINT_191960], **kwargs_191961)
            
            # Assigning a type to the variable 'initial' (line 61)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'initial', getsignal_call_result_191962)
            
            # Call to removeHandler(...): (line 62)
            # Processing the call keyword arguments (line 62)
            kwargs_191964 = {}
            # Getting the type of 'removeHandler' (line 62)
            removeHandler_191963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'removeHandler', False)
            # Calling removeHandler(args, kwargs) (line 62)
            removeHandler_call_result_191965 = invoke(stypy.reporting.localization.Localization(__file__, 62, 12), removeHandler_191963, *[], **kwargs_191964)
            
            
            # Try-finally block (line 63)
            
            # Call to method(...): (line 64)
            # Getting the type of 'args' (line 64)
            args_191967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 31), 'args', False)
            # Processing the call keyword arguments (line 64)
            # Getting the type of 'kwargs' (line 64)
            kwargs_191968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 39), 'kwargs', False)
            kwargs_191969 = {'kwargs_191968': kwargs_191968}
            # Getting the type of 'method' (line 64)
            method_191966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 23), 'method', False)
            # Calling method(args, kwargs) (line 64)
            method_call_result_191970 = invoke(stypy.reporting.localization.Localization(__file__, 64, 23), method_191966, *[args_191967], **kwargs_191969)
            
            # Assigning a type to the variable 'stypy_return_type' (line 64)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 16), 'stypy_return_type', method_call_result_191970)
            
            # finally branch of the try-finally block (line 63)
            
            # Call to signal(...): (line 66)
            # Processing the call arguments (line 66)
            # Getting the type of 'signal' (line 66)
            signal_191973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 30), 'signal', False)
            # Obtaining the member 'SIGINT' of a type (line 66)
            SIGINT_191974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 30), signal_191973, 'SIGINT')
            # Getting the type of 'initial' (line 66)
            initial_191975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 45), 'initial', False)
            # Processing the call keyword arguments (line 66)
            kwargs_191976 = {}
            # Getting the type of 'signal' (line 66)
            signal_191971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 16), 'signal', False)
            # Obtaining the member 'signal' of a type (line 66)
            signal_191972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 16), signal_191971, 'signal')
            # Calling signal(args, kwargs) (line 66)
            signal_call_result_191977 = invoke(stypy.reporting.localization.Localization(__file__, 66, 16), signal_191972, *[SIGINT_191974, initial_191975], **kwargs_191976)
            
            
            
            # ################# End of 'inner(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'inner' in the type store
            # Getting the type of 'stypy_return_type' (line 59)
            stypy_return_type_191978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_191978)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'inner'
            return stypy_return_type_191978

        # Assigning a type to the variable 'inner' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'inner', inner)
        # Getting the type of 'inner' (line 67)
        inner_191979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 15), 'inner')
        # Assigning a type to the variable 'stypy_return_type' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'stypy_return_type', inner_191979)

        if more_types_in_union_191956:
            # SSA join for if statement (line 58)
            module_type_store = module_type_store.join_ssa_context()


    
    # Marking variables as global (line 69)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 69, 4), '_interrupt_handler')
    
    # Type idiom detected: calculating its left and rigth part (line 70)
    # Getting the type of '_interrupt_handler' (line 70)
    _interrupt_handler_191980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), '_interrupt_handler')
    # Getting the type of 'None' (line 70)
    None_191981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 33), 'None')
    
    (may_be_191982, more_types_in_union_191983) = may_not_be_none(_interrupt_handler_191980, None_191981)

    if may_be_191982:

        if more_types_in_union_191983:
            # Runtime conditional SSA (line 70)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to signal(...): (line 71)
        # Processing the call arguments (line 71)
        # Getting the type of 'signal' (line 71)
        signal_191986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 22), 'signal', False)
        # Obtaining the member 'SIGINT' of a type (line 71)
        SIGINT_191987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 22), signal_191986, 'SIGINT')
        # Getting the type of '_interrupt_handler' (line 71)
        _interrupt_handler_191988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 37), '_interrupt_handler', False)
        # Obtaining the member 'original_handler' of a type (line 71)
        original_handler_191989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 37), _interrupt_handler_191988, 'original_handler')
        # Processing the call keyword arguments (line 71)
        kwargs_191990 = {}
        # Getting the type of 'signal' (line 71)
        signal_191984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'signal', False)
        # Obtaining the member 'signal' of a type (line 71)
        signal_191985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 8), signal_191984, 'signal')
        # Calling signal(args, kwargs) (line 71)
        signal_call_result_191991 = invoke(stypy.reporting.localization.Localization(__file__, 71, 8), signal_191985, *[SIGINT_191987, original_handler_191989], **kwargs_191990)
        

        if more_types_in_union_191983:
            # SSA join for if statement (line 70)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # ################# End of 'removeHandler(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'removeHandler' in the type store
    # Getting the type of 'stypy_return_type' (line 57)
    stypy_return_type_191992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_191992)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'removeHandler'
    return stypy_return_type_191992

# Assigning a type to the variable 'removeHandler' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'removeHandler', removeHandler)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
