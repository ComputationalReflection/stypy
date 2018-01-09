
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import (absolute_import, division, print_function,
2:                         unicode_literals)
3: 
4: import os
5: import six
6: import sys
7: from .. import _copy_metadata
8: from . import knownfail
9: from .exceptions import KnownFailureDidNotFailTest
10: 
11: 
12: def knownfailureif(fail_condition, msg=None, known_exception_class=None):
13:     # based on numpy.testing.dec.knownfailureif
14:     if msg is None:
15:         msg = 'Test known to fail'
16: 
17:     def known_fail_decorator(f):
18:         def failer(*args, **kwargs):
19:             try:
20:                 # Always run the test (to generate images).
21:                 result = f(*args, **kwargs)
22:             except Exception as err:
23:                 if fail_condition:
24:                     if known_exception_class is not None:
25:                         if not isinstance(err, known_exception_class):
26:                             # This is not the expected exception
27:                             raise
28:                     knownfail(msg)
29:                 else:
30:                     raise
31:             if fail_condition and fail_condition != 'indeterminate':
32:                 raise KnownFailureDidNotFailTest(msg)
33:             return result
34:         return _copy_metadata(f, failer)
35:     return known_fail_decorator
36: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import os' statement (line 4)
import os

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import six' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/_nose/')
import_294068 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'six')

if (type(import_294068) is not StypyTypeError):

    if (import_294068 != 'pyd_module'):
        __import__(import_294068)
        sys_modules_294069 = sys.modules[import_294068]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'six', sys_modules_294069.module_type_store, module_type_store)
    else:
        import six

        import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'six', six, module_type_store)

else:
    # Assigning a type to the variable 'six' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'six', import_294068)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/_nose/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import sys' statement (line 6)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from matplotlib.testing import _copy_metadata' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/_nose/')
import_294070 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib.testing')

if (type(import_294070) is not StypyTypeError):

    if (import_294070 != 'pyd_module'):
        __import__(import_294070)
        sys_modules_294071 = sys.modules[import_294070]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib.testing', sys_modules_294071.module_type_store, module_type_store, ['_copy_metadata'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_294071, sys_modules_294071.module_type_store, module_type_store)
    else:
        from matplotlib.testing import _copy_metadata

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib.testing', None, module_type_store, ['_copy_metadata'], [_copy_metadata])

else:
    # Assigning a type to the variable 'matplotlib.testing' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'matplotlib.testing', import_294070)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/_nose/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from matplotlib.testing._nose import knownfail' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/_nose/')
import_294072 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'matplotlib.testing._nose')

if (type(import_294072) is not StypyTypeError):

    if (import_294072 != 'pyd_module'):
        __import__(import_294072)
        sys_modules_294073 = sys.modules[import_294072]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'matplotlib.testing._nose', sys_modules_294073.module_type_store, module_type_store, ['knownfail'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_294073, sys_modules_294073.module_type_store, module_type_store)
    else:
        from matplotlib.testing._nose import knownfail

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'matplotlib.testing._nose', None, module_type_store, ['knownfail'], [knownfail])

else:
    # Assigning a type to the variable 'matplotlib.testing._nose' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'matplotlib.testing._nose', import_294072)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/_nose/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from matplotlib.testing._nose.exceptions import KnownFailureDidNotFailTest' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/testing/_nose/')
import_294074 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.testing._nose.exceptions')

if (type(import_294074) is not StypyTypeError):

    if (import_294074 != 'pyd_module'):
        __import__(import_294074)
        sys_modules_294075 = sys.modules[import_294074]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.testing._nose.exceptions', sys_modules_294075.module_type_store, module_type_store, ['KnownFailureDidNotFailTest'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_294075, sys_modules_294075.module_type_store, module_type_store)
    else:
        from matplotlib.testing._nose.exceptions import KnownFailureDidNotFailTest

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.testing._nose.exceptions', None, module_type_store, ['KnownFailureDidNotFailTest'], [KnownFailureDidNotFailTest])

else:
    # Assigning a type to the variable 'matplotlib.testing._nose.exceptions' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'matplotlib.testing._nose.exceptions', import_294074)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/testing/_nose/')


@norecursion
def knownfailureif(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 12)
    None_294076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 39), 'None')
    # Getting the type of 'None' (line 12)
    None_294077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 67), 'None')
    defaults = [None_294076, None_294077]
    # Create a new context for function 'knownfailureif'
    module_type_store = module_type_store.open_function_context('knownfailureif', 12, 0, False)
    
    # Passed parameters checking function
    knownfailureif.stypy_localization = localization
    knownfailureif.stypy_type_of_self = None
    knownfailureif.stypy_type_store = module_type_store
    knownfailureif.stypy_function_name = 'knownfailureif'
    knownfailureif.stypy_param_names_list = ['fail_condition', 'msg', 'known_exception_class']
    knownfailureif.stypy_varargs_param_name = None
    knownfailureif.stypy_kwargs_param_name = None
    knownfailureif.stypy_call_defaults = defaults
    knownfailureif.stypy_call_varargs = varargs
    knownfailureif.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'knownfailureif', ['fail_condition', 'msg', 'known_exception_class'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'knownfailureif', localization, ['fail_condition', 'msg', 'known_exception_class'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'knownfailureif(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 14)
    # Getting the type of 'msg' (line 14)
    msg_294078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 7), 'msg')
    # Getting the type of 'None' (line 14)
    None_294079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 14), 'None')
    
    (may_be_294080, more_types_in_union_294081) = may_be_none(msg_294078, None_294079)

    if may_be_294080:

        if more_types_in_union_294081:
            # Runtime conditional SSA (line 14)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Str to a Name (line 15):
        unicode_294082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 14), 'unicode', u'Test known to fail')
        # Assigning a type to the variable 'msg' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'msg', unicode_294082)

        if more_types_in_union_294081:
            # SSA join for if statement (line 14)
            module_type_store = module_type_store.join_ssa_context()


    

    @norecursion
    def known_fail_decorator(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'known_fail_decorator'
        module_type_store = module_type_store.open_function_context('known_fail_decorator', 17, 4, False)
        
        # Passed parameters checking function
        known_fail_decorator.stypy_localization = localization
        known_fail_decorator.stypy_type_of_self = None
        known_fail_decorator.stypy_type_store = module_type_store
        known_fail_decorator.stypy_function_name = 'known_fail_decorator'
        known_fail_decorator.stypy_param_names_list = ['f']
        known_fail_decorator.stypy_varargs_param_name = None
        known_fail_decorator.stypy_kwargs_param_name = None
        known_fail_decorator.stypy_call_defaults = defaults
        known_fail_decorator.stypy_call_varargs = varargs
        known_fail_decorator.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'known_fail_decorator', ['f'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'known_fail_decorator', localization, ['f'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'known_fail_decorator(...)' code ##################


        @norecursion
        def failer(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'failer'
            module_type_store = module_type_store.open_function_context('failer', 18, 8, False)
            
            # Passed parameters checking function
            failer.stypy_localization = localization
            failer.stypy_type_of_self = None
            failer.stypy_type_store = module_type_store
            failer.stypy_function_name = 'failer'
            failer.stypy_param_names_list = []
            failer.stypy_varargs_param_name = 'args'
            failer.stypy_kwargs_param_name = 'kwargs'
            failer.stypy_call_defaults = defaults
            failer.stypy_call_varargs = varargs
            failer.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'failer', [], 'args', 'kwargs', defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'failer', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'failer(...)' code ##################

            
            
            # SSA begins for try-except statement (line 19)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Call to a Name (line 21):
            
            # Call to f(...): (line 21)
            # Getting the type of 'args' (line 21)
            args_294084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 28), 'args', False)
            # Processing the call keyword arguments (line 21)
            # Getting the type of 'kwargs' (line 21)
            kwargs_294085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 36), 'kwargs', False)
            kwargs_294086 = {'kwargs_294085': kwargs_294085}
            # Getting the type of 'f' (line 21)
            f_294083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 25), 'f', False)
            # Calling f(args, kwargs) (line 21)
            f_call_result_294087 = invoke(stypy.reporting.localization.Localization(__file__, 21, 25), f_294083, *[args_294084], **kwargs_294086)
            
            # Assigning a type to the variable 'result' (line 21)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'result', f_call_result_294087)
            # SSA branch for the except part of a try statement (line 19)
            # SSA branch for the except 'Exception' branch of a try statement (line 19)
            # Storing handler type
            module_type_store.open_ssa_branch('except')
            # Getting the type of 'Exception' (line 22)
            Exception_294088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 19), 'Exception')
            # Assigning a type to the variable 'err' (line 22)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'err', Exception_294088)
            
            # Getting the type of 'fail_condition' (line 23)
            fail_condition_294089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 19), 'fail_condition')
            # Testing the type of an if condition (line 23)
            if_condition_294090 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 23, 16), fail_condition_294089)
            # Assigning a type to the variable 'if_condition_294090' (line 23)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 16), 'if_condition_294090', if_condition_294090)
            # SSA begins for if statement (line 23)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Type idiom detected: calculating its left and rigth part (line 24)
            # Getting the type of 'known_exception_class' (line 24)
            known_exception_class_294091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 20), 'known_exception_class')
            # Getting the type of 'None' (line 24)
            None_294092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 52), 'None')
            
            (may_be_294093, more_types_in_union_294094) = may_not_be_none(known_exception_class_294091, None_294092)

            if may_be_294093:

                if more_types_in_union_294094:
                    # Runtime conditional SSA (line 24)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                
                
                
                # Call to isinstance(...): (line 25)
                # Processing the call arguments (line 25)
                # Getting the type of 'err' (line 25)
                err_294096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 42), 'err', False)
                # Getting the type of 'known_exception_class' (line 25)
                known_exception_class_294097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 47), 'known_exception_class', False)
                # Processing the call keyword arguments (line 25)
                kwargs_294098 = {}
                # Getting the type of 'isinstance' (line 25)
                isinstance_294095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 31), 'isinstance', False)
                # Calling isinstance(args, kwargs) (line 25)
                isinstance_call_result_294099 = invoke(stypy.reporting.localization.Localization(__file__, 25, 31), isinstance_294095, *[err_294096, known_exception_class_294097], **kwargs_294098)
                
                # Applying the 'not' unary operator (line 25)
                result_not__294100 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 27), 'not', isinstance_call_result_294099)
                
                # Testing the type of an if condition (line 25)
                if_condition_294101 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 25, 24), result_not__294100)
                # Assigning a type to the variable 'if_condition_294101' (line 25)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 24), 'if_condition_294101', if_condition_294101)
                # SSA begins for if statement (line 25)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # SSA join for if statement (line 25)
                module_type_store = module_type_store.join_ssa_context()
                

                if more_types_in_union_294094:
                    # SSA join for if statement (line 24)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            # Call to knownfail(...): (line 28)
            # Processing the call arguments (line 28)
            # Getting the type of 'msg' (line 28)
            msg_294103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 30), 'msg', False)
            # Processing the call keyword arguments (line 28)
            kwargs_294104 = {}
            # Getting the type of 'knownfail' (line 28)
            knownfail_294102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 20), 'knownfail', False)
            # Calling knownfail(args, kwargs) (line 28)
            knownfail_call_result_294105 = invoke(stypy.reporting.localization.Localization(__file__, 28, 20), knownfail_294102, *[msg_294103], **kwargs_294104)
            
            # SSA branch for the else part of an if statement (line 23)
            module_type_store.open_ssa_branch('else')
            # SSA join for if statement (line 23)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for try-except statement (line 19)
            module_type_store = module_type_store.join_ssa_context()
            
            
            
            # Evaluating a boolean operation
            # Getting the type of 'fail_condition' (line 31)
            fail_condition_294106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 15), 'fail_condition')
            
            # Getting the type of 'fail_condition' (line 31)
            fail_condition_294107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 34), 'fail_condition')
            unicode_294108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 52), 'unicode', u'indeterminate')
            # Applying the binary operator '!=' (line 31)
            result_ne_294109 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 34), '!=', fail_condition_294107, unicode_294108)
            
            # Applying the binary operator 'and' (line 31)
            result_and_keyword_294110 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 15), 'and', fail_condition_294106, result_ne_294109)
            
            # Testing the type of an if condition (line 31)
            if_condition_294111 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 31, 12), result_and_keyword_294110)
            # Assigning a type to the variable 'if_condition_294111' (line 31)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 12), 'if_condition_294111', if_condition_294111)
            # SSA begins for if statement (line 31)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to KnownFailureDidNotFailTest(...): (line 32)
            # Processing the call arguments (line 32)
            # Getting the type of 'msg' (line 32)
            msg_294113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 49), 'msg', False)
            # Processing the call keyword arguments (line 32)
            kwargs_294114 = {}
            # Getting the type of 'KnownFailureDidNotFailTest' (line 32)
            KnownFailureDidNotFailTest_294112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 22), 'KnownFailureDidNotFailTest', False)
            # Calling KnownFailureDidNotFailTest(args, kwargs) (line 32)
            KnownFailureDidNotFailTest_call_result_294115 = invoke(stypy.reporting.localization.Localization(__file__, 32, 22), KnownFailureDidNotFailTest_294112, *[msg_294113], **kwargs_294114)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 32, 16), KnownFailureDidNotFailTest_call_result_294115, 'raise parameter', BaseException)
            # SSA join for if statement (line 31)
            module_type_store = module_type_store.join_ssa_context()
            
            # Getting the type of 'result' (line 33)
            result_294116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 19), 'result')
            # Assigning a type to the variable 'stypy_return_type' (line 33)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'stypy_return_type', result_294116)
            
            # ################# End of 'failer(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'failer' in the type store
            # Getting the type of 'stypy_return_type' (line 18)
            stypy_return_type_294117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_294117)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'failer'
            return stypy_return_type_294117

        # Assigning a type to the variable 'failer' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'failer', failer)
        
        # Call to _copy_metadata(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'f' (line 34)
        f_294119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 30), 'f', False)
        # Getting the type of 'failer' (line 34)
        failer_294120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 33), 'failer', False)
        # Processing the call keyword arguments (line 34)
        kwargs_294121 = {}
        # Getting the type of '_copy_metadata' (line 34)
        _copy_metadata_294118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 15), '_copy_metadata', False)
        # Calling _copy_metadata(args, kwargs) (line 34)
        _copy_metadata_call_result_294122 = invoke(stypy.reporting.localization.Localization(__file__, 34, 15), _copy_metadata_294118, *[f_294119, failer_294120], **kwargs_294121)
        
        # Assigning a type to the variable 'stypy_return_type' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'stypy_return_type', _copy_metadata_call_result_294122)
        
        # ################# End of 'known_fail_decorator(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'known_fail_decorator' in the type store
        # Getting the type of 'stypy_return_type' (line 17)
        stypy_return_type_294123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_294123)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'known_fail_decorator'
        return stypy_return_type_294123

    # Assigning a type to the variable 'known_fail_decorator' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'known_fail_decorator', known_fail_decorator)
    # Getting the type of 'known_fail_decorator' (line 35)
    known_fail_decorator_294124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 11), 'known_fail_decorator')
    # Assigning a type to the variable 'stypy_return_type' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type', known_fail_decorator_294124)
    
    # ################# End of 'knownfailureif(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'knownfailureif' in the type store
    # Getting the type of 'stypy_return_type' (line 12)
    stypy_return_type_294125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_294125)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'knownfailureif'
    return stypy_return_type_294125

# Assigning a type to the variable 'knownfailureif' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'knownfailureif', knownfailureif)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
