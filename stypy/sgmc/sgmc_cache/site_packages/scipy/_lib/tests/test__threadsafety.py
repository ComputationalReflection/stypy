
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import threading
4: import time
5: import traceback
6: 
7: from numpy.testing import assert_
8: from pytest import raises as assert_raises
9: 
10: from scipy._lib._threadsafety import ReentrancyLock, non_reentrant, ReentrancyError
11: 
12: 
13: def test_parallel_threads():
14:     # Check that ReentrancyLock serializes work in parallel threads.
15:     #
16:     # The test is not fully deterministic, and may succeed falsely if
17:     # the timings go wrong.
18: 
19:     lock = ReentrancyLock("failure")
20: 
21:     failflag = [False]
22:     exceptions_raised = []
23: 
24:     def worker(k):
25:         try:
26:             with lock:
27:                 assert_(not failflag[0])
28:                 failflag[0] = True
29:                 time.sleep(0.1 * k)
30:                 assert_(failflag[0])
31:                 failflag[0] = False
32:         except:
33:             exceptions_raised.append(traceback.format_exc(2))
34: 
35:     threads = [threading.Thread(target=lambda k=k: worker(k))
36:                for k in range(3)]
37:     for t in threads:
38:         t.start()
39:     for t in threads:
40:         t.join()
41: 
42:     exceptions_raised = "\n".join(exceptions_raised)
43:     assert_(not exceptions_raised, exceptions_raised)
44: 
45: 
46: def test_reentering():
47:     # Check that ReentrancyLock prevents re-entering from the same thread.
48: 
49:     @non_reentrant()
50:     def func(x):
51:         return func(x)
52: 
53:     assert_raises(ReentrancyError, func, 0)
54: 

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

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import time' statement (line 4)
import time

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'time', time, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import traceback' statement (line 5)
import traceback

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'traceback', traceback, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.testing import assert_' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/tests/')
import_712505 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing')

if (type(import_712505) is not StypyTypeError):

    if (import_712505 != 'pyd_module'):
        __import__(import_712505)
        sys_modules_712506 = sys.modules[import_712505]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', sys_modules_712506.module_type_store, module_type_store, ['assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_712506, sys_modules_712506.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', None, module_type_store, ['assert_'], [assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.testing', import_712505)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from pytest import assert_raises' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/tests/')
import_712507 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest')

if (type(import_712507) is not StypyTypeError):

    if (import_712507 != 'pyd_module'):
        __import__(import_712507)
        sys_modules_712508 = sys.modules[import_712507]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', sys_modules_712508.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_712508, sys_modules_712508.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'pytest', import_712507)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from scipy._lib._threadsafety import ReentrancyLock, non_reentrant, ReentrancyError' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/tests/')
import_712509 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib._threadsafety')

if (type(import_712509) is not StypyTypeError):

    if (import_712509 != 'pyd_module'):
        __import__(import_712509)
        sys_modules_712510 = sys.modules[import_712509]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib._threadsafety', sys_modules_712510.module_type_store, module_type_store, ['ReentrancyLock', 'non_reentrant', 'ReentrancyError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_712510, sys_modules_712510.module_type_store, module_type_store)
    else:
        from scipy._lib._threadsafety import ReentrancyLock, non_reentrant, ReentrancyError

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib._threadsafety', None, module_type_store, ['ReentrancyLock', 'non_reentrant', 'ReentrancyError'], [ReentrancyLock, non_reentrant, ReentrancyError])

else:
    # Assigning a type to the variable 'scipy._lib._threadsafety' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy._lib._threadsafety', import_712509)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/tests/')


@norecursion
def test_parallel_threads(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_parallel_threads'
    module_type_store = module_type_store.open_function_context('test_parallel_threads', 13, 0, False)
    
    # Passed parameters checking function
    test_parallel_threads.stypy_localization = localization
    test_parallel_threads.stypy_type_of_self = None
    test_parallel_threads.stypy_type_store = module_type_store
    test_parallel_threads.stypy_function_name = 'test_parallel_threads'
    test_parallel_threads.stypy_param_names_list = []
    test_parallel_threads.stypy_varargs_param_name = None
    test_parallel_threads.stypy_kwargs_param_name = None
    test_parallel_threads.stypy_call_defaults = defaults
    test_parallel_threads.stypy_call_varargs = varargs
    test_parallel_threads.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_parallel_threads', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_parallel_threads', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_parallel_threads(...)' code ##################

    
    # Assigning a Call to a Name (line 19):
    
    # Call to ReentrancyLock(...): (line 19)
    # Processing the call arguments (line 19)
    str_712512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 26), 'str', 'failure')
    # Processing the call keyword arguments (line 19)
    kwargs_712513 = {}
    # Getting the type of 'ReentrancyLock' (line 19)
    ReentrancyLock_712511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 11), 'ReentrancyLock', False)
    # Calling ReentrancyLock(args, kwargs) (line 19)
    ReentrancyLock_call_result_712514 = invoke(stypy.reporting.localization.Localization(__file__, 19, 11), ReentrancyLock_712511, *[str_712512], **kwargs_712513)
    
    # Assigning a type to the variable 'lock' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'lock', ReentrancyLock_call_result_712514)
    
    # Assigning a List to a Name (line 21):
    
    # Obtaining an instance of the builtin type 'list' (line 21)
    list_712515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 21)
    # Adding element type (line 21)
    # Getting the type of 'False' (line 21)
    False_712516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 15), list_712515, False_712516)
    
    # Assigning a type to the variable 'failflag' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'failflag', list_712515)
    
    # Assigning a List to a Name (line 22):
    
    # Obtaining an instance of the builtin type 'list' (line 22)
    list_712517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 22)
    
    # Assigning a type to the variable 'exceptions_raised' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'exceptions_raised', list_712517)

    @norecursion
    def worker(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'worker'
        module_type_store = module_type_store.open_function_context('worker', 24, 4, False)
        
        # Passed parameters checking function
        worker.stypy_localization = localization
        worker.stypy_type_of_self = None
        worker.stypy_type_store = module_type_store
        worker.stypy_function_name = 'worker'
        worker.stypy_param_names_list = ['k']
        worker.stypy_varargs_param_name = None
        worker.stypy_kwargs_param_name = None
        worker.stypy_call_defaults = defaults
        worker.stypy_call_varargs = varargs
        worker.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'worker', ['k'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'worker', localization, ['k'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'worker(...)' code ##################

        
        
        # SSA begins for try-except statement (line 25)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        # Getting the type of 'lock' (line 26)
        lock_712518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 17), 'lock')
        with_712519 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 26, 17), lock_712518, 'with parameter', '__enter__', '__exit__')

        if with_712519:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 26)
            enter___712520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 17), lock_712518, '__enter__')
            with_enter_712521 = invoke(stypy.reporting.localization.Localization(__file__, 26, 17), enter___712520)
            
            # Call to assert_(...): (line 27)
            # Processing the call arguments (line 27)
            
            
            # Obtaining the type of the subscript
            int_712523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 37), 'int')
            # Getting the type of 'failflag' (line 27)
            failflag_712524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 28), 'failflag', False)
            # Obtaining the member '__getitem__' of a type (line 27)
            getitem___712525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 28), failflag_712524, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 27)
            subscript_call_result_712526 = invoke(stypy.reporting.localization.Localization(__file__, 27, 28), getitem___712525, int_712523)
            
            # Applying the 'not' unary operator (line 27)
            result_not__712527 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 24), 'not', subscript_call_result_712526)
            
            # Processing the call keyword arguments (line 27)
            kwargs_712528 = {}
            # Getting the type of 'assert_' (line 27)
            assert__712522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 16), 'assert_', False)
            # Calling assert_(args, kwargs) (line 27)
            assert__call_result_712529 = invoke(stypy.reporting.localization.Localization(__file__, 27, 16), assert__712522, *[result_not__712527], **kwargs_712528)
            
            
            # Assigning a Name to a Subscript (line 28):
            # Getting the type of 'True' (line 28)
            True_712530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 30), 'True')
            # Getting the type of 'failflag' (line 28)
            failflag_712531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 16), 'failflag')
            int_712532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 25), 'int')
            # Storing an element on a container (line 28)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 16), failflag_712531, (int_712532, True_712530))
            
            # Call to sleep(...): (line 29)
            # Processing the call arguments (line 29)
            float_712535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 27), 'float')
            # Getting the type of 'k' (line 29)
            k_712536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 33), 'k', False)
            # Applying the binary operator '*' (line 29)
            result_mul_712537 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 27), '*', float_712535, k_712536)
            
            # Processing the call keyword arguments (line 29)
            kwargs_712538 = {}
            # Getting the type of 'time' (line 29)
            time_712533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 16), 'time', False)
            # Obtaining the member 'sleep' of a type (line 29)
            sleep_712534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 16), time_712533, 'sleep')
            # Calling sleep(args, kwargs) (line 29)
            sleep_call_result_712539 = invoke(stypy.reporting.localization.Localization(__file__, 29, 16), sleep_712534, *[result_mul_712537], **kwargs_712538)
            
            
            # Call to assert_(...): (line 30)
            # Processing the call arguments (line 30)
            
            # Obtaining the type of the subscript
            int_712541 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 33), 'int')
            # Getting the type of 'failflag' (line 30)
            failflag_712542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 24), 'failflag', False)
            # Obtaining the member '__getitem__' of a type (line 30)
            getitem___712543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 24), failflag_712542, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 30)
            subscript_call_result_712544 = invoke(stypy.reporting.localization.Localization(__file__, 30, 24), getitem___712543, int_712541)
            
            # Processing the call keyword arguments (line 30)
            kwargs_712545 = {}
            # Getting the type of 'assert_' (line 30)
            assert__712540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 16), 'assert_', False)
            # Calling assert_(args, kwargs) (line 30)
            assert__call_result_712546 = invoke(stypy.reporting.localization.Localization(__file__, 30, 16), assert__712540, *[subscript_call_result_712544], **kwargs_712545)
            
            
            # Assigning a Name to a Subscript (line 31):
            # Getting the type of 'False' (line 31)
            False_712547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 30), 'False')
            # Getting the type of 'failflag' (line 31)
            failflag_712548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 16), 'failflag')
            int_712549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 25), 'int')
            # Storing an element on a container (line 31)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 16), failflag_712548, (int_712549, False_712547))
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 26)
            exit___712550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 17), lock_712518, '__exit__')
            with_exit_712551 = invoke(stypy.reporting.localization.Localization(__file__, 26, 17), exit___712550, None, None, None)

        # SSA branch for the except part of a try statement (line 25)
        # SSA branch for the except '<any exception>' branch of a try statement (line 25)
        module_type_store.open_ssa_branch('except')
        
        # Call to append(...): (line 33)
        # Processing the call arguments (line 33)
        
        # Call to format_exc(...): (line 33)
        # Processing the call arguments (line 33)
        int_712556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 58), 'int')
        # Processing the call keyword arguments (line 33)
        kwargs_712557 = {}
        # Getting the type of 'traceback' (line 33)
        traceback_712554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 37), 'traceback', False)
        # Obtaining the member 'format_exc' of a type (line 33)
        format_exc_712555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 37), traceback_712554, 'format_exc')
        # Calling format_exc(args, kwargs) (line 33)
        format_exc_call_result_712558 = invoke(stypy.reporting.localization.Localization(__file__, 33, 37), format_exc_712555, *[int_712556], **kwargs_712557)
        
        # Processing the call keyword arguments (line 33)
        kwargs_712559 = {}
        # Getting the type of 'exceptions_raised' (line 33)
        exceptions_raised_712552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'exceptions_raised', False)
        # Obtaining the member 'append' of a type (line 33)
        append_712553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 12), exceptions_raised_712552, 'append')
        # Calling append(args, kwargs) (line 33)
        append_call_result_712560 = invoke(stypy.reporting.localization.Localization(__file__, 33, 12), append_712553, *[format_exc_call_result_712558], **kwargs_712559)
        
        # SSA join for try-except statement (line 25)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'worker(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'worker' in the type store
        # Getting the type of 'stypy_return_type' (line 24)
        stypy_return_type_712561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_712561)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'worker'
        return stypy_return_type_712561

    # Assigning a type to the variable 'worker' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'worker', worker)
    
    # Assigning a ListComp to a Name (line 35):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to range(...): (line 36)
    # Processing the call arguments (line 36)
    int_712575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 30), 'int')
    # Processing the call keyword arguments (line 36)
    kwargs_712576 = {}
    # Getting the type of 'range' (line 36)
    range_712574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 24), 'range', False)
    # Calling range(args, kwargs) (line 36)
    range_call_result_712577 = invoke(stypy.reporting.localization.Localization(__file__, 36, 24), range_712574, *[int_712575], **kwargs_712576)
    
    comprehension_712578 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 15), range_call_result_712577)
    # Assigning a type to the variable 'k' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 15), 'k', comprehension_712578)
    
    # Call to Thread(...): (line 35)
    # Processing the call keyword arguments (line 35)

    @norecursion
    def _stypy_temp_lambda_594(localization, *varargs, **kwargs):
        global module_type_store
        # Getting the type of 'k' (line 35)
        k_712564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 48), 'k', False)
        # Assign values to the parameters with defaults
        defaults = [k_712564]
        # Create a new context for function '_stypy_temp_lambda_594'
        module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_594', 35, 39, True)
        # Passed parameters checking function
        _stypy_temp_lambda_594.stypy_localization = localization
        _stypy_temp_lambda_594.stypy_type_of_self = None
        _stypy_temp_lambda_594.stypy_type_store = module_type_store
        _stypy_temp_lambda_594.stypy_function_name = '_stypy_temp_lambda_594'
        _stypy_temp_lambda_594.stypy_param_names_list = ['k']
        _stypy_temp_lambda_594.stypy_varargs_param_name = None
        _stypy_temp_lambda_594.stypy_kwargs_param_name = None
        _stypy_temp_lambda_594.stypy_call_defaults = defaults
        _stypy_temp_lambda_594.stypy_call_varargs = varargs
        _stypy_temp_lambda_594.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_594', ['k'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Stacktrace push for error reporting
        localization.set_stack_trace('_stypy_temp_lambda_594', ['k'], arguments)
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of the lambda function code ##################

        
        # Call to worker(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'k' (line 35)
        k_712566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 58), 'k', False)
        # Processing the call keyword arguments (line 35)
        kwargs_712567 = {}
        # Getting the type of 'worker' (line 35)
        worker_712565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 51), 'worker', False)
        # Calling worker(args, kwargs) (line 35)
        worker_call_result_712568 = invoke(stypy.reporting.localization.Localization(__file__, 35, 51), worker_712565, *[k_712566], **kwargs_712567)
        
        # Assigning the return type of the lambda function
        # Assigning a type to the variable 'stypy_return_type' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 39), 'stypy_return_type', worker_call_result_712568)
        
        # ################# End of the lambda function code ##################

        # Stacktrace pop (error reporting)
        localization.unset_stack_trace()
        
        # Storing the return type of function '_stypy_temp_lambda_594' in the type store
        # Getting the type of 'stypy_return_type' (line 35)
        stypy_return_type_712569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 39), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_712569)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_stypy_temp_lambda_594'
        return stypy_return_type_712569

    # Assigning a type to the variable '_stypy_temp_lambda_594' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 39), '_stypy_temp_lambda_594', _stypy_temp_lambda_594)
    # Getting the type of '_stypy_temp_lambda_594' (line 35)
    _stypy_temp_lambda_594_712570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 39), '_stypy_temp_lambda_594')
    keyword_712571 = _stypy_temp_lambda_594_712570
    kwargs_712572 = {'target': keyword_712571}
    # Getting the type of 'threading' (line 35)
    threading_712562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 15), 'threading', False)
    # Obtaining the member 'Thread' of a type (line 35)
    Thread_712563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 15), threading_712562, 'Thread')
    # Calling Thread(args, kwargs) (line 35)
    Thread_call_result_712573 = invoke(stypy.reporting.localization.Localization(__file__, 35, 15), Thread_712563, *[], **kwargs_712572)
    
    list_712579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 15), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 15), list_712579, Thread_call_result_712573)
    # Assigning a type to the variable 'threads' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'threads', list_712579)
    
    # Getting the type of 'threads' (line 37)
    threads_712580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 13), 'threads')
    # Testing the type of a for loop iterable (line 37)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 37, 4), threads_712580)
    # Getting the type of the for loop variable (line 37)
    for_loop_var_712581 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 37, 4), threads_712580)
    # Assigning a type to the variable 't' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 't', for_loop_var_712581)
    # SSA begins for a for statement (line 37)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to start(...): (line 38)
    # Processing the call keyword arguments (line 38)
    kwargs_712584 = {}
    # Getting the type of 't' (line 38)
    t_712582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 't', False)
    # Obtaining the member 'start' of a type (line 38)
    start_712583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), t_712582, 'start')
    # Calling start(args, kwargs) (line 38)
    start_call_result_712585 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), start_712583, *[], **kwargs_712584)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'threads' (line 39)
    threads_712586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 13), 'threads')
    # Testing the type of a for loop iterable (line 39)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 39, 4), threads_712586)
    # Getting the type of the for loop variable (line 39)
    for_loop_var_712587 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 39, 4), threads_712586)
    # Assigning a type to the variable 't' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 't', for_loop_var_712587)
    # SSA begins for a for statement (line 39)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to join(...): (line 40)
    # Processing the call keyword arguments (line 40)
    kwargs_712590 = {}
    # Getting the type of 't' (line 40)
    t_712588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 't', False)
    # Obtaining the member 'join' of a type (line 40)
    join_712589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), t_712588, 'join')
    # Calling join(args, kwargs) (line 40)
    join_call_result_712591 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), join_712589, *[], **kwargs_712590)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 42):
    
    # Call to join(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'exceptions_raised' (line 42)
    exceptions_raised_712594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 34), 'exceptions_raised', False)
    # Processing the call keyword arguments (line 42)
    kwargs_712595 = {}
    str_712592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 24), 'str', '\n')
    # Obtaining the member 'join' of a type (line 42)
    join_712593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 24), str_712592, 'join')
    # Calling join(args, kwargs) (line 42)
    join_call_result_712596 = invoke(stypy.reporting.localization.Localization(__file__, 42, 24), join_712593, *[exceptions_raised_712594], **kwargs_712595)
    
    # Assigning a type to the variable 'exceptions_raised' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'exceptions_raised', join_call_result_712596)
    
    # Call to assert_(...): (line 43)
    # Processing the call arguments (line 43)
    
    # Getting the type of 'exceptions_raised' (line 43)
    exceptions_raised_712598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 16), 'exceptions_raised', False)
    # Applying the 'not' unary operator (line 43)
    result_not__712599 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 12), 'not', exceptions_raised_712598)
    
    # Getting the type of 'exceptions_raised' (line 43)
    exceptions_raised_712600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 35), 'exceptions_raised', False)
    # Processing the call keyword arguments (line 43)
    kwargs_712601 = {}
    # Getting the type of 'assert_' (line 43)
    assert__712597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 43)
    assert__call_result_712602 = invoke(stypy.reporting.localization.Localization(__file__, 43, 4), assert__712597, *[result_not__712599, exceptions_raised_712600], **kwargs_712601)
    
    
    # ################# End of 'test_parallel_threads(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_parallel_threads' in the type store
    # Getting the type of 'stypy_return_type' (line 13)
    stypy_return_type_712603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_712603)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_parallel_threads'
    return stypy_return_type_712603

# Assigning a type to the variable 'test_parallel_threads' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'test_parallel_threads', test_parallel_threads)

@norecursion
def test_reentering(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_reentering'
    module_type_store = module_type_store.open_function_context('test_reentering', 46, 0, False)
    
    # Passed parameters checking function
    test_reentering.stypy_localization = localization
    test_reentering.stypy_type_of_self = None
    test_reentering.stypy_type_store = module_type_store
    test_reentering.stypy_function_name = 'test_reentering'
    test_reentering.stypy_param_names_list = []
    test_reentering.stypy_varargs_param_name = None
    test_reentering.stypy_kwargs_param_name = None
    test_reentering.stypy_call_defaults = defaults
    test_reentering.stypy_call_varargs = varargs
    test_reentering.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_reentering', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_reentering', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_reentering(...)' code ##################


    @norecursion
    def func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'func'
        module_type_store = module_type_store.open_function_context('func', 49, 4, False)
        
        # Passed parameters checking function
        func.stypy_localization = localization
        func.stypy_type_of_self = None
        func.stypy_type_store = module_type_store
        func.stypy_function_name = 'func'
        func.stypy_param_names_list = ['x']
        func.stypy_varargs_param_name = None
        func.stypy_kwargs_param_name = None
        func.stypy_call_defaults = defaults
        func.stypy_call_varargs = varargs
        func.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'func', ['x'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'func', localization, ['x'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'func(...)' code ##################

        
        # Call to func(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'x' (line 51)
        x_712605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 20), 'x', False)
        # Processing the call keyword arguments (line 51)
        kwargs_712606 = {}
        # Getting the type of 'func' (line 51)
        func_712604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 15), 'func', False)
        # Calling func(args, kwargs) (line 51)
        func_call_result_712607 = invoke(stypy.reporting.localization.Localization(__file__, 51, 15), func_712604, *[x_712605], **kwargs_712606)
        
        # Assigning a type to the variable 'stypy_return_type' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'stypy_return_type', func_call_result_712607)
        
        # ################# End of 'func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'func' in the type store
        # Getting the type of 'stypy_return_type' (line 49)
        stypy_return_type_712608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_712608)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'func'
        return stypy_return_type_712608

    # Assigning a type to the variable 'func' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'func', func)
    
    # Call to assert_raises(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'ReentrancyError' (line 53)
    ReentrancyError_712610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 18), 'ReentrancyError', False)
    # Getting the type of 'func' (line 53)
    func_712611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 35), 'func', False)
    int_712612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 41), 'int')
    # Processing the call keyword arguments (line 53)
    kwargs_712613 = {}
    # Getting the type of 'assert_raises' (line 53)
    assert_raises_712609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'assert_raises', False)
    # Calling assert_raises(args, kwargs) (line 53)
    assert_raises_call_result_712614 = invoke(stypy.reporting.localization.Localization(__file__, 53, 4), assert_raises_712609, *[ReentrancyError_712610, func_712611, int_712612], **kwargs_712613)
    
    
    # ################# End of 'test_reentering(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_reentering' in the type store
    # Getting the type of 'stypy_return_type' (line 46)
    stypy_return_type_712615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_712615)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_reentering'
    return stypy_return_type_712615

# Assigning a type to the variable 'test_reentering' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'test_reentering', test_reentering)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
