
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: import itertools
4: import threading
5: import time
6: 
7: import numpy as np
8: from numpy.testing import assert_equal
9: import pytest
10: import scipy.interpolate
11: 
12: 
13: class TestGIL(object):
14:     '''Check if the GIL is properly released by scipy.interpolate functions.'''
15: 
16:     def setup_method(self):
17:         self.messages = []
18: 
19:     def log(self, message):
20:         self.messages.append(message)
21: 
22:     def make_worker_thread(self, target, args):
23:         log = self.log
24: 
25:         class WorkerThread(threading.Thread):
26:             def run(self):
27:                 log('interpolation started')
28:                 target(*args)
29:                 log('interpolation complete')
30: 
31:         return WorkerThread()
32: 
33:     @pytest.mark.slow
34:     @pytest.mark.xfail(reason='race conditions, may depend on system load')
35:     def test_rectbivariatespline(self):
36:         def generate_params(n_points):
37:             x = y = np.linspace(0, 1000, n_points)
38:             x_grid, y_grid = np.meshgrid(x, y)
39:             z = x_grid * y_grid
40:             return x, y, z
41: 
42:         def calibrate_delay(requested_time):
43:             for n_points in itertools.count(5000, 1000):
44:                 args = generate_params(n_points)
45:                 time_started = time.time()
46:                 interpolate(*args)
47:                 if time.time() - time_started > requested_time:
48:                     return args
49: 
50:         def interpolate(x, y, z):
51:             scipy.interpolate.RectBivariateSpline(x, y, z)
52: 
53:         args = calibrate_delay(requested_time=3)
54:         worker_thread = self.make_worker_thread(interpolate, args)
55:         worker_thread.start()
56:         for i in range(3):
57:             time.sleep(0.5)
58:             self.log('working')
59:         worker_thread.join()
60:         assert_equal(self.messages, [
61:             'interpolation started',
62:             'working',
63:             'working',
64:             'working',
65:             'interpolation complete',
66:         ])
67: 
68: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import itertools' statement (line 3)
import itertools

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'itertools', itertools, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import threading' statement (line 4)
import threading

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'threading', threading, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import time' statement (line 5)
import time

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'time', time, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import numpy' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/tests/')
import_95413 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy')

if (type(import_95413) is not StypyTypeError):

    if (import_95413 != 'pyd_module'):
        __import__(import_95413)
        sys_modules_95414 = sys.modules[import_95413]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'np', sys_modules_95414.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', import_95413)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from numpy.testing import assert_equal' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/tests/')
import_95415 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.testing')

if (type(import_95415) is not StypyTypeError):

    if (import_95415 != 'pyd_module'):
        __import__(import_95415)
        sys_modules_95416 = sys.modules[import_95415]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.testing', sys_modules_95416.module_type_store, module_type_store, ['assert_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_95416, sys_modules_95416.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.testing', None, module_type_store, ['assert_equal'], [assert_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.testing', import_95415)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import pytest' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/tests/')
import_95417 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest')

if (type(import_95417) is not StypyTypeError):

    if (import_95417 != 'pyd_module'):
        __import__(import_95417)
        sys_modules_95418 = sys.modules[import_95417]
        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', sys_modules_95418.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'pytest', import_95417)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import scipy.interpolate' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/interpolate/tests/')
import_95419 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.interpolate')

if (type(import_95419) is not StypyTypeError):

    if (import_95419 != 'pyd_module'):
        __import__(import_95419)
        sys_modules_95420 = sys.modules[import_95419]
        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.interpolate', sys_modules_95420.module_type_store, module_type_store)
    else:
        import scipy.interpolate

        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.interpolate', scipy.interpolate, module_type_store)

else:
    # Assigning a type to the variable 'scipy.interpolate' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'scipy.interpolate', import_95419)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/interpolate/tests/')

# Declaration of the 'TestGIL' class

class TestGIL(object, ):
    str_95421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 4), 'str', 'Check if the GIL is properly released by scipy.interpolate functions.')

    @norecursion
    def setup_method(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setup_method'
        module_type_store = module_type_store.open_function_context('setup_method', 16, 4, False)
        # Assigning a type to the variable 'self' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestGIL.setup_method.__dict__.__setitem__('stypy_localization', localization)
        TestGIL.setup_method.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestGIL.setup_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestGIL.setup_method.__dict__.__setitem__('stypy_function_name', 'TestGIL.setup_method')
        TestGIL.setup_method.__dict__.__setitem__('stypy_param_names_list', [])
        TestGIL.setup_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestGIL.setup_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestGIL.setup_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestGIL.setup_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestGIL.setup_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestGIL.setup_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGIL.setup_method', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setup_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setup_method(...)' code ##################

        
        # Assigning a List to a Attribute (line 17):
        
        # Assigning a List to a Attribute (line 17):
        
        # Obtaining an instance of the builtin type 'list' (line 17)
        list_95422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 17)
        
        # Getting the type of 'self' (line 17)
        self_95423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'self')
        # Setting the type of the member 'messages' of a type (line 17)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 8), self_95423, 'messages', list_95422)
        
        # ################# End of 'setup_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setup_method' in the type store
        # Getting the type of 'stypy_return_type' (line 16)
        stypy_return_type_95424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_95424)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setup_method'
        return stypy_return_type_95424


    @norecursion
    def log(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'log'
        module_type_store = module_type_store.open_function_context('log', 19, 4, False)
        # Assigning a type to the variable 'self' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestGIL.log.__dict__.__setitem__('stypy_localization', localization)
        TestGIL.log.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestGIL.log.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestGIL.log.__dict__.__setitem__('stypy_function_name', 'TestGIL.log')
        TestGIL.log.__dict__.__setitem__('stypy_param_names_list', ['message'])
        TestGIL.log.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestGIL.log.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestGIL.log.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestGIL.log.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestGIL.log.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestGIL.log.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGIL.log', ['message'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'log', localization, ['message'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'log(...)' code ##################

        
        # Call to append(...): (line 20)
        # Processing the call arguments (line 20)
        # Getting the type of 'message' (line 20)
        message_95428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 29), 'message', False)
        # Processing the call keyword arguments (line 20)
        kwargs_95429 = {}
        # Getting the type of 'self' (line 20)
        self_95425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'self', False)
        # Obtaining the member 'messages' of a type (line 20)
        messages_95426 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), self_95425, 'messages')
        # Obtaining the member 'append' of a type (line 20)
        append_95427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), messages_95426, 'append')
        # Calling append(args, kwargs) (line 20)
        append_call_result_95430 = invoke(stypy.reporting.localization.Localization(__file__, 20, 8), append_95427, *[message_95428], **kwargs_95429)
        
        
        # ################# End of 'log(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'log' in the type store
        # Getting the type of 'stypy_return_type' (line 19)
        stypy_return_type_95431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_95431)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'log'
        return stypy_return_type_95431


    @norecursion
    def make_worker_thread(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'make_worker_thread'
        module_type_store = module_type_store.open_function_context('make_worker_thread', 22, 4, False)
        # Assigning a type to the variable 'self' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestGIL.make_worker_thread.__dict__.__setitem__('stypy_localization', localization)
        TestGIL.make_worker_thread.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestGIL.make_worker_thread.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestGIL.make_worker_thread.__dict__.__setitem__('stypy_function_name', 'TestGIL.make_worker_thread')
        TestGIL.make_worker_thread.__dict__.__setitem__('stypy_param_names_list', ['target', 'args'])
        TestGIL.make_worker_thread.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestGIL.make_worker_thread.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestGIL.make_worker_thread.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestGIL.make_worker_thread.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestGIL.make_worker_thread.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestGIL.make_worker_thread.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGIL.make_worker_thread', ['target', 'args'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'make_worker_thread', localization, ['target', 'args'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'make_worker_thread(...)' code ##################

        
        # Assigning a Attribute to a Name (line 23):
        
        # Assigning a Attribute to a Name (line 23):
        # Getting the type of 'self' (line 23)
        self_95432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 14), 'self')
        # Obtaining the member 'log' of a type (line 23)
        log_95433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 14), self_95432, 'log')
        # Assigning a type to the variable 'log' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'log', log_95433)
        # Declaration of the 'WorkerThread' class
        # Getting the type of 'threading' (line 25)
        threading_95434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 27), 'threading')
        # Obtaining the member 'Thread' of a type (line 25)
        Thread_95435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 27), threading_95434, 'Thread')

        class WorkerThread(Thread_95435, ):

            @norecursion
            def run(type_of_self, localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'run'
                module_type_store = module_type_store.open_function_context('run', 26, 12, False)
                # Assigning a type to the variable 'self' (line 27)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'self', type_of_self)
                
                # Passed parameters checking function
                WorkerThread.run.__dict__.__setitem__('stypy_localization', localization)
                WorkerThread.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
                WorkerThread.run.__dict__.__setitem__('stypy_type_store', module_type_store)
                WorkerThread.run.__dict__.__setitem__('stypy_function_name', 'WorkerThread.run')
                WorkerThread.run.__dict__.__setitem__('stypy_param_names_list', [])
                WorkerThread.run.__dict__.__setitem__('stypy_varargs_param_name', None)
                WorkerThread.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
                WorkerThread.run.__dict__.__setitem__('stypy_call_defaults', defaults)
                WorkerThread.run.__dict__.__setitem__('stypy_call_varargs', varargs)
                WorkerThread.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
                WorkerThread.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
                arguments = process_argument_values(localization, type_of_self, module_type_store, 'WorkerThread.run', [], None, None, defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'run', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'run(...)' code ##################

                
                # Call to log(...): (line 27)
                # Processing the call arguments (line 27)
                str_95437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 20), 'str', 'interpolation started')
                # Processing the call keyword arguments (line 27)
                kwargs_95438 = {}
                # Getting the type of 'log' (line 27)
                log_95436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 16), 'log', False)
                # Calling log(args, kwargs) (line 27)
                log_call_result_95439 = invoke(stypy.reporting.localization.Localization(__file__, 27, 16), log_95436, *[str_95437], **kwargs_95438)
                
                
                # Call to target(...): (line 28)
                # Getting the type of 'args' (line 28)
                args_95441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 24), 'args', False)
                # Processing the call keyword arguments (line 28)
                kwargs_95442 = {}
                # Getting the type of 'target' (line 28)
                target_95440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 16), 'target', False)
                # Calling target(args, kwargs) (line 28)
                target_call_result_95443 = invoke(stypy.reporting.localization.Localization(__file__, 28, 16), target_95440, *[args_95441], **kwargs_95442)
                
                
                # Call to log(...): (line 29)
                # Processing the call arguments (line 29)
                str_95445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 20), 'str', 'interpolation complete')
                # Processing the call keyword arguments (line 29)
                kwargs_95446 = {}
                # Getting the type of 'log' (line 29)
                log_95444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 16), 'log', False)
                # Calling log(args, kwargs) (line 29)
                log_call_result_95447 = invoke(stypy.reporting.localization.Localization(__file__, 29, 16), log_95444, *[str_95445], **kwargs_95446)
                
                
                # ################# End of 'run(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'run' in the type store
                # Getting the type of 'stypy_return_type' (line 26)
                stypy_return_type_95448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_95448)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'run'
                return stypy_return_type_95448

        
        # Assigning a type to the variable 'WorkerThread' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'WorkerThread', WorkerThread)
        
        # Call to WorkerThread(...): (line 31)
        # Processing the call keyword arguments (line 31)
        kwargs_95450 = {}
        # Getting the type of 'WorkerThread' (line 31)
        WorkerThread_95449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 15), 'WorkerThread', False)
        # Calling WorkerThread(args, kwargs) (line 31)
        WorkerThread_call_result_95451 = invoke(stypy.reporting.localization.Localization(__file__, 31, 15), WorkerThread_95449, *[], **kwargs_95450)
        
        # Assigning a type to the variable 'stypy_return_type' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'stypy_return_type', WorkerThread_call_result_95451)
        
        # ################# End of 'make_worker_thread(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'make_worker_thread' in the type store
        # Getting the type of 'stypy_return_type' (line 22)
        stypy_return_type_95452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_95452)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'make_worker_thread'
        return stypy_return_type_95452


    @norecursion
    def test_rectbivariatespline(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_rectbivariatespline'
        module_type_store = module_type_store.open_function_context('test_rectbivariatespline', 33, 4, False)
        # Assigning a type to the variable 'self' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestGIL.test_rectbivariatespline.__dict__.__setitem__('stypy_localization', localization)
        TestGIL.test_rectbivariatespline.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestGIL.test_rectbivariatespline.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestGIL.test_rectbivariatespline.__dict__.__setitem__('stypy_function_name', 'TestGIL.test_rectbivariatespline')
        TestGIL.test_rectbivariatespline.__dict__.__setitem__('stypy_param_names_list', [])
        TestGIL.test_rectbivariatespline.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestGIL.test_rectbivariatespline.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestGIL.test_rectbivariatespline.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestGIL.test_rectbivariatespline.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestGIL.test_rectbivariatespline.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestGIL.test_rectbivariatespline.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGIL.test_rectbivariatespline', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_rectbivariatespline', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_rectbivariatespline(...)' code ##################


        @norecursion
        def generate_params(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'generate_params'
            module_type_store = module_type_store.open_function_context('generate_params', 36, 8, False)
            
            # Passed parameters checking function
            generate_params.stypy_localization = localization
            generate_params.stypy_type_of_self = None
            generate_params.stypy_type_store = module_type_store
            generate_params.stypy_function_name = 'generate_params'
            generate_params.stypy_param_names_list = ['n_points']
            generate_params.stypy_varargs_param_name = None
            generate_params.stypy_kwargs_param_name = None
            generate_params.stypy_call_defaults = defaults
            generate_params.stypy_call_varargs = varargs
            generate_params.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'generate_params', ['n_points'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'generate_params', localization, ['n_points'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'generate_params(...)' code ##################

            
            # Multiple assignment of 2 elements.
            
            # Assigning a Call to a Name (line 37):
            
            # Call to linspace(...): (line 37)
            # Processing the call arguments (line 37)
            int_95455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 32), 'int')
            int_95456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 35), 'int')
            # Getting the type of 'n_points' (line 37)
            n_points_95457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 41), 'n_points', False)
            # Processing the call keyword arguments (line 37)
            kwargs_95458 = {}
            # Getting the type of 'np' (line 37)
            np_95453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 20), 'np', False)
            # Obtaining the member 'linspace' of a type (line 37)
            linspace_95454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 20), np_95453, 'linspace')
            # Calling linspace(args, kwargs) (line 37)
            linspace_call_result_95459 = invoke(stypy.reporting.localization.Localization(__file__, 37, 20), linspace_95454, *[int_95455, int_95456, n_points_95457], **kwargs_95458)
            
            # Assigning a type to the variable 'y' (line 37)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 16), 'y', linspace_call_result_95459)
            
            # Assigning a Name to a Name (line 37):
            # Getting the type of 'y' (line 37)
            y_95460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 16), 'y')
            # Assigning a type to the variable 'x' (line 37)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'x', y_95460)
            
            # Assigning a Call to a Tuple (line 38):
            
            # Assigning a Subscript to a Name (line 38):
            
            # Obtaining the type of the subscript
            int_95461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 12), 'int')
            
            # Call to meshgrid(...): (line 38)
            # Processing the call arguments (line 38)
            # Getting the type of 'x' (line 38)
            x_95464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 41), 'x', False)
            # Getting the type of 'y' (line 38)
            y_95465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 44), 'y', False)
            # Processing the call keyword arguments (line 38)
            kwargs_95466 = {}
            # Getting the type of 'np' (line 38)
            np_95462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 29), 'np', False)
            # Obtaining the member 'meshgrid' of a type (line 38)
            meshgrid_95463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 29), np_95462, 'meshgrid')
            # Calling meshgrid(args, kwargs) (line 38)
            meshgrid_call_result_95467 = invoke(stypy.reporting.localization.Localization(__file__, 38, 29), meshgrid_95463, *[x_95464, y_95465], **kwargs_95466)
            
            # Obtaining the member '__getitem__' of a type (line 38)
            getitem___95468 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 12), meshgrid_call_result_95467, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 38)
            subscript_call_result_95469 = invoke(stypy.reporting.localization.Localization(__file__, 38, 12), getitem___95468, int_95461)
            
            # Assigning a type to the variable 'tuple_var_assignment_95411' (line 38)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'tuple_var_assignment_95411', subscript_call_result_95469)
            
            # Assigning a Subscript to a Name (line 38):
            
            # Obtaining the type of the subscript
            int_95470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 12), 'int')
            
            # Call to meshgrid(...): (line 38)
            # Processing the call arguments (line 38)
            # Getting the type of 'x' (line 38)
            x_95473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 41), 'x', False)
            # Getting the type of 'y' (line 38)
            y_95474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 44), 'y', False)
            # Processing the call keyword arguments (line 38)
            kwargs_95475 = {}
            # Getting the type of 'np' (line 38)
            np_95471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 29), 'np', False)
            # Obtaining the member 'meshgrid' of a type (line 38)
            meshgrid_95472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 29), np_95471, 'meshgrid')
            # Calling meshgrid(args, kwargs) (line 38)
            meshgrid_call_result_95476 = invoke(stypy.reporting.localization.Localization(__file__, 38, 29), meshgrid_95472, *[x_95473, y_95474], **kwargs_95475)
            
            # Obtaining the member '__getitem__' of a type (line 38)
            getitem___95477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 12), meshgrid_call_result_95476, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 38)
            subscript_call_result_95478 = invoke(stypy.reporting.localization.Localization(__file__, 38, 12), getitem___95477, int_95470)
            
            # Assigning a type to the variable 'tuple_var_assignment_95412' (line 38)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'tuple_var_assignment_95412', subscript_call_result_95478)
            
            # Assigning a Name to a Name (line 38):
            # Getting the type of 'tuple_var_assignment_95411' (line 38)
            tuple_var_assignment_95411_95479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'tuple_var_assignment_95411')
            # Assigning a type to the variable 'x_grid' (line 38)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'x_grid', tuple_var_assignment_95411_95479)
            
            # Assigning a Name to a Name (line 38):
            # Getting the type of 'tuple_var_assignment_95412' (line 38)
            tuple_var_assignment_95412_95480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'tuple_var_assignment_95412')
            # Assigning a type to the variable 'y_grid' (line 38)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 20), 'y_grid', tuple_var_assignment_95412_95480)
            
            # Assigning a BinOp to a Name (line 39):
            
            # Assigning a BinOp to a Name (line 39):
            # Getting the type of 'x_grid' (line 39)
            x_grid_95481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 16), 'x_grid')
            # Getting the type of 'y_grid' (line 39)
            y_grid_95482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 25), 'y_grid')
            # Applying the binary operator '*' (line 39)
            result_mul_95483 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 16), '*', x_grid_95481, y_grid_95482)
            
            # Assigning a type to the variable 'z' (line 39)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'z', result_mul_95483)
            
            # Obtaining an instance of the builtin type 'tuple' (line 40)
            tuple_95484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 19), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 40)
            # Adding element type (line 40)
            # Getting the type of 'x' (line 40)
            x_95485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 19), 'x')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 19), tuple_95484, x_95485)
            # Adding element type (line 40)
            # Getting the type of 'y' (line 40)
            y_95486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 22), 'y')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 19), tuple_95484, y_95486)
            # Adding element type (line 40)
            # Getting the type of 'z' (line 40)
            z_95487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 25), 'z')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 19), tuple_95484, z_95487)
            
            # Assigning a type to the variable 'stypy_return_type' (line 40)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'stypy_return_type', tuple_95484)
            
            # ################# End of 'generate_params(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'generate_params' in the type store
            # Getting the type of 'stypy_return_type' (line 36)
            stypy_return_type_95488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_95488)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'generate_params'
            return stypy_return_type_95488

        # Assigning a type to the variable 'generate_params' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'generate_params', generate_params)

        @norecursion
        def calibrate_delay(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'calibrate_delay'
            module_type_store = module_type_store.open_function_context('calibrate_delay', 42, 8, False)
            
            # Passed parameters checking function
            calibrate_delay.stypy_localization = localization
            calibrate_delay.stypy_type_of_self = None
            calibrate_delay.stypy_type_store = module_type_store
            calibrate_delay.stypy_function_name = 'calibrate_delay'
            calibrate_delay.stypy_param_names_list = ['requested_time']
            calibrate_delay.stypy_varargs_param_name = None
            calibrate_delay.stypy_kwargs_param_name = None
            calibrate_delay.stypy_call_defaults = defaults
            calibrate_delay.stypy_call_varargs = varargs
            calibrate_delay.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'calibrate_delay', ['requested_time'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'calibrate_delay', localization, ['requested_time'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'calibrate_delay(...)' code ##################

            
            
            # Call to count(...): (line 43)
            # Processing the call arguments (line 43)
            int_95491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 44), 'int')
            int_95492 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 50), 'int')
            # Processing the call keyword arguments (line 43)
            kwargs_95493 = {}
            # Getting the type of 'itertools' (line 43)
            itertools_95489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 28), 'itertools', False)
            # Obtaining the member 'count' of a type (line 43)
            count_95490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 28), itertools_95489, 'count')
            # Calling count(args, kwargs) (line 43)
            count_call_result_95494 = invoke(stypy.reporting.localization.Localization(__file__, 43, 28), count_95490, *[int_95491, int_95492], **kwargs_95493)
            
            # Testing the type of a for loop iterable (line 43)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 43, 12), count_call_result_95494)
            # Getting the type of the for loop variable (line 43)
            for_loop_var_95495 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 43, 12), count_call_result_95494)
            # Assigning a type to the variable 'n_points' (line 43)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'n_points', for_loop_var_95495)
            # SSA begins for a for statement (line 43)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 44):
            
            # Assigning a Call to a Name (line 44):
            
            # Call to generate_params(...): (line 44)
            # Processing the call arguments (line 44)
            # Getting the type of 'n_points' (line 44)
            n_points_95497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 39), 'n_points', False)
            # Processing the call keyword arguments (line 44)
            kwargs_95498 = {}
            # Getting the type of 'generate_params' (line 44)
            generate_params_95496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 23), 'generate_params', False)
            # Calling generate_params(args, kwargs) (line 44)
            generate_params_call_result_95499 = invoke(stypy.reporting.localization.Localization(__file__, 44, 23), generate_params_95496, *[n_points_95497], **kwargs_95498)
            
            # Assigning a type to the variable 'args' (line 44)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 16), 'args', generate_params_call_result_95499)
            
            # Assigning a Call to a Name (line 45):
            
            # Assigning a Call to a Name (line 45):
            
            # Call to time(...): (line 45)
            # Processing the call keyword arguments (line 45)
            kwargs_95502 = {}
            # Getting the type of 'time' (line 45)
            time_95500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 31), 'time', False)
            # Obtaining the member 'time' of a type (line 45)
            time_95501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 31), time_95500, 'time')
            # Calling time(args, kwargs) (line 45)
            time_call_result_95503 = invoke(stypy.reporting.localization.Localization(__file__, 45, 31), time_95501, *[], **kwargs_95502)
            
            # Assigning a type to the variable 'time_started' (line 45)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 16), 'time_started', time_call_result_95503)
            
            # Call to interpolate(...): (line 46)
            # Getting the type of 'args' (line 46)
            args_95505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 29), 'args', False)
            # Processing the call keyword arguments (line 46)
            kwargs_95506 = {}
            # Getting the type of 'interpolate' (line 46)
            interpolate_95504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 16), 'interpolate', False)
            # Calling interpolate(args, kwargs) (line 46)
            interpolate_call_result_95507 = invoke(stypy.reporting.localization.Localization(__file__, 46, 16), interpolate_95504, *[args_95505], **kwargs_95506)
            
            
            
            
            # Call to time(...): (line 47)
            # Processing the call keyword arguments (line 47)
            kwargs_95510 = {}
            # Getting the type of 'time' (line 47)
            time_95508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 19), 'time', False)
            # Obtaining the member 'time' of a type (line 47)
            time_95509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 19), time_95508, 'time')
            # Calling time(args, kwargs) (line 47)
            time_call_result_95511 = invoke(stypy.reporting.localization.Localization(__file__, 47, 19), time_95509, *[], **kwargs_95510)
            
            # Getting the type of 'time_started' (line 47)
            time_started_95512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 33), 'time_started')
            # Applying the binary operator '-' (line 47)
            result_sub_95513 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 19), '-', time_call_result_95511, time_started_95512)
            
            # Getting the type of 'requested_time' (line 47)
            requested_time_95514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 48), 'requested_time')
            # Applying the binary operator '>' (line 47)
            result_gt_95515 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 19), '>', result_sub_95513, requested_time_95514)
            
            # Testing the type of an if condition (line 47)
            if_condition_95516 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 47, 16), result_gt_95515)
            # Assigning a type to the variable 'if_condition_95516' (line 47)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 16), 'if_condition_95516', if_condition_95516)
            # SSA begins for if statement (line 47)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'args' (line 48)
            args_95517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 27), 'args')
            # Assigning a type to the variable 'stypy_return_type' (line 48)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 20), 'stypy_return_type', args_95517)
            # SSA join for if statement (line 47)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            
            # ################# End of 'calibrate_delay(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'calibrate_delay' in the type store
            # Getting the type of 'stypy_return_type' (line 42)
            stypy_return_type_95518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_95518)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'calibrate_delay'
            return stypy_return_type_95518

        # Assigning a type to the variable 'calibrate_delay' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'calibrate_delay', calibrate_delay)

        @norecursion
        def interpolate(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'interpolate'
            module_type_store = module_type_store.open_function_context('interpolate', 50, 8, False)
            
            # Passed parameters checking function
            interpolate.stypy_localization = localization
            interpolate.stypy_type_of_self = None
            interpolate.stypy_type_store = module_type_store
            interpolate.stypy_function_name = 'interpolate'
            interpolate.stypy_param_names_list = ['x', 'y', 'z']
            interpolate.stypy_varargs_param_name = None
            interpolate.stypy_kwargs_param_name = None
            interpolate.stypy_call_defaults = defaults
            interpolate.stypy_call_varargs = varargs
            interpolate.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'interpolate', ['x', 'y', 'z'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'interpolate', localization, ['x', 'y', 'z'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'interpolate(...)' code ##################

            
            # Call to RectBivariateSpline(...): (line 51)
            # Processing the call arguments (line 51)
            # Getting the type of 'x' (line 51)
            x_95522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 50), 'x', False)
            # Getting the type of 'y' (line 51)
            y_95523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 53), 'y', False)
            # Getting the type of 'z' (line 51)
            z_95524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 56), 'z', False)
            # Processing the call keyword arguments (line 51)
            kwargs_95525 = {}
            # Getting the type of 'scipy' (line 51)
            scipy_95519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'scipy', False)
            # Obtaining the member 'interpolate' of a type (line 51)
            interpolate_95520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 12), scipy_95519, 'interpolate')
            # Obtaining the member 'RectBivariateSpline' of a type (line 51)
            RectBivariateSpline_95521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 12), interpolate_95520, 'RectBivariateSpline')
            # Calling RectBivariateSpline(args, kwargs) (line 51)
            RectBivariateSpline_call_result_95526 = invoke(stypy.reporting.localization.Localization(__file__, 51, 12), RectBivariateSpline_95521, *[x_95522, y_95523, z_95524], **kwargs_95525)
            
            
            # ################# End of 'interpolate(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'interpolate' in the type store
            # Getting the type of 'stypy_return_type' (line 50)
            stypy_return_type_95527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_95527)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'interpolate'
            return stypy_return_type_95527

        # Assigning a type to the variable 'interpolate' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'interpolate', interpolate)
        
        # Assigning a Call to a Name (line 53):
        
        # Assigning a Call to a Name (line 53):
        
        # Call to calibrate_delay(...): (line 53)
        # Processing the call keyword arguments (line 53)
        int_95529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 46), 'int')
        keyword_95530 = int_95529
        kwargs_95531 = {'requested_time': keyword_95530}
        # Getting the type of 'calibrate_delay' (line 53)
        calibrate_delay_95528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 15), 'calibrate_delay', False)
        # Calling calibrate_delay(args, kwargs) (line 53)
        calibrate_delay_call_result_95532 = invoke(stypy.reporting.localization.Localization(__file__, 53, 15), calibrate_delay_95528, *[], **kwargs_95531)
        
        # Assigning a type to the variable 'args' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'args', calibrate_delay_call_result_95532)
        
        # Assigning a Call to a Name (line 54):
        
        # Assigning a Call to a Name (line 54):
        
        # Call to make_worker_thread(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'interpolate' (line 54)
        interpolate_95535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 48), 'interpolate', False)
        # Getting the type of 'args' (line 54)
        args_95536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 61), 'args', False)
        # Processing the call keyword arguments (line 54)
        kwargs_95537 = {}
        # Getting the type of 'self' (line 54)
        self_95533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 24), 'self', False)
        # Obtaining the member 'make_worker_thread' of a type (line 54)
        make_worker_thread_95534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 24), self_95533, 'make_worker_thread')
        # Calling make_worker_thread(args, kwargs) (line 54)
        make_worker_thread_call_result_95538 = invoke(stypy.reporting.localization.Localization(__file__, 54, 24), make_worker_thread_95534, *[interpolate_95535, args_95536], **kwargs_95537)
        
        # Assigning a type to the variable 'worker_thread' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'worker_thread', make_worker_thread_call_result_95538)
        
        # Call to start(...): (line 55)
        # Processing the call keyword arguments (line 55)
        kwargs_95541 = {}
        # Getting the type of 'worker_thread' (line 55)
        worker_thread_95539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'worker_thread', False)
        # Obtaining the member 'start' of a type (line 55)
        start_95540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), worker_thread_95539, 'start')
        # Calling start(args, kwargs) (line 55)
        start_call_result_95542 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), start_95540, *[], **kwargs_95541)
        
        
        
        # Call to range(...): (line 56)
        # Processing the call arguments (line 56)
        int_95544 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 23), 'int')
        # Processing the call keyword arguments (line 56)
        kwargs_95545 = {}
        # Getting the type of 'range' (line 56)
        range_95543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 17), 'range', False)
        # Calling range(args, kwargs) (line 56)
        range_call_result_95546 = invoke(stypy.reporting.localization.Localization(__file__, 56, 17), range_95543, *[int_95544], **kwargs_95545)
        
        # Testing the type of a for loop iterable (line 56)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 56, 8), range_call_result_95546)
        # Getting the type of the for loop variable (line 56)
        for_loop_var_95547 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 56, 8), range_call_result_95546)
        # Assigning a type to the variable 'i' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'i', for_loop_var_95547)
        # SSA begins for a for statement (line 56)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to sleep(...): (line 57)
        # Processing the call arguments (line 57)
        float_95550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 23), 'float')
        # Processing the call keyword arguments (line 57)
        kwargs_95551 = {}
        # Getting the type of 'time' (line 57)
        time_95548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'time', False)
        # Obtaining the member 'sleep' of a type (line 57)
        sleep_95549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 12), time_95548, 'sleep')
        # Calling sleep(args, kwargs) (line 57)
        sleep_call_result_95552 = invoke(stypy.reporting.localization.Localization(__file__, 57, 12), sleep_95549, *[float_95550], **kwargs_95551)
        
        
        # Call to log(...): (line 58)
        # Processing the call arguments (line 58)
        str_95555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 21), 'str', 'working')
        # Processing the call keyword arguments (line 58)
        kwargs_95556 = {}
        # Getting the type of 'self' (line 58)
        self_95553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'self', False)
        # Obtaining the member 'log' of a type (line 58)
        log_95554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 12), self_95553, 'log')
        # Calling log(args, kwargs) (line 58)
        log_call_result_95557 = invoke(stypy.reporting.localization.Localization(__file__, 58, 12), log_95554, *[str_95555], **kwargs_95556)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to join(...): (line 59)
        # Processing the call keyword arguments (line 59)
        kwargs_95560 = {}
        # Getting the type of 'worker_thread' (line 59)
        worker_thread_95558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'worker_thread', False)
        # Obtaining the member 'join' of a type (line 59)
        join_95559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), worker_thread_95558, 'join')
        # Calling join(args, kwargs) (line 59)
        join_call_result_95561 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), join_95559, *[], **kwargs_95560)
        
        
        # Call to assert_equal(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'self' (line 60)
        self_95563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 21), 'self', False)
        # Obtaining the member 'messages' of a type (line 60)
        messages_95564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 21), self_95563, 'messages')
        
        # Obtaining an instance of the builtin type 'list' (line 60)
        list_95565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 60)
        # Adding element type (line 60)
        str_95566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 12), 'str', 'interpolation started')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 36), list_95565, str_95566)
        # Adding element type (line 60)
        str_95567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 12), 'str', 'working')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 36), list_95565, str_95567)
        # Adding element type (line 60)
        str_95568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 12), 'str', 'working')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 36), list_95565, str_95568)
        # Adding element type (line 60)
        str_95569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 12), 'str', 'working')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 36), list_95565, str_95569)
        # Adding element type (line 60)
        str_95570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 12), 'str', 'interpolation complete')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 36), list_95565, str_95570)
        
        # Processing the call keyword arguments (line 60)
        kwargs_95571 = {}
        # Getting the type of 'assert_equal' (line 60)
        assert_equal_95562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 60)
        assert_equal_call_result_95572 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), assert_equal_95562, *[messages_95564, list_95565], **kwargs_95571)
        
        
        # ################# End of 'test_rectbivariatespline(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_rectbivariatespline' in the type store
        # Getting the type of 'stypy_return_type' (line 33)
        stypy_return_type_95573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_95573)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_rectbivariatespline'
        return stypy_return_type_95573


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 13, 0, False)
        # Assigning a type to the variable 'self' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestGIL.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestGIL' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'TestGIL', TestGIL)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
