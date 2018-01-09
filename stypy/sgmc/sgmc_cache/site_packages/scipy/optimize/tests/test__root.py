
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Unit tests for optimization routines from _root.py.
3: '''
4: from __future__ import division, print_function, absolute_import
5: 
6: from numpy.testing import assert_
7: import numpy as np
8: 
9: from scipy.optimize import root
10: 
11: 
12: class TestRoot(object):
13:     def test_tol_parameter(self):
14:         # Check that the minimize() tol= argument does something
15:         def func(z):
16:             x, y = z
17:             return np.array([x**3 - 1, y**3 - 1])
18: 
19:         def dfunc(z):
20:             x, y = z
21:             return np.array([[3*x**2, 0], [0, 3*y**2]])
22: 
23:         for method in ['hybr', 'lm', 'broyden1', 'broyden2', 'anderson',
24:                        'diagbroyden', 'krylov']:
25:             if method in ('linearmixing', 'excitingmixing'):
26:                 # doesn't converge
27:                 continue
28: 
29:             if method in ('hybr', 'lm'):
30:                 jac = dfunc
31:             else:
32:                 jac = None
33: 
34:             sol1 = root(func, [1.1,1.1], jac=jac, tol=1e-4, method=method)
35:             sol2 = root(func, [1.1,1.1], jac=jac, tol=0.5, method=method)
36:             msg = "%s: %s vs. %s" % (method, func(sol1.x), func(sol2.x))
37:             assert_(sol1.success, msg)
38:             assert_(sol2.success, msg)
39:             assert_(abs(func(sol1.x)).max() < abs(func(sol2.x)).max(),
40:                     msg)
41: 
42:     def test_minimize_scalar_coerce_args_param(self):
43:         # github issue #3503
44:         def func(z, f=1):
45:             x, y = z
46:             return np.array([x**3 - 1, y**3 - f])
47:         root(func, [1.1, 1.1], args=1.5)
48: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_245679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', '\nUnit tests for optimization routines from _root.py.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy.testing import assert_' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_245680 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing')

if (type(import_245680) is not StypyTypeError):

    if (import_245680 != 'pyd_module'):
        __import__(import_245680)
        sys_modules_245681 = sys.modules[import_245680]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', sys_modules_245681.module_type_store, module_type_store, ['assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_245681, sys_modules_245681.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', None, module_type_store, ['assert_'], [assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.testing', import_245680)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import numpy' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_245682 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy')

if (type(import_245682) is not StypyTypeError):

    if (import_245682 != 'pyd_module'):
        __import__(import_245682)
        sys_modules_245683 = sys.modules[import_245682]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'np', sys_modules_245683.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy', import_245682)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from scipy.optimize import root' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/optimize/tests/')
import_245684 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.optimize')

if (type(import_245684) is not StypyTypeError):

    if (import_245684 != 'pyd_module'):
        __import__(import_245684)
        sys_modules_245685 = sys.modules[import_245684]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.optimize', sys_modules_245685.module_type_store, module_type_store, ['root'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_245685, sys_modules_245685.module_type_store, module_type_store)
    else:
        from scipy.optimize import root

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.optimize', None, module_type_store, ['root'], [root])

else:
    # Assigning a type to the variable 'scipy.optimize' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'scipy.optimize', import_245684)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/optimize/tests/')

# Declaration of the 'TestRoot' class

class TestRoot(object, ):

    @norecursion
    def test_tol_parameter(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_tol_parameter'
        module_type_store = module_type_store.open_function_context('test_tol_parameter', 13, 4, False)
        # Assigning a type to the variable 'self' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRoot.test_tol_parameter.__dict__.__setitem__('stypy_localization', localization)
        TestRoot.test_tol_parameter.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRoot.test_tol_parameter.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRoot.test_tol_parameter.__dict__.__setitem__('stypy_function_name', 'TestRoot.test_tol_parameter')
        TestRoot.test_tol_parameter.__dict__.__setitem__('stypy_param_names_list', [])
        TestRoot.test_tol_parameter.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRoot.test_tol_parameter.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRoot.test_tol_parameter.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRoot.test_tol_parameter.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRoot.test_tol_parameter.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRoot.test_tol_parameter.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRoot.test_tol_parameter', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_tol_parameter', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_tol_parameter(...)' code ##################


        @norecursion
        def func(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'func'
            module_type_store = module_type_store.open_function_context('func', 15, 8, False)
            
            # Passed parameters checking function
            func.stypy_localization = localization
            func.stypy_type_of_self = None
            func.stypy_type_store = module_type_store
            func.stypy_function_name = 'func'
            func.stypy_param_names_list = ['z']
            func.stypy_varargs_param_name = None
            func.stypy_kwargs_param_name = None
            func.stypy_call_defaults = defaults
            func.stypy_call_varargs = varargs
            func.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'func', ['z'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'func', localization, ['z'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'func(...)' code ##################

            
            # Assigning a Name to a Tuple (line 16):
            
            # Assigning a Subscript to a Name (line 16):
            
            # Obtaining the type of the subscript
            int_245686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 12), 'int')
            # Getting the type of 'z' (line 16)
            z_245687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 19), 'z')
            # Obtaining the member '__getitem__' of a type (line 16)
            getitem___245688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 12), z_245687, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 16)
            subscript_call_result_245689 = invoke(stypy.reporting.localization.Localization(__file__, 16, 12), getitem___245688, int_245686)
            
            # Assigning a type to the variable 'tuple_var_assignment_245673' (line 16)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 12), 'tuple_var_assignment_245673', subscript_call_result_245689)
            
            # Assigning a Subscript to a Name (line 16):
            
            # Obtaining the type of the subscript
            int_245690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 12), 'int')
            # Getting the type of 'z' (line 16)
            z_245691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 19), 'z')
            # Obtaining the member '__getitem__' of a type (line 16)
            getitem___245692 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 12), z_245691, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 16)
            subscript_call_result_245693 = invoke(stypy.reporting.localization.Localization(__file__, 16, 12), getitem___245692, int_245690)
            
            # Assigning a type to the variable 'tuple_var_assignment_245674' (line 16)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 12), 'tuple_var_assignment_245674', subscript_call_result_245693)
            
            # Assigning a Name to a Name (line 16):
            # Getting the type of 'tuple_var_assignment_245673' (line 16)
            tuple_var_assignment_245673_245694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 12), 'tuple_var_assignment_245673')
            # Assigning a type to the variable 'x' (line 16)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 12), 'x', tuple_var_assignment_245673_245694)
            
            # Assigning a Name to a Name (line 16):
            # Getting the type of 'tuple_var_assignment_245674' (line 16)
            tuple_var_assignment_245674_245695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 12), 'tuple_var_assignment_245674')
            # Assigning a type to the variable 'y' (line 16)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 15), 'y', tuple_var_assignment_245674_245695)
            
            # Call to array(...): (line 17)
            # Processing the call arguments (line 17)
            
            # Obtaining an instance of the builtin type 'list' (line 17)
            list_245698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 28), 'list')
            # Adding type elements to the builtin type 'list' instance (line 17)
            # Adding element type (line 17)
            # Getting the type of 'x' (line 17)
            x_245699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 29), 'x', False)
            int_245700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 32), 'int')
            # Applying the binary operator '**' (line 17)
            result_pow_245701 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 29), '**', x_245699, int_245700)
            
            int_245702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 36), 'int')
            # Applying the binary operator '-' (line 17)
            result_sub_245703 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 29), '-', result_pow_245701, int_245702)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 28), list_245698, result_sub_245703)
            # Adding element type (line 17)
            # Getting the type of 'y' (line 17)
            y_245704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 39), 'y', False)
            int_245705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 42), 'int')
            # Applying the binary operator '**' (line 17)
            result_pow_245706 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 39), '**', y_245704, int_245705)
            
            int_245707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 46), 'int')
            # Applying the binary operator '-' (line 17)
            result_sub_245708 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 39), '-', result_pow_245706, int_245707)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 28), list_245698, result_sub_245708)
            
            # Processing the call keyword arguments (line 17)
            kwargs_245709 = {}
            # Getting the type of 'np' (line 17)
            np_245696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 19), 'np', False)
            # Obtaining the member 'array' of a type (line 17)
            array_245697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 19), np_245696, 'array')
            # Calling array(args, kwargs) (line 17)
            array_call_result_245710 = invoke(stypy.reporting.localization.Localization(__file__, 17, 19), array_245697, *[list_245698], **kwargs_245709)
            
            # Assigning a type to the variable 'stypy_return_type' (line 17)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'stypy_return_type', array_call_result_245710)
            
            # ################# End of 'func(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'func' in the type store
            # Getting the type of 'stypy_return_type' (line 15)
            stypy_return_type_245711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_245711)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'func'
            return stypy_return_type_245711

        # Assigning a type to the variable 'func' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'func', func)

        @norecursion
        def dfunc(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'dfunc'
            module_type_store = module_type_store.open_function_context('dfunc', 19, 8, False)
            
            # Passed parameters checking function
            dfunc.stypy_localization = localization
            dfunc.stypy_type_of_self = None
            dfunc.stypy_type_store = module_type_store
            dfunc.stypy_function_name = 'dfunc'
            dfunc.stypy_param_names_list = ['z']
            dfunc.stypy_varargs_param_name = None
            dfunc.stypy_kwargs_param_name = None
            dfunc.stypy_call_defaults = defaults
            dfunc.stypy_call_varargs = varargs
            dfunc.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'dfunc', ['z'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'dfunc', localization, ['z'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'dfunc(...)' code ##################

            
            # Assigning a Name to a Tuple (line 20):
            
            # Assigning a Subscript to a Name (line 20):
            
            # Obtaining the type of the subscript
            int_245712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 12), 'int')
            # Getting the type of 'z' (line 20)
            z_245713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 19), 'z')
            # Obtaining the member '__getitem__' of a type (line 20)
            getitem___245714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 12), z_245713, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 20)
            subscript_call_result_245715 = invoke(stypy.reporting.localization.Localization(__file__, 20, 12), getitem___245714, int_245712)
            
            # Assigning a type to the variable 'tuple_var_assignment_245675' (line 20)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'tuple_var_assignment_245675', subscript_call_result_245715)
            
            # Assigning a Subscript to a Name (line 20):
            
            # Obtaining the type of the subscript
            int_245716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 12), 'int')
            # Getting the type of 'z' (line 20)
            z_245717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 19), 'z')
            # Obtaining the member '__getitem__' of a type (line 20)
            getitem___245718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 12), z_245717, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 20)
            subscript_call_result_245719 = invoke(stypy.reporting.localization.Localization(__file__, 20, 12), getitem___245718, int_245716)
            
            # Assigning a type to the variable 'tuple_var_assignment_245676' (line 20)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'tuple_var_assignment_245676', subscript_call_result_245719)
            
            # Assigning a Name to a Name (line 20):
            # Getting the type of 'tuple_var_assignment_245675' (line 20)
            tuple_var_assignment_245675_245720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'tuple_var_assignment_245675')
            # Assigning a type to the variable 'x' (line 20)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'x', tuple_var_assignment_245675_245720)
            
            # Assigning a Name to a Name (line 20):
            # Getting the type of 'tuple_var_assignment_245676' (line 20)
            tuple_var_assignment_245676_245721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'tuple_var_assignment_245676')
            # Assigning a type to the variable 'y' (line 20)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 15), 'y', tuple_var_assignment_245676_245721)
            
            # Call to array(...): (line 21)
            # Processing the call arguments (line 21)
            
            # Obtaining an instance of the builtin type 'list' (line 21)
            list_245724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 28), 'list')
            # Adding type elements to the builtin type 'list' instance (line 21)
            # Adding element type (line 21)
            
            # Obtaining an instance of the builtin type 'list' (line 21)
            list_245725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 29), 'list')
            # Adding type elements to the builtin type 'list' instance (line 21)
            # Adding element type (line 21)
            int_245726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 30), 'int')
            # Getting the type of 'x' (line 21)
            x_245727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 32), 'x', False)
            int_245728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 35), 'int')
            # Applying the binary operator '**' (line 21)
            result_pow_245729 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 32), '**', x_245727, int_245728)
            
            # Applying the binary operator '*' (line 21)
            result_mul_245730 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 30), '*', int_245726, result_pow_245729)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 29), list_245725, result_mul_245730)
            # Adding element type (line 21)
            int_245731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 38), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 29), list_245725, int_245731)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 28), list_245724, list_245725)
            # Adding element type (line 21)
            
            # Obtaining an instance of the builtin type 'list' (line 21)
            list_245732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 42), 'list')
            # Adding type elements to the builtin type 'list' instance (line 21)
            # Adding element type (line 21)
            int_245733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 43), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 42), list_245732, int_245733)
            # Adding element type (line 21)
            int_245734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 46), 'int')
            # Getting the type of 'y' (line 21)
            y_245735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 48), 'y', False)
            int_245736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 51), 'int')
            # Applying the binary operator '**' (line 21)
            result_pow_245737 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 48), '**', y_245735, int_245736)
            
            # Applying the binary operator '*' (line 21)
            result_mul_245738 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 46), '*', int_245734, result_pow_245737)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 42), list_245732, result_mul_245738)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 28), list_245724, list_245732)
            
            # Processing the call keyword arguments (line 21)
            kwargs_245739 = {}
            # Getting the type of 'np' (line 21)
            np_245722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 19), 'np', False)
            # Obtaining the member 'array' of a type (line 21)
            array_245723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 19), np_245722, 'array')
            # Calling array(args, kwargs) (line 21)
            array_call_result_245740 = invoke(stypy.reporting.localization.Localization(__file__, 21, 19), array_245723, *[list_245724], **kwargs_245739)
            
            # Assigning a type to the variable 'stypy_return_type' (line 21)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'stypy_return_type', array_call_result_245740)
            
            # ################# End of 'dfunc(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'dfunc' in the type store
            # Getting the type of 'stypy_return_type' (line 19)
            stypy_return_type_245741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_245741)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'dfunc'
            return stypy_return_type_245741

        # Assigning a type to the variable 'dfunc' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'dfunc', dfunc)
        
        
        # Obtaining an instance of the builtin type 'list' (line 23)
        list_245742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 23)
        # Adding element type (line 23)
        str_245743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 23), 'str', 'hybr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 22), list_245742, str_245743)
        # Adding element type (line 23)
        str_245744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 31), 'str', 'lm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 22), list_245742, str_245744)
        # Adding element type (line 23)
        str_245745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 37), 'str', 'broyden1')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 22), list_245742, str_245745)
        # Adding element type (line 23)
        str_245746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 49), 'str', 'broyden2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 22), list_245742, str_245746)
        # Adding element type (line 23)
        str_245747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 61), 'str', 'anderson')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 22), list_245742, str_245747)
        # Adding element type (line 23)
        str_245748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 23), 'str', 'diagbroyden')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 22), list_245742, str_245748)
        # Adding element type (line 23)
        str_245749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 38), 'str', 'krylov')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 22), list_245742, str_245749)
        
        # Testing the type of a for loop iterable (line 23)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 23, 8), list_245742)
        # Getting the type of the for loop variable (line 23)
        for_loop_var_245750 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 23, 8), list_245742)
        # Assigning a type to the variable 'method' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'method', for_loop_var_245750)
        # SSA begins for a for statement (line 23)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'method' (line 25)
        method_245751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'method')
        
        # Obtaining an instance of the builtin type 'tuple' (line 25)
        tuple_245752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 25)
        # Adding element type (line 25)
        str_245753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 26), 'str', 'linearmixing')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 26), tuple_245752, str_245753)
        # Adding element type (line 25)
        str_245754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 42), 'str', 'excitingmixing')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 26), tuple_245752, str_245754)
        
        # Applying the binary operator 'in' (line 25)
        result_contains_245755 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 15), 'in', method_245751, tuple_245752)
        
        # Testing the type of an if condition (line 25)
        if_condition_245756 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 25, 12), result_contains_245755)
        # Assigning a type to the variable 'if_condition_245756' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'if_condition_245756', if_condition_245756)
        # SSA begins for if statement (line 25)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 25)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'method' (line 29)
        method_245757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 15), 'method')
        
        # Obtaining an instance of the builtin type 'tuple' (line 29)
        tuple_245758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 29)
        # Adding element type (line 29)
        str_245759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 26), 'str', 'hybr')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 26), tuple_245758, str_245759)
        # Adding element type (line 29)
        str_245760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 34), 'str', 'lm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 26), tuple_245758, str_245760)
        
        # Applying the binary operator 'in' (line 29)
        result_contains_245761 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 15), 'in', method_245757, tuple_245758)
        
        # Testing the type of an if condition (line 29)
        if_condition_245762 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 29, 12), result_contains_245761)
        # Assigning a type to the variable 'if_condition_245762' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'if_condition_245762', if_condition_245762)
        # SSA begins for if statement (line 29)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 30):
        
        # Assigning a Name to a Name (line 30):
        # Getting the type of 'dfunc' (line 30)
        dfunc_245763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 22), 'dfunc')
        # Assigning a type to the variable 'jac' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 16), 'jac', dfunc_245763)
        # SSA branch for the else part of an if statement (line 29)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Name to a Name (line 32):
        
        # Assigning a Name to a Name (line 32):
        # Getting the type of 'None' (line 32)
        None_245764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 22), 'None')
        # Assigning a type to the variable 'jac' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 16), 'jac', None_245764)
        # SSA join for if statement (line 29)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 34):
        
        # Assigning a Call to a Name (line 34):
        
        # Call to root(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'func' (line 34)
        func_245766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 24), 'func', False)
        
        # Obtaining an instance of the builtin type 'list' (line 34)
        list_245767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 34)
        # Adding element type (line 34)
        float_245768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 30), list_245767, float_245768)
        # Adding element type (line 34)
        float_245769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 30), list_245767, float_245769)
        
        # Processing the call keyword arguments (line 34)
        # Getting the type of 'jac' (line 34)
        jac_245770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 45), 'jac', False)
        keyword_245771 = jac_245770
        float_245772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 54), 'float')
        keyword_245773 = float_245772
        # Getting the type of 'method' (line 34)
        method_245774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 67), 'method', False)
        keyword_245775 = method_245774
        kwargs_245776 = {'jac': keyword_245771, 'tol': keyword_245773, 'method': keyword_245775}
        # Getting the type of 'root' (line 34)
        root_245765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 19), 'root', False)
        # Calling root(args, kwargs) (line 34)
        root_call_result_245777 = invoke(stypy.reporting.localization.Localization(__file__, 34, 19), root_245765, *[func_245766, list_245767], **kwargs_245776)
        
        # Assigning a type to the variable 'sol1' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'sol1', root_call_result_245777)
        
        # Assigning a Call to a Name (line 35):
        
        # Assigning a Call to a Name (line 35):
        
        # Call to root(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'func' (line 35)
        func_245779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 24), 'func', False)
        
        # Obtaining an instance of the builtin type 'list' (line 35)
        list_245780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 35)
        # Adding element type (line 35)
        float_245781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 31), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 30), list_245780, float_245781)
        # Adding element type (line 35)
        float_245782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 35), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 30), list_245780, float_245782)
        
        # Processing the call keyword arguments (line 35)
        # Getting the type of 'jac' (line 35)
        jac_245783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 45), 'jac', False)
        keyword_245784 = jac_245783
        float_245785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 54), 'float')
        keyword_245786 = float_245785
        # Getting the type of 'method' (line 35)
        method_245787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 66), 'method', False)
        keyword_245788 = method_245787
        kwargs_245789 = {'jac': keyword_245784, 'tol': keyword_245786, 'method': keyword_245788}
        # Getting the type of 'root' (line 35)
        root_245778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 19), 'root', False)
        # Calling root(args, kwargs) (line 35)
        root_call_result_245790 = invoke(stypy.reporting.localization.Localization(__file__, 35, 19), root_245778, *[func_245779, list_245780], **kwargs_245789)
        
        # Assigning a type to the variable 'sol2' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'sol2', root_call_result_245790)
        
        # Assigning a BinOp to a Name (line 36):
        
        # Assigning a BinOp to a Name (line 36):
        str_245791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 18), 'str', '%s: %s vs. %s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 36)
        tuple_245792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 36)
        # Adding element type (line 36)
        # Getting the type of 'method' (line 36)
        method_245793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 37), 'method')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 37), tuple_245792, method_245793)
        # Adding element type (line 36)
        
        # Call to func(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'sol1' (line 36)
        sol1_245795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 50), 'sol1', False)
        # Obtaining the member 'x' of a type (line 36)
        x_245796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 50), sol1_245795, 'x')
        # Processing the call keyword arguments (line 36)
        kwargs_245797 = {}
        # Getting the type of 'func' (line 36)
        func_245794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 45), 'func', False)
        # Calling func(args, kwargs) (line 36)
        func_call_result_245798 = invoke(stypy.reporting.localization.Localization(__file__, 36, 45), func_245794, *[x_245796], **kwargs_245797)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 37), tuple_245792, func_call_result_245798)
        # Adding element type (line 36)
        
        # Call to func(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'sol2' (line 36)
        sol2_245800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 64), 'sol2', False)
        # Obtaining the member 'x' of a type (line 36)
        x_245801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 64), sol2_245800, 'x')
        # Processing the call keyword arguments (line 36)
        kwargs_245802 = {}
        # Getting the type of 'func' (line 36)
        func_245799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 59), 'func', False)
        # Calling func(args, kwargs) (line 36)
        func_call_result_245803 = invoke(stypy.reporting.localization.Localization(__file__, 36, 59), func_245799, *[x_245801], **kwargs_245802)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 37), tuple_245792, func_call_result_245803)
        
        # Applying the binary operator '%' (line 36)
        result_mod_245804 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 18), '%', str_245791, tuple_245792)
        
        # Assigning a type to the variable 'msg' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'msg', result_mod_245804)
        
        # Call to assert_(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'sol1' (line 37)
        sol1_245806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 20), 'sol1', False)
        # Obtaining the member 'success' of a type (line 37)
        success_245807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 20), sol1_245806, 'success')
        # Getting the type of 'msg' (line 37)
        msg_245808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 34), 'msg', False)
        # Processing the call keyword arguments (line 37)
        kwargs_245809 = {}
        # Getting the type of 'assert_' (line 37)
        assert__245805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 37)
        assert__call_result_245810 = invoke(stypy.reporting.localization.Localization(__file__, 37, 12), assert__245805, *[success_245807, msg_245808], **kwargs_245809)
        
        
        # Call to assert_(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'sol2' (line 38)
        sol2_245812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 20), 'sol2', False)
        # Obtaining the member 'success' of a type (line 38)
        success_245813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 20), sol2_245812, 'success')
        # Getting the type of 'msg' (line 38)
        msg_245814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 34), 'msg', False)
        # Processing the call keyword arguments (line 38)
        kwargs_245815 = {}
        # Getting the type of 'assert_' (line 38)
        assert__245811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 38)
        assert__call_result_245816 = invoke(stypy.reporting.localization.Localization(__file__, 38, 12), assert__245811, *[success_245813, msg_245814], **kwargs_245815)
        
        
        # Call to assert_(...): (line 39)
        # Processing the call arguments (line 39)
        
        
        # Call to max(...): (line 39)
        # Processing the call keyword arguments (line 39)
        kwargs_245827 = {}
        
        # Call to abs(...): (line 39)
        # Processing the call arguments (line 39)
        
        # Call to func(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'sol1' (line 39)
        sol1_245820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 29), 'sol1', False)
        # Obtaining the member 'x' of a type (line 39)
        x_245821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 29), sol1_245820, 'x')
        # Processing the call keyword arguments (line 39)
        kwargs_245822 = {}
        # Getting the type of 'func' (line 39)
        func_245819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 24), 'func', False)
        # Calling func(args, kwargs) (line 39)
        func_call_result_245823 = invoke(stypy.reporting.localization.Localization(__file__, 39, 24), func_245819, *[x_245821], **kwargs_245822)
        
        # Processing the call keyword arguments (line 39)
        kwargs_245824 = {}
        # Getting the type of 'abs' (line 39)
        abs_245818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 20), 'abs', False)
        # Calling abs(args, kwargs) (line 39)
        abs_call_result_245825 = invoke(stypy.reporting.localization.Localization(__file__, 39, 20), abs_245818, *[func_call_result_245823], **kwargs_245824)
        
        # Obtaining the member 'max' of a type (line 39)
        max_245826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 20), abs_call_result_245825, 'max')
        # Calling max(args, kwargs) (line 39)
        max_call_result_245828 = invoke(stypy.reporting.localization.Localization(__file__, 39, 20), max_245826, *[], **kwargs_245827)
        
        
        # Call to max(...): (line 39)
        # Processing the call keyword arguments (line 39)
        kwargs_245838 = {}
        
        # Call to abs(...): (line 39)
        # Processing the call arguments (line 39)
        
        # Call to func(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'sol2' (line 39)
        sol2_245831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 55), 'sol2', False)
        # Obtaining the member 'x' of a type (line 39)
        x_245832 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 55), sol2_245831, 'x')
        # Processing the call keyword arguments (line 39)
        kwargs_245833 = {}
        # Getting the type of 'func' (line 39)
        func_245830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 50), 'func', False)
        # Calling func(args, kwargs) (line 39)
        func_call_result_245834 = invoke(stypy.reporting.localization.Localization(__file__, 39, 50), func_245830, *[x_245832], **kwargs_245833)
        
        # Processing the call keyword arguments (line 39)
        kwargs_245835 = {}
        # Getting the type of 'abs' (line 39)
        abs_245829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 46), 'abs', False)
        # Calling abs(args, kwargs) (line 39)
        abs_call_result_245836 = invoke(stypy.reporting.localization.Localization(__file__, 39, 46), abs_245829, *[func_call_result_245834], **kwargs_245835)
        
        # Obtaining the member 'max' of a type (line 39)
        max_245837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 46), abs_call_result_245836, 'max')
        # Calling max(args, kwargs) (line 39)
        max_call_result_245839 = invoke(stypy.reporting.localization.Localization(__file__, 39, 46), max_245837, *[], **kwargs_245838)
        
        # Applying the binary operator '<' (line 39)
        result_lt_245840 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 20), '<', max_call_result_245828, max_call_result_245839)
        
        # Getting the type of 'msg' (line 40)
        msg_245841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 20), 'msg', False)
        # Processing the call keyword arguments (line 39)
        kwargs_245842 = {}
        # Getting the type of 'assert_' (line 39)
        assert__245817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'assert_', False)
        # Calling assert_(args, kwargs) (line 39)
        assert__call_result_245843 = invoke(stypy.reporting.localization.Localization(__file__, 39, 12), assert__245817, *[result_lt_245840, msg_245841], **kwargs_245842)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_tol_parameter(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_tol_parameter' in the type store
        # Getting the type of 'stypy_return_type' (line 13)
        stypy_return_type_245844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_245844)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_tol_parameter'
        return stypy_return_type_245844


    @norecursion
    def test_minimize_scalar_coerce_args_param(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_minimize_scalar_coerce_args_param'
        module_type_store = module_type_store.open_function_context('test_minimize_scalar_coerce_args_param', 42, 4, False)
        # Assigning a type to the variable 'self' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestRoot.test_minimize_scalar_coerce_args_param.__dict__.__setitem__('stypy_localization', localization)
        TestRoot.test_minimize_scalar_coerce_args_param.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestRoot.test_minimize_scalar_coerce_args_param.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestRoot.test_minimize_scalar_coerce_args_param.__dict__.__setitem__('stypy_function_name', 'TestRoot.test_minimize_scalar_coerce_args_param')
        TestRoot.test_minimize_scalar_coerce_args_param.__dict__.__setitem__('stypy_param_names_list', [])
        TestRoot.test_minimize_scalar_coerce_args_param.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestRoot.test_minimize_scalar_coerce_args_param.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestRoot.test_minimize_scalar_coerce_args_param.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestRoot.test_minimize_scalar_coerce_args_param.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestRoot.test_minimize_scalar_coerce_args_param.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestRoot.test_minimize_scalar_coerce_args_param.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRoot.test_minimize_scalar_coerce_args_param', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_minimize_scalar_coerce_args_param', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_minimize_scalar_coerce_args_param(...)' code ##################


        @norecursion
        def func(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            int_245845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 22), 'int')
            defaults = [int_245845]
            # Create a new context for function 'func'
            module_type_store = module_type_store.open_function_context('func', 44, 8, False)
            
            # Passed parameters checking function
            func.stypy_localization = localization
            func.stypy_type_of_self = None
            func.stypy_type_store = module_type_store
            func.stypy_function_name = 'func'
            func.stypy_param_names_list = ['z', 'f']
            func.stypy_varargs_param_name = None
            func.stypy_kwargs_param_name = None
            func.stypy_call_defaults = defaults
            func.stypy_call_varargs = varargs
            func.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'func', ['z', 'f'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'func', localization, ['z', 'f'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'func(...)' code ##################

            
            # Assigning a Name to a Tuple (line 45):
            
            # Assigning a Subscript to a Name (line 45):
            
            # Obtaining the type of the subscript
            int_245846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 12), 'int')
            # Getting the type of 'z' (line 45)
            z_245847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 19), 'z')
            # Obtaining the member '__getitem__' of a type (line 45)
            getitem___245848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 12), z_245847, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 45)
            subscript_call_result_245849 = invoke(stypy.reporting.localization.Localization(__file__, 45, 12), getitem___245848, int_245846)
            
            # Assigning a type to the variable 'tuple_var_assignment_245677' (line 45)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'tuple_var_assignment_245677', subscript_call_result_245849)
            
            # Assigning a Subscript to a Name (line 45):
            
            # Obtaining the type of the subscript
            int_245850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 12), 'int')
            # Getting the type of 'z' (line 45)
            z_245851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 19), 'z')
            # Obtaining the member '__getitem__' of a type (line 45)
            getitem___245852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 12), z_245851, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 45)
            subscript_call_result_245853 = invoke(stypy.reporting.localization.Localization(__file__, 45, 12), getitem___245852, int_245850)
            
            # Assigning a type to the variable 'tuple_var_assignment_245678' (line 45)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'tuple_var_assignment_245678', subscript_call_result_245853)
            
            # Assigning a Name to a Name (line 45):
            # Getting the type of 'tuple_var_assignment_245677' (line 45)
            tuple_var_assignment_245677_245854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'tuple_var_assignment_245677')
            # Assigning a type to the variable 'x' (line 45)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'x', tuple_var_assignment_245677_245854)
            
            # Assigning a Name to a Name (line 45):
            # Getting the type of 'tuple_var_assignment_245678' (line 45)
            tuple_var_assignment_245678_245855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'tuple_var_assignment_245678')
            # Assigning a type to the variable 'y' (line 45)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'y', tuple_var_assignment_245678_245855)
            
            # Call to array(...): (line 46)
            # Processing the call arguments (line 46)
            
            # Obtaining an instance of the builtin type 'list' (line 46)
            list_245858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 28), 'list')
            # Adding type elements to the builtin type 'list' instance (line 46)
            # Adding element type (line 46)
            # Getting the type of 'x' (line 46)
            x_245859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 29), 'x', False)
            int_245860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 32), 'int')
            # Applying the binary operator '**' (line 46)
            result_pow_245861 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 29), '**', x_245859, int_245860)
            
            int_245862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 36), 'int')
            # Applying the binary operator '-' (line 46)
            result_sub_245863 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 29), '-', result_pow_245861, int_245862)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 28), list_245858, result_sub_245863)
            # Adding element type (line 46)
            # Getting the type of 'y' (line 46)
            y_245864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 39), 'y', False)
            int_245865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 42), 'int')
            # Applying the binary operator '**' (line 46)
            result_pow_245866 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 39), '**', y_245864, int_245865)
            
            # Getting the type of 'f' (line 46)
            f_245867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 46), 'f', False)
            # Applying the binary operator '-' (line 46)
            result_sub_245868 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 39), '-', result_pow_245866, f_245867)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 28), list_245858, result_sub_245868)
            
            # Processing the call keyword arguments (line 46)
            kwargs_245869 = {}
            # Getting the type of 'np' (line 46)
            np_245856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 19), 'np', False)
            # Obtaining the member 'array' of a type (line 46)
            array_245857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 19), np_245856, 'array')
            # Calling array(args, kwargs) (line 46)
            array_call_result_245870 = invoke(stypy.reporting.localization.Localization(__file__, 46, 19), array_245857, *[list_245858], **kwargs_245869)
            
            # Assigning a type to the variable 'stypy_return_type' (line 46)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'stypy_return_type', array_call_result_245870)
            
            # ################# End of 'func(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'func' in the type store
            # Getting the type of 'stypy_return_type' (line 44)
            stypy_return_type_245871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_245871)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'func'
            return stypy_return_type_245871

        # Assigning a type to the variable 'func' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'func', func)
        
        # Call to root(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'func' (line 47)
        func_245873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 13), 'func', False)
        
        # Obtaining an instance of the builtin type 'list' (line 47)
        list_245874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 47)
        # Adding element type (line 47)
        float_245875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 20), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 19), list_245874, float_245875)
        # Adding element type (line 47)
        float_245876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 25), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 19), list_245874, float_245876)
        
        # Processing the call keyword arguments (line 47)
        float_245877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 36), 'float')
        keyword_245878 = float_245877
        kwargs_245879 = {'args': keyword_245878}
        # Getting the type of 'root' (line 47)
        root_245872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'root', False)
        # Calling root(args, kwargs) (line 47)
        root_call_result_245880 = invoke(stypy.reporting.localization.Localization(__file__, 47, 8), root_245872, *[func_245873, list_245874], **kwargs_245879)
        
        
        # ################# End of 'test_minimize_scalar_coerce_args_param(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_minimize_scalar_coerce_args_param' in the type store
        # Getting the type of 'stypy_return_type' (line 42)
        stypy_return_type_245881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_245881)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_minimize_scalar_coerce_args_param'
        return stypy_return_type_245881


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 12, 0, False)
        # Assigning a type to the variable 'self' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestRoot.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestRoot' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'TestRoot', TestRoot)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
