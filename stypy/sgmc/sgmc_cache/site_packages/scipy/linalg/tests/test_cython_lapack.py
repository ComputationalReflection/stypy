
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from numpy.testing import assert_allclose
2: from scipy.linalg import cython_lapack as cython_lapack
3: from scipy.linalg import lapack
4: 
5: 
6: class TestLamch(object):
7: 
8:     def test_slamch(self):
9:         for c in [b'e', b's', b'b', b'p', b'n', b'r', b'm', b'u', b'l', b'o']:
10:             assert_allclose(cython_lapack._test_slamch(c),
11:                             lapack.slamch(c))
12: 
13:     def test_dlamch(self):
14:         for c in [b'e', b's', b'b', b'p', b'n', b'r', b'm', b'u', b'l', b'o']:
15:             assert_allclose(cython_lapack._test_dlamch(c),
16:                             lapack.dlamch(c))
17: 
18: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from numpy.testing import assert_allclose' statement (line 1)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_54208 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy.testing')

if (type(import_54208) is not StypyTypeError):

    if (import_54208 != 'pyd_module'):
        __import__(import_54208)
        sys_modules_54209 = sys.modules[import_54208]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy.testing', sys_modules_54209.module_type_store, module_type_store, ['assert_allclose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_54209, sys_modules_54209.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_allclose

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy.testing', None, module_type_store, ['assert_allclose'], [assert_allclose])

else:
    # Assigning a type to the variable 'numpy.testing' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'numpy.testing', import_54208)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'from scipy.linalg import cython_lapack' statement (line 2)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_54210 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'scipy.linalg')

if (type(import_54210) is not StypyTypeError):

    if (import_54210 != 'pyd_module'):
        __import__(import_54210)
        sys_modules_54211 = sys.modules[import_54210]
        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'scipy.linalg', sys_modules_54211.module_type_store, module_type_store, ['cython_lapack'])
        nest_module(stypy.reporting.localization.Localization(__file__, 2, 0), __file__, sys_modules_54211, sys_modules_54211.module_type_store, module_type_store)
    else:
        from scipy.linalg import cython_lapack as cython_lapack

        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'scipy.linalg', None, module_type_store, ['cython_lapack'], [cython_lapack])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'scipy.linalg', import_54210)

# Adding an alias
module_type_store.add_alias('cython_lapack', 'cython_lapack')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from scipy.linalg import lapack' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_54212 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.linalg')

if (type(import_54212) is not StypyTypeError):

    if (import_54212 != 'pyd_module'):
        __import__(import_54212)
        sys_modules_54213 = sys.modules[import_54212]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.linalg', sys_modules_54213.module_type_store, module_type_store, ['lapack'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_54213, sys_modules_54213.module_type_store, module_type_store)
    else:
        from scipy.linalg import lapack

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.linalg', None, module_type_store, ['lapack'], [lapack])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.linalg', import_54212)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

# Declaration of the 'TestLamch' class

class TestLamch(object, ):

    @norecursion
    def test_slamch(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_slamch'
        module_type_store = module_type_store.open_function_context('test_slamch', 8, 4, False)
        # Assigning a type to the variable 'self' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLamch.test_slamch.__dict__.__setitem__('stypy_localization', localization)
        TestLamch.test_slamch.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLamch.test_slamch.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLamch.test_slamch.__dict__.__setitem__('stypy_function_name', 'TestLamch.test_slamch')
        TestLamch.test_slamch.__dict__.__setitem__('stypy_param_names_list', [])
        TestLamch.test_slamch.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLamch.test_slamch.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLamch.test_slamch.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLamch.test_slamch.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLamch.test_slamch.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLamch.test_slamch.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLamch.test_slamch', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_slamch', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_slamch(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 9)
        list_54214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 9)
        # Adding element type (line 9)
        str_54215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 18), 'str', 'e')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 17), list_54214, str_54215)
        # Adding element type (line 9)
        str_54216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 24), 'str', 's')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 17), list_54214, str_54216)
        # Adding element type (line 9)
        str_54217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 30), 'str', 'b')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 17), list_54214, str_54217)
        # Adding element type (line 9)
        str_54218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 36), 'str', 'p')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 17), list_54214, str_54218)
        # Adding element type (line 9)
        str_54219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 42), 'str', 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 17), list_54214, str_54219)
        # Adding element type (line 9)
        str_54220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 48), 'str', 'r')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 17), list_54214, str_54220)
        # Adding element type (line 9)
        str_54221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 54), 'str', 'm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 17), list_54214, str_54221)
        # Adding element type (line 9)
        str_54222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 60), 'str', 'u')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 17), list_54214, str_54222)
        # Adding element type (line 9)
        str_54223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 66), 'str', 'l')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 17), list_54214, str_54223)
        # Adding element type (line 9)
        str_54224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 72), 'str', 'o')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 9, 17), list_54214, str_54224)
        
        # Testing the type of a for loop iterable (line 9)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 9, 8), list_54214)
        # Getting the type of the for loop variable (line 9)
        for_loop_var_54225 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 9, 8), list_54214)
        # Assigning a type to the variable 'c' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'c', for_loop_var_54225)
        # SSA begins for a for statement (line 9)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_allclose(...): (line 10)
        # Processing the call arguments (line 10)
        
        # Call to _test_slamch(...): (line 10)
        # Processing the call arguments (line 10)
        # Getting the type of 'c' (line 10)
        c_54229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 55), 'c', False)
        # Processing the call keyword arguments (line 10)
        kwargs_54230 = {}
        # Getting the type of 'cython_lapack' (line 10)
        cython_lapack_54227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 28), 'cython_lapack', False)
        # Obtaining the member '_test_slamch' of a type (line 10)
        _test_slamch_54228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 28), cython_lapack_54227, '_test_slamch')
        # Calling _test_slamch(args, kwargs) (line 10)
        _test_slamch_call_result_54231 = invoke(stypy.reporting.localization.Localization(__file__, 10, 28), _test_slamch_54228, *[c_54229], **kwargs_54230)
        
        
        # Call to slamch(...): (line 11)
        # Processing the call arguments (line 11)
        # Getting the type of 'c' (line 11)
        c_54234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 42), 'c', False)
        # Processing the call keyword arguments (line 11)
        kwargs_54235 = {}
        # Getting the type of 'lapack' (line 11)
        lapack_54232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 28), 'lapack', False)
        # Obtaining the member 'slamch' of a type (line 11)
        slamch_54233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 28), lapack_54232, 'slamch')
        # Calling slamch(args, kwargs) (line 11)
        slamch_call_result_54236 = invoke(stypy.reporting.localization.Localization(__file__, 11, 28), slamch_54233, *[c_54234], **kwargs_54235)
        
        # Processing the call keyword arguments (line 10)
        kwargs_54237 = {}
        # Getting the type of 'assert_allclose' (line 10)
        assert_allclose_54226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 10)
        assert_allclose_call_result_54238 = invoke(stypy.reporting.localization.Localization(__file__, 10, 12), assert_allclose_54226, *[_test_slamch_call_result_54231, slamch_call_result_54236], **kwargs_54237)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_slamch(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_slamch' in the type store
        # Getting the type of 'stypy_return_type' (line 8)
        stypy_return_type_54239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_54239)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_slamch'
        return stypy_return_type_54239


    @norecursion
    def test_dlamch(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_dlamch'
        module_type_store = module_type_store.open_function_context('test_dlamch', 13, 4, False)
        # Assigning a type to the variable 'self' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestLamch.test_dlamch.__dict__.__setitem__('stypy_localization', localization)
        TestLamch.test_dlamch.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestLamch.test_dlamch.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestLamch.test_dlamch.__dict__.__setitem__('stypy_function_name', 'TestLamch.test_dlamch')
        TestLamch.test_dlamch.__dict__.__setitem__('stypy_param_names_list', [])
        TestLamch.test_dlamch.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestLamch.test_dlamch.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestLamch.test_dlamch.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestLamch.test_dlamch.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestLamch.test_dlamch.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestLamch.test_dlamch.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLamch.test_dlamch', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_dlamch', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_dlamch(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'list' (line 14)
        list_54240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 14)
        # Adding element type (line 14)
        str_54241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 18), 'str', 'e')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 17), list_54240, str_54241)
        # Adding element type (line 14)
        str_54242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 24), 'str', 's')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 17), list_54240, str_54242)
        # Adding element type (line 14)
        str_54243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 30), 'str', 'b')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 17), list_54240, str_54243)
        # Adding element type (line 14)
        str_54244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 36), 'str', 'p')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 17), list_54240, str_54244)
        # Adding element type (line 14)
        str_54245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 42), 'str', 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 17), list_54240, str_54245)
        # Adding element type (line 14)
        str_54246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 48), 'str', 'r')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 17), list_54240, str_54246)
        # Adding element type (line 14)
        str_54247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 54), 'str', 'm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 17), list_54240, str_54247)
        # Adding element type (line 14)
        str_54248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 60), 'str', 'u')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 17), list_54240, str_54248)
        # Adding element type (line 14)
        str_54249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 66), 'str', 'l')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 17), list_54240, str_54249)
        # Adding element type (line 14)
        str_54250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 72), 'str', 'o')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 17), list_54240, str_54250)
        
        # Testing the type of a for loop iterable (line 14)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 14, 8), list_54240)
        # Getting the type of the for loop variable (line 14)
        for_loop_var_54251 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 14, 8), list_54240)
        # Assigning a type to the variable 'c' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'c', for_loop_var_54251)
        # SSA begins for a for statement (line 14)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to assert_allclose(...): (line 15)
        # Processing the call arguments (line 15)
        
        # Call to _test_dlamch(...): (line 15)
        # Processing the call arguments (line 15)
        # Getting the type of 'c' (line 15)
        c_54255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 55), 'c', False)
        # Processing the call keyword arguments (line 15)
        kwargs_54256 = {}
        # Getting the type of 'cython_lapack' (line 15)
        cython_lapack_54253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 28), 'cython_lapack', False)
        # Obtaining the member '_test_dlamch' of a type (line 15)
        _test_dlamch_54254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 28), cython_lapack_54253, '_test_dlamch')
        # Calling _test_dlamch(args, kwargs) (line 15)
        _test_dlamch_call_result_54257 = invoke(stypy.reporting.localization.Localization(__file__, 15, 28), _test_dlamch_54254, *[c_54255], **kwargs_54256)
        
        
        # Call to dlamch(...): (line 16)
        # Processing the call arguments (line 16)
        # Getting the type of 'c' (line 16)
        c_54260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 42), 'c', False)
        # Processing the call keyword arguments (line 16)
        kwargs_54261 = {}
        # Getting the type of 'lapack' (line 16)
        lapack_54258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 28), 'lapack', False)
        # Obtaining the member 'dlamch' of a type (line 16)
        dlamch_54259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 28), lapack_54258, 'dlamch')
        # Calling dlamch(args, kwargs) (line 16)
        dlamch_call_result_54262 = invoke(stypy.reporting.localization.Localization(__file__, 16, 28), dlamch_54259, *[c_54260], **kwargs_54261)
        
        # Processing the call keyword arguments (line 15)
        kwargs_54263 = {}
        # Getting the type of 'assert_allclose' (line 15)
        assert_allclose_54252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 15)
        assert_allclose_call_result_54264 = invoke(stypy.reporting.localization.Localization(__file__, 15, 12), assert_allclose_54252, *[_test_dlamch_call_result_54257, dlamch_call_result_54262], **kwargs_54263)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'test_dlamch(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_dlamch' in the type store
        # Getting the type of 'stypy_return_type' (line 13)
        stypy_return_type_54265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_54265)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_dlamch'
        return stypy_return_type_54265


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 6, 0, False)
        # Assigning a type to the variable 'self' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestLamch.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestLamch' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'TestLamch', TestLamch)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
