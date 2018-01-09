
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: from numpy.testing import assert_equal, assert_allclose
4: from scipy._lib._numpy_compat import suppress_warnings
5: 
6: from scipy.misc import pade, logsumexp, face, ascent
7: from scipy.special import logsumexp as sc_logsumexp
8: 
9: 
10: def test_logsumexp():
11:     # make sure logsumexp can be imported from either scipy.misc or
12:     # scipy.special
13:     with suppress_warnings() as sup:
14:         sup.filter(DeprecationWarning, "`logsumexp` is deprecated")
15:         assert_allclose(logsumexp([0, 1]), sc_logsumexp([0, 1]), atol=1e-16)
16: 
17: 
18: def test_pade():
19:     # make sure scipy.misc.pade exists
20:     with suppress_warnings() as sup:
21:         sup.filter(DeprecationWarning, "`pade` is deprecated")
22:         pade([1, 2], 1)
23: 
24: 
25: def test_face():
26:     assert_equal(face().shape, (768, 1024, 3))
27: 
28: 
29: def test_ascent():
30:     assert_equal(ascent().shape, (512, 512))
31: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from numpy.testing import assert_equal, assert_allclose' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/misc/tests/')
import_115376 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing')

if (type(import_115376) is not StypyTypeError):

    if (import_115376 != 'pyd_module'):
        __import__(import_115376)
        sys_modules_115377 = sys.modules[import_115376]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing', sys_modules_115377.module_type_store, module_type_store, ['assert_equal', 'assert_allclose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_115377, sys_modules_115377.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_equal, assert_allclose

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing', None, module_type_store, ['assert_equal', 'assert_allclose'], [assert_equal, assert_allclose])

else:
    # Assigning a type to the variable 'numpy.testing' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.testing', import_115376)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/misc/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from scipy._lib._numpy_compat import suppress_warnings' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/misc/tests/')
import_115378 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy._lib._numpy_compat')

if (type(import_115378) is not StypyTypeError):

    if (import_115378 != 'pyd_module'):
        __import__(import_115378)
        sys_modules_115379 = sys.modules[import_115378]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy._lib._numpy_compat', sys_modules_115379.module_type_store, module_type_store, ['suppress_warnings'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_115379, sys_modules_115379.module_type_store, module_type_store)
    else:
        from scipy._lib._numpy_compat import suppress_warnings

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy._lib._numpy_compat', None, module_type_store, ['suppress_warnings'], [suppress_warnings])

else:
    # Assigning a type to the variable 'scipy._lib._numpy_compat' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'scipy._lib._numpy_compat', import_115378)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/misc/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from scipy.misc import pade, logsumexp, face, ascent' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/misc/tests/')
import_115380 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.misc')

if (type(import_115380) is not StypyTypeError):

    if (import_115380 != 'pyd_module'):
        __import__(import_115380)
        sys_modules_115381 = sys.modules[import_115380]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.misc', sys_modules_115381.module_type_store, module_type_store, ['pade', 'logsumexp', 'face', 'ascent'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_115381, sys_modules_115381.module_type_store, module_type_store)
    else:
        from scipy.misc import pade, logsumexp, face, ascent

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.misc', None, module_type_store, ['pade', 'logsumexp', 'face', 'ascent'], [pade, logsumexp, face, ascent])

else:
    # Assigning a type to the variable 'scipy.misc' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'scipy.misc', import_115380)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/misc/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy.special import sc_logsumexp' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/misc/tests/')
import_115382 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.special')

if (type(import_115382) is not StypyTypeError):

    if (import_115382 != 'pyd_module'):
        __import__(import_115382)
        sys_modules_115383 = sys.modules[import_115382]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.special', sys_modules_115383.module_type_store, module_type_store, ['logsumexp'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_115383, sys_modules_115383.module_type_store, module_type_store)
    else:
        from scipy.special import logsumexp as sc_logsumexp

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.special', None, module_type_store, ['logsumexp'], [sc_logsumexp])

else:
    # Assigning a type to the variable 'scipy.special' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy.special', import_115382)

# Adding an alias
module_type_store.add_alias('sc_logsumexp', 'logsumexp')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/misc/tests/')


@norecursion
def test_logsumexp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_logsumexp'
    module_type_store = module_type_store.open_function_context('test_logsumexp', 10, 0, False)
    
    # Passed parameters checking function
    test_logsumexp.stypy_localization = localization
    test_logsumexp.stypy_type_of_self = None
    test_logsumexp.stypy_type_store = module_type_store
    test_logsumexp.stypy_function_name = 'test_logsumexp'
    test_logsumexp.stypy_param_names_list = []
    test_logsumexp.stypy_varargs_param_name = None
    test_logsumexp.stypy_kwargs_param_name = None
    test_logsumexp.stypy_call_defaults = defaults
    test_logsumexp.stypy_call_varargs = varargs
    test_logsumexp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_logsumexp', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_logsumexp', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_logsumexp(...)' code ##################

    
    # Call to suppress_warnings(...): (line 13)
    # Processing the call keyword arguments (line 13)
    kwargs_115385 = {}
    # Getting the type of 'suppress_warnings' (line 13)
    suppress_warnings_115384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 9), 'suppress_warnings', False)
    # Calling suppress_warnings(args, kwargs) (line 13)
    suppress_warnings_call_result_115386 = invoke(stypy.reporting.localization.Localization(__file__, 13, 9), suppress_warnings_115384, *[], **kwargs_115385)
    
    with_115387 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 13, 9), suppress_warnings_call_result_115386, 'with parameter', '__enter__', '__exit__')

    if with_115387:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 13)
        enter___115388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 9), suppress_warnings_call_result_115386, '__enter__')
        with_enter_115389 = invoke(stypy.reporting.localization.Localization(__file__, 13, 9), enter___115388)
        # Assigning a type to the variable 'sup' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 9), 'sup', with_enter_115389)
        
        # Call to filter(...): (line 14)
        # Processing the call arguments (line 14)
        # Getting the type of 'DeprecationWarning' (line 14)
        DeprecationWarning_115392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 19), 'DeprecationWarning', False)
        str_115393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 39), 'str', '`logsumexp` is deprecated')
        # Processing the call keyword arguments (line 14)
        kwargs_115394 = {}
        # Getting the type of 'sup' (line 14)
        sup_115390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'sup', False)
        # Obtaining the member 'filter' of a type (line 14)
        filter_115391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 8), sup_115390, 'filter')
        # Calling filter(args, kwargs) (line 14)
        filter_call_result_115395 = invoke(stypy.reporting.localization.Localization(__file__, 14, 8), filter_115391, *[DeprecationWarning_115392, str_115393], **kwargs_115394)
        
        
        # Call to assert_allclose(...): (line 15)
        # Processing the call arguments (line 15)
        
        # Call to logsumexp(...): (line 15)
        # Processing the call arguments (line 15)
        
        # Obtaining an instance of the builtin type 'list' (line 15)
        list_115398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 15)
        # Adding element type (line 15)
        int_115399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 35), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 34), list_115398, int_115399)
        # Adding element type (line 15)
        int_115400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 38), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 34), list_115398, int_115400)
        
        # Processing the call keyword arguments (line 15)
        kwargs_115401 = {}
        # Getting the type of 'logsumexp' (line 15)
        logsumexp_115397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 24), 'logsumexp', False)
        # Calling logsumexp(args, kwargs) (line 15)
        logsumexp_call_result_115402 = invoke(stypy.reporting.localization.Localization(__file__, 15, 24), logsumexp_115397, *[list_115398], **kwargs_115401)
        
        
        # Call to sc_logsumexp(...): (line 15)
        # Processing the call arguments (line 15)
        
        # Obtaining an instance of the builtin type 'list' (line 15)
        list_115404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 56), 'list')
        # Adding type elements to the builtin type 'list' instance (line 15)
        # Adding element type (line 15)
        int_115405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 57), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 56), list_115404, int_115405)
        # Adding element type (line 15)
        int_115406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 60), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 56), list_115404, int_115406)
        
        # Processing the call keyword arguments (line 15)
        kwargs_115407 = {}
        # Getting the type of 'sc_logsumexp' (line 15)
        sc_logsumexp_115403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 43), 'sc_logsumexp', False)
        # Calling sc_logsumexp(args, kwargs) (line 15)
        sc_logsumexp_call_result_115408 = invoke(stypy.reporting.localization.Localization(__file__, 15, 43), sc_logsumexp_115403, *[list_115404], **kwargs_115407)
        
        # Processing the call keyword arguments (line 15)
        float_115409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 70), 'float')
        keyword_115410 = float_115409
        kwargs_115411 = {'atol': keyword_115410}
        # Getting the type of 'assert_allclose' (line 15)
        assert_allclose_115396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'assert_allclose', False)
        # Calling assert_allclose(args, kwargs) (line 15)
        assert_allclose_call_result_115412 = invoke(stypy.reporting.localization.Localization(__file__, 15, 8), assert_allclose_115396, *[logsumexp_call_result_115402, sc_logsumexp_call_result_115408], **kwargs_115411)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 13)
        exit___115413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 9), suppress_warnings_call_result_115386, '__exit__')
        with_exit_115414 = invoke(stypy.reporting.localization.Localization(__file__, 13, 9), exit___115413, None, None, None)

    
    # ################# End of 'test_logsumexp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_logsumexp' in the type store
    # Getting the type of 'stypy_return_type' (line 10)
    stypy_return_type_115415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_115415)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_logsumexp'
    return stypy_return_type_115415

# Assigning a type to the variable 'test_logsumexp' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'test_logsumexp', test_logsumexp)

@norecursion
def test_pade(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_pade'
    module_type_store = module_type_store.open_function_context('test_pade', 18, 0, False)
    
    # Passed parameters checking function
    test_pade.stypy_localization = localization
    test_pade.stypy_type_of_self = None
    test_pade.stypy_type_store = module_type_store
    test_pade.stypy_function_name = 'test_pade'
    test_pade.stypy_param_names_list = []
    test_pade.stypy_varargs_param_name = None
    test_pade.stypy_kwargs_param_name = None
    test_pade.stypy_call_defaults = defaults
    test_pade.stypy_call_varargs = varargs
    test_pade.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_pade', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_pade', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_pade(...)' code ##################

    
    # Call to suppress_warnings(...): (line 20)
    # Processing the call keyword arguments (line 20)
    kwargs_115417 = {}
    # Getting the type of 'suppress_warnings' (line 20)
    suppress_warnings_115416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 9), 'suppress_warnings', False)
    # Calling suppress_warnings(args, kwargs) (line 20)
    suppress_warnings_call_result_115418 = invoke(stypy.reporting.localization.Localization(__file__, 20, 9), suppress_warnings_115416, *[], **kwargs_115417)
    
    with_115419 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 20, 9), suppress_warnings_call_result_115418, 'with parameter', '__enter__', '__exit__')

    if with_115419:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 20)
        enter___115420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 9), suppress_warnings_call_result_115418, '__enter__')
        with_enter_115421 = invoke(stypy.reporting.localization.Localization(__file__, 20, 9), enter___115420)
        # Assigning a type to the variable 'sup' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 9), 'sup', with_enter_115421)
        
        # Call to filter(...): (line 21)
        # Processing the call arguments (line 21)
        # Getting the type of 'DeprecationWarning' (line 21)
        DeprecationWarning_115424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 19), 'DeprecationWarning', False)
        str_115425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 39), 'str', '`pade` is deprecated')
        # Processing the call keyword arguments (line 21)
        kwargs_115426 = {}
        # Getting the type of 'sup' (line 21)
        sup_115422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'sup', False)
        # Obtaining the member 'filter' of a type (line 21)
        filter_115423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 8), sup_115422, 'filter')
        # Calling filter(args, kwargs) (line 21)
        filter_call_result_115427 = invoke(stypy.reporting.localization.Localization(__file__, 21, 8), filter_115423, *[DeprecationWarning_115424, str_115425], **kwargs_115426)
        
        
        # Call to pade(...): (line 22)
        # Processing the call arguments (line 22)
        
        # Obtaining an instance of the builtin type 'list' (line 22)
        list_115429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 13), 'list')
        # Adding type elements to the builtin type 'list' instance (line 22)
        # Adding element type (line 22)
        int_115430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 14), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 13), list_115429, int_115430)
        # Adding element type (line 22)
        int_115431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 13), list_115429, int_115431)
        
        int_115432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 21), 'int')
        # Processing the call keyword arguments (line 22)
        kwargs_115433 = {}
        # Getting the type of 'pade' (line 22)
        pade_115428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'pade', False)
        # Calling pade(args, kwargs) (line 22)
        pade_call_result_115434 = invoke(stypy.reporting.localization.Localization(__file__, 22, 8), pade_115428, *[list_115429, int_115432], **kwargs_115433)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 20)
        exit___115435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 9), suppress_warnings_call_result_115418, '__exit__')
        with_exit_115436 = invoke(stypy.reporting.localization.Localization(__file__, 20, 9), exit___115435, None, None, None)

    
    # ################# End of 'test_pade(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_pade' in the type store
    # Getting the type of 'stypy_return_type' (line 18)
    stypy_return_type_115437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_115437)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_pade'
    return stypy_return_type_115437

# Assigning a type to the variable 'test_pade' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'test_pade', test_pade)

@norecursion
def test_face(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_face'
    module_type_store = module_type_store.open_function_context('test_face', 25, 0, False)
    
    # Passed parameters checking function
    test_face.stypy_localization = localization
    test_face.stypy_type_of_self = None
    test_face.stypy_type_store = module_type_store
    test_face.stypy_function_name = 'test_face'
    test_face.stypy_param_names_list = []
    test_face.stypy_varargs_param_name = None
    test_face.stypy_kwargs_param_name = None
    test_face.stypy_call_defaults = defaults
    test_face.stypy_call_varargs = varargs
    test_face.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_face', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_face', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_face(...)' code ##################

    
    # Call to assert_equal(...): (line 26)
    # Processing the call arguments (line 26)
    
    # Call to face(...): (line 26)
    # Processing the call keyword arguments (line 26)
    kwargs_115440 = {}
    # Getting the type of 'face' (line 26)
    face_115439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 17), 'face', False)
    # Calling face(args, kwargs) (line 26)
    face_call_result_115441 = invoke(stypy.reporting.localization.Localization(__file__, 26, 17), face_115439, *[], **kwargs_115440)
    
    # Obtaining the member 'shape' of a type (line 26)
    shape_115442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 17), face_call_result_115441, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 26)
    tuple_115443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 26)
    # Adding element type (line 26)
    int_115444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 32), tuple_115443, int_115444)
    # Adding element type (line 26)
    int_115445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 32), tuple_115443, int_115445)
    # Adding element type (line 26)
    int_115446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 32), tuple_115443, int_115446)
    
    # Processing the call keyword arguments (line 26)
    kwargs_115447 = {}
    # Getting the type of 'assert_equal' (line 26)
    assert_equal_115438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 26)
    assert_equal_call_result_115448 = invoke(stypy.reporting.localization.Localization(__file__, 26, 4), assert_equal_115438, *[shape_115442, tuple_115443], **kwargs_115447)
    
    
    # ################# End of 'test_face(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_face' in the type store
    # Getting the type of 'stypy_return_type' (line 25)
    stypy_return_type_115449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_115449)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_face'
    return stypy_return_type_115449

# Assigning a type to the variable 'test_face' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'test_face', test_face)

@norecursion
def test_ascent(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_ascent'
    module_type_store = module_type_store.open_function_context('test_ascent', 29, 0, False)
    
    # Passed parameters checking function
    test_ascent.stypy_localization = localization
    test_ascent.stypy_type_of_self = None
    test_ascent.stypy_type_store = module_type_store
    test_ascent.stypy_function_name = 'test_ascent'
    test_ascent.stypy_param_names_list = []
    test_ascent.stypy_varargs_param_name = None
    test_ascent.stypy_kwargs_param_name = None
    test_ascent.stypy_call_defaults = defaults
    test_ascent.stypy_call_varargs = varargs
    test_ascent.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_ascent', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_ascent', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_ascent(...)' code ##################

    
    # Call to assert_equal(...): (line 30)
    # Processing the call arguments (line 30)
    
    # Call to ascent(...): (line 30)
    # Processing the call keyword arguments (line 30)
    kwargs_115452 = {}
    # Getting the type of 'ascent' (line 30)
    ascent_115451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 17), 'ascent', False)
    # Calling ascent(args, kwargs) (line 30)
    ascent_call_result_115453 = invoke(stypy.reporting.localization.Localization(__file__, 30, 17), ascent_115451, *[], **kwargs_115452)
    
    # Obtaining the member 'shape' of a type (line 30)
    shape_115454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 17), ascent_call_result_115453, 'shape')
    
    # Obtaining an instance of the builtin type 'tuple' (line 30)
    tuple_115455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 30)
    # Adding element type (line 30)
    int_115456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 34), tuple_115455, int_115456)
    # Adding element type (line 30)
    int_115457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 39), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 34), tuple_115455, int_115457)
    
    # Processing the call keyword arguments (line 30)
    kwargs_115458 = {}
    # Getting the type of 'assert_equal' (line 30)
    assert_equal_115450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 30)
    assert_equal_call_result_115459 = invoke(stypy.reporting.localization.Localization(__file__, 30, 4), assert_equal_115450, *[shape_115454, tuple_115455], **kwargs_115458)
    
    
    # ################# End of 'test_ascent(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_ascent' in the type store
    # Getting the type of 'stypy_return_type' (line 29)
    stypy_return_type_115460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_115460)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_ascent'
    return stypy_return_type_115460

# Assigning a type to the variable 'test_ascent' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'test_ascent', test_ascent)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
