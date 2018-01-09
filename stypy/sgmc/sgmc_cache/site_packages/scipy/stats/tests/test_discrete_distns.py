
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import 
2: 
3: from scipy.stats import hypergeom, bernoulli
4: import numpy as np
5: from numpy.testing import assert_almost_equal
6: 
7: def test_hypergeom_logpmf():
8:     # symmetries test
9:     # f(k,N,K,n) = f(n-k,N,N-K,n) = f(K-k,N,K,N-n) = f(k,N,n,K)
10:     k = 5
11:     N = 50
12:     K = 10
13:     n = 5
14:     logpmf1 = hypergeom.logpmf(k,N,K,n)
15:     logpmf2 = hypergeom.logpmf(n-k,N,N-K,n)
16:     logpmf3 = hypergeom.logpmf(K-k,N,K,N-n)
17:     logpmf4 = hypergeom.logpmf(k,N,n,K)
18:     assert_almost_equal(logpmf1, logpmf2, decimal=12)
19:     assert_almost_equal(logpmf1, logpmf3, decimal=12)
20:     assert_almost_equal(logpmf1, logpmf4, decimal=12)
21: 
22:     # test related distribution
23:     # Bernoulli distribution if n = 1
24:     k = 1
25:     N = 10
26:     K = 7
27:     n = 1
28:     hypergeom_logpmf = hypergeom.logpmf(k,N,K,n)
29:     bernoulli_logpmf = bernoulli.logpmf(k,K/N)
30:     assert_almost_equal(hypergeom_logpmf, bernoulli_logpmf, decimal=12)
31: 
32: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from scipy.stats import hypergeom, bernoulli' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_635614 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.stats')

if (type(import_635614) is not StypyTypeError):

    if (import_635614 != 'pyd_module'):
        __import__(import_635614)
        sys_modules_635615 = sys.modules[import_635614]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.stats', sys_modules_635615.module_type_store, module_type_store, ['hypergeom', 'bernoulli'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_635615, sys_modules_635615.module_type_store, module_type_store)
    else:
        from scipy.stats import hypergeom, bernoulli

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.stats', None, module_type_store, ['hypergeom', 'bernoulli'], [hypergeom, bernoulli])

else:
    # Assigning a type to the variable 'scipy.stats' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'scipy.stats', import_635614)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import numpy' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_635616 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy')

if (type(import_635616) is not StypyTypeError):

    if (import_635616 != 'pyd_module'):
        __import__(import_635616)
        sys_modules_635617 = sys.modules[import_635616]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', sys_modules_635617.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy', import_635616)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from numpy.testing import assert_almost_equal' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/stats/tests/')
import_635618 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing')

if (type(import_635618) is not StypyTypeError):

    if (import_635618 != 'pyd_module'):
        __import__(import_635618)
        sys_modules_635619 = sys.modules[import_635618]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', sys_modules_635619.module_type_store, module_type_store, ['assert_almost_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_635619, sys_modules_635619.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_almost_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', None, module_type_store, ['assert_almost_equal'], [assert_almost_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.testing', import_635618)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/stats/tests/')


@norecursion
def test_hypergeom_logpmf(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_hypergeom_logpmf'
    module_type_store = module_type_store.open_function_context('test_hypergeom_logpmf', 7, 0, False)
    
    # Passed parameters checking function
    test_hypergeom_logpmf.stypy_localization = localization
    test_hypergeom_logpmf.stypy_type_of_self = None
    test_hypergeom_logpmf.stypy_type_store = module_type_store
    test_hypergeom_logpmf.stypy_function_name = 'test_hypergeom_logpmf'
    test_hypergeom_logpmf.stypy_param_names_list = []
    test_hypergeom_logpmf.stypy_varargs_param_name = None
    test_hypergeom_logpmf.stypy_kwargs_param_name = None
    test_hypergeom_logpmf.stypy_call_defaults = defaults
    test_hypergeom_logpmf.stypy_call_varargs = varargs
    test_hypergeom_logpmf.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_hypergeom_logpmf', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_hypergeom_logpmf', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_hypergeom_logpmf(...)' code ##################

    
    # Assigning a Num to a Name (line 10):
    int_635620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 8), 'int')
    # Assigning a type to the variable 'k' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'k', int_635620)
    
    # Assigning a Num to a Name (line 11):
    int_635621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 8), 'int')
    # Assigning a type to the variable 'N' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'N', int_635621)
    
    # Assigning a Num to a Name (line 12):
    int_635622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 8), 'int')
    # Assigning a type to the variable 'K' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'K', int_635622)
    
    # Assigning a Num to a Name (line 13):
    int_635623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 8), 'int')
    # Assigning a type to the variable 'n' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'n', int_635623)
    
    # Assigning a Call to a Name (line 14):
    
    # Call to logpmf(...): (line 14)
    # Processing the call arguments (line 14)
    # Getting the type of 'k' (line 14)
    k_635626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 31), 'k', False)
    # Getting the type of 'N' (line 14)
    N_635627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 33), 'N', False)
    # Getting the type of 'K' (line 14)
    K_635628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 35), 'K', False)
    # Getting the type of 'n' (line 14)
    n_635629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 37), 'n', False)
    # Processing the call keyword arguments (line 14)
    kwargs_635630 = {}
    # Getting the type of 'hypergeom' (line 14)
    hypergeom_635624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 14), 'hypergeom', False)
    # Obtaining the member 'logpmf' of a type (line 14)
    logpmf_635625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 14), hypergeom_635624, 'logpmf')
    # Calling logpmf(args, kwargs) (line 14)
    logpmf_call_result_635631 = invoke(stypy.reporting.localization.Localization(__file__, 14, 14), logpmf_635625, *[k_635626, N_635627, K_635628, n_635629], **kwargs_635630)
    
    # Assigning a type to the variable 'logpmf1' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'logpmf1', logpmf_call_result_635631)
    
    # Assigning a Call to a Name (line 15):
    
    # Call to logpmf(...): (line 15)
    # Processing the call arguments (line 15)
    # Getting the type of 'n' (line 15)
    n_635634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 31), 'n', False)
    # Getting the type of 'k' (line 15)
    k_635635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 33), 'k', False)
    # Applying the binary operator '-' (line 15)
    result_sub_635636 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 31), '-', n_635634, k_635635)
    
    # Getting the type of 'N' (line 15)
    N_635637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 35), 'N', False)
    # Getting the type of 'N' (line 15)
    N_635638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 37), 'N', False)
    # Getting the type of 'K' (line 15)
    K_635639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 39), 'K', False)
    # Applying the binary operator '-' (line 15)
    result_sub_635640 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 37), '-', N_635638, K_635639)
    
    # Getting the type of 'n' (line 15)
    n_635641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 41), 'n', False)
    # Processing the call keyword arguments (line 15)
    kwargs_635642 = {}
    # Getting the type of 'hypergeom' (line 15)
    hypergeom_635632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 14), 'hypergeom', False)
    # Obtaining the member 'logpmf' of a type (line 15)
    logpmf_635633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 14), hypergeom_635632, 'logpmf')
    # Calling logpmf(args, kwargs) (line 15)
    logpmf_call_result_635643 = invoke(stypy.reporting.localization.Localization(__file__, 15, 14), logpmf_635633, *[result_sub_635636, N_635637, result_sub_635640, n_635641], **kwargs_635642)
    
    # Assigning a type to the variable 'logpmf2' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'logpmf2', logpmf_call_result_635643)
    
    # Assigning a Call to a Name (line 16):
    
    # Call to logpmf(...): (line 16)
    # Processing the call arguments (line 16)
    # Getting the type of 'K' (line 16)
    K_635646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 31), 'K', False)
    # Getting the type of 'k' (line 16)
    k_635647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 33), 'k', False)
    # Applying the binary operator '-' (line 16)
    result_sub_635648 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 31), '-', K_635646, k_635647)
    
    # Getting the type of 'N' (line 16)
    N_635649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 35), 'N', False)
    # Getting the type of 'K' (line 16)
    K_635650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 37), 'K', False)
    # Getting the type of 'N' (line 16)
    N_635651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 39), 'N', False)
    # Getting the type of 'n' (line 16)
    n_635652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 41), 'n', False)
    # Applying the binary operator '-' (line 16)
    result_sub_635653 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 39), '-', N_635651, n_635652)
    
    # Processing the call keyword arguments (line 16)
    kwargs_635654 = {}
    # Getting the type of 'hypergeom' (line 16)
    hypergeom_635644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 14), 'hypergeom', False)
    # Obtaining the member 'logpmf' of a type (line 16)
    logpmf_635645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 14), hypergeom_635644, 'logpmf')
    # Calling logpmf(args, kwargs) (line 16)
    logpmf_call_result_635655 = invoke(stypy.reporting.localization.Localization(__file__, 16, 14), logpmf_635645, *[result_sub_635648, N_635649, K_635650, result_sub_635653], **kwargs_635654)
    
    # Assigning a type to the variable 'logpmf3' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'logpmf3', logpmf_call_result_635655)
    
    # Assigning a Call to a Name (line 17):
    
    # Call to logpmf(...): (line 17)
    # Processing the call arguments (line 17)
    # Getting the type of 'k' (line 17)
    k_635658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 31), 'k', False)
    # Getting the type of 'N' (line 17)
    N_635659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 33), 'N', False)
    # Getting the type of 'n' (line 17)
    n_635660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 35), 'n', False)
    # Getting the type of 'K' (line 17)
    K_635661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 37), 'K', False)
    # Processing the call keyword arguments (line 17)
    kwargs_635662 = {}
    # Getting the type of 'hypergeom' (line 17)
    hypergeom_635656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 14), 'hypergeom', False)
    # Obtaining the member 'logpmf' of a type (line 17)
    logpmf_635657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 14), hypergeom_635656, 'logpmf')
    # Calling logpmf(args, kwargs) (line 17)
    logpmf_call_result_635663 = invoke(stypy.reporting.localization.Localization(__file__, 17, 14), logpmf_635657, *[k_635658, N_635659, n_635660, K_635661], **kwargs_635662)
    
    # Assigning a type to the variable 'logpmf4' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'logpmf4', logpmf_call_result_635663)
    
    # Call to assert_almost_equal(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'logpmf1' (line 18)
    logpmf1_635665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 24), 'logpmf1', False)
    # Getting the type of 'logpmf2' (line 18)
    logpmf2_635666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 33), 'logpmf2', False)
    # Processing the call keyword arguments (line 18)
    int_635667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 50), 'int')
    keyword_635668 = int_635667
    kwargs_635669 = {'decimal': keyword_635668}
    # Getting the type of 'assert_almost_equal' (line 18)
    assert_almost_equal_635664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 18)
    assert_almost_equal_call_result_635670 = invoke(stypy.reporting.localization.Localization(__file__, 18, 4), assert_almost_equal_635664, *[logpmf1_635665, logpmf2_635666], **kwargs_635669)
    
    
    # Call to assert_almost_equal(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'logpmf1' (line 19)
    logpmf1_635672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 24), 'logpmf1', False)
    # Getting the type of 'logpmf3' (line 19)
    logpmf3_635673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 33), 'logpmf3', False)
    # Processing the call keyword arguments (line 19)
    int_635674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 50), 'int')
    keyword_635675 = int_635674
    kwargs_635676 = {'decimal': keyword_635675}
    # Getting the type of 'assert_almost_equal' (line 19)
    assert_almost_equal_635671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 19)
    assert_almost_equal_call_result_635677 = invoke(stypy.reporting.localization.Localization(__file__, 19, 4), assert_almost_equal_635671, *[logpmf1_635672, logpmf3_635673], **kwargs_635676)
    
    
    # Call to assert_almost_equal(...): (line 20)
    # Processing the call arguments (line 20)
    # Getting the type of 'logpmf1' (line 20)
    logpmf1_635679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 24), 'logpmf1', False)
    # Getting the type of 'logpmf4' (line 20)
    logpmf4_635680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 33), 'logpmf4', False)
    # Processing the call keyword arguments (line 20)
    int_635681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 50), 'int')
    keyword_635682 = int_635681
    kwargs_635683 = {'decimal': keyword_635682}
    # Getting the type of 'assert_almost_equal' (line 20)
    assert_almost_equal_635678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 20)
    assert_almost_equal_call_result_635684 = invoke(stypy.reporting.localization.Localization(__file__, 20, 4), assert_almost_equal_635678, *[logpmf1_635679, logpmf4_635680], **kwargs_635683)
    
    
    # Assigning a Num to a Name (line 24):
    int_635685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 8), 'int')
    # Assigning a type to the variable 'k' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'k', int_635685)
    
    # Assigning a Num to a Name (line 25):
    int_635686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 8), 'int')
    # Assigning a type to the variable 'N' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'N', int_635686)
    
    # Assigning a Num to a Name (line 26):
    int_635687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 8), 'int')
    # Assigning a type to the variable 'K' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'K', int_635687)
    
    # Assigning a Num to a Name (line 27):
    int_635688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 8), 'int')
    # Assigning a type to the variable 'n' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'n', int_635688)
    
    # Assigning a Call to a Name (line 28):
    
    # Call to logpmf(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'k' (line 28)
    k_635691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 40), 'k', False)
    # Getting the type of 'N' (line 28)
    N_635692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 42), 'N', False)
    # Getting the type of 'K' (line 28)
    K_635693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 44), 'K', False)
    # Getting the type of 'n' (line 28)
    n_635694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 46), 'n', False)
    # Processing the call keyword arguments (line 28)
    kwargs_635695 = {}
    # Getting the type of 'hypergeom' (line 28)
    hypergeom_635689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 23), 'hypergeom', False)
    # Obtaining the member 'logpmf' of a type (line 28)
    logpmf_635690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 23), hypergeom_635689, 'logpmf')
    # Calling logpmf(args, kwargs) (line 28)
    logpmf_call_result_635696 = invoke(stypy.reporting.localization.Localization(__file__, 28, 23), logpmf_635690, *[k_635691, N_635692, K_635693, n_635694], **kwargs_635695)
    
    # Assigning a type to the variable 'hypergeom_logpmf' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'hypergeom_logpmf', logpmf_call_result_635696)
    
    # Assigning a Call to a Name (line 29):
    
    # Call to logpmf(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of 'k' (line 29)
    k_635699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 40), 'k', False)
    # Getting the type of 'K' (line 29)
    K_635700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 42), 'K', False)
    # Getting the type of 'N' (line 29)
    N_635701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 44), 'N', False)
    # Applying the binary operator 'div' (line 29)
    result_div_635702 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 42), 'div', K_635700, N_635701)
    
    # Processing the call keyword arguments (line 29)
    kwargs_635703 = {}
    # Getting the type of 'bernoulli' (line 29)
    bernoulli_635697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 23), 'bernoulli', False)
    # Obtaining the member 'logpmf' of a type (line 29)
    logpmf_635698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 23), bernoulli_635697, 'logpmf')
    # Calling logpmf(args, kwargs) (line 29)
    logpmf_call_result_635704 = invoke(stypy.reporting.localization.Localization(__file__, 29, 23), logpmf_635698, *[k_635699, result_div_635702], **kwargs_635703)
    
    # Assigning a type to the variable 'bernoulli_logpmf' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'bernoulli_logpmf', logpmf_call_result_635704)
    
    # Call to assert_almost_equal(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'hypergeom_logpmf' (line 30)
    hypergeom_logpmf_635706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 24), 'hypergeom_logpmf', False)
    # Getting the type of 'bernoulli_logpmf' (line 30)
    bernoulli_logpmf_635707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 42), 'bernoulli_logpmf', False)
    # Processing the call keyword arguments (line 30)
    int_635708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 68), 'int')
    keyword_635709 = int_635708
    kwargs_635710 = {'decimal': keyword_635709}
    # Getting the type of 'assert_almost_equal' (line 30)
    assert_almost_equal_635705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'assert_almost_equal', False)
    # Calling assert_almost_equal(args, kwargs) (line 30)
    assert_almost_equal_call_result_635711 = invoke(stypy.reporting.localization.Localization(__file__, 30, 4), assert_almost_equal_635705, *[hypergeom_logpmf_635706, bernoulli_logpmf_635707], **kwargs_635710)
    
    
    # ################# End of 'test_hypergeom_logpmf(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_hypergeom_logpmf' in the type store
    # Getting the type of 'stypy_return_type' (line 7)
    stypy_return_type_635712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_635712)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_hypergeom_logpmf'
    return stypy_return_type_635712

# Assigning a type to the variable 'test_hypergeom_logpmf' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'test_hypergeom_logpmf', test_hypergeom_logpmf)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
