
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Test reading of files not conforming to matlab specification
2: 
3: We try and read any file that matlab reads, these files included
4: '''
5: from __future__ import division, print_function, absolute_import
6: 
7: from os.path import dirname, join as pjoin
8: 
9: from numpy.testing import assert_
10: from pytest import raises as assert_raises
11: 
12: from scipy.io.matlab.mio import loadmat
13: 
14: TEST_DATA_PATH = pjoin(dirname(__file__), 'data')
15: 
16: 
17: def test_multiple_fieldnames():
18:     # Example provided by Dharhas Pothina
19:     # Extracted using mio5.varmats_from_mat
20:     multi_fname = pjoin(TEST_DATA_PATH, 'nasty_duplicate_fieldnames.mat')
21:     vars = loadmat(multi_fname)
22:     funny_names = vars['Summary'].dtype.names
23:     assert_(set(['_1_Station_Q', '_2_Station_Q',
24:                      '_3_Station_Q']).issubset(funny_names))
25: 
26: 
27: def test_malformed1():
28:     # Example from gh-6072
29:     # Contains malformed header data, which previously resulted into a
30:     # buffer overflow.
31:     #
32:     # Should raise an exception, not segfault
33:     fname = pjoin(TEST_DATA_PATH, 'malformed1.mat')
34:     with open(fname, 'rb') as f:
35:         assert_raises(ValueError, loadmat, f)
36: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_144578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', ' Test reading of files not conforming to matlab specification\n\nWe try and read any file that matlab reads, these files included\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from os.path import dirname, pjoin' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')
import_144579 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'os.path')

if (type(import_144579) is not StypyTypeError):

    if (import_144579 != 'pyd_module'):
        __import__(import_144579)
        sys_modules_144580 = sys.modules[import_144579]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'os.path', sys_modules_144580.module_type_store, module_type_store, ['dirname', 'join'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_144580, sys_modules_144580.module_type_store, module_type_store)
    else:
        from os.path import dirname, join as pjoin

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'os.path', None, module_type_store, ['dirname', 'join'], [dirname, pjoin])

else:
    # Assigning a type to the variable 'os.path' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'os.path', import_144579)

# Adding an alias
module_type_store.add_alias('pjoin', 'join')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from numpy.testing import assert_' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')
import_144581 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing')

if (type(import_144581) is not StypyTypeError):

    if (import_144581 != 'pyd_module'):
        __import__(import_144581)
        sys_modules_144582 = sys.modules[import_144581]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing', sys_modules_144582.module_type_store, module_type_store, ['assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_144582, sys_modules_144582.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing', None, module_type_store, ['assert_'], [assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing', import_144581)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from pytest import assert_raises' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')
import_144583 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'pytest')

if (type(import_144583) is not StypyTypeError):

    if (import_144583 != 'pyd_module'):
        __import__(import_144583)
        sys_modules_144584 = sys.modules[import_144583]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'pytest', sys_modules_144584.module_type_store, module_type_store, ['raises'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_144584, sys_modules_144584.module_type_store, module_type_store)
    else:
        from pytest import raises as assert_raises

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'pytest', None, module_type_store, ['raises'], [assert_raises])

else:
    # Assigning a type to the variable 'pytest' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'pytest', import_144583)

# Adding an alias
module_type_store.add_alias('assert_raises', 'raises')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from scipy.io.matlab.mio import loadmat' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')
import_144585 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.io.matlab.mio')

if (type(import_144585) is not StypyTypeError):

    if (import_144585 != 'pyd_module'):
        __import__(import_144585)
        sys_modules_144586 = sys.modules[import_144585]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.io.matlab.mio', sys_modules_144586.module_type_store, module_type_store, ['loadmat'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_144586, sys_modules_144586.module_type_store, module_type_store)
    else:
        from scipy.io.matlab.mio import loadmat

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.io.matlab.mio', None, module_type_store, ['loadmat'], [loadmat])

else:
    # Assigning a type to the variable 'scipy.io.matlab.mio' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'scipy.io.matlab.mio', import_144585)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/io/matlab/tests/')


# Assigning a Call to a Name (line 14):

# Call to pjoin(...): (line 14)
# Processing the call arguments (line 14)

# Call to dirname(...): (line 14)
# Processing the call arguments (line 14)
# Getting the type of '__file__' (line 14)
file___144589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 31), '__file__', False)
# Processing the call keyword arguments (line 14)
kwargs_144590 = {}
# Getting the type of 'dirname' (line 14)
dirname_144588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 23), 'dirname', False)
# Calling dirname(args, kwargs) (line 14)
dirname_call_result_144591 = invoke(stypy.reporting.localization.Localization(__file__, 14, 23), dirname_144588, *[file___144589], **kwargs_144590)

str_144592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 42), 'str', 'data')
# Processing the call keyword arguments (line 14)
kwargs_144593 = {}
# Getting the type of 'pjoin' (line 14)
pjoin_144587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 17), 'pjoin', False)
# Calling pjoin(args, kwargs) (line 14)
pjoin_call_result_144594 = invoke(stypy.reporting.localization.Localization(__file__, 14, 17), pjoin_144587, *[dirname_call_result_144591, str_144592], **kwargs_144593)

# Assigning a type to the variable 'TEST_DATA_PATH' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'TEST_DATA_PATH', pjoin_call_result_144594)

@norecursion
def test_multiple_fieldnames(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_multiple_fieldnames'
    module_type_store = module_type_store.open_function_context('test_multiple_fieldnames', 17, 0, False)
    
    # Passed parameters checking function
    test_multiple_fieldnames.stypy_localization = localization
    test_multiple_fieldnames.stypy_type_of_self = None
    test_multiple_fieldnames.stypy_type_store = module_type_store
    test_multiple_fieldnames.stypy_function_name = 'test_multiple_fieldnames'
    test_multiple_fieldnames.stypy_param_names_list = []
    test_multiple_fieldnames.stypy_varargs_param_name = None
    test_multiple_fieldnames.stypy_kwargs_param_name = None
    test_multiple_fieldnames.stypy_call_defaults = defaults
    test_multiple_fieldnames.stypy_call_varargs = varargs
    test_multiple_fieldnames.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_multiple_fieldnames', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_multiple_fieldnames', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_multiple_fieldnames(...)' code ##################

    
    # Assigning a Call to a Name (line 20):
    
    # Call to pjoin(...): (line 20)
    # Processing the call arguments (line 20)
    # Getting the type of 'TEST_DATA_PATH' (line 20)
    TEST_DATA_PATH_144596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 24), 'TEST_DATA_PATH', False)
    str_144597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 40), 'str', 'nasty_duplicate_fieldnames.mat')
    # Processing the call keyword arguments (line 20)
    kwargs_144598 = {}
    # Getting the type of 'pjoin' (line 20)
    pjoin_144595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 18), 'pjoin', False)
    # Calling pjoin(args, kwargs) (line 20)
    pjoin_call_result_144599 = invoke(stypy.reporting.localization.Localization(__file__, 20, 18), pjoin_144595, *[TEST_DATA_PATH_144596, str_144597], **kwargs_144598)
    
    # Assigning a type to the variable 'multi_fname' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'multi_fname', pjoin_call_result_144599)
    
    # Assigning a Call to a Name (line 21):
    
    # Call to loadmat(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 'multi_fname' (line 21)
    multi_fname_144601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 19), 'multi_fname', False)
    # Processing the call keyword arguments (line 21)
    kwargs_144602 = {}
    # Getting the type of 'loadmat' (line 21)
    loadmat_144600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 11), 'loadmat', False)
    # Calling loadmat(args, kwargs) (line 21)
    loadmat_call_result_144603 = invoke(stypy.reporting.localization.Localization(__file__, 21, 11), loadmat_144600, *[multi_fname_144601], **kwargs_144602)
    
    # Assigning a type to the variable 'vars' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'vars', loadmat_call_result_144603)
    
    # Assigning a Attribute to a Name (line 22):
    
    # Obtaining the type of the subscript
    str_144604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 23), 'str', 'Summary')
    # Getting the type of 'vars' (line 22)
    vars_144605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 18), 'vars')
    # Obtaining the member '__getitem__' of a type (line 22)
    getitem___144606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 18), vars_144605, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 22)
    subscript_call_result_144607 = invoke(stypy.reporting.localization.Localization(__file__, 22, 18), getitem___144606, str_144604)
    
    # Obtaining the member 'dtype' of a type (line 22)
    dtype_144608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 18), subscript_call_result_144607, 'dtype')
    # Obtaining the member 'names' of a type (line 22)
    names_144609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 18), dtype_144608, 'names')
    # Assigning a type to the variable 'funny_names' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'funny_names', names_144609)
    
    # Call to assert_(...): (line 23)
    # Processing the call arguments (line 23)
    
    # Call to issubset(...): (line 23)
    # Processing the call arguments (line 23)
    # Getting the type of 'funny_names' (line 24)
    funny_names_144619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 47), 'funny_names', False)
    # Processing the call keyword arguments (line 23)
    kwargs_144620 = {}
    
    # Call to set(...): (line 23)
    # Processing the call arguments (line 23)
    
    # Obtaining an instance of the builtin type 'list' (line 23)
    list_144612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 23)
    # Adding element type (line 23)
    str_144613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 17), 'str', '_1_Station_Q')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 16), list_144612, str_144613)
    # Adding element type (line 23)
    str_144614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 33), 'str', '_2_Station_Q')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 16), list_144612, str_144614)
    # Adding element type (line 23)
    str_144615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 21), 'str', '_3_Station_Q')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 16), list_144612, str_144615)
    
    # Processing the call keyword arguments (line 23)
    kwargs_144616 = {}
    # Getting the type of 'set' (line 23)
    set_144611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'set', False)
    # Calling set(args, kwargs) (line 23)
    set_call_result_144617 = invoke(stypy.reporting.localization.Localization(__file__, 23, 12), set_144611, *[list_144612], **kwargs_144616)
    
    # Obtaining the member 'issubset' of a type (line 23)
    issubset_144618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 12), set_call_result_144617, 'issubset')
    # Calling issubset(args, kwargs) (line 23)
    issubset_call_result_144621 = invoke(stypy.reporting.localization.Localization(__file__, 23, 12), issubset_144618, *[funny_names_144619], **kwargs_144620)
    
    # Processing the call keyword arguments (line 23)
    kwargs_144622 = {}
    # Getting the type of 'assert_' (line 23)
    assert__144610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 23)
    assert__call_result_144623 = invoke(stypy.reporting.localization.Localization(__file__, 23, 4), assert__144610, *[issubset_call_result_144621], **kwargs_144622)
    
    
    # ################# End of 'test_multiple_fieldnames(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_multiple_fieldnames' in the type store
    # Getting the type of 'stypy_return_type' (line 17)
    stypy_return_type_144624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_144624)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_multiple_fieldnames'
    return stypy_return_type_144624

# Assigning a type to the variable 'test_multiple_fieldnames' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'test_multiple_fieldnames', test_multiple_fieldnames)

@norecursion
def test_malformed1(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_malformed1'
    module_type_store = module_type_store.open_function_context('test_malformed1', 27, 0, False)
    
    # Passed parameters checking function
    test_malformed1.stypy_localization = localization
    test_malformed1.stypy_type_of_self = None
    test_malformed1.stypy_type_store = module_type_store
    test_malformed1.stypy_function_name = 'test_malformed1'
    test_malformed1.stypy_param_names_list = []
    test_malformed1.stypy_varargs_param_name = None
    test_malformed1.stypy_kwargs_param_name = None
    test_malformed1.stypy_call_defaults = defaults
    test_malformed1.stypy_call_varargs = varargs
    test_malformed1.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_malformed1', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_malformed1', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_malformed1(...)' code ##################

    
    # Assigning a Call to a Name (line 33):
    
    # Call to pjoin(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'TEST_DATA_PATH' (line 33)
    TEST_DATA_PATH_144626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 18), 'TEST_DATA_PATH', False)
    str_144627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 34), 'str', 'malformed1.mat')
    # Processing the call keyword arguments (line 33)
    kwargs_144628 = {}
    # Getting the type of 'pjoin' (line 33)
    pjoin_144625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'pjoin', False)
    # Calling pjoin(args, kwargs) (line 33)
    pjoin_call_result_144629 = invoke(stypy.reporting.localization.Localization(__file__, 33, 12), pjoin_144625, *[TEST_DATA_PATH_144626, str_144627], **kwargs_144628)
    
    # Assigning a type to the variable 'fname' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'fname', pjoin_call_result_144629)
    
    # Call to open(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'fname' (line 34)
    fname_144631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 14), 'fname', False)
    str_144632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 21), 'str', 'rb')
    # Processing the call keyword arguments (line 34)
    kwargs_144633 = {}
    # Getting the type of 'open' (line 34)
    open_144630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 9), 'open', False)
    # Calling open(args, kwargs) (line 34)
    open_call_result_144634 = invoke(stypy.reporting.localization.Localization(__file__, 34, 9), open_144630, *[fname_144631, str_144632], **kwargs_144633)
    
    with_144635 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 34, 9), open_call_result_144634, 'with parameter', '__enter__', '__exit__')

    if with_144635:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 34)
        enter___144636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 9), open_call_result_144634, '__enter__')
        with_enter_144637 = invoke(stypy.reporting.localization.Localization(__file__, 34, 9), enter___144636)
        # Assigning a type to the variable 'f' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 9), 'f', with_enter_144637)
        
        # Call to assert_raises(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'ValueError' (line 35)
        ValueError_144639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 22), 'ValueError', False)
        # Getting the type of 'loadmat' (line 35)
        loadmat_144640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 34), 'loadmat', False)
        # Getting the type of 'f' (line 35)
        f_144641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 43), 'f', False)
        # Processing the call keyword arguments (line 35)
        kwargs_144642 = {}
        # Getting the type of 'assert_raises' (line 35)
        assert_raises_144638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'assert_raises', False)
        # Calling assert_raises(args, kwargs) (line 35)
        assert_raises_call_result_144643 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), assert_raises_144638, *[ValueError_144639, loadmat_144640, f_144641], **kwargs_144642)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 34)
        exit___144644 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 9), open_call_result_144634, '__exit__')
        with_exit_144645 = invoke(stypy.reporting.localization.Localization(__file__, 34, 9), exit___144644, None, None, None)

    
    # ################# End of 'test_malformed1(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_malformed1' in the type store
    # Getting the type of 'stypy_return_type' (line 27)
    stypy_return_type_144646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_144646)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_malformed1'
    return stypy_return_type_144646

# Assigning a type to the variable 'test_malformed1' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'test_malformed1', test_malformed1)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
