
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Test tmpdirs module '''
2: from __future__ import division, print_function, absolute_import
3: 
4: from os import getcwd
5: from os.path import realpath, abspath, dirname, isfile, join as pjoin, exists
6: 
7: from scipy._lib._tmpdirs import tempdir, in_tempdir, in_dir
8: 
9: from numpy.testing import assert_, assert_equal
10: 
11: MY_PATH = abspath(__file__)
12: MY_DIR = dirname(MY_PATH)
13: 
14: 
15: def test_tempdir():
16:     with tempdir() as tmpdir:
17:         fname = pjoin(tmpdir, 'example_file.txt')
18:         with open(fname, 'wt') as fobj:
19:             fobj.write('a string\\n')
20:     assert_(not exists(tmpdir))
21: 
22: 
23: def test_in_tempdir():
24:     my_cwd = getcwd()
25:     with in_tempdir() as tmpdir:
26:         with open('test.txt', 'wt') as f:
27:             f.write('some text')
28:         assert_(isfile('test.txt'))
29:         assert_(isfile(pjoin(tmpdir, 'test.txt')))
30:     assert_(not exists(tmpdir))
31:     assert_equal(getcwd(), my_cwd)
32: 
33: 
34: def test_given_directory():
35:     # Test InGivenDirectory
36:     cwd = getcwd()
37:     with in_dir() as tmpdir:
38:         assert_equal(tmpdir, abspath(cwd))
39:         assert_equal(tmpdir, abspath(getcwd()))
40:     with in_dir(MY_DIR) as tmpdir:
41:         assert_equal(tmpdir, MY_DIR)
42:         assert_equal(realpath(MY_DIR), realpath(abspath(getcwd())))
43:     # We were deleting the given directory!  Check not so now.
44:     assert_(isfile(MY_PATH))
45: 
46: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_711669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', ' Test tmpdirs module ')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from os import getcwd' statement (line 4)
try:
    from os import getcwd

except:
    getcwd = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'os', None, module_type_store, ['getcwd'], [getcwd])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from os.path import realpath, abspath, dirname, isfile, pjoin, exists' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/tests/')
import_711670 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'os.path')

if (type(import_711670) is not StypyTypeError):

    if (import_711670 != 'pyd_module'):
        __import__(import_711670)
        sys_modules_711671 = sys.modules[import_711670]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'os.path', sys_modules_711671.module_type_store, module_type_store, ['realpath', 'abspath', 'dirname', 'isfile', 'join', 'exists'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_711671, sys_modules_711671.module_type_store, module_type_store)
    else:
        from os.path import realpath, abspath, dirname, isfile, join as pjoin, exists

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'os.path', None, module_type_store, ['realpath', 'abspath', 'dirname', 'isfile', 'join', 'exists'], [realpath, abspath, dirname, isfile, pjoin, exists])

else:
    # Assigning a type to the variable 'os.path' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'os.path', import_711670)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from scipy._lib._tmpdirs import tempdir, in_tempdir, in_dir' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/tests/')
import_711672 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib._tmpdirs')

if (type(import_711672) is not StypyTypeError):

    if (import_711672 != 'pyd_module'):
        __import__(import_711672)
        sys_modules_711673 = sys.modules[import_711672]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib._tmpdirs', sys_modules_711673.module_type_store, module_type_store, ['tempdir', 'in_tempdir', 'in_dir'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_711673, sys_modules_711673.module_type_store, module_type_store)
    else:
        from scipy._lib._tmpdirs import tempdir, in_tempdir, in_dir

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib._tmpdirs', None, module_type_store, ['tempdir', 'in_tempdir', 'in_dir'], [tempdir, in_tempdir, in_dir])

else:
    # Assigning a type to the variable 'scipy._lib._tmpdirs' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'scipy._lib._tmpdirs', import_711672)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from numpy.testing import assert_, assert_equal' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/tests/')
import_711674 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing')

if (type(import_711674) is not StypyTypeError):

    if (import_711674 != 'pyd_module'):
        __import__(import_711674)
        sys_modules_711675 = sys.modules[import_711674]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing', sys_modules_711675.module_type_store, module_type_store, ['assert_', 'assert_equal'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_711675, sys_modules_711675.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_, assert_equal

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing', None, module_type_store, ['assert_', 'assert_equal'], [assert_, assert_equal])

else:
    # Assigning a type to the variable 'numpy.testing' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.testing', import_711674)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/tests/')


# Assigning a Call to a Name (line 11):

# Call to abspath(...): (line 11)
# Processing the call arguments (line 11)
# Getting the type of '__file__' (line 11)
file___711677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 18), '__file__', False)
# Processing the call keyword arguments (line 11)
kwargs_711678 = {}
# Getting the type of 'abspath' (line 11)
abspath_711676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 10), 'abspath', False)
# Calling abspath(args, kwargs) (line 11)
abspath_call_result_711679 = invoke(stypy.reporting.localization.Localization(__file__, 11, 10), abspath_711676, *[file___711677], **kwargs_711678)

# Assigning a type to the variable 'MY_PATH' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'MY_PATH', abspath_call_result_711679)

# Assigning a Call to a Name (line 12):

# Call to dirname(...): (line 12)
# Processing the call arguments (line 12)
# Getting the type of 'MY_PATH' (line 12)
MY_PATH_711681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 17), 'MY_PATH', False)
# Processing the call keyword arguments (line 12)
kwargs_711682 = {}
# Getting the type of 'dirname' (line 12)
dirname_711680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 9), 'dirname', False)
# Calling dirname(args, kwargs) (line 12)
dirname_call_result_711683 = invoke(stypy.reporting.localization.Localization(__file__, 12, 9), dirname_711680, *[MY_PATH_711681], **kwargs_711682)

# Assigning a type to the variable 'MY_DIR' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'MY_DIR', dirname_call_result_711683)

@norecursion
def test_tempdir(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_tempdir'
    module_type_store = module_type_store.open_function_context('test_tempdir', 15, 0, False)
    
    # Passed parameters checking function
    test_tempdir.stypy_localization = localization
    test_tempdir.stypy_type_of_self = None
    test_tempdir.stypy_type_store = module_type_store
    test_tempdir.stypy_function_name = 'test_tempdir'
    test_tempdir.stypy_param_names_list = []
    test_tempdir.stypy_varargs_param_name = None
    test_tempdir.stypy_kwargs_param_name = None
    test_tempdir.stypy_call_defaults = defaults
    test_tempdir.stypy_call_varargs = varargs
    test_tempdir.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_tempdir', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_tempdir', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_tempdir(...)' code ##################

    
    # Call to tempdir(...): (line 16)
    # Processing the call keyword arguments (line 16)
    kwargs_711685 = {}
    # Getting the type of 'tempdir' (line 16)
    tempdir_711684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 9), 'tempdir', False)
    # Calling tempdir(args, kwargs) (line 16)
    tempdir_call_result_711686 = invoke(stypy.reporting.localization.Localization(__file__, 16, 9), tempdir_711684, *[], **kwargs_711685)
    
    with_711687 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 16, 9), tempdir_call_result_711686, 'with parameter', '__enter__', '__exit__')

    if with_711687:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 16)
        enter___711688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 9), tempdir_call_result_711686, '__enter__')
        with_enter_711689 = invoke(stypy.reporting.localization.Localization(__file__, 16, 9), enter___711688)
        # Assigning a type to the variable 'tmpdir' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 9), 'tmpdir', with_enter_711689)
        
        # Assigning a Call to a Name (line 17):
        
        # Call to pjoin(...): (line 17)
        # Processing the call arguments (line 17)
        # Getting the type of 'tmpdir' (line 17)
        tmpdir_711691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 22), 'tmpdir', False)
        str_711692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 30), 'str', 'example_file.txt')
        # Processing the call keyword arguments (line 17)
        kwargs_711693 = {}
        # Getting the type of 'pjoin' (line 17)
        pjoin_711690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 16), 'pjoin', False)
        # Calling pjoin(args, kwargs) (line 17)
        pjoin_call_result_711694 = invoke(stypy.reporting.localization.Localization(__file__, 17, 16), pjoin_711690, *[tmpdir_711691, str_711692], **kwargs_711693)
        
        # Assigning a type to the variable 'fname' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'fname', pjoin_call_result_711694)
        
        # Call to open(...): (line 18)
        # Processing the call arguments (line 18)
        # Getting the type of 'fname' (line 18)
        fname_711696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 18), 'fname', False)
        str_711697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 25), 'str', 'wt')
        # Processing the call keyword arguments (line 18)
        kwargs_711698 = {}
        # Getting the type of 'open' (line 18)
        open_711695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 13), 'open', False)
        # Calling open(args, kwargs) (line 18)
        open_call_result_711699 = invoke(stypy.reporting.localization.Localization(__file__, 18, 13), open_711695, *[fname_711696, str_711697], **kwargs_711698)
        
        with_711700 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 18, 13), open_call_result_711699, 'with parameter', '__enter__', '__exit__')

        if with_711700:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 18)
            enter___711701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 13), open_call_result_711699, '__enter__')
            with_enter_711702 = invoke(stypy.reporting.localization.Localization(__file__, 18, 13), enter___711701)
            # Assigning a type to the variable 'fobj' (line 18)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 13), 'fobj', with_enter_711702)
            
            # Call to write(...): (line 19)
            # Processing the call arguments (line 19)
            str_711705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 23), 'str', 'a string\\n')
            # Processing the call keyword arguments (line 19)
            kwargs_711706 = {}
            # Getting the type of 'fobj' (line 19)
            fobj_711703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'fobj', False)
            # Obtaining the member 'write' of a type (line 19)
            write_711704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 12), fobj_711703, 'write')
            # Calling write(args, kwargs) (line 19)
            write_call_result_711707 = invoke(stypy.reporting.localization.Localization(__file__, 19, 12), write_711704, *[str_711705], **kwargs_711706)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 18)
            exit___711708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 13), open_call_result_711699, '__exit__')
            with_exit_711709 = invoke(stypy.reporting.localization.Localization(__file__, 18, 13), exit___711708, None, None, None)

        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 16)
        exit___711710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 9), tempdir_call_result_711686, '__exit__')
        with_exit_711711 = invoke(stypy.reporting.localization.Localization(__file__, 16, 9), exit___711710, None, None, None)

    
    # Call to assert_(...): (line 20)
    # Processing the call arguments (line 20)
    
    
    # Call to exists(...): (line 20)
    # Processing the call arguments (line 20)
    # Getting the type of 'tmpdir' (line 20)
    tmpdir_711714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 23), 'tmpdir', False)
    # Processing the call keyword arguments (line 20)
    kwargs_711715 = {}
    # Getting the type of 'exists' (line 20)
    exists_711713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'exists', False)
    # Calling exists(args, kwargs) (line 20)
    exists_call_result_711716 = invoke(stypy.reporting.localization.Localization(__file__, 20, 16), exists_711713, *[tmpdir_711714], **kwargs_711715)
    
    # Applying the 'not' unary operator (line 20)
    result_not__711717 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 12), 'not', exists_call_result_711716)
    
    # Processing the call keyword arguments (line 20)
    kwargs_711718 = {}
    # Getting the type of 'assert_' (line 20)
    assert__711712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 20)
    assert__call_result_711719 = invoke(stypy.reporting.localization.Localization(__file__, 20, 4), assert__711712, *[result_not__711717], **kwargs_711718)
    
    
    # ################# End of 'test_tempdir(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_tempdir' in the type store
    # Getting the type of 'stypy_return_type' (line 15)
    stypy_return_type_711720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_711720)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_tempdir'
    return stypy_return_type_711720

# Assigning a type to the variable 'test_tempdir' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'test_tempdir', test_tempdir)

@norecursion
def test_in_tempdir(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_in_tempdir'
    module_type_store = module_type_store.open_function_context('test_in_tempdir', 23, 0, False)
    
    # Passed parameters checking function
    test_in_tempdir.stypy_localization = localization
    test_in_tempdir.stypy_type_of_self = None
    test_in_tempdir.stypy_type_store = module_type_store
    test_in_tempdir.stypy_function_name = 'test_in_tempdir'
    test_in_tempdir.stypy_param_names_list = []
    test_in_tempdir.stypy_varargs_param_name = None
    test_in_tempdir.stypy_kwargs_param_name = None
    test_in_tempdir.stypy_call_defaults = defaults
    test_in_tempdir.stypy_call_varargs = varargs
    test_in_tempdir.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_in_tempdir', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_in_tempdir', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_in_tempdir(...)' code ##################

    
    # Assigning a Call to a Name (line 24):
    
    # Call to getcwd(...): (line 24)
    # Processing the call keyword arguments (line 24)
    kwargs_711722 = {}
    # Getting the type of 'getcwd' (line 24)
    getcwd_711721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 13), 'getcwd', False)
    # Calling getcwd(args, kwargs) (line 24)
    getcwd_call_result_711723 = invoke(stypy.reporting.localization.Localization(__file__, 24, 13), getcwd_711721, *[], **kwargs_711722)
    
    # Assigning a type to the variable 'my_cwd' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'my_cwd', getcwd_call_result_711723)
    
    # Call to in_tempdir(...): (line 25)
    # Processing the call keyword arguments (line 25)
    kwargs_711725 = {}
    # Getting the type of 'in_tempdir' (line 25)
    in_tempdir_711724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 9), 'in_tempdir', False)
    # Calling in_tempdir(args, kwargs) (line 25)
    in_tempdir_call_result_711726 = invoke(stypy.reporting.localization.Localization(__file__, 25, 9), in_tempdir_711724, *[], **kwargs_711725)
    
    with_711727 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 25, 9), in_tempdir_call_result_711726, 'with parameter', '__enter__', '__exit__')

    if with_711727:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 25)
        enter___711728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 9), in_tempdir_call_result_711726, '__enter__')
        with_enter_711729 = invoke(stypy.reporting.localization.Localization(__file__, 25, 9), enter___711728)
        # Assigning a type to the variable 'tmpdir' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 9), 'tmpdir', with_enter_711729)
        
        # Call to open(...): (line 26)
        # Processing the call arguments (line 26)
        str_711731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 18), 'str', 'test.txt')
        str_711732 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 30), 'str', 'wt')
        # Processing the call keyword arguments (line 26)
        kwargs_711733 = {}
        # Getting the type of 'open' (line 26)
        open_711730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 13), 'open', False)
        # Calling open(args, kwargs) (line 26)
        open_call_result_711734 = invoke(stypy.reporting.localization.Localization(__file__, 26, 13), open_711730, *[str_711731, str_711732], **kwargs_711733)
        
        with_711735 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 26, 13), open_call_result_711734, 'with parameter', '__enter__', '__exit__')

        if with_711735:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 26)
            enter___711736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 13), open_call_result_711734, '__enter__')
            with_enter_711737 = invoke(stypy.reporting.localization.Localization(__file__, 26, 13), enter___711736)
            # Assigning a type to the variable 'f' (line 26)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 13), 'f', with_enter_711737)
            
            # Call to write(...): (line 27)
            # Processing the call arguments (line 27)
            str_711740 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 20), 'str', 'some text')
            # Processing the call keyword arguments (line 27)
            kwargs_711741 = {}
            # Getting the type of 'f' (line 27)
            f_711738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'f', False)
            # Obtaining the member 'write' of a type (line 27)
            write_711739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 12), f_711738, 'write')
            # Calling write(args, kwargs) (line 27)
            write_call_result_711742 = invoke(stypy.reporting.localization.Localization(__file__, 27, 12), write_711739, *[str_711740], **kwargs_711741)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 26)
            exit___711743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 13), open_call_result_711734, '__exit__')
            with_exit_711744 = invoke(stypy.reporting.localization.Localization(__file__, 26, 13), exit___711743, None, None, None)

        
        # Call to assert_(...): (line 28)
        # Processing the call arguments (line 28)
        
        # Call to isfile(...): (line 28)
        # Processing the call arguments (line 28)
        str_711747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 23), 'str', 'test.txt')
        # Processing the call keyword arguments (line 28)
        kwargs_711748 = {}
        # Getting the type of 'isfile' (line 28)
        isfile_711746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 16), 'isfile', False)
        # Calling isfile(args, kwargs) (line 28)
        isfile_call_result_711749 = invoke(stypy.reporting.localization.Localization(__file__, 28, 16), isfile_711746, *[str_711747], **kwargs_711748)
        
        # Processing the call keyword arguments (line 28)
        kwargs_711750 = {}
        # Getting the type of 'assert_' (line 28)
        assert__711745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 28)
        assert__call_result_711751 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), assert__711745, *[isfile_call_result_711749], **kwargs_711750)
        
        
        # Call to assert_(...): (line 29)
        # Processing the call arguments (line 29)
        
        # Call to isfile(...): (line 29)
        # Processing the call arguments (line 29)
        
        # Call to pjoin(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'tmpdir' (line 29)
        tmpdir_711755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 29), 'tmpdir', False)
        str_711756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 37), 'str', 'test.txt')
        # Processing the call keyword arguments (line 29)
        kwargs_711757 = {}
        # Getting the type of 'pjoin' (line 29)
        pjoin_711754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 23), 'pjoin', False)
        # Calling pjoin(args, kwargs) (line 29)
        pjoin_call_result_711758 = invoke(stypy.reporting.localization.Localization(__file__, 29, 23), pjoin_711754, *[tmpdir_711755, str_711756], **kwargs_711757)
        
        # Processing the call keyword arguments (line 29)
        kwargs_711759 = {}
        # Getting the type of 'isfile' (line 29)
        isfile_711753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 16), 'isfile', False)
        # Calling isfile(args, kwargs) (line 29)
        isfile_call_result_711760 = invoke(stypy.reporting.localization.Localization(__file__, 29, 16), isfile_711753, *[pjoin_call_result_711758], **kwargs_711759)
        
        # Processing the call keyword arguments (line 29)
        kwargs_711761 = {}
        # Getting the type of 'assert_' (line 29)
        assert__711752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 29)
        assert__call_result_711762 = invoke(stypy.reporting.localization.Localization(__file__, 29, 8), assert__711752, *[isfile_call_result_711760], **kwargs_711761)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 25)
        exit___711763 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 9), in_tempdir_call_result_711726, '__exit__')
        with_exit_711764 = invoke(stypy.reporting.localization.Localization(__file__, 25, 9), exit___711763, None, None, None)

    
    # Call to assert_(...): (line 30)
    # Processing the call arguments (line 30)
    
    
    # Call to exists(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'tmpdir' (line 30)
    tmpdir_711767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 23), 'tmpdir', False)
    # Processing the call keyword arguments (line 30)
    kwargs_711768 = {}
    # Getting the type of 'exists' (line 30)
    exists_711766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 16), 'exists', False)
    # Calling exists(args, kwargs) (line 30)
    exists_call_result_711769 = invoke(stypy.reporting.localization.Localization(__file__, 30, 16), exists_711766, *[tmpdir_711767], **kwargs_711768)
    
    # Applying the 'not' unary operator (line 30)
    result_not__711770 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 12), 'not', exists_call_result_711769)
    
    # Processing the call keyword arguments (line 30)
    kwargs_711771 = {}
    # Getting the type of 'assert_' (line 30)
    assert__711765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 30)
    assert__call_result_711772 = invoke(stypy.reporting.localization.Localization(__file__, 30, 4), assert__711765, *[result_not__711770], **kwargs_711771)
    
    
    # Call to assert_equal(...): (line 31)
    # Processing the call arguments (line 31)
    
    # Call to getcwd(...): (line 31)
    # Processing the call keyword arguments (line 31)
    kwargs_711775 = {}
    # Getting the type of 'getcwd' (line 31)
    getcwd_711774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 17), 'getcwd', False)
    # Calling getcwd(args, kwargs) (line 31)
    getcwd_call_result_711776 = invoke(stypy.reporting.localization.Localization(__file__, 31, 17), getcwd_711774, *[], **kwargs_711775)
    
    # Getting the type of 'my_cwd' (line 31)
    my_cwd_711777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 27), 'my_cwd', False)
    # Processing the call keyword arguments (line 31)
    kwargs_711778 = {}
    # Getting the type of 'assert_equal' (line 31)
    assert_equal_711773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'assert_equal', False)
    # Calling assert_equal(args, kwargs) (line 31)
    assert_equal_call_result_711779 = invoke(stypy.reporting.localization.Localization(__file__, 31, 4), assert_equal_711773, *[getcwd_call_result_711776, my_cwd_711777], **kwargs_711778)
    
    
    # ################# End of 'test_in_tempdir(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_in_tempdir' in the type store
    # Getting the type of 'stypy_return_type' (line 23)
    stypy_return_type_711780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_711780)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_in_tempdir'
    return stypy_return_type_711780

# Assigning a type to the variable 'test_in_tempdir' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'test_in_tempdir', test_in_tempdir)

@norecursion
def test_given_directory(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'test_given_directory'
    module_type_store = module_type_store.open_function_context('test_given_directory', 34, 0, False)
    
    # Passed parameters checking function
    test_given_directory.stypy_localization = localization
    test_given_directory.stypy_type_of_self = None
    test_given_directory.stypy_type_store = module_type_store
    test_given_directory.stypy_function_name = 'test_given_directory'
    test_given_directory.stypy_param_names_list = []
    test_given_directory.stypy_varargs_param_name = None
    test_given_directory.stypy_kwargs_param_name = None
    test_given_directory.stypy_call_defaults = defaults
    test_given_directory.stypy_call_varargs = varargs
    test_given_directory.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'test_given_directory', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'test_given_directory', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'test_given_directory(...)' code ##################

    
    # Assigning a Call to a Name (line 36):
    
    # Call to getcwd(...): (line 36)
    # Processing the call keyword arguments (line 36)
    kwargs_711782 = {}
    # Getting the type of 'getcwd' (line 36)
    getcwd_711781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 10), 'getcwd', False)
    # Calling getcwd(args, kwargs) (line 36)
    getcwd_call_result_711783 = invoke(stypy.reporting.localization.Localization(__file__, 36, 10), getcwd_711781, *[], **kwargs_711782)
    
    # Assigning a type to the variable 'cwd' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'cwd', getcwd_call_result_711783)
    
    # Call to in_dir(...): (line 37)
    # Processing the call keyword arguments (line 37)
    kwargs_711785 = {}
    # Getting the type of 'in_dir' (line 37)
    in_dir_711784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 9), 'in_dir', False)
    # Calling in_dir(args, kwargs) (line 37)
    in_dir_call_result_711786 = invoke(stypy.reporting.localization.Localization(__file__, 37, 9), in_dir_711784, *[], **kwargs_711785)
    
    with_711787 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 37, 9), in_dir_call_result_711786, 'with parameter', '__enter__', '__exit__')

    if with_711787:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 37)
        enter___711788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 9), in_dir_call_result_711786, '__enter__')
        with_enter_711789 = invoke(stypy.reporting.localization.Localization(__file__, 37, 9), enter___711788)
        # Assigning a type to the variable 'tmpdir' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 9), 'tmpdir', with_enter_711789)
        
        # Call to assert_equal(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'tmpdir' (line 38)
        tmpdir_711791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 21), 'tmpdir', False)
        
        # Call to abspath(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'cwd' (line 38)
        cwd_711793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 37), 'cwd', False)
        # Processing the call keyword arguments (line 38)
        kwargs_711794 = {}
        # Getting the type of 'abspath' (line 38)
        abspath_711792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 29), 'abspath', False)
        # Calling abspath(args, kwargs) (line 38)
        abspath_call_result_711795 = invoke(stypy.reporting.localization.Localization(__file__, 38, 29), abspath_711792, *[cwd_711793], **kwargs_711794)
        
        # Processing the call keyword arguments (line 38)
        kwargs_711796 = {}
        # Getting the type of 'assert_equal' (line 38)
        assert_equal_711790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 38)
        assert_equal_call_result_711797 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), assert_equal_711790, *[tmpdir_711791, abspath_call_result_711795], **kwargs_711796)
        
        
        # Call to assert_equal(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'tmpdir' (line 39)
        tmpdir_711799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 21), 'tmpdir', False)
        
        # Call to abspath(...): (line 39)
        # Processing the call arguments (line 39)
        
        # Call to getcwd(...): (line 39)
        # Processing the call keyword arguments (line 39)
        kwargs_711802 = {}
        # Getting the type of 'getcwd' (line 39)
        getcwd_711801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 37), 'getcwd', False)
        # Calling getcwd(args, kwargs) (line 39)
        getcwd_call_result_711803 = invoke(stypy.reporting.localization.Localization(__file__, 39, 37), getcwd_711801, *[], **kwargs_711802)
        
        # Processing the call keyword arguments (line 39)
        kwargs_711804 = {}
        # Getting the type of 'abspath' (line 39)
        abspath_711800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 29), 'abspath', False)
        # Calling abspath(args, kwargs) (line 39)
        abspath_call_result_711805 = invoke(stypy.reporting.localization.Localization(__file__, 39, 29), abspath_711800, *[getcwd_call_result_711803], **kwargs_711804)
        
        # Processing the call keyword arguments (line 39)
        kwargs_711806 = {}
        # Getting the type of 'assert_equal' (line 39)
        assert_equal_711798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 39)
        assert_equal_call_result_711807 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), assert_equal_711798, *[tmpdir_711799, abspath_call_result_711805], **kwargs_711806)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 37)
        exit___711808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 9), in_dir_call_result_711786, '__exit__')
        with_exit_711809 = invoke(stypy.reporting.localization.Localization(__file__, 37, 9), exit___711808, None, None, None)

    
    # Call to in_dir(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'MY_DIR' (line 40)
    MY_DIR_711811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 16), 'MY_DIR', False)
    # Processing the call keyword arguments (line 40)
    kwargs_711812 = {}
    # Getting the type of 'in_dir' (line 40)
    in_dir_711810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 9), 'in_dir', False)
    # Calling in_dir(args, kwargs) (line 40)
    in_dir_call_result_711813 = invoke(stypy.reporting.localization.Localization(__file__, 40, 9), in_dir_711810, *[MY_DIR_711811], **kwargs_711812)
    
    with_711814 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 40, 9), in_dir_call_result_711813, 'with parameter', '__enter__', '__exit__')

    if with_711814:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 40)
        enter___711815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 9), in_dir_call_result_711813, '__enter__')
        with_enter_711816 = invoke(stypy.reporting.localization.Localization(__file__, 40, 9), enter___711815)
        # Assigning a type to the variable 'tmpdir' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 9), 'tmpdir', with_enter_711816)
        
        # Call to assert_equal(...): (line 41)
        # Processing the call arguments (line 41)
        # Getting the type of 'tmpdir' (line 41)
        tmpdir_711818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 21), 'tmpdir', False)
        # Getting the type of 'MY_DIR' (line 41)
        MY_DIR_711819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 29), 'MY_DIR', False)
        # Processing the call keyword arguments (line 41)
        kwargs_711820 = {}
        # Getting the type of 'assert_equal' (line 41)
        assert_equal_711817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 41)
        assert_equal_call_result_711821 = invoke(stypy.reporting.localization.Localization(__file__, 41, 8), assert_equal_711817, *[tmpdir_711818, MY_DIR_711819], **kwargs_711820)
        
        
        # Call to assert_equal(...): (line 42)
        # Processing the call arguments (line 42)
        
        # Call to realpath(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'MY_DIR' (line 42)
        MY_DIR_711824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 30), 'MY_DIR', False)
        # Processing the call keyword arguments (line 42)
        kwargs_711825 = {}
        # Getting the type of 'realpath' (line 42)
        realpath_711823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 21), 'realpath', False)
        # Calling realpath(args, kwargs) (line 42)
        realpath_call_result_711826 = invoke(stypy.reporting.localization.Localization(__file__, 42, 21), realpath_711823, *[MY_DIR_711824], **kwargs_711825)
        
        
        # Call to realpath(...): (line 42)
        # Processing the call arguments (line 42)
        
        # Call to abspath(...): (line 42)
        # Processing the call arguments (line 42)
        
        # Call to getcwd(...): (line 42)
        # Processing the call keyword arguments (line 42)
        kwargs_711830 = {}
        # Getting the type of 'getcwd' (line 42)
        getcwd_711829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 56), 'getcwd', False)
        # Calling getcwd(args, kwargs) (line 42)
        getcwd_call_result_711831 = invoke(stypy.reporting.localization.Localization(__file__, 42, 56), getcwd_711829, *[], **kwargs_711830)
        
        # Processing the call keyword arguments (line 42)
        kwargs_711832 = {}
        # Getting the type of 'abspath' (line 42)
        abspath_711828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 48), 'abspath', False)
        # Calling abspath(args, kwargs) (line 42)
        abspath_call_result_711833 = invoke(stypy.reporting.localization.Localization(__file__, 42, 48), abspath_711828, *[getcwd_call_result_711831], **kwargs_711832)
        
        # Processing the call keyword arguments (line 42)
        kwargs_711834 = {}
        # Getting the type of 'realpath' (line 42)
        realpath_711827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 39), 'realpath', False)
        # Calling realpath(args, kwargs) (line 42)
        realpath_call_result_711835 = invoke(stypy.reporting.localization.Localization(__file__, 42, 39), realpath_711827, *[abspath_call_result_711833], **kwargs_711834)
        
        # Processing the call keyword arguments (line 42)
        kwargs_711836 = {}
        # Getting the type of 'assert_equal' (line 42)
        assert_equal_711822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'assert_equal', False)
        # Calling assert_equal(args, kwargs) (line 42)
        assert_equal_call_result_711837 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), assert_equal_711822, *[realpath_call_result_711826, realpath_call_result_711835], **kwargs_711836)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 40)
        exit___711838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 9), in_dir_call_result_711813, '__exit__')
        with_exit_711839 = invoke(stypy.reporting.localization.Localization(__file__, 40, 9), exit___711838, None, None, None)

    
    # Call to assert_(...): (line 44)
    # Processing the call arguments (line 44)
    
    # Call to isfile(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'MY_PATH' (line 44)
    MY_PATH_711842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 19), 'MY_PATH', False)
    # Processing the call keyword arguments (line 44)
    kwargs_711843 = {}
    # Getting the type of 'isfile' (line 44)
    isfile_711841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'isfile', False)
    # Calling isfile(args, kwargs) (line 44)
    isfile_call_result_711844 = invoke(stypy.reporting.localization.Localization(__file__, 44, 12), isfile_711841, *[MY_PATH_711842], **kwargs_711843)
    
    # Processing the call keyword arguments (line 44)
    kwargs_711845 = {}
    # Getting the type of 'assert_' (line 44)
    assert__711840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'assert_', False)
    # Calling assert_(args, kwargs) (line 44)
    assert__call_result_711846 = invoke(stypy.reporting.localization.Localization(__file__, 44, 4), assert__711840, *[isfile_call_result_711844], **kwargs_711845)
    
    
    # ################# End of 'test_given_directory(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'test_given_directory' in the type store
    # Getting the type of 'stypy_return_type' (line 34)
    stypy_return_type_711847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_711847)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'test_given_directory'
    return stypy_return_type_711847

# Assigning a type to the variable 'test_given_directory' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'test_given_directory', test_given_directory)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
