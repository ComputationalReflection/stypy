
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Contexts for *with* statement providing temporary directories
2: '''
3: from __future__ import division, print_function, absolute_import
4: import os
5: from contextlib import contextmanager
6: from shutil import rmtree
7: from tempfile import mkdtemp
8: 
9: 
10: @contextmanager
11: def tempdir():
12:     '''Create and return a temporary directory.  This has the same
13:     behavior as mkdtemp but can be used as a context manager.
14: 
15:     Upon exiting the context, the directory and everthing contained
16:     in it are removed.
17: 
18:     Examples
19:     --------
20:     >>> import os
21:     >>> with tempdir() as tmpdir:
22:     ...     fname = os.path.join(tmpdir, 'example_file.txt')
23:     ...     with open(fname, 'wt') as fobj:
24:     ...         _ = fobj.write('a string\\n')
25:     >>> os.path.exists(tmpdir)
26:     False
27:     '''
28:     d = mkdtemp()
29:     yield d
30:     rmtree(d)
31: 
32: 
33: @contextmanager
34: def in_tempdir():
35:     ''' Create, return, and change directory to a temporary directory
36: 
37:     Examples
38:     --------
39:     >>> import os
40:     >>> my_cwd = os.getcwd()
41:     >>> with in_tempdir() as tmpdir:
42:     ...     _ = open('test.txt', 'wt').write('some text')
43:     ...     assert os.path.isfile('test.txt')
44:     ...     assert os.path.isfile(os.path.join(tmpdir, 'test.txt'))
45:     >>> os.path.exists(tmpdir)
46:     False
47:     >>> os.getcwd() == my_cwd
48:     True
49:     '''
50:     pwd = os.getcwd()
51:     d = mkdtemp()
52:     os.chdir(d)
53:     yield d
54:     os.chdir(pwd)
55:     rmtree(d)
56: 
57: 
58: @contextmanager
59: def in_dir(dir=None):
60:     ''' Change directory to given directory for duration of ``with`` block
61: 
62:     Useful when you want to use `in_tempdir` for the final test, but
63:     you are still debugging.  For example, you may want to do this in the end:
64: 
65:     >>> with in_tempdir() as tmpdir:
66:     ...     # do something complicated which might break
67:     ...     pass
68: 
69:     But indeed the complicated thing does break, and meanwhile the
70:     ``in_tempdir`` context manager wiped out the directory with the
71:     temporary files that you wanted for debugging.  So, while debugging, you
72:     replace with something like:
73: 
74:     >>> with in_dir() as tmpdir: # Use working directory by default
75:     ...     # do something complicated which might break
76:     ...     pass
77: 
78:     You can then look at the temporary file outputs to debug what is happening,
79:     fix, and finally replace ``in_dir`` with ``in_tempdir`` again.
80:     '''
81:     cwd = os.getcwd()
82:     if dir is None:
83:         yield cwd
84:         return
85:     os.chdir(dir)
86:     yield dir
87:     os.chdir(cwd)
88: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_710013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, (-1)), 'str', ' Contexts for *with* statement providing temporary directories\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import os' statement (line 4)
import os

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from contextlib import contextmanager' statement (line 5)
try:
    from contextlib import contextmanager

except:
    contextmanager = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'contextlib', None, module_type_store, ['contextmanager'], [contextmanager])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from shutil import rmtree' statement (line 6)
try:
    from shutil import rmtree

except:
    rmtree = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'shutil', None, module_type_store, ['rmtree'], [rmtree])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from tempfile import mkdtemp' statement (line 7)
try:
    from tempfile import mkdtemp

except:
    mkdtemp = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'tempfile', None, module_type_store, ['mkdtemp'], [mkdtemp])


@norecursion
def tempdir(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'tempdir'
    module_type_store = module_type_store.open_function_context('tempdir', 10, 0, False)
    
    # Passed parameters checking function
    tempdir.stypy_localization = localization
    tempdir.stypy_type_of_self = None
    tempdir.stypy_type_store = module_type_store
    tempdir.stypy_function_name = 'tempdir'
    tempdir.stypy_param_names_list = []
    tempdir.stypy_varargs_param_name = None
    tempdir.stypy_kwargs_param_name = None
    tempdir.stypy_call_defaults = defaults
    tempdir.stypy_call_varargs = varargs
    tempdir.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'tempdir', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'tempdir', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'tempdir(...)' code ##################

    str_710014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, (-1)), 'str', "Create and return a temporary directory.  This has the same\n    behavior as mkdtemp but can be used as a context manager.\n\n    Upon exiting the context, the directory and everthing contained\n    in it are removed.\n\n    Examples\n    --------\n    >>> import os\n    >>> with tempdir() as tmpdir:\n    ...     fname = os.path.join(tmpdir, 'example_file.txt')\n    ...     with open(fname, 'wt') as fobj:\n    ...         _ = fobj.write('a string\\n')\n    >>> os.path.exists(tmpdir)\n    False\n    ")
    
    # Assigning a Call to a Name (line 28):
    
    # Call to mkdtemp(...): (line 28)
    # Processing the call keyword arguments (line 28)
    kwargs_710016 = {}
    # Getting the type of 'mkdtemp' (line 28)
    mkdtemp_710015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'mkdtemp', False)
    # Calling mkdtemp(args, kwargs) (line 28)
    mkdtemp_call_result_710017 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), mkdtemp_710015, *[], **kwargs_710016)
    
    # Assigning a type to the variable 'd' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'd', mkdtemp_call_result_710017)
    # Creating a generator
    # Getting the type of 'd' (line 29)
    d_710018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 10), 'd')
    GeneratorType_710019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 4), 'GeneratorType')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 4), GeneratorType_710019, d_710018)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'stypy_return_type', GeneratorType_710019)
    
    # Call to rmtree(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'd' (line 30)
    d_710021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 11), 'd', False)
    # Processing the call keyword arguments (line 30)
    kwargs_710022 = {}
    # Getting the type of 'rmtree' (line 30)
    rmtree_710020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'rmtree', False)
    # Calling rmtree(args, kwargs) (line 30)
    rmtree_call_result_710023 = invoke(stypy.reporting.localization.Localization(__file__, 30, 4), rmtree_710020, *[d_710021], **kwargs_710022)
    
    
    # ################# End of 'tempdir(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'tempdir' in the type store
    # Getting the type of 'stypy_return_type' (line 10)
    stypy_return_type_710024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_710024)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'tempdir'
    return stypy_return_type_710024

# Assigning a type to the variable 'tempdir' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'tempdir', tempdir)

@norecursion
def in_tempdir(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'in_tempdir'
    module_type_store = module_type_store.open_function_context('in_tempdir', 33, 0, False)
    
    # Passed parameters checking function
    in_tempdir.stypy_localization = localization
    in_tempdir.stypy_type_of_self = None
    in_tempdir.stypy_type_store = module_type_store
    in_tempdir.stypy_function_name = 'in_tempdir'
    in_tempdir.stypy_param_names_list = []
    in_tempdir.stypy_varargs_param_name = None
    in_tempdir.stypy_kwargs_param_name = None
    in_tempdir.stypy_call_defaults = defaults
    in_tempdir.stypy_call_varargs = varargs
    in_tempdir.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'in_tempdir', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'in_tempdir', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'in_tempdir(...)' code ##################

    str_710025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, (-1)), 'str', " Create, return, and change directory to a temporary directory\n\n    Examples\n    --------\n    >>> import os\n    >>> my_cwd = os.getcwd()\n    >>> with in_tempdir() as tmpdir:\n    ...     _ = open('test.txt', 'wt').write('some text')\n    ...     assert os.path.isfile('test.txt')\n    ...     assert os.path.isfile(os.path.join(tmpdir, 'test.txt'))\n    >>> os.path.exists(tmpdir)\n    False\n    >>> os.getcwd() == my_cwd\n    True\n    ")
    
    # Assigning a Call to a Name (line 50):
    
    # Call to getcwd(...): (line 50)
    # Processing the call keyword arguments (line 50)
    kwargs_710028 = {}
    # Getting the type of 'os' (line 50)
    os_710026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 10), 'os', False)
    # Obtaining the member 'getcwd' of a type (line 50)
    getcwd_710027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 10), os_710026, 'getcwd')
    # Calling getcwd(args, kwargs) (line 50)
    getcwd_call_result_710029 = invoke(stypy.reporting.localization.Localization(__file__, 50, 10), getcwd_710027, *[], **kwargs_710028)
    
    # Assigning a type to the variable 'pwd' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'pwd', getcwd_call_result_710029)
    
    # Assigning a Call to a Name (line 51):
    
    # Call to mkdtemp(...): (line 51)
    # Processing the call keyword arguments (line 51)
    kwargs_710031 = {}
    # Getting the type of 'mkdtemp' (line 51)
    mkdtemp_710030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'mkdtemp', False)
    # Calling mkdtemp(args, kwargs) (line 51)
    mkdtemp_call_result_710032 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), mkdtemp_710030, *[], **kwargs_710031)
    
    # Assigning a type to the variable 'd' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'd', mkdtemp_call_result_710032)
    
    # Call to chdir(...): (line 52)
    # Processing the call arguments (line 52)
    # Getting the type of 'd' (line 52)
    d_710035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 13), 'd', False)
    # Processing the call keyword arguments (line 52)
    kwargs_710036 = {}
    # Getting the type of 'os' (line 52)
    os_710033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'os', False)
    # Obtaining the member 'chdir' of a type (line 52)
    chdir_710034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 4), os_710033, 'chdir')
    # Calling chdir(args, kwargs) (line 52)
    chdir_call_result_710037 = invoke(stypy.reporting.localization.Localization(__file__, 52, 4), chdir_710034, *[d_710035], **kwargs_710036)
    
    # Creating a generator
    # Getting the type of 'd' (line 53)
    d_710038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 10), 'd')
    GeneratorType_710039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 4), 'GeneratorType')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 4), GeneratorType_710039, d_710038)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'stypy_return_type', GeneratorType_710039)
    
    # Call to chdir(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 'pwd' (line 54)
    pwd_710042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 13), 'pwd', False)
    # Processing the call keyword arguments (line 54)
    kwargs_710043 = {}
    # Getting the type of 'os' (line 54)
    os_710040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'os', False)
    # Obtaining the member 'chdir' of a type (line 54)
    chdir_710041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 4), os_710040, 'chdir')
    # Calling chdir(args, kwargs) (line 54)
    chdir_call_result_710044 = invoke(stypy.reporting.localization.Localization(__file__, 54, 4), chdir_710041, *[pwd_710042], **kwargs_710043)
    
    
    # Call to rmtree(...): (line 55)
    # Processing the call arguments (line 55)
    # Getting the type of 'd' (line 55)
    d_710046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 11), 'd', False)
    # Processing the call keyword arguments (line 55)
    kwargs_710047 = {}
    # Getting the type of 'rmtree' (line 55)
    rmtree_710045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'rmtree', False)
    # Calling rmtree(args, kwargs) (line 55)
    rmtree_call_result_710048 = invoke(stypy.reporting.localization.Localization(__file__, 55, 4), rmtree_710045, *[d_710046], **kwargs_710047)
    
    
    # ################# End of 'in_tempdir(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'in_tempdir' in the type store
    # Getting the type of 'stypy_return_type' (line 33)
    stypy_return_type_710049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_710049)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'in_tempdir'
    return stypy_return_type_710049

# Assigning a type to the variable 'in_tempdir' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'in_tempdir', in_tempdir)

@norecursion
def in_dir(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 59)
    None_710050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 15), 'None')
    defaults = [None_710050]
    # Create a new context for function 'in_dir'
    module_type_store = module_type_store.open_function_context('in_dir', 58, 0, False)
    
    # Passed parameters checking function
    in_dir.stypy_localization = localization
    in_dir.stypy_type_of_self = None
    in_dir.stypy_type_store = module_type_store
    in_dir.stypy_function_name = 'in_dir'
    in_dir.stypy_param_names_list = ['dir']
    in_dir.stypy_varargs_param_name = None
    in_dir.stypy_kwargs_param_name = None
    in_dir.stypy_call_defaults = defaults
    in_dir.stypy_call_varargs = varargs
    in_dir.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'in_dir', ['dir'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'in_dir', localization, ['dir'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'in_dir(...)' code ##################

    str_710051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, (-1)), 'str', ' Change directory to given directory for duration of ``with`` block\n\n    Useful when you want to use `in_tempdir` for the final test, but\n    you are still debugging.  For example, you may want to do this in the end:\n\n    >>> with in_tempdir() as tmpdir:\n    ...     # do something complicated which might break\n    ...     pass\n\n    But indeed the complicated thing does break, and meanwhile the\n    ``in_tempdir`` context manager wiped out the directory with the\n    temporary files that you wanted for debugging.  So, while debugging, you\n    replace with something like:\n\n    >>> with in_dir() as tmpdir: # Use working directory by default\n    ...     # do something complicated which might break\n    ...     pass\n\n    You can then look at the temporary file outputs to debug what is happening,\n    fix, and finally replace ``in_dir`` with ``in_tempdir`` again.\n    ')
    
    # Assigning a Call to a Name (line 81):
    
    # Call to getcwd(...): (line 81)
    # Processing the call keyword arguments (line 81)
    kwargs_710054 = {}
    # Getting the type of 'os' (line 81)
    os_710052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 10), 'os', False)
    # Obtaining the member 'getcwd' of a type (line 81)
    getcwd_710053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 10), os_710052, 'getcwd')
    # Calling getcwd(args, kwargs) (line 81)
    getcwd_call_result_710055 = invoke(stypy.reporting.localization.Localization(__file__, 81, 10), getcwd_710053, *[], **kwargs_710054)
    
    # Assigning a type to the variable 'cwd' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'cwd', getcwd_call_result_710055)
    
    # Type idiom detected: calculating its left and rigth part (line 82)
    # Getting the type of 'dir' (line 82)
    dir_710056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 7), 'dir')
    # Getting the type of 'None' (line 82)
    None_710057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 14), 'None')
    
    (may_be_710058, more_types_in_union_710059) = may_be_none(dir_710056, None_710057)

    if may_be_710058:

        if more_types_in_union_710059:
            # Runtime conditional SSA (line 82)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Creating a generator
        # Getting the type of 'cwd' (line 83)
        cwd_710060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 14), 'cwd')
        GeneratorType_710061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 8), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 8), GeneratorType_710061, cwd_710060)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'stypy_return_type', GeneratorType_710061)
        # Assigning a type to the variable 'stypy_return_type' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'stypy_return_type', types.NoneType)

        if more_types_in_union_710059:
            # SSA join for if statement (line 82)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to chdir(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'dir' (line 85)
    dir_710064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 13), 'dir', False)
    # Processing the call keyword arguments (line 85)
    kwargs_710065 = {}
    # Getting the type of 'os' (line 85)
    os_710062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'os', False)
    # Obtaining the member 'chdir' of a type (line 85)
    chdir_710063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 4), os_710062, 'chdir')
    # Calling chdir(args, kwargs) (line 85)
    chdir_call_result_710066 = invoke(stypy.reporting.localization.Localization(__file__, 85, 4), chdir_710063, *[dir_710064], **kwargs_710065)
    
    # Creating a generator
    # Getting the type of 'dir' (line 86)
    dir_710067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 10), 'dir')
    GeneratorType_710068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 4), 'GeneratorType')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 4), GeneratorType_710068, dir_710067)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'stypy_return_type', GeneratorType_710068)
    
    # Call to chdir(...): (line 87)
    # Processing the call arguments (line 87)
    # Getting the type of 'cwd' (line 87)
    cwd_710071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 13), 'cwd', False)
    # Processing the call keyword arguments (line 87)
    kwargs_710072 = {}
    # Getting the type of 'os' (line 87)
    os_710069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'os', False)
    # Obtaining the member 'chdir' of a type (line 87)
    chdir_710070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 4), os_710069, 'chdir')
    # Calling chdir(args, kwargs) (line 87)
    chdir_call_result_710073 = invoke(stypy.reporting.localization.Localization(__file__, 87, 4), chdir_710070, *[cwd_710071], **kwargs_710072)
    
    
    # ################# End of 'in_dir(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'in_dir' in the type store
    # Getting the type of 'stypy_return_type' (line 58)
    stypy_return_type_710074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_710074)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'in_dir'
    return stypy_return_type_710074

# Assigning a type to the variable 'in_dir' (line 58)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'in_dir', in_dir)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
