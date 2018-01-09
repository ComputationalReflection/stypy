
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import logging
2: import os
3: import tempfile
4: import shutil
5: import json
6: from subprocess import check_call
7: 
8: from dateutil.zoneinfo import tar_open, METADATA_FN, ZONEFILENAME
9: 
10: 
11: def rebuild(filename, tag=None, format="gz", zonegroups=[], metadata=None):
12:     '''Rebuild the internal timezone info in dateutil/zoneinfo/zoneinfo*tar*
13: 
14:     filename is the timezone tarball from ftp.iana.org/tz.
15: 
16:     '''
17:     tmpdir = tempfile.mkdtemp()
18:     zonedir = os.path.join(tmpdir, "zoneinfo")
19:     moduledir = os.path.dirname(__file__)
20:     try:
21:         with tar_open(filename) as tf:
22:             for name in zonegroups:
23:                 tf.extract(name, tmpdir)
24:             filepaths = [os.path.join(tmpdir, n) for n in zonegroups]
25:             try:
26:                 check_call(["zic", "-d", zonedir] + filepaths)
27:             except OSError as e:
28:                 _print_on_nosuchfile(e)
29:                 raise
30:         # write metadata file
31:         with open(os.path.join(zonedir, METADATA_FN), 'w') as f:
32:             json.dump(metadata, f, indent=4, sort_keys=True)
33:         target = os.path.join(moduledir, ZONEFILENAME)
34:         with tar_open(target, "w:%s" % format) as tf:
35:             for entry in os.listdir(zonedir):
36:                 entrypath = os.path.join(zonedir, entry)
37:                 tf.add(entrypath, entry)
38:     finally:
39:         shutil.rmtree(tmpdir)
40: 
41: 
42: def _print_on_nosuchfile(e):
43:     '''Print helpful troubleshooting message
44: 
45:     e is an exception raised by subprocess.check_call()
46: 
47:     '''
48:     if e.errno == 2:
49:         logging.error(
50:             "Could not find zic. Perhaps you need to install "
51:             "libc-bin or some other package that provides it, "
52:             "or it's not in your PATH?")
53: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import logging' statement (line 1)
import logging

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'logging', logging, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import os' statement (line 2)
import os

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import tempfile' statement (line 3)
import tempfile

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'tempfile', tempfile, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import shutil' statement (line 4)
import shutil

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'shutil', shutil, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import json' statement (line 5)
import json

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'json', json, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from subprocess import check_call' statement (line 6)
try:
    from subprocess import check_call

except:
    check_call = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'subprocess', None, module_type_store, ['check_call'], [check_call])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from dateutil.zoneinfo import tar_open, METADATA_FN, ZONEFILENAME' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/dateutil/zoneinfo/')
import_324826 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'dateutil.zoneinfo')

if (type(import_324826) is not StypyTypeError):

    if (import_324826 != 'pyd_module'):
        __import__(import_324826)
        sys_modules_324827 = sys.modules[import_324826]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'dateutil.zoneinfo', sys_modules_324827.module_type_store, module_type_store, ['tar_open', 'METADATA_FN', 'ZONEFILENAME'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_324827, sys_modules_324827.module_type_store, module_type_store)
    else:
        from dateutil.zoneinfo import tar_open, METADATA_FN, ZONEFILENAME

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'dateutil.zoneinfo', None, module_type_store, ['tar_open', 'METADATA_FN', 'ZONEFILENAME'], [tar_open, METADATA_FN, ZONEFILENAME])

else:
    # Assigning a type to the variable 'dateutil.zoneinfo' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'dateutil.zoneinfo', import_324826)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/dateutil/zoneinfo/')


@norecursion
def rebuild(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 11)
    None_324828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 26), 'None')
    str_324829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 39), 'str', 'gz')
    
    # Obtaining an instance of the builtin type 'list' (line 11)
    list_324830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 56), 'list')
    # Adding type elements to the builtin type 'list' instance (line 11)
    
    # Getting the type of 'None' (line 11)
    None_324831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 69), 'None')
    defaults = [None_324828, str_324829, list_324830, None_324831]
    # Create a new context for function 'rebuild'
    module_type_store = module_type_store.open_function_context('rebuild', 11, 0, False)
    
    # Passed parameters checking function
    rebuild.stypy_localization = localization
    rebuild.stypy_type_of_self = None
    rebuild.stypy_type_store = module_type_store
    rebuild.stypy_function_name = 'rebuild'
    rebuild.stypy_param_names_list = ['filename', 'tag', 'format', 'zonegroups', 'metadata']
    rebuild.stypy_varargs_param_name = None
    rebuild.stypy_kwargs_param_name = None
    rebuild.stypy_call_defaults = defaults
    rebuild.stypy_call_varargs = varargs
    rebuild.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'rebuild', ['filename', 'tag', 'format', 'zonegroups', 'metadata'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'rebuild', localization, ['filename', 'tag', 'format', 'zonegroups', 'metadata'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'rebuild(...)' code ##################

    str_324832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, (-1)), 'str', 'Rebuild the internal timezone info in dateutil/zoneinfo/zoneinfo*tar*\n\n    filename is the timezone tarball from ftp.iana.org/tz.\n\n    ')
    
    # Assigning a Call to a Name (line 17):
    
    # Call to mkdtemp(...): (line 17)
    # Processing the call keyword arguments (line 17)
    kwargs_324835 = {}
    # Getting the type of 'tempfile' (line 17)
    tempfile_324833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 13), 'tempfile', False)
    # Obtaining the member 'mkdtemp' of a type (line 17)
    mkdtemp_324834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 13), tempfile_324833, 'mkdtemp')
    # Calling mkdtemp(args, kwargs) (line 17)
    mkdtemp_call_result_324836 = invoke(stypy.reporting.localization.Localization(__file__, 17, 13), mkdtemp_324834, *[], **kwargs_324835)
    
    # Assigning a type to the variable 'tmpdir' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'tmpdir', mkdtemp_call_result_324836)
    
    # Assigning a Call to a Name (line 18):
    
    # Call to join(...): (line 18)
    # Processing the call arguments (line 18)
    # Getting the type of 'tmpdir' (line 18)
    tmpdir_324840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 27), 'tmpdir', False)
    str_324841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 35), 'str', 'zoneinfo')
    # Processing the call keyword arguments (line 18)
    kwargs_324842 = {}
    # Getting the type of 'os' (line 18)
    os_324837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 14), 'os', False)
    # Obtaining the member 'path' of a type (line 18)
    path_324838 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 14), os_324837, 'path')
    # Obtaining the member 'join' of a type (line 18)
    join_324839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 14), path_324838, 'join')
    # Calling join(args, kwargs) (line 18)
    join_call_result_324843 = invoke(stypy.reporting.localization.Localization(__file__, 18, 14), join_324839, *[tmpdir_324840, str_324841], **kwargs_324842)
    
    # Assigning a type to the variable 'zonedir' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'zonedir', join_call_result_324843)
    
    # Assigning a Call to a Name (line 19):
    
    # Call to dirname(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of '__file__' (line 19)
    file___324847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 32), '__file__', False)
    # Processing the call keyword arguments (line 19)
    kwargs_324848 = {}
    # Getting the type of 'os' (line 19)
    os_324844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 16), 'os', False)
    # Obtaining the member 'path' of a type (line 19)
    path_324845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 16), os_324844, 'path')
    # Obtaining the member 'dirname' of a type (line 19)
    dirname_324846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 16), path_324845, 'dirname')
    # Calling dirname(args, kwargs) (line 19)
    dirname_call_result_324849 = invoke(stypy.reporting.localization.Localization(__file__, 19, 16), dirname_324846, *[file___324847], **kwargs_324848)
    
    # Assigning a type to the variable 'moduledir' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'moduledir', dirname_call_result_324849)
    
    # Try-finally block (line 20)
    
    # Call to tar_open(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 'filename' (line 21)
    filename_324851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 22), 'filename', False)
    # Processing the call keyword arguments (line 21)
    kwargs_324852 = {}
    # Getting the type of 'tar_open' (line 21)
    tar_open_324850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 13), 'tar_open', False)
    # Calling tar_open(args, kwargs) (line 21)
    tar_open_call_result_324853 = invoke(stypy.reporting.localization.Localization(__file__, 21, 13), tar_open_324850, *[filename_324851], **kwargs_324852)
    
    with_324854 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 21, 13), tar_open_call_result_324853, 'with parameter', '__enter__', '__exit__')

    if with_324854:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 21)
        enter___324855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 13), tar_open_call_result_324853, '__enter__')
        with_enter_324856 = invoke(stypy.reporting.localization.Localization(__file__, 21, 13), enter___324855)
        # Assigning a type to the variable 'tf' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 13), 'tf', with_enter_324856)
        
        # Getting the type of 'zonegroups' (line 22)
        zonegroups_324857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 24), 'zonegroups')
        # Testing the type of a for loop iterable (line 22)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 22, 12), zonegroups_324857)
        # Getting the type of the for loop variable (line 22)
        for_loop_var_324858 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 22, 12), zonegroups_324857)
        # Assigning a type to the variable 'name' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'name', for_loop_var_324858)
        # SSA begins for a for statement (line 22)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to extract(...): (line 23)
        # Processing the call arguments (line 23)
        # Getting the type of 'name' (line 23)
        name_324861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 27), 'name', False)
        # Getting the type of 'tmpdir' (line 23)
        tmpdir_324862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 33), 'tmpdir', False)
        # Processing the call keyword arguments (line 23)
        kwargs_324863 = {}
        # Getting the type of 'tf' (line 23)
        tf_324859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 16), 'tf', False)
        # Obtaining the member 'extract' of a type (line 23)
        extract_324860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 16), tf_324859, 'extract')
        # Calling extract(args, kwargs) (line 23)
        extract_call_result_324864 = invoke(stypy.reporting.localization.Localization(__file__, 23, 16), extract_324860, *[name_324861, tmpdir_324862], **kwargs_324863)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a ListComp to a Name (line 24):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'zonegroups' (line 24)
        zonegroups_324872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 58), 'zonegroups')
        comprehension_324873 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 25), zonegroups_324872)
        # Assigning a type to the variable 'n' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 25), 'n', comprehension_324873)
        
        # Call to join(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'tmpdir' (line 24)
        tmpdir_324868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 38), 'tmpdir', False)
        # Getting the type of 'n' (line 24)
        n_324869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 46), 'n', False)
        # Processing the call keyword arguments (line 24)
        kwargs_324870 = {}
        # Getting the type of 'os' (line 24)
        os_324865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 25), 'os', False)
        # Obtaining the member 'path' of a type (line 24)
        path_324866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 25), os_324865, 'path')
        # Obtaining the member 'join' of a type (line 24)
        join_324867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 25), path_324866, 'join')
        # Calling join(args, kwargs) (line 24)
        join_call_result_324871 = invoke(stypy.reporting.localization.Localization(__file__, 24, 25), join_324867, *[tmpdir_324868, n_324869], **kwargs_324870)
        
        list_324874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 25), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 25), list_324874, join_call_result_324871)
        # Assigning a type to the variable 'filepaths' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'filepaths', list_324874)
        
        
        # SSA begins for try-except statement (line 25)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to check_call(...): (line 26)
        # Processing the call arguments (line 26)
        
        # Obtaining an instance of the builtin type 'list' (line 26)
        list_324876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 26)
        # Adding element type (line 26)
        str_324877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 28), 'str', 'zic')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 27), list_324876, str_324877)
        # Adding element type (line 26)
        str_324878 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 35), 'str', '-d')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 27), list_324876, str_324878)
        # Adding element type (line 26)
        # Getting the type of 'zonedir' (line 26)
        zonedir_324879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 41), 'zonedir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 27), list_324876, zonedir_324879)
        
        # Getting the type of 'filepaths' (line 26)
        filepaths_324880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 52), 'filepaths', False)
        # Applying the binary operator '+' (line 26)
        result_add_324881 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 27), '+', list_324876, filepaths_324880)
        
        # Processing the call keyword arguments (line 26)
        kwargs_324882 = {}
        # Getting the type of 'check_call' (line 26)
        check_call_324875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 16), 'check_call', False)
        # Calling check_call(args, kwargs) (line 26)
        check_call_call_result_324883 = invoke(stypy.reporting.localization.Localization(__file__, 26, 16), check_call_324875, *[result_add_324881], **kwargs_324882)
        
        # SSA branch for the except part of a try statement (line 25)
        # SSA branch for the except 'OSError' branch of a try statement (line 25)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'OSError' (line 27)
        OSError_324884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 19), 'OSError')
        # Assigning a type to the variable 'e' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'e', OSError_324884)
        
        # Call to _print_on_nosuchfile(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'e' (line 28)
        e_324886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 37), 'e', False)
        # Processing the call keyword arguments (line 28)
        kwargs_324887 = {}
        # Getting the type of '_print_on_nosuchfile' (line 28)
        _print_on_nosuchfile_324885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 16), '_print_on_nosuchfile', False)
        # Calling _print_on_nosuchfile(args, kwargs) (line 28)
        _print_on_nosuchfile_call_result_324888 = invoke(stypy.reporting.localization.Localization(__file__, 28, 16), _print_on_nosuchfile_324885, *[e_324886], **kwargs_324887)
        
        # SSA join for try-except statement (line 25)
        module_type_store = module_type_store.join_ssa_context()
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 21)
        exit___324889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 13), tar_open_call_result_324853, '__exit__')
        with_exit_324890 = invoke(stypy.reporting.localization.Localization(__file__, 21, 13), exit___324889, None, None, None)

    
    # Call to open(...): (line 31)
    # Processing the call arguments (line 31)
    
    # Call to join(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'zonedir' (line 31)
    zonedir_324895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 31), 'zonedir', False)
    # Getting the type of 'METADATA_FN' (line 31)
    METADATA_FN_324896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 40), 'METADATA_FN', False)
    # Processing the call keyword arguments (line 31)
    kwargs_324897 = {}
    # Getting the type of 'os' (line 31)
    os_324892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 18), 'os', False)
    # Obtaining the member 'path' of a type (line 31)
    path_324893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 18), os_324892, 'path')
    # Obtaining the member 'join' of a type (line 31)
    join_324894 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 18), path_324893, 'join')
    # Calling join(args, kwargs) (line 31)
    join_call_result_324898 = invoke(stypy.reporting.localization.Localization(__file__, 31, 18), join_324894, *[zonedir_324895, METADATA_FN_324896], **kwargs_324897)
    
    str_324899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 54), 'str', 'w')
    # Processing the call keyword arguments (line 31)
    kwargs_324900 = {}
    # Getting the type of 'open' (line 31)
    open_324891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 13), 'open', False)
    # Calling open(args, kwargs) (line 31)
    open_call_result_324901 = invoke(stypy.reporting.localization.Localization(__file__, 31, 13), open_324891, *[join_call_result_324898, str_324899], **kwargs_324900)
    
    with_324902 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 31, 13), open_call_result_324901, 'with parameter', '__enter__', '__exit__')

    if with_324902:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 31)
        enter___324903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 13), open_call_result_324901, '__enter__')
        with_enter_324904 = invoke(stypy.reporting.localization.Localization(__file__, 31, 13), enter___324903)
        # Assigning a type to the variable 'f' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 13), 'f', with_enter_324904)
        
        # Call to dump(...): (line 32)
        # Processing the call arguments (line 32)
        # Getting the type of 'metadata' (line 32)
        metadata_324907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 22), 'metadata', False)
        # Getting the type of 'f' (line 32)
        f_324908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 32), 'f', False)
        # Processing the call keyword arguments (line 32)
        int_324909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 42), 'int')
        keyword_324910 = int_324909
        # Getting the type of 'True' (line 32)
        True_324911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 55), 'True', False)
        keyword_324912 = True_324911
        kwargs_324913 = {'sort_keys': keyword_324912, 'indent': keyword_324910}
        # Getting the type of 'json' (line 32)
        json_324905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'json', False)
        # Obtaining the member 'dump' of a type (line 32)
        dump_324906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 12), json_324905, 'dump')
        # Calling dump(args, kwargs) (line 32)
        dump_call_result_324914 = invoke(stypy.reporting.localization.Localization(__file__, 32, 12), dump_324906, *[metadata_324907, f_324908], **kwargs_324913)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 31)
        exit___324915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 13), open_call_result_324901, '__exit__')
        with_exit_324916 = invoke(stypy.reporting.localization.Localization(__file__, 31, 13), exit___324915, None, None, None)

    
    # Assigning a Call to a Name (line 33):
    
    # Call to join(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'moduledir' (line 33)
    moduledir_324920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 30), 'moduledir', False)
    # Getting the type of 'ZONEFILENAME' (line 33)
    ZONEFILENAME_324921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 41), 'ZONEFILENAME', False)
    # Processing the call keyword arguments (line 33)
    kwargs_324922 = {}
    # Getting the type of 'os' (line 33)
    os_324917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 17), 'os', False)
    # Obtaining the member 'path' of a type (line 33)
    path_324918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 17), os_324917, 'path')
    # Obtaining the member 'join' of a type (line 33)
    join_324919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 17), path_324918, 'join')
    # Calling join(args, kwargs) (line 33)
    join_call_result_324923 = invoke(stypy.reporting.localization.Localization(__file__, 33, 17), join_324919, *[moduledir_324920, ZONEFILENAME_324921], **kwargs_324922)
    
    # Assigning a type to the variable 'target' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'target', join_call_result_324923)
    
    # Call to tar_open(...): (line 34)
    # Processing the call arguments (line 34)
    # Getting the type of 'target' (line 34)
    target_324925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 22), 'target', False)
    str_324926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 30), 'str', 'w:%s')
    # Getting the type of 'format' (line 34)
    format_324927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 39), 'format', False)
    # Applying the binary operator '%' (line 34)
    result_mod_324928 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 30), '%', str_324926, format_324927)
    
    # Processing the call keyword arguments (line 34)
    kwargs_324929 = {}
    # Getting the type of 'tar_open' (line 34)
    tar_open_324924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 13), 'tar_open', False)
    # Calling tar_open(args, kwargs) (line 34)
    tar_open_call_result_324930 = invoke(stypy.reporting.localization.Localization(__file__, 34, 13), tar_open_324924, *[target_324925, result_mod_324928], **kwargs_324929)
    
    with_324931 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 34, 13), tar_open_call_result_324930, 'with parameter', '__enter__', '__exit__')

    if with_324931:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 34)
        enter___324932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 13), tar_open_call_result_324930, '__enter__')
        with_enter_324933 = invoke(stypy.reporting.localization.Localization(__file__, 34, 13), enter___324932)
        # Assigning a type to the variable 'tf' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 13), 'tf', with_enter_324933)
        
        
        # Call to listdir(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'zonedir' (line 35)
        zonedir_324936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 36), 'zonedir', False)
        # Processing the call keyword arguments (line 35)
        kwargs_324937 = {}
        # Getting the type of 'os' (line 35)
        os_324934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 25), 'os', False)
        # Obtaining the member 'listdir' of a type (line 35)
        listdir_324935 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 25), os_324934, 'listdir')
        # Calling listdir(args, kwargs) (line 35)
        listdir_call_result_324938 = invoke(stypy.reporting.localization.Localization(__file__, 35, 25), listdir_324935, *[zonedir_324936], **kwargs_324937)
        
        # Testing the type of a for loop iterable (line 35)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 35, 12), listdir_call_result_324938)
        # Getting the type of the for loop variable (line 35)
        for_loop_var_324939 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 35, 12), listdir_call_result_324938)
        # Assigning a type to the variable 'entry' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'entry', for_loop_var_324939)
        # SSA begins for a for statement (line 35)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 36):
        
        # Call to join(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'zonedir' (line 36)
        zonedir_324943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 41), 'zonedir', False)
        # Getting the type of 'entry' (line 36)
        entry_324944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 50), 'entry', False)
        # Processing the call keyword arguments (line 36)
        kwargs_324945 = {}
        # Getting the type of 'os' (line 36)
        os_324940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 28), 'os', False)
        # Obtaining the member 'path' of a type (line 36)
        path_324941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 28), os_324940, 'path')
        # Obtaining the member 'join' of a type (line 36)
        join_324942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 28), path_324941, 'join')
        # Calling join(args, kwargs) (line 36)
        join_call_result_324946 = invoke(stypy.reporting.localization.Localization(__file__, 36, 28), join_324942, *[zonedir_324943, entry_324944], **kwargs_324945)
        
        # Assigning a type to the variable 'entrypath' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'entrypath', join_call_result_324946)
        
        # Call to add(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'entrypath' (line 37)
        entrypath_324949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 23), 'entrypath', False)
        # Getting the type of 'entry' (line 37)
        entry_324950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 34), 'entry', False)
        # Processing the call keyword arguments (line 37)
        kwargs_324951 = {}
        # Getting the type of 'tf' (line 37)
        tf_324947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 16), 'tf', False)
        # Obtaining the member 'add' of a type (line 37)
        add_324948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 16), tf_324947, 'add')
        # Calling add(args, kwargs) (line 37)
        add_call_result_324952 = invoke(stypy.reporting.localization.Localization(__file__, 37, 16), add_324948, *[entrypath_324949, entry_324950], **kwargs_324951)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 34)
        exit___324953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 13), tar_open_call_result_324930, '__exit__')
        with_exit_324954 = invoke(stypy.reporting.localization.Localization(__file__, 34, 13), exit___324953, None, None, None)

    
    # finally branch of the try-finally block (line 20)
    
    # Call to rmtree(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'tmpdir' (line 39)
    tmpdir_324957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 22), 'tmpdir', False)
    # Processing the call keyword arguments (line 39)
    kwargs_324958 = {}
    # Getting the type of 'shutil' (line 39)
    shutil_324955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'shutil', False)
    # Obtaining the member 'rmtree' of a type (line 39)
    rmtree_324956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), shutil_324955, 'rmtree')
    # Calling rmtree(args, kwargs) (line 39)
    rmtree_call_result_324959 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), rmtree_324956, *[tmpdir_324957], **kwargs_324958)
    
    
    
    # ################# End of 'rebuild(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'rebuild' in the type store
    # Getting the type of 'stypy_return_type' (line 11)
    stypy_return_type_324960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_324960)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'rebuild'
    return stypy_return_type_324960

# Assigning a type to the variable 'rebuild' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'rebuild', rebuild)

@norecursion
def _print_on_nosuchfile(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_print_on_nosuchfile'
    module_type_store = module_type_store.open_function_context('_print_on_nosuchfile', 42, 0, False)
    
    # Passed parameters checking function
    _print_on_nosuchfile.stypy_localization = localization
    _print_on_nosuchfile.stypy_type_of_self = None
    _print_on_nosuchfile.stypy_type_store = module_type_store
    _print_on_nosuchfile.stypy_function_name = '_print_on_nosuchfile'
    _print_on_nosuchfile.stypy_param_names_list = ['e']
    _print_on_nosuchfile.stypy_varargs_param_name = None
    _print_on_nosuchfile.stypy_kwargs_param_name = None
    _print_on_nosuchfile.stypy_call_defaults = defaults
    _print_on_nosuchfile.stypy_call_varargs = varargs
    _print_on_nosuchfile.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_print_on_nosuchfile', ['e'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_print_on_nosuchfile', localization, ['e'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_print_on_nosuchfile(...)' code ##################

    str_324961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, (-1)), 'str', 'Print helpful troubleshooting message\n\n    e is an exception raised by subprocess.check_call()\n\n    ')
    
    
    # Getting the type of 'e' (line 48)
    e_324962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 7), 'e')
    # Obtaining the member 'errno' of a type (line 48)
    errno_324963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 7), e_324962, 'errno')
    int_324964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 18), 'int')
    # Applying the binary operator '==' (line 48)
    result_eq_324965 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 7), '==', errno_324963, int_324964)
    
    # Testing the type of an if condition (line 48)
    if_condition_324966 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 48, 4), result_eq_324965)
    # Assigning a type to the variable 'if_condition_324966' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'if_condition_324966', if_condition_324966)
    # SSA begins for if statement (line 48)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to error(...): (line 49)
    # Processing the call arguments (line 49)
    str_324969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 12), 'str', "Could not find zic. Perhaps you need to install libc-bin or some other package that provides it, or it's not in your PATH?")
    # Processing the call keyword arguments (line 49)
    kwargs_324970 = {}
    # Getting the type of 'logging' (line 49)
    logging_324967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'logging', False)
    # Obtaining the member 'error' of a type (line 49)
    error_324968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), logging_324967, 'error')
    # Calling error(args, kwargs) (line 49)
    error_call_result_324971 = invoke(stypy.reporting.localization.Localization(__file__, 49, 8), error_324968, *[str_324969], **kwargs_324970)
    
    # SSA join for if statement (line 48)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_print_on_nosuchfile(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_print_on_nosuchfile' in the type store
    # Getting the type of 'stypy_return_type' (line 42)
    stypy_return_type_324972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_324972)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_print_on_nosuchfile'
    return stypy_return_type_324972

# Assigning a type to the variable '_print_on_nosuchfile' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), '_print_on_nosuchfile', _print_on_nosuchfile)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
