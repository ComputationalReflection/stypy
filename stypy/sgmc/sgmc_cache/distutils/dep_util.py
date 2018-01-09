
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.dep_util
2: 
3: Utility functions for simple, timestamp-based dependency of files
4: and groups of files; also, function based entirely on such
5: timestamp dependency analysis.'''
6: 
7: __revision__ = "$Id$"
8: 
9: import os
10: from stat import ST_MTIME
11: from distutils.errors import DistutilsFileError
12: 
13: def newer(source, target):
14:     '''Tells if the target is newer than the source.
15: 
16:     Return true if 'source' exists and is more recently modified than
17:     'target', or if 'source' exists and 'target' doesn't.
18: 
19:     Return false if both exist and 'target' is the same age or younger
20:     than 'source'. Raise DistutilsFileError if 'source' does not exist.
21: 
22:     Note that this test is not very accurate: files created in the same second
23:     will have the same "age".
24:     '''
25:     if not os.path.exists(source):
26:         raise DistutilsFileError("file '%s' does not exist" %
27:                                  os.path.abspath(source))
28:     if not os.path.exists(target):
29:         return True
30: 
31:     return os.stat(source)[ST_MTIME] > os.stat(target)[ST_MTIME]
32: 
33: def newer_pairwise(sources, targets):
34:     '''Walk two filename lists in parallel, testing if each source is newer
35:     than its corresponding target.  Return a pair of lists (sources,
36:     targets) where source is newer than target, according to the semantics
37:     of 'newer()'.
38:     '''
39:     if len(sources) != len(targets):
40:         raise ValueError, "'sources' and 'targets' must be same length"
41: 
42:     # build a pair of lists (sources, targets) where  source is newer
43:     n_sources = []
44:     n_targets = []
45:     for source, target in zip(sources, targets):
46:         if newer(source, target):
47:             n_sources.append(source)
48:             n_targets.append(target)
49: 
50:     return n_sources, n_targets
51: 
52: def newer_group(sources, target, missing='error'):
53:     '''Return true if 'target' is out-of-date with respect to any file
54:     listed in 'sources'.
55: 
56:     In other words, if 'target' exists and is newer
57:     than every file in 'sources', return false; otherwise return true.
58:     'missing' controls what we do when a source file is missing; the
59:     default ("error") is to blow up with an OSError from inside 'stat()';
60:     if it is "ignore", we silently drop any missing source files; if it is
61:     "newer", any missing source files make us assume that 'target' is
62:     out-of-date (this is handy in "dry-run" mode: it'll make you pretend to
63:     carry out commands that wouldn't work because inputs are missing, but
64:     that doesn't matter because you're not actually going to run the
65:     commands).
66:     '''
67:     # If the target doesn't even exist, then it's definitely out-of-date.
68:     if not os.path.exists(target):
69:         return True
70: 
71:     # Otherwise we have to find out the hard way: if *any* source file
72:     # is more recent than 'target', then 'target' is out-of-date and
73:     # we can immediately return true.  If we fall through to the end
74:     # of the loop, then 'target' is up-to-date and we return false.
75:     target_mtime = os.stat(target)[ST_MTIME]
76: 
77:     for source in sources:
78:         if not os.path.exists(source):
79:             if missing == 'error':      # blow up when we stat() the file
80:                 pass
81:             elif missing == 'ignore':   # missing source dropped from
82:                 continue                #  target's dependency list
83:             elif missing == 'newer':    # missing source means target is
84:                 return True             #  out-of-date
85: 
86:         if os.stat(source)[ST_MTIME] > target_mtime:
87:             return True
88: 
89:     return False
90: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_307500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, (-1)), 'str', 'distutils.dep_util\n\nUtility functions for simple, timestamp-based dependency of files\nand groups of files; also, function based entirely on such\ntimestamp dependency analysis.')

# Assigning a Str to a Name (line 7):
str_307501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__revision__', str_307501)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import os' statement (line 9)
import os

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from stat import ST_MTIME' statement (line 10)
try:
    from stat import ST_MTIME

except:
    ST_MTIME = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'stat', None, module_type_store, ['ST_MTIME'], [ST_MTIME])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from distutils.errors import DistutilsFileError' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_307502 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.errors')

if (type(import_307502) is not StypyTypeError):

    if (import_307502 != 'pyd_module'):
        __import__(import_307502)
        sys_modules_307503 = sys.modules[import_307502]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.errors', sys_modules_307503.module_type_store, module_type_store, ['DistutilsFileError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_307503, sys_modules_307503.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsFileError

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.errors', None, module_type_store, ['DistutilsFileError'], [DistutilsFileError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.errors', import_307502)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')


@norecursion
def newer(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'newer'
    module_type_store = module_type_store.open_function_context('newer', 13, 0, False)
    
    # Passed parameters checking function
    newer.stypy_localization = localization
    newer.stypy_type_of_self = None
    newer.stypy_type_store = module_type_store
    newer.stypy_function_name = 'newer'
    newer.stypy_param_names_list = ['source', 'target']
    newer.stypy_varargs_param_name = None
    newer.stypy_kwargs_param_name = None
    newer.stypy_call_defaults = defaults
    newer.stypy_call_varargs = varargs
    newer.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'newer', ['source', 'target'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'newer', localization, ['source', 'target'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'newer(...)' code ##################

    str_307504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, (-1)), 'str', 'Tells if the target is newer than the source.\n\n    Return true if \'source\' exists and is more recently modified than\n    \'target\', or if \'source\' exists and \'target\' doesn\'t.\n\n    Return false if both exist and \'target\' is the same age or younger\n    than \'source\'. Raise DistutilsFileError if \'source\' does not exist.\n\n    Note that this test is not very accurate: files created in the same second\n    will have the same "age".\n    ')
    
    
    
    # Call to exists(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'source' (line 25)
    source_307508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 26), 'source', False)
    # Processing the call keyword arguments (line 25)
    kwargs_307509 = {}
    # Getting the type of 'os' (line 25)
    os_307505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 25)
    path_307506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 11), os_307505, 'path')
    # Obtaining the member 'exists' of a type (line 25)
    exists_307507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 11), path_307506, 'exists')
    # Calling exists(args, kwargs) (line 25)
    exists_call_result_307510 = invoke(stypy.reporting.localization.Localization(__file__, 25, 11), exists_307507, *[source_307508], **kwargs_307509)
    
    # Applying the 'not' unary operator (line 25)
    result_not__307511 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 7), 'not', exists_call_result_307510)
    
    # Testing the type of an if condition (line 25)
    if_condition_307512 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 25, 4), result_not__307511)
    # Assigning a type to the variable 'if_condition_307512' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'if_condition_307512', if_condition_307512)
    # SSA begins for if statement (line 25)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to DistutilsFileError(...): (line 26)
    # Processing the call arguments (line 26)
    str_307514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 33), 'str', "file '%s' does not exist")
    
    # Call to abspath(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'source' (line 27)
    source_307518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 49), 'source', False)
    # Processing the call keyword arguments (line 27)
    kwargs_307519 = {}
    # Getting the type of 'os' (line 27)
    os_307515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 33), 'os', False)
    # Obtaining the member 'path' of a type (line 27)
    path_307516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 33), os_307515, 'path')
    # Obtaining the member 'abspath' of a type (line 27)
    abspath_307517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 33), path_307516, 'abspath')
    # Calling abspath(args, kwargs) (line 27)
    abspath_call_result_307520 = invoke(stypy.reporting.localization.Localization(__file__, 27, 33), abspath_307517, *[source_307518], **kwargs_307519)
    
    # Applying the binary operator '%' (line 26)
    result_mod_307521 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 33), '%', str_307514, abspath_call_result_307520)
    
    # Processing the call keyword arguments (line 26)
    kwargs_307522 = {}
    # Getting the type of 'DistutilsFileError' (line 26)
    DistutilsFileError_307513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 14), 'DistutilsFileError', False)
    # Calling DistutilsFileError(args, kwargs) (line 26)
    DistutilsFileError_call_result_307523 = invoke(stypy.reporting.localization.Localization(__file__, 26, 14), DistutilsFileError_307513, *[result_mod_307521], **kwargs_307522)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 26, 8), DistutilsFileError_call_result_307523, 'raise parameter', BaseException)
    # SSA join for if statement (line 25)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to exists(...): (line 28)
    # Processing the call arguments (line 28)
    # Getting the type of 'target' (line 28)
    target_307527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 26), 'target', False)
    # Processing the call keyword arguments (line 28)
    kwargs_307528 = {}
    # Getting the type of 'os' (line 28)
    os_307524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 28)
    path_307525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 11), os_307524, 'path')
    # Obtaining the member 'exists' of a type (line 28)
    exists_307526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 11), path_307525, 'exists')
    # Calling exists(args, kwargs) (line 28)
    exists_call_result_307529 = invoke(stypy.reporting.localization.Localization(__file__, 28, 11), exists_307526, *[target_307527], **kwargs_307528)
    
    # Applying the 'not' unary operator (line 28)
    result_not__307530 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 7), 'not', exists_call_result_307529)
    
    # Testing the type of an if condition (line 28)
    if_condition_307531 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 28, 4), result_not__307530)
    # Assigning a type to the variable 'if_condition_307531' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'if_condition_307531', if_condition_307531)
    # SSA begins for if statement (line 28)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'True' (line 29)
    True_307532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 15), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'stypy_return_type', True_307532)
    # SSA join for if statement (line 28)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'ST_MTIME' (line 31)
    ST_MTIME_307533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 27), 'ST_MTIME')
    
    # Call to stat(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'source' (line 31)
    source_307536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 19), 'source', False)
    # Processing the call keyword arguments (line 31)
    kwargs_307537 = {}
    # Getting the type of 'os' (line 31)
    os_307534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 11), 'os', False)
    # Obtaining the member 'stat' of a type (line 31)
    stat_307535 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 11), os_307534, 'stat')
    # Calling stat(args, kwargs) (line 31)
    stat_call_result_307538 = invoke(stypy.reporting.localization.Localization(__file__, 31, 11), stat_307535, *[source_307536], **kwargs_307537)
    
    # Obtaining the member '__getitem__' of a type (line 31)
    getitem___307539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 11), stat_call_result_307538, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 31)
    subscript_call_result_307540 = invoke(stypy.reporting.localization.Localization(__file__, 31, 11), getitem___307539, ST_MTIME_307533)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'ST_MTIME' (line 31)
    ST_MTIME_307541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 55), 'ST_MTIME')
    
    # Call to stat(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'target' (line 31)
    target_307544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 47), 'target', False)
    # Processing the call keyword arguments (line 31)
    kwargs_307545 = {}
    # Getting the type of 'os' (line 31)
    os_307542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 39), 'os', False)
    # Obtaining the member 'stat' of a type (line 31)
    stat_307543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 39), os_307542, 'stat')
    # Calling stat(args, kwargs) (line 31)
    stat_call_result_307546 = invoke(stypy.reporting.localization.Localization(__file__, 31, 39), stat_307543, *[target_307544], **kwargs_307545)
    
    # Obtaining the member '__getitem__' of a type (line 31)
    getitem___307547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 39), stat_call_result_307546, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 31)
    subscript_call_result_307548 = invoke(stypy.reporting.localization.Localization(__file__, 31, 39), getitem___307547, ST_MTIME_307541)
    
    # Applying the binary operator '>' (line 31)
    result_gt_307549 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 11), '>', subscript_call_result_307540, subscript_call_result_307548)
    
    # Assigning a type to the variable 'stypy_return_type' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'stypy_return_type', result_gt_307549)
    
    # ################# End of 'newer(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'newer' in the type store
    # Getting the type of 'stypy_return_type' (line 13)
    stypy_return_type_307550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_307550)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'newer'
    return stypy_return_type_307550

# Assigning a type to the variable 'newer' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'newer', newer)

@norecursion
def newer_pairwise(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'newer_pairwise'
    module_type_store = module_type_store.open_function_context('newer_pairwise', 33, 0, False)
    
    # Passed parameters checking function
    newer_pairwise.stypy_localization = localization
    newer_pairwise.stypy_type_of_self = None
    newer_pairwise.stypy_type_store = module_type_store
    newer_pairwise.stypy_function_name = 'newer_pairwise'
    newer_pairwise.stypy_param_names_list = ['sources', 'targets']
    newer_pairwise.stypy_varargs_param_name = None
    newer_pairwise.stypy_kwargs_param_name = None
    newer_pairwise.stypy_call_defaults = defaults
    newer_pairwise.stypy_call_varargs = varargs
    newer_pairwise.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'newer_pairwise', ['sources', 'targets'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'newer_pairwise', localization, ['sources', 'targets'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'newer_pairwise(...)' code ##################

    str_307551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, (-1)), 'str', "Walk two filename lists in parallel, testing if each source is newer\n    than its corresponding target.  Return a pair of lists (sources,\n    targets) where source is newer than target, according to the semantics\n    of 'newer()'.\n    ")
    
    
    
    # Call to len(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'sources' (line 39)
    sources_307553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 11), 'sources', False)
    # Processing the call keyword arguments (line 39)
    kwargs_307554 = {}
    # Getting the type of 'len' (line 39)
    len_307552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 7), 'len', False)
    # Calling len(args, kwargs) (line 39)
    len_call_result_307555 = invoke(stypy.reporting.localization.Localization(__file__, 39, 7), len_307552, *[sources_307553], **kwargs_307554)
    
    
    # Call to len(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'targets' (line 39)
    targets_307557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 27), 'targets', False)
    # Processing the call keyword arguments (line 39)
    kwargs_307558 = {}
    # Getting the type of 'len' (line 39)
    len_307556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 23), 'len', False)
    # Calling len(args, kwargs) (line 39)
    len_call_result_307559 = invoke(stypy.reporting.localization.Localization(__file__, 39, 23), len_307556, *[targets_307557], **kwargs_307558)
    
    # Applying the binary operator '!=' (line 39)
    result_ne_307560 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 7), '!=', len_call_result_307555, len_call_result_307559)
    
    # Testing the type of an if condition (line 39)
    if_condition_307561 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 39, 4), result_ne_307560)
    # Assigning a type to the variable 'if_condition_307561' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'if_condition_307561', if_condition_307561)
    # SSA begins for if statement (line 39)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'ValueError' (line 40)
    ValueError_307562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 14), 'ValueError')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 40, 8), ValueError_307562, 'raise parameter', BaseException)
    # SSA join for if statement (line 39)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 43):
    
    # Obtaining an instance of the builtin type 'list' (line 43)
    list_307563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 43)
    
    # Assigning a type to the variable 'n_sources' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'n_sources', list_307563)
    
    # Assigning a List to a Name (line 44):
    
    # Obtaining an instance of the builtin type 'list' (line 44)
    list_307564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 44)
    
    # Assigning a type to the variable 'n_targets' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'n_targets', list_307564)
    
    
    # Call to zip(...): (line 45)
    # Processing the call arguments (line 45)
    # Getting the type of 'sources' (line 45)
    sources_307566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 30), 'sources', False)
    # Getting the type of 'targets' (line 45)
    targets_307567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 39), 'targets', False)
    # Processing the call keyword arguments (line 45)
    kwargs_307568 = {}
    # Getting the type of 'zip' (line 45)
    zip_307565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 26), 'zip', False)
    # Calling zip(args, kwargs) (line 45)
    zip_call_result_307569 = invoke(stypy.reporting.localization.Localization(__file__, 45, 26), zip_307565, *[sources_307566, targets_307567], **kwargs_307568)
    
    # Testing the type of a for loop iterable (line 45)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 45, 4), zip_call_result_307569)
    # Getting the type of the for loop variable (line 45)
    for_loop_var_307570 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 45, 4), zip_call_result_307569)
    # Assigning a type to the variable 'source' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'source', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 4), for_loop_var_307570))
    # Assigning a type to the variable 'target' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'target', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 4), for_loop_var_307570))
    # SSA begins for a for statement (line 45)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to newer(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'source' (line 46)
    source_307572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 17), 'source', False)
    # Getting the type of 'target' (line 46)
    target_307573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 25), 'target', False)
    # Processing the call keyword arguments (line 46)
    kwargs_307574 = {}
    # Getting the type of 'newer' (line 46)
    newer_307571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 11), 'newer', False)
    # Calling newer(args, kwargs) (line 46)
    newer_call_result_307575 = invoke(stypy.reporting.localization.Localization(__file__, 46, 11), newer_307571, *[source_307572, target_307573], **kwargs_307574)
    
    # Testing the type of an if condition (line 46)
    if_condition_307576 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 46, 8), newer_call_result_307575)
    # Assigning a type to the variable 'if_condition_307576' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'if_condition_307576', if_condition_307576)
    # SSA begins for if statement (line 46)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 47)
    # Processing the call arguments (line 47)
    # Getting the type of 'source' (line 47)
    source_307579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 29), 'source', False)
    # Processing the call keyword arguments (line 47)
    kwargs_307580 = {}
    # Getting the type of 'n_sources' (line 47)
    n_sources_307577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'n_sources', False)
    # Obtaining the member 'append' of a type (line 47)
    append_307578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 12), n_sources_307577, 'append')
    # Calling append(args, kwargs) (line 47)
    append_call_result_307581 = invoke(stypy.reporting.localization.Localization(__file__, 47, 12), append_307578, *[source_307579], **kwargs_307580)
    
    
    # Call to append(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'target' (line 48)
    target_307584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 29), 'target', False)
    # Processing the call keyword arguments (line 48)
    kwargs_307585 = {}
    # Getting the type of 'n_targets' (line 48)
    n_targets_307582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'n_targets', False)
    # Obtaining the member 'append' of a type (line 48)
    append_307583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 12), n_targets_307582, 'append')
    # Calling append(args, kwargs) (line 48)
    append_call_result_307586 = invoke(stypy.reporting.localization.Localization(__file__, 48, 12), append_307583, *[target_307584], **kwargs_307585)
    
    # SSA join for if statement (line 46)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 50)
    tuple_307587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 50)
    # Adding element type (line 50)
    # Getting the type of 'n_sources' (line 50)
    n_sources_307588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 11), 'n_sources')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 11), tuple_307587, n_sources_307588)
    # Adding element type (line 50)
    # Getting the type of 'n_targets' (line 50)
    n_targets_307589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 22), 'n_targets')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 11), tuple_307587, n_targets_307589)
    
    # Assigning a type to the variable 'stypy_return_type' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'stypy_return_type', tuple_307587)
    
    # ################# End of 'newer_pairwise(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'newer_pairwise' in the type store
    # Getting the type of 'stypy_return_type' (line 33)
    stypy_return_type_307590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_307590)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'newer_pairwise'
    return stypy_return_type_307590

# Assigning a type to the variable 'newer_pairwise' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'newer_pairwise', newer_pairwise)

@norecursion
def newer_group(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_307591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 41), 'str', 'error')
    defaults = [str_307591]
    # Create a new context for function 'newer_group'
    module_type_store = module_type_store.open_function_context('newer_group', 52, 0, False)
    
    # Passed parameters checking function
    newer_group.stypy_localization = localization
    newer_group.stypy_type_of_self = None
    newer_group.stypy_type_store = module_type_store
    newer_group.stypy_function_name = 'newer_group'
    newer_group.stypy_param_names_list = ['sources', 'target', 'missing']
    newer_group.stypy_varargs_param_name = None
    newer_group.stypy_kwargs_param_name = None
    newer_group.stypy_call_defaults = defaults
    newer_group.stypy_call_varargs = varargs
    newer_group.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'newer_group', ['sources', 'target', 'missing'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'newer_group', localization, ['sources', 'target', 'missing'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'newer_group(...)' code ##################

    str_307592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, (-1)), 'str', 'Return true if \'target\' is out-of-date with respect to any file\n    listed in \'sources\'.\n\n    In other words, if \'target\' exists and is newer\n    than every file in \'sources\', return false; otherwise return true.\n    \'missing\' controls what we do when a source file is missing; the\n    default ("error") is to blow up with an OSError from inside \'stat()\';\n    if it is "ignore", we silently drop any missing source files; if it is\n    "newer", any missing source files make us assume that \'target\' is\n    out-of-date (this is handy in "dry-run" mode: it\'ll make you pretend to\n    carry out commands that wouldn\'t work because inputs are missing, but\n    that doesn\'t matter because you\'re not actually going to run the\n    commands).\n    ')
    
    
    
    # Call to exists(...): (line 68)
    # Processing the call arguments (line 68)
    # Getting the type of 'target' (line 68)
    target_307596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 26), 'target', False)
    # Processing the call keyword arguments (line 68)
    kwargs_307597 = {}
    # Getting the type of 'os' (line 68)
    os_307593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 68)
    path_307594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 11), os_307593, 'path')
    # Obtaining the member 'exists' of a type (line 68)
    exists_307595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 11), path_307594, 'exists')
    # Calling exists(args, kwargs) (line 68)
    exists_call_result_307598 = invoke(stypy.reporting.localization.Localization(__file__, 68, 11), exists_307595, *[target_307596], **kwargs_307597)
    
    # Applying the 'not' unary operator (line 68)
    result_not__307599 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 7), 'not', exists_call_result_307598)
    
    # Testing the type of an if condition (line 68)
    if_condition_307600 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 68, 4), result_not__307599)
    # Assigning a type to the variable 'if_condition_307600' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'if_condition_307600', if_condition_307600)
    # SSA begins for if statement (line 68)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'True' (line 69)
    True_307601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 15), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'stypy_return_type', True_307601)
    # SSA join for if statement (line 68)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 75):
    
    # Obtaining the type of the subscript
    # Getting the type of 'ST_MTIME' (line 75)
    ST_MTIME_307602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 35), 'ST_MTIME')
    
    # Call to stat(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'target' (line 75)
    target_307605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 27), 'target', False)
    # Processing the call keyword arguments (line 75)
    kwargs_307606 = {}
    # Getting the type of 'os' (line 75)
    os_307603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 19), 'os', False)
    # Obtaining the member 'stat' of a type (line 75)
    stat_307604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 19), os_307603, 'stat')
    # Calling stat(args, kwargs) (line 75)
    stat_call_result_307607 = invoke(stypy.reporting.localization.Localization(__file__, 75, 19), stat_307604, *[target_307605], **kwargs_307606)
    
    # Obtaining the member '__getitem__' of a type (line 75)
    getitem___307608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 19), stat_call_result_307607, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 75)
    subscript_call_result_307609 = invoke(stypy.reporting.localization.Localization(__file__, 75, 19), getitem___307608, ST_MTIME_307602)
    
    # Assigning a type to the variable 'target_mtime' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'target_mtime', subscript_call_result_307609)
    
    # Getting the type of 'sources' (line 77)
    sources_307610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 18), 'sources')
    # Testing the type of a for loop iterable (line 77)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 77, 4), sources_307610)
    # Getting the type of the for loop variable (line 77)
    for_loop_var_307611 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 77, 4), sources_307610)
    # Assigning a type to the variable 'source' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'source', for_loop_var_307611)
    # SSA begins for a for statement (line 77)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Call to exists(...): (line 78)
    # Processing the call arguments (line 78)
    # Getting the type of 'source' (line 78)
    source_307615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 30), 'source', False)
    # Processing the call keyword arguments (line 78)
    kwargs_307616 = {}
    # Getting the type of 'os' (line 78)
    os_307612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 78)
    path_307613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 15), os_307612, 'path')
    # Obtaining the member 'exists' of a type (line 78)
    exists_307614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 15), path_307613, 'exists')
    # Calling exists(args, kwargs) (line 78)
    exists_call_result_307617 = invoke(stypy.reporting.localization.Localization(__file__, 78, 15), exists_307614, *[source_307615], **kwargs_307616)
    
    # Applying the 'not' unary operator (line 78)
    result_not__307618 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 11), 'not', exists_call_result_307617)
    
    # Testing the type of an if condition (line 78)
    if_condition_307619 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 78, 8), result_not__307618)
    # Assigning a type to the variable 'if_condition_307619' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'if_condition_307619', if_condition_307619)
    # SSA begins for if statement (line 78)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'missing' (line 79)
    missing_307620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 15), 'missing')
    str_307621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 26), 'str', 'error')
    # Applying the binary operator '==' (line 79)
    result_eq_307622 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 15), '==', missing_307620, str_307621)
    
    # Testing the type of an if condition (line 79)
    if_condition_307623 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 12), result_eq_307622)
    # Assigning a type to the variable 'if_condition_307623' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'if_condition_307623', if_condition_307623)
    # SSA begins for if statement (line 79)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    pass
    # SSA branch for the else part of an if statement (line 79)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'missing' (line 81)
    missing_307624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 17), 'missing')
    str_307625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 28), 'str', 'ignore')
    # Applying the binary operator '==' (line 81)
    result_eq_307626 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 17), '==', missing_307624, str_307625)
    
    # Testing the type of an if condition (line 81)
    if_condition_307627 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 17), result_eq_307626)
    # Assigning a type to the variable 'if_condition_307627' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 17), 'if_condition_307627', if_condition_307627)
    # SSA begins for if statement (line 81)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA branch for the else part of an if statement (line 81)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'missing' (line 83)
    missing_307628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 17), 'missing')
    str_307629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 28), 'str', 'newer')
    # Applying the binary operator '==' (line 83)
    result_eq_307630 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 17), '==', missing_307628, str_307629)
    
    # Testing the type of an if condition (line 83)
    if_condition_307631 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 17), result_eq_307630)
    # Assigning a type to the variable 'if_condition_307631' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 17), 'if_condition_307631', if_condition_307631)
    # SSA begins for if statement (line 83)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'True' (line 84)
    True_307632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 23), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 16), 'stypy_return_type', True_307632)
    # SSA join for if statement (line 83)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 81)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 79)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 78)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'ST_MTIME' (line 86)
    ST_MTIME_307633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 27), 'ST_MTIME')
    
    # Call to stat(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'source' (line 86)
    source_307636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 19), 'source', False)
    # Processing the call keyword arguments (line 86)
    kwargs_307637 = {}
    # Getting the type of 'os' (line 86)
    os_307634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 11), 'os', False)
    # Obtaining the member 'stat' of a type (line 86)
    stat_307635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 11), os_307634, 'stat')
    # Calling stat(args, kwargs) (line 86)
    stat_call_result_307638 = invoke(stypy.reporting.localization.Localization(__file__, 86, 11), stat_307635, *[source_307636], **kwargs_307637)
    
    # Obtaining the member '__getitem__' of a type (line 86)
    getitem___307639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 11), stat_call_result_307638, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 86)
    subscript_call_result_307640 = invoke(stypy.reporting.localization.Localization(__file__, 86, 11), getitem___307639, ST_MTIME_307633)
    
    # Getting the type of 'target_mtime' (line 86)
    target_mtime_307641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 39), 'target_mtime')
    # Applying the binary operator '>' (line 86)
    result_gt_307642 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 11), '>', subscript_call_result_307640, target_mtime_307641)
    
    # Testing the type of an if condition (line 86)
    if_condition_307643 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 86, 8), result_gt_307642)
    # Assigning a type to the variable 'if_condition_307643' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'if_condition_307643', if_condition_307643)
    # SSA begins for if statement (line 86)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'True' (line 87)
    True_307644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 19), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'stypy_return_type', True_307644)
    # SSA join for if statement (line 86)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'False' (line 89)
    False_307645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 11), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'stypy_return_type', False_307645)
    
    # ################# End of 'newer_group(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'newer_group' in the type store
    # Getting the type of 'stypy_return_type' (line 52)
    stypy_return_type_307646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_307646)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'newer_group'
    return stypy_return_type_307646

# Assigning a type to the variable 'newer_group' (line 52)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'newer_group', newer_group)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
