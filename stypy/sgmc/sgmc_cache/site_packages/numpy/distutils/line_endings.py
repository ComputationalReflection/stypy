
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Functions for converting from DOS to UNIX line endings
2: 
3: '''
4: from __future__ import division, absolute_import, print_function
5: 
6: import sys, re, os
7: 
8: def dos2unix(file):
9:     "Replace CRLF with LF in argument files.  Print names of changed files."
10:     if os.path.isdir(file):
11:         print(file, "Directory!")
12:         return
13: 
14:     data = open(file, "rb").read()
15:     if '\0' in data:
16:         print(file, "Binary!")
17:         return
18: 
19:     newdata = re.sub("\r\n", "\n", data)
20:     if newdata != data:
21:         print('dos2unix:', file)
22:         f = open(file, "wb")
23:         f.write(newdata)
24:         f.close()
25:         return file
26:     else:
27:         print(file, 'ok')
28: 
29: def dos2unix_one_dir(modified_files, dir_name, file_names):
30:     for file in file_names:
31:         full_path = os.path.join(dir_name, file)
32:         file = dos2unix(full_path)
33:         if file is not None:
34:             modified_files.append(file)
35: 
36: def dos2unix_dir(dir_name):
37:     modified_files = []
38:     os.path.walk(dir_name, dos2unix_one_dir, modified_files)
39:     return modified_files
40: #----------------------------------
41: 
42: def unix2dos(file):
43:     "Replace LF with CRLF in argument files.  Print names of changed files."
44:     if os.path.isdir(file):
45:         print(file, "Directory!")
46:         return
47: 
48:     data = open(file, "rb").read()
49:     if '\0' in data:
50:         print(file, "Binary!")
51:         return
52:     newdata = re.sub("\r\n", "\n", data)
53:     newdata = re.sub("\n", "\r\n", newdata)
54:     if newdata != data:
55:         print('unix2dos:', file)
56:         f = open(file, "wb")
57:         f.write(newdata)
58:         f.close()
59:         return file
60:     else:
61:         print(file, 'ok')
62: 
63: def unix2dos_one_dir(modified_files, dir_name, file_names):
64:     for file in file_names:
65:         full_path = os.path.join(dir_name, file)
66:         unix2dos(full_path)
67:         if file is not None:
68:             modified_files.append(file)
69: 
70: def unix2dos_dir(dir_name):
71:     modified_files = []
72:     os.path.walk(dir_name, unix2dos_one_dir, modified_files)
73:     return modified_files
74: 
75: if __name__ == "__main__":
76:     dos2unix_dir(sys.argv[1])
77: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_36456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', ' Functions for converting from DOS to UNIX line endings\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# Multiple import statement. import sys (1/3) (line 6)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'sys', sys, module_type_store)
# Multiple import statement. import re (2/3) (line 6)
import re

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 're', re, module_type_store)
# Multiple import statement. import os (3/3) (line 6)
import os

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'os', os, module_type_store)


@norecursion
def dos2unix(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'dos2unix'
    module_type_store = module_type_store.open_function_context('dos2unix', 8, 0, False)
    
    # Passed parameters checking function
    dos2unix.stypy_localization = localization
    dos2unix.stypy_type_of_self = None
    dos2unix.stypy_type_store = module_type_store
    dos2unix.stypy_function_name = 'dos2unix'
    dos2unix.stypy_param_names_list = ['file']
    dos2unix.stypy_varargs_param_name = None
    dos2unix.stypy_kwargs_param_name = None
    dos2unix.stypy_call_defaults = defaults
    dos2unix.stypy_call_varargs = varargs
    dos2unix.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'dos2unix', ['file'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'dos2unix', localization, ['file'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'dos2unix(...)' code ##################

    str_36457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 4), 'str', 'Replace CRLF with LF in argument files.  Print names of changed files.')
    
    
    # Call to isdir(...): (line 10)
    # Processing the call arguments (line 10)
    # Getting the type of 'file' (line 10)
    file_36461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 21), 'file', False)
    # Processing the call keyword arguments (line 10)
    kwargs_36462 = {}
    # Getting the type of 'os' (line 10)
    os_36458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 7), 'os', False)
    # Obtaining the member 'path' of a type (line 10)
    path_36459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 7), os_36458, 'path')
    # Obtaining the member 'isdir' of a type (line 10)
    isdir_36460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 7), path_36459, 'isdir')
    # Calling isdir(args, kwargs) (line 10)
    isdir_call_result_36463 = invoke(stypy.reporting.localization.Localization(__file__, 10, 7), isdir_36460, *[file_36461], **kwargs_36462)
    
    # Testing the type of an if condition (line 10)
    if_condition_36464 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 10, 4), isdir_call_result_36463)
    # Assigning a type to the variable 'if_condition_36464' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'if_condition_36464', if_condition_36464)
    # SSA begins for if statement (line 10)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 11)
    # Processing the call arguments (line 11)
    # Getting the type of 'file' (line 11)
    file_36466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 14), 'file', False)
    str_36467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 20), 'str', 'Directory!')
    # Processing the call keyword arguments (line 11)
    kwargs_36468 = {}
    # Getting the type of 'print' (line 11)
    print_36465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'print', False)
    # Calling print(args, kwargs) (line 11)
    print_call_result_36469 = invoke(stypy.reporting.localization.Localization(__file__, 11, 8), print_36465, *[file_36466, str_36467], **kwargs_36468)
    
    # Assigning a type to the variable 'stypy_return_type' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 10)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 14):
    
    # Call to read(...): (line 14)
    # Processing the call keyword arguments (line 14)
    kwargs_36476 = {}
    
    # Call to open(...): (line 14)
    # Processing the call arguments (line 14)
    # Getting the type of 'file' (line 14)
    file_36471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 16), 'file', False)
    str_36472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 22), 'str', 'rb')
    # Processing the call keyword arguments (line 14)
    kwargs_36473 = {}
    # Getting the type of 'open' (line 14)
    open_36470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 11), 'open', False)
    # Calling open(args, kwargs) (line 14)
    open_call_result_36474 = invoke(stypy.reporting.localization.Localization(__file__, 14, 11), open_36470, *[file_36471, str_36472], **kwargs_36473)
    
    # Obtaining the member 'read' of a type (line 14)
    read_36475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 11), open_call_result_36474, 'read')
    # Calling read(args, kwargs) (line 14)
    read_call_result_36477 = invoke(stypy.reporting.localization.Localization(__file__, 14, 11), read_36475, *[], **kwargs_36476)
    
    # Assigning a type to the variable 'data' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'data', read_call_result_36477)
    
    
    str_36478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 7), 'str', '\x00')
    # Getting the type of 'data' (line 15)
    data_36479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 15), 'data')
    # Applying the binary operator 'in' (line 15)
    result_contains_36480 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 7), 'in', str_36478, data_36479)
    
    # Testing the type of an if condition (line 15)
    if_condition_36481 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 15, 4), result_contains_36480)
    # Assigning a type to the variable 'if_condition_36481' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'if_condition_36481', if_condition_36481)
    # SSA begins for if statement (line 15)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 16)
    # Processing the call arguments (line 16)
    # Getting the type of 'file' (line 16)
    file_36483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 14), 'file', False)
    str_36484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 20), 'str', 'Binary!')
    # Processing the call keyword arguments (line 16)
    kwargs_36485 = {}
    # Getting the type of 'print' (line 16)
    print_36482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'print', False)
    # Calling print(args, kwargs) (line 16)
    print_call_result_36486 = invoke(stypy.reporting.localization.Localization(__file__, 16, 8), print_36482, *[file_36483, str_36484], **kwargs_36485)
    
    # Assigning a type to the variable 'stypy_return_type' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 15)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 19):
    
    # Call to sub(...): (line 19)
    # Processing the call arguments (line 19)
    str_36489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 21), 'str', '\r\n')
    str_36490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 29), 'str', '\n')
    # Getting the type of 'data' (line 19)
    data_36491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 35), 'data', False)
    # Processing the call keyword arguments (line 19)
    kwargs_36492 = {}
    # Getting the type of 're' (line 19)
    re_36487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 14), 're', False)
    # Obtaining the member 'sub' of a type (line 19)
    sub_36488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 14), re_36487, 'sub')
    # Calling sub(args, kwargs) (line 19)
    sub_call_result_36493 = invoke(stypy.reporting.localization.Localization(__file__, 19, 14), sub_36488, *[str_36489, str_36490, data_36491], **kwargs_36492)
    
    # Assigning a type to the variable 'newdata' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'newdata', sub_call_result_36493)
    
    
    # Getting the type of 'newdata' (line 20)
    newdata_36494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 7), 'newdata')
    # Getting the type of 'data' (line 20)
    data_36495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 18), 'data')
    # Applying the binary operator '!=' (line 20)
    result_ne_36496 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 7), '!=', newdata_36494, data_36495)
    
    # Testing the type of an if condition (line 20)
    if_condition_36497 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 20, 4), result_ne_36496)
    # Assigning a type to the variable 'if_condition_36497' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'if_condition_36497', if_condition_36497)
    # SSA begins for if statement (line 20)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 21)
    # Processing the call arguments (line 21)
    str_36499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 14), 'str', 'dos2unix:')
    # Getting the type of 'file' (line 21)
    file_36500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 27), 'file', False)
    # Processing the call keyword arguments (line 21)
    kwargs_36501 = {}
    # Getting the type of 'print' (line 21)
    print_36498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'print', False)
    # Calling print(args, kwargs) (line 21)
    print_call_result_36502 = invoke(stypy.reporting.localization.Localization(__file__, 21, 8), print_36498, *[str_36499, file_36500], **kwargs_36501)
    
    
    # Assigning a Call to a Name (line 22):
    
    # Call to open(...): (line 22)
    # Processing the call arguments (line 22)
    # Getting the type of 'file' (line 22)
    file_36504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 17), 'file', False)
    str_36505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 23), 'str', 'wb')
    # Processing the call keyword arguments (line 22)
    kwargs_36506 = {}
    # Getting the type of 'open' (line 22)
    open_36503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'open', False)
    # Calling open(args, kwargs) (line 22)
    open_call_result_36507 = invoke(stypy.reporting.localization.Localization(__file__, 22, 12), open_36503, *[file_36504, str_36505], **kwargs_36506)
    
    # Assigning a type to the variable 'f' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'f', open_call_result_36507)
    
    # Call to write(...): (line 23)
    # Processing the call arguments (line 23)
    # Getting the type of 'newdata' (line 23)
    newdata_36510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 16), 'newdata', False)
    # Processing the call keyword arguments (line 23)
    kwargs_36511 = {}
    # Getting the type of 'f' (line 23)
    f_36508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'f', False)
    # Obtaining the member 'write' of a type (line 23)
    write_36509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), f_36508, 'write')
    # Calling write(args, kwargs) (line 23)
    write_call_result_36512 = invoke(stypy.reporting.localization.Localization(__file__, 23, 8), write_36509, *[newdata_36510], **kwargs_36511)
    
    
    # Call to close(...): (line 24)
    # Processing the call keyword arguments (line 24)
    kwargs_36515 = {}
    # Getting the type of 'f' (line 24)
    f_36513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'f', False)
    # Obtaining the member 'close' of a type (line 24)
    close_36514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), f_36513, 'close')
    # Calling close(args, kwargs) (line 24)
    close_call_result_36516 = invoke(stypy.reporting.localization.Localization(__file__, 24, 8), close_36514, *[], **kwargs_36515)
    
    # Getting the type of 'file' (line 25)
    file_36517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'file')
    # Assigning a type to the variable 'stypy_return_type' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'stypy_return_type', file_36517)
    # SSA branch for the else part of an if statement (line 20)
    module_type_store.open_ssa_branch('else')
    
    # Call to print(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'file' (line 27)
    file_36519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 14), 'file', False)
    str_36520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 20), 'str', 'ok')
    # Processing the call keyword arguments (line 27)
    kwargs_36521 = {}
    # Getting the type of 'print' (line 27)
    print_36518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'print', False)
    # Calling print(args, kwargs) (line 27)
    print_call_result_36522 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), print_36518, *[file_36519, str_36520], **kwargs_36521)
    
    # SSA join for if statement (line 20)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'dos2unix(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'dos2unix' in the type store
    # Getting the type of 'stypy_return_type' (line 8)
    stypy_return_type_36523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_36523)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'dos2unix'
    return stypy_return_type_36523

# Assigning a type to the variable 'dos2unix' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'dos2unix', dos2unix)

@norecursion
def dos2unix_one_dir(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'dos2unix_one_dir'
    module_type_store = module_type_store.open_function_context('dos2unix_one_dir', 29, 0, False)
    
    # Passed parameters checking function
    dos2unix_one_dir.stypy_localization = localization
    dos2unix_one_dir.stypy_type_of_self = None
    dos2unix_one_dir.stypy_type_store = module_type_store
    dos2unix_one_dir.stypy_function_name = 'dos2unix_one_dir'
    dos2unix_one_dir.stypy_param_names_list = ['modified_files', 'dir_name', 'file_names']
    dos2unix_one_dir.stypy_varargs_param_name = None
    dos2unix_one_dir.stypy_kwargs_param_name = None
    dos2unix_one_dir.stypy_call_defaults = defaults
    dos2unix_one_dir.stypy_call_varargs = varargs
    dos2unix_one_dir.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'dos2unix_one_dir', ['modified_files', 'dir_name', 'file_names'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'dos2unix_one_dir', localization, ['modified_files', 'dir_name', 'file_names'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'dos2unix_one_dir(...)' code ##################

    
    # Getting the type of 'file_names' (line 30)
    file_names_36524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 16), 'file_names')
    # Testing the type of a for loop iterable (line 30)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 30, 4), file_names_36524)
    # Getting the type of the for loop variable (line 30)
    for_loop_var_36525 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 30, 4), file_names_36524)
    # Assigning a type to the variable 'file' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'file', for_loop_var_36525)
    # SSA begins for a for statement (line 30)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 31):
    
    # Call to join(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'dir_name' (line 31)
    dir_name_36529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 33), 'dir_name', False)
    # Getting the type of 'file' (line 31)
    file_36530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 43), 'file', False)
    # Processing the call keyword arguments (line 31)
    kwargs_36531 = {}
    # Getting the type of 'os' (line 31)
    os_36526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 20), 'os', False)
    # Obtaining the member 'path' of a type (line 31)
    path_36527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 20), os_36526, 'path')
    # Obtaining the member 'join' of a type (line 31)
    join_36528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 20), path_36527, 'join')
    # Calling join(args, kwargs) (line 31)
    join_call_result_36532 = invoke(stypy.reporting.localization.Localization(__file__, 31, 20), join_36528, *[dir_name_36529, file_36530], **kwargs_36531)
    
    # Assigning a type to the variable 'full_path' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'full_path', join_call_result_36532)
    
    # Assigning a Call to a Name (line 32):
    
    # Call to dos2unix(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'full_path' (line 32)
    full_path_36534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 24), 'full_path', False)
    # Processing the call keyword arguments (line 32)
    kwargs_36535 = {}
    # Getting the type of 'dos2unix' (line 32)
    dos2unix_36533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 15), 'dos2unix', False)
    # Calling dos2unix(args, kwargs) (line 32)
    dos2unix_call_result_36536 = invoke(stypy.reporting.localization.Localization(__file__, 32, 15), dos2unix_36533, *[full_path_36534], **kwargs_36535)
    
    # Assigning a type to the variable 'file' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'file', dos2unix_call_result_36536)
    
    # Type idiom detected: calculating its left and rigth part (line 33)
    # Getting the type of 'file' (line 33)
    file_36537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'file')
    # Getting the type of 'None' (line 33)
    None_36538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 23), 'None')
    
    (may_be_36539, more_types_in_union_36540) = may_not_be_none(file_36537, None_36538)

    if may_be_36539:

        if more_types_in_union_36540:
            # Runtime conditional SSA (line 33)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to append(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'file' (line 34)
        file_36543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 34), 'file', False)
        # Processing the call keyword arguments (line 34)
        kwargs_36544 = {}
        # Getting the type of 'modified_files' (line 34)
        modified_files_36541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'modified_files', False)
        # Obtaining the member 'append' of a type (line 34)
        append_36542 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 12), modified_files_36541, 'append')
        # Calling append(args, kwargs) (line 34)
        append_call_result_36545 = invoke(stypy.reporting.localization.Localization(__file__, 34, 12), append_36542, *[file_36543], **kwargs_36544)
        

        if more_types_in_union_36540:
            # SSA join for if statement (line 33)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'dos2unix_one_dir(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'dos2unix_one_dir' in the type store
    # Getting the type of 'stypy_return_type' (line 29)
    stypy_return_type_36546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_36546)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'dos2unix_one_dir'
    return stypy_return_type_36546

# Assigning a type to the variable 'dos2unix_one_dir' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'dos2unix_one_dir', dos2unix_one_dir)

@norecursion
def dos2unix_dir(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'dos2unix_dir'
    module_type_store = module_type_store.open_function_context('dos2unix_dir', 36, 0, False)
    
    # Passed parameters checking function
    dos2unix_dir.stypy_localization = localization
    dos2unix_dir.stypy_type_of_self = None
    dos2unix_dir.stypy_type_store = module_type_store
    dos2unix_dir.stypy_function_name = 'dos2unix_dir'
    dos2unix_dir.stypy_param_names_list = ['dir_name']
    dos2unix_dir.stypy_varargs_param_name = None
    dos2unix_dir.stypy_kwargs_param_name = None
    dos2unix_dir.stypy_call_defaults = defaults
    dos2unix_dir.stypy_call_varargs = varargs
    dos2unix_dir.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'dos2unix_dir', ['dir_name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'dos2unix_dir', localization, ['dir_name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'dos2unix_dir(...)' code ##################

    
    # Assigning a List to a Name (line 37):
    
    # Obtaining an instance of the builtin type 'list' (line 37)
    list_36547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 37)
    
    # Assigning a type to the variable 'modified_files' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'modified_files', list_36547)
    
    # Call to walk(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'dir_name' (line 38)
    dir_name_36551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 17), 'dir_name', False)
    # Getting the type of 'dos2unix_one_dir' (line 38)
    dos2unix_one_dir_36552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 27), 'dos2unix_one_dir', False)
    # Getting the type of 'modified_files' (line 38)
    modified_files_36553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 45), 'modified_files', False)
    # Processing the call keyword arguments (line 38)
    kwargs_36554 = {}
    # Getting the type of 'os' (line 38)
    os_36548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'os', False)
    # Obtaining the member 'path' of a type (line 38)
    path_36549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 4), os_36548, 'path')
    # Obtaining the member 'walk' of a type (line 38)
    walk_36550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 4), path_36549, 'walk')
    # Calling walk(args, kwargs) (line 38)
    walk_call_result_36555 = invoke(stypy.reporting.localization.Localization(__file__, 38, 4), walk_36550, *[dir_name_36551, dos2unix_one_dir_36552, modified_files_36553], **kwargs_36554)
    
    # Getting the type of 'modified_files' (line 39)
    modified_files_36556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 11), 'modified_files')
    # Assigning a type to the variable 'stypy_return_type' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'stypy_return_type', modified_files_36556)
    
    # ################# End of 'dos2unix_dir(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'dos2unix_dir' in the type store
    # Getting the type of 'stypy_return_type' (line 36)
    stypy_return_type_36557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_36557)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'dos2unix_dir'
    return stypy_return_type_36557

# Assigning a type to the variable 'dos2unix_dir' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 0), 'dos2unix_dir', dos2unix_dir)

@norecursion
def unix2dos(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'unix2dos'
    module_type_store = module_type_store.open_function_context('unix2dos', 42, 0, False)
    
    # Passed parameters checking function
    unix2dos.stypy_localization = localization
    unix2dos.stypy_type_of_self = None
    unix2dos.stypy_type_store = module_type_store
    unix2dos.stypy_function_name = 'unix2dos'
    unix2dos.stypy_param_names_list = ['file']
    unix2dos.stypy_varargs_param_name = None
    unix2dos.stypy_kwargs_param_name = None
    unix2dos.stypy_call_defaults = defaults
    unix2dos.stypy_call_varargs = varargs
    unix2dos.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'unix2dos', ['file'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'unix2dos', localization, ['file'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'unix2dos(...)' code ##################

    str_36558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 4), 'str', 'Replace LF with CRLF in argument files.  Print names of changed files.')
    
    
    # Call to isdir(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'file' (line 44)
    file_36562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 21), 'file', False)
    # Processing the call keyword arguments (line 44)
    kwargs_36563 = {}
    # Getting the type of 'os' (line 44)
    os_36559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 7), 'os', False)
    # Obtaining the member 'path' of a type (line 44)
    path_36560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 7), os_36559, 'path')
    # Obtaining the member 'isdir' of a type (line 44)
    isdir_36561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 7), path_36560, 'isdir')
    # Calling isdir(args, kwargs) (line 44)
    isdir_call_result_36564 = invoke(stypy.reporting.localization.Localization(__file__, 44, 7), isdir_36561, *[file_36562], **kwargs_36563)
    
    # Testing the type of an if condition (line 44)
    if_condition_36565 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 44, 4), isdir_call_result_36564)
    # Assigning a type to the variable 'if_condition_36565' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'if_condition_36565', if_condition_36565)
    # SSA begins for if statement (line 44)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 45)
    # Processing the call arguments (line 45)
    # Getting the type of 'file' (line 45)
    file_36567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 14), 'file', False)
    str_36568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 20), 'str', 'Directory!')
    # Processing the call keyword arguments (line 45)
    kwargs_36569 = {}
    # Getting the type of 'print' (line 45)
    print_36566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'print', False)
    # Calling print(args, kwargs) (line 45)
    print_call_result_36570 = invoke(stypy.reporting.localization.Localization(__file__, 45, 8), print_36566, *[file_36567, str_36568], **kwargs_36569)
    
    # Assigning a type to the variable 'stypy_return_type' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 44)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 48):
    
    # Call to read(...): (line 48)
    # Processing the call keyword arguments (line 48)
    kwargs_36577 = {}
    
    # Call to open(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'file' (line 48)
    file_36572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 16), 'file', False)
    str_36573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 22), 'str', 'rb')
    # Processing the call keyword arguments (line 48)
    kwargs_36574 = {}
    # Getting the type of 'open' (line 48)
    open_36571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 11), 'open', False)
    # Calling open(args, kwargs) (line 48)
    open_call_result_36575 = invoke(stypy.reporting.localization.Localization(__file__, 48, 11), open_36571, *[file_36572, str_36573], **kwargs_36574)
    
    # Obtaining the member 'read' of a type (line 48)
    read_36576 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 11), open_call_result_36575, 'read')
    # Calling read(args, kwargs) (line 48)
    read_call_result_36578 = invoke(stypy.reporting.localization.Localization(__file__, 48, 11), read_36576, *[], **kwargs_36577)
    
    # Assigning a type to the variable 'data' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'data', read_call_result_36578)
    
    
    str_36579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 7), 'str', '\x00')
    # Getting the type of 'data' (line 49)
    data_36580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 15), 'data')
    # Applying the binary operator 'in' (line 49)
    result_contains_36581 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 7), 'in', str_36579, data_36580)
    
    # Testing the type of an if condition (line 49)
    if_condition_36582 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 4), result_contains_36581)
    # Assigning a type to the variable 'if_condition_36582' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'if_condition_36582', if_condition_36582)
    # SSA begins for if statement (line 49)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 50)
    # Processing the call arguments (line 50)
    # Getting the type of 'file' (line 50)
    file_36584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 14), 'file', False)
    str_36585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 20), 'str', 'Binary!')
    # Processing the call keyword arguments (line 50)
    kwargs_36586 = {}
    # Getting the type of 'print' (line 50)
    print_36583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'print', False)
    # Calling print(args, kwargs) (line 50)
    print_call_result_36587 = invoke(stypy.reporting.localization.Localization(__file__, 50, 8), print_36583, *[file_36584, str_36585], **kwargs_36586)
    
    # Assigning a type to the variable 'stypy_return_type' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 49)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 52):
    
    # Call to sub(...): (line 52)
    # Processing the call arguments (line 52)
    str_36590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 21), 'str', '\r\n')
    str_36591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 29), 'str', '\n')
    # Getting the type of 'data' (line 52)
    data_36592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 35), 'data', False)
    # Processing the call keyword arguments (line 52)
    kwargs_36593 = {}
    # Getting the type of 're' (line 52)
    re_36588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 14), 're', False)
    # Obtaining the member 'sub' of a type (line 52)
    sub_36589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 14), re_36588, 'sub')
    # Calling sub(args, kwargs) (line 52)
    sub_call_result_36594 = invoke(stypy.reporting.localization.Localization(__file__, 52, 14), sub_36589, *[str_36590, str_36591, data_36592], **kwargs_36593)
    
    # Assigning a type to the variable 'newdata' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'newdata', sub_call_result_36594)
    
    # Assigning a Call to a Name (line 53):
    
    # Call to sub(...): (line 53)
    # Processing the call arguments (line 53)
    str_36597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 21), 'str', '\n')
    str_36598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 27), 'str', '\r\n')
    # Getting the type of 'newdata' (line 53)
    newdata_36599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 35), 'newdata', False)
    # Processing the call keyword arguments (line 53)
    kwargs_36600 = {}
    # Getting the type of 're' (line 53)
    re_36595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 14), 're', False)
    # Obtaining the member 'sub' of a type (line 53)
    sub_36596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 14), re_36595, 'sub')
    # Calling sub(args, kwargs) (line 53)
    sub_call_result_36601 = invoke(stypy.reporting.localization.Localization(__file__, 53, 14), sub_36596, *[str_36597, str_36598, newdata_36599], **kwargs_36600)
    
    # Assigning a type to the variable 'newdata' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'newdata', sub_call_result_36601)
    
    
    # Getting the type of 'newdata' (line 54)
    newdata_36602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 7), 'newdata')
    # Getting the type of 'data' (line 54)
    data_36603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 18), 'data')
    # Applying the binary operator '!=' (line 54)
    result_ne_36604 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 7), '!=', newdata_36602, data_36603)
    
    # Testing the type of an if condition (line 54)
    if_condition_36605 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 54, 4), result_ne_36604)
    # Assigning a type to the variable 'if_condition_36605' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'if_condition_36605', if_condition_36605)
    # SSA begins for if statement (line 54)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 55)
    # Processing the call arguments (line 55)
    str_36607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 14), 'str', 'unix2dos:')
    # Getting the type of 'file' (line 55)
    file_36608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 27), 'file', False)
    # Processing the call keyword arguments (line 55)
    kwargs_36609 = {}
    # Getting the type of 'print' (line 55)
    print_36606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'print', False)
    # Calling print(args, kwargs) (line 55)
    print_call_result_36610 = invoke(stypy.reporting.localization.Localization(__file__, 55, 8), print_36606, *[str_36607, file_36608], **kwargs_36609)
    
    
    # Assigning a Call to a Name (line 56):
    
    # Call to open(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'file' (line 56)
    file_36612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 17), 'file', False)
    str_36613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 23), 'str', 'wb')
    # Processing the call keyword arguments (line 56)
    kwargs_36614 = {}
    # Getting the type of 'open' (line 56)
    open_36611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'open', False)
    # Calling open(args, kwargs) (line 56)
    open_call_result_36615 = invoke(stypy.reporting.localization.Localization(__file__, 56, 12), open_36611, *[file_36612, str_36613], **kwargs_36614)
    
    # Assigning a type to the variable 'f' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'f', open_call_result_36615)
    
    # Call to write(...): (line 57)
    # Processing the call arguments (line 57)
    # Getting the type of 'newdata' (line 57)
    newdata_36618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 16), 'newdata', False)
    # Processing the call keyword arguments (line 57)
    kwargs_36619 = {}
    # Getting the type of 'f' (line 57)
    f_36616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'f', False)
    # Obtaining the member 'write' of a type (line 57)
    write_36617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), f_36616, 'write')
    # Calling write(args, kwargs) (line 57)
    write_call_result_36620 = invoke(stypy.reporting.localization.Localization(__file__, 57, 8), write_36617, *[newdata_36618], **kwargs_36619)
    
    
    # Call to close(...): (line 58)
    # Processing the call keyword arguments (line 58)
    kwargs_36623 = {}
    # Getting the type of 'f' (line 58)
    f_36621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'f', False)
    # Obtaining the member 'close' of a type (line 58)
    close_36622 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), f_36621, 'close')
    # Calling close(args, kwargs) (line 58)
    close_call_result_36624 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), close_36622, *[], **kwargs_36623)
    
    # Getting the type of 'file' (line 59)
    file_36625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 15), 'file')
    # Assigning a type to the variable 'stypy_return_type' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'stypy_return_type', file_36625)
    # SSA branch for the else part of an if statement (line 54)
    module_type_store.open_ssa_branch('else')
    
    # Call to print(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'file' (line 61)
    file_36627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 14), 'file', False)
    str_36628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 20), 'str', 'ok')
    # Processing the call keyword arguments (line 61)
    kwargs_36629 = {}
    # Getting the type of 'print' (line 61)
    print_36626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'print', False)
    # Calling print(args, kwargs) (line 61)
    print_call_result_36630 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), print_36626, *[file_36627, str_36628], **kwargs_36629)
    
    # SSA join for if statement (line 54)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'unix2dos(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'unix2dos' in the type store
    # Getting the type of 'stypy_return_type' (line 42)
    stypy_return_type_36631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_36631)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'unix2dos'
    return stypy_return_type_36631

# Assigning a type to the variable 'unix2dos' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), 'unix2dos', unix2dos)

@norecursion
def unix2dos_one_dir(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'unix2dos_one_dir'
    module_type_store = module_type_store.open_function_context('unix2dos_one_dir', 63, 0, False)
    
    # Passed parameters checking function
    unix2dos_one_dir.stypy_localization = localization
    unix2dos_one_dir.stypy_type_of_self = None
    unix2dos_one_dir.stypy_type_store = module_type_store
    unix2dos_one_dir.stypy_function_name = 'unix2dos_one_dir'
    unix2dos_one_dir.stypy_param_names_list = ['modified_files', 'dir_name', 'file_names']
    unix2dos_one_dir.stypy_varargs_param_name = None
    unix2dos_one_dir.stypy_kwargs_param_name = None
    unix2dos_one_dir.stypy_call_defaults = defaults
    unix2dos_one_dir.stypy_call_varargs = varargs
    unix2dos_one_dir.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'unix2dos_one_dir', ['modified_files', 'dir_name', 'file_names'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'unix2dos_one_dir', localization, ['modified_files', 'dir_name', 'file_names'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'unix2dos_one_dir(...)' code ##################

    
    # Getting the type of 'file_names' (line 64)
    file_names_36632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 16), 'file_names')
    # Testing the type of a for loop iterable (line 64)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 64, 4), file_names_36632)
    # Getting the type of the for loop variable (line 64)
    for_loop_var_36633 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 64, 4), file_names_36632)
    # Assigning a type to the variable 'file' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'file', for_loop_var_36633)
    # SSA begins for a for statement (line 64)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 65):
    
    # Call to join(...): (line 65)
    # Processing the call arguments (line 65)
    # Getting the type of 'dir_name' (line 65)
    dir_name_36637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 33), 'dir_name', False)
    # Getting the type of 'file' (line 65)
    file_36638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 43), 'file', False)
    # Processing the call keyword arguments (line 65)
    kwargs_36639 = {}
    # Getting the type of 'os' (line 65)
    os_36634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 20), 'os', False)
    # Obtaining the member 'path' of a type (line 65)
    path_36635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 20), os_36634, 'path')
    # Obtaining the member 'join' of a type (line 65)
    join_36636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 20), path_36635, 'join')
    # Calling join(args, kwargs) (line 65)
    join_call_result_36640 = invoke(stypy.reporting.localization.Localization(__file__, 65, 20), join_36636, *[dir_name_36637, file_36638], **kwargs_36639)
    
    # Assigning a type to the variable 'full_path' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'full_path', join_call_result_36640)
    
    # Call to unix2dos(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 'full_path' (line 66)
    full_path_36642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 17), 'full_path', False)
    # Processing the call keyword arguments (line 66)
    kwargs_36643 = {}
    # Getting the type of 'unix2dos' (line 66)
    unix2dos_36641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'unix2dos', False)
    # Calling unix2dos(args, kwargs) (line 66)
    unix2dos_call_result_36644 = invoke(stypy.reporting.localization.Localization(__file__, 66, 8), unix2dos_36641, *[full_path_36642], **kwargs_36643)
    
    
    # Type idiom detected: calculating its left and rigth part (line 67)
    # Getting the type of 'file' (line 67)
    file_36645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'file')
    # Getting the type of 'None' (line 67)
    None_36646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 23), 'None')
    
    (may_be_36647, more_types_in_union_36648) = may_not_be_none(file_36645, None_36646)

    if may_be_36647:

        if more_types_in_union_36648:
            # Runtime conditional SSA (line 67)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to append(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'file' (line 68)
        file_36651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 34), 'file', False)
        # Processing the call keyword arguments (line 68)
        kwargs_36652 = {}
        # Getting the type of 'modified_files' (line 68)
        modified_files_36649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'modified_files', False)
        # Obtaining the member 'append' of a type (line 68)
        append_36650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 12), modified_files_36649, 'append')
        # Calling append(args, kwargs) (line 68)
        append_call_result_36653 = invoke(stypy.reporting.localization.Localization(__file__, 68, 12), append_36650, *[file_36651], **kwargs_36652)
        

        if more_types_in_union_36648:
            # SSA join for if statement (line 67)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'unix2dos_one_dir(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'unix2dos_one_dir' in the type store
    # Getting the type of 'stypy_return_type' (line 63)
    stypy_return_type_36654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_36654)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'unix2dos_one_dir'
    return stypy_return_type_36654

# Assigning a type to the variable 'unix2dos_one_dir' (line 63)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'unix2dos_one_dir', unix2dos_one_dir)

@norecursion
def unix2dos_dir(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'unix2dos_dir'
    module_type_store = module_type_store.open_function_context('unix2dos_dir', 70, 0, False)
    
    # Passed parameters checking function
    unix2dos_dir.stypy_localization = localization
    unix2dos_dir.stypy_type_of_self = None
    unix2dos_dir.stypy_type_store = module_type_store
    unix2dos_dir.stypy_function_name = 'unix2dos_dir'
    unix2dos_dir.stypy_param_names_list = ['dir_name']
    unix2dos_dir.stypy_varargs_param_name = None
    unix2dos_dir.stypy_kwargs_param_name = None
    unix2dos_dir.stypy_call_defaults = defaults
    unix2dos_dir.stypy_call_varargs = varargs
    unix2dos_dir.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'unix2dos_dir', ['dir_name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'unix2dos_dir', localization, ['dir_name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'unix2dos_dir(...)' code ##################

    
    # Assigning a List to a Name (line 71):
    
    # Obtaining an instance of the builtin type 'list' (line 71)
    list_36655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 71)
    
    # Assigning a type to the variable 'modified_files' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'modified_files', list_36655)
    
    # Call to walk(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'dir_name' (line 72)
    dir_name_36659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 17), 'dir_name', False)
    # Getting the type of 'unix2dos_one_dir' (line 72)
    unix2dos_one_dir_36660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 27), 'unix2dos_one_dir', False)
    # Getting the type of 'modified_files' (line 72)
    modified_files_36661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 45), 'modified_files', False)
    # Processing the call keyword arguments (line 72)
    kwargs_36662 = {}
    # Getting the type of 'os' (line 72)
    os_36656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'os', False)
    # Obtaining the member 'path' of a type (line 72)
    path_36657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 4), os_36656, 'path')
    # Obtaining the member 'walk' of a type (line 72)
    walk_36658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 4), path_36657, 'walk')
    # Calling walk(args, kwargs) (line 72)
    walk_call_result_36663 = invoke(stypy.reporting.localization.Localization(__file__, 72, 4), walk_36658, *[dir_name_36659, unix2dos_one_dir_36660, modified_files_36661], **kwargs_36662)
    
    # Getting the type of 'modified_files' (line 73)
    modified_files_36664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 11), 'modified_files')
    # Assigning a type to the variable 'stypy_return_type' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'stypy_return_type', modified_files_36664)
    
    # ################# End of 'unix2dos_dir(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'unix2dos_dir' in the type store
    # Getting the type of 'stypy_return_type' (line 70)
    stypy_return_type_36665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_36665)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'unix2dos_dir'
    return stypy_return_type_36665

# Assigning a type to the variable 'unix2dos_dir' (line 70)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'unix2dos_dir', unix2dos_dir)

if (__name__ == '__main__'):
    
    # Call to dos2unix_dir(...): (line 76)
    # Processing the call arguments (line 76)
    
    # Obtaining the type of the subscript
    int_36667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 26), 'int')
    # Getting the type of 'sys' (line 76)
    sys_36668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 17), 'sys', False)
    # Obtaining the member 'argv' of a type (line 76)
    argv_36669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 17), sys_36668, 'argv')
    # Obtaining the member '__getitem__' of a type (line 76)
    getitem___36670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 17), argv_36669, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 76)
    subscript_call_result_36671 = invoke(stypy.reporting.localization.Localization(__file__, 76, 17), getitem___36670, int_36667)
    
    # Processing the call keyword arguments (line 76)
    kwargs_36672 = {}
    # Getting the type of 'dos2unix_dir' (line 76)
    dos2unix_dir_36666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'dos2unix_dir', False)
    # Calling dos2unix_dir(args, kwargs) (line 76)
    dos2unix_dir_call_result_36673 = invoke(stypy.reporting.localization.Localization(__file__, 76, 4), dos2unix_dir_36666, *[subscript_call_result_36671], **kwargs_36672)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
