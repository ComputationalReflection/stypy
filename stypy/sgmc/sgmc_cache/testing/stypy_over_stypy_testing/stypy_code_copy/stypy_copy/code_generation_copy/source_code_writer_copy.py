
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import os
2: 
3: 
4: def write_python_source_code(source_file_path, src):
5:     '''
6:     Writes Python source code to the provided file
7:     :param source_file_path: Destination .py file
8:     :param src: Source code
9:     :return:
10:     '''
11:     dirname = os.path.dirname(source_file_path)
12:     if not os.path.exists(dirname):
13:         os.makedirs(dirname)
14: 
15:     with open(source_file_path, 'w') as outfile:
16:         outfile.write(src)
17: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import os' statement (line 1)
import os

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'os', os, module_type_store)


@norecursion
def write_python_source_code(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'write_python_source_code'
    module_type_store = module_type_store.open_function_context('write_python_source_code', 4, 0, False)
    
    # Passed parameters checking function
    write_python_source_code.stypy_localization = localization
    write_python_source_code.stypy_type_of_self = None
    write_python_source_code.stypy_type_store = module_type_store
    write_python_source_code.stypy_function_name = 'write_python_source_code'
    write_python_source_code.stypy_param_names_list = ['source_file_path', 'src']
    write_python_source_code.stypy_varargs_param_name = None
    write_python_source_code.stypy_kwargs_param_name = None
    write_python_source_code.stypy_call_defaults = defaults
    write_python_source_code.stypy_call_varargs = varargs
    write_python_source_code.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'write_python_source_code', ['source_file_path', 'src'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'write_python_source_code', localization, ['source_file_path', 'src'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'write_python_source_code(...)' code ##################

    str_1549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, (-1)), 'str', '\n    Writes Python source code to the provided file\n    :param source_file_path: Destination .py file\n    :param src: Source code\n    :return:\n    ')
    
    # Assigning a Call to a Name (line 11):
    
    # Call to dirname(...): (line 11)
    # Processing the call arguments (line 11)
    # Getting the type of 'source_file_path' (line 11)
    source_file_path_1553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 30), 'source_file_path', False)
    # Processing the call keyword arguments (line 11)
    kwargs_1554 = {}
    # Getting the type of 'os' (line 11)
    os_1550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 14), 'os', False)
    # Obtaining the member 'path' of a type (line 11)
    path_1551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 14), os_1550, 'path')
    # Obtaining the member 'dirname' of a type (line 11)
    dirname_1552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 14), path_1551, 'dirname')
    # Calling dirname(args, kwargs) (line 11)
    dirname_call_result_1555 = invoke(stypy.reporting.localization.Localization(__file__, 11, 14), dirname_1552, *[source_file_path_1553], **kwargs_1554)
    
    # Assigning a type to the variable 'dirname' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'dirname', dirname_call_result_1555)
    
    
    # Call to exists(...): (line 12)
    # Processing the call arguments (line 12)
    # Getting the type of 'dirname' (line 12)
    dirname_1559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 26), 'dirname', False)
    # Processing the call keyword arguments (line 12)
    kwargs_1560 = {}
    # Getting the type of 'os' (line 12)
    os_1556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 12)
    path_1557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 11), os_1556, 'path')
    # Obtaining the member 'exists' of a type (line 12)
    exists_1558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 11), path_1557, 'exists')
    # Calling exists(args, kwargs) (line 12)
    exists_call_result_1561 = invoke(stypy.reporting.localization.Localization(__file__, 12, 11), exists_1558, *[dirname_1559], **kwargs_1560)
    
    # Applying the 'not' unary operator (line 12)
    result_not__1562 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 7), 'not', exists_call_result_1561)
    
    # Testing if the type of an if condition is none (line 12)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 12, 4), result_not__1562):
        pass
    else:
        
        # Testing the type of an if condition (line 12)
        if_condition_1563 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 12, 4), result_not__1562)
        # Assigning a type to the variable 'if_condition_1563' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'if_condition_1563', if_condition_1563)
        # SSA begins for if statement (line 12)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to makedirs(...): (line 13)
        # Processing the call arguments (line 13)
        # Getting the type of 'dirname' (line 13)
        dirname_1566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 20), 'dirname', False)
        # Processing the call keyword arguments (line 13)
        kwargs_1567 = {}
        # Getting the type of 'os' (line 13)
        os_1564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'os', False)
        # Obtaining the member 'makedirs' of a type (line 13)
        makedirs_1565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 8), os_1564, 'makedirs')
        # Calling makedirs(args, kwargs) (line 13)
        makedirs_call_result_1568 = invoke(stypy.reporting.localization.Localization(__file__, 13, 8), makedirs_1565, *[dirname_1566], **kwargs_1567)
        
        # SSA join for if statement (line 12)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to open(...): (line 15)
    # Processing the call arguments (line 15)
    # Getting the type of 'source_file_path' (line 15)
    source_file_path_1570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 14), 'source_file_path', False)
    str_1571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 32), 'str', 'w')
    # Processing the call keyword arguments (line 15)
    kwargs_1572 = {}
    # Getting the type of 'open' (line 15)
    open_1569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 9), 'open', False)
    # Calling open(args, kwargs) (line 15)
    open_call_result_1573 = invoke(stypy.reporting.localization.Localization(__file__, 15, 9), open_1569, *[source_file_path_1570, str_1571], **kwargs_1572)
    
    with_1574 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 15, 9), open_call_result_1573, 'with parameter', '__enter__', '__exit__')

    if with_1574:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 15)
        enter___1575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 9), open_call_result_1573, '__enter__')
        with_enter_1576 = invoke(stypy.reporting.localization.Localization(__file__, 15, 9), enter___1575)
        # Assigning a type to the variable 'outfile' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 9), 'outfile', with_enter_1576)
        
        # Call to write(...): (line 16)
        # Processing the call arguments (line 16)
        # Getting the type of 'src' (line 16)
        src_1579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 22), 'src', False)
        # Processing the call keyword arguments (line 16)
        kwargs_1580 = {}
        # Getting the type of 'outfile' (line 16)
        outfile_1577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'outfile', False)
        # Obtaining the member 'write' of a type (line 16)
        write_1578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), outfile_1577, 'write')
        # Calling write(args, kwargs) (line 16)
        write_call_result_1581 = invoke(stypy.reporting.localization.Localization(__file__, 16, 8), write_1578, *[src_1579], **kwargs_1580)
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 15)
        exit___1582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 9), open_call_result_1573, '__exit__')
        with_exit_1583 = invoke(stypy.reporting.localization.Localization(__file__, 15, 9), exit___1582, None, None, None)

    
    # ################# End of 'write_python_source_code(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'write_python_source_code' in the type store
    # Getting the type of 'stypy_return_type' (line 4)
    stypy_return_type_1584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1584)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'write_python_source_code'
    return stypy_return_type_1584

# Assigning a type to the variable 'write_python_source_code' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'write_python_source_code', write_python_source_code)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
