
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import os
2: import sys
3: import unittest
4: 
5: 
6: here = os.path.dirname(__file__)
7: loader = unittest.defaultTestLoader
8: 
9: def suite():
10:     suite = unittest.TestSuite()
11:     for fn in os.listdir(here):
12:         if fn.startswith("test") and fn.endswith(".py"):
13:             modname = "unittest.test." + fn[:-3]
14:             __import__(modname)
15:             module = sys.modules[modname]
16:             suite.addTest(loader.loadTestsFromModule(module))
17:     return suite
18: 
19: 
20: if __name__ == "__main__":
21:     unittest.main(defaultTest="suite")
22: 

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

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import sys' statement (line 2)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import unittest' statement (line 3)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'unittest', unittest, module_type_store)


# Assigning a Call to a Name (line 6):

# Call to dirname(...): (line 6)
# Processing the call arguments (line 6)
# Getting the type of '__file__' (line 6)
file___208999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 23), '__file__', False)
# Processing the call keyword arguments (line 6)
kwargs_209000 = {}
# Getting the type of 'os' (line 6)
os_208996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 7), 'os', False)
# Obtaining the member 'path' of a type (line 6)
path_208997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 7), os_208996, 'path')
# Obtaining the member 'dirname' of a type (line 6)
dirname_208998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 7), path_208997, 'dirname')
# Calling dirname(args, kwargs) (line 6)
dirname_call_result_209001 = invoke(stypy.reporting.localization.Localization(__file__, 6, 7), dirname_208998, *[file___208999], **kwargs_209000)

# Assigning a type to the variable 'here' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'here', dirname_call_result_209001)

# Assigning a Attribute to a Name (line 7):
# Getting the type of 'unittest' (line 7)
unittest_209002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 9), 'unittest')
# Obtaining the member 'defaultTestLoader' of a type (line 7)
defaultTestLoader_209003 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 9), unittest_209002, 'defaultTestLoader')
# Assigning a type to the variable 'loader' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'loader', defaultTestLoader_209003)

@norecursion
def suite(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'suite'
    module_type_store = module_type_store.open_function_context('suite', 9, 0, False)
    
    # Passed parameters checking function
    suite.stypy_localization = localization
    suite.stypy_type_of_self = None
    suite.stypy_type_store = module_type_store
    suite.stypy_function_name = 'suite'
    suite.stypy_param_names_list = []
    suite.stypy_varargs_param_name = None
    suite.stypy_kwargs_param_name = None
    suite.stypy_call_defaults = defaults
    suite.stypy_call_varargs = varargs
    suite.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'suite', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'suite', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'suite(...)' code ##################

    
    # Assigning a Call to a Name (line 10):
    
    # Call to TestSuite(...): (line 10)
    # Processing the call keyword arguments (line 10)
    kwargs_209006 = {}
    # Getting the type of 'unittest' (line 10)
    unittest_209004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 12), 'unittest', False)
    # Obtaining the member 'TestSuite' of a type (line 10)
    TestSuite_209005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 12), unittest_209004, 'TestSuite')
    # Calling TestSuite(args, kwargs) (line 10)
    TestSuite_call_result_209007 = invoke(stypy.reporting.localization.Localization(__file__, 10, 12), TestSuite_209005, *[], **kwargs_209006)
    
    # Assigning a type to the variable 'suite' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'suite', TestSuite_call_result_209007)
    
    
    # Call to listdir(...): (line 11)
    # Processing the call arguments (line 11)
    # Getting the type of 'here' (line 11)
    here_209010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 25), 'here', False)
    # Processing the call keyword arguments (line 11)
    kwargs_209011 = {}
    # Getting the type of 'os' (line 11)
    os_209008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 14), 'os', False)
    # Obtaining the member 'listdir' of a type (line 11)
    listdir_209009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 14), os_209008, 'listdir')
    # Calling listdir(args, kwargs) (line 11)
    listdir_call_result_209012 = invoke(stypy.reporting.localization.Localization(__file__, 11, 14), listdir_209009, *[here_209010], **kwargs_209011)
    
    # Testing the type of a for loop iterable (line 11)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 11, 4), listdir_call_result_209012)
    # Getting the type of the for loop variable (line 11)
    for_loop_var_209013 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 11, 4), listdir_call_result_209012)
    # Assigning a type to the variable 'fn' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'fn', for_loop_var_209013)
    # SSA begins for a for statement (line 11)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    # Call to startswith(...): (line 12)
    # Processing the call arguments (line 12)
    str_209016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 25), 'str', 'test')
    # Processing the call keyword arguments (line 12)
    kwargs_209017 = {}
    # Getting the type of 'fn' (line 12)
    fn_209014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 11), 'fn', False)
    # Obtaining the member 'startswith' of a type (line 12)
    startswith_209015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 11), fn_209014, 'startswith')
    # Calling startswith(args, kwargs) (line 12)
    startswith_call_result_209018 = invoke(stypy.reporting.localization.Localization(__file__, 12, 11), startswith_209015, *[str_209016], **kwargs_209017)
    
    
    # Call to endswith(...): (line 12)
    # Processing the call arguments (line 12)
    str_209021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 49), 'str', '.py')
    # Processing the call keyword arguments (line 12)
    kwargs_209022 = {}
    # Getting the type of 'fn' (line 12)
    fn_209019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 37), 'fn', False)
    # Obtaining the member 'endswith' of a type (line 12)
    endswith_209020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 37), fn_209019, 'endswith')
    # Calling endswith(args, kwargs) (line 12)
    endswith_call_result_209023 = invoke(stypy.reporting.localization.Localization(__file__, 12, 37), endswith_209020, *[str_209021], **kwargs_209022)
    
    # Applying the binary operator 'and' (line 12)
    result_and_keyword_209024 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 11), 'and', startswith_call_result_209018, endswith_call_result_209023)
    
    # Testing the type of an if condition (line 12)
    if_condition_209025 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 12, 8), result_and_keyword_209024)
    # Assigning a type to the variable 'if_condition_209025' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'if_condition_209025', if_condition_209025)
    # SSA begins for if statement (line 12)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 13):
    str_209026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 22), 'str', 'unittest.test.')
    
    # Obtaining the type of the subscript
    int_209027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 45), 'int')
    slice_209028 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 13, 41), None, int_209027, None)
    # Getting the type of 'fn' (line 13)
    fn_209029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 41), 'fn')
    # Obtaining the member '__getitem__' of a type (line 13)
    getitem___209030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 41), fn_209029, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 13)
    subscript_call_result_209031 = invoke(stypy.reporting.localization.Localization(__file__, 13, 41), getitem___209030, slice_209028)
    
    # Applying the binary operator '+' (line 13)
    result_add_209032 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 22), '+', str_209026, subscript_call_result_209031)
    
    # Assigning a type to the variable 'modname' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 12), 'modname', result_add_209032)
    
    # Call to __import__(...): (line 14)
    # Processing the call arguments (line 14)
    # Getting the type of 'modname' (line 14)
    modname_209034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 23), 'modname', False)
    # Processing the call keyword arguments (line 14)
    kwargs_209035 = {}
    # Getting the type of '__import__' (line 14)
    import___209033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 12), '__import__', False)
    # Calling __import__(args, kwargs) (line 14)
    import___call_result_209036 = invoke(stypy.reporting.localization.Localization(__file__, 14, 12), import___209033, *[modname_209034], **kwargs_209035)
    
    
    # Assigning a Subscript to a Name (line 15):
    
    # Obtaining the type of the subscript
    # Getting the type of 'modname' (line 15)
    modname_209037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 33), 'modname')
    # Getting the type of 'sys' (line 15)
    sys_209038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 21), 'sys')
    # Obtaining the member 'modules' of a type (line 15)
    modules_209039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 21), sys_209038, 'modules')
    # Obtaining the member '__getitem__' of a type (line 15)
    getitem___209040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 21), modules_209039, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 15)
    subscript_call_result_209041 = invoke(stypy.reporting.localization.Localization(__file__, 15, 21), getitem___209040, modname_209037)
    
    # Assigning a type to the variable 'module' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), 'module', subscript_call_result_209041)
    
    # Call to addTest(...): (line 16)
    # Processing the call arguments (line 16)
    
    # Call to loadTestsFromModule(...): (line 16)
    # Processing the call arguments (line 16)
    # Getting the type of 'module' (line 16)
    module_209046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 53), 'module', False)
    # Processing the call keyword arguments (line 16)
    kwargs_209047 = {}
    # Getting the type of 'loader' (line 16)
    loader_209044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 26), 'loader', False)
    # Obtaining the member 'loadTestsFromModule' of a type (line 16)
    loadTestsFromModule_209045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 26), loader_209044, 'loadTestsFromModule')
    # Calling loadTestsFromModule(args, kwargs) (line 16)
    loadTestsFromModule_call_result_209048 = invoke(stypy.reporting.localization.Localization(__file__, 16, 26), loadTestsFromModule_209045, *[module_209046], **kwargs_209047)
    
    # Processing the call keyword arguments (line 16)
    kwargs_209049 = {}
    # Getting the type of 'suite' (line 16)
    suite_209042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 12), 'suite', False)
    # Obtaining the member 'addTest' of a type (line 16)
    addTest_209043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 12), suite_209042, 'addTest')
    # Calling addTest(args, kwargs) (line 16)
    addTest_call_result_209050 = invoke(stypy.reporting.localization.Localization(__file__, 16, 12), addTest_209043, *[loadTestsFromModule_call_result_209048], **kwargs_209049)
    
    # SSA join for if statement (line 12)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'suite' (line 17)
    suite_209051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 11), 'suite')
    # Assigning a type to the variable 'stypy_return_type' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type', suite_209051)
    
    # ################# End of 'suite(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'suite' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_209052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_209052)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'suite'
    return stypy_return_type_209052

# Assigning a type to the variable 'suite' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'suite', suite)

if (__name__ == '__main__'):
    
    # Call to main(...): (line 21)
    # Processing the call keyword arguments (line 21)
    str_209055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 30), 'str', 'suite')
    keyword_209056 = str_209055
    kwargs_209057 = {'defaultTest': keyword_209056}
    # Getting the type of 'unittest' (line 21)
    unittest_209053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'unittest', False)
    # Obtaining the member 'main' of a type (line 21)
    main_209054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 4), unittest_209053, 'main')
    # Calling main(args, kwargs) (line 21)
    main_call_result_209058 = invoke(stypy.reporting.localization.Localization(__file__, 21, 4), main_209054, *[], **kwargs_209057)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
