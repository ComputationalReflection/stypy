
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: A replacement wrapper around the subprocess module, with a number of
3: work-arounds:
4: - Provides a stub implementation of subprocess members on Google App Engine
5:   (which are missing in subprocess).
6: - Use subprocess32, backport from python 3.2 on Linux/Mac work-around for
7:   https://github.com/matplotlib/matplotlib/issues/5314
8: 
9: Instead of importing subprocess, other modules should use this as follows:
10: 
11: from matplotlib.compat import subprocess
12: 
13: This module is safe to import from anywhere within matplotlib.
14: '''
15: 
16: from __future__ import absolute_import    # Required to import subprocess
17: from __future__ import print_function
18: import os
19: import sys
20: if os.name == 'posix' and sys.version_info[0] < 3:
21:     # work around for https://github.com/matplotlib/matplotlib/issues/5314
22:     try:
23:         import subprocess32 as subprocess
24:     except ImportError:
25:         import subprocess
26: else:
27:     import subprocess
28: 
29: __all__ = ['Popen', 'PIPE', 'STDOUT', 'check_output', 'CalledProcessError']
30: 
31: 
32: if hasattr(subprocess, 'Popen'):
33:     Popen = subprocess.Popen
34:     # Assume that it also has the other constants.
35:     PIPE = subprocess.PIPE
36:     STDOUT = subprocess.STDOUT
37:     CalledProcessError = subprocess.CalledProcessError
38:     check_output = subprocess.check_output
39: else:
40:     # In restricted environments (such as Google App Engine), these are
41:     # non-existent. Replace them with dummy versions that always raise OSError.
42:     def Popen(*args, **kwargs):
43:         raise OSError("subprocess.Popen is not supported")
44: 
45:     def check_output(*args, **kwargs):
46:         raise OSError("subprocess.check_output is not supported")
47:     PIPE = -1
48:     STDOUT = -2
49:     # There is no need to catch CalledProcessError. These stubs cannot raise
50:     # it. None in an except clause will simply not match any exceptions.
51:     CalledProcessError = None
52: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_279151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, (-1)), 'str', '\nA replacement wrapper around the subprocess module, with a number of\nwork-arounds:\n- Provides a stub implementation of subprocess members on Google App Engine\n  (which are missing in subprocess).\n- Use subprocess32, backport from python 3.2 on Linux/Mac work-around for\n  https://github.com/matplotlib/matplotlib/issues/5314\n\nInstead of importing subprocess, other modules should use this as follows:\n\nfrom matplotlib.compat import subprocess\n\nThis module is safe to import from anywhere within matplotlib.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 0))

# 'import os' statement (line 18)
import os

import_module(stypy.reporting.localization.Localization(__file__, 18, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'import sys' statement (line 19)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'sys', sys, module_type_store)



# Evaluating a boolean operation

# Getting the type of 'os' (line 20)
os_279152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 3), 'os')
# Obtaining the member 'name' of a type (line 20)
name_279153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 3), os_279152, 'name')
str_279154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 14), 'str', 'posix')
# Applying the binary operator '==' (line 20)
result_eq_279155 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 3), '==', name_279153, str_279154)



# Obtaining the type of the subscript
int_279156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 43), 'int')
# Getting the type of 'sys' (line 20)
sys_279157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 26), 'sys')
# Obtaining the member 'version_info' of a type (line 20)
version_info_279158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 26), sys_279157, 'version_info')
# Obtaining the member '__getitem__' of a type (line 20)
getitem___279159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 26), version_info_279158, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 20)
subscript_call_result_279160 = invoke(stypy.reporting.localization.Localization(__file__, 20, 26), getitem___279159, int_279156)

int_279161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 48), 'int')
# Applying the binary operator '<' (line 20)
result_lt_279162 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 26), '<', subscript_call_result_279160, int_279161)

# Applying the binary operator 'and' (line 20)
result_and_keyword_279163 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 3), 'and', result_eq_279155, result_lt_279162)

# Testing the type of an if condition (line 20)
if_condition_279164 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 20, 0), result_and_keyword_279163)
# Assigning a type to the variable 'if_condition_279164' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'if_condition_279164', if_condition_279164)
# SSA begins for if statement (line 20)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')


# SSA begins for try-except statement (line 22)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 8))

# 'import subprocess32' statement (line 23)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/compat/')
import_279165 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 23, 8), 'subprocess32')

if (type(import_279165) is not StypyTypeError):

    if (import_279165 != 'pyd_module'):
        __import__(import_279165)
        sys_modules_279166 = sys.modules[import_279165]
        import_module(stypy.reporting.localization.Localization(__file__, 23, 8), 'subprocess', sys_modules_279166.module_type_store, module_type_store)
    else:
        import subprocess32 as subprocess

        import_module(stypy.reporting.localization.Localization(__file__, 23, 8), 'subprocess', subprocess32, module_type_store)

else:
    # Assigning a type to the variable 'subprocess32' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'subprocess32', import_279165)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/compat/')

# SSA branch for the except part of a try statement (line 22)
# SSA branch for the except 'ImportError' branch of a try statement (line 22)
module_type_store.open_ssa_branch('except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 8))

# 'import subprocess' statement (line 25)
import subprocess

import_module(stypy.reporting.localization.Localization(__file__, 25, 8), 'subprocess', subprocess, module_type_store)

# SSA join for try-except statement (line 22)
module_type_store = module_type_store.join_ssa_context()

# SSA branch for the else part of an if statement (line 20)
module_type_store.open_ssa_branch('else')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 27, 4))

# 'import subprocess' statement (line 27)
import subprocess

import_module(stypy.reporting.localization.Localization(__file__, 27, 4), 'subprocess', subprocess, module_type_store)

# SSA join for if statement (line 20)
module_type_store = module_type_store.join_ssa_context()


# Assigning a List to a Name (line 29):
__all__ = ['Popen', 'PIPE', 'STDOUT', 'check_output', 'CalledProcessError']
module_type_store.set_exportable_members(['Popen', 'PIPE', 'STDOUT', 'check_output', 'CalledProcessError'])

# Obtaining an instance of the builtin type 'list' (line 29)
list_279167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 29)
# Adding element type (line 29)
str_279168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 11), 'str', 'Popen')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_279167, str_279168)
# Adding element type (line 29)
str_279169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 20), 'str', 'PIPE')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_279167, str_279169)
# Adding element type (line 29)
str_279170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 28), 'str', 'STDOUT')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_279167, str_279170)
# Adding element type (line 29)
str_279171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 38), 'str', 'check_output')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_279167, str_279171)
# Adding element type (line 29)
str_279172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 54), 'str', 'CalledProcessError')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 10), list_279167, str_279172)

# Assigning a type to the variable '__all__' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), '__all__', list_279167)

# Type idiom detected: calculating its left and rigth part (line 32)
str_279173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 23), 'str', 'Popen')
# Getting the type of 'subprocess' (line 32)
subprocess_279174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 11), 'subprocess')

(may_be_279175, more_types_in_union_279176) = may_provide_member(str_279173, subprocess_279174)

if may_be_279175:

    if more_types_in_union_279176:
        # Runtime conditional SSA (line 32)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
    else:
        module_type_store = module_type_store

    # Assigning a type to the variable 'subprocess' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'subprocess', remove_not_member_provider_from_union(subprocess_279174, 'Popen'))
    
    # Assigning a Attribute to a Name (line 33):
    # Getting the type of 'subprocess' (line 33)
    subprocess_279177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'subprocess')
    # Obtaining the member 'Popen' of a type (line 33)
    Popen_279178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 12), subprocess_279177, 'Popen')
    # Assigning a type to the variable 'Popen' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'Popen', Popen_279178)
    
    # Assigning a Attribute to a Name (line 35):
    # Getting the type of 'subprocess' (line 35)
    subprocess_279179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 11), 'subprocess')
    # Obtaining the member 'PIPE' of a type (line 35)
    PIPE_279180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 11), subprocess_279179, 'PIPE')
    # Assigning a type to the variable 'PIPE' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'PIPE', PIPE_279180)
    
    # Assigning a Attribute to a Name (line 36):
    # Getting the type of 'subprocess' (line 36)
    subprocess_279181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 13), 'subprocess')
    # Obtaining the member 'STDOUT' of a type (line 36)
    STDOUT_279182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 13), subprocess_279181, 'STDOUT')
    # Assigning a type to the variable 'STDOUT' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'STDOUT', STDOUT_279182)
    
    # Assigning a Attribute to a Name (line 37):
    # Getting the type of 'subprocess' (line 37)
    subprocess_279183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 25), 'subprocess')
    # Obtaining the member 'CalledProcessError' of a type (line 37)
    CalledProcessError_279184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 25), subprocess_279183, 'CalledProcessError')
    # Assigning a type to the variable 'CalledProcessError' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'CalledProcessError', CalledProcessError_279184)
    
    # Assigning a Attribute to a Name (line 38):
    # Getting the type of 'subprocess' (line 38)
    subprocess_279185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 19), 'subprocess')
    # Obtaining the member 'check_output' of a type (line 38)
    check_output_279186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 19), subprocess_279185, 'check_output')
    # Assigning a type to the variable 'check_output' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'check_output', check_output_279186)

    if more_types_in_union_279176:
        # Runtime conditional SSA for else branch (line 32)
        module_type_store.open_ssa_branch('idiom else')



if ((not may_be_279175) or more_types_in_union_279176):
    # Assigning a type to the variable 'subprocess' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'subprocess', remove_member_provider_from_union(subprocess_279174, 'Popen'))

    @norecursion
    def Popen(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'Popen'
        module_type_store = module_type_store.open_function_context('Popen', 42, 4, False)
        
        # Passed parameters checking function
        Popen.stypy_localization = localization
        Popen.stypy_type_of_self = None
        Popen.stypy_type_store = module_type_store
        Popen.stypy_function_name = 'Popen'
        Popen.stypy_param_names_list = []
        Popen.stypy_varargs_param_name = 'args'
        Popen.stypy_kwargs_param_name = 'kwargs'
        Popen.stypy_call_defaults = defaults
        Popen.stypy_call_varargs = varargs
        Popen.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'Popen', [], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'Popen', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'Popen(...)' code ##################

        
        # Call to OSError(...): (line 43)
        # Processing the call arguments (line 43)
        str_279188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 22), 'str', 'subprocess.Popen is not supported')
        # Processing the call keyword arguments (line 43)
        kwargs_279189 = {}
        # Getting the type of 'OSError' (line 43)
        OSError_279187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 14), 'OSError', False)
        # Calling OSError(args, kwargs) (line 43)
        OSError_call_result_279190 = invoke(stypy.reporting.localization.Localization(__file__, 43, 14), OSError_279187, *[str_279188], **kwargs_279189)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 43, 8), OSError_call_result_279190, 'raise parameter', BaseException)
        
        # ################# End of 'Popen(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'Popen' in the type store
        # Getting the type of 'stypy_return_type' (line 42)
        stypy_return_type_279191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_279191)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'Popen'
        return stypy_return_type_279191

    # Assigning a type to the variable 'Popen' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'Popen', Popen)

    @norecursion
    def check_output(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_output'
        module_type_store = module_type_store.open_function_context('check_output', 45, 4, False)
        
        # Passed parameters checking function
        check_output.stypy_localization = localization
        check_output.stypy_type_of_self = None
        check_output.stypy_type_store = module_type_store
        check_output.stypy_function_name = 'check_output'
        check_output.stypy_param_names_list = []
        check_output.stypy_varargs_param_name = 'args'
        check_output.stypy_kwargs_param_name = 'kwargs'
        check_output.stypy_call_defaults = defaults
        check_output.stypy_call_varargs = varargs
        check_output.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'check_output', [], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_output', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_output(...)' code ##################

        
        # Call to OSError(...): (line 46)
        # Processing the call arguments (line 46)
        str_279193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 22), 'str', 'subprocess.check_output is not supported')
        # Processing the call keyword arguments (line 46)
        kwargs_279194 = {}
        # Getting the type of 'OSError' (line 46)
        OSError_279192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 14), 'OSError', False)
        # Calling OSError(args, kwargs) (line 46)
        OSError_call_result_279195 = invoke(stypy.reporting.localization.Localization(__file__, 46, 14), OSError_279192, *[str_279193], **kwargs_279194)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 46, 8), OSError_call_result_279195, 'raise parameter', BaseException)
        
        # ################# End of 'check_output(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_output' in the type store
        # Getting the type of 'stypy_return_type' (line 45)
        stypy_return_type_279196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_279196)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_output'
        return stypy_return_type_279196

    # Assigning a type to the variable 'check_output' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'check_output', check_output)
    
    # Assigning a Num to a Name (line 47):
    int_279197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 11), 'int')
    # Assigning a type to the variable 'PIPE' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'PIPE', int_279197)
    
    # Assigning a Num to a Name (line 48):
    int_279198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 13), 'int')
    # Assigning a type to the variable 'STDOUT' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'STDOUT', int_279198)
    
    # Assigning a Name to a Name (line 51):
    # Getting the type of 'None' (line 51)
    None_279199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 25), 'None')
    # Assigning a type to the variable 'CalledProcessError' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'CalledProcessError', None_279199)

    if (may_be_279175 and more_types_in_union_279176):
        # SSA join for if statement (line 32)
        module_type_store = module_type_store.join_ssa_context()




# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
