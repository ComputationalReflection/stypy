
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Generic test utilities.
3: 
4: '''
5: 
6: from __future__ import division, print_function, absolute_import
7: 
8: import os
9: import re
10: import sys
11: 
12: 
13: __all__ = ['PytestTester', 'check_free_memory']
14: 
15: 
16: class FPUModeChangeWarning(RuntimeWarning):
17:     '''Warning about FPU mode change'''
18:     pass
19: 
20: 
21: class PytestTester(object):
22:     '''
23:     Pytest test runner entry point.
24:     '''
25: 
26:     def __init__(self, module_name):
27:         self.module_name = module_name
28: 
29:     def __call__(self, label="fast", verbose=1, extra_argv=None, doctests=False,
30:                  coverage=False, tests=None):
31:         import pytest
32: 
33:         module = sys.modules[self.module_name]
34:         module_path = os.path.abspath(module.__path__[0])
35: 
36:         pytest_args = ['-l']
37: 
38:         if doctests:
39:             raise ValueError("Doctests not supported")
40: 
41:         if extra_argv:
42:             pytest_args += list(extra_argv)
43: 
44:         if verbose and int(verbose) > 1:
45:             pytest_args += ["-" + "v"*(int(verbose)-1)]
46: 
47:         if coverage:
48:             pytest_args += ["--cov=" + module_path]
49: 
50:         if label == "fast":
51:             pytest_args += ["-m", "not slow"]
52:         elif label != "full":
53:             pytest_args += ["-m", label]
54: 
55:         if tests is None:
56:             tests = [self.module_name]
57: 
58:         pytest_args += ['--pyargs'] + list(tests)
59: 
60:         try:
61:             code = pytest.main(pytest_args)
62:         except SystemExit as exc:
63:             code = exc.code
64: 
65:         return (code == 0)
66: 
67: 
68: def check_free_memory(free_mb):
69:     '''
70:     Check *free_mb* of memory is available, otherwise do pytest.skip
71:     '''
72:     import pytest
73: 
74:     try:
75:         mem_free = _parse_size(os.environ['SCIPY_AVAILABLE_MEM'])
76:         msg = '{0} MB memory required, but environment SCIPY_AVAILABLE_MEM={1}'.format(
77:             free_mb, os.environ['SCIPY_AVAILABLE_MEM'])
78:     except KeyError:
79:         mem_free = _get_mem_available()
80:         if mem_free is None:
81:             pytest.skip("Could not determine available memory; set SCIPY_AVAILABLE_MEM "
82:                         "variable to free memory in MB to run the test.")
83:         msg = '{0} MB memory required, but {1} MB available'.format(
84:             free_mb, mem_free/1e6)
85: 
86:     if mem_free < free_mb * 1e6:
87:         pytest.skip(msg)
88: 
89: 
90: def _parse_size(size_str):
91:     suffixes = {'': 1e6,
92:                 'b': 1.0,
93:                 'k': 1e3, 'M': 1e6, 'G': 1e9, 'T': 1e12,
94:                 'kb': 1e3, 'Mb': 1e6, 'Gb': 1e9, 'Tb': 1e12,
95:                 'kib': 1024.0, 'Mib': 1024.0**2, 'Gib': 1024.0**3, 'Tib': 1024.0**4}
96:     m = re.match(r'^\s*(\d+)\s*({0})\s*$'.format('|'.join(suffixes.keys())),
97:                  size_str,
98:                  re.I)
99:     if not m or m.group(2) not in suffixes:
100:         raise ValueError("Invalid size string")
101: 
102:     return float(m.group(1)) * suffixes[m.group(2)]
103: 
104: 
105: def _get_mem_available():
106:     '''
107:     Get information about memory available, not counting swap.
108:     '''
109:     try:
110:         import psutil
111:         return psutil.virtual_memory().available
112:     except (ImportError, AttributeError):
113:         pass
114: 
115:     if sys.platform.startswith('linux'):
116:         info = {}
117:         with open('/proc/meminfo', 'r') as f:
118:             for line in f:
119:                 p = line.split()
120:                 info[p[0].strip(':').lower()] = float(p[1]) * 1e3
121: 
122:         if 'memavailable' in info:
123:             # Linux >= 3.14
124:             return info['memavailable']
125:         else:
126:             return info['memfree'] + info['cached']
127: 
128:     return None
129: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_709586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', '\nGeneric test utilities.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import os' statement (line 8)
import os

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import re' statement (line 9)
import re

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import sys' statement (line 10)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'sys', sys, module_type_store)


# Assigning a List to a Name (line 13):
__all__ = ['PytestTester', 'check_free_memory']
module_type_store.set_exportable_members(['PytestTester', 'check_free_memory'])

# Obtaining an instance of the builtin type 'list' (line 13)
list_709587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 13)
# Adding element type (line 13)
str_709588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 11), 'str', 'PytestTester')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_709587, str_709588)
# Adding element type (line 13)
str_709589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 27), 'str', 'check_free_memory')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 10), list_709587, str_709589)

# Assigning a type to the variable '__all__' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), '__all__', list_709587)
# Declaration of the 'FPUModeChangeWarning' class
# Getting the type of 'RuntimeWarning' (line 16)
RuntimeWarning_709590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 27), 'RuntimeWarning')

class FPUModeChangeWarning(RuntimeWarning_709590, ):
    str_709591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 4), 'str', 'Warning about FPU mode change')
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 16, 0, False)
        # Assigning a type to the variable 'self' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FPUModeChangeWarning.__init__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'FPUModeChangeWarning' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'FPUModeChangeWarning', FPUModeChangeWarning)
# Declaration of the 'PytestTester' class

class PytestTester(object, ):
    str_709592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, (-1)), 'str', '\n    Pytest test runner entry point.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 26, 4, False)
        # Assigning a type to the variable 'self' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PytestTester.__init__', ['module_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['module_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 27):
        # Getting the type of 'module_name' (line 27)
        module_name_709593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 27), 'module_name')
        # Getting the type of 'self' (line 27)
        self_709594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'self')
        # Setting the type of the member 'module_name' of a type (line 27)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), self_709594, 'module_name', module_name_709593)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_709595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 29), 'str', 'fast')
        int_709596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 45), 'int')
        # Getting the type of 'None' (line 29)
        None_709597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 59), 'None')
        # Getting the type of 'False' (line 29)
        False_709598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 74), 'False')
        # Getting the type of 'False' (line 30)
        False_709599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 26), 'False')
        # Getting the type of 'None' (line 30)
        None_709600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 39), 'None')
        defaults = [str_709595, int_709596, None_709597, False_709598, False_709599, None_709600]
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 29, 4, False)
        # Assigning a type to the variable 'self' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PytestTester.__call__.__dict__.__setitem__('stypy_localization', localization)
        PytestTester.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PytestTester.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        PytestTester.__call__.__dict__.__setitem__('stypy_function_name', 'PytestTester.__call__')
        PytestTester.__call__.__dict__.__setitem__('stypy_param_names_list', ['label', 'verbose', 'extra_argv', 'doctests', 'coverage', 'tests'])
        PytestTester.__call__.__dict__.__setitem__('stypy_varargs_param_name', None)
        PytestTester.__call__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PytestTester.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        PytestTester.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        PytestTester.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PytestTester.__call__.__dict__.__setitem__('stypy_declared_arg_number', 7)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PytestTester.__call__', ['label', 'verbose', 'extra_argv', 'doctests', 'coverage', 'tests'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['label', 'verbose', 'extra_argv', 'doctests', 'coverage', 'tests'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 31, 8))
        
        # 'import pytest' statement (line 31)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/')
        import_709601 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 31, 8), 'pytest')

        if (type(import_709601) is not StypyTypeError):

            if (import_709601 != 'pyd_module'):
                __import__(import_709601)
                sys_modules_709602 = sys.modules[import_709601]
                import_module(stypy.reporting.localization.Localization(__file__, 31, 8), 'pytest', sys_modules_709602.module_type_store, module_type_store)
            else:
                import pytest

                import_module(stypy.reporting.localization.Localization(__file__, 31, 8), 'pytest', pytest, module_type_store)

        else:
            # Assigning a type to the variable 'pytest' (line 31)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'pytest', import_709601)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/')
        
        
        # Assigning a Subscript to a Name (line 33):
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 33)
        self_709603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 29), 'self')
        # Obtaining the member 'module_name' of a type (line 33)
        module_name_709604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 29), self_709603, 'module_name')
        # Getting the type of 'sys' (line 33)
        sys_709605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 17), 'sys')
        # Obtaining the member 'modules' of a type (line 33)
        modules_709606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 17), sys_709605, 'modules')
        # Obtaining the member '__getitem__' of a type (line 33)
        getitem___709607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 17), modules_709606, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 33)
        subscript_call_result_709608 = invoke(stypy.reporting.localization.Localization(__file__, 33, 17), getitem___709607, module_name_709604)
        
        # Assigning a type to the variable 'module' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'module', subscript_call_result_709608)
        
        # Assigning a Call to a Name (line 34):
        
        # Call to abspath(...): (line 34)
        # Processing the call arguments (line 34)
        
        # Obtaining the type of the subscript
        int_709612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 54), 'int')
        # Getting the type of 'module' (line 34)
        module_709613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 38), 'module', False)
        # Obtaining the member '__path__' of a type (line 34)
        path___709614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 38), module_709613, '__path__')
        # Obtaining the member '__getitem__' of a type (line 34)
        getitem___709615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 38), path___709614, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 34)
        subscript_call_result_709616 = invoke(stypy.reporting.localization.Localization(__file__, 34, 38), getitem___709615, int_709612)
        
        # Processing the call keyword arguments (line 34)
        kwargs_709617 = {}
        # Getting the type of 'os' (line 34)
        os_709609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 22), 'os', False)
        # Obtaining the member 'path' of a type (line 34)
        path_709610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 22), os_709609, 'path')
        # Obtaining the member 'abspath' of a type (line 34)
        abspath_709611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 22), path_709610, 'abspath')
        # Calling abspath(args, kwargs) (line 34)
        abspath_call_result_709618 = invoke(stypy.reporting.localization.Localization(__file__, 34, 22), abspath_709611, *[subscript_call_result_709616], **kwargs_709617)
        
        # Assigning a type to the variable 'module_path' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'module_path', abspath_call_result_709618)
        
        # Assigning a List to a Name (line 36):
        
        # Obtaining an instance of the builtin type 'list' (line 36)
        list_709619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 36)
        # Adding element type (line 36)
        str_709620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 23), 'str', '-l')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 22), list_709619, str_709620)
        
        # Assigning a type to the variable 'pytest_args' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'pytest_args', list_709619)
        
        # Getting the type of 'doctests' (line 38)
        doctests_709621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 11), 'doctests')
        # Testing the type of an if condition (line 38)
        if_condition_709622 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 38, 8), doctests_709621)
        # Assigning a type to the variable 'if_condition_709622' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'if_condition_709622', if_condition_709622)
        # SSA begins for if statement (line 38)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 39)
        # Processing the call arguments (line 39)
        str_709624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 29), 'str', 'Doctests not supported')
        # Processing the call keyword arguments (line 39)
        kwargs_709625 = {}
        # Getting the type of 'ValueError' (line 39)
        ValueError_709623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 39)
        ValueError_call_result_709626 = invoke(stypy.reporting.localization.Localization(__file__, 39, 18), ValueError_709623, *[str_709624], **kwargs_709625)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 39, 12), ValueError_call_result_709626, 'raise parameter', BaseException)
        # SSA join for if statement (line 38)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'extra_argv' (line 41)
        extra_argv_709627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 11), 'extra_argv')
        # Testing the type of an if condition (line 41)
        if_condition_709628 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 8), extra_argv_709627)
        # Assigning a type to the variable 'if_condition_709628' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'if_condition_709628', if_condition_709628)
        # SSA begins for if statement (line 41)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'pytest_args' (line 42)
        pytest_args_709629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'pytest_args')
        
        # Call to list(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'extra_argv' (line 42)
        extra_argv_709631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 32), 'extra_argv', False)
        # Processing the call keyword arguments (line 42)
        kwargs_709632 = {}
        # Getting the type of 'list' (line 42)
        list_709630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 27), 'list', False)
        # Calling list(args, kwargs) (line 42)
        list_call_result_709633 = invoke(stypy.reporting.localization.Localization(__file__, 42, 27), list_709630, *[extra_argv_709631], **kwargs_709632)
        
        # Applying the binary operator '+=' (line 42)
        result_iadd_709634 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 12), '+=', pytest_args_709629, list_call_result_709633)
        # Assigning a type to the variable 'pytest_args' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'pytest_args', result_iadd_709634)
        
        # SSA join for if statement (line 41)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'verbose' (line 44)
        verbose_709635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 11), 'verbose')
        
        
        # Call to int(...): (line 44)
        # Processing the call arguments (line 44)
        # Getting the type of 'verbose' (line 44)
        verbose_709637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 27), 'verbose', False)
        # Processing the call keyword arguments (line 44)
        kwargs_709638 = {}
        # Getting the type of 'int' (line 44)
        int_709636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 23), 'int', False)
        # Calling int(args, kwargs) (line 44)
        int_call_result_709639 = invoke(stypy.reporting.localization.Localization(__file__, 44, 23), int_709636, *[verbose_709637], **kwargs_709638)
        
        int_709640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 38), 'int')
        # Applying the binary operator '>' (line 44)
        result_gt_709641 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 23), '>', int_call_result_709639, int_709640)
        
        # Applying the binary operator 'and' (line 44)
        result_and_keyword_709642 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 11), 'and', verbose_709635, result_gt_709641)
        
        # Testing the type of an if condition (line 44)
        if_condition_709643 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 44, 8), result_and_keyword_709642)
        # Assigning a type to the variable 'if_condition_709643' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'if_condition_709643', if_condition_709643)
        # SSA begins for if statement (line 44)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'pytest_args' (line 45)
        pytest_args_709644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'pytest_args')
        
        # Obtaining an instance of the builtin type 'list' (line 45)
        list_709645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 45)
        # Adding element type (line 45)
        str_709646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 28), 'str', '-')
        str_709647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 34), 'str', 'v')
        
        # Call to int(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'verbose' (line 45)
        verbose_709649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 43), 'verbose', False)
        # Processing the call keyword arguments (line 45)
        kwargs_709650 = {}
        # Getting the type of 'int' (line 45)
        int_709648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 39), 'int', False)
        # Calling int(args, kwargs) (line 45)
        int_call_result_709651 = invoke(stypy.reporting.localization.Localization(__file__, 45, 39), int_709648, *[verbose_709649], **kwargs_709650)
        
        int_709652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 52), 'int')
        # Applying the binary operator '-' (line 45)
        result_sub_709653 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 39), '-', int_call_result_709651, int_709652)
        
        # Applying the binary operator '*' (line 45)
        result_mul_709654 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 34), '*', str_709647, result_sub_709653)
        
        # Applying the binary operator '+' (line 45)
        result_add_709655 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 28), '+', str_709646, result_mul_709654)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 27), list_709645, result_add_709655)
        
        # Applying the binary operator '+=' (line 45)
        result_iadd_709656 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 12), '+=', pytest_args_709644, list_709645)
        # Assigning a type to the variable 'pytest_args' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'pytest_args', result_iadd_709656)
        
        # SSA join for if statement (line 44)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'coverage' (line 47)
        coverage_709657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 11), 'coverage')
        # Testing the type of an if condition (line 47)
        if_condition_709658 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 47, 8), coverage_709657)
        # Assigning a type to the variable 'if_condition_709658' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'if_condition_709658', if_condition_709658)
        # SSA begins for if statement (line 47)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'pytest_args' (line 48)
        pytest_args_709659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'pytest_args')
        
        # Obtaining an instance of the builtin type 'list' (line 48)
        list_709660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 48)
        # Adding element type (line 48)
        str_709661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 28), 'str', '--cov=')
        # Getting the type of 'module_path' (line 48)
        module_path_709662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 39), 'module_path')
        # Applying the binary operator '+' (line 48)
        result_add_709663 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 28), '+', str_709661, module_path_709662)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 27), list_709660, result_add_709663)
        
        # Applying the binary operator '+=' (line 48)
        result_iadd_709664 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 12), '+=', pytest_args_709659, list_709660)
        # Assigning a type to the variable 'pytest_args' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'pytest_args', result_iadd_709664)
        
        # SSA join for if statement (line 47)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'label' (line 50)
        label_709665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 11), 'label')
        str_709666 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 20), 'str', 'fast')
        # Applying the binary operator '==' (line 50)
        result_eq_709667 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 11), '==', label_709665, str_709666)
        
        # Testing the type of an if condition (line 50)
        if_condition_709668 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 8), result_eq_709667)
        # Assigning a type to the variable 'if_condition_709668' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'if_condition_709668', if_condition_709668)
        # SSA begins for if statement (line 50)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'pytest_args' (line 51)
        pytest_args_709669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'pytest_args')
        
        # Obtaining an instance of the builtin type 'list' (line 51)
        list_709670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 51)
        # Adding element type (line 51)
        str_709671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 28), 'str', '-m')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 27), list_709670, str_709671)
        # Adding element type (line 51)
        str_709672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 34), 'str', 'not slow')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 27), list_709670, str_709672)
        
        # Applying the binary operator '+=' (line 51)
        result_iadd_709673 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 12), '+=', pytest_args_709669, list_709670)
        # Assigning a type to the variable 'pytest_args' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'pytest_args', result_iadd_709673)
        
        # SSA branch for the else part of an if statement (line 50)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'label' (line 52)
        label_709674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 13), 'label')
        str_709675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 22), 'str', 'full')
        # Applying the binary operator '!=' (line 52)
        result_ne_709676 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 13), '!=', label_709674, str_709675)
        
        # Testing the type of an if condition (line 52)
        if_condition_709677 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 52, 13), result_ne_709676)
        # Assigning a type to the variable 'if_condition_709677' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 13), 'if_condition_709677', if_condition_709677)
        # SSA begins for if statement (line 52)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'pytest_args' (line 53)
        pytest_args_709678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'pytest_args')
        
        # Obtaining an instance of the builtin type 'list' (line 53)
        list_709679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 53)
        # Adding element type (line 53)
        str_709680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 28), 'str', '-m')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 27), list_709679, str_709680)
        # Adding element type (line 53)
        # Getting the type of 'label' (line 53)
        label_709681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 34), 'label')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 27), list_709679, label_709681)
        
        # Applying the binary operator '+=' (line 53)
        result_iadd_709682 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 12), '+=', pytest_args_709678, list_709679)
        # Assigning a type to the variable 'pytest_args' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'pytest_args', result_iadd_709682)
        
        # SSA join for if statement (line 52)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 50)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 55)
        # Getting the type of 'tests' (line 55)
        tests_709683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 11), 'tests')
        # Getting the type of 'None' (line 55)
        None_709684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 20), 'None')
        
        (may_be_709685, more_types_in_union_709686) = may_be_none(tests_709683, None_709684)

        if may_be_709685:

            if more_types_in_union_709686:
                # Runtime conditional SSA (line 55)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a List to a Name (line 56):
            
            # Obtaining an instance of the builtin type 'list' (line 56)
            list_709687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 20), 'list')
            # Adding type elements to the builtin type 'list' instance (line 56)
            # Adding element type (line 56)
            # Getting the type of 'self' (line 56)
            self_709688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 21), 'self')
            # Obtaining the member 'module_name' of a type (line 56)
            module_name_709689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 21), self_709688, 'module_name')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 20), list_709687, module_name_709689)
            
            # Assigning a type to the variable 'tests' (line 56)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'tests', list_709687)

            if more_types_in_union_709686:
                # SSA join for if statement (line 55)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'pytest_args' (line 58)
        pytest_args_709690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'pytest_args')
        
        # Obtaining an instance of the builtin type 'list' (line 58)
        list_709691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 58)
        # Adding element type (line 58)
        str_709692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 24), 'str', '--pyargs')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 23), list_709691, str_709692)
        
        
        # Call to list(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'tests' (line 58)
        tests_709694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 43), 'tests', False)
        # Processing the call keyword arguments (line 58)
        kwargs_709695 = {}
        # Getting the type of 'list' (line 58)
        list_709693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 38), 'list', False)
        # Calling list(args, kwargs) (line 58)
        list_call_result_709696 = invoke(stypy.reporting.localization.Localization(__file__, 58, 38), list_709693, *[tests_709694], **kwargs_709695)
        
        # Applying the binary operator '+' (line 58)
        result_add_709697 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 23), '+', list_709691, list_call_result_709696)
        
        # Applying the binary operator '+=' (line 58)
        result_iadd_709698 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 8), '+=', pytest_args_709690, result_add_709697)
        # Assigning a type to the variable 'pytest_args' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'pytest_args', result_iadd_709698)
        
        
        
        # SSA begins for try-except statement (line 60)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 61):
        
        # Call to main(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'pytest_args' (line 61)
        pytest_args_709701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 31), 'pytest_args', False)
        # Processing the call keyword arguments (line 61)
        kwargs_709702 = {}
        # Getting the type of 'pytest' (line 61)
        pytest_709699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 19), 'pytest', False)
        # Obtaining the member 'main' of a type (line 61)
        main_709700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 19), pytest_709699, 'main')
        # Calling main(args, kwargs) (line 61)
        main_call_result_709703 = invoke(stypy.reporting.localization.Localization(__file__, 61, 19), main_709700, *[pytest_args_709701], **kwargs_709702)
        
        # Assigning a type to the variable 'code' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'code', main_call_result_709703)
        # SSA branch for the except part of a try statement (line 60)
        # SSA branch for the except 'SystemExit' branch of a try statement (line 60)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'SystemExit' (line 62)
        SystemExit_709704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 15), 'SystemExit')
        # Assigning a type to the variable 'exc' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'exc', SystemExit_709704)
        
        # Assigning a Attribute to a Name (line 63):
        # Getting the type of 'exc' (line 63)
        exc_709705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 19), 'exc')
        # Obtaining the member 'code' of a type (line 63)
        code_709706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 19), exc_709705, 'code')
        # Assigning a type to the variable 'code' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'code', code_709706)
        # SSA join for try-except statement (line 60)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'code' (line 65)
        code_709707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 16), 'code')
        int_709708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 24), 'int')
        # Applying the binary operator '==' (line 65)
        result_eq_709709 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 16), '==', code_709707, int_709708)
        
        # Assigning a type to the variable 'stypy_return_type' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'stypy_return_type', result_eq_709709)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 29)
        stypy_return_type_709710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_709710)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_709710


# Assigning a type to the variable 'PytestTester' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'PytestTester', PytestTester)

@norecursion
def check_free_memory(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_free_memory'
    module_type_store = module_type_store.open_function_context('check_free_memory', 68, 0, False)
    
    # Passed parameters checking function
    check_free_memory.stypy_localization = localization
    check_free_memory.stypy_type_of_self = None
    check_free_memory.stypy_type_store = module_type_store
    check_free_memory.stypy_function_name = 'check_free_memory'
    check_free_memory.stypy_param_names_list = ['free_mb']
    check_free_memory.stypy_varargs_param_name = None
    check_free_memory.stypy_kwargs_param_name = None
    check_free_memory.stypy_call_defaults = defaults
    check_free_memory.stypy_call_varargs = varargs
    check_free_memory.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_free_memory', ['free_mb'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_free_memory', localization, ['free_mb'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_free_memory(...)' code ##################

    str_709711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, (-1)), 'str', '\n    Check *free_mb* of memory is available, otherwise do pytest.skip\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 72, 4))
    
    # 'import pytest' statement (line 72)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/')
    import_709712 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 72, 4), 'pytest')

    if (type(import_709712) is not StypyTypeError):

        if (import_709712 != 'pyd_module'):
            __import__(import_709712)
            sys_modules_709713 = sys.modules[import_709712]
            import_module(stypy.reporting.localization.Localization(__file__, 72, 4), 'pytest', sys_modules_709713.module_type_store, module_type_store)
        else:
            import pytest

            import_module(stypy.reporting.localization.Localization(__file__, 72, 4), 'pytest', pytest, module_type_store)

    else:
        # Assigning a type to the variable 'pytest' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'pytest', import_709712)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/')
    
    
    
    # SSA begins for try-except statement (line 74)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 75):
    
    # Call to _parse_size(...): (line 75)
    # Processing the call arguments (line 75)
    
    # Obtaining the type of the subscript
    str_709715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 42), 'str', 'SCIPY_AVAILABLE_MEM')
    # Getting the type of 'os' (line 75)
    os_709716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 31), 'os', False)
    # Obtaining the member 'environ' of a type (line 75)
    environ_709717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 31), os_709716, 'environ')
    # Obtaining the member '__getitem__' of a type (line 75)
    getitem___709718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 31), environ_709717, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 75)
    subscript_call_result_709719 = invoke(stypy.reporting.localization.Localization(__file__, 75, 31), getitem___709718, str_709715)
    
    # Processing the call keyword arguments (line 75)
    kwargs_709720 = {}
    # Getting the type of '_parse_size' (line 75)
    _parse_size_709714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 19), '_parse_size', False)
    # Calling _parse_size(args, kwargs) (line 75)
    _parse_size_call_result_709721 = invoke(stypy.reporting.localization.Localization(__file__, 75, 19), _parse_size_709714, *[subscript_call_result_709719], **kwargs_709720)
    
    # Assigning a type to the variable 'mem_free' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'mem_free', _parse_size_call_result_709721)
    
    # Assigning a Call to a Name (line 76):
    
    # Call to format(...): (line 76)
    # Processing the call arguments (line 76)
    # Getting the type of 'free_mb' (line 77)
    free_mb_709724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'free_mb', False)
    
    # Obtaining the type of the subscript
    str_709725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 32), 'str', 'SCIPY_AVAILABLE_MEM')
    # Getting the type of 'os' (line 77)
    os_709726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 21), 'os', False)
    # Obtaining the member 'environ' of a type (line 77)
    environ_709727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 21), os_709726, 'environ')
    # Obtaining the member '__getitem__' of a type (line 77)
    getitem___709728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 21), environ_709727, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 77)
    subscript_call_result_709729 = invoke(stypy.reporting.localization.Localization(__file__, 77, 21), getitem___709728, str_709725)
    
    # Processing the call keyword arguments (line 76)
    kwargs_709730 = {}
    str_709722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 14), 'str', '{0} MB memory required, but environment SCIPY_AVAILABLE_MEM={1}')
    # Obtaining the member 'format' of a type (line 76)
    format_709723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 14), str_709722, 'format')
    # Calling format(args, kwargs) (line 76)
    format_call_result_709731 = invoke(stypy.reporting.localization.Localization(__file__, 76, 14), format_709723, *[free_mb_709724, subscript_call_result_709729], **kwargs_709730)
    
    # Assigning a type to the variable 'msg' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'msg', format_call_result_709731)
    # SSA branch for the except part of a try statement (line 74)
    # SSA branch for the except 'KeyError' branch of a try statement (line 74)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Call to a Name (line 79):
    
    # Call to _get_mem_available(...): (line 79)
    # Processing the call keyword arguments (line 79)
    kwargs_709733 = {}
    # Getting the type of '_get_mem_available' (line 79)
    _get_mem_available_709732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 19), '_get_mem_available', False)
    # Calling _get_mem_available(args, kwargs) (line 79)
    _get_mem_available_call_result_709734 = invoke(stypy.reporting.localization.Localization(__file__, 79, 19), _get_mem_available_709732, *[], **kwargs_709733)
    
    # Assigning a type to the variable 'mem_free' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'mem_free', _get_mem_available_call_result_709734)
    
    # Type idiom detected: calculating its left and rigth part (line 80)
    # Getting the type of 'mem_free' (line 80)
    mem_free_709735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 11), 'mem_free')
    # Getting the type of 'None' (line 80)
    None_709736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 23), 'None')
    
    (may_be_709737, more_types_in_union_709738) = may_be_none(mem_free_709735, None_709736)

    if may_be_709737:

        if more_types_in_union_709738:
            # Runtime conditional SSA (line 80)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to skip(...): (line 81)
        # Processing the call arguments (line 81)
        str_709741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 24), 'str', 'Could not determine available memory; set SCIPY_AVAILABLE_MEM variable to free memory in MB to run the test.')
        # Processing the call keyword arguments (line 81)
        kwargs_709742 = {}
        # Getting the type of 'pytest' (line 81)
        pytest_709739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'pytest', False)
        # Obtaining the member 'skip' of a type (line 81)
        skip_709740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 12), pytest_709739, 'skip')
        # Calling skip(args, kwargs) (line 81)
        skip_call_result_709743 = invoke(stypy.reporting.localization.Localization(__file__, 81, 12), skip_709740, *[str_709741], **kwargs_709742)
        

        if more_types_in_union_709738:
            # SSA join for if statement (line 80)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 83):
    
    # Call to format(...): (line 83)
    # Processing the call arguments (line 83)
    # Getting the type of 'free_mb' (line 84)
    free_mb_709746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'free_mb', False)
    # Getting the type of 'mem_free' (line 84)
    mem_free_709747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 21), 'mem_free', False)
    float_709748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 30), 'float')
    # Applying the binary operator 'div' (line 84)
    result_div_709749 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 21), 'div', mem_free_709747, float_709748)
    
    # Processing the call keyword arguments (line 83)
    kwargs_709750 = {}
    str_709744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 14), 'str', '{0} MB memory required, but {1} MB available')
    # Obtaining the member 'format' of a type (line 83)
    format_709745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 14), str_709744, 'format')
    # Calling format(args, kwargs) (line 83)
    format_call_result_709751 = invoke(stypy.reporting.localization.Localization(__file__, 83, 14), format_709745, *[free_mb_709746, result_div_709749], **kwargs_709750)
    
    # Assigning a type to the variable 'msg' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'msg', format_call_result_709751)
    # SSA join for try-except statement (line 74)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'mem_free' (line 86)
    mem_free_709752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 7), 'mem_free')
    # Getting the type of 'free_mb' (line 86)
    free_mb_709753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 18), 'free_mb')
    float_709754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 28), 'float')
    # Applying the binary operator '*' (line 86)
    result_mul_709755 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 18), '*', free_mb_709753, float_709754)
    
    # Applying the binary operator '<' (line 86)
    result_lt_709756 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 7), '<', mem_free_709752, result_mul_709755)
    
    # Testing the type of an if condition (line 86)
    if_condition_709757 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 86, 4), result_lt_709756)
    # Assigning a type to the variable 'if_condition_709757' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'if_condition_709757', if_condition_709757)
    # SSA begins for if statement (line 86)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to skip(...): (line 87)
    # Processing the call arguments (line 87)
    # Getting the type of 'msg' (line 87)
    msg_709760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), 'msg', False)
    # Processing the call keyword arguments (line 87)
    kwargs_709761 = {}
    # Getting the type of 'pytest' (line 87)
    pytest_709758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'pytest', False)
    # Obtaining the member 'skip' of a type (line 87)
    skip_709759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 8), pytest_709758, 'skip')
    # Calling skip(args, kwargs) (line 87)
    skip_call_result_709762 = invoke(stypy.reporting.localization.Localization(__file__, 87, 8), skip_709759, *[msg_709760], **kwargs_709761)
    
    # SSA join for if statement (line 86)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'check_free_memory(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_free_memory' in the type store
    # Getting the type of 'stypy_return_type' (line 68)
    stypy_return_type_709763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_709763)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_free_memory'
    return stypy_return_type_709763

# Assigning a type to the variable 'check_free_memory' (line 68)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'check_free_memory', check_free_memory)

@norecursion
def _parse_size(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_parse_size'
    module_type_store = module_type_store.open_function_context('_parse_size', 90, 0, False)
    
    # Passed parameters checking function
    _parse_size.stypy_localization = localization
    _parse_size.stypy_type_of_self = None
    _parse_size.stypy_type_store = module_type_store
    _parse_size.stypy_function_name = '_parse_size'
    _parse_size.stypy_param_names_list = ['size_str']
    _parse_size.stypy_varargs_param_name = None
    _parse_size.stypy_kwargs_param_name = None
    _parse_size.stypy_call_defaults = defaults
    _parse_size.stypy_call_varargs = varargs
    _parse_size.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_parse_size', ['size_str'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_parse_size', localization, ['size_str'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_parse_size(...)' code ##################

    
    # Assigning a Dict to a Name (line 91):
    
    # Obtaining an instance of the builtin type 'dict' (line 91)
    dict_709764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 15), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 91)
    # Adding element type (key, value) (line 91)
    str_709765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 16), 'str', '')
    float_709766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 20), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 15), dict_709764, (str_709765, float_709766))
    # Adding element type (key, value) (line 91)
    str_709767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 16), 'str', 'b')
    float_709768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 21), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 15), dict_709764, (str_709767, float_709768))
    # Adding element type (key, value) (line 91)
    str_709769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 16), 'str', 'k')
    float_709770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 21), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 15), dict_709764, (str_709769, float_709770))
    # Adding element type (key, value) (line 91)
    str_709771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 26), 'str', 'M')
    float_709772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 31), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 15), dict_709764, (str_709771, float_709772))
    # Adding element type (key, value) (line 91)
    str_709773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 36), 'str', 'G')
    float_709774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 41), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 15), dict_709764, (str_709773, float_709774))
    # Adding element type (key, value) (line 91)
    str_709775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 46), 'str', 'T')
    float_709776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 51), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 15), dict_709764, (str_709775, float_709776))
    # Adding element type (key, value) (line 91)
    str_709777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 16), 'str', 'kb')
    float_709778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 22), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 15), dict_709764, (str_709777, float_709778))
    # Adding element type (key, value) (line 91)
    str_709779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 27), 'str', 'Mb')
    float_709780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 33), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 15), dict_709764, (str_709779, float_709780))
    # Adding element type (key, value) (line 91)
    str_709781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 38), 'str', 'Gb')
    float_709782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 44), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 15), dict_709764, (str_709781, float_709782))
    # Adding element type (key, value) (line 91)
    str_709783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 49), 'str', 'Tb')
    float_709784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 55), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 15), dict_709764, (str_709783, float_709784))
    # Adding element type (key, value) (line 91)
    str_709785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 16), 'str', 'kib')
    float_709786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 23), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 15), dict_709764, (str_709785, float_709786))
    # Adding element type (key, value) (line 91)
    str_709787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 31), 'str', 'Mib')
    float_709788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 38), 'float')
    int_709789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 46), 'int')
    # Applying the binary operator '**' (line 95)
    result_pow_709790 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 38), '**', float_709788, int_709789)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 15), dict_709764, (str_709787, result_pow_709790))
    # Adding element type (key, value) (line 91)
    str_709791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 49), 'str', 'Gib')
    float_709792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 56), 'float')
    int_709793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 64), 'int')
    # Applying the binary operator '**' (line 95)
    result_pow_709794 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 56), '**', float_709792, int_709793)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 15), dict_709764, (str_709791, result_pow_709794))
    # Adding element type (key, value) (line 91)
    str_709795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 67), 'str', 'Tib')
    float_709796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 74), 'float')
    int_709797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 82), 'int')
    # Applying the binary operator '**' (line 95)
    result_pow_709798 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 74), '**', float_709796, int_709797)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 15), dict_709764, (str_709795, result_pow_709798))
    
    # Assigning a type to the variable 'suffixes' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'suffixes', dict_709764)
    
    # Assigning a Call to a Name (line 96):
    
    # Call to match(...): (line 96)
    # Processing the call arguments (line 96)
    
    # Call to format(...): (line 96)
    # Processing the call arguments (line 96)
    
    # Call to join(...): (line 96)
    # Processing the call arguments (line 96)
    
    # Call to keys(...): (line 96)
    # Processing the call keyword arguments (line 96)
    kwargs_709807 = {}
    # Getting the type of 'suffixes' (line 96)
    suffixes_709805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 58), 'suffixes', False)
    # Obtaining the member 'keys' of a type (line 96)
    keys_709806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 58), suffixes_709805, 'keys')
    # Calling keys(args, kwargs) (line 96)
    keys_call_result_709808 = invoke(stypy.reporting.localization.Localization(__file__, 96, 58), keys_709806, *[], **kwargs_709807)
    
    # Processing the call keyword arguments (line 96)
    kwargs_709809 = {}
    str_709803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 49), 'str', '|')
    # Obtaining the member 'join' of a type (line 96)
    join_709804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 49), str_709803, 'join')
    # Calling join(args, kwargs) (line 96)
    join_call_result_709810 = invoke(stypy.reporting.localization.Localization(__file__, 96, 49), join_709804, *[keys_call_result_709808], **kwargs_709809)
    
    # Processing the call keyword arguments (line 96)
    kwargs_709811 = {}
    str_709801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 17), 'str', '^\\s*(\\d+)\\s*({0})\\s*$')
    # Obtaining the member 'format' of a type (line 96)
    format_709802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 17), str_709801, 'format')
    # Calling format(args, kwargs) (line 96)
    format_call_result_709812 = invoke(stypy.reporting.localization.Localization(__file__, 96, 17), format_709802, *[join_call_result_709810], **kwargs_709811)
    
    # Getting the type of 'size_str' (line 97)
    size_str_709813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 17), 'size_str', False)
    # Getting the type of 're' (line 98)
    re_709814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 17), 're', False)
    # Obtaining the member 'I' of a type (line 98)
    I_709815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 17), re_709814, 'I')
    # Processing the call keyword arguments (line 96)
    kwargs_709816 = {}
    # Getting the type of 're' (line 96)
    re_709799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 're', False)
    # Obtaining the member 'match' of a type (line 96)
    match_709800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 8), re_709799, 'match')
    # Calling match(args, kwargs) (line 96)
    match_call_result_709817 = invoke(stypy.reporting.localization.Localization(__file__, 96, 8), match_709800, *[format_call_result_709812, size_str_709813, I_709815], **kwargs_709816)
    
    # Assigning a type to the variable 'm' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'm', match_call_result_709817)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'm' (line 99)
    m_709818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 11), 'm')
    # Applying the 'not' unary operator (line 99)
    result_not__709819 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 7), 'not', m_709818)
    
    
    
    # Call to group(...): (line 99)
    # Processing the call arguments (line 99)
    int_709822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 24), 'int')
    # Processing the call keyword arguments (line 99)
    kwargs_709823 = {}
    # Getting the type of 'm' (line 99)
    m_709820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'm', False)
    # Obtaining the member 'group' of a type (line 99)
    group_709821 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 16), m_709820, 'group')
    # Calling group(args, kwargs) (line 99)
    group_call_result_709824 = invoke(stypy.reporting.localization.Localization(__file__, 99, 16), group_709821, *[int_709822], **kwargs_709823)
    
    # Getting the type of 'suffixes' (line 99)
    suffixes_709825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 34), 'suffixes')
    # Applying the binary operator 'notin' (line 99)
    result_contains_709826 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 16), 'notin', group_call_result_709824, suffixes_709825)
    
    # Applying the binary operator 'or' (line 99)
    result_or_keyword_709827 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 7), 'or', result_not__709819, result_contains_709826)
    
    # Testing the type of an if condition (line 99)
    if_condition_709828 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 99, 4), result_or_keyword_709827)
    # Assigning a type to the variable 'if_condition_709828' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'if_condition_709828', if_condition_709828)
    # SSA begins for if statement (line 99)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 100)
    # Processing the call arguments (line 100)
    str_709830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 25), 'str', 'Invalid size string')
    # Processing the call keyword arguments (line 100)
    kwargs_709831 = {}
    # Getting the type of 'ValueError' (line 100)
    ValueError_709829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 100)
    ValueError_call_result_709832 = invoke(stypy.reporting.localization.Localization(__file__, 100, 14), ValueError_709829, *[str_709830], **kwargs_709831)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 100, 8), ValueError_call_result_709832, 'raise parameter', BaseException)
    # SSA join for if statement (line 99)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to float(...): (line 102)
    # Processing the call arguments (line 102)
    
    # Call to group(...): (line 102)
    # Processing the call arguments (line 102)
    int_709836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 25), 'int')
    # Processing the call keyword arguments (line 102)
    kwargs_709837 = {}
    # Getting the type of 'm' (line 102)
    m_709834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 17), 'm', False)
    # Obtaining the member 'group' of a type (line 102)
    group_709835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 17), m_709834, 'group')
    # Calling group(args, kwargs) (line 102)
    group_call_result_709838 = invoke(stypy.reporting.localization.Localization(__file__, 102, 17), group_709835, *[int_709836], **kwargs_709837)
    
    # Processing the call keyword arguments (line 102)
    kwargs_709839 = {}
    # Getting the type of 'float' (line 102)
    float_709833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 11), 'float', False)
    # Calling float(args, kwargs) (line 102)
    float_call_result_709840 = invoke(stypy.reporting.localization.Localization(__file__, 102, 11), float_709833, *[group_call_result_709838], **kwargs_709839)
    
    
    # Obtaining the type of the subscript
    
    # Call to group(...): (line 102)
    # Processing the call arguments (line 102)
    int_709843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 48), 'int')
    # Processing the call keyword arguments (line 102)
    kwargs_709844 = {}
    # Getting the type of 'm' (line 102)
    m_709841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 40), 'm', False)
    # Obtaining the member 'group' of a type (line 102)
    group_709842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 40), m_709841, 'group')
    # Calling group(args, kwargs) (line 102)
    group_call_result_709845 = invoke(stypy.reporting.localization.Localization(__file__, 102, 40), group_709842, *[int_709843], **kwargs_709844)
    
    # Getting the type of 'suffixes' (line 102)
    suffixes_709846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 31), 'suffixes')
    # Obtaining the member '__getitem__' of a type (line 102)
    getitem___709847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 31), suffixes_709846, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 102)
    subscript_call_result_709848 = invoke(stypy.reporting.localization.Localization(__file__, 102, 31), getitem___709847, group_call_result_709845)
    
    # Applying the binary operator '*' (line 102)
    result_mul_709849 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 11), '*', float_call_result_709840, subscript_call_result_709848)
    
    # Assigning a type to the variable 'stypy_return_type' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'stypy_return_type', result_mul_709849)
    
    # ################# End of '_parse_size(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_parse_size' in the type store
    # Getting the type of 'stypy_return_type' (line 90)
    stypy_return_type_709850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_709850)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_parse_size'
    return stypy_return_type_709850

# Assigning a type to the variable '_parse_size' (line 90)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 0), '_parse_size', _parse_size)

@norecursion
def _get_mem_available(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_get_mem_available'
    module_type_store = module_type_store.open_function_context('_get_mem_available', 105, 0, False)
    
    # Passed parameters checking function
    _get_mem_available.stypy_localization = localization
    _get_mem_available.stypy_type_of_self = None
    _get_mem_available.stypy_type_store = module_type_store
    _get_mem_available.stypy_function_name = '_get_mem_available'
    _get_mem_available.stypy_param_names_list = []
    _get_mem_available.stypy_varargs_param_name = None
    _get_mem_available.stypy_kwargs_param_name = None
    _get_mem_available.stypy_call_defaults = defaults
    _get_mem_available.stypy_call_varargs = varargs
    _get_mem_available.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_get_mem_available', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_get_mem_available', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_get_mem_available(...)' code ##################

    str_709851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, (-1)), 'str', '\n    Get information about memory available, not counting swap.\n    ')
    
    
    # SSA begins for try-except statement (line 109)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 110, 8))
    
    # 'import psutil' statement (line 110)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/')
    import_709852 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 110, 8), 'psutil')

    if (type(import_709852) is not StypyTypeError):

        if (import_709852 != 'pyd_module'):
            __import__(import_709852)
            sys_modules_709853 = sys.modules[import_709852]
            import_module(stypy.reporting.localization.Localization(__file__, 110, 8), 'psutil', sys_modules_709853.module_type_store, module_type_store)
        else:
            import psutil

            import_module(stypy.reporting.localization.Localization(__file__, 110, 8), 'psutil', psutil, module_type_store)

    else:
        # Assigning a type to the variable 'psutil' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'psutil', import_709852)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/')
    
    
    # Call to virtual_memory(...): (line 111)
    # Processing the call keyword arguments (line 111)
    kwargs_709856 = {}
    # Getting the type of 'psutil' (line 111)
    psutil_709854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 15), 'psutil', False)
    # Obtaining the member 'virtual_memory' of a type (line 111)
    virtual_memory_709855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 15), psutil_709854, 'virtual_memory')
    # Calling virtual_memory(args, kwargs) (line 111)
    virtual_memory_call_result_709857 = invoke(stypy.reporting.localization.Localization(__file__, 111, 15), virtual_memory_709855, *[], **kwargs_709856)
    
    # Obtaining the member 'available' of a type (line 111)
    available_709858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 15), virtual_memory_call_result_709857, 'available')
    # Assigning a type to the variable 'stypy_return_type' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'stypy_return_type', available_709858)
    # SSA branch for the except part of a try statement (line 109)
    # SSA branch for the except 'Tuple' branch of a try statement (line 109)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 109)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to startswith(...): (line 115)
    # Processing the call arguments (line 115)
    str_709862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 31), 'str', 'linux')
    # Processing the call keyword arguments (line 115)
    kwargs_709863 = {}
    # Getting the type of 'sys' (line 115)
    sys_709859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 7), 'sys', False)
    # Obtaining the member 'platform' of a type (line 115)
    platform_709860 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 7), sys_709859, 'platform')
    # Obtaining the member 'startswith' of a type (line 115)
    startswith_709861 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 7), platform_709860, 'startswith')
    # Calling startswith(args, kwargs) (line 115)
    startswith_call_result_709864 = invoke(stypy.reporting.localization.Localization(__file__, 115, 7), startswith_709861, *[str_709862], **kwargs_709863)
    
    # Testing the type of an if condition (line 115)
    if_condition_709865 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 115, 4), startswith_call_result_709864)
    # Assigning a type to the variable 'if_condition_709865' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'if_condition_709865', if_condition_709865)
    # SSA begins for if statement (line 115)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Dict to a Name (line 116):
    
    # Obtaining an instance of the builtin type 'dict' (line 116)
    dict_709866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 15), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 116)
    
    # Assigning a type to the variable 'info' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'info', dict_709866)
    
    # Call to open(...): (line 117)
    # Processing the call arguments (line 117)
    str_709868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 18), 'str', '/proc/meminfo')
    str_709869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 35), 'str', 'r')
    # Processing the call keyword arguments (line 117)
    kwargs_709870 = {}
    # Getting the type of 'open' (line 117)
    open_709867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 13), 'open', False)
    # Calling open(args, kwargs) (line 117)
    open_call_result_709871 = invoke(stypy.reporting.localization.Localization(__file__, 117, 13), open_709867, *[str_709868, str_709869], **kwargs_709870)
    
    with_709872 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 117, 13), open_call_result_709871, 'with parameter', '__enter__', '__exit__')

    if with_709872:
        # Calling the __enter__ method to initiate a with section
        # Obtaining the member '__enter__' of a type (line 117)
        enter___709873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 13), open_call_result_709871, '__enter__')
        with_enter_709874 = invoke(stypy.reporting.localization.Localization(__file__, 117, 13), enter___709873)
        # Assigning a type to the variable 'f' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 13), 'f', with_enter_709874)
        
        # Getting the type of 'f' (line 118)
        f_709875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 24), 'f')
        # Testing the type of a for loop iterable (line 118)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 118, 12), f_709875)
        # Getting the type of the for loop variable (line 118)
        for_loop_var_709876 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 118, 12), f_709875)
        # Assigning a type to the variable 'line' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'line', for_loop_var_709876)
        # SSA begins for a for statement (line 118)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 119):
        
        # Call to split(...): (line 119)
        # Processing the call keyword arguments (line 119)
        kwargs_709879 = {}
        # Getting the type of 'line' (line 119)
        line_709877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 20), 'line', False)
        # Obtaining the member 'split' of a type (line 119)
        split_709878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 20), line_709877, 'split')
        # Calling split(args, kwargs) (line 119)
        split_call_result_709880 = invoke(stypy.reporting.localization.Localization(__file__, 119, 20), split_709878, *[], **kwargs_709879)
        
        # Assigning a type to the variable 'p' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 16), 'p', split_call_result_709880)
        
        # Assigning a BinOp to a Subscript (line 120):
        
        # Call to float(...): (line 120)
        # Processing the call arguments (line 120)
        
        # Obtaining the type of the subscript
        int_709882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 56), 'int')
        # Getting the type of 'p' (line 120)
        p_709883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 54), 'p', False)
        # Obtaining the member '__getitem__' of a type (line 120)
        getitem___709884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 54), p_709883, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 120)
        subscript_call_result_709885 = invoke(stypy.reporting.localization.Localization(__file__, 120, 54), getitem___709884, int_709882)
        
        # Processing the call keyword arguments (line 120)
        kwargs_709886 = {}
        # Getting the type of 'float' (line 120)
        float_709881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 48), 'float', False)
        # Calling float(args, kwargs) (line 120)
        float_call_result_709887 = invoke(stypy.reporting.localization.Localization(__file__, 120, 48), float_709881, *[subscript_call_result_709885], **kwargs_709886)
        
        float_709888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 62), 'float')
        # Applying the binary operator '*' (line 120)
        result_mul_709889 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 48), '*', float_call_result_709887, float_709888)
        
        # Getting the type of 'info' (line 120)
        info_709890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'info')
        
        # Call to lower(...): (line 120)
        # Processing the call keyword arguments (line 120)
        kwargs_709900 = {}
        
        # Call to strip(...): (line 120)
        # Processing the call arguments (line 120)
        str_709896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 32), 'str', ':')
        # Processing the call keyword arguments (line 120)
        kwargs_709897 = {}
        
        # Obtaining the type of the subscript
        int_709891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 23), 'int')
        # Getting the type of 'p' (line 120)
        p_709892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 21), 'p', False)
        # Obtaining the member '__getitem__' of a type (line 120)
        getitem___709893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 21), p_709892, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 120)
        subscript_call_result_709894 = invoke(stypy.reporting.localization.Localization(__file__, 120, 21), getitem___709893, int_709891)
        
        # Obtaining the member 'strip' of a type (line 120)
        strip_709895 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 21), subscript_call_result_709894, 'strip')
        # Calling strip(args, kwargs) (line 120)
        strip_call_result_709898 = invoke(stypy.reporting.localization.Localization(__file__, 120, 21), strip_709895, *[str_709896], **kwargs_709897)
        
        # Obtaining the member 'lower' of a type (line 120)
        lower_709899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 21), strip_call_result_709898, 'lower')
        # Calling lower(args, kwargs) (line 120)
        lower_call_result_709901 = invoke(stypy.reporting.localization.Localization(__file__, 120, 21), lower_709899, *[], **kwargs_709900)
        
        # Storing an element on a container (line 120)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 16), info_709890, (lower_call_result_709901, result_mul_709889))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Calling the __exit__ method to finish a with section
        # Obtaining the member '__exit__' of a type (line 117)
        exit___709902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 13), open_call_result_709871, '__exit__')
        with_exit_709903 = invoke(stypy.reporting.localization.Localization(__file__, 117, 13), exit___709902, None, None, None)

    
    
    str_709904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 11), 'str', 'memavailable')
    # Getting the type of 'info' (line 122)
    info_709905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 29), 'info')
    # Applying the binary operator 'in' (line 122)
    result_contains_709906 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 11), 'in', str_709904, info_709905)
    
    # Testing the type of an if condition (line 122)
    if_condition_709907 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 122, 8), result_contains_709906)
    # Assigning a type to the variable 'if_condition_709907' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'if_condition_709907', if_condition_709907)
    # SSA begins for if statement (line 122)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    str_709908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 24), 'str', 'memavailable')
    # Getting the type of 'info' (line 124)
    info_709909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 19), 'info')
    # Obtaining the member '__getitem__' of a type (line 124)
    getitem___709910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 19), info_709909, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 124)
    subscript_call_result_709911 = invoke(stypy.reporting.localization.Localization(__file__, 124, 19), getitem___709910, str_709908)
    
    # Assigning a type to the variable 'stypy_return_type' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'stypy_return_type', subscript_call_result_709911)
    # SSA branch for the else part of an if statement (line 122)
    module_type_store.open_ssa_branch('else')
    
    # Obtaining the type of the subscript
    str_709912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 24), 'str', 'memfree')
    # Getting the type of 'info' (line 126)
    info_709913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 19), 'info')
    # Obtaining the member '__getitem__' of a type (line 126)
    getitem___709914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 19), info_709913, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 126)
    subscript_call_result_709915 = invoke(stypy.reporting.localization.Localization(__file__, 126, 19), getitem___709914, str_709912)
    
    
    # Obtaining the type of the subscript
    str_709916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 42), 'str', 'cached')
    # Getting the type of 'info' (line 126)
    info_709917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 37), 'info')
    # Obtaining the member '__getitem__' of a type (line 126)
    getitem___709918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 37), info_709917, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 126)
    subscript_call_result_709919 = invoke(stypy.reporting.localization.Localization(__file__, 126, 37), getitem___709918, str_709916)
    
    # Applying the binary operator '+' (line 126)
    result_add_709920 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 19), '+', subscript_call_result_709915, subscript_call_result_709919)
    
    # Assigning a type to the variable 'stypy_return_type' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'stypy_return_type', result_add_709920)
    # SSA join for if statement (line 122)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 115)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'None' (line 128)
    None_709921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 11), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'stypy_return_type', None_709921)
    
    # ################# End of '_get_mem_available(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_get_mem_available' in the type store
    # Getting the type of 'stypy_return_type' (line 105)
    stypy_return_type_709922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_709922)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_get_mem_available'
    return stypy_return_type_709922

# Assigning a type to the variable '_get_mem_available' (line 105)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), '_get_mem_available', _get_mem_available)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
