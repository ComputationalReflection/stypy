
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, print_function, absolute_import
2: 
3: from subprocess import call, PIPE, Popen
4: import sys
5: import re
6: 
7: import pytest
8: from numpy.testing import assert_
9: from numpy.compat import asbytes
10: 
11: from scipy.linalg import _flapack as flapack
12: 
13: # XXX: this is copied from numpy trunk. Can be removed when we will depend on
14: # numpy 1.3
15: 
16: 
17: class FindDependenciesLdd:
18:     def __init__(self):
19:         self.cmd = ['ldd']
20: 
21:         try:
22:             st = call(self.cmd, stdout=PIPE, stderr=PIPE)
23:         except OSError:
24:             raise RuntimeError("command %s cannot be run" % self.cmd)
25: 
26:     def get_dependencies(self, file):
27:         p = Popen(self.cmd + [file], stdout=PIPE, stderr=PIPE)
28:         stdout, stderr = p.communicate()
29:         if not (p.returncode == 0):
30:             raise RuntimeError("Failed to check dependencies for %s" % file)
31: 
32:         return stdout
33: 
34:     def grep_dependencies(self, file, deps):
35:         stdout = self.get_dependencies(file)
36: 
37:         rdeps = dict([(asbytes(dep), re.compile(asbytes(dep))) for dep in deps])
38:         founds = []
39:         for l in stdout.splitlines():
40:             for k, v in rdeps.items():
41:                 if v.search(l):
42:                     founds.append(k)
43: 
44:         return founds
45: 
46: 
47: class TestF77Mismatch(object):
48:     @pytest.mark.skipif(not(sys.platform[:5] == 'linux'),
49:                         reason="Skipping fortran compiler mismatch on non Linux platform")
50:     def test_lapack(self):
51:         f = FindDependenciesLdd()
52:         deps = f.grep_dependencies(flapack.__file__,
53:                                    ['libg2c', 'libgfortran'])
54:         assert_(not (len(deps) > 1),
55: '''Both g77 and gfortran runtimes linked in scipy.linalg.flapack ! This is
56: likely to cause random crashes and wrong results. See numpy INSTALL.rst.txt for
57: more information.''')
58: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from subprocess import call, PIPE, Popen' statement (line 3)
try:
    from subprocess import call, PIPE, Popen

except:
    call = UndefinedType
    PIPE = UndefinedType
    Popen = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'subprocess', None, module_type_store, ['call', 'PIPE', 'Popen'], [call, PIPE, Popen])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import sys' statement (line 4)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import re' statement (line 5)
import re

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import pytest' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_53278 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'pytest')

if (type(import_53278) is not StypyTypeError):

    if (import_53278 != 'pyd_module'):
        __import__(import_53278)
        sys_modules_53279 = sys.modules[import_53278]
        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'pytest', sys_modules_53279.module_type_store, module_type_store)
    else:
        import pytest

        import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'pytest', pytest, module_type_store)

else:
    # Assigning a type to the variable 'pytest' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'pytest', import_53278)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from numpy.testing import assert_' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_53280 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.testing')

if (type(import_53280) is not StypyTypeError):

    if (import_53280 != 'pyd_module'):
        __import__(import_53280)
        sys_modules_53281 = sys.modules[import_53280]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.testing', sys_modules_53281.module_type_store, module_type_store, ['assert_'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_53281, sys_modules_53281.module_type_store, module_type_store)
    else:
        from numpy.testing import assert_

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.testing', None, module_type_store, ['assert_'], [assert_])

else:
    # Assigning a type to the variable 'numpy.testing' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.testing', import_53280)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from numpy.compat import asbytes' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_53282 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.compat')

if (type(import_53282) is not StypyTypeError):

    if (import_53282 != 'pyd_module'):
        __import__(import_53282)
        sys_modules_53283 = sys.modules[import_53282]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.compat', sys_modules_53283.module_type_store, module_type_store, ['asbytes'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_53283, sys_modules_53283.module_type_store, module_type_store)
    else:
        from numpy.compat import asbytes

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.compat', None, module_type_store, ['asbytes'], [asbytes])

else:
    # Assigning a type to the variable 'numpy.compat' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.compat', import_53282)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy.linalg import flapack' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/linalg/tests/')
import_53284 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg')

if (type(import_53284) is not StypyTypeError):

    if (import_53284 != 'pyd_module'):
        __import__(import_53284)
        sys_modules_53285 = sys.modules[import_53284]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg', sys_modules_53285.module_type_store, module_type_store, ['_flapack'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_53285, sys_modules_53285.module_type_store, module_type_store)
    else:
        from scipy.linalg import _flapack as flapack

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg', None, module_type_store, ['_flapack'], [flapack])

else:
    # Assigning a type to the variable 'scipy.linalg' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy.linalg', import_53284)

# Adding an alias
module_type_store.add_alias('flapack', '_flapack')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/linalg/tests/')

# Declaration of the 'FindDependenciesLdd' class

class FindDependenciesLdd:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 18, 4, False)
        # Assigning a type to the variable 'self' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FindDependenciesLdd.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a List to a Attribute (line 19):
        
        # Assigning a List to a Attribute (line 19):
        
        # Obtaining an instance of the builtin type 'list' (line 19)
        list_53286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 19)
        # Adding element type (line 19)
        str_53287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 20), 'str', 'ldd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 19), list_53286, str_53287)
        
        # Getting the type of 'self' (line 19)
        self_53288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'self')
        # Setting the type of the member 'cmd' of a type (line 19)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 8), self_53288, 'cmd', list_53286)
        
        
        # SSA begins for try-except statement (line 21)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 22):
        
        # Assigning a Call to a Name (line 22):
        
        # Call to call(...): (line 22)
        # Processing the call arguments (line 22)
        # Getting the type of 'self' (line 22)
        self_53290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 22), 'self', False)
        # Obtaining the member 'cmd' of a type (line 22)
        cmd_53291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 22), self_53290, 'cmd')
        # Processing the call keyword arguments (line 22)
        # Getting the type of 'PIPE' (line 22)
        PIPE_53292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 39), 'PIPE', False)
        keyword_53293 = PIPE_53292
        # Getting the type of 'PIPE' (line 22)
        PIPE_53294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 52), 'PIPE', False)
        keyword_53295 = PIPE_53294
        kwargs_53296 = {'stderr': keyword_53295, 'stdout': keyword_53293}
        # Getting the type of 'call' (line 22)
        call_53289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 17), 'call', False)
        # Calling call(args, kwargs) (line 22)
        call_call_result_53297 = invoke(stypy.reporting.localization.Localization(__file__, 22, 17), call_53289, *[cmd_53291], **kwargs_53296)
        
        # Assigning a type to the variable 'st' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'st', call_call_result_53297)
        # SSA branch for the except part of a try statement (line 21)
        # SSA branch for the except 'OSError' branch of a try statement (line 21)
        module_type_store.open_ssa_branch('except')
        
        # Call to RuntimeError(...): (line 24)
        # Processing the call arguments (line 24)
        str_53299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 31), 'str', 'command %s cannot be run')
        # Getting the type of 'self' (line 24)
        self_53300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 60), 'self', False)
        # Obtaining the member 'cmd' of a type (line 24)
        cmd_53301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 60), self_53300, 'cmd')
        # Applying the binary operator '%' (line 24)
        result_mod_53302 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 31), '%', str_53299, cmd_53301)
        
        # Processing the call keyword arguments (line 24)
        kwargs_53303 = {}
        # Getting the type of 'RuntimeError' (line 24)
        RuntimeError_53298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 18), 'RuntimeError', False)
        # Calling RuntimeError(args, kwargs) (line 24)
        RuntimeError_call_result_53304 = invoke(stypy.reporting.localization.Localization(__file__, 24, 18), RuntimeError_53298, *[result_mod_53302], **kwargs_53303)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 24, 12), RuntimeError_call_result_53304, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 21)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def get_dependencies(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_dependencies'
        module_type_store = module_type_store.open_function_context('get_dependencies', 26, 4, False)
        # Assigning a type to the variable 'self' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FindDependenciesLdd.get_dependencies.__dict__.__setitem__('stypy_localization', localization)
        FindDependenciesLdd.get_dependencies.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FindDependenciesLdd.get_dependencies.__dict__.__setitem__('stypy_type_store', module_type_store)
        FindDependenciesLdd.get_dependencies.__dict__.__setitem__('stypy_function_name', 'FindDependenciesLdd.get_dependencies')
        FindDependenciesLdd.get_dependencies.__dict__.__setitem__('stypy_param_names_list', ['file'])
        FindDependenciesLdd.get_dependencies.__dict__.__setitem__('stypy_varargs_param_name', None)
        FindDependenciesLdd.get_dependencies.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FindDependenciesLdd.get_dependencies.__dict__.__setitem__('stypy_call_defaults', defaults)
        FindDependenciesLdd.get_dependencies.__dict__.__setitem__('stypy_call_varargs', varargs)
        FindDependenciesLdd.get_dependencies.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FindDependenciesLdd.get_dependencies.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FindDependenciesLdd.get_dependencies', ['file'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_dependencies', localization, ['file'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_dependencies(...)' code ##################

        
        # Assigning a Call to a Name (line 27):
        
        # Assigning a Call to a Name (line 27):
        
        # Call to Popen(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'self' (line 27)
        self_53306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 18), 'self', False)
        # Obtaining the member 'cmd' of a type (line 27)
        cmd_53307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 18), self_53306, 'cmd')
        
        # Obtaining an instance of the builtin type 'list' (line 27)
        list_53308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 27)
        # Adding element type (line 27)
        # Getting the type of 'file' (line 27)
        file_53309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 30), 'file', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 29), list_53308, file_53309)
        
        # Applying the binary operator '+' (line 27)
        result_add_53310 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 18), '+', cmd_53307, list_53308)
        
        # Processing the call keyword arguments (line 27)
        # Getting the type of 'PIPE' (line 27)
        PIPE_53311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 44), 'PIPE', False)
        keyword_53312 = PIPE_53311
        # Getting the type of 'PIPE' (line 27)
        PIPE_53313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 57), 'PIPE', False)
        keyword_53314 = PIPE_53313
        kwargs_53315 = {'stderr': keyword_53314, 'stdout': keyword_53312}
        # Getting the type of 'Popen' (line 27)
        Popen_53305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'Popen', False)
        # Calling Popen(args, kwargs) (line 27)
        Popen_call_result_53316 = invoke(stypy.reporting.localization.Localization(__file__, 27, 12), Popen_53305, *[result_add_53310], **kwargs_53315)
        
        # Assigning a type to the variable 'p' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'p', Popen_call_result_53316)
        
        # Assigning a Call to a Tuple (line 28):
        
        # Assigning a Subscript to a Name (line 28):
        
        # Obtaining the type of the subscript
        int_53317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 8), 'int')
        
        # Call to communicate(...): (line 28)
        # Processing the call keyword arguments (line 28)
        kwargs_53320 = {}
        # Getting the type of 'p' (line 28)
        p_53318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 25), 'p', False)
        # Obtaining the member 'communicate' of a type (line 28)
        communicate_53319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 25), p_53318, 'communicate')
        # Calling communicate(args, kwargs) (line 28)
        communicate_call_result_53321 = invoke(stypy.reporting.localization.Localization(__file__, 28, 25), communicate_53319, *[], **kwargs_53320)
        
        # Obtaining the member '__getitem__' of a type (line 28)
        getitem___53322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), communicate_call_result_53321, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 28)
        subscript_call_result_53323 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), getitem___53322, int_53317)
        
        # Assigning a type to the variable 'tuple_var_assignment_53276' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'tuple_var_assignment_53276', subscript_call_result_53323)
        
        # Assigning a Subscript to a Name (line 28):
        
        # Obtaining the type of the subscript
        int_53324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 8), 'int')
        
        # Call to communicate(...): (line 28)
        # Processing the call keyword arguments (line 28)
        kwargs_53327 = {}
        # Getting the type of 'p' (line 28)
        p_53325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 25), 'p', False)
        # Obtaining the member 'communicate' of a type (line 28)
        communicate_53326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 25), p_53325, 'communicate')
        # Calling communicate(args, kwargs) (line 28)
        communicate_call_result_53328 = invoke(stypy.reporting.localization.Localization(__file__, 28, 25), communicate_53326, *[], **kwargs_53327)
        
        # Obtaining the member '__getitem__' of a type (line 28)
        getitem___53329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), communicate_call_result_53328, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 28)
        subscript_call_result_53330 = invoke(stypy.reporting.localization.Localization(__file__, 28, 8), getitem___53329, int_53324)
        
        # Assigning a type to the variable 'tuple_var_assignment_53277' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'tuple_var_assignment_53277', subscript_call_result_53330)
        
        # Assigning a Name to a Name (line 28):
        # Getting the type of 'tuple_var_assignment_53276' (line 28)
        tuple_var_assignment_53276_53331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'tuple_var_assignment_53276')
        # Assigning a type to the variable 'stdout' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'stdout', tuple_var_assignment_53276_53331)
        
        # Assigning a Name to a Name (line 28):
        # Getting the type of 'tuple_var_assignment_53277' (line 28)
        tuple_var_assignment_53277_53332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'tuple_var_assignment_53277')
        # Assigning a type to the variable 'stderr' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 16), 'stderr', tuple_var_assignment_53277_53332)
        
        
        
        # Getting the type of 'p' (line 29)
        p_53333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 16), 'p')
        # Obtaining the member 'returncode' of a type (line 29)
        returncode_53334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 16), p_53333, 'returncode')
        int_53335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 32), 'int')
        # Applying the binary operator '==' (line 29)
        result_eq_53336 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 16), '==', returncode_53334, int_53335)
        
        # Applying the 'not' unary operator (line 29)
        result_not__53337 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 11), 'not', result_eq_53336)
        
        # Testing the type of an if condition (line 29)
        if_condition_53338 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 29, 8), result_not__53337)
        # Assigning a type to the variable 'if_condition_53338' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'if_condition_53338', if_condition_53338)
        # SSA begins for if statement (line 29)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to RuntimeError(...): (line 30)
        # Processing the call arguments (line 30)
        str_53340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 31), 'str', 'Failed to check dependencies for %s')
        # Getting the type of 'file' (line 30)
        file_53341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 71), 'file', False)
        # Applying the binary operator '%' (line 30)
        result_mod_53342 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 31), '%', str_53340, file_53341)
        
        # Processing the call keyword arguments (line 30)
        kwargs_53343 = {}
        # Getting the type of 'RuntimeError' (line 30)
        RuntimeError_53339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 18), 'RuntimeError', False)
        # Calling RuntimeError(args, kwargs) (line 30)
        RuntimeError_call_result_53344 = invoke(stypy.reporting.localization.Localization(__file__, 30, 18), RuntimeError_53339, *[result_mod_53342], **kwargs_53343)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 30, 12), RuntimeError_call_result_53344, 'raise parameter', BaseException)
        # SSA join for if statement (line 29)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'stdout' (line 32)
        stdout_53345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 15), 'stdout')
        # Assigning a type to the variable 'stypy_return_type' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'stypy_return_type', stdout_53345)
        
        # ################# End of 'get_dependencies(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_dependencies' in the type store
        # Getting the type of 'stypy_return_type' (line 26)
        stypy_return_type_53346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_53346)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_dependencies'
        return stypy_return_type_53346


    @norecursion
    def grep_dependencies(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'grep_dependencies'
        module_type_store = module_type_store.open_function_context('grep_dependencies', 34, 4, False)
        # Assigning a type to the variable 'self' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FindDependenciesLdd.grep_dependencies.__dict__.__setitem__('stypy_localization', localization)
        FindDependenciesLdd.grep_dependencies.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FindDependenciesLdd.grep_dependencies.__dict__.__setitem__('stypy_type_store', module_type_store)
        FindDependenciesLdd.grep_dependencies.__dict__.__setitem__('stypy_function_name', 'FindDependenciesLdd.grep_dependencies')
        FindDependenciesLdd.grep_dependencies.__dict__.__setitem__('stypy_param_names_list', ['file', 'deps'])
        FindDependenciesLdd.grep_dependencies.__dict__.__setitem__('stypy_varargs_param_name', None)
        FindDependenciesLdd.grep_dependencies.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FindDependenciesLdd.grep_dependencies.__dict__.__setitem__('stypy_call_defaults', defaults)
        FindDependenciesLdd.grep_dependencies.__dict__.__setitem__('stypy_call_varargs', varargs)
        FindDependenciesLdd.grep_dependencies.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FindDependenciesLdd.grep_dependencies.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FindDependenciesLdd.grep_dependencies', ['file', 'deps'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'grep_dependencies', localization, ['file', 'deps'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'grep_dependencies(...)' code ##################

        
        # Assigning a Call to a Name (line 35):
        
        # Assigning a Call to a Name (line 35):
        
        # Call to get_dependencies(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'file' (line 35)
        file_53349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 39), 'file', False)
        # Processing the call keyword arguments (line 35)
        kwargs_53350 = {}
        # Getting the type of 'self' (line 35)
        self_53347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 17), 'self', False)
        # Obtaining the member 'get_dependencies' of a type (line 35)
        get_dependencies_53348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 17), self_53347, 'get_dependencies')
        # Calling get_dependencies(args, kwargs) (line 35)
        get_dependencies_call_result_53351 = invoke(stypy.reporting.localization.Localization(__file__, 35, 17), get_dependencies_53348, *[file_53349], **kwargs_53350)
        
        # Assigning a type to the variable 'stdout' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'stdout', get_dependencies_call_result_53351)
        
        # Assigning a Call to a Name (line 37):
        
        # Assigning a Call to a Name (line 37):
        
        # Call to dict(...): (line 37)
        # Processing the call arguments (line 37)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'deps' (line 37)
        deps_53366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 74), 'deps', False)
        comprehension_53367 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 22), deps_53366)
        # Assigning a type to the variable 'dep' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 22), 'dep', comprehension_53367)
        
        # Obtaining an instance of the builtin type 'tuple' (line 37)
        tuple_53353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 37)
        # Adding element type (line 37)
        
        # Call to asbytes(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'dep' (line 37)
        dep_53355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 31), 'dep', False)
        # Processing the call keyword arguments (line 37)
        kwargs_53356 = {}
        # Getting the type of 'asbytes' (line 37)
        asbytes_53354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 23), 'asbytes', False)
        # Calling asbytes(args, kwargs) (line 37)
        asbytes_call_result_53357 = invoke(stypy.reporting.localization.Localization(__file__, 37, 23), asbytes_53354, *[dep_53355], **kwargs_53356)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 23), tuple_53353, asbytes_call_result_53357)
        # Adding element type (line 37)
        
        # Call to compile(...): (line 37)
        # Processing the call arguments (line 37)
        
        # Call to asbytes(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'dep' (line 37)
        dep_53361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 56), 'dep', False)
        # Processing the call keyword arguments (line 37)
        kwargs_53362 = {}
        # Getting the type of 'asbytes' (line 37)
        asbytes_53360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 48), 'asbytes', False)
        # Calling asbytes(args, kwargs) (line 37)
        asbytes_call_result_53363 = invoke(stypy.reporting.localization.Localization(__file__, 37, 48), asbytes_53360, *[dep_53361], **kwargs_53362)
        
        # Processing the call keyword arguments (line 37)
        kwargs_53364 = {}
        # Getting the type of 're' (line 37)
        re_53358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 37), 're', False)
        # Obtaining the member 'compile' of a type (line 37)
        compile_53359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 37), re_53358, 'compile')
        # Calling compile(args, kwargs) (line 37)
        compile_call_result_53365 = invoke(stypy.reporting.localization.Localization(__file__, 37, 37), compile_53359, *[asbytes_call_result_53363], **kwargs_53364)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 23), tuple_53353, compile_call_result_53365)
        
        list_53368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 22), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 22), list_53368, tuple_53353)
        # Processing the call keyword arguments (line 37)
        kwargs_53369 = {}
        # Getting the type of 'dict' (line 37)
        dict_53352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 16), 'dict', False)
        # Calling dict(args, kwargs) (line 37)
        dict_call_result_53370 = invoke(stypy.reporting.localization.Localization(__file__, 37, 16), dict_53352, *[list_53368], **kwargs_53369)
        
        # Assigning a type to the variable 'rdeps' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'rdeps', dict_call_result_53370)
        
        # Assigning a List to a Name (line 38):
        
        # Assigning a List to a Name (line 38):
        
        # Obtaining an instance of the builtin type 'list' (line 38)
        list_53371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 38)
        
        # Assigning a type to the variable 'founds' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'founds', list_53371)
        
        
        # Call to splitlines(...): (line 39)
        # Processing the call keyword arguments (line 39)
        kwargs_53374 = {}
        # Getting the type of 'stdout' (line 39)
        stdout_53372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 17), 'stdout', False)
        # Obtaining the member 'splitlines' of a type (line 39)
        splitlines_53373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 17), stdout_53372, 'splitlines')
        # Calling splitlines(args, kwargs) (line 39)
        splitlines_call_result_53375 = invoke(stypy.reporting.localization.Localization(__file__, 39, 17), splitlines_53373, *[], **kwargs_53374)
        
        # Testing the type of a for loop iterable (line 39)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 39, 8), splitlines_call_result_53375)
        # Getting the type of the for loop variable (line 39)
        for_loop_var_53376 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 39, 8), splitlines_call_result_53375)
        # Assigning a type to the variable 'l' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'l', for_loop_var_53376)
        # SSA begins for a for statement (line 39)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to items(...): (line 40)
        # Processing the call keyword arguments (line 40)
        kwargs_53379 = {}
        # Getting the type of 'rdeps' (line 40)
        rdeps_53377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 24), 'rdeps', False)
        # Obtaining the member 'items' of a type (line 40)
        items_53378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 24), rdeps_53377, 'items')
        # Calling items(args, kwargs) (line 40)
        items_call_result_53380 = invoke(stypy.reporting.localization.Localization(__file__, 40, 24), items_53378, *[], **kwargs_53379)
        
        # Testing the type of a for loop iterable (line 40)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 40, 12), items_call_result_53380)
        # Getting the type of the for loop variable (line 40)
        for_loop_var_53381 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 40, 12), items_call_result_53380)
        # Assigning a type to the variable 'k' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 12), for_loop_var_53381))
        # Assigning a type to the variable 'v' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 12), for_loop_var_53381))
        # SSA begins for a for statement (line 40)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to search(...): (line 41)
        # Processing the call arguments (line 41)
        # Getting the type of 'l' (line 41)
        l_53384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 28), 'l', False)
        # Processing the call keyword arguments (line 41)
        kwargs_53385 = {}
        # Getting the type of 'v' (line 41)
        v_53382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 19), 'v', False)
        # Obtaining the member 'search' of a type (line 41)
        search_53383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 19), v_53382, 'search')
        # Calling search(args, kwargs) (line 41)
        search_call_result_53386 = invoke(stypy.reporting.localization.Localization(__file__, 41, 19), search_53383, *[l_53384], **kwargs_53385)
        
        # Testing the type of an if condition (line 41)
        if_condition_53387 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 16), search_call_result_53386)
        # Assigning a type to the variable 'if_condition_53387' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 16), 'if_condition_53387', if_condition_53387)
        # SSA begins for if statement (line 41)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'k' (line 42)
        k_53390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 34), 'k', False)
        # Processing the call keyword arguments (line 42)
        kwargs_53391 = {}
        # Getting the type of 'founds' (line 42)
        founds_53388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 20), 'founds', False)
        # Obtaining the member 'append' of a type (line 42)
        append_53389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 20), founds_53388, 'append')
        # Calling append(args, kwargs) (line 42)
        append_call_result_53392 = invoke(stypy.reporting.localization.Localization(__file__, 42, 20), append_53389, *[k_53390], **kwargs_53391)
        
        # SSA join for if statement (line 41)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'founds' (line 44)
        founds_53393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 15), 'founds')
        # Assigning a type to the variable 'stypy_return_type' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'stypy_return_type', founds_53393)
        
        # ################# End of 'grep_dependencies(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'grep_dependencies' in the type store
        # Getting the type of 'stypy_return_type' (line 34)
        stypy_return_type_53394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_53394)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'grep_dependencies'
        return stypy_return_type_53394


# Assigning a type to the variable 'FindDependenciesLdd' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'FindDependenciesLdd', FindDependenciesLdd)
# Declaration of the 'TestF77Mismatch' class

class TestF77Mismatch(object, ):

    @norecursion
    def test_lapack(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'test_lapack'
        module_type_store = module_type_store.open_function_context('test_lapack', 48, 4, False)
        # Assigning a type to the variable 'self' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TestF77Mismatch.test_lapack.__dict__.__setitem__('stypy_localization', localization)
        TestF77Mismatch.test_lapack.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TestF77Mismatch.test_lapack.__dict__.__setitem__('stypy_type_store', module_type_store)
        TestF77Mismatch.test_lapack.__dict__.__setitem__('stypy_function_name', 'TestF77Mismatch.test_lapack')
        TestF77Mismatch.test_lapack.__dict__.__setitem__('stypy_param_names_list', [])
        TestF77Mismatch.test_lapack.__dict__.__setitem__('stypy_varargs_param_name', None)
        TestF77Mismatch.test_lapack.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TestF77Mismatch.test_lapack.__dict__.__setitem__('stypy_call_defaults', defaults)
        TestF77Mismatch.test_lapack.__dict__.__setitem__('stypy_call_varargs', varargs)
        TestF77Mismatch.test_lapack.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TestF77Mismatch.test_lapack.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestF77Mismatch.test_lapack', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'test_lapack', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'test_lapack(...)' code ##################

        
        # Assigning a Call to a Name (line 51):
        
        # Assigning a Call to a Name (line 51):
        
        # Call to FindDependenciesLdd(...): (line 51)
        # Processing the call keyword arguments (line 51)
        kwargs_53396 = {}
        # Getting the type of 'FindDependenciesLdd' (line 51)
        FindDependenciesLdd_53395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'FindDependenciesLdd', False)
        # Calling FindDependenciesLdd(args, kwargs) (line 51)
        FindDependenciesLdd_call_result_53397 = invoke(stypy.reporting.localization.Localization(__file__, 51, 12), FindDependenciesLdd_53395, *[], **kwargs_53396)
        
        # Assigning a type to the variable 'f' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'f', FindDependenciesLdd_call_result_53397)
        
        # Assigning a Call to a Name (line 52):
        
        # Assigning a Call to a Name (line 52):
        
        # Call to grep_dependencies(...): (line 52)
        # Processing the call arguments (line 52)
        # Getting the type of 'flapack' (line 52)
        flapack_53400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 35), 'flapack', False)
        # Obtaining the member '__file__' of a type (line 52)
        file___53401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 35), flapack_53400, '__file__')
        
        # Obtaining an instance of the builtin type 'list' (line 53)
        list_53402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 53)
        # Adding element type (line 53)
        str_53403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 36), 'str', 'libg2c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 35), list_53402, str_53403)
        # Adding element type (line 53)
        str_53404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 46), 'str', 'libgfortran')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 35), list_53402, str_53404)
        
        # Processing the call keyword arguments (line 52)
        kwargs_53405 = {}
        # Getting the type of 'f' (line 52)
        f_53398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 15), 'f', False)
        # Obtaining the member 'grep_dependencies' of a type (line 52)
        grep_dependencies_53399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 15), f_53398, 'grep_dependencies')
        # Calling grep_dependencies(args, kwargs) (line 52)
        grep_dependencies_call_result_53406 = invoke(stypy.reporting.localization.Localization(__file__, 52, 15), grep_dependencies_53399, *[file___53401, list_53402], **kwargs_53405)
        
        # Assigning a type to the variable 'deps' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'deps', grep_dependencies_call_result_53406)
        
        # Call to assert_(...): (line 54)
        # Processing the call arguments (line 54)
        
        
        
        # Call to len(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'deps' (line 54)
        deps_53409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 25), 'deps', False)
        # Processing the call keyword arguments (line 54)
        kwargs_53410 = {}
        # Getting the type of 'len' (line 54)
        len_53408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 21), 'len', False)
        # Calling len(args, kwargs) (line 54)
        len_call_result_53411 = invoke(stypy.reporting.localization.Localization(__file__, 54, 21), len_53408, *[deps_53409], **kwargs_53410)
        
        int_53412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 33), 'int')
        # Applying the binary operator '>' (line 54)
        result_gt_53413 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 21), '>', len_call_result_53411, int_53412)
        
        # Applying the 'not' unary operator (line 54)
        result_not__53414 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 16), 'not', result_gt_53413)
        
        str_53415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, (-1)), 'str', 'Both g77 and gfortran runtimes linked in scipy.linalg.flapack ! This is\nlikely to cause random crashes and wrong results. See numpy INSTALL.rst.txt for\nmore information.')
        # Processing the call keyword arguments (line 54)
        kwargs_53416 = {}
        # Getting the type of 'assert_' (line 54)
        assert__53407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'assert_', False)
        # Calling assert_(args, kwargs) (line 54)
        assert__call_result_53417 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), assert__53407, *[result_not__53414, str_53415], **kwargs_53416)
        
        
        # ################# End of 'test_lapack(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'test_lapack' in the type store
        # Getting the type of 'stypy_return_type' (line 48)
        stypy_return_type_53418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_53418)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'test_lapack'
        return stypy_return_type_53418


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 47, 0, False)
        # Assigning a type to the variable 'self' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TestF77Mismatch.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TestF77Mismatch' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'TestF77Mismatch', TestF77Mismatch)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
