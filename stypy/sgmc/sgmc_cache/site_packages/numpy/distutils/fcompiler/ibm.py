
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: import os
4: import re
5: import sys
6: 
7: from numpy.distutils.fcompiler import FCompiler
8: from numpy.distutils.exec_command import exec_command, find_executable
9: from numpy.distutils.misc_util import make_temp_file
10: from distutils import log
11: 
12: compilers = ['IBMFCompiler']
13: 
14: class IBMFCompiler(FCompiler):
15:     compiler_type = 'ibm'
16:     description = 'IBM XL Fortran Compiler'
17:     version_pattern =  r'(xlf\(1\)\s*|)IBM XL Fortran ((Advanced Edition |)Version |Enterprise Edition V|for AIX, V)(?P<version>[^\s*]*)'
18:     #IBM XL Fortran Enterprise Edition V10.1 for AIX \nVersion: 10.01.0000.0004
19: 
20:     executables = {
21:         'version_cmd'  : ["<F77>", "-qversion"],
22:         'compiler_f77' : ["xlf"],
23:         'compiler_fix' : ["xlf90", "-qfixed"],
24:         'compiler_f90' : ["xlf90"],
25:         'linker_so'    : ["xlf95"],
26:         'archiver'     : ["ar", "-cr"],
27:         'ranlib'       : ["ranlib"]
28:         }
29: 
30:     def get_version(self,*args,**kwds):
31:         version = FCompiler.get_version(self,*args,**kwds)
32: 
33:         if version is None and sys.platform.startswith('aix'):
34:             # use lslpp to find out xlf version
35:             lslpp = find_executable('lslpp')
36:             xlf = find_executable('xlf')
37:             if os.path.exists(xlf) and os.path.exists(lslpp):
38:                 s, o = exec_command(lslpp + ' -Lc xlfcmp')
39:                 m = re.search('xlfcmp:(?P<version>\d+([.]\d+)+)', o)
40:                 if m: version = m.group('version')
41: 
42:         xlf_dir = '/etc/opt/ibmcmp/xlf'
43:         if version is None and os.path.isdir(xlf_dir):
44:             # linux:
45:             # If the output of xlf does not contain version info
46:             # (that's the case with xlf 8.1, for instance) then
47:             # let's try another method:
48:             l = sorted(os.listdir(xlf_dir))
49:             l.reverse()
50:             l = [d for d in l if os.path.isfile(os.path.join(xlf_dir, d, 'xlf.cfg'))]
51:             if l:
52:                 from distutils.version import LooseVersion
53:                 self.version = version = LooseVersion(l[0])
54:         return version
55: 
56:     def get_flags(self):
57:         return ['-qextname']
58: 
59:     def get_flags_debug(self):
60:         return ['-g']
61: 
62:     def get_flags_linker_so(self):
63:         opt = []
64:         if sys.platform=='darwin':
65:             opt.append('-Wl,-bundle,-flat_namespace,-undefined,suppress')
66:         else:
67:             opt.append('-bshared')
68:         version = self.get_version(ok_status=[0, 40])
69:         if version is not None:
70:             if sys.platform.startswith('aix'):
71:                 xlf_cfg = '/etc/xlf.cfg'
72:             else:
73:                 xlf_cfg = '/etc/opt/ibmcmp/xlf/%s/xlf.cfg' % version
74:             fo, new_cfg = make_temp_file(suffix='_xlf.cfg')
75:             log.info('Creating '+new_cfg)
76:             fi = open(xlf_cfg, 'r')
77:             crt1_match = re.compile(r'\s*crt\s*[=]\s*(?P<path>.*)/crt1.o').match
78:             for line in fi:
79:                 m = crt1_match(line)
80:                 if m:
81:                     fo.write('crt = %s/bundle1.o\n' % (m.group('path')))
82:                 else:
83:                     fo.write(line)
84:             fi.close()
85:             fo.close()
86:             opt.append('-F'+new_cfg)
87:         return opt
88: 
89:     def get_flags_opt(self):
90:         return ['-O3']
91: 
92: if __name__ == '__main__':
93:     log.set_verbosity(2)
94:     compiler = IBMFCompiler()
95:     compiler.customize()
96:     print(compiler.get_version())
97: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import re' statement (line 4)
import re

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import sys' statement (line 5)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.distutils.fcompiler import FCompiler' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_61883 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.distutils.fcompiler')

if (type(import_61883) is not StypyTypeError):

    if (import_61883 != 'pyd_module'):
        __import__(import_61883)
        sys_modules_61884 = sys.modules[import_61883]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.distutils.fcompiler', sys_modules_61884.module_type_store, module_type_store, ['FCompiler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_61884, sys_modules_61884.module_type_store, module_type_store)
    else:
        from numpy.distutils.fcompiler import FCompiler

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.distutils.fcompiler', None, module_type_store, ['FCompiler'], [FCompiler])

else:
    # Assigning a type to the variable 'numpy.distutils.fcompiler' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.distutils.fcompiler', import_61883)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from numpy.distutils.exec_command import exec_command, find_executable' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_61885 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.distutils.exec_command')

if (type(import_61885) is not StypyTypeError):

    if (import_61885 != 'pyd_module'):
        __import__(import_61885)
        sys_modules_61886 = sys.modules[import_61885]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.distutils.exec_command', sys_modules_61886.module_type_store, module_type_store, ['exec_command', 'find_executable'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_61886, sys_modules_61886.module_type_store, module_type_store)
    else:
        from numpy.distutils.exec_command import exec_command, find_executable

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.distutils.exec_command', None, module_type_store, ['exec_command', 'find_executable'], [exec_command, find_executable])

else:
    # Assigning a type to the variable 'numpy.distutils.exec_command' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.distutils.exec_command', import_61885)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from numpy.distutils.misc_util import make_temp_file' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_61887 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.distutils.misc_util')

if (type(import_61887) is not StypyTypeError):

    if (import_61887 != 'pyd_module'):
        __import__(import_61887)
        sys_modules_61888 = sys.modules[import_61887]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.distutils.misc_util', sys_modules_61888.module_type_store, module_type_store, ['make_temp_file'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_61888, sys_modules_61888.module_type_store, module_type_store)
    else:
        from numpy.distutils.misc_util import make_temp_file

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.distutils.misc_util', None, module_type_store, ['make_temp_file'], [make_temp_file])

else:
    # Assigning a type to the variable 'numpy.distutils.misc_util' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.distutils.misc_util', import_61887)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from distutils import log' statement (line 10)
from distutils import log

import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils', None, module_type_store, ['log'], [log])


# Assigning a List to a Name (line 12):

# Assigning a List to a Name (line 12):

# Obtaining an instance of the builtin type 'list' (line 12)
list_61889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 12)
# Adding element type (line 12)
str_61890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 13), 'str', 'IBMFCompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 12), list_61889, str_61890)

# Assigning a type to the variable 'compilers' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'compilers', list_61889)
# Declaration of the 'IBMFCompiler' class
# Getting the type of 'FCompiler' (line 14)
FCompiler_61891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 19), 'FCompiler')

class IBMFCompiler(FCompiler_61891, ):
    
    # Assigning a Str to a Name (line 15):
    
    # Assigning a Str to a Name (line 16):
    
    # Assigning a Str to a Name (line 17):
    
    # Assigning a Dict to a Name (line 20):

    @norecursion
    def get_version(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_version'
        module_type_store = module_type_store.open_function_context('get_version', 30, 4, False)
        # Assigning a type to the variable 'self' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IBMFCompiler.get_version.__dict__.__setitem__('stypy_localization', localization)
        IBMFCompiler.get_version.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IBMFCompiler.get_version.__dict__.__setitem__('stypy_type_store', module_type_store)
        IBMFCompiler.get_version.__dict__.__setitem__('stypy_function_name', 'IBMFCompiler.get_version')
        IBMFCompiler.get_version.__dict__.__setitem__('stypy_param_names_list', [])
        IBMFCompiler.get_version.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        IBMFCompiler.get_version.__dict__.__setitem__('stypy_kwargs_param_name', 'kwds')
        IBMFCompiler.get_version.__dict__.__setitem__('stypy_call_defaults', defaults)
        IBMFCompiler.get_version.__dict__.__setitem__('stypy_call_varargs', varargs)
        IBMFCompiler.get_version.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IBMFCompiler.get_version.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IBMFCompiler.get_version', [], 'args', 'kwds', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_version', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_version(...)' code ##################

        
        # Assigning a Call to a Name (line 31):
        
        # Assigning a Call to a Name (line 31):
        
        # Call to get_version(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'self' (line 31)
        self_61894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 40), 'self', False)
        # Getting the type of 'args' (line 31)
        args_61895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 46), 'args', False)
        # Processing the call keyword arguments (line 31)
        # Getting the type of 'kwds' (line 31)
        kwds_61896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 53), 'kwds', False)
        kwargs_61897 = {'kwds_61896': kwds_61896}
        # Getting the type of 'FCompiler' (line 31)
        FCompiler_61892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 18), 'FCompiler', False)
        # Obtaining the member 'get_version' of a type (line 31)
        get_version_61893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 18), FCompiler_61892, 'get_version')
        # Calling get_version(args, kwargs) (line 31)
        get_version_call_result_61898 = invoke(stypy.reporting.localization.Localization(__file__, 31, 18), get_version_61893, *[self_61894, args_61895], **kwargs_61897)
        
        # Assigning a type to the variable 'version' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'version', get_version_call_result_61898)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'version' (line 33)
        version_61899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 11), 'version')
        # Getting the type of 'None' (line 33)
        None_61900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 22), 'None')
        # Applying the binary operator 'is' (line 33)
        result_is__61901 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 11), 'is', version_61899, None_61900)
        
        
        # Call to startswith(...): (line 33)
        # Processing the call arguments (line 33)
        str_61905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 55), 'str', 'aix')
        # Processing the call keyword arguments (line 33)
        kwargs_61906 = {}
        # Getting the type of 'sys' (line 33)
        sys_61902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 31), 'sys', False)
        # Obtaining the member 'platform' of a type (line 33)
        platform_61903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 31), sys_61902, 'platform')
        # Obtaining the member 'startswith' of a type (line 33)
        startswith_61904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 31), platform_61903, 'startswith')
        # Calling startswith(args, kwargs) (line 33)
        startswith_call_result_61907 = invoke(stypy.reporting.localization.Localization(__file__, 33, 31), startswith_61904, *[str_61905], **kwargs_61906)
        
        # Applying the binary operator 'and' (line 33)
        result_and_keyword_61908 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 11), 'and', result_is__61901, startswith_call_result_61907)
        
        # Testing the type of an if condition (line 33)
        if_condition_61909 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 33, 8), result_and_keyword_61908)
        # Assigning a type to the variable 'if_condition_61909' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'if_condition_61909', if_condition_61909)
        # SSA begins for if statement (line 33)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 35):
        
        # Assigning a Call to a Name (line 35):
        
        # Call to find_executable(...): (line 35)
        # Processing the call arguments (line 35)
        str_61911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 36), 'str', 'lslpp')
        # Processing the call keyword arguments (line 35)
        kwargs_61912 = {}
        # Getting the type of 'find_executable' (line 35)
        find_executable_61910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 20), 'find_executable', False)
        # Calling find_executable(args, kwargs) (line 35)
        find_executable_call_result_61913 = invoke(stypy.reporting.localization.Localization(__file__, 35, 20), find_executable_61910, *[str_61911], **kwargs_61912)
        
        # Assigning a type to the variable 'lslpp' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'lslpp', find_executable_call_result_61913)
        
        # Assigning a Call to a Name (line 36):
        
        # Assigning a Call to a Name (line 36):
        
        # Call to find_executable(...): (line 36)
        # Processing the call arguments (line 36)
        str_61915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 34), 'str', 'xlf')
        # Processing the call keyword arguments (line 36)
        kwargs_61916 = {}
        # Getting the type of 'find_executable' (line 36)
        find_executable_61914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 18), 'find_executable', False)
        # Calling find_executable(args, kwargs) (line 36)
        find_executable_call_result_61917 = invoke(stypy.reporting.localization.Localization(__file__, 36, 18), find_executable_61914, *[str_61915], **kwargs_61916)
        
        # Assigning a type to the variable 'xlf' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'xlf', find_executable_call_result_61917)
        
        
        # Evaluating a boolean operation
        
        # Call to exists(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'xlf' (line 37)
        xlf_61921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 30), 'xlf', False)
        # Processing the call keyword arguments (line 37)
        kwargs_61922 = {}
        # Getting the type of 'os' (line 37)
        os_61918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 37)
        path_61919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 15), os_61918, 'path')
        # Obtaining the member 'exists' of a type (line 37)
        exists_61920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 15), path_61919, 'exists')
        # Calling exists(args, kwargs) (line 37)
        exists_call_result_61923 = invoke(stypy.reporting.localization.Localization(__file__, 37, 15), exists_61920, *[xlf_61921], **kwargs_61922)
        
        
        # Call to exists(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'lslpp' (line 37)
        lslpp_61927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 54), 'lslpp', False)
        # Processing the call keyword arguments (line 37)
        kwargs_61928 = {}
        # Getting the type of 'os' (line 37)
        os_61924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 39), 'os', False)
        # Obtaining the member 'path' of a type (line 37)
        path_61925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 39), os_61924, 'path')
        # Obtaining the member 'exists' of a type (line 37)
        exists_61926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 39), path_61925, 'exists')
        # Calling exists(args, kwargs) (line 37)
        exists_call_result_61929 = invoke(stypy.reporting.localization.Localization(__file__, 37, 39), exists_61926, *[lslpp_61927], **kwargs_61928)
        
        # Applying the binary operator 'and' (line 37)
        result_and_keyword_61930 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 15), 'and', exists_call_result_61923, exists_call_result_61929)
        
        # Testing the type of an if condition (line 37)
        if_condition_61931 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 37, 12), result_and_keyword_61930)
        # Assigning a type to the variable 'if_condition_61931' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'if_condition_61931', if_condition_61931)
        # SSA begins for if statement (line 37)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 38):
        
        # Assigning a Call to a Name:
        
        # Call to exec_command(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'lslpp' (line 38)
        lslpp_61933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 36), 'lslpp', False)
        str_61934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 44), 'str', ' -Lc xlfcmp')
        # Applying the binary operator '+' (line 38)
        result_add_61935 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 36), '+', lslpp_61933, str_61934)
        
        # Processing the call keyword arguments (line 38)
        kwargs_61936 = {}
        # Getting the type of 'exec_command' (line 38)
        exec_command_61932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 23), 'exec_command', False)
        # Calling exec_command(args, kwargs) (line 38)
        exec_command_call_result_61937 = invoke(stypy.reporting.localization.Localization(__file__, 38, 23), exec_command_61932, *[result_add_61935], **kwargs_61936)
        
        # Assigning a type to the variable 'call_assignment_61877' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'call_assignment_61877', exec_command_call_result_61937)
        
        # Assigning a Call to a Name (line 38):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_61940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 16), 'int')
        # Processing the call keyword arguments
        kwargs_61941 = {}
        # Getting the type of 'call_assignment_61877' (line 38)
        call_assignment_61877_61938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'call_assignment_61877', False)
        # Obtaining the member '__getitem__' of a type (line 38)
        getitem___61939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 16), call_assignment_61877_61938, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_61942 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___61939, *[int_61940], **kwargs_61941)
        
        # Assigning a type to the variable 'call_assignment_61878' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'call_assignment_61878', getitem___call_result_61942)
        
        # Assigning a Name to a Name (line 38):
        # Getting the type of 'call_assignment_61878' (line 38)
        call_assignment_61878_61943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'call_assignment_61878')
        # Assigning a type to the variable 's' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 's', call_assignment_61878_61943)
        
        # Assigning a Call to a Name (line 38):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_61946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 16), 'int')
        # Processing the call keyword arguments
        kwargs_61947 = {}
        # Getting the type of 'call_assignment_61877' (line 38)
        call_assignment_61877_61944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'call_assignment_61877', False)
        # Obtaining the member '__getitem__' of a type (line 38)
        getitem___61945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 16), call_assignment_61877_61944, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_61948 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___61945, *[int_61946], **kwargs_61947)
        
        # Assigning a type to the variable 'call_assignment_61879' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'call_assignment_61879', getitem___call_result_61948)
        
        # Assigning a Name to a Name (line 38):
        # Getting the type of 'call_assignment_61879' (line 38)
        call_assignment_61879_61949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'call_assignment_61879')
        # Assigning a type to the variable 'o' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 19), 'o', call_assignment_61879_61949)
        
        # Assigning a Call to a Name (line 39):
        
        # Assigning a Call to a Name (line 39):
        
        # Call to search(...): (line 39)
        # Processing the call arguments (line 39)
        str_61952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 30), 'str', 'xlfcmp:(?P<version>\\d+([.]\\d+)+)')
        # Getting the type of 'o' (line 39)
        o_61953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 66), 'o', False)
        # Processing the call keyword arguments (line 39)
        kwargs_61954 = {}
        # Getting the type of 're' (line 39)
        re_61950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 20), 're', False)
        # Obtaining the member 'search' of a type (line 39)
        search_61951 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 20), re_61950, 'search')
        # Calling search(args, kwargs) (line 39)
        search_call_result_61955 = invoke(stypy.reporting.localization.Localization(__file__, 39, 20), search_61951, *[str_61952, o_61953], **kwargs_61954)
        
        # Assigning a type to the variable 'm' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 16), 'm', search_call_result_61955)
        
        # Getting the type of 'm' (line 40)
        m_61956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 19), 'm')
        # Testing the type of an if condition (line 40)
        if_condition_61957 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 40, 16), m_61956)
        # Assigning a type to the variable 'if_condition_61957' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 16), 'if_condition_61957', if_condition_61957)
        # SSA begins for if statement (line 40)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 40):
        
        # Assigning a Call to a Name (line 40):
        
        # Call to group(...): (line 40)
        # Processing the call arguments (line 40)
        str_61960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 40), 'str', 'version')
        # Processing the call keyword arguments (line 40)
        kwargs_61961 = {}
        # Getting the type of 'm' (line 40)
        m_61958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 32), 'm', False)
        # Obtaining the member 'group' of a type (line 40)
        group_61959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 32), m_61958, 'group')
        # Calling group(args, kwargs) (line 40)
        group_call_result_61962 = invoke(stypy.reporting.localization.Localization(__file__, 40, 32), group_61959, *[str_61960], **kwargs_61961)
        
        # Assigning a type to the variable 'version' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 22), 'version', group_call_result_61962)
        # SSA join for if statement (line 40)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 37)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 33)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Str to a Name (line 42):
        
        # Assigning a Str to a Name (line 42):
        str_61963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 18), 'str', '/etc/opt/ibmcmp/xlf')
        # Assigning a type to the variable 'xlf_dir' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'xlf_dir', str_61963)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'version' (line 43)
        version_61964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 11), 'version')
        # Getting the type of 'None' (line 43)
        None_61965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 22), 'None')
        # Applying the binary operator 'is' (line 43)
        result_is__61966 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 11), 'is', version_61964, None_61965)
        
        
        # Call to isdir(...): (line 43)
        # Processing the call arguments (line 43)
        # Getting the type of 'xlf_dir' (line 43)
        xlf_dir_61970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 45), 'xlf_dir', False)
        # Processing the call keyword arguments (line 43)
        kwargs_61971 = {}
        # Getting the type of 'os' (line 43)
        os_61967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 31), 'os', False)
        # Obtaining the member 'path' of a type (line 43)
        path_61968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 31), os_61967, 'path')
        # Obtaining the member 'isdir' of a type (line 43)
        isdir_61969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 31), path_61968, 'isdir')
        # Calling isdir(args, kwargs) (line 43)
        isdir_call_result_61972 = invoke(stypy.reporting.localization.Localization(__file__, 43, 31), isdir_61969, *[xlf_dir_61970], **kwargs_61971)
        
        # Applying the binary operator 'and' (line 43)
        result_and_keyword_61973 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 11), 'and', result_is__61966, isdir_call_result_61972)
        
        # Testing the type of an if condition (line 43)
        if_condition_61974 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 43, 8), result_and_keyword_61973)
        # Assigning a type to the variable 'if_condition_61974' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'if_condition_61974', if_condition_61974)
        # SSA begins for if statement (line 43)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 48):
        
        # Assigning a Call to a Name (line 48):
        
        # Call to sorted(...): (line 48)
        # Processing the call arguments (line 48)
        
        # Call to listdir(...): (line 48)
        # Processing the call arguments (line 48)
        # Getting the type of 'xlf_dir' (line 48)
        xlf_dir_61978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 34), 'xlf_dir', False)
        # Processing the call keyword arguments (line 48)
        kwargs_61979 = {}
        # Getting the type of 'os' (line 48)
        os_61976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 23), 'os', False)
        # Obtaining the member 'listdir' of a type (line 48)
        listdir_61977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 23), os_61976, 'listdir')
        # Calling listdir(args, kwargs) (line 48)
        listdir_call_result_61980 = invoke(stypy.reporting.localization.Localization(__file__, 48, 23), listdir_61977, *[xlf_dir_61978], **kwargs_61979)
        
        # Processing the call keyword arguments (line 48)
        kwargs_61981 = {}
        # Getting the type of 'sorted' (line 48)
        sorted_61975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 16), 'sorted', False)
        # Calling sorted(args, kwargs) (line 48)
        sorted_call_result_61982 = invoke(stypy.reporting.localization.Localization(__file__, 48, 16), sorted_61975, *[listdir_call_result_61980], **kwargs_61981)
        
        # Assigning a type to the variable 'l' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'l', sorted_call_result_61982)
        
        # Call to reverse(...): (line 49)
        # Processing the call keyword arguments (line 49)
        kwargs_61985 = {}
        # Getting the type of 'l' (line 49)
        l_61983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 12), 'l', False)
        # Obtaining the member 'reverse' of a type (line 49)
        reverse_61984 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 12), l_61983, 'reverse')
        # Calling reverse(args, kwargs) (line 49)
        reverse_call_result_61986 = invoke(stypy.reporting.localization.Localization(__file__, 49, 12), reverse_61984, *[], **kwargs_61985)
        
        
        # Assigning a ListComp to a Name (line 50):
        
        # Assigning a ListComp to a Name (line 50):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'l' (line 50)
        l_62001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 28), 'l')
        comprehension_62002 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), l_62001)
        # Assigning a type to the variable 'd' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 17), 'd', comprehension_62002)
        
        # Call to isfile(...): (line 50)
        # Processing the call arguments (line 50)
        
        # Call to join(...): (line 50)
        # Processing the call arguments (line 50)
        # Getting the type of 'xlf_dir' (line 50)
        xlf_dir_61994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 61), 'xlf_dir', False)
        # Getting the type of 'd' (line 50)
        d_61995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 70), 'd', False)
        str_61996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 73), 'str', 'xlf.cfg')
        # Processing the call keyword arguments (line 50)
        kwargs_61997 = {}
        # Getting the type of 'os' (line 50)
        os_61991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 48), 'os', False)
        # Obtaining the member 'path' of a type (line 50)
        path_61992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 48), os_61991, 'path')
        # Obtaining the member 'join' of a type (line 50)
        join_61993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 48), path_61992, 'join')
        # Calling join(args, kwargs) (line 50)
        join_call_result_61998 = invoke(stypy.reporting.localization.Localization(__file__, 50, 48), join_61993, *[xlf_dir_61994, d_61995, str_61996], **kwargs_61997)
        
        # Processing the call keyword arguments (line 50)
        kwargs_61999 = {}
        # Getting the type of 'os' (line 50)
        os_61988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 33), 'os', False)
        # Obtaining the member 'path' of a type (line 50)
        path_61989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 33), os_61988, 'path')
        # Obtaining the member 'isfile' of a type (line 50)
        isfile_61990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 33), path_61989, 'isfile')
        # Calling isfile(args, kwargs) (line 50)
        isfile_call_result_62000 = invoke(stypy.reporting.localization.Localization(__file__, 50, 33), isfile_61990, *[join_call_result_61998], **kwargs_61999)
        
        # Getting the type of 'd' (line 50)
        d_61987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 17), 'd')
        list_62003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 17), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), list_62003, d_61987)
        # Assigning a type to the variable 'l' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'l', list_62003)
        
        # Getting the type of 'l' (line 51)
        l_62004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 15), 'l')
        # Testing the type of an if condition (line 51)
        if_condition_62005 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 51, 12), l_62004)
        # Assigning a type to the variable 'if_condition_62005' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'if_condition_62005', if_condition_62005)
        # SSA begins for if statement (line 51)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 52, 16))
        
        # 'from distutils.version import LooseVersion' statement (line 52)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
        import_62006 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 52, 16), 'distutils.version')

        if (type(import_62006) is not StypyTypeError):

            if (import_62006 != 'pyd_module'):
                __import__(import_62006)
                sys_modules_62007 = sys.modules[import_62006]
                import_from_module(stypy.reporting.localization.Localization(__file__, 52, 16), 'distutils.version', sys_modules_62007.module_type_store, module_type_store, ['LooseVersion'])
                nest_module(stypy.reporting.localization.Localization(__file__, 52, 16), __file__, sys_modules_62007, sys_modules_62007.module_type_store, module_type_store)
            else:
                from distutils.version import LooseVersion

                import_from_module(stypy.reporting.localization.Localization(__file__, 52, 16), 'distutils.version', None, module_type_store, ['LooseVersion'], [LooseVersion])

        else:
            # Assigning a type to the variable 'distutils.version' (line 52)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'distutils.version', import_62006)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
        
        
        # Multiple assignment of 2 elements.
        
        # Assigning a Call to a Name (line 53):
        
        # Call to LooseVersion(...): (line 53)
        # Processing the call arguments (line 53)
        
        # Obtaining the type of the subscript
        int_62009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 56), 'int')
        # Getting the type of 'l' (line 53)
        l_62010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 54), 'l', False)
        # Obtaining the member '__getitem__' of a type (line 53)
        getitem___62011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 54), l_62010, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 53)
        subscript_call_result_62012 = invoke(stypy.reporting.localization.Localization(__file__, 53, 54), getitem___62011, int_62009)
        
        # Processing the call keyword arguments (line 53)
        kwargs_62013 = {}
        # Getting the type of 'LooseVersion' (line 53)
        LooseVersion_62008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 41), 'LooseVersion', False)
        # Calling LooseVersion(args, kwargs) (line 53)
        LooseVersion_call_result_62014 = invoke(stypy.reporting.localization.Localization(__file__, 53, 41), LooseVersion_62008, *[subscript_call_result_62012], **kwargs_62013)
        
        # Assigning a type to the variable 'version' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 31), 'version', LooseVersion_call_result_62014)
        
        # Assigning a Name to a Attribute (line 53):
        # Getting the type of 'version' (line 53)
        version_62015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 31), 'version')
        # Getting the type of 'self' (line 53)
        self_62016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 16), 'self')
        # Setting the type of the member 'version' of a type (line 53)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 16), self_62016, 'version', version_62015)
        # SSA join for if statement (line 51)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 43)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'version' (line 54)
        version_62017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 15), 'version')
        # Assigning a type to the variable 'stypy_return_type' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'stypy_return_type', version_62017)
        
        # ################# End of 'get_version(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_version' in the type store
        # Getting the type of 'stypy_return_type' (line 30)
        stypy_return_type_62018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62018)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_version'
        return stypy_return_type_62018


    @norecursion
    def get_flags(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags'
        module_type_store = module_type_store.open_function_context('get_flags', 56, 4, False)
        # Assigning a type to the variable 'self' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IBMFCompiler.get_flags.__dict__.__setitem__('stypy_localization', localization)
        IBMFCompiler.get_flags.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IBMFCompiler.get_flags.__dict__.__setitem__('stypy_type_store', module_type_store)
        IBMFCompiler.get_flags.__dict__.__setitem__('stypy_function_name', 'IBMFCompiler.get_flags')
        IBMFCompiler.get_flags.__dict__.__setitem__('stypy_param_names_list', [])
        IBMFCompiler.get_flags.__dict__.__setitem__('stypy_varargs_param_name', None)
        IBMFCompiler.get_flags.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IBMFCompiler.get_flags.__dict__.__setitem__('stypy_call_defaults', defaults)
        IBMFCompiler.get_flags.__dict__.__setitem__('stypy_call_varargs', varargs)
        IBMFCompiler.get_flags.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IBMFCompiler.get_flags.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IBMFCompiler.get_flags', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags(...)' code ##################

        
        # Obtaining an instance of the builtin type 'list' (line 57)
        list_62019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 57)
        # Adding element type (line 57)
        str_62020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 16), 'str', '-qextname')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 15), list_62019, str_62020)
        
        # Assigning a type to the variable 'stypy_return_type' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'stypy_return_type', list_62019)
        
        # ################# End of 'get_flags(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags' in the type store
        # Getting the type of 'stypy_return_type' (line 56)
        stypy_return_type_62021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62021)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags'
        return stypy_return_type_62021


    @norecursion
    def get_flags_debug(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_debug'
        module_type_store = module_type_store.open_function_context('get_flags_debug', 59, 4, False)
        # Assigning a type to the variable 'self' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IBMFCompiler.get_flags_debug.__dict__.__setitem__('stypy_localization', localization)
        IBMFCompiler.get_flags_debug.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IBMFCompiler.get_flags_debug.__dict__.__setitem__('stypy_type_store', module_type_store)
        IBMFCompiler.get_flags_debug.__dict__.__setitem__('stypy_function_name', 'IBMFCompiler.get_flags_debug')
        IBMFCompiler.get_flags_debug.__dict__.__setitem__('stypy_param_names_list', [])
        IBMFCompiler.get_flags_debug.__dict__.__setitem__('stypy_varargs_param_name', None)
        IBMFCompiler.get_flags_debug.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IBMFCompiler.get_flags_debug.__dict__.__setitem__('stypy_call_defaults', defaults)
        IBMFCompiler.get_flags_debug.__dict__.__setitem__('stypy_call_varargs', varargs)
        IBMFCompiler.get_flags_debug.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IBMFCompiler.get_flags_debug.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IBMFCompiler.get_flags_debug', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_debug', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_debug(...)' code ##################

        
        # Obtaining an instance of the builtin type 'list' (line 60)
        list_62022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 60)
        # Adding element type (line 60)
        str_62023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 16), 'str', '-g')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 15), list_62022, str_62023)
        
        # Assigning a type to the variable 'stypy_return_type' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'stypy_return_type', list_62022)
        
        # ################# End of 'get_flags_debug(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_debug' in the type store
        # Getting the type of 'stypy_return_type' (line 59)
        stypy_return_type_62024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62024)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_debug'
        return stypy_return_type_62024


    @norecursion
    def get_flags_linker_so(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_linker_so'
        module_type_store = module_type_store.open_function_context('get_flags_linker_so', 62, 4, False)
        # Assigning a type to the variable 'self' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IBMFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_localization', localization)
        IBMFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IBMFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_type_store', module_type_store)
        IBMFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_function_name', 'IBMFCompiler.get_flags_linker_so')
        IBMFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_param_names_list', [])
        IBMFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_varargs_param_name', None)
        IBMFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IBMFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_call_defaults', defaults)
        IBMFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_call_varargs', varargs)
        IBMFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IBMFCompiler.get_flags_linker_so.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IBMFCompiler.get_flags_linker_so', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_linker_so', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_linker_so(...)' code ##################

        
        # Assigning a List to a Name (line 63):
        
        # Assigning a List to a Name (line 63):
        
        # Obtaining an instance of the builtin type 'list' (line 63)
        list_62025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 63)
        
        # Assigning a type to the variable 'opt' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'opt', list_62025)
        
        
        # Getting the type of 'sys' (line 64)
        sys_62026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 11), 'sys')
        # Obtaining the member 'platform' of a type (line 64)
        platform_62027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 11), sys_62026, 'platform')
        str_62028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 25), 'str', 'darwin')
        # Applying the binary operator '==' (line 64)
        result_eq_62029 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 11), '==', platform_62027, str_62028)
        
        # Testing the type of an if condition (line 64)
        if_condition_62030 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 64, 8), result_eq_62029)
        # Assigning a type to the variable 'if_condition_62030' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'if_condition_62030', if_condition_62030)
        # SSA begins for if statement (line 64)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 65)
        # Processing the call arguments (line 65)
        str_62033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 23), 'str', '-Wl,-bundle,-flat_namespace,-undefined,suppress')
        # Processing the call keyword arguments (line 65)
        kwargs_62034 = {}
        # Getting the type of 'opt' (line 65)
        opt_62031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'opt', False)
        # Obtaining the member 'append' of a type (line 65)
        append_62032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 12), opt_62031, 'append')
        # Calling append(args, kwargs) (line 65)
        append_call_result_62035 = invoke(stypy.reporting.localization.Localization(__file__, 65, 12), append_62032, *[str_62033], **kwargs_62034)
        
        # SSA branch for the else part of an if statement (line 64)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 67)
        # Processing the call arguments (line 67)
        str_62038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 23), 'str', '-bshared')
        # Processing the call keyword arguments (line 67)
        kwargs_62039 = {}
        # Getting the type of 'opt' (line 67)
        opt_62036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'opt', False)
        # Obtaining the member 'append' of a type (line 67)
        append_62037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 12), opt_62036, 'append')
        # Calling append(args, kwargs) (line 67)
        append_call_result_62040 = invoke(stypy.reporting.localization.Localization(__file__, 67, 12), append_62037, *[str_62038], **kwargs_62039)
        
        # SSA join for if statement (line 64)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 68):
        
        # Assigning a Call to a Name (line 68):
        
        # Call to get_version(...): (line 68)
        # Processing the call keyword arguments (line 68)
        
        # Obtaining an instance of the builtin type 'list' (line 68)
        list_62043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 68)
        # Adding element type (line 68)
        int_62044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 45), list_62043, int_62044)
        # Adding element type (line 68)
        int_62045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 49), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 45), list_62043, int_62045)
        
        keyword_62046 = list_62043
        kwargs_62047 = {'ok_status': keyword_62046}
        # Getting the type of 'self' (line 68)
        self_62041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 18), 'self', False)
        # Obtaining the member 'get_version' of a type (line 68)
        get_version_62042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 18), self_62041, 'get_version')
        # Calling get_version(args, kwargs) (line 68)
        get_version_call_result_62048 = invoke(stypy.reporting.localization.Localization(__file__, 68, 18), get_version_62042, *[], **kwargs_62047)
        
        # Assigning a type to the variable 'version' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'version', get_version_call_result_62048)
        
        # Type idiom detected: calculating its left and rigth part (line 69)
        # Getting the type of 'version' (line 69)
        version_62049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'version')
        # Getting the type of 'None' (line 69)
        None_62050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 26), 'None')
        
        (may_be_62051, more_types_in_union_62052) = may_not_be_none(version_62049, None_62050)

        if may_be_62051:

            if more_types_in_union_62052:
                # Runtime conditional SSA (line 69)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # Call to startswith(...): (line 70)
            # Processing the call arguments (line 70)
            str_62056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 39), 'str', 'aix')
            # Processing the call keyword arguments (line 70)
            kwargs_62057 = {}
            # Getting the type of 'sys' (line 70)
            sys_62053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 15), 'sys', False)
            # Obtaining the member 'platform' of a type (line 70)
            platform_62054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 15), sys_62053, 'platform')
            # Obtaining the member 'startswith' of a type (line 70)
            startswith_62055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 15), platform_62054, 'startswith')
            # Calling startswith(args, kwargs) (line 70)
            startswith_call_result_62058 = invoke(stypy.reporting.localization.Localization(__file__, 70, 15), startswith_62055, *[str_62056], **kwargs_62057)
            
            # Testing the type of an if condition (line 70)
            if_condition_62059 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 70, 12), startswith_call_result_62058)
            # Assigning a type to the variable 'if_condition_62059' (line 70)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'if_condition_62059', if_condition_62059)
            # SSA begins for if statement (line 70)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Str to a Name (line 71):
            
            # Assigning a Str to a Name (line 71):
            str_62060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 26), 'str', '/etc/xlf.cfg')
            # Assigning a type to the variable 'xlf_cfg' (line 71)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 16), 'xlf_cfg', str_62060)
            # SSA branch for the else part of an if statement (line 70)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a BinOp to a Name (line 73):
            
            # Assigning a BinOp to a Name (line 73):
            str_62061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 26), 'str', '/etc/opt/ibmcmp/xlf/%s/xlf.cfg')
            # Getting the type of 'version' (line 73)
            version_62062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 61), 'version')
            # Applying the binary operator '%' (line 73)
            result_mod_62063 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 26), '%', str_62061, version_62062)
            
            # Assigning a type to the variable 'xlf_cfg' (line 73)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 16), 'xlf_cfg', result_mod_62063)
            # SSA join for if statement (line 70)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Call to a Tuple (line 74):
            
            # Assigning a Call to a Name:
            
            # Call to make_temp_file(...): (line 74)
            # Processing the call keyword arguments (line 74)
            str_62065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 48), 'str', '_xlf.cfg')
            keyword_62066 = str_62065
            kwargs_62067 = {'suffix': keyword_62066}
            # Getting the type of 'make_temp_file' (line 74)
            make_temp_file_62064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 26), 'make_temp_file', False)
            # Calling make_temp_file(args, kwargs) (line 74)
            make_temp_file_call_result_62068 = invoke(stypy.reporting.localization.Localization(__file__, 74, 26), make_temp_file_62064, *[], **kwargs_62067)
            
            # Assigning a type to the variable 'call_assignment_61880' (line 74)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'call_assignment_61880', make_temp_file_call_result_62068)
            
            # Assigning a Call to a Name (line 74):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_62071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 12), 'int')
            # Processing the call keyword arguments
            kwargs_62072 = {}
            # Getting the type of 'call_assignment_61880' (line 74)
            call_assignment_61880_62069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'call_assignment_61880', False)
            # Obtaining the member '__getitem__' of a type (line 74)
            getitem___62070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 12), call_assignment_61880_62069, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_62073 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___62070, *[int_62071], **kwargs_62072)
            
            # Assigning a type to the variable 'call_assignment_61881' (line 74)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'call_assignment_61881', getitem___call_result_62073)
            
            # Assigning a Name to a Name (line 74):
            # Getting the type of 'call_assignment_61881' (line 74)
            call_assignment_61881_62074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'call_assignment_61881')
            # Assigning a type to the variable 'fo' (line 74)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'fo', call_assignment_61881_62074)
            
            # Assigning a Call to a Name (line 74):
            
            # Call to __getitem__(...):
            # Processing the call arguments
            int_62077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 12), 'int')
            # Processing the call keyword arguments
            kwargs_62078 = {}
            # Getting the type of 'call_assignment_61880' (line 74)
            call_assignment_61880_62075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'call_assignment_61880', False)
            # Obtaining the member '__getitem__' of a type (line 74)
            getitem___62076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 12), call_assignment_61880_62075, '__getitem__')
            # Calling __getitem__(args, kwargs)
            getitem___call_result_62079 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___62076, *[int_62077], **kwargs_62078)
            
            # Assigning a type to the variable 'call_assignment_61882' (line 74)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'call_assignment_61882', getitem___call_result_62079)
            
            # Assigning a Name to a Name (line 74):
            # Getting the type of 'call_assignment_61882' (line 74)
            call_assignment_61882_62080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'call_assignment_61882')
            # Assigning a type to the variable 'new_cfg' (line 74)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 16), 'new_cfg', call_assignment_61882_62080)
            
            # Call to info(...): (line 75)
            # Processing the call arguments (line 75)
            str_62083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 21), 'str', 'Creating ')
            # Getting the type of 'new_cfg' (line 75)
            new_cfg_62084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 33), 'new_cfg', False)
            # Applying the binary operator '+' (line 75)
            result_add_62085 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 21), '+', str_62083, new_cfg_62084)
            
            # Processing the call keyword arguments (line 75)
            kwargs_62086 = {}
            # Getting the type of 'log' (line 75)
            log_62081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'log', False)
            # Obtaining the member 'info' of a type (line 75)
            info_62082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 12), log_62081, 'info')
            # Calling info(args, kwargs) (line 75)
            info_call_result_62087 = invoke(stypy.reporting.localization.Localization(__file__, 75, 12), info_62082, *[result_add_62085], **kwargs_62086)
            
            
            # Assigning a Call to a Name (line 76):
            
            # Assigning a Call to a Name (line 76):
            
            # Call to open(...): (line 76)
            # Processing the call arguments (line 76)
            # Getting the type of 'xlf_cfg' (line 76)
            xlf_cfg_62089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 22), 'xlf_cfg', False)
            str_62090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 31), 'str', 'r')
            # Processing the call keyword arguments (line 76)
            kwargs_62091 = {}
            # Getting the type of 'open' (line 76)
            open_62088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 17), 'open', False)
            # Calling open(args, kwargs) (line 76)
            open_call_result_62092 = invoke(stypy.reporting.localization.Localization(__file__, 76, 17), open_62088, *[xlf_cfg_62089, str_62090], **kwargs_62091)
            
            # Assigning a type to the variable 'fi' (line 76)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'fi', open_call_result_62092)
            
            # Assigning a Attribute to a Name (line 77):
            
            # Assigning a Attribute to a Name (line 77):
            
            # Call to compile(...): (line 77)
            # Processing the call arguments (line 77)
            str_62095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 36), 'str', '\\s*crt\\s*[=]\\s*(?P<path>.*)/crt1.o')
            # Processing the call keyword arguments (line 77)
            kwargs_62096 = {}
            # Getting the type of 're' (line 77)
            re_62093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 25), 're', False)
            # Obtaining the member 'compile' of a type (line 77)
            compile_62094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 25), re_62093, 'compile')
            # Calling compile(args, kwargs) (line 77)
            compile_call_result_62097 = invoke(stypy.reporting.localization.Localization(__file__, 77, 25), compile_62094, *[str_62095], **kwargs_62096)
            
            # Obtaining the member 'match' of a type (line 77)
            match_62098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 25), compile_call_result_62097, 'match')
            # Assigning a type to the variable 'crt1_match' (line 77)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'crt1_match', match_62098)
            
            # Getting the type of 'fi' (line 78)
            fi_62099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 24), 'fi')
            # Testing the type of a for loop iterable (line 78)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 78, 12), fi_62099)
            # Getting the type of the for loop variable (line 78)
            for_loop_var_62100 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 78, 12), fi_62099)
            # Assigning a type to the variable 'line' (line 78)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'line', for_loop_var_62100)
            # SSA begins for a for statement (line 78)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 79):
            
            # Assigning a Call to a Name (line 79):
            
            # Call to crt1_match(...): (line 79)
            # Processing the call arguments (line 79)
            # Getting the type of 'line' (line 79)
            line_62102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 31), 'line', False)
            # Processing the call keyword arguments (line 79)
            kwargs_62103 = {}
            # Getting the type of 'crt1_match' (line 79)
            crt1_match_62101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 20), 'crt1_match', False)
            # Calling crt1_match(args, kwargs) (line 79)
            crt1_match_call_result_62104 = invoke(stypy.reporting.localization.Localization(__file__, 79, 20), crt1_match_62101, *[line_62102], **kwargs_62103)
            
            # Assigning a type to the variable 'm' (line 79)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 16), 'm', crt1_match_call_result_62104)
            
            # Getting the type of 'm' (line 80)
            m_62105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 19), 'm')
            # Testing the type of an if condition (line 80)
            if_condition_62106 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 16), m_62105)
            # Assigning a type to the variable 'if_condition_62106' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 16), 'if_condition_62106', if_condition_62106)
            # SSA begins for if statement (line 80)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to write(...): (line 81)
            # Processing the call arguments (line 81)
            str_62109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 29), 'str', 'crt = %s/bundle1.o\n')
            
            # Call to group(...): (line 81)
            # Processing the call arguments (line 81)
            str_62112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 63), 'str', 'path')
            # Processing the call keyword arguments (line 81)
            kwargs_62113 = {}
            # Getting the type of 'm' (line 81)
            m_62110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 55), 'm', False)
            # Obtaining the member 'group' of a type (line 81)
            group_62111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 55), m_62110, 'group')
            # Calling group(args, kwargs) (line 81)
            group_call_result_62114 = invoke(stypy.reporting.localization.Localization(__file__, 81, 55), group_62111, *[str_62112], **kwargs_62113)
            
            # Applying the binary operator '%' (line 81)
            result_mod_62115 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 29), '%', str_62109, group_call_result_62114)
            
            # Processing the call keyword arguments (line 81)
            kwargs_62116 = {}
            # Getting the type of 'fo' (line 81)
            fo_62107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 20), 'fo', False)
            # Obtaining the member 'write' of a type (line 81)
            write_62108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 20), fo_62107, 'write')
            # Calling write(args, kwargs) (line 81)
            write_call_result_62117 = invoke(stypy.reporting.localization.Localization(__file__, 81, 20), write_62108, *[result_mod_62115], **kwargs_62116)
            
            # SSA branch for the else part of an if statement (line 80)
            module_type_store.open_ssa_branch('else')
            
            # Call to write(...): (line 83)
            # Processing the call arguments (line 83)
            # Getting the type of 'line' (line 83)
            line_62120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 29), 'line', False)
            # Processing the call keyword arguments (line 83)
            kwargs_62121 = {}
            # Getting the type of 'fo' (line 83)
            fo_62118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), 'fo', False)
            # Obtaining the member 'write' of a type (line 83)
            write_62119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 20), fo_62118, 'write')
            # Calling write(args, kwargs) (line 83)
            write_call_result_62122 = invoke(stypy.reporting.localization.Localization(__file__, 83, 20), write_62119, *[line_62120], **kwargs_62121)
            
            # SSA join for if statement (line 80)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Call to close(...): (line 84)
            # Processing the call keyword arguments (line 84)
            kwargs_62125 = {}
            # Getting the type of 'fi' (line 84)
            fi_62123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'fi', False)
            # Obtaining the member 'close' of a type (line 84)
            close_62124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 12), fi_62123, 'close')
            # Calling close(args, kwargs) (line 84)
            close_call_result_62126 = invoke(stypy.reporting.localization.Localization(__file__, 84, 12), close_62124, *[], **kwargs_62125)
            
            
            # Call to close(...): (line 85)
            # Processing the call keyword arguments (line 85)
            kwargs_62129 = {}
            # Getting the type of 'fo' (line 85)
            fo_62127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'fo', False)
            # Obtaining the member 'close' of a type (line 85)
            close_62128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 12), fo_62127, 'close')
            # Calling close(args, kwargs) (line 85)
            close_call_result_62130 = invoke(stypy.reporting.localization.Localization(__file__, 85, 12), close_62128, *[], **kwargs_62129)
            
            
            # Call to append(...): (line 86)
            # Processing the call arguments (line 86)
            str_62133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 23), 'str', '-F')
            # Getting the type of 'new_cfg' (line 86)
            new_cfg_62134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 28), 'new_cfg', False)
            # Applying the binary operator '+' (line 86)
            result_add_62135 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 23), '+', str_62133, new_cfg_62134)
            
            # Processing the call keyword arguments (line 86)
            kwargs_62136 = {}
            # Getting the type of 'opt' (line 86)
            opt_62131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'opt', False)
            # Obtaining the member 'append' of a type (line 86)
            append_62132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 12), opt_62131, 'append')
            # Calling append(args, kwargs) (line 86)
            append_call_result_62137 = invoke(stypy.reporting.localization.Localization(__file__, 86, 12), append_62132, *[result_add_62135], **kwargs_62136)
            

            if more_types_in_union_62052:
                # SSA join for if statement (line 69)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'opt' (line 87)
        opt_62138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 15), 'opt')
        # Assigning a type to the variable 'stypy_return_type' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'stypy_return_type', opt_62138)
        
        # ################# End of 'get_flags_linker_so(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_linker_so' in the type store
        # Getting the type of 'stypy_return_type' (line 62)
        stypy_return_type_62139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62139)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_linker_so'
        return stypy_return_type_62139


    @norecursion
    def get_flags_opt(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_opt'
        module_type_store = module_type_store.open_function_context('get_flags_opt', 89, 4, False)
        # Assigning a type to the variable 'self' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        IBMFCompiler.get_flags_opt.__dict__.__setitem__('stypy_localization', localization)
        IBMFCompiler.get_flags_opt.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        IBMFCompiler.get_flags_opt.__dict__.__setitem__('stypy_type_store', module_type_store)
        IBMFCompiler.get_flags_opt.__dict__.__setitem__('stypy_function_name', 'IBMFCompiler.get_flags_opt')
        IBMFCompiler.get_flags_opt.__dict__.__setitem__('stypy_param_names_list', [])
        IBMFCompiler.get_flags_opt.__dict__.__setitem__('stypy_varargs_param_name', None)
        IBMFCompiler.get_flags_opt.__dict__.__setitem__('stypy_kwargs_param_name', None)
        IBMFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_defaults', defaults)
        IBMFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_varargs', varargs)
        IBMFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        IBMFCompiler.get_flags_opt.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IBMFCompiler.get_flags_opt', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_opt', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_opt(...)' code ##################

        
        # Obtaining an instance of the builtin type 'list' (line 90)
        list_62140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 90)
        # Adding element type (line 90)
        str_62141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 16), 'str', '-O3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 15), list_62140, str_62141)
        
        # Assigning a type to the variable 'stypy_return_type' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'stypy_return_type', list_62140)
        
        # ################# End of 'get_flags_opt(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_opt' in the type store
        # Getting the type of 'stypy_return_type' (line 89)
        stypy_return_type_62142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62142)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_opt'
        return stypy_return_type_62142


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 14, 0, False)
        # Assigning a type to the variable 'self' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'IBMFCompiler.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'IBMFCompiler' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'IBMFCompiler', IBMFCompiler)

# Assigning a Str to a Name (line 15):
str_62143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 20), 'str', 'ibm')
# Getting the type of 'IBMFCompiler'
IBMFCompiler_62144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IBMFCompiler')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IBMFCompiler_62144, 'compiler_type', str_62143)

# Assigning a Str to a Name (line 16):
str_62145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 18), 'str', 'IBM XL Fortran Compiler')
# Getting the type of 'IBMFCompiler'
IBMFCompiler_62146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IBMFCompiler')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IBMFCompiler_62146, 'description', str_62145)

# Assigning a Str to a Name (line 17):
str_62147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 23), 'str', '(xlf\\(1\\)\\s*|)IBM XL Fortran ((Advanced Edition |)Version |Enterprise Edition V|for AIX, V)(?P<version>[^\\s*]*)')
# Getting the type of 'IBMFCompiler'
IBMFCompiler_62148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IBMFCompiler')
# Setting the type of the member 'version_pattern' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IBMFCompiler_62148, 'version_pattern', str_62147)

# Assigning a Dict to a Name (line 20):

# Obtaining an instance of the builtin type 'dict' (line 20)
dict_62149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 20)
# Adding element type (key, value) (line 20)
str_62150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 8), 'str', 'version_cmd')

# Obtaining an instance of the builtin type 'list' (line 21)
list_62151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 21)
# Adding element type (line 21)
str_62152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 26), 'str', '<F77>')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 25), list_62151, str_62152)
# Adding element type (line 21)
str_62153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 35), 'str', '-qversion')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 25), list_62151, str_62153)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 18), dict_62149, (str_62150, list_62151))
# Adding element type (key, value) (line 20)
str_62154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 8), 'str', 'compiler_f77')

# Obtaining an instance of the builtin type 'list' (line 22)
list_62155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 22)
# Adding element type (line 22)
str_62156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 26), 'str', 'xlf')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 25), list_62155, str_62156)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 18), dict_62149, (str_62154, list_62155))
# Adding element type (key, value) (line 20)
str_62157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 8), 'str', 'compiler_fix')

# Obtaining an instance of the builtin type 'list' (line 23)
list_62158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 23)
# Adding element type (line 23)
str_62159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 26), 'str', 'xlf90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 25), list_62158, str_62159)
# Adding element type (line 23)
str_62160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 35), 'str', '-qfixed')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 25), list_62158, str_62160)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 18), dict_62149, (str_62157, list_62158))
# Adding element type (key, value) (line 20)
str_62161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 8), 'str', 'compiler_f90')

# Obtaining an instance of the builtin type 'list' (line 24)
list_62162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 24)
# Adding element type (line 24)
str_62163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 26), 'str', 'xlf90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 25), list_62162, str_62163)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 18), dict_62149, (str_62161, list_62162))
# Adding element type (key, value) (line 20)
str_62164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 8), 'str', 'linker_so')

# Obtaining an instance of the builtin type 'list' (line 25)
list_62165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 25)
# Adding element type (line 25)
str_62166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 26), 'str', 'xlf95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 25), list_62165, str_62166)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 18), dict_62149, (str_62164, list_62165))
# Adding element type (key, value) (line 20)
str_62167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 8), 'str', 'archiver')

# Obtaining an instance of the builtin type 'list' (line 26)
list_62168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 26)
# Adding element type (line 26)
str_62169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 26), 'str', 'ar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 25), list_62168, str_62169)
# Adding element type (line 26)
str_62170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 32), 'str', '-cr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 25), list_62168, str_62170)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 18), dict_62149, (str_62167, list_62168))
# Adding element type (key, value) (line 20)
str_62171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 8), 'str', 'ranlib')

# Obtaining an instance of the builtin type 'list' (line 27)
list_62172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 27)
# Adding element type (line 27)
str_62173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 26), 'str', 'ranlib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 25), list_62172, str_62173)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 18), dict_62149, (str_62171, list_62172))

# Getting the type of 'IBMFCompiler'
IBMFCompiler_62174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'IBMFCompiler')
# Setting the type of the member 'executables' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), IBMFCompiler_62174, 'executables', dict_62149)

if (__name__ == '__main__'):
    
    # Call to set_verbosity(...): (line 93)
    # Processing the call arguments (line 93)
    int_62177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 22), 'int')
    # Processing the call keyword arguments (line 93)
    kwargs_62178 = {}
    # Getting the type of 'log' (line 93)
    log_62175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'log', False)
    # Obtaining the member 'set_verbosity' of a type (line 93)
    set_verbosity_62176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 4), log_62175, 'set_verbosity')
    # Calling set_verbosity(args, kwargs) (line 93)
    set_verbosity_call_result_62179 = invoke(stypy.reporting.localization.Localization(__file__, 93, 4), set_verbosity_62176, *[int_62177], **kwargs_62178)
    
    
    # Assigning a Call to a Name (line 94):
    
    # Assigning a Call to a Name (line 94):
    
    # Call to IBMFCompiler(...): (line 94)
    # Processing the call keyword arguments (line 94)
    kwargs_62181 = {}
    # Getting the type of 'IBMFCompiler' (line 94)
    IBMFCompiler_62180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 15), 'IBMFCompiler', False)
    # Calling IBMFCompiler(args, kwargs) (line 94)
    IBMFCompiler_call_result_62182 = invoke(stypy.reporting.localization.Localization(__file__, 94, 15), IBMFCompiler_62180, *[], **kwargs_62181)
    
    # Assigning a type to the variable 'compiler' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'compiler', IBMFCompiler_call_result_62182)
    
    # Call to customize(...): (line 95)
    # Processing the call keyword arguments (line 95)
    kwargs_62185 = {}
    # Getting the type of 'compiler' (line 95)
    compiler_62183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'compiler', False)
    # Obtaining the member 'customize' of a type (line 95)
    customize_62184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 4), compiler_62183, 'customize')
    # Calling customize(args, kwargs) (line 95)
    customize_call_result_62186 = invoke(stypy.reporting.localization.Localization(__file__, 95, 4), customize_62184, *[], **kwargs_62185)
    
    
    # Call to print(...): (line 96)
    # Processing the call arguments (line 96)
    
    # Call to get_version(...): (line 96)
    # Processing the call keyword arguments (line 96)
    kwargs_62190 = {}
    # Getting the type of 'compiler' (line 96)
    compiler_62188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 10), 'compiler', False)
    # Obtaining the member 'get_version' of a type (line 96)
    get_version_62189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 10), compiler_62188, 'get_version')
    # Calling get_version(args, kwargs) (line 96)
    get_version_call_result_62191 = invoke(stypy.reporting.localization.Localization(__file__, 96, 10), get_version_62189, *[], **kwargs_62190)
    
    # Processing the call keyword arguments (line 96)
    kwargs_62192 = {}
    # Getting the type of 'print' (line 96)
    print_62187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'print', False)
    # Calling print(args, kwargs) (line 96)
    print_call_result_62193 = invoke(stypy.reporting.localization.Localization(__file__, 96, 4), print_62187, *[get_version_call_result_62191], **kwargs_62192)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
