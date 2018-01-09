
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: unixccompiler - can handle very long argument lists for ar.
3: 
4: '''
5: from __future__ import division, absolute_import, print_function
6: 
7: import os
8: 
9: from distutils.errors import DistutilsExecError, CompileError
10: from distutils.unixccompiler import *
11: from numpy.distutils.ccompiler import replace_method
12: from numpy.distutils.compat import get_exception
13: 
14: if sys.version_info[0] < 3:
15:     from . import log
16: else:
17:     from numpy.distutils import log
18: 
19: # Note that UnixCCompiler._compile appeared in Python 2.3
20: def UnixCCompiler__compile(self, obj, src, ext, cc_args, extra_postargs, pp_opts):
21:     '''Compile a single source files with a Unix-style compiler.'''
22:     # HP ad-hoc fix, see ticket 1383
23:     ccomp = self.compiler_so
24:     if ccomp[0] == 'aCC':
25:         # remove flags that will trigger ANSI-C mode for aCC
26:         if '-Ae' in ccomp:
27:             ccomp.remove('-Ae')
28:         if '-Aa' in ccomp:
29:             ccomp.remove('-Aa')
30:         # add flags for (almost) sane C++ handling
31:         ccomp += ['-AA']
32:         self.compiler_so = ccomp
33:     # ensure OPT environment variable is read
34:     if 'OPT' in os.environ:
35:         from distutils.sysconfig import get_config_vars
36:         opt = " ".join(os.environ['OPT'].split())
37:         gcv_opt = " ".join(get_config_vars('OPT')[0].split())
38:         ccomp_s = " ".join(self.compiler_so)
39:         if opt not in ccomp_s:
40:             ccomp_s = ccomp_s.replace(gcv_opt, opt)
41:             self.compiler_so = ccomp_s.split()
42:         llink_s = " ".join(self.linker_so)
43:         if opt not in llink_s:
44:             self.linker_so = llink_s.split() + opt.split()
45: 
46:     display = '%s: %s' % (os.path.basename(self.compiler_so[0]), src)
47:     try:
48:         self.spawn(self.compiler_so + cc_args + [src, '-o', obj] +
49:                    extra_postargs, display = display)
50:     except DistutilsExecError:
51:         msg = str(get_exception())
52:         raise CompileError(msg)
53: 
54: replace_method(UnixCCompiler, '_compile', UnixCCompiler__compile)
55: 
56: 
57: def UnixCCompiler_create_static_lib(self, objects, output_libname,
58:                                     output_dir=None, debug=0, target_lang=None):
59:     '''
60:     Build a static library in a separate sub-process.
61: 
62:     Parameters
63:     ----------
64:     objects : list or tuple of str
65:         List of paths to object files used to build the static library.
66:     output_libname : str
67:         The library name as an absolute or relative (if `output_dir` is used)
68:         path.
69:     output_dir : str, optional
70:         The path to the output directory. Default is None, in which case
71:         the ``output_dir`` attribute of the UnixCCompiler instance.
72:     debug : bool, optional
73:         This parameter is not used.
74:     target_lang : str, optional
75:         This parameter is not used.
76: 
77:     Returns
78:     -------
79:     None
80: 
81:     '''
82:     objects, output_dir = self._fix_object_args(objects, output_dir)
83: 
84:     output_filename = \
85:                     self.library_filename(output_libname, output_dir=output_dir)
86: 
87:     if self._need_link(objects, output_filename):
88:         try:
89:             # previous .a may be screwed up; best to remove it first
90:             # and recreate.
91:             # Also, ar on OS X doesn't handle updating universal archives
92:             os.unlink(output_filename)
93:         except (IOError, OSError):
94:             pass
95:         self.mkpath(os.path.dirname(output_filename))
96:         tmp_objects = objects + self.objects
97:         while tmp_objects:
98:             objects = tmp_objects[:50]
99:             tmp_objects = tmp_objects[50:]
100:             display = '%s: adding %d object files to %s' % (
101:                            os.path.basename(self.archiver[0]),
102:                            len(objects), output_filename)
103:             self.spawn(self.archiver + [output_filename] + objects,
104:                        display = display)
105: 
106:         # Not many Unices required ranlib anymore -- SunOS 4.x is, I
107:         # think the only major Unix that does.  Maybe we need some
108:         # platform intelligence here to skip ranlib if it's not
109:         # needed -- or maybe Python's configure script took care of
110:         # it for us, hence the check for leading colon.
111:         if self.ranlib:
112:             display = '%s:@ %s' % (os.path.basename(self.ranlib[0]),
113:                                    output_filename)
114:             try:
115:                 self.spawn(self.ranlib + [output_filename],
116:                            display = display)
117:             except DistutilsExecError:
118:                 msg = str(get_exception())
119:                 raise LibError(msg)
120:     else:
121:         log.debug("skipping %s (up-to-date)", output_filename)
122:     return
123: 
124: replace_method(UnixCCompiler, 'create_static_lib',
125:                UnixCCompiler_create_static_lib)
126: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_51768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', '\nunixccompiler - can handle very long argument lists for ar.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import os' statement (line 7)
import os

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from distutils.errors import DistutilsExecError, CompileError' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_51769 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.errors')

if (type(import_51769) is not StypyTypeError):

    if (import_51769 != 'pyd_module'):
        __import__(import_51769)
        sys_modules_51770 = sys.modules[import_51769]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.errors', sys_modules_51770.module_type_store, module_type_store, ['DistutilsExecError', 'CompileError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_51770, sys_modules_51770.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsExecError, CompileError

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.errors', None, module_type_store, ['DistutilsExecError', 'CompileError'], [DistutilsExecError, CompileError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.errors', import_51769)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from distutils.unixccompiler import ' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_51771 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.unixccompiler')

if (type(import_51771) is not StypyTypeError):

    if (import_51771 != 'pyd_module'):
        __import__(import_51771)
        sys_modules_51772 = sys.modules[import_51771]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.unixccompiler', sys_modules_51772.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_51772, sys_modules_51772.module_type_store, module_type_store)
    else:
        from distutils.unixccompiler import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.unixccompiler', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'distutils.unixccompiler' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.unixccompiler', import_51771)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from numpy.distutils.ccompiler import replace_method' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_51773 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.distutils.ccompiler')

if (type(import_51773) is not StypyTypeError):

    if (import_51773 != 'pyd_module'):
        __import__(import_51773)
        sys_modules_51774 = sys.modules[import_51773]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.distutils.ccompiler', sys_modules_51774.module_type_store, module_type_store, ['replace_method'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_51774, sys_modules_51774.module_type_store, module_type_store)
    else:
        from numpy.distutils.ccompiler import replace_method

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.distutils.ccompiler', None, module_type_store, ['replace_method'], [replace_method])

else:
    # Assigning a type to the variable 'numpy.distutils.ccompiler' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.distutils.ccompiler', import_51773)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from numpy.distutils.compat import get_exception' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_51775 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.distutils.compat')

if (type(import_51775) is not StypyTypeError):

    if (import_51775 != 'pyd_module'):
        __import__(import_51775)
        sys_modules_51776 = sys.modules[import_51775]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.distutils.compat', sys_modules_51776.module_type_store, module_type_store, ['get_exception'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_51776, sys_modules_51776.module_type_store, module_type_store)
    else:
        from numpy.distutils.compat import get_exception

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.distutils.compat', None, module_type_store, ['get_exception'], [get_exception])

else:
    # Assigning a type to the variable 'numpy.distutils.compat' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.distutils.compat', import_51775)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')




# Obtaining the type of the subscript
int_51777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 20), 'int')
# Getting the type of 'sys' (line 14)
sys_51778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 3), 'sys')
# Obtaining the member 'version_info' of a type (line 14)
version_info_51779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 3), sys_51778, 'version_info')
# Obtaining the member '__getitem__' of a type (line 14)
getitem___51780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 3), version_info_51779, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 14)
subscript_call_result_51781 = invoke(stypy.reporting.localization.Localization(__file__, 14, 3), getitem___51780, int_51777)

int_51782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 25), 'int')
# Applying the binary operator '<' (line 14)
result_lt_51783 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 3), '<', subscript_call_result_51781, int_51782)

# Testing the type of an if condition (line 14)
if_condition_51784 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 14, 0), result_lt_51783)
# Assigning a type to the variable 'if_condition_51784' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'if_condition_51784', if_condition_51784)
# SSA begins for if statement (line 14)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 4))

# 'from numpy.distutils import log' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_51785 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 4), 'numpy.distutils')

if (type(import_51785) is not StypyTypeError):

    if (import_51785 != 'pyd_module'):
        __import__(import_51785)
        sys_modules_51786 = sys.modules[import_51785]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 4), 'numpy.distutils', sys_modules_51786.module_type_store, module_type_store, ['log'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 4), __file__, sys_modules_51786, sys_modules_51786.module_type_store, module_type_store)
    else:
        from numpy.distutils import log

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 4), 'numpy.distutils', None, module_type_store, ['log'], [log])

else:
    # Assigning a type to the variable 'numpy.distutils' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'numpy.distutils', import_51785)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

# SSA branch for the else part of an if statement (line 14)
module_type_store.open_ssa_branch('else')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 4))

# 'from numpy.distutils import log' statement (line 17)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_51787 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 17, 4), 'numpy.distutils')

if (type(import_51787) is not StypyTypeError):

    if (import_51787 != 'pyd_module'):
        __import__(import_51787)
        sys_modules_51788 = sys.modules[import_51787]
        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 4), 'numpy.distutils', sys_modules_51788.module_type_store, module_type_store, ['log'])
        nest_module(stypy.reporting.localization.Localization(__file__, 17, 4), __file__, sys_modules_51788, sys_modules_51788.module_type_store, module_type_store)
    else:
        from numpy.distutils import log

        import_from_module(stypy.reporting.localization.Localization(__file__, 17, 4), 'numpy.distutils', None, module_type_store, ['log'], [log])

else:
    # Assigning a type to the variable 'numpy.distutils' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'numpy.distutils', import_51787)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

# SSA join for if statement (line 14)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def UnixCCompiler__compile(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'UnixCCompiler__compile'
    module_type_store = module_type_store.open_function_context('UnixCCompiler__compile', 20, 0, False)
    
    # Passed parameters checking function
    UnixCCompiler__compile.stypy_localization = localization
    UnixCCompiler__compile.stypy_type_of_self = None
    UnixCCompiler__compile.stypy_type_store = module_type_store
    UnixCCompiler__compile.stypy_function_name = 'UnixCCompiler__compile'
    UnixCCompiler__compile.stypy_param_names_list = ['self', 'obj', 'src', 'ext', 'cc_args', 'extra_postargs', 'pp_opts']
    UnixCCompiler__compile.stypy_varargs_param_name = None
    UnixCCompiler__compile.stypy_kwargs_param_name = None
    UnixCCompiler__compile.stypy_call_defaults = defaults
    UnixCCompiler__compile.stypy_call_varargs = varargs
    UnixCCompiler__compile.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'UnixCCompiler__compile', ['self', 'obj', 'src', 'ext', 'cc_args', 'extra_postargs', 'pp_opts'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'UnixCCompiler__compile', localization, ['self', 'obj', 'src', 'ext', 'cc_args', 'extra_postargs', 'pp_opts'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'UnixCCompiler__compile(...)' code ##################

    str_51789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 4), 'str', 'Compile a single source files with a Unix-style compiler.')
    
    # Assigning a Attribute to a Name (line 23):
    
    # Assigning a Attribute to a Name (line 23):
    # Getting the type of 'self' (line 23)
    self_51790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'self')
    # Obtaining the member 'compiler_so' of a type (line 23)
    compiler_so_51791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 12), self_51790, 'compiler_so')
    # Assigning a type to the variable 'ccomp' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'ccomp', compiler_so_51791)
    
    
    
    # Obtaining the type of the subscript
    int_51792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 13), 'int')
    # Getting the type of 'ccomp' (line 24)
    ccomp_51793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 7), 'ccomp')
    # Obtaining the member '__getitem__' of a type (line 24)
    getitem___51794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 7), ccomp_51793, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 24)
    subscript_call_result_51795 = invoke(stypy.reporting.localization.Localization(__file__, 24, 7), getitem___51794, int_51792)
    
    str_51796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 19), 'str', 'aCC')
    # Applying the binary operator '==' (line 24)
    result_eq_51797 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 7), '==', subscript_call_result_51795, str_51796)
    
    # Testing the type of an if condition (line 24)
    if_condition_51798 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 24, 4), result_eq_51797)
    # Assigning a type to the variable 'if_condition_51798' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'if_condition_51798', if_condition_51798)
    # SSA begins for if statement (line 24)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    str_51799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 11), 'str', '-Ae')
    # Getting the type of 'ccomp' (line 26)
    ccomp_51800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 20), 'ccomp')
    # Applying the binary operator 'in' (line 26)
    result_contains_51801 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 11), 'in', str_51799, ccomp_51800)
    
    # Testing the type of an if condition (line 26)
    if_condition_51802 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 26, 8), result_contains_51801)
    # Assigning a type to the variable 'if_condition_51802' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'if_condition_51802', if_condition_51802)
    # SSA begins for if statement (line 26)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to remove(...): (line 27)
    # Processing the call arguments (line 27)
    str_51805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 25), 'str', '-Ae')
    # Processing the call keyword arguments (line 27)
    kwargs_51806 = {}
    # Getting the type of 'ccomp' (line 27)
    ccomp_51803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'ccomp', False)
    # Obtaining the member 'remove' of a type (line 27)
    remove_51804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 12), ccomp_51803, 'remove')
    # Calling remove(args, kwargs) (line 27)
    remove_call_result_51807 = invoke(stypy.reporting.localization.Localization(__file__, 27, 12), remove_51804, *[str_51805], **kwargs_51806)
    
    # SSA join for if statement (line 26)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_51808 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 11), 'str', '-Aa')
    # Getting the type of 'ccomp' (line 28)
    ccomp_51809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 20), 'ccomp')
    # Applying the binary operator 'in' (line 28)
    result_contains_51810 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 11), 'in', str_51808, ccomp_51809)
    
    # Testing the type of an if condition (line 28)
    if_condition_51811 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 28, 8), result_contains_51810)
    # Assigning a type to the variable 'if_condition_51811' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'if_condition_51811', if_condition_51811)
    # SSA begins for if statement (line 28)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to remove(...): (line 29)
    # Processing the call arguments (line 29)
    str_51814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 25), 'str', '-Aa')
    # Processing the call keyword arguments (line 29)
    kwargs_51815 = {}
    # Getting the type of 'ccomp' (line 29)
    ccomp_51812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'ccomp', False)
    # Obtaining the member 'remove' of a type (line 29)
    remove_51813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 12), ccomp_51812, 'remove')
    # Calling remove(args, kwargs) (line 29)
    remove_call_result_51816 = invoke(stypy.reporting.localization.Localization(__file__, 29, 12), remove_51813, *[str_51814], **kwargs_51815)
    
    # SSA join for if statement (line 28)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'ccomp' (line 31)
    ccomp_51817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'ccomp')
    
    # Obtaining an instance of the builtin type 'list' (line 31)
    list_51818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 31)
    # Adding element type (line 31)
    str_51819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 18), 'str', '-AA')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 17), list_51818, str_51819)
    
    # Applying the binary operator '+=' (line 31)
    result_iadd_51820 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 8), '+=', ccomp_51817, list_51818)
    # Assigning a type to the variable 'ccomp' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'ccomp', result_iadd_51820)
    
    
    # Assigning a Name to a Attribute (line 32):
    
    # Assigning a Name to a Attribute (line 32):
    # Getting the type of 'ccomp' (line 32)
    ccomp_51821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 27), 'ccomp')
    # Getting the type of 'self' (line 32)
    self_51822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'self')
    # Setting the type of the member 'compiler_so' of a type (line 32)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 8), self_51822, 'compiler_so', ccomp_51821)
    # SSA join for if statement (line 24)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_51823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 7), 'str', 'OPT')
    # Getting the type of 'os' (line 34)
    os_51824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 16), 'os')
    # Obtaining the member 'environ' of a type (line 34)
    environ_51825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 16), os_51824, 'environ')
    # Applying the binary operator 'in' (line 34)
    result_contains_51826 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 7), 'in', str_51823, environ_51825)
    
    # Testing the type of an if condition (line 34)
    if_condition_51827 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 34, 4), result_contains_51826)
    # Assigning a type to the variable 'if_condition_51827' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'if_condition_51827', if_condition_51827)
    # SSA begins for if statement (line 34)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 35, 8))
    
    # 'from distutils.sysconfig import get_config_vars' statement (line 35)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
    import_51828 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 35, 8), 'distutils.sysconfig')

    if (type(import_51828) is not StypyTypeError):

        if (import_51828 != 'pyd_module'):
            __import__(import_51828)
            sys_modules_51829 = sys.modules[import_51828]
            import_from_module(stypy.reporting.localization.Localization(__file__, 35, 8), 'distutils.sysconfig', sys_modules_51829.module_type_store, module_type_store, ['get_config_vars'])
            nest_module(stypy.reporting.localization.Localization(__file__, 35, 8), __file__, sys_modules_51829, sys_modules_51829.module_type_store, module_type_store)
        else:
            from distutils.sysconfig import get_config_vars

            import_from_module(stypy.reporting.localization.Localization(__file__, 35, 8), 'distutils.sysconfig', None, module_type_store, ['get_config_vars'], [get_config_vars])

    else:
        # Assigning a type to the variable 'distutils.sysconfig' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'distutils.sysconfig', import_51828)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')
    
    
    # Assigning a Call to a Name (line 36):
    
    # Assigning a Call to a Name (line 36):
    
    # Call to join(...): (line 36)
    # Processing the call arguments (line 36)
    
    # Call to split(...): (line 36)
    # Processing the call keyword arguments (line 36)
    kwargs_51838 = {}
    
    # Obtaining the type of the subscript
    str_51832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 34), 'str', 'OPT')
    # Getting the type of 'os' (line 36)
    os_51833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 23), 'os', False)
    # Obtaining the member 'environ' of a type (line 36)
    environ_51834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 23), os_51833, 'environ')
    # Obtaining the member '__getitem__' of a type (line 36)
    getitem___51835 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 23), environ_51834, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 36)
    subscript_call_result_51836 = invoke(stypy.reporting.localization.Localization(__file__, 36, 23), getitem___51835, str_51832)
    
    # Obtaining the member 'split' of a type (line 36)
    split_51837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 23), subscript_call_result_51836, 'split')
    # Calling split(args, kwargs) (line 36)
    split_call_result_51839 = invoke(stypy.reporting.localization.Localization(__file__, 36, 23), split_51837, *[], **kwargs_51838)
    
    # Processing the call keyword arguments (line 36)
    kwargs_51840 = {}
    str_51830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 14), 'str', ' ')
    # Obtaining the member 'join' of a type (line 36)
    join_51831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 14), str_51830, 'join')
    # Calling join(args, kwargs) (line 36)
    join_call_result_51841 = invoke(stypy.reporting.localization.Localization(__file__, 36, 14), join_51831, *[split_call_result_51839], **kwargs_51840)
    
    # Assigning a type to the variable 'opt' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'opt', join_call_result_51841)
    
    # Assigning a Call to a Name (line 37):
    
    # Assigning a Call to a Name (line 37):
    
    # Call to join(...): (line 37)
    # Processing the call arguments (line 37)
    
    # Call to split(...): (line 37)
    # Processing the call keyword arguments (line 37)
    kwargs_51852 = {}
    
    # Obtaining the type of the subscript
    int_51844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 50), 'int')
    
    # Call to get_config_vars(...): (line 37)
    # Processing the call arguments (line 37)
    str_51846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 43), 'str', 'OPT')
    # Processing the call keyword arguments (line 37)
    kwargs_51847 = {}
    # Getting the type of 'get_config_vars' (line 37)
    get_config_vars_51845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 27), 'get_config_vars', False)
    # Calling get_config_vars(args, kwargs) (line 37)
    get_config_vars_call_result_51848 = invoke(stypy.reporting.localization.Localization(__file__, 37, 27), get_config_vars_51845, *[str_51846], **kwargs_51847)
    
    # Obtaining the member '__getitem__' of a type (line 37)
    getitem___51849 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 27), get_config_vars_call_result_51848, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 37)
    subscript_call_result_51850 = invoke(stypy.reporting.localization.Localization(__file__, 37, 27), getitem___51849, int_51844)
    
    # Obtaining the member 'split' of a type (line 37)
    split_51851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 27), subscript_call_result_51850, 'split')
    # Calling split(args, kwargs) (line 37)
    split_call_result_51853 = invoke(stypy.reporting.localization.Localization(__file__, 37, 27), split_51851, *[], **kwargs_51852)
    
    # Processing the call keyword arguments (line 37)
    kwargs_51854 = {}
    str_51842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 18), 'str', ' ')
    # Obtaining the member 'join' of a type (line 37)
    join_51843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 18), str_51842, 'join')
    # Calling join(args, kwargs) (line 37)
    join_call_result_51855 = invoke(stypy.reporting.localization.Localization(__file__, 37, 18), join_51843, *[split_call_result_51853], **kwargs_51854)
    
    # Assigning a type to the variable 'gcv_opt' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'gcv_opt', join_call_result_51855)
    
    # Assigning a Call to a Name (line 38):
    
    # Assigning a Call to a Name (line 38):
    
    # Call to join(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'self' (line 38)
    self_51858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 27), 'self', False)
    # Obtaining the member 'compiler_so' of a type (line 38)
    compiler_so_51859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 27), self_51858, 'compiler_so')
    # Processing the call keyword arguments (line 38)
    kwargs_51860 = {}
    str_51856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 18), 'str', ' ')
    # Obtaining the member 'join' of a type (line 38)
    join_51857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 18), str_51856, 'join')
    # Calling join(args, kwargs) (line 38)
    join_call_result_51861 = invoke(stypy.reporting.localization.Localization(__file__, 38, 18), join_51857, *[compiler_so_51859], **kwargs_51860)
    
    # Assigning a type to the variable 'ccomp_s' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'ccomp_s', join_call_result_51861)
    
    
    # Getting the type of 'opt' (line 39)
    opt_51862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 11), 'opt')
    # Getting the type of 'ccomp_s' (line 39)
    ccomp_s_51863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 22), 'ccomp_s')
    # Applying the binary operator 'notin' (line 39)
    result_contains_51864 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 11), 'notin', opt_51862, ccomp_s_51863)
    
    # Testing the type of an if condition (line 39)
    if_condition_51865 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 39, 8), result_contains_51864)
    # Assigning a type to the variable 'if_condition_51865' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'if_condition_51865', if_condition_51865)
    # SSA begins for if statement (line 39)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 40):
    
    # Assigning a Call to a Name (line 40):
    
    # Call to replace(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'gcv_opt' (line 40)
    gcv_opt_51868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 38), 'gcv_opt', False)
    # Getting the type of 'opt' (line 40)
    opt_51869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 47), 'opt', False)
    # Processing the call keyword arguments (line 40)
    kwargs_51870 = {}
    # Getting the type of 'ccomp_s' (line 40)
    ccomp_s_51866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 22), 'ccomp_s', False)
    # Obtaining the member 'replace' of a type (line 40)
    replace_51867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 22), ccomp_s_51866, 'replace')
    # Calling replace(args, kwargs) (line 40)
    replace_call_result_51871 = invoke(stypy.reporting.localization.Localization(__file__, 40, 22), replace_51867, *[gcv_opt_51868, opt_51869], **kwargs_51870)
    
    # Assigning a type to the variable 'ccomp_s' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'ccomp_s', replace_call_result_51871)
    
    # Assigning a Call to a Attribute (line 41):
    
    # Assigning a Call to a Attribute (line 41):
    
    # Call to split(...): (line 41)
    # Processing the call keyword arguments (line 41)
    kwargs_51874 = {}
    # Getting the type of 'ccomp_s' (line 41)
    ccomp_s_51872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 31), 'ccomp_s', False)
    # Obtaining the member 'split' of a type (line 41)
    split_51873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 31), ccomp_s_51872, 'split')
    # Calling split(args, kwargs) (line 41)
    split_call_result_51875 = invoke(stypy.reporting.localization.Localization(__file__, 41, 31), split_51873, *[], **kwargs_51874)
    
    # Getting the type of 'self' (line 41)
    self_51876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'self')
    # Setting the type of the member 'compiler_so' of a type (line 41)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 12), self_51876, 'compiler_so', split_call_result_51875)
    # SSA join for if statement (line 39)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 42):
    
    # Assigning a Call to a Name (line 42):
    
    # Call to join(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'self' (line 42)
    self_51879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 27), 'self', False)
    # Obtaining the member 'linker_so' of a type (line 42)
    linker_so_51880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 27), self_51879, 'linker_so')
    # Processing the call keyword arguments (line 42)
    kwargs_51881 = {}
    str_51877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 18), 'str', ' ')
    # Obtaining the member 'join' of a type (line 42)
    join_51878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 18), str_51877, 'join')
    # Calling join(args, kwargs) (line 42)
    join_call_result_51882 = invoke(stypy.reporting.localization.Localization(__file__, 42, 18), join_51878, *[linker_so_51880], **kwargs_51881)
    
    # Assigning a type to the variable 'llink_s' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'llink_s', join_call_result_51882)
    
    
    # Getting the type of 'opt' (line 43)
    opt_51883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 11), 'opt')
    # Getting the type of 'llink_s' (line 43)
    llink_s_51884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 22), 'llink_s')
    # Applying the binary operator 'notin' (line 43)
    result_contains_51885 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 11), 'notin', opt_51883, llink_s_51884)
    
    # Testing the type of an if condition (line 43)
    if_condition_51886 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 43, 8), result_contains_51885)
    # Assigning a type to the variable 'if_condition_51886' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'if_condition_51886', if_condition_51886)
    # SSA begins for if statement (line 43)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Attribute (line 44):
    
    # Assigning a BinOp to a Attribute (line 44):
    
    # Call to split(...): (line 44)
    # Processing the call keyword arguments (line 44)
    kwargs_51889 = {}
    # Getting the type of 'llink_s' (line 44)
    llink_s_51887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 29), 'llink_s', False)
    # Obtaining the member 'split' of a type (line 44)
    split_51888 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 29), llink_s_51887, 'split')
    # Calling split(args, kwargs) (line 44)
    split_call_result_51890 = invoke(stypy.reporting.localization.Localization(__file__, 44, 29), split_51888, *[], **kwargs_51889)
    
    
    # Call to split(...): (line 44)
    # Processing the call keyword arguments (line 44)
    kwargs_51893 = {}
    # Getting the type of 'opt' (line 44)
    opt_51891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 47), 'opt', False)
    # Obtaining the member 'split' of a type (line 44)
    split_51892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 47), opt_51891, 'split')
    # Calling split(args, kwargs) (line 44)
    split_call_result_51894 = invoke(stypy.reporting.localization.Localization(__file__, 44, 47), split_51892, *[], **kwargs_51893)
    
    # Applying the binary operator '+' (line 44)
    result_add_51895 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 29), '+', split_call_result_51890, split_call_result_51894)
    
    # Getting the type of 'self' (line 44)
    self_51896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'self')
    # Setting the type of the member 'linker_so' of a type (line 44)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 12), self_51896, 'linker_so', result_add_51895)
    # SSA join for if statement (line 43)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 34)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 46):
    
    # Assigning a BinOp to a Name (line 46):
    str_51897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 14), 'str', '%s: %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 46)
    tuple_51898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 46)
    # Adding element type (line 46)
    
    # Call to basename(...): (line 46)
    # Processing the call arguments (line 46)
    
    # Obtaining the type of the subscript
    int_51902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 60), 'int')
    # Getting the type of 'self' (line 46)
    self_51903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 43), 'self', False)
    # Obtaining the member 'compiler_so' of a type (line 46)
    compiler_so_51904 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 43), self_51903, 'compiler_so')
    # Obtaining the member '__getitem__' of a type (line 46)
    getitem___51905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 43), compiler_so_51904, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 46)
    subscript_call_result_51906 = invoke(stypy.reporting.localization.Localization(__file__, 46, 43), getitem___51905, int_51902)
    
    # Processing the call keyword arguments (line 46)
    kwargs_51907 = {}
    # Getting the type of 'os' (line 46)
    os_51899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 26), 'os', False)
    # Obtaining the member 'path' of a type (line 46)
    path_51900 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 26), os_51899, 'path')
    # Obtaining the member 'basename' of a type (line 46)
    basename_51901 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 26), path_51900, 'basename')
    # Calling basename(args, kwargs) (line 46)
    basename_call_result_51908 = invoke(stypy.reporting.localization.Localization(__file__, 46, 26), basename_51901, *[subscript_call_result_51906], **kwargs_51907)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 26), tuple_51898, basename_call_result_51908)
    # Adding element type (line 46)
    # Getting the type of 'src' (line 46)
    src_51909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 65), 'src')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 26), tuple_51898, src_51909)
    
    # Applying the binary operator '%' (line 46)
    result_mod_51910 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 14), '%', str_51897, tuple_51898)
    
    # Assigning a type to the variable 'display' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'display', result_mod_51910)
    
    
    # SSA begins for try-except statement (line 47)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to spawn(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'self' (line 48)
    self_51913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 19), 'self', False)
    # Obtaining the member 'compiler_so' of a type (line 48)
    compiler_so_51914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 19), self_51913, 'compiler_so')
    # Getting the type of 'cc_args' (line 48)
    cc_args_51915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 38), 'cc_args', False)
    # Applying the binary operator '+' (line 48)
    result_add_51916 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 19), '+', compiler_so_51914, cc_args_51915)
    
    
    # Obtaining an instance of the builtin type 'list' (line 48)
    list_51917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 48), 'list')
    # Adding type elements to the builtin type 'list' instance (line 48)
    # Adding element type (line 48)
    # Getting the type of 'src' (line 48)
    src_51918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 49), 'src', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 48), list_51917, src_51918)
    # Adding element type (line 48)
    str_51919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 54), 'str', '-o')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 48), list_51917, str_51919)
    # Adding element type (line 48)
    # Getting the type of 'obj' (line 48)
    obj_51920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 60), 'obj', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 48, 48), list_51917, obj_51920)
    
    # Applying the binary operator '+' (line 48)
    result_add_51921 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 46), '+', result_add_51916, list_51917)
    
    # Getting the type of 'extra_postargs' (line 49)
    extra_postargs_51922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 19), 'extra_postargs', False)
    # Applying the binary operator '+' (line 48)
    result_add_51923 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 65), '+', result_add_51921, extra_postargs_51922)
    
    # Processing the call keyword arguments (line 48)
    # Getting the type of 'display' (line 49)
    display_51924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 45), 'display', False)
    keyword_51925 = display_51924
    kwargs_51926 = {'display': keyword_51925}
    # Getting the type of 'self' (line 48)
    self_51911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'self', False)
    # Obtaining the member 'spawn' of a type (line 48)
    spawn_51912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), self_51911, 'spawn')
    # Calling spawn(args, kwargs) (line 48)
    spawn_call_result_51927 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), spawn_51912, *[result_add_51923], **kwargs_51926)
    
    # SSA branch for the except part of a try statement (line 47)
    # SSA branch for the except 'DistutilsExecError' branch of a try statement (line 47)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Call to a Name (line 51):
    
    # Assigning a Call to a Name (line 51):
    
    # Call to str(...): (line 51)
    # Processing the call arguments (line 51)
    
    # Call to get_exception(...): (line 51)
    # Processing the call keyword arguments (line 51)
    kwargs_51930 = {}
    # Getting the type of 'get_exception' (line 51)
    get_exception_51929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 18), 'get_exception', False)
    # Calling get_exception(args, kwargs) (line 51)
    get_exception_call_result_51931 = invoke(stypy.reporting.localization.Localization(__file__, 51, 18), get_exception_51929, *[], **kwargs_51930)
    
    # Processing the call keyword arguments (line 51)
    kwargs_51932 = {}
    # Getting the type of 'str' (line 51)
    str_51928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 14), 'str', False)
    # Calling str(args, kwargs) (line 51)
    str_call_result_51933 = invoke(stypy.reporting.localization.Localization(__file__, 51, 14), str_51928, *[get_exception_call_result_51931], **kwargs_51932)
    
    # Assigning a type to the variable 'msg' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'msg', str_call_result_51933)
    
    # Call to CompileError(...): (line 52)
    # Processing the call arguments (line 52)
    # Getting the type of 'msg' (line 52)
    msg_51935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 27), 'msg', False)
    # Processing the call keyword arguments (line 52)
    kwargs_51936 = {}
    # Getting the type of 'CompileError' (line 52)
    CompileError_51934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 14), 'CompileError', False)
    # Calling CompileError(args, kwargs) (line 52)
    CompileError_call_result_51937 = invoke(stypy.reporting.localization.Localization(__file__, 52, 14), CompileError_51934, *[msg_51935], **kwargs_51936)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 52, 8), CompileError_call_result_51937, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 47)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'UnixCCompiler__compile(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'UnixCCompiler__compile' in the type store
    # Getting the type of 'stypy_return_type' (line 20)
    stypy_return_type_51938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_51938)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'UnixCCompiler__compile'
    return stypy_return_type_51938

# Assigning a type to the variable 'UnixCCompiler__compile' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'UnixCCompiler__compile', UnixCCompiler__compile)

# Call to replace_method(...): (line 54)
# Processing the call arguments (line 54)
# Getting the type of 'UnixCCompiler' (line 54)
UnixCCompiler_51940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 15), 'UnixCCompiler', False)
str_51941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 30), 'str', '_compile')
# Getting the type of 'UnixCCompiler__compile' (line 54)
UnixCCompiler__compile_51942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 42), 'UnixCCompiler__compile', False)
# Processing the call keyword arguments (line 54)
kwargs_51943 = {}
# Getting the type of 'replace_method' (line 54)
replace_method_51939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'replace_method', False)
# Calling replace_method(args, kwargs) (line 54)
replace_method_call_result_51944 = invoke(stypy.reporting.localization.Localization(__file__, 54, 0), replace_method_51939, *[UnixCCompiler_51940, str_51941, UnixCCompiler__compile_51942], **kwargs_51943)


@norecursion
def UnixCCompiler_create_static_lib(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 58)
    None_51945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 47), 'None')
    int_51946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 59), 'int')
    # Getting the type of 'None' (line 58)
    None_51947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 74), 'None')
    defaults = [None_51945, int_51946, None_51947]
    # Create a new context for function 'UnixCCompiler_create_static_lib'
    module_type_store = module_type_store.open_function_context('UnixCCompiler_create_static_lib', 57, 0, False)
    
    # Passed parameters checking function
    UnixCCompiler_create_static_lib.stypy_localization = localization
    UnixCCompiler_create_static_lib.stypy_type_of_self = None
    UnixCCompiler_create_static_lib.stypy_type_store = module_type_store
    UnixCCompiler_create_static_lib.stypy_function_name = 'UnixCCompiler_create_static_lib'
    UnixCCompiler_create_static_lib.stypy_param_names_list = ['self', 'objects', 'output_libname', 'output_dir', 'debug', 'target_lang']
    UnixCCompiler_create_static_lib.stypy_varargs_param_name = None
    UnixCCompiler_create_static_lib.stypy_kwargs_param_name = None
    UnixCCompiler_create_static_lib.stypy_call_defaults = defaults
    UnixCCompiler_create_static_lib.stypy_call_varargs = varargs
    UnixCCompiler_create_static_lib.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'UnixCCompiler_create_static_lib', ['self', 'objects', 'output_libname', 'output_dir', 'debug', 'target_lang'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'UnixCCompiler_create_static_lib', localization, ['self', 'objects', 'output_libname', 'output_dir', 'debug', 'target_lang'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'UnixCCompiler_create_static_lib(...)' code ##################

    str_51948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, (-1)), 'str', '\n    Build a static library in a separate sub-process.\n\n    Parameters\n    ----------\n    objects : list or tuple of str\n        List of paths to object files used to build the static library.\n    output_libname : str\n        The library name as an absolute or relative (if `output_dir` is used)\n        path.\n    output_dir : str, optional\n        The path to the output directory. Default is None, in which case\n        the ``output_dir`` attribute of the UnixCCompiler instance.\n    debug : bool, optional\n        This parameter is not used.\n    target_lang : str, optional\n        This parameter is not used.\n\n    Returns\n    -------\n    None\n\n    ')
    
    # Assigning a Call to a Tuple (line 82):
    
    # Assigning a Call to a Name:
    
    # Call to _fix_object_args(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 'objects' (line 82)
    objects_51951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 48), 'objects', False)
    # Getting the type of 'output_dir' (line 82)
    output_dir_51952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 57), 'output_dir', False)
    # Processing the call keyword arguments (line 82)
    kwargs_51953 = {}
    # Getting the type of 'self' (line 82)
    self_51949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 26), 'self', False)
    # Obtaining the member '_fix_object_args' of a type (line 82)
    _fix_object_args_51950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 26), self_51949, '_fix_object_args')
    # Calling _fix_object_args(args, kwargs) (line 82)
    _fix_object_args_call_result_51954 = invoke(stypy.reporting.localization.Localization(__file__, 82, 26), _fix_object_args_51950, *[objects_51951, output_dir_51952], **kwargs_51953)
    
    # Assigning a type to the variable 'call_assignment_51765' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'call_assignment_51765', _fix_object_args_call_result_51954)
    
    # Assigning a Call to a Name (line 82):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_51957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 4), 'int')
    # Processing the call keyword arguments
    kwargs_51958 = {}
    # Getting the type of 'call_assignment_51765' (line 82)
    call_assignment_51765_51955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'call_assignment_51765', False)
    # Obtaining the member '__getitem__' of a type (line 82)
    getitem___51956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 4), call_assignment_51765_51955, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_51959 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___51956, *[int_51957], **kwargs_51958)
    
    # Assigning a type to the variable 'call_assignment_51766' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'call_assignment_51766', getitem___call_result_51959)
    
    # Assigning a Name to a Name (line 82):
    # Getting the type of 'call_assignment_51766' (line 82)
    call_assignment_51766_51960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'call_assignment_51766')
    # Assigning a type to the variable 'objects' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'objects', call_assignment_51766_51960)
    
    # Assigning a Call to a Name (line 82):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_51963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 4), 'int')
    # Processing the call keyword arguments
    kwargs_51964 = {}
    # Getting the type of 'call_assignment_51765' (line 82)
    call_assignment_51765_51961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'call_assignment_51765', False)
    # Obtaining the member '__getitem__' of a type (line 82)
    getitem___51962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 4), call_assignment_51765_51961, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_51965 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___51962, *[int_51963], **kwargs_51964)
    
    # Assigning a type to the variable 'call_assignment_51767' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'call_assignment_51767', getitem___call_result_51965)
    
    # Assigning a Name to a Name (line 82):
    # Getting the type of 'call_assignment_51767' (line 82)
    call_assignment_51767_51966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'call_assignment_51767')
    # Assigning a type to the variable 'output_dir' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 13), 'output_dir', call_assignment_51767_51966)
    
    # Assigning a Call to a Name (line 84):
    
    # Assigning a Call to a Name (line 84):
    
    # Call to library_filename(...): (line 85)
    # Processing the call arguments (line 85)
    # Getting the type of 'output_libname' (line 85)
    output_libname_51969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 42), 'output_libname', False)
    # Processing the call keyword arguments (line 85)
    # Getting the type of 'output_dir' (line 85)
    output_dir_51970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 69), 'output_dir', False)
    keyword_51971 = output_dir_51970
    kwargs_51972 = {'output_dir': keyword_51971}
    # Getting the type of 'self' (line 85)
    self_51967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 20), 'self', False)
    # Obtaining the member 'library_filename' of a type (line 85)
    library_filename_51968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 20), self_51967, 'library_filename')
    # Calling library_filename(args, kwargs) (line 85)
    library_filename_call_result_51973 = invoke(stypy.reporting.localization.Localization(__file__, 85, 20), library_filename_51968, *[output_libname_51969], **kwargs_51972)
    
    # Assigning a type to the variable 'output_filename' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'output_filename', library_filename_call_result_51973)
    
    
    # Call to _need_link(...): (line 87)
    # Processing the call arguments (line 87)
    # Getting the type of 'objects' (line 87)
    objects_51976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 23), 'objects', False)
    # Getting the type of 'output_filename' (line 87)
    output_filename_51977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 32), 'output_filename', False)
    # Processing the call keyword arguments (line 87)
    kwargs_51978 = {}
    # Getting the type of 'self' (line 87)
    self_51974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 7), 'self', False)
    # Obtaining the member '_need_link' of a type (line 87)
    _need_link_51975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 7), self_51974, '_need_link')
    # Calling _need_link(args, kwargs) (line 87)
    _need_link_call_result_51979 = invoke(stypy.reporting.localization.Localization(__file__, 87, 7), _need_link_51975, *[objects_51976, output_filename_51977], **kwargs_51978)
    
    # Testing the type of an if condition (line 87)
    if_condition_51980 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 4), _need_link_call_result_51979)
    # Assigning a type to the variable 'if_condition_51980' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'if_condition_51980', if_condition_51980)
    # SSA begins for if statement (line 87)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # SSA begins for try-except statement (line 88)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to unlink(...): (line 92)
    # Processing the call arguments (line 92)
    # Getting the type of 'output_filename' (line 92)
    output_filename_51983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 22), 'output_filename', False)
    # Processing the call keyword arguments (line 92)
    kwargs_51984 = {}
    # Getting the type of 'os' (line 92)
    os_51981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'os', False)
    # Obtaining the member 'unlink' of a type (line 92)
    unlink_51982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 12), os_51981, 'unlink')
    # Calling unlink(args, kwargs) (line 92)
    unlink_call_result_51985 = invoke(stypy.reporting.localization.Localization(__file__, 92, 12), unlink_51982, *[output_filename_51983], **kwargs_51984)
    
    # SSA branch for the except part of a try statement (line 88)
    # SSA branch for the except 'Tuple' branch of a try statement (line 88)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 88)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to mkpath(...): (line 95)
    # Processing the call arguments (line 95)
    
    # Call to dirname(...): (line 95)
    # Processing the call arguments (line 95)
    # Getting the type of 'output_filename' (line 95)
    output_filename_51991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 36), 'output_filename', False)
    # Processing the call keyword arguments (line 95)
    kwargs_51992 = {}
    # Getting the type of 'os' (line 95)
    os_51988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 20), 'os', False)
    # Obtaining the member 'path' of a type (line 95)
    path_51989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 20), os_51988, 'path')
    # Obtaining the member 'dirname' of a type (line 95)
    dirname_51990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 20), path_51989, 'dirname')
    # Calling dirname(args, kwargs) (line 95)
    dirname_call_result_51993 = invoke(stypy.reporting.localization.Localization(__file__, 95, 20), dirname_51990, *[output_filename_51991], **kwargs_51992)
    
    # Processing the call keyword arguments (line 95)
    kwargs_51994 = {}
    # Getting the type of 'self' (line 95)
    self_51986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'self', False)
    # Obtaining the member 'mkpath' of a type (line 95)
    mkpath_51987 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), self_51986, 'mkpath')
    # Calling mkpath(args, kwargs) (line 95)
    mkpath_call_result_51995 = invoke(stypy.reporting.localization.Localization(__file__, 95, 8), mkpath_51987, *[dirname_call_result_51993], **kwargs_51994)
    
    
    # Assigning a BinOp to a Name (line 96):
    
    # Assigning a BinOp to a Name (line 96):
    # Getting the type of 'objects' (line 96)
    objects_51996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 22), 'objects')
    # Getting the type of 'self' (line 96)
    self_51997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 32), 'self')
    # Obtaining the member 'objects' of a type (line 96)
    objects_51998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 32), self_51997, 'objects')
    # Applying the binary operator '+' (line 96)
    result_add_51999 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 22), '+', objects_51996, objects_51998)
    
    # Assigning a type to the variable 'tmp_objects' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'tmp_objects', result_add_51999)
    
    # Getting the type of 'tmp_objects' (line 97)
    tmp_objects_52000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 14), 'tmp_objects')
    # Testing the type of an if condition (line 97)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 8), tmp_objects_52000)
    # SSA begins for while statement (line 97)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Subscript to a Name (line 98):
    
    # Assigning a Subscript to a Name (line 98):
    
    # Obtaining the type of the subscript
    int_52001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 35), 'int')
    slice_52002 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 98, 22), None, int_52001, None)
    # Getting the type of 'tmp_objects' (line 98)
    tmp_objects_52003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 22), 'tmp_objects')
    # Obtaining the member '__getitem__' of a type (line 98)
    getitem___52004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 22), tmp_objects_52003, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 98)
    subscript_call_result_52005 = invoke(stypy.reporting.localization.Localization(__file__, 98, 22), getitem___52004, slice_52002)
    
    # Assigning a type to the variable 'objects' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'objects', subscript_call_result_52005)
    
    # Assigning a Subscript to a Name (line 99):
    
    # Assigning a Subscript to a Name (line 99):
    
    # Obtaining the type of the subscript
    int_52006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 38), 'int')
    slice_52007 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 99, 26), int_52006, None, None)
    # Getting the type of 'tmp_objects' (line 99)
    tmp_objects_52008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 26), 'tmp_objects')
    # Obtaining the member '__getitem__' of a type (line 99)
    getitem___52009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 26), tmp_objects_52008, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 99)
    subscript_call_result_52010 = invoke(stypy.reporting.localization.Localization(__file__, 99, 26), getitem___52009, slice_52007)
    
    # Assigning a type to the variable 'tmp_objects' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'tmp_objects', subscript_call_result_52010)
    
    # Assigning a BinOp to a Name (line 100):
    
    # Assigning a BinOp to a Name (line 100):
    str_52011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 22), 'str', '%s: adding %d object files to %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 101)
    tuple_52012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 101)
    # Adding element type (line 101)
    
    # Call to basename(...): (line 101)
    # Processing the call arguments (line 101)
    
    # Obtaining the type of the subscript
    int_52016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 58), 'int')
    # Getting the type of 'self' (line 101)
    self_52017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 44), 'self', False)
    # Obtaining the member 'archiver' of a type (line 101)
    archiver_52018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 44), self_52017, 'archiver')
    # Obtaining the member '__getitem__' of a type (line 101)
    getitem___52019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 44), archiver_52018, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 101)
    subscript_call_result_52020 = invoke(stypy.reporting.localization.Localization(__file__, 101, 44), getitem___52019, int_52016)
    
    # Processing the call keyword arguments (line 101)
    kwargs_52021 = {}
    # Getting the type of 'os' (line 101)
    os_52013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 27), 'os', False)
    # Obtaining the member 'path' of a type (line 101)
    path_52014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 27), os_52013, 'path')
    # Obtaining the member 'basename' of a type (line 101)
    basename_52015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 27), path_52014, 'basename')
    # Calling basename(args, kwargs) (line 101)
    basename_call_result_52022 = invoke(stypy.reporting.localization.Localization(__file__, 101, 27), basename_52015, *[subscript_call_result_52020], **kwargs_52021)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 27), tuple_52012, basename_call_result_52022)
    # Adding element type (line 101)
    
    # Call to len(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 'objects' (line 102)
    objects_52024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 31), 'objects', False)
    # Processing the call keyword arguments (line 102)
    kwargs_52025 = {}
    # Getting the type of 'len' (line 102)
    len_52023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 27), 'len', False)
    # Calling len(args, kwargs) (line 102)
    len_call_result_52026 = invoke(stypy.reporting.localization.Localization(__file__, 102, 27), len_52023, *[objects_52024], **kwargs_52025)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 27), tuple_52012, len_call_result_52026)
    # Adding element type (line 101)
    # Getting the type of 'output_filename' (line 102)
    output_filename_52027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 41), 'output_filename')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 27), tuple_52012, output_filename_52027)
    
    # Applying the binary operator '%' (line 100)
    result_mod_52028 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 22), '%', str_52011, tuple_52012)
    
    # Assigning a type to the variable 'display' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'display', result_mod_52028)
    
    # Call to spawn(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'self' (line 103)
    self_52031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 23), 'self', False)
    # Obtaining the member 'archiver' of a type (line 103)
    archiver_52032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 23), self_52031, 'archiver')
    
    # Obtaining an instance of the builtin type 'list' (line 103)
    list_52033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 39), 'list')
    # Adding type elements to the builtin type 'list' instance (line 103)
    # Adding element type (line 103)
    # Getting the type of 'output_filename' (line 103)
    output_filename_52034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 40), 'output_filename', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 39), list_52033, output_filename_52034)
    
    # Applying the binary operator '+' (line 103)
    result_add_52035 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 23), '+', archiver_52032, list_52033)
    
    # Getting the type of 'objects' (line 103)
    objects_52036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 59), 'objects', False)
    # Applying the binary operator '+' (line 103)
    result_add_52037 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 57), '+', result_add_52035, objects_52036)
    
    # Processing the call keyword arguments (line 103)
    # Getting the type of 'display' (line 104)
    display_52038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 33), 'display', False)
    keyword_52039 = display_52038
    kwargs_52040 = {'display': keyword_52039}
    # Getting the type of 'self' (line 103)
    self_52029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'self', False)
    # Obtaining the member 'spawn' of a type (line 103)
    spawn_52030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), self_52029, 'spawn')
    # Calling spawn(args, kwargs) (line 103)
    spawn_call_result_52041 = invoke(stypy.reporting.localization.Localization(__file__, 103, 12), spawn_52030, *[result_add_52037], **kwargs_52040)
    
    # SSA join for while statement (line 97)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'self' (line 111)
    self_52042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 11), 'self')
    # Obtaining the member 'ranlib' of a type (line 111)
    ranlib_52043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 11), self_52042, 'ranlib')
    # Testing the type of an if condition (line 111)
    if_condition_52044 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 111, 8), ranlib_52043)
    # Assigning a type to the variable 'if_condition_52044' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'if_condition_52044', if_condition_52044)
    # SSA begins for if statement (line 111)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 112):
    
    # Assigning a BinOp to a Name (line 112):
    str_52045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 22), 'str', '%s:@ %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 112)
    tuple_52046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 112)
    # Adding element type (line 112)
    
    # Call to basename(...): (line 112)
    # Processing the call arguments (line 112)
    
    # Obtaining the type of the subscript
    int_52050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 64), 'int')
    # Getting the type of 'self' (line 112)
    self_52051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 52), 'self', False)
    # Obtaining the member 'ranlib' of a type (line 112)
    ranlib_52052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 52), self_52051, 'ranlib')
    # Obtaining the member '__getitem__' of a type (line 112)
    getitem___52053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 52), ranlib_52052, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 112)
    subscript_call_result_52054 = invoke(stypy.reporting.localization.Localization(__file__, 112, 52), getitem___52053, int_52050)
    
    # Processing the call keyword arguments (line 112)
    kwargs_52055 = {}
    # Getting the type of 'os' (line 112)
    os_52047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 35), 'os', False)
    # Obtaining the member 'path' of a type (line 112)
    path_52048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 35), os_52047, 'path')
    # Obtaining the member 'basename' of a type (line 112)
    basename_52049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 35), path_52048, 'basename')
    # Calling basename(args, kwargs) (line 112)
    basename_call_result_52056 = invoke(stypy.reporting.localization.Localization(__file__, 112, 35), basename_52049, *[subscript_call_result_52054], **kwargs_52055)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 35), tuple_52046, basename_call_result_52056)
    # Adding element type (line 112)
    # Getting the type of 'output_filename' (line 113)
    output_filename_52057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 35), 'output_filename')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 35), tuple_52046, output_filename_52057)
    
    # Applying the binary operator '%' (line 112)
    result_mod_52058 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 22), '%', str_52045, tuple_52046)
    
    # Assigning a type to the variable 'display' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'display', result_mod_52058)
    
    
    # SSA begins for try-except statement (line 114)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to spawn(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'self' (line 115)
    self_52061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 27), 'self', False)
    # Obtaining the member 'ranlib' of a type (line 115)
    ranlib_52062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 27), self_52061, 'ranlib')
    
    # Obtaining an instance of the builtin type 'list' (line 115)
    list_52063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 41), 'list')
    # Adding type elements to the builtin type 'list' instance (line 115)
    # Adding element type (line 115)
    # Getting the type of 'output_filename' (line 115)
    output_filename_52064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 42), 'output_filename', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 41), list_52063, output_filename_52064)
    
    # Applying the binary operator '+' (line 115)
    result_add_52065 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 27), '+', ranlib_52062, list_52063)
    
    # Processing the call keyword arguments (line 115)
    # Getting the type of 'display' (line 116)
    display_52066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 37), 'display', False)
    keyword_52067 = display_52066
    kwargs_52068 = {'display': keyword_52067}
    # Getting the type of 'self' (line 115)
    self_52059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 16), 'self', False)
    # Obtaining the member 'spawn' of a type (line 115)
    spawn_52060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 16), self_52059, 'spawn')
    # Calling spawn(args, kwargs) (line 115)
    spawn_call_result_52069 = invoke(stypy.reporting.localization.Localization(__file__, 115, 16), spawn_52060, *[result_add_52065], **kwargs_52068)
    
    # SSA branch for the except part of a try statement (line 114)
    # SSA branch for the except 'DistutilsExecError' branch of a try statement (line 114)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Call to a Name (line 118):
    
    # Assigning a Call to a Name (line 118):
    
    # Call to str(...): (line 118)
    # Processing the call arguments (line 118)
    
    # Call to get_exception(...): (line 118)
    # Processing the call keyword arguments (line 118)
    kwargs_52072 = {}
    # Getting the type of 'get_exception' (line 118)
    get_exception_52071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 26), 'get_exception', False)
    # Calling get_exception(args, kwargs) (line 118)
    get_exception_call_result_52073 = invoke(stypy.reporting.localization.Localization(__file__, 118, 26), get_exception_52071, *[], **kwargs_52072)
    
    # Processing the call keyword arguments (line 118)
    kwargs_52074 = {}
    # Getting the type of 'str' (line 118)
    str_52070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 22), 'str', False)
    # Calling str(args, kwargs) (line 118)
    str_call_result_52075 = invoke(stypy.reporting.localization.Localization(__file__, 118, 22), str_52070, *[get_exception_call_result_52073], **kwargs_52074)
    
    # Assigning a type to the variable 'msg' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 16), 'msg', str_call_result_52075)
    
    # Call to LibError(...): (line 119)
    # Processing the call arguments (line 119)
    # Getting the type of 'msg' (line 119)
    msg_52077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 31), 'msg', False)
    # Processing the call keyword arguments (line 119)
    kwargs_52078 = {}
    # Getting the type of 'LibError' (line 119)
    LibError_52076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 22), 'LibError', False)
    # Calling LibError(args, kwargs) (line 119)
    LibError_call_result_52079 = invoke(stypy.reporting.localization.Localization(__file__, 119, 22), LibError_52076, *[msg_52077], **kwargs_52078)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 119, 16), LibError_call_result_52079, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 114)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 111)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 87)
    module_type_store.open_ssa_branch('else')
    
    # Call to debug(...): (line 121)
    # Processing the call arguments (line 121)
    str_52082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 18), 'str', 'skipping %s (up-to-date)')
    # Getting the type of 'output_filename' (line 121)
    output_filename_52083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 46), 'output_filename', False)
    # Processing the call keyword arguments (line 121)
    kwargs_52084 = {}
    # Getting the type of 'log' (line 121)
    log_52080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'log', False)
    # Obtaining the member 'debug' of a type (line 121)
    debug_52081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 8), log_52080, 'debug')
    # Calling debug(args, kwargs) (line 121)
    debug_call_result_52085 = invoke(stypy.reporting.localization.Localization(__file__, 121, 8), debug_52081, *[str_52082, output_filename_52083], **kwargs_52084)
    
    # SSA join for if statement (line 87)
    module_type_store = module_type_store.join_ssa_context()
    
    # Assigning a type to the variable 'stypy_return_type' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'stypy_return_type', types.NoneType)
    
    # ################# End of 'UnixCCompiler_create_static_lib(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'UnixCCompiler_create_static_lib' in the type store
    # Getting the type of 'stypy_return_type' (line 57)
    stypy_return_type_52086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_52086)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'UnixCCompiler_create_static_lib'
    return stypy_return_type_52086

# Assigning a type to the variable 'UnixCCompiler_create_static_lib' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'UnixCCompiler_create_static_lib', UnixCCompiler_create_static_lib)

# Call to replace_method(...): (line 124)
# Processing the call arguments (line 124)
# Getting the type of 'UnixCCompiler' (line 124)
UnixCCompiler_52088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 15), 'UnixCCompiler', False)
str_52089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 30), 'str', 'create_static_lib')
# Getting the type of 'UnixCCompiler_create_static_lib' (line 125)
UnixCCompiler_create_static_lib_52090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 15), 'UnixCCompiler_create_static_lib', False)
# Processing the call keyword arguments (line 124)
kwargs_52091 = {}
# Getting the type of 'replace_method' (line 124)
replace_method_52087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 0), 'replace_method', False)
# Calling replace_method(args, kwargs) (line 124)
replace_method_call_result_52092 = invoke(stypy.reporting.localization.Localization(__file__, 124, 0), replace_method_52087, *[UnixCCompiler_52088, str_52089, UnixCCompiler_create_static_lib_52090], **kwargs_52091)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
