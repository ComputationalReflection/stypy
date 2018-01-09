
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #!/usr/bin/env python
2: '''
3: setup.py for installing F2PY
4: 
5: Usage:
6:    python setup.py install
7: 
8: Copyright 2001-2005 Pearu Peterson all rights reserved,
9: Pearu Peterson <pearu@cens.ioc.ee>
10: Permission to use, modify, and distribute this software is given under the
11: terms of the NumPy License.
12: 
13: NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.
14: $Revision: 1.32 $
15: $Date: 2005/01/30 17:22:14 $
16: Pearu Peterson
17: 
18: '''
19: from __future__ import division, print_function
20: 
21: __version__ = "$Id: setup.py,v 1.32 2005/01/30 17:22:14 pearu Exp $"
22: 
23: import os
24: import sys
25: from distutils.dep_util import newer
26: from numpy.distutils import log
27: from numpy.distutils.core import setup
28: from numpy.distutils.misc_util import Configuration
29: 
30: from __version__ import version
31: 
32: 
33: def _get_f2py_shebang():
34:     ''' Return shebang line for f2py script
35: 
36:     If we are building a binary distribution format, then the shebang line
37:     should be ``#!python`` rather than ``#!`` followed by the contents of
38:     ``sys.executable``.
39:     '''
40:     if set(('bdist_wheel', 'bdist_egg', 'bdist_wininst',
41:             'bdist_rpm')).intersection(sys.argv):
42:         return '#!python'
43:     return '#!' + sys.executable
44: 
45: 
46: def configuration(parent_package='', top_path=None):
47:     config = Configuration('f2py', parent_package, top_path)
48: 
49:     config.add_data_dir('tests')
50: 
51:     config.add_data_files('src/fortranobject.c',
52:                           'src/fortranobject.h',
53:                           )
54: 
55:     config.make_svn_version_py()
56: 
57:     def generate_f2py_py(build_dir):
58:         f2py_exe = 'f2py' + os.path.basename(sys.executable)[6:]
59:         if f2py_exe[-4:] == '.exe':
60:             f2py_exe = f2py_exe[:-4] + '.py'
61:         if 'bdist_wininst' in sys.argv and f2py_exe[-3:] != '.py':
62:             f2py_exe = f2py_exe + '.py'
63:         target = os.path.join(build_dir, f2py_exe)
64:         if newer(__file__, target):
65:             log.info('Creating %s', target)
66:             f = open(target, 'w')
67:             f.write(_get_f2py_shebang() + '\n')
68:             mainloc = os.path.join(os.path.dirname(__file__), "__main__.py")
69:             with open(mainloc) as mf:
70:                 f.write(mf.read())
71:             f.close()
72:         return target
73: 
74:     config.add_scripts(generate_f2py_py)
75: 
76:     log.info('F2PY Version %s', config.get_version())
77: 
78:     return config
79: 
80: if __name__ == "__main__":
81: 
82:     config = configuration(top_path='')
83:     print('F2PY Version', version)
84:     config = config.todict()
85: 
86:     config['download_url'] = "http://cens.ioc.ee/projects/f2py2e/2.x"\
87:                              "/F2PY-2-latest.tar.gz"
88:     config['classifiers'] = [
89:         'Development Status :: 5 - Production/Stable',
90:         'Intended Audience :: Developers',
91:         'Intended Audience :: Science/Research',
92:         'License :: OSI Approved :: NumPy License',
93:         'Natural Language :: English',
94:         'Operating System :: OS Independent',
95:         'Programming Language :: C',
96:         'Programming Language :: Fortran',
97:         'Programming Language :: Python',
98:         'Topic :: Scientific/Engineering',
99:         'Topic :: Software Development :: Code Generators',
100:     ]
101:     setup(version=version,
102:           description="F2PY - Fortran to Python Interface Generaton",
103:           author="Pearu Peterson",
104:           author_email="pearu@cens.ioc.ee",
105:           maintainer="Pearu Peterson",
106:           maintainer_email="pearu@cens.ioc.ee",
107:           license="BSD",
108:           platforms="Unix, Windows (mingw|cygwin), Mac OSX",
109:           long_description='''\
110: The Fortran to Python Interface Generator, or F2PY for short, is a
111: command line tool (f2py) for generating Python C/API modules for
112: wrapping Fortran 77/90/95 subroutines, accessing common blocks from
113: Python, and calling Python functions from Fortran (call-backs).
114: Interfacing subroutines/data from Fortran 90/95 modules is supported.''',
115:           url="http://cens.ioc.ee/projects/f2py2e/",
116:           keywords=['Fortran', 'f2py'],
117:           **config)
118: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_99199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, (-1)), 'str', '\nsetup.py for installing F2PY\n\nUsage:\n   python setup.py install\n\nCopyright 2001-2005 Pearu Peterson all rights reserved,\nPearu Peterson <pearu@cens.ioc.ee>\nPermission to use, modify, and distribute this software is given under the\nterms of the NumPy License.\n\nNO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.\n$Revision: 1.32 $\n$Date: 2005/01/30 17:22:14 $\nPearu Peterson\n\n')

# Assigning a Str to a Name (line 21):
str_99200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 14), 'str', '$Id: setup.py,v 1.32 2005/01/30 17:22:14 pearu Exp $')
# Assigning a type to the variable '__version__' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), '__version__', str_99200)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'import os' statement (line 23)
import os

import_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 24, 0))

# 'import sys' statement (line 24)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 24, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 25, 0))

# 'from distutils.dep_util import newer' statement (line 25)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_99201 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'distutils.dep_util')

if (type(import_99201) is not StypyTypeError):

    if (import_99201 != 'pyd_module'):
        __import__(import_99201)
        sys_modules_99202 = sys.modules[import_99201]
        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'distutils.dep_util', sys_modules_99202.module_type_store, module_type_store, ['newer'])
        nest_module(stypy.reporting.localization.Localization(__file__, 25, 0), __file__, sys_modules_99202, sys_modules_99202.module_type_store, module_type_store)
    else:
        from distutils.dep_util import newer

        import_from_module(stypy.reporting.localization.Localization(__file__, 25, 0), 'distutils.dep_util', None, module_type_store, ['newer'], [newer])

else:
    # Assigning a type to the variable 'distutils.dep_util' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'distutils.dep_util', import_99201)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 0))

# 'from numpy.distutils import log' statement (line 26)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_99203 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy.distutils')

if (type(import_99203) is not StypyTypeError):

    if (import_99203 != 'pyd_module'):
        __import__(import_99203)
        sys_modules_99204 = sys.modules[import_99203]
        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy.distutils', sys_modules_99204.module_type_store, module_type_store, ['log'])
        nest_module(stypy.reporting.localization.Localization(__file__, 26, 0), __file__, sys_modules_99204, sys_modules_99204.module_type_store, module_type_store)
    else:
        from numpy.distutils import log

        import_from_module(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy.distutils', None, module_type_store, ['log'], [log])

else:
    # Assigning a type to the variable 'numpy.distutils' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'numpy.distutils', import_99203)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 27, 0))

# 'from numpy.distutils.core import setup' statement (line 27)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_99205 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'numpy.distutils.core')

if (type(import_99205) is not StypyTypeError):

    if (import_99205 != 'pyd_module'):
        __import__(import_99205)
        sys_modules_99206 = sys.modules[import_99205]
        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'numpy.distutils.core', sys_modules_99206.module_type_store, module_type_store, ['setup'])
        nest_module(stypy.reporting.localization.Localization(__file__, 27, 0), __file__, sys_modules_99206, sys_modules_99206.module_type_store, module_type_store)
    else:
        from numpy.distutils.core import setup

        import_from_module(stypy.reporting.localization.Localization(__file__, 27, 0), 'numpy.distutils.core', None, module_type_store, ['setup'], [setup])

else:
    # Assigning a type to the variable 'numpy.distutils.core' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'numpy.distutils.core', import_99205)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 28, 0))

# 'from numpy.distutils.misc_util import Configuration' statement (line 28)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_99207 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'numpy.distutils.misc_util')

if (type(import_99207) is not StypyTypeError):

    if (import_99207 != 'pyd_module'):
        __import__(import_99207)
        sys_modules_99208 = sys.modules[import_99207]
        import_from_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'numpy.distutils.misc_util', sys_modules_99208.module_type_store, module_type_store, ['Configuration'])
        nest_module(stypy.reporting.localization.Localization(__file__, 28, 0), __file__, sys_modules_99208, sys_modules_99208.module_type_store, module_type_store)
    else:
        from numpy.distutils.misc_util import Configuration

        import_from_module(stypy.reporting.localization.Localization(__file__, 28, 0), 'numpy.distutils.misc_util', None, module_type_store, ['Configuration'], [Configuration])

else:
    # Assigning a type to the variable 'numpy.distutils.misc_util' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'numpy.distutils.misc_util', import_99207)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 30, 0))

# 'from __version__ import version' statement (line 30)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
import_99209 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 30, 0), '__version__')

if (type(import_99209) is not StypyTypeError):

    if (import_99209 != 'pyd_module'):
        __import__(import_99209)
        sys_modules_99210 = sys.modules[import_99209]
        import_from_module(stypy.reporting.localization.Localization(__file__, 30, 0), '__version__', sys_modules_99210.module_type_store, module_type_store, ['version'])
        nest_module(stypy.reporting.localization.Localization(__file__, 30, 0), __file__, sys_modules_99210, sys_modules_99210.module_type_store, module_type_store)
    else:
        from __version__ import version

        import_from_module(stypy.reporting.localization.Localization(__file__, 30, 0), '__version__', None, module_type_store, ['version'], [version])

else:
    # Assigning a type to the variable '__version__' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), '__version__', import_99209)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')


@norecursion
def _get_f2py_shebang(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_get_f2py_shebang'
    module_type_store = module_type_store.open_function_context('_get_f2py_shebang', 33, 0, False)
    
    # Passed parameters checking function
    _get_f2py_shebang.stypy_localization = localization
    _get_f2py_shebang.stypy_type_of_self = None
    _get_f2py_shebang.stypy_type_store = module_type_store
    _get_f2py_shebang.stypy_function_name = '_get_f2py_shebang'
    _get_f2py_shebang.stypy_param_names_list = []
    _get_f2py_shebang.stypy_varargs_param_name = None
    _get_f2py_shebang.stypy_kwargs_param_name = None
    _get_f2py_shebang.stypy_call_defaults = defaults
    _get_f2py_shebang.stypy_call_varargs = varargs
    _get_f2py_shebang.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_get_f2py_shebang', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_get_f2py_shebang', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_get_f2py_shebang(...)' code ##################

    str_99211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, (-1)), 'str', ' Return shebang line for f2py script\n\n    If we are building a binary distribution format, then the shebang line\n    should be ``#!python`` rather than ``#!`` followed by the contents of\n    ``sys.executable``.\n    ')
    
    
    # Call to intersection(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'sys' (line 41)
    sys_99221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 39), 'sys', False)
    # Obtaining the member 'argv' of a type (line 41)
    argv_99222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 39), sys_99221, 'argv')
    # Processing the call keyword arguments (line 40)
    kwargs_99223 = {}
    
    # Call to set(...): (line 40)
    # Processing the call arguments (line 40)
    
    # Obtaining an instance of the builtin type 'tuple' (line 40)
    tuple_99213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 40)
    # Adding element type (line 40)
    str_99214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 12), 'str', 'bdist_wheel')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 12), tuple_99213, str_99214)
    # Adding element type (line 40)
    str_99215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 27), 'str', 'bdist_egg')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 12), tuple_99213, str_99215)
    # Adding element type (line 40)
    str_99216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 40), 'str', 'bdist_wininst')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 12), tuple_99213, str_99216)
    # Adding element type (line 40)
    str_99217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 12), 'str', 'bdist_rpm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 12), tuple_99213, str_99217)
    
    # Processing the call keyword arguments (line 40)
    kwargs_99218 = {}
    # Getting the type of 'set' (line 40)
    set_99212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 7), 'set', False)
    # Calling set(args, kwargs) (line 40)
    set_call_result_99219 = invoke(stypy.reporting.localization.Localization(__file__, 40, 7), set_99212, *[tuple_99213], **kwargs_99218)
    
    # Obtaining the member 'intersection' of a type (line 40)
    intersection_99220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 7), set_call_result_99219, 'intersection')
    # Calling intersection(args, kwargs) (line 40)
    intersection_call_result_99224 = invoke(stypy.reporting.localization.Localization(__file__, 40, 7), intersection_99220, *[argv_99222], **kwargs_99223)
    
    # Testing the type of an if condition (line 40)
    if_condition_99225 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 40, 4), intersection_call_result_99224)
    # Assigning a type to the variable 'if_condition_99225' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'if_condition_99225', if_condition_99225)
    # SSA begins for if statement (line 40)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_99226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 15), 'str', '#!python')
    # Assigning a type to the variable 'stypy_return_type' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'stypy_return_type', str_99226)
    # SSA join for if statement (line 40)
    module_type_store = module_type_store.join_ssa_context()
    
    str_99227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 11), 'str', '#!')
    # Getting the type of 'sys' (line 43)
    sys_99228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 18), 'sys')
    # Obtaining the member 'executable' of a type (line 43)
    executable_99229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 18), sys_99228, 'executable')
    # Applying the binary operator '+' (line 43)
    result_add_99230 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 11), '+', str_99227, executable_99229)
    
    # Assigning a type to the variable 'stypy_return_type' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'stypy_return_type', result_add_99230)
    
    # ################# End of '_get_f2py_shebang(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_get_f2py_shebang' in the type store
    # Getting the type of 'stypy_return_type' (line 33)
    stypy_return_type_99231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_99231)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_get_f2py_shebang'
    return stypy_return_type_99231

# Assigning a type to the variable '_get_f2py_shebang' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), '_get_f2py_shebang', _get_f2py_shebang)

@norecursion
def configuration(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_99232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 33), 'str', '')
    # Getting the type of 'None' (line 46)
    None_99233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 46), 'None')
    defaults = [str_99232, None_99233]
    # Create a new context for function 'configuration'
    module_type_store = module_type_store.open_function_context('configuration', 46, 0, False)
    
    # Passed parameters checking function
    configuration.stypy_localization = localization
    configuration.stypy_type_of_self = None
    configuration.stypy_type_store = module_type_store
    configuration.stypy_function_name = 'configuration'
    configuration.stypy_param_names_list = ['parent_package', 'top_path']
    configuration.stypy_varargs_param_name = None
    configuration.stypy_kwargs_param_name = None
    configuration.stypy_call_defaults = defaults
    configuration.stypy_call_varargs = varargs
    configuration.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'configuration', ['parent_package', 'top_path'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'configuration', localization, ['parent_package', 'top_path'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'configuration(...)' code ##################

    
    # Assigning a Call to a Name (line 47):
    
    # Call to Configuration(...): (line 47)
    # Processing the call arguments (line 47)
    str_99235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 27), 'str', 'f2py')
    # Getting the type of 'parent_package' (line 47)
    parent_package_99236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 35), 'parent_package', False)
    # Getting the type of 'top_path' (line 47)
    top_path_99237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 51), 'top_path', False)
    # Processing the call keyword arguments (line 47)
    kwargs_99238 = {}
    # Getting the type of 'Configuration' (line 47)
    Configuration_99234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 13), 'Configuration', False)
    # Calling Configuration(args, kwargs) (line 47)
    Configuration_call_result_99239 = invoke(stypy.reporting.localization.Localization(__file__, 47, 13), Configuration_99234, *[str_99235, parent_package_99236, top_path_99237], **kwargs_99238)
    
    # Assigning a type to the variable 'config' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'config', Configuration_call_result_99239)
    
    # Call to add_data_dir(...): (line 49)
    # Processing the call arguments (line 49)
    str_99242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 24), 'str', 'tests')
    # Processing the call keyword arguments (line 49)
    kwargs_99243 = {}
    # Getting the type of 'config' (line 49)
    config_99240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'config', False)
    # Obtaining the member 'add_data_dir' of a type (line 49)
    add_data_dir_99241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 4), config_99240, 'add_data_dir')
    # Calling add_data_dir(args, kwargs) (line 49)
    add_data_dir_call_result_99244 = invoke(stypy.reporting.localization.Localization(__file__, 49, 4), add_data_dir_99241, *[str_99242], **kwargs_99243)
    
    
    # Call to add_data_files(...): (line 51)
    # Processing the call arguments (line 51)
    str_99247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 26), 'str', 'src/fortranobject.c')
    str_99248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 26), 'str', 'src/fortranobject.h')
    # Processing the call keyword arguments (line 51)
    kwargs_99249 = {}
    # Getting the type of 'config' (line 51)
    config_99245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'config', False)
    # Obtaining the member 'add_data_files' of a type (line 51)
    add_data_files_99246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 4), config_99245, 'add_data_files')
    # Calling add_data_files(args, kwargs) (line 51)
    add_data_files_call_result_99250 = invoke(stypy.reporting.localization.Localization(__file__, 51, 4), add_data_files_99246, *[str_99247, str_99248], **kwargs_99249)
    
    
    # Call to make_svn_version_py(...): (line 55)
    # Processing the call keyword arguments (line 55)
    kwargs_99253 = {}
    # Getting the type of 'config' (line 55)
    config_99251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'config', False)
    # Obtaining the member 'make_svn_version_py' of a type (line 55)
    make_svn_version_py_99252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 4), config_99251, 'make_svn_version_py')
    # Calling make_svn_version_py(args, kwargs) (line 55)
    make_svn_version_py_call_result_99254 = invoke(stypy.reporting.localization.Localization(__file__, 55, 4), make_svn_version_py_99252, *[], **kwargs_99253)
    

    @norecursion
    def generate_f2py_py(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'generate_f2py_py'
        module_type_store = module_type_store.open_function_context('generate_f2py_py', 57, 4, False)
        
        # Passed parameters checking function
        generate_f2py_py.stypy_localization = localization
        generate_f2py_py.stypy_type_of_self = None
        generate_f2py_py.stypy_type_store = module_type_store
        generate_f2py_py.stypy_function_name = 'generate_f2py_py'
        generate_f2py_py.stypy_param_names_list = ['build_dir']
        generate_f2py_py.stypy_varargs_param_name = None
        generate_f2py_py.stypy_kwargs_param_name = None
        generate_f2py_py.stypy_call_defaults = defaults
        generate_f2py_py.stypy_call_varargs = varargs
        generate_f2py_py.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'generate_f2py_py', ['build_dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'generate_f2py_py', localization, ['build_dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'generate_f2py_py(...)' code ##################

        
        # Assigning a BinOp to a Name (line 58):
        str_99255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 19), 'str', 'f2py')
        
        # Obtaining the type of the subscript
        int_99256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 61), 'int')
        slice_99257 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 58, 28), int_99256, None, None)
        
        # Call to basename(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'sys' (line 58)
        sys_99261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 45), 'sys', False)
        # Obtaining the member 'executable' of a type (line 58)
        executable_99262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 45), sys_99261, 'executable')
        # Processing the call keyword arguments (line 58)
        kwargs_99263 = {}
        # Getting the type of 'os' (line 58)
        os_99258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 28), 'os', False)
        # Obtaining the member 'path' of a type (line 58)
        path_99259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 28), os_99258, 'path')
        # Obtaining the member 'basename' of a type (line 58)
        basename_99260 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 28), path_99259, 'basename')
        # Calling basename(args, kwargs) (line 58)
        basename_call_result_99264 = invoke(stypy.reporting.localization.Localization(__file__, 58, 28), basename_99260, *[executable_99262], **kwargs_99263)
        
        # Obtaining the member '__getitem__' of a type (line 58)
        getitem___99265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 28), basename_call_result_99264, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 58)
        subscript_call_result_99266 = invoke(stypy.reporting.localization.Localization(__file__, 58, 28), getitem___99265, slice_99257)
        
        # Applying the binary operator '+' (line 58)
        result_add_99267 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 19), '+', str_99255, subscript_call_result_99266)
        
        # Assigning a type to the variable 'f2py_exe' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'f2py_exe', result_add_99267)
        
        
        
        # Obtaining the type of the subscript
        int_99268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 20), 'int')
        slice_99269 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 59, 11), int_99268, None, None)
        # Getting the type of 'f2py_exe' (line 59)
        f2py_exe_99270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 11), 'f2py_exe')
        # Obtaining the member '__getitem__' of a type (line 59)
        getitem___99271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 11), f2py_exe_99270, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 59)
        subscript_call_result_99272 = invoke(stypy.reporting.localization.Localization(__file__, 59, 11), getitem___99271, slice_99269)
        
        str_99273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 28), 'str', '.exe')
        # Applying the binary operator '==' (line 59)
        result_eq_99274 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 11), '==', subscript_call_result_99272, str_99273)
        
        # Testing the type of an if condition (line 59)
        if_condition_99275 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 59, 8), result_eq_99274)
        # Assigning a type to the variable 'if_condition_99275' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'if_condition_99275', if_condition_99275)
        # SSA begins for if statement (line 59)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 60):
        
        # Obtaining the type of the subscript
        int_99276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 33), 'int')
        slice_99277 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 60, 23), None, int_99276, None)
        # Getting the type of 'f2py_exe' (line 60)
        f2py_exe_99278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 23), 'f2py_exe')
        # Obtaining the member '__getitem__' of a type (line 60)
        getitem___99279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 23), f2py_exe_99278, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 60)
        subscript_call_result_99280 = invoke(stypy.reporting.localization.Localization(__file__, 60, 23), getitem___99279, slice_99277)
        
        str_99281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 39), 'str', '.py')
        # Applying the binary operator '+' (line 60)
        result_add_99282 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 23), '+', subscript_call_result_99280, str_99281)
        
        # Assigning a type to the variable 'f2py_exe' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'f2py_exe', result_add_99282)
        # SSA join for if statement (line 59)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        
        str_99283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 11), 'str', 'bdist_wininst')
        # Getting the type of 'sys' (line 61)
        sys_99284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 30), 'sys')
        # Obtaining the member 'argv' of a type (line 61)
        argv_99285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 30), sys_99284, 'argv')
        # Applying the binary operator 'in' (line 61)
        result_contains_99286 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 11), 'in', str_99283, argv_99285)
        
        
        
        # Obtaining the type of the subscript
        int_99287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 52), 'int')
        slice_99288 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 61, 43), int_99287, None, None)
        # Getting the type of 'f2py_exe' (line 61)
        f2py_exe_99289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 43), 'f2py_exe')
        # Obtaining the member '__getitem__' of a type (line 61)
        getitem___99290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 43), f2py_exe_99289, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 61)
        subscript_call_result_99291 = invoke(stypy.reporting.localization.Localization(__file__, 61, 43), getitem___99290, slice_99288)
        
        str_99292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 60), 'str', '.py')
        # Applying the binary operator '!=' (line 61)
        result_ne_99293 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 43), '!=', subscript_call_result_99291, str_99292)
        
        # Applying the binary operator 'and' (line 61)
        result_and_keyword_99294 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 11), 'and', result_contains_99286, result_ne_99293)
        
        # Testing the type of an if condition (line 61)
        if_condition_99295 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 61, 8), result_and_keyword_99294)
        # Assigning a type to the variable 'if_condition_99295' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'if_condition_99295', if_condition_99295)
        # SSA begins for if statement (line 61)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Name (line 62):
        # Getting the type of 'f2py_exe' (line 62)
        f2py_exe_99296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 23), 'f2py_exe')
        str_99297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 34), 'str', '.py')
        # Applying the binary operator '+' (line 62)
        result_add_99298 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 23), '+', f2py_exe_99296, str_99297)
        
        # Assigning a type to the variable 'f2py_exe' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'f2py_exe', result_add_99298)
        # SSA join for if statement (line 61)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 63):
        
        # Call to join(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'build_dir' (line 63)
        build_dir_99302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 30), 'build_dir', False)
        # Getting the type of 'f2py_exe' (line 63)
        f2py_exe_99303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 41), 'f2py_exe', False)
        # Processing the call keyword arguments (line 63)
        kwargs_99304 = {}
        # Getting the type of 'os' (line 63)
        os_99299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 63)
        path_99300 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 17), os_99299, 'path')
        # Obtaining the member 'join' of a type (line 63)
        join_99301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 17), path_99300, 'join')
        # Calling join(args, kwargs) (line 63)
        join_call_result_99305 = invoke(stypy.reporting.localization.Localization(__file__, 63, 17), join_99301, *[build_dir_99302, f2py_exe_99303], **kwargs_99304)
        
        # Assigning a type to the variable 'target' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'target', join_call_result_99305)
        
        
        # Call to newer(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of '__file__' (line 64)
        file___99307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 17), '__file__', False)
        # Getting the type of 'target' (line 64)
        target_99308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 27), 'target', False)
        # Processing the call keyword arguments (line 64)
        kwargs_99309 = {}
        # Getting the type of 'newer' (line 64)
        newer_99306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 11), 'newer', False)
        # Calling newer(args, kwargs) (line 64)
        newer_call_result_99310 = invoke(stypy.reporting.localization.Localization(__file__, 64, 11), newer_99306, *[file___99307, target_99308], **kwargs_99309)
        
        # Testing the type of an if condition (line 64)
        if_condition_99311 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 64, 8), newer_call_result_99310)
        # Assigning a type to the variable 'if_condition_99311' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'if_condition_99311', if_condition_99311)
        # SSA begins for if statement (line 64)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to info(...): (line 65)
        # Processing the call arguments (line 65)
        str_99314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 21), 'str', 'Creating %s')
        # Getting the type of 'target' (line 65)
        target_99315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 36), 'target', False)
        # Processing the call keyword arguments (line 65)
        kwargs_99316 = {}
        # Getting the type of 'log' (line 65)
        log_99312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'log', False)
        # Obtaining the member 'info' of a type (line 65)
        info_99313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 12), log_99312, 'info')
        # Calling info(args, kwargs) (line 65)
        info_call_result_99317 = invoke(stypy.reporting.localization.Localization(__file__, 65, 12), info_99313, *[str_99314, target_99315], **kwargs_99316)
        
        
        # Assigning a Call to a Name (line 66):
        
        # Call to open(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'target' (line 66)
        target_99319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 21), 'target', False)
        str_99320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 29), 'str', 'w')
        # Processing the call keyword arguments (line 66)
        kwargs_99321 = {}
        # Getting the type of 'open' (line 66)
        open_99318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 16), 'open', False)
        # Calling open(args, kwargs) (line 66)
        open_call_result_99322 = invoke(stypy.reporting.localization.Localization(__file__, 66, 16), open_99318, *[target_99319, str_99320], **kwargs_99321)
        
        # Assigning a type to the variable 'f' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'f', open_call_result_99322)
        
        # Call to write(...): (line 67)
        # Processing the call arguments (line 67)
        
        # Call to _get_f2py_shebang(...): (line 67)
        # Processing the call keyword arguments (line 67)
        kwargs_99326 = {}
        # Getting the type of '_get_f2py_shebang' (line 67)
        _get_f2py_shebang_99325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 20), '_get_f2py_shebang', False)
        # Calling _get_f2py_shebang(args, kwargs) (line 67)
        _get_f2py_shebang_call_result_99327 = invoke(stypy.reporting.localization.Localization(__file__, 67, 20), _get_f2py_shebang_99325, *[], **kwargs_99326)
        
        str_99328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 42), 'str', '\n')
        # Applying the binary operator '+' (line 67)
        result_add_99329 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 20), '+', _get_f2py_shebang_call_result_99327, str_99328)
        
        # Processing the call keyword arguments (line 67)
        kwargs_99330 = {}
        # Getting the type of 'f' (line 67)
        f_99323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'f', False)
        # Obtaining the member 'write' of a type (line 67)
        write_99324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 12), f_99323, 'write')
        # Calling write(args, kwargs) (line 67)
        write_call_result_99331 = invoke(stypy.reporting.localization.Localization(__file__, 67, 12), write_99324, *[result_add_99329], **kwargs_99330)
        
        
        # Assigning a Call to a Name (line 68):
        
        # Call to join(...): (line 68)
        # Processing the call arguments (line 68)
        
        # Call to dirname(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of '__file__' (line 68)
        file___99338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 51), '__file__', False)
        # Processing the call keyword arguments (line 68)
        kwargs_99339 = {}
        # Getting the type of 'os' (line 68)
        os_99335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 35), 'os', False)
        # Obtaining the member 'path' of a type (line 68)
        path_99336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 35), os_99335, 'path')
        # Obtaining the member 'dirname' of a type (line 68)
        dirname_99337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 35), path_99336, 'dirname')
        # Calling dirname(args, kwargs) (line 68)
        dirname_call_result_99340 = invoke(stypy.reporting.localization.Localization(__file__, 68, 35), dirname_99337, *[file___99338], **kwargs_99339)
        
        str_99341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 62), 'str', '__main__.py')
        # Processing the call keyword arguments (line 68)
        kwargs_99342 = {}
        # Getting the type of 'os' (line 68)
        os_99332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 22), 'os', False)
        # Obtaining the member 'path' of a type (line 68)
        path_99333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 22), os_99332, 'path')
        # Obtaining the member 'join' of a type (line 68)
        join_99334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 22), path_99333, 'join')
        # Calling join(args, kwargs) (line 68)
        join_call_result_99343 = invoke(stypy.reporting.localization.Localization(__file__, 68, 22), join_99334, *[dirname_call_result_99340, str_99341], **kwargs_99342)
        
        # Assigning a type to the variable 'mainloc' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'mainloc', join_call_result_99343)
        
        # Call to open(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'mainloc' (line 69)
        mainloc_99345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 22), 'mainloc', False)
        # Processing the call keyword arguments (line 69)
        kwargs_99346 = {}
        # Getting the type of 'open' (line 69)
        open_99344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 17), 'open', False)
        # Calling open(args, kwargs) (line 69)
        open_call_result_99347 = invoke(stypy.reporting.localization.Localization(__file__, 69, 17), open_99344, *[mainloc_99345], **kwargs_99346)
        
        with_99348 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 69, 17), open_call_result_99347, 'with parameter', '__enter__', '__exit__')

        if with_99348:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 69)
            enter___99349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 17), open_call_result_99347, '__enter__')
            with_enter_99350 = invoke(stypy.reporting.localization.Localization(__file__, 69, 17), enter___99349)
            # Assigning a type to the variable 'mf' (line 69)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 17), 'mf', with_enter_99350)
            
            # Call to write(...): (line 70)
            # Processing the call arguments (line 70)
            
            # Call to read(...): (line 70)
            # Processing the call keyword arguments (line 70)
            kwargs_99355 = {}
            # Getting the type of 'mf' (line 70)
            mf_99353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 24), 'mf', False)
            # Obtaining the member 'read' of a type (line 70)
            read_99354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 24), mf_99353, 'read')
            # Calling read(args, kwargs) (line 70)
            read_call_result_99356 = invoke(stypy.reporting.localization.Localization(__file__, 70, 24), read_99354, *[], **kwargs_99355)
            
            # Processing the call keyword arguments (line 70)
            kwargs_99357 = {}
            # Getting the type of 'f' (line 70)
            f_99351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'f', False)
            # Obtaining the member 'write' of a type (line 70)
            write_99352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 16), f_99351, 'write')
            # Calling write(args, kwargs) (line 70)
            write_call_result_99358 = invoke(stypy.reporting.localization.Localization(__file__, 70, 16), write_99352, *[read_call_result_99356], **kwargs_99357)
            
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 69)
            exit___99359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 17), open_call_result_99347, '__exit__')
            with_exit_99360 = invoke(stypy.reporting.localization.Localization(__file__, 69, 17), exit___99359, None, None, None)

        
        # Call to close(...): (line 71)
        # Processing the call keyword arguments (line 71)
        kwargs_99363 = {}
        # Getting the type of 'f' (line 71)
        f_99361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'f', False)
        # Obtaining the member 'close' of a type (line 71)
        close_99362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 12), f_99361, 'close')
        # Calling close(args, kwargs) (line 71)
        close_call_result_99364 = invoke(stypy.reporting.localization.Localization(__file__, 71, 12), close_99362, *[], **kwargs_99363)
        
        # SSA join for if statement (line 64)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'target' (line 72)
        target_99365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 15), 'target')
        # Assigning a type to the variable 'stypy_return_type' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'stypy_return_type', target_99365)
        
        # ################# End of 'generate_f2py_py(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'generate_f2py_py' in the type store
        # Getting the type of 'stypy_return_type' (line 57)
        stypy_return_type_99366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_99366)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'generate_f2py_py'
        return stypy_return_type_99366

    # Assigning a type to the variable 'generate_f2py_py' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'generate_f2py_py', generate_f2py_py)
    
    # Call to add_scripts(...): (line 74)
    # Processing the call arguments (line 74)
    # Getting the type of 'generate_f2py_py' (line 74)
    generate_f2py_py_99369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 23), 'generate_f2py_py', False)
    # Processing the call keyword arguments (line 74)
    kwargs_99370 = {}
    # Getting the type of 'config' (line 74)
    config_99367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'config', False)
    # Obtaining the member 'add_scripts' of a type (line 74)
    add_scripts_99368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 4), config_99367, 'add_scripts')
    # Calling add_scripts(args, kwargs) (line 74)
    add_scripts_call_result_99371 = invoke(stypy.reporting.localization.Localization(__file__, 74, 4), add_scripts_99368, *[generate_f2py_py_99369], **kwargs_99370)
    
    
    # Call to info(...): (line 76)
    # Processing the call arguments (line 76)
    str_99374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 13), 'str', 'F2PY Version %s')
    
    # Call to get_version(...): (line 76)
    # Processing the call keyword arguments (line 76)
    kwargs_99377 = {}
    # Getting the type of 'config' (line 76)
    config_99375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 32), 'config', False)
    # Obtaining the member 'get_version' of a type (line 76)
    get_version_99376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 32), config_99375, 'get_version')
    # Calling get_version(args, kwargs) (line 76)
    get_version_call_result_99378 = invoke(stypy.reporting.localization.Localization(__file__, 76, 32), get_version_99376, *[], **kwargs_99377)
    
    # Processing the call keyword arguments (line 76)
    kwargs_99379 = {}
    # Getting the type of 'log' (line 76)
    log_99372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'log', False)
    # Obtaining the member 'info' of a type (line 76)
    info_99373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 4), log_99372, 'info')
    # Calling info(args, kwargs) (line 76)
    info_call_result_99380 = invoke(stypy.reporting.localization.Localization(__file__, 76, 4), info_99373, *[str_99374, get_version_call_result_99378], **kwargs_99379)
    
    # Getting the type of 'config' (line 78)
    config_99381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 11), 'config')
    # Assigning a type to the variable 'stypy_return_type' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type', config_99381)
    
    # ################# End of 'configuration(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'configuration' in the type store
    # Getting the type of 'stypy_return_type' (line 46)
    stypy_return_type_99382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_99382)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'configuration'
    return stypy_return_type_99382

# Assigning a type to the variable 'configuration' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'configuration', configuration)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 82):
    
    # Call to configuration(...): (line 82)
    # Processing the call keyword arguments (line 82)
    str_99384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 36), 'str', '')
    keyword_99385 = str_99384
    kwargs_99386 = {'top_path': keyword_99385}
    # Getting the type of 'configuration' (line 82)
    configuration_99383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 13), 'configuration', False)
    # Calling configuration(args, kwargs) (line 82)
    configuration_call_result_99387 = invoke(stypy.reporting.localization.Localization(__file__, 82, 13), configuration_99383, *[], **kwargs_99386)
    
    # Assigning a type to the variable 'config' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'config', configuration_call_result_99387)
    
    # Call to print(...): (line 83)
    # Processing the call arguments (line 83)
    str_99389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 10), 'str', 'F2PY Version')
    # Getting the type of 'version' (line 83)
    version_99390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 26), 'version', False)
    # Processing the call keyword arguments (line 83)
    kwargs_99391 = {}
    # Getting the type of 'print' (line 83)
    print_99388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'print', False)
    # Calling print(args, kwargs) (line 83)
    print_call_result_99392 = invoke(stypy.reporting.localization.Localization(__file__, 83, 4), print_99388, *[str_99389, version_99390], **kwargs_99391)
    
    
    # Assigning a Call to a Name (line 84):
    
    # Call to todict(...): (line 84)
    # Processing the call keyword arguments (line 84)
    kwargs_99395 = {}
    # Getting the type of 'config' (line 84)
    config_99393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 13), 'config', False)
    # Obtaining the member 'todict' of a type (line 84)
    todict_99394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 13), config_99393, 'todict')
    # Calling todict(args, kwargs) (line 84)
    todict_call_result_99396 = invoke(stypy.reporting.localization.Localization(__file__, 84, 13), todict_99394, *[], **kwargs_99395)
    
    # Assigning a type to the variable 'config' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'config', todict_call_result_99396)
    
    # Assigning a Str to a Subscript (line 86):
    str_99397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 29), 'str', 'http://cens.ioc.ee/projects/f2py2e/2.x/F2PY-2-latest.tar.gz')
    # Getting the type of 'config' (line 86)
    config_99398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'config')
    str_99399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 11), 'str', 'download_url')
    # Storing an element on a container (line 86)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 4), config_99398, (str_99399, str_99397))
    
    # Assigning a List to a Subscript (line 88):
    
    # Obtaining an instance of the builtin type 'list' (line 88)
    list_99400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 88)
    # Adding element type (line 88)
    str_99401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 8), 'str', 'Development Status :: 5 - Production/Stable')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 28), list_99400, str_99401)
    # Adding element type (line 88)
    str_99402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 8), 'str', 'Intended Audience :: Developers')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 28), list_99400, str_99402)
    # Adding element type (line 88)
    str_99403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 8), 'str', 'Intended Audience :: Science/Research')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 28), list_99400, str_99403)
    # Adding element type (line 88)
    str_99404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 8), 'str', 'License :: OSI Approved :: NumPy License')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 28), list_99400, str_99404)
    # Adding element type (line 88)
    str_99405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 8), 'str', 'Natural Language :: English')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 28), list_99400, str_99405)
    # Adding element type (line 88)
    str_99406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 8), 'str', 'Operating System :: OS Independent')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 28), list_99400, str_99406)
    # Adding element type (line 88)
    str_99407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 8), 'str', 'Programming Language :: C')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 28), list_99400, str_99407)
    # Adding element type (line 88)
    str_99408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 8), 'str', 'Programming Language :: Fortran')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 28), list_99400, str_99408)
    # Adding element type (line 88)
    str_99409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 8), 'str', 'Programming Language :: Python')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 28), list_99400, str_99409)
    # Adding element type (line 88)
    str_99410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 8), 'str', 'Topic :: Scientific/Engineering')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 28), list_99400, str_99410)
    # Adding element type (line 88)
    str_99411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 8), 'str', 'Topic :: Software Development :: Code Generators')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 28), list_99400, str_99411)
    
    # Getting the type of 'config' (line 88)
    config_99412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'config')
    str_99413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 11), 'str', 'classifiers')
    # Storing an element on a container (line 88)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 4), config_99412, (str_99413, list_99400))
    
    # Call to setup(...): (line 101)
    # Processing the call keyword arguments (line 101)
    # Getting the type of 'version' (line 101)
    version_99415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 18), 'version', False)
    keyword_99416 = version_99415
    str_99417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 22), 'str', 'F2PY - Fortran to Python Interface Generaton')
    keyword_99418 = str_99417
    str_99419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 17), 'str', 'Pearu Peterson')
    keyword_99420 = str_99419
    str_99421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 23), 'str', 'pearu@cens.ioc.ee')
    keyword_99422 = str_99421
    str_99423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 21), 'str', 'Pearu Peterson')
    keyword_99424 = str_99423
    str_99425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 27), 'str', 'pearu@cens.ioc.ee')
    keyword_99426 = str_99425
    str_99427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 18), 'str', 'BSD')
    keyword_99428 = str_99427
    str_99429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 20), 'str', 'Unix, Windows (mingw|cygwin), Mac OSX')
    keyword_99430 = str_99429
    str_99431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, (-1)), 'str', 'The Fortran to Python Interface Generator, or F2PY for short, is a\ncommand line tool (f2py) for generating Python C/API modules for\nwrapping Fortran 77/90/95 subroutines, accessing common blocks from\nPython, and calling Python functions from Fortran (call-backs).\nInterfacing subroutines/data from Fortran 90/95 modules is supported.')
    keyword_99432 = str_99431
    str_99433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 14), 'str', 'http://cens.ioc.ee/projects/f2py2e/')
    keyword_99434 = str_99433
    
    # Obtaining an instance of the builtin type 'list' (line 116)
    list_99435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 116)
    # Adding element type (line 116)
    str_99436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 20), 'str', 'Fortran')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 19), list_99435, str_99436)
    # Adding element type (line 116)
    str_99437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 31), 'str', 'f2py')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 19), list_99435, str_99437)
    
    keyword_99438 = list_99435
    # Getting the type of 'config' (line 117)
    config_99439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 12), 'config', False)
    kwargs_99440 = {'maintainer': keyword_99424, 'description': keyword_99418, 'license': keyword_99428, 'author': keyword_99420, 'author_email': keyword_99422, 'maintainer_email': keyword_99426, 'platforms': keyword_99430, 'version': keyword_99416, 'url': keyword_99434, 'keywords': keyword_99438, 'long_description': keyword_99432, 'config_99439': config_99439}
    # Getting the type of 'setup' (line 101)
    setup_99414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'setup', False)
    # Calling setup(args, kwargs) (line 101)
    setup_call_result_99441 = invoke(stypy.reporting.localization.Localization(__file__, 101, 4), setup_99414, *[], **kwargs_99440)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
