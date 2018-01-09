
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: NumPy
3: =====
4: 
5: Provides
6:   1. An array object of arbitrary homogeneous items
7:   2. Fast mathematical operations over arrays
8:   3. Linear Algebra, Fourier Transforms, Random Number Generation
9: 
10: How to use the documentation
11: ----------------------------
12: Documentation is available in two forms: docstrings provided
13: with the code, and a loose standing reference guide, available from
14: `the NumPy homepage <http://www.scipy.org>`_.
15: 
16: We recommend exploring the docstrings using
17: `IPython <http://ipython.scipy.org>`_, an advanced Python shell with
18: TAB-completion and introspection capabilities.  See below for further
19: instructions.
20: 
21: The docstring examples assume that `numpy` has been imported as `np`::
22: 
23:   >>> import numpy as np
24: 
25: Code snippets are indicated by three greater-than signs::
26: 
27:   >>> x = 42
28:   >>> x = x + 1
29: 
30: Use the built-in ``help`` function to view a function's docstring::
31: 
32:   >>> help(np.sort)
33:   ... # doctest: +SKIP
34: 
35: For some objects, ``np.info(obj)`` may provide additional help.  This is
36: particularly true if you see the line "Help on ufunc object:" at the top
37: of the help() page.  Ufuncs are implemented in C, not Python, for speed.
38: The native Python help() does not know how to view their help, but our
39: np.info() function does.
40: 
41: To search for documents containing a keyword, do::
42: 
43:   >>> np.lookfor('keyword')
44:   ... # doctest: +SKIP
45: 
46: General-purpose documents like a glossary and help on the basic concepts
47: of numpy are available under the ``doc`` sub-module::
48: 
49:   >>> from numpy import doc
50:   >>> help(doc)
51:   ... # doctest: +SKIP
52: 
53: Available subpackages
54: ---------------------
55: doc
56:     Topical documentation on broadcasting, indexing, etc.
57: lib
58:     Basic functions used by several sub-packages.
59: random
60:     Core Random Tools
61: linalg
62:     Core Linear Algebra Tools
63: fft
64:     Core FFT routines
65: polynomial
66:     Polynomial tools
67: testing
68:     Numpy testing tools
69: f2py
70:     Fortran to Python Interface Generator.
71: distutils
72:     Enhancements to distutils with support for
73:     Fortran compilers support and more.
74: 
75: Utilities
76: ---------
77: test
78:     Run numpy unittests
79: show_config
80:     Show numpy build configuration
81: dual
82:     Overwrite certain functions with high-performance Scipy tools
83: matlib
84:     Make everything matrices.
85: __version__
86:     Numpy version string
87: 
88: Viewing documentation using IPython
89: -----------------------------------
90: Start IPython with the NumPy profile (``ipython -p numpy``), which will
91: import `numpy` under the alias `np`.  Then, use the ``cpaste`` command to
92: paste examples into the shell.  To see which functions are available in
93: `numpy`, type ``np.<TAB>`` (where ``<TAB>`` refers to the TAB key), or use
94: ``np.*cos*?<ENTER>`` (where ``<ENTER>`` refers to the ENTER key) to narrow
95: down the list.  To view the docstring for a function, use
96: ``np.cos?<ENTER>`` (to view the docstring) and ``np.cos??<ENTER>`` (to view
97: the source code).
98: 
99: Copies vs. in-place operation
100: -----------------------------
101: Most of the functions in `numpy` return a copy of the array argument
102: (e.g., `np.sort`).  In-place versions of these functions are often
103: available as array methods, i.e. ``x = np.array([1,2,3]); x.sort()``.
104: Exceptions to this rule are documented.
105: 
106: '''
107: from __future__ import division, absolute_import, print_function
108: 
109: import sys
110: 
111: 
112: class ModuleDeprecationWarning(DeprecationWarning):
113:     '''Module deprecation warning.
114: 
115:     The nose tester turns ordinary Deprecation warnings into test failures.
116:     That makes it hard to deprecate whole modules, because they get
117:     imported by default. So this is a special Deprecation warning that the
118:     nose tester will let pass without making tests fail.
119: 
120:     '''
121:     pass
122: 
123: 
124: class VisibleDeprecationWarning(UserWarning):
125:     '''Visible deprecation warning.
126: 
127:     By default, python will not show deprecation warnings, so this class
128:     can be used when a very visible warning is helpful, for example because
129:     the usage is most likely a user bug.
130: 
131:     '''
132:     pass
133: 
134: 
135: class _NoValue:
136:     '''Special keyword value.
137: 
138:     This class may be used as the default value assigned to a
139:     deprecated keyword in order to check if it has been given a user
140:     defined value.
141:     '''
142:     pass
143: 
144: 
145: # oldnumeric and numarray were removed in 1.9. In case some packages import
146: # but do not use them, we define them here for backward compatibility.
147: oldnumeric = 'removed'
148: numarray = 'removed'
149: 
150: 
151: # We first need to detect if we're being called as part of the numpy setup
152: # procedure itself in a reliable manner.
153: try:
154:     __NUMPY_SETUP__
155: except NameError:
156:     __NUMPY_SETUP__ = False
157: 
158: 
159: if __NUMPY_SETUP__:
160:     import sys as _sys
161:     _sys.stderr.write('Running from numpy source directory.\n')
162:     del _sys
163: else:
164:     try:
165:         from numpy.__config__ import show as show_config
166:     except ImportError:
167:         msg = '''Error importing numpy: you should not try to import numpy from
168:         its source directory; please exit the numpy source tree, and relaunch
169:         your python interpreter from there.'''
170:         raise ImportError(msg)
171:     from .version import git_revision as __git_revision__
172:     from .version import version as __version__
173: 
174:     from ._import_tools import PackageLoader
175: 
176:     def pkgload(*packages, **options):
177:         loader = PackageLoader(infunc=True)
178:         return loader(*packages, **options)
179: 
180:     from . import add_newdocs
181:     __all__ = ['add_newdocs',
182:                'ModuleDeprecationWarning',
183:                'VisibleDeprecationWarning']
184: 
185:     pkgload.__doc__ = PackageLoader.__call__.__doc__
186: 
187:     # We don't actually use this ourselves anymore, but I'm not 100% sure that
188:     # no-one else in the world is using it (though I hope not)
189:     from .testing import Tester
190:     test = testing.nosetester._numpy_tester().test
191:     bench = testing.nosetester._numpy_tester().bench
192: 
193:     from . import core
194:     from .core import *
195:     from . import compat
196:     from . import lib
197:     from .lib import *
198:     from . import linalg
199:     from . import fft
200:     from . import polynomial
201:     from . import random
202:     from . import ctypeslib
203:     from . import ma
204:     from . import matrixlib as _mat
205:     from .matrixlib import *
206:     from .compat import long
207: 
208:     # Make these accessible from numpy name-space
209:     #  but not imported in from numpy import *
210:     if sys.version_info[0] >= 3:
211:         from builtins import bool, int, float, complex, object, str
212:         unicode = str
213:     else:
214:         from __builtin__ import bool, int, float, complex, object, unicode, str
215: 
216:     from .core import round, abs, max, min
217: 
218:     __all__.extend(['__version__', 'pkgload', 'PackageLoader',
219:                'show_config'])
220:     __all__.extend(core.__all__)
221:     __all__.extend(_mat.__all__)
222:     __all__.extend(lib.__all__)
223:     __all__.extend(['linalg', 'fft', 'random', 'ctypeslib', 'ma'])
224: 
225:     # Filter annoying Cython warnings that serve no good purpose.
226:     import warnings
227:     warnings.filterwarnings("ignore", message="numpy.dtype size changed")
228:     warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
229:     warnings.filterwarnings("ignore", message="numpy.ndarray size changed")
230: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, (-1)), 'str', '\nNumPy\n=====\n\nProvides\n  1. An array object of arbitrary homogeneous items\n  2. Fast mathematical operations over arrays\n  3. Linear Algebra, Fourier Transforms, Random Number Generation\n\nHow to use the documentation\n----------------------------\nDocumentation is available in two forms: docstrings provided\nwith the code, and a loose standing reference guide, available from\n`the NumPy homepage <http://www.scipy.org>`_.\n\nWe recommend exploring the docstrings using\n`IPython <http://ipython.scipy.org>`_, an advanced Python shell with\nTAB-completion and introspection capabilities.  See below for further\ninstructions.\n\nThe docstring examples assume that `numpy` has been imported as `np`::\n\n  >>> import numpy as np\n\nCode snippets are indicated by three greater-than signs::\n\n  >>> x = 42\n  >>> x = x + 1\n\nUse the built-in ``help`` function to view a function\'s docstring::\n\n  >>> help(np.sort)\n  ... # doctest: +SKIP\n\nFor some objects, ``np.info(obj)`` may provide additional help.  This is\nparticularly true if you see the line "Help on ufunc object:" at the top\nof the help() page.  Ufuncs are implemented in C, not Python, for speed.\nThe native Python help() does not know how to view their help, but our\nnp.info() function does.\n\nTo search for documents containing a keyword, do::\n\n  >>> np.lookfor(\'keyword\')\n  ... # doctest: +SKIP\n\nGeneral-purpose documents like a glossary and help on the basic concepts\nof numpy are available under the ``doc`` sub-module::\n\n  >>> from numpy import doc\n  >>> help(doc)\n  ... # doctest: +SKIP\n\nAvailable subpackages\n---------------------\ndoc\n    Topical documentation on broadcasting, indexing, etc.\nlib\n    Basic functions used by several sub-packages.\nrandom\n    Core Random Tools\nlinalg\n    Core Linear Algebra Tools\nfft\n    Core FFT routines\npolynomial\n    Polynomial tools\ntesting\n    Numpy testing tools\nf2py\n    Fortran to Python Interface Generator.\ndistutils\n    Enhancements to distutils with support for\n    Fortran compilers support and more.\n\nUtilities\n---------\ntest\n    Run numpy unittests\nshow_config\n    Show numpy build configuration\ndual\n    Overwrite certain functions with high-performance Scipy tools\nmatlib\n    Make everything matrices.\n__version__\n    Numpy version string\n\nViewing documentation using IPython\n-----------------------------------\nStart IPython with the NumPy profile (``ipython -p numpy``), which will\nimport `numpy` under the alias `np`.  Then, use the ``cpaste`` command to\npaste examples into the shell.  To see which functions are available in\n`numpy`, type ``np.<TAB>`` (where ``<TAB>`` refers to the TAB key), or use\n``np.*cos*?<ENTER>`` (where ``<ENTER>`` refers to the ENTER key) to narrow\ndown the list.  To view the docstring for a function, use\n``np.cos?<ENTER>`` (to view the docstring) and ``np.cos??<ENTER>`` (to view\nthe source code).\n\nCopies vs. in-place operation\n-----------------------------\nMost of the functions in `numpy` return a copy of the array argument\n(e.g., `np.sort`).  In-place versions of these functions are often\navailable as array methods, i.e. ``x = np.array([1,2,3]); x.sort()``.\nExceptions to this rule are documented.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 109, 0))

# 'import sys' statement (line 109)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 109, 0), 'sys', sys, module_type_store)

# Declaration of the 'ModuleDeprecationWarning' class
# Getting the type of 'DeprecationWarning' (line 112)
DeprecationWarning_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 31), 'DeprecationWarning')

class ModuleDeprecationWarning(DeprecationWarning_12, ):
    str_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, (-1)), 'str', 'Module deprecation warning.\n\n    The nose tester turns ordinary Deprecation warnings into test failures.\n    That makes it hard to deprecate whole modules, because they get\n    imported by default. So this is a special Deprecation warning that the\n    nose tester will let pass without making tests fail.\n\n    ')
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 112, 0, False)
        # Assigning a type to the variable 'self' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ModuleDeprecationWarning.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'ModuleDeprecationWarning' (line 112)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 0), 'ModuleDeprecationWarning', ModuleDeprecationWarning)
# Declaration of the 'VisibleDeprecationWarning' class
# Getting the type of 'UserWarning' (line 124)
UserWarning_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 32), 'UserWarning')

class VisibleDeprecationWarning(UserWarning_14, ):
    str_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, (-1)), 'str', 'Visible deprecation warning.\n\n    By default, python will not show deprecation warnings, so this class\n    can be used when a very visible warning is helpful, for example because\n    the usage is most likely a user bug.\n\n    ')
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 124, 0, False)
        # Assigning a type to the variable 'self' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VisibleDeprecationWarning.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'VisibleDeprecationWarning' (line 124)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 0), 'VisibleDeprecationWarning', VisibleDeprecationWarning)
# Declaration of the '_NoValue' class

class _NoValue:
    str_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, (-1)), 'str', 'Special keyword value.\n\n    This class may be used as the default value assigned to a\n    deprecated keyword in order to check if it has been given a user\n    defined value.\n    ')
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 135, 0, False)
        # Assigning a type to the variable 'self' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_NoValue.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable '_NoValue' (line 135)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 0), '_NoValue', _NoValue)

# Assigning a Str to a Name (line 147):
str_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 13), 'str', 'removed')
# Assigning a type to the variable 'oldnumeric' (line 147)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 0), 'oldnumeric', str_17)

# Assigning a Str to a Name (line 148):
str_18 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 11), 'str', 'removed')
# Assigning a type to the variable 'numarray' (line 148)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 0), 'numarray', str_18)


# SSA begins for try-except statement (line 153)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
# Getting the type of '__NUMPY_SETUP__' (line 154)
NUMPY_SETUP___19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), '__NUMPY_SETUP__')
# SSA branch for the except part of a try statement (line 153)
# SSA branch for the except 'NameError' branch of a try statement (line 153)
module_type_store.open_ssa_branch('except')

# Assigning a Name to a Name (line 156):
# Getting the type of 'False' (line 156)
False_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 22), 'False')
# Assigning a type to the variable '__NUMPY_SETUP__' (line 156)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), '__NUMPY_SETUP__', False_20)
# SSA join for try-except statement (line 153)
module_type_store = module_type_store.join_ssa_context()


# Getting the type of '__NUMPY_SETUP__' (line 159)
NUMPY_SETUP___21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 3), '__NUMPY_SETUP__')
# Testing the type of an if condition (line 159)
if_condition_22 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 0), NUMPY_SETUP___21)
# Assigning a type to the variable 'if_condition_22' (line 159)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 0), 'if_condition_22', if_condition_22)
# SSA begins for if statement (line 159)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 160, 4))

# 'import sys' statement (line 160)
import sys as _sys

import_module(stypy.reporting.localization.Localization(__file__, 160, 4), '_sys', _sys, module_type_store)


# Call to write(...): (line 161)
# Processing the call arguments (line 161)
str_26 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 22), 'str', 'Running from numpy source directory.\n')
# Processing the call keyword arguments (line 161)
kwargs_27 = {}
# Getting the type of '_sys' (line 161)
_sys_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), '_sys', False)
# Obtaining the member 'stderr' of a type (line 161)
stderr_24 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 4), _sys_23, 'stderr')
# Obtaining the member 'write' of a type (line 161)
write_25 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 4), stderr_24, 'write')
# Calling write(args, kwargs) (line 161)
write_call_result_28 = invoke(stypy.reporting.localization.Localization(__file__, 161, 4), write_25, *[str_26], **kwargs_27)

# Deleting a member
module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 162, 4), module_type_store, '_sys')
# SSA branch for the else part of an if statement (line 159)
module_type_store.open_ssa_branch('else')


# SSA begins for try-except statement (line 164)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 165, 8))

# 'from numpy.__config__ import show_config' statement (line 165)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/')
import_29 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 165, 8), 'numpy.__config__')

if (type(import_29) is not StypyTypeError):

    if (import_29 != 'pyd_module'):
        __import__(import_29)
        sys_modules_30 = sys.modules[import_29]
        import_from_module(stypy.reporting.localization.Localization(__file__, 165, 8), 'numpy.__config__', sys_modules_30.module_type_store, module_type_store, ['show'])
        nest_module(stypy.reporting.localization.Localization(__file__, 165, 8), __file__, sys_modules_30, sys_modules_30.module_type_store, module_type_store)
    else:
        from numpy.__config__ import show as show_config

        import_from_module(stypy.reporting.localization.Localization(__file__, 165, 8), 'numpy.__config__', None, module_type_store, ['show'], [show_config])

else:
    # Assigning a type to the variable 'numpy.__config__' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'numpy.__config__', import_29)

# Adding an alias
module_type_store.add_alias('show_config', 'show')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/')

# SSA branch for the except part of a try statement (line 164)
# SSA branch for the except 'ImportError' branch of a try statement (line 164)
module_type_store.open_ssa_branch('except')

# Assigning a Str to a Name (line 167):
str_31 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, (-1)), 'str', 'Error importing numpy: you should not try to import numpy from\n        its source directory; please exit the numpy source tree, and relaunch\n        your python interpreter from there.')
# Assigning a type to the variable 'msg' (line 167)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'msg', str_31)

# Call to ImportError(...): (line 170)
# Processing the call arguments (line 170)
# Getting the type of 'msg' (line 170)
msg_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 26), 'msg', False)
# Processing the call keyword arguments (line 170)
kwargs_34 = {}
# Getting the type of 'ImportError' (line 170)
ImportError_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 14), 'ImportError', False)
# Calling ImportError(args, kwargs) (line 170)
ImportError_call_result_35 = invoke(stypy.reporting.localization.Localization(__file__, 170, 14), ImportError_32, *[msg_33], **kwargs_34)

ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 170, 8), ImportError_call_result_35, 'raise parameter', BaseException)
# SSA join for try-except statement (line 164)
module_type_store = module_type_store.join_ssa_context()

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 171, 4))

# 'from numpy.version import __git_revision__' statement (line 171)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/')
import_36 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 171, 4), 'numpy.version')

if (type(import_36) is not StypyTypeError):

    if (import_36 != 'pyd_module'):
        __import__(import_36)
        sys_modules_37 = sys.modules[import_36]
        import_from_module(stypy.reporting.localization.Localization(__file__, 171, 4), 'numpy.version', sys_modules_37.module_type_store, module_type_store, ['git_revision'])
        nest_module(stypy.reporting.localization.Localization(__file__, 171, 4), __file__, sys_modules_37, sys_modules_37.module_type_store, module_type_store)
    else:
        from numpy.version import git_revision as __git_revision__

        import_from_module(stypy.reporting.localization.Localization(__file__, 171, 4), 'numpy.version', None, module_type_store, ['git_revision'], [__git_revision__])

else:
    # Assigning a type to the variable 'numpy.version' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'numpy.version', import_36)

# Adding an alias
module_type_store.add_alias('__git_revision__', 'git_revision')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 172, 4))

# 'from numpy.version import __version__' statement (line 172)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/')
import_38 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 172, 4), 'numpy.version')

if (type(import_38) is not StypyTypeError):

    if (import_38 != 'pyd_module'):
        __import__(import_38)
        sys_modules_39 = sys.modules[import_38]
        import_from_module(stypy.reporting.localization.Localization(__file__, 172, 4), 'numpy.version', sys_modules_39.module_type_store, module_type_store, ['version'])
        nest_module(stypy.reporting.localization.Localization(__file__, 172, 4), __file__, sys_modules_39, sys_modules_39.module_type_store, module_type_store)
    else:
        from numpy.version import version as __version__

        import_from_module(stypy.reporting.localization.Localization(__file__, 172, 4), 'numpy.version', None, module_type_store, ['version'], [__version__])

else:
    # Assigning a type to the variable 'numpy.version' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'numpy.version', import_38)

# Adding an alias
module_type_store.add_alias('__version__', 'version')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 174, 4))

# 'from numpy._import_tools import PackageLoader' statement (line 174)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/')
import_40 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 174, 4), 'numpy._import_tools')

if (type(import_40) is not StypyTypeError):

    if (import_40 != 'pyd_module'):
        __import__(import_40)
        sys_modules_41 = sys.modules[import_40]
        import_from_module(stypy.reporting.localization.Localization(__file__, 174, 4), 'numpy._import_tools', sys_modules_41.module_type_store, module_type_store, ['PackageLoader'])
        nest_module(stypy.reporting.localization.Localization(__file__, 174, 4), __file__, sys_modules_41, sys_modules_41.module_type_store, module_type_store)
    else:
        from numpy._import_tools import PackageLoader

        import_from_module(stypy.reporting.localization.Localization(__file__, 174, 4), 'numpy._import_tools', None, module_type_store, ['PackageLoader'], [PackageLoader])

else:
    # Assigning a type to the variable 'numpy._import_tools' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'numpy._import_tools', import_40)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/')


@norecursion
def pkgload(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'pkgload'
    module_type_store = module_type_store.open_function_context('pkgload', 176, 4, False)
    
    # Passed parameters checking function
    pkgload.stypy_localization = localization
    pkgload.stypy_type_of_self = None
    pkgload.stypy_type_store = module_type_store
    pkgload.stypy_function_name = 'pkgload'
    pkgload.stypy_param_names_list = []
    pkgload.stypy_varargs_param_name = 'packages'
    pkgload.stypy_kwargs_param_name = 'options'
    pkgload.stypy_call_defaults = defaults
    pkgload.stypy_call_varargs = varargs
    pkgload.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'pkgload', [], 'packages', 'options', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'pkgload', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'pkgload(...)' code ##################

    
    # Assigning a Call to a Name (line 177):
    
    # Call to PackageLoader(...): (line 177)
    # Processing the call keyword arguments (line 177)
    # Getting the type of 'True' (line 177)
    True_43 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 38), 'True', False)
    keyword_44 = True_43
    kwargs_45 = {'infunc': keyword_44}
    # Getting the type of 'PackageLoader' (line 177)
    PackageLoader_42 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 17), 'PackageLoader', False)
    # Calling PackageLoader(args, kwargs) (line 177)
    PackageLoader_call_result_46 = invoke(stypy.reporting.localization.Localization(__file__, 177, 17), PackageLoader_42, *[], **kwargs_45)
    
    # Assigning a type to the variable 'loader' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'loader', PackageLoader_call_result_46)
    
    # Call to loader(...): (line 178)
    # Getting the type of 'packages' (line 178)
    packages_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 23), 'packages', False)
    # Processing the call keyword arguments (line 178)
    # Getting the type of 'options' (line 178)
    options_49 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 35), 'options', False)
    kwargs_50 = {'options_49': options_49}
    # Getting the type of 'loader' (line 178)
    loader_47 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 15), 'loader', False)
    # Calling loader(args, kwargs) (line 178)
    loader_call_result_51 = invoke(stypy.reporting.localization.Localization(__file__, 178, 15), loader_47, *[packages_48], **kwargs_50)
    
    # Assigning a type to the variable 'stypy_return_type' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'stypy_return_type', loader_call_result_51)
    
    # ################# End of 'pkgload(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'pkgload' in the type store
    # Getting the type of 'stypy_return_type' (line 176)
    stypy_return_type_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_52)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'pkgload'
    return stypy_return_type_52

# Assigning a type to the variable 'pkgload' (line 176)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'pkgload', pkgload)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 180, 4))

# 'from numpy import add_newdocs' statement (line 180)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/')
import_53 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 180, 4), 'numpy')

if (type(import_53) is not StypyTypeError):

    if (import_53 != 'pyd_module'):
        __import__(import_53)
        sys_modules_54 = sys.modules[import_53]
        import_from_module(stypy.reporting.localization.Localization(__file__, 180, 4), 'numpy', sys_modules_54.module_type_store, module_type_store, ['add_newdocs'])
        nest_module(stypy.reporting.localization.Localization(__file__, 180, 4), __file__, sys_modules_54, sys_modules_54.module_type_store, module_type_store)
    else:
        from numpy import add_newdocs

        import_from_module(stypy.reporting.localization.Localization(__file__, 180, 4), 'numpy', None, module_type_store, ['add_newdocs'], [add_newdocs])

else:
    # Assigning a type to the variable 'numpy' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'numpy', import_53)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/')


# Assigning a List to a Name (line 181):
__all__ = ['add_newdocs', 'ModuleDeprecationWarning', 'VisibleDeprecationWarning']
module_type_store.set_exportable_members(['add_newdocs', 'ModuleDeprecationWarning', 'VisibleDeprecationWarning'])

# Obtaining an instance of the builtin type 'list' (line 181)
list_55 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 181)
# Adding element type (line 181)
str_56 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 15), 'str', 'add_newdocs')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 14), list_55, str_56)
# Adding element type (line 181)
str_57 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 15), 'str', 'ModuleDeprecationWarning')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 14), list_55, str_57)
# Adding element type (line 181)
str_58 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 15), 'str', 'VisibleDeprecationWarning')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 181, 14), list_55, str_58)

# Assigning a type to the variable '__all__' (line 181)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), '__all__', list_55)

# Assigning a Attribute to a Attribute (line 185):
# Getting the type of 'PackageLoader' (line 185)
PackageLoader_59 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 22), 'PackageLoader')
# Obtaining the member '__call__' of a type (line 185)
call___60 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 22), PackageLoader_59, '__call__')
# Obtaining the member '__doc__' of a type (line 185)
doc___61 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 22), call___60, '__doc__')
# Getting the type of 'pkgload' (line 185)
pkgload_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'pkgload')
# Setting the type of the member '__doc__' of a type (line 185)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 4), pkgload_62, '__doc__', doc___61)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 189, 4))

# 'from numpy.testing import Tester' statement (line 189)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/')
import_63 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 189, 4), 'numpy.testing')

if (type(import_63) is not StypyTypeError):

    if (import_63 != 'pyd_module'):
        __import__(import_63)
        sys_modules_64 = sys.modules[import_63]
        import_from_module(stypy.reporting.localization.Localization(__file__, 189, 4), 'numpy.testing', sys_modules_64.module_type_store, module_type_store, ['Tester'])
        nest_module(stypy.reporting.localization.Localization(__file__, 189, 4), __file__, sys_modules_64, sys_modules_64.module_type_store, module_type_store)
    else:
        from numpy.testing import Tester

        import_from_module(stypy.reporting.localization.Localization(__file__, 189, 4), 'numpy.testing', None, module_type_store, ['Tester'], [Tester])

else:
    # Assigning a type to the variable 'numpy.testing' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'numpy.testing', import_63)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/')


# Assigning a Attribute to a Name (line 190):

# Call to _numpy_tester(...): (line 190)
# Processing the call keyword arguments (line 190)
kwargs_68 = {}
# Getting the type of 'testing' (line 190)
testing_65 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 11), 'testing', False)
# Obtaining the member 'nosetester' of a type (line 190)
nosetester_66 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 11), testing_65, 'nosetester')
# Obtaining the member '_numpy_tester' of a type (line 190)
_numpy_tester_67 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 11), nosetester_66, '_numpy_tester')
# Calling _numpy_tester(args, kwargs) (line 190)
_numpy_tester_call_result_69 = invoke(stypy.reporting.localization.Localization(__file__, 190, 11), _numpy_tester_67, *[], **kwargs_68)

# Obtaining the member 'test' of a type (line 190)
test_70 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 190, 11), _numpy_tester_call_result_69, 'test')
# Assigning a type to the variable 'test' (line 190)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'test', test_70)

# Assigning a Attribute to a Name (line 191):

# Call to _numpy_tester(...): (line 191)
# Processing the call keyword arguments (line 191)
kwargs_74 = {}
# Getting the type of 'testing' (line 191)
testing_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'testing', False)
# Obtaining the member 'nosetester' of a type (line 191)
nosetester_72 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 12), testing_71, 'nosetester')
# Obtaining the member '_numpy_tester' of a type (line 191)
_numpy_tester_73 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 12), nosetester_72, '_numpy_tester')
# Calling _numpy_tester(args, kwargs) (line 191)
_numpy_tester_call_result_75 = invoke(stypy.reporting.localization.Localization(__file__, 191, 12), _numpy_tester_73, *[], **kwargs_74)

# Obtaining the member 'bench' of a type (line 191)
bench_76 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 12), _numpy_tester_call_result_75, 'bench')
# Assigning a type to the variable 'bench' (line 191)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'bench', bench_76)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 193, 4))

# 'from numpy import core' statement (line 193)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/')
import_77 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 193, 4), 'numpy')

if (type(import_77) is not StypyTypeError):

    if (import_77 != 'pyd_module'):
        __import__(import_77)
        sys_modules_78 = sys.modules[import_77]
        import_from_module(stypy.reporting.localization.Localization(__file__, 193, 4), 'numpy', sys_modules_78.module_type_store, module_type_store, ['core'])
        nest_module(stypy.reporting.localization.Localization(__file__, 193, 4), __file__, sys_modules_78, sys_modules_78.module_type_store, module_type_store)
    else:
        from numpy import core

        import_from_module(stypy.reporting.localization.Localization(__file__, 193, 4), 'numpy', None, module_type_store, ['core'], [core])

else:
    # Assigning a type to the variable 'numpy' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'numpy', import_77)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 194, 4))

# 'from numpy.core import ' statement (line 194)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/')
import_79 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 194, 4), 'numpy.core')

if (type(import_79) is not StypyTypeError):

    if (import_79 != 'pyd_module'):
        __import__(import_79)
        sys_modules_80 = sys.modules[import_79]
        import_from_module(stypy.reporting.localization.Localization(__file__, 194, 4), 'numpy.core', sys_modules_80.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 194, 4), __file__, sys_modules_80, sys_modules_80.module_type_store, module_type_store)
    else:
        from numpy.core import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 194, 4), 'numpy.core', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.core' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'numpy.core', import_79)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 195, 4))

# 'from numpy import compat' statement (line 195)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/')
import_81 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 195, 4), 'numpy')

if (type(import_81) is not StypyTypeError):

    if (import_81 != 'pyd_module'):
        __import__(import_81)
        sys_modules_82 = sys.modules[import_81]
        import_from_module(stypy.reporting.localization.Localization(__file__, 195, 4), 'numpy', sys_modules_82.module_type_store, module_type_store, ['compat'])
        nest_module(stypy.reporting.localization.Localization(__file__, 195, 4), __file__, sys_modules_82, sys_modules_82.module_type_store, module_type_store)
    else:
        from numpy import compat

        import_from_module(stypy.reporting.localization.Localization(__file__, 195, 4), 'numpy', None, module_type_store, ['compat'], [compat])

else:
    # Assigning a type to the variable 'numpy' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'numpy', import_81)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 196, 4))

# 'from numpy import lib' statement (line 196)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/')
import_83 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 196, 4), 'numpy')

if (type(import_83) is not StypyTypeError):

    if (import_83 != 'pyd_module'):
        __import__(import_83)
        sys_modules_84 = sys.modules[import_83]
        import_from_module(stypy.reporting.localization.Localization(__file__, 196, 4), 'numpy', sys_modules_84.module_type_store, module_type_store, ['lib'])
        nest_module(stypy.reporting.localization.Localization(__file__, 196, 4), __file__, sys_modules_84, sys_modules_84.module_type_store, module_type_store)
    else:
        from numpy import lib

        import_from_module(stypy.reporting.localization.Localization(__file__, 196, 4), 'numpy', None, module_type_store, ['lib'], [lib])

else:
    # Assigning a type to the variable 'numpy' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'numpy', import_83)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 197, 4))

# 'from numpy.lib import ' statement (line 197)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/')
import_85 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 197, 4), 'numpy.lib')

if (type(import_85) is not StypyTypeError):

    if (import_85 != 'pyd_module'):
        __import__(import_85)
        sys_modules_86 = sys.modules[import_85]
        import_from_module(stypy.reporting.localization.Localization(__file__, 197, 4), 'numpy.lib', sys_modules_86.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 197, 4), __file__, sys_modules_86, sys_modules_86.module_type_store, module_type_store)
    else:
        from numpy.lib import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 197, 4), 'numpy.lib', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.lib' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'numpy.lib', import_85)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 198, 4))

# 'from numpy import linalg' statement (line 198)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/')
import_87 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 198, 4), 'numpy')

if (type(import_87) is not StypyTypeError):

    if (import_87 != 'pyd_module'):
        __import__(import_87)
        sys_modules_88 = sys.modules[import_87]
        import_from_module(stypy.reporting.localization.Localization(__file__, 198, 4), 'numpy', sys_modules_88.module_type_store, module_type_store, ['linalg'])
        nest_module(stypy.reporting.localization.Localization(__file__, 198, 4), __file__, sys_modules_88, sys_modules_88.module_type_store, module_type_store)
    else:
        from numpy import linalg

        import_from_module(stypy.reporting.localization.Localization(__file__, 198, 4), 'numpy', None, module_type_store, ['linalg'], [linalg])

else:
    # Assigning a type to the variable 'numpy' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'numpy', import_87)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 199, 4))

# 'from numpy import fft' statement (line 199)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/')
import_89 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 199, 4), 'numpy')

if (type(import_89) is not StypyTypeError):

    if (import_89 != 'pyd_module'):
        __import__(import_89)
        sys_modules_90 = sys.modules[import_89]
        import_from_module(stypy.reporting.localization.Localization(__file__, 199, 4), 'numpy', sys_modules_90.module_type_store, module_type_store, ['fft'])
        nest_module(stypy.reporting.localization.Localization(__file__, 199, 4), __file__, sys_modules_90, sys_modules_90.module_type_store, module_type_store)
    else:
        from numpy import fft

        import_from_module(stypy.reporting.localization.Localization(__file__, 199, 4), 'numpy', None, module_type_store, ['fft'], [fft])

else:
    # Assigning a type to the variable 'numpy' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'numpy', import_89)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 200, 4))

# 'from numpy import polynomial' statement (line 200)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/')
import_91 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 200, 4), 'numpy')

if (type(import_91) is not StypyTypeError):

    if (import_91 != 'pyd_module'):# and False:
        __import__(import_91)
        sys_modules_92 = sys.modules[import_91]
        import_from_module(stypy.reporting.localization.Localization(__file__, 200, 4), 'numpy', sys_modules_92.module_type_store, module_type_store, ['polynomial'])
        nest_module(stypy.reporting.localization.Localization(__file__, 200, 4), __file__, sys_modules_92, sys_modules_92.module_type_store, module_type_store)
    else:
        from numpy import polynomial

        import_from_module(stypy.reporting.localization.Localization(__file__, 200, 4), 'numpy', None, module_type_store, ['polynomial'], [polynomial])

else:
    # Assigning a type to the variable 'numpy' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 4), 'numpy', import_91)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 201, 4))

# 'from numpy import random' statement (line 201)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/')
import_93 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 201, 4), 'numpy')

if (type(import_93) is not StypyTypeError):

    if (import_93 != 'pyd_module'):
        __import__(import_93)
        sys_modules_94 = sys.modules[import_93]
        import_from_module(stypy.reporting.localization.Localization(__file__, 201, 4), 'numpy', sys_modules_94.module_type_store, module_type_store, ['random'])
        nest_module(stypy.reporting.localization.Localization(__file__, 201, 4), __file__, sys_modules_94, sys_modules_94.module_type_store, module_type_store)
    else:
        from numpy import random

        import_from_module(stypy.reporting.localization.Localization(__file__, 201, 4), 'numpy', None, module_type_store, ['random'], [random])

else:
    # Assigning a type to the variable 'numpy' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'numpy', import_93)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 202, 4))

# 'from numpy import ctypeslib' statement (line 202)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/')
import_95 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 202, 4), 'numpy')

if (type(import_95) is not StypyTypeError):

    if (import_95 != 'pyd_module'):
        __import__(import_95)
        sys_modules_96 = sys.modules[import_95]
        import_from_module(stypy.reporting.localization.Localization(__file__, 202, 4), 'numpy', sys_modules_96.module_type_store, module_type_store, ['ctypeslib'])
        nest_module(stypy.reporting.localization.Localization(__file__, 202, 4), __file__, sys_modules_96, sys_modules_96.module_type_store, module_type_store)
    else:
        from numpy import ctypeslib

        import_from_module(stypy.reporting.localization.Localization(__file__, 202, 4), 'numpy', None, module_type_store, ['ctypeslib'], [ctypeslib])

else:
    # Assigning a type to the variable 'numpy' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'numpy', import_95)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 203, 4))

# 'from numpy import ma' statement (line 203)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/')
import_97 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 203, 4), 'numpy')

if (type(import_97) is not StypyTypeError):

    if (import_97 != 'pyd_module'):
        __import__(import_97)
        sys_modules_98 = sys.modules[import_97]
        import_from_module(stypy.reporting.localization.Localization(__file__, 203, 4), 'numpy', sys_modules_98.module_type_store, module_type_store, ['ma'])
        nest_module(stypy.reporting.localization.Localization(__file__, 203, 4), __file__, sys_modules_98, sys_modules_98.module_type_store, module_type_store)
    else:
        from numpy import ma

        import_from_module(stypy.reporting.localization.Localization(__file__, 203, 4), 'numpy', None, module_type_store, ['ma'], [ma])

else:
    # Assigning a type to the variable 'numpy' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'numpy', import_97)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 204, 4))

# 'from numpy import _mat' statement (line 204)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/')
import_99 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 204, 4), 'numpy')

if (type(import_99) is not StypyTypeError):

    if (import_99 != 'pyd_module'):
        __import__(import_99)
        sys_modules_100 = sys.modules[import_99]
        import_from_module(stypy.reporting.localization.Localization(__file__, 204, 4), 'numpy', sys_modules_100.module_type_store, module_type_store, ['matrixlib'])
        nest_module(stypy.reporting.localization.Localization(__file__, 204, 4), __file__, sys_modules_100, sys_modules_100.module_type_store, module_type_store)
    else:
        from numpy import matrixlib as _mat

        import_from_module(stypy.reporting.localization.Localization(__file__, 204, 4), 'numpy', None, module_type_store, ['matrixlib'], [_mat])

else:
    # Assigning a type to the variable 'numpy' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'numpy', import_99)

# Adding an alias
module_type_store.add_alias('_mat', 'matrixlib')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 205, 4))

# 'from numpy.matrixlib import ' statement (line 205)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/')
import_101 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 205, 4), 'numpy.matrixlib')

if (type(import_101) is not StypyTypeError):

    if (import_101 != 'pyd_module'):
        __import__(import_101)
        sys_modules_102 = sys.modules[import_101]
        import_from_module(stypy.reporting.localization.Localization(__file__, 205, 4), 'numpy.matrixlib', sys_modules_102.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 205, 4), __file__, sys_modules_102, sys_modules_102.module_type_store, module_type_store)
    else:
        from numpy.matrixlib import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 205, 4), 'numpy.matrixlib', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'numpy.matrixlib' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'numpy.matrixlib', import_101)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 206, 4))

# 'from numpy.compat import long' statement (line 206)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/')
import_103 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 206, 4), 'numpy.compat')

if (type(import_103) is not StypyTypeError):

    if (import_103 != 'pyd_module'):
        __import__(import_103)
        sys_modules_104 = sys.modules[import_103]
        import_from_module(stypy.reporting.localization.Localization(__file__, 206, 4), 'numpy.compat', sys_modules_104.module_type_store, module_type_store, ['long'])
        nest_module(stypy.reporting.localization.Localization(__file__, 206, 4), __file__, sys_modules_104, sys_modules_104.module_type_store, module_type_store)
    else:
        from numpy.compat import long

        import_from_module(stypy.reporting.localization.Localization(__file__, 206, 4), 'numpy.compat', None, module_type_store, ['long'], [long])

else:
    # Assigning a type to the variable 'numpy.compat' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), 'numpy.compat', import_103)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/')




# Obtaining the type of the subscript
int_105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 24), 'int')
# Getting the type of 'sys' (line 210)
sys_106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 7), 'sys')
# Obtaining the member 'version_info' of a type (line 210)
version_info_107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 7), sys_106, 'version_info')
# Obtaining the member '__getitem__' of a type (line 210)
getitem___108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 7), version_info_107, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 210)
subscript_call_result_109 = invoke(stypy.reporting.localization.Localization(__file__, 210, 7), getitem___108, int_105)

int_110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 30), 'int')
# Applying the binary operator '>=' (line 210)
result_ge_111 = python_operator(stypy.reporting.localization.Localization(__file__, 210, 7), '>=', subscript_call_result_109, int_110)

# Testing the type of an if condition (line 210)
if_condition_112 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 210, 4), result_ge_111)
# Assigning a type to the variable 'if_condition_112' (line 210)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'if_condition_112', if_condition_112)
# SSA begins for if statement (line 210)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 211, 8))

# 'from builtins import bool, int, float, complex, object, str' statement (line 211)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/')
import_113 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 211, 8), 'builtins')

if (type(import_113) is not StypyTypeError):

    if (import_113 != 'pyd_module'):
        __import__(import_113)
        sys_modules_114 = sys.modules[import_113]
        import_from_module(stypy.reporting.localization.Localization(__file__, 211, 8), 'builtins', sys_modules_114.module_type_store, module_type_store, ['bool', 'int', 'float', 'complex', 'object', 'str'])
        nest_module(stypy.reporting.localization.Localization(__file__, 211, 8), __file__, sys_modules_114, sys_modules_114.module_type_store, module_type_store)
    else:
        from builtins import bool, int, float, complex, object, str

        import_from_module(stypy.reporting.localization.Localization(__file__, 211, 8), 'builtins', None, module_type_store, ['bool', 'int', 'float', 'complex', 'object', 'str'], [bool, int, float, complex, object, str])

else:
    # Assigning a type to the variable 'builtins' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'builtins', import_113)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/')


# Assigning a Name to a Name (line 212):
# Getting the type of 'str' (line 212)
str_115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 18), 'str')
# Assigning a type to the variable 'unicode' (line 212)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'unicode', str_115)
# SSA branch for the else part of an if statement (line 210)
module_type_store.open_ssa_branch('else')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 214, 8))

# 'from __builtin__ import bool, int, float, complex, object, unicode, str' statement (line 214)
from __builtin__ import bool, int, float, complex, object, unicode, str

import_from_module(stypy.reporting.localization.Localization(__file__, 214, 8), '__builtin__', None, module_type_store, ['bool', 'int', 'float', 'complex', 'object', 'unicode', 'str'], [bool, int, float, complex, object, unicode, str])

# SSA join for if statement (line 210)
module_type_store = module_type_store.join_ssa_context()

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 216, 4))

# 'from numpy.core import round, abs, max, min' statement (line 216)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/')
import_116 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 216, 4), 'numpy.core')

if (type(import_116) is not StypyTypeError):

    if (import_116 != 'pyd_module'):
        __import__(import_116)
        sys_modules_117 = sys.modules[import_116]
        import_from_module(stypy.reporting.localization.Localization(__file__, 216, 4), 'numpy.core', sys_modules_117.module_type_store, module_type_store, ['round', 'abs', 'max', 'min'])
        nest_module(stypy.reporting.localization.Localization(__file__, 216, 4), __file__, sys_modules_117, sys_modules_117.module_type_store, module_type_store)
    else:
        from numpy.core import round, abs, max, min

        import_from_module(stypy.reporting.localization.Localization(__file__, 216, 4), 'numpy.core', None, module_type_store, ['round', 'abs', 'max', 'min'], [round, abs, max, min])

else:
    # Assigning a type to the variable 'numpy.core' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'numpy.core', import_116)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/')


# Call to extend(...): (line 218)
# Processing the call arguments (line 218)

# Obtaining an instance of the builtin type 'list' (line 218)
list_120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 218)
# Adding element type (line 218)
str_121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 20), 'str', '__version__')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 19), list_120, str_121)
# Adding element type (line 218)
str_122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 35), 'str', 'pkgload')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 19), list_120, str_122)
# Adding element type (line 218)
str_123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 46), 'str', 'PackageLoader')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 19), list_120, str_123)
# Adding element type (line 218)
str_124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 15), 'str', 'show_config')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 218, 19), list_120, str_124)

# Processing the call keyword arguments (line 218)
kwargs_125 = {}
# Getting the type of '__all__' (line 218)
all___118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), '__all__', False)
# Obtaining the member 'extend' of a type (line 218)
extend_119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 4), all___118, 'extend')
# Calling extend(args, kwargs) (line 218)
extend_call_result_126 = invoke(stypy.reporting.localization.Localization(__file__, 218, 4), extend_119, *[list_120], **kwargs_125)


# Call to extend(...): (line 220)
# Processing the call arguments (line 220)
# Getting the type of 'core' (line 220)
core_129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 19), 'core', False)
# Obtaining the member '__all__' of a type (line 220)
all___130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 19), core_129, '__all__')
# Processing the call keyword arguments (line 220)
kwargs_131 = {}
# Getting the type of '__all__' (line 220)
all___127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), '__all__', False)
# Obtaining the member 'extend' of a type (line 220)
extend_128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 4), all___127, 'extend')
# Calling extend(args, kwargs) (line 220)
extend_call_result_132 = invoke(stypy.reporting.localization.Localization(__file__, 220, 4), extend_128, *[all___130], **kwargs_131)


# Call to extend(...): (line 221)
# Processing the call arguments (line 221)
# Getting the type of '_mat' (line 221)
_mat_135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 19), '_mat', False)
# Obtaining the member '__all__' of a type (line 221)
all___136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 19), _mat_135, '__all__')
# Processing the call keyword arguments (line 221)
kwargs_137 = {}
# Getting the type of '__all__' (line 221)
all___133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), '__all__', False)
# Obtaining the member 'extend' of a type (line 221)
extend_134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 4), all___133, 'extend')
# Calling extend(args, kwargs) (line 221)
extend_call_result_138 = invoke(stypy.reporting.localization.Localization(__file__, 221, 4), extend_134, *[all___136], **kwargs_137)


# Call to extend(...): (line 222)
# Processing the call arguments (line 222)
# Getting the type of 'lib' (line 222)
lib_141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 19), 'lib', False)
# Obtaining the member '__all__' of a type (line 222)
all___142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 19), lib_141, '__all__')
# Processing the call keyword arguments (line 222)
kwargs_143 = {}
# Getting the type of '__all__' (line 222)
all___139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), '__all__', False)
# Obtaining the member 'extend' of a type (line 222)
extend_140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 4), all___139, 'extend')
# Calling extend(args, kwargs) (line 222)
extend_call_result_144 = invoke(stypy.reporting.localization.Localization(__file__, 222, 4), extend_140, *[all___142], **kwargs_143)


# Call to extend(...): (line 223)
# Processing the call arguments (line 223)

# Obtaining an instance of the builtin type 'list' (line 223)
list_147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 223)
# Adding element type (line 223)
str_148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 20), 'str', 'linalg')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 19), list_147, str_148)
# Adding element type (line 223)
str_149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 30), 'str', 'fft')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 19), list_147, str_149)
# Adding element type (line 223)
str_150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 37), 'str', 'random')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 19), list_147, str_150)
# Adding element type (line 223)
str_151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 47), 'str', 'ctypeslib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 19), list_147, str_151)
# Adding element type (line 223)
str_152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 60), 'str', 'ma')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 19), list_147, str_152)

# Processing the call keyword arguments (line 223)
kwargs_153 = {}
# Getting the type of '__all__' (line 223)
all___145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), '__all__', False)
# Obtaining the member 'extend' of a type (line 223)
extend_146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 4), all___145, 'extend')
# Calling extend(args, kwargs) (line 223)
extend_call_result_154 = invoke(stypy.reporting.localization.Localization(__file__, 223, 4), extend_146, *[list_147], **kwargs_153)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 226, 4))

# 'import warnings' statement (line 226)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 226, 4), 'warnings', warnings, module_type_store)


# Call to filterwarnings(...): (line 227)
# Processing the call arguments (line 227)
str_157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 28), 'str', 'ignore')
# Processing the call keyword arguments (line 227)
str_158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 46), 'str', 'numpy.dtype size changed')
keyword_159 = str_158
kwargs_160 = {'message': keyword_159}
# Getting the type of 'warnings' (line 227)
warnings_155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'warnings', False)
# Obtaining the member 'filterwarnings' of a type (line 227)
filterwarnings_156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 4), warnings_155, 'filterwarnings')
# Calling filterwarnings(args, kwargs) (line 227)
filterwarnings_call_result_161 = invoke(stypy.reporting.localization.Localization(__file__, 227, 4), filterwarnings_156, *[str_157], **kwargs_160)


# Call to filterwarnings(...): (line 228)
# Processing the call arguments (line 228)
str_164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 28), 'str', 'ignore')
# Processing the call keyword arguments (line 228)
str_165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 46), 'str', 'numpy.ufunc size changed')
keyword_166 = str_165
kwargs_167 = {'message': keyword_166}
# Getting the type of 'warnings' (line 228)
warnings_162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'warnings', False)
# Obtaining the member 'filterwarnings' of a type (line 228)
filterwarnings_163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 4), warnings_162, 'filterwarnings')
# Calling filterwarnings(args, kwargs) (line 228)
filterwarnings_call_result_168 = invoke(stypy.reporting.localization.Localization(__file__, 228, 4), filterwarnings_163, *[str_164], **kwargs_167)


# Call to filterwarnings(...): (line 229)
# Processing the call arguments (line 229)
str_171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 28), 'str', 'ignore')
# Processing the call keyword arguments (line 229)
str_172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 46), 'str', 'numpy.ndarray size changed')
keyword_173 = str_172
kwargs_174 = {'message': keyword_173}
# Getting the type of 'warnings' (line 229)
warnings_169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'warnings', False)
# Obtaining the member 'filterwarnings' of a type (line 229)
filterwarnings_170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 4), warnings_169, 'filterwarnings')
# Calling filterwarnings(args, kwargs) (line 229)
filterwarnings_call_result_175 = invoke(stypy.reporting.localization.Localization(__file__, 229, 4), filterwarnings_170, *[str_171], **kwargs_174)

# SSA join for if statement (line 159)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
