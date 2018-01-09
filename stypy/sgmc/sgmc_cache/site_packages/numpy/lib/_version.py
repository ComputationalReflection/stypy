
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Utility to compare (Numpy) version strings.
2: 
3: The NumpyVersion class allows properly comparing numpy version strings.
4: The LooseVersion and StrictVersion classes that distutils provides don't
5: work; they don't recognize anything like alpha/beta/rc/dev versions.
6: 
7: '''
8: from __future__ import division, absolute_import, print_function
9: 
10: import re
11: 
12: from numpy.compat import basestring
13: 
14: 
15: __all__ = ['NumpyVersion']
16: 
17: 
18: class NumpyVersion():
19:     '''Parse and compare numpy version strings.
20: 
21:     Numpy has the following versioning scheme (numbers given are examples; they
22:     can be > 9) in principle):
23: 
24:     - Released version: '1.8.0', '1.8.1', etc.
25:     - Alpha: '1.8.0a1', '1.8.0a2', etc.
26:     - Beta: '1.8.0b1', '1.8.0b2', etc.
27:     - Release candidates: '1.8.0rc1', '1.8.0rc2', etc.
28:     - Development versions: '1.8.0.dev-f1234afa' (git commit hash appended)
29:     - Development versions after a1: '1.8.0a1.dev-f1234afa',
30:                                      '1.8.0b2.dev-f1234afa',
31:                                      '1.8.1rc1.dev-f1234afa', etc.
32:     - Development versions (no git hash available): '1.8.0.dev-Unknown'
33: 
34:     Comparing needs to be done against a valid version string or other
35:     `NumpyVersion` instance. Note that all development versions of the same
36:     (pre-)release compare equal.
37: 
38:     .. versionadded:: 1.9.0
39: 
40:     Parameters
41:     ----------
42:     vstring : str
43:         Numpy version string (``np.__version__``).
44: 
45:     Examples
46:     --------
47:     >>> from numpy.lib import NumpyVersion
48:     >>> if NumpyVersion(np.__version__) < '1.7.0'):
49:     ...     print('skip')
50:     skip
51: 
52:     >>> NumpyVersion('1.7')  # raises ValueError, add ".0"
53: 
54:     '''
55: 
56:     def __init__(self, vstring):
57:         self.vstring = vstring
58:         ver_main = re.match(r'\d[.]\d+[.]\d+', vstring)
59:         if not ver_main:
60:             raise ValueError("Not a valid numpy version string")
61: 
62:         self.version = ver_main.group()
63:         self.major, self.minor, self.bugfix = [int(x) for x in
64:             self.version.split('.')]
65:         if len(vstring) == ver_main.end():
66:             self.pre_release = 'final'
67:         else:
68:             alpha = re.match(r'a\d', vstring[ver_main.end():])
69:             beta = re.match(r'b\d', vstring[ver_main.end():])
70:             rc = re.match(r'rc\d', vstring[ver_main.end():])
71:             pre_rel = [m for m in [alpha, beta, rc] if m is not None]
72:             if pre_rel:
73:                 self.pre_release = pre_rel[0].group()
74:             else:
75:                 self.pre_release = ''
76: 
77:         self.is_devversion = bool(re.search(r'.dev', vstring))
78: 
79:     def _compare_version(self, other):
80:         '''Compare major.minor.bugfix'''
81:         if self.major == other.major:
82:             if self.minor == other.minor:
83:                 if self.bugfix == other.bugfix:
84:                     vercmp = 0
85:                 elif self.bugfix > other.bugfix:
86:                     vercmp = 1
87:                 else:
88:                     vercmp = -1
89:             elif self.minor > other.minor:
90:                 vercmp = 1
91:             else:
92:                 vercmp = -1
93:         elif self.major > other.major:
94:             vercmp = 1
95:         else:
96:             vercmp = -1
97: 
98:         return vercmp
99: 
100:     def _compare_pre_release(self, other):
101:         '''Compare alpha/beta/rc/final.'''
102:         if self.pre_release == other.pre_release:
103:             vercmp = 0
104:         elif self.pre_release == 'final':
105:             vercmp = 1
106:         elif other.pre_release == 'final':
107:             vercmp = -1
108:         elif self.pre_release > other.pre_release:
109:             vercmp = 1
110:         else:
111:             vercmp = -1
112: 
113:         return vercmp
114: 
115:     def _compare(self, other):
116:         if not isinstance(other, (basestring, NumpyVersion)):
117:             raise ValueError("Invalid object to compare with NumpyVersion.")
118: 
119:         if isinstance(other, basestring):
120:             other = NumpyVersion(other)
121: 
122:         vercmp = self._compare_version(other)
123:         if vercmp == 0:
124:             # Same x.y.z version, check for alpha/beta/rc
125:             vercmp = self._compare_pre_release(other)
126:             if vercmp == 0:
127:                 # Same version and same pre-release, check if dev version
128:                 if self.is_devversion is other.is_devversion:
129:                     vercmp = 0
130:                 elif self.is_devversion:
131:                     vercmp = -1
132:                 else:
133:                     vercmp = 1
134: 
135:         return vercmp
136: 
137:     def __lt__(self, other):
138:         return self._compare(other) < 0
139: 
140:     def __le__(self, other):
141:         return self._compare(other) <= 0
142: 
143:     def __eq__(self, other):
144:         return self._compare(other) == 0
145: 
146:     def __ne__(self, other):
147:         return self._compare(other) != 0
148: 
149:     def __gt__(self, other):
150:         return self._compare(other) > 0
151: 
152:     def __ge__(self, other):
153:         return self._compare(other) >= 0
154: 
155:     def __repr(self):
156:         return "NumpyVersion(%s)" % self.vstring
157: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_133815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, (-1)), 'str', "Utility to compare (Numpy) version strings.\n\nThe NumpyVersion class allows properly comparing numpy version strings.\nThe LooseVersion and StrictVersion classes that distutils provides don't\nwork; they don't recognize anything like alpha/beta/rc/dev versions.\n\n")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import re' statement (line 10)
import re

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from numpy.compat import basestring' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_133816 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.compat')

if (type(import_133816) is not StypyTypeError):

    if (import_133816 != 'pyd_module'):
        __import__(import_133816)
        sys_modules_133817 = sys.modules[import_133816]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.compat', sys_modules_133817.module_type_store, module_type_store, ['basestring'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_133817, sys_modules_133817.module_type_store, module_type_store)
    else:
        from numpy.compat import basestring

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.compat', None, module_type_store, ['basestring'], [basestring])

else:
    # Assigning a type to the variable 'numpy.compat' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'numpy.compat', import_133816)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')


# Assigning a List to a Name (line 15):

# Assigning a List to a Name (line 15):
__all__ = ['NumpyVersion']
module_type_store.set_exportable_members(['NumpyVersion'])

# Obtaining an instance of the builtin type 'list' (line 15)
list_133818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 15)
# Adding element type (line 15)
str_133819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 11), 'str', 'NumpyVersion')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 10), list_133818, str_133819)

# Assigning a type to the variable '__all__' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), '__all__', list_133818)
# Declaration of the 'NumpyVersion' class

class NumpyVersion:
    str_133820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, (-1)), 'str', 'Parse and compare numpy version strings.\n\n    Numpy has the following versioning scheme (numbers given are examples; they\n    can be > 9) in principle):\n\n    - Released version: \'1.8.0\', \'1.8.1\', etc.\n    - Alpha: \'1.8.0a1\', \'1.8.0a2\', etc.\n    - Beta: \'1.8.0b1\', \'1.8.0b2\', etc.\n    - Release candidates: \'1.8.0rc1\', \'1.8.0rc2\', etc.\n    - Development versions: \'1.8.0.dev-f1234afa\' (git commit hash appended)\n    - Development versions after a1: \'1.8.0a1.dev-f1234afa\',\n                                     \'1.8.0b2.dev-f1234afa\',\n                                     \'1.8.1rc1.dev-f1234afa\', etc.\n    - Development versions (no git hash available): \'1.8.0.dev-Unknown\'\n\n    Comparing needs to be done against a valid version string or other\n    `NumpyVersion` instance. Note that all development versions of the same\n    (pre-)release compare equal.\n\n    .. versionadded:: 1.9.0\n\n    Parameters\n    ----------\n    vstring : str\n        Numpy version string (``np.__version__``).\n\n    Examples\n    --------\n    >>> from numpy.lib import NumpyVersion\n    >>> if NumpyVersion(np.__version__) < \'1.7.0\'):\n    ...     print(\'skip\')\n    skip\n\n    >>> NumpyVersion(\'1.7\')  # raises ValueError, add ".0"\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 56, 4, False)
        # Assigning a type to the variable 'self' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NumpyVersion.__init__', ['vstring'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['vstring'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 57):
        
        # Assigning a Name to a Attribute (line 57):
        # Getting the type of 'vstring' (line 57)
        vstring_133821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 23), 'vstring')
        # Getting the type of 'self' (line 57)
        self_133822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'self')
        # Setting the type of the member 'vstring' of a type (line 57)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 8), self_133822, 'vstring', vstring_133821)
        
        # Assigning a Call to a Name (line 58):
        
        # Assigning a Call to a Name (line 58):
        
        # Call to match(...): (line 58)
        # Processing the call arguments (line 58)
        str_133825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 28), 'str', '\\d[.]\\d+[.]\\d+')
        # Getting the type of 'vstring' (line 58)
        vstring_133826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 47), 'vstring', False)
        # Processing the call keyword arguments (line 58)
        kwargs_133827 = {}
        # Getting the type of 're' (line 58)
        re_133823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 19), 're', False)
        # Obtaining the member 'match' of a type (line 58)
        match_133824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 19), re_133823, 'match')
        # Calling match(args, kwargs) (line 58)
        match_call_result_133828 = invoke(stypy.reporting.localization.Localization(__file__, 58, 19), match_133824, *[str_133825, vstring_133826], **kwargs_133827)
        
        # Assigning a type to the variable 'ver_main' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'ver_main', match_call_result_133828)
        
        
        # Getting the type of 'ver_main' (line 59)
        ver_main_133829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 15), 'ver_main')
        # Applying the 'not' unary operator (line 59)
        result_not__133830 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 11), 'not', ver_main_133829)
        
        # Testing the type of an if condition (line 59)
        if_condition_133831 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 59, 8), result_not__133830)
        # Assigning a type to the variable 'if_condition_133831' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'if_condition_133831', if_condition_133831)
        # SSA begins for if statement (line 59)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 60)
        # Processing the call arguments (line 60)
        str_133833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 29), 'str', 'Not a valid numpy version string')
        # Processing the call keyword arguments (line 60)
        kwargs_133834 = {}
        # Getting the type of 'ValueError' (line 60)
        ValueError_133832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 60)
        ValueError_call_result_133835 = invoke(stypy.reporting.localization.Localization(__file__, 60, 18), ValueError_133832, *[str_133833], **kwargs_133834)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 60, 12), ValueError_call_result_133835, 'raise parameter', BaseException)
        # SSA join for if statement (line 59)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 62):
        
        # Assigning a Call to a Attribute (line 62):
        
        # Call to group(...): (line 62)
        # Processing the call keyword arguments (line 62)
        kwargs_133838 = {}
        # Getting the type of 'ver_main' (line 62)
        ver_main_133836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 23), 'ver_main', False)
        # Obtaining the member 'group' of a type (line 62)
        group_133837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 23), ver_main_133836, 'group')
        # Calling group(args, kwargs) (line 62)
        group_call_result_133839 = invoke(stypy.reporting.localization.Localization(__file__, 62, 23), group_133837, *[], **kwargs_133838)
        
        # Getting the type of 'self' (line 62)
        self_133840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'self')
        # Setting the type of the member 'version' of a type (line 62)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), self_133840, 'version', group_call_result_133839)
        
        # Assigning a ListComp to a Tuple (line 63):
        
        # Assigning a Subscript to a Name (line 63):
        
        # Obtaining the type of the subscript
        int_133841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 8), 'int')
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to split(...): (line 64)
        # Processing the call arguments (line 64)
        str_133849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 31), 'str', '.')
        # Processing the call keyword arguments (line 64)
        kwargs_133850 = {}
        # Getting the type of 'self' (line 64)
        self_133846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'self', False)
        # Obtaining the member 'version' of a type (line 64)
        version_133847 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 12), self_133846, 'version')
        # Obtaining the member 'split' of a type (line 64)
        split_133848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 12), version_133847, 'split')
        # Calling split(args, kwargs) (line 64)
        split_call_result_133851 = invoke(stypy.reporting.localization.Localization(__file__, 64, 12), split_133848, *[str_133849], **kwargs_133850)
        
        comprehension_133852 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 47), split_call_result_133851)
        # Assigning a type to the variable 'x' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 47), 'x', comprehension_133852)
        
        # Call to int(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'x' (line 63)
        x_133843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 51), 'x', False)
        # Processing the call keyword arguments (line 63)
        kwargs_133844 = {}
        # Getting the type of 'int' (line 63)
        int_133842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 47), 'int', False)
        # Calling int(args, kwargs) (line 63)
        int_call_result_133845 = invoke(stypy.reporting.localization.Localization(__file__, 63, 47), int_133842, *[x_133843], **kwargs_133844)
        
        list_133853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 47), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 47), list_133853, int_call_result_133845)
        # Obtaining the member '__getitem__' of a type (line 63)
        getitem___133854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), list_133853, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 63)
        subscript_call_result_133855 = invoke(stypy.reporting.localization.Localization(__file__, 63, 8), getitem___133854, int_133841)
        
        # Assigning a type to the variable 'tuple_var_assignment_133812' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'tuple_var_assignment_133812', subscript_call_result_133855)
        
        # Assigning a Subscript to a Name (line 63):
        
        # Obtaining the type of the subscript
        int_133856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 8), 'int')
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to split(...): (line 64)
        # Processing the call arguments (line 64)
        str_133864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 31), 'str', '.')
        # Processing the call keyword arguments (line 64)
        kwargs_133865 = {}
        # Getting the type of 'self' (line 64)
        self_133861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'self', False)
        # Obtaining the member 'version' of a type (line 64)
        version_133862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 12), self_133861, 'version')
        # Obtaining the member 'split' of a type (line 64)
        split_133863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 12), version_133862, 'split')
        # Calling split(args, kwargs) (line 64)
        split_call_result_133866 = invoke(stypy.reporting.localization.Localization(__file__, 64, 12), split_133863, *[str_133864], **kwargs_133865)
        
        comprehension_133867 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 47), split_call_result_133866)
        # Assigning a type to the variable 'x' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 47), 'x', comprehension_133867)
        
        # Call to int(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'x' (line 63)
        x_133858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 51), 'x', False)
        # Processing the call keyword arguments (line 63)
        kwargs_133859 = {}
        # Getting the type of 'int' (line 63)
        int_133857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 47), 'int', False)
        # Calling int(args, kwargs) (line 63)
        int_call_result_133860 = invoke(stypy.reporting.localization.Localization(__file__, 63, 47), int_133857, *[x_133858], **kwargs_133859)
        
        list_133868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 47), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 47), list_133868, int_call_result_133860)
        # Obtaining the member '__getitem__' of a type (line 63)
        getitem___133869 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), list_133868, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 63)
        subscript_call_result_133870 = invoke(stypy.reporting.localization.Localization(__file__, 63, 8), getitem___133869, int_133856)
        
        # Assigning a type to the variable 'tuple_var_assignment_133813' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'tuple_var_assignment_133813', subscript_call_result_133870)
        
        # Assigning a Subscript to a Name (line 63):
        
        # Obtaining the type of the subscript
        int_133871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 8), 'int')
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to split(...): (line 64)
        # Processing the call arguments (line 64)
        str_133879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 31), 'str', '.')
        # Processing the call keyword arguments (line 64)
        kwargs_133880 = {}
        # Getting the type of 'self' (line 64)
        self_133876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'self', False)
        # Obtaining the member 'version' of a type (line 64)
        version_133877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 12), self_133876, 'version')
        # Obtaining the member 'split' of a type (line 64)
        split_133878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 12), version_133877, 'split')
        # Calling split(args, kwargs) (line 64)
        split_call_result_133881 = invoke(stypy.reporting.localization.Localization(__file__, 64, 12), split_133878, *[str_133879], **kwargs_133880)
        
        comprehension_133882 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 47), split_call_result_133881)
        # Assigning a type to the variable 'x' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 47), 'x', comprehension_133882)
        
        # Call to int(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'x' (line 63)
        x_133873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 51), 'x', False)
        # Processing the call keyword arguments (line 63)
        kwargs_133874 = {}
        # Getting the type of 'int' (line 63)
        int_133872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 47), 'int', False)
        # Calling int(args, kwargs) (line 63)
        int_call_result_133875 = invoke(stypy.reporting.localization.Localization(__file__, 63, 47), int_133872, *[x_133873], **kwargs_133874)
        
        list_133883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 47), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 47), list_133883, int_call_result_133875)
        # Obtaining the member '__getitem__' of a type (line 63)
        getitem___133884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), list_133883, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 63)
        subscript_call_result_133885 = invoke(stypy.reporting.localization.Localization(__file__, 63, 8), getitem___133884, int_133871)
        
        # Assigning a type to the variable 'tuple_var_assignment_133814' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'tuple_var_assignment_133814', subscript_call_result_133885)
        
        # Assigning a Name to a Attribute (line 63):
        # Getting the type of 'tuple_var_assignment_133812' (line 63)
        tuple_var_assignment_133812_133886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'tuple_var_assignment_133812')
        # Getting the type of 'self' (line 63)
        self_133887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'self')
        # Setting the type of the member 'major' of a type (line 63)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), self_133887, 'major', tuple_var_assignment_133812_133886)
        
        # Assigning a Name to a Attribute (line 63):
        # Getting the type of 'tuple_var_assignment_133813' (line 63)
        tuple_var_assignment_133813_133888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'tuple_var_assignment_133813')
        # Getting the type of 'self' (line 63)
        self_133889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 20), 'self')
        # Setting the type of the member 'minor' of a type (line 63)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 20), self_133889, 'minor', tuple_var_assignment_133813_133888)
        
        # Assigning a Name to a Attribute (line 63):
        # Getting the type of 'tuple_var_assignment_133814' (line 63)
        tuple_var_assignment_133814_133890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'tuple_var_assignment_133814')
        # Getting the type of 'self' (line 63)
        self_133891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 32), 'self')
        # Setting the type of the member 'bugfix' of a type (line 63)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 32), self_133891, 'bugfix', tuple_var_assignment_133814_133890)
        
        
        
        # Call to len(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'vstring' (line 65)
        vstring_133893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 15), 'vstring', False)
        # Processing the call keyword arguments (line 65)
        kwargs_133894 = {}
        # Getting the type of 'len' (line 65)
        len_133892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 11), 'len', False)
        # Calling len(args, kwargs) (line 65)
        len_call_result_133895 = invoke(stypy.reporting.localization.Localization(__file__, 65, 11), len_133892, *[vstring_133893], **kwargs_133894)
        
        
        # Call to end(...): (line 65)
        # Processing the call keyword arguments (line 65)
        kwargs_133898 = {}
        # Getting the type of 'ver_main' (line 65)
        ver_main_133896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 27), 'ver_main', False)
        # Obtaining the member 'end' of a type (line 65)
        end_133897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 27), ver_main_133896, 'end')
        # Calling end(args, kwargs) (line 65)
        end_call_result_133899 = invoke(stypy.reporting.localization.Localization(__file__, 65, 27), end_133897, *[], **kwargs_133898)
        
        # Applying the binary operator '==' (line 65)
        result_eq_133900 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 11), '==', len_call_result_133895, end_call_result_133899)
        
        # Testing the type of an if condition (line 65)
        if_condition_133901 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 65, 8), result_eq_133900)
        # Assigning a type to the variable 'if_condition_133901' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'if_condition_133901', if_condition_133901)
        # SSA begins for if statement (line 65)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Attribute (line 66):
        
        # Assigning a Str to a Attribute (line 66):
        str_133902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 31), 'str', 'final')
        # Getting the type of 'self' (line 66)
        self_133903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'self')
        # Setting the type of the member 'pre_release' of a type (line 66)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 12), self_133903, 'pre_release', str_133902)
        # SSA branch for the else part of an if statement (line 65)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 68):
        
        # Assigning a Call to a Name (line 68):
        
        # Call to match(...): (line 68)
        # Processing the call arguments (line 68)
        str_133906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 29), 'str', 'a\\d')
        
        # Obtaining the type of the subscript
        
        # Call to end(...): (line 68)
        # Processing the call keyword arguments (line 68)
        kwargs_133909 = {}
        # Getting the type of 'ver_main' (line 68)
        ver_main_133907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 45), 'ver_main', False)
        # Obtaining the member 'end' of a type (line 68)
        end_133908 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 45), ver_main_133907, 'end')
        # Calling end(args, kwargs) (line 68)
        end_call_result_133910 = invoke(stypy.reporting.localization.Localization(__file__, 68, 45), end_133908, *[], **kwargs_133909)
        
        slice_133911 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 68, 37), end_call_result_133910, None, None)
        # Getting the type of 'vstring' (line 68)
        vstring_133912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 37), 'vstring', False)
        # Obtaining the member '__getitem__' of a type (line 68)
        getitem___133913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 37), vstring_133912, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 68)
        subscript_call_result_133914 = invoke(stypy.reporting.localization.Localization(__file__, 68, 37), getitem___133913, slice_133911)
        
        # Processing the call keyword arguments (line 68)
        kwargs_133915 = {}
        # Getting the type of 're' (line 68)
        re_133904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 20), 're', False)
        # Obtaining the member 'match' of a type (line 68)
        match_133905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 20), re_133904, 'match')
        # Calling match(args, kwargs) (line 68)
        match_call_result_133916 = invoke(stypy.reporting.localization.Localization(__file__, 68, 20), match_133905, *[str_133906, subscript_call_result_133914], **kwargs_133915)
        
        # Assigning a type to the variable 'alpha' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'alpha', match_call_result_133916)
        
        # Assigning a Call to a Name (line 69):
        
        # Assigning a Call to a Name (line 69):
        
        # Call to match(...): (line 69)
        # Processing the call arguments (line 69)
        str_133919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 28), 'str', 'b\\d')
        
        # Obtaining the type of the subscript
        
        # Call to end(...): (line 69)
        # Processing the call keyword arguments (line 69)
        kwargs_133922 = {}
        # Getting the type of 'ver_main' (line 69)
        ver_main_133920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 44), 'ver_main', False)
        # Obtaining the member 'end' of a type (line 69)
        end_133921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 44), ver_main_133920, 'end')
        # Calling end(args, kwargs) (line 69)
        end_call_result_133923 = invoke(stypy.reporting.localization.Localization(__file__, 69, 44), end_133921, *[], **kwargs_133922)
        
        slice_133924 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 69, 36), end_call_result_133923, None, None)
        # Getting the type of 'vstring' (line 69)
        vstring_133925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 36), 'vstring', False)
        # Obtaining the member '__getitem__' of a type (line 69)
        getitem___133926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 36), vstring_133925, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 69)
        subscript_call_result_133927 = invoke(stypy.reporting.localization.Localization(__file__, 69, 36), getitem___133926, slice_133924)
        
        # Processing the call keyword arguments (line 69)
        kwargs_133928 = {}
        # Getting the type of 're' (line 69)
        re_133917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 19), 're', False)
        # Obtaining the member 'match' of a type (line 69)
        match_133918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 19), re_133917, 'match')
        # Calling match(args, kwargs) (line 69)
        match_call_result_133929 = invoke(stypy.reporting.localization.Localization(__file__, 69, 19), match_133918, *[str_133919, subscript_call_result_133927], **kwargs_133928)
        
        # Assigning a type to the variable 'beta' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'beta', match_call_result_133929)
        
        # Assigning a Call to a Name (line 70):
        
        # Assigning a Call to a Name (line 70):
        
        # Call to match(...): (line 70)
        # Processing the call arguments (line 70)
        str_133932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 26), 'str', 'rc\\d')
        
        # Obtaining the type of the subscript
        
        # Call to end(...): (line 70)
        # Processing the call keyword arguments (line 70)
        kwargs_133935 = {}
        # Getting the type of 'ver_main' (line 70)
        ver_main_133933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 43), 'ver_main', False)
        # Obtaining the member 'end' of a type (line 70)
        end_133934 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 43), ver_main_133933, 'end')
        # Calling end(args, kwargs) (line 70)
        end_call_result_133936 = invoke(stypy.reporting.localization.Localization(__file__, 70, 43), end_133934, *[], **kwargs_133935)
        
        slice_133937 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 70, 35), end_call_result_133936, None, None)
        # Getting the type of 'vstring' (line 70)
        vstring_133938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 35), 'vstring', False)
        # Obtaining the member '__getitem__' of a type (line 70)
        getitem___133939 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 35), vstring_133938, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 70)
        subscript_call_result_133940 = invoke(stypy.reporting.localization.Localization(__file__, 70, 35), getitem___133939, slice_133937)
        
        # Processing the call keyword arguments (line 70)
        kwargs_133941 = {}
        # Getting the type of 're' (line 70)
        re_133930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 17), 're', False)
        # Obtaining the member 'match' of a type (line 70)
        match_133931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 17), re_133930, 'match')
        # Calling match(args, kwargs) (line 70)
        match_call_result_133942 = invoke(stypy.reporting.localization.Localization(__file__, 70, 17), match_133931, *[str_133932, subscript_call_result_133940], **kwargs_133941)
        
        # Assigning a type to the variable 'rc' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'rc', match_call_result_133942)
        
        # Assigning a ListComp to a Name (line 71):
        
        # Assigning a ListComp to a Name (line 71):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Obtaining an instance of the builtin type 'list' (line 71)
        list_133947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 71)
        # Adding element type (line 71)
        # Getting the type of 'alpha' (line 71)
        alpha_133948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 35), 'alpha')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 34), list_133947, alpha_133948)
        # Adding element type (line 71)
        # Getting the type of 'beta' (line 71)
        beta_133949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 42), 'beta')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 34), list_133947, beta_133949)
        # Adding element type (line 71)
        # Getting the type of 'rc' (line 71)
        rc_133950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 48), 'rc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 34), list_133947, rc_133950)
        
        comprehension_133951 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 23), list_133947)
        # Assigning a type to the variable 'm' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 23), 'm', comprehension_133951)
        
        # Getting the type of 'm' (line 71)
        m_133944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 55), 'm')
        # Getting the type of 'None' (line 71)
        None_133945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 64), 'None')
        # Applying the binary operator 'isnot' (line 71)
        result_is_not_133946 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 55), 'isnot', m_133944, None_133945)
        
        # Getting the type of 'm' (line 71)
        m_133943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 23), 'm')
        list_133952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 23), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 23), list_133952, m_133943)
        # Assigning a type to the variable 'pre_rel' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'pre_rel', list_133952)
        
        # Getting the type of 'pre_rel' (line 72)
        pre_rel_133953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 15), 'pre_rel')
        # Testing the type of an if condition (line 72)
        if_condition_133954 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 12), pre_rel_133953)
        # Assigning a type to the variable 'if_condition_133954' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'if_condition_133954', if_condition_133954)
        # SSA begins for if statement (line 72)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 73):
        
        # Assigning a Call to a Attribute (line 73):
        
        # Call to group(...): (line 73)
        # Processing the call keyword arguments (line 73)
        kwargs_133960 = {}
        
        # Obtaining the type of the subscript
        int_133955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 43), 'int')
        # Getting the type of 'pre_rel' (line 73)
        pre_rel_133956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 35), 'pre_rel', False)
        # Obtaining the member '__getitem__' of a type (line 73)
        getitem___133957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 35), pre_rel_133956, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 73)
        subscript_call_result_133958 = invoke(stypy.reporting.localization.Localization(__file__, 73, 35), getitem___133957, int_133955)
        
        # Obtaining the member 'group' of a type (line 73)
        group_133959 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 35), subscript_call_result_133958, 'group')
        # Calling group(args, kwargs) (line 73)
        group_call_result_133961 = invoke(stypy.reporting.localization.Localization(__file__, 73, 35), group_133959, *[], **kwargs_133960)
        
        # Getting the type of 'self' (line 73)
        self_133962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 16), 'self')
        # Setting the type of the member 'pre_release' of a type (line 73)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 16), self_133962, 'pre_release', group_call_result_133961)
        # SSA branch for the else part of an if statement (line 72)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Attribute (line 75):
        
        # Assigning a Str to a Attribute (line 75):
        str_133963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 35), 'str', '')
        # Getting the type of 'self' (line 75)
        self_133964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 16), 'self')
        # Setting the type of the member 'pre_release' of a type (line 75)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 16), self_133964, 'pre_release', str_133963)
        # SSA join for if statement (line 72)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 65)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 77):
        
        # Assigning a Call to a Attribute (line 77):
        
        # Call to bool(...): (line 77)
        # Processing the call arguments (line 77)
        
        # Call to search(...): (line 77)
        # Processing the call arguments (line 77)
        str_133968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 44), 'str', '.dev')
        # Getting the type of 'vstring' (line 77)
        vstring_133969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 53), 'vstring', False)
        # Processing the call keyword arguments (line 77)
        kwargs_133970 = {}
        # Getting the type of 're' (line 77)
        re_133966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 34), 're', False)
        # Obtaining the member 'search' of a type (line 77)
        search_133967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 34), re_133966, 'search')
        # Calling search(args, kwargs) (line 77)
        search_call_result_133971 = invoke(stypy.reporting.localization.Localization(__file__, 77, 34), search_133967, *[str_133968, vstring_133969], **kwargs_133970)
        
        # Processing the call keyword arguments (line 77)
        kwargs_133972 = {}
        # Getting the type of 'bool' (line 77)
        bool_133965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 29), 'bool', False)
        # Calling bool(args, kwargs) (line 77)
        bool_call_result_133973 = invoke(stypy.reporting.localization.Localization(__file__, 77, 29), bool_133965, *[search_call_result_133971], **kwargs_133972)
        
        # Getting the type of 'self' (line 77)
        self_133974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'self')
        # Setting the type of the member 'is_devversion' of a type (line 77)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 8), self_133974, 'is_devversion', bool_call_result_133973)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def _compare_version(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_compare_version'
        module_type_store = module_type_store.open_function_context('_compare_version', 79, 4, False)
        # Assigning a type to the variable 'self' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NumpyVersion._compare_version.__dict__.__setitem__('stypy_localization', localization)
        NumpyVersion._compare_version.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NumpyVersion._compare_version.__dict__.__setitem__('stypy_type_store', module_type_store)
        NumpyVersion._compare_version.__dict__.__setitem__('stypy_function_name', 'NumpyVersion._compare_version')
        NumpyVersion._compare_version.__dict__.__setitem__('stypy_param_names_list', ['other'])
        NumpyVersion._compare_version.__dict__.__setitem__('stypy_varargs_param_name', None)
        NumpyVersion._compare_version.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NumpyVersion._compare_version.__dict__.__setitem__('stypy_call_defaults', defaults)
        NumpyVersion._compare_version.__dict__.__setitem__('stypy_call_varargs', varargs)
        NumpyVersion._compare_version.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NumpyVersion._compare_version.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NumpyVersion._compare_version', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_compare_version', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_compare_version(...)' code ##################

        str_133975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 8), 'str', 'Compare major.minor.bugfix')
        
        
        # Getting the type of 'self' (line 81)
        self_133976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 11), 'self')
        # Obtaining the member 'major' of a type (line 81)
        major_133977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 11), self_133976, 'major')
        # Getting the type of 'other' (line 81)
        other_133978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 25), 'other')
        # Obtaining the member 'major' of a type (line 81)
        major_133979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 25), other_133978, 'major')
        # Applying the binary operator '==' (line 81)
        result_eq_133980 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 11), '==', major_133977, major_133979)
        
        # Testing the type of an if condition (line 81)
        if_condition_133981 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 8), result_eq_133980)
        # Assigning a type to the variable 'if_condition_133981' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'if_condition_133981', if_condition_133981)
        # SSA begins for if statement (line 81)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'self' (line 82)
        self_133982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), 'self')
        # Obtaining the member 'minor' of a type (line 82)
        minor_133983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 15), self_133982, 'minor')
        # Getting the type of 'other' (line 82)
        other_133984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 29), 'other')
        # Obtaining the member 'minor' of a type (line 82)
        minor_133985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 29), other_133984, 'minor')
        # Applying the binary operator '==' (line 82)
        result_eq_133986 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 15), '==', minor_133983, minor_133985)
        
        # Testing the type of an if condition (line 82)
        if_condition_133987 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 12), result_eq_133986)
        # Assigning a type to the variable 'if_condition_133987' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'if_condition_133987', if_condition_133987)
        # SSA begins for if statement (line 82)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'self' (line 83)
        self_133988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 19), 'self')
        # Obtaining the member 'bugfix' of a type (line 83)
        bugfix_133989 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 19), self_133988, 'bugfix')
        # Getting the type of 'other' (line 83)
        other_133990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 34), 'other')
        # Obtaining the member 'bugfix' of a type (line 83)
        bugfix_133991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 34), other_133990, 'bugfix')
        # Applying the binary operator '==' (line 83)
        result_eq_133992 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 19), '==', bugfix_133989, bugfix_133991)
        
        # Testing the type of an if condition (line 83)
        if_condition_133993 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 83, 16), result_eq_133992)
        # Assigning a type to the variable 'if_condition_133993' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 16), 'if_condition_133993', if_condition_133993)
        # SSA begins for if statement (line 83)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 84):
        
        # Assigning a Num to a Name (line 84):
        int_133994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 29), 'int')
        # Assigning a type to the variable 'vercmp' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 20), 'vercmp', int_133994)
        # SSA branch for the else part of an if statement (line 83)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 85)
        self_133995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 21), 'self')
        # Obtaining the member 'bugfix' of a type (line 85)
        bugfix_133996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 21), self_133995, 'bugfix')
        # Getting the type of 'other' (line 85)
        other_133997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 35), 'other')
        # Obtaining the member 'bugfix' of a type (line 85)
        bugfix_133998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 35), other_133997, 'bugfix')
        # Applying the binary operator '>' (line 85)
        result_gt_133999 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 21), '>', bugfix_133996, bugfix_133998)
        
        # Testing the type of an if condition (line 85)
        if_condition_134000 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 85, 21), result_gt_133999)
        # Assigning a type to the variable 'if_condition_134000' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 21), 'if_condition_134000', if_condition_134000)
        # SSA begins for if statement (line 85)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 86):
        
        # Assigning a Num to a Name (line 86):
        int_134001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 29), 'int')
        # Assigning a type to the variable 'vercmp' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 20), 'vercmp', int_134001)
        # SSA branch for the else part of an if statement (line 85)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 88):
        
        # Assigning a Num to a Name (line 88):
        int_134002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 29), 'int')
        # Assigning a type to the variable 'vercmp' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 20), 'vercmp', int_134002)
        # SSA join for if statement (line 85)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 83)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 82)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 89)
        self_134003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 17), 'self')
        # Obtaining the member 'minor' of a type (line 89)
        minor_134004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 17), self_134003, 'minor')
        # Getting the type of 'other' (line 89)
        other_134005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 30), 'other')
        # Obtaining the member 'minor' of a type (line 89)
        minor_134006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 30), other_134005, 'minor')
        # Applying the binary operator '>' (line 89)
        result_gt_134007 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 17), '>', minor_134004, minor_134006)
        
        # Testing the type of an if condition (line 89)
        if_condition_134008 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 17), result_gt_134007)
        # Assigning a type to the variable 'if_condition_134008' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 17), 'if_condition_134008', if_condition_134008)
        # SSA begins for if statement (line 89)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 90):
        
        # Assigning a Num to a Name (line 90):
        int_134009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 25), 'int')
        # Assigning a type to the variable 'vercmp' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 16), 'vercmp', int_134009)
        # SSA branch for the else part of an if statement (line 89)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 92):
        
        # Assigning a Num to a Name (line 92):
        int_134010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 25), 'int')
        # Assigning a type to the variable 'vercmp' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 16), 'vercmp', int_134010)
        # SSA join for if statement (line 89)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 82)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 81)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 93)
        self_134011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 13), 'self')
        # Obtaining the member 'major' of a type (line 93)
        major_134012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 13), self_134011, 'major')
        # Getting the type of 'other' (line 93)
        other_134013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 26), 'other')
        # Obtaining the member 'major' of a type (line 93)
        major_134014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 26), other_134013, 'major')
        # Applying the binary operator '>' (line 93)
        result_gt_134015 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 13), '>', major_134012, major_134014)
        
        # Testing the type of an if condition (line 93)
        if_condition_134016 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 93, 13), result_gt_134015)
        # Assigning a type to the variable 'if_condition_134016' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 13), 'if_condition_134016', if_condition_134016)
        # SSA begins for if statement (line 93)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 94):
        
        # Assigning a Num to a Name (line 94):
        int_134017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 21), 'int')
        # Assigning a type to the variable 'vercmp' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'vercmp', int_134017)
        # SSA branch for the else part of an if statement (line 93)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 96):
        
        # Assigning a Num to a Name (line 96):
        int_134018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 21), 'int')
        # Assigning a type to the variable 'vercmp' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'vercmp', int_134018)
        # SSA join for if statement (line 93)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 81)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'vercmp' (line 98)
        vercmp_134019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 15), 'vercmp')
        # Assigning a type to the variable 'stypy_return_type' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'stypy_return_type', vercmp_134019)
        
        # ################# End of '_compare_version(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_compare_version' in the type store
        # Getting the type of 'stypy_return_type' (line 79)
        stypy_return_type_134020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134020)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_compare_version'
        return stypy_return_type_134020


    @norecursion
    def _compare_pre_release(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_compare_pre_release'
        module_type_store = module_type_store.open_function_context('_compare_pre_release', 100, 4, False)
        # Assigning a type to the variable 'self' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NumpyVersion._compare_pre_release.__dict__.__setitem__('stypy_localization', localization)
        NumpyVersion._compare_pre_release.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NumpyVersion._compare_pre_release.__dict__.__setitem__('stypy_type_store', module_type_store)
        NumpyVersion._compare_pre_release.__dict__.__setitem__('stypy_function_name', 'NumpyVersion._compare_pre_release')
        NumpyVersion._compare_pre_release.__dict__.__setitem__('stypy_param_names_list', ['other'])
        NumpyVersion._compare_pre_release.__dict__.__setitem__('stypy_varargs_param_name', None)
        NumpyVersion._compare_pre_release.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NumpyVersion._compare_pre_release.__dict__.__setitem__('stypy_call_defaults', defaults)
        NumpyVersion._compare_pre_release.__dict__.__setitem__('stypy_call_varargs', varargs)
        NumpyVersion._compare_pre_release.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NumpyVersion._compare_pre_release.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NumpyVersion._compare_pre_release', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_compare_pre_release', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_compare_pre_release(...)' code ##################

        str_134021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 8), 'str', 'Compare alpha/beta/rc/final.')
        
        
        # Getting the type of 'self' (line 102)
        self_134022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 11), 'self')
        # Obtaining the member 'pre_release' of a type (line 102)
        pre_release_134023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 11), self_134022, 'pre_release')
        # Getting the type of 'other' (line 102)
        other_134024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 31), 'other')
        # Obtaining the member 'pre_release' of a type (line 102)
        pre_release_134025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 31), other_134024, 'pre_release')
        # Applying the binary operator '==' (line 102)
        result_eq_134026 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 11), '==', pre_release_134023, pre_release_134025)
        
        # Testing the type of an if condition (line 102)
        if_condition_134027 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 102, 8), result_eq_134026)
        # Assigning a type to the variable 'if_condition_134027' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'if_condition_134027', if_condition_134027)
        # SSA begins for if statement (line 102)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 103):
        
        # Assigning a Num to a Name (line 103):
        int_134028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 21), 'int')
        # Assigning a type to the variable 'vercmp' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'vercmp', int_134028)
        # SSA branch for the else part of an if statement (line 102)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 104)
        self_134029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 13), 'self')
        # Obtaining the member 'pre_release' of a type (line 104)
        pre_release_134030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 13), self_134029, 'pre_release')
        str_134031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 33), 'str', 'final')
        # Applying the binary operator '==' (line 104)
        result_eq_134032 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 13), '==', pre_release_134030, str_134031)
        
        # Testing the type of an if condition (line 104)
        if_condition_134033 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 104, 13), result_eq_134032)
        # Assigning a type to the variable 'if_condition_134033' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 13), 'if_condition_134033', if_condition_134033)
        # SSA begins for if statement (line 104)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 105):
        
        # Assigning a Num to a Name (line 105):
        int_134034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 21), 'int')
        # Assigning a type to the variable 'vercmp' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'vercmp', int_134034)
        # SSA branch for the else part of an if statement (line 104)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'other' (line 106)
        other_134035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 13), 'other')
        # Obtaining the member 'pre_release' of a type (line 106)
        pre_release_134036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 13), other_134035, 'pre_release')
        str_134037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 34), 'str', 'final')
        # Applying the binary operator '==' (line 106)
        result_eq_134038 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 13), '==', pre_release_134036, str_134037)
        
        # Testing the type of an if condition (line 106)
        if_condition_134039 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 13), result_eq_134038)
        # Assigning a type to the variable 'if_condition_134039' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 13), 'if_condition_134039', if_condition_134039)
        # SSA begins for if statement (line 106)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 107):
        
        # Assigning a Num to a Name (line 107):
        int_134040 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 21), 'int')
        # Assigning a type to the variable 'vercmp' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'vercmp', int_134040)
        # SSA branch for the else part of an if statement (line 106)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 108)
        self_134041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'self')
        # Obtaining the member 'pre_release' of a type (line 108)
        pre_release_134042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 13), self_134041, 'pre_release')
        # Getting the type of 'other' (line 108)
        other_134043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 32), 'other')
        # Obtaining the member 'pre_release' of a type (line 108)
        pre_release_134044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 32), other_134043, 'pre_release')
        # Applying the binary operator '>' (line 108)
        result_gt_134045 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 13), '>', pre_release_134042, pre_release_134044)
        
        # Testing the type of an if condition (line 108)
        if_condition_134046 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 13), result_gt_134045)
        # Assigning a type to the variable 'if_condition_134046' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 13), 'if_condition_134046', if_condition_134046)
        # SSA begins for if statement (line 108)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 109):
        
        # Assigning a Num to a Name (line 109):
        int_134047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 21), 'int')
        # Assigning a type to the variable 'vercmp' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'vercmp', int_134047)
        # SSA branch for the else part of an if statement (line 108)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 111):
        
        # Assigning a Num to a Name (line 111):
        int_134048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 21), 'int')
        # Assigning a type to the variable 'vercmp' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'vercmp', int_134048)
        # SSA join for if statement (line 108)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 106)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 104)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 102)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'vercmp' (line 113)
        vercmp_134049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 15), 'vercmp')
        # Assigning a type to the variable 'stypy_return_type' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'stypy_return_type', vercmp_134049)
        
        # ################# End of '_compare_pre_release(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_compare_pre_release' in the type store
        # Getting the type of 'stypy_return_type' (line 100)
        stypy_return_type_134050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134050)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_compare_pre_release'
        return stypy_return_type_134050


    @norecursion
    def _compare(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_compare'
        module_type_store = module_type_store.open_function_context('_compare', 115, 4, False)
        # Assigning a type to the variable 'self' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NumpyVersion._compare.__dict__.__setitem__('stypy_localization', localization)
        NumpyVersion._compare.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NumpyVersion._compare.__dict__.__setitem__('stypy_type_store', module_type_store)
        NumpyVersion._compare.__dict__.__setitem__('stypy_function_name', 'NumpyVersion._compare')
        NumpyVersion._compare.__dict__.__setitem__('stypy_param_names_list', ['other'])
        NumpyVersion._compare.__dict__.__setitem__('stypy_varargs_param_name', None)
        NumpyVersion._compare.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NumpyVersion._compare.__dict__.__setitem__('stypy_call_defaults', defaults)
        NumpyVersion._compare.__dict__.__setitem__('stypy_call_varargs', varargs)
        NumpyVersion._compare.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NumpyVersion._compare.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NumpyVersion._compare', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_compare', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_compare(...)' code ##################

        
        
        
        # Call to isinstance(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'other' (line 116)
        other_134052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 26), 'other', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 116)
        tuple_134053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 116)
        # Adding element type (line 116)
        # Getting the type of 'basestring' (line 116)
        basestring_134054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 34), 'basestring', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 34), tuple_134053, basestring_134054)
        # Adding element type (line 116)
        # Getting the type of 'NumpyVersion' (line 116)
        NumpyVersion_134055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 46), 'NumpyVersion', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 34), tuple_134053, NumpyVersion_134055)
        
        # Processing the call keyword arguments (line 116)
        kwargs_134056 = {}
        # Getting the type of 'isinstance' (line 116)
        isinstance_134051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 116)
        isinstance_call_result_134057 = invoke(stypy.reporting.localization.Localization(__file__, 116, 15), isinstance_134051, *[other_134052, tuple_134053], **kwargs_134056)
        
        # Applying the 'not' unary operator (line 116)
        result_not__134058 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 11), 'not', isinstance_call_result_134057)
        
        # Testing the type of an if condition (line 116)
        if_condition_134059 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 116, 8), result_not__134058)
        # Assigning a type to the variable 'if_condition_134059' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'if_condition_134059', if_condition_134059)
        # SSA begins for if statement (line 116)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 117)
        # Processing the call arguments (line 117)
        str_134061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 29), 'str', 'Invalid object to compare with NumpyVersion.')
        # Processing the call keyword arguments (line 117)
        kwargs_134062 = {}
        # Getting the type of 'ValueError' (line 117)
        ValueError_134060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 117)
        ValueError_call_result_134063 = invoke(stypy.reporting.localization.Localization(__file__, 117, 18), ValueError_134060, *[str_134061], **kwargs_134062)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 117, 12), ValueError_call_result_134063, 'raise parameter', BaseException)
        # SSA join for if statement (line 116)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 119)
        # Getting the type of 'basestring' (line 119)
        basestring_134064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 29), 'basestring')
        # Getting the type of 'other' (line 119)
        other_134065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 22), 'other')
        
        (may_be_134066, more_types_in_union_134067) = may_be_subtype(basestring_134064, other_134065)

        if may_be_134066:

            if more_types_in_union_134067:
                # Runtime conditional SSA (line 119)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'other' (line 119)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'other', remove_not_subtype_from_union(other_134065, basestring))
            
            # Assigning a Call to a Name (line 120):
            
            # Assigning a Call to a Name (line 120):
            
            # Call to NumpyVersion(...): (line 120)
            # Processing the call arguments (line 120)
            # Getting the type of 'other' (line 120)
            other_134069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 33), 'other', False)
            # Processing the call keyword arguments (line 120)
            kwargs_134070 = {}
            # Getting the type of 'NumpyVersion' (line 120)
            NumpyVersion_134068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 20), 'NumpyVersion', False)
            # Calling NumpyVersion(args, kwargs) (line 120)
            NumpyVersion_call_result_134071 = invoke(stypy.reporting.localization.Localization(__file__, 120, 20), NumpyVersion_134068, *[other_134069], **kwargs_134070)
            
            # Assigning a type to the variable 'other' (line 120)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'other', NumpyVersion_call_result_134071)

            if more_types_in_union_134067:
                # SSA join for if statement (line 119)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 122):
        
        # Assigning a Call to a Name (line 122):
        
        # Call to _compare_version(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'other' (line 122)
        other_134074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 39), 'other', False)
        # Processing the call keyword arguments (line 122)
        kwargs_134075 = {}
        # Getting the type of 'self' (line 122)
        self_134072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 17), 'self', False)
        # Obtaining the member '_compare_version' of a type (line 122)
        _compare_version_134073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 17), self_134072, '_compare_version')
        # Calling _compare_version(args, kwargs) (line 122)
        _compare_version_call_result_134076 = invoke(stypy.reporting.localization.Localization(__file__, 122, 17), _compare_version_134073, *[other_134074], **kwargs_134075)
        
        # Assigning a type to the variable 'vercmp' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'vercmp', _compare_version_call_result_134076)
        
        
        # Getting the type of 'vercmp' (line 123)
        vercmp_134077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 11), 'vercmp')
        int_134078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 21), 'int')
        # Applying the binary operator '==' (line 123)
        result_eq_134079 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 11), '==', vercmp_134077, int_134078)
        
        # Testing the type of an if condition (line 123)
        if_condition_134080 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 123, 8), result_eq_134079)
        # Assigning a type to the variable 'if_condition_134080' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'if_condition_134080', if_condition_134080)
        # SSA begins for if statement (line 123)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 125):
        
        # Assigning a Call to a Name (line 125):
        
        # Call to _compare_pre_release(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'other' (line 125)
        other_134083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 47), 'other', False)
        # Processing the call keyword arguments (line 125)
        kwargs_134084 = {}
        # Getting the type of 'self' (line 125)
        self_134081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 21), 'self', False)
        # Obtaining the member '_compare_pre_release' of a type (line 125)
        _compare_pre_release_134082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 21), self_134081, '_compare_pre_release')
        # Calling _compare_pre_release(args, kwargs) (line 125)
        _compare_pre_release_call_result_134085 = invoke(stypy.reporting.localization.Localization(__file__, 125, 21), _compare_pre_release_134082, *[other_134083], **kwargs_134084)
        
        # Assigning a type to the variable 'vercmp' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'vercmp', _compare_pre_release_call_result_134085)
        
        
        # Getting the type of 'vercmp' (line 126)
        vercmp_134086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 15), 'vercmp')
        int_134087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 25), 'int')
        # Applying the binary operator '==' (line 126)
        result_eq_134088 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 15), '==', vercmp_134086, int_134087)
        
        # Testing the type of an if condition (line 126)
        if_condition_134089 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 126, 12), result_eq_134088)
        # Assigning a type to the variable 'if_condition_134089' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'if_condition_134089', if_condition_134089)
        # SSA begins for if statement (line 126)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'self' (line 128)
        self_134090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 19), 'self')
        # Obtaining the member 'is_devversion' of a type (line 128)
        is_devversion_134091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 19), self_134090, 'is_devversion')
        # Getting the type of 'other' (line 128)
        other_134092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 41), 'other')
        # Obtaining the member 'is_devversion' of a type (line 128)
        is_devversion_134093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 41), other_134092, 'is_devversion')
        # Applying the binary operator 'is' (line 128)
        result_is__134094 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 19), 'is', is_devversion_134091, is_devversion_134093)
        
        # Testing the type of an if condition (line 128)
        if_condition_134095 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 128, 16), result_is__134094)
        # Assigning a type to the variable 'if_condition_134095' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'if_condition_134095', if_condition_134095)
        # SSA begins for if statement (line 128)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 129):
        
        # Assigning a Num to a Name (line 129):
        int_134096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 29), 'int')
        # Assigning a type to the variable 'vercmp' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 20), 'vercmp', int_134096)
        # SSA branch for the else part of an if statement (line 128)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'self' (line 130)
        self_134097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 21), 'self')
        # Obtaining the member 'is_devversion' of a type (line 130)
        is_devversion_134098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 21), self_134097, 'is_devversion')
        # Testing the type of an if condition (line 130)
        if_condition_134099 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 21), is_devversion_134098)
        # Assigning a type to the variable 'if_condition_134099' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 21), 'if_condition_134099', if_condition_134099)
        # SSA begins for if statement (line 130)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 131):
        
        # Assigning a Num to a Name (line 131):
        int_134100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 29), 'int')
        # Assigning a type to the variable 'vercmp' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 20), 'vercmp', int_134100)
        # SSA branch for the else part of an if statement (line 130)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 133):
        
        # Assigning a Num to a Name (line 133):
        int_134101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 29), 'int')
        # Assigning a type to the variable 'vercmp' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 20), 'vercmp', int_134101)
        # SSA join for if statement (line 130)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 128)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 126)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 123)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'vercmp' (line 135)
        vercmp_134102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 15), 'vercmp')
        # Assigning a type to the variable 'stypy_return_type' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'stypy_return_type', vercmp_134102)
        
        # ################# End of '_compare(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_compare' in the type store
        # Getting the type of 'stypy_return_type' (line 115)
        stypy_return_type_134103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134103)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_compare'
        return stypy_return_type_134103


    @norecursion
    def __lt__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__lt__'
        module_type_store = module_type_store.open_function_context('__lt__', 137, 4, False)
        # Assigning a type to the variable 'self' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NumpyVersion.__lt__.__dict__.__setitem__('stypy_localization', localization)
        NumpyVersion.__lt__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NumpyVersion.__lt__.__dict__.__setitem__('stypy_type_store', module_type_store)
        NumpyVersion.__lt__.__dict__.__setitem__('stypy_function_name', 'NumpyVersion.__lt__')
        NumpyVersion.__lt__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        NumpyVersion.__lt__.__dict__.__setitem__('stypy_varargs_param_name', None)
        NumpyVersion.__lt__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NumpyVersion.__lt__.__dict__.__setitem__('stypy_call_defaults', defaults)
        NumpyVersion.__lt__.__dict__.__setitem__('stypy_call_varargs', varargs)
        NumpyVersion.__lt__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NumpyVersion.__lt__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NumpyVersion.__lt__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__lt__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__lt__(...)' code ##################

        
        
        # Call to _compare(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'other' (line 138)
        other_134106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 29), 'other', False)
        # Processing the call keyword arguments (line 138)
        kwargs_134107 = {}
        # Getting the type of 'self' (line 138)
        self_134104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 15), 'self', False)
        # Obtaining the member '_compare' of a type (line 138)
        _compare_134105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 15), self_134104, '_compare')
        # Calling _compare(args, kwargs) (line 138)
        _compare_call_result_134108 = invoke(stypy.reporting.localization.Localization(__file__, 138, 15), _compare_134105, *[other_134106], **kwargs_134107)
        
        int_134109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 38), 'int')
        # Applying the binary operator '<' (line 138)
        result_lt_134110 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 15), '<', _compare_call_result_134108, int_134109)
        
        # Assigning a type to the variable 'stypy_return_type' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'stypy_return_type', result_lt_134110)
        
        # ################# End of '__lt__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__lt__' in the type store
        # Getting the type of 'stypy_return_type' (line 137)
        stypy_return_type_134111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134111)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__lt__'
        return stypy_return_type_134111


    @norecursion
    def __le__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__le__'
        module_type_store = module_type_store.open_function_context('__le__', 140, 4, False)
        # Assigning a type to the variable 'self' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NumpyVersion.__le__.__dict__.__setitem__('stypy_localization', localization)
        NumpyVersion.__le__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NumpyVersion.__le__.__dict__.__setitem__('stypy_type_store', module_type_store)
        NumpyVersion.__le__.__dict__.__setitem__('stypy_function_name', 'NumpyVersion.__le__')
        NumpyVersion.__le__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        NumpyVersion.__le__.__dict__.__setitem__('stypy_varargs_param_name', None)
        NumpyVersion.__le__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NumpyVersion.__le__.__dict__.__setitem__('stypy_call_defaults', defaults)
        NumpyVersion.__le__.__dict__.__setitem__('stypy_call_varargs', varargs)
        NumpyVersion.__le__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NumpyVersion.__le__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NumpyVersion.__le__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__le__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__le__(...)' code ##################

        
        
        # Call to _compare(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'other' (line 141)
        other_134114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 29), 'other', False)
        # Processing the call keyword arguments (line 141)
        kwargs_134115 = {}
        # Getting the type of 'self' (line 141)
        self_134112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 15), 'self', False)
        # Obtaining the member '_compare' of a type (line 141)
        _compare_134113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 15), self_134112, '_compare')
        # Calling _compare(args, kwargs) (line 141)
        _compare_call_result_134116 = invoke(stypy.reporting.localization.Localization(__file__, 141, 15), _compare_134113, *[other_134114], **kwargs_134115)
        
        int_134117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 39), 'int')
        # Applying the binary operator '<=' (line 141)
        result_le_134118 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 15), '<=', _compare_call_result_134116, int_134117)
        
        # Assigning a type to the variable 'stypy_return_type' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'stypy_return_type', result_le_134118)
        
        # ################# End of '__le__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__le__' in the type store
        # Getting the type of 'stypy_return_type' (line 140)
        stypy_return_type_134119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134119)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__le__'
        return stypy_return_type_134119


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 143, 4, False)
        # Assigning a type to the variable 'self' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NumpyVersion.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        NumpyVersion.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NumpyVersion.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        NumpyVersion.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'NumpyVersion.__eq__')
        NumpyVersion.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        NumpyVersion.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        NumpyVersion.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NumpyVersion.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        NumpyVersion.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        NumpyVersion.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NumpyVersion.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NumpyVersion.__eq__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__eq__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__eq__(...)' code ##################

        
        
        # Call to _compare(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'other' (line 144)
        other_134122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 29), 'other', False)
        # Processing the call keyword arguments (line 144)
        kwargs_134123 = {}
        # Getting the type of 'self' (line 144)
        self_134120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 15), 'self', False)
        # Obtaining the member '_compare' of a type (line 144)
        _compare_134121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 15), self_134120, '_compare')
        # Calling _compare(args, kwargs) (line 144)
        _compare_call_result_134124 = invoke(stypy.reporting.localization.Localization(__file__, 144, 15), _compare_134121, *[other_134122], **kwargs_134123)
        
        int_134125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 39), 'int')
        # Applying the binary operator '==' (line 144)
        result_eq_134126 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 15), '==', _compare_call_result_134124, int_134125)
        
        # Assigning a type to the variable 'stypy_return_type' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'stypy_return_type', result_eq_134126)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 143)
        stypy_return_type_134127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134127)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_134127


    @norecursion
    def __ne__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__ne__'
        module_type_store = module_type_store.open_function_context('__ne__', 146, 4, False)
        # Assigning a type to the variable 'self' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NumpyVersion.__ne__.__dict__.__setitem__('stypy_localization', localization)
        NumpyVersion.__ne__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NumpyVersion.__ne__.__dict__.__setitem__('stypy_type_store', module_type_store)
        NumpyVersion.__ne__.__dict__.__setitem__('stypy_function_name', 'NumpyVersion.__ne__')
        NumpyVersion.__ne__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        NumpyVersion.__ne__.__dict__.__setitem__('stypy_varargs_param_name', None)
        NumpyVersion.__ne__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NumpyVersion.__ne__.__dict__.__setitem__('stypy_call_defaults', defaults)
        NumpyVersion.__ne__.__dict__.__setitem__('stypy_call_varargs', varargs)
        NumpyVersion.__ne__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NumpyVersion.__ne__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NumpyVersion.__ne__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__ne__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__ne__(...)' code ##################

        
        
        # Call to _compare(...): (line 147)
        # Processing the call arguments (line 147)
        # Getting the type of 'other' (line 147)
        other_134130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 29), 'other', False)
        # Processing the call keyword arguments (line 147)
        kwargs_134131 = {}
        # Getting the type of 'self' (line 147)
        self_134128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 15), 'self', False)
        # Obtaining the member '_compare' of a type (line 147)
        _compare_134129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 15), self_134128, '_compare')
        # Calling _compare(args, kwargs) (line 147)
        _compare_call_result_134132 = invoke(stypy.reporting.localization.Localization(__file__, 147, 15), _compare_134129, *[other_134130], **kwargs_134131)
        
        int_134133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 39), 'int')
        # Applying the binary operator '!=' (line 147)
        result_ne_134134 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 15), '!=', _compare_call_result_134132, int_134133)
        
        # Assigning a type to the variable 'stypy_return_type' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'stypy_return_type', result_ne_134134)
        
        # ################# End of '__ne__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__ne__' in the type store
        # Getting the type of 'stypy_return_type' (line 146)
        stypy_return_type_134135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134135)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__ne__'
        return stypy_return_type_134135


    @norecursion
    def __gt__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__gt__'
        module_type_store = module_type_store.open_function_context('__gt__', 149, 4, False)
        # Assigning a type to the variable 'self' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NumpyVersion.__gt__.__dict__.__setitem__('stypy_localization', localization)
        NumpyVersion.__gt__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NumpyVersion.__gt__.__dict__.__setitem__('stypy_type_store', module_type_store)
        NumpyVersion.__gt__.__dict__.__setitem__('stypy_function_name', 'NumpyVersion.__gt__')
        NumpyVersion.__gt__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        NumpyVersion.__gt__.__dict__.__setitem__('stypy_varargs_param_name', None)
        NumpyVersion.__gt__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NumpyVersion.__gt__.__dict__.__setitem__('stypy_call_defaults', defaults)
        NumpyVersion.__gt__.__dict__.__setitem__('stypy_call_varargs', varargs)
        NumpyVersion.__gt__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NumpyVersion.__gt__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NumpyVersion.__gt__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__gt__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__gt__(...)' code ##################

        
        
        # Call to _compare(...): (line 150)
        # Processing the call arguments (line 150)
        # Getting the type of 'other' (line 150)
        other_134138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 29), 'other', False)
        # Processing the call keyword arguments (line 150)
        kwargs_134139 = {}
        # Getting the type of 'self' (line 150)
        self_134136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 15), 'self', False)
        # Obtaining the member '_compare' of a type (line 150)
        _compare_134137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 15), self_134136, '_compare')
        # Calling _compare(args, kwargs) (line 150)
        _compare_call_result_134140 = invoke(stypy.reporting.localization.Localization(__file__, 150, 15), _compare_134137, *[other_134138], **kwargs_134139)
        
        int_134141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 38), 'int')
        # Applying the binary operator '>' (line 150)
        result_gt_134142 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 15), '>', _compare_call_result_134140, int_134141)
        
        # Assigning a type to the variable 'stypy_return_type' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'stypy_return_type', result_gt_134142)
        
        # ################# End of '__gt__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__gt__' in the type store
        # Getting the type of 'stypy_return_type' (line 149)
        stypy_return_type_134143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134143)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__gt__'
        return stypy_return_type_134143


    @norecursion
    def __ge__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__ge__'
        module_type_store = module_type_store.open_function_context('__ge__', 152, 4, False)
        # Assigning a type to the variable 'self' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NumpyVersion.__ge__.__dict__.__setitem__('stypy_localization', localization)
        NumpyVersion.__ge__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NumpyVersion.__ge__.__dict__.__setitem__('stypy_type_store', module_type_store)
        NumpyVersion.__ge__.__dict__.__setitem__('stypy_function_name', 'NumpyVersion.__ge__')
        NumpyVersion.__ge__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        NumpyVersion.__ge__.__dict__.__setitem__('stypy_varargs_param_name', None)
        NumpyVersion.__ge__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NumpyVersion.__ge__.__dict__.__setitem__('stypy_call_defaults', defaults)
        NumpyVersion.__ge__.__dict__.__setitem__('stypy_call_varargs', varargs)
        NumpyVersion.__ge__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NumpyVersion.__ge__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NumpyVersion.__ge__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__ge__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__ge__(...)' code ##################

        
        
        # Call to _compare(...): (line 153)
        # Processing the call arguments (line 153)
        # Getting the type of 'other' (line 153)
        other_134146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 29), 'other', False)
        # Processing the call keyword arguments (line 153)
        kwargs_134147 = {}
        # Getting the type of 'self' (line 153)
        self_134144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 15), 'self', False)
        # Obtaining the member '_compare' of a type (line 153)
        _compare_134145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 15), self_134144, '_compare')
        # Calling _compare(args, kwargs) (line 153)
        _compare_call_result_134148 = invoke(stypy.reporting.localization.Localization(__file__, 153, 15), _compare_134145, *[other_134146], **kwargs_134147)
        
        int_134149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 39), 'int')
        # Applying the binary operator '>=' (line 153)
        result_ge_134150 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 15), '>=', _compare_call_result_134148, int_134149)
        
        # Assigning a type to the variable 'stypy_return_type' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'stypy_return_type', result_ge_134150)
        
        # ################# End of '__ge__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__ge__' in the type store
        # Getting the type of 'stypy_return_type' (line 152)
        stypy_return_type_134151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134151)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__ge__'
        return stypy_return_type_134151


    @norecursion
    def __repr(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr'
        module_type_store = module_type_store.open_function_context('__repr', 155, 4, False)
        # Assigning a type to the variable 'self' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NumpyVersion.__repr.__dict__.__setitem__('stypy_localization', localization)
        NumpyVersion.__repr.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NumpyVersion.__repr.__dict__.__setitem__('stypy_type_store', module_type_store)
        NumpyVersion.__repr.__dict__.__setitem__('stypy_function_name', 'NumpyVersion.__repr')
        NumpyVersion.__repr.__dict__.__setitem__('stypy_param_names_list', [])
        NumpyVersion.__repr.__dict__.__setitem__('stypy_varargs_param_name', None)
        NumpyVersion.__repr.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NumpyVersion.__repr.__dict__.__setitem__('stypy_call_defaults', defaults)
        NumpyVersion.__repr.__dict__.__setitem__('stypy_call_varargs', varargs)
        NumpyVersion.__repr.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NumpyVersion.__repr.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NumpyVersion.__repr', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__repr', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__repr(...)' code ##################

        str_134152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 15), 'str', 'NumpyVersion(%s)')
        # Getting the type of 'self' (line 156)
        self_134153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 36), 'self')
        # Obtaining the member 'vstring' of a type (line 156)
        vstring_134154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 36), self_134153, 'vstring')
        # Applying the binary operator '%' (line 156)
        result_mod_134155 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 15), '%', str_134152, vstring_134154)
        
        # Assigning a type to the variable 'stypy_return_type' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'stypy_return_type', result_mod_134155)
        
        # ################# End of '__repr(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr' in the type store
        # Getting the type of 'stypy_return_type' (line 155)
        stypy_return_type_134156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_134156)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr'
        return stypy_return_type_134156


# Assigning a type to the variable 'NumpyVersion' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'NumpyVersion', NumpyVersion)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
