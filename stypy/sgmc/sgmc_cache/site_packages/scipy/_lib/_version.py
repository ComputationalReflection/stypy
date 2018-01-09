
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
8: 
9: import re
10: 
11: from scipy._lib.six import string_types
12: 
13: 
14: __all__ = ['NumpyVersion']
15: 
16: 
17: class NumpyVersion():
18:     '''Parse and compare numpy version strings.
19: 
20:     Numpy has the following versioning scheme (numbers given are examples; they
21:     can be >9) in principle):
22: 
23:     - Released version: '1.8.0', '1.8.1', etc.
24:     - Alpha: '1.8.0a1', '1.8.0a2', etc.
25:     - Beta: '1.8.0b1', '1.8.0b2', etc.
26:     - Release candidates: '1.8.0rc1', '1.8.0rc2', etc.
27:     - Development versions: '1.8.0.dev-f1234afa' (git commit hash appended)
28:     - Development versions after a1: '1.8.0a1.dev-f1234afa',
29:                                      '1.8.0b2.dev-f1234afa',
30:                                      '1.8.1rc1.dev-f1234afa', etc.
31:     - Development versions (no git hash available): '1.8.0.dev-Unknown'
32: 
33:     Comparing needs to be done against a valid version string or other
34:     `NumpyVersion` instance.
35: 
36:     Parameters
37:     ----------
38:     vstring : str
39:         Numpy version string (``np.__version__``).
40: 
41:     Notes
42:     -----
43:     All dev versions of the same (pre-)release compare equal.
44: 
45:     Examples
46:     --------
47:     >>> from scipy._lib._version import NumpyVersion
48:     >>> if NumpyVersion(np.__version__) < '1.7.0':
49:     ...     print('skip')
50:     skip
51: 
52:     >>> NumpyVersion('1.7')  # raises ValueError, add ".0"
53: 
54:     '''
55:     def __init__(self, vstring):
56:         self.vstring = vstring
57:         ver_main = re.match(r'\d[.]\d+[.]\d+', vstring)
58:         if not ver_main:
59:             raise ValueError("Not a valid numpy version string")
60: 
61:         self.version = ver_main.group()
62:         self.major, self.minor, self.bugfix = [int(x) for x in
63:             self.version.split('.')]
64:         if len(vstring) == ver_main.end():
65:             self.pre_release = 'final'
66:         else:
67:             alpha = re.match(r'a\d', vstring[ver_main.end():])
68:             beta = re.match(r'b\d', vstring[ver_main.end():])
69:             rc = re.match(r'rc\d', vstring[ver_main.end():])
70:             pre_rel = [m for m in [alpha, beta, rc] if m is not None]
71:             if pre_rel:
72:                 self.pre_release = pre_rel[0].group()
73:             else:
74:                 self.pre_release = ''
75: 
76:         self.is_devversion = bool(re.search(r'.dev', vstring))
77: 
78:     def _compare_version(self, other):
79:         '''Compare major.minor.bugfix'''
80:         if self.major == other.major:
81:             if self.minor == other.minor:
82:                 if self.bugfix == other.bugfix:
83:                     vercmp = 0
84:                 elif self.bugfix > other.bugfix:
85:                     vercmp = 1
86:                 else:
87:                     vercmp = -1
88:             elif self.minor > other.minor:
89:                 vercmp = 1
90:             else:
91:                 vercmp = -1
92:         elif self.major > other.major:
93:             vercmp = 1
94:         else:
95:             vercmp = -1
96: 
97:         return vercmp
98: 
99:     def _compare_pre_release(self, other):
100:         '''Compare alpha/beta/rc/final.'''
101:         if self.pre_release == other.pre_release:
102:             vercmp = 0
103:         elif self.pre_release == 'final':
104:             vercmp = 1
105:         elif other.pre_release == 'final':
106:             vercmp = -1
107:         elif self.pre_release > other.pre_release:
108:             vercmp = 1
109:         else:
110:             vercmp = -1
111: 
112:         return vercmp
113: 
114:     def _compare(self, other):
115:         if not isinstance(other, (string_types, NumpyVersion)):
116:             raise ValueError("Invalid object to compare with NumpyVersion.")
117: 
118:         if isinstance(other, string_types):
119:             other = NumpyVersion(other)
120: 
121:         vercmp = self._compare_version(other)
122:         if vercmp == 0:
123:             # Same x.y.z version, check for alpha/beta/rc
124:             vercmp = self._compare_pre_release(other)
125:             if vercmp == 0:
126:                 # Same version and same pre-release, check if dev version
127:                 if self.is_devversion is other.is_devversion:
128:                     vercmp = 0
129:                 elif self.is_devversion:
130:                     vercmp = -1
131:                 else:
132:                     vercmp = 1
133: 
134:         return vercmp
135: 
136:     def __lt__(self, other):
137:         return self._compare(other) < 0
138: 
139:     def __le__(self, other):
140:         return self._compare(other) <= 0
141: 
142:     def __eq__(self, other):
143:         return self._compare(other) == 0
144: 
145:     def __ne__(self, other):
146:         return self._compare(other) != 0
147: 
148:     def __gt__(self, other):
149:         return self._compare(other) > 0
150: 
151:     def __ge__(self, other):
152:         return self._compare(other) >= 0
153: 
154:     def __repr__(self):
155:         return "NumpyVersion(%s)" % self.vstring
156: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_710747 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, (-1)), 'str', "Utility to compare (Numpy) version strings.\n\nThe NumpyVersion class allows properly comparing numpy version strings.\nThe LooseVersion and StrictVersion classes that distutils provides don't\nwork; they don't recognize anything like alpha/beta/rc/dev versions.\n\n")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import re' statement (line 9)
import re

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from scipy._lib.six import string_types' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/')
import_710748 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib.six')

if (type(import_710748) is not StypyTypeError):

    if (import_710748 != 'pyd_module'):
        __import__(import_710748)
        sys_modules_710749 = sys.modules[import_710748]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib.six', sys_modules_710749.module_type_store, module_type_store, ['string_types'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_710749, sys_modules_710749.module_type_store, module_type_store)
    else:
        from scipy._lib.six import string_types

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib.six', None, module_type_store, ['string_types'], [string_types])

else:
    # Assigning a type to the variable 'scipy._lib.six' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'scipy._lib.six', import_710748)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/')


# Assigning a List to a Name (line 14):

# Assigning a List to a Name (line 14):
__all__ = ['NumpyVersion']
module_type_store.set_exportable_members(['NumpyVersion'])

# Obtaining an instance of the builtin type 'list' (line 14)
list_710750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 14)
# Adding element type (line 14)
str_710751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 11), 'str', 'NumpyVersion')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 10), list_710750, str_710751)

# Assigning a type to the variable '__all__' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), '__all__', list_710750)
# Declaration of the 'NumpyVersion' class

class NumpyVersion:
    str_710752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, (-1)), 'str', 'Parse and compare numpy version strings.\n\n    Numpy has the following versioning scheme (numbers given are examples; they\n    can be >9) in principle):\n\n    - Released version: \'1.8.0\', \'1.8.1\', etc.\n    - Alpha: \'1.8.0a1\', \'1.8.0a2\', etc.\n    - Beta: \'1.8.0b1\', \'1.8.0b2\', etc.\n    - Release candidates: \'1.8.0rc1\', \'1.8.0rc2\', etc.\n    - Development versions: \'1.8.0.dev-f1234afa\' (git commit hash appended)\n    - Development versions after a1: \'1.8.0a1.dev-f1234afa\',\n                                     \'1.8.0b2.dev-f1234afa\',\n                                     \'1.8.1rc1.dev-f1234afa\', etc.\n    - Development versions (no git hash available): \'1.8.0.dev-Unknown\'\n\n    Comparing needs to be done against a valid version string or other\n    `NumpyVersion` instance.\n\n    Parameters\n    ----------\n    vstring : str\n        Numpy version string (``np.__version__``).\n\n    Notes\n    -----\n    All dev versions of the same (pre-)release compare equal.\n\n    Examples\n    --------\n    >>> from scipy._lib._version import NumpyVersion\n    >>> if NumpyVersion(np.__version__) < \'1.7.0\':\n    ...     print(\'skip\')\n    skip\n\n    >>> NumpyVersion(\'1.7\')  # raises ValueError, add ".0"\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 55, 4, False)
        # Assigning a type to the variable 'self' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'self', type_of_self)
        
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

        
        # Assigning a Name to a Attribute (line 56):
        
        # Assigning a Name to a Attribute (line 56):
        # Getting the type of 'vstring' (line 56)
        vstring_710753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 23), 'vstring')
        # Getting the type of 'self' (line 56)
        self_710754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'self')
        # Setting the type of the member 'vstring' of a type (line 56)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), self_710754, 'vstring', vstring_710753)
        
        # Assigning a Call to a Name (line 57):
        
        # Assigning a Call to a Name (line 57):
        
        # Call to match(...): (line 57)
        # Processing the call arguments (line 57)
        str_710757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 28), 'str', '\\d[.]\\d+[.]\\d+')
        # Getting the type of 'vstring' (line 57)
        vstring_710758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 47), 'vstring', False)
        # Processing the call keyword arguments (line 57)
        kwargs_710759 = {}
        # Getting the type of 're' (line 57)
        re_710755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 19), 're', False)
        # Obtaining the member 'match' of a type (line 57)
        match_710756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 19), re_710755, 'match')
        # Calling match(args, kwargs) (line 57)
        match_call_result_710760 = invoke(stypy.reporting.localization.Localization(__file__, 57, 19), match_710756, *[str_710757, vstring_710758], **kwargs_710759)
        
        # Assigning a type to the variable 'ver_main' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'ver_main', match_call_result_710760)
        
        
        # Getting the type of 'ver_main' (line 58)
        ver_main_710761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 15), 'ver_main')
        # Applying the 'not' unary operator (line 58)
        result_not__710762 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 11), 'not', ver_main_710761)
        
        # Testing the type of an if condition (line 58)
        if_condition_710763 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 8), result_not__710762)
        # Assigning a type to the variable 'if_condition_710763' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'if_condition_710763', if_condition_710763)
        # SSA begins for if statement (line 58)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 59)
        # Processing the call arguments (line 59)
        str_710765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 29), 'str', 'Not a valid numpy version string')
        # Processing the call keyword arguments (line 59)
        kwargs_710766 = {}
        # Getting the type of 'ValueError' (line 59)
        ValueError_710764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 59)
        ValueError_call_result_710767 = invoke(stypy.reporting.localization.Localization(__file__, 59, 18), ValueError_710764, *[str_710765], **kwargs_710766)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 59, 12), ValueError_call_result_710767, 'raise parameter', BaseException)
        # SSA join for if statement (line 58)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 61):
        
        # Assigning a Call to a Attribute (line 61):
        
        # Call to group(...): (line 61)
        # Processing the call keyword arguments (line 61)
        kwargs_710770 = {}
        # Getting the type of 'ver_main' (line 61)
        ver_main_710768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 23), 'ver_main', False)
        # Obtaining the member 'group' of a type (line 61)
        group_710769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 23), ver_main_710768, 'group')
        # Calling group(args, kwargs) (line 61)
        group_call_result_710771 = invoke(stypy.reporting.localization.Localization(__file__, 61, 23), group_710769, *[], **kwargs_710770)
        
        # Getting the type of 'self' (line 61)
        self_710772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'self')
        # Setting the type of the member 'version' of a type (line 61)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), self_710772, 'version', group_call_result_710771)
        
        # Assigning a ListComp to a Tuple (line 62):
        
        # Assigning a Subscript to a Name (line 62):
        
        # Obtaining the type of the subscript
        int_710773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 8), 'int')
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to split(...): (line 63)
        # Processing the call arguments (line 63)
        str_710781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 31), 'str', '.')
        # Processing the call keyword arguments (line 63)
        kwargs_710782 = {}
        # Getting the type of 'self' (line 63)
        self_710778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'self', False)
        # Obtaining the member 'version' of a type (line 63)
        version_710779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 12), self_710778, 'version')
        # Obtaining the member 'split' of a type (line 63)
        split_710780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 12), version_710779, 'split')
        # Calling split(args, kwargs) (line 63)
        split_call_result_710783 = invoke(stypy.reporting.localization.Localization(__file__, 63, 12), split_710780, *[str_710781], **kwargs_710782)
        
        comprehension_710784 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 47), split_call_result_710783)
        # Assigning a type to the variable 'x' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 47), 'x', comprehension_710784)
        
        # Call to int(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'x' (line 62)
        x_710775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 51), 'x', False)
        # Processing the call keyword arguments (line 62)
        kwargs_710776 = {}
        # Getting the type of 'int' (line 62)
        int_710774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 47), 'int', False)
        # Calling int(args, kwargs) (line 62)
        int_call_result_710777 = invoke(stypy.reporting.localization.Localization(__file__, 62, 47), int_710774, *[x_710775], **kwargs_710776)
        
        list_710785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 47), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 47), list_710785, int_call_result_710777)
        # Obtaining the member '__getitem__' of a type (line 62)
        getitem___710786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), list_710785, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 62)
        subscript_call_result_710787 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), getitem___710786, int_710773)
        
        # Assigning a type to the variable 'tuple_var_assignment_710744' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'tuple_var_assignment_710744', subscript_call_result_710787)
        
        # Assigning a Subscript to a Name (line 62):
        
        # Obtaining the type of the subscript
        int_710788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 8), 'int')
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to split(...): (line 63)
        # Processing the call arguments (line 63)
        str_710796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 31), 'str', '.')
        # Processing the call keyword arguments (line 63)
        kwargs_710797 = {}
        # Getting the type of 'self' (line 63)
        self_710793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'self', False)
        # Obtaining the member 'version' of a type (line 63)
        version_710794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 12), self_710793, 'version')
        # Obtaining the member 'split' of a type (line 63)
        split_710795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 12), version_710794, 'split')
        # Calling split(args, kwargs) (line 63)
        split_call_result_710798 = invoke(stypy.reporting.localization.Localization(__file__, 63, 12), split_710795, *[str_710796], **kwargs_710797)
        
        comprehension_710799 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 47), split_call_result_710798)
        # Assigning a type to the variable 'x' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 47), 'x', comprehension_710799)
        
        # Call to int(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'x' (line 62)
        x_710790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 51), 'x', False)
        # Processing the call keyword arguments (line 62)
        kwargs_710791 = {}
        # Getting the type of 'int' (line 62)
        int_710789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 47), 'int', False)
        # Calling int(args, kwargs) (line 62)
        int_call_result_710792 = invoke(stypy.reporting.localization.Localization(__file__, 62, 47), int_710789, *[x_710790], **kwargs_710791)
        
        list_710800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 47), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 47), list_710800, int_call_result_710792)
        # Obtaining the member '__getitem__' of a type (line 62)
        getitem___710801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), list_710800, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 62)
        subscript_call_result_710802 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), getitem___710801, int_710788)
        
        # Assigning a type to the variable 'tuple_var_assignment_710745' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'tuple_var_assignment_710745', subscript_call_result_710802)
        
        # Assigning a Subscript to a Name (line 62):
        
        # Obtaining the type of the subscript
        int_710803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 8), 'int')
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to split(...): (line 63)
        # Processing the call arguments (line 63)
        str_710811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 31), 'str', '.')
        # Processing the call keyword arguments (line 63)
        kwargs_710812 = {}
        # Getting the type of 'self' (line 63)
        self_710808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'self', False)
        # Obtaining the member 'version' of a type (line 63)
        version_710809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 12), self_710808, 'version')
        # Obtaining the member 'split' of a type (line 63)
        split_710810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 12), version_710809, 'split')
        # Calling split(args, kwargs) (line 63)
        split_call_result_710813 = invoke(stypy.reporting.localization.Localization(__file__, 63, 12), split_710810, *[str_710811], **kwargs_710812)
        
        comprehension_710814 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 47), split_call_result_710813)
        # Assigning a type to the variable 'x' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 47), 'x', comprehension_710814)
        
        # Call to int(...): (line 62)
        # Processing the call arguments (line 62)
        # Getting the type of 'x' (line 62)
        x_710805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 51), 'x', False)
        # Processing the call keyword arguments (line 62)
        kwargs_710806 = {}
        # Getting the type of 'int' (line 62)
        int_710804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 47), 'int', False)
        # Calling int(args, kwargs) (line 62)
        int_call_result_710807 = invoke(stypy.reporting.localization.Localization(__file__, 62, 47), int_710804, *[x_710805], **kwargs_710806)
        
        list_710815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 47), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 47), list_710815, int_call_result_710807)
        # Obtaining the member '__getitem__' of a type (line 62)
        getitem___710816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), list_710815, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 62)
        subscript_call_result_710817 = invoke(stypy.reporting.localization.Localization(__file__, 62, 8), getitem___710816, int_710803)
        
        # Assigning a type to the variable 'tuple_var_assignment_710746' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'tuple_var_assignment_710746', subscript_call_result_710817)
        
        # Assigning a Name to a Attribute (line 62):
        # Getting the type of 'tuple_var_assignment_710744' (line 62)
        tuple_var_assignment_710744_710818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'tuple_var_assignment_710744')
        # Getting the type of 'self' (line 62)
        self_710819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'self')
        # Setting the type of the member 'major' of a type (line 62)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), self_710819, 'major', tuple_var_assignment_710744_710818)
        
        # Assigning a Name to a Attribute (line 62):
        # Getting the type of 'tuple_var_assignment_710745' (line 62)
        tuple_var_assignment_710745_710820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'tuple_var_assignment_710745')
        # Getting the type of 'self' (line 62)
        self_710821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 20), 'self')
        # Setting the type of the member 'minor' of a type (line 62)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 20), self_710821, 'minor', tuple_var_assignment_710745_710820)
        
        # Assigning a Name to a Attribute (line 62):
        # Getting the type of 'tuple_var_assignment_710746' (line 62)
        tuple_var_assignment_710746_710822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'tuple_var_assignment_710746')
        # Getting the type of 'self' (line 62)
        self_710823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 32), 'self')
        # Setting the type of the member 'bugfix' of a type (line 62)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 32), self_710823, 'bugfix', tuple_var_assignment_710746_710822)
        
        
        
        # Call to len(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'vstring' (line 64)
        vstring_710825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 15), 'vstring', False)
        # Processing the call keyword arguments (line 64)
        kwargs_710826 = {}
        # Getting the type of 'len' (line 64)
        len_710824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 11), 'len', False)
        # Calling len(args, kwargs) (line 64)
        len_call_result_710827 = invoke(stypy.reporting.localization.Localization(__file__, 64, 11), len_710824, *[vstring_710825], **kwargs_710826)
        
        
        # Call to end(...): (line 64)
        # Processing the call keyword arguments (line 64)
        kwargs_710830 = {}
        # Getting the type of 'ver_main' (line 64)
        ver_main_710828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 27), 'ver_main', False)
        # Obtaining the member 'end' of a type (line 64)
        end_710829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 27), ver_main_710828, 'end')
        # Calling end(args, kwargs) (line 64)
        end_call_result_710831 = invoke(stypy.reporting.localization.Localization(__file__, 64, 27), end_710829, *[], **kwargs_710830)
        
        # Applying the binary operator '==' (line 64)
        result_eq_710832 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 11), '==', len_call_result_710827, end_call_result_710831)
        
        # Testing the type of an if condition (line 64)
        if_condition_710833 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 64, 8), result_eq_710832)
        # Assigning a type to the variable 'if_condition_710833' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'if_condition_710833', if_condition_710833)
        # SSA begins for if statement (line 64)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Attribute (line 65):
        
        # Assigning a Str to a Attribute (line 65):
        str_710834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 31), 'str', 'final')
        # Getting the type of 'self' (line 65)
        self_710835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'self')
        # Setting the type of the member 'pre_release' of a type (line 65)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 12), self_710835, 'pre_release', str_710834)
        # SSA branch for the else part of an if statement (line 64)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 67):
        
        # Assigning a Call to a Name (line 67):
        
        # Call to match(...): (line 67)
        # Processing the call arguments (line 67)
        str_710838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 29), 'str', 'a\\d')
        
        # Obtaining the type of the subscript
        
        # Call to end(...): (line 67)
        # Processing the call keyword arguments (line 67)
        kwargs_710841 = {}
        # Getting the type of 'ver_main' (line 67)
        ver_main_710839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 45), 'ver_main', False)
        # Obtaining the member 'end' of a type (line 67)
        end_710840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 45), ver_main_710839, 'end')
        # Calling end(args, kwargs) (line 67)
        end_call_result_710842 = invoke(stypy.reporting.localization.Localization(__file__, 67, 45), end_710840, *[], **kwargs_710841)
        
        slice_710843 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 67, 37), end_call_result_710842, None, None)
        # Getting the type of 'vstring' (line 67)
        vstring_710844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 37), 'vstring', False)
        # Obtaining the member '__getitem__' of a type (line 67)
        getitem___710845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 37), vstring_710844, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 67)
        subscript_call_result_710846 = invoke(stypy.reporting.localization.Localization(__file__, 67, 37), getitem___710845, slice_710843)
        
        # Processing the call keyword arguments (line 67)
        kwargs_710847 = {}
        # Getting the type of 're' (line 67)
        re_710836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 20), 're', False)
        # Obtaining the member 'match' of a type (line 67)
        match_710837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 20), re_710836, 'match')
        # Calling match(args, kwargs) (line 67)
        match_call_result_710848 = invoke(stypy.reporting.localization.Localization(__file__, 67, 20), match_710837, *[str_710838, subscript_call_result_710846], **kwargs_710847)
        
        # Assigning a type to the variable 'alpha' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'alpha', match_call_result_710848)
        
        # Assigning a Call to a Name (line 68):
        
        # Assigning a Call to a Name (line 68):
        
        # Call to match(...): (line 68)
        # Processing the call arguments (line 68)
        str_710851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 28), 'str', 'b\\d')
        
        # Obtaining the type of the subscript
        
        # Call to end(...): (line 68)
        # Processing the call keyword arguments (line 68)
        kwargs_710854 = {}
        # Getting the type of 'ver_main' (line 68)
        ver_main_710852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 44), 'ver_main', False)
        # Obtaining the member 'end' of a type (line 68)
        end_710853 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 44), ver_main_710852, 'end')
        # Calling end(args, kwargs) (line 68)
        end_call_result_710855 = invoke(stypy.reporting.localization.Localization(__file__, 68, 44), end_710853, *[], **kwargs_710854)
        
        slice_710856 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 68, 36), end_call_result_710855, None, None)
        # Getting the type of 'vstring' (line 68)
        vstring_710857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 36), 'vstring', False)
        # Obtaining the member '__getitem__' of a type (line 68)
        getitem___710858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 36), vstring_710857, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 68)
        subscript_call_result_710859 = invoke(stypy.reporting.localization.Localization(__file__, 68, 36), getitem___710858, slice_710856)
        
        # Processing the call keyword arguments (line 68)
        kwargs_710860 = {}
        # Getting the type of 're' (line 68)
        re_710849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 19), 're', False)
        # Obtaining the member 'match' of a type (line 68)
        match_710850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 19), re_710849, 'match')
        # Calling match(args, kwargs) (line 68)
        match_call_result_710861 = invoke(stypy.reporting.localization.Localization(__file__, 68, 19), match_710850, *[str_710851, subscript_call_result_710859], **kwargs_710860)
        
        # Assigning a type to the variable 'beta' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'beta', match_call_result_710861)
        
        # Assigning a Call to a Name (line 69):
        
        # Assigning a Call to a Name (line 69):
        
        # Call to match(...): (line 69)
        # Processing the call arguments (line 69)
        str_710864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 26), 'str', 'rc\\d')
        
        # Obtaining the type of the subscript
        
        # Call to end(...): (line 69)
        # Processing the call keyword arguments (line 69)
        kwargs_710867 = {}
        # Getting the type of 'ver_main' (line 69)
        ver_main_710865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 43), 'ver_main', False)
        # Obtaining the member 'end' of a type (line 69)
        end_710866 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 43), ver_main_710865, 'end')
        # Calling end(args, kwargs) (line 69)
        end_call_result_710868 = invoke(stypy.reporting.localization.Localization(__file__, 69, 43), end_710866, *[], **kwargs_710867)
        
        slice_710869 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 69, 35), end_call_result_710868, None, None)
        # Getting the type of 'vstring' (line 69)
        vstring_710870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 35), 'vstring', False)
        # Obtaining the member '__getitem__' of a type (line 69)
        getitem___710871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 35), vstring_710870, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 69)
        subscript_call_result_710872 = invoke(stypy.reporting.localization.Localization(__file__, 69, 35), getitem___710871, slice_710869)
        
        # Processing the call keyword arguments (line 69)
        kwargs_710873 = {}
        # Getting the type of 're' (line 69)
        re_710862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 17), 're', False)
        # Obtaining the member 'match' of a type (line 69)
        match_710863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 17), re_710862, 'match')
        # Calling match(args, kwargs) (line 69)
        match_call_result_710874 = invoke(stypy.reporting.localization.Localization(__file__, 69, 17), match_710863, *[str_710864, subscript_call_result_710872], **kwargs_710873)
        
        # Assigning a type to the variable 'rc' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'rc', match_call_result_710874)
        
        # Assigning a ListComp to a Name (line 70):
        
        # Assigning a ListComp to a Name (line 70):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Obtaining an instance of the builtin type 'list' (line 70)
        list_710879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 70)
        # Adding element type (line 70)
        # Getting the type of 'alpha' (line 70)
        alpha_710880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 35), 'alpha')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 34), list_710879, alpha_710880)
        # Adding element type (line 70)
        # Getting the type of 'beta' (line 70)
        beta_710881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 42), 'beta')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 34), list_710879, beta_710881)
        # Adding element type (line 70)
        # Getting the type of 'rc' (line 70)
        rc_710882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 48), 'rc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 34), list_710879, rc_710882)
        
        comprehension_710883 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 23), list_710879)
        # Assigning a type to the variable 'm' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 23), 'm', comprehension_710883)
        
        # Getting the type of 'm' (line 70)
        m_710876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 55), 'm')
        # Getting the type of 'None' (line 70)
        None_710877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 64), 'None')
        # Applying the binary operator 'isnot' (line 70)
        result_is_not_710878 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 55), 'isnot', m_710876, None_710877)
        
        # Getting the type of 'm' (line 70)
        m_710875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 23), 'm')
        list_710884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 23), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 23), list_710884, m_710875)
        # Assigning a type to the variable 'pre_rel' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'pre_rel', list_710884)
        
        # Getting the type of 'pre_rel' (line 71)
        pre_rel_710885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 15), 'pre_rel')
        # Testing the type of an if condition (line 71)
        if_condition_710886 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 12), pre_rel_710885)
        # Assigning a type to the variable 'if_condition_710886' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'if_condition_710886', if_condition_710886)
        # SSA begins for if statement (line 71)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 72):
        
        # Assigning a Call to a Attribute (line 72):
        
        # Call to group(...): (line 72)
        # Processing the call keyword arguments (line 72)
        kwargs_710892 = {}
        
        # Obtaining the type of the subscript
        int_710887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 43), 'int')
        # Getting the type of 'pre_rel' (line 72)
        pre_rel_710888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 35), 'pre_rel', False)
        # Obtaining the member '__getitem__' of a type (line 72)
        getitem___710889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 35), pre_rel_710888, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 72)
        subscript_call_result_710890 = invoke(stypy.reporting.localization.Localization(__file__, 72, 35), getitem___710889, int_710887)
        
        # Obtaining the member 'group' of a type (line 72)
        group_710891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 35), subscript_call_result_710890, 'group')
        # Calling group(args, kwargs) (line 72)
        group_call_result_710893 = invoke(stypy.reporting.localization.Localization(__file__, 72, 35), group_710891, *[], **kwargs_710892)
        
        # Getting the type of 'self' (line 72)
        self_710894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 16), 'self')
        # Setting the type of the member 'pre_release' of a type (line 72)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 16), self_710894, 'pre_release', group_call_result_710893)
        # SSA branch for the else part of an if statement (line 71)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Attribute (line 74):
        
        # Assigning a Str to a Attribute (line 74):
        str_710895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 35), 'str', '')
        # Getting the type of 'self' (line 74)
        self_710896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 16), 'self')
        # Setting the type of the member 'pre_release' of a type (line 74)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 16), self_710896, 'pre_release', str_710895)
        # SSA join for if statement (line 71)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 64)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 76):
        
        # Assigning a Call to a Attribute (line 76):
        
        # Call to bool(...): (line 76)
        # Processing the call arguments (line 76)
        
        # Call to search(...): (line 76)
        # Processing the call arguments (line 76)
        str_710900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 44), 'str', '.dev')
        # Getting the type of 'vstring' (line 76)
        vstring_710901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 53), 'vstring', False)
        # Processing the call keyword arguments (line 76)
        kwargs_710902 = {}
        # Getting the type of 're' (line 76)
        re_710898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 34), 're', False)
        # Obtaining the member 'search' of a type (line 76)
        search_710899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 34), re_710898, 'search')
        # Calling search(args, kwargs) (line 76)
        search_call_result_710903 = invoke(stypy.reporting.localization.Localization(__file__, 76, 34), search_710899, *[str_710900, vstring_710901], **kwargs_710902)
        
        # Processing the call keyword arguments (line 76)
        kwargs_710904 = {}
        # Getting the type of 'bool' (line 76)
        bool_710897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 29), 'bool', False)
        # Calling bool(args, kwargs) (line 76)
        bool_call_result_710905 = invoke(stypy.reporting.localization.Localization(__file__, 76, 29), bool_710897, *[search_call_result_710903], **kwargs_710904)
        
        # Getting the type of 'self' (line 76)
        self_710906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'self')
        # Setting the type of the member 'is_devversion' of a type (line 76)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), self_710906, 'is_devversion', bool_call_result_710905)
        
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
        module_type_store = module_type_store.open_function_context('_compare_version', 78, 4, False)
        # Assigning a type to the variable 'self' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'self', type_of_self)
        
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

        str_710907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 8), 'str', 'Compare major.minor.bugfix')
        
        
        # Getting the type of 'self' (line 80)
        self_710908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 11), 'self')
        # Obtaining the member 'major' of a type (line 80)
        major_710909 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 11), self_710908, 'major')
        # Getting the type of 'other' (line 80)
        other_710910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 25), 'other')
        # Obtaining the member 'major' of a type (line 80)
        major_710911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 25), other_710910, 'major')
        # Applying the binary operator '==' (line 80)
        result_eq_710912 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 11), '==', major_710909, major_710911)
        
        # Testing the type of an if condition (line 80)
        if_condition_710913 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 8), result_eq_710912)
        # Assigning a type to the variable 'if_condition_710913' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'if_condition_710913', if_condition_710913)
        # SSA begins for if statement (line 80)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'self' (line 81)
        self_710914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 15), 'self')
        # Obtaining the member 'minor' of a type (line 81)
        minor_710915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 15), self_710914, 'minor')
        # Getting the type of 'other' (line 81)
        other_710916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 29), 'other')
        # Obtaining the member 'minor' of a type (line 81)
        minor_710917 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 29), other_710916, 'minor')
        # Applying the binary operator '==' (line 81)
        result_eq_710918 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 15), '==', minor_710915, minor_710917)
        
        # Testing the type of an if condition (line 81)
        if_condition_710919 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 12), result_eq_710918)
        # Assigning a type to the variable 'if_condition_710919' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'if_condition_710919', if_condition_710919)
        # SSA begins for if statement (line 81)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'self' (line 82)
        self_710920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 19), 'self')
        # Obtaining the member 'bugfix' of a type (line 82)
        bugfix_710921 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 19), self_710920, 'bugfix')
        # Getting the type of 'other' (line 82)
        other_710922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 34), 'other')
        # Obtaining the member 'bugfix' of a type (line 82)
        bugfix_710923 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 34), other_710922, 'bugfix')
        # Applying the binary operator '==' (line 82)
        result_eq_710924 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 19), '==', bugfix_710921, bugfix_710923)
        
        # Testing the type of an if condition (line 82)
        if_condition_710925 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 16), result_eq_710924)
        # Assigning a type to the variable 'if_condition_710925' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 16), 'if_condition_710925', if_condition_710925)
        # SSA begins for if statement (line 82)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 83):
        
        # Assigning a Num to a Name (line 83):
        int_710926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 29), 'int')
        # Assigning a type to the variable 'vercmp' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), 'vercmp', int_710926)
        # SSA branch for the else part of an if statement (line 82)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 84)
        self_710927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 21), 'self')
        # Obtaining the member 'bugfix' of a type (line 84)
        bugfix_710928 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 21), self_710927, 'bugfix')
        # Getting the type of 'other' (line 84)
        other_710929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 35), 'other')
        # Obtaining the member 'bugfix' of a type (line 84)
        bugfix_710930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 35), other_710929, 'bugfix')
        # Applying the binary operator '>' (line 84)
        result_gt_710931 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 21), '>', bugfix_710928, bugfix_710930)
        
        # Testing the type of an if condition (line 84)
        if_condition_710932 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 84, 21), result_gt_710931)
        # Assigning a type to the variable 'if_condition_710932' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 21), 'if_condition_710932', if_condition_710932)
        # SSA begins for if statement (line 84)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 85):
        
        # Assigning a Num to a Name (line 85):
        int_710933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 29), 'int')
        # Assigning a type to the variable 'vercmp' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 20), 'vercmp', int_710933)
        # SSA branch for the else part of an if statement (line 84)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 87):
        
        # Assigning a Num to a Name (line 87):
        int_710934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 29), 'int')
        # Assigning a type to the variable 'vercmp' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), 'vercmp', int_710934)
        # SSA join for if statement (line 84)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 82)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 81)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 88)
        self_710935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 17), 'self')
        # Obtaining the member 'minor' of a type (line 88)
        minor_710936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 17), self_710935, 'minor')
        # Getting the type of 'other' (line 88)
        other_710937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 30), 'other')
        # Obtaining the member 'minor' of a type (line 88)
        minor_710938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 30), other_710937, 'minor')
        # Applying the binary operator '>' (line 88)
        result_gt_710939 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 17), '>', minor_710936, minor_710938)
        
        # Testing the type of an if condition (line 88)
        if_condition_710940 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 88, 17), result_gt_710939)
        # Assigning a type to the variable 'if_condition_710940' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 17), 'if_condition_710940', if_condition_710940)
        # SSA begins for if statement (line 88)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 89):
        
        # Assigning a Num to a Name (line 89):
        int_710941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 25), 'int')
        # Assigning a type to the variable 'vercmp' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 16), 'vercmp', int_710941)
        # SSA branch for the else part of an if statement (line 88)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 91):
        
        # Assigning a Num to a Name (line 91):
        int_710942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 25), 'int')
        # Assigning a type to the variable 'vercmp' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 16), 'vercmp', int_710942)
        # SSA join for if statement (line 88)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 81)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 80)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 92)
        self_710943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 13), 'self')
        # Obtaining the member 'major' of a type (line 92)
        major_710944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 13), self_710943, 'major')
        # Getting the type of 'other' (line 92)
        other_710945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 26), 'other')
        # Obtaining the member 'major' of a type (line 92)
        major_710946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 26), other_710945, 'major')
        # Applying the binary operator '>' (line 92)
        result_gt_710947 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 13), '>', major_710944, major_710946)
        
        # Testing the type of an if condition (line 92)
        if_condition_710948 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 92, 13), result_gt_710947)
        # Assigning a type to the variable 'if_condition_710948' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 13), 'if_condition_710948', if_condition_710948)
        # SSA begins for if statement (line 92)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 93):
        
        # Assigning a Num to a Name (line 93):
        int_710949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 21), 'int')
        # Assigning a type to the variable 'vercmp' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'vercmp', int_710949)
        # SSA branch for the else part of an if statement (line 92)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 95):
        
        # Assigning a Num to a Name (line 95):
        int_710950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 21), 'int')
        # Assigning a type to the variable 'vercmp' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'vercmp', int_710950)
        # SSA join for if statement (line 92)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 80)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'vercmp' (line 97)
        vercmp_710951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 15), 'vercmp')
        # Assigning a type to the variable 'stypy_return_type' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'stypy_return_type', vercmp_710951)
        
        # ################# End of '_compare_version(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_compare_version' in the type store
        # Getting the type of 'stypy_return_type' (line 78)
        stypy_return_type_710952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_710952)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_compare_version'
        return stypy_return_type_710952


    @norecursion
    def _compare_pre_release(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_compare_pre_release'
        module_type_store = module_type_store.open_function_context('_compare_pre_release', 99, 4, False)
        # Assigning a type to the variable 'self' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'self', type_of_self)
        
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

        str_710953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 8), 'str', 'Compare alpha/beta/rc/final.')
        
        
        # Getting the type of 'self' (line 101)
        self_710954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 11), 'self')
        # Obtaining the member 'pre_release' of a type (line 101)
        pre_release_710955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 11), self_710954, 'pre_release')
        # Getting the type of 'other' (line 101)
        other_710956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 31), 'other')
        # Obtaining the member 'pre_release' of a type (line 101)
        pre_release_710957 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 31), other_710956, 'pre_release')
        # Applying the binary operator '==' (line 101)
        result_eq_710958 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 11), '==', pre_release_710955, pre_release_710957)
        
        # Testing the type of an if condition (line 101)
        if_condition_710959 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 101, 8), result_eq_710958)
        # Assigning a type to the variable 'if_condition_710959' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'if_condition_710959', if_condition_710959)
        # SSA begins for if statement (line 101)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 102):
        
        # Assigning a Num to a Name (line 102):
        int_710960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 21), 'int')
        # Assigning a type to the variable 'vercmp' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 12), 'vercmp', int_710960)
        # SSA branch for the else part of an if statement (line 101)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 103)
        self_710961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 13), 'self')
        # Obtaining the member 'pre_release' of a type (line 103)
        pre_release_710962 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 13), self_710961, 'pre_release')
        str_710963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 33), 'str', 'final')
        # Applying the binary operator '==' (line 103)
        result_eq_710964 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 13), '==', pre_release_710962, str_710963)
        
        # Testing the type of an if condition (line 103)
        if_condition_710965 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 13), result_eq_710964)
        # Assigning a type to the variable 'if_condition_710965' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 13), 'if_condition_710965', if_condition_710965)
        # SSA begins for if statement (line 103)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 104):
        
        # Assigning a Num to a Name (line 104):
        int_710966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 21), 'int')
        # Assigning a type to the variable 'vercmp' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'vercmp', int_710966)
        # SSA branch for the else part of an if statement (line 103)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'other' (line 105)
        other_710967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 13), 'other')
        # Obtaining the member 'pre_release' of a type (line 105)
        pre_release_710968 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 13), other_710967, 'pre_release')
        str_710969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 34), 'str', 'final')
        # Applying the binary operator '==' (line 105)
        result_eq_710970 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 13), '==', pre_release_710968, str_710969)
        
        # Testing the type of an if condition (line 105)
        if_condition_710971 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 105, 13), result_eq_710970)
        # Assigning a type to the variable 'if_condition_710971' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 13), 'if_condition_710971', if_condition_710971)
        # SSA begins for if statement (line 105)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 106):
        
        # Assigning a Num to a Name (line 106):
        int_710972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 21), 'int')
        # Assigning a type to the variable 'vercmp' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'vercmp', int_710972)
        # SSA branch for the else part of an if statement (line 105)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'self' (line 107)
        self_710973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 13), 'self')
        # Obtaining the member 'pre_release' of a type (line 107)
        pre_release_710974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 13), self_710973, 'pre_release')
        # Getting the type of 'other' (line 107)
        other_710975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 32), 'other')
        # Obtaining the member 'pre_release' of a type (line 107)
        pre_release_710976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 32), other_710975, 'pre_release')
        # Applying the binary operator '>' (line 107)
        result_gt_710977 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 13), '>', pre_release_710974, pre_release_710976)
        
        # Testing the type of an if condition (line 107)
        if_condition_710978 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 107, 13), result_gt_710977)
        # Assigning a type to the variable 'if_condition_710978' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 13), 'if_condition_710978', if_condition_710978)
        # SSA begins for if statement (line 107)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 108):
        
        # Assigning a Num to a Name (line 108):
        int_710979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 21), 'int')
        # Assigning a type to the variable 'vercmp' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'vercmp', int_710979)
        # SSA branch for the else part of an if statement (line 107)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 110):
        
        # Assigning a Num to a Name (line 110):
        int_710980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 21), 'int')
        # Assigning a type to the variable 'vercmp' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'vercmp', int_710980)
        # SSA join for if statement (line 107)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 105)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 103)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 101)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'vercmp' (line 112)
        vercmp_710981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 15), 'vercmp')
        # Assigning a type to the variable 'stypy_return_type' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'stypy_return_type', vercmp_710981)
        
        # ################# End of '_compare_pre_release(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_compare_pre_release' in the type store
        # Getting the type of 'stypy_return_type' (line 99)
        stypy_return_type_710982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_710982)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_compare_pre_release'
        return stypy_return_type_710982


    @norecursion
    def _compare(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_compare'
        module_type_store = module_type_store.open_function_context('_compare', 114, 4, False)
        # Assigning a type to the variable 'self' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'self', type_of_self)
        
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

        
        
        
        # Call to isinstance(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'other' (line 115)
        other_710984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 26), 'other', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 115)
        tuple_710985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 115)
        # Adding element type (line 115)
        # Getting the type of 'string_types' (line 115)
        string_types_710986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 34), 'string_types', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 34), tuple_710985, string_types_710986)
        # Adding element type (line 115)
        # Getting the type of 'NumpyVersion' (line 115)
        NumpyVersion_710987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 48), 'NumpyVersion', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 34), tuple_710985, NumpyVersion_710987)
        
        # Processing the call keyword arguments (line 115)
        kwargs_710988 = {}
        # Getting the type of 'isinstance' (line 115)
        isinstance_710983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 115)
        isinstance_call_result_710989 = invoke(stypy.reporting.localization.Localization(__file__, 115, 15), isinstance_710983, *[other_710984, tuple_710985], **kwargs_710988)
        
        # Applying the 'not' unary operator (line 115)
        result_not__710990 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 11), 'not', isinstance_call_result_710989)
        
        # Testing the type of an if condition (line 115)
        if_condition_710991 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 115, 8), result_not__710990)
        # Assigning a type to the variable 'if_condition_710991' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'if_condition_710991', if_condition_710991)
        # SSA begins for if statement (line 115)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 116)
        # Processing the call arguments (line 116)
        str_710993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 29), 'str', 'Invalid object to compare with NumpyVersion.')
        # Processing the call keyword arguments (line 116)
        kwargs_710994 = {}
        # Getting the type of 'ValueError' (line 116)
        ValueError_710992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 116)
        ValueError_call_result_710995 = invoke(stypy.reporting.localization.Localization(__file__, 116, 18), ValueError_710992, *[str_710993], **kwargs_710994)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 116, 12), ValueError_call_result_710995, 'raise parameter', BaseException)
        # SSA join for if statement (line 115)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to isinstance(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'other' (line 118)
        other_710997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 22), 'other', False)
        # Getting the type of 'string_types' (line 118)
        string_types_710998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 29), 'string_types', False)
        # Processing the call keyword arguments (line 118)
        kwargs_710999 = {}
        # Getting the type of 'isinstance' (line 118)
        isinstance_710996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 118)
        isinstance_call_result_711000 = invoke(stypy.reporting.localization.Localization(__file__, 118, 11), isinstance_710996, *[other_710997, string_types_710998], **kwargs_710999)
        
        # Testing the type of an if condition (line 118)
        if_condition_711001 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 118, 8), isinstance_call_result_711000)
        # Assigning a type to the variable 'if_condition_711001' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'if_condition_711001', if_condition_711001)
        # SSA begins for if statement (line 118)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 119):
        
        # Assigning a Call to a Name (line 119):
        
        # Call to NumpyVersion(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'other' (line 119)
        other_711003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 33), 'other', False)
        # Processing the call keyword arguments (line 119)
        kwargs_711004 = {}
        # Getting the type of 'NumpyVersion' (line 119)
        NumpyVersion_711002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 20), 'NumpyVersion', False)
        # Calling NumpyVersion(args, kwargs) (line 119)
        NumpyVersion_call_result_711005 = invoke(stypy.reporting.localization.Localization(__file__, 119, 20), NumpyVersion_711002, *[other_711003], **kwargs_711004)
        
        # Assigning a type to the variable 'other' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'other', NumpyVersion_call_result_711005)
        # SSA join for if statement (line 118)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 121):
        
        # Assigning a Call to a Name (line 121):
        
        # Call to _compare_version(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'other' (line 121)
        other_711008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 39), 'other', False)
        # Processing the call keyword arguments (line 121)
        kwargs_711009 = {}
        # Getting the type of 'self' (line 121)
        self_711006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 17), 'self', False)
        # Obtaining the member '_compare_version' of a type (line 121)
        _compare_version_711007 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 17), self_711006, '_compare_version')
        # Calling _compare_version(args, kwargs) (line 121)
        _compare_version_call_result_711010 = invoke(stypy.reporting.localization.Localization(__file__, 121, 17), _compare_version_711007, *[other_711008], **kwargs_711009)
        
        # Assigning a type to the variable 'vercmp' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'vercmp', _compare_version_call_result_711010)
        
        
        # Getting the type of 'vercmp' (line 122)
        vercmp_711011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 11), 'vercmp')
        int_711012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 21), 'int')
        # Applying the binary operator '==' (line 122)
        result_eq_711013 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 11), '==', vercmp_711011, int_711012)
        
        # Testing the type of an if condition (line 122)
        if_condition_711014 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 122, 8), result_eq_711013)
        # Assigning a type to the variable 'if_condition_711014' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'if_condition_711014', if_condition_711014)
        # SSA begins for if statement (line 122)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 124):
        
        # Assigning a Call to a Name (line 124):
        
        # Call to _compare_pre_release(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'other' (line 124)
        other_711017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 47), 'other', False)
        # Processing the call keyword arguments (line 124)
        kwargs_711018 = {}
        # Getting the type of 'self' (line 124)
        self_711015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 21), 'self', False)
        # Obtaining the member '_compare_pre_release' of a type (line 124)
        _compare_pre_release_711016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 21), self_711015, '_compare_pre_release')
        # Calling _compare_pre_release(args, kwargs) (line 124)
        _compare_pre_release_call_result_711019 = invoke(stypy.reporting.localization.Localization(__file__, 124, 21), _compare_pre_release_711016, *[other_711017], **kwargs_711018)
        
        # Assigning a type to the variable 'vercmp' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'vercmp', _compare_pre_release_call_result_711019)
        
        
        # Getting the type of 'vercmp' (line 125)
        vercmp_711020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 15), 'vercmp')
        int_711021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 25), 'int')
        # Applying the binary operator '==' (line 125)
        result_eq_711022 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 15), '==', vercmp_711020, int_711021)
        
        # Testing the type of an if condition (line 125)
        if_condition_711023 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 12), result_eq_711022)
        # Assigning a type to the variable 'if_condition_711023' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'if_condition_711023', if_condition_711023)
        # SSA begins for if statement (line 125)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'self' (line 127)
        self_711024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 19), 'self')
        # Obtaining the member 'is_devversion' of a type (line 127)
        is_devversion_711025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 19), self_711024, 'is_devversion')
        # Getting the type of 'other' (line 127)
        other_711026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 41), 'other')
        # Obtaining the member 'is_devversion' of a type (line 127)
        is_devversion_711027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 41), other_711026, 'is_devversion')
        # Applying the binary operator 'is' (line 127)
        result_is__711028 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 19), 'is', is_devversion_711025, is_devversion_711027)
        
        # Testing the type of an if condition (line 127)
        if_condition_711029 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 127, 16), result_is__711028)
        # Assigning a type to the variable 'if_condition_711029' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 16), 'if_condition_711029', if_condition_711029)
        # SSA begins for if statement (line 127)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 128):
        
        # Assigning a Num to a Name (line 128):
        int_711030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 29), 'int')
        # Assigning a type to the variable 'vercmp' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 20), 'vercmp', int_711030)
        # SSA branch for the else part of an if statement (line 127)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'self' (line 129)
        self_711031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 21), 'self')
        # Obtaining the member 'is_devversion' of a type (line 129)
        is_devversion_711032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 21), self_711031, 'is_devversion')
        # Testing the type of an if condition (line 129)
        if_condition_711033 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 129, 21), is_devversion_711032)
        # Assigning a type to the variable 'if_condition_711033' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 21), 'if_condition_711033', if_condition_711033)
        # SSA begins for if statement (line 129)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Name (line 130):
        
        # Assigning a Num to a Name (line 130):
        int_711034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 29), 'int')
        # Assigning a type to the variable 'vercmp' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 20), 'vercmp', int_711034)
        # SSA branch for the else part of an if statement (line 129)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Num to a Name (line 132):
        
        # Assigning a Num to a Name (line 132):
        int_711035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 29), 'int')
        # Assigning a type to the variable 'vercmp' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 20), 'vercmp', int_711035)
        # SSA join for if statement (line 129)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 127)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 125)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 122)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'vercmp' (line 134)
        vercmp_711036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 15), 'vercmp')
        # Assigning a type to the variable 'stypy_return_type' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'stypy_return_type', vercmp_711036)
        
        # ################# End of '_compare(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_compare' in the type store
        # Getting the type of 'stypy_return_type' (line 114)
        stypy_return_type_711037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_711037)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_compare'
        return stypy_return_type_711037


    @norecursion
    def __lt__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__lt__'
        module_type_store = module_type_store.open_function_context('__lt__', 136, 4, False)
        # Assigning a type to the variable 'self' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'self', type_of_self)
        
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

        
        
        # Call to _compare(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'other' (line 137)
        other_711040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 29), 'other', False)
        # Processing the call keyword arguments (line 137)
        kwargs_711041 = {}
        # Getting the type of 'self' (line 137)
        self_711038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 15), 'self', False)
        # Obtaining the member '_compare' of a type (line 137)
        _compare_711039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 15), self_711038, '_compare')
        # Calling _compare(args, kwargs) (line 137)
        _compare_call_result_711042 = invoke(stypy.reporting.localization.Localization(__file__, 137, 15), _compare_711039, *[other_711040], **kwargs_711041)
        
        int_711043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 38), 'int')
        # Applying the binary operator '<' (line 137)
        result_lt_711044 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 15), '<', _compare_call_result_711042, int_711043)
        
        # Assigning a type to the variable 'stypy_return_type' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'stypy_return_type', result_lt_711044)
        
        # ################# End of '__lt__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__lt__' in the type store
        # Getting the type of 'stypy_return_type' (line 136)
        stypy_return_type_711045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_711045)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__lt__'
        return stypy_return_type_711045


    @norecursion
    def __le__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__le__'
        module_type_store = module_type_store.open_function_context('__le__', 139, 4, False)
        # Assigning a type to the variable 'self' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'self', type_of_self)
        
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

        
        
        # Call to _compare(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'other' (line 140)
        other_711048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 29), 'other', False)
        # Processing the call keyword arguments (line 140)
        kwargs_711049 = {}
        # Getting the type of 'self' (line 140)
        self_711046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 15), 'self', False)
        # Obtaining the member '_compare' of a type (line 140)
        _compare_711047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 15), self_711046, '_compare')
        # Calling _compare(args, kwargs) (line 140)
        _compare_call_result_711050 = invoke(stypy.reporting.localization.Localization(__file__, 140, 15), _compare_711047, *[other_711048], **kwargs_711049)
        
        int_711051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 39), 'int')
        # Applying the binary operator '<=' (line 140)
        result_le_711052 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 15), '<=', _compare_call_result_711050, int_711051)
        
        # Assigning a type to the variable 'stypy_return_type' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'stypy_return_type', result_le_711052)
        
        # ################# End of '__le__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__le__' in the type store
        # Getting the type of 'stypy_return_type' (line 139)
        stypy_return_type_711053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_711053)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__le__'
        return stypy_return_type_711053


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 142, 4, False)
        # Assigning a type to the variable 'self' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NumpyVersion.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        NumpyVersion.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NumpyVersion.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        NumpyVersion.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'NumpyVersion.stypy__eq__')
        NumpyVersion.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        NumpyVersion.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        NumpyVersion.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NumpyVersion.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        NumpyVersion.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        NumpyVersion.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NumpyVersion.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NumpyVersion.stypy__eq__', ['other'], None, None, defaults, varargs, kwargs)

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

        
        
        # Call to _compare(...): (line 143)
        # Processing the call arguments (line 143)
        # Getting the type of 'other' (line 143)
        other_711056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 29), 'other', False)
        # Processing the call keyword arguments (line 143)
        kwargs_711057 = {}
        # Getting the type of 'self' (line 143)
        self_711054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 15), 'self', False)
        # Obtaining the member '_compare' of a type (line 143)
        _compare_711055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 15), self_711054, '_compare')
        # Calling _compare(args, kwargs) (line 143)
        _compare_call_result_711058 = invoke(stypy.reporting.localization.Localization(__file__, 143, 15), _compare_711055, *[other_711056], **kwargs_711057)
        
        int_711059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 39), 'int')
        # Applying the binary operator '==' (line 143)
        result_eq_711060 = python_operator(stypy.reporting.localization.Localization(__file__, 143, 15), '==', _compare_call_result_711058, int_711059)
        
        # Assigning a type to the variable 'stypy_return_type' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'stypy_return_type', result_eq_711060)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 142)
        stypy_return_type_711061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_711061)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_711061


    @norecursion
    def __ne__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__ne__'
        module_type_store = module_type_store.open_function_context('__ne__', 145, 4, False)
        # Assigning a type to the variable 'self' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'self', type_of_self)
        
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

        
        
        # Call to _compare(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'other' (line 146)
        other_711064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 29), 'other', False)
        # Processing the call keyword arguments (line 146)
        kwargs_711065 = {}
        # Getting the type of 'self' (line 146)
        self_711062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 15), 'self', False)
        # Obtaining the member '_compare' of a type (line 146)
        _compare_711063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 15), self_711062, '_compare')
        # Calling _compare(args, kwargs) (line 146)
        _compare_call_result_711066 = invoke(stypy.reporting.localization.Localization(__file__, 146, 15), _compare_711063, *[other_711064], **kwargs_711065)
        
        int_711067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 39), 'int')
        # Applying the binary operator '!=' (line 146)
        result_ne_711068 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 15), '!=', _compare_call_result_711066, int_711067)
        
        # Assigning a type to the variable 'stypy_return_type' (line 146)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'stypy_return_type', result_ne_711068)
        
        # ################# End of '__ne__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__ne__' in the type store
        # Getting the type of 'stypy_return_type' (line 145)
        stypy_return_type_711069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_711069)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__ne__'
        return stypy_return_type_711069


    @norecursion
    def __gt__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__gt__'
        module_type_store = module_type_store.open_function_context('__gt__', 148, 4, False)
        # Assigning a type to the variable 'self' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'self', type_of_self)
        
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

        
        
        # Call to _compare(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'other' (line 149)
        other_711072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 29), 'other', False)
        # Processing the call keyword arguments (line 149)
        kwargs_711073 = {}
        # Getting the type of 'self' (line 149)
        self_711070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 15), 'self', False)
        # Obtaining the member '_compare' of a type (line 149)
        _compare_711071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 15), self_711070, '_compare')
        # Calling _compare(args, kwargs) (line 149)
        _compare_call_result_711074 = invoke(stypy.reporting.localization.Localization(__file__, 149, 15), _compare_711071, *[other_711072], **kwargs_711073)
        
        int_711075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 38), 'int')
        # Applying the binary operator '>' (line 149)
        result_gt_711076 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 15), '>', _compare_call_result_711074, int_711075)
        
        # Assigning a type to the variable 'stypy_return_type' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'stypy_return_type', result_gt_711076)
        
        # ################# End of '__gt__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__gt__' in the type store
        # Getting the type of 'stypy_return_type' (line 148)
        stypy_return_type_711077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_711077)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__gt__'
        return stypy_return_type_711077


    @norecursion
    def __ge__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__ge__'
        module_type_store = module_type_store.open_function_context('__ge__', 151, 4, False)
        # Assigning a type to the variable 'self' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'self', type_of_self)
        
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

        
        
        # Call to _compare(...): (line 152)
        # Processing the call arguments (line 152)
        # Getting the type of 'other' (line 152)
        other_711080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 29), 'other', False)
        # Processing the call keyword arguments (line 152)
        kwargs_711081 = {}
        # Getting the type of 'self' (line 152)
        self_711078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 15), 'self', False)
        # Obtaining the member '_compare' of a type (line 152)
        _compare_711079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 15), self_711078, '_compare')
        # Calling _compare(args, kwargs) (line 152)
        _compare_call_result_711082 = invoke(stypy.reporting.localization.Localization(__file__, 152, 15), _compare_711079, *[other_711080], **kwargs_711081)
        
        int_711083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 39), 'int')
        # Applying the binary operator '>=' (line 152)
        result_ge_711084 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 15), '>=', _compare_call_result_711082, int_711083)
        
        # Assigning a type to the variable 'stypy_return_type' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'stypy_return_type', result_ge_711084)
        
        # ################# End of '__ge__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__ge__' in the type store
        # Getting the type of 'stypy_return_type' (line 151)
        stypy_return_type_711085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_711085)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__ge__'
        return stypy_return_type_711085


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 154, 4, False)
        # Assigning a type to the variable 'self' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NumpyVersion.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        NumpyVersion.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NumpyVersion.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        NumpyVersion.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'NumpyVersion.stypy__repr__')
        NumpyVersion.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        NumpyVersion.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        NumpyVersion.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NumpyVersion.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        NumpyVersion.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        NumpyVersion.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NumpyVersion.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NumpyVersion.stypy__repr__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__repr__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__repr__(...)' code ##################

        str_711086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 15), 'str', 'NumpyVersion(%s)')
        # Getting the type of 'self' (line 155)
        self_711087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 36), 'self')
        # Obtaining the member 'vstring' of a type (line 155)
        vstring_711088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 36), self_711087, 'vstring')
        # Applying the binary operator '%' (line 155)
        result_mod_711089 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 15), '%', str_711086, vstring_711088)
        
        # Assigning a type to the variable 'stypy_return_type' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), 'stypy_return_type', result_mod_711089)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 154)
        stypy_return_type_711090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_711090)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_711090


# Assigning a type to the variable 'NumpyVersion' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'NumpyVersion', NumpyVersion)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
