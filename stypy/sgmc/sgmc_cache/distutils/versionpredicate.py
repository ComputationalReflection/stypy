
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Module for parsing and testing package version predicate strings.
2: '''
3: import re
4: import distutils.version
5: import operator
6: 
7: 
8: re_validPackage = re.compile(r"(?i)^\s*([a-z_]\w*(?:\.[a-z_]\w*)*)(.*)")
9: # (package) (rest)
10: 
11: re_paren = re.compile(r"^\s*\((.*)\)\s*$") # (list) inside of parentheses
12: re_splitComparison = re.compile(r"^\s*(<=|>=|<|>|!=|==)\s*([^\s,]+)\s*$")
13: # (comp) (version)
14: 
15: 
16: def splitUp(pred):
17:     '''Parse a single version comparison.
18: 
19:     Return (comparison string, StrictVersion)
20:     '''
21:     res = re_splitComparison.match(pred)
22:     if not res:
23:         raise ValueError("bad package restriction syntax: %r" % pred)
24:     comp, verStr = res.groups()
25:     return (comp, distutils.version.StrictVersion(verStr))
26: 
27: compmap = {"<": operator.lt, "<=": operator.le, "==": operator.eq,
28:            ">": operator.gt, ">=": operator.ge, "!=": operator.ne}
29: 
30: class VersionPredicate:
31:     '''Parse and test package version predicates.
32: 
33:     >>> v = VersionPredicate('pyepat.abc (>1.0, <3333.3a1, !=1555.1b3)')
34: 
35:     The `name` attribute provides the full dotted name that is given::
36: 
37:     >>> v.name
38:     'pyepat.abc'
39: 
40:     The str() of a `VersionPredicate` provides a normalized
41:     human-readable version of the expression::
42: 
43:     >>> print v
44:     pyepat.abc (> 1.0, < 3333.3a1, != 1555.1b3)
45: 
46:     The `satisfied_by()` method can be used to determine with a given
47:     version number is included in the set described by the version
48:     restrictions::
49: 
50:     >>> v.satisfied_by('1.1')
51:     True
52:     >>> v.satisfied_by('1.4')
53:     True
54:     >>> v.satisfied_by('1.0')
55:     False
56:     >>> v.satisfied_by('4444.4')
57:     False
58:     >>> v.satisfied_by('1555.1b3')
59:     False
60: 
61:     `VersionPredicate` is flexible in accepting extra whitespace::
62: 
63:     >>> v = VersionPredicate(' pat( ==  0.1  )  ')
64:     >>> v.name
65:     'pat'
66:     >>> v.satisfied_by('0.1')
67:     True
68:     >>> v.satisfied_by('0.2')
69:     False
70: 
71:     If any version numbers passed in do not conform to the
72:     restrictions of `StrictVersion`, a `ValueError` is raised::
73: 
74:     >>> v = VersionPredicate('p1.p2.p3.p4(>=1.0, <=1.3a1, !=1.2zb3)')
75:     Traceback (most recent call last):
76:       ...
77:     ValueError: invalid version number '1.2zb3'
78: 
79:     It the module or package name given does not conform to what's
80:     allowed as a legal module or package name, `ValueError` is
81:     raised::
82: 
83:     >>> v = VersionPredicate('foo-bar')
84:     Traceback (most recent call last):
85:       ...
86:     ValueError: expected parenthesized list: '-bar'
87: 
88:     >>> v = VersionPredicate('foo bar (12.21)')
89:     Traceback (most recent call last):
90:       ...
91:     ValueError: expected parenthesized list: 'bar (12.21)'
92: 
93:     '''
94: 
95:     def __init__(self, versionPredicateStr):
96:         '''Parse a version predicate string.
97:         '''
98:         # Fields:
99:         #    name:  package name
100:         #    pred:  list of (comparison string, StrictVersion)
101: 
102:         versionPredicateStr = versionPredicateStr.strip()
103:         if not versionPredicateStr:
104:             raise ValueError("empty package restriction")
105:         match = re_validPackage.match(versionPredicateStr)
106:         if not match:
107:             raise ValueError("bad package name in %r" % versionPredicateStr)
108:         self.name, paren = match.groups()
109:         paren = paren.strip()
110:         if paren:
111:             match = re_paren.match(paren)
112:             if not match:
113:                 raise ValueError("expected parenthesized list: %r" % paren)
114:             str = match.groups()[0]
115:             self.pred = [splitUp(aPred) for aPred in str.split(",")]
116:             if not self.pred:
117:                 raise ValueError("empty parenthesized list in %r"
118:                                  % versionPredicateStr)
119:         else:
120:             self.pred = []
121: 
122:     def __str__(self):
123:         if self.pred:
124:             seq = [cond + " " + str(ver) for cond, ver in self.pred]
125:             return self.name + " (" + ", ".join(seq) + ")"
126:         else:
127:             return self.name
128: 
129:     def satisfied_by(self, version):
130:         '''True if version is compatible with all the predicates in self.
131:         The parameter version must be acceptable to the StrictVersion
132:         constructor.  It may be either a string or StrictVersion.
133:         '''
134:         for cond, ver in self.pred:
135:             if not compmap[cond](version, ver):
136:                 return False
137:         return True
138: 
139: 
140: _provision_rx = None
141: 
142: def split_provision(value):
143:     '''Return the name and optional version number of a provision.
144: 
145:     The version number, if given, will be returned as a `StrictVersion`
146:     instance, otherwise it will be `None`.
147: 
148:     >>> split_provision('mypkg')
149:     ('mypkg', None)
150:     >>> split_provision(' mypkg( 1.2 ) ')
151:     ('mypkg', StrictVersion ('1.2'))
152:     '''
153:     global _provision_rx
154:     if _provision_rx is None:
155:         _provision_rx = re.compile(
156:             "([a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)*)(?:\s*\(\s*([^)\s]+)\s*\))?$")
157:     value = value.strip()
158:     m = _provision_rx.match(value)
159:     if not m:
160:         raise ValueError("illegal provides specification: %r" % value)
161:     ver = m.group(2) or None
162:     if ver:
163:         ver = distutils.version.StrictVersion(ver)
164:     return m.group(1), ver
165: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_11394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, (-1)), 'str', 'Module for parsing and testing package version predicate strings.\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import re' statement (line 3)
import re

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 're', re, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import distutils.version' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_11395 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.version')

if (type(import_11395) is not StypyTypeError):

    if (import_11395 != 'pyd_module'):
        __import__(import_11395)
        sys_modules_11396 = sys.modules[import_11395]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.version', sys_modules_11396.module_type_store, module_type_store)
    else:
        import distutils.version

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.version', distutils.version, module_type_store)

else:
    # Assigning a type to the variable 'distutils.version' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.version', import_11395)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import operator' statement (line 5)
import operator

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'operator', operator, module_type_store)


# Assigning a Call to a Name (line 8):

# Assigning a Call to a Name (line 8):

# Call to compile(...): (line 8)
# Processing the call arguments (line 8)
str_11399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 29), 'str', '(?i)^\\s*([a-z_]\\w*(?:\\.[a-z_]\\w*)*)(.*)')
# Processing the call keyword arguments (line 8)
kwargs_11400 = {}
# Getting the type of 're' (line 8)
re_11397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 18), 're', False)
# Obtaining the member 'compile' of a type (line 8)
compile_11398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 18), re_11397, 'compile')
# Calling compile(args, kwargs) (line 8)
compile_call_result_11401 = invoke(stypy.reporting.localization.Localization(__file__, 8, 18), compile_11398, *[str_11399], **kwargs_11400)

# Assigning a type to the variable 're_validPackage' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 're_validPackage', compile_call_result_11401)

# Assigning a Call to a Name (line 11):

# Assigning a Call to a Name (line 11):

# Call to compile(...): (line 11)
# Processing the call arguments (line 11)
str_11404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 22), 'str', '^\\s*\\((.*)\\)\\s*$')
# Processing the call keyword arguments (line 11)
kwargs_11405 = {}
# Getting the type of 're' (line 11)
re_11402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 11), 're', False)
# Obtaining the member 'compile' of a type (line 11)
compile_11403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 11), re_11402, 'compile')
# Calling compile(args, kwargs) (line 11)
compile_call_result_11406 = invoke(stypy.reporting.localization.Localization(__file__, 11, 11), compile_11403, *[str_11404], **kwargs_11405)

# Assigning a type to the variable 're_paren' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 're_paren', compile_call_result_11406)

# Assigning a Call to a Name (line 12):

# Assigning a Call to a Name (line 12):

# Call to compile(...): (line 12)
# Processing the call arguments (line 12)
str_11409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 32), 'str', '^\\s*(<=|>=|<|>|!=|==)\\s*([^\\s,]+)\\s*$')
# Processing the call keyword arguments (line 12)
kwargs_11410 = {}
# Getting the type of 're' (line 12)
re_11407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 21), 're', False)
# Obtaining the member 'compile' of a type (line 12)
compile_11408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 21), re_11407, 'compile')
# Calling compile(args, kwargs) (line 12)
compile_call_result_11411 = invoke(stypy.reporting.localization.Localization(__file__, 12, 21), compile_11408, *[str_11409], **kwargs_11410)

# Assigning a type to the variable 're_splitComparison' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 're_splitComparison', compile_call_result_11411)

@norecursion
def splitUp(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'splitUp'
    module_type_store = module_type_store.open_function_context('splitUp', 16, 0, False)
    
    # Passed parameters checking function
    splitUp.stypy_localization = localization
    splitUp.stypy_type_of_self = None
    splitUp.stypy_type_store = module_type_store
    splitUp.stypy_function_name = 'splitUp'
    splitUp.stypy_param_names_list = ['pred']
    splitUp.stypy_varargs_param_name = None
    splitUp.stypy_kwargs_param_name = None
    splitUp.stypy_call_defaults = defaults
    splitUp.stypy_call_varargs = varargs
    splitUp.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'splitUp', ['pred'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'splitUp', localization, ['pred'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'splitUp(...)' code ##################

    str_11412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, (-1)), 'str', 'Parse a single version comparison.\n\n    Return (comparison string, StrictVersion)\n    ')
    
    # Assigning a Call to a Name (line 21):
    
    # Assigning a Call to a Name (line 21):
    
    # Call to match(...): (line 21)
    # Processing the call arguments (line 21)
    # Getting the type of 'pred' (line 21)
    pred_11415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 35), 'pred', False)
    # Processing the call keyword arguments (line 21)
    kwargs_11416 = {}
    # Getting the type of 're_splitComparison' (line 21)
    re_splitComparison_11413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 10), 're_splitComparison', False)
    # Obtaining the member 'match' of a type (line 21)
    match_11414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 10), re_splitComparison_11413, 'match')
    # Calling match(args, kwargs) (line 21)
    match_call_result_11417 = invoke(stypy.reporting.localization.Localization(__file__, 21, 10), match_11414, *[pred_11415], **kwargs_11416)
    
    # Assigning a type to the variable 'res' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'res', match_call_result_11417)
    
    
    # Getting the type of 'res' (line 22)
    res_11418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 11), 'res')
    # Applying the 'not' unary operator (line 22)
    result_not__11419 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 7), 'not', res_11418)
    
    # Testing the type of an if condition (line 22)
    if_condition_11420 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 22, 4), result_not__11419)
    # Assigning a type to the variable 'if_condition_11420' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'if_condition_11420', if_condition_11420)
    # SSA begins for if statement (line 22)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 23)
    # Processing the call arguments (line 23)
    str_11422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 25), 'str', 'bad package restriction syntax: %r')
    # Getting the type of 'pred' (line 23)
    pred_11423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 64), 'pred', False)
    # Applying the binary operator '%' (line 23)
    result_mod_11424 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 25), '%', str_11422, pred_11423)
    
    # Processing the call keyword arguments (line 23)
    kwargs_11425 = {}
    # Getting the type of 'ValueError' (line 23)
    ValueError_11421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 23)
    ValueError_call_result_11426 = invoke(stypy.reporting.localization.Localization(__file__, 23, 14), ValueError_11421, *[result_mod_11424], **kwargs_11425)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 23, 8), ValueError_call_result_11426, 'raise parameter', BaseException)
    # SSA join for if statement (line 22)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 24):
    
    # Assigning a Subscript to a Name (line 24):
    
    # Obtaining the type of the subscript
    int_11427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 4), 'int')
    
    # Call to groups(...): (line 24)
    # Processing the call keyword arguments (line 24)
    kwargs_11430 = {}
    # Getting the type of 'res' (line 24)
    res_11428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 19), 'res', False)
    # Obtaining the member 'groups' of a type (line 24)
    groups_11429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 19), res_11428, 'groups')
    # Calling groups(args, kwargs) (line 24)
    groups_call_result_11431 = invoke(stypy.reporting.localization.Localization(__file__, 24, 19), groups_11429, *[], **kwargs_11430)
    
    # Obtaining the member '__getitem__' of a type (line 24)
    getitem___11432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 4), groups_call_result_11431, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 24)
    subscript_call_result_11433 = invoke(stypy.reporting.localization.Localization(__file__, 24, 4), getitem___11432, int_11427)
    
    # Assigning a type to the variable 'tuple_var_assignment_11390' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'tuple_var_assignment_11390', subscript_call_result_11433)
    
    # Assigning a Subscript to a Name (line 24):
    
    # Obtaining the type of the subscript
    int_11434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 4), 'int')
    
    # Call to groups(...): (line 24)
    # Processing the call keyword arguments (line 24)
    kwargs_11437 = {}
    # Getting the type of 'res' (line 24)
    res_11435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 19), 'res', False)
    # Obtaining the member 'groups' of a type (line 24)
    groups_11436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 19), res_11435, 'groups')
    # Calling groups(args, kwargs) (line 24)
    groups_call_result_11438 = invoke(stypy.reporting.localization.Localization(__file__, 24, 19), groups_11436, *[], **kwargs_11437)
    
    # Obtaining the member '__getitem__' of a type (line 24)
    getitem___11439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 4), groups_call_result_11438, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 24)
    subscript_call_result_11440 = invoke(stypy.reporting.localization.Localization(__file__, 24, 4), getitem___11439, int_11434)
    
    # Assigning a type to the variable 'tuple_var_assignment_11391' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'tuple_var_assignment_11391', subscript_call_result_11440)
    
    # Assigning a Name to a Name (line 24):
    # Getting the type of 'tuple_var_assignment_11390' (line 24)
    tuple_var_assignment_11390_11441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'tuple_var_assignment_11390')
    # Assigning a type to the variable 'comp' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'comp', tuple_var_assignment_11390_11441)
    
    # Assigning a Name to a Name (line 24):
    # Getting the type of 'tuple_var_assignment_11391' (line 24)
    tuple_var_assignment_11391_11442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'tuple_var_assignment_11391')
    # Assigning a type to the variable 'verStr' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 10), 'verStr', tuple_var_assignment_11391_11442)
    
    # Obtaining an instance of the builtin type 'tuple' (line 25)
    tuple_11443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 25)
    # Adding element type (line 25)
    # Getting the type of 'comp' (line 25)
    comp_11444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'comp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 12), tuple_11443, comp_11444)
    # Adding element type (line 25)
    
    # Call to StrictVersion(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'verStr' (line 25)
    verStr_11448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 50), 'verStr', False)
    # Processing the call keyword arguments (line 25)
    kwargs_11449 = {}
    # Getting the type of 'distutils' (line 25)
    distutils_11445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 18), 'distutils', False)
    # Obtaining the member 'version' of a type (line 25)
    version_11446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 18), distutils_11445, 'version')
    # Obtaining the member 'StrictVersion' of a type (line 25)
    StrictVersion_11447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 18), version_11446, 'StrictVersion')
    # Calling StrictVersion(args, kwargs) (line 25)
    StrictVersion_call_result_11450 = invoke(stypy.reporting.localization.Localization(__file__, 25, 18), StrictVersion_11447, *[verStr_11448], **kwargs_11449)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 12), tuple_11443, StrictVersion_call_result_11450)
    
    # Assigning a type to the variable 'stypy_return_type' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type', tuple_11443)
    
    # ################# End of 'splitUp(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'splitUp' in the type store
    # Getting the type of 'stypy_return_type' (line 16)
    stypy_return_type_11451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_11451)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'splitUp'
    return stypy_return_type_11451

# Assigning a type to the variable 'splitUp' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'splitUp', splitUp)

# Assigning a Dict to a Name (line 27):

# Assigning a Dict to a Name (line 27):

# Obtaining an instance of the builtin type 'dict' (line 27)
dict_11452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 10), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 27)
# Adding element type (key, value) (line 27)
str_11453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 11), 'str', '<')
# Getting the type of 'operator' (line 27)
operator_11454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 16), 'operator')
# Obtaining the member 'lt' of a type (line 27)
lt_11455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 16), operator_11454, 'lt')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), dict_11452, (str_11453, lt_11455))
# Adding element type (key, value) (line 27)
str_11456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 29), 'str', '<=')
# Getting the type of 'operator' (line 27)
operator_11457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 35), 'operator')
# Obtaining the member 'le' of a type (line 27)
le_11458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 35), operator_11457, 'le')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), dict_11452, (str_11456, le_11458))
# Adding element type (key, value) (line 27)
str_11459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 48), 'str', '==')
# Getting the type of 'operator' (line 27)
operator_11460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 54), 'operator')
# Obtaining the member 'eq' of a type (line 27)
eq_11461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 54), operator_11460, 'eq')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), dict_11452, (str_11459, eq_11461))
# Adding element type (key, value) (line 27)
str_11462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 11), 'str', '>')
# Getting the type of 'operator' (line 28)
operator_11463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 16), 'operator')
# Obtaining the member 'gt' of a type (line 28)
gt_11464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 16), operator_11463, 'gt')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), dict_11452, (str_11462, gt_11464))
# Adding element type (key, value) (line 27)
str_11465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 29), 'str', '>=')
# Getting the type of 'operator' (line 28)
operator_11466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 35), 'operator')
# Obtaining the member 'ge' of a type (line 28)
ge_11467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 35), operator_11466, 'ge')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), dict_11452, (str_11465, ge_11467))
# Adding element type (key, value) (line 27)
str_11468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 48), 'str', '!=')
# Getting the type of 'operator' (line 28)
operator_11469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 54), 'operator')
# Obtaining the member 'ne' of a type (line 28)
ne_11470 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 54), operator_11469, 'ne')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 10), dict_11452, (str_11468, ne_11470))

# Assigning a type to the variable 'compmap' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'compmap', dict_11452)
# Declaration of the 'VersionPredicate' class

class VersionPredicate:
    str_11471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, (-1)), 'str', "Parse and test package version predicates.\n\n    >>> v = VersionPredicate('pyepat.abc (>1.0, <3333.3a1, !=1555.1b3)')\n\n    The `name` attribute provides the full dotted name that is given::\n\n    >>> v.name\n    'pyepat.abc'\n\n    The str() of a `VersionPredicate` provides a normalized\n    human-readable version of the expression::\n\n    >>> print v\n    pyepat.abc (> 1.0, < 3333.3a1, != 1555.1b3)\n\n    The `satisfied_by()` method can be used to determine with a given\n    version number is included in the set described by the version\n    restrictions::\n\n    >>> v.satisfied_by('1.1')\n    True\n    >>> v.satisfied_by('1.4')\n    True\n    >>> v.satisfied_by('1.0')\n    False\n    >>> v.satisfied_by('4444.4')\n    False\n    >>> v.satisfied_by('1555.1b3')\n    False\n\n    `VersionPredicate` is flexible in accepting extra whitespace::\n\n    >>> v = VersionPredicate(' pat( ==  0.1  )  ')\n    >>> v.name\n    'pat'\n    >>> v.satisfied_by('0.1')\n    True\n    >>> v.satisfied_by('0.2')\n    False\n\n    If any version numbers passed in do not conform to the\n    restrictions of `StrictVersion`, a `ValueError` is raised::\n\n    >>> v = VersionPredicate('p1.p2.p3.p4(>=1.0, <=1.3a1, !=1.2zb3)')\n    Traceback (most recent call last):\n      ...\n    ValueError: invalid version number '1.2zb3'\n\n    It the module or package name given does not conform to what's\n    allowed as a legal module or package name, `ValueError` is\n    raised::\n\n    >>> v = VersionPredicate('foo-bar')\n    Traceback (most recent call last):\n      ...\n    ValueError: expected parenthesized list: '-bar'\n\n    >>> v = VersionPredicate('foo bar (12.21)')\n    Traceback (most recent call last):\n      ...\n    ValueError: expected parenthesized list: 'bar (12.21)'\n\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 95, 4, False)
        # Assigning a type to the variable 'self' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VersionPredicate.__init__', ['versionPredicateStr'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['versionPredicateStr'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_11472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, (-1)), 'str', 'Parse a version predicate string.\n        ')
        
        # Assigning a Call to a Name (line 102):
        
        # Assigning a Call to a Name (line 102):
        
        # Call to strip(...): (line 102)
        # Processing the call keyword arguments (line 102)
        kwargs_11475 = {}
        # Getting the type of 'versionPredicateStr' (line 102)
        versionPredicateStr_11473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 30), 'versionPredicateStr', False)
        # Obtaining the member 'strip' of a type (line 102)
        strip_11474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 30), versionPredicateStr_11473, 'strip')
        # Calling strip(args, kwargs) (line 102)
        strip_call_result_11476 = invoke(stypy.reporting.localization.Localization(__file__, 102, 30), strip_11474, *[], **kwargs_11475)
        
        # Assigning a type to the variable 'versionPredicateStr' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'versionPredicateStr', strip_call_result_11476)
        
        
        # Getting the type of 'versionPredicateStr' (line 103)
        versionPredicateStr_11477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 15), 'versionPredicateStr')
        # Applying the 'not' unary operator (line 103)
        result_not__11478 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 11), 'not', versionPredicateStr_11477)
        
        # Testing the type of an if condition (line 103)
        if_condition_11479 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 8), result_not__11478)
        # Assigning a type to the variable 'if_condition_11479' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'if_condition_11479', if_condition_11479)
        # SSA begins for if statement (line 103)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 104)
        # Processing the call arguments (line 104)
        str_11481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 29), 'str', 'empty package restriction')
        # Processing the call keyword arguments (line 104)
        kwargs_11482 = {}
        # Getting the type of 'ValueError' (line 104)
        ValueError_11480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 104)
        ValueError_call_result_11483 = invoke(stypy.reporting.localization.Localization(__file__, 104, 18), ValueError_11480, *[str_11481], **kwargs_11482)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 104, 12), ValueError_call_result_11483, 'raise parameter', BaseException)
        # SSA join for if statement (line 103)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 105):
        
        # Assigning a Call to a Name (line 105):
        
        # Call to match(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'versionPredicateStr' (line 105)
        versionPredicateStr_11486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 38), 'versionPredicateStr', False)
        # Processing the call keyword arguments (line 105)
        kwargs_11487 = {}
        # Getting the type of 're_validPackage' (line 105)
        re_validPackage_11484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 're_validPackage', False)
        # Obtaining the member 'match' of a type (line 105)
        match_11485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 16), re_validPackage_11484, 'match')
        # Calling match(args, kwargs) (line 105)
        match_call_result_11488 = invoke(stypy.reporting.localization.Localization(__file__, 105, 16), match_11485, *[versionPredicateStr_11486], **kwargs_11487)
        
        # Assigning a type to the variable 'match' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'match', match_call_result_11488)
        
        
        # Getting the type of 'match' (line 106)
        match_11489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 15), 'match')
        # Applying the 'not' unary operator (line 106)
        result_not__11490 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 11), 'not', match_11489)
        
        # Testing the type of an if condition (line 106)
        if_condition_11491 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 8), result_not__11490)
        # Assigning a type to the variable 'if_condition_11491' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'if_condition_11491', if_condition_11491)
        # SSA begins for if statement (line 106)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 107)
        # Processing the call arguments (line 107)
        str_11493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 29), 'str', 'bad package name in %r')
        # Getting the type of 'versionPredicateStr' (line 107)
        versionPredicateStr_11494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 56), 'versionPredicateStr', False)
        # Applying the binary operator '%' (line 107)
        result_mod_11495 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 29), '%', str_11493, versionPredicateStr_11494)
        
        # Processing the call keyword arguments (line 107)
        kwargs_11496 = {}
        # Getting the type of 'ValueError' (line 107)
        ValueError_11492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 107)
        ValueError_call_result_11497 = invoke(stypy.reporting.localization.Localization(__file__, 107, 18), ValueError_11492, *[result_mod_11495], **kwargs_11496)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 107, 12), ValueError_call_result_11497, 'raise parameter', BaseException)
        # SSA join for if statement (line 106)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Tuple (line 108):
        
        # Assigning a Subscript to a Name (line 108):
        
        # Obtaining the type of the subscript
        int_11498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 8), 'int')
        
        # Call to groups(...): (line 108)
        # Processing the call keyword arguments (line 108)
        kwargs_11501 = {}
        # Getting the type of 'match' (line 108)
        match_11499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 27), 'match', False)
        # Obtaining the member 'groups' of a type (line 108)
        groups_11500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 27), match_11499, 'groups')
        # Calling groups(args, kwargs) (line 108)
        groups_call_result_11502 = invoke(stypy.reporting.localization.Localization(__file__, 108, 27), groups_11500, *[], **kwargs_11501)
        
        # Obtaining the member '__getitem__' of a type (line 108)
        getitem___11503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 8), groups_call_result_11502, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 108)
        subscript_call_result_11504 = invoke(stypy.reporting.localization.Localization(__file__, 108, 8), getitem___11503, int_11498)
        
        # Assigning a type to the variable 'tuple_var_assignment_11392' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'tuple_var_assignment_11392', subscript_call_result_11504)
        
        # Assigning a Subscript to a Name (line 108):
        
        # Obtaining the type of the subscript
        int_11505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 8), 'int')
        
        # Call to groups(...): (line 108)
        # Processing the call keyword arguments (line 108)
        kwargs_11508 = {}
        # Getting the type of 'match' (line 108)
        match_11506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 27), 'match', False)
        # Obtaining the member 'groups' of a type (line 108)
        groups_11507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 27), match_11506, 'groups')
        # Calling groups(args, kwargs) (line 108)
        groups_call_result_11509 = invoke(stypy.reporting.localization.Localization(__file__, 108, 27), groups_11507, *[], **kwargs_11508)
        
        # Obtaining the member '__getitem__' of a type (line 108)
        getitem___11510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 8), groups_call_result_11509, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 108)
        subscript_call_result_11511 = invoke(stypy.reporting.localization.Localization(__file__, 108, 8), getitem___11510, int_11505)
        
        # Assigning a type to the variable 'tuple_var_assignment_11393' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'tuple_var_assignment_11393', subscript_call_result_11511)
        
        # Assigning a Name to a Attribute (line 108):
        # Getting the type of 'tuple_var_assignment_11392' (line 108)
        tuple_var_assignment_11392_11512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'tuple_var_assignment_11392')
        # Getting the type of 'self' (line 108)
        self_11513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'self')
        # Setting the type of the member 'name' of a type (line 108)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 8), self_11513, 'name', tuple_var_assignment_11392_11512)
        
        # Assigning a Name to a Name (line 108):
        # Getting the type of 'tuple_var_assignment_11393' (line 108)
        tuple_var_assignment_11393_11514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'tuple_var_assignment_11393')
        # Assigning a type to the variable 'paren' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 19), 'paren', tuple_var_assignment_11393_11514)
        
        # Assigning a Call to a Name (line 109):
        
        # Assigning a Call to a Name (line 109):
        
        # Call to strip(...): (line 109)
        # Processing the call keyword arguments (line 109)
        kwargs_11517 = {}
        # Getting the type of 'paren' (line 109)
        paren_11515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), 'paren', False)
        # Obtaining the member 'strip' of a type (line 109)
        strip_11516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 16), paren_11515, 'strip')
        # Calling strip(args, kwargs) (line 109)
        strip_call_result_11518 = invoke(stypy.reporting.localization.Localization(__file__, 109, 16), strip_11516, *[], **kwargs_11517)
        
        # Assigning a type to the variable 'paren' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'paren', strip_call_result_11518)
        
        # Getting the type of 'paren' (line 110)
        paren_11519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 11), 'paren')
        # Testing the type of an if condition (line 110)
        if_condition_11520 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 110, 8), paren_11519)
        # Assigning a type to the variable 'if_condition_11520' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'if_condition_11520', if_condition_11520)
        # SSA begins for if statement (line 110)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 111):
        
        # Assigning a Call to a Name (line 111):
        
        # Call to match(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'paren' (line 111)
        paren_11523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 35), 'paren', False)
        # Processing the call keyword arguments (line 111)
        kwargs_11524 = {}
        # Getting the type of 're_paren' (line 111)
        re_paren_11521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 20), 're_paren', False)
        # Obtaining the member 'match' of a type (line 111)
        match_11522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 20), re_paren_11521, 'match')
        # Calling match(args, kwargs) (line 111)
        match_call_result_11525 = invoke(stypy.reporting.localization.Localization(__file__, 111, 20), match_11522, *[paren_11523], **kwargs_11524)
        
        # Assigning a type to the variable 'match' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'match', match_call_result_11525)
        
        
        # Getting the type of 'match' (line 112)
        match_11526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 19), 'match')
        # Applying the 'not' unary operator (line 112)
        result_not__11527 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 15), 'not', match_11526)
        
        # Testing the type of an if condition (line 112)
        if_condition_11528 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 12), result_not__11527)
        # Assigning a type to the variable 'if_condition_11528' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'if_condition_11528', if_condition_11528)
        # SSA begins for if statement (line 112)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 113)
        # Processing the call arguments (line 113)
        str_11530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 33), 'str', 'expected parenthesized list: %r')
        # Getting the type of 'paren' (line 113)
        paren_11531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 69), 'paren', False)
        # Applying the binary operator '%' (line 113)
        result_mod_11532 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 33), '%', str_11530, paren_11531)
        
        # Processing the call keyword arguments (line 113)
        kwargs_11533 = {}
        # Getting the type of 'ValueError' (line 113)
        ValueError_11529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 113)
        ValueError_call_result_11534 = invoke(stypy.reporting.localization.Localization(__file__, 113, 22), ValueError_11529, *[result_mod_11532], **kwargs_11533)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 113, 16), ValueError_call_result_11534, 'raise parameter', BaseException)
        # SSA join for if statement (line 112)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 114):
        
        # Assigning a Subscript to a Name (line 114):
        
        # Obtaining the type of the subscript
        int_11535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 33), 'int')
        
        # Call to groups(...): (line 114)
        # Processing the call keyword arguments (line 114)
        kwargs_11538 = {}
        # Getting the type of 'match' (line 114)
        match_11536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 18), 'match', False)
        # Obtaining the member 'groups' of a type (line 114)
        groups_11537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 18), match_11536, 'groups')
        # Calling groups(args, kwargs) (line 114)
        groups_call_result_11539 = invoke(stypy.reporting.localization.Localization(__file__, 114, 18), groups_11537, *[], **kwargs_11538)
        
        # Obtaining the member '__getitem__' of a type (line 114)
        getitem___11540 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 18), groups_call_result_11539, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 114)
        subscript_call_result_11541 = invoke(stypy.reporting.localization.Localization(__file__, 114, 18), getitem___11540, int_11535)
        
        # Assigning a type to the variable 'str' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'str', subscript_call_result_11541)
        
        # Assigning a ListComp to a Attribute (line 115):
        
        # Assigning a ListComp to a Attribute (line 115):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to split(...): (line 115)
        # Processing the call arguments (line 115)
        str_11548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 63), 'str', ',')
        # Processing the call keyword arguments (line 115)
        kwargs_11549 = {}
        # Getting the type of 'str' (line 115)
        str_11546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 53), 'str', False)
        # Obtaining the member 'split' of a type (line 115)
        split_11547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 53), str_11546, 'split')
        # Calling split(args, kwargs) (line 115)
        split_call_result_11550 = invoke(stypy.reporting.localization.Localization(__file__, 115, 53), split_11547, *[str_11548], **kwargs_11549)
        
        comprehension_11551 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 25), split_call_result_11550)
        # Assigning a type to the variable 'aPred' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 25), 'aPred', comprehension_11551)
        
        # Call to splitUp(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'aPred' (line 115)
        aPred_11543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 33), 'aPred', False)
        # Processing the call keyword arguments (line 115)
        kwargs_11544 = {}
        # Getting the type of 'splitUp' (line 115)
        splitUp_11542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 25), 'splitUp', False)
        # Calling splitUp(args, kwargs) (line 115)
        splitUp_call_result_11545 = invoke(stypy.reporting.localization.Localization(__file__, 115, 25), splitUp_11542, *[aPred_11543], **kwargs_11544)
        
        list_11552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 25), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 25), list_11552, splitUp_call_result_11545)
        # Getting the type of 'self' (line 115)
        self_11553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'self')
        # Setting the type of the member 'pred' of a type (line 115)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 12), self_11553, 'pred', list_11552)
        
        
        # Getting the type of 'self' (line 116)
        self_11554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 19), 'self')
        # Obtaining the member 'pred' of a type (line 116)
        pred_11555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 19), self_11554, 'pred')
        # Applying the 'not' unary operator (line 116)
        result_not__11556 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 15), 'not', pred_11555)
        
        # Testing the type of an if condition (line 116)
        if_condition_11557 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 116, 12), result_not__11556)
        # Assigning a type to the variable 'if_condition_11557' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'if_condition_11557', if_condition_11557)
        # SSA begins for if statement (line 116)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 117)
        # Processing the call arguments (line 117)
        str_11559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 33), 'str', 'empty parenthesized list in %r')
        # Getting the type of 'versionPredicateStr' (line 118)
        versionPredicateStr_11560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 35), 'versionPredicateStr', False)
        # Applying the binary operator '%' (line 117)
        result_mod_11561 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 33), '%', str_11559, versionPredicateStr_11560)
        
        # Processing the call keyword arguments (line 117)
        kwargs_11562 = {}
        # Getting the type of 'ValueError' (line 117)
        ValueError_11558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 117)
        ValueError_call_result_11563 = invoke(stypy.reporting.localization.Localization(__file__, 117, 22), ValueError_11558, *[result_mod_11561], **kwargs_11562)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 117, 16), ValueError_call_result_11563, 'raise parameter', BaseException)
        # SSA join for if statement (line 116)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 110)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Attribute (line 120):
        
        # Assigning a List to a Attribute (line 120):
        
        # Obtaining an instance of the builtin type 'list' (line 120)
        list_11564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 120)
        
        # Getting the type of 'self' (line 120)
        self_11565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'self')
        # Setting the type of the member 'pred' of a type (line 120)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 12), self_11565, 'pred', list_11564)
        # SSA join for if statement (line 110)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 122, 4, False)
        # Assigning a type to the variable 'self' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VersionPredicate.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        VersionPredicate.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VersionPredicate.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        VersionPredicate.stypy__str__.__dict__.__setitem__('stypy_function_name', 'VersionPredicate.stypy__str__')
        VersionPredicate.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        VersionPredicate.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        VersionPredicate.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VersionPredicate.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        VersionPredicate.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        VersionPredicate.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VersionPredicate.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VersionPredicate.stypy__str__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__str__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__str__(...)' code ##################

        
        # Getting the type of 'self' (line 123)
        self_11566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 11), 'self')
        # Obtaining the member 'pred' of a type (line 123)
        pred_11567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 11), self_11566, 'pred')
        # Testing the type of an if condition (line 123)
        if_condition_11568 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 123, 8), pred_11567)
        # Assigning a type to the variable 'if_condition_11568' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'if_condition_11568', if_condition_11568)
        # SSA begins for if statement (line 123)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a ListComp to a Name (line 124):
        
        # Assigning a ListComp to a Name (line 124):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 124)
        self_11577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 58), 'self')
        # Obtaining the member 'pred' of a type (line 124)
        pred_11578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 58), self_11577, 'pred')
        comprehension_11579 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 19), pred_11578)
        # Assigning a type to the variable 'cond' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 19), 'cond', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 19), comprehension_11579))
        # Assigning a type to the variable 'ver' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 19), 'ver', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 19), comprehension_11579))
        # Getting the type of 'cond' (line 124)
        cond_11569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 19), 'cond')
        str_11570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 26), 'str', ' ')
        # Applying the binary operator '+' (line 124)
        result_add_11571 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 19), '+', cond_11569, str_11570)
        
        
        # Call to str(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'ver' (line 124)
        ver_11573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 36), 'ver', False)
        # Processing the call keyword arguments (line 124)
        kwargs_11574 = {}
        # Getting the type of 'str' (line 124)
        str_11572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 32), 'str', False)
        # Calling str(args, kwargs) (line 124)
        str_call_result_11575 = invoke(stypy.reporting.localization.Localization(__file__, 124, 32), str_11572, *[ver_11573], **kwargs_11574)
        
        # Applying the binary operator '+' (line 124)
        result_add_11576 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 30), '+', result_add_11571, str_call_result_11575)
        
        list_11580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 19), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 19), list_11580, result_add_11576)
        # Assigning a type to the variable 'seq' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'seq', list_11580)
        # Getting the type of 'self' (line 125)
        self_11581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 19), 'self')
        # Obtaining the member 'name' of a type (line 125)
        name_11582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 19), self_11581, 'name')
        str_11583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 31), 'str', ' (')
        # Applying the binary operator '+' (line 125)
        result_add_11584 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 19), '+', name_11582, str_11583)
        
        
        # Call to join(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'seq' (line 125)
        seq_11587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 48), 'seq', False)
        # Processing the call keyword arguments (line 125)
        kwargs_11588 = {}
        str_11585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 38), 'str', ', ')
        # Obtaining the member 'join' of a type (line 125)
        join_11586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 38), str_11585, 'join')
        # Calling join(args, kwargs) (line 125)
        join_call_result_11589 = invoke(stypy.reporting.localization.Localization(__file__, 125, 38), join_11586, *[seq_11587], **kwargs_11588)
        
        # Applying the binary operator '+' (line 125)
        result_add_11590 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 36), '+', result_add_11584, join_call_result_11589)
        
        str_11591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 55), 'str', ')')
        # Applying the binary operator '+' (line 125)
        result_add_11592 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 53), '+', result_add_11590, str_11591)
        
        # Assigning a type to the variable 'stypy_return_type' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'stypy_return_type', result_add_11592)
        # SSA branch for the else part of an if statement (line 123)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'self' (line 127)
        self_11593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 19), 'self')
        # Obtaining the member 'name' of a type (line 127)
        name_11594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 19), self_11593, 'name')
        # Assigning a type to the variable 'stypy_return_type' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'stypy_return_type', name_11594)
        # SSA join for if statement (line 123)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 122)
        stypy_return_type_11595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_11595)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_11595


    @norecursion
    def satisfied_by(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'satisfied_by'
        module_type_store = module_type_store.open_function_context('satisfied_by', 129, 4, False)
        # Assigning a type to the variable 'self' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VersionPredicate.satisfied_by.__dict__.__setitem__('stypy_localization', localization)
        VersionPredicate.satisfied_by.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VersionPredicate.satisfied_by.__dict__.__setitem__('stypy_type_store', module_type_store)
        VersionPredicate.satisfied_by.__dict__.__setitem__('stypy_function_name', 'VersionPredicate.satisfied_by')
        VersionPredicate.satisfied_by.__dict__.__setitem__('stypy_param_names_list', ['version'])
        VersionPredicate.satisfied_by.__dict__.__setitem__('stypy_varargs_param_name', None)
        VersionPredicate.satisfied_by.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VersionPredicate.satisfied_by.__dict__.__setitem__('stypy_call_defaults', defaults)
        VersionPredicate.satisfied_by.__dict__.__setitem__('stypy_call_varargs', varargs)
        VersionPredicate.satisfied_by.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VersionPredicate.satisfied_by.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VersionPredicate.satisfied_by', ['version'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'satisfied_by', localization, ['version'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'satisfied_by(...)' code ##################

        str_11596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, (-1)), 'str', 'True if version is compatible with all the predicates in self.\n        The parameter version must be acceptable to the StrictVersion\n        constructor.  It may be either a string or StrictVersion.\n        ')
        
        # Getting the type of 'self' (line 134)
        self_11597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 25), 'self')
        # Obtaining the member 'pred' of a type (line 134)
        pred_11598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 25), self_11597, 'pred')
        # Testing the type of a for loop iterable (line 134)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 134, 8), pred_11598)
        # Getting the type of the for loop variable (line 134)
        for_loop_var_11599 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 134, 8), pred_11598)
        # Assigning a type to the variable 'cond' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'cond', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 8), for_loop_var_11599))
        # Assigning a type to the variable 'ver' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'ver', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 8), for_loop_var_11599))
        # SSA begins for a for statement (line 134)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Call to (...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'version' (line 135)
        version_11604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 33), 'version', False)
        # Getting the type of 'ver' (line 135)
        ver_11605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 42), 'ver', False)
        # Processing the call keyword arguments (line 135)
        kwargs_11606 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'cond' (line 135)
        cond_11600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 27), 'cond', False)
        # Getting the type of 'compmap' (line 135)
        compmap_11601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 19), 'compmap', False)
        # Obtaining the member '__getitem__' of a type (line 135)
        getitem___11602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 19), compmap_11601, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 135)
        subscript_call_result_11603 = invoke(stypy.reporting.localization.Localization(__file__, 135, 19), getitem___11602, cond_11600)
        
        # Calling (args, kwargs) (line 135)
        _call_result_11607 = invoke(stypy.reporting.localization.Localization(__file__, 135, 19), subscript_call_result_11603, *[version_11604, ver_11605], **kwargs_11606)
        
        # Applying the 'not' unary operator (line 135)
        result_not__11608 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 15), 'not', _call_result_11607)
        
        # Testing the type of an if condition (line 135)
        if_condition_11609 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 135, 12), result_not__11608)
        # Assigning a type to the variable 'if_condition_11609' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'if_condition_11609', if_condition_11609)
        # SSA begins for if statement (line 135)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 136)
        False_11610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 23), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 16), 'stypy_return_type', False_11610)
        # SSA join for if statement (line 135)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'True' (line 137)
        True_11611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'stypy_return_type', True_11611)
        
        # ################# End of 'satisfied_by(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'satisfied_by' in the type store
        # Getting the type of 'stypy_return_type' (line 129)
        stypy_return_type_11612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_11612)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'satisfied_by'
        return stypy_return_type_11612


# Assigning a type to the variable 'VersionPredicate' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'VersionPredicate', VersionPredicate)

# Assigning a Name to a Name (line 140):

# Assigning a Name to a Name (line 140):
# Getting the type of 'None' (line 140)
None_11613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 16), 'None')
# Assigning a type to the variable '_provision_rx' (line 140)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 0), '_provision_rx', None_11613)

@norecursion
def split_provision(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'split_provision'
    module_type_store = module_type_store.open_function_context('split_provision', 142, 0, False)
    
    # Passed parameters checking function
    split_provision.stypy_localization = localization
    split_provision.stypy_type_of_self = None
    split_provision.stypy_type_store = module_type_store
    split_provision.stypy_function_name = 'split_provision'
    split_provision.stypy_param_names_list = ['value']
    split_provision.stypy_varargs_param_name = None
    split_provision.stypy_kwargs_param_name = None
    split_provision.stypy_call_defaults = defaults
    split_provision.stypy_call_varargs = varargs
    split_provision.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'split_provision', ['value'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'split_provision', localization, ['value'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'split_provision(...)' code ##################

    str_11614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, (-1)), 'str', "Return the name and optional version number of a provision.\n\n    The version number, if given, will be returned as a `StrictVersion`\n    instance, otherwise it will be `None`.\n\n    >>> split_provision('mypkg')\n    ('mypkg', None)\n    >>> split_provision(' mypkg( 1.2 ) ')\n    ('mypkg', StrictVersion ('1.2'))\n    ")
    # Marking variables as global (line 153)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 153, 4), '_provision_rx')
    
    # Type idiom detected: calculating its left and rigth part (line 154)
    # Getting the type of '_provision_rx' (line 154)
    _provision_rx_11615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 7), '_provision_rx')
    # Getting the type of 'None' (line 154)
    None_11616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 24), 'None')
    
    (may_be_11617, more_types_in_union_11618) = may_be_none(_provision_rx_11615, None_11616)

    if may_be_11617:

        if more_types_in_union_11618:
            # Runtime conditional SSA (line 154)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 155):
        
        # Assigning a Call to a Name (line 155):
        
        # Call to compile(...): (line 155)
        # Processing the call arguments (line 155)
        str_11621 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 12), 'str', '([a-zA-Z_]\\w*(?:\\.[a-zA-Z_]\\w*)*)(?:\\s*\\(\\s*([^)\\s]+)\\s*\\))?$')
        # Processing the call keyword arguments (line 155)
        kwargs_11622 = {}
        # Getting the type of 're' (line 155)
        re_11619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 24), 're', False)
        # Obtaining the member 'compile' of a type (line 155)
        compile_11620 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 24), re_11619, 'compile')
        # Calling compile(args, kwargs) (line 155)
        compile_call_result_11623 = invoke(stypy.reporting.localization.Localization(__file__, 155, 24), compile_11620, *[str_11621], **kwargs_11622)
        
        # Assigning a type to the variable '_provision_rx' (line 155)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 8), '_provision_rx', compile_call_result_11623)

        if more_types_in_union_11618:
            # SSA join for if statement (line 154)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 157):
    
    # Assigning a Call to a Name (line 157):
    
    # Call to strip(...): (line 157)
    # Processing the call keyword arguments (line 157)
    kwargs_11626 = {}
    # Getting the type of 'value' (line 157)
    value_11624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'value', False)
    # Obtaining the member 'strip' of a type (line 157)
    strip_11625 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 12), value_11624, 'strip')
    # Calling strip(args, kwargs) (line 157)
    strip_call_result_11627 = invoke(stypy.reporting.localization.Localization(__file__, 157, 12), strip_11625, *[], **kwargs_11626)
    
    # Assigning a type to the variable 'value' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'value', strip_call_result_11627)
    
    # Assigning a Call to a Name (line 158):
    
    # Assigning a Call to a Name (line 158):
    
    # Call to match(...): (line 158)
    # Processing the call arguments (line 158)
    # Getting the type of 'value' (line 158)
    value_11630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 28), 'value', False)
    # Processing the call keyword arguments (line 158)
    kwargs_11631 = {}
    # Getting the type of '_provision_rx' (line 158)
    _provision_rx_11628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), '_provision_rx', False)
    # Obtaining the member 'match' of a type (line 158)
    match_11629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), _provision_rx_11628, 'match')
    # Calling match(args, kwargs) (line 158)
    match_call_result_11632 = invoke(stypy.reporting.localization.Localization(__file__, 158, 8), match_11629, *[value_11630], **kwargs_11631)
    
    # Assigning a type to the variable 'm' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'm', match_call_result_11632)
    
    
    # Getting the type of 'm' (line 159)
    m_11633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 11), 'm')
    # Applying the 'not' unary operator (line 159)
    result_not__11634 = python_operator(stypy.reporting.localization.Localization(__file__, 159, 7), 'not', m_11633)
    
    # Testing the type of an if condition (line 159)
    if_condition_11635 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 4), result_not__11634)
    # Assigning a type to the variable 'if_condition_11635' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'if_condition_11635', if_condition_11635)
    # SSA begins for if statement (line 159)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 160)
    # Processing the call arguments (line 160)
    str_11637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 25), 'str', 'illegal provides specification: %r')
    # Getting the type of 'value' (line 160)
    value_11638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 64), 'value', False)
    # Applying the binary operator '%' (line 160)
    result_mod_11639 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 25), '%', str_11637, value_11638)
    
    # Processing the call keyword arguments (line 160)
    kwargs_11640 = {}
    # Getting the type of 'ValueError' (line 160)
    ValueError_11636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 160)
    ValueError_call_result_11641 = invoke(stypy.reporting.localization.Localization(__file__, 160, 14), ValueError_11636, *[result_mod_11639], **kwargs_11640)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 160, 8), ValueError_call_result_11641, 'raise parameter', BaseException)
    # SSA join for if statement (line 159)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BoolOp to a Name (line 161):
    
    # Assigning a BoolOp to a Name (line 161):
    
    # Evaluating a boolean operation
    
    # Call to group(...): (line 161)
    # Processing the call arguments (line 161)
    int_11644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 18), 'int')
    # Processing the call keyword arguments (line 161)
    kwargs_11645 = {}
    # Getting the type of 'm' (line 161)
    m_11642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 10), 'm', False)
    # Obtaining the member 'group' of a type (line 161)
    group_11643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 10), m_11642, 'group')
    # Calling group(args, kwargs) (line 161)
    group_call_result_11646 = invoke(stypy.reporting.localization.Localization(__file__, 161, 10), group_11643, *[int_11644], **kwargs_11645)
    
    # Getting the type of 'None' (line 161)
    None_11647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 24), 'None')
    # Applying the binary operator 'or' (line 161)
    result_or_keyword_11648 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 10), 'or', group_call_result_11646, None_11647)
    
    # Assigning a type to the variable 'ver' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'ver', result_or_keyword_11648)
    
    # Getting the type of 'ver' (line 162)
    ver_11649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 7), 'ver')
    # Testing the type of an if condition (line 162)
    if_condition_11650 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 162, 4), ver_11649)
    # Assigning a type to the variable 'if_condition_11650' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'if_condition_11650', if_condition_11650)
    # SSA begins for if statement (line 162)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 163):
    
    # Assigning a Call to a Name (line 163):
    
    # Call to StrictVersion(...): (line 163)
    # Processing the call arguments (line 163)
    # Getting the type of 'ver' (line 163)
    ver_11654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 46), 'ver', False)
    # Processing the call keyword arguments (line 163)
    kwargs_11655 = {}
    # Getting the type of 'distutils' (line 163)
    distutils_11651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 14), 'distutils', False)
    # Obtaining the member 'version' of a type (line 163)
    version_11652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 14), distutils_11651, 'version')
    # Obtaining the member 'StrictVersion' of a type (line 163)
    StrictVersion_11653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 14), version_11652, 'StrictVersion')
    # Calling StrictVersion(args, kwargs) (line 163)
    StrictVersion_call_result_11656 = invoke(stypy.reporting.localization.Localization(__file__, 163, 14), StrictVersion_11653, *[ver_11654], **kwargs_11655)
    
    # Assigning a type to the variable 'ver' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'ver', StrictVersion_call_result_11656)
    # SSA join for if statement (line 162)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 164)
    tuple_11657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 164)
    # Adding element type (line 164)
    
    # Call to group(...): (line 164)
    # Processing the call arguments (line 164)
    int_11660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 19), 'int')
    # Processing the call keyword arguments (line 164)
    kwargs_11661 = {}
    # Getting the type of 'm' (line 164)
    m_11658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 11), 'm', False)
    # Obtaining the member 'group' of a type (line 164)
    group_11659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 11), m_11658, 'group')
    # Calling group(args, kwargs) (line 164)
    group_call_result_11662 = invoke(stypy.reporting.localization.Localization(__file__, 164, 11), group_11659, *[int_11660], **kwargs_11661)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 11), tuple_11657, group_call_result_11662)
    # Adding element type (line 164)
    # Getting the type of 'ver' (line 164)
    ver_11663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 23), 'ver')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 11), tuple_11657, ver_11663)
    
    # Assigning a type to the variable 'stypy_return_type' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'stypy_return_type', tuple_11657)
    
    # ################# End of 'split_provision(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'split_provision' in the type store
    # Getting the type of 'stypy_return_type' (line 142)
    stypy_return_type_11664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_11664)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'split_provision'
    return stypy_return_type_11664

# Assigning a type to the variable 'split_provision' (line 142)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 0), 'split_provision', split_provision)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
