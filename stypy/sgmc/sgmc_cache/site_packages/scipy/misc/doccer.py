
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Utilities to allow inserting docstring fragments for common
2: parameters into function and method docstrings'''
3: 
4: from __future__ import division, print_function, absolute_import
5: 
6: import sys
7: 
8: __all__ = ['docformat', 'inherit_docstring_from', 'indentcount_lines',
9:            'filldoc', 'unindent_dict', 'unindent_string']
10: 
11: 
12: def docformat(docstring, docdict=None):
13:     ''' Fill a function docstring from variables in dictionary
14: 
15:     Adapt the indent of the inserted docs
16: 
17:     Parameters
18:     ----------
19:     docstring : string
20:         docstring from function, possibly with dict formatting strings
21:     docdict : dict, optional
22:         dictionary with keys that match the dict formatting strings
23:         and values that are docstring fragments to be inserted.  The
24:         indentation of the inserted docstrings is set to match the
25:         minimum indentation of the ``docstring`` by adding this
26:         indentation to all lines of the inserted string, except the
27:         first
28: 
29:     Returns
30:     -------
31:     outstring : string
32:         string with requested ``docdict`` strings inserted
33: 
34:     Examples
35:     --------
36:     >>> docformat(' Test string with %(value)s', {'value':'inserted value'})
37:     ' Test string with inserted value'
38:     >>> docstring = 'First line\\n    Second line\\n    %(value)s'
39:     >>> inserted_string = "indented\\nstring"
40:     >>> docdict = {'value': inserted_string}
41:     >>> docformat(docstring, docdict)
42:     'First line\\n    Second line\\n    indented\\n    string'
43:     '''
44:     if not docstring:
45:         return docstring
46:     if docdict is None:
47:         docdict = {}
48:     if not docdict:
49:         return docstring
50:     lines = docstring.expandtabs().splitlines()
51:     # Find the minimum indent of the main docstring, after first line
52:     if len(lines) < 2:
53:         icount = 0
54:     else:
55:         icount = indentcount_lines(lines[1:])
56:     indent = ' ' * icount
57:     # Insert this indent to dictionary docstrings
58:     indented = {}
59:     for name, dstr in docdict.items():
60:         lines = dstr.expandtabs().splitlines()
61:         try:
62:             newlines = [lines[0]]
63:             for line in lines[1:]:
64:                 newlines.append(indent+line)
65:             indented[name] = '\n'.join(newlines)
66:         except IndexError:
67:             indented[name] = dstr
68:     return docstring % indented
69: 
70: 
71: def inherit_docstring_from(cls):
72:     '''
73:     This decorator modifies the decorated function's docstring by
74:     replacing occurrences of '%(super)s' with the docstring of the
75:     method of the same name from the class `cls`.
76: 
77:     If the decorated method has no docstring, it is simply given the
78:     docstring of `cls`s method.
79: 
80:     Parameters
81:     ----------
82:     cls : Python class or instance
83:         A class with a method with the same name as the decorated method.
84:         The docstring of the method in this class replaces '%(super)s' in the
85:         docstring of the decorated method.
86: 
87:     Returns
88:     -------
89:     f : function
90:         The decorator function that modifies the __doc__ attribute
91:         of its argument.
92: 
93:     Examples
94:     --------
95:     In the following, the docstring for Bar.func created using the
96:     docstring of `Foo.func`.
97: 
98:     >>> class Foo(object):
99:     ...     def func(self):
100:     ...         '''Do something useful.'''
101:     ...         return
102:     ...
103:     >>> class Bar(Foo):
104:     ...     @inherit_docstring_from(Foo)
105:     ...     def func(self):
106:     ...         '''%(super)s
107:     ...         Do it fast.
108:     ...         '''
109:     ...         return
110:     ...
111:     >>> b = Bar()
112:     >>> b.func.__doc__
113:     'Do something useful.\n        Do it fast.\n        '
114: 
115:     '''
116:     def _doc(func):
117:         cls_docstring = getattr(cls, func.__name__).__doc__
118:         func_docstring = func.__doc__
119:         if func_docstring is None:
120:             func.__doc__ = cls_docstring
121:         else:
122:             new_docstring = func_docstring % dict(super=cls_docstring)
123:             func.__doc__ = new_docstring
124:         return func
125:     return _doc
126: 
127: 
128: def indentcount_lines(lines):
129:     ''' Minimum indent for all lines in line list
130: 
131:     >>> lines = [' one', '  two', '   three']
132:     >>> indentcount_lines(lines)
133:     1
134:     >>> lines = []
135:     >>> indentcount_lines(lines)
136:     0
137:     >>> lines = [' one']
138:     >>> indentcount_lines(lines)
139:     1
140:     >>> indentcount_lines(['    '])
141:     0
142:     '''
143:     indentno = sys.maxsize
144:     for line in lines:
145:         stripped = line.lstrip()
146:         if stripped:
147:             indentno = min(indentno, len(line) - len(stripped))
148:     if indentno == sys.maxsize:
149:         return 0
150:     return indentno
151: 
152: 
153: def filldoc(docdict, unindent_params=True):
154:     ''' Return docstring decorator using docdict variable dictionary
155: 
156:     Parameters
157:     ----------
158:     docdict : dictionary
159:         dictionary containing name, docstring fragment pairs
160:     unindent_params : {False, True}, boolean, optional
161:         If True, strip common indentation from all parameters in
162:         docdict
163: 
164:     Returns
165:     -------
166:     decfunc : function
167:         decorator that applies dictionary to input function docstring
168: 
169:     '''
170:     if unindent_params:
171:         docdict = unindent_dict(docdict)
172: 
173:     def decorate(f):
174:         f.__doc__ = docformat(f.__doc__, docdict)
175:         return f
176:     return decorate
177: 
178: 
179: def unindent_dict(docdict):
180:     ''' Unindent all strings in a docdict '''
181:     can_dict = {}
182:     for name, dstr in docdict.items():
183:         can_dict[name] = unindent_string(dstr)
184:     return can_dict
185: 
186: 
187: def unindent_string(docstring):
188:     ''' Set docstring to minimum indent for all lines, including first
189: 
190:     >>> unindent_string(' two')
191:     'two'
192:     >>> unindent_string('  two\\n   three')
193:     'two\\n three'
194:     '''
195:     lines = docstring.expandtabs().splitlines()
196:     icount = indentcount_lines(lines)
197:     if icount == 0:
198:         return docstring
199:     return '\n'.join([line[icount:] for line in lines])
200: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_114023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 2, (-1)), 'str', ' Utilities to allow inserting docstring fragments for common\nparameters into function and method docstrings')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import sys' statement (line 6)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'sys', sys, module_type_store)


# Assigning a List to a Name (line 8):
__all__ = ['docformat', 'inherit_docstring_from', 'indentcount_lines', 'filldoc', 'unindent_dict', 'unindent_string']
module_type_store.set_exportable_members(['docformat', 'inherit_docstring_from', 'indentcount_lines', 'filldoc', 'unindent_dict', 'unindent_string'])

# Obtaining an instance of the builtin type 'list' (line 8)
list_114024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 8)
# Adding element type (line 8)
str_114025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 11), 'str', 'docformat')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_114024, str_114025)
# Adding element type (line 8)
str_114026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 24), 'str', 'inherit_docstring_from')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_114024, str_114026)
# Adding element type (line 8)
str_114027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 50), 'str', 'indentcount_lines')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_114024, str_114027)
# Adding element type (line 8)
str_114028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 11), 'str', 'filldoc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_114024, str_114028)
# Adding element type (line 8)
str_114029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 22), 'str', 'unindent_dict')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_114024, str_114029)
# Adding element type (line 8)
str_114030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 39), 'str', 'unindent_string')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 8, 10), list_114024, str_114030)

# Assigning a type to the variable '__all__' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), '__all__', list_114024)

@norecursion
def docformat(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 12)
    None_114031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 33), 'None')
    defaults = [None_114031]
    # Create a new context for function 'docformat'
    module_type_store = module_type_store.open_function_context('docformat', 12, 0, False)
    
    # Passed parameters checking function
    docformat.stypy_localization = localization
    docformat.stypy_type_of_self = None
    docformat.stypy_type_store = module_type_store
    docformat.stypy_function_name = 'docformat'
    docformat.stypy_param_names_list = ['docstring', 'docdict']
    docformat.stypy_varargs_param_name = None
    docformat.stypy_kwargs_param_name = None
    docformat.stypy_call_defaults = defaults
    docformat.stypy_call_varargs = varargs
    docformat.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'docformat', ['docstring', 'docdict'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'docformat', localization, ['docstring', 'docdict'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'docformat(...)' code ##################

    str_114032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, (-1)), 'str', ' Fill a function docstring from variables in dictionary\n\n    Adapt the indent of the inserted docs\n\n    Parameters\n    ----------\n    docstring : string\n        docstring from function, possibly with dict formatting strings\n    docdict : dict, optional\n        dictionary with keys that match the dict formatting strings\n        and values that are docstring fragments to be inserted.  The\n        indentation of the inserted docstrings is set to match the\n        minimum indentation of the ``docstring`` by adding this\n        indentation to all lines of the inserted string, except the\n        first\n\n    Returns\n    -------\n    outstring : string\n        string with requested ``docdict`` strings inserted\n\n    Examples\n    --------\n    >>> docformat(\' Test string with %(value)s\', {\'value\':\'inserted value\'})\n    \' Test string with inserted value\'\n    >>> docstring = \'First line\\n    Second line\\n    %(value)s\'\n    >>> inserted_string = "indented\\nstring"\n    >>> docdict = {\'value\': inserted_string}\n    >>> docformat(docstring, docdict)\n    \'First line\\n    Second line\\n    indented\\n    string\'\n    ')
    
    
    # Getting the type of 'docstring' (line 44)
    docstring_114033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 11), 'docstring')
    # Applying the 'not' unary operator (line 44)
    result_not__114034 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 7), 'not', docstring_114033)
    
    # Testing the type of an if condition (line 44)
    if_condition_114035 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 44, 4), result_not__114034)
    # Assigning a type to the variable 'if_condition_114035' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'if_condition_114035', if_condition_114035)
    # SSA begins for if statement (line 44)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'docstring' (line 45)
    docstring_114036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'docstring')
    # Assigning a type to the variable 'stypy_return_type' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'stypy_return_type', docstring_114036)
    # SSA join for if statement (line 44)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 46)
    # Getting the type of 'docdict' (line 46)
    docdict_114037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 7), 'docdict')
    # Getting the type of 'None' (line 46)
    None_114038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 18), 'None')
    
    (may_be_114039, more_types_in_union_114040) = may_be_none(docdict_114037, None_114038)

    if may_be_114039:

        if more_types_in_union_114040:
            # Runtime conditional SSA (line 46)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Dict to a Name (line 47):
        
        # Obtaining an instance of the builtin type 'dict' (line 47)
        dict_114041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 18), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 47)
        
        # Assigning a type to the variable 'docdict' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'docdict', dict_114041)

        if more_types_in_union_114040:
            # SSA join for if statement (line 46)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'docdict' (line 48)
    docdict_114042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 11), 'docdict')
    # Applying the 'not' unary operator (line 48)
    result_not__114043 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 7), 'not', docdict_114042)
    
    # Testing the type of an if condition (line 48)
    if_condition_114044 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 48, 4), result_not__114043)
    # Assigning a type to the variable 'if_condition_114044' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'if_condition_114044', if_condition_114044)
    # SSA begins for if statement (line 48)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'docstring' (line 49)
    docstring_114045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 15), 'docstring')
    # Assigning a type to the variable 'stypy_return_type' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'stypy_return_type', docstring_114045)
    # SSA join for if statement (line 48)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 50):
    
    # Call to splitlines(...): (line 50)
    # Processing the call keyword arguments (line 50)
    kwargs_114051 = {}
    
    # Call to expandtabs(...): (line 50)
    # Processing the call keyword arguments (line 50)
    kwargs_114048 = {}
    # Getting the type of 'docstring' (line 50)
    docstring_114046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'docstring', False)
    # Obtaining the member 'expandtabs' of a type (line 50)
    expandtabs_114047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 12), docstring_114046, 'expandtabs')
    # Calling expandtabs(args, kwargs) (line 50)
    expandtabs_call_result_114049 = invoke(stypy.reporting.localization.Localization(__file__, 50, 12), expandtabs_114047, *[], **kwargs_114048)
    
    # Obtaining the member 'splitlines' of a type (line 50)
    splitlines_114050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 12), expandtabs_call_result_114049, 'splitlines')
    # Calling splitlines(args, kwargs) (line 50)
    splitlines_call_result_114052 = invoke(stypy.reporting.localization.Localization(__file__, 50, 12), splitlines_114050, *[], **kwargs_114051)
    
    # Assigning a type to the variable 'lines' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'lines', splitlines_call_result_114052)
    
    
    
    # Call to len(...): (line 52)
    # Processing the call arguments (line 52)
    # Getting the type of 'lines' (line 52)
    lines_114054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 11), 'lines', False)
    # Processing the call keyword arguments (line 52)
    kwargs_114055 = {}
    # Getting the type of 'len' (line 52)
    len_114053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 7), 'len', False)
    # Calling len(args, kwargs) (line 52)
    len_call_result_114056 = invoke(stypy.reporting.localization.Localization(__file__, 52, 7), len_114053, *[lines_114054], **kwargs_114055)
    
    int_114057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 20), 'int')
    # Applying the binary operator '<' (line 52)
    result_lt_114058 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 7), '<', len_call_result_114056, int_114057)
    
    # Testing the type of an if condition (line 52)
    if_condition_114059 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 52, 4), result_lt_114058)
    # Assigning a type to the variable 'if_condition_114059' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'if_condition_114059', if_condition_114059)
    # SSA begins for if statement (line 52)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 53):
    int_114060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 17), 'int')
    # Assigning a type to the variable 'icount' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'icount', int_114060)
    # SSA branch for the else part of an if statement (line 52)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 55):
    
    # Call to indentcount_lines(...): (line 55)
    # Processing the call arguments (line 55)
    
    # Obtaining the type of the subscript
    int_114062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 41), 'int')
    slice_114063 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 55, 35), int_114062, None, None)
    # Getting the type of 'lines' (line 55)
    lines_114064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 35), 'lines', False)
    # Obtaining the member '__getitem__' of a type (line 55)
    getitem___114065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 35), lines_114064, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 55)
    subscript_call_result_114066 = invoke(stypy.reporting.localization.Localization(__file__, 55, 35), getitem___114065, slice_114063)
    
    # Processing the call keyword arguments (line 55)
    kwargs_114067 = {}
    # Getting the type of 'indentcount_lines' (line 55)
    indentcount_lines_114061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 17), 'indentcount_lines', False)
    # Calling indentcount_lines(args, kwargs) (line 55)
    indentcount_lines_call_result_114068 = invoke(stypy.reporting.localization.Localization(__file__, 55, 17), indentcount_lines_114061, *[subscript_call_result_114066], **kwargs_114067)
    
    # Assigning a type to the variable 'icount' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'icount', indentcount_lines_call_result_114068)
    # SSA join for if statement (line 52)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 56):
    str_114069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 13), 'str', ' ')
    # Getting the type of 'icount' (line 56)
    icount_114070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 19), 'icount')
    # Applying the binary operator '*' (line 56)
    result_mul_114071 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 13), '*', str_114069, icount_114070)
    
    # Assigning a type to the variable 'indent' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'indent', result_mul_114071)
    
    # Assigning a Dict to a Name (line 58):
    
    # Obtaining an instance of the builtin type 'dict' (line 58)
    dict_114072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 15), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 58)
    
    # Assigning a type to the variable 'indented' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'indented', dict_114072)
    
    
    # Call to items(...): (line 59)
    # Processing the call keyword arguments (line 59)
    kwargs_114075 = {}
    # Getting the type of 'docdict' (line 59)
    docdict_114073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 22), 'docdict', False)
    # Obtaining the member 'items' of a type (line 59)
    items_114074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 22), docdict_114073, 'items')
    # Calling items(args, kwargs) (line 59)
    items_call_result_114076 = invoke(stypy.reporting.localization.Localization(__file__, 59, 22), items_114074, *[], **kwargs_114075)
    
    # Testing the type of a for loop iterable (line 59)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 59, 4), items_call_result_114076)
    # Getting the type of the for loop variable (line 59)
    for_loop_var_114077 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 59, 4), items_call_result_114076)
    # Assigning a type to the variable 'name' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 4), for_loop_var_114077))
    # Assigning a type to the variable 'dstr' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'dstr', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 4), for_loop_var_114077))
    # SSA begins for a for statement (line 59)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 60):
    
    # Call to splitlines(...): (line 60)
    # Processing the call keyword arguments (line 60)
    kwargs_114083 = {}
    
    # Call to expandtabs(...): (line 60)
    # Processing the call keyword arguments (line 60)
    kwargs_114080 = {}
    # Getting the type of 'dstr' (line 60)
    dstr_114078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'dstr', False)
    # Obtaining the member 'expandtabs' of a type (line 60)
    expandtabs_114079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 16), dstr_114078, 'expandtabs')
    # Calling expandtabs(args, kwargs) (line 60)
    expandtabs_call_result_114081 = invoke(stypy.reporting.localization.Localization(__file__, 60, 16), expandtabs_114079, *[], **kwargs_114080)
    
    # Obtaining the member 'splitlines' of a type (line 60)
    splitlines_114082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 16), expandtabs_call_result_114081, 'splitlines')
    # Calling splitlines(args, kwargs) (line 60)
    splitlines_call_result_114084 = invoke(stypy.reporting.localization.Localization(__file__, 60, 16), splitlines_114082, *[], **kwargs_114083)
    
    # Assigning a type to the variable 'lines' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'lines', splitlines_call_result_114084)
    
    
    # SSA begins for try-except statement (line 61)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a List to a Name (line 62):
    
    # Obtaining an instance of the builtin type 'list' (line 62)
    list_114085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 23), 'list')
    # Adding type elements to the builtin type 'list' instance (line 62)
    # Adding element type (line 62)
    
    # Obtaining the type of the subscript
    int_114086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 30), 'int')
    # Getting the type of 'lines' (line 62)
    lines_114087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 24), 'lines')
    # Obtaining the member '__getitem__' of a type (line 62)
    getitem___114088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 24), lines_114087, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 62)
    subscript_call_result_114089 = invoke(stypy.reporting.localization.Localization(__file__, 62, 24), getitem___114088, int_114086)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 23), list_114085, subscript_call_result_114089)
    
    # Assigning a type to the variable 'newlines' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'newlines', list_114085)
    
    
    # Obtaining the type of the subscript
    int_114090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 30), 'int')
    slice_114091 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 63, 24), int_114090, None, None)
    # Getting the type of 'lines' (line 63)
    lines_114092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 24), 'lines')
    # Obtaining the member '__getitem__' of a type (line 63)
    getitem___114093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 24), lines_114092, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 63)
    subscript_call_result_114094 = invoke(stypy.reporting.localization.Localization(__file__, 63, 24), getitem___114093, slice_114091)
    
    # Testing the type of a for loop iterable (line 63)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 63, 12), subscript_call_result_114094)
    # Getting the type of the for loop variable (line 63)
    for_loop_var_114095 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 63, 12), subscript_call_result_114094)
    # Assigning a type to the variable 'line' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'line', for_loop_var_114095)
    # SSA begins for a for statement (line 63)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'indent' (line 64)
    indent_114098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 32), 'indent', False)
    # Getting the type of 'line' (line 64)
    line_114099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 39), 'line', False)
    # Applying the binary operator '+' (line 64)
    result_add_114100 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 32), '+', indent_114098, line_114099)
    
    # Processing the call keyword arguments (line 64)
    kwargs_114101 = {}
    # Getting the type of 'newlines' (line 64)
    newlines_114096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 16), 'newlines', False)
    # Obtaining the member 'append' of a type (line 64)
    append_114097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 16), newlines_114096, 'append')
    # Calling append(args, kwargs) (line 64)
    append_call_result_114102 = invoke(stypy.reporting.localization.Localization(__file__, 64, 16), append_114097, *[result_add_114100], **kwargs_114101)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Subscript (line 65):
    
    # Call to join(...): (line 65)
    # Processing the call arguments (line 65)
    # Getting the type of 'newlines' (line 65)
    newlines_114105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 39), 'newlines', False)
    # Processing the call keyword arguments (line 65)
    kwargs_114106 = {}
    str_114103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 29), 'str', '\n')
    # Obtaining the member 'join' of a type (line 65)
    join_114104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 29), str_114103, 'join')
    # Calling join(args, kwargs) (line 65)
    join_call_result_114107 = invoke(stypy.reporting.localization.Localization(__file__, 65, 29), join_114104, *[newlines_114105], **kwargs_114106)
    
    # Getting the type of 'indented' (line 65)
    indented_114108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'indented')
    # Getting the type of 'name' (line 65)
    name_114109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 21), 'name')
    # Storing an element on a container (line 65)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 12), indented_114108, (name_114109, join_call_result_114107))
    # SSA branch for the except part of a try statement (line 61)
    # SSA branch for the except 'IndexError' branch of a try statement (line 61)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Name to a Subscript (line 67):
    # Getting the type of 'dstr' (line 67)
    dstr_114110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 29), 'dstr')
    # Getting the type of 'indented' (line 67)
    indented_114111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'indented')
    # Getting the type of 'name' (line 67)
    name_114112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 21), 'name')
    # Storing an element on a container (line 67)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 12), indented_114111, (name_114112, dstr_114110))
    # SSA join for try-except statement (line 61)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'docstring' (line 68)
    docstring_114113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 11), 'docstring')
    # Getting the type of 'indented' (line 68)
    indented_114114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 23), 'indented')
    # Applying the binary operator '%' (line 68)
    result_mod_114115 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 11), '%', docstring_114113, indented_114114)
    
    # Assigning a type to the variable 'stypy_return_type' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'stypy_return_type', result_mod_114115)
    
    # ################# End of 'docformat(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'docformat' in the type store
    # Getting the type of 'stypy_return_type' (line 12)
    stypy_return_type_114116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_114116)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'docformat'
    return stypy_return_type_114116

# Assigning a type to the variable 'docformat' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'docformat', docformat)

@norecursion
def inherit_docstring_from(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'inherit_docstring_from'
    module_type_store = module_type_store.open_function_context('inherit_docstring_from', 71, 0, False)
    
    # Passed parameters checking function
    inherit_docstring_from.stypy_localization = localization
    inherit_docstring_from.stypy_type_of_self = None
    inherit_docstring_from.stypy_type_store = module_type_store
    inherit_docstring_from.stypy_function_name = 'inherit_docstring_from'
    inherit_docstring_from.stypy_param_names_list = ['cls']
    inherit_docstring_from.stypy_varargs_param_name = None
    inherit_docstring_from.stypy_kwargs_param_name = None
    inherit_docstring_from.stypy_call_defaults = defaults
    inherit_docstring_from.stypy_call_varargs = varargs
    inherit_docstring_from.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'inherit_docstring_from', ['cls'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'inherit_docstring_from', localization, ['cls'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'inherit_docstring_from(...)' code ##################

    str_114117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, (-1)), 'str', "\n    This decorator modifies the decorated function's docstring by\n    replacing occurrences of '%(super)s' with the docstring of the\n    method of the same name from the class `cls`.\n\n    If the decorated method has no docstring, it is simply given the\n    docstring of `cls`s method.\n\n    Parameters\n    ----------\n    cls : Python class or instance\n        A class with a method with the same name as the decorated method.\n        The docstring of the method in this class replaces '%(super)s' in the\n        docstring of the decorated method.\n\n    Returns\n    -------\n    f : function\n        The decorator function that modifies the __doc__ attribute\n        of its argument.\n\n    Examples\n    --------\n    In the following, the docstring for Bar.func created using the\n    docstring of `Foo.func`.\n\n    >>> class Foo(object):\n    ...     def func(self):\n    ...         '''Do something useful.'''\n    ...         return\n    ...\n    >>> class Bar(Foo):\n    ...     @inherit_docstring_from(Foo)\n    ...     def func(self):\n    ...         '''%(super)s\n    ...         Do it fast.\n    ...         '''\n    ...         return\n    ...\n    >>> b = Bar()\n    >>> b.func.__doc__\n    'Do something useful.\n        Do it fast.\n        '\n\n    ")

    @norecursion
    def _doc(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_doc'
        module_type_store = module_type_store.open_function_context('_doc', 116, 4, False)
        
        # Passed parameters checking function
        _doc.stypy_localization = localization
        _doc.stypy_type_of_self = None
        _doc.stypy_type_store = module_type_store
        _doc.stypy_function_name = '_doc'
        _doc.stypy_param_names_list = ['func']
        _doc.stypy_varargs_param_name = None
        _doc.stypy_kwargs_param_name = None
        _doc.stypy_call_defaults = defaults
        _doc.stypy_call_varargs = varargs
        _doc.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_doc', ['func'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_doc', localization, ['func'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_doc(...)' code ##################

        
        # Assigning a Attribute to a Name (line 117):
        
        # Call to getattr(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'cls' (line 117)
        cls_114119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 32), 'cls', False)
        # Getting the type of 'func' (line 117)
        func_114120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 37), 'func', False)
        # Obtaining the member '__name__' of a type (line 117)
        name___114121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 37), func_114120, '__name__')
        # Processing the call keyword arguments (line 117)
        kwargs_114122 = {}
        # Getting the type of 'getattr' (line 117)
        getattr_114118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 24), 'getattr', False)
        # Calling getattr(args, kwargs) (line 117)
        getattr_call_result_114123 = invoke(stypy.reporting.localization.Localization(__file__, 117, 24), getattr_114118, *[cls_114119, name___114121], **kwargs_114122)
        
        # Obtaining the member '__doc__' of a type (line 117)
        doc___114124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 24), getattr_call_result_114123, '__doc__')
        # Assigning a type to the variable 'cls_docstring' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'cls_docstring', doc___114124)
        
        # Assigning a Attribute to a Name (line 118):
        # Getting the type of 'func' (line 118)
        func_114125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 25), 'func')
        # Obtaining the member '__doc__' of a type (line 118)
        doc___114126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 25), func_114125, '__doc__')
        # Assigning a type to the variable 'func_docstring' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'func_docstring', doc___114126)
        
        # Type idiom detected: calculating its left and rigth part (line 119)
        # Getting the type of 'func_docstring' (line 119)
        func_docstring_114127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 11), 'func_docstring')
        # Getting the type of 'None' (line 119)
        None_114128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 29), 'None')
        
        (may_be_114129, more_types_in_union_114130) = may_be_none(func_docstring_114127, None_114128)

        if may_be_114129:

            if more_types_in_union_114130:
                # Runtime conditional SSA (line 119)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Attribute (line 120):
            # Getting the type of 'cls_docstring' (line 120)
            cls_docstring_114131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 27), 'cls_docstring')
            # Getting the type of 'func' (line 120)
            func_114132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'func')
            # Setting the type of the member '__doc__' of a type (line 120)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 12), func_114132, '__doc__', cls_docstring_114131)

            if more_types_in_union_114130:
                # Runtime conditional SSA for else branch (line 119)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_114129) or more_types_in_union_114130):
            
            # Assigning a BinOp to a Name (line 122):
            # Getting the type of 'func_docstring' (line 122)
            func_docstring_114133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 28), 'func_docstring')
            
            # Call to dict(...): (line 122)
            # Processing the call keyword arguments (line 122)
            # Getting the type of 'cls_docstring' (line 122)
            cls_docstring_114135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 56), 'cls_docstring', False)
            keyword_114136 = cls_docstring_114135
            kwargs_114137 = {'super': keyword_114136}
            # Getting the type of 'dict' (line 122)
            dict_114134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 45), 'dict', False)
            # Calling dict(args, kwargs) (line 122)
            dict_call_result_114138 = invoke(stypy.reporting.localization.Localization(__file__, 122, 45), dict_114134, *[], **kwargs_114137)
            
            # Applying the binary operator '%' (line 122)
            result_mod_114139 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 28), '%', func_docstring_114133, dict_call_result_114138)
            
            # Assigning a type to the variable 'new_docstring' (line 122)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'new_docstring', result_mod_114139)
            
            # Assigning a Name to a Attribute (line 123):
            # Getting the type of 'new_docstring' (line 123)
            new_docstring_114140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 27), 'new_docstring')
            # Getting the type of 'func' (line 123)
            func_114141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'func')
            # Setting the type of the member '__doc__' of a type (line 123)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 12), func_114141, '__doc__', new_docstring_114140)

            if (may_be_114129 and more_types_in_union_114130):
                # SSA join for if statement (line 119)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'func' (line 124)
        func_114142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 15), 'func')
        # Assigning a type to the variable 'stypy_return_type' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'stypy_return_type', func_114142)
        
        # ################# End of '_doc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_doc' in the type store
        # Getting the type of 'stypy_return_type' (line 116)
        stypy_return_type_114143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_114143)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_doc'
        return stypy_return_type_114143

    # Assigning a type to the variable '_doc' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), '_doc', _doc)
    # Getting the type of '_doc' (line 125)
    _doc_114144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 11), '_doc')
    # Assigning a type to the variable 'stypy_return_type' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'stypy_return_type', _doc_114144)
    
    # ################# End of 'inherit_docstring_from(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'inherit_docstring_from' in the type store
    # Getting the type of 'stypy_return_type' (line 71)
    stypy_return_type_114145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_114145)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'inherit_docstring_from'
    return stypy_return_type_114145

# Assigning a type to the variable 'inherit_docstring_from' (line 71)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'inherit_docstring_from', inherit_docstring_from)

@norecursion
def indentcount_lines(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'indentcount_lines'
    module_type_store = module_type_store.open_function_context('indentcount_lines', 128, 0, False)
    
    # Passed parameters checking function
    indentcount_lines.stypy_localization = localization
    indentcount_lines.stypy_type_of_self = None
    indentcount_lines.stypy_type_store = module_type_store
    indentcount_lines.stypy_function_name = 'indentcount_lines'
    indentcount_lines.stypy_param_names_list = ['lines']
    indentcount_lines.stypy_varargs_param_name = None
    indentcount_lines.stypy_kwargs_param_name = None
    indentcount_lines.stypy_call_defaults = defaults
    indentcount_lines.stypy_call_varargs = varargs
    indentcount_lines.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'indentcount_lines', ['lines'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'indentcount_lines', localization, ['lines'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'indentcount_lines(...)' code ##################

    str_114146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, (-1)), 'str', " Minimum indent for all lines in line list\n\n    >>> lines = [' one', '  two', '   three']\n    >>> indentcount_lines(lines)\n    1\n    >>> lines = []\n    >>> indentcount_lines(lines)\n    0\n    >>> lines = [' one']\n    >>> indentcount_lines(lines)\n    1\n    >>> indentcount_lines(['    '])\n    0\n    ")
    
    # Assigning a Attribute to a Name (line 143):
    # Getting the type of 'sys' (line 143)
    sys_114147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 15), 'sys')
    # Obtaining the member 'maxsize' of a type (line 143)
    maxsize_114148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 15), sys_114147, 'maxsize')
    # Assigning a type to the variable 'indentno' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'indentno', maxsize_114148)
    
    # Getting the type of 'lines' (line 144)
    lines_114149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 16), 'lines')
    # Testing the type of a for loop iterable (line 144)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 144, 4), lines_114149)
    # Getting the type of the for loop variable (line 144)
    for_loop_var_114150 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 144, 4), lines_114149)
    # Assigning a type to the variable 'line' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'line', for_loop_var_114150)
    # SSA begins for a for statement (line 144)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 145):
    
    # Call to lstrip(...): (line 145)
    # Processing the call keyword arguments (line 145)
    kwargs_114153 = {}
    # Getting the type of 'line' (line 145)
    line_114151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 19), 'line', False)
    # Obtaining the member 'lstrip' of a type (line 145)
    lstrip_114152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 19), line_114151, 'lstrip')
    # Calling lstrip(args, kwargs) (line 145)
    lstrip_call_result_114154 = invoke(stypy.reporting.localization.Localization(__file__, 145, 19), lstrip_114152, *[], **kwargs_114153)
    
    # Assigning a type to the variable 'stripped' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'stripped', lstrip_call_result_114154)
    
    # Getting the type of 'stripped' (line 146)
    stripped_114155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 11), 'stripped')
    # Testing the type of an if condition (line 146)
    if_condition_114156 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 146, 8), stripped_114155)
    # Assigning a type to the variable 'if_condition_114156' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'if_condition_114156', if_condition_114156)
    # SSA begins for if statement (line 146)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 147):
    
    # Call to min(...): (line 147)
    # Processing the call arguments (line 147)
    # Getting the type of 'indentno' (line 147)
    indentno_114158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 27), 'indentno', False)
    
    # Call to len(...): (line 147)
    # Processing the call arguments (line 147)
    # Getting the type of 'line' (line 147)
    line_114160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 41), 'line', False)
    # Processing the call keyword arguments (line 147)
    kwargs_114161 = {}
    # Getting the type of 'len' (line 147)
    len_114159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 37), 'len', False)
    # Calling len(args, kwargs) (line 147)
    len_call_result_114162 = invoke(stypy.reporting.localization.Localization(__file__, 147, 37), len_114159, *[line_114160], **kwargs_114161)
    
    
    # Call to len(...): (line 147)
    # Processing the call arguments (line 147)
    # Getting the type of 'stripped' (line 147)
    stripped_114164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 53), 'stripped', False)
    # Processing the call keyword arguments (line 147)
    kwargs_114165 = {}
    # Getting the type of 'len' (line 147)
    len_114163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 49), 'len', False)
    # Calling len(args, kwargs) (line 147)
    len_call_result_114166 = invoke(stypy.reporting.localization.Localization(__file__, 147, 49), len_114163, *[stripped_114164], **kwargs_114165)
    
    # Applying the binary operator '-' (line 147)
    result_sub_114167 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 37), '-', len_call_result_114162, len_call_result_114166)
    
    # Processing the call keyword arguments (line 147)
    kwargs_114168 = {}
    # Getting the type of 'min' (line 147)
    min_114157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 23), 'min', False)
    # Calling min(args, kwargs) (line 147)
    min_call_result_114169 = invoke(stypy.reporting.localization.Localization(__file__, 147, 23), min_114157, *[indentno_114158, result_sub_114167], **kwargs_114168)
    
    # Assigning a type to the variable 'indentno' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'indentno', min_call_result_114169)
    # SSA join for if statement (line 146)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'indentno' (line 148)
    indentno_114170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 7), 'indentno')
    # Getting the type of 'sys' (line 148)
    sys_114171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 19), 'sys')
    # Obtaining the member 'maxsize' of a type (line 148)
    maxsize_114172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 19), sys_114171, 'maxsize')
    # Applying the binary operator '==' (line 148)
    result_eq_114173 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 7), '==', indentno_114170, maxsize_114172)
    
    # Testing the type of an if condition (line 148)
    if_condition_114174 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 148, 4), result_eq_114173)
    # Assigning a type to the variable 'if_condition_114174' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'if_condition_114174', if_condition_114174)
    # SSA begins for if statement (line 148)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    int_114175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 15), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'stypy_return_type', int_114175)
    # SSA join for if statement (line 148)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'indentno' (line 150)
    indentno_114176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 11), 'indentno')
    # Assigning a type to the variable 'stypy_return_type' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'stypy_return_type', indentno_114176)
    
    # ################# End of 'indentcount_lines(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'indentcount_lines' in the type store
    # Getting the type of 'stypy_return_type' (line 128)
    stypy_return_type_114177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_114177)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'indentcount_lines'
    return stypy_return_type_114177

# Assigning a type to the variable 'indentcount_lines' (line 128)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 0), 'indentcount_lines', indentcount_lines)

@norecursion
def filldoc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 153)
    True_114178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 37), 'True')
    defaults = [True_114178]
    # Create a new context for function 'filldoc'
    module_type_store = module_type_store.open_function_context('filldoc', 153, 0, False)
    
    # Passed parameters checking function
    filldoc.stypy_localization = localization
    filldoc.stypy_type_of_self = None
    filldoc.stypy_type_store = module_type_store
    filldoc.stypy_function_name = 'filldoc'
    filldoc.stypy_param_names_list = ['docdict', 'unindent_params']
    filldoc.stypy_varargs_param_name = None
    filldoc.stypy_kwargs_param_name = None
    filldoc.stypy_call_defaults = defaults
    filldoc.stypy_call_varargs = varargs
    filldoc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'filldoc', ['docdict', 'unindent_params'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'filldoc', localization, ['docdict', 'unindent_params'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'filldoc(...)' code ##################

    str_114179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, (-1)), 'str', ' Return docstring decorator using docdict variable dictionary\n\n    Parameters\n    ----------\n    docdict : dictionary\n        dictionary containing name, docstring fragment pairs\n    unindent_params : {False, True}, boolean, optional\n        If True, strip common indentation from all parameters in\n        docdict\n\n    Returns\n    -------\n    decfunc : function\n        decorator that applies dictionary to input function docstring\n\n    ')
    
    # Getting the type of 'unindent_params' (line 170)
    unindent_params_114180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 7), 'unindent_params')
    # Testing the type of an if condition (line 170)
    if_condition_114181 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 170, 4), unindent_params_114180)
    # Assigning a type to the variable 'if_condition_114181' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'if_condition_114181', if_condition_114181)
    # SSA begins for if statement (line 170)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 171):
    
    # Call to unindent_dict(...): (line 171)
    # Processing the call arguments (line 171)
    # Getting the type of 'docdict' (line 171)
    docdict_114183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 32), 'docdict', False)
    # Processing the call keyword arguments (line 171)
    kwargs_114184 = {}
    # Getting the type of 'unindent_dict' (line 171)
    unindent_dict_114182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 18), 'unindent_dict', False)
    # Calling unindent_dict(args, kwargs) (line 171)
    unindent_dict_call_result_114185 = invoke(stypy.reporting.localization.Localization(__file__, 171, 18), unindent_dict_114182, *[docdict_114183], **kwargs_114184)
    
    # Assigning a type to the variable 'docdict' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'docdict', unindent_dict_call_result_114185)
    # SSA join for if statement (line 170)
    module_type_store = module_type_store.join_ssa_context()
    

    @norecursion
    def decorate(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'decorate'
        module_type_store = module_type_store.open_function_context('decorate', 173, 4, False)
        
        # Passed parameters checking function
        decorate.stypy_localization = localization
        decorate.stypy_type_of_self = None
        decorate.stypy_type_store = module_type_store
        decorate.stypy_function_name = 'decorate'
        decorate.stypy_param_names_list = ['f']
        decorate.stypy_varargs_param_name = None
        decorate.stypy_kwargs_param_name = None
        decorate.stypy_call_defaults = defaults
        decorate.stypy_call_varargs = varargs
        decorate.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'decorate', ['f'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'decorate', localization, ['f'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'decorate(...)' code ##################

        
        # Assigning a Call to a Attribute (line 174):
        
        # Call to docformat(...): (line 174)
        # Processing the call arguments (line 174)
        # Getting the type of 'f' (line 174)
        f_114187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 30), 'f', False)
        # Obtaining the member '__doc__' of a type (line 174)
        doc___114188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 30), f_114187, '__doc__')
        # Getting the type of 'docdict' (line 174)
        docdict_114189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 41), 'docdict', False)
        # Processing the call keyword arguments (line 174)
        kwargs_114190 = {}
        # Getting the type of 'docformat' (line 174)
        docformat_114186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 20), 'docformat', False)
        # Calling docformat(args, kwargs) (line 174)
        docformat_call_result_114191 = invoke(stypy.reporting.localization.Localization(__file__, 174, 20), docformat_114186, *[doc___114188, docdict_114189], **kwargs_114190)
        
        # Getting the type of 'f' (line 174)
        f_114192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'f')
        # Setting the type of the member '__doc__' of a type (line 174)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 8), f_114192, '__doc__', docformat_call_result_114191)
        # Getting the type of 'f' (line 175)
        f_114193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 15), 'f')
        # Assigning a type to the variable 'stypy_return_type' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'stypy_return_type', f_114193)
        
        # ################# End of 'decorate(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'decorate' in the type store
        # Getting the type of 'stypy_return_type' (line 173)
        stypy_return_type_114194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_114194)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'decorate'
        return stypy_return_type_114194

    # Assigning a type to the variable 'decorate' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'decorate', decorate)
    # Getting the type of 'decorate' (line 176)
    decorate_114195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 11), 'decorate')
    # Assigning a type to the variable 'stypy_return_type' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'stypy_return_type', decorate_114195)
    
    # ################# End of 'filldoc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'filldoc' in the type store
    # Getting the type of 'stypy_return_type' (line 153)
    stypy_return_type_114196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_114196)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'filldoc'
    return stypy_return_type_114196

# Assigning a type to the variable 'filldoc' (line 153)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 0), 'filldoc', filldoc)

@norecursion
def unindent_dict(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'unindent_dict'
    module_type_store = module_type_store.open_function_context('unindent_dict', 179, 0, False)
    
    # Passed parameters checking function
    unindent_dict.stypy_localization = localization
    unindent_dict.stypy_type_of_self = None
    unindent_dict.stypy_type_store = module_type_store
    unindent_dict.stypy_function_name = 'unindent_dict'
    unindent_dict.stypy_param_names_list = ['docdict']
    unindent_dict.stypy_varargs_param_name = None
    unindent_dict.stypy_kwargs_param_name = None
    unindent_dict.stypy_call_defaults = defaults
    unindent_dict.stypy_call_varargs = varargs
    unindent_dict.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'unindent_dict', ['docdict'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'unindent_dict', localization, ['docdict'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'unindent_dict(...)' code ##################

    str_114197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 4), 'str', ' Unindent all strings in a docdict ')
    
    # Assigning a Dict to a Name (line 181):
    
    # Obtaining an instance of the builtin type 'dict' (line 181)
    dict_114198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 15), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 181)
    
    # Assigning a type to the variable 'can_dict' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'can_dict', dict_114198)
    
    
    # Call to items(...): (line 182)
    # Processing the call keyword arguments (line 182)
    kwargs_114201 = {}
    # Getting the type of 'docdict' (line 182)
    docdict_114199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 22), 'docdict', False)
    # Obtaining the member 'items' of a type (line 182)
    items_114200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 22), docdict_114199, 'items')
    # Calling items(args, kwargs) (line 182)
    items_call_result_114202 = invoke(stypy.reporting.localization.Localization(__file__, 182, 22), items_114200, *[], **kwargs_114201)
    
    # Testing the type of a for loop iterable (line 182)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 182, 4), items_call_result_114202)
    # Getting the type of the for loop variable (line 182)
    for_loop_var_114203 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 182, 4), items_call_result_114202)
    # Assigning a type to the variable 'name' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 4), for_loop_var_114203))
    # Assigning a type to the variable 'dstr' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'dstr', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 4), for_loop_var_114203))
    # SSA begins for a for statement (line 182)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Subscript (line 183):
    
    # Call to unindent_string(...): (line 183)
    # Processing the call arguments (line 183)
    # Getting the type of 'dstr' (line 183)
    dstr_114205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 41), 'dstr', False)
    # Processing the call keyword arguments (line 183)
    kwargs_114206 = {}
    # Getting the type of 'unindent_string' (line 183)
    unindent_string_114204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 25), 'unindent_string', False)
    # Calling unindent_string(args, kwargs) (line 183)
    unindent_string_call_result_114207 = invoke(stypy.reporting.localization.Localization(__file__, 183, 25), unindent_string_114204, *[dstr_114205], **kwargs_114206)
    
    # Getting the type of 'can_dict' (line 183)
    can_dict_114208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'can_dict')
    # Getting the type of 'name' (line 183)
    name_114209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 17), 'name')
    # Storing an element on a container (line 183)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 8), can_dict_114208, (name_114209, unindent_string_call_result_114207))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'can_dict' (line 184)
    can_dict_114210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 11), 'can_dict')
    # Assigning a type to the variable 'stypy_return_type' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'stypy_return_type', can_dict_114210)
    
    # ################# End of 'unindent_dict(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'unindent_dict' in the type store
    # Getting the type of 'stypy_return_type' (line 179)
    stypy_return_type_114211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_114211)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'unindent_dict'
    return stypy_return_type_114211

# Assigning a type to the variable 'unindent_dict' (line 179)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 0), 'unindent_dict', unindent_dict)

@norecursion
def unindent_string(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'unindent_string'
    module_type_store = module_type_store.open_function_context('unindent_string', 187, 0, False)
    
    # Passed parameters checking function
    unindent_string.stypy_localization = localization
    unindent_string.stypy_type_of_self = None
    unindent_string.stypy_type_store = module_type_store
    unindent_string.stypy_function_name = 'unindent_string'
    unindent_string.stypy_param_names_list = ['docstring']
    unindent_string.stypy_varargs_param_name = None
    unindent_string.stypy_kwargs_param_name = None
    unindent_string.stypy_call_defaults = defaults
    unindent_string.stypy_call_varargs = varargs
    unindent_string.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'unindent_string', ['docstring'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'unindent_string', localization, ['docstring'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'unindent_string(...)' code ##################

    str_114212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, (-1)), 'str', " Set docstring to minimum indent for all lines, including first\n\n    >>> unindent_string(' two')\n    'two'\n    >>> unindent_string('  two\\n   three')\n    'two\\n three'\n    ")
    
    # Assigning a Call to a Name (line 195):
    
    # Call to splitlines(...): (line 195)
    # Processing the call keyword arguments (line 195)
    kwargs_114218 = {}
    
    # Call to expandtabs(...): (line 195)
    # Processing the call keyword arguments (line 195)
    kwargs_114215 = {}
    # Getting the type of 'docstring' (line 195)
    docstring_114213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 12), 'docstring', False)
    # Obtaining the member 'expandtabs' of a type (line 195)
    expandtabs_114214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 12), docstring_114213, 'expandtabs')
    # Calling expandtabs(args, kwargs) (line 195)
    expandtabs_call_result_114216 = invoke(stypy.reporting.localization.Localization(__file__, 195, 12), expandtabs_114214, *[], **kwargs_114215)
    
    # Obtaining the member 'splitlines' of a type (line 195)
    splitlines_114217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 12), expandtabs_call_result_114216, 'splitlines')
    # Calling splitlines(args, kwargs) (line 195)
    splitlines_call_result_114219 = invoke(stypy.reporting.localization.Localization(__file__, 195, 12), splitlines_114217, *[], **kwargs_114218)
    
    # Assigning a type to the variable 'lines' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'lines', splitlines_call_result_114219)
    
    # Assigning a Call to a Name (line 196):
    
    # Call to indentcount_lines(...): (line 196)
    # Processing the call arguments (line 196)
    # Getting the type of 'lines' (line 196)
    lines_114221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 31), 'lines', False)
    # Processing the call keyword arguments (line 196)
    kwargs_114222 = {}
    # Getting the type of 'indentcount_lines' (line 196)
    indentcount_lines_114220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 13), 'indentcount_lines', False)
    # Calling indentcount_lines(args, kwargs) (line 196)
    indentcount_lines_call_result_114223 = invoke(stypy.reporting.localization.Localization(__file__, 196, 13), indentcount_lines_114220, *[lines_114221], **kwargs_114222)
    
    # Assigning a type to the variable 'icount' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'icount', indentcount_lines_call_result_114223)
    
    
    # Getting the type of 'icount' (line 197)
    icount_114224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 7), 'icount')
    int_114225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 17), 'int')
    # Applying the binary operator '==' (line 197)
    result_eq_114226 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 7), '==', icount_114224, int_114225)
    
    # Testing the type of an if condition (line 197)
    if_condition_114227 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 197, 4), result_eq_114226)
    # Assigning a type to the variable 'if_condition_114227' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 4), 'if_condition_114227', if_condition_114227)
    # SSA begins for if statement (line 197)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'docstring' (line 198)
    docstring_114228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 15), 'docstring')
    # Assigning a type to the variable 'stypy_return_type' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'stypy_return_type', docstring_114228)
    # SSA join for if statement (line 197)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to join(...): (line 199)
    # Processing the call arguments (line 199)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'lines' (line 199)
    lines_114236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 48), 'lines', False)
    comprehension_114237 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 22), lines_114236)
    # Assigning a type to the variable 'line' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 22), 'line', comprehension_114237)
    
    # Obtaining the type of the subscript
    # Getting the type of 'icount' (line 199)
    icount_114231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 27), 'icount', False)
    slice_114232 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 199, 22), icount_114231, None, None)
    # Getting the type of 'line' (line 199)
    line_114233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 22), 'line', False)
    # Obtaining the member '__getitem__' of a type (line 199)
    getitem___114234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 22), line_114233, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 199)
    subscript_call_result_114235 = invoke(stypy.reporting.localization.Localization(__file__, 199, 22), getitem___114234, slice_114232)
    
    list_114238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 22), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 22), list_114238, subscript_call_result_114235)
    # Processing the call keyword arguments (line 199)
    kwargs_114239 = {}
    str_114229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 11), 'str', '\n')
    # Obtaining the member 'join' of a type (line 199)
    join_114230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 11), str_114229, 'join')
    # Calling join(args, kwargs) (line 199)
    join_call_result_114240 = invoke(stypy.reporting.localization.Localization(__file__, 199, 11), join_114230, *[list_114238], **kwargs_114239)
    
    # Assigning a type to the variable 'stypy_return_type' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'stypy_return_type', join_call_result_114240)
    
    # ################# End of 'unindent_string(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'unindent_string' in the type store
    # Getting the type of 'stypy_return_type' (line 187)
    stypy_return_type_114241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_114241)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'unindent_string'
    return stypy_return_type_114241

# Assigning a type to the variable 'unindent_string' (line 187)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 0), 'unindent_string', unindent_string)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
