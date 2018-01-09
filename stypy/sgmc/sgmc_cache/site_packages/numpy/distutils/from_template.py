
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #!/usr/bin/python
2: '''
3: 
4: process_file(filename)
5: 
6:   takes templated file .xxx.src and produces .xxx file where .xxx
7:   is .pyf .f90 or .f using the following template rules:
8: 
9:   '<..>' denotes a template.
10: 
11:   All function and subroutine blocks in a source file with names that
12:   contain '<..>' will be replicated according to the rules in '<..>'.
13: 
14:   The number of comma-separeted words in '<..>' will determine the number of
15:   replicates.
16: 
17:   '<..>' may have two different forms, named and short. For example,
18: 
19:   named:
20:    <p=d,s,z,c> where anywhere inside a block '<p>' will be replaced with
21:    'd', 's', 'z', and 'c' for each replicate of the block.
22: 
23:    <_c>  is already defined: <_c=s,d,c,z>
24:    <_t>  is already defined: <_t=real,double precision,complex,double complex>
25: 
26:   short:
27:    <s,d,c,z>, a short form of the named, useful when no <p> appears inside
28:    a block.
29: 
30:   In general, '<..>' contains a comma separated list of arbitrary
31:   expressions. If these expression must contain a comma|leftarrow|rightarrow,
32:   then prepend the comma|leftarrow|rightarrow with a backslash.
33: 
34:   If an expression matches '\\<index>' then it will be replaced
35:   by <index>-th expression.
36: 
37:   Note that all '<..>' forms in a block must have the same number of
38:   comma-separated entries.
39: 
40:  Predefined named template rules:
41:   <prefix=s,d,c,z>
42:   <ftype=real,double precision,complex,double complex>
43:   <ftypereal=real,double precision,\\0,\\1>
44:   <ctype=float,double,complex_float,complex_double>
45:   <ctypereal=float,double,\\0,\\1>
46: 
47: '''
48: from __future__ import division, absolute_import, print_function
49: 
50: __all__ = ['process_str', 'process_file']
51: 
52: import os
53: import sys
54: import re
55: 
56: routine_start_re = re.compile(r'(\n|\A)((     (\$|\*))|)\s*(subroutine|function)\b', re.I)
57: routine_end_re = re.compile(r'\n\s*end\s*(subroutine|function)\b.*(\n|\Z)', re.I)
58: function_start_re = re.compile(r'\n     (\$|\*)\s*function\b', re.I)
59: 
60: def parse_structure(astr):
61:     ''' Return a list of tuples for each function or subroutine each
62:     tuple is the start and end of a subroutine or function to be
63:     expanded.
64:     '''
65: 
66:     spanlist = []
67:     ind = 0
68:     while True:
69:         m = routine_start_re.search(astr, ind)
70:         if m is None:
71:             break
72:         start = m.start()
73:         if function_start_re.match(astr, start, m.end()):
74:             while True:
75:                 i = astr.rfind('\n', ind, start)
76:                 if i==-1:
77:                     break
78:                 start = i
79:                 if astr[i:i+7]!='\n     $':
80:                     break
81:         start += 1
82:         m = routine_end_re.search(astr, m.end())
83:         ind = end = m and m.end()-1 or len(astr)
84:         spanlist.append((start, end))
85:     return spanlist
86: 
87: template_re = re.compile(r"<\s*(\w[\w\d]*)\s*>")
88: named_re = re.compile(r"<\s*(\w[\w\d]*)\s*=\s*(.*?)\s*>")
89: list_re = re.compile(r"<\s*((.*?))\s*>")
90: 
91: def find_repl_patterns(astr):
92:     reps = named_re.findall(astr)
93:     names = {}
94:     for rep in reps:
95:         name = rep[0].strip() or unique_key(names)
96:         repl = rep[1].replace('\,', '@comma@')
97:         thelist = conv(repl)
98:         names[name] = thelist
99:     return names
100: 
101: item_re = re.compile(r"\A\\(?P<index>\d+)\Z")
102: def conv(astr):
103:     b = astr.split(',')
104:     l = [x.strip() for x in b]
105:     for i in range(len(l)):
106:         m = item_re.match(l[i])
107:         if m:
108:             j = int(m.group('index'))
109:             l[i] = l[j]
110:     return ','.join(l)
111: 
112: def unique_key(adict):
113:     ''' Obtain a unique key given a dictionary.'''
114:     allkeys = list(adict.keys())
115:     done = False
116:     n = 1
117:     while not done:
118:         newkey = '__l%s' % (n)
119:         if newkey in allkeys:
120:             n += 1
121:         else:
122:             done = True
123:     return newkey
124: 
125: 
126: template_name_re = re.compile(r'\A\s*(\w[\w\d]*)\s*\Z')
127: def expand_sub(substr, names):
128:     substr = substr.replace('\>', '@rightarrow@')
129:     substr = substr.replace('\<', '@leftarrow@')
130:     lnames = find_repl_patterns(substr)
131:     substr = named_re.sub(r"<\1>", substr)  # get rid of definition templates
132: 
133:     def listrepl(mobj):
134:         thelist = conv(mobj.group(1).replace('\,', '@comma@'))
135:         if template_name_re.match(thelist):
136:             return "<%s>" % (thelist)
137:         name = None
138:         for key in lnames.keys():    # see if list is already in dictionary
139:             if lnames[key] == thelist:
140:                 name = key
141:         if name is None:      # this list is not in the dictionary yet
142:             name = unique_key(lnames)
143:             lnames[name] = thelist
144:         return "<%s>" % name
145: 
146:     substr = list_re.sub(listrepl, substr) # convert all lists to named templates
147:                                            # newnames are constructed as needed
148: 
149:     numsubs = None
150:     base_rule = None
151:     rules = {}
152:     for r in template_re.findall(substr):
153:         if r not in rules:
154:             thelist = lnames.get(r, names.get(r, None))
155:             if thelist is None:
156:                 raise ValueError('No replicates found for <%s>' % (r))
157:             if r not in names and not thelist.startswith('_'):
158:                 names[r] = thelist
159:             rule = [i.replace('@comma@', ',') for i in thelist.split(',')]
160:             num = len(rule)
161: 
162:             if numsubs is None:
163:                 numsubs = num
164:                 rules[r] = rule
165:                 base_rule = r
166:             elif num == numsubs:
167:                 rules[r] = rule
168:             else:
169:                 print("Mismatch in number of replacements (base <%s=%s>)"
170:                       " for <%s=%s>. Ignoring." %
171:                       (base_rule, ','.join(rules[base_rule]), r, thelist))
172:     if not rules:
173:         return substr
174: 
175:     def namerepl(mobj):
176:         name = mobj.group(1)
177:         return rules.get(name, (k+1)*[name])[k]
178: 
179:     newstr = ''
180:     for k in range(numsubs):
181:         newstr += template_re.sub(namerepl, substr) + '\n\n'
182: 
183:     newstr = newstr.replace('@rightarrow@', '>')
184:     newstr = newstr.replace('@leftarrow@', '<')
185:     return newstr
186: 
187: def process_str(allstr):
188:     newstr = allstr
189:     writestr = '' #_head # using _head will break free-format files
190: 
191:     struct = parse_structure(newstr)
192: 
193:     oldend = 0
194:     names = {}
195:     names.update(_special_names)
196:     for sub in struct:
197:         writestr += newstr[oldend:sub[0]]
198:         names.update(find_repl_patterns(newstr[oldend:sub[0]]))
199:         writestr += expand_sub(newstr[sub[0]:sub[1]], names)
200:         oldend =  sub[1]
201:     writestr += newstr[oldend:]
202: 
203:     return writestr
204: 
205: include_src_re = re.compile(r"(\n|\A)\s*include\s*['\"](?P<name>[\w\d./\\]+[.]src)['\"]", re.I)
206: 
207: def resolve_includes(source):
208:     d = os.path.dirname(source)
209:     fid = open(source)
210:     lines = []
211:     for line in fid:
212:         m = include_src_re.match(line)
213:         if m:
214:             fn = m.group('name')
215:             if not os.path.isabs(fn):
216:                 fn = os.path.join(d, fn)
217:             if os.path.isfile(fn):
218:                 print('Including file', fn)
219:                 lines.extend(resolve_includes(fn))
220:             else:
221:                 lines.append(line)
222:         else:
223:             lines.append(line)
224:     fid.close()
225:     return lines
226: 
227: def process_file(source):
228:     lines = resolve_includes(source)
229:     return process_str(''.join(lines))
230: 
231: _special_names = find_repl_patterns('''
232: <_c=s,d,c,z>
233: <_t=real,double precision,complex,double complex>
234: <prefix=s,d,c,z>
235: <ftype=real,double precision,complex,double complex>
236: <ctype=float,double,complex_float,complex_double>
237: <ftypereal=real,double precision,\\0,\\1>
238: <ctypereal=float,double,\\0,\\1>
239: ''')
240: 
241: if __name__ == "__main__":
242: 
243:     try:
244:         file = sys.argv[1]
245:     except IndexError:
246:         fid = sys.stdin
247:         outfile = sys.stdout
248:     else:
249:         fid = open(file, 'r')
250:         (base, ext) = os.path.splitext(file)
251:         newname = base
252:         outfile = open(newname, 'w')
253: 
254:     allstr = fid.read()
255:     writestr = process_str(allstr)
256:     outfile.write(writestr)
257: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_35098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, (-1)), 'str', "\n\nprocess_file(filename)\n\n  takes templated file .xxx.src and produces .xxx file where .xxx\n  is .pyf .f90 or .f using the following template rules:\n\n  '<..>' denotes a template.\n\n  All function and subroutine blocks in a source file with names that\n  contain '<..>' will be replicated according to the rules in '<..>'.\n\n  The number of comma-separeted words in '<..>' will determine the number of\n  replicates.\n\n  '<..>' may have two different forms, named and short. For example,\n\n  named:\n   <p=d,s,z,c> where anywhere inside a block '<p>' will be replaced with\n   'd', 's', 'z', and 'c' for each replicate of the block.\n\n   <_c>  is already defined: <_c=s,d,c,z>\n   <_t>  is already defined: <_t=real,double precision,complex,double complex>\n\n  short:\n   <s,d,c,z>, a short form of the named, useful when no <p> appears inside\n   a block.\n\n  In general, '<..>' contains a comma separated list of arbitrary\n  expressions. If these expression must contain a comma|leftarrow|rightarrow,\n  then prepend the comma|leftarrow|rightarrow with a backslash.\n\n  If an expression matches '\\<index>' then it will be replaced\n  by <index>-th expression.\n\n  Note that all '<..>' forms in a block must have the same number of\n  comma-separated entries.\n\n Predefined named template rules:\n  <prefix=s,d,c,z>\n  <ftype=real,double precision,complex,double complex>\n  <ftypereal=real,double precision,\\0,\\1>\n  <ctype=float,double,complex_float,complex_double>\n  <ctypereal=float,double,\\0,\\1>\n\n")

# Assigning a List to a Name (line 50):

# Assigning a List to a Name (line 50):
__all__ = ['process_str', 'process_file']
module_type_store.set_exportable_members(['process_str', 'process_file'])

# Obtaining an instance of the builtin type 'list' (line 50)
list_35099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 50)
# Adding element type (line 50)
str_35100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 11), 'str', 'process_str')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 10), list_35099, str_35100)
# Adding element type (line 50)
str_35101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 26), 'str', 'process_file')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 10), list_35099, str_35101)

# Assigning a type to the variable '__all__' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), '__all__', list_35099)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 52, 0))

# 'import os' statement (line 52)
import os

import_module(stypy.reporting.localization.Localization(__file__, 52, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 53, 0))

# 'import sys' statement (line 53)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 53, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 54, 0))

# 'import re' statement (line 54)
import re

import_module(stypy.reporting.localization.Localization(__file__, 54, 0), 're', re, module_type_store)


# Assigning a Call to a Name (line 56):

# Assigning a Call to a Name (line 56):

# Call to compile(...): (line 56)
# Processing the call arguments (line 56)
str_35104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 30), 'str', '(\\n|\\A)((     (\\$|\\*))|)\\s*(subroutine|function)\\b')
# Getting the type of 're' (line 56)
re_35105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 85), 're', False)
# Obtaining the member 'I' of a type (line 56)
I_35106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 85), re_35105, 'I')
# Processing the call keyword arguments (line 56)
kwargs_35107 = {}
# Getting the type of 're' (line 56)
re_35102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 19), 're', False)
# Obtaining the member 'compile' of a type (line 56)
compile_35103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 19), re_35102, 'compile')
# Calling compile(args, kwargs) (line 56)
compile_call_result_35108 = invoke(stypy.reporting.localization.Localization(__file__, 56, 19), compile_35103, *[str_35104, I_35106], **kwargs_35107)

# Assigning a type to the variable 'routine_start_re' (line 56)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'routine_start_re', compile_call_result_35108)

# Assigning a Call to a Name (line 57):

# Assigning a Call to a Name (line 57):

# Call to compile(...): (line 57)
# Processing the call arguments (line 57)
str_35111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 28), 'str', '\\n\\s*end\\s*(subroutine|function)\\b.*(\\n|\\Z)')
# Getting the type of 're' (line 57)
re_35112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 76), 're', False)
# Obtaining the member 'I' of a type (line 57)
I_35113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 76), re_35112, 'I')
# Processing the call keyword arguments (line 57)
kwargs_35114 = {}
# Getting the type of 're' (line 57)
re_35109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 17), 're', False)
# Obtaining the member 'compile' of a type (line 57)
compile_35110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 17), re_35109, 'compile')
# Calling compile(args, kwargs) (line 57)
compile_call_result_35115 = invoke(stypy.reporting.localization.Localization(__file__, 57, 17), compile_35110, *[str_35111, I_35113], **kwargs_35114)

# Assigning a type to the variable 'routine_end_re' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'routine_end_re', compile_call_result_35115)

# Assigning a Call to a Name (line 58):

# Assigning a Call to a Name (line 58):

# Call to compile(...): (line 58)
# Processing the call arguments (line 58)
str_35118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 31), 'str', '\\n     (\\$|\\*)\\s*function\\b')
# Getting the type of 're' (line 58)
re_35119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 63), 're', False)
# Obtaining the member 'I' of a type (line 58)
I_35120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 63), re_35119, 'I')
# Processing the call keyword arguments (line 58)
kwargs_35121 = {}
# Getting the type of 're' (line 58)
re_35116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 20), 're', False)
# Obtaining the member 'compile' of a type (line 58)
compile_35117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 20), re_35116, 'compile')
# Calling compile(args, kwargs) (line 58)
compile_call_result_35122 = invoke(stypy.reporting.localization.Localization(__file__, 58, 20), compile_35117, *[str_35118, I_35120], **kwargs_35121)

# Assigning a type to the variable 'function_start_re' (line 58)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'function_start_re', compile_call_result_35122)

@norecursion
def parse_structure(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'parse_structure'
    module_type_store = module_type_store.open_function_context('parse_structure', 60, 0, False)
    
    # Passed parameters checking function
    parse_structure.stypy_localization = localization
    parse_structure.stypy_type_of_self = None
    parse_structure.stypy_type_store = module_type_store
    parse_structure.stypy_function_name = 'parse_structure'
    parse_structure.stypy_param_names_list = ['astr']
    parse_structure.stypy_varargs_param_name = None
    parse_structure.stypy_kwargs_param_name = None
    parse_structure.stypy_call_defaults = defaults
    parse_structure.stypy_call_varargs = varargs
    parse_structure.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'parse_structure', ['astr'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'parse_structure', localization, ['astr'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'parse_structure(...)' code ##################

    str_35123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, (-1)), 'str', ' Return a list of tuples for each function or subroutine each\n    tuple is the start and end of a subroutine or function to be\n    expanded.\n    ')
    
    # Assigning a List to a Name (line 66):
    
    # Assigning a List to a Name (line 66):
    
    # Obtaining an instance of the builtin type 'list' (line 66)
    list_35124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 66)
    
    # Assigning a type to the variable 'spanlist' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'spanlist', list_35124)
    
    # Assigning a Num to a Name (line 67):
    
    # Assigning a Num to a Name (line 67):
    int_35125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 10), 'int')
    # Assigning a type to the variable 'ind' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'ind', int_35125)
    
    # Getting the type of 'True' (line 68)
    True_35126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 10), 'True')
    # Testing the type of an if condition (line 68)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 68, 4), True_35126)
    # SSA begins for while statement (line 68)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Name (line 69):
    
    # Assigning a Call to a Name (line 69):
    
    # Call to search(...): (line 69)
    # Processing the call arguments (line 69)
    # Getting the type of 'astr' (line 69)
    astr_35129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 36), 'astr', False)
    # Getting the type of 'ind' (line 69)
    ind_35130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 42), 'ind', False)
    # Processing the call keyword arguments (line 69)
    kwargs_35131 = {}
    # Getting the type of 'routine_start_re' (line 69)
    routine_start_re_35127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'routine_start_re', False)
    # Obtaining the member 'search' of a type (line 69)
    search_35128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 12), routine_start_re_35127, 'search')
    # Calling search(args, kwargs) (line 69)
    search_call_result_35132 = invoke(stypy.reporting.localization.Localization(__file__, 69, 12), search_35128, *[astr_35129, ind_35130], **kwargs_35131)
    
    # Assigning a type to the variable 'm' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'm', search_call_result_35132)
    
    # Type idiom detected: calculating its left and rigth part (line 70)
    # Getting the type of 'm' (line 70)
    m_35133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 11), 'm')
    # Getting the type of 'None' (line 70)
    None_35134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'None')
    
    (may_be_35135, more_types_in_union_35136) = may_be_none(m_35133, None_35134)

    if may_be_35135:

        if more_types_in_union_35136:
            # Runtime conditional SSA (line 70)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store


        if more_types_in_union_35136:
            # SSA join for if statement (line 70)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 72):
    
    # Assigning a Call to a Name (line 72):
    
    # Call to start(...): (line 72)
    # Processing the call keyword arguments (line 72)
    kwargs_35139 = {}
    # Getting the type of 'm' (line 72)
    m_35137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 16), 'm', False)
    # Obtaining the member 'start' of a type (line 72)
    start_35138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 16), m_35137, 'start')
    # Calling start(args, kwargs) (line 72)
    start_call_result_35140 = invoke(stypy.reporting.localization.Localization(__file__, 72, 16), start_35138, *[], **kwargs_35139)
    
    # Assigning a type to the variable 'start' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'start', start_call_result_35140)
    
    
    # Call to match(...): (line 73)
    # Processing the call arguments (line 73)
    # Getting the type of 'astr' (line 73)
    astr_35143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 35), 'astr', False)
    # Getting the type of 'start' (line 73)
    start_35144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 41), 'start', False)
    
    # Call to end(...): (line 73)
    # Processing the call keyword arguments (line 73)
    kwargs_35147 = {}
    # Getting the type of 'm' (line 73)
    m_35145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 48), 'm', False)
    # Obtaining the member 'end' of a type (line 73)
    end_35146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 48), m_35145, 'end')
    # Calling end(args, kwargs) (line 73)
    end_call_result_35148 = invoke(stypy.reporting.localization.Localization(__file__, 73, 48), end_35146, *[], **kwargs_35147)
    
    # Processing the call keyword arguments (line 73)
    kwargs_35149 = {}
    # Getting the type of 'function_start_re' (line 73)
    function_start_re_35141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 11), 'function_start_re', False)
    # Obtaining the member 'match' of a type (line 73)
    match_35142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 11), function_start_re_35141, 'match')
    # Calling match(args, kwargs) (line 73)
    match_call_result_35150 = invoke(stypy.reporting.localization.Localization(__file__, 73, 11), match_35142, *[astr_35143, start_35144, end_call_result_35148], **kwargs_35149)
    
    # Testing the type of an if condition (line 73)
    if_condition_35151 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 73, 8), match_call_result_35150)
    # Assigning a type to the variable 'if_condition_35151' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'if_condition_35151', if_condition_35151)
    # SSA begins for if statement (line 73)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'True' (line 74)
    True_35152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 18), 'True')
    # Testing the type of an if condition (line 74)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 12), True_35152)
    # SSA begins for while statement (line 74)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Name (line 75):
    
    # Assigning a Call to a Name (line 75):
    
    # Call to rfind(...): (line 75)
    # Processing the call arguments (line 75)
    str_35155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 31), 'str', '\n')
    # Getting the type of 'ind' (line 75)
    ind_35156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 37), 'ind', False)
    # Getting the type of 'start' (line 75)
    start_35157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 42), 'start', False)
    # Processing the call keyword arguments (line 75)
    kwargs_35158 = {}
    # Getting the type of 'astr' (line 75)
    astr_35153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 20), 'astr', False)
    # Obtaining the member 'rfind' of a type (line 75)
    rfind_35154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 20), astr_35153, 'rfind')
    # Calling rfind(args, kwargs) (line 75)
    rfind_call_result_35159 = invoke(stypy.reporting.localization.Localization(__file__, 75, 20), rfind_35154, *[str_35155, ind_35156, start_35157], **kwargs_35158)
    
    # Assigning a type to the variable 'i' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 16), 'i', rfind_call_result_35159)
    
    
    # Getting the type of 'i' (line 76)
    i_35160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 19), 'i')
    int_35161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 22), 'int')
    # Applying the binary operator '==' (line 76)
    result_eq_35162 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 19), '==', i_35160, int_35161)
    
    # Testing the type of an if condition (line 76)
    if_condition_35163 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 16), result_eq_35162)
    # Assigning a type to the variable 'if_condition_35163' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'if_condition_35163', if_condition_35163)
    # SSA begins for if statement (line 76)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 76)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Name (line 78):
    
    # Assigning a Name to a Name (line 78):
    # Getting the type of 'i' (line 78)
    i_35164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 24), 'i')
    # Assigning a type to the variable 'start' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'start', i_35164)
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 79)
    i_35165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 24), 'i')
    # Getting the type of 'i' (line 79)
    i_35166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 26), 'i')
    int_35167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 28), 'int')
    # Applying the binary operator '+' (line 79)
    result_add_35168 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 26), '+', i_35166, int_35167)
    
    slice_35169 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 79, 19), i_35165, result_add_35168, None)
    # Getting the type of 'astr' (line 79)
    astr_35170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 19), 'astr')
    # Obtaining the member '__getitem__' of a type (line 79)
    getitem___35171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 19), astr_35170, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 79)
    subscript_call_result_35172 = invoke(stypy.reporting.localization.Localization(__file__, 79, 19), getitem___35171, slice_35169)
    
    str_35173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 32), 'str', '\n     $')
    # Applying the binary operator '!=' (line 79)
    result_ne_35174 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 19), '!=', subscript_call_result_35172, str_35173)
    
    # Testing the type of an if condition (line 79)
    if_condition_35175 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 16), result_ne_35174)
    # Assigning a type to the variable 'if_condition_35175' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 16), 'if_condition_35175', if_condition_35175)
    # SSA begins for if statement (line 79)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 79)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 74)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 73)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'start' (line 81)
    start_35176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'start')
    int_35177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 17), 'int')
    # Applying the binary operator '+=' (line 81)
    result_iadd_35178 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 8), '+=', start_35176, int_35177)
    # Assigning a type to the variable 'start' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'start', result_iadd_35178)
    
    
    # Assigning a Call to a Name (line 82):
    
    # Assigning a Call to a Name (line 82):
    
    # Call to search(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 'astr' (line 82)
    astr_35181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 34), 'astr', False)
    
    # Call to end(...): (line 82)
    # Processing the call keyword arguments (line 82)
    kwargs_35184 = {}
    # Getting the type of 'm' (line 82)
    m_35182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 40), 'm', False)
    # Obtaining the member 'end' of a type (line 82)
    end_35183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 40), m_35182, 'end')
    # Calling end(args, kwargs) (line 82)
    end_call_result_35185 = invoke(stypy.reporting.localization.Localization(__file__, 82, 40), end_35183, *[], **kwargs_35184)
    
    # Processing the call keyword arguments (line 82)
    kwargs_35186 = {}
    # Getting the type of 'routine_end_re' (line 82)
    routine_end_re_35179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'routine_end_re', False)
    # Obtaining the member 'search' of a type (line 82)
    search_35180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 12), routine_end_re_35179, 'search')
    # Calling search(args, kwargs) (line 82)
    search_call_result_35187 = invoke(stypy.reporting.localization.Localization(__file__, 82, 12), search_35180, *[astr_35181, end_call_result_35185], **kwargs_35186)
    
    # Assigning a type to the variable 'm' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'm', search_call_result_35187)
    
    # Multiple assignment of 2 elements.
    
    # Assigning a BoolOp to a Name (line 83):
    
    # Evaluating a boolean operation
    
    # Evaluating a boolean operation
    # Getting the type of 'm' (line 83)
    m_35188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), 'm')
    
    # Call to end(...): (line 83)
    # Processing the call keyword arguments (line 83)
    kwargs_35191 = {}
    # Getting the type of 'm' (line 83)
    m_35189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 26), 'm', False)
    # Obtaining the member 'end' of a type (line 83)
    end_35190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 26), m_35189, 'end')
    # Calling end(args, kwargs) (line 83)
    end_call_result_35192 = invoke(stypy.reporting.localization.Localization(__file__, 83, 26), end_35190, *[], **kwargs_35191)
    
    int_35193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 34), 'int')
    # Applying the binary operator '-' (line 83)
    result_sub_35194 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 26), '-', end_call_result_35192, int_35193)
    
    # Applying the binary operator 'and' (line 83)
    result_and_keyword_35195 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 20), 'and', m_35188, result_sub_35194)
    
    
    # Call to len(...): (line 83)
    # Processing the call arguments (line 83)
    # Getting the type of 'astr' (line 83)
    astr_35197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 43), 'astr', False)
    # Processing the call keyword arguments (line 83)
    kwargs_35198 = {}
    # Getting the type of 'len' (line 83)
    len_35196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 39), 'len', False)
    # Calling len(args, kwargs) (line 83)
    len_call_result_35199 = invoke(stypy.reporting.localization.Localization(__file__, 83, 39), len_35196, *[astr_35197], **kwargs_35198)
    
    # Applying the binary operator 'or' (line 83)
    result_or_keyword_35200 = python_operator(stypy.reporting.localization.Localization(__file__, 83, 20), 'or', result_and_keyword_35195, len_call_result_35199)
    
    # Assigning a type to the variable 'end' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 14), 'end', result_or_keyword_35200)
    
    # Assigning a Name to a Name (line 83):
    # Getting the type of 'end' (line 83)
    end_35201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 14), 'end')
    # Assigning a type to the variable 'ind' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'ind', end_35201)
    
    # Call to append(...): (line 84)
    # Processing the call arguments (line 84)
    
    # Obtaining an instance of the builtin type 'tuple' (line 84)
    tuple_35204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 25), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 84)
    # Adding element type (line 84)
    # Getting the type of 'start' (line 84)
    start_35205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 25), 'start', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 25), tuple_35204, start_35205)
    # Adding element type (line 84)
    # Getting the type of 'end' (line 84)
    end_35206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 32), 'end', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 25), tuple_35204, end_35206)
    
    # Processing the call keyword arguments (line 84)
    kwargs_35207 = {}
    # Getting the type of 'spanlist' (line 84)
    spanlist_35202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'spanlist', False)
    # Obtaining the member 'append' of a type (line 84)
    append_35203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), spanlist_35202, 'append')
    # Calling append(args, kwargs) (line 84)
    append_call_result_35208 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), append_35203, *[tuple_35204], **kwargs_35207)
    
    # SSA join for while statement (line 68)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'spanlist' (line 85)
    spanlist_35209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 11), 'spanlist')
    # Assigning a type to the variable 'stypy_return_type' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'stypy_return_type', spanlist_35209)
    
    # ################# End of 'parse_structure(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'parse_structure' in the type store
    # Getting the type of 'stypy_return_type' (line 60)
    stypy_return_type_35210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_35210)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'parse_structure'
    return stypy_return_type_35210

# Assigning a type to the variable 'parse_structure' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'parse_structure', parse_structure)

# Assigning a Call to a Name (line 87):

# Assigning a Call to a Name (line 87):

# Call to compile(...): (line 87)
# Processing the call arguments (line 87)
str_35213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 25), 'str', '<\\s*(\\w[\\w\\d]*)\\s*>')
# Processing the call keyword arguments (line 87)
kwargs_35214 = {}
# Getting the type of 're' (line 87)
re_35211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 14), 're', False)
# Obtaining the member 'compile' of a type (line 87)
compile_35212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 14), re_35211, 'compile')
# Calling compile(args, kwargs) (line 87)
compile_call_result_35215 = invoke(stypy.reporting.localization.Localization(__file__, 87, 14), compile_35212, *[str_35213], **kwargs_35214)

# Assigning a type to the variable 'template_re' (line 87)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'template_re', compile_call_result_35215)

# Assigning a Call to a Name (line 88):

# Assigning a Call to a Name (line 88):

# Call to compile(...): (line 88)
# Processing the call arguments (line 88)
str_35218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 22), 'str', '<\\s*(\\w[\\w\\d]*)\\s*=\\s*(.*?)\\s*>')
# Processing the call keyword arguments (line 88)
kwargs_35219 = {}
# Getting the type of 're' (line 88)
re_35216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 11), 're', False)
# Obtaining the member 'compile' of a type (line 88)
compile_35217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 11), re_35216, 'compile')
# Calling compile(args, kwargs) (line 88)
compile_call_result_35220 = invoke(stypy.reporting.localization.Localization(__file__, 88, 11), compile_35217, *[str_35218], **kwargs_35219)

# Assigning a type to the variable 'named_re' (line 88)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 0), 'named_re', compile_call_result_35220)

# Assigning a Call to a Name (line 89):

# Assigning a Call to a Name (line 89):

# Call to compile(...): (line 89)
# Processing the call arguments (line 89)
str_35223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 21), 'str', '<\\s*((.*?))\\s*>')
# Processing the call keyword arguments (line 89)
kwargs_35224 = {}
# Getting the type of 're' (line 89)
re_35221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 10), 're', False)
# Obtaining the member 'compile' of a type (line 89)
compile_35222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 10), re_35221, 'compile')
# Calling compile(args, kwargs) (line 89)
compile_call_result_35225 = invoke(stypy.reporting.localization.Localization(__file__, 89, 10), compile_35222, *[str_35223], **kwargs_35224)

# Assigning a type to the variable 'list_re' (line 89)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 0), 'list_re', compile_call_result_35225)

@norecursion
def find_repl_patterns(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'find_repl_patterns'
    module_type_store = module_type_store.open_function_context('find_repl_patterns', 91, 0, False)
    
    # Passed parameters checking function
    find_repl_patterns.stypy_localization = localization
    find_repl_patterns.stypy_type_of_self = None
    find_repl_patterns.stypy_type_store = module_type_store
    find_repl_patterns.stypy_function_name = 'find_repl_patterns'
    find_repl_patterns.stypy_param_names_list = ['astr']
    find_repl_patterns.stypy_varargs_param_name = None
    find_repl_patterns.stypy_kwargs_param_name = None
    find_repl_patterns.stypy_call_defaults = defaults
    find_repl_patterns.stypy_call_varargs = varargs
    find_repl_patterns.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'find_repl_patterns', ['astr'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'find_repl_patterns', localization, ['astr'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'find_repl_patterns(...)' code ##################

    
    # Assigning a Call to a Name (line 92):
    
    # Assigning a Call to a Name (line 92):
    
    # Call to findall(...): (line 92)
    # Processing the call arguments (line 92)
    # Getting the type of 'astr' (line 92)
    astr_35228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 28), 'astr', False)
    # Processing the call keyword arguments (line 92)
    kwargs_35229 = {}
    # Getting the type of 'named_re' (line 92)
    named_re_35226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 11), 'named_re', False)
    # Obtaining the member 'findall' of a type (line 92)
    findall_35227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 11), named_re_35226, 'findall')
    # Calling findall(args, kwargs) (line 92)
    findall_call_result_35230 = invoke(stypy.reporting.localization.Localization(__file__, 92, 11), findall_35227, *[astr_35228], **kwargs_35229)
    
    # Assigning a type to the variable 'reps' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'reps', findall_call_result_35230)
    
    # Assigning a Dict to a Name (line 93):
    
    # Assigning a Dict to a Name (line 93):
    
    # Obtaining an instance of the builtin type 'dict' (line 93)
    dict_35231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 12), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 93)
    
    # Assigning a type to the variable 'names' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'names', dict_35231)
    
    # Getting the type of 'reps' (line 94)
    reps_35232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 15), 'reps')
    # Testing the type of a for loop iterable (line 94)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 94, 4), reps_35232)
    # Getting the type of the for loop variable (line 94)
    for_loop_var_35233 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 94, 4), reps_35232)
    # Assigning a type to the variable 'rep' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'rep', for_loop_var_35233)
    # SSA begins for a for statement (line 94)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BoolOp to a Name (line 95):
    
    # Assigning a BoolOp to a Name (line 95):
    
    # Evaluating a boolean operation
    
    # Call to strip(...): (line 95)
    # Processing the call keyword arguments (line 95)
    kwargs_35239 = {}
    
    # Obtaining the type of the subscript
    int_35234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 19), 'int')
    # Getting the type of 'rep' (line 95)
    rep_35235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 15), 'rep', False)
    # Obtaining the member '__getitem__' of a type (line 95)
    getitem___35236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 15), rep_35235, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 95)
    subscript_call_result_35237 = invoke(stypy.reporting.localization.Localization(__file__, 95, 15), getitem___35236, int_35234)
    
    # Obtaining the member 'strip' of a type (line 95)
    strip_35238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 15), subscript_call_result_35237, 'strip')
    # Calling strip(args, kwargs) (line 95)
    strip_call_result_35240 = invoke(stypy.reporting.localization.Localization(__file__, 95, 15), strip_35238, *[], **kwargs_35239)
    
    
    # Call to unique_key(...): (line 95)
    # Processing the call arguments (line 95)
    # Getting the type of 'names' (line 95)
    names_35242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 44), 'names', False)
    # Processing the call keyword arguments (line 95)
    kwargs_35243 = {}
    # Getting the type of 'unique_key' (line 95)
    unique_key_35241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 33), 'unique_key', False)
    # Calling unique_key(args, kwargs) (line 95)
    unique_key_call_result_35244 = invoke(stypy.reporting.localization.Localization(__file__, 95, 33), unique_key_35241, *[names_35242], **kwargs_35243)
    
    # Applying the binary operator 'or' (line 95)
    result_or_keyword_35245 = python_operator(stypy.reporting.localization.Localization(__file__, 95, 15), 'or', strip_call_result_35240, unique_key_call_result_35244)
    
    # Assigning a type to the variable 'name' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'name', result_or_keyword_35245)
    
    # Assigning a Call to a Name (line 96):
    
    # Assigning a Call to a Name (line 96):
    
    # Call to replace(...): (line 96)
    # Processing the call arguments (line 96)
    str_35251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 30), 'str', '\\,')
    str_35252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 36), 'str', '@comma@')
    # Processing the call keyword arguments (line 96)
    kwargs_35253 = {}
    
    # Obtaining the type of the subscript
    int_35246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 19), 'int')
    # Getting the type of 'rep' (line 96)
    rep_35247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 15), 'rep', False)
    # Obtaining the member '__getitem__' of a type (line 96)
    getitem___35248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 15), rep_35247, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 96)
    subscript_call_result_35249 = invoke(stypy.reporting.localization.Localization(__file__, 96, 15), getitem___35248, int_35246)
    
    # Obtaining the member 'replace' of a type (line 96)
    replace_35250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 15), subscript_call_result_35249, 'replace')
    # Calling replace(args, kwargs) (line 96)
    replace_call_result_35254 = invoke(stypy.reporting.localization.Localization(__file__, 96, 15), replace_35250, *[str_35251, str_35252], **kwargs_35253)
    
    # Assigning a type to the variable 'repl' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'repl', replace_call_result_35254)
    
    # Assigning a Call to a Name (line 97):
    
    # Assigning a Call to a Name (line 97):
    
    # Call to conv(...): (line 97)
    # Processing the call arguments (line 97)
    # Getting the type of 'repl' (line 97)
    repl_35256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 23), 'repl', False)
    # Processing the call keyword arguments (line 97)
    kwargs_35257 = {}
    # Getting the type of 'conv' (line 97)
    conv_35255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 18), 'conv', False)
    # Calling conv(args, kwargs) (line 97)
    conv_call_result_35258 = invoke(stypy.reporting.localization.Localization(__file__, 97, 18), conv_35255, *[repl_35256], **kwargs_35257)
    
    # Assigning a type to the variable 'thelist' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'thelist', conv_call_result_35258)
    
    # Assigning a Name to a Subscript (line 98):
    
    # Assigning a Name to a Subscript (line 98):
    # Getting the type of 'thelist' (line 98)
    thelist_35259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 22), 'thelist')
    # Getting the type of 'names' (line 98)
    names_35260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'names')
    # Getting the type of 'name' (line 98)
    name_35261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 14), 'name')
    # Storing an element on a container (line 98)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 8), names_35260, (name_35261, thelist_35259))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'names' (line 99)
    names_35262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 11), 'names')
    # Assigning a type to the variable 'stypy_return_type' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'stypy_return_type', names_35262)
    
    # ################# End of 'find_repl_patterns(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'find_repl_patterns' in the type store
    # Getting the type of 'stypy_return_type' (line 91)
    stypy_return_type_35263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_35263)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'find_repl_patterns'
    return stypy_return_type_35263

# Assigning a type to the variable 'find_repl_patterns' (line 91)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 0), 'find_repl_patterns', find_repl_patterns)

# Assigning a Call to a Name (line 101):

# Assigning a Call to a Name (line 101):

# Call to compile(...): (line 101)
# Processing the call arguments (line 101)
str_35266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 21), 'str', '\\A\\\\(?P<index>\\d+)\\Z')
# Processing the call keyword arguments (line 101)
kwargs_35267 = {}
# Getting the type of 're' (line 101)
re_35264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 10), 're', False)
# Obtaining the member 'compile' of a type (line 101)
compile_35265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 10), re_35264, 'compile')
# Calling compile(args, kwargs) (line 101)
compile_call_result_35268 = invoke(stypy.reporting.localization.Localization(__file__, 101, 10), compile_35265, *[str_35266], **kwargs_35267)

# Assigning a type to the variable 'item_re' (line 101)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 0), 'item_re', compile_call_result_35268)

@norecursion
def conv(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'conv'
    module_type_store = module_type_store.open_function_context('conv', 102, 0, False)
    
    # Passed parameters checking function
    conv.stypy_localization = localization
    conv.stypy_type_of_self = None
    conv.stypy_type_store = module_type_store
    conv.stypy_function_name = 'conv'
    conv.stypy_param_names_list = ['astr']
    conv.stypy_varargs_param_name = None
    conv.stypy_kwargs_param_name = None
    conv.stypy_call_defaults = defaults
    conv.stypy_call_varargs = varargs
    conv.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'conv', ['astr'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'conv', localization, ['astr'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'conv(...)' code ##################

    
    # Assigning a Call to a Name (line 103):
    
    # Assigning a Call to a Name (line 103):
    
    # Call to split(...): (line 103)
    # Processing the call arguments (line 103)
    str_35271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 19), 'str', ',')
    # Processing the call keyword arguments (line 103)
    kwargs_35272 = {}
    # Getting the type of 'astr' (line 103)
    astr_35269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'astr', False)
    # Obtaining the member 'split' of a type (line 103)
    split_35270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 8), astr_35269, 'split')
    # Calling split(args, kwargs) (line 103)
    split_call_result_35273 = invoke(stypy.reporting.localization.Localization(__file__, 103, 8), split_35270, *[str_35271], **kwargs_35272)
    
    # Assigning a type to the variable 'b' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'b', split_call_result_35273)
    
    # Assigning a ListComp to a Name (line 104):
    
    # Assigning a ListComp to a Name (line 104):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'b' (line 104)
    b_35278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 28), 'b')
    comprehension_35279 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 9), b_35278)
    # Assigning a type to the variable 'x' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 9), 'x', comprehension_35279)
    
    # Call to strip(...): (line 104)
    # Processing the call keyword arguments (line 104)
    kwargs_35276 = {}
    # Getting the type of 'x' (line 104)
    x_35274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 9), 'x', False)
    # Obtaining the member 'strip' of a type (line 104)
    strip_35275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 9), x_35274, 'strip')
    # Calling strip(args, kwargs) (line 104)
    strip_call_result_35277 = invoke(stypy.reporting.localization.Localization(__file__, 104, 9), strip_35275, *[], **kwargs_35276)
    
    list_35280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 9), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 9), list_35280, strip_call_result_35277)
    # Assigning a type to the variable 'l' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'l', list_35280)
    
    
    # Call to range(...): (line 105)
    # Processing the call arguments (line 105)
    
    # Call to len(...): (line 105)
    # Processing the call arguments (line 105)
    # Getting the type of 'l' (line 105)
    l_35283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 23), 'l', False)
    # Processing the call keyword arguments (line 105)
    kwargs_35284 = {}
    # Getting the type of 'len' (line 105)
    len_35282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 19), 'len', False)
    # Calling len(args, kwargs) (line 105)
    len_call_result_35285 = invoke(stypy.reporting.localization.Localization(__file__, 105, 19), len_35282, *[l_35283], **kwargs_35284)
    
    # Processing the call keyword arguments (line 105)
    kwargs_35286 = {}
    # Getting the type of 'range' (line 105)
    range_35281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 13), 'range', False)
    # Calling range(args, kwargs) (line 105)
    range_call_result_35287 = invoke(stypy.reporting.localization.Localization(__file__, 105, 13), range_35281, *[len_call_result_35285], **kwargs_35286)
    
    # Testing the type of a for loop iterable (line 105)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 105, 4), range_call_result_35287)
    # Getting the type of the for loop variable (line 105)
    for_loop_var_35288 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 105, 4), range_call_result_35287)
    # Assigning a type to the variable 'i' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'i', for_loop_var_35288)
    # SSA begins for a for statement (line 105)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 106):
    
    # Assigning a Call to a Name (line 106):
    
    # Call to match(...): (line 106)
    # Processing the call arguments (line 106)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 106)
    i_35291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 28), 'i', False)
    # Getting the type of 'l' (line 106)
    l_35292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 26), 'l', False)
    # Obtaining the member '__getitem__' of a type (line 106)
    getitem___35293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 26), l_35292, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 106)
    subscript_call_result_35294 = invoke(stypy.reporting.localization.Localization(__file__, 106, 26), getitem___35293, i_35291)
    
    # Processing the call keyword arguments (line 106)
    kwargs_35295 = {}
    # Getting the type of 'item_re' (line 106)
    item_re_35289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'item_re', False)
    # Obtaining the member 'match' of a type (line 106)
    match_35290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 12), item_re_35289, 'match')
    # Calling match(args, kwargs) (line 106)
    match_call_result_35296 = invoke(stypy.reporting.localization.Localization(__file__, 106, 12), match_35290, *[subscript_call_result_35294], **kwargs_35295)
    
    # Assigning a type to the variable 'm' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'm', match_call_result_35296)
    
    # Getting the type of 'm' (line 107)
    m_35297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 11), 'm')
    # Testing the type of an if condition (line 107)
    if_condition_35298 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 107, 8), m_35297)
    # Assigning a type to the variable 'if_condition_35298' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'if_condition_35298', if_condition_35298)
    # SSA begins for if statement (line 107)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 108):
    
    # Assigning a Call to a Name (line 108):
    
    # Call to int(...): (line 108)
    # Processing the call arguments (line 108)
    
    # Call to group(...): (line 108)
    # Processing the call arguments (line 108)
    str_35302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 28), 'str', 'index')
    # Processing the call keyword arguments (line 108)
    kwargs_35303 = {}
    # Getting the type of 'm' (line 108)
    m_35300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 20), 'm', False)
    # Obtaining the member 'group' of a type (line 108)
    group_35301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 20), m_35300, 'group')
    # Calling group(args, kwargs) (line 108)
    group_call_result_35304 = invoke(stypy.reporting.localization.Localization(__file__, 108, 20), group_35301, *[str_35302], **kwargs_35303)
    
    # Processing the call keyword arguments (line 108)
    kwargs_35305 = {}
    # Getting the type of 'int' (line 108)
    int_35299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 16), 'int', False)
    # Calling int(args, kwargs) (line 108)
    int_call_result_35306 = invoke(stypy.reporting.localization.Localization(__file__, 108, 16), int_35299, *[group_call_result_35304], **kwargs_35305)
    
    # Assigning a type to the variable 'j' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'j', int_call_result_35306)
    
    # Assigning a Subscript to a Subscript (line 109):
    
    # Assigning a Subscript to a Subscript (line 109):
    
    # Obtaining the type of the subscript
    # Getting the type of 'j' (line 109)
    j_35307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 21), 'j')
    # Getting the type of 'l' (line 109)
    l_35308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'l')
    # Obtaining the member '__getitem__' of a type (line 109)
    getitem___35309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 19), l_35308, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 109)
    subscript_call_result_35310 = invoke(stypy.reporting.localization.Localization(__file__, 109, 19), getitem___35309, j_35307)
    
    # Getting the type of 'l' (line 109)
    l_35311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'l')
    # Getting the type of 'i' (line 109)
    i_35312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 14), 'i')
    # Storing an element on a container (line 109)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 12), l_35311, (i_35312, subscript_call_result_35310))
    # SSA join for if statement (line 107)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to join(...): (line 110)
    # Processing the call arguments (line 110)
    # Getting the type of 'l' (line 110)
    l_35315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 20), 'l', False)
    # Processing the call keyword arguments (line 110)
    kwargs_35316 = {}
    str_35313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 11), 'str', ',')
    # Obtaining the member 'join' of a type (line 110)
    join_35314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 11), str_35313, 'join')
    # Calling join(args, kwargs) (line 110)
    join_call_result_35317 = invoke(stypy.reporting.localization.Localization(__file__, 110, 11), join_35314, *[l_35315], **kwargs_35316)
    
    # Assigning a type to the variable 'stypy_return_type' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'stypy_return_type', join_call_result_35317)
    
    # ################# End of 'conv(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'conv' in the type store
    # Getting the type of 'stypy_return_type' (line 102)
    stypy_return_type_35318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_35318)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'conv'
    return stypy_return_type_35318

# Assigning a type to the variable 'conv' (line 102)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 0), 'conv', conv)

@norecursion
def unique_key(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'unique_key'
    module_type_store = module_type_store.open_function_context('unique_key', 112, 0, False)
    
    # Passed parameters checking function
    unique_key.stypy_localization = localization
    unique_key.stypy_type_of_self = None
    unique_key.stypy_type_store = module_type_store
    unique_key.stypy_function_name = 'unique_key'
    unique_key.stypy_param_names_list = ['adict']
    unique_key.stypy_varargs_param_name = None
    unique_key.stypy_kwargs_param_name = None
    unique_key.stypy_call_defaults = defaults
    unique_key.stypy_call_varargs = varargs
    unique_key.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'unique_key', ['adict'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'unique_key', localization, ['adict'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'unique_key(...)' code ##################

    str_35319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 4), 'str', ' Obtain a unique key given a dictionary.')
    
    # Assigning a Call to a Name (line 114):
    
    # Assigning a Call to a Name (line 114):
    
    # Call to list(...): (line 114)
    # Processing the call arguments (line 114)
    
    # Call to keys(...): (line 114)
    # Processing the call keyword arguments (line 114)
    kwargs_35323 = {}
    # Getting the type of 'adict' (line 114)
    adict_35321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 19), 'adict', False)
    # Obtaining the member 'keys' of a type (line 114)
    keys_35322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 19), adict_35321, 'keys')
    # Calling keys(args, kwargs) (line 114)
    keys_call_result_35324 = invoke(stypy.reporting.localization.Localization(__file__, 114, 19), keys_35322, *[], **kwargs_35323)
    
    # Processing the call keyword arguments (line 114)
    kwargs_35325 = {}
    # Getting the type of 'list' (line 114)
    list_35320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 14), 'list', False)
    # Calling list(args, kwargs) (line 114)
    list_call_result_35326 = invoke(stypy.reporting.localization.Localization(__file__, 114, 14), list_35320, *[keys_call_result_35324], **kwargs_35325)
    
    # Assigning a type to the variable 'allkeys' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'allkeys', list_call_result_35326)
    
    # Assigning a Name to a Name (line 115):
    
    # Assigning a Name to a Name (line 115):
    # Getting the type of 'False' (line 115)
    False_35327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 11), 'False')
    # Assigning a type to the variable 'done' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'done', False_35327)
    
    # Assigning a Num to a Name (line 116):
    
    # Assigning a Num to a Name (line 116):
    int_35328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 8), 'int')
    # Assigning a type to the variable 'n' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'n', int_35328)
    
    
    # Getting the type of 'done' (line 117)
    done_35329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 14), 'done')
    # Applying the 'not' unary operator (line 117)
    result_not__35330 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 10), 'not', done_35329)
    
    # Testing the type of an if condition (line 117)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 117, 4), result_not__35330)
    # SSA begins for while statement (line 117)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a BinOp to a Name (line 118):
    
    # Assigning a BinOp to a Name (line 118):
    str_35331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 17), 'str', '__l%s')
    # Getting the type of 'n' (line 118)
    n_35332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 28), 'n')
    # Applying the binary operator '%' (line 118)
    result_mod_35333 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 17), '%', str_35331, n_35332)
    
    # Assigning a type to the variable 'newkey' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'newkey', result_mod_35333)
    
    
    # Getting the type of 'newkey' (line 119)
    newkey_35334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 11), 'newkey')
    # Getting the type of 'allkeys' (line 119)
    allkeys_35335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 21), 'allkeys')
    # Applying the binary operator 'in' (line 119)
    result_contains_35336 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 11), 'in', newkey_35334, allkeys_35335)
    
    # Testing the type of an if condition (line 119)
    if_condition_35337 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 119, 8), result_contains_35336)
    # Assigning a type to the variable 'if_condition_35337' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'if_condition_35337', if_condition_35337)
    # SSA begins for if statement (line 119)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'n' (line 120)
    n_35338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'n')
    int_35339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 17), 'int')
    # Applying the binary operator '+=' (line 120)
    result_iadd_35340 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 12), '+=', n_35338, int_35339)
    # Assigning a type to the variable 'n' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'n', result_iadd_35340)
    
    # SSA branch for the else part of an if statement (line 119)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 122):
    
    # Assigning a Name to a Name (line 122):
    # Getting the type of 'True' (line 122)
    True_35341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 19), 'True')
    # Assigning a type to the variable 'done' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'done', True_35341)
    # SSA join for if statement (line 119)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 117)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'newkey' (line 123)
    newkey_35342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 11), 'newkey')
    # Assigning a type to the variable 'stypy_return_type' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'stypy_return_type', newkey_35342)
    
    # ################# End of 'unique_key(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'unique_key' in the type store
    # Getting the type of 'stypy_return_type' (line 112)
    stypy_return_type_35343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_35343)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'unique_key'
    return stypy_return_type_35343

# Assigning a type to the variable 'unique_key' (line 112)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 0), 'unique_key', unique_key)

# Assigning a Call to a Name (line 126):

# Assigning a Call to a Name (line 126):

# Call to compile(...): (line 126)
# Processing the call arguments (line 126)
str_35346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 30), 'str', '\\A\\s*(\\w[\\w\\d]*)\\s*\\Z')
# Processing the call keyword arguments (line 126)
kwargs_35347 = {}
# Getting the type of 're' (line 126)
re_35344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 19), 're', False)
# Obtaining the member 'compile' of a type (line 126)
compile_35345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 19), re_35344, 'compile')
# Calling compile(args, kwargs) (line 126)
compile_call_result_35348 = invoke(stypy.reporting.localization.Localization(__file__, 126, 19), compile_35345, *[str_35346], **kwargs_35347)

# Assigning a type to the variable 'template_name_re' (line 126)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 0), 'template_name_re', compile_call_result_35348)

@norecursion
def expand_sub(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'expand_sub'
    module_type_store = module_type_store.open_function_context('expand_sub', 127, 0, False)
    
    # Passed parameters checking function
    expand_sub.stypy_localization = localization
    expand_sub.stypy_type_of_self = None
    expand_sub.stypy_type_store = module_type_store
    expand_sub.stypy_function_name = 'expand_sub'
    expand_sub.stypy_param_names_list = ['substr', 'names']
    expand_sub.stypy_varargs_param_name = None
    expand_sub.stypy_kwargs_param_name = None
    expand_sub.stypy_call_defaults = defaults
    expand_sub.stypy_call_varargs = varargs
    expand_sub.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'expand_sub', ['substr', 'names'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'expand_sub', localization, ['substr', 'names'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'expand_sub(...)' code ##################

    
    # Assigning a Call to a Name (line 128):
    
    # Assigning a Call to a Name (line 128):
    
    # Call to replace(...): (line 128)
    # Processing the call arguments (line 128)
    str_35351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 28), 'str', '\\>')
    str_35352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 34), 'str', '@rightarrow@')
    # Processing the call keyword arguments (line 128)
    kwargs_35353 = {}
    # Getting the type of 'substr' (line 128)
    substr_35349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 13), 'substr', False)
    # Obtaining the member 'replace' of a type (line 128)
    replace_35350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 13), substr_35349, 'replace')
    # Calling replace(args, kwargs) (line 128)
    replace_call_result_35354 = invoke(stypy.reporting.localization.Localization(__file__, 128, 13), replace_35350, *[str_35351, str_35352], **kwargs_35353)
    
    # Assigning a type to the variable 'substr' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'substr', replace_call_result_35354)
    
    # Assigning a Call to a Name (line 129):
    
    # Assigning a Call to a Name (line 129):
    
    # Call to replace(...): (line 129)
    # Processing the call arguments (line 129)
    str_35357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 28), 'str', '\\<')
    str_35358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 34), 'str', '@leftarrow@')
    # Processing the call keyword arguments (line 129)
    kwargs_35359 = {}
    # Getting the type of 'substr' (line 129)
    substr_35355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 13), 'substr', False)
    # Obtaining the member 'replace' of a type (line 129)
    replace_35356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 13), substr_35355, 'replace')
    # Calling replace(args, kwargs) (line 129)
    replace_call_result_35360 = invoke(stypy.reporting.localization.Localization(__file__, 129, 13), replace_35356, *[str_35357, str_35358], **kwargs_35359)
    
    # Assigning a type to the variable 'substr' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'substr', replace_call_result_35360)
    
    # Assigning a Call to a Name (line 130):
    
    # Assigning a Call to a Name (line 130):
    
    # Call to find_repl_patterns(...): (line 130)
    # Processing the call arguments (line 130)
    # Getting the type of 'substr' (line 130)
    substr_35362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 32), 'substr', False)
    # Processing the call keyword arguments (line 130)
    kwargs_35363 = {}
    # Getting the type of 'find_repl_patterns' (line 130)
    find_repl_patterns_35361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 13), 'find_repl_patterns', False)
    # Calling find_repl_patterns(args, kwargs) (line 130)
    find_repl_patterns_call_result_35364 = invoke(stypy.reporting.localization.Localization(__file__, 130, 13), find_repl_patterns_35361, *[substr_35362], **kwargs_35363)
    
    # Assigning a type to the variable 'lnames' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'lnames', find_repl_patterns_call_result_35364)
    
    # Assigning a Call to a Name (line 131):
    
    # Assigning a Call to a Name (line 131):
    
    # Call to sub(...): (line 131)
    # Processing the call arguments (line 131)
    str_35367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 26), 'str', '<\\1>')
    # Getting the type of 'substr' (line 131)
    substr_35368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 35), 'substr', False)
    # Processing the call keyword arguments (line 131)
    kwargs_35369 = {}
    # Getting the type of 'named_re' (line 131)
    named_re_35365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 13), 'named_re', False)
    # Obtaining the member 'sub' of a type (line 131)
    sub_35366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 13), named_re_35365, 'sub')
    # Calling sub(args, kwargs) (line 131)
    sub_call_result_35370 = invoke(stypy.reporting.localization.Localization(__file__, 131, 13), sub_35366, *[str_35367, substr_35368], **kwargs_35369)
    
    # Assigning a type to the variable 'substr' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'substr', sub_call_result_35370)

    @norecursion
    def listrepl(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'listrepl'
        module_type_store = module_type_store.open_function_context('listrepl', 133, 4, False)
        
        # Passed parameters checking function
        listrepl.stypy_localization = localization
        listrepl.stypy_type_of_self = None
        listrepl.stypy_type_store = module_type_store
        listrepl.stypy_function_name = 'listrepl'
        listrepl.stypy_param_names_list = ['mobj']
        listrepl.stypy_varargs_param_name = None
        listrepl.stypy_kwargs_param_name = None
        listrepl.stypy_call_defaults = defaults
        listrepl.stypy_call_varargs = varargs
        listrepl.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'listrepl', ['mobj'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'listrepl', localization, ['mobj'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'listrepl(...)' code ##################

        
        # Assigning a Call to a Name (line 134):
        
        # Assigning a Call to a Name (line 134):
        
        # Call to conv(...): (line 134)
        # Processing the call arguments (line 134)
        
        # Call to replace(...): (line 134)
        # Processing the call arguments (line 134)
        str_35378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 45), 'str', '\\,')
        str_35379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 51), 'str', '@comma@')
        # Processing the call keyword arguments (line 134)
        kwargs_35380 = {}
        
        # Call to group(...): (line 134)
        # Processing the call arguments (line 134)
        int_35374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 34), 'int')
        # Processing the call keyword arguments (line 134)
        kwargs_35375 = {}
        # Getting the type of 'mobj' (line 134)
        mobj_35372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 23), 'mobj', False)
        # Obtaining the member 'group' of a type (line 134)
        group_35373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 23), mobj_35372, 'group')
        # Calling group(args, kwargs) (line 134)
        group_call_result_35376 = invoke(stypy.reporting.localization.Localization(__file__, 134, 23), group_35373, *[int_35374], **kwargs_35375)
        
        # Obtaining the member 'replace' of a type (line 134)
        replace_35377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 23), group_call_result_35376, 'replace')
        # Calling replace(args, kwargs) (line 134)
        replace_call_result_35381 = invoke(stypy.reporting.localization.Localization(__file__, 134, 23), replace_35377, *[str_35378, str_35379], **kwargs_35380)
        
        # Processing the call keyword arguments (line 134)
        kwargs_35382 = {}
        # Getting the type of 'conv' (line 134)
        conv_35371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 18), 'conv', False)
        # Calling conv(args, kwargs) (line 134)
        conv_call_result_35383 = invoke(stypy.reporting.localization.Localization(__file__, 134, 18), conv_35371, *[replace_call_result_35381], **kwargs_35382)
        
        # Assigning a type to the variable 'thelist' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'thelist', conv_call_result_35383)
        
        
        # Call to match(...): (line 135)
        # Processing the call arguments (line 135)
        # Getting the type of 'thelist' (line 135)
        thelist_35386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 34), 'thelist', False)
        # Processing the call keyword arguments (line 135)
        kwargs_35387 = {}
        # Getting the type of 'template_name_re' (line 135)
        template_name_re_35384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 11), 'template_name_re', False)
        # Obtaining the member 'match' of a type (line 135)
        match_35385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 11), template_name_re_35384, 'match')
        # Calling match(args, kwargs) (line 135)
        match_call_result_35388 = invoke(stypy.reporting.localization.Localization(__file__, 135, 11), match_35385, *[thelist_35386], **kwargs_35387)
        
        # Testing the type of an if condition (line 135)
        if_condition_35389 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 135, 8), match_call_result_35388)
        # Assigning a type to the variable 'if_condition_35389' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'if_condition_35389', if_condition_35389)
        # SSA begins for if statement (line 135)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_35390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 19), 'str', '<%s>')
        # Getting the type of 'thelist' (line 136)
        thelist_35391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 29), 'thelist')
        # Applying the binary operator '%' (line 136)
        result_mod_35392 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 19), '%', str_35390, thelist_35391)
        
        # Assigning a type to the variable 'stypy_return_type' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'stypy_return_type', result_mod_35392)
        # SSA join for if statement (line 135)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Name (line 137):
        
        # Assigning a Name to a Name (line 137):
        # Getting the type of 'None' (line 137)
        None_35393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 15), 'None')
        # Assigning a type to the variable 'name' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'name', None_35393)
        
        
        # Call to keys(...): (line 138)
        # Processing the call keyword arguments (line 138)
        kwargs_35396 = {}
        # Getting the type of 'lnames' (line 138)
        lnames_35394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 19), 'lnames', False)
        # Obtaining the member 'keys' of a type (line 138)
        keys_35395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 19), lnames_35394, 'keys')
        # Calling keys(args, kwargs) (line 138)
        keys_call_result_35397 = invoke(stypy.reporting.localization.Localization(__file__, 138, 19), keys_35395, *[], **kwargs_35396)
        
        # Testing the type of a for loop iterable (line 138)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 138, 8), keys_call_result_35397)
        # Getting the type of the for loop variable (line 138)
        for_loop_var_35398 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 138, 8), keys_call_result_35397)
        # Assigning a type to the variable 'key' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 8), 'key', for_loop_var_35398)
        # SSA begins for a for statement (line 138)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'key' (line 139)
        key_35399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 22), 'key')
        # Getting the type of 'lnames' (line 139)
        lnames_35400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 15), 'lnames')
        # Obtaining the member '__getitem__' of a type (line 139)
        getitem___35401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 15), lnames_35400, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 139)
        subscript_call_result_35402 = invoke(stypy.reporting.localization.Localization(__file__, 139, 15), getitem___35401, key_35399)
        
        # Getting the type of 'thelist' (line 139)
        thelist_35403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 30), 'thelist')
        # Applying the binary operator '==' (line 139)
        result_eq_35404 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 15), '==', subscript_call_result_35402, thelist_35403)
        
        # Testing the type of an if condition (line 139)
        if_condition_35405 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 139, 12), result_eq_35404)
        # Assigning a type to the variable 'if_condition_35405' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'if_condition_35405', if_condition_35405)
        # SSA begins for if statement (line 139)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 140):
        
        # Assigning a Name to a Name (line 140):
        # Getting the type of 'key' (line 140)
        key_35406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 23), 'key')
        # Assigning a type to the variable 'name' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 16), 'name', key_35406)
        # SSA join for if statement (line 139)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 141)
        # Getting the type of 'name' (line 141)
        name_35407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 11), 'name')
        # Getting the type of 'None' (line 141)
        None_35408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 19), 'None')
        
        (may_be_35409, more_types_in_union_35410) = may_be_none(name_35407, None_35408)

        if may_be_35409:

            if more_types_in_union_35410:
                # Runtime conditional SSA (line 141)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 142):
            
            # Assigning a Call to a Name (line 142):
            
            # Call to unique_key(...): (line 142)
            # Processing the call arguments (line 142)
            # Getting the type of 'lnames' (line 142)
            lnames_35412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 30), 'lnames', False)
            # Processing the call keyword arguments (line 142)
            kwargs_35413 = {}
            # Getting the type of 'unique_key' (line 142)
            unique_key_35411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 19), 'unique_key', False)
            # Calling unique_key(args, kwargs) (line 142)
            unique_key_call_result_35414 = invoke(stypy.reporting.localization.Localization(__file__, 142, 19), unique_key_35411, *[lnames_35412], **kwargs_35413)
            
            # Assigning a type to the variable 'name' (line 142)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'name', unique_key_call_result_35414)
            
            # Assigning a Name to a Subscript (line 143):
            
            # Assigning a Name to a Subscript (line 143):
            # Getting the type of 'thelist' (line 143)
            thelist_35415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 27), 'thelist')
            # Getting the type of 'lnames' (line 143)
            lnames_35416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'lnames')
            # Getting the type of 'name' (line 143)
            name_35417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 19), 'name')
            # Storing an element on a container (line 143)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 12), lnames_35416, (name_35417, thelist_35415))

            if more_types_in_union_35410:
                # SSA join for if statement (line 141)
                module_type_store = module_type_store.join_ssa_context()


        
        str_35418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 15), 'str', '<%s>')
        # Getting the type of 'name' (line 144)
        name_35419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 24), 'name')
        # Applying the binary operator '%' (line 144)
        result_mod_35420 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 15), '%', str_35418, name_35419)
        
        # Assigning a type to the variable 'stypy_return_type' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'stypy_return_type', result_mod_35420)
        
        # ################# End of 'listrepl(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'listrepl' in the type store
        # Getting the type of 'stypy_return_type' (line 133)
        stypy_return_type_35421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_35421)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'listrepl'
        return stypy_return_type_35421

    # Assigning a type to the variable 'listrepl' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'listrepl', listrepl)
    
    # Assigning a Call to a Name (line 146):
    
    # Assigning a Call to a Name (line 146):
    
    # Call to sub(...): (line 146)
    # Processing the call arguments (line 146)
    # Getting the type of 'listrepl' (line 146)
    listrepl_35424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 25), 'listrepl', False)
    # Getting the type of 'substr' (line 146)
    substr_35425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 35), 'substr', False)
    # Processing the call keyword arguments (line 146)
    kwargs_35426 = {}
    # Getting the type of 'list_re' (line 146)
    list_re_35422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 13), 'list_re', False)
    # Obtaining the member 'sub' of a type (line 146)
    sub_35423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 13), list_re_35422, 'sub')
    # Calling sub(args, kwargs) (line 146)
    sub_call_result_35427 = invoke(stypy.reporting.localization.Localization(__file__, 146, 13), sub_35423, *[listrepl_35424, substr_35425], **kwargs_35426)
    
    # Assigning a type to the variable 'substr' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'substr', sub_call_result_35427)
    
    # Assigning a Name to a Name (line 149):
    
    # Assigning a Name to a Name (line 149):
    # Getting the type of 'None' (line 149)
    None_35428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 14), 'None')
    # Assigning a type to the variable 'numsubs' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'numsubs', None_35428)
    
    # Assigning a Name to a Name (line 150):
    
    # Assigning a Name to a Name (line 150):
    # Getting the type of 'None' (line 150)
    None_35429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'None')
    # Assigning a type to the variable 'base_rule' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'base_rule', None_35429)
    
    # Assigning a Dict to a Name (line 151):
    
    # Assigning a Dict to a Name (line 151):
    
    # Obtaining an instance of the builtin type 'dict' (line 151)
    dict_35430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 12), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 151)
    
    # Assigning a type to the variable 'rules' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'rules', dict_35430)
    
    
    # Call to findall(...): (line 152)
    # Processing the call arguments (line 152)
    # Getting the type of 'substr' (line 152)
    substr_35433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 33), 'substr', False)
    # Processing the call keyword arguments (line 152)
    kwargs_35434 = {}
    # Getting the type of 'template_re' (line 152)
    template_re_35431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 13), 'template_re', False)
    # Obtaining the member 'findall' of a type (line 152)
    findall_35432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 13), template_re_35431, 'findall')
    # Calling findall(args, kwargs) (line 152)
    findall_call_result_35435 = invoke(stypy.reporting.localization.Localization(__file__, 152, 13), findall_35432, *[substr_35433], **kwargs_35434)
    
    # Testing the type of a for loop iterable (line 152)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 152, 4), findall_call_result_35435)
    # Getting the type of the for loop variable (line 152)
    for_loop_var_35436 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 152, 4), findall_call_result_35435)
    # Assigning a type to the variable 'r' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'r', for_loop_var_35436)
    # SSA begins for a for statement (line 152)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'r' (line 153)
    r_35437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 11), 'r')
    # Getting the type of 'rules' (line 153)
    rules_35438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 20), 'rules')
    # Applying the binary operator 'notin' (line 153)
    result_contains_35439 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 11), 'notin', r_35437, rules_35438)
    
    # Testing the type of an if condition (line 153)
    if_condition_35440 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 8), result_contains_35439)
    # Assigning a type to the variable 'if_condition_35440' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'if_condition_35440', if_condition_35440)
    # SSA begins for if statement (line 153)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 154):
    
    # Assigning a Call to a Name (line 154):
    
    # Call to get(...): (line 154)
    # Processing the call arguments (line 154)
    # Getting the type of 'r' (line 154)
    r_35443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 33), 'r', False)
    
    # Call to get(...): (line 154)
    # Processing the call arguments (line 154)
    # Getting the type of 'r' (line 154)
    r_35446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 46), 'r', False)
    # Getting the type of 'None' (line 154)
    None_35447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 49), 'None', False)
    # Processing the call keyword arguments (line 154)
    kwargs_35448 = {}
    # Getting the type of 'names' (line 154)
    names_35444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 36), 'names', False)
    # Obtaining the member 'get' of a type (line 154)
    get_35445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 36), names_35444, 'get')
    # Calling get(args, kwargs) (line 154)
    get_call_result_35449 = invoke(stypy.reporting.localization.Localization(__file__, 154, 36), get_35445, *[r_35446, None_35447], **kwargs_35448)
    
    # Processing the call keyword arguments (line 154)
    kwargs_35450 = {}
    # Getting the type of 'lnames' (line 154)
    lnames_35441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 22), 'lnames', False)
    # Obtaining the member 'get' of a type (line 154)
    get_35442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 22), lnames_35441, 'get')
    # Calling get(args, kwargs) (line 154)
    get_call_result_35451 = invoke(stypy.reporting.localization.Localization(__file__, 154, 22), get_35442, *[r_35443, get_call_result_35449], **kwargs_35450)
    
    # Assigning a type to the variable 'thelist' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'thelist', get_call_result_35451)
    
    # Type idiom detected: calculating its left and rigth part (line 155)
    # Getting the type of 'thelist' (line 155)
    thelist_35452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 15), 'thelist')
    # Getting the type of 'None' (line 155)
    None_35453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 26), 'None')
    
    (may_be_35454, more_types_in_union_35455) = may_be_none(thelist_35452, None_35453)

    if may_be_35454:

        if more_types_in_union_35455:
            # Runtime conditional SSA (line 155)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to ValueError(...): (line 156)
        # Processing the call arguments (line 156)
        str_35457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 33), 'str', 'No replicates found for <%s>')
        # Getting the type of 'r' (line 156)
        r_35458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 67), 'r', False)
        # Applying the binary operator '%' (line 156)
        result_mod_35459 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 33), '%', str_35457, r_35458)
        
        # Processing the call keyword arguments (line 156)
        kwargs_35460 = {}
        # Getting the type of 'ValueError' (line 156)
        ValueError_35456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 156)
        ValueError_call_result_35461 = invoke(stypy.reporting.localization.Localization(__file__, 156, 22), ValueError_35456, *[result_mod_35459], **kwargs_35460)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 156, 16), ValueError_call_result_35461, 'raise parameter', BaseException)

        if more_types_in_union_35455:
            # SSA join for if statement (line 155)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'r' (line 157)
    r_35462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 15), 'r')
    # Getting the type of 'names' (line 157)
    names_35463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 24), 'names')
    # Applying the binary operator 'notin' (line 157)
    result_contains_35464 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 15), 'notin', r_35462, names_35463)
    
    
    
    # Call to startswith(...): (line 157)
    # Processing the call arguments (line 157)
    str_35467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 57), 'str', '_')
    # Processing the call keyword arguments (line 157)
    kwargs_35468 = {}
    # Getting the type of 'thelist' (line 157)
    thelist_35465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 38), 'thelist', False)
    # Obtaining the member 'startswith' of a type (line 157)
    startswith_35466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 38), thelist_35465, 'startswith')
    # Calling startswith(args, kwargs) (line 157)
    startswith_call_result_35469 = invoke(stypy.reporting.localization.Localization(__file__, 157, 38), startswith_35466, *[str_35467], **kwargs_35468)
    
    # Applying the 'not' unary operator (line 157)
    result_not__35470 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 34), 'not', startswith_call_result_35469)
    
    # Applying the binary operator 'and' (line 157)
    result_and_keyword_35471 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 15), 'and', result_contains_35464, result_not__35470)
    
    # Testing the type of an if condition (line 157)
    if_condition_35472 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 157, 12), result_and_keyword_35471)
    # Assigning a type to the variable 'if_condition_35472' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'if_condition_35472', if_condition_35472)
    # SSA begins for if statement (line 157)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 158):
    
    # Assigning a Name to a Subscript (line 158):
    # Getting the type of 'thelist' (line 158)
    thelist_35473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 27), 'thelist')
    # Getting the type of 'names' (line 158)
    names_35474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 16), 'names')
    # Getting the type of 'r' (line 158)
    r_35475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 22), 'r')
    # Storing an element on a container (line 158)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 16), names_35474, (r_35475, thelist_35473))
    # SSA join for if statement (line 157)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a ListComp to a Name (line 159):
    
    # Assigning a ListComp to a Name (line 159):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to split(...): (line 159)
    # Processing the call arguments (line 159)
    str_35484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 69), 'str', ',')
    # Processing the call keyword arguments (line 159)
    kwargs_35485 = {}
    # Getting the type of 'thelist' (line 159)
    thelist_35482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 55), 'thelist', False)
    # Obtaining the member 'split' of a type (line 159)
    split_35483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 55), thelist_35482, 'split')
    # Calling split(args, kwargs) (line 159)
    split_call_result_35486 = invoke(stypy.reporting.localization.Localization(__file__, 159, 55), split_35483, *[str_35484], **kwargs_35485)
    
    comprehension_35487 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 20), split_call_result_35486)
    # Assigning a type to the variable 'i' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 20), 'i', comprehension_35487)
    
    # Call to replace(...): (line 159)
    # Processing the call arguments (line 159)
    str_35478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 30), 'str', '@comma@')
    str_35479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 41), 'str', ',')
    # Processing the call keyword arguments (line 159)
    kwargs_35480 = {}
    # Getting the type of 'i' (line 159)
    i_35476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 20), 'i', False)
    # Obtaining the member 'replace' of a type (line 159)
    replace_35477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 20), i_35476, 'replace')
    # Calling replace(args, kwargs) (line 159)
    replace_call_result_35481 = invoke(stypy.reporting.localization.Localization(__file__, 159, 20), replace_35477, *[str_35478, str_35479], **kwargs_35480)
    
    list_35488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 20), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 159, 20), list_35488, replace_call_result_35481)
    # Assigning a type to the variable 'rule' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'rule', list_35488)
    
    # Assigning a Call to a Name (line 160):
    
    # Assigning a Call to a Name (line 160):
    
    # Call to len(...): (line 160)
    # Processing the call arguments (line 160)
    # Getting the type of 'rule' (line 160)
    rule_35490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 22), 'rule', False)
    # Processing the call keyword arguments (line 160)
    kwargs_35491 = {}
    # Getting the type of 'len' (line 160)
    len_35489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 18), 'len', False)
    # Calling len(args, kwargs) (line 160)
    len_call_result_35492 = invoke(stypy.reporting.localization.Localization(__file__, 160, 18), len_35489, *[rule_35490], **kwargs_35491)
    
    # Assigning a type to the variable 'num' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'num', len_call_result_35492)
    
    # Type idiom detected: calculating its left and rigth part (line 162)
    # Getting the type of 'numsubs' (line 162)
    numsubs_35493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 15), 'numsubs')
    # Getting the type of 'None' (line 162)
    None_35494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 26), 'None')
    
    (may_be_35495, more_types_in_union_35496) = may_be_none(numsubs_35493, None_35494)

    if may_be_35495:

        if more_types_in_union_35496:
            # Runtime conditional SSA (line 162)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Name (line 163):
        
        # Assigning a Name to a Name (line 163):
        # Getting the type of 'num' (line 163)
        num_35497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 26), 'num')
        # Assigning a type to the variable 'numsubs' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 16), 'numsubs', num_35497)
        
        # Assigning a Name to a Subscript (line 164):
        
        # Assigning a Name to a Subscript (line 164):
        # Getting the type of 'rule' (line 164)
        rule_35498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 27), 'rule')
        # Getting the type of 'rules' (line 164)
        rules_35499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 16), 'rules')
        # Getting the type of 'r' (line 164)
        r_35500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 22), 'r')
        # Storing an element on a container (line 164)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 164, 16), rules_35499, (r_35500, rule_35498))
        
        # Assigning a Name to a Name (line 165):
        
        # Assigning a Name to a Name (line 165):
        # Getting the type of 'r' (line 165)
        r_35501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 28), 'r')
        # Assigning a type to the variable 'base_rule' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 16), 'base_rule', r_35501)

        if more_types_in_union_35496:
            # Runtime conditional SSA for else branch (line 162)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_35495) or more_types_in_union_35496):
        
        
        # Getting the type of 'num' (line 166)
        num_35502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 17), 'num')
        # Getting the type of 'numsubs' (line 166)
        numsubs_35503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 24), 'numsubs')
        # Applying the binary operator '==' (line 166)
        result_eq_35504 = python_operator(stypy.reporting.localization.Localization(__file__, 166, 17), '==', num_35502, numsubs_35503)
        
        # Testing the type of an if condition (line 166)
        if_condition_35505 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 166, 17), result_eq_35504)
        # Assigning a type to the variable 'if_condition_35505' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 17), 'if_condition_35505', if_condition_35505)
        # SSA begins for if statement (line 166)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 167):
        
        # Assigning a Name to a Subscript (line 167):
        # Getting the type of 'rule' (line 167)
        rule_35506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 27), 'rule')
        # Getting the type of 'rules' (line 167)
        rules_35507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 16), 'rules')
        # Getting the type of 'r' (line 167)
        r_35508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 22), 'r')
        # Storing an element on a container (line 167)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 16), rules_35507, (r_35508, rule_35506))
        # SSA branch for the else part of an if statement (line 166)
        module_type_store.open_ssa_branch('else')
        
        # Call to print(...): (line 169)
        # Processing the call arguments (line 169)
        str_35510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 22), 'str', 'Mismatch in number of replacements (base <%s=%s>) for <%s=%s>. Ignoring.')
        
        # Obtaining an instance of the builtin type 'tuple' (line 171)
        tuple_35511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 171)
        # Adding element type (line 171)
        # Getting the type of 'base_rule' (line 171)
        base_rule_35512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 23), 'base_rule', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 23), tuple_35511, base_rule_35512)
        # Adding element type (line 171)
        
        # Call to join(...): (line 171)
        # Processing the call arguments (line 171)
        
        # Obtaining the type of the subscript
        # Getting the type of 'base_rule' (line 171)
        base_rule_35515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 49), 'base_rule', False)
        # Getting the type of 'rules' (line 171)
        rules_35516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 43), 'rules', False)
        # Obtaining the member '__getitem__' of a type (line 171)
        getitem___35517 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 43), rules_35516, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 171)
        subscript_call_result_35518 = invoke(stypy.reporting.localization.Localization(__file__, 171, 43), getitem___35517, base_rule_35515)
        
        # Processing the call keyword arguments (line 171)
        kwargs_35519 = {}
        str_35513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 34), 'str', ',')
        # Obtaining the member 'join' of a type (line 171)
        join_35514 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 34), str_35513, 'join')
        # Calling join(args, kwargs) (line 171)
        join_call_result_35520 = invoke(stypy.reporting.localization.Localization(__file__, 171, 34), join_35514, *[subscript_call_result_35518], **kwargs_35519)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 23), tuple_35511, join_call_result_35520)
        # Adding element type (line 171)
        # Getting the type of 'r' (line 171)
        r_35521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 62), 'r', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 23), tuple_35511, r_35521)
        # Adding element type (line 171)
        # Getting the type of 'thelist' (line 171)
        thelist_35522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 65), 'thelist', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 23), tuple_35511, thelist_35522)
        
        # Applying the binary operator '%' (line 169)
        result_mod_35523 = python_operator(stypy.reporting.localization.Localization(__file__, 169, 22), '%', str_35510, tuple_35511)
        
        # Processing the call keyword arguments (line 169)
        kwargs_35524 = {}
        # Getting the type of 'print' (line 169)
        print_35509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 16), 'print', False)
        # Calling print(args, kwargs) (line 169)
        print_call_result_35525 = invoke(stypy.reporting.localization.Localization(__file__, 169, 16), print_35509, *[result_mod_35523], **kwargs_35524)
        
        # SSA join for if statement (line 166)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_35495 and more_types_in_union_35496):
            # SSA join for if statement (line 162)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 153)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'rules' (line 172)
    rules_35526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 11), 'rules')
    # Applying the 'not' unary operator (line 172)
    result_not__35527 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 7), 'not', rules_35526)
    
    # Testing the type of an if condition (line 172)
    if_condition_35528 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 172, 4), result_not__35527)
    # Assigning a type to the variable 'if_condition_35528' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'if_condition_35528', if_condition_35528)
    # SSA begins for if statement (line 172)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'substr' (line 173)
    substr_35529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 15), 'substr')
    # Assigning a type to the variable 'stypy_return_type' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'stypy_return_type', substr_35529)
    # SSA join for if statement (line 172)
    module_type_store = module_type_store.join_ssa_context()
    

    @norecursion
    def namerepl(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'namerepl'
        module_type_store = module_type_store.open_function_context('namerepl', 175, 4, False)
        
        # Passed parameters checking function
        namerepl.stypy_localization = localization
        namerepl.stypy_type_of_self = None
        namerepl.stypy_type_store = module_type_store
        namerepl.stypy_function_name = 'namerepl'
        namerepl.stypy_param_names_list = ['mobj']
        namerepl.stypy_varargs_param_name = None
        namerepl.stypy_kwargs_param_name = None
        namerepl.stypy_call_defaults = defaults
        namerepl.stypy_call_varargs = varargs
        namerepl.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'namerepl', ['mobj'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'namerepl', localization, ['mobj'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'namerepl(...)' code ##################

        
        # Assigning a Call to a Name (line 176):
        
        # Assigning a Call to a Name (line 176):
        
        # Call to group(...): (line 176)
        # Processing the call arguments (line 176)
        int_35532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 26), 'int')
        # Processing the call keyword arguments (line 176)
        kwargs_35533 = {}
        # Getting the type of 'mobj' (line 176)
        mobj_35530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 15), 'mobj', False)
        # Obtaining the member 'group' of a type (line 176)
        group_35531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 15), mobj_35530, 'group')
        # Calling group(args, kwargs) (line 176)
        group_call_result_35534 = invoke(stypy.reporting.localization.Localization(__file__, 176, 15), group_35531, *[int_35532], **kwargs_35533)
        
        # Assigning a type to the variable 'name' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'name', group_call_result_35534)
        
        # Obtaining the type of the subscript
        # Getting the type of 'k' (line 177)
        k_35535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 45), 'k')
        
        # Call to get(...): (line 177)
        # Processing the call arguments (line 177)
        # Getting the type of 'name' (line 177)
        name_35538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 25), 'name', False)
        # Getting the type of 'k' (line 177)
        k_35539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 32), 'k', False)
        int_35540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 34), 'int')
        # Applying the binary operator '+' (line 177)
        result_add_35541 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 32), '+', k_35539, int_35540)
        
        
        # Obtaining an instance of the builtin type 'list' (line 177)
        list_35542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 37), 'list')
        # Adding type elements to the builtin type 'list' instance (line 177)
        # Adding element type (line 177)
        # Getting the type of 'name' (line 177)
        name_35543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 38), 'name', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 37), list_35542, name_35543)
        
        # Applying the binary operator '*' (line 177)
        result_mul_35544 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 31), '*', result_add_35541, list_35542)
        
        # Processing the call keyword arguments (line 177)
        kwargs_35545 = {}
        # Getting the type of 'rules' (line 177)
        rules_35536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 15), 'rules', False)
        # Obtaining the member 'get' of a type (line 177)
        get_35537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 15), rules_35536, 'get')
        # Calling get(args, kwargs) (line 177)
        get_call_result_35546 = invoke(stypy.reporting.localization.Localization(__file__, 177, 15), get_35537, *[name_35538, result_mul_35544], **kwargs_35545)
        
        # Obtaining the member '__getitem__' of a type (line 177)
        getitem___35547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 177, 15), get_call_result_35546, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 177)
        subscript_call_result_35548 = invoke(stypy.reporting.localization.Localization(__file__, 177, 15), getitem___35547, k_35535)
        
        # Assigning a type to the variable 'stypy_return_type' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 8), 'stypy_return_type', subscript_call_result_35548)
        
        # ################# End of 'namerepl(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'namerepl' in the type store
        # Getting the type of 'stypy_return_type' (line 175)
        stypy_return_type_35549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_35549)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'namerepl'
        return stypy_return_type_35549

    # Assigning a type to the variable 'namerepl' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'namerepl', namerepl)
    
    # Assigning a Str to a Name (line 179):
    
    # Assigning a Str to a Name (line 179):
    str_35550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 13), 'str', '')
    # Assigning a type to the variable 'newstr' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'newstr', str_35550)
    
    
    # Call to range(...): (line 180)
    # Processing the call arguments (line 180)
    # Getting the type of 'numsubs' (line 180)
    numsubs_35552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 19), 'numsubs', False)
    # Processing the call keyword arguments (line 180)
    kwargs_35553 = {}
    # Getting the type of 'range' (line 180)
    range_35551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 13), 'range', False)
    # Calling range(args, kwargs) (line 180)
    range_call_result_35554 = invoke(stypy.reporting.localization.Localization(__file__, 180, 13), range_35551, *[numsubs_35552], **kwargs_35553)
    
    # Testing the type of a for loop iterable (line 180)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 180, 4), range_call_result_35554)
    # Getting the type of the for loop variable (line 180)
    for_loop_var_35555 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 180, 4), range_call_result_35554)
    # Assigning a type to the variable 'k' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'k', for_loop_var_35555)
    # SSA begins for a for statement (line 180)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'newstr' (line 181)
    newstr_35556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'newstr')
    
    # Call to sub(...): (line 181)
    # Processing the call arguments (line 181)
    # Getting the type of 'namerepl' (line 181)
    namerepl_35559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 34), 'namerepl', False)
    # Getting the type of 'substr' (line 181)
    substr_35560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 44), 'substr', False)
    # Processing the call keyword arguments (line 181)
    kwargs_35561 = {}
    # Getting the type of 'template_re' (line 181)
    template_re_35557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 18), 'template_re', False)
    # Obtaining the member 'sub' of a type (line 181)
    sub_35558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 18), template_re_35557, 'sub')
    # Calling sub(args, kwargs) (line 181)
    sub_call_result_35562 = invoke(stypy.reporting.localization.Localization(__file__, 181, 18), sub_35558, *[namerepl_35559, substr_35560], **kwargs_35561)
    
    str_35563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 54), 'str', '\n\n')
    # Applying the binary operator '+' (line 181)
    result_add_35564 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 18), '+', sub_call_result_35562, str_35563)
    
    # Applying the binary operator '+=' (line 181)
    result_iadd_35565 = python_operator(stypy.reporting.localization.Localization(__file__, 181, 8), '+=', newstr_35556, result_add_35564)
    # Assigning a type to the variable 'newstr' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'newstr', result_iadd_35565)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 183):
    
    # Assigning a Call to a Name (line 183):
    
    # Call to replace(...): (line 183)
    # Processing the call arguments (line 183)
    str_35568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 28), 'str', '@rightarrow@')
    str_35569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 44), 'str', '>')
    # Processing the call keyword arguments (line 183)
    kwargs_35570 = {}
    # Getting the type of 'newstr' (line 183)
    newstr_35566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 13), 'newstr', False)
    # Obtaining the member 'replace' of a type (line 183)
    replace_35567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 13), newstr_35566, 'replace')
    # Calling replace(args, kwargs) (line 183)
    replace_call_result_35571 = invoke(stypy.reporting.localization.Localization(__file__, 183, 13), replace_35567, *[str_35568, str_35569], **kwargs_35570)
    
    # Assigning a type to the variable 'newstr' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'newstr', replace_call_result_35571)
    
    # Assigning a Call to a Name (line 184):
    
    # Assigning a Call to a Name (line 184):
    
    # Call to replace(...): (line 184)
    # Processing the call arguments (line 184)
    str_35574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 28), 'str', '@leftarrow@')
    str_35575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 43), 'str', '<')
    # Processing the call keyword arguments (line 184)
    kwargs_35576 = {}
    # Getting the type of 'newstr' (line 184)
    newstr_35572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 13), 'newstr', False)
    # Obtaining the member 'replace' of a type (line 184)
    replace_35573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 13), newstr_35572, 'replace')
    # Calling replace(args, kwargs) (line 184)
    replace_call_result_35577 = invoke(stypy.reporting.localization.Localization(__file__, 184, 13), replace_35573, *[str_35574, str_35575], **kwargs_35576)
    
    # Assigning a type to the variable 'newstr' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'newstr', replace_call_result_35577)
    # Getting the type of 'newstr' (line 185)
    newstr_35578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 11), 'newstr')
    # Assigning a type to the variable 'stypy_return_type' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'stypy_return_type', newstr_35578)
    
    # ################# End of 'expand_sub(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'expand_sub' in the type store
    # Getting the type of 'stypy_return_type' (line 127)
    stypy_return_type_35579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_35579)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'expand_sub'
    return stypy_return_type_35579

# Assigning a type to the variable 'expand_sub' (line 127)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 0), 'expand_sub', expand_sub)

@norecursion
def process_str(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'process_str'
    module_type_store = module_type_store.open_function_context('process_str', 187, 0, False)
    
    # Passed parameters checking function
    process_str.stypy_localization = localization
    process_str.stypy_type_of_self = None
    process_str.stypy_type_store = module_type_store
    process_str.stypy_function_name = 'process_str'
    process_str.stypy_param_names_list = ['allstr']
    process_str.stypy_varargs_param_name = None
    process_str.stypy_kwargs_param_name = None
    process_str.stypy_call_defaults = defaults
    process_str.stypy_call_varargs = varargs
    process_str.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'process_str', ['allstr'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'process_str', localization, ['allstr'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'process_str(...)' code ##################

    
    # Assigning a Name to a Name (line 188):
    
    # Assigning a Name to a Name (line 188):
    # Getting the type of 'allstr' (line 188)
    allstr_35580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 13), 'allstr')
    # Assigning a type to the variable 'newstr' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'newstr', allstr_35580)
    
    # Assigning a Str to a Name (line 189):
    
    # Assigning a Str to a Name (line 189):
    str_35581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 15), 'str', '')
    # Assigning a type to the variable 'writestr' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'writestr', str_35581)
    
    # Assigning a Call to a Name (line 191):
    
    # Assigning a Call to a Name (line 191):
    
    # Call to parse_structure(...): (line 191)
    # Processing the call arguments (line 191)
    # Getting the type of 'newstr' (line 191)
    newstr_35583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 29), 'newstr', False)
    # Processing the call keyword arguments (line 191)
    kwargs_35584 = {}
    # Getting the type of 'parse_structure' (line 191)
    parse_structure_35582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 13), 'parse_structure', False)
    # Calling parse_structure(args, kwargs) (line 191)
    parse_structure_call_result_35585 = invoke(stypy.reporting.localization.Localization(__file__, 191, 13), parse_structure_35582, *[newstr_35583], **kwargs_35584)
    
    # Assigning a type to the variable 'struct' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'struct', parse_structure_call_result_35585)
    
    # Assigning a Num to a Name (line 193):
    
    # Assigning a Num to a Name (line 193):
    int_35586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 13), 'int')
    # Assigning a type to the variable 'oldend' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'oldend', int_35586)
    
    # Assigning a Dict to a Name (line 194):
    
    # Assigning a Dict to a Name (line 194):
    
    # Obtaining an instance of the builtin type 'dict' (line 194)
    dict_35587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 12), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 194)
    
    # Assigning a type to the variable 'names' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'names', dict_35587)
    
    # Call to update(...): (line 195)
    # Processing the call arguments (line 195)
    # Getting the type of '_special_names' (line 195)
    _special_names_35590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 17), '_special_names', False)
    # Processing the call keyword arguments (line 195)
    kwargs_35591 = {}
    # Getting the type of 'names' (line 195)
    names_35588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), 'names', False)
    # Obtaining the member 'update' of a type (line 195)
    update_35589 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 4), names_35588, 'update')
    # Calling update(args, kwargs) (line 195)
    update_call_result_35592 = invoke(stypy.reporting.localization.Localization(__file__, 195, 4), update_35589, *[_special_names_35590], **kwargs_35591)
    
    
    # Getting the type of 'struct' (line 196)
    struct_35593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 15), 'struct')
    # Testing the type of a for loop iterable (line 196)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 196, 4), struct_35593)
    # Getting the type of the for loop variable (line 196)
    for_loop_var_35594 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 196, 4), struct_35593)
    # Assigning a type to the variable 'sub' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'sub', for_loop_var_35594)
    # SSA begins for a for statement (line 196)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'writestr' (line 197)
    writestr_35595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'writestr')
    
    # Obtaining the type of the subscript
    # Getting the type of 'oldend' (line 197)
    oldend_35596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 27), 'oldend')
    
    # Obtaining the type of the subscript
    int_35597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 38), 'int')
    # Getting the type of 'sub' (line 197)
    sub_35598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 34), 'sub')
    # Obtaining the member '__getitem__' of a type (line 197)
    getitem___35599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 34), sub_35598, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 197)
    subscript_call_result_35600 = invoke(stypy.reporting.localization.Localization(__file__, 197, 34), getitem___35599, int_35597)
    
    slice_35601 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 197, 20), oldend_35596, subscript_call_result_35600, None)
    # Getting the type of 'newstr' (line 197)
    newstr_35602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 20), 'newstr')
    # Obtaining the member '__getitem__' of a type (line 197)
    getitem___35603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 20), newstr_35602, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 197)
    subscript_call_result_35604 = invoke(stypy.reporting.localization.Localization(__file__, 197, 20), getitem___35603, slice_35601)
    
    # Applying the binary operator '+=' (line 197)
    result_iadd_35605 = python_operator(stypy.reporting.localization.Localization(__file__, 197, 8), '+=', writestr_35595, subscript_call_result_35604)
    # Assigning a type to the variable 'writestr' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'writestr', result_iadd_35605)
    
    
    # Call to update(...): (line 198)
    # Processing the call arguments (line 198)
    
    # Call to find_repl_patterns(...): (line 198)
    # Processing the call arguments (line 198)
    
    # Obtaining the type of the subscript
    # Getting the type of 'oldend' (line 198)
    oldend_35609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 47), 'oldend', False)
    
    # Obtaining the type of the subscript
    int_35610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 58), 'int')
    # Getting the type of 'sub' (line 198)
    sub_35611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 54), 'sub', False)
    # Obtaining the member '__getitem__' of a type (line 198)
    getitem___35612 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 54), sub_35611, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 198)
    subscript_call_result_35613 = invoke(stypy.reporting.localization.Localization(__file__, 198, 54), getitem___35612, int_35610)
    
    slice_35614 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 198, 40), oldend_35609, subscript_call_result_35613, None)
    # Getting the type of 'newstr' (line 198)
    newstr_35615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 40), 'newstr', False)
    # Obtaining the member '__getitem__' of a type (line 198)
    getitem___35616 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 40), newstr_35615, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 198)
    subscript_call_result_35617 = invoke(stypy.reporting.localization.Localization(__file__, 198, 40), getitem___35616, slice_35614)
    
    # Processing the call keyword arguments (line 198)
    kwargs_35618 = {}
    # Getting the type of 'find_repl_patterns' (line 198)
    find_repl_patterns_35608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 21), 'find_repl_patterns', False)
    # Calling find_repl_patterns(args, kwargs) (line 198)
    find_repl_patterns_call_result_35619 = invoke(stypy.reporting.localization.Localization(__file__, 198, 21), find_repl_patterns_35608, *[subscript_call_result_35617], **kwargs_35618)
    
    # Processing the call keyword arguments (line 198)
    kwargs_35620 = {}
    # Getting the type of 'names' (line 198)
    names_35606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 8), 'names', False)
    # Obtaining the member 'update' of a type (line 198)
    update_35607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 8), names_35606, 'update')
    # Calling update(args, kwargs) (line 198)
    update_call_result_35621 = invoke(stypy.reporting.localization.Localization(__file__, 198, 8), update_35607, *[find_repl_patterns_call_result_35619], **kwargs_35620)
    
    
    # Getting the type of 'writestr' (line 199)
    writestr_35622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'writestr')
    
    # Call to expand_sub(...): (line 199)
    # Processing the call arguments (line 199)
    
    # Obtaining the type of the subscript
    
    # Obtaining the type of the subscript
    int_35624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 42), 'int')
    # Getting the type of 'sub' (line 199)
    sub_35625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 38), 'sub', False)
    # Obtaining the member '__getitem__' of a type (line 199)
    getitem___35626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 38), sub_35625, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 199)
    subscript_call_result_35627 = invoke(stypy.reporting.localization.Localization(__file__, 199, 38), getitem___35626, int_35624)
    
    
    # Obtaining the type of the subscript
    int_35628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 49), 'int')
    # Getting the type of 'sub' (line 199)
    sub_35629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 45), 'sub', False)
    # Obtaining the member '__getitem__' of a type (line 199)
    getitem___35630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 45), sub_35629, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 199)
    subscript_call_result_35631 = invoke(stypy.reporting.localization.Localization(__file__, 199, 45), getitem___35630, int_35628)
    
    slice_35632 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 199, 31), subscript_call_result_35627, subscript_call_result_35631, None)
    # Getting the type of 'newstr' (line 199)
    newstr_35633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 31), 'newstr', False)
    # Obtaining the member '__getitem__' of a type (line 199)
    getitem___35634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 31), newstr_35633, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 199)
    subscript_call_result_35635 = invoke(stypy.reporting.localization.Localization(__file__, 199, 31), getitem___35634, slice_35632)
    
    # Getting the type of 'names' (line 199)
    names_35636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 54), 'names', False)
    # Processing the call keyword arguments (line 199)
    kwargs_35637 = {}
    # Getting the type of 'expand_sub' (line 199)
    expand_sub_35623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 20), 'expand_sub', False)
    # Calling expand_sub(args, kwargs) (line 199)
    expand_sub_call_result_35638 = invoke(stypy.reporting.localization.Localization(__file__, 199, 20), expand_sub_35623, *[subscript_call_result_35635, names_35636], **kwargs_35637)
    
    # Applying the binary operator '+=' (line 199)
    result_iadd_35639 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 8), '+=', writestr_35622, expand_sub_call_result_35638)
    # Assigning a type to the variable 'writestr' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'writestr', result_iadd_35639)
    
    
    # Assigning a Subscript to a Name (line 200):
    
    # Assigning a Subscript to a Name (line 200):
    
    # Obtaining the type of the subscript
    int_35640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 22), 'int')
    # Getting the type of 'sub' (line 200)
    sub_35641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 18), 'sub')
    # Obtaining the member '__getitem__' of a type (line 200)
    getitem___35642 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 18), sub_35641, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 200)
    subscript_call_result_35643 = invoke(stypy.reporting.localization.Localization(__file__, 200, 18), getitem___35642, int_35640)
    
    # Assigning a type to the variable 'oldend' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'oldend', subscript_call_result_35643)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'writestr' (line 201)
    writestr_35644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'writestr')
    
    # Obtaining the type of the subscript
    # Getting the type of 'oldend' (line 201)
    oldend_35645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 23), 'oldend')
    slice_35646 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 201, 16), oldend_35645, None, None)
    # Getting the type of 'newstr' (line 201)
    newstr_35647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'newstr')
    # Obtaining the member '__getitem__' of a type (line 201)
    getitem___35648 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 16), newstr_35647, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 201)
    subscript_call_result_35649 = invoke(stypy.reporting.localization.Localization(__file__, 201, 16), getitem___35648, slice_35646)
    
    # Applying the binary operator '+=' (line 201)
    result_iadd_35650 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 4), '+=', writestr_35644, subscript_call_result_35649)
    # Assigning a type to the variable 'writestr' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'writestr', result_iadd_35650)
    
    # Getting the type of 'writestr' (line 203)
    writestr_35651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 11), 'writestr')
    # Assigning a type to the variable 'stypy_return_type' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 4), 'stypy_return_type', writestr_35651)
    
    # ################# End of 'process_str(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'process_str' in the type store
    # Getting the type of 'stypy_return_type' (line 187)
    stypy_return_type_35652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_35652)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'process_str'
    return stypy_return_type_35652

# Assigning a type to the variable 'process_str' (line 187)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 0), 'process_str', process_str)

# Assigning a Call to a Name (line 205):

# Assigning a Call to a Name (line 205):

# Call to compile(...): (line 205)
# Processing the call arguments (line 205)
str_35655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 28), 'str', '(\\n|\\A)\\s*include\\s*[\'\\"](?P<name>[\\w\\d./\\\\]+[.]src)[\'\\"]')
# Getting the type of 're' (line 205)
re_35656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 90), 're', False)
# Obtaining the member 'I' of a type (line 205)
I_35657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 90), re_35656, 'I')
# Processing the call keyword arguments (line 205)
kwargs_35658 = {}
# Getting the type of 're' (line 205)
re_35653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 17), 're', False)
# Obtaining the member 'compile' of a type (line 205)
compile_35654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 17), re_35653, 'compile')
# Calling compile(args, kwargs) (line 205)
compile_call_result_35659 = invoke(stypy.reporting.localization.Localization(__file__, 205, 17), compile_35654, *[str_35655, I_35657], **kwargs_35658)

# Assigning a type to the variable 'include_src_re' (line 205)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 0), 'include_src_re', compile_call_result_35659)

@norecursion
def resolve_includes(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'resolve_includes'
    module_type_store = module_type_store.open_function_context('resolve_includes', 207, 0, False)
    
    # Passed parameters checking function
    resolve_includes.stypy_localization = localization
    resolve_includes.stypy_type_of_self = None
    resolve_includes.stypy_type_store = module_type_store
    resolve_includes.stypy_function_name = 'resolve_includes'
    resolve_includes.stypy_param_names_list = ['source']
    resolve_includes.stypy_varargs_param_name = None
    resolve_includes.stypy_kwargs_param_name = None
    resolve_includes.stypy_call_defaults = defaults
    resolve_includes.stypy_call_varargs = varargs
    resolve_includes.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'resolve_includes', ['source'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'resolve_includes', localization, ['source'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'resolve_includes(...)' code ##################

    
    # Assigning a Call to a Name (line 208):
    
    # Assigning a Call to a Name (line 208):
    
    # Call to dirname(...): (line 208)
    # Processing the call arguments (line 208)
    # Getting the type of 'source' (line 208)
    source_35663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 24), 'source', False)
    # Processing the call keyword arguments (line 208)
    kwargs_35664 = {}
    # Getting the type of 'os' (line 208)
    os_35660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'os', False)
    # Obtaining the member 'path' of a type (line 208)
    path_35661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 8), os_35660, 'path')
    # Obtaining the member 'dirname' of a type (line 208)
    dirname_35662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 8), path_35661, 'dirname')
    # Calling dirname(args, kwargs) (line 208)
    dirname_call_result_35665 = invoke(stypy.reporting.localization.Localization(__file__, 208, 8), dirname_35662, *[source_35663], **kwargs_35664)
    
    # Assigning a type to the variable 'd' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'd', dirname_call_result_35665)
    
    # Assigning a Call to a Name (line 209):
    
    # Assigning a Call to a Name (line 209):
    
    # Call to open(...): (line 209)
    # Processing the call arguments (line 209)
    # Getting the type of 'source' (line 209)
    source_35667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 15), 'source', False)
    # Processing the call keyword arguments (line 209)
    kwargs_35668 = {}
    # Getting the type of 'open' (line 209)
    open_35666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 10), 'open', False)
    # Calling open(args, kwargs) (line 209)
    open_call_result_35669 = invoke(stypy.reporting.localization.Localization(__file__, 209, 10), open_35666, *[source_35667], **kwargs_35668)
    
    # Assigning a type to the variable 'fid' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'fid', open_call_result_35669)
    
    # Assigning a List to a Name (line 210):
    
    # Assigning a List to a Name (line 210):
    
    # Obtaining an instance of the builtin type 'list' (line 210)
    list_35670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 210)
    
    # Assigning a type to the variable 'lines' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'lines', list_35670)
    
    # Getting the type of 'fid' (line 211)
    fid_35671 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 16), 'fid')
    # Testing the type of a for loop iterable (line 211)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 211, 4), fid_35671)
    # Getting the type of the for loop variable (line 211)
    for_loop_var_35672 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 211, 4), fid_35671)
    # Assigning a type to the variable 'line' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'line', for_loop_var_35672)
    # SSA begins for a for statement (line 211)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 212):
    
    # Assigning a Call to a Name (line 212):
    
    # Call to match(...): (line 212)
    # Processing the call arguments (line 212)
    # Getting the type of 'line' (line 212)
    line_35675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 33), 'line', False)
    # Processing the call keyword arguments (line 212)
    kwargs_35676 = {}
    # Getting the type of 'include_src_re' (line 212)
    include_src_re_35673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'include_src_re', False)
    # Obtaining the member 'match' of a type (line 212)
    match_35674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 12), include_src_re_35673, 'match')
    # Calling match(args, kwargs) (line 212)
    match_call_result_35677 = invoke(stypy.reporting.localization.Localization(__file__, 212, 12), match_35674, *[line_35675], **kwargs_35676)
    
    # Assigning a type to the variable 'm' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 8), 'm', match_call_result_35677)
    
    # Getting the type of 'm' (line 213)
    m_35678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 11), 'm')
    # Testing the type of an if condition (line 213)
    if_condition_35679 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 213, 8), m_35678)
    # Assigning a type to the variable 'if_condition_35679' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'if_condition_35679', if_condition_35679)
    # SSA begins for if statement (line 213)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 214):
    
    # Assigning a Call to a Name (line 214):
    
    # Call to group(...): (line 214)
    # Processing the call arguments (line 214)
    str_35682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 25), 'str', 'name')
    # Processing the call keyword arguments (line 214)
    kwargs_35683 = {}
    # Getting the type of 'm' (line 214)
    m_35680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 17), 'm', False)
    # Obtaining the member 'group' of a type (line 214)
    group_35681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 17), m_35680, 'group')
    # Calling group(args, kwargs) (line 214)
    group_call_result_35684 = invoke(stypy.reporting.localization.Localization(__file__, 214, 17), group_35681, *[str_35682], **kwargs_35683)
    
    # Assigning a type to the variable 'fn' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 12), 'fn', group_call_result_35684)
    
    
    
    # Call to isabs(...): (line 215)
    # Processing the call arguments (line 215)
    # Getting the type of 'fn' (line 215)
    fn_35688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 33), 'fn', False)
    # Processing the call keyword arguments (line 215)
    kwargs_35689 = {}
    # Getting the type of 'os' (line 215)
    os_35685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 19), 'os', False)
    # Obtaining the member 'path' of a type (line 215)
    path_35686 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 19), os_35685, 'path')
    # Obtaining the member 'isabs' of a type (line 215)
    isabs_35687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 19), path_35686, 'isabs')
    # Calling isabs(args, kwargs) (line 215)
    isabs_call_result_35690 = invoke(stypy.reporting.localization.Localization(__file__, 215, 19), isabs_35687, *[fn_35688], **kwargs_35689)
    
    # Applying the 'not' unary operator (line 215)
    result_not__35691 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 15), 'not', isabs_call_result_35690)
    
    # Testing the type of an if condition (line 215)
    if_condition_35692 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 215, 12), result_not__35691)
    # Assigning a type to the variable 'if_condition_35692' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'if_condition_35692', if_condition_35692)
    # SSA begins for if statement (line 215)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 216):
    
    # Assigning a Call to a Name (line 216):
    
    # Call to join(...): (line 216)
    # Processing the call arguments (line 216)
    # Getting the type of 'd' (line 216)
    d_35696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 34), 'd', False)
    # Getting the type of 'fn' (line 216)
    fn_35697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 37), 'fn', False)
    # Processing the call keyword arguments (line 216)
    kwargs_35698 = {}
    # Getting the type of 'os' (line 216)
    os_35693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 21), 'os', False)
    # Obtaining the member 'path' of a type (line 216)
    path_35694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 21), os_35693, 'path')
    # Obtaining the member 'join' of a type (line 216)
    join_35695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 21), path_35694, 'join')
    # Calling join(args, kwargs) (line 216)
    join_call_result_35699 = invoke(stypy.reporting.localization.Localization(__file__, 216, 21), join_35695, *[d_35696, fn_35697], **kwargs_35698)
    
    # Assigning a type to the variable 'fn' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 16), 'fn', join_call_result_35699)
    # SSA join for if statement (line 215)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isfile(...): (line 217)
    # Processing the call arguments (line 217)
    # Getting the type of 'fn' (line 217)
    fn_35703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 30), 'fn', False)
    # Processing the call keyword arguments (line 217)
    kwargs_35704 = {}
    # Getting the type of 'os' (line 217)
    os_35700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 217)
    path_35701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 15), os_35700, 'path')
    # Obtaining the member 'isfile' of a type (line 217)
    isfile_35702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 15), path_35701, 'isfile')
    # Calling isfile(args, kwargs) (line 217)
    isfile_call_result_35705 = invoke(stypy.reporting.localization.Localization(__file__, 217, 15), isfile_35702, *[fn_35703], **kwargs_35704)
    
    # Testing the type of an if condition (line 217)
    if_condition_35706 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 217, 12), isfile_call_result_35705)
    # Assigning a type to the variable 'if_condition_35706' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'if_condition_35706', if_condition_35706)
    # SSA begins for if statement (line 217)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 218)
    # Processing the call arguments (line 218)
    str_35708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 22), 'str', 'Including file')
    # Getting the type of 'fn' (line 218)
    fn_35709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 40), 'fn', False)
    # Processing the call keyword arguments (line 218)
    kwargs_35710 = {}
    # Getting the type of 'print' (line 218)
    print_35707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 16), 'print', False)
    # Calling print(args, kwargs) (line 218)
    print_call_result_35711 = invoke(stypy.reporting.localization.Localization(__file__, 218, 16), print_35707, *[str_35708, fn_35709], **kwargs_35710)
    
    
    # Call to extend(...): (line 219)
    # Processing the call arguments (line 219)
    
    # Call to resolve_includes(...): (line 219)
    # Processing the call arguments (line 219)
    # Getting the type of 'fn' (line 219)
    fn_35715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 46), 'fn', False)
    # Processing the call keyword arguments (line 219)
    kwargs_35716 = {}
    # Getting the type of 'resolve_includes' (line 219)
    resolve_includes_35714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 29), 'resolve_includes', False)
    # Calling resolve_includes(args, kwargs) (line 219)
    resolve_includes_call_result_35717 = invoke(stypy.reporting.localization.Localization(__file__, 219, 29), resolve_includes_35714, *[fn_35715], **kwargs_35716)
    
    # Processing the call keyword arguments (line 219)
    kwargs_35718 = {}
    # Getting the type of 'lines' (line 219)
    lines_35712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 16), 'lines', False)
    # Obtaining the member 'extend' of a type (line 219)
    extend_35713 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 16), lines_35712, 'extend')
    # Calling extend(args, kwargs) (line 219)
    extend_call_result_35719 = invoke(stypy.reporting.localization.Localization(__file__, 219, 16), extend_35713, *[resolve_includes_call_result_35717], **kwargs_35718)
    
    # SSA branch for the else part of an if statement (line 217)
    module_type_store.open_ssa_branch('else')
    
    # Call to append(...): (line 221)
    # Processing the call arguments (line 221)
    # Getting the type of 'line' (line 221)
    line_35722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 29), 'line', False)
    # Processing the call keyword arguments (line 221)
    kwargs_35723 = {}
    # Getting the type of 'lines' (line 221)
    lines_35720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'lines', False)
    # Obtaining the member 'append' of a type (line 221)
    append_35721 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 16), lines_35720, 'append')
    # Calling append(args, kwargs) (line 221)
    append_call_result_35724 = invoke(stypy.reporting.localization.Localization(__file__, 221, 16), append_35721, *[line_35722], **kwargs_35723)
    
    # SSA join for if statement (line 217)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 213)
    module_type_store.open_ssa_branch('else')
    
    # Call to append(...): (line 223)
    # Processing the call arguments (line 223)
    # Getting the type of 'line' (line 223)
    line_35727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 25), 'line', False)
    # Processing the call keyword arguments (line 223)
    kwargs_35728 = {}
    # Getting the type of 'lines' (line 223)
    lines_35725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), 'lines', False)
    # Obtaining the member 'append' of a type (line 223)
    append_35726 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 12), lines_35725, 'append')
    # Calling append(args, kwargs) (line 223)
    append_call_result_35729 = invoke(stypy.reporting.localization.Localization(__file__, 223, 12), append_35726, *[line_35727], **kwargs_35728)
    
    # SSA join for if statement (line 213)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to close(...): (line 224)
    # Processing the call keyword arguments (line 224)
    kwargs_35732 = {}
    # Getting the type of 'fid' (line 224)
    fid_35730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 4), 'fid', False)
    # Obtaining the member 'close' of a type (line 224)
    close_35731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 4), fid_35730, 'close')
    # Calling close(args, kwargs) (line 224)
    close_call_result_35733 = invoke(stypy.reporting.localization.Localization(__file__, 224, 4), close_35731, *[], **kwargs_35732)
    
    # Getting the type of 'lines' (line 225)
    lines_35734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 11), 'lines')
    # Assigning a type to the variable 'stypy_return_type' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'stypy_return_type', lines_35734)
    
    # ################# End of 'resolve_includes(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'resolve_includes' in the type store
    # Getting the type of 'stypy_return_type' (line 207)
    stypy_return_type_35735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_35735)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'resolve_includes'
    return stypy_return_type_35735

# Assigning a type to the variable 'resolve_includes' (line 207)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 0), 'resolve_includes', resolve_includes)

@norecursion
def process_file(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'process_file'
    module_type_store = module_type_store.open_function_context('process_file', 227, 0, False)
    
    # Passed parameters checking function
    process_file.stypy_localization = localization
    process_file.stypy_type_of_self = None
    process_file.stypy_type_store = module_type_store
    process_file.stypy_function_name = 'process_file'
    process_file.stypy_param_names_list = ['source']
    process_file.stypy_varargs_param_name = None
    process_file.stypy_kwargs_param_name = None
    process_file.stypy_call_defaults = defaults
    process_file.stypy_call_varargs = varargs
    process_file.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'process_file', ['source'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'process_file', localization, ['source'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'process_file(...)' code ##################

    
    # Assigning a Call to a Name (line 228):
    
    # Assigning a Call to a Name (line 228):
    
    # Call to resolve_includes(...): (line 228)
    # Processing the call arguments (line 228)
    # Getting the type of 'source' (line 228)
    source_35737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 29), 'source', False)
    # Processing the call keyword arguments (line 228)
    kwargs_35738 = {}
    # Getting the type of 'resolve_includes' (line 228)
    resolve_includes_35736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 12), 'resolve_includes', False)
    # Calling resolve_includes(args, kwargs) (line 228)
    resolve_includes_call_result_35739 = invoke(stypy.reporting.localization.Localization(__file__, 228, 12), resolve_includes_35736, *[source_35737], **kwargs_35738)
    
    # Assigning a type to the variable 'lines' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'lines', resolve_includes_call_result_35739)
    
    # Call to process_str(...): (line 229)
    # Processing the call arguments (line 229)
    
    # Call to join(...): (line 229)
    # Processing the call arguments (line 229)
    # Getting the type of 'lines' (line 229)
    lines_35743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 31), 'lines', False)
    # Processing the call keyword arguments (line 229)
    kwargs_35744 = {}
    str_35741 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 23), 'str', '')
    # Obtaining the member 'join' of a type (line 229)
    join_35742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 23), str_35741, 'join')
    # Calling join(args, kwargs) (line 229)
    join_call_result_35745 = invoke(stypy.reporting.localization.Localization(__file__, 229, 23), join_35742, *[lines_35743], **kwargs_35744)
    
    # Processing the call keyword arguments (line 229)
    kwargs_35746 = {}
    # Getting the type of 'process_str' (line 229)
    process_str_35740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 11), 'process_str', False)
    # Calling process_str(args, kwargs) (line 229)
    process_str_call_result_35747 = invoke(stypy.reporting.localization.Localization(__file__, 229, 11), process_str_35740, *[join_call_result_35745], **kwargs_35746)
    
    # Assigning a type to the variable 'stypy_return_type' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'stypy_return_type', process_str_call_result_35747)
    
    # ################# End of 'process_file(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'process_file' in the type store
    # Getting the type of 'stypy_return_type' (line 227)
    stypy_return_type_35748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_35748)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'process_file'
    return stypy_return_type_35748

# Assigning a type to the variable 'process_file' (line 227)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 0), 'process_file', process_file)

# Assigning a Call to a Name (line 231):

# Assigning a Call to a Name (line 231):

# Call to find_repl_patterns(...): (line 231)
# Processing the call arguments (line 231)
str_35750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, (-1)), 'str', '\n<_c=s,d,c,z>\n<_t=real,double precision,complex,double complex>\n<prefix=s,d,c,z>\n<ftype=real,double precision,complex,double complex>\n<ctype=float,double,complex_float,complex_double>\n<ftypereal=real,double precision,\\0,\\1>\n<ctypereal=float,double,\\0,\\1>\n')
# Processing the call keyword arguments (line 231)
kwargs_35751 = {}
# Getting the type of 'find_repl_patterns' (line 231)
find_repl_patterns_35749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 17), 'find_repl_patterns', False)
# Calling find_repl_patterns(args, kwargs) (line 231)
find_repl_patterns_call_result_35752 = invoke(stypy.reporting.localization.Localization(__file__, 231, 17), find_repl_patterns_35749, *[str_35750], **kwargs_35751)

# Assigning a type to the variable '_special_names' (line 231)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 0), '_special_names', find_repl_patterns_call_result_35752)

if (__name__ == '__main__'):
    
    
    # SSA begins for try-except statement (line 243)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 244):
    
    # Assigning a Subscript to a Name (line 244):
    
    # Obtaining the type of the subscript
    int_35753 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 244, 24), 'int')
    # Getting the type of 'sys' (line 244)
    sys_35754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 15), 'sys')
    # Obtaining the member 'argv' of a type (line 244)
    argv_35755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 15), sys_35754, 'argv')
    # Obtaining the member '__getitem__' of a type (line 244)
    getitem___35756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 15), argv_35755, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 244)
    subscript_call_result_35757 = invoke(stypy.reporting.localization.Localization(__file__, 244, 15), getitem___35756, int_35753)
    
    # Assigning a type to the variable 'file' (line 244)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'file', subscript_call_result_35757)
    # SSA branch for the except part of a try statement (line 243)
    # SSA branch for the except 'IndexError' branch of a try statement (line 243)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Attribute to a Name (line 246):
    
    # Assigning a Attribute to a Name (line 246):
    # Getting the type of 'sys' (line 246)
    sys_35758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 14), 'sys')
    # Obtaining the member 'stdin' of a type (line 246)
    stdin_35759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 14), sys_35758, 'stdin')
    # Assigning a type to the variable 'fid' (line 246)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 8), 'fid', stdin_35759)
    
    # Assigning a Attribute to a Name (line 247):
    
    # Assigning a Attribute to a Name (line 247):
    # Getting the type of 'sys' (line 247)
    sys_35760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 18), 'sys')
    # Obtaining the member 'stdout' of a type (line 247)
    stdout_35761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 18), sys_35760, 'stdout')
    # Assigning a type to the variable 'outfile' (line 247)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 8), 'outfile', stdout_35761)
    # SSA branch for the else branch of a try statement (line 243)
    module_type_store.open_ssa_branch('except else')
    
    # Assigning a Call to a Name (line 249):
    
    # Assigning a Call to a Name (line 249):
    
    # Call to open(...): (line 249)
    # Processing the call arguments (line 249)
    # Getting the type of 'file' (line 249)
    file_35763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 19), 'file', False)
    str_35764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 249, 25), 'str', 'r')
    # Processing the call keyword arguments (line 249)
    kwargs_35765 = {}
    # Getting the type of 'open' (line 249)
    open_35762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 14), 'open', False)
    # Calling open(args, kwargs) (line 249)
    open_call_result_35766 = invoke(stypy.reporting.localization.Localization(__file__, 249, 14), open_35762, *[file_35763, str_35764], **kwargs_35765)
    
    # Assigning a type to the variable 'fid' (line 249)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 249, 8), 'fid', open_call_result_35766)
    
    # Assigning a Call to a Tuple (line 250):
    
    # Assigning a Call to a Name:
    
    # Call to splitext(...): (line 250)
    # Processing the call arguments (line 250)
    # Getting the type of 'file' (line 250)
    file_35770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 39), 'file', False)
    # Processing the call keyword arguments (line 250)
    kwargs_35771 = {}
    # Getting the type of 'os' (line 250)
    os_35767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 22), 'os', False)
    # Obtaining the member 'path' of a type (line 250)
    path_35768 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 22), os_35767, 'path')
    # Obtaining the member 'splitext' of a type (line 250)
    splitext_35769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 22), path_35768, 'splitext')
    # Calling splitext(args, kwargs) (line 250)
    splitext_call_result_35772 = invoke(stypy.reporting.localization.Localization(__file__, 250, 22), splitext_35769, *[file_35770], **kwargs_35771)
    
    # Assigning a type to the variable 'call_assignment_35095' (line 250)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'call_assignment_35095', splitext_call_result_35772)
    
    # Assigning a Call to a Name (line 250):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_35775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 8), 'int')
    # Processing the call keyword arguments
    kwargs_35776 = {}
    # Getting the type of 'call_assignment_35095' (line 250)
    call_assignment_35095_35773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'call_assignment_35095', False)
    # Obtaining the member '__getitem__' of a type (line 250)
    getitem___35774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 8), call_assignment_35095_35773, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_35777 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___35774, *[int_35775], **kwargs_35776)
    
    # Assigning a type to the variable 'call_assignment_35096' (line 250)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'call_assignment_35096', getitem___call_result_35777)
    
    # Assigning a Name to a Name (line 250):
    # Getting the type of 'call_assignment_35096' (line 250)
    call_assignment_35096_35778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'call_assignment_35096')
    # Assigning a type to the variable 'base' (line 250)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 9), 'base', call_assignment_35096_35778)
    
    # Assigning a Call to a Name (line 250):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_35781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 250, 8), 'int')
    # Processing the call keyword arguments
    kwargs_35782 = {}
    # Getting the type of 'call_assignment_35095' (line 250)
    call_assignment_35095_35779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'call_assignment_35095', False)
    # Obtaining the member '__getitem__' of a type (line 250)
    getitem___35780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 8), call_assignment_35095_35779, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_35783 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___35780, *[int_35781], **kwargs_35782)
    
    # Assigning a type to the variable 'call_assignment_35097' (line 250)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'call_assignment_35097', getitem___call_result_35783)
    
    # Assigning a Name to a Name (line 250):
    # Getting the type of 'call_assignment_35097' (line 250)
    call_assignment_35097_35784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'call_assignment_35097')
    # Assigning a type to the variable 'ext' (line 250)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 250, 15), 'ext', call_assignment_35097_35784)
    
    # Assigning a Name to a Name (line 251):
    
    # Assigning a Name to a Name (line 251):
    # Getting the type of 'base' (line 251)
    base_35785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 18), 'base')
    # Assigning a type to the variable 'newname' (line 251)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'newname', base_35785)
    
    # Assigning a Call to a Name (line 252):
    
    # Assigning a Call to a Name (line 252):
    
    # Call to open(...): (line 252)
    # Processing the call arguments (line 252)
    # Getting the type of 'newname' (line 252)
    newname_35787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 23), 'newname', False)
    str_35788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 32), 'str', 'w')
    # Processing the call keyword arguments (line 252)
    kwargs_35789 = {}
    # Getting the type of 'open' (line 252)
    open_35786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 18), 'open', False)
    # Calling open(args, kwargs) (line 252)
    open_call_result_35790 = invoke(stypy.reporting.localization.Localization(__file__, 252, 18), open_35786, *[newname_35787, str_35788], **kwargs_35789)
    
    # Assigning a type to the variable 'outfile' (line 252)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'outfile', open_call_result_35790)
    # SSA join for try-except statement (line 243)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 254):
    
    # Assigning a Call to a Name (line 254):
    
    # Call to read(...): (line 254)
    # Processing the call keyword arguments (line 254)
    kwargs_35793 = {}
    # Getting the type of 'fid' (line 254)
    fid_35791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 13), 'fid', False)
    # Obtaining the member 'read' of a type (line 254)
    read_35792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 13), fid_35791, 'read')
    # Calling read(args, kwargs) (line 254)
    read_call_result_35794 = invoke(stypy.reporting.localization.Localization(__file__, 254, 13), read_35792, *[], **kwargs_35793)
    
    # Assigning a type to the variable 'allstr' (line 254)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 4), 'allstr', read_call_result_35794)
    
    # Assigning a Call to a Name (line 255):
    
    # Assigning a Call to a Name (line 255):
    
    # Call to process_str(...): (line 255)
    # Processing the call arguments (line 255)
    # Getting the type of 'allstr' (line 255)
    allstr_35796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 27), 'allstr', False)
    # Processing the call keyword arguments (line 255)
    kwargs_35797 = {}
    # Getting the type of 'process_str' (line 255)
    process_str_35795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 15), 'process_str', False)
    # Calling process_str(args, kwargs) (line 255)
    process_str_call_result_35798 = invoke(stypy.reporting.localization.Localization(__file__, 255, 15), process_str_35795, *[allstr_35796], **kwargs_35797)
    
    # Assigning a type to the variable 'writestr' (line 255)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 4), 'writestr', process_str_call_result_35798)
    
    # Call to write(...): (line 256)
    # Processing the call arguments (line 256)
    # Getting the type of 'writestr' (line 256)
    writestr_35801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 18), 'writestr', False)
    # Processing the call keyword arguments (line 256)
    kwargs_35802 = {}
    # Getting the type of 'outfile' (line 256)
    outfile_35799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'outfile', False)
    # Obtaining the member 'write' of a type (line 256)
    write_35800 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 256, 4), outfile_35799, 'write')
    # Calling write(args, kwargs) (line 256)
    write_call_result_35803 = invoke(stypy.reporting.localization.Localization(__file__, 256, 4), write_35800, *[writestr_35801], **kwargs_35802)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
