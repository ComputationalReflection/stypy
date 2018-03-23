
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # pisang - a simple sat solver in Python
2: # (c) mark.dufour@gmail.com
3: import os
4: 
5: 
6: def Relative(path):
7:     return os.path.join(os.path.dirname(__file__), path)
8: 
9: 
10: argv = ['', Relative('testdata/uuf250-010.cnf')]
11: 
12: cnf = [l.strip().split() for l in file(argv[1]) if l[0] not in 'c%0\n']
13: clauses = [[int(x) for x in m[:-1]] for m in cnf if m[0] != 'p']
14: nrofvars = [int(n[2]) for n in cnf if n[0] == 'p'][0]
15: vars = range(nrofvars + 1)
16: occurrence = [[] for l in vars + range(-nrofvars, 0)]
17: for clause in clauses:
18:     for lit in clause: occurrence[lit].append(clause)
19: fixedt = [-1 for var in vars]
20: 
21: nodecount = 0
22: 
23: 
24: def solve_rec():
25:     global nodecount
26:     nodecount += 1
27: 
28:     if not -1 in fixedt[1:]:
29:         ##        print 'v', ' '.join([str((2*fixedt[i]-1)*i) for i in vars[1:]])
30:         return 1
31: 
32:     la_mods = []
33:     var = lookahead(la_mods)
34:     if not var:
35:         return backtrack(la_mods)
36:     for choice in [var, -var]:
37:         prop_mods = []
38:         if propagate(choice, prop_mods) and solve_rec():
39:             return 1
40:         backtrack(prop_mods)
41:     return backtrack(la_mods)
42: 
43: 
44: def propagate(lit, mods):
45:     global bincount
46: 
47:     current = len(mods)
48:     mods.append(lit)
49: 
50:     while 1:
51:         if fixedt[abs(lit)] == -1:
52:             fixedt[abs(lit)] = int(lit > 0)
53:             for clause in occurrence[-lit]:
54:                 cl_len = length(clause)
55:                 if cl_len == 0:
56:                     return 0
57:                 elif cl_len == 1:
58:                     mods.append(unfixed(clause))
59:                 elif cl_len == 2:
60:                     bincount += 1
61: 
62:         elif fixedt[abs(lit)] != int(lit > 0):
63:             return 0
64: 
65:         current += 1
66:         if current == len(mods):
67:             break
68:         lit = mods[current]
69: 
70:     return 1
71: 
72: 
73: def lookahead(mods):
74:     global bincount
75: 
76:     dif = [-1 for var in vars]
77:     for var in unfixed_vars():
78:         score = []
79:         for choice in [var, -var]:
80:             prop_mods = []
81:             bincount = 0
82:             prop = propagate(choice, prop_mods)
83:             backtrack(prop_mods)
84:             if not prop:
85:                 if not propagate(-choice, mods):
86:                     return 0
87:                 break
88:             score.append(bincount)
89:         dif[var] = reduce(lambda x, y: 1024 * x * y + x + y, score, 0)
90: 
91:     return dif.index(max(dif))
92: 
93: 
94: def backtrack(mods):
95:     for lit in mods:
96:         fixedt[abs(lit)] = -1
97:     return 0
98: 
99: 
100: def length(clause):
101:     len = 0
102:     for lit in clause:
103:         fixed = fixedt[abs(lit)]
104:         if fixed == int(lit > 0):
105:             return -1
106:         if fixed == -1:
107:             len += 1
108:     return len
109: 
110: 
111: def unfixed(clause):
112:     for lit in clause:
113:         fixed = fixedt[abs(lit)]
114:         if fixed == -1:
115:             return lit
116: 
117: 
118: def unfixed_vars():
119:     return [var for var in vars[1:] if fixedt[var] == -1]
120: 
121: 
122: class SatSolverRun:
123:     def run(self):
124:         nodecount = 0
125:         if not solve_rec():
126:             pass
127:         ##        print 'unsatisfiable', nodecount
128:         return True
129: 
130: 
131: def run():
132:     s = SatSolverRun()
133:     s.run()
134: 
135: 
136: run()
137: 

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


@norecursion
def Relative(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Relative'
    module_type_store = module_type_store.open_function_context('Relative', 6, 0, False)
    
    # Passed parameters checking function
    Relative.stypy_localization = localization
    Relative.stypy_type_of_self = None
    Relative.stypy_type_store = module_type_store
    Relative.stypy_function_name = 'Relative'
    Relative.stypy_param_names_list = ['path']
    Relative.stypy_varargs_param_name = None
    Relative.stypy_kwargs_param_name = None
    Relative.stypy_call_defaults = defaults
    Relative.stypy_call_varargs = varargs
    Relative.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'Relative', ['path'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'Relative', localization, ['path'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'Relative(...)' code ##################

    
    # Call to join(...): (line 7)
    # Processing the call arguments (line 7)
    
    # Call to dirname(...): (line 7)
    # Processing the call arguments (line 7)
    # Getting the type of '__file__' (line 7)
    file___7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 40), '__file__', False)
    # Processing the call keyword arguments (line 7)
    kwargs_8 = {}
    # Getting the type of 'os' (line 7)
    os_4 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 24), 'os', False)
    # Obtaining the member 'path' of a type (line 7)
    path_5 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 24), os_4, 'path')
    # Obtaining the member 'dirname' of a type (line 7)
    dirname_6 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 24), path_5, 'dirname')
    # Calling dirname(args, kwargs) (line 7)
    dirname_call_result_9 = invoke(stypy.reporting.localization.Localization(__file__, 7, 24), dirname_6, *[file___7], **kwargs_8)
    
    # Getting the type of 'path' (line 7)
    path_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 51), 'path', False)
    # Processing the call keyword arguments (line 7)
    kwargs_11 = {}
    # Getting the type of 'os' (line 7)
    os_1 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 7)
    path_2 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 11), os_1, 'path')
    # Obtaining the member 'join' of a type (line 7)
    join_3 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 11), path_2, 'join')
    # Calling join(args, kwargs) (line 7)
    join_call_result_12 = invoke(stypy.reporting.localization.Localization(__file__, 7, 11), join_3, *[dirname_call_result_9, path_10], **kwargs_11)
    
    # Assigning a type to the variable 'stypy_return_type' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'stypy_return_type', join_call_result_12)
    
    # ################# End of 'Relative(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Relative' in the type store
    # Getting the type of 'stypy_return_type' (line 6)
    stypy_return_type_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_13)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Relative'
    return stypy_return_type_13

# Assigning a type to the variable 'Relative' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'Relative', Relative)

# Assigning a List to a Name (line 10):

# Obtaining an instance of the builtin type 'list' (line 10)
list_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 7), 'list')
# Adding type elements to the builtin type 'list' instance (line 10)
# Adding element type (line 10)
str_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 8), 'str', '')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 7), list_14, str_15)
# Adding element type (line 10)

# Call to Relative(...): (line 10)
# Processing the call arguments (line 10)
str_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 21), 'str', 'testdata/uuf250-010.cnf')
# Processing the call keyword arguments (line 10)
kwargs_18 = {}
# Getting the type of 'Relative' (line 10)
Relative_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 12), 'Relative', False)
# Calling Relative(args, kwargs) (line 10)
Relative_call_result_19 = invoke(stypy.reporting.localization.Localization(__file__, 10, 12), Relative_16, *[str_17], **kwargs_18)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 10, 7), list_14, Relative_call_result_19)

# Assigning a type to the variable 'argv' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'argv', list_14)

# Assigning a ListComp to a Name (line 12):
# Calculating list comprehension
# Calculating comprehension expression

# Call to file(...): (line 12)
# Processing the call arguments (line 12)

# Obtaining the type of the subscript
int_34 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 44), 'int')
# Getting the type of 'argv' (line 12)
argv_35 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 39), 'argv', False)
# Obtaining the member '__getitem__' of a type (line 12)
getitem___36 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 39), argv_35, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 12)
subscript_call_result_37 = invoke(stypy.reporting.localization.Localization(__file__, 12, 39), getitem___36, int_34)

# Processing the call keyword arguments (line 12)
kwargs_38 = {}
# Getting the type of 'file' (line 12)
file_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 34), 'file', False)
# Calling file(args, kwargs) (line 12)
file_call_result_39 = invoke(stypy.reporting.localization.Localization(__file__, 12, 34), file_33, *[subscript_call_result_37], **kwargs_38)

comprehension_40 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 7), file_call_result_39)
# Assigning a type to the variable 'l' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 7), 'l', comprehension_40)


# Obtaining the type of the subscript
int_27 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 53), 'int')
# Getting the type of 'l' (line 12)
l_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 51), 'l')
# Obtaining the member '__getitem__' of a type (line 12)
getitem___29 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 51), l_28, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 12)
subscript_call_result_30 = invoke(stypy.reporting.localization.Localization(__file__, 12, 51), getitem___29, int_27)

str_31 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 63), 'str', 'c%0\n')
# Applying the binary operator 'notin' (line 12)
result_contains_32 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 51), 'notin', subscript_call_result_30, str_31)


# Call to split(...): (line 12)
# Processing the call keyword arguments (line 12)
kwargs_25 = {}

# Call to strip(...): (line 12)
# Processing the call keyword arguments (line 12)
kwargs_22 = {}
# Getting the type of 'l' (line 12)
l_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 7), 'l', False)
# Obtaining the member 'strip' of a type (line 12)
strip_21 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 7), l_20, 'strip')
# Calling strip(args, kwargs) (line 12)
strip_call_result_23 = invoke(stypy.reporting.localization.Localization(__file__, 12, 7), strip_21, *[], **kwargs_22)

# Obtaining the member 'split' of a type (line 12)
split_24 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 7), strip_call_result_23, 'split')
# Calling split(args, kwargs) (line 12)
split_call_result_26 = invoke(stypy.reporting.localization.Localization(__file__, 12, 7), split_24, *[], **kwargs_25)

list_41 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 7), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 7), list_41, split_call_result_26)
# Assigning a type to the variable 'cnf' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'cnf', list_41)

# Assigning a ListComp to a Name (line 13):
# Calculating list comprehension
# Calculating comprehension expression
# Getting the type of 'cnf' (line 13)
cnf_59 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 45), 'cnf')
comprehension_60 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 11), cnf_59)
# Assigning a type to the variable 'm' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 11), 'm', comprehension_60)


# Obtaining the type of the subscript
int_53 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 54), 'int')
# Getting the type of 'm' (line 13)
m_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 52), 'm')
# Obtaining the member '__getitem__' of a type (line 13)
getitem___55 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 52), m_54, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 13)
subscript_call_result_56 = invoke(stypy.reporting.localization.Localization(__file__, 13, 52), getitem___55, int_53)

str_57 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 60), 'str', 'p')
# Applying the binary operator '!=' (line 13)
result_ne_58 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 52), '!=', subscript_call_result_56, str_57)

# Calculating list comprehension
# Calculating comprehension expression

# Obtaining the type of the subscript
int_46 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 31), 'int')
slice_47 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 13, 28), None, int_46, None)
# Getting the type of 'm' (line 13)
m_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 28), 'm')
# Obtaining the member '__getitem__' of a type (line 13)
getitem___49 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 28), m_48, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 13)
subscript_call_result_50 = invoke(stypy.reporting.localization.Localization(__file__, 13, 28), getitem___49, slice_47)

comprehension_51 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 12), subscript_call_result_50)
# Assigning a type to the variable 'x' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 12), 'x', comprehension_51)

# Call to int(...): (line 13)
# Processing the call arguments (line 13)
# Getting the type of 'x' (line 13)
x_43 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 16), 'x', False)
# Processing the call keyword arguments (line 13)
kwargs_44 = {}
# Getting the type of 'int' (line 13)
int_42 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 12), 'int', False)
# Calling int(args, kwargs) (line 13)
int_call_result_45 = invoke(stypy.reporting.localization.Localization(__file__, 13, 12), int_42, *[x_43], **kwargs_44)

list_52 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 12), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 12), list_52, int_call_result_45)
list_61 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 11), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 11), list_61, list_52)
# Assigning a type to the variable 'clauses' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'clauses', list_61)

# Assigning a Subscript to a Name (line 14):

# Obtaining the type of the subscript
int_62 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 51), 'int')
# Calculating list comprehension
# Calculating comprehension expression
# Getting the type of 'cnf' (line 14)
cnf_76 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 31), 'cnf')
comprehension_77 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), cnf_76)
# Assigning a type to the variable 'n' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 12), 'n', comprehension_77)


# Obtaining the type of the subscript
int_70 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 40), 'int')
# Getting the type of 'n' (line 14)
n_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 38), 'n')
# Obtaining the member '__getitem__' of a type (line 14)
getitem___72 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 38), n_71, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 14)
subscript_call_result_73 = invoke(stypy.reporting.localization.Localization(__file__, 14, 38), getitem___72, int_70)

str_74 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 46), 'str', 'p')
# Applying the binary operator '==' (line 14)
result_eq_75 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 38), '==', subscript_call_result_73, str_74)


# Call to int(...): (line 14)
# Processing the call arguments (line 14)

# Obtaining the type of the subscript
int_64 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 18), 'int')
# Getting the type of 'n' (line 14)
n_65 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 16), 'n', False)
# Obtaining the member '__getitem__' of a type (line 14)
getitem___66 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 16), n_65, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 14)
subscript_call_result_67 = invoke(stypy.reporting.localization.Localization(__file__, 14, 16), getitem___66, int_64)

# Processing the call keyword arguments (line 14)
kwargs_68 = {}
# Getting the type of 'int' (line 14)
int_63 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 12), 'int', False)
# Calling int(args, kwargs) (line 14)
int_call_result_69 = invoke(stypy.reporting.localization.Localization(__file__, 14, 12), int_63, *[subscript_call_result_67], **kwargs_68)

list_78 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 12), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), list_78, int_call_result_69)
# Obtaining the member '__getitem__' of a type (line 14)
getitem___79 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 12), list_78, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 14)
subscript_call_result_80 = invoke(stypy.reporting.localization.Localization(__file__, 14, 12), getitem___79, int_62)

# Assigning a type to the variable 'nrofvars' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'nrofvars', subscript_call_result_80)

# Assigning a Call to a Name (line 15):

# Call to range(...): (line 15)
# Processing the call arguments (line 15)
# Getting the type of 'nrofvars' (line 15)
nrofvars_82 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 13), 'nrofvars', False)
int_83 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 24), 'int')
# Applying the binary operator '+' (line 15)
result_add_84 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 13), '+', nrofvars_82, int_83)

# Processing the call keyword arguments (line 15)
kwargs_85 = {}
# Getting the type of 'range' (line 15)
range_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 7), 'range', False)
# Calling range(args, kwargs) (line 15)
range_call_result_86 = invoke(stypy.reporting.localization.Localization(__file__, 15, 7), range_81, *[result_add_84], **kwargs_85)

# Assigning a type to the variable 'vars' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'vars', range_call_result_86)

# Assigning a ListComp to a Name (line 16):
# Calculating list comprehension
# Calculating comprehension expression
# Getting the type of 'vars' (line 16)
vars_88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 26), 'vars')

# Call to range(...): (line 16)
# Processing the call arguments (line 16)

# Getting the type of 'nrofvars' (line 16)
nrofvars_90 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 40), 'nrofvars', False)
# Applying the 'usub' unary operator (line 16)
result___neg___91 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 39), 'usub', nrofvars_90)

int_92 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 50), 'int')
# Processing the call keyword arguments (line 16)
kwargs_93 = {}
# Getting the type of 'range' (line 16)
range_89 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 33), 'range', False)
# Calling range(args, kwargs) (line 16)
range_call_result_94 = invoke(stypy.reporting.localization.Localization(__file__, 16, 33), range_89, *[result___neg___91, int_92], **kwargs_93)

# Applying the binary operator '+' (line 16)
result_add_95 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 26), '+', vars_88, range_call_result_94)

comprehension_96 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 14), result_add_95)
# Assigning a type to the variable 'l' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 14), 'l', comprehension_96)

# Obtaining an instance of the builtin type 'list' (line 16)
list_87 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 16)

list_97 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 14), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 14), list_97, list_87)
# Assigning a type to the variable 'occurrence' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'occurrence', list_97)

# Getting the type of 'clauses' (line 17)
clauses_98 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 14), 'clauses')
# Assigning a type to the variable 'clauses_98' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'clauses_98', clauses_98)
# Testing if the for loop is going to be iterated (line 17)
# Testing the type of a for loop iterable (line 17)
is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 17, 0), clauses_98)

if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 17, 0), clauses_98):
    # Getting the type of the for loop variable (line 17)
    for_loop_var_99 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 17, 0), clauses_98)
    # Assigning a type to the variable 'clause' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'clause', for_loop_var_99)
    # SSA begins for a for statement (line 17)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'clause' (line 18)
    clause_100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 15), 'clause')
    # Assigning a type to the variable 'clause_100' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'clause_100', clause_100)
    # Testing if the for loop is going to be iterated (line 18)
    # Testing the type of a for loop iterable (line 18)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 18, 4), clause_100)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 18, 4), clause_100):
        # Getting the type of the for loop variable (line 18)
        for_loop_var_101 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 18, 4), clause_100)
        # Assigning a type to the variable 'lit' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'lit', for_loop_var_101)
        # SSA begins for a for statement (line 18)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 18)
        # Processing the call arguments (line 18)
        # Getting the type of 'clause' (line 18)
        clause_107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 46), 'clause', False)
        # Processing the call keyword arguments (line 18)
        kwargs_108 = {}
        
        # Obtaining the type of the subscript
        # Getting the type of 'lit' (line 18)
        lit_102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 34), 'lit', False)
        # Getting the type of 'occurrence' (line 18)
        occurrence_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 23), 'occurrence', False)
        # Obtaining the member '__getitem__' of a type (line 18)
        getitem___104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 23), occurrence_103, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 18)
        subscript_call_result_105 = invoke(stypy.reporting.localization.Localization(__file__, 18, 23), getitem___104, lit_102)
        
        # Obtaining the member 'append' of a type (line 18)
        append_106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 23), subscript_call_result_105, 'append')
        # Calling append(args, kwargs) (line 18)
        append_call_result_109 = invoke(stypy.reporting.localization.Localization(__file__, 18, 23), append_106, *[clause_107], **kwargs_108)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()



# Assigning a ListComp to a Name (line 19):
# Calculating list comprehension
# Calculating comprehension expression
# Getting the type of 'vars' (line 19)
vars_111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 24), 'vars')
comprehension_112 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 10), vars_111)
# Assigning a type to the variable 'var' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 10), 'var', comprehension_112)
int_110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 10), 'int')
list_113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 10), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 10), list_113, int_110)
# Assigning a type to the variable 'fixedt' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'fixedt', list_113)

# Assigning a Num to a Name (line 21):
int_114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 12), 'int')
# Assigning a type to the variable 'nodecount' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'nodecount', int_114)

@norecursion
def solve_rec(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'solve_rec'
    module_type_store = module_type_store.open_function_context('solve_rec', 24, 0, False)
    
    # Passed parameters checking function
    solve_rec.stypy_localization = localization
    solve_rec.stypy_type_of_self = None
    solve_rec.stypy_type_store = module_type_store
    solve_rec.stypy_function_name = 'solve_rec'
    solve_rec.stypy_param_names_list = []
    solve_rec.stypy_varargs_param_name = None
    solve_rec.stypy_kwargs_param_name = None
    solve_rec.stypy_call_defaults = defaults
    solve_rec.stypy_call_varargs = varargs
    solve_rec.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'solve_rec', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'solve_rec', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'solve_rec(...)' code ##################

    # Marking variables as global (line 25)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 25, 4), 'nodecount')
    
    # Getting the type of 'nodecount' (line 26)
    nodecount_115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'nodecount')
    int_116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 17), 'int')
    # Applying the binary operator '+=' (line 26)
    result_iadd_117 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 4), '+=', nodecount_115, int_116)
    # Assigning a type to the variable 'nodecount' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'nodecount', result_iadd_117)
    
    
    
    int_118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 11), 'int')
    
    # Obtaining the type of the subscript
    int_119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 24), 'int')
    slice_120 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 28, 17), int_119, None, None)
    # Getting the type of 'fixedt' (line 28)
    fixedt_121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 17), 'fixedt')
    # Obtaining the member '__getitem__' of a type (line 28)
    getitem___122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 17), fixedt_121, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 28)
    subscript_call_result_123 = invoke(stypy.reporting.localization.Localization(__file__, 28, 17), getitem___122, slice_120)
    
    # Applying the binary operator 'in' (line 28)
    result_contains_124 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 11), 'in', int_118, subscript_call_result_123)
    
    # Applying the 'not' unary operator (line 28)
    result_not__125 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 7), 'not', result_contains_124)
    
    # Testing if the type of an if condition is none (line 28)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 28, 4), result_not__125):
        pass
    else:
        
        # Testing the type of an if condition (line 28)
        if_condition_126 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 28, 4), result_not__125)
        # Assigning a type to the variable 'if_condition_126' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'if_condition_126', if_condition_126)
        # SSA begins for if statement (line 28)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        int_127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 15), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'stypy_return_type', int_127)
        # SSA join for if statement (line 28)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a List to a Name (line 32):
    
    # Obtaining an instance of the builtin type 'list' (line 32)
    list_128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 32)
    
    # Assigning a type to the variable 'la_mods' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'la_mods', list_128)
    
    # Assigning a Call to a Name (line 33):
    
    # Call to lookahead(...): (line 33)
    # Processing the call arguments (line 33)
    # Getting the type of 'la_mods' (line 33)
    la_mods_130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 20), 'la_mods', False)
    # Processing the call keyword arguments (line 33)
    kwargs_131 = {}
    # Getting the type of 'lookahead' (line 33)
    lookahead_129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 10), 'lookahead', False)
    # Calling lookahead(args, kwargs) (line 33)
    lookahead_call_result_132 = invoke(stypy.reporting.localization.Localization(__file__, 33, 10), lookahead_129, *[la_mods_130], **kwargs_131)
    
    # Assigning a type to the variable 'var' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'var', lookahead_call_result_132)
    
    # Getting the type of 'var' (line 34)
    var_133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 11), 'var')
    # Applying the 'not' unary operator (line 34)
    result_not__134 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 7), 'not', var_133)
    
    # Testing if the type of an if condition is none (line 34)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 34, 4), result_not__134):
        pass
    else:
        
        # Testing the type of an if condition (line 34)
        if_condition_135 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 34, 4), result_not__134)
        # Assigning a type to the variable 'if_condition_135' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'if_condition_135', if_condition_135)
        # SSA begins for if statement (line 34)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to backtrack(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'la_mods' (line 35)
        la_mods_137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 25), 'la_mods', False)
        # Processing the call keyword arguments (line 35)
        kwargs_138 = {}
        # Getting the type of 'backtrack' (line 35)
        backtrack_136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 15), 'backtrack', False)
        # Calling backtrack(args, kwargs) (line 35)
        backtrack_call_result_139 = invoke(stypy.reporting.localization.Localization(__file__, 35, 15), backtrack_136, *[la_mods_137], **kwargs_138)
        
        # Assigning a type to the variable 'stypy_return_type' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'stypy_return_type', backtrack_call_result_139)
        # SSA join for if statement (line 34)
        module_type_store = module_type_store.join_ssa_context()
        

    
    
    # Obtaining an instance of the builtin type 'list' (line 36)
    list_140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 36)
    # Adding element type (line 36)
    # Getting the type of 'var' (line 36)
    var_141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 19), 'var')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 18), list_140, var_141)
    # Adding element type (line 36)
    
    # Getting the type of 'var' (line 36)
    var_142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 25), 'var')
    # Applying the 'usub' unary operator (line 36)
    result___neg___143 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 24), 'usub', var_142)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 18), list_140, result___neg___143)
    
    # Assigning a type to the variable 'list_140' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'list_140', list_140)
    # Testing if the for loop is going to be iterated (line 36)
    # Testing the type of a for loop iterable (line 36)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 36, 4), list_140)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 36, 4), list_140):
        # Getting the type of the for loop variable (line 36)
        for_loop_var_144 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 36, 4), list_140)
        # Assigning a type to the variable 'choice' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'choice', for_loop_var_144)
        # SSA begins for a for statement (line 36)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a List to a Name (line 37):
        
        # Obtaining an instance of the builtin type 'list' (line 37)
        list_145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 37)
        
        # Assigning a type to the variable 'prop_mods' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'prop_mods', list_145)
        
        # Evaluating a boolean operation
        
        # Call to propagate(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'choice' (line 38)
        choice_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 21), 'choice', False)
        # Getting the type of 'prop_mods' (line 38)
        prop_mods_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 29), 'prop_mods', False)
        # Processing the call keyword arguments (line 38)
        kwargs_149 = {}
        # Getting the type of 'propagate' (line 38)
        propagate_146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 11), 'propagate', False)
        # Calling propagate(args, kwargs) (line 38)
        propagate_call_result_150 = invoke(stypy.reporting.localization.Localization(__file__, 38, 11), propagate_146, *[choice_147, prop_mods_148], **kwargs_149)
        
        
        # Call to solve_rec(...): (line 38)
        # Processing the call keyword arguments (line 38)
        kwargs_152 = {}
        # Getting the type of 'solve_rec' (line 38)
        solve_rec_151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 44), 'solve_rec', False)
        # Calling solve_rec(args, kwargs) (line 38)
        solve_rec_call_result_153 = invoke(stypy.reporting.localization.Localization(__file__, 38, 44), solve_rec_151, *[], **kwargs_152)
        
        # Applying the binary operator 'and' (line 38)
        result_and_keyword_154 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 11), 'and', propagate_call_result_150, solve_rec_call_result_153)
        
        # Testing if the type of an if condition is none (line 38)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 38, 8), result_and_keyword_154):
            pass
        else:
            
            # Testing the type of an if condition (line 38)
            if_condition_155 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 38, 8), result_and_keyword_154)
            # Assigning a type to the variable 'if_condition_155' (line 38)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'if_condition_155', if_condition_155)
            # SSA begins for if statement (line 38)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            int_156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 39)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'stypy_return_type', int_156)
            # SSA join for if statement (line 38)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to backtrack(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'prop_mods' (line 40)
        prop_mods_158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 18), 'prop_mods', False)
        # Processing the call keyword arguments (line 40)
        kwargs_159 = {}
        # Getting the type of 'backtrack' (line 40)
        backtrack_157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'backtrack', False)
        # Calling backtrack(args, kwargs) (line 40)
        backtrack_call_result_160 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), backtrack_157, *[prop_mods_158], **kwargs_159)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Call to backtrack(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 'la_mods' (line 41)
    la_mods_162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 21), 'la_mods', False)
    # Processing the call keyword arguments (line 41)
    kwargs_163 = {}
    # Getting the type of 'backtrack' (line 41)
    backtrack_161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 11), 'backtrack', False)
    # Calling backtrack(args, kwargs) (line 41)
    backtrack_call_result_164 = invoke(stypy.reporting.localization.Localization(__file__, 41, 11), backtrack_161, *[la_mods_162], **kwargs_163)
    
    # Assigning a type to the variable 'stypy_return_type' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type', backtrack_call_result_164)
    
    # ################# End of 'solve_rec(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'solve_rec' in the type store
    # Getting the type of 'stypy_return_type' (line 24)
    stypy_return_type_165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_165)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'solve_rec'
    return stypy_return_type_165

# Assigning a type to the variable 'solve_rec' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'solve_rec', solve_rec)

@norecursion
def propagate(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'propagate'
    module_type_store = module_type_store.open_function_context('propagate', 44, 0, False)
    
    # Passed parameters checking function
    propagate.stypy_localization = localization
    propagate.stypy_type_of_self = None
    propagate.stypy_type_store = module_type_store
    propagate.stypy_function_name = 'propagate'
    propagate.stypy_param_names_list = ['lit', 'mods']
    propagate.stypy_varargs_param_name = None
    propagate.stypy_kwargs_param_name = None
    propagate.stypy_call_defaults = defaults
    propagate.stypy_call_varargs = varargs
    propagate.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'propagate', ['lit', 'mods'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'propagate', localization, ['lit', 'mods'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'propagate(...)' code ##################

    # Marking variables as global (line 45)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 45, 4), 'bincount')
    
    # Assigning a Call to a Name (line 47):
    
    # Call to len(...): (line 47)
    # Processing the call arguments (line 47)
    # Getting the type of 'mods' (line 47)
    mods_167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 18), 'mods', False)
    # Processing the call keyword arguments (line 47)
    kwargs_168 = {}
    # Getting the type of 'len' (line 47)
    len_166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 14), 'len', False)
    # Calling len(args, kwargs) (line 47)
    len_call_result_169 = invoke(stypy.reporting.localization.Localization(__file__, 47, 14), len_166, *[mods_167], **kwargs_168)
    
    # Assigning a type to the variable 'current' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'current', len_call_result_169)
    
    # Call to append(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'lit' (line 48)
    lit_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 16), 'lit', False)
    # Processing the call keyword arguments (line 48)
    kwargs_173 = {}
    # Getting the type of 'mods' (line 48)
    mods_170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'mods', False)
    # Obtaining the member 'append' of a type (line 48)
    append_171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 4), mods_170, 'append')
    # Calling append(args, kwargs) (line 48)
    append_call_result_174 = invoke(stypy.reporting.localization.Localization(__file__, 48, 4), append_171, *[lit_172], **kwargs_173)
    
    
    int_175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 10), 'int')
    # Assigning a type to the variable 'int_175' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'int_175', int_175)
    # Testing if the while is going to be iterated (line 50)
    # Testing the type of an if condition (line 50)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 4), int_175)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 50, 4), int_175):
        # SSA begins for while statement (line 50)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        
        # Obtaining the type of the subscript
        
        # Call to abs(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'lit' (line 51)
        lit_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 22), 'lit', False)
        # Processing the call keyword arguments (line 51)
        kwargs_178 = {}
        # Getting the type of 'abs' (line 51)
        abs_176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 18), 'abs', False)
        # Calling abs(args, kwargs) (line 51)
        abs_call_result_179 = invoke(stypy.reporting.localization.Localization(__file__, 51, 18), abs_176, *[lit_177], **kwargs_178)
        
        # Getting the type of 'fixedt' (line 51)
        fixedt_180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 11), 'fixedt')
        # Obtaining the member '__getitem__' of a type (line 51)
        getitem___181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 11), fixedt_180, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
        subscript_call_result_182 = invoke(stypy.reporting.localization.Localization(__file__, 51, 11), getitem___181, abs_call_result_179)
        
        int_183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 31), 'int')
        # Applying the binary operator '==' (line 51)
        result_eq_184 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 11), '==', subscript_call_result_182, int_183)
        
        # Testing if the type of an if condition is none (line 51)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 51, 8), result_eq_184):
            
            
            # Obtaining the type of the subscript
            
            # Call to abs(...): (line 62)
            # Processing the call arguments (line 62)
            # Getting the type of 'lit' (line 62)
            lit_232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 24), 'lit', False)
            # Processing the call keyword arguments (line 62)
            kwargs_233 = {}
            # Getting the type of 'abs' (line 62)
            abs_231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 20), 'abs', False)
            # Calling abs(args, kwargs) (line 62)
            abs_call_result_234 = invoke(stypy.reporting.localization.Localization(__file__, 62, 20), abs_231, *[lit_232], **kwargs_233)
            
            # Getting the type of 'fixedt' (line 62)
            fixedt_235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 13), 'fixedt')
            # Obtaining the member '__getitem__' of a type (line 62)
            getitem___236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 13), fixedt_235, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 62)
            subscript_call_result_237 = invoke(stypy.reporting.localization.Localization(__file__, 62, 13), getitem___236, abs_call_result_234)
            
            
            # Call to int(...): (line 62)
            # Processing the call arguments (line 62)
            
            # Getting the type of 'lit' (line 62)
            lit_239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 37), 'lit', False)
            int_240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 43), 'int')
            # Applying the binary operator '>' (line 62)
            result_gt_241 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 37), '>', lit_239, int_240)
            
            # Processing the call keyword arguments (line 62)
            kwargs_242 = {}
            # Getting the type of 'int' (line 62)
            int_238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 33), 'int', False)
            # Calling int(args, kwargs) (line 62)
            int_call_result_243 = invoke(stypy.reporting.localization.Localization(__file__, 62, 33), int_238, *[result_gt_241], **kwargs_242)
            
            # Applying the binary operator '!=' (line 62)
            result_ne_244 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 13), '!=', subscript_call_result_237, int_call_result_243)
            
            # Testing if the type of an if condition is none (line 62)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 62, 13), result_ne_244):
                pass
            else:
                
                # Testing the type of an if condition (line 62)
                if_condition_245 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 62, 13), result_ne_244)
                # Assigning a type to the variable 'if_condition_245' (line 62)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 13), 'if_condition_245', if_condition_245)
                # SSA begins for if statement (line 62)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                int_246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 19), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 63)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'stypy_return_type', int_246)
                # SSA join for if statement (line 62)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 51)
            if_condition_185 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 51, 8), result_eq_184)
            # Assigning a type to the variable 'if_condition_185' (line 51)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'if_condition_185', if_condition_185)
            # SSA begins for if statement (line 51)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Subscript (line 52):
            
            # Call to int(...): (line 52)
            # Processing the call arguments (line 52)
            
            # Getting the type of 'lit' (line 52)
            lit_187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 35), 'lit', False)
            int_188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 41), 'int')
            # Applying the binary operator '>' (line 52)
            result_gt_189 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 35), '>', lit_187, int_188)
            
            # Processing the call keyword arguments (line 52)
            kwargs_190 = {}
            # Getting the type of 'int' (line 52)
            int_186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 31), 'int', False)
            # Calling int(args, kwargs) (line 52)
            int_call_result_191 = invoke(stypy.reporting.localization.Localization(__file__, 52, 31), int_186, *[result_gt_189], **kwargs_190)
            
            # Getting the type of 'fixedt' (line 52)
            fixedt_192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'fixedt')
            
            # Call to abs(...): (line 52)
            # Processing the call arguments (line 52)
            # Getting the type of 'lit' (line 52)
            lit_194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 23), 'lit', False)
            # Processing the call keyword arguments (line 52)
            kwargs_195 = {}
            # Getting the type of 'abs' (line 52)
            abs_193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 19), 'abs', False)
            # Calling abs(args, kwargs) (line 52)
            abs_call_result_196 = invoke(stypy.reporting.localization.Localization(__file__, 52, 19), abs_193, *[lit_194], **kwargs_195)
            
            # Storing an element on a container (line 52)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 12), fixedt_192, (abs_call_result_196, int_call_result_191))
            
            
            # Obtaining the type of the subscript
            
            # Getting the type of 'lit' (line 53)
            lit_197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 38), 'lit')
            # Applying the 'usub' unary operator (line 53)
            result___neg___198 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 37), 'usub', lit_197)
            
            # Getting the type of 'occurrence' (line 53)
            occurrence_199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 26), 'occurrence')
            # Obtaining the member '__getitem__' of a type (line 53)
            getitem___200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 26), occurrence_199, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 53)
            subscript_call_result_201 = invoke(stypy.reporting.localization.Localization(__file__, 53, 26), getitem___200, result___neg___198)
            
            # Assigning a type to the variable 'subscript_call_result_201' (line 53)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'subscript_call_result_201', subscript_call_result_201)
            # Testing if the for loop is going to be iterated (line 53)
            # Testing the type of a for loop iterable (line 53)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 53, 12), subscript_call_result_201)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 53, 12), subscript_call_result_201):
                # Getting the type of the for loop variable (line 53)
                for_loop_var_202 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 53, 12), subscript_call_result_201)
                # Assigning a type to the variable 'clause' (line 53)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'clause', for_loop_var_202)
                # SSA begins for a for statement (line 53)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Call to a Name (line 54):
                
                # Call to length(...): (line 54)
                # Processing the call arguments (line 54)
                # Getting the type of 'clause' (line 54)
                clause_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 32), 'clause', False)
                # Processing the call keyword arguments (line 54)
                kwargs_205 = {}
                # Getting the type of 'length' (line 54)
                length_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 25), 'length', False)
                # Calling length(args, kwargs) (line 54)
                length_call_result_206 = invoke(stypy.reporting.localization.Localization(__file__, 54, 25), length_203, *[clause_204], **kwargs_205)
                
                # Assigning a type to the variable 'cl_len' (line 54)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'cl_len', length_call_result_206)
                
                # Getting the type of 'cl_len' (line 55)
                cl_len_207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 19), 'cl_len')
                int_208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 29), 'int')
                # Applying the binary operator '==' (line 55)
                result_eq_209 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 19), '==', cl_len_207, int_208)
                
                # Testing if the type of an if condition is none (line 55)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 55, 16), result_eq_209):
                    
                    # Getting the type of 'cl_len' (line 57)
                    cl_len_212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 21), 'cl_len')
                    int_213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 31), 'int')
                    # Applying the binary operator '==' (line 57)
                    result_eq_214 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 21), '==', cl_len_212, int_213)
                    
                    # Testing if the type of an if condition is none (line 57)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 57, 21), result_eq_214):
                        
                        # Getting the type of 'cl_len' (line 59)
                        cl_len_224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 21), 'cl_len')
                        int_225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 31), 'int')
                        # Applying the binary operator '==' (line 59)
                        result_eq_226 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 21), '==', cl_len_224, int_225)
                        
                        # Testing if the type of an if condition is none (line 59)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 59, 21), result_eq_226):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 59)
                            if_condition_227 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 59, 21), result_eq_226)
                            # Assigning a type to the variable 'if_condition_227' (line 59)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 21), 'if_condition_227', if_condition_227)
                            # SSA begins for if statement (line 59)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Getting the type of 'bincount' (line 60)
                            bincount_228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 20), 'bincount')
                            int_229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 32), 'int')
                            # Applying the binary operator '+=' (line 60)
                            result_iadd_230 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 20), '+=', bincount_228, int_229)
                            # Assigning a type to the variable 'bincount' (line 60)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 20), 'bincount', result_iadd_230)
                            
                            # SSA join for if statement (line 59)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 57)
                        if_condition_215 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 57, 21), result_eq_214)
                        # Assigning a type to the variable 'if_condition_215' (line 57)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 21), 'if_condition_215', if_condition_215)
                        # SSA begins for if statement (line 57)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to append(...): (line 58)
                        # Processing the call arguments (line 58)
                        
                        # Call to unfixed(...): (line 58)
                        # Processing the call arguments (line 58)
                        # Getting the type of 'clause' (line 58)
                        clause_219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 40), 'clause', False)
                        # Processing the call keyword arguments (line 58)
                        kwargs_220 = {}
                        # Getting the type of 'unfixed' (line 58)
                        unfixed_218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 32), 'unfixed', False)
                        # Calling unfixed(args, kwargs) (line 58)
                        unfixed_call_result_221 = invoke(stypy.reporting.localization.Localization(__file__, 58, 32), unfixed_218, *[clause_219], **kwargs_220)
                        
                        # Processing the call keyword arguments (line 58)
                        kwargs_222 = {}
                        # Getting the type of 'mods' (line 58)
                        mods_216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 20), 'mods', False)
                        # Obtaining the member 'append' of a type (line 58)
                        append_217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 20), mods_216, 'append')
                        # Calling append(args, kwargs) (line 58)
                        append_call_result_223 = invoke(stypy.reporting.localization.Localization(__file__, 58, 20), append_217, *[unfixed_call_result_221], **kwargs_222)
                        
                        # SSA branch for the else part of an if statement (line 57)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'cl_len' (line 59)
                        cl_len_224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 21), 'cl_len')
                        int_225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 31), 'int')
                        # Applying the binary operator '==' (line 59)
                        result_eq_226 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 21), '==', cl_len_224, int_225)
                        
                        # Testing if the type of an if condition is none (line 59)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 59, 21), result_eq_226):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 59)
                            if_condition_227 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 59, 21), result_eq_226)
                            # Assigning a type to the variable 'if_condition_227' (line 59)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 21), 'if_condition_227', if_condition_227)
                            # SSA begins for if statement (line 59)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Getting the type of 'bincount' (line 60)
                            bincount_228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 20), 'bincount')
                            int_229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 32), 'int')
                            # Applying the binary operator '+=' (line 60)
                            result_iadd_230 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 20), '+=', bincount_228, int_229)
                            # Assigning a type to the variable 'bincount' (line 60)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 20), 'bincount', result_iadd_230)
                            
                            # SSA join for if statement (line 59)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 57)
                        module_type_store = module_type_store.join_ssa_context()
                        

                else:
                    
                    # Testing the type of an if condition (line 55)
                    if_condition_210 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 55, 16), result_eq_209)
                    # Assigning a type to the variable 'if_condition_210' (line 55)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), 'if_condition_210', if_condition_210)
                    # SSA begins for if statement (line 55)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    int_211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 27), 'int')
                    # Assigning a type to the variable 'stypy_return_type' (line 56)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 20), 'stypy_return_type', int_211)
                    # SSA branch for the else part of an if statement (line 55)
                    module_type_store.open_ssa_branch('else')
                    
                    # Getting the type of 'cl_len' (line 57)
                    cl_len_212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 21), 'cl_len')
                    int_213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 31), 'int')
                    # Applying the binary operator '==' (line 57)
                    result_eq_214 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 21), '==', cl_len_212, int_213)
                    
                    # Testing if the type of an if condition is none (line 57)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 57, 21), result_eq_214):
                        
                        # Getting the type of 'cl_len' (line 59)
                        cl_len_224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 21), 'cl_len')
                        int_225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 31), 'int')
                        # Applying the binary operator '==' (line 59)
                        result_eq_226 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 21), '==', cl_len_224, int_225)
                        
                        # Testing if the type of an if condition is none (line 59)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 59, 21), result_eq_226):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 59)
                            if_condition_227 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 59, 21), result_eq_226)
                            # Assigning a type to the variable 'if_condition_227' (line 59)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 21), 'if_condition_227', if_condition_227)
                            # SSA begins for if statement (line 59)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Getting the type of 'bincount' (line 60)
                            bincount_228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 20), 'bincount')
                            int_229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 32), 'int')
                            # Applying the binary operator '+=' (line 60)
                            result_iadd_230 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 20), '+=', bincount_228, int_229)
                            # Assigning a type to the variable 'bincount' (line 60)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 20), 'bincount', result_iadd_230)
                            
                            # SSA join for if statement (line 59)
                            module_type_store = module_type_store.join_ssa_context()
                            

                    else:
                        
                        # Testing the type of an if condition (line 57)
                        if_condition_215 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 57, 21), result_eq_214)
                        # Assigning a type to the variable 'if_condition_215' (line 57)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 21), 'if_condition_215', if_condition_215)
                        # SSA begins for if statement (line 57)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to append(...): (line 58)
                        # Processing the call arguments (line 58)
                        
                        # Call to unfixed(...): (line 58)
                        # Processing the call arguments (line 58)
                        # Getting the type of 'clause' (line 58)
                        clause_219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 40), 'clause', False)
                        # Processing the call keyword arguments (line 58)
                        kwargs_220 = {}
                        # Getting the type of 'unfixed' (line 58)
                        unfixed_218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 32), 'unfixed', False)
                        # Calling unfixed(args, kwargs) (line 58)
                        unfixed_call_result_221 = invoke(stypy.reporting.localization.Localization(__file__, 58, 32), unfixed_218, *[clause_219], **kwargs_220)
                        
                        # Processing the call keyword arguments (line 58)
                        kwargs_222 = {}
                        # Getting the type of 'mods' (line 58)
                        mods_216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 20), 'mods', False)
                        # Obtaining the member 'append' of a type (line 58)
                        append_217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 20), mods_216, 'append')
                        # Calling append(args, kwargs) (line 58)
                        append_call_result_223 = invoke(stypy.reporting.localization.Localization(__file__, 58, 20), append_217, *[unfixed_call_result_221], **kwargs_222)
                        
                        # SSA branch for the else part of an if statement (line 57)
                        module_type_store.open_ssa_branch('else')
                        
                        # Getting the type of 'cl_len' (line 59)
                        cl_len_224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 21), 'cl_len')
                        int_225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 31), 'int')
                        # Applying the binary operator '==' (line 59)
                        result_eq_226 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 21), '==', cl_len_224, int_225)
                        
                        # Testing if the type of an if condition is none (line 59)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 59, 21), result_eq_226):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 59)
                            if_condition_227 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 59, 21), result_eq_226)
                            # Assigning a type to the variable 'if_condition_227' (line 59)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 21), 'if_condition_227', if_condition_227)
                            # SSA begins for if statement (line 59)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Getting the type of 'bincount' (line 60)
                            bincount_228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 20), 'bincount')
                            int_229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 32), 'int')
                            # Applying the binary operator '+=' (line 60)
                            result_iadd_230 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 20), '+=', bincount_228, int_229)
                            # Assigning a type to the variable 'bincount' (line 60)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 20), 'bincount', result_iadd_230)
                            
                            # SSA join for if statement (line 59)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 57)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 55)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA branch for the else part of an if statement (line 51)
            module_type_store.open_ssa_branch('else')
            
            
            # Obtaining the type of the subscript
            
            # Call to abs(...): (line 62)
            # Processing the call arguments (line 62)
            # Getting the type of 'lit' (line 62)
            lit_232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 24), 'lit', False)
            # Processing the call keyword arguments (line 62)
            kwargs_233 = {}
            # Getting the type of 'abs' (line 62)
            abs_231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 20), 'abs', False)
            # Calling abs(args, kwargs) (line 62)
            abs_call_result_234 = invoke(stypy.reporting.localization.Localization(__file__, 62, 20), abs_231, *[lit_232], **kwargs_233)
            
            # Getting the type of 'fixedt' (line 62)
            fixedt_235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 13), 'fixedt')
            # Obtaining the member '__getitem__' of a type (line 62)
            getitem___236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 13), fixedt_235, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 62)
            subscript_call_result_237 = invoke(stypy.reporting.localization.Localization(__file__, 62, 13), getitem___236, abs_call_result_234)
            
            
            # Call to int(...): (line 62)
            # Processing the call arguments (line 62)
            
            # Getting the type of 'lit' (line 62)
            lit_239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 37), 'lit', False)
            int_240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 43), 'int')
            # Applying the binary operator '>' (line 62)
            result_gt_241 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 37), '>', lit_239, int_240)
            
            # Processing the call keyword arguments (line 62)
            kwargs_242 = {}
            # Getting the type of 'int' (line 62)
            int_238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 33), 'int', False)
            # Calling int(args, kwargs) (line 62)
            int_call_result_243 = invoke(stypy.reporting.localization.Localization(__file__, 62, 33), int_238, *[result_gt_241], **kwargs_242)
            
            # Applying the binary operator '!=' (line 62)
            result_ne_244 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 13), '!=', subscript_call_result_237, int_call_result_243)
            
            # Testing if the type of an if condition is none (line 62)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 62, 13), result_ne_244):
                pass
            else:
                
                # Testing the type of an if condition (line 62)
                if_condition_245 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 62, 13), result_ne_244)
                # Assigning a type to the variable 'if_condition_245' (line 62)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 13), 'if_condition_245', if_condition_245)
                # SSA begins for if statement (line 62)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                int_246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 19), 'int')
                # Assigning a type to the variable 'stypy_return_type' (line 63)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'stypy_return_type', int_246)
                # SSA join for if statement (line 62)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 51)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'current' (line 65)
        current_247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'current')
        int_248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 19), 'int')
        # Applying the binary operator '+=' (line 65)
        result_iadd_249 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 8), '+=', current_247, int_248)
        # Assigning a type to the variable 'current' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'current', result_iadd_249)
        
        
        # Getting the type of 'current' (line 66)
        current_250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 11), 'current')
        
        # Call to len(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'mods' (line 66)
        mods_252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 26), 'mods', False)
        # Processing the call keyword arguments (line 66)
        kwargs_253 = {}
        # Getting the type of 'len' (line 66)
        len_251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 22), 'len', False)
        # Calling len(args, kwargs) (line 66)
        len_call_result_254 = invoke(stypy.reporting.localization.Localization(__file__, 66, 22), len_251, *[mods_252], **kwargs_253)
        
        # Applying the binary operator '==' (line 66)
        result_eq_255 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 11), '==', current_250, len_call_result_254)
        
        # Testing if the type of an if condition is none (line 66)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 66, 8), result_eq_255):
            pass
        else:
            
            # Testing the type of an if condition (line 66)
            if_condition_256 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 8), result_eq_255)
            # Assigning a type to the variable 'if_condition_256' (line 66)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'if_condition_256', if_condition_256)
            # SSA begins for if statement (line 66)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 66)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Subscript to a Name (line 68):
        
        # Obtaining the type of the subscript
        # Getting the type of 'current' (line 68)
        current_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 19), 'current')
        # Getting the type of 'mods' (line 68)
        mods_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 14), 'mods')
        # Obtaining the member '__getitem__' of a type (line 68)
        getitem___259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 14), mods_258, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 68)
        subscript_call_result_260 = invoke(stypy.reporting.localization.Localization(__file__, 68, 14), getitem___259, current_257)
        
        # Assigning a type to the variable 'lit' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'lit', subscript_call_result_260)
        # SSA join for while statement (line 50)
        module_type_store = module_type_store.join_ssa_context()

    
    int_261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 11), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'stypy_return_type', int_261)
    
    # ################# End of 'propagate(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'propagate' in the type store
    # Getting the type of 'stypy_return_type' (line 44)
    stypy_return_type_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_262)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'propagate'
    return stypy_return_type_262

# Assigning a type to the variable 'propagate' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'propagate', propagate)

@norecursion
def lookahead(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'lookahead'
    module_type_store = module_type_store.open_function_context('lookahead', 73, 0, False)
    
    # Passed parameters checking function
    lookahead.stypy_localization = localization
    lookahead.stypy_type_of_self = None
    lookahead.stypy_type_store = module_type_store
    lookahead.stypy_function_name = 'lookahead'
    lookahead.stypy_param_names_list = ['mods']
    lookahead.stypy_varargs_param_name = None
    lookahead.stypy_kwargs_param_name = None
    lookahead.stypy_call_defaults = defaults
    lookahead.stypy_call_varargs = varargs
    lookahead.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lookahead', ['mods'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lookahead', localization, ['mods'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lookahead(...)' code ##################

    # Marking variables as global (line 74)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 74, 4), 'bincount')
    
    # Assigning a ListComp to a Name (line 76):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'vars' (line 76)
    vars_264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 25), 'vars')
    comprehension_265 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 11), vars_264)
    # Assigning a type to the variable 'var' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 11), 'var', comprehension_265)
    int_263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 11), 'int')
    list_266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 11), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 11), list_266, int_263)
    # Assigning a type to the variable 'dif' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'dif', list_266)
    
    
    # Call to unfixed_vars(...): (line 77)
    # Processing the call keyword arguments (line 77)
    kwargs_268 = {}
    # Getting the type of 'unfixed_vars' (line 77)
    unfixed_vars_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 15), 'unfixed_vars', False)
    # Calling unfixed_vars(args, kwargs) (line 77)
    unfixed_vars_call_result_269 = invoke(stypy.reporting.localization.Localization(__file__, 77, 15), unfixed_vars_267, *[], **kwargs_268)
    
    # Assigning a type to the variable 'unfixed_vars_call_result_269' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'unfixed_vars_call_result_269', unfixed_vars_call_result_269)
    # Testing if the for loop is going to be iterated (line 77)
    # Testing the type of a for loop iterable (line 77)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 77, 4), unfixed_vars_call_result_269)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 77, 4), unfixed_vars_call_result_269):
        # Getting the type of the for loop variable (line 77)
        for_loop_var_270 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 77, 4), unfixed_vars_call_result_269)
        # Assigning a type to the variable 'var' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'var', for_loop_var_270)
        # SSA begins for a for statement (line 77)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a List to a Name (line 78):
        
        # Obtaining an instance of the builtin type 'list' (line 78)
        list_271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 78)
        
        # Assigning a type to the variable 'score' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'score', list_271)
        
        
        # Obtaining an instance of the builtin type 'list' (line 79)
        list_272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 79)
        # Adding element type (line 79)
        # Getting the type of 'var' (line 79)
        var_273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 23), 'var')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 22), list_272, var_273)
        # Adding element type (line 79)
        
        # Getting the type of 'var' (line 79)
        var_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 29), 'var')
        # Applying the 'usub' unary operator (line 79)
        result___neg___275 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 28), 'usub', var_274)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 22), list_272, result___neg___275)
        
        # Assigning a type to the variable 'list_272' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'list_272', list_272)
        # Testing if the for loop is going to be iterated (line 79)
        # Testing the type of a for loop iterable (line 79)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 79, 8), list_272)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 79, 8), list_272):
            # Getting the type of the for loop variable (line 79)
            for_loop_var_276 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 79, 8), list_272)
            # Assigning a type to the variable 'choice' (line 79)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'choice', for_loop_var_276)
            # SSA begins for a for statement (line 79)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a List to a Name (line 80):
            
            # Obtaining an instance of the builtin type 'list' (line 80)
            list_277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 24), 'list')
            # Adding type elements to the builtin type 'list' instance (line 80)
            
            # Assigning a type to the variable 'prop_mods' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'prop_mods', list_277)
            
            # Assigning a Num to a Name (line 81):
            int_278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 23), 'int')
            # Assigning a type to the variable 'bincount' (line 81)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'bincount', int_278)
            
            # Assigning a Call to a Name (line 82):
            
            # Call to propagate(...): (line 82)
            # Processing the call arguments (line 82)
            # Getting the type of 'choice' (line 82)
            choice_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 29), 'choice', False)
            # Getting the type of 'prop_mods' (line 82)
            prop_mods_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 37), 'prop_mods', False)
            # Processing the call keyword arguments (line 82)
            kwargs_282 = {}
            # Getting the type of 'propagate' (line 82)
            propagate_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 19), 'propagate', False)
            # Calling propagate(args, kwargs) (line 82)
            propagate_call_result_283 = invoke(stypy.reporting.localization.Localization(__file__, 82, 19), propagate_279, *[choice_280, prop_mods_281], **kwargs_282)
            
            # Assigning a type to the variable 'prop' (line 82)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'prop', propagate_call_result_283)
            
            # Call to backtrack(...): (line 83)
            # Processing the call arguments (line 83)
            # Getting the type of 'prop_mods' (line 83)
            prop_mods_285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 22), 'prop_mods', False)
            # Processing the call keyword arguments (line 83)
            kwargs_286 = {}
            # Getting the type of 'backtrack' (line 83)
            backtrack_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'backtrack', False)
            # Calling backtrack(args, kwargs) (line 83)
            backtrack_call_result_287 = invoke(stypy.reporting.localization.Localization(__file__, 83, 12), backtrack_284, *[prop_mods_285], **kwargs_286)
            
            
            # Getting the type of 'prop' (line 84)
            prop_288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 19), 'prop')
            # Applying the 'not' unary operator (line 84)
            result_not__289 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 15), 'not', prop_288)
            
            # Testing if the type of an if condition is none (line 84)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 84, 12), result_not__289):
                pass
            else:
                
                # Testing the type of an if condition (line 84)
                if_condition_290 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 84, 12), result_not__289)
                # Assigning a type to the variable 'if_condition_290' (line 84)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'if_condition_290', if_condition_290)
                # SSA begins for if statement (line 84)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                
                # Call to propagate(...): (line 85)
                # Processing the call arguments (line 85)
                
                # Getting the type of 'choice' (line 85)
                choice_292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 34), 'choice', False)
                # Applying the 'usub' unary operator (line 85)
                result___neg___293 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 33), 'usub', choice_292)
                
                # Getting the type of 'mods' (line 85)
                mods_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 42), 'mods', False)
                # Processing the call keyword arguments (line 85)
                kwargs_295 = {}
                # Getting the type of 'propagate' (line 85)
                propagate_291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 23), 'propagate', False)
                # Calling propagate(args, kwargs) (line 85)
                propagate_call_result_296 = invoke(stypy.reporting.localization.Localization(__file__, 85, 23), propagate_291, *[result___neg___293, mods_294], **kwargs_295)
                
                # Applying the 'not' unary operator (line 85)
                result_not__297 = python_operator(stypy.reporting.localization.Localization(__file__, 85, 19), 'not', propagate_call_result_296)
                
                # Testing if the type of an if condition is none (line 85)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 85, 16), result_not__297):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 85)
                    if_condition_298 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 85, 16), result_not__297)
                    # Assigning a type to the variable 'if_condition_298' (line 85)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 16), 'if_condition_298', if_condition_298)
                    # SSA begins for if statement (line 85)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    int_299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 27), 'int')
                    # Assigning a type to the variable 'stypy_return_type' (line 86)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 20), 'stypy_return_type', int_299)
                    # SSA join for if statement (line 85)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 84)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to append(...): (line 88)
            # Processing the call arguments (line 88)
            # Getting the type of 'bincount' (line 88)
            bincount_302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 25), 'bincount', False)
            # Processing the call keyword arguments (line 88)
            kwargs_303 = {}
            # Getting the type of 'score' (line 88)
            score_300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'score', False)
            # Obtaining the member 'append' of a type (line 88)
            append_301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 12), score_300, 'append')
            # Calling append(args, kwargs) (line 88)
            append_call_result_304 = invoke(stypy.reporting.localization.Localization(__file__, 88, 12), append_301, *[bincount_302], **kwargs_303)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Call to a Subscript (line 89):
        
        # Call to reduce(...): (line 89)
        # Processing the call arguments (line 89)

        @norecursion
        def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_1'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 89, 26, True)
            # Passed parameters checking function
            _stypy_temp_lambda_1.stypy_localization = localization
            _stypy_temp_lambda_1.stypy_type_of_self = None
            _stypy_temp_lambda_1.stypy_type_store = module_type_store
            _stypy_temp_lambda_1.stypy_function_name = '_stypy_temp_lambda_1'
            _stypy_temp_lambda_1.stypy_param_names_list = ['x', 'y']
            _stypy_temp_lambda_1.stypy_varargs_param_name = None
            _stypy_temp_lambda_1.stypy_kwargs_param_name = None
            _stypy_temp_lambda_1.stypy_call_defaults = defaults
            _stypy_temp_lambda_1.stypy_call_varargs = varargs
            _stypy_temp_lambda_1.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_1', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_1', ['x', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            int_306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 39), 'int')
            # Getting the type of 'x' (line 89)
            x_307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 46), 'x', False)
            # Applying the binary operator '*' (line 89)
            result_mul_308 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 39), '*', int_306, x_307)
            
            # Getting the type of 'y' (line 89)
            y_309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 50), 'y', False)
            # Applying the binary operator '*' (line 89)
            result_mul_310 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 48), '*', result_mul_308, y_309)
            
            # Getting the type of 'x' (line 89)
            x_311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 54), 'x', False)
            # Applying the binary operator '+' (line 89)
            result_add_312 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 39), '+', result_mul_310, x_311)
            
            # Getting the type of 'y' (line 89)
            y_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 58), 'y', False)
            # Applying the binary operator '+' (line 89)
            result_add_314 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 56), '+', result_add_312, y_313)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 89)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 26), 'stypy_return_type', result_add_314)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_1' in the type store
            # Getting the type of 'stypy_return_type' (line 89)
            stypy_return_type_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 26), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_315)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_1'
            return stypy_return_type_315

        # Assigning a type to the variable '_stypy_temp_lambda_1' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 26), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
        # Getting the type of '_stypy_temp_lambda_1' (line 89)
        _stypy_temp_lambda_1_316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 26), '_stypy_temp_lambda_1')
        # Getting the type of 'score' (line 89)
        score_317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 61), 'score', False)
        int_318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 68), 'int')
        # Processing the call keyword arguments (line 89)
        kwargs_319 = {}
        # Getting the type of 'reduce' (line 89)
        reduce_305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 19), 'reduce', False)
        # Calling reduce(args, kwargs) (line 89)
        reduce_call_result_320 = invoke(stypy.reporting.localization.Localization(__file__, 89, 19), reduce_305, *[_stypy_temp_lambda_1_316, score_317, int_318], **kwargs_319)
        
        # Getting the type of 'dif' (line 89)
        dif_321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'dif')
        # Getting the type of 'var' (line 89)
        var_322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'var')
        # Storing an element on a container (line 89)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 8), dif_321, (var_322, reduce_call_result_320))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Call to index(...): (line 91)
    # Processing the call arguments (line 91)
    
    # Call to max(...): (line 91)
    # Processing the call arguments (line 91)
    # Getting the type of 'dif' (line 91)
    dif_326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 25), 'dif', False)
    # Processing the call keyword arguments (line 91)
    kwargs_327 = {}
    # Getting the type of 'max' (line 91)
    max_325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 21), 'max', False)
    # Calling max(args, kwargs) (line 91)
    max_call_result_328 = invoke(stypy.reporting.localization.Localization(__file__, 91, 21), max_325, *[dif_326], **kwargs_327)
    
    # Processing the call keyword arguments (line 91)
    kwargs_329 = {}
    # Getting the type of 'dif' (line 91)
    dif_323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 11), 'dif', False)
    # Obtaining the member 'index' of a type (line 91)
    index_324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 11), dif_323, 'index')
    # Calling index(args, kwargs) (line 91)
    index_call_result_330 = invoke(stypy.reporting.localization.Localization(__file__, 91, 11), index_324, *[max_call_result_328], **kwargs_329)
    
    # Assigning a type to the variable 'stypy_return_type' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'stypy_return_type', index_call_result_330)
    
    # ################# End of 'lookahead(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lookahead' in the type store
    # Getting the type of 'stypy_return_type' (line 73)
    stypy_return_type_331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_331)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lookahead'
    return stypy_return_type_331

# Assigning a type to the variable 'lookahead' (line 73)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), 'lookahead', lookahead)

@norecursion
def backtrack(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'backtrack'
    module_type_store = module_type_store.open_function_context('backtrack', 94, 0, False)
    
    # Passed parameters checking function
    backtrack.stypy_localization = localization
    backtrack.stypy_type_of_self = None
    backtrack.stypy_type_store = module_type_store
    backtrack.stypy_function_name = 'backtrack'
    backtrack.stypy_param_names_list = ['mods']
    backtrack.stypy_varargs_param_name = None
    backtrack.stypy_kwargs_param_name = None
    backtrack.stypy_call_defaults = defaults
    backtrack.stypy_call_varargs = varargs
    backtrack.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'backtrack', ['mods'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'backtrack', localization, ['mods'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'backtrack(...)' code ##################

    
    # Getting the type of 'mods' (line 95)
    mods_332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 15), 'mods')
    # Assigning a type to the variable 'mods_332' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'mods_332', mods_332)
    # Testing if the for loop is going to be iterated (line 95)
    # Testing the type of a for loop iterable (line 95)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 95, 4), mods_332)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 95, 4), mods_332):
        # Getting the type of the for loop variable (line 95)
        for_loop_var_333 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 95, 4), mods_332)
        # Assigning a type to the variable 'lit' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'lit', for_loop_var_333)
        # SSA begins for a for statement (line 95)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Num to a Subscript (line 96):
        int_334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 27), 'int')
        # Getting the type of 'fixedt' (line 96)
        fixedt_335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'fixedt')
        
        # Call to abs(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'lit' (line 96)
        lit_337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 19), 'lit', False)
        # Processing the call keyword arguments (line 96)
        kwargs_338 = {}
        # Getting the type of 'abs' (line 96)
        abs_336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 15), 'abs', False)
        # Calling abs(args, kwargs) (line 96)
        abs_call_result_339 = invoke(stypy.reporting.localization.Localization(__file__, 96, 15), abs_336, *[lit_337], **kwargs_338)
        
        # Storing an element on a container (line 96)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 96, 8), fixedt_335, (abs_call_result_339, int_334))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    int_340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 11), 'int')
    # Assigning a type to the variable 'stypy_return_type' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'stypy_return_type', int_340)
    
    # ################# End of 'backtrack(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'backtrack' in the type store
    # Getting the type of 'stypy_return_type' (line 94)
    stypy_return_type_341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_341)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'backtrack'
    return stypy_return_type_341

# Assigning a type to the variable 'backtrack' (line 94)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'backtrack', backtrack)

@norecursion
def length(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'length'
    module_type_store = module_type_store.open_function_context('length', 100, 0, False)
    
    # Passed parameters checking function
    length.stypy_localization = localization
    length.stypy_type_of_self = None
    length.stypy_type_store = module_type_store
    length.stypy_function_name = 'length'
    length.stypy_param_names_list = ['clause']
    length.stypy_varargs_param_name = None
    length.stypy_kwargs_param_name = None
    length.stypy_call_defaults = defaults
    length.stypy_call_varargs = varargs
    length.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'length', ['clause'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'length', localization, ['clause'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'length(...)' code ##################

    
    # Assigning a Num to a Name (line 101):
    int_342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 10), 'int')
    # Assigning a type to the variable 'len' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'len', int_342)
    
    # Getting the type of 'clause' (line 102)
    clause_343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 15), 'clause')
    # Assigning a type to the variable 'clause_343' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'clause_343', clause_343)
    # Testing if the for loop is going to be iterated (line 102)
    # Testing the type of a for loop iterable (line 102)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 102, 4), clause_343)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 102, 4), clause_343):
        # Getting the type of the for loop variable (line 102)
        for_loop_var_344 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 102, 4), clause_343)
        # Assigning a type to the variable 'lit' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'lit', for_loop_var_344)
        # SSA begins for a for statement (line 102)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 103):
        
        # Obtaining the type of the subscript
        
        # Call to abs(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'lit' (line 103)
        lit_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 27), 'lit', False)
        # Processing the call keyword arguments (line 103)
        kwargs_347 = {}
        # Getting the type of 'abs' (line 103)
        abs_345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 23), 'abs', False)
        # Calling abs(args, kwargs) (line 103)
        abs_call_result_348 = invoke(stypy.reporting.localization.Localization(__file__, 103, 23), abs_345, *[lit_346], **kwargs_347)
        
        # Getting the type of 'fixedt' (line 103)
        fixedt_349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 16), 'fixedt')
        # Obtaining the member '__getitem__' of a type (line 103)
        getitem___350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 16), fixedt_349, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 103)
        subscript_call_result_351 = invoke(stypy.reporting.localization.Localization(__file__, 103, 16), getitem___350, abs_call_result_348)
        
        # Assigning a type to the variable 'fixed' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'fixed', subscript_call_result_351)
        
        # Getting the type of 'fixed' (line 104)
        fixed_352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 11), 'fixed')
        
        # Call to int(...): (line 104)
        # Processing the call arguments (line 104)
        
        # Getting the type of 'lit' (line 104)
        lit_354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 24), 'lit', False)
        int_355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 30), 'int')
        # Applying the binary operator '>' (line 104)
        result_gt_356 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 24), '>', lit_354, int_355)
        
        # Processing the call keyword arguments (line 104)
        kwargs_357 = {}
        # Getting the type of 'int' (line 104)
        int_353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 20), 'int', False)
        # Calling int(args, kwargs) (line 104)
        int_call_result_358 = invoke(stypy.reporting.localization.Localization(__file__, 104, 20), int_353, *[result_gt_356], **kwargs_357)
        
        # Applying the binary operator '==' (line 104)
        result_eq_359 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 11), '==', fixed_352, int_call_result_358)
        
        # Testing if the type of an if condition is none (line 104)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 104, 8), result_eq_359):
            pass
        else:
            
            # Testing the type of an if condition (line 104)
            if_condition_360 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 104, 8), result_eq_359)
            # Assigning a type to the variable 'if_condition_360' (line 104)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'if_condition_360', if_condition_360)
            # SSA begins for if statement (line 104)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            int_361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 19), 'int')
            # Assigning a type to the variable 'stypy_return_type' (line 105)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'stypy_return_type', int_361)
            # SSA join for if statement (line 104)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'fixed' (line 106)
        fixed_362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 11), 'fixed')
        int_363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 20), 'int')
        # Applying the binary operator '==' (line 106)
        result_eq_364 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 11), '==', fixed_362, int_363)
        
        # Testing if the type of an if condition is none (line 106)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 106, 8), result_eq_364):
            pass
        else:
            
            # Testing the type of an if condition (line 106)
            if_condition_365 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 8), result_eq_364)
            # Assigning a type to the variable 'if_condition_365' (line 106)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'if_condition_365', if_condition_365)
            # SSA begins for if statement (line 106)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'len' (line 107)
            len_366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'len')
            int_367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 19), 'int')
            # Applying the binary operator '+=' (line 107)
            result_iadd_368 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 12), '+=', len_366, int_367)
            # Assigning a type to the variable 'len' (line 107)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'len', result_iadd_368)
            
            # SSA join for if statement (line 106)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'len' (line 108)
    len_369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 11), 'len')
    # Assigning a type to the variable 'stypy_return_type' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'stypy_return_type', len_369)
    
    # ################# End of 'length(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'length' in the type store
    # Getting the type of 'stypy_return_type' (line 100)
    stypy_return_type_370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_370)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'length'
    return stypy_return_type_370

# Assigning a type to the variable 'length' (line 100)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 0), 'length', length)

@norecursion
def unfixed(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'unfixed'
    module_type_store = module_type_store.open_function_context('unfixed', 111, 0, False)
    
    # Passed parameters checking function
    unfixed.stypy_localization = localization
    unfixed.stypy_type_of_self = None
    unfixed.stypy_type_store = module_type_store
    unfixed.stypy_function_name = 'unfixed'
    unfixed.stypy_param_names_list = ['clause']
    unfixed.stypy_varargs_param_name = None
    unfixed.stypy_kwargs_param_name = None
    unfixed.stypy_call_defaults = defaults
    unfixed.stypy_call_varargs = varargs
    unfixed.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'unfixed', ['clause'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'unfixed', localization, ['clause'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'unfixed(...)' code ##################

    
    # Getting the type of 'clause' (line 112)
    clause_371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 15), 'clause')
    # Assigning a type to the variable 'clause_371' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'clause_371', clause_371)
    # Testing if the for loop is going to be iterated (line 112)
    # Testing the type of a for loop iterable (line 112)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 112, 4), clause_371)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 112, 4), clause_371):
        # Getting the type of the for loop variable (line 112)
        for_loop_var_372 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 112, 4), clause_371)
        # Assigning a type to the variable 'lit' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'lit', for_loop_var_372)
        # SSA begins for a for statement (line 112)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 113):
        
        # Obtaining the type of the subscript
        
        # Call to abs(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'lit' (line 113)
        lit_374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 27), 'lit', False)
        # Processing the call keyword arguments (line 113)
        kwargs_375 = {}
        # Getting the type of 'abs' (line 113)
        abs_373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 23), 'abs', False)
        # Calling abs(args, kwargs) (line 113)
        abs_call_result_376 = invoke(stypy.reporting.localization.Localization(__file__, 113, 23), abs_373, *[lit_374], **kwargs_375)
        
        # Getting the type of 'fixedt' (line 113)
        fixedt_377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 16), 'fixedt')
        # Obtaining the member '__getitem__' of a type (line 113)
        getitem___378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 16), fixedt_377, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 113)
        subscript_call_result_379 = invoke(stypy.reporting.localization.Localization(__file__, 113, 16), getitem___378, abs_call_result_376)
        
        # Assigning a type to the variable 'fixed' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'fixed', subscript_call_result_379)
        
        # Getting the type of 'fixed' (line 114)
        fixed_380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 11), 'fixed')
        int_381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 20), 'int')
        # Applying the binary operator '==' (line 114)
        result_eq_382 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 11), '==', fixed_380, int_381)
        
        # Testing if the type of an if condition is none (line 114)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 114, 8), result_eq_382):
            pass
        else:
            
            # Testing the type of an if condition (line 114)
            if_condition_383 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 114, 8), result_eq_382)
            # Assigning a type to the variable 'if_condition_383' (line 114)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'if_condition_383', if_condition_383)
            # SSA begins for if statement (line 114)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'lit' (line 115)
            lit_384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 19), 'lit')
            # Assigning a type to the variable 'stypy_return_type' (line 115)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'stypy_return_type', lit_384)
            # SSA join for if statement (line 114)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'unfixed(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'unfixed' in the type store
    # Getting the type of 'stypy_return_type' (line 111)
    stypy_return_type_385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_385)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'unfixed'
    return stypy_return_type_385

# Assigning a type to the variable 'unfixed' (line 111)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 0), 'unfixed', unfixed)

@norecursion
def unfixed_vars(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'unfixed_vars'
    module_type_store = module_type_store.open_function_context('unfixed_vars', 118, 0, False)
    
    # Passed parameters checking function
    unfixed_vars.stypy_localization = localization
    unfixed_vars.stypy_type_of_self = None
    unfixed_vars.stypy_type_store = module_type_store
    unfixed_vars.stypy_function_name = 'unfixed_vars'
    unfixed_vars.stypy_param_names_list = []
    unfixed_vars.stypy_varargs_param_name = None
    unfixed_vars.stypy_kwargs_param_name = None
    unfixed_vars.stypy_call_defaults = defaults
    unfixed_vars.stypy_call_varargs = varargs
    unfixed_vars.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'unfixed_vars', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'unfixed_vars', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'unfixed_vars(...)' code ##################

    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    int_393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 32), 'int')
    slice_394 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 119, 27), int_393, None, None)
    # Getting the type of 'vars' (line 119)
    vars_395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 27), 'vars')
    # Obtaining the member '__getitem__' of a type (line 119)
    getitem___396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 27), vars_395, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 119)
    subscript_call_result_397 = invoke(stypy.reporting.localization.Localization(__file__, 119, 27), getitem___396, slice_394)
    
    comprehension_398 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 12), subscript_call_result_397)
    # Assigning a type to the variable 'var' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'var', comprehension_398)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'var' (line 119)
    var_387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 46), 'var')
    # Getting the type of 'fixedt' (line 119)
    fixedt_388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 39), 'fixedt')
    # Obtaining the member '__getitem__' of a type (line 119)
    getitem___389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 39), fixedt_388, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 119)
    subscript_call_result_390 = invoke(stypy.reporting.localization.Localization(__file__, 119, 39), getitem___389, var_387)
    
    int_391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 54), 'int')
    # Applying the binary operator '==' (line 119)
    result_eq_392 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 39), '==', subscript_call_result_390, int_391)
    
    # Getting the type of 'var' (line 119)
    var_386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'var')
    list_399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 12), list_399, var_386)
    # Assigning a type to the variable 'stypy_return_type' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'stypy_return_type', list_399)
    
    # ################# End of 'unfixed_vars(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'unfixed_vars' in the type store
    # Getting the type of 'stypy_return_type' (line 118)
    stypy_return_type_400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_400)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'unfixed_vars'
    return stypy_return_type_400

# Assigning a type to the variable 'unfixed_vars' (line 118)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 0), 'unfixed_vars', unfixed_vars)
# Declaration of the 'SatSolverRun' class

class SatSolverRun:

    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 123, 4, False)
        # Assigning a type to the variable 'self' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SatSolverRun.run.__dict__.__setitem__('stypy_localization', localization)
        SatSolverRun.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SatSolverRun.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        SatSolverRun.run.__dict__.__setitem__('stypy_function_name', 'SatSolverRun.run')
        SatSolverRun.run.__dict__.__setitem__('stypy_param_names_list', [])
        SatSolverRun.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        SatSolverRun.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SatSolverRun.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        SatSolverRun.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        SatSolverRun.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SatSolverRun.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SatSolverRun.run', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'run', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'run(...)' code ##################

        
        # Assigning a Num to a Name (line 124):
        int_401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 20), 'int')
        # Assigning a type to the variable 'nodecount' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'nodecount', int_401)
        
        
        # Call to solve_rec(...): (line 125)
        # Processing the call keyword arguments (line 125)
        kwargs_403 = {}
        # Getting the type of 'solve_rec' (line 125)
        solve_rec_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 15), 'solve_rec', False)
        # Calling solve_rec(args, kwargs) (line 125)
        solve_rec_call_result_404 = invoke(stypy.reporting.localization.Localization(__file__, 125, 15), solve_rec_402, *[], **kwargs_403)
        
        # Applying the 'not' unary operator (line 125)
        result_not__405 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 11), 'not', solve_rec_call_result_404)
        
        # Testing if the type of an if condition is none (line 125)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 125, 8), result_not__405):
            pass
        else:
            
            # Testing the type of an if condition (line 125)
            if_condition_406 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 8), result_not__405)
            # Assigning a type to the variable 'if_condition_406' (line 125)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'if_condition_406', if_condition_406)
            # SSA begins for if statement (line 125)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            pass
            # SSA join for if statement (line 125)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'True' (line 128)
        True_407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'stypy_return_type', True_407)
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 123)
        stypy_return_type_408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_408)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_408


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 122, 0, False)
        # Assigning a type to the variable 'self' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SatSolverRun.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'SatSolverRun' (line 122)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 0), 'SatSolverRun', SatSolverRun)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 131, 0, False)
    
    # Passed parameters checking function
    run.stypy_localization = localization
    run.stypy_type_of_self = None
    run.stypy_type_store = module_type_store
    run.stypy_function_name = 'run'
    run.stypy_param_names_list = []
    run.stypy_varargs_param_name = None
    run.stypy_kwargs_param_name = None
    run.stypy_call_defaults = defaults
    run.stypy_call_varargs = varargs
    run.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'run', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'run', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'run(...)' code ##################

    
    # Assigning a Call to a Name (line 132):
    
    # Call to SatSolverRun(...): (line 132)
    # Processing the call keyword arguments (line 132)
    kwargs_410 = {}
    # Getting the type of 'SatSolverRun' (line 132)
    SatSolverRun_409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'SatSolverRun', False)
    # Calling SatSolverRun(args, kwargs) (line 132)
    SatSolverRun_call_result_411 = invoke(stypy.reporting.localization.Localization(__file__, 132, 8), SatSolverRun_409, *[], **kwargs_410)
    
    # Assigning a type to the variable 's' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 's', SatSolverRun_call_result_411)
    
    # Call to run(...): (line 133)
    # Processing the call keyword arguments (line 133)
    kwargs_414 = {}
    # Getting the type of 's' (line 133)
    s_412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 's', False)
    # Obtaining the member 'run' of a type (line 133)
    run_413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 4), s_412, 'run')
    # Calling run(args, kwargs) (line 133)
    run_call_result_415 = invoke(stypy.reporting.localization.Localization(__file__, 133, 4), run_413, *[], **kwargs_414)
    
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 131)
    stypy_return_type_416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_416)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_416

# Assigning a type to the variable 'run' (line 131)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 0), 'run', run)

# Call to run(...): (line 136)
# Processing the call keyword arguments (line 136)
kwargs_418 = {}
# Getting the type of 'run' (line 136)
run_417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 0), 'run', False)
# Calling run(args, kwargs) (line 136)
run_call_result_419 = invoke(stypy.reporting.localization.Localization(__file__, 136, 0), run_417, *[], **kwargs_418)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
