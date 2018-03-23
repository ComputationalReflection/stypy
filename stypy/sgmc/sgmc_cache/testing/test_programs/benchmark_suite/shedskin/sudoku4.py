
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ## Solve Every Sudoku Puzzle
2: 
3: ## See http://norvig.com/sudoku.html
4: 
5: ## Throughout this program we have:
6: ##   r is a row,    e.g. 'A'
7: ##   c is a column, e.g. '3'
8: ##   s is a square, e.g. 'A3'
9: ##   d is a digit,  e.g. '9'
10: ##   u is a unit,   e.g. ['A1','B1','C1','D1','E1','F1','G1','H1','I1']
11: ##   g is a grid,   e.g. 81 non-blank chars, e.g. starting with '.18...7...
12: ##   values is a dict of possible values, e.g. {'A1':'123489', 'A2':'8', ...}
13: 
14: def cross(A, B):
15:     return [a + b for a in A for b in B]
16: 
17: 
18: rows = 'ABCDEFGHI'
19: cols = '123456789'
20: digits = '123456789'
21: squares = cross(rows, cols)
22: unitlist = ([cross(rows, c) for c in cols] +
23:             [cross(r, cols) for r in rows] +
24:             [cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')])
25: units = dict([(s, [u for u in unitlist if s in u])
26:               for s in squares])
27: peers = dict([(s, set([s2 for u in units[s] for s2 in u if s2 != s]))
28:               for s in squares])
29: 
30: 
31: def search(values):
32:     "Using depth-first search and propagation, try all possible values."
33:     if values is None:
34:         return None  ## Failed earlier
35:     if all([len(values[s]) == 1 for s in squares]):
36:         return values  ## Solved!
37:     ## Chose the unfilled square s with the fewest possibilities
38:     _, s = min([(len(values[s]), s) for s in squares if len(values[s]) > 1])
39:     for d in values[s]:
40:         r = search(assign(values.copy(), s, d))
41:         if r: return r
42: 
43: 
44: def assign(values, s, d):
45:     "Eliminate all the other values (except d) from values[s] and propagate."
46:     if all([eliminate(values, s, d2) for d2 in values[s] if d2 != d]):
47:         return values
48:     else:
49:         return None
50: 
51: 
52: def eliminate(values, s, d):
53:     "Eliminate d from values[s]; propagate when values or places <= 2."
54:     if d not in values[s]:
55:         return values  ## Already eliminated
56:     values[s] = values[s].replace(d, '')
57:     if len(values[s]) == 0:
58:         return None  ## Contradiction: removed last value
59:     elif len(values[s]) == 1:
60:         ## If there is only one value (d2) left in square, remove it from peers
61:         d2, = values[s]
62:         if not all([eliminate(values, s2, d2) for s2 in peers[s]]):
63:             return None
64:     ## Now check the places where d appears in the units of s
65:     for u in units[s]:
66:         dplaces = [s for s in u if d in values[s]]
67:         if len(dplaces) == 0:
68:             return None
69:         elif len(dplaces) == 1:
70:             # d can only be in one place in unit; assign it there
71:             if not assign(values, dplaces[0], d):
72:                 return None
73:     return values
74: 
75: 
76: def parse_grid(grid):
77:     "Given a string of 81 digits (or .0-), return a dict of {cell:values}"
78:     grid = [c for c in grid if c in '0.-123456789']
79:     values = dict([(s, digits) for s in squares])  ## Each square can be any digit
80:     for s, d in zip(squares, grid):
81:         if d in digits and not assign(values, s, d):
82:             return None
83:     return values
84: 
85: 
86: def solve_file(filename, sep, action):  # =lambda x: x):
87:     "Parse a file into a sequence of 81-char descriptions and solve them."
88:     results = [action(search(parse_grid(grid)))
89:                for grid in file(filename).read().strip().split(sep)]
90:     ##    print "## Got %d out of %d" % (
91:     ##          sum([(r is not None) for r in results]), len(results))
92:     return results
93: 
94: 
95: def printboard(values):
96:     "Used for debugging."
97:     width = 1 + max([len(values[s]) for s in squares])
98:     line = '\n' + '+'.join(['-' * (width * 3)] * 3)
99:     for r in rows:
100:         pass
101:     ##        print ''.join([values[r+c].center(width) + ('|' if c in '36' else '')
102:     ##                      for c in cols]) + (line if r in 'CF' else '')
103:     ##    print
104:     return values
105: 
106: 
107: import os
108: 
109: 
110: def Relative(path):
111:     return os.path.join(os.path.dirname(__file__), path)
112: 
113: 
114: def run():
115:     solve_file(Relative("testdata/top95.txt"), '\n', printboard)
116:     return True
117: 
118: 
119: ## References used:
120: ## http://www.scanraid.com/BasicStrategies.htm
121: ## http://www.krazydad.com/blog/2005/09/29/an-index-of-sudoku-strategies/
122: ## http://www2.warwick.ac.uk/fac/sci/moac/currentstudents/peter_cock/python/sudoku/
123: 
124: run()
125: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################


@norecursion
def cross(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'cross'
    module_type_store = module_type_store.open_function_context('cross', 14, 0, False)
    
    # Passed parameters checking function
    cross.stypy_localization = localization
    cross.stypy_type_of_self = None
    cross.stypy_type_store = module_type_store
    cross.stypy_function_name = 'cross'
    cross.stypy_param_names_list = ['A', 'B']
    cross.stypy_varargs_param_name = None
    cross.stypy_kwargs_param_name = None
    cross.stypy_call_defaults = defaults
    cross.stypy_call_varargs = varargs
    cross.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'cross', ['A', 'B'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'cross', localization, ['A', 'B'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'cross(...)' code ##################

    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'A' (line 15)
    A_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 27), 'A')
    comprehension_9 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 12), A_8)
    # Assigning a type to the variable 'a' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), 'a', comprehension_9)
    # Calculating comprehension expression
    # Getting the type of 'B' (line 15)
    B_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 38), 'B')
    comprehension_11 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 12), B_10)
    # Assigning a type to the variable 'b' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), 'b', comprehension_11)
    # Getting the type of 'a' (line 15)
    a_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 12), 'a')
    # Getting the type of 'b' (line 15)
    b_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 16), 'b')
    # Applying the binary operator '+' (line 15)
    result_add_7 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 12), '+', a_5, b_6)
    
    list_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 12), list_12, result_add_7)
    # Assigning a type to the variable 'stypy_return_type' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'stypy_return_type', list_12)
    
    # ################# End of 'cross(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'cross' in the type store
    # Getting the type of 'stypy_return_type' (line 14)
    stypy_return_type_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_13)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'cross'
    return stypy_return_type_13

# Assigning a type to the variable 'cross' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'cross', cross)

# Assigning a Str to a Name (line 18):

# Assigning a Str to a Name (line 18):
str_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 7), 'str', 'ABCDEFGHI')
# Assigning a type to the variable 'rows' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'rows', str_14)

# Assigning a Str to a Name (line 19):

# Assigning a Str to a Name (line 19):
str_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 7), 'str', '123456789')
# Assigning a type to the variable 'cols' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'cols', str_15)

# Assigning a Str to a Name (line 20):

# Assigning a Str to a Name (line 20):
str_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 9), 'str', '123456789')
# Assigning a type to the variable 'digits' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'digits', str_16)

# Assigning a Call to a Name (line 21):

# Assigning a Call to a Name (line 21):

# Call to cross(...): (line 21)
# Processing the call arguments (line 21)
# Getting the type of 'rows' (line 21)
rows_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'rows', False)
# Getting the type of 'cols' (line 21)
cols_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 22), 'cols', False)
# Processing the call keyword arguments (line 21)
kwargs_20 = {}
# Getting the type of 'cross' (line 21)
cross_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 10), 'cross', False)
# Calling cross(args, kwargs) (line 21)
cross_call_result_21 = invoke(stypy.reporting.localization.Localization(__file__, 21, 10), cross_17, *[rows_18, cols_19], **kwargs_20)

# Assigning a type to the variable 'squares' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'squares', cross_call_result_21)

# Assigning a BinOp to a Name (line 22):

# Assigning a BinOp to a Name (line 22):
# Calculating list comprehension
# Calculating comprehension expression
# Getting the type of 'cols' (line 22)
cols_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 37), 'cols')
comprehension_28 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 13), cols_27)
# Assigning a type to the variable 'c' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 13), 'c', comprehension_28)

# Call to cross(...): (line 22)
# Processing the call arguments (line 22)
# Getting the type of 'rows' (line 22)
rows_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 19), 'rows', False)
# Getting the type of 'c' (line 22)
c_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 25), 'c', False)
# Processing the call keyword arguments (line 22)
kwargs_25 = {}
# Getting the type of 'cross' (line 22)
cross_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 13), 'cross', False)
# Calling cross(args, kwargs) (line 22)
cross_call_result_26 = invoke(stypy.reporting.localization.Localization(__file__, 22, 13), cross_22, *[rows_23, c_24], **kwargs_25)

list_29 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 13), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 13), list_29, cross_call_result_26)
# Calculating list comprehension
# Calculating comprehension expression
# Getting the type of 'rows' (line 23)
rows_35 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 37), 'rows')
comprehension_36 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 13), rows_35)
# Assigning a type to the variable 'r' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 13), 'r', comprehension_36)

# Call to cross(...): (line 23)
# Processing the call arguments (line 23)
# Getting the type of 'r' (line 23)
r_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 19), 'r', False)
# Getting the type of 'cols' (line 23)
cols_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 22), 'cols', False)
# Processing the call keyword arguments (line 23)
kwargs_33 = {}
# Getting the type of 'cross' (line 23)
cross_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 13), 'cross', False)
# Calling cross(args, kwargs) (line 23)
cross_call_result_34 = invoke(stypy.reporting.localization.Localization(__file__, 23, 13), cross_30, *[r_31, cols_32], **kwargs_33)

list_37 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 13), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 13), list_37, cross_call_result_34)
# Applying the binary operator '+' (line 22)
result_add_38 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 12), '+', list_29, list_37)

# Calculating list comprehension
# Calculating comprehension expression

# Obtaining an instance of the builtin type 'tuple' (line 24)
tuple_44 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 38), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 24)
# Adding element type (line 24)
str_45 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 38), 'str', 'ABC')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 38), tuple_44, str_45)
# Adding element type (line 24)
str_46 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 45), 'str', 'DEF')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 38), tuple_44, str_46)
# Adding element type (line 24)
str_47 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 52), 'str', 'GHI')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 38), tuple_44, str_47)

comprehension_48 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 13), tuple_44)
# Assigning a type to the variable 'rs' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 13), 'rs', comprehension_48)
# Calculating comprehension expression

# Obtaining an instance of the builtin type 'tuple' (line 24)
tuple_49 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 70), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 24)
# Adding element type (line 24)
str_50 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 70), 'str', '123')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 70), tuple_49, str_50)
# Adding element type (line 24)
str_51 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 77), 'str', '456')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 70), tuple_49, str_51)
# Adding element type (line 24)
str_52 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 84), 'str', '789')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 70), tuple_49, str_52)

comprehension_53 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 13), tuple_49)
# Assigning a type to the variable 'cs' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 13), 'cs', comprehension_53)

# Call to cross(...): (line 24)
# Processing the call arguments (line 24)
# Getting the type of 'rs' (line 24)
rs_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 19), 'rs', False)
# Getting the type of 'cs' (line 24)
cs_41 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 23), 'cs', False)
# Processing the call keyword arguments (line 24)
kwargs_42 = {}
# Getting the type of 'cross' (line 24)
cross_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 13), 'cross', False)
# Calling cross(args, kwargs) (line 24)
cross_call_result_43 = invoke(stypy.reporting.localization.Localization(__file__, 24, 13), cross_39, *[rs_40, cs_41], **kwargs_42)

list_54 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 13), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 13), list_54, cross_call_result_43)
# Applying the binary operator '+' (line 23)
result_add_55 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 43), '+', result_add_38, list_54)

# Assigning a type to the variable 'unitlist' (line 22)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'unitlist', result_add_55)

# Assigning a Call to a Name (line 25):

# Assigning a Call to a Name (line 25):

# Call to dict(...): (line 25)
# Processing the call arguments (line 25)
# Calculating list comprehension
# Calculating comprehension expression
# Getting the type of 'squares' (line 26)
squares_66 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 23), 'squares', False)
comprehension_67 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 14), squares_66)
# Assigning a type to the variable 's' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 14), 's', comprehension_67)

# Obtaining an instance of the builtin type 'tuple' (line 25)
tuple_57 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 25)
# Adding element type (line 25)
# Getting the type of 's' (line 25)
s_58 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 's', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 15), tuple_57, s_58)
# Adding element type (line 25)
# Calculating list comprehension
# Calculating comprehension expression
# Getting the type of 'unitlist' (line 25)
unitlist_63 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 30), 'unitlist', False)
comprehension_64 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 19), unitlist_63)
# Assigning a type to the variable 'u' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 19), 'u', comprehension_64)

# Getting the type of 's' (line 25)
s_60 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 42), 's', False)
# Getting the type of 'u' (line 25)
u_61 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 47), 'u', False)
# Applying the binary operator 'in' (line 25)
result_contains_62 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 42), 'in', s_60, u_61)

# Getting the type of 'u' (line 25)
u_59 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 19), 'u', False)
list_65 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 19), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 19), list_65, u_59)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 15), tuple_57, list_65)

list_68 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 14), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 14), list_68, tuple_57)
# Processing the call keyword arguments (line 25)
kwargs_69 = {}
# Getting the type of 'dict' (line 25)
dict_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'dict', False)
# Calling dict(args, kwargs) (line 25)
dict_call_result_70 = invoke(stypy.reporting.localization.Localization(__file__, 25, 8), dict_56, *[list_68], **kwargs_69)

# Assigning a type to the variable 'units' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'units', dict_call_result_70)

# Assigning a Call to a Name (line 27):

# Assigning a Call to a Name (line 27):

# Call to dict(...): (line 27)
# Processing the call arguments (line 27)
# Calculating list comprehension
# Calculating comprehension expression
# Getting the type of 'squares' (line 28)
squares_89 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 23), 'squares', False)
comprehension_90 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 14), squares_89)
# Assigning a type to the variable 's' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 14), 's', comprehension_90)

# Obtaining an instance of the builtin type 'tuple' (line 27)
tuple_72 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 27)
# Adding element type (line 27)
# Getting the type of 's' (line 27)
s_73 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 15), 's', False)
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 15), tuple_72, s_73)
# Adding element type (line 27)

# Call to set(...): (line 27)
# Processing the call arguments (line 27)
# Calculating list comprehension
# Calculating comprehension expression

# Obtaining the type of the subscript
# Getting the type of 's' (line 27)
s_76 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 41), 's', False)
# Getting the type of 'units' (line 27)
units_77 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 35), 'units', False)
# Obtaining the member '__getitem__' of a type (line 27)
getitem___78 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 35), units_77, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 27)
subscript_call_result_79 = invoke(stypy.reporting.localization.Localization(__file__, 27, 35), getitem___78, s_76)

comprehension_80 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 23), subscript_call_result_79)
# Assigning a type to the variable 'u' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 23), 'u', comprehension_80)
# Calculating comprehension expression
# Getting the type of 'u' (line 27)
u_84 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 54), 'u', False)
comprehension_85 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 23), u_84)
# Assigning a type to the variable 's2' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 23), 's2', comprehension_85)

# Getting the type of 's2' (line 27)
s2_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 59), 's2', False)
# Getting the type of 's' (line 27)
s_82 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 65), 's', False)
# Applying the binary operator '!=' (line 27)
result_ne_83 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 59), '!=', s2_81, s_82)

# Getting the type of 's2' (line 27)
s2_75 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 23), 's2', False)
list_86 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 23), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 23), list_86, s2_75)
# Processing the call keyword arguments (line 27)
kwargs_87 = {}
# Getting the type of 'set' (line 27)
set_74 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 18), 'set', False)
# Calling set(args, kwargs) (line 27)
set_call_result_88 = invoke(stypy.reporting.localization.Localization(__file__, 27, 18), set_74, *[list_86], **kwargs_87)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 15), tuple_72, set_call_result_88)

list_91 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 14), 'list')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 14), list_91, tuple_72)
# Processing the call keyword arguments (line 27)
kwargs_92 = {}
# Getting the type of 'dict' (line 27)
dict_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'dict', False)
# Calling dict(args, kwargs) (line 27)
dict_call_result_93 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), dict_71, *[list_91], **kwargs_92)

# Assigning a type to the variable 'peers' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'peers', dict_call_result_93)

@norecursion
def search(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'search'
    module_type_store = module_type_store.open_function_context('search', 31, 0, False)
    
    # Passed parameters checking function
    search.stypy_localization = localization
    search.stypy_type_of_self = None
    search.stypy_type_store = module_type_store
    search.stypy_function_name = 'search'
    search.stypy_param_names_list = ['values']
    search.stypy_varargs_param_name = None
    search.stypy_kwargs_param_name = None
    search.stypy_call_defaults = defaults
    search.stypy_call_varargs = varargs
    search.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'search', ['values'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'search', localization, ['values'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'search(...)' code ##################

    str_94 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 4), 'str', 'Using depth-first search and propagation, try all possible values.')
    
    # Type idiom detected: calculating its left and rigth part (line 33)
    # Getting the type of 'values' (line 33)
    values_95 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 7), 'values')
    # Getting the type of 'None' (line 33)
    None_96 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 17), 'None')
    
    (may_be_97, more_types_in_union_98) = may_be_none(values_95, None_96)

    if may_be_97:

        if more_types_in_union_98:
            # Runtime conditional SSA (line 33)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'None' (line 34)
        None_99 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'stypy_return_type', None_99)

        if more_types_in_union_98:
            # SSA join for if statement (line 33)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'values' (line 33)
    values_100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'values')
    # Assigning a type to the variable 'values' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'values', remove_type_from_union(values_100, types.NoneType))
    
    # Call to all(...): (line 35)
    # Processing the call arguments (line 35)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'squares' (line 35)
    squares_111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 41), 'squares', False)
    comprehension_112 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 12), squares_111)
    # Assigning a type to the variable 's' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 's', comprehension_112)
    
    
    # Call to len(...): (line 35)
    # Processing the call arguments (line 35)
    
    # Obtaining the type of the subscript
    # Getting the type of 's' (line 35)
    s_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 23), 's', False)
    # Getting the type of 'values' (line 35)
    values_104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 16), 'values', False)
    # Obtaining the member '__getitem__' of a type (line 35)
    getitem___105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 16), values_104, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 35)
    subscript_call_result_106 = invoke(stypy.reporting.localization.Localization(__file__, 35, 16), getitem___105, s_103)
    
    # Processing the call keyword arguments (line 35)
    kwargs_107 = {}
    # Getting the type of 'len' (line 35)
    len_102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'len', False)
    # Calling len(args, kwargs) (line 35)
    len_call_result_108 = invoke(stypy.reporting.localization.Localization(__file__, 35, 12), len_102, *[subscript_call_result_106], **kwargs_107)
    
    int_109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 30), 'int')
    # Applying the binary operator '==' (line 35)
    result_eq_110 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 12), '==', len_call_result_108, int_109)
    
    list_113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 12), list_113, result_eq_110)
    # Processing the call keyword arguments (line 35)
    kwargs_114 = {}
    # Getting the type of 'all' (line 35)
    all_101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 7), 'all', False)
    # Calling all(args, kwargs) (line 35)
    all_call_result_115 = invoke(stypy.reporting.localization.Localization(__file__, 35, 7), all_101, *[list_113], **kwargs_114)
    
    # Testing if the type of an if condition is none (line 35)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 35, 4), all_call_result_115):
        pass
    else:
        
        # Testing the type of an if condition (line 35)
        if_condition_116 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 35, 4), all_call_result_115)
        # Assigning a type to the variable 'if_condition_116' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'if_condition_116', if_condition_116)
        # SSA begins for if statement (line 35)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'values' (line 36)
        values_117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 15), 'values')
        # Assigning a type to the variable 'stypy_return_type' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'stypy_return_type', values_117)
        # SSA join for if statement (line 35)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Tuple (line 38):
    
    # Assigning a Call to a Name:
    
    # Call to min(...): (line 38)
    # Processing the call arguments (line 38)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'squares' (line 38)
    squares_137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 45), 'squares', False)
    comprehension_138 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 16), squares_137)
    # Assigning a type to the variable 's' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 's', comprehension_138)
    
    
    # Call to len(...): (line 38)
    # Processing the call arguments (line 38)
    
    # Obtaining the type of the subscript
    # Getting the type of 's' (line 38)
    s_129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 67), 's', False)
    # Getting the type of 'values' (line 38)
    values_130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 60), 'values', False)
    # Obtaining the member '__getitem__' of a type (line 38)
    getitem___131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 60), values_130, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 38)
    subscript_call_result_132 = invoke(stypy.reporting.localization.Localization(__file__, 38, 60), getitem___131, s_129)
    
    # Processing the call keyword arguments (line 38)
    kwargs_133 = {}
    # Getting the type of 'len' (line 38)
    len_128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 56), 'len', False)
    # Calling len(args, kwargs) (line 38)
    len_call_result_134 = invoke(stypy.reporting.localization.Localization(__file__, 38, 56), len_128, *[subscript_call_result_132], **kwargs_133)
    
    int_135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 73), 'int')
    # Applying the binary operator '>' (line 38)
    result_gt_136 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 56), '>', len_call_result_134, int_135)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 38)
    tuple_119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 17), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 38)
    # Adding element type (line 38)
    
    # Call to len(...): (line 38)
    # Processing the call arguments (line 38)
    
    # Obtaining the type of the subscript
    # Getting the type of 's' (line 38)
    s_121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 28), 's', False)
    # Getting the type of 'values' (line 38)
    values_122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 21), 'values', False)
    # Obtaining the member '__getitem__' of a type (line 38)
    getitem___123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 21), values_122, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 38)
    subscript_call_result_124 = invoke(stypy.reporting.localization.Localization(__file__, 38, 21), getitem___123, s_121)
    
    # Processing the call keyword arguments (line 38)
    kwargs_125 = {}
    # Getting the type of 'len' (line 38)
    len_120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 17), 'len', False)
    # Calling len(args, kwargs) (line 38)
    len_call_result_126 = invoke(stypy.reporting.localization.Localization(__file__, 38, 17), len_120, *[subscript_call_result_124], **kwargs_125)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 17), tuple_119, len_call_result_126)
    # Adding element type (line 38)
    # Getting the type of 's' (line 38)
    s_127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 33), 's', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 17), tuple_119, s_127)
    
    list_139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 16), list_139, tuple_119)
    # Processing the call keyword arguments (line 38)
    kwargs_140 = {}
    # Getting the type of 'min' (line 38)
    min_118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 11), 'min', False)
    # Calling min(args, kwargs) (line 38)
    min_call_result_141 = invoke(stypy.reporting.localization.Localization(__file__, 38, 11), min_118, *[list_139], **kwargs_140)
    
    # Assigning a type to the variable 'call_assignment_1' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'call_assignment_1', min_call_result_141)
    
    # Assigning a Call to a Name (line 38):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_1' (line 38)
    call_assignment_1_142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'call_assignment_1', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_143 = stypy_get_value_from_tuple(call_assignment_1_142, 2, 0)
    
    # Assigning a type to the variable 'call_assignment_2' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'call_assignment_2', stypy_get_value_from_tuple_call_result_143)
    
    # Assigning a Name to a Name (line 38):
    # Getting the type of 'call_assignment_2' (line 38)
    call_assignment_2_144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'call_assignment_2')
    # Assigning a type to the variable '_' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), '_', call_assignment_2_144)
    
    # Assigning a Call to a Name (line 38):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_1' (line 38)
    call_assignment_1_145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'call_assignment_1', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_146 = stypy_get_value_from_tuple(call_assignment_1_145, 2, 1)
    
    # Assigning a type to the variable 'call_assignment_3' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'call_assignment_3', stypy_get_value_from_tuple_call_result_146)
    
    # Assigning a Name to a Name (line 38):
    # Getting the type of 'call_assignment_3' (line 38)
    call_assignment_3_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'call_assignment_3')
    # Assigning a type to the variable 's' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 7), 's', call_assignment_3_147)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 's' (line 39)
    s_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 20), 's')
    # Getting the type of 'values' (line 39)
    values_149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 13), 'values')
    # Obtaining the member '__getitem__' of a type (line 39)
    getitem___150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 13), values_149, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 39)
    subscript_call_result_151 = invoke(stypy.reporting.localization.Localization(__file__, 39, 13), getitem___150, s_148)
    
    # Assigning a type to the variable 'subscript_call_result_151' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'subscript_call_result_151', subscript_call_result_151)
    # Testing if the for loop is going to be iterated (line 39)
    # Testing the type of a for loop iterable (line 39)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 39, 4), subscript_call_result_151)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 39, 4), subscript_call_result_151):
        # Getting the type of the for loop variable (line 39)
        for_loop_var_152 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 39, 4), subscript_call_result_151)
        # Assigning a type to the variable 'd' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'd', for_loop_var_152)
        # SSA begins for a for statement (line 39)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 40):
        
        # Assigning a Call to a Name (line 40):
        
        # Call to search(...): (line 40)
        # Processing the call arguments (line 40)
        
        # Call to assign(...): (line 40)
        # Processing the call arguments (line 40)
        
        # Call to copy(...): (line 40)
        # Processing the call keyword arguments (line 40)
        kwargs_157 = {}
        # Getting the type of 'values' (line 40)
        values_155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 26), 'values', False)
        # Obtaining the member 'copy' of a type (line 40)
        copy_156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 26), values_155, 'copy')
        # Calling copy(args, kwargs) (line 40)
        copy_call_result_158 = invoke(stypy.reporting.localization.Localization(__file__, 40, 26), copy_156, *[], **kwargs_157)
        
        # Getting the type of 's' (line 40)
        s_159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 41), 's', False)
        # Getting the type of 'd' (line 40)
        d_160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 44), 'd', False)
        # Processing the call keyword arguments (line 40)
        kwargs_161 = {}
        # Getting the type of 'assign' (line 40)
        assign_154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 19), 'assign', False)
        # Calling assign(args, kwargs) (line 40)
        assign_call_result_162 = invoke(stypy.reporting.localization.Localization(__file__, 40, 19), assign_154, *[copy_call_result_158, s_159, d_160], **kwargs_161)
        
        # Processing the call keyword arguments (line 40)
        kwargs_163 = {}
        # Getting the type of 'search' (line 40)
        search_153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'search', False)
        # Calling search(args, kwargs) (line 40)
        search_call_result_164 = invoke(stypy.reporting.localization.Localization(__file__, 40, 12), search_153, *[assign_call_result_162], **kwargs_163)
        
        # Assigning a type to the variable 'r' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'r', search_call_result_164)
        # Getting the type of 'r' (line 41)
        r_165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 11), 'r')
        # Testing if the type of an if condition is none (line 41)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 41, 8), r_165):
            pass
        else:
            
            # Testing the type of an if condition (line 41)
            if_condition_166 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 8), r_165)
            # Assigning a type to the variable 'if_condition_166' (line 41)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'if_condition_166', if_condition_166)
            # SSA begins for if statement (line 41)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'r' (line 41)
            r_167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 21), 'r')
            # Assigning a type to the variable 'stypy_return_type' (line 41)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 14), 'stypy_return_type', r_167)
            # SSA join for if statement (line 41)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'search(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'search' in the type store
    # Getting the type of 'stypy_return_type' (line 31)
    stypy_return_type_168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_168)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'search'
    return stypy_return_type_168

# Assigning a type to the variable 'search' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'search', search)

@norecursion
def assign(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'assign'
    module_type_store = module_type_store.open_function_context('assign', 44, 0, False)
    
    # Passed parameters checking function
    assign.stypy_localization = localization
    assign.stypy_type_of_self = None
    assign.stypy_type_store = module_type_store
    assign.stypy_function_name = 'assign'
    assign.stypy_param_names_list = ['values', 's', 'd']
    assign.stypy_varargs_param_name = None
    assign.stypy_kwargs_param_name = None
    assign.stypy_call_defaults = defaults
    assign.stypy_call_varargs = varargs
    assign.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'assign', ['values', 's', 'd'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'assign', localization, ['values', 's', 'd'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'assign(...)' code ##################

    str_169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 4), 'str', 'Eliminate all the other values (except d) from values[s] and propagate.')
    
    # Call to all(...): (line 46)
    # Processing the call arguments (line 46)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Obtaining the type of the subscript
    # Getting the type of 's' (line 46)
    s_180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 54), 's', False)
    # Getting the type of 'values' (line 46)
    values_181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 47), 'values', False)
    # Obtaining the member '__getitem__' of a type (line 46)
    getitem___182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 47), values_181, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 46)
    subscript_call_result_183 = invoke(stypy.reporting.localization.Localization(__file__, 46, 47), getitem___182, s_180)
    
    comprehension_184 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 12), subscript_call_result_183)
    # Assigning a type to the variable 'd2' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'd2', comprehension_184)
    
    # Getting the type of 'd2' (line 46)
    d2_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 60), 'd2', False)
    # Getting the type of 'd' (line 46)
    d_178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 66), 'd', False)
    # Applying the binary operator '!=' (line 46)
    result_ne_179 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 60), '!=', d2_177, d_178)
    
    
    # Call to eliminate(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'values' (line 46)
    values_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 22), 'values', False)
    # Getting the type of 's' (line 46)
    s_173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 30), 's', False)
    # Getting the type of 'd2' (line 46)
    d2_174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 33), 'd2', False)
    # Processing the call keyword arguments (line 46)
    kwargs_175 = {}
    # Getting the type of 'eliminate' (line 46)
    eliminate_171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'eliminate', False)
    # Calling eliminate(args, kwargs) (line 46)
    eliminate_call_result_176 = invoke(stypy.reporting.localization.Localization(__file__, 46, 12), eliminate_171, *[values_172, s_173, d2_174], **kwargs_175)
    
    list_185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 12), list_185, eliminate_call_result_176)
    # Processing the call keyword arguments (line 46)
    kwargs_186 = {}
    # Getting the type of 'all' (line 46)
    all_170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 7), 'all', False)
    # Calling all(args, kwargs) (line 46)
    all_call_result_187 = invoke(stypy.reporting.localization.Localization(__file__, 46, 7), all_170, *[list_185], **kwargs_186)
    
    # Testing if the type of an if condition is none (line 46)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 46, 4), all_call_result_187):
        # Getting the type of 'None' (line 49)
        None_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'stypy_return_type', None_190)
    else:
        
        # Testing the type of an if condition (line 46)
        if_condition_188 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 46, 4), all_call_result_187)
        # Assigning a type to the variable 'if_condition_188' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'if_condition_188', if_condition_188)
        # SSA begins for if statement (line 46)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'values' (line 47)
        values_189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 15), 'values')
        # Assigning a type to the variable 'stypy_return_type' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'stypy_return_type', values_189)
        # SSA branch for the else part of an if statement (line 46)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'None' (line 49)
        None_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'stypy_return_type', None_190)
        # SSA join for if statement (line 46)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'assign(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'assign' in the type store
    # Getting the type of 'stypy_return_type' (line 44)
    stypy_return_type_191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_191)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'assign'
    return stypy_return_type_191

# Assigning a type to the variable 'assign' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'assign', assign)

@norecursion
def eliminate(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'eliminate'
    module_type_store = module_type_store.open_function_context('eliminate', 52, 0, False)
    
    # Passed parameters checking function
    eliminate.stypy_localization = localization
    eliminate.stypy_type_of_self = None
    eliminate.stypy_type_store = module_type_store
    eliminate.stypy_function_name = 'eliminate'
    eliminate.stypy_param_names_list = ['values', 's', 'd']
    eliminate.stypy_varargs_param_name = None
    eliminate.stypy_kwargs_param_name = None
    eliminate.stypy_call_defaults = defaults
    eliminate.stypy_call_varargs = varargs
    eliminate.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'eliminate', ['values', 's', 'd'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'eliminate', localization, ['values', 's', 'd'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'eliminate(...)' code ##################

    str_192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 4), 'str', 'Eliminate d from values[s]; propagate when values or places <= 2.')
    
    # Getting the type of 'd' (line 54)
    d_193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 7), 'd')
    
    # Obtaining the type of the subscript
    # Getting the type of 's' (line 54)
    s_194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 23), 's')
    # Getting the type of 'values' (line 54)
    values_195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'values')
    # Obtaining the member '__getitem__' of a type (line 54)
    getitem___196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 16), values_195, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 54)
    subscript_call_result_197 = invoke(stypy.reporting.localization.Localization(__file__, 54, 16), getitem___196, s_194)
    
    # Applying the binary operator 'notin' (line 54)
    result_contains_198 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 7), 'notin', d_193, subscript_call_result_197)
    
    # Testing if the type of an if condition is none (line 54)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 54, 4), result_contains_198):
        pass
    else:
        
        # Testing the type of an if condition (line 54)
        if_condition_199 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 54, 4), result_contains_198)
        # Assigning a type to the variable 'if_condition_199' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'if_condition_199', if_condition_199)
        # SSA begins for if statement (line 54)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'values' (line 55)
        values_200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 15), 'values')
        # Assigning a type to the variable 'stypy_return_type' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'stypy_return_type', values_200)
        # SSA join for if statement (line 54)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Subscript (line 56):
    
    # Assigning a Call to a Subscript (line 56):
    
    # Call to replace(...): (line 56)
    # Processing the call arguments (line 56)
    # Getting the type of 'd' (line 56)
    d_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 34), 'd', False)
    str_207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 37), 'str', '')
    # Processing the call keyword arguments (line 56)
    kwargs_208 = {}
    
    # Obtaining the type of the subscript
    # Getting the type of 's' (line 56)
    s_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 23), 's', False)
    # Getting the type of 'values' (line 56)
    values_202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 16), 'values', False)
    # Obtaining the member '__getitem__' of a type (line 56)
    getitem___203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 16), values_202, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 56)
    subscript_call_result_204 = invoke(stypy.reporting.localization.Localization(__file__, 56, 16), getitem___203, s_201)
    
    # Obtaining the member 'replace' of a type (line 56)
    replace_205 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 16), subscript_call_result_204, 'replace')
    # Calling replace(args, kwargs) (line 56)
    replace_call_result_209 = invoke(stypy.reporting.localization.Localization(__file__, 56, 16), replace_205, *[d_206, str_207], **kwargs_208)
    
    # Getting the type of 'values' (line 56)
    values_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'values')
    # Getting the type of 's' (line 56)
    s_211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 11), 's')
    # Storing an element on a container (line 56)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 4), values_210, (s_211, replace_call_result_209))
    
    
    # Call to len(...): (line 57)
    # Processing the call arguments (line 57)
    
    # Obtaining the type of the subscript
    # Getting the type of 's' (line 57)
    s_213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 18), 's', False)
    # Getting the type of 'values' (line 57)
    values_214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 11), 'values', False)
    # Obtaining the member '__getitem__' of a type (line 57)
    getitem___215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 11), values_214, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 57)
    subscript_call_result_216 = invoke(stypy.reporting.localization.Localization(__file__, 57, 11), getitem___215, s_213)
    
    # Processing the call keyword arguments (line 57)
    kwargs_217 = {}
    # Getting the type of 'len' (line 57)
    len_212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 7), 'len', False)
    # Calling len(args, kwargs) (line 57)
    len_call_result_218 = invoke(stypy.reporting.localization.Localization(__file__, 57, 7), len_212, *[subscript_call_result_216], **kwargs_217)
    
    int_219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 25), 'int')
    # Applying the binary operator '==' (line 57)
    result_eq_220 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 7), '==', len_call_result_218, int_219)
    
    # Testing if the type of an if condition is none (line 57)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 57, 4), result_eq_220):
        
        
        # Call to len(...): (line 59)
        # Processing the call arguments (line 59)
        
        # Obtaining the type of the subscript
        # Getting the type of 's' (line 59)
        s_224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 20), 's', False)
        # Getting the type of 'values' (line 59)
        values_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 13), 'values', False)
        # Obtaining the member '__getitem__' of a type (line 59)
        getitem___226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 13), values_225, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 59)
        subscript_call_result_227 = invoke(stypy.reporting.localization.Localization(__file__, 59, 13), getitem___226, s_224)
        
        # Processing the call keyword arguments (line 59)
        kwargs_228 = {}
        # Getting the type of 'len' (line 59)
        len_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 9), 'len', False)
        # Calling len(args, kwargs) (line 59)
        len_call_result_229 = invoke(stypy.reporting.localization.Localization(__file__, 59, 9), len_223, *[subscript_call_result_227], **kwargs_228)
        
        int_230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 27), 'int')
        # Applying the binary operator '==' (line 59)
        result_eq_231 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 9), '==', len_call_result_229, int_230)
        
        # Testing if the type of an if condition is none (line 59)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 59, 9), result_eq_231):
            pass
        else:
            
            # Testing the type of an if condition (line 59)
            if_condition_232 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 59, 9), result_eq_231)
            # Assigning a type to the variable 'if_condition_232' (line 59)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 9), 'if_condition_232', if_condition_232)
            # SSA begins for if statement (line 59)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Tuple (line 61):
            
            # Assigning a Subscript to a Name (line 61):
            
            # Obtaining the type of the subscript
            int_233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 8), 'int')
            
            # Obtaining the type of the subscript
            # Getting the type of 's' (line 61)
            s_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 21), 's')
            # Getting the type of 'values' (line 61)
            values_235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 14), 'values')
            # Obtaining the member '__getitem__' of a type (line 61)
            getitem___236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 14), values_235, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 61)
            subscript_call_result_237 = invoke(stypy.reporting.localization.Localization(__file__, 61, 14), getitem___236, s_234)
            
            # Obtaining the member '__getitem__' of a type (line 61)
            getitem___238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), subscript_call_result_237, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 61)
            subscript_call_result_239 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), getitem___238, int_233)
            
            # Assigning a type to the variable 'tuple_var_assignment_4' (line 61)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'tuple_var_assignment_4', subscript_call_result_239)
            
            # Assigning a Name to a Name (line 61):
            # Getting the type of 'tuple_var_assignment_4' (line 61)
            tuple_var_assignment_4_240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'tuple_var_assignment_4')
            # Assigning a type to the variable 'd2' (line 61)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'd2', tuple_var_assignment_4_240)
            
            
            # Call to all(...): (line 62)
            # Processing the call arguments (line 62)
            # Calculating list comprehension
            # Calculating comprehension expression
            
            # Obtaining the type of the subscript
            # Getting the type of 's' (line 62)
            s_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 62), 's', False)
            # Getting the type of 'peers' (line 62)
            peers_249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 56), 'peers', False)
            # Obtaining the member '__getitem__' of a type (line 62)
            getitem___250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 56), peers_249, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 62)
            subscript_call_result_251 = invoke(stypy.reporting.localization.Localization(__file__, 62, 56), getitem___250, s_248)
            
            comprehension_252 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 20), subscript_call_result_251)
            # Assigning a type to the variable 's2' (line 62)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 20), 's2', comprehension_252)
            
            # Call to eliminate(...): (line 62)
            # Processing the call arguments (line 62)
            # Getting the type of 'values' (line 62)
            values_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 30), 'values', False)
            # Getting the type of 's2' (line 62)
            s2_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 38), 's2', False)
            # Getting the type of 'd2' (line 62)
            d2_245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 42), 'd2', False)
            # Processing the call keyword arguments (line 62)
            kwargs_246 = {}
            # Getting the type of 'eliminate' (line 62)
            eliminate_242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 20), 'eliminate', False)
            # Calling eliminate(args, kwargs) (line 62)
            eliminate_call_result_247 = invoke(stypy.reporting.localization.Localization(__file__, 62, 20), eliminate_242, *[values_243, s2_244, d2_245], **kwargs_246)
            
            list_253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 20), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 20), list_253, eliminate_call_result_247)
            # Processing the call keyword arguments (line 62)
            kwargs_254 = {}
            # Getting the type of 'all' (line 62)
            all_241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 15), 'all', False)
            # Calling all(args, kwargs) (line 62)
            all_call_result_255 = invoke(stypy.reporting.localization.Localization(__file__, 62, 15), all_241, *[list_253], **kwargs_254)
            
            # Applying the 'not' unary operator (line 62)
            result_not__256 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 11), 'not', all_call_result_255)
            
            # Testing if the type of an if condition is none (line 62)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 62, 8), result_not__256):
                pass
            else:
                
                # Testing the type of an if condition (line 62)
                if_condition_257 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 62, 8), result_not__256)
                # Assigning a type to the variable 'if_condition_257' (line 62)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'if_condition_257', if_condition_257)
                # SSA begins for if statement (line 62)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'None' (line 63)
                None_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 19), 'None')
                # Assigning a type to the variable 'stypy_return_type' (line 63)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'stypy_return_type', None_258)
                # SSA join for if statement (line 62)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 59)
            module_type_store = module_type_store.join_ssa_context()
            

    else:
        
        # Testing the type of an if condition (line 57)
        if_condition_221 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 57, 4), result_eq_220)
        # Assigning a type to the variable 'if_condition_221' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'if_condition_221', if_condition_221)
        # SSA begins for if statement (line 57)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'None' (line 58)
        None_222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'stypy_return_type', None_222)
        # SSA branch for the else part of an if statement (line 57)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to len(...): (line 59)
        # Processing the call arguments (line 59)
        
        # Obtaining the type of the subscript
        # Getting the type of 's' (line 59)
        s_224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 20), 's', False)
        # Getting the type of 'values' (line 59)
        values_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 13), 'values', False)
        # Obtaining the member '__getitem__' of a type (line 59)
        getitem___226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 13), values_225, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 59)
        subscript_call_result_227 = invoke(stypy.reporting.localization.Localization(__file__, 59, 13), getitem___226, s_224)
        
        # Processing the call keyword arguments (line 59)
        kwargs_228 = {}
        # Getting the type of 'len' (line 59)
        len_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 9), 'len', False)
        # Calling len(args, kwargs) (line 59)
        len_call_result_229 = invoke(stypy.reporting.localization.Localization(__file__, 59, 9), len_223, *[subscript_call_result_227], **kwargs_228)
        
        int_230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 27), 'int')
        # Applying the binary operator '==' (line 59)
        result_eq_231 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 9), '==', len_call_result_229, int_230)
        
        # Testing if the type of an if condition is none (line 59)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 59, 9), result_eq_231):
            pass
        else:
            
            # Testing the type of an if condition (line 59)
            if_condition_232 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 59, 9), result_eq_231)
            # Assigning a type to the variable 'if_condition_232' (line 59)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 9), 'if_condition_232', if_condition_232)
            # SSA begins for if statement (line 59)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Subscript to a Tuple (line 61):
            
            # Assigning a Subscript to a Name (line 61):
            
            # Obtaining the type of the subscript
            int_233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 8), 'int')
            
            # Obtaining the type of the subscript
            # Getting the type of 's' (line 61)
            s_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 21), 's')
            # Getting the type of 'values' (line 61)
            values_235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 14), 'values')
            # Obtaining the member '__getitem__' of a type (line 61)
            getitem___236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 14), values_235, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 61)
            subscript_call_result_237 = invoke(stypy.reporting.localization.Localization(__file__, 61, 14), getitem___236, s_234)
            
            # Obtaining the member '__getitem__' of a type (line 61)
            getitem___238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), subscript_call_result_237, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 61)
            subscript_call_result_239 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), getitem___238, int_233)
            
            # Assigning a type to the variable 'tuple_var_assignment_4' (line 61)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'tuple_var_assignment_4', subscript_call_result_239)
            
            # Assigning a Name to a Name (line 61):
            # Getting the type of 'tuple_var_assignment_4' (line 61)
            tuple_var_assignment_4_240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'tuple_var_assignment_4')
            # Assigning a type to the variable 'd2' (line 61)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'd2', tuple_var_assignment_4_240)
            
            
            # Call to all(...): (line 62)
            # Processing the call arguments (line 62)
            # Calculating list comprehension
            # Calculating comprehension expression
            
            # Obtaining the type of the subscript
            # Getting the type of 's' (line 62)
            s_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 62), 's', False)
            # Getting the type of 'peers' (line 62)
            peers_249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 56), 'peers', False)
            # Obtaining the member '__getitem__' of a type (line 62)
            getitem___250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 56), peers_249, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 62)
            subscript_call_result_251 = invoke(stypy.reporting.localization.Localization(__file__, 62, 56), getitem___250, s_248)
            
            comprehension_252 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 20), subscript_call_result_251)
            # Assigning a type to the variable 's2' (line 62)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 20), 's2', comprehension_252)
            
            # Call to eliminate(...): (line 62)
            # Processing the call arguments (line 62)
            # Getting the type of 'values' (line 62)
            values_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 30), 'values', False)
            # Getting the type of 's2' (line 62)
            s2_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 38), 's2', False)
            # Getting the type of 'd2' (line 62)
            d2_245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 42), 'd2', False)
            # Processing the call keyword arguments (line 62)
            kwargs_246 = {}
            # Getting the type of 'eliminate' (line 62)
            eliminate_242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 20), 'eliminate', False)
            # Calling eliminate(args, kwargs) (line 62)
            eliminate_call_result_247 = invoke(stypy.reporting.localization.Localization(__file__, 62, 20), eliminate_242, *[values_243, s2_244, d2_245], **kwargs_246)
            
            list_253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 20), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 20), list_253, eliminate_call_result_247)
            # Processing the call keyword arguments (line 62)
            kwargs_254 = {}
            # Getting the type of 'all' (line 62)
            all_241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 15), 'all', False)
            # Calling all(args, kwargs) (line 62)
            all_call_result_255 = invoke(stypy.reporting.localization.Localization(__file__, 62, 15), all_241, *[list_253], **kwargs_254)
            
            # Applying the 'not' unary operator (line 62)
            result_not__256 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 11), 'not', all_call_result_255)
            
            # Testing if the type of an if condition is none (line 62)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 62, 8), result_not__256):
                pass
            else:
                
                # Testing the type of an if condition (line 62)
                if_condition_257 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 62, 8), result_not__256)
                # Assigning a type to the variable 'if_condition_257' (line 62)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'if_condition_257', if_condition_257)
                # SSA begins for if statement (line 62)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'None' (line 63)
                None_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 19), 'None')
                # Assigning a type to the variable 'stypy_return_type' (line 63)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'stypy_return_type', None_258)
                # SSA join for if statement (line 62)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 59)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for if statement (line 57)
        module_type_store = module_type_store.join_ssa_context()
        

    
    
    # Obtaining the type of the subscript
    # Getting the type of 's' (line 65)
    s_259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 19), 's')
    # Getting the type of 'units' (line 65)
    units_260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 13), 'units')
    # Obtaining the member '__getitem__' of a type (line 65)
    getitem___261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 13), units_260, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 65)
    subscript_call_result_262 = invoke(stypy.reporting.localization.Localization(__file__, 65, 13), getitem___261, s_259)
    
    # Assigning a type to the variable 'subscript_call_result_262' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'subscript_call_result_262', subscript_call_result_262)
    # Testing if the for loop is going to be iterated (line 65)
    # Testing the type of a for loop iterable (line 65)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 65, 4), subscript_call_result_262)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 65, 4), subscript_call_result_262):
        # Getting the type of the for loop variable (line 65)
        for_loop_var_263 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 65, 4), subscript_call_result_262)
        # Assigning a type to the variable 'u' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'u', for_loop_var_263)
        # SSA begins for a for statement (line 65)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a ListComp to a Name (line 66):
        
        # Assigning a ListComp to a Name (line 66):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'u' (line 66)
        u_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 30), 'u')
        comprehension_272 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 19), u_271)
        # Assigning a type to the variable 's' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 19), 's', comprehension_272)
        
        # Getting the type of 'd' (line 66)
        d_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 35), 'd')
        
        # Obtaining the type of the subscript
        # Getting the type of 's' (line 66)
        s_266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 47), 's')
        # Getting the type of 'values' (line 66)
        values_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 40), 'values')
        # Obtaining the member '__getitem__' of a type (line 66)
        getitem___268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 40), values_267, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 66)
        subscript_call_result_269 = invoke(stypy.reporting.localization.Localization(__file__, 66, 40), getitem___268, s_266)
        
        # Applying the binary operator 'in' (line 66)
        result_contains_270 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 35), 'in', d_265, subscript_call_result_269)
        
        # Getting the type of 's' (line 66)
        s_264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 19), 's')
        list_273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 19), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 19), list_273, s_264)
        # Assigning a type to the variable 'dplaces' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'dplaces', list_273)
        
        
        # Call to len(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'dplaces' (line 67)
        dplaces_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 15), 'dplaces', False)
        # Processing the call keyword arguments (line 67)
        kwargs_276 = {}
        # Getting the type of 'len' (line 67)
        len_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 11), 'len', False)
        # Calling len(args, kwargs) (line 67)
        len_call_result_277 = invoke(stypy.reporting.localization.Localization(__file__, 67, 11), len_274, *[dplaces_275], **kwargs_276)
        
        int_278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 27), 'int')
        # Applying the binary operator '==' (line 67)
        result_eq_279 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 11), '==', len_call_result_277, int_278)
        
        # Testing if the type of an if condition is none (line 67)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 67, 8), result_eq_279):
            
            
            # Call to len(...): (line 69)
            # Processing the call arguments (line 69)
            # Getting the type of 'dplaces' (line 69)
            dplaces_283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 17), 'dplaces', False)
            # Processing the call keyword arguments (line 69)
            kwargs_284 = {}
            # Getting the type of 'len' (line 69)
            len_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 13), 'len', False)
            # Calling len(args, kwargs) (line 69)
            len_call_result_285 = invoke(stypy.reporting.localization.Localization(__file__, 69, 13), len_282, *[dplaces_283], **kwargs_284)
            
            int_286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 29), 'int')
            # Applying the binary operator '==' (line 69)
            result_eq_287 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 13), '==', len_call_result_285, int_286)
            
            # Testing if the type of an if condition is none (line 69)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 69, 13), result_eq_287):
                pass
            else:
                
                # Testing the type of an if condition (line 69)
                if_condition_288 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 13), result_eq_287)
                # Assigning a type to the variable 'if_condition_288' (line 69)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 13), 'if_condition_288', if_condition_288)
                # SSA begins for if statement (line 69)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                
                # Call to assign(...): (line 71)
                # Processing the call arguments (line 71)
                # Getting the type of 'values' (line 71)
                values_290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 26), 'values', False)
                
                # Obtaining the type of the subscript
                int_291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 42), 'int')
                # Getting the type of 'dplaces' (line 71)
                dplaces_292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 34), 'dplaces', False)
                # Obtaining the member '__getitem__' of a type (line 71)
                getitem___293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 34), dplaces_292, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 71)
                subscript_call_result_294 = invoke(stypy.reporting.localization.Localization(__file__, 71, 34), getitem___293, int_291)
                
                # Getting the type of 'd' (line 71)
                d_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 46), 'd', False)
                # Processing the call keyword arguments (line 71)
                kwargs_296 = {}
                # Getting the type of 'assign' (line 71)
                assign_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 19), 'assign', False)
                # Calling assign(args, kwargs) (line 71)
                assign_call_result_297 = invoke(stypy.reporting.localization.Localization(__file__, 71, 19), assign_289, *[values_290, subscript_call_result_294, d_295], **kwargs_296)
                
                # Applying the 'not' unary operator (line 71)
                result_not__298 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 15), 'not', assign_call_result_297)
                
                # Testing if the type of an if condition is none (line 71)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 71, 12), result_not__298):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 71)
                    if_condition_299 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 12), result_not__298)
                    # Assigning a type to the variable 'if_condition_299' (line 71)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'if_condition_299', if_condition_299)
                    # SSA begins for if statement (line 71)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # Getting the type of 'None' (line 72)
                    None_300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 23), 'None')
                    # Assigning a type to the variable 'stypy_return_type' (line 72)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 16), 'stypy_return_type', None_300)
                    # SSA join for if statement (line 71)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 69)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 67)
            if_condition_280 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 8), result_eq_279)
            # Assigning a type to the variable 'if_condition_280' (line 67)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'if_condition_280', if_condition_280)
            # SSA begins for if statement (line 67)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'None' (line 68)
            None_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 19), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 68)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'stypy_return_type', None_281)
            # SSA branch for the else part of an if statement (line 67)
            module_type_store.open_ssa_branch('else')
            
            
            # Call to len(...): (line 69)
            # Processing the call arguments (line 69)
            # Getting the type of 'dplaces' (line 69)
            dplaces_283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 17), 'dplaces', False)
            # Processing the call keyword arguments (line 69)
            kwargs_284 = {}
            # Getting the type of 'len' (line 69)
            len_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 13), 'len', False)
            # Calling len(args, kwargs) (line 69)
            len_call_result_285 = invoke(stypy.reporting.localization.Localization(__file__, 69, 13), len_282, *[dplaces_283], **kwargs_284)
            
            int_286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 29), 'int')
            # Applying the binary operator '==' (line 69)
            result_eq_287 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 13), '==', len_call_result_285, int_286)
            
            # Testing if the type of an if condition is none (line 69)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 69, 13), result_eq_287):
                pass
            else:
                
                # Testing the type of an if condition (line 69)
                if_condition_288 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 13), result_eq_287)
                # Assigning a type to the variable 'if_condition_288' (line 69)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 13), 'if_condition_288', if_condition_288)
                # SSA begins for if statement (line 69)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                
                # Call to assign(...): (line 71)
                # Processing the call arguments (line 71)
                # Getting the type of 'values' (line 71)
                values_290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 26), 'values', False)
                
                # Obtaining the type of the subscript
                int_291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 42), 'int')
                # Getting the type of 'dplaces' (line 71)
                dplaces_292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 34), 'dplaces', False)
                # Obtaining the member '__getitem__' of a type (line 71)
                getitem___293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 34), dplaces_292, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 71)
                subscript_call_result_294 = invoke(stypy.reporting.localization.Localization(__file__, 71, 34), getitem___293, int_291)
                
                # Getting the type of 'd' (line 71)
                d_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 46), 'd', False)
                # Processing the call keyword arguments (line 71)
                kwargs_296 = {}
                # Getting the type of 'assign' (line 71)
                assign_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 19), 'assign', False)
                # Calling assign(args, kwargs) (line 71)
                assign_call_result_297 = invoke(stypy.reporting.localization.Localization(__file__, 71, 19), assign_289, *[values_290, subscript_call_result_294, d_295], **kwargs_296)
                
                # Applying the 'not' unary operator (line 71)
                result_not__298 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 15), 'not', assign_call_result_297)
                
                # Testing if the type of an if condition is none (line 71)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 71, 12), result_not__298):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 71)
                    if_condition_299 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 12), result_not__298)
                    # Assigning a type to the variable 'if_condition_299' (line 71)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'if_condition_299', if_condition_299)
                    # SSA begins for if statement (line 71)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # Getting the type of 'None' (line 72)
                    None_300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 23), 'None')
                    # Assigning a type to the variable 'stypy_return_type' (line 72)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 16), 'stypy_return_type', None_300)
                    # SSA join for if statement (line 71)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 69)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 67)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'values' (line 73)
    values_301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 11), 'values')
    # Assigning a type to the variable 'stypy_return_type' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'stypy_return_type', values_301)
    
    # ################# End of 'eliminate(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'eliminate' in the type store
    # Getting the type of 'stypy_return_type' (line 52)
    stypy_return_type_302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_302)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'eliminate'
    return stypy_return_type_302

# Assigning a type to the variable 'eliminate' (line 52)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'eliminate', eliminate)

@norecursion
def parse_grid(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'parse_grid'
    module_type_store = module_type_store.open_function_context('parse_grid', 76, 0, False)
    
    # Passed parameters checking function
    parse_grid.stypy_localization = localization
    parse_grid.stypy_type_of_self = None
    parse_grid.stypy_type_store = module_type_store
    parse_grid.stypy_function_name = 'parse_grid'
    parse_grid.stypy_param_names_list = ['grid']
    parse_grid.stypy_varargs_param_name = None
    parse_grid.stypy_kwargs_param_name = None
    parse_grid.stypy_call_defaults = defaults
    parse_grid.stypy_call_varargs = varargs
    parse_grid.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'parse_grid', ['grid'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'parse_grid', localization, ['grid'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'parse_grid(...)' code ##################

    str_303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 4), 'str', 'Given a string of 81 digits (or .0-), return a dict of {cell:values}')
    
    # Assigning a ListComp to a Name (line 78):
    
    # Assigning a ListComp to a Name (line 78):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'grid' (line 78)
    grid_308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 23), 'grid')
    comprehension_309 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 12), grid_308)
    # Assigning a type to the variable 'c' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'c', comprehension_309)
    
    # Getting the type of 'c' (line 78)
    c_305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 31), 'c')
    str_306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 36), 'str', '0.-123456789')
    # Applying the binary operator 'in' (line 78)
    result_contains_307 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 31), 'in', c_305, str_306)
    
    # Getting the type of 'c' (line 78)
    c_304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'c')
    list_310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 12), list_310, c_304)
    # Assigning a type to the variable 'grid' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'grid', list_310)
    
    # Assigning a Call to a Name (line 79):
    
    # Assigning a Call to a Name (line 79):
    
    # Call to dict(...): (line 79)
    # Processing the call arguments (line 79)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'squares' (line 79)
    squares_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 40), 'squares', False)
    comprehension_316 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 19), squares_315)
    # Assigning a type to the variable 's' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 19), 's', comprehension_316)
    
    # Obtaining an instance of the builtin type 'tuple' (line 79)
    tuple_312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 79)
    # Adding element type (line 79)
    # Getting the type of 's' (line 79)
    s_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 20), 's', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 20), tuple_312, s_313)
    # Adding element type (line 79)
    # Getting the type of 'digits' (line 79)
    digits_314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 23), 'digits', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 20), tuple_312, digits_314)
    
    list_317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 19), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 19), list_317, tuple_312)
    # Processing the call keyword arguments (line 79)
    kwargs_318 = {}
    # Getting the type of 'dict' (line 79)
    dict_311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 13), 'dict', False)
    # Calling dict(args, kwargs) (line 79)
    dict_call_result_319 = invoke(stypy.reporting.localization.Localization(__file__, 79, 13), dict_311, *[list_317], **kwargs_318)
    
    # Assigning a type to the variable 'values' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'values', dict_call_result_319)
    
    
    # Call to zip(...): (line 80)
    # Processing the call arguments (line 80)
    # Getting the type of 'squares' (line 80)
    squares_321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 20), 'squares', False)
    # Getting the type of 'grid' (line 80)
    grid_322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 29), 'grid', False)
    # Processing the call keyword arguments (line 80)
    kwargs_323 = {}
    # Getting the type of 'zip' (line 80)
    zip_320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 16), 'zip', False)
    # Calling zip(args, kwargs) (line 80)
    zip_call_result_324 = invoke(stypy.reporting.localization.Localization(__file__, 80, 16), zip_320, *[squares_321, grid_322], **kwargs_323)
    
    # Assigning a type to the variable 'zip_call_result_324' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'zip_call_result_324', zip_call_result_324)
    # Testing if the for loop is going to be iterated (line 80)
    # Testing the type of a for loop iterable (line 80)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 80, 4), zip_call_result_324)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 80, 4), zip_call_result_324):
        # Getting the type of the for loop variable (line 80)
        for_loop_var_325 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 80, 4), zip_call_result_324)
        # Assigning a type to the variable 's' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 's', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 4), for_loop_var_325, 2, 0))
        # Assigning a type to the variable 'd' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'd', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 4), for_loop_var_325, 2, 1))
        # SSA begins for a for statement (line 80)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Evaluating a boolean operation
        
        # Getting the type of 'd' (line 81)
        d_326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 11), 'd')
        # Getting the type of 'digits' (line 81)
        digits_327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'digits')
        # Applying the binary operator 'in' (line 81)
        result_contains_328 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 11), 'in', d_326, digits_327)
        
        
        
        # Call to assign(...): (line 81)
        # Processing the call arguments (line 81)
        # Getting the type of 'values' (line 81)
        values_330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 38), 'values', False)
        # Getting the type of 's' (line 81)
        s_331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 46), 's', False)
        # Getting the type of 'd' (line 81)
        d_332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 49), 'd', False)
        # Processing the call keyword arguments (line 81)
        kwargs_333 = {}
        # Getting the type of 'assign' (line 81)
        assign_329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 31), 'assign', False)
        # Calling assign(args, kwargs) (line 81)
        assign_call_result_334 = invoke(stypy.reporting.localization.Localization(__file__, 81, 31), assign_329, *[values_330, s_331, d_332], **kwargs_333)
        
        # Applying the 'not' unary operator (line 81)
        result_not__335 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 27), 'not', assign_call_result_334)
        
        # Applying the binary operator 'and' (line 81)
        result_and_keyword_336 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 11), 'and', result_contains_328, result_not__335)
        
        # Testing if the type of an if condition is none (line 81)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 81, 8), result_and_keyword_336):
            pass
        else:
            
            # Testing the type of an if condition (line 81)
            if_condition_337 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 8), result_and_keyword_336)
            # Assigning a type to the variable 'if_condition_337' (line 81)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'if_condition_337', if_condition_337)
            # SSA begins for if statement (line 81)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'None' (line 82)
            None_338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 19), 'None')
            # Assigning a type to the variable 'stypy_return_type' (line 82)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'stypy_return_type', None_338)
            # SSA join for if statement (line 81)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'values' (line 83)
    values_339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 11), 'values')
    # Assigning a type to the variable 'stypy_return_type' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'stypy_return_type', values_339)
    
    # ################# End of 'parse_grid(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'parse_grid' in the type store
    # Getting the type of 'stypy_return_type' (line 76)
    stypy_return_type_340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_340)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'parse_grid'
    return stypy_return_type_340

# Assigning a type to the variable 'parse_grid' (line 76)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'parse_grid', parse_grid)

@norecursion
def solve_file(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'solve_file'
    module_type_store = module_type_store.open_function_context('solve_file', 86, 0, False)
    
    # Passed parameters checking function
    solve_file.stypy_localization = localization
    solve_file.stypy_type_of_self = None
    solve_file.stypy_type_store = module_type_store
    solve_file.stypy_function_name = 'solve_file'
    solve_file.stypy_param_names_list = ['filename', 'sep', 'action']
    solve_file.stypy_varargs_param_name = None
    solve_file.stypy_kwargs_param_name = None
    solve_file.stypy_call_defaults = defaults
    solve_file.stypy_call_varargs = varargs
    solve_file.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'solve_file', ['filename', 'sep', 'action'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'solve_file', localization, ['filename', 'sep', 'action'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'solve_file(...)' code ##################

    str_341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 4), 'str', 'Parse a file into a sequence of 81-char descriptions and solve them.')
    
    # Assigning a ListComp to a Name (line 88):
    
    # Assigning a ListComp to a Name (line 88):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to split(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'sep' (line 89)
    sep_363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 63), 'sep', False)
    # Processing the call keyword arguments (line 89)
    kwargs_364 = {}
    
    # Call to strip(...): (line 89)
    # Processing the call keyword arguments (line 89)
    kwargs_360 = {}
    
    # Call to read(...): (line 89)
    # Processing the call keyword arguments (line 89)
    kwargs_357 = {}
    
    # Call to file(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'filename' (line 89)
    filename_353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 32), 'filename', False)
    # Processing the call keyword arguments (line 89)
    kwargs_354 = {}
    # Getting the type of 'file' (line 89)
    file_352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 27), 'file', False)
    # Calling file(args, kwargs) (line 89)
    file_call_result_355 = invoke(stypy.reporting.localization.Localization(__file__, 89, 27), file_352, *[filename_353], **kwargs_354)
    
    # Obtaining the member 'read' of a type (line 89)
    read_356 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 27), file_call_result_355, 'read')
    # Calling read(args, kwargs) (line 89)
    read_call_result_358 = invoke(stypy.reporting.localization.Localization(__file__, 89, 27), read_356, *[], **kwargs_357)
    
    # Obtaining the member 'strip' of a type (line 89)
    strip_359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 27), read_call_result_358, 'strip')
    # Calling strip(args, kwargs) (line 89)
    strip_call_result_361 = invoke(stypy.reporting.localization.Localization(__file__, 89, 27), strip_359, *[], **kwargs_360)
    
    # Obtaining the member 'split' of a type (line 89)
    split_362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 27), strip_call_result_361, 'split')
    # Calling split(args, kwargs) (line 89)
    split_call_result_365 = invoke(stypy.reporting.localization.Localization(__file__, 89, 27), split_362, *[sep_363], **kwargs_364)
    
    comprehension_366 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 15), split_call_result_365)
    # Assigning a type to the variable 'grid' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 15), 'grid', comprehension_366)
    
    # Call to action(...): (line 88)
    # Processing the call arguments (line 88)
    
    # Call to search(...): (line 88)
    # Processing the call arguments (line 88)
    
    # Call to parse_grid(...): (line 88)
    # Processing the call arguments (line 88)
    # Getting the type of 'grid' (line 88)
    grid_345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 40), 'grid', False)
    # Processing the call keyword arguments (line 88)
    kwargs_346 = {}
    # Getting the type of 'parse_grid' (line 88)
    parse_grid_344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 29), 'parse_grid', False)
    # Calling parse_grid(args, kwargs) (line 88)
    parse_grid_call_result_347 = invoke(stypy.reporting.localization.Localization(__file__, 88, 29), parse_grid_344, *[grid_345], **kwargs_346)
    
    # Processing the call keyword arguments (line 88)
    kwargs_348 = {}
    # Getting the type of 'search' (line 88)
    search_343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 22), 'search', False)
    # Calling search(args, kwargs) (line 88)
    search_call_result_349 = invoke(stypy.reporting.localization.Localization(__file__, 88, 22), search_343, *[parse_grid_call_result_347], **kwargs_348)
    
    # Processing the call keyword arguments (line 88)
    kwargs_350 = {}
    # Getting the type of 'action' (line 88)
    action_342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 15), 'action', False)
    # Calling action(args, kwargs) (line 88)
    action_call_result_351 = invoke(stypy.reporting.localization.Localization(__file__, 88, 15), action_342, *[search_call_result_349], **kwargs_350)
    
    list_367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 15), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 15), list_367, action_call_result_351)
    # Assigning a type to the variable 'results' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'results', list_367)
    # Getting the type of 'results' (line 92)
    results_368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 11), 'results')
    # Assigning a type to the variable 'stypy_return_type' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'stypy_return_type', results_368)
    
    # ################# End of 'solve_file(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'solve_file' in the type store
    # Getting the type of 'stypy_return_type' (line 86)
    stypy_return_type_369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_369)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'solve_file'
    return stypy_return_type_369

# Assigning a type to the variable 'solve_file' (line 86)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'solve_file', solve_file)

@norecursion
def printboard(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'printboard'
    module_type_store = module_type_store.open_function_context('printboard', 95, 0, False)
    
    # Passed parameters checking function
    printboard.stypy_localization = localization
    printboard.stypy_type_of_self = None
    printboard.stypy_type_store = module_type_store
    printboard.stypy_function_name = 'printboard'
    printboard.stypy_param_names_list = ['values']
    printboard.stypy_varargs_param_name = None
    printboard.stypy_kwargs_param_name = None
    printboard.stypy_call_defaults = defaults
    printboard.stypy_call_varargs = varargs
    printboard.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'printboard', ['values'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'printboard', localization, ['values'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'printboard(...)' code ##################

    str_370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 4), 'str', 'Used for debugging.')
    
    # Assigning a BinOp to a Name (line 97):
    
    # Assigning a BinOp to a Name (line 97):
    int_371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 12), 'int')
    
    # Call to max(...): (line 97)
    # Processing the call arguments (line 97)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'squares' (line 97)
    squares_380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 45), 'squares', False)
    comprehension_381 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 21), squares_380)
    # Assigning a type to the variable 's' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 21), 's', comprehension_381)
    
    # Call to len(...): (line 97)
    # Processing the call arguments (line 97)
    
    # Obtaining the type of the subscript
    # Getting the type of 's' (line 97)
    s_374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 32), 's', False)
    # Getting the type of 'values' (line 97)
    values_375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 25), 'values', False)
    # Obtaining the member '__getitem__' of a type (line 97)
    getitem___376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 25), values_375, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 97)
    subscript_call_result_377 = invoke(stypy.reporting.localization.Localization(__file__, 97, 25), getitem___376, s_374)
    
    # Processing the call keyword arguments (line 97)
    kwargs_378 = {}
    # Getting the type of 'len' (line 97)
    len_373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 21), 'len', False)
    # Calling len(args, kwargs) (line 97)
    len_call_result_379 = invoke(stypy.reporting.localization.Localization(__file__, 97, 21), len_373, *[subscript_call_result_377], **kwargs_378)
    
    list_382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 21), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 21), list_382, len_call_result_379)
    # Processing the call keyword arguments (line 97)
    kwargs_383 = {}
    # Getting the type of 'max' (line 97)
    max_372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 16), 'max', False)
    # Calling max(args, kwargs) (line 97)
    max_call_result_384 = invoke(stypy.reporting.localization.Localization(__file__, 97, 16), max_372, *[list_382], **kwargs_383)
    
    # Applying the binary operator '+' (line 97)
    result_add_385 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 12), '+', int_371, max_call_result_384)
    
    # Assigning a type to the variable 'width' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'width', result_add_385)
    
    # Assigning a BinOp to a Name (line 98):
    
    # Assigning a BinOp to a Name (line 98):
    str_386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 11), 'str', '\n')
    
    # Call to join(...): (line 98)
    # Processing the call arguments (line 98)
    
    # Obtaining an instance of the builtin type 'list' (line 98)
    list_389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 27), 'list')
    # Adding type elements to the builtin type 'list' instance (line 98)
    # Adding element type (line 98)
    str_390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 28), 'str', '-')
    # Getting the type of 'width' (line 98)
    width_391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 35), 'width', False)
    int_392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 43), 'int')
    # Applying the binary operator '*' (line 98)
    result_mul_393 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 35), '*', width_391, int_392)
    
    # Applying the binary operator '*' (line 98)
    result_mul_394 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 28), '*', str_390, result_mul_393)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 27), list_389, result_mul_394)
    
    int_395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 49), 'int')
    # Applying the binary operator '*' (line 98)
    result_mul_396 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 27), '*', list_389, int_395)
    
    # Processing the call keyword arguments (line 98)
    kwargs_397 = {}
    str_387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 18), 'str', '+')
    # Obtaining the member 'join' of a type (line 98)
    join_388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 18), str_387, 'join')
    # Calling join(args, kwargs) (line 98)
    join_call_result_398 = invoke(stypy.reporting.localization.Localization(__file__, 98, 18), join_388, *[result_mul_396], **kwargs_397)
    
    # Applying the binary operator '+' (line 98)
    result_add_399 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 11), '+', str_386, join_call_result_398)
    
    # Assigning a type to the variable 'line' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'line', result_add_399)
    
    # Getting the type of 'rows' (line 99)
    rows_400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 13), 'rows')
    # Assigning a type to the variable 'rows_400' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'rows_400', rows_400)
    # Testing if the for loop is going to be iterated (line 99)
    # Testing the type of a for loop iterable (line 99)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 99, 4), rows_400)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 99, 4), rows_400):
        # Getting the type of the for loop variable (line 99)
        for_loop_var_401 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 99, 4), rows_400)
        # Assigning a type to the variable 'r' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'r', for_loop_var_401)
        # SSA begins for a for statement (line 99)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        pass
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'values' (line 104)
    values_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 11), 'values')
    # Assigning a type to the variable 'stypy_return_type' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'stypy_return_type', values_402)
    
    # ################# End of 'printboard(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'printboard' in the type store
    # Getting the type of 'stypy_return_type' (line 95)
    stypy_return_type_403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_403)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'printboard'
    return stypy_return_type_403

# Assigning a type to the variable 'printboard' (line 95)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 0), 'printboard', printboard)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 107, 0))

# 'import os' statement (line 107)
import os

import_module(stypy.reporting.localization.Localization(__file__, 107, 0), 'os', os, module_type_store)


@norecursion
def Relative(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Relative'
    module_type_store = module_type_store.open_function_context('Relative', 110, 0, False)
    
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

    
    # Call to join(...): (line 111)
    # Processing the call arguments (line 111)
    
    # Call to dirname(...): (line 111)
    # Processing the call arguments (line 111)
    # Getting the type of '__file__' (line 111)
    file___410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 40), '__file__', False)
    # Processing the call keyword arguments (line 111)
    kwargs_411 = {}
    # Getting the type of 'os' (line 111)
    os_407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 24), 'os', False)
    # Obtaining the member 'path' of a type (line 111)
    path_408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 24), os_407, 'path')
    # Obtaining the member 'dirname' of a type (line 111)
    dirname_409 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 24), path_408, 'dirname')
    # Calling dirname(args, kwargs) (line 111)
    dirname_call_result_412 = invoke(stypy.reporting.localization.Localization(__file__, 111, 24), dirname_409, *[file___410], **kwargs_411)
    
    # Getting the type of 'path' (line 111)
    path_413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 51), 'path', False)
    # Processing the call keyword arguments (line 111)
    kwargs_414 = {}
    # Getting the type of 'os' (line 111)
    os_404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 111)
    path_405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 11), os_404, 'path')
    # Obtaining the member 'join' of a type (line 111)
    join_406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 11), path_405, 'join')
    # Calling join(args, kwargs) (line 111)
    join_call_result_415 = invoke(stypy.reporting.localization.Localization(__file__, 111, 11), join_406, *[dirname_call_result_412, path_413], **kwargs_414)
    
    # Assigning a type to the variable 'stypy_return_type' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'stypy_return_type', join_call_result_415)
    
    # ################# End of 'Relative(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Relative' in the type store
    # Getting the type of 'stypy_return_type' (line 110)
    stypy_return_type_416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_416)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Relative'
    return stypy_return_type_416

# Assigning a type to the variable 'Relative' (line 110)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 0), 'Relative', Relative)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 114, 0, False)
    
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

    
    # Call to solve_file(...): (line 115)
    # Processing the call arguments (line 115)
    
    # Call to Relative(...): (line 115)
    # Processing the call arguments (line 115)
    str_419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 24), 'str', 'testdata/top95.txt')
    # Processing the call keyword arguments (line 115)
    kwargs_420 = {}
    # Getting the type of 'Relative' (line 115)
    Relative_418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 15), 'Relative', False)
    # Calling Relative(args, kwargs) (line 115)
    Relative_call_result_421 = invoke(stypy.reporting.localization.Localization(__file__, 115, 15), Relative_418, *[str_419], **kwargs_420)
    
    str_422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 47), 'str', '\n')
    # Getting the type of 'printboard' (line 115)
    printboard_423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 53), 'printboard', False)
    # Processing the call keyword arguments (line 115)
    kwargs_424 = {}
    # Getting the type of 'solve_file' (line 115)
    solve_file_417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'solve_file', False)
    # Calling solve_file(args, kwargs) (line 115)
    solve_file_call_result_425 = invoke(stypy.reporting.localization.Localization(__file__, 115, 4), solve_file_417, *[Relative_call_result_421, str_422, printboard_423], **kwargs_424)
    
    # Getting the type of 'True' (line 116)
    True_426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'stypy_return_type', True_426)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 114)
    stypy_return_type_427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_427)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_427

# Assigning a type to the variable 'run' (line 114)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 0), 'run', run)

# Call to run(...): (line 124)
# Processing the call keyword arguments (line 124)
kwargs_429 = {}
# Getting the type of 'run' (line 124)
run_428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 0), 'run', False)
# Calling run(args, kwargs) (line 124)
run_call_result_430 = invoke(stypy.reporting.localization.Localization(__file__, 124, 0), run_428, *[], **kwargs_429)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
