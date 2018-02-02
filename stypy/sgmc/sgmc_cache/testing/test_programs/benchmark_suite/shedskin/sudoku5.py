
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Author: Ali Assaf <ali.assaf.mail@gmail.com>
3: Copyright: (C) 2010 Ali Assaf
4: License: GNU General Public License <http://www.gnu.org/licenses/>
5: '''
6: 
7: from itertools import product
8: 
9: 
10: def solve_sudoku(size, grid):
11:     '''An efficient Sudoku solver using Algorithm X.'''
12:     R, C = size
13:     N = R * C
14:     X1 = ([("rc", rc) for rc in product(xrange(N), xrange(N))] +
15:           [("rn", rn) for rn in product(xrange(N), xrange(1, N + 1))] +
16:           [("cn", cn) for cn in product(xrange(N), xrange(1, N + 1))] +
17:           [("bn", bn) for bn in product(xrange(N), xrange(1, N + 1))])
18:     Y = dict()
19:     for r, c, n in product(xrange(N), xrange(N), xrange(1, N + 1)):
20:         b = (r // R) * R + (c // C)  # Box number
21:         Y[(r, c, n)] = [
22:             ("rc", (r, c)),
23:             ("rn", (r, n)),
24:             ("cn", (c, n)),
25:             ("bn", (b, n))]
26:     X, Y = exact_cover(X1, Y)
27:     for i, row in enumerate(grid):
28:         for j, n in enumerate(row):
29:             if n:
30:                 select(X, Y, (i, j, n))
31:     for solution in solve(X, Y, []):
32:         for (r, c, n) in solution:
33:             grid[r][c] = n
34:         yield grid
35: 
36: 
37: def exact_cover(X1, Y):
38:     X = dict((j, set()) for j in X1)
39:     for i, row in Y.iteritems():
40:         for j in row:
41:             X[j].add(i)
42:     return X, Y
43: 
44: 
45: def solve(X, Y, solution):
46:     if not X:
47:         yield list(solution)
48:     else:
49:         # c = min(X, key=lambda c: len(X[c])) # shedskin doesn't support closures!
50:         c = min([(len(X[c]), c) for c in X])[1]
51:         for r in list(X[c]):
52:             solution.append(r)
53:             cols = select(X, Y, r)
54:             for solution in solve(X, Y, solution):
55:                 yield solution
56:             deselect(X, Y, r, cols)
57:             solution.pop()
58: 
59: 
60: def select(X, Y, r):
61:     cols = []
62:     for j in Y[r]:
63:         for i in X[j]:
64:             for k in Y[i]:
65:                 if k != j:
66:                     X[k].remove(i)
67:         cols.append(X.pop(j))
68:     return cols
69: 
70: 
71: def deselect(X, Y, r, cols):
72:     for j in reversed(Y[r]):
73:         X[j] = cols.pop()
74:         for i in X[j]:
75:             for k in Y[i]:
76:                 if k != j:
77:                     X[k].add(i)
78: 
79: 
80: def main():
81:     grid = [
82:         [5, 3, 0, 0, 7, 0, 0, 0, 0],
83:         [6, 0, 0, 1, 9, 5, 0, 0, 0],
84:         [0, 9, 8, 0, 0, 0, 0, 6, 0],
85:         [8, 0, 0, 0, 6, 0, 0, 0, 3],
86:         [4, 0, 0, 8, 0, 3, 0, 0, 1],
87:         [7, 0, 0, 0, 2, 0, 0, 0, 6],
88:         [0, 6, 0, 0, 0, 0, 2, 8, 0],
89:         [0, 0, 0, 4, 1, 9, 0, 0, 5],
90:         [0, 0, 0, 0, 8, 0, 0, 7, 9],
91:     ]
92:     for solution in solve_sudoku((3, 3), grid):
93:         a = "\n".join(str(s) for s in solution)
94:         # pass#print "\n".join(str(s) for s in solution)
95: 
96: 
97: def run():
98:     for i in range(100):
99:         main()
100:     return True
101: 
102: 
103: run()
104: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, (-1)), 'str', '\nAuthor: Ali Assaf <ali.assaf.mail@gmail.com>\nCopyright: (C) 2010 Ali Assaf\nLicense: GNU General Public License <http://www.gnu.org/licenses/>\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from itertools import product' statement (line 7)
try:
    from itertools import product

except:
    product = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'itertools', None, module_type_store, ['product'], [product])


@norecursion
def solve_sudoku(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'solve_sudoku'
    module_type_store = module_type_store.open_function_context('solve_sudoku', 10, 0, False)
    
    # Passed parameters checking function
    solve_sudoku.stypy_localization = localization
    solve_sudoku.stypy_type_of_self = None
    solve_sudoku.stypy_type_store = module_type_store
    solve_sudoku.stypy_function_name = 'solve_sudoku'
    solve_sudoku.stypy_param_names_list = ['size', 'grid']
    solve_sudoku.stypy_varargs_param_name = None
    solve_sudoku.stypy_kwargs_param_name = None
    solve_sudoku.stypy_call_defaults = defaults
    solve_sudoku.stypy_call_varargs = varargs
    solve_sudoku.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'solve_sudoku', ['size', 'grid'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'solve_sudoku', localization, ['size', 'grid'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'solve_sudoku(...)' code ##################

    str_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 4), 'str', 'An efficient Sudoku solver using Algorithm X.')
    
    # Assigning a Name to a Tuple (line 12):
    
    # Assigning a Subscript to a Name (line 12):
    
    # Obtaining the type of the subscript
    int_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 4), 'int')
    # Getting the type of 'size' (line 12)
    size_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 11), 'size')
    # Obtaining the member '__getitem__' of a type (line 12)
    getitem___10 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 4), size_9, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 12)
    subscript_call_result_11 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), getitem___10, int_8)
    
    # Assigning a type to the variable 'tuple_var_assignment_1' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'tuple_var_assignment_1', subscript_call_result_11)
    
    # Assigning a Subscript to a Name (line 12):
    
    # Obtaining the type of the subscript
    int_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 4), 'int')
    # Getting the type of 'size' (line 12)
    size_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 11), 'size')
    # Obtaining the member '__getitem__' of a type (line 12)
    getitem___14 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 4), size_13, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 12)
    subscript_call_result_15 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), getitem___14, int_12)
    
    # Assigning a type to the variable 'tuple_var_assignment_2' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'tuple_var_assignment_2', subscript_call_result_15)
    
    # Assigning a Name to a Name (line 12):
    # Getting the type of 'tuple_var_assignment_1' (line 12)
    tuple_var_assignment_1_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'tuple_var_assignment_1')
    # Assigning a type to the variable 'R' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'R', tuple_var_assignment_1_16)
    
    # Assigning a Name to a Name (line 12):
    # Getting the type of 'tuple_var_assignment_2' (line 12)
    tuple_var_assignment_2_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'tuple_var_assignment_2')
    # Assigning a type to the variable 'C' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 7), 'C', tuple_var_assignment_2_17)
    
    # Assigning a BinOp to a Name (line 13):
    
    # Assigning a BinOp to a Name (line 13):
    # Getting the type of 'R' (line 13)
    R_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'R')
    # Getting the type of 'C' (line 13)
    C_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 12), 'C')
    # Applying the binary operator '*' (line 13)
    result_mul_20 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 8), '*', R_18, C_19)
    
    # Assigning a type to the variable 'N' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'N', result_mul_20)
    
    # Assigning a BinOp to a Name (line 14):
    
    # Assigning a BinOp to a Name (line 14):
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to product(...): (line 14)
    # Processing the call arguments (line 14)
    
    # Call to xrange(...): (line 14)
    # Processing the call arguments (line 14)
    # Getting the type of 'N' (line 14)
    N_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 47), 'N', False)
    # Processing the call keyword arguments (line 14)
    kwargs_27 = {}
    # Getting the type of 'xrange' (line 14)
    xrange_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 40), 'xrange', False)
    # Calling xrange(args, kwargs) (line 14)
    xrange_call_result_28 = invoke(stypy.reporting.localization.Localization(__file__, 14, 40), xrange_25, *[N_26], **kwargs_27)
    
    
    # Call to xrange(...): (line 14)
    # Processing the call arguments (line 14)
    # Getting the type of 'N' (line 14)
    N_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 58), 'N', False)
    # Processing the call keyword arguments (line 14)
    kwargs_31 = {}
    # Getting the type of 'xrange' (line 14)
    xrange_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 51), 'xrange', False)
    # Calling xrange(args, kwargs) (line 14)
    xrange_call_result_32 = invoke(stypy.reporting.localization.Localization(__file__, 14, 51), xrange_29, *[N_30], **kwargs_31)
    
    # Processing the call keyword arguments (line 14)
    kwargs_33 = {}
    # Getting the type of 'product' (line 14)
    product_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 32), 'product', False)
    # Calling product(args, kwargs) (line 14)
    product_call_result_34 = invoke(stypy.reporting.localization.Localization(__file__, 14, 32), product_24, *[xrange_call_result_28, xrange_call_result_32], **kwargs_33)
    
    comprehension_35 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 11), product_call_result_34)
    # Assigning a type to the variable 'rc' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 11), 'rc', comprehension_35)
    
    # Obtaining an instance of the builtin type 'tuple' (line 14)
    tuple_21 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 14)
    # Adding element type (line 14)
    str_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 12), 'str', 'rc')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), tuple_21, str_22)
    # Adding element type (line 14)
    # Getting the type of 'rc' (line 14)
    rc_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 18), 'rc')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 12), tuple_21, rc_23)
    
    list_36 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 11), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 11), list_36, tuple_21)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to product(...): (line 15)
    # Processing the call arguments (line 15)
    
    # Call to xrange(...): (line 15)
    # Processing the call arguments (line 15)
    # Getting the type of 'N' (line 15)
    N_42 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 47), 'N', False)
    # Processing the call keyword arguments (line 15)
    kwargs_43 = {}
    # Getting the type of 'xrange' (line 15)
    xrange_41 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 40), 'xrange', False)
    # Calling xrange(args, kwargs) (line 15)
    xrange_call_result_44 = invoke(stypy.reporting.localization.Localization(__file__, 15, 40), xrange_41, *[N_42], **kwargs_43)
    
    
    # Call to xrange(...): (line 15)
    # Processing the call arguments (line 15)
    int_46 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 58), 'int')
    # Getting the type of 'N' (line 15)
    N_47 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 61), 'N', False)
    int_48 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 65), 'int')
    # Applying the binary operator '+' (line 15)
    result_add_49 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 61), '+', N_47, int_48)
    
    # Processing the call keyword arguments (line 15)
    kwargs_50 = {}
    # Getting the type of 'xrange' (line 15)
    xrange_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 51), 'xrange', False)
    # Calling xrange(args, kwargs) (line 15)
    xrange_call_result_51 = invoke(stypy.reporting.localization.Localization(__file__, 15, 51), xrange_45, *[int_46, result_add_49], **kwargs_50)
    
    # Processing the call keyword arguments (line 15)
    kwargs_52 = {}
    # Getting the type of 'product' (line 15)
    product_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 32), 'product', False)
    # Calling product(args, kwargs) (line 15)
    product_call_result_53 = invoke(stypy.reporting.localization.Localization(__file__, 15, 32), product_40, *[xrange_call_result_44, xrange_call_result_51], **kwargs_52)
    
    comprehension_54 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 11), product_call_result_53)
    # Assigning a type to the variable 'rn' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 11), 'rn', comprehension_54)
    
    # Obtaining an instance of the builtin type 'tuple' (line 15)
    tuple_37 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 15)
    # Adding element type (line 15)
    str_38 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 12), 'str', 'rn')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 12), tuple_37, str_38)
    # Adding element type (line 15)
    # Getting the type of 'rn' (line 15)
    rn_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 18), 'rn')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 12), tuple_37, rn_39)
    
    list_55 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 11), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 11), list_55, tuple_37)
    # Applying the binary operator '+' (line 14)
    result_add_56 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 10), '+', list_36, list_55)
    
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to product(...): (line 16)
    # Processing the call arguments (line 16)
    
    # Call to xrange(...): (line 16)
    # Processing the call arguments (line 16)
    # Getting the type of 'N' (line 16)
    N_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 47), 'N', False)
    # Processing the call keyword arguments (line 16)
    kwargs_63 = {}
    # Getting the type of 'xrange' (line 16)
    xrange_61 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 40), 'xrange', False)
    # Calling xrange(args, kwargs) (line 16)
    xrange_call_result_64 = invoke(stypy.reporting.localization.Localization(__file__, 16, 40), xrange_61, *[N_62], **kwargs_63)
    
    
    # Call to xrange(...): (line 16)
    # Processing the call arguments (line 16)
    int_66 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 58), 'int')
    # Getting the type of 'N' (line 16)
    N_67 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 61), 'N', False)
    int_68 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 65), 'int')
    # Applying the binary operator '+' (line 16)
    result_add_69 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 61), '+', N_67, int_68)
    
    # Processing the call keyword arguments (line 16)
    kwargs_70 = {}
    # Getting the type of 'xrange' (line 16)
    xrange_65 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 51), 'xrange', False)
    # Calling xrange(args, kwargs) (line 16)
    xrange_call_result_71 = invoke(stypy.reporting.localization.Localization(__file__, 16, 51), xrange_65, *[int_66, result_add_69], **kwargs_70)
    
    # Processing the call keyword arguments (line 16)
    kwargs_72 = {}
    # Getting the type of 'product' (line 16)
    product_60 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 32), 'product', False)
    # Calling product(args, kwargs) (line 16)
    product_call_result_73 = invoke(stypy.reporting.localization.Localization(__file__, 16, 32), product_60, *[xrange_call_result_64, xrange_call_result_71], **kwargs_72)
    
    comprehension_74 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 11), product_call_result_73)
    # Assigning a type to the variable 'cn' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 11), 'cn', comprehension_74)
    
    # Obtaining an instance of the builtin type 'tuple' (line 16)
    tuple_57 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 16)
    # Adding element type (line 16)
    str_58 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 12), 'str', 'cn')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 12), tuple_57, str_58)
    # Adding element type (line 16)
    # Getting the type of 'cn' (line 16)
    cn_59 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 18), 'cn')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 12), tuple_57, cn_59)
    
    list_75 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 11), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 11), list_75, tuple_57)
    # Applying the binary operator '+' (line 15)
    result_add_76 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 70), '+', result_add_56, list_75)
    
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to product(...): (line 17)
    # Processing the call arguments (line 17)
    
    # Call to xrange(...): (line 17)
    # Processing the call arguments (line 17)
    # Getting the type of 'N' (line 17)
    N_82 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 47), 'N', False)
    # Processing the call keyword arguments (line 17)
    kwargs_83 = {}
    # Getting the type of 'xrange' (line 17)
    xrange_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 40), 'xrange', False)
    # Calling xrange(args, kwargs) (line 17)
    xrange_call_result_84 = invoke(stypy.reporting.localization.Localization(__file__, 17, 40), xrange_81, *[N_82], **kwargs_83)
    
    
    # Call to xrange(...): (line 17)
    # Processing the call arguments (line 17)
    int_86 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 58), 'int')
    # Getting the type of 'N' (line 17)
    N_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 61), 'N', False)
    int_88 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 65), 'int')
    # Applying the binary operator '+' (line 17)
    result_add_89 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 61), '+', N_87, int_88)
    
    # Processing the call keyword arguments (line 17)
    kwargs_90 = {}
    # Getting the type of 'xrange' (line 17)
    xrange_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 51), 'xrange', False)
    # Calling xrange(args, kwargs) (line 17)
    xrange_call_result_91 = invoke(stypy.reporting.localization.Localization(__file__, 17, 51), xrange_85, *[int_86, result_add_89], **kwargs_90)
    
    # Processing the call keyword arguments (line 17)
    kwargs_92 = {}
    # Getting the type of 'product' (line 17)
    product_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 32), 'product', False)
    # Calling product(args, kwargs) (line 17)
    product_call_result_93 = invoke(stypy.reporting.localization.Localization(__file__, 17, 32), product_80, *[xrange_call_result_84, xrange_call_result_91], **kwargs_92)
    
    comprehension_94 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 11), product_call_result_93)
    # Assigning a type to the variable 'bn' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 11), 'bn', comprehension_94)
    
    # Obtaining an instance of the builtin type 'tuple' (line 17)
    tuple_77 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 17)
    # Adding element type (line 17)
    str_78 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 12), 'str', 'bn')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 12), tuple_77, str_78)
    # Adding element type (line 17)
    # Getting the type of 'bn' (line 17)
    bn_79 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 18), 'bn')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 12), tuple_77, bn_79)
    
    list_95 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 11), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 11), list_95, tuple_77)
    # Applying the binary operator '+' (line 16)
    result_add_96 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 70), '+', result_add_76, list_95)
    
    # Assigning a type to the variable 'X1' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'X1', result_add_96)
    
    # Assigning a Call to a Name (line 18):
    
    # Assigning a Call to a Name (line 18):
    
    # Call to dict(...): (line 18)
    # Processing the call keyword arguments (line 18)
    kwargs_98 = {}
    # Getting the type of 'dict' (line 18)
    dict_97 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'dict', False)
    # Calling dict(args, kwargs) (line 18)
    dict_call_result_99 = invoke(stypy.reporting.localization.Localization(__file__, 18, 8), dict_97, *[], **kwargs_98)
    
    # Assigning a type to the variable 'Y' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'Y', dict_call_result_99)
    
    
    # Call to product(...): (line 19)
    # Processing the call arguments (line 19)
    
    # Call to xrange(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'N' (line 19)
    N_102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 34), 'N', False)
    # Processing the call keyword arguments (line 19)
    kwargs_103 = {}
    # Getting the type of 'xrange' (line 19)
    xrange_101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 27), 'xrange', False)
    # Calling xrange(args, kwargs) (line 19)
    xrange_call_result_104 = invoke(stypy.reporting.localization.Localization(__file__, 19, 27), xrange_101, *[N_102], **kwargs_103)
    
    
    # Call to xrange(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'N' (line 19)
    N_106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 45), 'N', False)
    # Processing the call keyword arguments (line 19)
    kwargs_107 = {}
    # Getting the type of 'xrange' (line 19)
    xrange_105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 38), 'xrange', False)
    # Calling xrange(args, kwargs) (line 19)
    xrange_call_result_108 = invoke(stypy.reporting.localization.Localization(__file__, 19, 38), xrange_105, *[N_106], **kwargs_107)
    
    
    # Call to xrange(...): (line 19)
    # Processing the call arguments (line 19)
    int_110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 56), 'int')
    # Getting the type of 'N' (line 19)
    N_111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 59), 'N', False)
    int_112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 63), 'int')
    # Applying the binary operator '+' (line 19)
    result_add_113 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 59), '+', N_111, int_112)
    
    # Processing the call keyword arguments (line 19)
    kwargs_114 = {}
    # Getting the type of 'xrange' (line 19)
    xrange_109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 49), 'xrange', False)
    # Calling xrange(args, kwargs) (line 19)
    xrange_call_result_115 = invoke(stypy.reporting.localization.Localization(__file__, 19, 49), xrange_109, *[int_110, result_add_113], **kwargs_114)
    
    # Processing the call keyword arguments (line 19)
    kwargs_116 = {}
    # Getting the type of 'product' (line 19)
    product_100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 19), 'product', False)
    # Calling product(args, kwargs) (line 19)
    product_call_result_117 = invoke(stypy.reporting.localization.Localization(__file__, 19, 19), product_100, *[xrange_call_result_104, xrange_call_result_108, xrange_call_result_115], **kwargs_116)
    
    # Testing if the for loop is going to be iterated (line 19)
    # Testing the type of a for loop iterable (line 19)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 19, 4), product_call_result_117)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 19, 4), product_call_result_117):
        # Getting the type of the for loop variable (line 19)
        for_loop_var_118 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 19, 4), product_call_result_117)
        # Assigning a type to the variable 'r' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'r', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 4), for_loop_var_118))
        # Assigning a type to the variable 'c' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'c', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 4), for_loop_var_118))
        # Assigning a type to the variable 'n' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'n', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 4), for_loop_var_118))
        # SSA begins for a for statement (line 19)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 20):
        
        # Assigning a BinOp to a Name (line 20):
        # Getting the type of 'r' (line 20)
        r_119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 13), 'r')
        # Getting the type of 'R' (line 20)
        R_120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 18), 'R')
        # Applying the binary operator '//' (line 20)
        result_floordiv_121 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 13), '//', r_119, R_120)
        
        # Getting the type of 'R' (line 20)
        R_122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 23), 'R')
        # Applying the binary operator '*' (line 20)
        result_mul_123 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 12), '*', result_floordiv_121, R_122)
        
        # Getting the type of 'c' (line 20)
        c_124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 28), 'c')
        # Getting the type of 'C' (line 20)
        C_125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 33), 'C')
        # Applying the binary operator '//' (line 20)
        result_floordiv_126 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 28), '//', c_124, C_125)
        
        # Applying the binary operator '+' (line 20)
        result_add_127 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 12), '+', result_mul_123, result_floordiv_126)
        
        # Assigning a type to the variable 'b' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'b', result_add_127)
        
        # Assigning a List to a Subscript (line 21):
        
        # Assigning a List to a Subscript (line 21):
        
        # Obtaining an instance of the builtin type 'list' (line 21)
        list_128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 21)
        # Adding element type (line 21)
        
        # Obtaining an instance of the builtin type 'tuple' (line 22)
        tuple_129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 22)
        # Adding element type (line 22)
        str_130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 13), 'str', 'rc')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 13), tuple_129, str_130)
        # Adding element type (line 22)
        
        # Obtaining an instance of the builtin type 'tuple' (line 22)
        tuple_131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 22)
        # Adding element type (line 22)
        # Getting the type of 'r' (line 22)
        r_132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 20), 'r')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 20), tuple_131, r_132)
        # Adding element type (line 22)
        # Getting the type of 'c' (line 22)
        c_133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 23), 'c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 20), tuple_131, c_133)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 13), tuple_129, tuple_131)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 23), list_128, tuple_129)
        # Adding element type (line 21)
        
        # Obtaining an instance of the builtin type 'tuple' (line 23)
        tuple_134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 23)
        # Adding element type (line 23)
        str_135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 13), 'str', 'rn')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 13), tuple_134, str_135)
        # Adding element type (line 23)
        
        # Obtaining an instance of the builtin type 'tuple' (line 23)
        tuple_136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 23)
        # Adding element type (line 23)
        # Getting the type of 'r' (line 23)
        r_137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 20), 'r')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 20), tuple_136, r_137)
        # Adding element type (line 23)
        # Getting the type of 'n' (line 23)
        n_138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 23), 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 20), tuple_136, n_138)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 13), tuple_134, tuple_136)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 23), list_128, tuple_134)
        # Adding element type (line 21)
        
        # Obtaining an instance of the builtin type 'tuple' (line 24)
        tuple_139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 24)
        # Adding element type (line 24)
        str_140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 13), 'str', 'cn')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 13), tuple_139, str_140)
        # Adding element type (line 24)
        
        # Obtaining an instance of the builtin type 'tuple' (line 24)
        tuple_141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 24)
        # Adding element type (line 24)
        # Getting the type of 'c' (line 24)
        c_142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 20), 'c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 20), tuple_141, c_142)
        # Adding element type (line 24)
        # Getting the type of 'n' (line 24)
        n_143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 23), 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 20), tuple_141, n_143)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 13), tuple_139, tuple_141)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 23), list_128, tuple_139)
        # Adding element type (line 21)
        
        # Obtaining an instance of the builtin type 'tuple' (line 25)
        tuple_144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 13), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 25)
        # Adding element type (line 25)
        str_145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 13), 'str', 'bn')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 13), tuple_144, str_145)
        # Adding element type (line 25)
        
        # Obtaining an instance of the builtin type 'tuple' (line 25)
        tuple_146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 25)
        # Adding element type (line 25)
        # Getting the type of 'b' (line 25)
        b_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 20), 'b')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 20), tuple_146, b_147)
        # Adding element type (line 25)
        # Getting the type of 'n' (line 25)
        n_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 23), 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 20), tuple_146, n_148)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 13), tuple_144, tuple_146)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 23), list_128, tuple_144)
        
        # Getting the type of 'Y' (line 21)
        Y_149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'Y')
        
        # Obtaining an instance of the builtin type 'tuple' (line 21)
        tuple_150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 11), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 21)
        # Adding element type (line 21)
        # Getting the type of 'r' (line 21)
        r_151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 11), 'r')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 11), tuple_150, r_151)
        # Adding element type (line 21)
        # Getting the type of 'c' (line 21)
        c_152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 14), 'c')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 11), tuple_150, c_152)
        # Adding element type (line 21)
        # Getting the type of 'n' (line 21)
        n_153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 17), 'n')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 11), tuple_150, n_153)
        
        # Storing an element on a container (line 21)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 8), Y_149, (tuple_150, list_128))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Assigning a Call to a Tuple (line 26):
    
    # Assigning a Call to a Name:
    
    # Call to exact_cover(...): (line 26)
    # Processing the call arguments (line 26)
    # Getting the type of 'X1' (line 26)
    X1_155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 23), 'X1', False)
    # Getting the type of 'Y' (line 26)
    Y_156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 27), 'Y', False)
    # Processing the call keyword arguments (line 26)
    kwargs_157 = {}
    # Getting the type of 'exact_cover' (line 26)
    exact_cover_154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 11), 'exact_cover', False)
    # Calling exact_cover(args, kwargs) (line 26)
    exact_cover_call_result_158 = invoke(stypy.reporting.localization.Localization(__file__, 26, 11), exact_cover_154, *[X1_155, Y_156], **kwargs_157)
    
    # Assigning a type to the variable 'call_assignment_3' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'call_assignment_3', exact_cover_call_result_158)
    
    # Assigning a Call to a Name (line 26):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 4), 'int')
    # Processing the call keyword arguments
    kwargs_162 = {}
    # Getting the type of 'call_assignment_3' (line 26)
    call_assignment_3_159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'call_assignment_3', False)
    # Obtaining the member '__getitem__' of a type (line 26)
    getitem___160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 4), call_assignment_3_159, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_163 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___160, *[int_161], **kwargs_162)
    
    # Assigning a type to the variable 'call_assignment_4' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'call_assignment_4', getitem___call_result_163)
    
    # Assigning a Name to a Name (line 26):
    # Getting the type of 'call_assignment_4' (line 26)
    call_assignment_4_164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'call_assignment_4')
    # Assigning a type to the variable 'X' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'X', call_assignment_4_164)
    
    # Assigning a Call to a Name (line 26):
    
    # Call to __getitem__(...):
    # Processing the call arguments
    int_167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 4), 'int')
    # Processing the call keyword arguments
    kwargs_168 = {}
    # Getting the type of 'call_assignment_3' (line 26)
    call_assignment_3_165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'call_assignment_3', False)
    # Obtaining the member '__getitem__' of a type (line 26)
    getitem___166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 4), call_assignment_3_165, '__getitem__')
    # Calling __getitem__(args, kwargs)
    getitem___call_result_169 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___166, *[int_167], **kwargs_168)
    
    # Assigning a type to the variable 'call_assignment_5' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'call_assignment_5', getitem___call_result_169)
    
    # Assigning a Name to a Name (line 26):
    # Getting the type of 'call_assignment_5' (line 26)
    call_assignment_5_170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'call_assignment_5')
    # Assigning a type to the variable 'Y' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 7), 'Y', call_assignment_5_170)
    
    
    # Call to enumerate(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'grid' (line 27)
    grid_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 28), 'grid', False)
    # Processing the call keyword arguments (line 27)
    kwargs_173 = {}
    # Getting the type of 'enumerate' (line 27)
    enumerate_171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 18), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 27)
    enumerate_call_result_174 = invoke(stypy.reporting.localization.Localization(__file__, 27, 18), enumerate_171, *[grid_172], **kwargs_173)
    
    # Testing if the for loop is going to be iterated (line 27)
    # Testing the type of a for loop iterable (line 27)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 27, 4), enumerate_call_result_174)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 27, 4), enumerate_call_result_174):
        # Getting the type of the for loop variable (line 27)
        for_loop_var_175 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 27, 4), enumerate_call_result_174)
        # Assigning a type to the variable 'i' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 4), for_loop_var_175))
        # Assigning a type to the variable 'row' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'row', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 4), for_loop_var_175))
        # SSA begins for a for statement (line 27)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to enumerate(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'row' (line 28)
        row_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 30), 'row', False)
        # Processing the call keyword arguments (line 28)
        kwargs_178 = {}
        # Getting the type of 'enumerate' (line 28)
        enumerate_176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 20), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 28)
        enumerate_call_result_179 = invoke(stypy.reporting.localization.Localization(__file__, 28, 20), enumerate_176, *[row_177], **kwargs_178)
        
        # Testing if the for loop is going to be iterated (line 28)
        # Testing the type of a for loop iterable (line 28)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 28, 8), enumerate_call_result_179)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 28, 8), enumerate_call_result_179):
            # Getting the type of the for loop variable (line 28)
            for_loop_var_180 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 28, 8), enumerate_call_result_179)
            # Assigning a type to the variable 'j' (line 28)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 8), for_loop_var_180))
            # Assigning a type to the variable 'n' (line 28)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'n', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 8), for_loop_var_180))
            # SSA begins for a for statement (line 28)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            # Getting the type of 'n' (line 29)
            n_181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 15), 'n')
            # Testing if the type of an if condition is none (line 29)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 29, 12), n_181):
                pass
            else:
                
                # Testing the type of an if condition (line 29)
                if_condition_182 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 29, 12), n_181)
                # Assigning a type to the variable 'if_condition_182' (line 29)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'if_condition_182', if_condition_182)
                # SSA begins for if statement (line 29)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to select(...): (line 30)
                # Processing the call arguments (line 30)
                # Getting the type of 'X' (line 30)
                X_184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 23), 'X', False)
                # Getting the type of 'Y' (line 30)
                Y_185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 26), 'Y', False)
                
                # Obtaining an instance of the builtin type 'tuple' (line 30)
                tuple_186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 30), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 30)
                # Adding element type (line 30)
                # Getting the type of 'i' (line 30)
                i_187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 30), 'i', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 30), tuple_186, i_187)
                # Adding element type (line 30)
                # Getting the type of 'j' (line 30)
                j_188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 33), 'j', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 30), tuple_186, j_188)
                # Adding element type (line 30)
                # Getting the type of 'n' (line 30)
                n_189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 36), 'n', False)
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 30), tuple_186, n_189)
                
                # Processing the call keyword arguments (line 30)
                kwargs_190 = {}
                # Getting the type of 'select' (line 30)
                select_183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 16), 'select', False)
                # Calling select(args, kwargs) (line 30)
                select_call_result_191 = invoke(stypy.reporting.localization.Localization(__file__, 30, 16), select_183, *[X_184, Y_185, tuple_186], **kwargs_190)
                
                # SSA join for if statement (line 29)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    
    # Call to solve(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'X' (line 31)
    X_193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 26), 'X', False)
    # Getting the type of 'Y' (line 31)
    Y_194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 29), 'Y', False)
    
    # Obtaining an instance of the builtin type 'list' (line 31)
    list_195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 32), 'list')
    # Adding type elements to the builtin type 'list' instance (line 31)
    
    # Processing the call keyword arguments (line 31)
    kwargs_196 = {}
    # Getting the type of 'solve' (line 31)
    solve_192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 20), 'solve', False)
    # Calling solve(args, kwargs) (line 31)
    solve_call_result_197 = invoke(stypy.reporting.localization.Localization(__file__, 31, 20), solve_192, *[X_193, Y_194, list_195], **kwargs_196)
    
    # Testing if the for loop is going to be iterated (line 31)
    # Testing the type of a for loop iterable (line 31)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 31, 4), solve_call_result_197)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 31, 4), solve_call_result_197):
        # Getting the type of the for loop variable (line 31)
        for_loop_var_198 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 31, 4), solve_call_result_197)
        # Assigning a type to the variable 'solution' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'solution', for_loop_var_198)
        # SSA begins for a for statement (line 31)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'solution' (line 32)
        solution_199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 25), 'solution')
        # Testing if the for loop is going to be iterated (line 32)
        # Testing the type of a for loop iterable (line 32)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 32, 8), solution_199)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 32, 8), solution_199):
            # Getting the type of the for loop variable (line 32)
            for_loop_var_200 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 32, 8), solution_199)
            # Assigning a type to the variable 'r' (line 32)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'r', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 8), for_loop_var_200))
            # Assigning a type to the variable 'c' (line 32)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'c', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 8), for_loop_var_200))
            # Assigning a type to the variable 'n' (line 32)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'n', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 8), for_loop_var_200))
            # SSA begins for a for statement (line 32)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Name to a Subscript (line 33):
            
            # Assigning a Name to a Subscript (line 33):
            # Getting the type of 'n' (line 33)
            n_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 25), 'n')
            
            # Obtaining the type of the subscript
            # Getting the type of 'r' (line 33)
            r_202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 17), 'r')
            # Getting the type of 'grid' (line 33)
            grid_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'grid')
            # Obtaining the member '__getitem__' of a type (line 33)
            getitem___204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 12), grid_203, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 33)
            subscript_call_result_205 = invoke(stypy.reporting.localization.Localization(__file__, 33, 12), getitem___204, r_202)
            
            # Getting the type of 'c' (line 33)
            c_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 20), 'c')
            # Storing an element on a container (line 33)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 12), subscript_call_result_205, (c_206, n_201))
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Creating a generator
        # Getting the type of 'grid' (line 34)
        grid_207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 14), 'grid')
        GeneratorType_208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 8), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 8), GeneratorType_208, grid_207)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'stypy_return_type', GeneratorType_208)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'solve_sudoku(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'solve_sudoku' in the type store
    # Getting the type of 'stypy_return_type' (line 10)
    stypy_return_type_209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_209)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'solve_sudoku'
    return stypy_return_type_209

# Assigning a type to the variable 'solve_sudoku' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'solve_sudoku', solve_sudoku)

@norecursion
def exact_cover(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'exact_cover'
    module_type_store = module_type_store.open_function_context('exact_cover', 37, 0, False)
    
    # Passed parameters checking function
    exact_cover.stypy_localization = localization
    exact_cover.stypy_type_of_self = None
    exact_cover.stypy_type_store = module_type_store
    exact_cover.stypy_function_name = 'exact_cover'
    exact_cover.stypy_param_names_list = ['X1', 'Y']
    exact_cover.stypy_varargs_param_name = None
    exact_cover.stypy_kwargs_param_name = None
    exact_cover.stypy_call_defaults = defaults
    exact_cover.stypy_call_varargs = varargs
    exact_cover.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'exact_cover', ['X1', 'Y'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'exact_cover', localization, ['X1', 'Y'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'exact_cover(...)' code ##################

    
    # Assigning a Call to a Name (line 38):
    
    # Assigning a Call to a Name (line 38):
    
    # Call to dict(...): (line 38)
    # Processing the call arguments (line 38)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 38, 13, True)
    # Calculating comprehension expression
    # Getting the type of 'X1' (line 38)
    X1_216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 33), 'X1', False)
    comprehension_217 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 13), X1_216)
    # Assigning a type to the variable 'j' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 13), 'j', comprehension_217)
    
    # Obtaining an instance of the builtin type 'tuple' (line 38)
    tuple_211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 14), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 38)
    # Adding element type (line 38)
    # Getting the type of 'j' (line 38)
    j_212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 14), 'j', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 14), tuple_211, j_212)
    # Adding element type (line 38)
    
    # Call to set(...): (line 38)
    # Processing the call keyword arguments (line 38)
    kwargs_214 = {}
    # Getting the type of 'set' (line 38)
    set_213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 17), 'set', False)
    # Calling set(args, kwargs) (line 38)
    set_call_result_215 = invoke(stypy.reporting.localization.Localization(__file__, 38, 17), set_213, *[], **kwargs_214)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 14), tuple_211, set_call_result_215)
    
    list_218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 13), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 13), list_218, tuple_211)
    # Processing the call keyword arguments (line 38)
    kwargs_219 = {}
    # Getting the type of 'dict' (line 38)
    dict_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'dict', False)
    # Calling dict(args, kwargs) (line 38)
    dict_call_result_220 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), dict_210, *[list_218], **kwargs_219)
    
    # Assigning a type to the variable 'X' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'X', dict_call_result_220)
    
    
    # Call to iteritems(...): (line 39)
    # Processing the call keyword arguments (line 39)
    kwargs_223 = {}
    # Getting the type of 'Y' (line 39)
    Y_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 18), 'Y', False)
    # Obtaining the member 'iteritems' of a type (line 39)
    iteritems_222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 18), Y_221, 'iteritems')
    # Calling iteritems(args, kwargs) (line 39)
    iteritems_call_result_224 = invoke(stypy.reporting.localization.Localization(__file__, 39, 18), iteritems_222, *[], **kwargs_223)
    
    # Testing if the for loop is going to be iterated (line 39)
    # Testing the type of a for loop iterable (line 39)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 39, 4), iteritems_call_result_224)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 39, 4), iteritems_call_result_224):
        # Getting the type of the for loop variable (line 39)
        for_loop_var_225 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 39, 4), iteritems_call_result_224)
        # Assigning a type to the variable 'i' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 4), for_loop_var_225))
        # Assigning a type to the variable 'row' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'row', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 4), for_loop_var_225))
        # SSA begins for a for statement (line 39)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'row' (line 40)
        row_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 17), 'row')
        # Testing if the for loop is going to be iterated (line 40)
        # Testing the type of a for loop iterable (line 40)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 40, 8), row_226)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 40, 8), row_226):
            # Getting the type of the for loop variable (line 40)
            for_loop_var_227 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 40, 8), row_226)
            # Assigning a type to the variable 'j' (line 40)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'j', for_loop_var_227)
            # SSA begins for a for statement (line 40)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to add(...): (line 41)
            # Processing the call arguments (line 41)
            # Getting the type of 'i' (line 41)
            i_233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 21), 'i', False)
            # Processing the call keyword arguments (line 41)
            kwargs_234 = {}
            
            # Obtaining the type of the subscript
            # Getting the type of 'j' (line 41)
            j_228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 14), 'j', False)
            # Getting the type of 'X' (line 41)
            X_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'X', False)
            # Obtaining the member '__getitem__' of a type (line 41)
            getitem___230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 12), X_229, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 41)
            subscript_call_result_231 = invoke(stypy.reporting.localization.Localization(__file__, 41, 12), getitem___230, j_228)
            
            # Obtaining the member 'add' of a type (line 41)
            add_232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 12), subscript_call_result_231, 'add')
            # Calling add(args, kwargs) (line 41)
            add_call_result_235 = invoke(stypy.reporting.localization.Localization(__file__, 41, 12), add_232, *[i_233], **kwargs_234)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Obtaining an instance of the builtin type 'tuple' (line 42)
    tuple_236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 42)
    # Adding element type (line 42)
    # Getting the type of 'X' (line 42)
    X_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 11), 'X')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 11), tuple_236, X_237)
    # Adding element type (line 42)
    # Getting the type of 'Y' (line 42)
    Y_238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 14), 'Y')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 11), tuple_236, Y_238)
    
    # Assigning a type to the variable 'stypy_return_type' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'stypy_return_type', tuple_236)
    
    # ################# End of 'exact_cover(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'exact_cover' in the type store
    # Getting the type of 'stypy_return_type' (line 37)
    stypy_return_type_239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_239)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'exact_cover'
    return stypy_return_type_239

# Assigning a type to the variable 'exact_cover' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'exact_cover', exact_cover)

@norecursion
def solve(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'solve'
    module_type_store = module_type_store.open_function_context('solve', 45, 0, False)
    
    # Passed parameters checking function
    solve.stypy_localization = localization
    solve.stypy_type_of_self = None
    solve.stypy_type_store = module_type_store
    solve.stypy_function_name = 'solve'
    solve.stypy_param_names_list = ['X', 'Y', 'solution']
    solve.stypy_varargs_param_name = None
    solve.stypy_kwargs_param_name = None
    solve.stypy_call_defaults = defaults
    solve.stypy_call_varargs = varargs
    solve.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'solve', ['X', 'Y', 'solution'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'solve', localization, ['X', 'Y', 'solution'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'solve(...)' code ##################

    
    # Getting the type of 'X' (line 46)
    X_240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 11), 'X')
    # Applying the 'not' unary operator (line 46)
    result_not__241 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 7), 'not', X_240)
    
    # Testing if the type of an if condition is none (line 46)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 46, 4), result_not__241):
        
        # Assigning a Subscript to a Name (line 50):
        
        # Assigning a Subscript to a Name (line 50):
        
        # Obtaining the type of the subscript
        int_248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 45), 'int')
        
        # Call to min(...): (line 50)
        # Processing the call arguments (line 50)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'X' (line 50)
        X_259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 41), 'X', False)
        comprehension_260 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), X_259)
        # Assigning a type to the variable 'c' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 17), 'c', comprehension_260)
        
        # Obtaining an instance of the builtin type 'tuple' (line 50)
        tuple_250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 50)
        # Adding element type (line 50)
        
        # Call to len(...): (line 50)
        # Processing the call arguments (line 50)
        
        # Obtaining the type of the subscript
        # Getting the type of 'c' (line 50)
        c_252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 24), 'c', False)
        # Getting the type of 'X' (line 50)
        X_253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 22), 'X', False)
        # Obtaining the member '__getitem__' of a type (line 50)
        getitem___254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 22), X_253, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 50)
        subscript_call_result_255 = invoke(stypy.reporting.localization.Localization(__file__, 50, 22), getitem___254, c_252)
        
        # Processing the call keyword arguments (line 50)
        kwargs_256 = {}
        # Getting the type of 'len' (line 50)
        len_251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 18), 'len', False)
        # Calling len(args, kwargs) (line 50)
        len_call_result_257 = invoke(stypy.reporting.localization.Localization(__file__, 50, 18), len_251, *[subscript_call_result_255], **kwargs_256)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 18), tuple_250, len_call_result_257)
        # Adding element type (line 50)
        # Getting the type of 'c' (line 50)
        c_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 29), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 18), tuple_250, c_258)
        
        list_261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 17), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), list_261, tuple_250)
        # Processing the call keyword arguments (line 50)
        kwargs_262 = {}
        # Getting the type of 'min' (line 50)
        min_249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'min', False)
        # Calling min(args, kwargs) (line 50)
        min_call_result_263 = invoke(stypy.reporting.localization.Localization(__file__, 50, 12), min_249, *[list_261], **kwargs_262)
        
        # Obtaining the member '__getitem__' of a type (line 50)
        getitem___264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 12), min_call_result_263, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 50)
        subscript_call_result_265 = invoke(stypy.reporting.localization.Localization(__file__, 50, 12), getitem___264, int_248)
        
        # Assigning a type to the variable 'c' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'c', subscript_call_result_265)
        
        
        # Call to list(...): (line 51)
        # Processing the call arguments (line 51)
        
        # Obtaining the type of the subscript
        # Getting the type of 'c' (line 51)
        c_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 24), 'c', False)
        # Getting the type of 'X' (line 51)
        X_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 22), 'X', False)
        # Obtaining the member '__getitem__' of a type (line 51)
        getitem___269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 22), X_268, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
        subscript_call_result_270 = invoke(stypy.reporting.localization.Localization(__file__, 51, 22), getitem___269, c_267)
        
        # Processing the call keyword arguments (line 51)
        kwargs_271 = {}
        # Getting the type of 'list' (line 51)
        list_266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 17), 'list', False)
        # Calling list(args, kwargs) (line 51)
        list_call_result_272 = invoke(stypy.reporting.localization.Localization(__file__, 51, 17), list_266, *[subscript_call_result_270], **kwargs_271)
        
        # Testing if the for loop is going to be iterated (line 51)
        # Testing the type of a for loop iterable (line 51)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 51, 8), list_call_result_272)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 51, 8), list_call_result_272):
            # Getting the type of the for loop variable (line 51)
            for_loop_var_273 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 51, 8), list_call_result_272)
            # Assigning a type to the variable 'r' (line 51)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'r', for_loop_var_273)
            # SSA begins for a for statement (line 51)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to append(...): (line 52)
            # Processing the call arguments (line 52)
            # Getting the type of 'r' (line 52)
            r_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 28), 'r', False)
            # Processing the call keyword arguments (line 52)
            kwargs_277 = {}
            # Getting the type of 'solution' (line 52)
            solution_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'solution', False)
            # Obtaining the member 'append' of a type (line 52)
            append_275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 12), solution_274, 'append')
            # Calling append(args, kwargs) (line 52)
            append_call_result_278 = invoke(stypy.reporting.localization.Localization(__file__, 52, 12), append_275, *[r_276], **kwargs_277)
            
            
            # Assigning a Call to a Name (line 53):
            
            # Assigning a Call to a Name (line 53):
            
            # Call to select(...): (line 53)
            # Processing the call arguments (line 53)
            # Getting the type of 'X' (line 53)
            X_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 26), 'X', False)
            # Getting the type of 'Y' (line 53)
            Y_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 29), 'Y', False)
            # Getting the type of 'r' (line 53)
            r_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 32), 'r', False)
            # Processing the call keyword arguments (line 53)
            kwargs_283 = {}
            # Getting the type of 'select' (line 53)
            select_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 19), 'select', False)
            # Calling select(args, kwargs) (line 53)
            select_call_result_284 = invoke(stypy.reporting.localization.Localization(__file__, 53, 19), select_279, *[X_280, Y_281, r_282], **kwargs_283)
            
            # Assigning a type to the variable 'cols' (line 53)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'cols', select_call_result_284)
            
            
            # Call to solve(...): (line 54)
            # Processing the call arguments (line 54)
            # Getting the type of 'X' (line 54)
            X_286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 34), 'X', False)
            # Getting the type of 'Y' (line 54)
            Y_287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 37), 'Y', False)
            # Getting the type of 'solution' (line 54)
            solution_288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 40), 'solution', False)
            # Processing the call keyword arguments (line 54)
            kwargs_289 = {}
            # Getting the type of 'solve' (line 54)
            solve_285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 28), 'solve', False)
            # Calling solve(args, kwargs) (line 54)
            solve_call_result_290 = invoke(stypy.reporting.localization.Localization(__file__, 54, 28), solve_285, *[X_286, Y_287, solution_288], **kwargs_289)
            
            # Testing if the for loop is going to be iterated (line 54)
            # Testing the type of a for loop iterable (line 54)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 54, 12), solve_call_result_290)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 54, 12), solve_call_result_290):
                # Getting the type of the for loop variable (line 54)
                for_loop_var_291 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 54, 12), solve_call_result_290)
                # Assigning a type to the variable 'solution' (line 54)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'solution', for_loop_var_291)
                # SSA begins for a for statement (line 54)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                # Creating a generator
                # Getting the type of 'solution' (line 55)
                solution_292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 22), 'solution')
                GeneratorType_293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 16), 'GeneratorType')
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 16), GeneratorType_293, solution_292)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), 'stypy_return_type', GeneratorType_293)
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Call to deselect(...): (line 56)
            # Processing the call arguments (line 56)
            # Getting the type of 'X' (line 56)
            X_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 21), 'X', False)
            # Getting the type of 'Y' (line 56)
            Y_296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 24), 'Y', False)
            # Getting the type of 'r' (line 56)
            r_297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 27), 'r', False)
            # Getting the type of 'cols' (line 56)
            cols_298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 30), 'cols', False)
            # Processing the call keyword arguments (line 56)
            kwargs_299 = {}
            # Getting the type of 'deselect' (line 56)
            deselect_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'deselect', False)
            # Calling deselect(args, kwargs) (line 56)
            deselect_call_result_300 = invoke(stypy.reporting.localization.Localization(__file__, 56, 12), deselect_294, *[X_295, Y_296, r_297, cols_298], **kwargs_299)
            
            
            # Call to pop(...): (line 57)
            # Processing the call keyword arguments (line 57)
            kwargs_303 = {}
            # Getting the type of 'solution' (line 57)
            solution_301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'solution', False)
            # Obtaining the member 'pop' of a type (line 57)
            pop_302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 12), solution_301, 'pop')
            # Calling pop(args, kwargs) (line 57)
            pop_call_result_304 = invoke(stypy.reporting.localization.Localization(__file__, 57, 12), pop_302, *[], **kwargs_303)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
    else:
        
        # Testing the type of an if condition (line 46)
        if_condition_242 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 46, 4), result_not__241)
        # Assigning a type to the variable 'if_condition_242' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'if_condition_242', if_condition_242)
        # SSA begins for if statement (line 46)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Creating a generator
        
        # Call to list(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'solution' (line 47)
        solution_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 19), 'solution', False)
        # Processing the call keyword arguments (line 47)
        kwargs_245 = {}
        # Getting the type of 'list' (line 47)
        list_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 14), 'list', False)
        # Calling list(args, kwargs) (line 47)
        list_call_result_246 = invoke(stypy.reporting.localization.Localization(__file__, 47, 14), list_243, *[solution_244], **kwargs_245)
        
        GeneratorType_247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 8), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 8), GeneratorType_247, list_call_result_246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'stypy_return_type', GeneratorType_247)
        # SSA branch for the else part of an if statement (line 46)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Subscript to a Name (line 50):
        
        # Assigning a Subscript to a Name (line 50):
        
        # Obtaining the type of the subscript
        int_248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 45), 'int')
        
        # Call to min(...): (line 50)
        # Processing the call arguments (line 50)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'X' (line 50)
        X_259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 41), 'X', False)
        comprehension_260 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), X_259)
        # Assigning a type to the variable 'c' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 17), 'c', comprehension_260)
        
        # Obtaining an instance of the builtin type 'tuple' (line 50)
        tuple_250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 18), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 50)
        # Adding element type (line 50)
        
        # Call to len(...): (line 50)
        # Processing the call arguments (line 50)
        
        # Obtaining the type of the subscript
        # Getting the type of 'c' (line 50)
        c_252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 24), 'c', False)
        # Getting the type of 'X' (line 50)
        X_253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 22), 'X', False)
        # Obtaining the member '__getitem__' of a type (line 50)
        getitem___254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 22), X_253, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 50)
        subscript_call_result_255 = invoke(stypy.reporting.localization.Localization(__file__, 50, 22), getitem___254, c_252)
        
        # Processing the call keyword arguments (line 50)
        kwargs_256 = {}
        # Getting the type of 'len' (line 50)
        len_251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 18), 'len', False)
        # Calling len(args, kwargs) (line 50)
        len_call_result_257 = invoke(stypy.reporting.localization.Localization(__file__, 50, 18), len_251, *[subscript_call_result_255], **kwargs_256)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 18), tuple_250, len_call_result_257)
        # Adding element type (line 50)
        # Getting the type of 'c' (line 50)
        c_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 29), 'c', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 18), tuple_250, c_258)
        
        list_261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 17), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 17), list_261, tuple_250)
        # Processing the call keyword arguments (line 50)
        kwargs_262 = {}
        # Getting the type of 'min' (line 50)
        min_249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'min', False)
        # Calling min(args, kwargs) (line 50)
        min_call_result_263 = invoke(stypy.reporting.localization.Localization(__file__, 50, 12), min_249, *[list_261], **kwargs_262)
        
        # Obtaining the member '__getitem__' of a type (line 50)
        getitem___264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 12), min_call_result_263, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 50)
        subscript_call_result_265 = invoke(stypy.reporting.localization.Localization(__file__, 50, 12), getitem___264, int_248)
        
        # Assigning a type to the variable 'c' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'c', subscript_call_result_265)
        
        
        # Call to list(...): (line 51)
        # Processing the call arguments (line 51)
        
        # Obtaining the type of the subscript
        # Getting the type of 'c' (line 51)
        c_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 24), 'c', False)
        # Getting the type of 'X' (line 51)
        X_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 22), 'X', False)
        # Obtaining the member '__getitem__' of a type (line 51)
        getitem___269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 22), X_268, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 51)
        subscript_call_result_270 = invoke(stypy.reporting.localization.Localization(__file__, 51, 22), getitem___269, c_267)
        
        # Processing the call keyword arguments (line 51)
        kwargs_271 = {}
        # Getting the type of 'list' (line 51)
        list_266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 17), 'list', False)
        # Calling list(args, kwargs) (line 51)
        list_call_result_272 = invoke(stypy.reporting.localization.Localization(__file__, 51, 17), list_266, *[subscript_call_result_270], **kwargs_271)
        
        # Testing if the for loop is going to be iterated (line 51)
        # Testing the type of a for loop iterable (line 51)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 51, 8), list_call_result_272)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 51, 8), list_call_result_272):
            # Getting the type of the for loop variable (line 51)
            for_loop_var_273 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 51, 8), list_call_result_272)
            # Assigning a type to the variable 'r' (line 51)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'r', for_loop_var_273)
            # SSA begins for a for statement (line 51)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to append(...): (line 52)
            # Processing the call arguments (line 52)
            # Getting the type of 'r' (line 52)
            r_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 28), 'r', False)
            # Processing the call keyword arguments (line 52)
            kwargs_277 = {}
            # Getting the type of 'solution' (line 52)
            solution_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'solution', False)
            # Obtaining the member 'append' of a type (line 52)
            append_275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 12), solution_274, 'append')
            # Calling append(args, kwargs) (line 52)
            append_call_result_278 = invoke(stypy.reporting.localization.Localization(__file__, 52, 12), append_275, *[r_276], **kwargs_277)
            
            
            # Assigning a Call to a Name (line 53):
            
            # Assigning a Call to a Name (line 53):
            
            # Call to select(...): (line 53)
            # Processing the call arguments (line 53)
            # Getting the type of 'X' (line 53)
            X_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 26), 'X', False)
            # Getting the type of 'Y' (line 53)
            Y_281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 29), 'Y', False)
            # Getting the type of 'r' (line 53)
            r_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 32), 'r', False)
            # Processing the call keyword arguments (line 53)
            kwargs_283 = {}
            # Getting the type of 'select' (line 53)
            select_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 19), 'select', False)
            # Calling select(args, kwargs) (line 53)
            select_call_result_284 = invoke(stypy.reporting.localization.Localization(__file__, 53, 19), select_279, *[X_280, Y_281, r_282], **kwargs_283)
            
            # Assigning a type to the variable 'cols' (line 53)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'cols', select_call_result_284)
            
            
            # Call to solve(...): (line 54)
            # Processing the call arguments (line 54)
            # Getting the type of 'X' (line 54)
            X_286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 34), 'X', False)
            # Getting the type of 'Y' (line 54)
            Y_287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 37), 'Y', False)
            # Getting the type of 'solution' (line 54)
            solution_288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 40), 'solution', False)
            # Processing the call keyword arguments (line 54)
            kwargs_289 = {}
            # Getting the type of 'solve' (line 54)
            solve_285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 28), 'solve', False)
            # Calling solve(args, kwargs) (line 54)
            solve_call_result_290 = invoke(stypy.reporting.localization.Localization(__file__, 54, 28), solve_285, *[X_286, Y_287, solution_288], **kwargs_289)
            
            # Testing if the for loop is going to be iterated (line 54)
            # Testing the type of a for loop iterable (line 54)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 54, 12), solve_call_result_290)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 54, 12), solve_call_result_290):
                # Getting the type of the for loop variable (line 54)
                for_loop_var_291 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 54, 12), solve_call_result_290)
                # Assigning a type to the variable 'solution' (line 54)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'solution', for_loop_var_291)
                # SSA begins for a for statement (line 54)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                # Creating a generator
                # Getting the type of 'solution' (line 55)
                solution_292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 22), 'solution')
                GeneratorType_293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 16), 'GeneratorType')
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 16), GeneratorType_293, solution_292)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 16), 'stypy_return_type', GeneratorType_293)
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            
            # Call to deselect(...): (line 56)
            # Processing the call arguments (line 56)
            # Getting the type of 'X' (line 56)
            X_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 21), 'X', False)
            # Getting the type of 'Y' (line 56)
            Y_296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 24), 'Y', False)
            # Getting the type of 'r' (line 56)
            r_297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 27), 'r', False)
            # Getting the type of 'cols' (line 56)
            cols_298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 30), 'cols', False)
            # Processing the call keyword arguments (line 56)
            kwargs_299 = {}
            # Getting the type of 'deselect' (line 56)
            deselect_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'deselect', False)
            # Calling deselect(args, kwargs) (line 56)
            deselect_call_result_300 = invoke(stypy.reporting.localization.Localization(__file__, 56, 12), deselect_294, *[X_295, Y_296, r_297, cols_298], **kwargs_299)
            
            
            # Call to pop(...): (line 57)
            # Processing the call keyword arguments (line 57)
            kwargs_303 = {}
            # Getting the type of 'solution' (line 57)
            solution_301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'solution', False)
            # Obtaining the member 'pop' of a type (line 57)
            pop_302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 12), solution_301, 'pop')
            # Calling pop(args, kwargs) (line 57)
            pop_call_result_304 = invoke(stypy.reporting.localization.Localization(__file__, 57, 12), pop_302, *[], **kwargs_303)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for if statement (line 46)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # ################# End of 'solve(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'solve' in the type store
    # Getting the type of 'stypy_return_type' (line 45)
    stypy_return_type_305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_305)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'solve'
    return stypy_return_type_305

# Assigning a type to the variable 'solve' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'solve', solve)

@norecursion
def select(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'select'
    module_type_store = module_type_store.open_function_context('select', 60, 0, False)
    
    # Passed parameters checking function
    select.stypy_localization = localization
    select.stypy_type_of_self = None
    select.stypy_type_store = module_type_store
    select.stypy_function_name = 'select'
    select.stypy_param_names_list = ['X', 'Y', 'r']
    select.stypy_varargs_param_name = None
    select.stypy_kwargs_param_name = None
    select.stypy_call_defaults = defaults
    select.stypy_call_varargs = varargs
    select.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'select', ['X', 'Y', 'r'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'select', localization, ['X', 'Y', 'r'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'select(...)' code ##################

    
    # Assigning a List to a Name (line 61):
    
    # Assigning a List to a Name (line 61):
    
    # Obtaining an instance of the builtin type 'list' (line 61)
    list_306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 61)
    
    # Assigning a type to the variable 'cols' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'cols', list_306)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'r' (line 62)
    r_307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 15), 'r')
    # Getting the type of 'Y' (line 62)
    Y_308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 13), 'Y')
    # Obtaining the member '__getitem__' of a type (line 62)
    getitem___309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 13), Y_308, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 62)
    subscript_call_result_310 = invoke(stypy.reporting.localization.Localization(__file__, 62, 13), getitem___309, r_307)
    
    # Testing if the for loop is going to be iterated (line 62)
    # Testing the type of a for loop iterable (line 62)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 62, 4), subscript_call_result_310)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 62, 4), subscript_call_result_310):
        # Getting the type of the for loop variable (line 62)
        for_loop_var_311 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 62, 4), subscript_call_result_310)
        # Assigning a type to the variable 'j' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'j', for_loop_var_311)
        # SSA begins for a for statement (line 62)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 63)
        j_312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 19), 'j')
        # Getting the type of 'X' (line 63)
        X_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 17), 'X')
        # Obtaining the member '__getitem__' of a type (line 63)
        getitem___314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 17), X_313, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 63)
        subscript_call_result_315 = invoke(stypy.reporting.localization.Localization(__file__, 63, 17), getitem___314, j_312)
        
        # Testing if the for loop is going to be iterated (line 63)
        # Testing the type of a for loop iterable (line 63)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 63, 8), subscript_call_result_315)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 63, 8), subscript_call_result_315):
            # Getting the type of the for loop variable (line 63)
            for_loop_var_316 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 63, 8), subscript_call_result_315)
            # Assigning a type to the variable 'i' (line 63)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'i', for_loop_var_316)
            # SSA begins for a for statement (line 63)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 64)
            i_317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 23), 'i')
            # Getting the type of 'Y' (line 64)
            Y_318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 21), 'Y')
            # Obtaining the member '__getitem__' of a type (line 64)
            getitem___319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 21), Y_318, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 64)
            subscript_call_result_320 = invoke(stypy.reporting.localization.Localization(__file__, 64, 21), getitem___319, i_317)
            
            # Testing if the for loop is going to be iterated (line 64)
            # Testing the type of a for loop iterable (line 64)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 64, 12), subscript_call_result_320)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 64, 12), subscript_call_result_320):
                # Getting the type of the for loop variable (line 64)
                for_loop_var_321 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 64, 12), subscript_call_result_320)
                # Assigning a type to the variable 'k' (line 64)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'k', for_loop_var_321)
                # SSA begins for a for statement (line 64)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Getting the type of 'k' (line 65)
                k_322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 19), 'k')
                # Getting the type of 'j' (line 65)
                j_323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 24), 'j')
                # Applying the binary operator '!=' (line 65)
                result_ne_324 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 19), '!=', k_322, j_323)
                
                # Testing if the type of an if condition is none (line 65)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 65, 16), result_ne_324):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 65)
                    if_condition_325 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 65, 16), result_ne_324)
                    # Assigning a type to the variable 'if_condition_325' (line 65)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 16), 'if_condition_325', if_condition_325)
                    # SSA begins for if statement (line 65)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to remove(...): (line 66)
                    # Processing the call arguments (line 66)
                    # Getting the type of 'i' (line 66)
                    i_331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 32), 'i', False)
                    # Processing the call keyword arguments (line 66)
                    kwargs_332 = {}
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'k' (line 66)
                    k_326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 22), 'k', False)
                    # Getting the type of 'X' (line 66)
                    X_327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 20), 'X', False)
                    # Obtaining the member '__getitem__' of a type (line 66)
                    getitem___328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 20), X_327, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 66)
                    subscript_call_result_329 = invoke(stypy.reporting.localization.Localization(__file__, 66, 20), getitem___328, k_326)
                    
                    # Obtaining the member 'remove' of a type (line 66)
                    remove_330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 20), subscript_call_result_329, 'remove')
                    # Calling remove(args, kwargs) (line 66)
                    remove_call_result_333 = invoke(stypy.reporting.localization.Localization(__file__, 66, 20), remove_330, *[i_331], **kwargs_332)
                    
                    # SSA join for if statement (line 65)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Call to append(...): (line 67)
        # Processing the call arguments (line 67)
        
        # Call to pop(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'j' (line 67)
        j_338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 26), 'j', False)
        # Processing the call keyword arguments (line 67)
        kwargs_339 = {}
        # Getting the type of 'X' (line 67)
        X_336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 20), 'X', False)
        # Obtaining the member 'pop' of a type (line 67)
        pop_337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 20), X_336, 'pop')
        # Calling pop(args, kwargs) (line 67)
        pop_call_result_340 = invoke(stypy.reporting.localization.Localization(__file__, 67, 20), pop_337, *[j_338], **kwargs_339)
        
        # Processing the call keyword arguments (line 67)
        kwargs_341 = {}
        # Getting the type of 'cols' (line 67)
        cols_334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'cols', False)
        # Obtaining the member 'append' of a type (line 67)
        append_335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), cols_334, 'append')
        # Calling append(args, kwargs) (line 67)
        append_call_result_342 = invoke(stypy.reporting.localization.Localization(__file__, 67, 8), append_335, *[pop_call_result_340], **kwargs_341)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'cols' (line 68)
    cols_343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 11), 'cols')
    # Assigning a type to the variable 'stypy_return_type' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'stypy_return_type', cols_343)
    
    # ################# End of 'select(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'select' in the type store
    # Getting the type of 'stypy_return_type' (line 60)
    stypy_return_type_344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_344)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'select'
    return stypy_return_type_344

# Assigning a type to the variable 'select' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'select', select)

@norecursion
def deselect(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'deselect'
    module_type_store = module_type_store.open_function_context('deselect', 71, 0, False)
    
    # Passed parameters checking function
    deselect.stypy_localization = localization
    deselect.stypy_type_of_self = None
    deselect.stypy_type_store = module_type_store
    deselect.stypy_function_name = 'deselect'
    deselect.stypy_param_names_list = ['X', 'Y', 'r', 'cols']
    deselect.stypy_varargs_param_name = None
    deselect.stypy_kwargs_param_name = None
    deselect.stypy_call_defaults = defaults
    deselect.stypy_call_varargs = varargs
    deselect.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'deselect', ['X', 'Y', 'r', 'cols'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'deselect', localization, ['X', 'Y', 'r', 'cols'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'deselect(...)' code ##################

    
    
    # Call to reversed(...): (line 72)
    # Processing the call arguments (line 72)
    
    # Obtaining the type of the subscript
    # Getting the type of 'r' (line 72)
    r_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 24), 'r', False)
    # Getting the type of 'Y' (line 72)
    Y_347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 22), 'Y', False)
    # Obtaining the member '__getitem__' of a type (line 72)
    getitem___348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 22), Y_347, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 72)
    subscript_call_result_349 = invoke(stypy.reporting.localization.Localization(__file__, 72, 22), getitem___348, r_346)
    
    # Processing the call keyword arguments (line 72)
    kwargs_350 = {}
    # Getting the type of 'reversed' (line 72)
    reversed_345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 13), 'reversed', False)
    # Calling reversed(args, kwargs) (line 72)
    reversed_call_result_351 = invoke(stypy.reporting.localization.Localization(__file__, 72, 13), reversed_345, *[subscript_call_result_349], **kwargs_350)
    
    # Testing if the for loop is going to be iterated (line 72)
    # Testing the type of a for loop iterable (line 72)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 72, 4), reversed_call_result_351)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 72, 4), reversed_call_result_351):
        # Getting the type of the for loop variable (line 72)
        for_loop_var_352 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 72, 4), reversed_call_result_351)
        # Assigning a type to the variable 'j' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'j', for_loop_var_352)
        # SSA begins for a for statement (line 72)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Subscript (line 73):
        
        # Assigning a Call to a Subscript (line 73):
        
        # Call to pop(...): (line 73)
        # Processing the call keyword arguments (line 73)
        kwargs_355 = {}
        # Getting the type of 'cols' (line 73)
        cols_353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 15), 'cols', False)
        # Obtaining the member 'pop' of a type (line 73)
        pop_354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 15), cols_353, 'pop')
        # Calling pop(args, kwargs) (line 73)
        pop_call_result_356 = invoke(stypy.reporting.localization.Localization(__file__, 73, 15), pop_354, *[], **kwargs_355)
        
        # Getting the type of 'X' (line 73)
        X_357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'X')
        # Getting the type of 'j' (line 73)
        j_358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 10), 'j')
        # Storing an element on a container (line 73)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 8), X_357, (j_358, pop_call_result_356))
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'j' (line 74)
        j_359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 19), 'j')
        # Getting the type of 'X' (line 74)
        X_360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 17), 'X')
        # Obtaining the member '__getitem__' of a type (line 74)
        getitem___361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 17), X_360, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 74)
        subscript_call_result_362 = invoke(stypy.reporting.localization.Localization(__file__, 74, 17), getitem___361, j_359)
        
        # Testing if the for loop is going to be iterated (line 74)
        # Testing the type of a for loop iterable (line 74)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 74, 8), subscript_call_result_362)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 74, 8), subscript_call_result_362):
            # Getting the type of the for loop variable (line 74)
            for_loop_var_363 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 74, 8), subscript_call_result_362)
            # Assigning a type to the variable 'i' (line 74)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'i', for_loop_var_363)
            # SSA begins for a for statement (line 74)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'i' (line 75)
            i_364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 23), 'i')
            # Getting the type of 'Y' (line 75)
            Y_365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 21), 'Y')
            # Obtaining the member '__getitem__' of a type (line 75)
            getitem___366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 21), Y_365, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 75)
            subscript_call_result_367 = invoke(stypy.reporting.localization.Localization(__file__, 75, 21), getitem___366, i_364)
            
            # Testing if the for loop is going to be iterated (line 75)
            # Testing the type of a for loop iterable (line 75)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 75, 12), subscript_call_result_367)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 75, 12), subscript_call_result_367):
                # Getting the type of the for loop variable (line 75)
                for_loop_var_368 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 75, 12), subscript_call_result_367)
                # Assigning a type to the variable 'k' (line 75)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'k', for_loop_var_368)
                # SSA begins for a for statement (line 75)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Getting the type of 'k' (line 76)
                k_369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 19), 'k')
                # Getting the type of 'j' (line 76)
                j_370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 24), 'j')
                # Applying the binary operator '!=' (line 76)
                result_ne_371 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 19), '!=', k_369, j_370)
                
                # Testing if the type of an if condition is none (line 76)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 76, 16), result_ne_371):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 76)
                    if_condition_372 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 16), result_ne_371)
                    # Assigning a type to the variable 'if_condition_372' (line 76)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'if_condition_372', if_condition_372)
                    # SSA begins for if statement (line 76)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to add(...): (line 77)
                    # Processing the call arguments (line 77)
                    # Getting the type of 'i' (line 77)
                    i_378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 29), 'i', False)
                    # Processing the call keyword arguments (line 77)
                    kwargs_379 = {}
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'k' (line 77)
                    k_373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 22), 'k', False)
                    # Getting the type of 'X' (line 77)
                    X_374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 20), 'X', False)
                    # Obtaining the member '__getitem__' of a type (line 77)
                    getitem___375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 20), X_374, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 77)
                    subscript_call_result_376 = invoke(stypy.reporting.localization.Localization(__file__, 77, 20), getitem___375, k_373)
                    
                    # Obtaining the member 'add' of a type (line 77)
                    add_377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 20), subscript_call_result_376, 'add')
                    # Calling add(args, kwargs) (line 77)
                    add_call_result_380 = invoke(stypy.reporting.localization.Localization(__file__, 77, 20), add_377, *[i_378], **kwargs_379)
                    
                    # SSA join for if statement (line 76)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'deselect(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'deselect' in the type store
    # Getting the type of 'stypy_return_type' (line 71)
    stypy_return_type_381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_381)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'deselect'
    return stypy_return_type_381

# Assigning a type to the variable 'deselect' (line 71)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'deselect', deselect)

@norecursion
def main(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'main'
    module_type_store = module_type_store.open_function_context('main', 80, 0, False)
    
    # Passed parameters checking function
    main.stypy_localization = localization
    main.stypy_type_of_self = None
    main.stypy_type_store = module_type_store
    main.stypy_function_name = 'main'
    main.stypy_param_names_list = []
    main.stypy_varargs_param_name = None
    main.stypy_kwargs_param_name = None
    main.stypy_call_defaults = defaults
    main.stypy_call_varargs = varargs
    main.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'main', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'main', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'main(...)' code ##################

    
    # Assigning a List to a Name (line 81):
    
    # Assigning a List to a Name (line 81):
    
    # Obtaining an instance of the builtin type 'list' (line 81)
    list_382 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 81)
    # Adding element type (line 81)
    
    # Obtaining an instance of the builtin type 'list' (line 82)
    list_383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 82)
    # Adding element type (line 82)
    int_384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 8), list_383, int_384)
    # Adding element type (line 82)
    int_385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 8), list_383, int_385)
    # Adding element type (line 82)
    int_386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 8), list_383, int_386)
    # Adding element type (line 82)
    int_387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 8), list_383, int_387)
    # Adding element type (line 82)
    int_388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 8), list_383, int_388)
    # Adding element type (line 82)
    int_389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 8), list_383, int_389)
    # Adding element type (line 82)
    int_390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 8), list_383, int_390)
    # Adding element type (line 82)
    int_391 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 8), list_383, int_391)
    # Adding element type (line 82)
    int_392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 8), list_383, int_392)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 11), list_382, list_383)
    # Adding element type (line 81)
    
    # Obtaining an instance of the builtin type 'list' (line 83)
    list_393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 83)
    # Adding element type (line 83)
    int_394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 8), list_393, int_394)
    # Adding element type (line 83)
    int_395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 8), list_393, int_395)
    # Adding element type (line 83)
    int_396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 8), list_393, int_396)
    # Adding element type (line 83)
    int_397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 8), list_393, int_397)
    # Adding element type (line 83)
    int_398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 8), list_393, int_398)
    # Adding element type (line 83)
    int_399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 8), list_393, int_399)
    # Adding element type (line 83)
    int_400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 8), list_393, int_400)
    # Adding element type (line 83)
    int_401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 8), list_393, int_401)
    # Adding element type (line 83)
    int_402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 8), list_393, int_402)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 11), list_382, list_393)
    # Adding element type (line 81)
    
    # Obtaining an instance of the builtin type 'list' (line 84)
    list_403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 84)
    # Adding element type (line 84)
    int_404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 8), list_403, int_404)
    # Adding element type (line 84)
    int_405 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 8), list_403, int_405)
    # Adding element type (line 84)
    int_406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 8), list_403, int_406)
    # Adding element type (line 84)
    int_407 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 8), list_403, int_407)
    # Adding element type (line 84)
    int_408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 8), list_403, int_408)
    # Adding element type (line 84)
    int_409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 8), list_403, int_409)
    # Adding element type (line 84)
    int_410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 8), list_403, int_410)
    # Adding element type (line 84)
    int_411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 8), list_403, int_411)
    # Adding element type (line 84)
    int_412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 8), list_403, int_412)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 11), list_382, list_403)
    # Adding element type (line 81)
    
    # Obtaining an instance of the builtin type 'list' (line 85)
    list_413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 85)
    # Adding element type (line 85)
    int_414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 8), list_413, int_414)
    # Adding element type (line 85)
    int_415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 8), list_413, int_415)
    # Adding element type (line 85)
    int_416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 8), list_413, int_416)
    # Adding element type (line 85)
    int_417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 8), list_413, int_417)
    # Adding element type (line 85)
    int_418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 8), list_413, int_418)
    # Adding element type (line 85)
    int_419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 8), list_413, int_419)
    # Adding element type (line 85)
    int_420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 8), list_413, int_420)
    # Adding element type (line 85)
    int_421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 8), list_413, int_421)
    # Adding element type (line 85)
    int_422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 8), list_413, int_422)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 11), list_382, list_413)
    # Adding element type (line 81)
    
    # Obtaining an instance of the builtin type 'list' (line 86)
    list_423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 86)
    # Adding element type (line 86)
    int_424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 8), list_423, int_424)
    # Adding element type (line 86)
    int_425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 8), list_423, int_425)
    # Adding element type (line 86)
    int_426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 8), list_423, int_426)
    # Adding element type (line 86)
    int_427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 8), list_423, int_427)
    # Adding element type (line 86)
    int_428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 8), list_423, int_428)
    # Adding element type (line 86)
    int_429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 8), list_423, int_429)
    # Adding element type (line 86)
    int_430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 8), list_423, int_430)
    # Adding element type (line 86)
    int_431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 8), list_423, int_431)
    # Adding element type (line 86)
    int_432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 8), list_423, int_432)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 11), list_382, list_423)
    # Adding element type (line 81)
    
    # Obtaining an instance of the builtin type 'list' (line 87)
    list_433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 87)
    # Adding element type (line 87)
    int_434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 8), list_433, int_434)
    # Adding element type (line 87)
    int_435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 8), list_433, int_435)
    # Adding element type (line 87)
    int_436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 8), list_433, int_436)
    # Adding element type (line 87)
    int_437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 8), list_433, int_437)
    # Adding element type (line 87)
    int_438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 8), list_433, int_438)
    # Adding element type (line 87)
    int_439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 8), list_433, int_439)
    # Adding element type (line 87)
    int_440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 8), list_433, int_440)
    # Adding element type (line 87)
    int_441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 8), list_433, int_441)
    # Adding element type (line 87)
    int_442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 8), list_433, int_442)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 11), list_382, list_433)
    # Adding element type (line 81)
    
    # Obtaining an instance of the builtin type 'list' (line 88)
    list_443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 88)
    # Adding element type (line 88)
    int_444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 8), list_443, int_444)
    # Adding element type (line 88)
    int_445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 8), list_443, int_445)
    # Adding element type (line 88)
    int_446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 8), list_443, int_446)
    # Adding element type (line 88)
    int_447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 8), list_443, int_447)
    # Adding element type (line 88)
    int_448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 8), list_443, int_448)
    # Adding element type (line 88)
    int_449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 8), list_443, int_449)
    # Adding element type (line 88)
    int_450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 8), list_443, int_450)
    # Adding element type (line 88)
    int_451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 8), list_443, int_451)
    # Adding element type (line 88)
    int_452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 8), list_443, int_452)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 11), list_382, list_443)
    # Adding element type (line 81)
    
    # Obtaining an instance of the builtin type 'list' (line 89)
    list_453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 89)
    # Adding element type (line 89)
    int_454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 8), list_453, int_454)
    # Adding element type (line 89)
    int_455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 8), list_453, int_455)
    # Adding element type (line 89)
    int_456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 8), list_453, int_456)
    # Adding element type (line 89)
    int_457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 8), list_453, int_457)
    # Adding element type (line 89)
    int_458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 8), list_453, int_458)
    # Adding element type (line 89)
    int_459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 8), list_453, int_459)
    # Adding element type (line 89)
    int_460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 8), list_453, int_460)
    # Adding element type (line 89)
    int_461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 8), list_453, int_461)
    # Adding element type (line 89)
    int_462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 8), list_453, int_462)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 11), list_382, list_453)
    # Adding element type (line 81)
    
    # Obtaining an instance of the builtin type 'list' (line 90)
    list_463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 90)
    # Adding element type (line 90)
    int_464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 9), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 8), list_463, int_464)
    # Adding element type (line 90)
    int_465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 8), list_463, int_465)
    # Adding element type (line 90)
    int_466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 15), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 8), list_463, int_466)
    # Adding element type (line 90)
    int_467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 18), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 8), list_463, int_467)
    # Adding element type (line 90)
    int_468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 8), list_463, int_468)
    # Adding element type (line 90)
    int_469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 24), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 8), list_463, int_469)
    # Adding element type (line 90)
    int_470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 27), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 8), list_463, int_470)
    # Adding element type (line 90)
    int_471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 8), list_463, int_471)
    # Adding element type (line 90)
    int_472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 33), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 8), list_463, int_472)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 11), list_382, list_463)
    
    # Assigning a type to the variable 'grid' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'grid', list_382)
    
    
    # Call to solve_sudoku(...): (line 92)
    # Processing the call arguments (line 92)
    
    # Obtaining an instance of the builtin type 'tuple' (line 92)
    tuple_474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 34), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 92)
    # Adding element type (line 92)
    int_475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 34), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 34), tuple_474, int_475)
    # Adding element type (line 92)
    int_476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 34), tuple_474, int_476)
    
    # Getting the type of 'grid' (line 92)
    grid_477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 41), 'grid', False)
    # Processing the call keyword arguments (line 92)
    kwargs_478 = {}
    # Getting the type of 'solve_sudoku' (line 92)
    solve_sudoku_473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 20), 'solve_sudoku', False)
    # Calling solve_sudoku(args, kwargs) (line 92)
    solve_sudoku_call_result_479 = invoke(stypy.reporting.localization.Localization(__file__, 92, 20), solve_sudoku_473, *[tuple_474, grid_477], **kwargs_478)
    
    # Testing if the for loop is going to be iterated (line 92)
    # Testing the type of a for loop iterable (line 92)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 92, 4), solve_sudoku_call_result_479)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 92, 4), solve_sudoku_call_result_479):
        # Getting the type of the for loop variable (line 92)
        for_loop_var_480 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 92, 4), solve_sudoku_call_result_479)
        # Assigning a type to the variable 'solution' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'solution', for_loop_var_480)
        # SSA begins for a for statement (line 92)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 93):
        
        # Assigning a Call to a Name (line 93):
        
        # Call to join(...): (line 93)
        # Processing the call arguments (line 93)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 93, 22, True)
        # Calculating comprehension expression
        # Getting the type of 'solution' (line 93)
        solution_487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 38), 'solution', False)
        comprehension_488 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 22), solution_487)
        # Assigning a type to the variable 's' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 22), 's', comprehension_488)
        
        # Call to str(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 's' (line 93)
        s_484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 26), 's', False)
        # Processing the call keyword arguments (line 93)
        kwargs_485 = {}
        # Getting the type of 'str' (line 93)
        str_483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 22), 'str', False)
        # Calling str(args, kwargs) (line 93)
        str_call_result_486 = invoke(stypy.reporting.localization.Localization(__file__, 93, 22), str_483, *[s_484], **kwargs_485)
        
        list_489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 22), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 22), list_489, str_call_result_486)
        # Processing the call keyword arguments (line 93)
        kwargs_490 = {}
        str_481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 12), 'str', '\n')
        # Obtaining the member 'join' of a type (line 93)
        join_482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 12), str_481, 'join')
        # Calling join(args, kwargs) (line 93)
        join_call_result_491 = invoke(stypy.reporting.localization.Localization(__file__, 93, 12), join_482, *[list_489], **kwargs_490)
        
        # Assigning a type to the variable 'a' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'a', join_call_result_491)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'main(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'main' in the type store
    # Getting the type of 'stypy_return_type' (line 80)
    stypy_return_type_492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_492)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'main'
    return stypy_return_type_492

# Assigning a type to the variable 'main' (line 80)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'main', main)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 97, 0, False)
    
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

    
    
    # Call to range(...): (line 98)
    # Processing the call arguments (line 98)
    int_494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 19), 'int')
    # Processing the call keyword arguments (line 98)
    kwargs_495 = {}
    # Getting the type of 'range' (line 98)
    range_493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 13), 'range', False)
    # Calling range(args, kwargs) (line 98)
    range_call_result_496 = invoke(stypy.reporting.localization.Localization(__file__, 98, 13), range_493, *[int_494], **kwargs_495)
    
    # Testing if the for loop is going to be iterated (line 98)
    # Testing the type of a for loop iterable (line 98)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 98, 4), range_call_result_496)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 98, 4), range_call_result_496):
        # Getting the type of the for loop variable (line 98)
        for_loop_var_497 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 98, 4), range_call_result_496)
        # Assigning a type to the variable 'i' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'i', for_loop_var_497)
        # SSA begins for a for statement (line 98)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to main(...): (line 99)
        # Processing the call keyword arguments (line 99)
        kwargs_499 = {}
        # Getting the type of 'main' (line 99)
        main_498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'main', False)
        # Calling main(args, kwargs) (line 99)
        main_call_result_500 = invoke(stypy.reporting.localization.Localization(__file__, 99, 8), main_498, *[], **kwargs_499)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'True' (line 100)
    True_501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'stypy_return_type', True_501)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 97)
    stypy_return_type_502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_502)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_502

# Assigning a type to the variable 'run' (line 97)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 0), 'run', run)

# Call to run(...): (line 103)
# Processing the call keyword arguments (line 103)
kwargs_504 = {}
# Getting the type of 'run' (line 103)
run_503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 0), 'run', False)
# Calling run(args, kwargs) (line 103)
run_call_result_505 = invoke(stypy.reporting.localization.Localization(__file__, 103, 0), run_503, *[], **kwargs_504)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
