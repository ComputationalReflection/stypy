
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from math import sqrt, ceil
2: from sys import argv
3: 
4: 
5: def sieveOfAtkin(end):
6:     '''sieveOfAtkin(end): return a list of all the prime numbers <end
7:     using the Sieve of Atkin.'''
8:     # Code by Steve Krenzel, <Sgk284@gmail.com>, improved
9:     # Code: http://krenzel.info/?p=83
10:     # Info: http://en.wikipedia.org/wiki/Sieve_of_Atkin
11:     assert end > 0, "end must be >0"
12:     lng = ((end // 2) - 1 + end % 2)
13:     sieve = [False] * (lng + 1)
14: 
15:     x_max, x2, xd = int(sqrt((end - 1) / 4.0)), 0, 4
16:     for xd in xrange(4, 8 * x_max + 2, 8):
17:         x2 += xd
18:         y_max = int(sqrt(end - x2))
19:         n, n_diff = x2 + y_max * y_max, (y_max << 1) - 1
20:         if not (n & 1):
21:             n -= n_diff
22:             n_diff -= 2
23:         for d in xrange((n_diff - 1) << 1, -1, -8):
24:             m = n % 12
25:             if m == 1 or m == 5:
26:                 m = n >> 1
27:                 sieve[m] = not sieve[m]
28:             n -= d
29: 
30:     x_max, x2, xd = int(sqrt((end - 1) / 3.0)), 0, 3
31:     for xd in xrange(3, 6 * x_max + 2, 6):
32:         x2 += xd
33:         y_max = int(sqrt(end - x2))
34:         n, n_diff = x2 + y_max * y_max, (y_max << 1) - 1
35:         if not (n & 1):
36:             n -= n_diff
37:             n_diff -= 2
38:         for d in xrange((n_diff - 1) << 1, -1, -8):
39:             if n % 12 == 7:
40:                 m = n >> 1
41:                 sieve[m] = not sieve[m]
42:             n -= d
43: 
44:     x_max, y_min, x2, xd = int((2 + sqrt(4 - 8 * (1 - end))) / 4), -1, 0, 3
45:     for x in xrange(1, x_max + 1):
46:         x2 += xd
47:         xd += 6
48:         if x2 >= end: y_min = (((int(ceil(sqrt(x2 - end))) - 1) << 1) - 2) << 1
49:         n, n_diff = ((x * x + x) << 1) - 1, (((x - 1) << 1) - 2) << 1
50:         for d in xrange(n_diff, y_min, -8):
51:             if n % 12 == 11:
52:                 m = n >> 1
53:                 sieve[m] = not sieve[m]
54:             n += d
55: 
56:     primes = [2, 3]
57:     if end <= 3:
58:         return primes[:max(0, end - 2)]
59: 
60:     for n in xrange(5 >> 1, (int(sqrt(end)) + 1) >> 1):
61:         if sieve[n]:
62:             primes.append((n << 1) + 1)
63:             aux = (n << 1) + 1
64:             aux *= aux
65:             for k in xrange(aux, end, 2 * aux):
66:                 sieve[k >> 1] = False
67: 
68:     s = int(sqrt(end)) + 1
69:     if s % 2 == 0:
70:         s += 1
71:     primes.extend([i for i in xrange(s, end, 2) if sieve[i >> 1]])
72: 
73:     return primes
74: 
75: 
76: def sieveOfEratostenes(n):
77:     '''sieveOfEratostenes(n): return the list of the primes < n.'''
78:     # Code from: <dickinsm@gmail.com>, Nov 30 2006
79:     # http://groups.google.com/group/comp.lang.python/msg/f1f10ced88c68c2d
80:     if n <= 2:
81:         return []
82:     sieve = range(3, n, 2)
83:     top = len(sieve)
84:     for si in sieve:
85:         if si:
86:             bottom = (si * si - 3) // 2
87:             if bottom >= top:
88:                 break
89:             sieve[bottom::si] = [0] * -((bottom - top) // si)
90:     return [2] + [el for el in sieve if el]
91: 
92: 
93: def run():
94:     # The Sieve of Atkin is supposed to be faster for big n.
95: 
96:     n = 10000000  # int(argv[1])
97:     # print "n:", n
98: 
99:     # if argv[2] == "1":
100:     ##    print "Sieve of Atkin"
101:     r = sieveOfAtkin(n)
102:     ##    print len(r)
103:     # else:
104:     ##    print "Sieve of Eratostenes"
105:     r = sieveOfEratostenes(n)
106:     ##    print len(r)
107: 
108:     # if argv[3] == "1":
109:     #    print r
110: 
111:     return True
112: 
113: 
114: run()
115: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from math import sqrt, ceil' statement (line 1)
try:
    from math import sqrt, ceil

except:
    sqrt = UndefinedType
    ceil = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'math', None, module_type_store, ['sqrt', 'ceil'], [sqrt, ceil])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'from sys import argv' statement (line 2)
try:
    from sys import argv

except:
    argv = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'sys', None, module_type_store, ['argv'], [argv])


@norecursion
def sieveOfAtkin(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'sieveOfAtkin'
    module_type_store = module_type_store.open_function_context('sieveOfAtkin', 5, 0, False)
    
    # Passed parameters checking function
    sieveOfAtkin.stypy_localization = localization
    sieveOfAtkin.stypy_type_of_self = None
    sieveOfAtkin.stypy_type_store = module_type_store
    sieveOfAtkin.stypy_function_name = 'sieveOfAtkin'
    sieveOfAtkin.stypy_param_names_list = ['end']
    sieveOfAtkin.stypy_varargs_param_name = None
    sieveOfAtkin.stypy_kwargs_param_name = None
    sieveOfAtkin.stypy_call_defaults = defaults
    sieveOfAtkin.stypy_call_varargs = varargs
    sieveOfAtkin.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'sieveOfAtkin', ['end'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'sieveOfAtkin', localization, ['end'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'sieveOfAtkin(...)' code ##################

    str_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, (-1)), 'str', 'sieveOfAtkin(end): return a list of all the prime numbers <end\n    using the Sieve of Atkin.')
    # Evaluating assert statement condition
    
    # Getting the type of 'end' (line 11)
    end_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 11), 'end')
    int_19 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 17), 'int')
    # Applying the binary operator '>' (line 11)
    result_gt_20 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 11), '>', end_18, int_19)
    
    assert_21 = result_gt_20
    # Assigning a type to the variable 'assert_21' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'assert_21', result_gt_20)
    
    # Assigning a BinOp to a Name (line 12):
    
    # Assigning a BinOp to a Name (line 12):
    # Getting the type of 'end' (line 12)
    end_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'end')
    int_23 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 19), 'int')
    # Applying the binary operator '//' (line 12)
    result_floordiv_24 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 12), '//', end_22, int_23)
    
    int_25 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 24), 'int')
    # Applying the binary operator '-' (line 12)
    result_sub_26 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 11), '-', result_floordiv_24, int_25)
    
    # Getting the type of 'end' (line 12)
    end_27 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 28), 'end')
    int_28 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 34), 'int')
    # Applying the binary operator '%' (line 12)
    result_mod_29 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 28), '%', end_27, int_28)
    
    # Applying the binary operator '+' (line 12)
    result_add_30 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 26), '+', result_sub_26, result_mod_29)
    
    # Assigning a type to the variable 'lng' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'lng', result_add_30)
    
    # Assigning a BinOp to a Name (line 13):
    
    # Assigning a BinOp to a Name (line 13):
    
    # Obtaining an instance of the builtin type 'list' (line 13)
    list_31 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 13)
    # Adding element type (line 13)
    # Getting the type of 'False' (line 13)
    False_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 13), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 12), list_31, False_32)
    
    # Getting the type of 'lng' (line 13)
    lng_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 23), 'lng')
    int_34 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 29), 'int')
    # Applying the binary operator '+' (line 13)
    result_add_35 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 23), '+', lng_33, int_34)
    
    # Applying the binary operator '*' (line 13)
    result_mul_36 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 12), '*', list_31, result_add_35)
    
    # Assigning a type to the variable 'sieve' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'sieve', result_mul_36)
    
    # Assigning a Tuple to a Tuple (line 15):
    
    # Assigning a Call to a Name (line 15):
    
    # Call to int(...): (line 15)
    # Processing the call arguments (line 15)
    
    # Call to sqrt(...): (line 15)
    # Processing the call arguments (line 15)
    # Getting the type of 'end' (line 15)
    end_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 30), 'end', False)
    int_40 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 36), 'int')
    # Applying the binary operator '-' (line 15)
    result_sub_41 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 30), '-', end_39, int_40)
    
    float_42 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 41), 'float')
    # Applying the binary operator 'div' (line 15)
    result_div_43 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 29), 'div', result_sub_41, float_42)
    
    # Processing the call keyword arguments (line 15)
    kwargs_44 = {}
    # Getting the type of 'sqrt' (line 15)
    sqrt_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 24), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 15)
    sqrt_call_result_45 = invoke(stypy.reporting.localization.Localization(__file__, 15, 24), sqrt_38, *[result_div_43], **kwargs_44)
    
    # Processing the call keyword arguments (line 15)
    kwargs_46 = {}
    # Getting the type of 'int' (line 15)
    int_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 20), 'int', False)
    # Calling int(args, kwargs) (line 15)
    int_call_result_47 = invoke(stypy.reporting.localization.Localization(__file__, 15, 20), int_37, *[sqrt_call_result_45], **kwargs_46)
    
    # Assigning a type to the variable 'tuple_assignment_1' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'tuple_assignment_1', int_call_result_47)
    
    # Assigning a Num to a Name (line 15):
    int_48 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 48), 'int')
    # Assigning a type to the variable 'tuple_assignment_2' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'tuple_assignment_2', int_48)
    
    # Assigning a Num to a Name (line 15):
    int_49 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 51), 'int')
    # Assigning a type to the variable 'tuple_assignment_3' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'tuple_assignment_3', int_49)
    
    # Assigning a Name to a Name (line 15):
    # Getting the type of 'tuple_assignment_1' (line 15)
    tuple_assignment_1_50 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'tuple_assignment_1')
    # Assigning a type to the variable 'x_max' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'x_max', tuple_assignment_1_50)
    
    # Assigning a Name to a Name (line 15):
    # Getting the type of 'tuple_assignment_2' (line 15)
    tuple_assignment_2_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'tuple_assignment_2')
    # Assigning a type to the variable 'x2' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 11), 'x2', tuple_assignment_2_51)
    
    # Assigning a Name to a Name (line 15):
    # Getting the type of 'tuple_assignment_3' (line 15)
    tuple_assignment_3_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'tuple_assignment_3')
    # Assigning a type to the variable 'xd' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 15), 'xd', tuple_assignment_3_52)
    
    
    # Call to xrange(...): (line 16)
    # Processing the call arguments (line 16)
    int_54 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 21), 'int')
    int_55 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 24), 'int')
    # Getting the type of 'x_max' (line 16)
    x_max_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 28), 'x_max', False)
    # Applying the binary operator '*' (line 16)
    result_mul_57 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 24), '*', int_55, x_max_56)
    
    int_58 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 36), 'int')
    # Applying the binary operator '+' (line 16)
    result_add_59 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 24), '+', result_mul_57, int_58)
    
    int_60 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 39), 'int')
    # Processing the call keyword arguments (line 16)
    kwargs_61 = {}
    # Getting the type of 'xrange' (line 16)
    xrange_53 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 14), 'xrange', False)
    # Calling xrange(args, kwargs) (line 16)
    xrange_call_result_62 = invoke(stypy.reporting.localization.Localization(__file__, 16, 14), xrange_53, *[int_54, result_add_59, int_60], **kwargs_61)
    
    # Assigning a type to the variable 'xrange_call_result_62' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'xrange_call_result_62', xrange_call_result_62)
    # Testing if the for loop is going to be iterated (line 16)
    # Testing the type of a for loop iterable (line 16)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 16, 4), xrange_call_result_62)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 16, 4), xrange_call_result_62):
        # Getting the type of the for loop variable (line 16)
        for_loop_var_63 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 16, 4), xrange_call_result_62)
        # Assigning a type to the variable 'xd' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'xd', for_loop_var_63)
        # SSA begins for a for statement (line 16)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'x2' (line 17)
        x2_64 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'x2')
        # Getting the type of 'xd' (line 17)
        xd_65 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 14), 'xd')
        # Applying the binary operator '+=' (line 17)
        result_iadd_66 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 8), '+=', x2_64, xd_65)
        # Assigning a type to the variable 'x2' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'x2', result_iadd_66)
        
        
        # Assigning a Call to a Name (line 18):
        
        # Assigning a Call to a Name (line 18):
        
        # Call to int(...): (line 18)
        # Processing the call arguments (line 18)
        
        # Call to sqrt(...): (line 18)
        # Processing the call arguments (line 18)
        # Getting the type of 'end' (line 18)
        end_69 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 25), 'end', False)
        # Getting the type of 'x2' (line 18)
        x2_70 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 31), 'x2', False)
        # Applying the binary operator '-' (line 18)
        result_sub_71 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 25), '-', end_69, x2_70)
        
        # Processing the call keyword arguments (line 18)
        kwargs_72 = {}
        # Getting the type of 'sqrt' (line 18)
        sqrt_68 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 20), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 18)
        sqrt_call_result_73 = invoke(stypy.reporting.localization.Localization(__file__, 18, 20), sqrt_68, *[result_sub_71], **kwargs_72)
        
        # Processing the call keyword arguments (line 18)
        kwargs_74 = {}
        # Getting the type of 'int' (line 18)
        int_67 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 16), 'int', False)
        # Calling int(args, kwargs) (line 18)
        int_call_result_75 = invoke(stypy.reporting.localization.Localization(__file__, 18, 16), int_67, *[sqrt_call_result_73], **kwargs_74)
        
        # Assigning a type to the variable 'y_max' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'y_max', int_call_result_75)
        
        # Assigning a Tuple to a Tuple (line 19):
        
        # Assigning a BinOp to a Name (line 19):
        # Getting the type of 'x2' (line 19)
        x2_76 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 20), 'x2')
        # Getting the type of 'y_max' (line 19)
        y_max_77 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 25), 'y_max')
        # Getting the type of 'y_max' (line 19)
        y_max_78 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 33), 'y_max')
        # Applying the binary operator '*' (line 19)
        result_mul_79 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 25), '*', y_max_77, y_max_78)
        
        # Applying the binary operator '+' (line 19)
        result_add_80 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 20), '+', x2_76, result_mul_79)
        
        # Assigning a type to the variable 'tuple_assignment_4' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'tuple_assignment_4', result_add_80)
        
        # Assigning a BinOp to a Name (line 19):
        # Getting the type of 'y_max' (line 19)
        y_max_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 41), 'y_max')
        int_82 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 50), 'int')
        # Applying the binary operator '<<' (line 19)
        result_lshift_83 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 41), '<<', y_max_81, int_82)
        
        int_84 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 55), 'int')
        # Applying the binary operator '-' (line 19)
        result_sub_85 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 40), '-', result_lshift_83, int_84)
        
        # Assigning a type to the variable 'tuple_assignment_5' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'tuple_assignment_5', result_sub_85)
        
        # Assigning a Name to a Name (line 19):
        # Getting the type of 'tuple_assignment_4' (line 19)
        tuple_assignment_4_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'tuple_assignment_4')
        # Assigning a type to the variable 'n' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'n', tuple_assignment_4_86)
        
        # Assigning a Name to a Name (line 19):
        # Getting the type of 'tuple_assignment_5' (line 19)
        tuple_assignment_5_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'tuple_assignment_5')
        # Assigning a type to the variable 'n_diff' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 11), 'n_diff', tuple_assignment_5_87)
        
        # Getting the type of 'n' (line 20)
        n_88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'n')
        int_89 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 20), 'int')
        # Applying the binary operator '&' (line 20)
        result_and__90 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 16), '&', n_88, int_89)
        
        # Applying the 'not' unary operator (line 20)
        result_not__91 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 11), 'not', result_and__90)
        
        # Testing if the type of an if condition is none (line 20)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 20, 8), result_not__91):
            pass
        else:
            
            # Testing the type of an if condition (line 20)
            if_condition_92 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 20, 8), result_not__91)
            # Assigning a type to the variable 'if_condition_92' (line 20)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'if_condition_92', if_condition_92)
            # SSA begins for if statement (line 20)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'n' (line 21)
            n_93 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'n')
            # Getting the type of 'n_diff' (line 21)
            n_diff_94 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 17), 'n_diff')
            # Applying the binary operator '-=' (line 21)
            result_isub_95 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 12), '-=', n_93, n_diff_94)
            # Assigning a type to the variable 'n' (line 21)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'n', result_isub_95)
            
            
            # Getting the type of 'n_diff' (line 22)
            n_diff_96 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'n_diff')
            int_97 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 22), 'int')
            # Applying the binary operator '-=' (line 22)
            result_isub_98 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 12), '-=', n_diff_96, int_97)
            # Assigning a type to the variable 'n_diff' (line 22)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'n_diff', result_isub_98)
            
            # SSA join for if statement (line 20)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to xrange(...): (line 23)
        # Processing the call arguments (line 23)
        # Getting the type of 'n_diff' (line 23)
        n_diff_100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 25), 'n_diff', False)
        int_101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 34), 'int')
        # Applying the binary operator '-' (line 23)
        result_sub_102 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 25), '-', n_diff_100, int_101)
        
        int_103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 40), 'int')
        # Applying the binary operator '<<' (line 23)
        result_lshift_104 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 24), '<<', result_sub_102, int_103)
        
        int_105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 43), 'int')
        int_106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 47), 'int')
        # Processing the call keyword arguments (line 23)
        kwargs_107 = {}
        # Getting the type of 'xrange' (line 23)
        xrange_99 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 23)
        xrange_call_result_108 = invoke(stypy.reporting.localization.Localization(__file__, 23, 17), xrange_99, *[result_lshift_104, int_105, int_106], **kwargs_107)
        
        # Assigning a type to the variable 'xrange_call_result_108' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'xrange_call_result_108', xrange_call_result_108)
        # Testing if the for loop is going to be iterated (line 23)
        # Testing the type of a for loop iterable (line 23)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 23, 8), xrange_call_result_108)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 23, 8), xrange_call_result_108):
            # Getting the type of the for loop variable (line 23)
            for_loop_var_109 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 23, 8), xrange_call_result_108)
            # Assigning a type to the variable 'd' (line 23)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'd', for_loop_var_109)
            # SSA begins for a for statement (line 23)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a BinOp to a Name (line 24):
            
            # Assigning a BinOp to a Name (line 24):
            # Getting the type of 'n' (line 24)
            n_110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 16), 'n')
            int_111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 20), 'int')
            # Applying the binary operator '%' (line 24)
            result_mod_112 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 16), '%', n_110, int_111)
            
            # Assigning a type to the variable 'm' (line 24)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'm', result_mod_112)
            
            # Evaluating a boolean operation
            
            # Getting the type of 'm' (line 25)
            m_113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'm')
            int_114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 20), 'int')
            # Applying the binary operator '==' (line 25)
            result_eq_115 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 15), '==', m_113, int_114)
            
            
            # Getting the type of 'm' (line 25)
            m_116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 25), 'm')
            int_117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 30), 'int')
            # Applying the binary operator '==' (line 25)
            result_eq_118 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 25), '==', m_116, int_117)
            
            # Applying the binary operator 'or' (line 25)
            result_or_keyword_119 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 15), 'or', result_eq_115, result_eq_118)
            
            # Testing if the type of an if condition is none (line 25)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 25, 12), result_or_keyword_119):
                pass
            else:
                
                # Testing the type of an if condition (line 25)
                if_condition_120 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 25, 12), result_or_keyword_119)
                # Assigning a type to the variable 'if_condition_120' (line 25)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'if_condition_120', if_condition_120)
                # SSA begins for if statement (line 25)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a BinOp to a Name (line 26):
                
                # Assigning a BinOp to a Name (line 26):
                # Getting the type of 'n' (line 26)
                n_121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 20), 'n')
                int_122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 25), 'int')
                # Applying the binary operator '>>' (line 26)
                result_rshift_123 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 20), '>>', n_121, int_122)
                
                # Assigning a type to the variable 'm' (line 26)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 16), 'm', result_rshift_123)
                
                # Assigning a UnaryOp to a Subscript (line 27):
                
                # Assigning a UnaryOp to a Subscript (line 27):
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'm' (line 27)
                m_124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 37), 'm')
                # Getting the type of 'sieve' (line 27)
                sieve_125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 31), 'sieve')
                # Obtaining the member '__getitem__' of a type (line 27)
                getitem___126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 31), sieve_125, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 27)
                subscript_call_result_127 = invoke(stypy.reporting.localization.Localization(__file__, 27, 31), getitem___126, m_124)
                
                # Applying the 'not' unary operator (line 27)
                result_not__128 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 27), 'not', subscript_call_result_127)
                
                # Getting the type of 'sieve' (line 27)
                sieve_129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 16), 'sieve')
                # Getting the type of 'm' (line 27)
                m_130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 22), 'm')
                # Storing an element on a container (line 27)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 16), sieve_129, (m_130, result_not__128))
                # SSA join for if statement (line 25)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'n' (line 28)
            n_131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'n')
            # Getting the type of 'd' (line 28)
            d_132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 17), 'd')
            # Applying the binary operator '-=' (line 28)
            result_isub_133 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 12), '-=', n_131, d_132)
            # Assigning a type to the variable 'n' (line 28)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'n', result_isub_133)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Assigning a Tuple to a Tuple (line 30):
    
    # Assigning a Call to a Name (line 30):
    
    # Call to int(...): (line 30)
    # Processing the call arguments (line 30)
    
    # Call to sqrt(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'end' (line 30)
    end_136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 30), 'end', False)
    int_137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 36), 'int')
    # Applying the binary operator '-' (line 30)
    result_sub_138 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 30), '-', end_136, int_137)
    
    float_139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 41), 'float')
    # Applying the binary operator 'div' (line 30)
    result_div_140 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 29), 'div', result_sub_138, float_139)
    
    # Processing the call keyword arguments (line 30)
    kwargs_141 = {}
    # Getting the type of 'sqrt' (line 30)
    sqrt_135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 24), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 30)
    sqrt_call_result_142 = invoke(stypy.reporting.localization.Localization(__file__, 30, 24), sqrt_135, *[result_div_140], **kwargs_141)
    
    # Processing the call keyword arguments (line 30)
    kwargs_143 = {}
    # Getting the type of 'int' (line 30)
    int_134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 20), 'int', False)
    # Calling int(args, kwargs) (line 30)
    int_call_result_144 = invoke(stypy.reporting.localization.Localization(__file__, 30, 20), int_134, *[sqrt_call_result_142], **kwargs_143)
    
    # Assigning a type to the variable 'tuple_assignment_6' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'tuple_assignment_6', int_call_result_144)
    
    # Assigning a Num to a Name (line 30):
    int_145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 48), 'int')
    # Assigning a type to the variable 'tuple_assignment_7' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'tuple_assignment_7', int_145)
    
    # Assigning a Num to a Name (line 30):
    int_146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 51), 'int')
    # Assigning a type to the variable 'tuple_assignment_8' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'tuple_assignment_8', int_146)
    
    # Assigning a Name to a Name (line 30):
    # Getting the type of 'tuple_assignment_6' (line 30)
    tuple_assignment_6_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'tuple_assignment_6')
    # Assigning a type to the variable 'x_max' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'x_max', tuple_assignment_6_147)
    
    # Assigning a Name to a Name (line 30):
    # Getting the type of 'tuple_assignment_7' (line 30)
    tuple_assignment_7_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'tuple_assignment_7')
    # Assigning a type to the variable 'x2' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 11), 'x2', tuple_assignment_7_148)
    
    # Assigning a Name to a Name (line 30):
    # Getting the type of 'tuple_assignment_8' (line 30)
    tuple_assignment_8_149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'tuple_assignment_8')
    # Assigning a type to the variable 'xd' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 15), 'xd', tuple_assignment_8_149)
    
    
    # Call to xrange(...): (line 31)
    # Processing the call arguments (line 31)
    int_151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 21), 'int')
    int_152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 24), 'int')
    # Getting the type of 'x_max' (line 31)
    x_max_153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 28), 'x_max', False)
    # Applying the binary operator '*' (line 31)
    result_mul_154 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 24), '*', int_152, x_max_153)
    
    int_155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 36), 'int')
    # Applying the binary operator '+' (line 31)
    result_add_156 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 24), '+', result_mul_154, int_155)
    
    int_157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 39), 'int')
    # Processing the call keyword arguments (line 31)
    kwargs_158 = {}
    # Getting the type of 'xrange' (line 31)
    xrange_150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 14), 'xrange', False)
    # Calling xrange(args, kwargs) (line 31)
    xrange_call_result_159 = invoke(stypy.reporting.localization.Localization(__file__, 31, 14), xrange_150, *[int_151, result_add_156, int_157], **kwargs_158)
    
    # Assigning a type to the variable 'xrange_call_result_159' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'xrange_call_result_159', xrange_call_result_159)
    # Testing if the for loop is going to be iterated (line 31)
    # Testing the type of a for loop iterable (line 31)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 31, 4), xrange_call_result_159)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 31, 4), xrange_call_result_159):
        # Getting the type of the for loop variable (line 31)
        for_loop_var_160 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 31, 4), xrange_call_result_159)
        # Assigning a type to the variable 'xd' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'xd', for_loop_var_160)
        # SSA begins for a for statement (line 31)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'x2' (line 32)
        x2_161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'x2')
        # Getting the type of 'xd' (line 32)
        xd_162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 14), 'xd')
        # Applying the binary operator '+=' (line 32)
        result_iadd_163 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 8), '+=', x2_161, xd_162)
        # Assigning a type to the variable 'x2' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'x2', result_iadd_163)
        
        
        # Assigning a Call to a Name (line 33):
        
        # Assigning a Call to a Name (line 33):
        
        # Call to int(...): (line 33)
        # Processing the call arguments (line 33)
        
        # Call to sqrt(...): (line 33)
        # Processing the call arguments (line 33)
        # Getting the type of 'end' (line 33)
        end_166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 25), 'end', False)
        # Getting the type of 'x2' (line 33)
        x2_167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 31), 'x2', False)
        # Applying the binary operator '-' (line 33)
        result_sub_168 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 25), '-', end_166, x2_167)
        
        # Processing the call keyword arguments (line 33)
        kwargs_169 = {}
        # Getting the type of 'sqrt' (line 33)
        sqrt_165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 20), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 33)
        sqrt_call_result_170 = invoke(stypy.reporting.localization.Localization(__file__, 33, 20), sqrt_165, *[result_sub_168], **kwargs_169)
        
        # Processing the call keyword arguments (line 33)
        kwargs_171 = {}
        # Getting the type of 'int' (line 33)
        int_164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 16), 'int', False)
        # Calling int(args, kwargs) (line 33)
        int_call_result_172 = invoke(stypy.reporting.localization.Localization(__file__, 33, 16), int_164, *[sqrt_call_result_170], **kwargs_171)
        
        # Assigning a type to the variable 'y_max' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'y_max', int_call_result_172)
        
        # Assigning a Tuple to a Tuple (line 34):
        
        # Assigning a BinOp to a Name (line 34):
        # Getting the type of 'x2' (line 34)
        x2_173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 20), 'x2')
        # Getting the type of 'y_max' (line 34)
        y_max_174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 25), 'y_max')
        # Getting the type of 'y_max' (line 34)
        y_max_175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 33), 'y_max')
        # Applying the binary operator '*' (line 34)
        result_mul_176 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 25), '*', y_max_174, y_max_175)
        
        # Applying the binary operator '+' (line 34)
        result_add_177 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 20), '+', x2_173, result_mul_176)
        
        # Assigning a type to the variable 'tuple_assignment_9' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'tuple_assignment_9', result_add_177)
        
        # Assigning a BinOp to a Name (line 34):
        # Getting the type of 'y_max' (line 34)
        y_max_178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 41), 'y_max')
        int_179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 50), 'int')
        # Applying the binary operator '<<' (line 34)
        result_lshift_180 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 41), '<<', y_max_178, int_179)
        
        int_181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 55), 'int')
        # Applying the binary operator '-' (line 34)
        result_sub_182 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 40), '-', result_lshift_180, int_181)
        
        # Assigning a type to the variable 'tuple_assignment_10' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'tuple_assignment_10', result_sub_182)
        
        # Assigning a Name to a Name (line 34):
        # Getting the type of 'tuple_assignment_9' (line 34)
        tuple_assignment_9_183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'tuple_assignment_9')
        # Assigning a type to the variable 'n' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'n', tuple_assignment_9_183)
        
        # Assigning a Name to a Name (line 34):
        # Getting the type of 'tuple_assignment_10' (line 34)
        tuple_assignment_10_184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'tuple_assignment_10')
        # Assigning a type to the variable 'n_diff' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 11), 'n_diff', tuple_assignment_10_184)
        
        # Getting the type of 'n' (line 35)
        n_185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 16), 'n')
        int_186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 20), 'int')
        # Applying the binary operator '&' (line 35)
        result_and__187 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 16), '&', n_185, int_186)
        
        # Applying the 'not' unary operator (line 35)
        result_not__188 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 11), 'not', result_and__187)
        
        # Testing if the type of an if condition is none (line 35)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 35, 8), result_not__188):
            pass
        else:
            
            # Testing the type of an if condition (line 35)
            if_condition_189 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 35, 8), result_not__188)
            # Assigning a type to the variable 'if_condition_189' (line 35)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'if_condition_189', if_condition_189)
            # SSA begins for if statement (line 35)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'n' (line 36)
            n_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'n')
            # Getting the type of 'n_diff' (line 36)
            n_diff_191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 17), 'n_diff')
            # Applying the binary operator '-=' (line 36)
            result_isub_192 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 12), '-=', n_190, n_diff_191)
            # Assigning a type to the variable 'n' (line 36)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'n', result_isub_192)
            
            
            # Getting the type of 'n_diff' (line 37)
            n_diff_193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'n_diff')
            int_194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 22), 'int')
            # Applying the binary operator '-=' (line 37)
            result_isub_195 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 12), '-=', n_diff_193, int_194)
            # Assigning a type to the variable 'n_diff' (line 37)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'n_diff', result_isub_195)
            
            # SSA join for if statement (line 35)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to xrange(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'n_diff' (line 38)
        n_diff_197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 25), 'n_diff', False)
        int_198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 34), 'int')
        # Applying the binary operator '-' (line 38)
        result_sub_199 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 25), '-', n_diff_197, int_198)
        
        int_200 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 40), 'int')
        # Applying the binary operator '<<' (line 38)
        result_lshift_201 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 24), '<<', result_sub_199, int_200)
        
        int_202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 43), 'int')
        int_203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 47), 'int')
        # Processing the call keyword arguments (line 38)
        kwargs_204 = {}
        # Getting the type of 'xrange' (line 38)
        xrange_196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 38)
        xrange_call_result_205 = invoke(stypy.reporting.localization.Localization(__file__, 38, 17), xrange_196, *[result_lshift_201, int_202, int_203], **kwargs_204)
        
        # Assigning a type to the variable 'xrange_call_result_205' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'xrange_call_result_205', xrange_call_result_205)
        # Testing if the for loop is going to be iterated (line 38)
        # Testing the type of a for loop iterable (line 38)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 38, 8), xrange_call_result_205)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 38, 8), xrange_call_result_205):
            # Getting the type of the for loop variable (line 38)
            for_loop_var_206 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 38, 8), xrange_call_result_205)
            # Assigning a type to the variable 'd' (line 38)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'd', for_loop_var_206)
            # SSA begins for a for statement (line 38)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'n' (line 39)
            n_207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 15), 'n')
            int_208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 19), 'int')
            # Applying the binary operator '%' (line 39)
            result_mod_209 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 15), '%', n_207, int_208)
            
            int_210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 25), 'int')
            # Applying the binary operator '==' (line 39)
            result_eq_211 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 15), '==', result_mod_209, int_210)
            
            # Testing if the type of an if condition is none (line 39)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 39, 12), result_eq_211):
                pass
            else:
                
                # Testing the type of an if condition (line 39)
                if_condition_212 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 39, 12), result_eq_211)
                # Assigning a type to the variable 'if_condition_212' (line 39)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'if_condition_212', if_condition_212)
                # SSA begins for if statement (line 39)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a BinOp to a Name (line 40):
                
                # Assigning a BinOp to a Name (line 40):
                # Getting the type of 'n' (line 40)
                n_213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 20), 'n')
                int_214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 25), 'int')
                # Applying the binary operator '>>' (line 40)
                result_rshift_215 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 20), '>>', n_213, int_214)
                
                # Assigning a type to the variable 'm' (line 40)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 16), 'm', result_rshift_215)
                
                # Assigning a UnaryOp to a Subscript (line 41):
                
                # Assigning a UnaryOp to a Subscript (line 41):
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'm' (line 41)
                m_216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 37), 'm')
                # Getting the type of 'sieve' (line 41)
                sieve_217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 31), 'sieve')
                # Obtaining the member '__getitem__' of a type (line 41)
                getitem___218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 31), sieve_217, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 41)
                subscript_call_result_219 = invoke(stypy.reporting.localization.Localization(__file__, 41, 31), getitem___218, m_216)
                
                # Applying the 'not' unary operator (line 41)
                result_not__220 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 27), 'not', subscript_call_result_219)
                
                # Getting the type of 'sieve' (line 41)
                sieve_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 16), 'sieve')
                # Getting the type of 'm' (line 41)
                m_222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 22), 'm')
                # Storing an element on a container (line 41)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 16), sieve_221, (m_222, result_not__220))
                # SSA join for if statement (line 39)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'n' (line 42)
            n_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'n')
            # Getting the type of 'd' (line 42)
            d_224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 17), 'd')
            # Applying the binary operator '-=' (line 42)
            result_isub_225 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 12), '-=', n_223, d_224)
            # Assigning a type to the variable 'n' (line 42)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'n', result_isub_225)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Assigning a Tuple to a Tuple (line 44):
    
    # Assigning a Call to a Name (line 44):
    
    # Call to int(...): (line 44)
    # Processing the call arguments (line 44)
    int_227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 32), 'int')
    
    # Call to sqrt(...): (line 44)
    # Processing the call arguments (line 44)
    int_229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 41), 'int')
    int_230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 45), 'int')
    int_231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 50), 'int')
    # Getting the type of 'end' (line 44)
    end_232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 54), 'end', False)
    # Applying the binary operator '-' (line 44)
    result_sub_233 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 50), '-', int_231, end_232)
    
    # Applying the binary operator '*' (line 44)
    result_mul_234 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 45), '*', int_230, result_sub_233)
    
    # Applying the binary operator '-' (line 44)
    result_sub_235 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 41), '-', int_229, result_mul_234)
    
    # Processing the call keyword arguments (line 44)
    kwargs_236 = {}
    # Getting the type of 'sqrt' (line 44)
    sqrt_228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 36), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 44)
    sqrt_call_result_237 = invoke(stypy.reporting.localization.Localization(__file__, 44, 36), sqrt_228, *[result_sub_235], **kwargs_236)
    
    # Applying the binary operator '+' (line 44)
    result_add_238 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 32), '+', int_227, sqrt_call_result_237)
    
    int_239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 63), 'int')
    # Applying the binary operator 'div' (line 44)
    result_div_240 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 31), 'div', result_add_238, int_239)
    
    # Processing the call keyword arguments (line 44)
    kwargs_241 = {}
    # Getting the type of 'int' (line 44)
    int_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 27), 'int', False)
    # Calling int(args, kwargs) (line 44)
    int_call_result_242 = invoke(stypy.reporting.localization.Localization(__file__, 44, 27), int_226, *[result_div_240], **kwargs_241)
    
    # Assigning a type to the variable 'tuple_assignment_11' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'tuple_assignment_11', int_call_result_242)
    
    # Assigning a Num to a Name (line 44):
    int_243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 67), 'int')
    # Assigning a type to the variable 'tuple_assignment_12' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'tuple_assignment_12', int_243)
    
    # Assigning a Num to a Name (line 44):
    int_244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 71), 'int')
    # Assigning a type to the variable 'tuple_assignment_13' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'tuple_assignment_13', int_244)
    
    # Assigning a Num to a Name (line 44):
    int_245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 74), 'int')
    # Assigning a type to the variable 'tuple_assignment_14' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'tuple_assignment_14', int_245)
    
    # Assigning a Name to a Name (line 44):
    # Getting the type of 'tuple_assignment_11' (line 44)
    tuple_assignment_11_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'tuple_assignment_11')
    # Assigning a type to the variable 'x_max' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'x_max', tuple_assignment_11_246)
    
    # Assigning a Name to a Name (line 44):
    # Getting the type of 'tuple_assignment_12' (line 44)
    tuple_assignment_12_247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'tuple_assignment_12')
    # Assigning a type to the variable 'y_min' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 11), 'y_min', tuple_assignment_12_247)
    
    # Assigning a Name to a Name (line 44):
    # Getting the type of 'tuple_assignment_13' (line 44)
    tuple_assignment_13_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'tuple_assignment_13')
    # Assigning a type to the variable 'x2' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 18), 'x2', tuple_assignment_13_248)
    
    # Assigning a Name to a Name (line 44):
    # Getting the type of 'tuple_assignment_14' (line 44)
    tuple_assignment_14_249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'tuple_assignment_14')
    # Assigning a type to the variable 'xd' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 22), 'xd', tuple_assignment_14_249)
    
    
    # Call to xrange(...): (line 45)
    # Processing the call arguments (line 45)
    int_251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 20), 'int')
    # Getting the type of 'x_max' (line 45)
    x_max_252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 23), 'x_max', False)
    int_253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 31), 'int')
    # Applying the binary operator '+' (line 45)
    result_add_254 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 23), '+', x_max_252, int_253)
    
    # Processing the call keyword arguments (line 45)
    kwargs_255 = {}
    # Getting the type of 'xrange' (line 45)
    xrange_250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 45)
    xrange_call_result_256 = invoke(stypy.reporting.localization.Localization(__file__, 45, 13), xrange_250, *[int_251, result_add_254], **kwargs_255)
    
    # Assigning a type to the variable 'xrange_call_result_256' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'xrange_call_result_256', xrange_call_result_256)
    # Testing if the for loop is going to be iterated (line 45)
    # Testing the type of a for loop iterable (line 45)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 45, 4), xrange_call_result_256)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 45, 4), xrange_call_result_256):
        # Getting the type of the for loop variable (line 45)
        for_loop_var_257 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 45, 4), xrange_call_result_256)
        # Assigning a type to the variable 'x' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'x', for_loop_var_257)
        # SSA begins for a for statement (line 45)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'x2' (line 46)
        x2_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'x2')
        # Getting the type of 'xd' (line 46)
        xd_259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 14), 'xd')
        # Applying the binary operator '+=' (line 46)
        result_iadd_260 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 8), '+=', x2_258, xd_259)
        # Assigning a type to the variable 'x2' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'x2', result_iadd_260)
        
        
        # Getting the type of 'xd' (line 47)
        xd_261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'xd')
        int_262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 14), 'int')
        # Applying the binary operator '+=' (line 47)
        result_iadd_263 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 8), '+=', xd_261, int_262)
        # Assigning a type to the variable 'xd' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'xd', result_iadd_263)
        
        
        # Getting the type of 'x2' (line 48)
        x2_264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 11), 'x2')
        # Getting the type of 'end' (line 48)
        end_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 17), 'end')
        # Applying the binary operator '>=' (line 48)
        result_ge_266 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 11), '>=', x2_264, end_265)
        
        # Testing if the type of an if condition is none (line 48)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 48, 8), result_ge_266):
            pass
        else:
            
            # Testing the type of an if condition (line 48)
            if_condition_267 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 48, 8), result_ge_266)
            # Assigning a type to the variable 'if_condition_267' (line 48)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'if_condition_267', if_condition_267)
            # SSA begins for if statement (line 48)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 48):
            
            # Assigning a BinOp to a Name (line 48):
            
            # Call to int(...): (line 48)
            # Processing the call arguments (line 48)
            
            # Call to ceil(...): (line 48)
            # Processing the call arguments (line 48)
            
            # Call to sqrt(...): (line 48)
            # Processing the call arguments (line 48)
            # Getting the type of 'x2' (line 48)
            x2_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 47), 'x2', False)
            # Getting the type of 'end' (line 48)
            end_272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 52), 'end', False)
            # Applying the binary operator '-' (line 48)
            result_sub_273 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 47), '-', x2_271, end_272)
            
            # Processing the call keyword arguments (line 48)
            kwargs_274 = {}
            # Getting the type of 'sqrt' (line 48)
            sqrt_270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 42), 'sqrt', False)
            # Calling sqrt(args, kwargs) (line 48)
            sqrt_call_result_275 = invoke(stypy.reporting.localization.Localization(__file__, 48, 42), sqrt_270, *[result_sub_273], **kwargs_274)
            
            # Processing the call keyword arguments (line 48)
            kwargs_276 = {}
            # Getting the type of 'ceil' (line 48)
            ceil_269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 37), 'ceil', False)
            # Calling ceil(args, kwargs) (line 48)
            ceil_call_result_277 = invoke(stypy.reporting.localization.Localization(__file__, 48, 37), ceil_269, *[sqrt_call_result_275], **kwargs_276)
            
            # Processing the call keyword arguments (line 48)
            kwargs_278 = {}
            # Getting the type of 'int' (line 48)
            int_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 33), 'int', False)
            # Calling int(args, kwargs) (line 48)
            int_call_result_279 = invoke(stypy.reporting.localization.Localization(__file__, 48, 33), int_268, *[ceil_call_result_277], **kwargs_278)
            
            int_280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 61), 'int')
            # Applying the binary operator '-' (line 48)
            result_sub_281 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 33), '-', int_call_result_279, int_280)
            
            int_282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 67), 'int')
            # Applying the binary operator '<<' (line 48)
            result_lshift_283 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 32), '<<', result_sub_281, int_282)
            
            int_284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 72), 'int')
            # Applying the binary operator '-' (line 48)
            result_sub_285 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 31), '-', result_lshift_283, int_284)
            
            int_286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 78), 'int')
            # Applying the binary operator '<<' (line 48)
            result_lshift_287 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 30), '<<', result_sub_285, int_286)
            
            # Assigning a type to the variable 'y_min' (line 48)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 22), 'y_min', result_lshift_287)
            # SSA join for if statement (line 48)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Tuple to a Tuple (line 49):
        
        # Assigning a BinOp to a Name (line 49):
        # Getting the type of 'x' (line 49)
        x_288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 22), 'x')
        # Getting the type of 'x' (line 49)
        x_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 26), 'x')
        # Applying the binary operator '*' (line 49)
        result_mul_290 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 22), '*', x_288, x_289)
        
        # Getting the type of 'x' (line 49)
        x_291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 30), 'x')
        # Applying the binary operator '+' (line 49)
        result_add_292 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 22), '+', result_mul_290, x_291)
        
        int_293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 36), 'int')
        # Applying the binary operator '<<' (line 49)
        result_lshift_294 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 21), '<<', result_add_292, int_293)
        
        int_295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 41), 'int')
        # Applying the binary operator '-' (line 49)
        result_sub_296 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 20), '-', result_lshift_294, int_295)
        
        # Assigning a type to the variable 'tuple_assignment_15' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_assignment_15', result_sub_296)
        
        # Assigning a BinOp to a Name (line 49):
        # Getting the type of 'x' (line 49)
        x_297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 47), 'x')
        int_298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 51), 'int')
        # Applying the binary operator '-' (line 49)
        result_sub_299 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 47), '-', x_297, int_298)
        
        int_300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 57), 'int')
        # Applying the binary operator '<<' (line 49)
        result_lshift_301 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 46), '<<', result_sub_299, int_300)
        
        int_302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 62), 'int')
        # Applying the binary operator '-' (line 49)
        result_sub_303 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 45), '-', result_lshift_301, int_302)
        
        int_304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 68), 'int')
        # Applying the binary operator '<<' (line 49)
        result_lshift_305 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 44), '<<', result_sub_303, int_304)
        
        # Assigning a type to the variable 'tuple_assignment_16' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_assignment_16', result_lshift_305)
        
        # Assigning a Name to a Name (line 49):
        # Getting the type of 'tuple_assignment_15' (line 49)
        tuple_assignment_15_306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_assignment_15')
        # Assigning a type to the variable 'n' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'n', tuple_assignment_15_306)
        
        # Assigning a Name to a Name (line 49):
        # Getting the type of 'tuple_assignment_16' (line 49)
        tuple_assignment_16_307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_assignment_16')
        # Assigning a type to the variable 'n_diff' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 11), 'n_diff', tuple_assignment_16_307)
        
        
        # Call to xrange(...): (line 50)
        # Processing the call arguments (line 50)
        # Getting the type of 'n_diff' (line 50)
        n_diff_309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 24), 'n_diff', False)
        # Getting the type of 'y_min' (line 50)
        y_min_310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 32), 'y_min', False)
        int_311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 39), 'int')
        # Processing the call keyword arguments (line 50)
        kwargs_312 = {}
        # Getting the type of 'xrange' (line 50)
        xrange_308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 50)
        xrange_call_result_313 = invoke(stypy.reporting.localization.Localization(__file__, 50, 17), xrange_308, *[n_diff_309, y_min_310, int_311], **kwargs_312)
        
        # Assigning a type to the variable 'xrange_call_result_313' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'xrange_call_result_313', xrange_call_result_313)
        # Testing if the for loop is going to be iterated (line 50)
        # Testing the type of a for loop iterable (line 50)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 50, 8), xrange_call_result_313)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 50, 8), xrange_call_result_313):
            # Getting the type of the for loop variable (line 50)
            for_loop_var_314 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 50, 8), xrange_call_result_313)
            # Assigning a type to the variable 'd' (line 50)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'd', for_loop_var_314)
            # SSA begins for a for statement (line 50)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'n' (line 51)
            n_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 15), 'n')
            int_316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 19), 'int')
            # Applying the binary operator '%' (line 51)
            result_mod_317 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 15), '%', n_315, int_316)
            
            int_318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 25), 'int')
            # Applying the binary operator '==' (line 51)
            result_eq_319 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 15), '==', result_mod_317, int_318)
            
            # Testing if the type of an if condition is none (line 51)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 51, 12), result_eq_319):
                pass
            else:
                
                # Testing the type of an if condition (line 51)
                if_condition_320 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 51, 12), result_eq_319)
                # Assigning a type to the variable 'if_condition_320' (line 51)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'if_condition_320', if_condition_320)
                # SSA begins for if statement (line 51)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a BinOp to a Name (line 52):
                
                # Assigning a BinOp to a Name (line 52):
                # Getting the type of 'n' (line 52)
                n_321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 20), 'n')
                int_322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 25), 'int')
                # Applying the binary operator '>>' (line 52)
                result_rshift_323 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 20), '>>', n_321, int_322)
                
                # Assigning a type to the variable 'm' (line 52)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'm', result_rshift_323)
                
                # Assigning a UnaryOp to a Subscript (line 53):
                
                # Assigning a UnaryOp to a Subscript (line 53):
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'm' (line 53)
                m_324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 37), 'm')
                # Getting the type of 'sieve' (line 53)
                sieve_325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 31), 'sieve')
                # Obtaining the member '__getitem__' of a type (line 53)
                getitem___326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 31), sieve_325, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 53)
                subscript_call_result_327 = invoke(stypy.reporting.localization.Localization(__file__, 53, 31), getitem___326, m_324)
                
                # Applying the 'not' unary operator (line 53)
                result_not__328 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 27), 'not', subscript_call_result_327)
                
                # Getting the type of 'sieve' (line 53)
                sieve_329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 16), 'sieve')
                # Getting the type of 'm' (line 53)
                m_330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 22), 'm')
                # Storing an element on a container (line 53)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 16), sieve_329, (m_330, result_not__328))
                # SSA join for if statement (line 51)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'n' (line 54)
            n_331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'n')
            # Getting the type of 'd' (line 54)
            d_332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 17), 'd')
            # Applying the binary operator '+=' (line 54)
            result_iadd_333 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 12), '+=', n_331, d_332)
            # Assigning a type to the variable 'n' (line 54)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'n', result_iadd_333)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Assigning a List to a Name (line 56):
    
    # Assigning a List to a Name (line 56):
    
    # Obtaining an instance of the builtin type 'list' (line 56)
    list_334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 56)
    # Adding element type (line 56)
    int_335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 13), list_334, int_335)
    # Adding element type (line 56)
    int_336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 13), list_334, int_336)
    
    # Assigning a type to the variable 'primes' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'primes', list_334)
    
    # Getting the type of 'end' (line 57)
    end_337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 7), 'end')
    int_338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 14), 'int')
    # Applying the binary operator '<=' (line 57)
    result_le_339 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 7), '<=', end_337, int_338)
    
    # Testing if the type of an if condition is none (line 57)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 57, 4), result_le_339):
        pass
    else:
        
        # Testing the type of an if condition (line 57)
        if_condition_340 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 57, 4), result_le_339)
        # Assigning a type to the variable 'if_condition_340' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'if_condition_340', if_condition_340)
        # SSA begins for if statement (line 57)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining the type of the subscript
        
        # Call to max(...): (line 58)
        # Processing the call arguments (line 58)
        int_342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 27), 'int')
        # Getting the type of 'end' (line 58)
        end_343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 30), 'end', False)
        int_344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 36), 'int')
        # Applying the binary operator '-' (line 58)
        result_sub_345 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 30), '-', end_343, int_344)
        
        # Processing the call keyword arguments (line 58)
        kwargs_346 = {}
        # Getting the type of 'max' (line 58)
        max_341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 23), 'max', False)
        # Calling max(args, kwargs) (line 58)
        max_call_result_347 = invoke(stypy.reporting.localization.Localization(__file__, 58, 23), max_341, *[int_342, result_sub_345], **kwargs_346)
        
        slice_348 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 58, 15), None, max_call_result_347, None)
        # Getting the type of 'primes' (line 58)
        primes_349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 15), 'primes')
        # Obtaining the member '__getitem__' of a type (line 58)
        getitem___350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 15), primes_349, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 58)
        subscript_call_result_351 = invoke(stypy.reporting.localization.Localization(__file__, 58, 15), getitem___350, slice_348)
        
        # Assigning a type to the variable 'stypy_return_type' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'stypy_return_type', subscript_call_result_351)
        # SSA join for if statement (line 57)
        module_type_store = module_type_store.join_ssa_context()
        

    
    
    # Call to xrange(...): (line 60)
    # Processing the call arguments (line 60)
    int_353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 20), 'int')
    int_354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 25), 'int')
    # Applying the binary operator '>>' (line 60)
    result_rshift_355 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 20), '>>', int_353, int_354)
    
    
    # Call to int(...): (line 60)
    # Processing the call arguments (line 60)
    
    # Call to sqrt(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'end' (line 60)
    end_358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 38), 'end', False)
    # Processing the call keyword arguments (line 60)
    kwargs_359 = {}
    # Getting the type of 'sqrt' (line 60)
    sqrt_357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 33), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 60)
    sqrt_call_result_360 = invoke(stypy.reporting.localization.Localization(__file__, 60, 33), sqrt_357, *[end_358], **kwargs_359)
    
    # Processing the call keyword arguments (line 60)
    kwargs_361 = {}
    # Getting the type of 'int' (line 60)
    int_356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 29), 'int', False)
    # Calling int(args, kwargs) (line 60)
    int_call_result_362 = invoke(stypy.reporting.localization.Localization(__file__, 60, 29), int_356, *[sqrt_call_result_360], **kwargs_361)
    
    int_363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 46), 'int')
    # Applying the binary operator '+' (line 60)
    result_add_364 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 29), '+', int_call_result_362, int_363)
    
    int_365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 52), 'int')
    # Applying the binary operator '>>' (line 60)
    result_rshift_366 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 28), '>>', result_add_364, int_365)
    
    # Processing the call keyword arguments (line 60)
    kwargs_367 = {}
    # Getting the type of 'xrange' (line 60)
    xrange_352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 60)
    xrange_call_result_368 = invoke(stypy.reporting.localization.Localization(__file__, 60, 13), xrange_352, *[result_rshift_355, result_rshift_366], **kwargs_367)
    
    # Assigning a type to the variable 'xrange_call_result_368' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'xrange_call_result_368', xrange_call_result_368)
    # Testing if the for loop is going to be iterated (line 60)
    # Testing the type of a for loop iterable (line 60)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 60, 4), xrange_call_result_368)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 60, 4), xrange_call_result_368):
        # Getting the type of the for loop variable (line 60)
        for_loop_var_369 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 60, 4), xrange_call_result_368)
        # Assigning a type to the variable 'n' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'n', for_loop_var_369)
        # SSA begins for a for statement (line 60)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Obtaining the type of the subscript
        # Getting the type of 'n' (line 61)
        n_370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 17), 'n')
        # Getting the type of 'sieve' (line 61)
        sieve_371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 11), 'sieve')
        # Obtaining the member '__getitem__' of a type (line 61)
        getitem___372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 11), sieve_371, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 61)
        subscript_call_result_373 = invoke(stypy.reporting.localization.Localization(__file__, 61, 11), getitem___372, n_370)
        
        # Testing if the type of an if condition is none (line 61)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 61, 8), subscript_call_result_373):
            pass
        else:
            
            # Testing the type of an if condition (line 61)
            if_condition_374 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 61, 8), subscript_call_result_373)
            # Assigning a type to the variable 'if_condition_374' (line 61)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'if_condition_374', if_condition_374)
            # SSA begins for if statement (line 61)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 62)
            # Processing the call arguments (line 62)
            # Getting the type of 'n' (line 62)
            n_377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 27), 'n', False)
            int_378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 32), 'int')
            # Applying the binary operator '<<' (line 62)
            result_lshift_379 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 27), '<<', n_377, int_378)
            
            int_380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 37), 'int')
            # Applying the binary operator '+' (line 62)
            result_add_381 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 26), '+', result_lshift_379, int_380)
            
            # Processing the call keyword arguments (line 62)
            kwargs_382 = {}
            # Getting the type of 'primes' (line 62)
            primes_375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'primes', False)
            # Obtaining the member 'append' of a type (line 62)
            append_376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 12), primes_375, 'append')
            # Calling append(args, kwargs) (line 62)
            append_call_result_383 = invoke(stypy.reporting.localization.Localization(__file__, 62, 12), append_376, *[result_add_381], **kwargs_382)
            
            
            # Assigning a BinOp to a Name (line 63):
            
            # Assigning a BinOp to a Name (line 63):
            # Getting the type of 'n' (line 63)
            n_384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 19), 'n')
            int_385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 24), 'int')
            # Applying the binary operator '<<' (line 63)
            result_lshift_386 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 19), '<<', n_384, int_385)
            
            int_387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 29), 'int')
            # Applying the binary operator '+' (line 63)
            result_add_388 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 18), '+', result_lshift_386, int_387)
            
            # Assigning a type to the variable 'aux' (line 63)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'aux', result_add_388)
            
            # Getting the type of 'aux' (line 64)
            aux_389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'aux')
            # Getting the type of 'aux' (line 64)
            aux_390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 19), 'aux')
            # Applying the binary operator '*=' (line 64)
            result_imul_391 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 12), '*=', aux_389, aux_390)
            # Assigning a type to the variable 'aux' (line 64)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'aux', result_imul_391)
            
            
            
            # Call to xrange(...): (line 65)
            # Processing the call arguments (line 65)
            # Getting the type of 'aux' (line 65)
            aux_393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 28), 'aux', False)
            # Getting the type of 'end' (line 65)
            end_394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 33), 'end', False)
            int_395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 38), 'int')
            # Getting the type of 'aux' (line 65)
            aux_396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 42), 'aux', False)
            # Applying the binary operator '*' (line 65)
            result_mul_397 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 38), '*', int_395, aux_396)
            
            # Processing the call keyword arguments (line 65)
            kwargs_398 = {}
            # Getting the type of 'xrange' (line 65)
            xrange_392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 21), 'xrange', False)
            # Calling xrange(args, kwargs) (line 65)
            xrange_call_result_399 = invoke(stypy.reporting.localization.Localization(__file__, 65, 21), xrange_392, *[aux_393, end_394, result_mul_397], **kwargs_398)
            
            # Assigning a type to the variable 'xrange_call_result_399' (line 65)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'xrange_call_result_399', xrange_call_result_399)
            # Testing if the for loop is going to be iterated (line 65)
            # Testing the type of a for loop iterable (line 65)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 65, 12), xrange_call_result_399)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 65, 12), xrange_call_result_399):
                # Getting the type of the for loop variable (line 65)
                for_loop_var_400 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 65, 12), xrange_call_result_399)
                # Assigning a type to the variable 'k' (line 65)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'k', for_loop_var_400)
                # SSA begins for a for statement (line 65)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Name to a Subscript (line 66):
                
                # Assigning a Name to a Subscript (line 66):
                # Getting the type of 'False' (line 66)
                False_401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 32), 'False')
                # Getting the type of 'sieve' (line 66)
                sieve_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 16), 'sieve')
                # Getting the type of 'k' (line 66)
                k_403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 22), 'k')
                int_404 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 27), 'int')
                # Applying the binary operator '>>' (line 66)
                result_rshift_405 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 22), '>>', k_403, int_404)
                
                # Storing an element on a container (line 66)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 16), sieve_402, (result_rshift_405, False_401))
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 61)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Assigning a BinOp to a Name (line 68):
    
    # Assigning a BinOp to a Name (line 68):
    
    # Call to int(...): (line 68)
    # Processing the call arguments (line 68)
    
    # Call to sqrt(...): (line 68)
    # Processing the call arguments (line 68)
    # Getting the type of 'end' (line 68)
    end_408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 17), 'end', False)
    # Processing the call keyword arguments (line 68)
    kwargs_409 = {}
    # Getting the type of 'sqrt' (line 68)
    sqrt_407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 68)
    sqrt_call_result_410 = invoke(stypy.reporting.localization.Localization(__file__, 68, 12), sqrt_407, *[end_408], **kwargs_409)
    
    # Processing the call keyword arguments (line 68)
    kwargs_411 = {}
    # Getting the type of 'int' (line 68)
    int_406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'int', False)
    # Calling int(args, kwargs) (line 68)
    int_call_result_412 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), int_406, *[sqrt_call_result_410], **kwargs_411)
    
    int_413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 25), 'int')
    # Applying the binary operator '+' (line 68)
    result_add_414 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 8), '+', int_call_result_412, int_413)
    
    # Assigning a type to the variable 's' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 's', result_add_414)
    
    # Getting the type of 's' (line 69)
    s_415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 7), 's')
    int_416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 11), 'int')
    # Applying the binary operator '%' (line 69)
    result_mod_417 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 7), '%', s_415, int_416)
    
    int_418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 16), 'int')
    # Applying the binary operator '==' (line 69)
    result_eq_419 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 7), '==', result_mod_417, int_418)
    
    # Testing if the type of an if condition is none (line 69)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 69, 4), result_eq_419):
        pass
    else:
        
        # Testing the type of an if condition (line 69)
        if_condition_420 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 4), result_eq_419)
        # Assigning a type to the variable 'if_condition_420' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'if_condition_420', if_condition_420)
        # SSA begins for if statement (line 69)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 's' (line 70)
        s_421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 's')
        int_422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 13), 'int')
        # Applying the binary operator '+=' (line 70)
        result_iadd_423 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 8), '+=', s_421, int_422)
        # Assigning a type to the variable 's' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 's', result_iadd_423)
        
        # SSA join for if statement (line 69)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to extend(...): (line 71)
    # Processing the call arguments (line 71)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to xrange(...): (line 71)
    # Processing the call arguments (line 71)
    # Getting the type of 's' (line 71)
    s_434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 37), 's', False)
    # Getting the type of 'end' (line 71)
    end_435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 40), 'end', False)
    int_436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 45), 'int')
    # Processing the call keyword arguments (line 71)
    kwargs_437 = {}
    # Getting the type of 'xrange' (line 71)
    xrange_433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 30), 'xrange', False)
    # Calling xrange(args, kwargs) (line 71)
    xrange_call_result_438 = invoke(stypy.reporting.localization.Localization(__file__, 71, 30), xrange_433, *[s_434, end_435, int_436], **kwargs_437)
    
    comprehension_439 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 19), xrange_call_result_438)
    # Assigning a type to the variable 'i' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 19), 'i', comprehension_439)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 71)
    i_427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 57), 'i', False)
    int_428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 62), 'int')
    # Applying the binary operator '>>' (line 71)
    result_rshift_429 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 57), '>>', i_427, int_428)
    
    # Getting the type of 'sieve' (line 71)
    sieve_430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 51), 'sieve', False)
    # Obtaining the member '__getitem__' of a type (line 71)
    getitem___431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 51), sieve_430, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 71)
    subscript_call_result_432 = invoke(stypy.reporting.localization.Localization(__file__, 71, 51), getitem___431, result_rshift_429)
    
    # Getting the type of 'i' (line 71)
    i_426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 19), 'i', False)
    list_440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 19), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 19), list_440, i_426)
    # Processing the call keyword arguments (line 71)
    kwargs_441 = {}
    # Getting the type of 'primes' (line 71)
    primes_424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'primes', False)
    # Obtaining the member 'extend' of a type (line 71)
    extend_425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 4), primes_424, 'extend')
    # Calling extend(args, kwargs) (line 71)
    extend_call_result_442 = invoke(stypy.reporting.localization.Localization(__file__, 71, 4), extend_425, *[list_440], **kwargs_441)
    
    # Getting the type of 'primes' (line 73)
    primes_443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 11), 'primes')
    # Assigning a type to the variable 'stypy_return_type' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'stypy_return_type', primes_443)
    
    # ################# End of 'sieveOfAtkin(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sieveOfAtkin' in the type store
    # Getting the type of 'stypy_return_type' (line 5)
    stypy_return_type_444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_444)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sieveOfAtkin'
    return stypy_return_type_444

# Assigning a type to the variable 'sieveOfAtkin' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'sieveOfAtkin', sieveOfAtkin)

@norecursion
def sieveOfEratostenes(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'sieveOfEratostenes'
    module_type_store = module_type_store.open_function_context('sieveOfEratostenes', 76, 0, False)
    
    # Passed parameters checking function
    sieveOfEratostenes.stypy_localization = localization
    sieveOfEratostenes.stypy_type_of_self = None
    sieveOfEratostenes.stypy_type_store = module_type_store
    sieveOfEratostenes.stypy_function_name = 'sieveOfEratostenes'
    sieveOfEratostenes.stypy_param_names_list = ['n']
    sieveOfEratostenes.stypy_varargs_param_name = None
    sieveOfEratostenes.stypy_kwargs_param_name = None
    sieveOfEratostenes.stypy_call_defaults = defaults
    sieveOfEratostenes.stypy_call_varargs = varargs
    sieveOfEratostenes.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'sieveOfEratostenes', ['n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'sieveOfEratostenes', localization, ['n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'sieveOfEratostenes(...)' code ##################

    str_445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 4), 'str', 'sieveOfEratostenes(n): return the list of the primes < n.')
    
    # Getting the type of 'n' (line 80)
    n_446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 7), 'n')
    int_447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 12), 'int')
    # Applying the binary operator '<=' (line 80)
    result_le_448 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 7), '<=', n_446, int_447)
    
    # Testing if the type of an if condition is none (line 80)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 80, 4), result_le_448):
        pass
    else:
        
        # Testing the type of an if condition (line 80)
        if_condition_449 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 4), result_le_448)
        # Assigning a type to the variable 'if_condition_449' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'if_condition_449', if_condition_449)
        # SSA begins for if statement (line 80)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'list' (line 81)
        list_450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 81)
        
        # Assigning a type to the variable 'stypy_return_type' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'stypy_return_type', list_450)
        # SSA join for if statement (line 80)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 82):
    
    # Assigning a Call to a Name (line 82):
    
    # Call to range(...): (line 82)
    # Processing the call arguments (line 82)
    int_452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 18), 'int')
    # Getting the type of 'n' (line 82)
    n_453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 21), 'n', False)
    int_454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 24), 'int')
    # Processing the call keyword arguments (line 82)
    kwargs_455 = {}
    # Getting the type of 'range' (line 82)
    range_451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'range', False)
    # Calling range(args, kwargs) (line 82)
    range_call_result_456 = invoke(stypy.reporting.localization.Localization(__file__, 82, 12), range_451, *[int_452, n_453, int_454], **kwargs_455)
    
    # Assigning a type to the variable 'sieve' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'sieve', range_call_result_456)
    
    # Assigning a Call to a Name (line 83):
    
    # Assigning a Call to a Name (line 83):
    
    # Call to len(...): (line 83)
    # Processing the call arguments (line 83)
    # Getting the type of 'sieve' (line 83)
    sieve_458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 14), 'sieve', False)
    # Processing the call keyword arguments (line 83)
    kwargs_459 = {}
    # Getting the type of 'len' (line 83)
    len_457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 10), 'len', False)
    # Calling len(args, kwargs) (line 83)
    len_call_result_460 = invoke(stypy.reporting.localization.Localization(__file__, 83, 10), len_457, *[sieve_458], **kwargs_459)
    
    # Assigning a type to the variable 'top' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'top', len_call_result_460)
    
    # Getting the type of 'sieve' (line 84)
    sieve_461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 14), 'sieve')
    # Assigning a type to the variable 'sieve_461' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'sieve_461', sieve_461)
    # Testing if the for loop is going to be iterated (line 84)
    # Testing the type of a for loop iterable (line 84)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 84, 4), sieve_461)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 84, 4), sieve_461):
        # Getting the type of the for loop variable (line 84)
        for_loop_var_462 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 84, 4), sieve_461)
        # Assigning a type to the variable 'si' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'si', for_loop_var_462)
        # SSA begins for a for statement (line 84)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        # Getting the type of 'si' (line 85)
        si_463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 11), 'si')
        # Testing if the type of an if condition is none (line 85)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 85, 8), si_463):
            pass
        else:
            
            # Testing the type of an if condition (line 85)
            if_condition_464 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 85, 8), si_463)
            # Assigning a type to the variable 'if_condition_464' (line 85)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'if_condition_464', if_condition_464)
            # SSA begins for if statement (line 85)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 86):
            
            # Assigning a BinOp to a Name (line 86):
            # Getting the type of 'si' (line 86)
            si_465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 22), 'si')
            # Getting the type of 'si' (line 86)
            si_466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 27), 'si')
            # Applying the binary operator '*' (line 86)
            result_mul_467 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 22), '*', si_465, si_466)
            
            int_468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 32), 'int')
            # Applying the binary operator '-' (line 86)
            result_sub_469 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 22), '-', result_mul_467, int_468)
            
            int_470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 38), 'int')
            # Applying the binary operator '//' (line 86)
            result_floordiv_471 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 21), '//', result_sub_469, int_470)
            
            # Assigning a type to the variable 'bottom' (line 86)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'bottom', result_floordiv_471)
            
            # Getting the type of 'bottom' (line 87)
            bottom_472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 15), 'bottom')
            # Getting the type of 'top' (line 87)
            top_473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 25), 'top')
            # Applying the binary operator '>=' (line 87)
            result_ge_474 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 15), '>=', bottom_472, top_473)
            
            # Testing if the type of an if condition is none (line 87)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 87, 12), result_ge_474):
                pass
            else:
                
                # Testing the type of an if condition (line 87)
                if_condition_475 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 12), result_ge_474)
                # Assigning a type to the variable 'if_condition_475' (line 87)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'if_condition_475', if_condition_475)
                # SSA begins for if statement (line 87)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # SSA join for if statement (line 87)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a BinOp to a Subscript (line 89):
            
            # Assigning a BinOp to a Subscript (line 89):
            
            # Obtaining an instance of the builtin type 'list' (line 89)
            list_476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 32), 'list')
            # Adding type elements to the builtin type 'list' instance (line 89)
            # Adding element type (line 89)
            int_477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 33), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 32), list_476, int_477)
            
            
            # Getting the type of 'bottom' (line 89)
            bottom_478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 41), 'bottom')
            # Getting the type of 'top' (line 89)
            top_479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 50), 'top')
            # Applying the binary operator '-' (line 89)
            result_sub_480 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 41), '-', bottom_478, top_479)
            
            # Getting the type of 'si' (line 89)
            si_481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 58), 'si')
            # Applying the binary operator '//' (line 89)
            result_floordiv_482 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 40), '//', result_sub_480, si_481)
            
            # Applying the 'usub' unary operator (line 89)
            result___neg___483 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 38), 'usub', result_floordiv_482)
            
            # Applying the binary operator '*' (line 89)
            result_mul_484 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 32), '*', list_476, result___neg___483)
            
            # Getting the type of 'sieve' (line 89)
            sieve_485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'sieve')
            # Getting the type of 'bottom' (line 89)
            bottom_486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 18), 'bottom')
            # Getting the type of 'si' (line 89)
            si_487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 26), 'si')
            slice_488 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 89, 12), bottom_486, None, si_487)
            # Storing an element on a container (line 89)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 12), sieve_485, (slice_488, result_mul_484))
            # SSA join for if statement (line 85)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Obtaining an instance of the builtin type 'list' (line 90)
    list_489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 90)
    # Adding element type (line 90)
    int_490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 11), list_489, int_490)
    
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'sieve' (line 90)
    sieve_493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 31), 'sieve')
    comprehension_494 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 18), sieve_493)
    # Assigning a type to the variable 'el' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 18), 'el', comprehension_494)
    # Getting the type of 'el' (line 90)
    el_492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 40), 'el')
    # Getting the type of 'el' (line 90)
    el_491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 18), 'el')
    list_495 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 18), list_495, el_491)
    # Applying the binary operator '+' (line 90)
    result_add_496 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 11), '+', list_489, list_495)
    
    # Assigning a type to the variable 'stypy_return_type' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'stypy_return_type', result_add_496)
    
    # ################# End of 'sieveOfEratostenes(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sieveOfEratostenes' in the type store
    # Getting the type of 'stypy_return_type' (line 76)
    stypy_return_type_497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_497)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sieveOfEratostenes'
    return stypy_return_type_497

# Assigning a type to the variable 'sieveOfEratostenes' (line 76)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'sieveOfEratostenes', sieveOfEratostenes)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 93, 0, False)
    
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

    
    # Assigning a Num to a Name (line 96):
    
    # Assigning a Num to a Name (line 96):
    int_498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 8), 'int')
    # Assigning a type to the variable 'n' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'n', int_498)
    
    # Assigning a Call to a Name (line 101):
    
    # Assigning a Call to a Name (line 101):
    
    # Call to sieveOfAtkin(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'n' (line 101)
    n_500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 21), 'n', False)
    # Processing the call keyword arguments (line 101)
    kwargs_501 = {}
    # Getting the type of 'sieveOfAtkin' (line 101)
    sieveOfAtkin_499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'sieveOfAtkin', False)
    # Calling sieveOfAtkin(args, kwargs) (line 101)
    sieveOfAtkin_call_result_502 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), sieveOfAtkin_499, *[n_500], **kwargs_501)
    
    # Assigning a type to the variable 'r' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'r', sieveOfAtkin_call_result_502)
    
    # Assigning a Call to a Name (line 105):
    
    # Assigning a Call to a Name (line 105):
    
    # Call to sieveOfEratostenes(...): (line 105)
    # Processing the call arguments (line 105)
    # Getting the type of 'n' (line 105)
    n_504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 27), 'n', False)
    # Processing the call keyword arguments (line 105)
    kwargs_505 = {}
    # Getting the type of 'sieveOfEratostenes' (line 105)
    sieveOfEratostenes_503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'sieveOfEratostenes', False)
    # Calling sieveOfEratostenes(args, kwargs) (line 105)
    sieveOfEratostenes_call_result_506 = invoke(stypy.reporting.localization.Localization(__file__, 105, 8), sieveOfEratostenes_503, *[n_504], **kwargs_505)
    
    # Assigning a type to the variable 'r' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'r', sieveOfEratostenes_call_result_506)
    # Getting the type of 'True' (line 111)
    True_507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'stypy_return_type', True_507)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 93)
    stypy_return_type_508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_508)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_508

# Assigning a type to the variable 'run' (line 93)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'run', run)

# Call to run(...): (line 114)
# Processing the call keyword arguments (line 114)
kwargs_510 = {}
# Getting the type of 'run' (line 114)
run_509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 0), 'run', False)
# Calling run(args, kwargs) (line 114)
run_call_result_511 = invoke(stypy.reporting.localization.Localization(__file__, 114, 0), run_509, *[], **kwargs_510)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
