
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
    
    
    # Assigning a BinOp to a Name (line 12):
    
    # Assigning a BinOp to a Name (line 12):
    # Getting the type of 'end' (line 12)
    end_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 12), 'end')
    int_22 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 19), 'int')
    # Applying the binary operator '//' (line 12)
    result_floordiv_23 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 12), '//', end_21, int_22)
    
    int_24 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 24), 'int')
    # Applying the binary operator '-' (line 12)
    result_sub_25 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 11), '-', result_floordiv_23, int_24)
    
    # Getting the type of 'end' (line 12)
    end_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 28), 'end')
    int_27 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 34), 'int')
    # Applying the binary operator '%' (line 12)
    result_mod_28 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 28), '%', end_26, int_27)
    
    # Applying the binary operator '+' (line 12)
    result_add_29 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 26), '+', result_sub_25, result_mod_28)
    
    # Assigning a type to the variable 'lng' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'lng', result_add_29)
    
    # Assigning a BinOp to a Name (line 13):
    
    # Assigning a BinOp to a Name (line 13):
    
    # Obtaining an instance of the builtin type 'list' (line 13)
    list_30 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 13)
    # Adding element type (line 13)
    # Getting the type of 'False' (line 13)
    False_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 13), 'False')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 12), list_30, False_31)
    
    # Getting the type of 'lng' (line 13)
    lng_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 23), 'lng')
    int_33 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 29), 'int')
    # Applying the binary operator '+' (line 13)
    result_add_34 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 23), '+', lng_32, int_33)
    
    # Applying the binary operator '*' (line 13)
    result_mul_35 = python_operator(stypy.reporting.localization.Localization(__file__, 13, 12), '*', list_30, result_add_34)
    
    # Assigning a type to the variable 'sieve' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'sieve', result_mul_35)
    
    # Assigning a Tuple to a Tuple (line 15):
    
    # Assigning a Call to a Name (line 15):
    
    # Call to int(...): (line 15)
    # Processing the call arguments (line 15)
    
    # Call to sqrt(...): (line 15)
    # Processing the call arguments (line 15)
    # Getting the type of 'end' (line 15)
    end_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 30), 'end', False)
    int_39 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 36), 'int')
    # Applying the binary operator '-' (line 15)
    result_sub_40 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 30), '-', end_38, int_39)
    
    float_41 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 41), 'float')
    # Applying the binary operator 'div' (line 15)
    result_div_42 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 29), 'div', result_sub_40, float_41)
    
    # Processing the call keyword arguments (line 15)
    kwargs_43 = {}
    # Getting the type of 'sqrt' (line 15)
    sqrt_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 24), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 15)
    sqrt_call_result_44 = invoke(stypy.reporting.localization.Localization(__file__, 15, 24), sqrt_37, *[result_div_42], **kwargs_43)
    
    # Processing the call keyword arguments (line 15)
    kwargs_45 = {}
    # Getting the type of 'int' (line 15)
    int_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 20), 'int', False)
    # Calling int(args, kwargs) (line 15)
    int_call_result_46 = invoke(stypy.reporting.localization.Localization(__file__, 15, 20), int_36, *[sqrt_call_result_44], **kwargs_45)
    
    # Assigning a type to the variable 'tuple_assignment_1' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'tuple_assignment_1', int_call_result_46)
    
    # Assigning a Num to a Name (line 15):
    int_47 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 48), 'int')
    # Assigning a type to the variable 'tuple_assignment_2' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'tuple_assignment_2', int_47)
    
    # Assigning a Num to a Name (line 15):
    int_48 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 51), 'int')
    # Assigning a type to the variable 'tuple_assignment_3' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'tuple_assignment_3', int_48)
    
    # Assigning a Name to a Name (line 15):
    # Getting the type of 'tuple_assignment_1' (line 15)
    tuple_assignment_1_49 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'tuple_assignment_1')
    # Assigning a type to the variable 'x_max' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'x_max', tuple_assignment_1_49)
    
    # Assigning a Name to a Name (line 15):
    # Getting the type of 'tuple_assignment_2' (line 15)
    tuple_assignment_2_50 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'tuple_assignment_2')
    # Assigning a type to the variable 'x2' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 11), 'x2', tuple_assignment_2_50)
    
    # Assigning a Name to a Name (line 15):
    # Getting the type of 'tuple_assignment_3' (line 15)
    tuple_assignment_3_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'tuple_assignment_3')
    # Assigning a type to the variable 'xd' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 15), 'xd', tuple_assignment_3_51)
    
    
    # Call to xrange(...): (line 16)
    # Processing the call arguments (line 16)
    int_53 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 21), 'int')
    int_54 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 24), 'int')
    # Getting the type of 'x_max' (line 16)
    x_max_55 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 28), 'x_max', False)
    # Applying the binary operator '*' (line 16)
    result_mul_56 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 24), '*', int_54, x_max_55)
    
    int_57 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 36), 'int')
    # Applying the binary operator '+' (line 16)
    result_add_58 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 24), '+', result_mul_56, int_57)
    
    int_59 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 39), 'int')
    # Processing the call keyword arguments (line 16)
    kwargs_60 = {}
    # Getting the type of 'xrange' (line 16)
    xrange_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 14), 'xrange', False)
    # Calling xrange(args, kwargs) (line 16)
    xrange_call_result_61 = invoke(stypy.reporting.localization.Localization(__file__, 16, 14), xrange_52, *[int_53, result_add_58, int_59], **kwargs_60)
    
    # Testing if the for loop is going to be iterated (line 16)
    # Testing the type of a for loop iterable (line 16)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 16, 4), xrange_call_result_61)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 16, 4), xrange_call_result_61):
        # Getting the type of the for loop variable (line 16)
        for_loop_var_62 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 16, 4), xrange_call_result_61)
        # Assigning a type to the variable 'xd' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'xd', for_loop_var_62)
        # SSA begins for a for statement (line 16)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'x2' (line 17)
        x2_63 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'x2')
        # Getting the type of 'xd' (line 17)
        xd_64 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 14), 'xd')
        # Applying the binary operator '+=' (line 17)
        result_iadd_65 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 8), '+=', x2_63, xd_64)
        # Assigning a type to the variable 'x2' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'x2', result_iadd_65)
        
        
        # Assigning a Call to a Name (line 18):
        
        # Assigning a Call to a Name (line 18):
        
        # Call to int(...): (line 18)
        # Processing the call arguments (line 18)
        
        # Call to sqrt(...): (line 18)
        # Processing the call arguments (line 18)
        # Getting the type of 'end' (line 18)
        end_68 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 25), 'end', False)
        # Getting the type of 'x2' (line 18)
        x2_69 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 31), 'x2', False)
        # Applying the binary operator '-' (line 18)
        result_sub_70 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 25), '-', end_68, x2_69)
        
        # Processing the call keyword arguments (line 18)
        kwargs_71 = {}
        # Getting the type of 'sqrt' (line 18)
        sqrt_67 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 20), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 18)
        sqrt_call_result_72 = invoke(stypy.reporting.localization.Localization(__file__, 18, 20), sqrt_67, *[result_sub_70], **kwargs_71)
        
        # Processing the call keyword arguments (line 18)
        kwargs_73 = {}
        # Getting the type of 'int' (line 18)
        int_66 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 16), 'int', False)
        # Calling int(args, kwargs) (line 18)
        int_call_result_74 = invoke(stypy.reporting.localization.Localization(__file__, 18, 16), int_66, *[sqrt_call_result_72], **kwargs_73)
        
        # Assigning a type to the variable 'y_max' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'y_max', int_call_result_74)
        
        # Assigning a Tuple to a Tuple (line 19):
        
        # Assigning a BinOp to a Name (line 19):
        # Getting the type of 'x2' (line 19)
        x2_75 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 20), 'x2')
        # Getting the type of 'y_max' (line 19)
        y_max_76 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 25), 'y_max')
        # Getting the type of 'y_max' (line 19)
        y_max_77 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 33), 'y_max')
        # Applying the binary operator '*' (line 19)
        result_mul_78 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 25), '*', y_max_76, y_max_77)
        
        # Applying the binary operator '+' (line 19)
        result_add_79 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 20), '+', x2_75, result_mul_78)
        
        # Assigning a type to the variable 'tuple_assignment_4' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'tuple_assignment_4', result_add_79)
        
        # Assigning a BinOp to a Name (line 19):
        # Getting the type of 'y_max' (line 19)
        y_max_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 41), 'y_max')
        int_81 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 50), 'int')
        # Applying the binary operator '<<' (line 19)
        result_lshift_82 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 41), '<<', y_max_80, int_81)
        
        int_83 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 55), 'int')
        # Applying the binary operator '-' (line 19)
        result_sub_84 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 40), '-', result_lshift_82, int_83)
        
        # Assigning a type to the variable 'tuple_assignment_5' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'tuple_assignment_5', result_sub_84)
        
        # Assigning a Name to a Name (line 19):
        # Getting the type of 'tuple_assignment_4' (line 19)
        tuple_assignment_4_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'tuple_assignment_4')
        # Assigning a type to the variable 'n' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'n', tuple_assignment_4_85)
        
        # Assigning a Name to a Name (line 19):
        # Getting the type of 'tuple_assignment_5' (line 19)
        tuple_assignment_5_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'tuple_assignment_5')
        # Assigning a type to the variable 'n_diff' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 11), 'n_diff', tuple_assignment_5_86)
        
        # Getting the type of 'n' (line 20)
        n_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 16), 'n')
        int_88 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 20), 'int')
        # Applying the binary operator '&' (line 20)
        result_and__89 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 16), '&', n_87, int_88)
        
        # Applying the 'not' unary operator (line 20)
        result_not__90 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 11), 'not', result_and__89)
        
        # Testing if the type of an if condition is none (line 20)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 20, 8), result_not__90):
            pass
        else:
            
            # Testing the type of an if condition (line 20)
            if_condition_91 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 20, 8), result_not__90)
            # Assigning a type to the variable 'if_condition_91' (line 20)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'if_condition_91', if_condition_91)
            # SSA begins for if statement (line 20)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'n' (line 21)
            n_92 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'n')
            # Getting the type of 'n_diff' (line 21)
            n_diff_93 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 17), 'n_diff')
            # Applying the binary operator '-=' (line 21)
            result_isub_94 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 12), '-=', n_92, n_diff_93)
            # Assigning a type to the variable 'n' (line 21)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'n', result_isub_94)
            
            
            # Getting the type of 'n_diff' (line 22)
            n_diff_95 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'n_diff')
            int_96 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 22), 'int')
            # Applying the binary operator '-=' (line 22)
            result_isub_97 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 12), '-=', n_diff_95, int_96)
            # Assigning a type to the variable 'n_diff' (line 22)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'n_diff', result_isub_97)
            
            # SSA join for if statement (line 20)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to xrange(...): (line 23)
        # Processing the call arguments (line 23)
        # Getting the type of 'n_diff' (line 23)
        n_diff_99 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 25), 'n_diff', False)
        int_100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 34), 'int')
        # Applying the binary operator '-' (line 23)
        result_sub_101 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 25), '-', n_diff_99, int_100)
        
        int_102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 40), 'int')
        # Applying the binary operator '<<' (line 23)
        result_lshift_103 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 24), '<<', result_sub_101, int_102)
        
        int_104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 43), 'int')
        int_105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 47), 'int')
        # Processing the call keyword arguments (line 23)
        kwargs_106 = {}
        # Getting the type of 'xrange' (line 23)
        xrange_98 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 23)
        xrange_call_result_107 = invoke(stypy.reporting.localization.Localization(__file__, 23, 17), xrange_98, *[result_lshift_103, int_104, int_105], **kwargs_106)
        
        # Testing if the for loop is going to be iterated (line 23)
        # Testing the type of a for loop iterable (line 23)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 23, 8), xrange_call_result_107)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 23, 8), xrange_call_result_107):
            # Getting the type of the for loop variable (line 23)
            for_loop_var_108 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 23, 8), xrange_call_result_107)
            # Assigning a type to the variable 'd' (line 23)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'd', for_loop_var_108)
            # SSA begins for a for statement (line 23)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a BinOp to a Name (line 24):
            
            # Assigning a BinOp to a Name (line 24):
            # Getting the type of 'n' (line 24)
            n_109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 16), 'n')
            int_110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 20), 'int')
            # Applying the binary operator '%' (line 24)
            result_mod_111 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 16), '%', n_109, int_110)
            
            # Assigning a type to the variable 'm' (line 24)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'm', result_mod_111)
            
            # Evaluating a boolean operation
            
            # Getting the type of 'm' (line 25)
            m_112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'm')
            int_113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 20), 'int')
            # Applying the binary operator '==' (line 25)
            result_eq_114 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 15), '==', m_112, int_113)
            
            
            # Getting the type of 'm' (line 25)
            m_115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 25), 'm')
            int_116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 30), 'int')
            # Applying the binary operator '==' (line 25)
            result_eq_117 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 25), '==', m_115, int_116)
            
            # Applying the binary operator 'or' (line 25)
            result_or_keyword_118 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 15), 'or', result_eq_114, result_eq_117)
            
            # Testing if the type of an if condition is none (line 25)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 25, 12), result_or_keyword_118):
                pass
            else:
                
                # Testing the type of an if condition (line 25)
                if_condition_119 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 25, 12), result_or_keyword_118)
                # Assigning a type to the variable 'if_condition_119' (line 25)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'if_condition_119', if_condition_119)
                # SSA begins for if statement (line 25)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a BinOp to a Name (line 26):
                
                # Assigning a BinOp to a Name (line 26):
                # Getting the type of 'n' (line 26)
                n_120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 20), 'n')
                int_121 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 25), 'int')
                # Applying the binary operator '>>' (line 26)
                result_rshift_122 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 20), '>>', n_120, int_121)
                
                # Assigning a type to the variable 'm' (line 26)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 16), 'm', result_rshift_122)
                
                # Assigning a UnaryOp to a Subscript (line 27):
                
                # Assigning a UnaryOp to a Subscript (line 27):
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'm' (line 27)
                m_123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 37), 'm')
                # Getting the type of 'sieve' (line 27)
                sieve_124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 31), 'sieve')
                # Obtaining the member '__getitem__' of a type (line 27)
                getitem___125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 31), sieve_124, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 27)
                subscript_call_result_126 = invoke(stypy.reporting.localization.Localization(__file__, 27, 31), getitem___125, m_123)
                
                # Applying the 'not' unary operator (line 27)
                result_not__127 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 27), 'not', subscript_call_result_126)
                
                # Getting the type of 'sieve' (line 27)
                sieve_128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 16), 'sieve')
                # Getting the type of 'm' (line 27)
                m_129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 22), 'm')
                # Storing an element on a container (line 27)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 16), sieve_128, (m_129, result_not__127))
                # SSA join for if statement (line 25)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'n' (line 28)
            n_130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'n')
            # Getting the type of 'd' (line 28)
            d_131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 17), 'd')
            # Applying the binary operator '-=' (line 28)
            result_isub_132 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 12), '-=', n_130, d_131)
            # Assigning a type to the variable 'n' (line 28)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'n', result_isub_132)
            
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
    end_135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 30), 'end', False)
    int_136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 36), 'int')
    # Applying the binary operator '-' (line 30)
    result_sub_137 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 30), '-', end_135, int_136)
    
    float_138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 41), 'float')
    # Applying the binary operator 'div' (line 30)
    result_div_139 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 29), 'div', result_sub_137, float_138)
    
    # Processing the call keyword arguments (line 30)
    kwargs_140 = {}
    # Getting the type of 'sqrt' (line 30)
    sqrt_134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 24), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 30)
    sqrt_call_result_141 = invoke(stypy.reporting.localization.Localization(__file__, 30, 24), sqrt_134, *[result_div_139], **kwargs_140)
    
    # Processing the call keyword arguments (line 30)
    kwargs_142 = {}
    # Getting the type of 'int' (line 30)
    int_133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 20), 'int', False)
    # Calling int(args, kwargs) (line 30)
    int_call_result_143 = invoke(stypy.reporting.localization.Localization(__file__, 30, 20), int_133, *[sqrt_call_result_141], **kwargs_142)
    
    # Assigning a type to the variable 'tuple_assignment_6' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'tuple_assignment_6', int_call_result_143)
    
    # Assigning a Num to a Name (line 30):
    int_144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 48), 'int')
    # Assigning a type to the variable 'tuple_assignment_7' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'tuple_assignment_7', int_144)
    
    # Assigning a Num to a Name (line 30):
    int_145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 51), 'int')
    # Assigning a type to the variable 'tuple_assignment_8' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'tuple_assignment_8', int_145)
    
    # Assigning a Name to a Name (line 30):
    # Getting the type of 'tuple_assignment_6' (line 30)
    tuple_assignment_6_146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'tuple_assignment_6')
    # Assigning a type to the variable 'x_max' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'x_max', tuple_assignment_6_146)
    
    # Assigning a Name to a Name (line 30):
    # Getting the type of 'tuple_assignment_7' (line 30)
    tuple_assignment_7_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'tuple_assignment_7')
    # Assigning a type to the variable 'x2' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 11), 'x2', tuple_assignment_7_147)
    
    # Assigning a Name to a Name (line 30):
    # Getting the type of 'tuple_assignment_8' (line 30)
    tuple_assignment_8_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'tuple_assignment_8')
    # Assigning a type to the variable 'xd' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 15), 'xd', tuple_assignment_8_148)
    
    
    # Call to xrange(...): (line 31)
    # Processing the call arguments (line 31)
    int_150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 21), 'int')
    int_151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 24), 'int')
    # Getting the type of 'x_max' (line 31)
    x_max_152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 28), 'x_max', False)
    # Applying the binary operator '*' (line 31)
    result_mul_153 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 24), '*', int_151, x_max_152)
    
    int_154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 36), 'int')
    # Applying the binary operator '+' (line 31)
    result_add_155 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 24), '+', result_mul_153, int_154)
    
    int_156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 39), 'int')
    # Processing the call keyword arguments (line 31)
    kwargs_157 = {}
    # Getting the type of 'xrange' (line 31)
    xrange_149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 14), 'xrange', False)
    # Calling xrange(args, kwargs) (line 31)
    xrange_call_result_158 = invoke(stypy.reporting.localization.Localization(__file__, 31, 14), xrange_149, *[int_150, result_add_155, int_156], **kwargs_157)
    
    # Testing if the for loop is going to be iterated (line 31)
    # Testing the type of a for loop iterable (line 31)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 31, 4), xrange_call_result_158)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 31, 4), xrange_call_result_158):
        # Getting the type of the for loop variable (line 31)
        for_loop_var_159 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 31, 4), xrange_call_result_158)
        # Assigning a type to the variable 'xd' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'xd', for_loop_var_159)
        # SSA begins for a for statement (line 31)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'x2' (line 32)
        x2_160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'x2')
        # Getting the type of 'xd' (line 32)
        xd_161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 14), 'xd')
        # Applying the binary operator '+=' (line 32)
        result_iadd_162 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 8), '+=', x2_160, xd_161)
        # Assigning a type to the variable 'x2' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'x2', result_iadd_162)
        
        
        # Assigning a Call to a Name (line 33):
        
        # Assigning a Call to a Name (line 33):
        
        # Call to int(...): (line 33)
        # Processing the call arguments (line 33)
        
        # Call to sqrt(...): (line 33)
        # Processing the call arguments (line 33)
        # Getting the type of 'end' (line 33)
        end_165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 25), 'end', False)
        # Getting the type of 'x2' (line 33)
        x2_166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 31), 'x2', False)
        # Applying the binary operator '-' (line 33)
        result_sub_167 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 25), '-', end_165, x2_166)
        
        # Processing the call keyword arguments (line 33)
        kwargs_168 = {}
        # Getting the type of 'sqrt' (line 33)
        sqrt_164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 20), 'sqrt', False)
        # Calling sqrt(args, kwargs) (line 33)
        sqrt_call_result_169 = invoke(stypy.reporting.localization.Localization(__file__, 33, 20), sqrt_164, *[result_sub_167], **kwargs_168)
        
        # Processing the call keyword arguments (line 33)
        kwargs_170 = {}
        # Getting the type of 'int' (line 33)
        int_163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 16), 'int', False)
        # Calling int(args, kwargs) (line 33)
        int_call_result_171 = invoke(stypy.reporting.localization.Localization(__file__, 33, 16), int_163, *[sqrt_call_result_169], **kwargs_170)
        
        # Assigning a type to the variable 'y_max' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'y_max', int_call_result_171)
        
        # Assigning a Tuple to a Tuple (line 34):
        
        # Assigning a BinOp to a Name (line 34):
        # Getting the type of 'x2' (line 34)
        x2_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 20), 'x2')
        # Getting the type of 'y_max' (line 34)
        y_max_173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 25), 'y_max')
        # Getting the type of 'y_max' (line 34)
        y_max_174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 33), 'y_max')
        # Applying the binary operator '*' (line 34)
        result_mul_175 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 25), '*', y_max_173, y_max_174)
        
        # Applying the binary operator '+' (line 34)
        result_add_176 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 20), '+', x2_172, result_mul_175)
        
        # Assigning a type to the variable 'tuple_assignment_9' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'tuple_assignment_9', result_add_176)
        
        # Assigning a BinOp to a Name (line 34):
        # Getting the type of 'y_max' (line 34)
        y_max_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 41), 'y_max')
        int_178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 50), 'int')
        # Applying the binary operator '<<' (line 34)
        result_lshift_179 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 41), '<<', y_max_177, int_178)
        
        int_180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 55), 'int')
        # Applying the binary operator '-' (line 34)
        result_sub_181 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 40), '-', result_lshift_179, int_180)
        
        # Assigning a type to the variable 'tuple_assignment_10' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'tuple_assignment_10', result_sub_181)
        
        # Assigning a Name to a Name (line 34):
        # Getting the type of 'tuple_assignment_9' (line 34)
        tuple_assignment_9_182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'tuple_assignment_9')
        # Assigning a type to the variable 'n' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'n', tuple_assignment_9_182)
        
        # Assigning a Name to a Name (line 34):
        # Getting the type of 'tuple_assignment_10' (line 34)
        tuple_assignment_10_183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'tuple_assignment_10')
        # Assigning a type to the variable 'n_diff' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 11), 'n_diff', tuple_assignment_10_183)
        
        # Getting the type of 'n' (line 35)
        n_184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 16), 'n')
        int_185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 20), 'int')
        # Applying the binary operator '&' (line 35)
        result_and__186 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 16), '&', n_184, int_185)
        
        # Applying the 'not' unary operator (line 35)
        result_not__187 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 11), 'not', result_and__186)
        
        # Testing if the type of an if condition is none (line 35)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 35, 8), result_not__187):
            pass
        else:
            
            # Testing the type of an if condition (line 35)
            if_condition_188 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 35, 8), result_not__187)
            # Assigning a type to the variable 'if_condition_188' (line 35)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'if_condition_188', if_condition_188)
            # SSA begins for if statement (line 35)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'n' (line 36)
            n_189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'n')
            # Getting the type of 'n_diff' (line 36)
            n_diff_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 17), 'n_diff')
            # Applying the binary operator '-=' (line 36)
            result_isub_191 = python_operator(stypy.reporting.localization.Localization(__file__, 36, 12), '-=', n_189, n_diff_190)
            # Assigning a type to the variable 'n' (line 36)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'n', result_isub_191)
            
            
            # Getting the type of 'n_diff' (line 37)
            n_diff_192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'n_diff')
            int_193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 22), 'int')
            # Applying the binary operator '-=' (line 37)
            result_isub_194 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 12), '-=', n_diff_192, int_193)
            # Assigning a type to the variable 'n_diff' (line 37)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'n_diff', result_isub_194)
            
            # SSA join for if statement (line 35)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Call to xrange(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'n_diff' (line 38)
        n_diff_196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 25), 'n_diff', False)
        int_197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 34), 'int')
        # Applying the binary operator '-' (line 38)
        result_sub_198 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 25), '-', n_diff_196, int_197)
        
        int_199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 40), 'int')
        # Applying the binary operator '<<' (line 38)
        result_lshift_200 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 24), '<<', result_sub_198, int_199)
        
        int_201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 43), 'int')
        int_202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 47), 'int')
        # Processing the call keyword arguments (line 38)
        kwargs_203 = {}
        # Getting the type of 'xrange' (line 38)
        xrange_195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 38)
        xrange_call_result_204 = invoke(stypy.reporting.localization.Localization(__file__, 38, 17), xrange_195, *[result_lshift_200, int_201, int_202], **kwargs_203)
        
        # Testing if the for loop is going to be iterated (line 38)
        # Testing the type of a for loop iterable (line 38)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 38, 8), xrange_call_result_204)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 38, 8), xrange_call_result_204):
            # Getting the type of the for loop variable (line 38)
            for_loop_var_205 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 38, 8), xrange_call_result_204)
            # Assigning a type to the variable 'd' (line 38)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'd', for_loop_var_205)
            # SSA begins for a for statement (line 38)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'n' (line 39)
            n_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 15), 'n')
            int_207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 19), 'int')
            # Applying the binary operator '%' (line 39)
            result_mod_208 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 15), '%', n_206, int_207)
            
            int_209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 25), 'int')
            # Applying the binary operator '==' (line 39)
            result_eq_210 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 15), '==', result_mod_208, int_209)
            
            # Testing if the type of an if condition is none (line 39)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 39, 12), result_eq_210):
                pass
            else:
                
                # Testing the type of an if condition (line 39)
                if_condition_211 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 39, 12), result_eq_210)
                # Assigning a type to the variable 'if_condition_211' (line 39)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'if_condition_211', if_condition_211)
                # SSA begins for if statement (line 39)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a BinOp to a Name (line 40):
                
                # Assigning a BinOp to a Name (line 40):
                # Getting the type of 'n' (line 40)
                n_212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 20), 'n')
                int_213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 25), 'int')
                # Applying the binary operator '>>' (line 40)
                result_rshift_214 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 20), '>>', n_212, int_213)
                
                # Assigning a type to the variable 'm' (line 40)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 16), 'm', result_rshift_214)
                
                # Assigning a UnaryOp to a Subscript (line 41):
                
                # Assigning a UnaryOp to a Subscript (line 41):
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'm' (line 41)
                m_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 37), 'm')
                # Getting the type of 'sieve' (line 41)
                sieve_216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 31), 'sieve')
                # Obtaining the member '__getitem__' of a type (line 41)
                getitem___217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 31), sieve_216, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 41)
                subscript_call_result_218 = invoke(stypy.reporting.localization.Localization(__file__, 41, 31), getitem___217, m_215)
                
                # Applying the 'not' unary operator (line 41)
                result_not__219 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 27), 'not', subscript_call_result_218)
                
                # Getting the type of 'sieve' (line 41)
                sieve_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 16), 'sieve')
                # Getting the type of 'm' (line 41)
                m_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 22), 'm')
                # Storing an element on a container (line 41)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 16), sieve_220, (m_221, result_not__219))
                # SSA join for if statement (line 39)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'n' (line 42)
            n_222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'n')
            # Getting the type of 'd' (line 42)
            d_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 17), 'd')
            # Applying the binary operator '-=' (line 42)
            result_isub_224 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 12), '-=', n_222, d_223)
            # Assigning a type to the variable 'n' (line 42)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'n', result_isub_224)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Assigning a Tuple to a Tuple (line 44):
    
    # Assigning a Call to a Name (line 44):
    
    # Call to int(...): (line 44)
    # Processing the call arguments (line 44)
    int_226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 32), 'int')
    
    # Call to sqrt(...): (line 44)
    # Processing the call arguments (line 44)
    int_228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 41), 'int')
    int_229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 45), 'int')
    int_230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 50), 'int')
    # Getting the type of 'end' (line 44)
    end_231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 54), 'end', False)
    # Applying the binary operator '-' (line 44)
    result_sub_232 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 50), '-', int_230, end_231)
    
    # Applying the binary operator '*' (line 44)
    result_mul_233 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 45), '*', int_229, result_sub_232)
    
    # Applying the binary operator '-' (line 44)
    result_sub_234 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 41), '-', int_228, result_mul_233)
    
    # Processing the call keyword arguments (line 44)
    kwargs_235 = {}
    # Getting the type of 'sqrt' (line 44)
    sqrt_227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 36), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 44)
    sqrt_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 44, 36), sqrt_227, *[result_sub_234], **kwargs_235)
    
    # Applying the binary operator '+' (line 44)
    result_add_237 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 32), '+', int_226, sqrt_call_result_236)
    
    int_238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 63), 'int')
    # Applying the binary operator 'div' (line 44)
    result_div_239 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 31), 'div', result_add_237, int_238)
    
    # Processing the call keyword arguments (line 44)
    kwargs_240 = {}
    # Getting the type of 'int' (line 44)
    int_225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 27), 'int', False)
    # Calling int(args, kwargs) (line 44)
    int_call_result_241 = invoke(stypy.reporting.localization.Localization(__file__, 44, 27), int_225, *[result_div_239], **kwargs_240)
    
    # Assigning a type to the variable 'tuple_assignment_11' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'tuple_assignment_11', int_call_result_241)
    
    # Assigning a Num to a Name (line 44):
    int_242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 67), 'int')
    # Assigning a type to the variable 'tuple_assignment_12' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'tuple_assignment_12', int_242)
    
    # Assigning a Num to a Name (line 44):
    int_243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 71), 'int')
    # Assigning a type to the variable 'tuple_assignment_13' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'tuple_assignment_13', int_243)
    
    # Assigning a Num to a Name (line 44):
    int_244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 74), 'int')
    # Assigning a type to the variable 'tuple_assignment_14' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'tuple_assignment_14', int_244)
    
    # Assigning a Name to a Name (line 44):
    # Getting the type of 'tuple_assignment_11' (line 44)
    tuple_assignment_11_245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'tuple_assignment_11')
    # Assigning a type to the variable 'x_max' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'x_max', tuple_assignment_11_245)
    
    # Assigning a Name to a Name (line 44):
    # Getting the type of 'tuple_assignment_12' (line 44)
    tuple_assignment_12_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'tuple_assignment_12')
    # Assigning a type to the variable 'y_min' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 11), 'y_min', tuple_assignment_12_246)
    
    # Assigning a Name to a Name (line 44):
    # Getting the type of 'tuple_assignment_13' (line 44)
    tuple_assignment_13_247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'tuple_assignment_13')
    # Assigning a type to the variable 'x2' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 18), 'x2', tuple_assignment_13_247)
    
    # Assigning a Name to a Name (line 44):
    # Getting the type of 'tuple_assignment_14' (line 44)
    tuple_assignment_14_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'tuple_assignment_14')
    # Assigning a type to the variable 'xd' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 22), 'xd', tuple_assignment_14_248)
    
    
    # Call to xrange(...): (line 45)
    # Processing the call arguments (line 45)
    int_250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 20), 'int')
    # Getting the type of 'x_max' (line 45)
    x_max_251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 23), 'x_max', False)
    int_252 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 31), 'int')
    # Applying the binary operator '+' (line 45)
    result_add_253 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 23), '+', x_max_251, int_252)
    
    # Processing the call keyword arguments (line 45)
    kwargs_254 = {}
    # Getting the type of 'xrange' (line 45)
    xrange_249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 45)
    xrange_call_result_255 = invoke(stypy.reporting.localization.Localization(__file__, 45, 13), xrange_249, *[int_250, result_add_253], **kwargs_254)
    
    # Testing if the for loop is going to be iterated (line 45)
    # Testing the type of a for loop iterable (line 45)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 45, 4), xrange_call_result_255)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 45, 4), xrange_call_result_255):
        # Getting the type of the for loop variable (line 45)
        for_loop_var_256 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 45, 4), xrange_call_result_255)
        # Assigning a type to the variable 'x' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'x', for_loop_var_256)
        # SSA begins for a for statement (line 45)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'x2' (line 46)
        x2_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'x2')
        # Getting the type of 'xd' (line 46)
        xd_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 14), 'xd')
        # Applying the binary operator '+=' (line 46)
        result_iadd_259 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 8), '+=', x2_257, xd_258)
        # Assigning a type to the variable 'x2' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'x2', result_iadd_259)
        
        
        # Getting the type of 'xd' (line 47)
        xd_260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'xd')
        int_261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 14), 'int')
        # Applying the binary operator '+=' (line 47)
        result_iadd_262 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 8), '+=', xd_260, int_261)
        # Assigning a type to the variable 'xd' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'xd', result_iadd_262)
        
        
        # Getting the type of 'x2' (line 48)
        x2_263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 11), 'x2')
        # Getting the type of 'end' (line 48)
        end_264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 17), 'end')
        # Applying the binary operator '>=' (line 48)
        result_ge_265 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 11), '>=', x2_263, end_264)
        
        # Testing if the type of an if condition is none (line 48)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 48, 8), result_ge_265):
            pass
        else:
            
            # Testing the type of an if condition (line 48)
            if_condition_266 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 48, 8), result_ge_265)
            # Assigning a type to the variable 'if_condition_266' (line 48)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'if_condition_266', if_condition_266)
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
            x2_270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 47), 'x2', False)
            # Getting the type of 'end' (line 48)
            end_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 52), 'end', False)
            # Applying the binary operator '-' (line 48)
            result_sub_272 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 47), '-', x2_270, end_271)
            
            # Processing the call keyword arguments (line 48)
            kwargs_273 = {}
            # Getting the type of 'sqrt' (line 48)
            sqrt_269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 42), 'sqrt', False)
            # Calling sqrt(args, kwargs) (line 48)
            sqrt_call_result_274 = invoke(stypy.reporting.localization.Localization(__file__, 48, 42), sqrt_269, *[result_sub_272], **kwargs_273)
            
            # Processing the call keyword arguments (line 48)
            kwargs_275 = {}
            # Getting the type of 'ceil' (line 48)
            ceil_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 37), 'ceil', False)
            # Calling ceil(args, kwargs) (line 48)
            ceil_call_result_276 = invoke(stypy.reporting.localization.Localization(__file__, 48, 37), ceil_268, *[sqrt_call_result_274], **kwargs_275)
            
            # Processing the call keyword arguments (line 48)
            kwargs_277 = {}
            # Getting the type of 'int' (line 48)
            int_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 33), 'int', False)
            # Calling int(args, kwargs) (line 48)
            int_call_result_278 = invoke(stypy.reporting.localization.Localization(__file__, 48, 33), int_267, *[ceil_call_result_276], **kwargs_277)
            
            int_279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 61), 'int')
            # Applying the binary operator '-' (line 48)
            result_sub_280 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 33), '-', int_call_result_278, int_279)
            
            int_281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 67), 'int')
            # Applying the binary operator '<<' (line 48)
            result_lshift_282 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 32), '<<', result_sub_280, int_281)
            
            int_283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 72), 'int')
            # Applying the binary operator '-' (line 48)
            result_sub_284 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 31), '-', result_lshift_282, int_283)
            
            int_285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 78), 'int')
            # Applying the binary operator '<<' (line 48)
            result_lshift_286 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 30), '<<', result_sub_284, int_285)
            
            # Assigning a type to the variable 'y_min' (line 48)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 22), 'y_min', result_lshift_286)
            # SSA join for if statement (line 48)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Tuple to a Tuple (line 49):
        
        # Assigning a BinOp to a Name (line 49):
        # Getting the type of 'x' (line 49)
        x_287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 22), 'x')
        # Getting the type of 'x' (line 49)
        x_288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 26), 'x')
        # Applying the binary operator '*' (line 49)
        result_mul_289 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 22), '*', x_287, x_288)
        
        # Getting the type of 'x' (line 49)
        x_290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 30), 'x')
        # Applying the binary operator '+' (line 49)
        result_add_291 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 22), '+', result_mul_289, x_290)
        
        int_292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 36), 'int')
        # Applying the binary operator '<<' (line 49)
        result_lshift_293 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 21), '<<', result_add_291, int_292)
        
        int_294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 41), 'int')
        # Applying the binary operator '-' (line 49)
        result_sub_295 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 20), '-', result_lshift_293, int_294)
        
        # Assigning a type to the variable 'tuple_assignment_15' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_assignment_15', result_sub_295)
        
        # Assigning a BinOp to a Name (line 49):
        # Getting the type of 'x' (line 49)
        x_296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 47), 'x')
        int_297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 51), 'int')
        # Applying the binary operator '-' (line 49)
        result_sub_298 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 47), '-', x_296, int_297)
        
        int_299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 57), 'int')
        # Applying the binary operator '<<' (line 49)
        result_lshift_300 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 46), '<<', result_sub_298, int_299)
        
        int_301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 62), 'int')
        # Applying the binary operator '-' (line 49)
        result_sub_302 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 45), '-', result_lshift_300, int_301)
        
        int_303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 68), 'int')
        # Applying the binary operator '<<' (line 49)
        result_lshift_304 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 44), '<<', result_sub_302, int_303)
        
        # Assigning a type to the variable 'tuple_assignment_16' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_assignment_16', result_lshift_304)
        
        # Assigning a Name to a Name (line 49):
        # Getting the type of 'tuple_assignment_15' (line 49)
        tuple_assignment_15_305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_assignment_15')
        # Assigning a type to the variable 'n' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'n', tuple_assignment_15_305)
        
        # Assigning a Name to a Name (line 49):
        # Getting the type of 'tuple_assignment_16' (line 49)
        tuple_assignment_16_306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'tuple_assignment_16')
        # Assigning a type to the variable 'n_diff' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 11), 'n_diff', tuple_assignment_16_306)
        
        
        # Call to xrange(...): (line 50)
        # Processing the call arguments (line 50)
        # Getting the type of 'n_diff' (line 50)
        n_diff_308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 24), 'n_diff', False)
        # Getting the type of 'y_min' (line 50)
        y_min_309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 32), 'y_min', False)
        int_310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 39), 'int')
        # Processing the call keyword arguments (line 50)
        kwargs_311 = {}
        # Getting the type of 'xrange' (line 50)
        xrange_307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 50)
        xrange_call_result_312 = invoke(stypy.reporting.localization.Localization(__file__, 50, 17), xrange_307, *[n_diff_308, y_min_309, int_310], **kwargs_311)
        
        # Testing if the for loop is going to be iterated (line 50)
        # Testing the type of a for loop iterable (line 50)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 50, 8), xrange_call_result_312)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 50, 8), xrange_call_result_312):
            # Getting the type of the for loop variable (line 50)
            for_loop_var_313 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 50, 8), xrange_call_result_312)
            # Assigning a type to the variable 'd' (line 50)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'd', for_loop_var_313)
            # SSA begins for a for statement (line 50)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'n' (line 51)
            n_314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 15), 'n')
            int_315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 19), 'int')
            # Applying the binary operator '%' (line 51)
            result_mod_316 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 15), '%', n_314, int_315)
            
            int_317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 25), 'int')
            # Applying the binary operator '==' (line 51)
            result_eq_318 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 15), '==', result_mod_316, int_317)
            
            # Testing if the type of an if condition is none (line 51)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 51, 12), result_eq_318):
                pass
            else:
                
                # Testing the type of an if condition (line 51)
                if_condition_319 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 51, 12), result_eq_318)
                # Assigning a type to the variable 'if_condition_319' (line 51)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'if_condition_319', if_condition_319)
                # SSA begins for if statement (line 51)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a BinOp to a Name (line 52):
                
                # Assigning a BinOp to a Name (line 52):
                # Getting the type of 'n' (line 52)
                n_320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 20), 'n')
                int_321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 25), 'int')
                # Applying the binary operator '>>' (line 52)
                result_rshift_322 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 20), '>>', n_320, int_321)
                
                # Assigning a type to the variable 'm' (line 52)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 16), 'm', result_rshift_322)
                
                # Assigning a UnaryOp to a Subscript (line 53):
                
                # Assigning a UnaryOp to a Subscript (line 53):
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'm' (line 53)
                m_323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 37), 'm')
                # Getting the type of 'sieve' (line 53)
                sieve_324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 31), 'sieve')
                # Obtaining the member '__getitem__' of a type (line 53)
                getitem___325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 31), sieve_324, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 53)
                subscript_call_result_326 = invoke(stypy.reporting.localization.Localization(__file__, 53, 31), getitem___325, m_323)
                
                # Applying the 'not' unary operator (line 53)
                result_not__327 = python_operator(stypy.reporting.localization.Localization(__file__, 53, 27), 'not', subscript_call_result_326)
                
                # Getting the type of 'sieve' (line 53)
                sieve_328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 16), 'sieve')
                # Getting the type of 'm' (line 53)
                m_329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 22), 'm')
                # Storing an element on a container (line 53)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 16), sieve_328, (m_329, result_not__327))
                # SSA join for if statement (line 51)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Getting the type of 'n' (line 54)
            n_330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'n')
            # Getting the type of 'd' (line 54)
            d_331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 17), 'd')
            # Applying the binary operator '+=' (line 54)
            result_iadd_332 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 12), '+=', n_330, d_331)
            # Assigning a type to the variable 'n' (line 54)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'n', result_iadd_332)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Assigning a List to a Name (line 56):
    
    # Assigning a List to a Name (line 56):
    
    # Obtaining an instance of the builtin type 'list' (line 56)
    list_333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 56)
    # Adding element type (line 56)
    int_334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 14), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 13), list_333, int_334)
    # Adding element type (line 56)
    int_335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 56, 13), list_333, int_335)
    
    # Assigning a type to the variable 'primes' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'primes', list_333)
    
    # Getting the type of 'end' (line 57)
    end_336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 7), 'end')
    int_337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 14), 'int')
    # Applying the binary operator '<=' (line 57)
    result_le_338 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 7), '<=', end_336, int_337)
    
    # Testing if the type of an if condition is none (line 57)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 57, 4), result_le_338):
        pass
    else:
        
        # Testing the type of an if condition (line 57)
        if_condition_339 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 57, 4), result_le_338)
        # Assigning a type to the variable 'if_condition_339' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'if_condition_339', if_condition_339)
        # SSA begins for if statement (line 57)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining the type of the subscript
        
        # Call to max(...): (line 58)
        # Processing the call arguments (line 58)
        int_341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 27), 'int')
        # Getting the type of 'end' (line 58)
        end_342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 30), 'end', False)
        int_343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 36), 'int')
        # Applying the binary operator '-' (line 58)
        result_sub_344 = python_operator(stypy.reporting.localization.Localization(__file__, 58, 30), '-', end_342, int_343)
        
        # Processing the call keyword arguments (line 58)
        kwargs_345 = {}
        # Getting the type of 'max' (line 58)
        max_340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 23), 'max', False)
        # Calling max(args, kwargs) (line 58)
        max_call_result_346 = invoke(stypy.reporting.localization.Localization(__file__, 58, 23), max_340, *[int_341, result_sub_344], **kwargs_345)
        
        slice_347 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 58, 15), None, max_call_result_346, None)
        # Getting the type of 'primes' (line 58)
        primes_348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 15), 'primes')
        # Obtaining the member '__getitem__' of a type (line 58)
        getitem___349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 15), primes_348, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 58)
        subscript_call_result_350 = invoke(stypy.reporting.localization.Localization(__file__, 58, 15), getitem___349, slice_347)
        
        # Assigning a type to the variable 'stypy_return_type' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'stypy_return_type', subscript_call_result_350)
        # SSA join for if statement (line 57)
        module_type_store = module_type_store.join_ssa_context()
        

    
    
    # Call to xrange(...): (line 60)
    # Processing the call arguments (line 60)
    int_352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 20), 'int')
    int_353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 25), 'int')
    # Applying the binary operator '>>' (line 60)
    result_rshift_354 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 20), '>>', int_352, int_353)
    
    
    # Call to int(...): (line 60)
    # Processing the call arguments (line 60)
    
    # Call to sqrt(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'end' (line 60)
    end_357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 38), 'end', False)
    # Processing the call keyword arguments (line 60)
    kwargs_358 = {}
    # Getting the type of 'sqrt' (line 60)
    sqrt_356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 33), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 60)
    sqrt_call_result_359 = invoke(stypy.reporting.localization.Localization(__file__, 60, 33), sqrt_356, *[end_357], **kwargs_358)
    
    # Processing the call keyword arguments (line 60)
    kwargs_360 = {}
    # Getting the type of 'int' (line 60)
    int_355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 29), 'int', False)
    # Calling int(args, kwargs) (line 60)
    int_call_result_361 = invoke(stypy.reporting.localization.Localization(__file__, 60, 29), int_355, *[sqrt_call_result_359], **kwargs_360)
    
    int_362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 46), 'int')
    # Applying the binary operator '+' (line 60)
    result_add_363 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 29), '+', int_call_result_361, int_362)
    
    int_364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 52), 'int')
    # Applying the binary operator '>>' (line 60)
    result_rshift_365 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 28), '>>', result_add_363, int_364)
    
    # Processing the call keyword arguments (line 60)
    kwargs_366 = {}
    # Getting the type of 'xrange' (line 60)
    xrange_351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 60)
    xrange_call_result_367 = invoke(stypy.reporting.localization.Localization(__file__, 60, 13), xrange_351, *[result_rshift_354, result_rshift_365], **kwargs_366)
    
    # Testing if the for loop is going to be iterated (line 60)
    # Testing the type of a for loop iterable (line 60)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 60, 4), xrange_call_result_367)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 60, 4), xrange_call_result_367):
        # Getting the type of the for loop variable (line 60)
        for_loop_var_368 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 60, 4), xrange_call_result_367)
        # Assigning a type to the variable 'n' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'n', for_loop_var_368)
        # SSA begins for a for statement (line 60)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Obtaining the type of the subscript
        # Getting the type of 'n' (line 61)
        n_369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 17), 'n')
        # Getting the type of 'sieve' (line 61)
        sieve_370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 11), 'sieve')
        # Obtaining the member '__getitem__' of a type (line 61)
        getitem___371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 11), sieve_370, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 61)
        subscript_call_result_372 = invoke(stypy.reporting.localization.Localization(__file__, 61, 11), getitem___371, n_369)
        
        # Testing if the type of an if condition is none (line 61)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 61, 8), subscript_call_result_372):
            pass
        else:
            
            # Testing the type of an if condition (line 61)
            if_condition_373 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 61, 8), subscript_call_result_372)
            # Assigning a type to the variable 'if_condition_373' (line 61)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'if_condition_373', if_condition_373)
            # SSA begins for if statement (line 61)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 62)
            # Processing the call arguments (line 62)
            # Getting the type of 'n' (line 62)
            n_376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 27), 'n', False)
            int_377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 32), 'int')
            # Applying the binary operator '<<' (line 62)
            result_lshift_378 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 27), '<<', n_376, int_377)
            
            int_379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 37), 'int')
            # Applying the binary operator '+' (line 62)
            result_add_380 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 26), '+', result_lshift_378, int_379)
            
            # Processing the call keyword arguments (line 62)
            kwargs_381 = {}
            # Getting the type of 'primes' (line 62)
            primes_374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'primes', False)
            # Obtaining the member 'append' of a type (line 62)
            append_375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 12), primes_374, 'append')
            # Calling append(args, kwargs) (line 62)
            append_call_result_382 = invoke(stypy.reporting.localization.Localization(__file__, 62, 12), append_375, *[result_add_380], **kwargs_381)
            
            
            # Assigning a BinOp to a Name (line 63):
            
            # Assigning a BinOp to a Name (line 63):
            # Getting the type of 'n' (line 63)
            n_383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 19), 'n')
            int_384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 24), 'int')
            # Applying the binary operator '<<' (line 63)
            result_lshift_385 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 19), '<<', n_383, int_384)
            
            int_386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 29), 'int')
            # Applying the binary operator '+' (line 63)
            result_add_387 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 18), '+', result_lshift_385, int_386)
            
            # Assigning a type to the variable 'aux' (line 63)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'aux', result_add_387)
            
            # Getting the type of 'aux' (line 64)
            aux_388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'aux')
            # Getting the type of 'aux' (line 64)
            aux_389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 19), 'aux')
            # Applying the binary operator '*=' (line 64)
            result_imul_390 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 12), '*=', aux_388, aux_389)
            # Assigning a type to the variable 'aux' (line 64)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'aux', result_imul_390)
            
            
            
            # Call to xrange(...): (line 65)
            # Processing the call arguments (line 65)
            # Getting the type of 'aux' (line 65)
            aux_392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 28), 'aux', False)
            # Getting the type of 'end' (line 65)
            end_393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 33), 'end', False)
            int_394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 38), 'int')
            # Getting the type of 'aux' (line 65)
            aux_395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 42), 'aux', False)
            # Applying the binary operator '*' (line 65)
            result_mul_396 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 38), '*', int_394, aux_395)
            
            # Processing the call keyword arguments (line 65)
            kwargs_397 = {}
            # Getting the type of 'xrange' (line 65)
            xrange_391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 21), 'xrange', False)
            # Calling xrange(args, kwargs) (line 65)
            xrange_call_result_398 = invoke(stypy.reporting.localization.Localization(__file__, 65, 21), xrange_391, *[aux_392, end_393, result_mul_396], **kwargs_397)
            
            # Testing if the for loop is going to be iterated (line 65)
            # Testing the type of a for loop iterable (line 65)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 65, 12), xrange_call_result_398)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 65, 12), xrange_call_result_398):
                # Getting the type of the for loop variable (line 65)
                for_loop_var_399 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 65, 12), xrange_call_result_398)
                # Assigning a type to the variable 'k' (line 65)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'k', for_loop_var_399)
                # SSA begins for a for statement (line 65)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Name to a Subscript (line 66):
                
                # Assigning a Name to a Subscript (line 66):
                # Getting the type of 'False' (line 66)
                False_400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 32), 'False')
                # Getting the type of 'sieve' (line 66)
                sieve_401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 16), 'sieve')
                # Getting the type of 'k' (line 66)
                k_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 22), 'k')
                int_403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 27), 'int')
                # Applying the binary operator '>>' (line 66)
                result_rshift_404 = python_operator(stypy.reporting.localization.Localization(__file__, 66, 22), '>>', k_402, int_403)
                
                # Storing an element on a container (line 66)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 16), sieve_401, (result_rshift_404, False_400))
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
    end_407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 17), 'end', False)
    # Processing the call keyword arguments (line 68)
    kwargs_408 = {}
    # Getting the type of 'sqrt' (line 68)
    sqrt_406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 68)
    sqrt_call_result_409 = invoke(stypy.reporting.localization.Localization(__file__, 68, 12), sqrt_406, *[end_407], **kwargs_408)
    
    # Processing the call keyword arguments (line 68)
    kwargs_410 = {}
    # Getting the type of 'int' (line 68)
    int_405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'int', False)
    # Calling int(args, kwargs) (line 68)
    int_call_result_411 = invoke(stypy.reporting.localization.Localization(__file__, 68, 8), int_405, *[sqrt_call_result_409], **kwargs_410)
    
    int_412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 25), 'int')
    # Applying the binary operator '+' (line 68)
    result_add_413 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 8), '+', int_call_result_411, int_412)
    
    # Assigning a type to the variable 's' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 's', result_add_413)
    
    # Getting the type of 's' (line 69)
    s_414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 7), 's')
    int_415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 11), 'int')
    # Applying the binary operator '%' (line 69)
    result_mod_416 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 7), '%', s_414, int_415)
    
    int_417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 16), 'int')
    # Applying the binary operator '==' (line 69)
    result_eq_418 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 7), '==', result_mod_416, int_417)
    
    # Testing if the type of an if condition is none (line 69)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 69, 4), result_eq_418):
        pass
    else:
        
        # Testing the type of an if condition (line 69)
        if_condition_419 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 4), result_eq_418)
        # Assigning a type to the variable 'if_condition_419' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'if_condition_419', if_condition_419)
        # SSA begins for if statement (line 69)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 's' (line 70)
        s_420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 's')
        int_421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 13), 'int')
        # Applying the binary operator '+=' (line 70)
        result_iadd_422 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 8), '+=', s_420, int_421)
        # Assigning a type to the variable 's' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 's', result_iadd_422)
        
        # SSA join for if statement (line 69)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Call to extend(...): (line 71)
    # Processing the call arguments (line 71)
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to xrange(...): (line 71)
    # Processing the call arguments (line 71)
    # Getting the type of 's' (line 71)
    s_433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 37), 's', False)
    # Getting the type of 'end' (line 71)
    end_434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 40), 'end', False)
    int_435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 45), 'int')
    # Processing the call keyword arguments (line 71)
    kwargs_436 = {}
    # Getting the type of 'xrange' (line 71)
    xrange_432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 30), 'xrange', False)
    # Calling xrange(args, kwargs) (line 71)
    xrange_call_result_437 = invoke(stypy.reporting.localization.Localization(__file__, 71, 30), xrange_432, *[s_433, end_434, int_435], **kwargs_436)
    
    comprehension_438 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 19), xrange_call_result_437)
    # Assigning a type to the variable 'i' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 19), 'i', comprehension_438)
    
    # Obtaining the type of the subscript
    # Getting the type of 'i' (line 71)
    i_426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 57), 'i', False)
    int_427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 62), 'int')
    # Applying the binary operator '>>' (line 71)
    result_rshift_428 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 57), '>>', i_426, int_427)
    
    # Getting the type of 'sieve' (line 71)
    sieve_429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 51), 'sieve', False)
    # Obtaining the member '__getitem__' of a type (line 71)
    getitem___430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 51), sieve_429, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 71)
    subscript_call_result_431 = invoke(stypy.reporting.localization.Localization(__file__, 71, 51), getitem___430, result_rshift_428)
    
    # Getting the type of 'i' (line 71)
    i_425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 19), 'i', False)
    list_439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 19), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 19), list_439, i_425)
    # Processing the call keyword arguments (line 71)
    kwargs_440 = {}
    # Getting the type of 'primes' (line 71)
    primes_423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'primes', False)
    # Obtaining the member 'extend' of a type (line 71)
    extend_424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 4), primes_423, 'extend')
    # Calling extend(args, kwargs) (line 71)
    extend_call_result_441 = invoke(stypy.reporting.localization.Localization(__file__, 71, 4), extend_424, *[list_439], **kwargs_440)
    
    # Getting the type of 'primes' (line 73)
    primes_442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 11), 'primes')
    # Assigning a type to the variable 'stypy_return_type' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'stypy_return_type', primes_442)
    
    # ################# End of 'sieveOfAtkin(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sieveOfAtkin' in the type store
    # Getting the type of 'stypy_return_type' (line 5)
    stypy_return_type_443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_443)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sieveOfAtkin'
    return stypy_return_type_443

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

    str_444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 4), 'str', 'sieveOfEratostenes(n): return the list of the primes < n.')
    
    # Getting the type of 'n' (line 80)
    n_445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 7), 'n')
    int_446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 12), 'int')
    # Applying the binary operator '<=' (line 80)
    result_le_447 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 7), '<=', n_445, int_446)
    
    # Testing if the type of an if condition is none (line 80)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 80, 4), result_le_447):
        pass
    else:
        
        # Testing the type of an if condition (line 80)
        if_condition_448 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 4), result_le_447)
        # Assigning a type to the variable 'if_condition_448' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'if_condition_448', if_condition_448)
        # SSA begins for if statement (line 80)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'list' (line 81)
        list_449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 81)
        
        # Assigning a type to the variable 'stypy_return_type' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'stypy_return_type', list_449)
        # SSA join for if statement (line 80)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Call to a Name (line 82):
    
    # Assigning a Call to a Name (line 82):
    
    # Call to range(...): (line 82)
    # Processing the call arguments (line 82)
    int_451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 18), 'int')
    # Getting the type of 'n' (line 82)
    n_452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 21), 'n', False)
    int_453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 24), 'int')
    # Processing the call keyword arguments (line 82)
    kwargs_454 = {}
    # Getting the type of 'range' (line 82)
    range_450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'range', False)
    # Calling range(args, kwargs) (line 82)
    range_call_result_455 = invoke(stypy.reporting.localization.Localization(__file__, 82, 12), range_450, *[int_451, n_452, int_453], **kwargs_454)
    
    # Assigning a type to the variable 'sieve' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'sieve', range_call_result_455)
    
    # Assigning a Call to a Name (line 83):
    
    # Assigning a Call to a Name (line 83):
    
    # Call to len(...): (line 83)
    # Processing the call arguments (line 83)
    # Getting the type of 'sieve' (line 83)
    sieve_457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 14), 'sieve', False)
    # Processing the call keyword arguments (line 83)
    kwargs_458 = {}
    # Getting the type of 'len' (line 83)
    len_456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 10), 'len', False)
    # Calling len(args, kwargs) (line 83)
    len_call_result_459 = invoke(stypy.reporting.localization.Localization(__file__, 83, 10), len_456, *[sieve_457], **kwargs_458)
    
    # Assigning a type to the variable 'top' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'top', len_call_result_459)
    
    # Getting the type of 'sieve' (line 84)
    sieve_460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 14), 'sieve')
    # Testing if the for loop is going to be iterated (line 84)
    # Testing the type of a for loop iterable (line 84)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 84, 4), sieve_460)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 84, 4), sieve_460):
        # Getting the type of the for loop variable (line 84)
        for_loop_var_461 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 84, 4), sieve_460)
        # Assigning a type to the variable 'si' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'si', for_loop_var_461)
        # SSA begins for a for statement (line 84)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        # Getting the type of 'si' (line 85)
        si_462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 11), 'si')
        # Testing if the type of an if condition is none (line 85)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 85, 8), si_462):
            pass
        else:
            
            # Testing the type of an if condition (line 85)
            if_condition_463 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 85, 8), si_462)
            # Assigning a type to the variable 'if_condition_463' (line 85)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'if_condition_463', if_condition_463)
            # SSA begins for if statement (line 85)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 86):
            
            # Assigning a BinOp to a Name (line 86):
            # Getting the type of 'si' (line 86)
            si_464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 22), 'si')
            # Getting the type of 'si' (line 86)
            si_465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 27), 'si')
            # Applying the binary operator '*' (line 86)
            result_mul_466 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 22), '*', si_464, si_465)
            
            int_467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 32), 'int')
            # Applying the binary operator '-' (line 86)
            result_sub_468 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 22), '-', result_mul_466, int_467)
            
            int_469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 38), 'int')
            # Applying the binary operator '//' (line 86)
            result_floordiv_470 = python_operator(stypy.reporting.localization.Localization(__file__, 86, 21), '//', result_sub_468, int_469)
            
            # Assigning a type to the variable 'bottom' (line 86)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'bottom', result_floordiv_470)
            
            # Getting the type of 'bottom' (line 87)
            bottom_471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 15), 'bottom')
            # Getting the type of 'top' (line 87)
            top_472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 25), 'top')
            # Applying the binary operator '>=' (line 87)
            result_ge_473 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 15), '>=', bottom_471, top_472)
            
            # Testing if the type of an if condition is none (line 87)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 87, 12), result_ge_473):
                pass
            else:
                
                # Testing the type of an if condition (line 87)
                if_condition_474 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 87, 12), result_ge_473)
                # Assigning a type to the variable 'if_condition_474' (line 87)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 12), 'if_condition_474', if_condition_474)
                # SSA begins for if statement (line 87)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # SSA join for if statement (line 87)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Assigning a BinOp to a Subscript (line 89):
            
            # Assigning a BinOp to a Subscript (line 89):
            
            # Obtaining an instance of the builtin type 'list' (line 89)
            list_475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 32), 'list')
            # Adding type elements to the builtin type 'list' instance (line 89)
            # Adding element type (line 89)
            int_476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 33), 'int')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 32), list_475, int_476)
            
            
            # Getting the type of 'bottom' (line 89)
            bottom_477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 41), 'bottom')
            # Getting the type of 'top' (line 89)
            top_478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 50), 'top')
            # Applying the binary operator '-' (line 89)
            result_sub_479 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 41), '-', bottom_477, top_478)
            
            # Getting the type of 'si' (line 89)
            si_480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 58), 'si')
            # Applying the binary operator '//' (line 89)
            result_floordiv_481 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 40), '//', result_sub_479, si_480)
            
            # Applying the 'usub' unary operator (line 89)
            result___neg___482 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 38), 'usub', result_floordiv_481)
            
            # Applying the binary operator '*' (line 89)
            result_mul_483 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 32), '*', list_475, result___neg___482)
            
            # Getting the type of 'sieve' (line 89)
            sieve_484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'sieve')
            # Getting the type of 'bottom' (line 89)
            bottom_485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 18), 'bottom')
            # Getting the type of 'si' (line 89)
            si_486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 26), 'si')
            slice_487 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 89, 12), bottom_485, None, si_486)
            # Storing an element on a container (line 89)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 12), sieve_484, (slice_487, result_mul_483))
            # SSA join for if statement (line 85)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Obtaining an instance of the builtin type 'list' (line 90)
    list_488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 90)
    # Adding element type (line 90)
    int_489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 12), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 11), list_488, int_489)
    
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'sieve' (line 90)
    sieve_492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 31), 'sieve')
    comprehension_493 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 18), sieve_492)
    # Assigning a type to the variable 'el' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 18), 'el', comprehension_493)
    # Getting the type of 'el' (line 90)
    el_491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 40), 'el')
    # Getting the type of 'el' (line 90)
    el_490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 18), 'el')
    list_494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 18), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 18), list_494, el_490)
    # Applying the binary operator '+' (line 90)
    result_add_495 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 11), '+', list_488, list_494)
    
    # Assigning a type to the variable 'stypy_return_type' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'stypy_return_type', result_add_495)
    
    # ################# End of 'sieveOfEratostenes(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'sieveOfEratostenes' in the type store
    # Getting the type of 'stypy_return_type' (line 76)
    stypy_return_type_496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_496)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'sieveOfEratostenes'
    return stypy_return_type_496

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
    int_497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 8), 'int')
    # Assigning a type to the variable 'n' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'n', int_497)
    
    # Assigning a Call to a Name (line 101):
    
    # Assigning a Call to a Name (line 101):
    
    # Call to sieveOfAtkin(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'n' (line 101)
    n_499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 21), 'n', False)
    # Processing the call keyword arguments (line 101)
    kwargs_500 = {}
    # Getting the type of 'sieveOfAtkin' (line 101)
    sieveOfAtkin_498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'sieveOfAtkin', False)
    # Calling sieveOfAtkin(args, kwargs) (line 101)
    sieveOfAtkin_call_result_501 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), sieveOfAtkin_498, *[n_499], **kwargs_500)
    
    # Assigning a type to the variable 'r' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'r', sieveOfAtkin_call_result_501)
    
    # Assigning a Call to a Name (line 105):
    
    # Assigning a Call to a Name (line 105):
    
    # Call to sieveOfEratostenes(...): (line 105)
    # Processing the call arguments (line 105)
    # Getting the type of 'n' (line 105)
    n_503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 27), 'n', False)
    # Processing the call keyword arguments (line 105)
    kwargs_504 = {}
    # Getting the type of 'sieveOfEratostenes' (line 105)
    sieveOfEratostenes_502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'sieveOfEratostenes', False)
    # Calling sieveOfEratostenes(args, kwargs) (line 105)
    sieveOfEratostenes_call_result_505 = invoke(stypy.reporting.localization.Localization(__file__, 105, 8), sieveOfEratostenes_502, *[n_503], **kwargs_504)
    
    # Assigning a type to the variable 'r' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'r', sieveOfEratostenes_call_result_505)
    # Getting the type of 'True' (line 111)
    True_506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'stypy_return_type', True_506)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 93)
    stypy_return_type_507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_507)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_507

# Assigning a type to the variable 'run' (line 93)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 0), 'run', run)

# Call to run(...): (line 114)
# Processing the call keyword arguments (line 114)
kwargs_509 = {}
# Getting the type of 'run' (line 114)
run_508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 0), 'run', False)
# Calling run(args, kwargs) (line 114)
run_call_result_510 = invoke(stypy.reporting.localization.Localization(__file__, 114, 0), run_508, *[], **kwargs_509)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
