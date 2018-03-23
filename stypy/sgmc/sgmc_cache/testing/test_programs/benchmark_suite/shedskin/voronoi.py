
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # Textual Voronoi code modified from: <abhishek@ocf.berkeley.edu>
2: # http://www.ocf.berkeley.edu/~Eabhishek/
3: 
4: from random import random  # for generateRandomPoints
5: from math import sqrt
6: 
7: 
8: def generateRandomPoints(npoints=6):
9:     '''Generate a few random points v1...vn'''
10:     ##    print npoints, "points x,y:"
11:     points = []
12:     for i in xrange(npoints):
13:         xrand, yrand = random(), random()
14:         ##        print xrand, yrand
15:         for xoff in range(-1, 2):
16:             for yoff in range(-1, 2):
17:                 points.append((xrand + xoff, yrand + yoff))
18:     return points
19: 
20: 
21: def closest(x, y, points):
22:     '''Function to find the closest of the vi.'''
23:     best, good = 99.0 * 99.0, 99.0 * 99.0
24:     for px, py in points:
25:         dist = (x - px) * (x - px) + (y - py) * (y - py)
26:         if dist < best:
27:             best, good = dist, best
28:         elif dist < good:
29:             good = dist
30:     return sqrt(best) / sqrt(good)
31: 
32: 
33: def generateScreen(points, rows=40, cols=80):
34:     yfact = 1.0 / cols
35:     xfact = 1.0 / rows
36:     screen = []
37:     chars = " -.,+*$&#~~"
38:     for i in xrange(rows):
39:         x = i * xfact
40:         line = [chars[int(10 * closest(x, j * yfact, points))] for j in xrange(cols)]
41:         screen.extend(line)
42:         screen.append("\n")
43:     return "".join(screen)
44: 
45: 
46: from time import clock
47: 
48: 
49: def run():
50:     points = generateRandomPoints(10)
51:     ##    print
52:     t1 = clock()
53:     ##    print generateScreen(points, 40, 80)
54:     generateScreen(points, 40, 80)
55:     t2 = clock()
56:     ##    print round(t2-t1, 3)
57:     return True
58: 
59: 
60: run()
61: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from random import random' statement (line 4)
try:
    from random import random

except:
    random = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'random', None, module_type_store, ['random'], [random])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from math import sqrt' statement (line 5)
try:
    from math import sqrt

except:
    sqrt = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'math', None, module_type_store, ['sqrt'], [sqrt])


@norecursion
def generateRandomPoints(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 33), 'int')
    defaults = [int_7]
    # Create a new context for function 'generateRandomPoints'
    module_type_store = module_type_store.open_function_context('generateRandomPoints', 8, 0, False)
    
    # Passed parameters checking function
    generateRandomPoints.stypy_localization = localization
    generateRandomPoints.stypy_type_of_self = None
    generateRandomPoints.stypy_type_store = module_type_store
    generateRandomPoints.stypy_function_name = 'generateRandomPoints'
    generateRandomPoints.stypy_param_names_list = ['npoints']
    generateRandomPoints.stypy_varargs_param_name = None
    generateRandomPoints.stypy_kwargs_param_name = None
    generateRandomPoints.stypy_call_defaults = defaults
    generateRandomPoints.stypy_call_varargs = varargs
    generateRandomPoints.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'generateRandomPoints', ['npoints'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'generateRandomPoints', localization, ['npoints'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'generateRandomPoints(...)' code ##################

    str_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 4), 'str', 'Generate a few random points v1...vn')
    
    # Assigning a List to a Name (line 11):
    
    # Assigning a List to a Name (line 11):
    
    # Obtaining an instance of the builtin type 'list' (line 11)
    list_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 11)
    
    # Assigning a type to the variable 'points' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'points', list_9)
    
    
    # Call to xrange(...): (line 12)
    # Processing the call arguments (line 12)
    # Getting the type of 'npoints' (line 12)
    npoints_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 20), 'npoints', False)
    # Processing the call keyword arguments (line 12)
    kwargs_12 = {}
    # Getting the type of 'xrange' (line 12)
    xrange_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 12)
    xrange_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 12, 13), xrange_10, *[npoints_11], **kwargs_12)
    
    # Assigning a type to the variable 'xrange_call_result_13' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'xrange_call_result_13', xrange_call_result_13)
    # Testing if the for loop is going to be iterated (line 12)
    # Testing the type of a for loop iterable (line 12)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 12, 4), xrange_call_result_13)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 12, 4), xrange_call_result_13):
        # Getting the type of the for loop variable (line 12)
        for_loop_var_14 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 12, 4), xrange_call_result_13)
        # Assigning a type to the variable 'i' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'i', for_loop_var_14)
        # SSA begins for a for statement (line 12)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Tuple to a Tuple (line 13):
        
        # Assigning a Call to a Name (line 13):
        
        # Call to random(...): (line 13)
        # Processing the call keyword arguments (line 13)
        kwargs_16 = {}
        # Getting the type of 'random' (line 13)
        random_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 23), 'random', False)
        # Calling random(args, kwargs) (line 13)
        random_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 13, 23), random_15, *[], **kwargs_16)
        
        # Assigning a type to the variable 'tuple_assignment_1' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'tuple_assignment_1', random_call_result_17)
        
        # Assigning a Call to a Name (line 13):
        
        # Call to random(...): (line 13)
        # Processing the call keyword arguments (line 13)
        kwargs_19 = {}
        # Getting the type of 'random' (line 13)
        random_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 33), 'random', False)
        # Calling random(args, kwargs) (line 13)
        random_call_result_20 = invoke(stypy.reporting.localization.Localization(__file__, 13, 33), random_18, *[], **kwargs_19)
        
        # Assigning a type to the variable 'tuple_assignment_2' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'tuple_assignment_2', random_call_result_20)
        
        # Assigning a Name to a Name (line 13):
        # Getting the type of 'tuple_assignment_1' (line 13)
        tuple_assignment_1_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'tuple_assignment_1')
        # Assigning a type to the variable 'xrand' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'xrand', tuple_assignment_1_21)
        
        # Assigning a Name to a Name (line 13):
        # Getting the type of 'tuple_assignment_2' (line 13)
        tuple_assignment_2_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'tuple_assignment_2')
        # Assigning a type to the variable 'yrand' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 15), 'yrand', tuple_assignment_2_22)
        
        
        # Call to range(...): (line 15)
        # Processing the call arguments (line 15)
        int_24 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 26), 'int')
        int_25 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 30), 'int')
        # Processing the call keyword arguments (line 15)
        kwargs_26 = {}
        # Getting the type of 'range' (line 15)
        range_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 20), 'range', False)
        # Calling range(args, kwargs) (line 15)
        range_call_result_27 = invoke(stypy.reporting.localization.Localization(__file__, 15, 20), range_23, *[int_24, int_25], **kwargs_26)
        
        # Assigning a type to the variable 'range_call_result_27' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'range_call_result_27', range_call_result_27)
        # Testing if the for loop is going to be iterated (line 15)
        # Testing the type of a for loop iterable (line 15)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 15, 8), range_call_result_27)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 15, 8), range_call_result_27):
            # Getting the type of the for loop variable (line 15)
            for_loop_var_28 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 15, 8), range_call_result_27)
            # Assigning a type to the variable 'xoff' (line 15)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'xoff', for_loop_var_28)
            # SSA begins for a for statement (line 15)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Call to range(...): (line 16)
            # Processing the call arguments (line 16)
            int_30 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 30), 'int')
            int_31 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 34), 'int')
            # Processing the call keyword arguments (line 16)
            kwargs_32 = {}
            # Getting the type of 'range' (line 16)
            range_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 24), 'range', False)
            # Calling range(args, kwargs) (line 16)
            range_call_result_33 = invoke(stypy.reporting.localization.Localization(__file__, 16, 24), range_29, *[int_30, int_31], **kwargs_32)
            
            # Assigning a type to the variable 'range_call_result_33' (line 16)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 12), 'range_call_result_33', range_call_result_33)
            # Testing if the for loop is going to be iterated (line 16)
            # Testing the type of a for loop iterable (line 16)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 16, 12), range_call_result_33)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 16, 12), range_call_result_33):
                # Getting the type of the for loop variable (line 16)
                for_loop_var_34 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 16, 12), range_call_result_33)
                # Assigning a type to the variable 'yoff' (line 16)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 12), 'yoff', for_loop_var_34)
                # SSA begins for a for statement (line 16)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Call to append(...): (line 17)
                # Processing the call arguments (line 17)
                
                # Obtaining an instance of the builtin type 'tuple' (line 17)
                tuple_37 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 31), 'tuple')
                # Adding type elements to the builtin type 'tuple' instance (line 17)
                # Adding element type (line 17)
                # Getting the type of 'xrand' (line 17)
                xrand_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 31), 'xrand', False)
                # Getting the type of 'xoff' (line 17)
                xoff_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 39), 'xoff', False)
                # Applying the binary operator '+' (line 17)
                result_add_40 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 31), '+', xrand_38, xoff_39)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 31), tuple_37, result_add_40)
                # Adding element type (line 17)
                # Getting the type of 'yrand' (line 17)
                yrand_41 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 45), 'yrand', False)
                # Getting the type of 'yoff' (line 17)
                yoff_42 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 53), 'yoff', False)
                # Applying the binary operator '+' (line 17)
                result_add_43 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 45), '+', yrand_41, yoff_42)
                
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 31), tuple_37, result_add_43)
                
                # Processing the call keyword arguments (line 17)
                kwargs_44 = {}
                # Getting the type of 'points' (line 17)
                points_35 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 16), 'points', False)
                # Obtaining the member 'append' of a type (line 17)
                append_36 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 16), points_35, 'append')
                # Calling append(args, kwargs) (line 17)
                append_call_result_45 = invoke(stypy.reporting.localization.Localization(__file__, 17, 16), append_36, *[tuple_37], **kwargs_44)
                
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'points' (line 18)
    points_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 11), 'points')
    # Assigning a type to the variable 'stypy_return_type' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'stypy_return_type', points_46)
    
    # ################# End of 'generateRandomPoints(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'generateRandomPoints' in the type store
    # Getting the type of 'stypy_return_type' (line 8)
    stypy_return_type_47 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_47)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'generateRandomPoints'
    return stypy_return_type_47

# Assigning a type to the variable 'generateRandomPoints' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'generateRandomPoints', generateRandomPoints)

@norecursion
def closest(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'closest'
    module_type_store = module_type_store.open_function_context('closest', 21, 0, False)
    
    # Passed parameters checking function
    closest.stypy_localization = localization
    closest.stypy_type_of_self = None
    closest.stypy_type_store = module_type_store
    closest.stypy_function_name = 'closest'
    closest.stypy_param_names_list = ['x', 'y', 'points']
    closest.stypy_varargs_param_name = None
    closest.stypy_kwargs_param_name = None
    closest.stypy_call_defaults = defaults
    closest.stypy_call_varargs = varargs
    closest.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'closest', ['x', 'y', 'points'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'closest', localization, ['x', 'y', 'points'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'closest(...)' code ##################

    str_48 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 4), 'str', 'Function to find the closest of the vi.')
    
    # Assigning a Tuple to a Tuple (line 23):
    
    # Assigning a BinOp to a Name (line 23):
    float_49 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 17), 'float')
    float_50 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 24), 'float')
    # Applying the binary operator '*' (line 23)
    result_mul_51 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 17), '*', float_49, float_50)
    
    # Assigning a type to the variable 'tuple_assignment_3' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'tuple_assignment_3', result_mul_51)
    
    # Assigning a BinOp to a Name (line 23):
    float_52 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 30), 'float')
    float_53 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 37), 'float')
    # Applying the binary operator '*' (line 23)
    result_mul_54 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 30), '*', float_52, float_53)
    
    # Assigning a type to the variable 'tuple_assignment_4' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'tuple_assignment_4', result_mul_54)
    
    # Assigning a Name to a Name (line 23):
    # Getting the type of 'tuple_assignment_3' (line 23)
    tuple_assignment_3_55 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'tuple_assignment_3')
    # Assigning a type to the variable 'best' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'best', tuple_assignment_3_55)
    
    # Assigning a Name to a Name (line 23):
    # Getting the type of 'tuple_assignment_4' (line 23)
    tuple_assignment_4_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'tuple_assignment_4')
    # Assigning a type to the variable 'good' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 10), 'good', tuple_assignment_4_56)
    
    # Getting the type of 'points' (line 24)
    points_57 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 18), 'points')
    # Assigning a type to the variable 'points_57' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'points_57', points_57)
    # Testing if the for loop is going to be iterated (line 24)
    # Testing the type of a for loop iterable (line 24)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 24, 4), points_57)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 24, 4), points_57):
        # Getting the type of the for loop variable (line 24)
        for_loop_var_58 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 24, 4), points_57)
        # Assigning a type to the variable 'px' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'px', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 4), for_loop_var_58, 2, 0))
        # Assigning a type to the variable 'py' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'py', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 4), for_loop_var_58, 2, 1))
        # SSA begins for a for statement (line 24)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 25):
        
        # Assigning a BinOp to a Name (line 25):
        # Getting the type of 'x' (line 25)
        x_59 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 16), 'x')
        # Getting the type of 'px' (line 25)
        px_60 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 20), 'px')
        # Applying the binary operator '-' (line 25)
        result_sub_61 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 16), '-', x_59, px_60)
        
        # Getting the type of 'x' (line 25)
        x_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 27), 'x')
        # Getting the type of 'px' (line 25)
        px_63 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 31), 'px')
        # Applying the binary operator '-' (line 25)
        result_sub_64 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 27), '-', x_62, px_63)
        
        # Applying the binary operator '*' (line 25)
        result_mul_65 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 15), '*', result_sub_61, result_sub_64)
        
        # Getting the type of 'y' (line 25)
        y_66 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 38), 'y')
        # Getting the type of 'py' (line 25)
        py_67 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 42), 'py')
        # Applying the binary operator '-' (line 25)
        result_sub_68 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 38), '-', y_66, py_67)
        
        # Getting the type of 'y' (line 25)
        y_69 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 49), 'y')
        # Getting the type of 'py' (line 25)
        py_70 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 53), 'py')
        # Applying the binary operator '-' (line 25)
        result_sub_71 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 49), '-', y_69, py_70)
        
        # Applying the binary operator '*' (line 25)
        result_mul_72 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 37), '*', result_sub_68, result_sub_71)
        
        # Applying the binary operator '+' (line 25)
        result_add_73 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 15), '+', result_mul_65, result_mul_72)
        
        # Assigning a type to the variable 'dist' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'dist', result_add_73)
        
        # Getting the type of 'dist' (line 26)
        dist_74 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 11), 'dist')
        # Getting the type of 'best' (line 26)
        best_75 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 18), 'best')
        # Applying the binary operator '<' (line 26)
        result_lt_76 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 11), '<', dist_74, best_75)
        
        # Testing if the type of an if condition is none (line 26)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 26, 8), result_lt_76):
            
            # Getting the type of 'dist' (line 28)
            dist_82 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 13), 'dist')
            # Getting the type of 'good' (line 28)
            good_83 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 20), 'good')
            # Applying the binary operator '<' (line 28)
            result_lt_84 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 13), '<', dist_82, good_83)
            
            # Testing if the type of an if condition is none (line 28)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 28, 13), result_lt_84):
                pass
            else:
                
                # Testing the type of an if condition (line 28)
                if_condition_85 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 28, 13), result_lt_84)
                # Assigning a type to the variable 'if_condition_85' (line 28)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 13), 'if_condition_85', if_condition_85)
                # SSA begins for if statement (line 28)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 29):
                
                # Assigning a Name to a Name (line 29):
                # Getting the type of 'dist' (line 29)
                dist_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 19), 'dist')
                # Assigning a type to the variable 'good' (line 29)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'good', dist_86)
                # SSA join for if statement (line 28)
                module_type_store = module_type_store.join_ssa_context()
                

        else:
            
            # Testing the type of an if condition (line 26)
            if_condition_77 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 26, 8), result_lt_76)
            # Assigning a type to the variable 'if_condition_77' (line 26)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'if_condition_77', if_condition_77)
            # SSA begins for if statement (line 26)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Tuple to a Tuple (line 27):
            
            # Assigning a Name to a Name (line 27):
            # Getting the type of 'dist' (line 27)
            dist_78 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 25), 'dist')
            # Assigning a type to the variable 'tuple_assignment_5' (line 27)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'tuple_assignment_5', dist_78)
            
            # Assigning a Name to a Name (line 27):
            # Getting the type of 'best' (line 27)
            best_79 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 31), 'best')
            # Assigning a type to the variable 'tuple_assignment_6' (line 27)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'tuple_assignment_6', best_79)
            
            # Assigning a Name to a Name (line 27):
            # Getting the type of 'tuple_assignment_5' (line 27)
            tuple_assignment_5_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'tuple_assignment_5')
            # Assigning a type to the variable 'best' (line 27)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'best', tuple_assignment_5_80)
            
            # Assigning a Name to a Name (line 27):
            # Getting the type of 'tuple_assignment_6' (line 27)
            tuple_assignment_6_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'tuple_assignment_6')
            # Assigning a type to the variable 'good' (line 27)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 18), 'good', tuple_assignment_6_81)
            # SSA branch for the else part of an if statement (line 26)
            module_type_store.open_ssa_branch('else')
            
            # Getting the type of 'dist' (line 28)
            dist_82 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 13), 'dist')
            # Getting the type of 'good' (line 28)
            good_83 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 20), 'good')
            # Applying the binary operator '<' (line 28)
            result_lt_84 = python_operator(stypy.reporting.localization.Localization(__file__, 28, 13), '<', dist_82, good_83)
            
            # Testing if the type of an if condition is none (line 28)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 28, 13), result_lt_84):
                pass
            else:
                
                # Testing the type of an if condition (line 28)
                if_condition_85 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 28, 13), result_lt_84)
                # Assigning a type to the variable 'if_condition_85' (line 28)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 13), 'if_condition_85', if_condition_85)
                # SSA begins for if statement (line 28)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Name (line 29):
                
                # Assigning a Name to a Name (line 29):
                # Getting the type of 'dist' (line 29)
                dist_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 19), 'dist')
                # Assigning a type to the variable 'good' (line 29)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'good', dist_86)
                # SSA join for if statement (line 28)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 26)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Call to sqrt(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'best' (line 30)
    best_88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 16), 'best', False)
    # Processing the call keyword arguments (line 30)
    kwargs_89 = {}
    # Getting the type of 'sqrt' (line 30)
    sqrt_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 11), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 30)
    sqrt_call_result_90 = invoke(stypy.reporting.localization.Localization(__file__, 30, 11), sqrt_87, *[best_88], **kwargs_89)
    
    
    # Call to sqrt(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'good' (line 30)
    good_92 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 29), 'good', False)
    # Processing the call keyword arguments (line 30)
    kwargs_93 = {}
    # Getting the type of 'sqrt' (line 30)
    sqrt_91 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 24), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 30)
    sqrt_call_result_94 = invoke(stypy.reporting.localization.Localization(__file__, 30, 24), sqrt_91, *[good_92], **kwargs_93)
    
    # Applying the binary operator 'div' (line 30)
    result_div_95 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 11), 'div', sqrt_call_result_90, sqrt_call_result_94)
    
    # Assigning a type to the variable 'stypy_return_type' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'stypy_return_type', result_div_95)
    
    # ################# End of 'closest(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'closest' in the type store
    # Getting the type of 'stypy_return_type' (line 21)
    stypy_return_type_96 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_96)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'closest'
    return stypy_return_type_96

# Assigning a type to the variable 'closest' (line 21)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'closest', closest)

@norecursion
def generateScreen(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_97 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 32), 'int')
    int_98 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 41), 'int')
    defaults = [int_97, int_98]
    # Create a new context for function 'generateScreen'
    module_type_store = module_type_store.open_function_context('generateScreen', 33, 0, False)
    
    # Passed parameters checking function
    generateScreen.stypy_localization = localization
    generateScreen.stypy_type_of_self = None
    generateScreen.stypy_type_store = module_type_store
    generateScreen.stypy_function_name = 'generateScreen'
    generateScreen.stypy_param_names_list = ['points', 'rows', 'cols']
    generateScreen.stypy_varargs_param_name = None
    generateScreen.stypy_kwargs_param_name = None
    generateScreen.stypy_call_defaults = defaults
    generateScreen.stypy_call_varargs = varargs
    generateScreen.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'generateScreen', ['points', 'rows', 'cols'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'generateScreen', localization, ['points', 'rows', 'cols'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'generateScreen(...)' code ##################

    
    # Assigning a BinOp to a Name (line 34):
    
    # Assigning a BinOp to a Name (line 34):
    float_99 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 12), 'float')
    # Getting the type of 'cols' (line 34)
    cols_100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 18), 'cols')
    # Applying the binary operator 'div' (line 34)
    result_div_101 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 12), 'div', float_99, cols_100)
    
    # Assigning a type to the variable 'yfact' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'yfact', result_div_101)
    
    # Assigning a BinOp to a Name (line 35):
    
    # Assigning a BinOp to a Name (line 35):
    float_102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 12), 'float')
    # Getting the type of 'rows' (line 35)
    rows_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 18), 'rows')
    # Applying the binary operator 'div' (line 35)
    result_div_104 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 12), 'div', float_102, rows_103)
    
    # Assigning a type to the variable 'xfact' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'xfact', result_div_104)
    
    # Assigning a List to a Name (line 36):
    
    # Assigning a List to a Name (line 36):
    
    # Obtaining an instance of the builtin type 'list' (line 36)
    list_105 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 36)
    
    # Assigning a type to the variable 'screen' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'screen', list_105)
    
    # Assigning a Str to a Name (line 37):
    
    # Assigning a Str to a Name (line 37):
    str_106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 12), 'str', ' -.,+*$&#~~')
    # Assigning a type to the variable 'chars' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'chars', str_106)
    
    
    # Call to xrange(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'rows' (line 38)
    rows_108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 20), 'rows', False)
    # Processing the call keyword arguments (line 38)
    kwargs_109 = {}
    # Getting the type of 'xrange' (line 38)
    xrange_107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 38)
    xrange_call_result_110 = invoke(stypy.reporting.localization.Localization(__file__, 38, 13), xrange_107, *[rows_108], **kwargs_109)
    
    # Assigning a type to the variable 'xrange_call_result_110' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'xrange_call_result_110', xrange_call_result_110)
    # Testing if the for loop is going to be iterated (line 38)
    # Testing the type of a for loop iterable (line 38)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 38, 4), xrange_call_result_110)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 38, 4), xrange_call_result_110):
        # Getting the type of the for loop variable (line 38)
        for_loop_var_111 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 38, 4), xrange_call_result_110)
        # Assigning a type to the variable 'i' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'i', for_loop_var_111)
        # SSA begins for a for statement (line 38)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Name (line 39):
        
        # Assigning a BinOp to a Name (line 39):
        # Getting the type of 'i' (line 39)
        i_112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'i')
        # Getting the type of 'xfact' (line 39)
        xfact_113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 16), 'xfact')
        # Applying the binary operator '*' (line 39)
        result_mul_114 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 12), '*', i_112, xfact_113)
        
        # Assigning a type to the variable 'x' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'x', result_mul_114)
        
        # Assigning a ListComp to a Name (line 40):
        
        # Assigning a ListComp to a Name (line 40):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to xrange(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'cols' (line 40)
        cols_132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 79), 'cols', False)
        # Processing the call keyword arguments (line 40)
        kwargs_133 = {}
        # Getting the type of 'xrange' (line 40)
        xrange_131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 72), 'xrange', False)
        # Calling xrange(args, kwargs) (line 40)
        xrange_call_result_134 = invoke(stypy.reporting.localization.Localization(__file__, 40, 72), xrange_131, *[cols_132], **kwargs_133)
        
        comprehension_135 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 16), xrange_call_result_134)
        # Assigning a type to the variable 'j' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 16), 'j', comprehension_135)
        
        # Obtaining the type of the subscript
        
        # Call to int(...): (line 40)
        # Processing the call arguments (line 40)
        int_116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 26), 'int')
        
        # Call to closest(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'x' (line 40)
        x_118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 39), 'x', False)
        # Getting the type of 'j' (line 40)
        j_119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 42), 'j', False)
        # Getting the type of 'yfact' (line 40)
        yfact_120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 46), 'yfact', False)
        # Applying the binary operator '*' (line 40)
        result_mul_121 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 42), '*', j_119, yfact_120)
        
        # Getting the type of 'points' (line 40)
        points_122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 53), 'points', False)
        # Processing the call keyword arguments (line 40)
        kwargs_123 = {}
        # Getting the type of 'closest' (line 40)
        closest_117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 31), 'closest', False)
        # Calling closest(args, kwargs) (line 40)
        closest_call_result_124 = invoke(stypy.reporting.localization.Localization(__file__, 40, 31), closest_117, *[x_118, result_mul_121, points_122], **kwargs_123)
        
        # Applying the binary operator '*' (line 40)
        result_mul_125 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 26), '*', int_116, closest_call_result_124)
        
        # Processing the call keyword arguments (line 40)
        kwargs_126 = {}
        # Getting the type of 'int' (line 40)
        int_115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 22), 'int', False)
        # Calling int(args, kwargs) (line 40)
        int_call_result_127 = invoke(stypy.reporting.localization.Localization(__file__, 40, 22), int_115, *[result_mul_125], **kwargs_126)
        
        # Getting the type of 'chars' (line 40)
        chars_128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 16), 'chars')
        # Obtaining the member '__getitem__' of a type (line 40)
        getitem___129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 16), chars_128, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 40)
        subscript_call_result_130 = invoke(stypy.reporting.localization.Localization(__file__, 40, 16), getitem___129, int_call_result_127)
        
        list_136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 16), list_136, subscript_call_result_130)
        # Assigning a type to the variable 'line' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'line', list_136)
        
        # Call to extend(...): (line 41)
        # Processing the call arguments (line 41)
        # Getting the type of 'line' (line 41)
        line_139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 22), 'line', False)
        # Processing the call keyword arguments (line 41)
        kwargs_140 = {}
        # Getting the type of 'screen' (line 41)
        screen_137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'screen', False)
        # Obtaining the member 'extend' of a type (line 41)
        extend_138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), screen_137, 'extend')
        # Calling extend(args, kwargs) (line 41)
        extend_call_result_141 = invoke(stypy.reporting.localization.Localization(__file__, 41, 8), extend_138, *[line_139], **kwargs_140)
        
        
        # Call to append(...): (line 42)
        # Processing the call arguments (line 42)
        str_144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 22), 'str', '\n')
        # Processing the call keyword arguments (line 42)
        kwargs_145 = {}
        # Getting the type of 'screen' (line 42)
        screen_142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'screen', False)
        # Obtaining the member 'append' of a type (line 42)
        append_143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), screen_142, 'append')
        # Calling append(args, kwargs) (line 42)
        append_call_result_146 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), append_143, *[str_144], **kwargs_145)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Call to join(...): (line 43)
    # Processing the call arguments (line 43)
    # Getting the type of 'screen' (line 43)
    screen_149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 19), 'screen', False)
    # Processing the call keyword arguments (line 43)
    kwargs_150 = {}
    str_147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 11), 'str', '')
    # Obtaining the member 'join' of a type (line 43)
    join_148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 11), str_147, 'join')
    # Calling join(args, kwargs) (line 43)
    join_call_result_151 = invoke(stypy.reporting.localization.Localization(__file__, 43, 11), join_148, *[screen_149], **kwargs_150)
    
    # Assigning a type to the variable 'stypy_return_type' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'stypy_return_type', join_call_result_151)
    
    # ################# End of 'generateScreen(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'generateScreen' in the type store
    # Getting the type of 'stypy_return_type' (line 33)
    stypy_return_type_152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_152)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'generateScreen'
    return stypy_return_type_152

# Assigning a type to the variable 'generateScreen' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'generateScreen', generateScreen)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 46, 0))

# 'from time import clock' statement (line 46)
try:
    from time import clock

except:
    clock = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 46, 0), 'time', None, module_type_store, ['clock'], [clock])


@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 49, 0, False)
    
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

    
    # Assigning a Call to a Name (line 50):
    
    # Assigning a Call to a Name (line 50):
    
    # Call to generateRandomPoints(...): (line 50)
    # Processing the call arguments (line 50)
    int_154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 34), 'int')
    # Processing the call keyword arguments (line 50)
    kwargs_155 = {}
    # Getting the type of 'generateRandomPoints' (line 50)
    generateRandomPoints_153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 13), 'generateRandomPoints', False)
    # Calling generateRandomPoints(args, kwargs) (line 50)
    generateRandomPoints_call_result_156 = invoke(stypy.reporting.localization.Localization(__file__, 50, 13), generateRandomPoints_153, *[int_154], **kwargs_155)
    
    # Assigning a type to the variable 'points' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'points', generateRandomPoints_call_result_156)
    
    # Assigning a Call to a Name (line 52):
    
    # Assigning a Call to a Name (line 52):
    
    # Call to clock(...): (line 52)
    # Processing the call keyword arguments (line 52)
    kwargs_158 = {}
    # Getting the type of 'clock' (line 52)
    clock_157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 9), 'clock', False)
    # Calling clock(args, kwargs) (line 52)
    clock_call_result_159 = invoke(stypy.reporting.localization.Localization(__file__, 52, 9), clock_157, *[], **kwargs_158)
    
    # Assigning a type to the variable 't1' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 't1', clock_call_result_159)
    
    # Call to generateScreen(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 'points' (line 54)
    points_161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 19), 'points', False)
    int_162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 27), 'int')
    int_163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 31), 'int')
    # Processing the call keyword arguments (line 54)
    kwargs_164 = {}
    # Getting the type of 'generateScreen' (line 54)
    generateScreen_160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'generateScreen', False)
    # Calling generateScreen(args, kwargs) (line 54)
    generateScreen_call_result_165 = invoke(stypy.reporting.localization.Localization(__file__, 54, 4), generateScreen_160, *[points_161, int_162, int_163], **kwargs_164)
    
    
    # Assigning a Call to a Name (line 55):
    
    # Assigning a Call to a Name (line 55):
    
    # Call to clock(...): (line 55)
    # Processing the call keyword arguments (line 55)
    kwargs_167 = {}
    # Getting the type of 'clock' (line 55)
    clock_166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 9), 'clock', False)
    # Calling clock(args, kwargs) (line 55)
    clock_call_result_168 = invoke(stypy.reporting.localization.Localization(__file__, 55, 9), clock_166, *[], **kwargs_167)
    
    # Assigning a type to the variable 't2' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 't2', clock_call_result_168)
    # Getting the type of 'True' (line 57)
    True_169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'stypy_return_type', True_169)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 49)
    stypy_return_type_170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_170)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_170

# Assigning a type to the variable 'run' (line 49)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'run', run)

# Call to run(...): (line 60)
# Processing the call keyword arguments (line 60)
kwargs_172 = {}
# Getting the type of 'run' (line 60)
run_171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'run', False)
# Calling run(args, kwargs) (line 60)
run_call_result_173 = invoke(stypy.reporting.localization.Localization(__file__, 60, 0), run_171, *[], **kwargs_172)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
