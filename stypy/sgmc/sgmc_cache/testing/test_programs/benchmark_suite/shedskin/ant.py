
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #!/usr/bin/python
2: 
3: # ant.py
4: # Eric Rollins 2008
5: 
6: #   This program generates a random array of distances between cities, then uses
7: #   Ant Colony Optimization to find a short path traversing all the cities --
8: #   the Travelling Salesman Problem.
9: #
10: #   In this version of Ant Colony Optimization each ant starts in a random city.
11: #   Paths are randomly chosed with probability inversely proportional to to the
12: #   distance to the next city.  At the end of its travel the ant updates the
13: #   pheromone matrix with its path if this path is the shortest one yet found.
14: #   The probability of later ants taking a path is increased by the pheromone
15: #   value on that path.  Pheromone values evaporate (decrease) over time.
16: #
17: #   In this impementation weights between cities actually represent
18: #   (maxDistance - dist), so we are trying to maximize the score.
19: #
20: #   Usage: ant seed boost iterations cities
21: #     seed         seed for random number generator (1,2,3...).
22: #                  This seed controls the city distance array.  Remote
23: #                  executions have their seed values fixed (1,2) so each will
24: #                  produce a different result.
25: #     boost        pheromone boost for best path.  5 appears good.
26: #                  0 disables pheromones, providing random search.
27: #     iterations   number of ants to be run.
28: #     cities       number of cities.
29: 
30: import random
31: 
32: # type Matrix = Array[Array[double]]
33: # type Path = List[int]
34: # type CitySet = HashSet[int]
35: 
36: # int * int * int -> Matrix
37: def randomMatrix(n, upperBound, seed):
38:     random.seed(seed)
39:     m = []
40:     for r in range(n):
41:         sm = []
42:         m.append(sm)
43:         for c in range(n):
44:              sm.append(upperBound * random.random())
45:     return m
46: 
47: # Path -> Path
48: def wrappedPath(path):
49:     return path[1:] + [path[0]]
50: 
51: # Matrix * Path -> double
52: def pathLength(cities, path):
53:     pairs = zip(path, wrappedPath(path))
54:     return sum([cities[r][c] for (r,c) in pairs])
55: 
56: # Boosts pheromones for cities on path.
57: # Matrix * Path * int -> unit
58: def updatePher(pher, path, boost):
59:     pairs = zip(path, wrappedPath(path))
60:     for (r,c) in pairs:
61:         pher[r][c] = pher[r][c] + boost
62: 
63: # Matrix * int * int -> unit
64: def evaporatePher(pher, maxIter, boost):
65:     decr = boost / float(maxIter)
66:     for r in range(len(pher)):
67:         for c in range(len(pher[r])):
68:             if pher[r][c] > decr:
69:                 pher[r][c] = pher[r][c] - decr
70:             else:
71:                 pher[r][c] = 0.0
72: 
73: # Sum weights for all paths to cities adjacent to current.
74: # Matrix * Matrix * CitySet * int -> double
75: def doSumWeight(cities, pher, used, current):
76:     runningTotal = 0.0
77:     for city in range(len(cities)):
78:         if not used.has_key(city):
79:             runningTotal = (runningTotal +
80:                             cities[current][city] * (1.0 + pher[current][city]))
81:     return runningTotal
82: 
83: # Returns city at soughtTotal.
84: # Matrix * Matrix * CitySet * int * double -> int
85: def findSumWeight(cities, pher, used, current, soughtTotal):
86:     runningTotal = 0.0
87:     next = 0
88:     for city in range(len(cities)):
89:         if runningTotal >= soughtTotal:
90:             break
91:         if not used.has_key(city):
92:             runningTotal = (runningTotal +
93:                             cities[current][city] * (1.0 + pher[current][city]))
94:             next = city
95:     return next
96: 
97: # Matrix * Matrix -> Path
98: def genPath(cities, pher):
99:     current = random.randint(0, len(cities)-1)
100:     path = [current]
101:     used = {current:1}
102:     while len(used) < len(cities):
103:         sumWeight = doSumWeight(cities, pher, used, current)
104:         rndValue = random.random() * sumWeight
105:         current = findSumWeight(cities, pher, used, current, rndValue)
106:         path.append(current)
107:         used[current] = 1
108:     return path
109: 
110: # Matrix * int * int * int ->Path
111: def bestPath(cities, seed, maxIter, boost):
112:     pher = randomMatrix(len(cities), 0, 0)
113:     random.seed(seed)
114:     bestLen = 0.0
115:     bestPath = []
116:     for iter in range(maxIter):
117:         path = genPath(cities, pher)
118:         pathLen = pathLength(cities, path)
119:         if pathLen > bestLen:
120:             # Remember we are trying to maximize score.
121:             updatePher(pher, path, boost)
122:             bestLen = pathLen
123:             bestPath = path
124:         evaporatePher(pher, maxIter, boost)
125:     return bestPath
126: 
127: def main():
128:     seed = 1
129:     boost = 5
130:     iter = 1000
131:     numCities = 200
132:     maxDistance = 100
133:     cityDistanceSeed = 1
134:     #print "starting"
135:     cities = randomMatrix(numCities, maxDistance, cityDistanceSeed)
136:     path = bestPath(cities, seed, iter, boost)
137:     #print path
138:     #print "len = ", pathLength(cities, path)
139:     pathLength(cities, path)
140: 
141: def run():
142:     main()
143:     return True
144: 
145: run()
146: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 30, 0))

# 'import random' statement (line 30)
import random

import_module(stypy.reporting.localization.Localization(__file__, 30, 0), 'random', random, module_type_store)


@norecursion
def randomMatrix(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'randomMatrix'
    module_type_store = module_type_store.open_function_context('randomMatrix', 37, 0, False)
    
    # Passed parameters checking function
    randomMatrix.stypy_localization = localization
    randomMatrix.stypy_type_of_self = None
    randomMatrix.stypy_type_store = module_type_store
    randomMatrix.stypy_function_name = 'randomMatrix'
    randomMatrix.stypy_param_names_list = ['n', 'upperBound', 'seed']
    randomMatrix.stypy_varargs_param_name = None
    randomMatrix.stypy_kwargs_param_name = None
    randomMatrix.stypy_call_defaults = defaults
    randomMatrix.stypy_call_varargs = varargs
    randomMatrix.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'randomMatrix', ['n', 'upperBound', 'seed'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'randomMatrix', localization, ['n', 'upperBound', 'seed'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'randomMatrix(...)' code ##################

    
    # Call to seed(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'seed' (line 38)
    seed_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'seed', False)
    # Processing the call keyword arguments (line 38)
    kwargs_4 = {}
    # Getting the type of 'random' (line 38)
    random_1 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'random', False)
    # Obtaining the member 'seed' of a type (line 38)
    seed_2 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 4), random_1, 'seed')
    # Calling seed(args, kwargs) (line 38)
    seed_call_result_5 = invoke(stypy.reporting.localization.Localization(__file__, 38, 4), seed_2, *[seed_3], **kwargs_4)
    
    
    # Assigning a List to a Name (line 39):
    
    # Obtaining an instance of the builtin type 'list' (line 39)
    list_6 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 39)
    
    # Assigning a type to the variable 'm' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'm', list_6)
    
    
    # Call to range(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'n' (line 40)
    n_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 19), 'n', False)
    # Processing the call keyword arguments (line 40)
    kwargs_9 = {}
    # Getting the type of 'range' (line 40)
    range_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 13), 'range', False)
    # Calling range(args, kwargs) (line 40)
    range_call_result_10 = invoke(stypy.reporting.localization.Localization(__file__, 40, 13), range_7, *[n_8], **kwargs_9)
    
    # Testing the type of a for loop iterable (line 40)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 40, 4), range_call_result_10)
    # Getting the type of the for loop variable (line 40)
    for_loop_var_11 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 40, 4), range_call_result_10)
    # Assigning a type to the variable 'r' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'r', for_loop_var_11)
    # SSA begins for a for statement (line 40)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a List to a Name (line 41):
    
    # Obtaining an instance of the builtin type 'list' (line 41)
    list_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 41)
    
    # Assigning a type to the variable 'sm' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'sm', list_12)
    
    # Call to append(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'sm' (line 42)
    sm_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 17), 'sm', False)
    # Processing the call keyword arguments (line 42)
    kwargs_16 = {}
    # Getting the type of 'm' (line 42)
    m_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'm', False)
    # Obtaining the member 'append' of a type (line 42)
    append_14 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), m_13, 'append')
    # Calling append(args, kwargs) (line 42)
    append_call_result_17 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), append_14, *[sm_15], **kwargs_16)
    
    
    
    # Call to range(...): (line 43)
    # Processing the call arguments (line 43)
    # Getting the type of 'n' (line 43)
    n_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 23), 'n', False)
    # Processing the call keyword arguments (line 43)
    kwargs_20 = {}
    # Getting the type of 'range' (line 43)
    range_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 17), 'range', False)
    # Calling range(args, kwargs) (line 43)
    range_call_result_21 = invoke(stypy.reporting.localization.Localization(__file__, 43, 17), range_18, *[n_19], **kwargs_20)
    
    # Testing the type of a for loop iterable (line 43)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 43, 8), range_call_result_21)
    # Getting the type of the for loop variable (line 43)
    for_loop_var_22 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 43, 8), range_call_result_21)
    # Assigning a type to the variable 'c' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'c', for_loop_var_22)
    # SSA begins for a for statement (line 43)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'upperBound' (line 44)
    upperBound_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 23), 'upperBound', False)
    
    # Call to random(...): (line 44)
    # Processing the call keyword arguments (line 44)
    kwargs_28 = {}
    # Getting the type of 'random' (line 44)
    random_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 36), 'random', False)
    # Obtaining the member 'random' of a type (line 44)
    random_27 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 36), random_26, 'random')
    # Calling random(args, kwargs) (line 44)
    random_call_result_29 = invoke(stypy.reporting.localization.Localization(__file__, 44, 36), random_27, *[], **kwargs_28)
    
    # Applying the binary operator '*' (line 44)
    result_mul_30 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 23), '*', upperBound_25, random_call_result_29)
    
    # Processing the call keyword arguments (line 44)
    kwargs_31 = {}
    # Getting the type of 'sm' (line 44)
    sm_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 13), 'sm', False)
    # Obtaining the member 'append' of a type (line 44)
    append_24 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 13), sm_23, 'append')
    # Calling append(args, kwargs) (line 44)
    append_call_result_32 = invoke(stypy.reporting.localization.Localization(__file__, 44, 13), append_24, *[result_mul_30], **kwargs_31)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'm' (line 45)
    m_33 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 11), 'm')
    # Assigning a type to the variable 'stypy_return_type' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'stypy_return_type', m_33)
    
    # ################# End of 'randomMatrix(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'randomMatrix' in the type store
    # Getting the type of 'stypy_return_type' (line 37)
    stypy_return_type_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_34)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'randomMatrix'
    return stypy_return_type_34

# Assigning a type to the variable 'randomMatrix' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'randomMatrix', randomMatrix)

@norecursion
def wrappedPath(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'wrappedPath'
    module_type_store = module_type_store.open_function_context('wrappedPath', 48, 0, False)
    
    # Passed parameters checking function
    wrappedPath.stypy_localization = localization
    wrappedPath.stypy_type_of_self = None
    wrappedPath.stypy_type_store = module_type_store
    wrappedPath.stypy_function_name = 'wrappedPath'
    wrappedPath.stypy_param_names_list = ['path']
    wrappedPath.stypy_varargs_param_name = None
    wrappedPath.stypy_kwargs_param_name = None
    wrappedPath.stypy_call_defaults = defaults
    wrappedPath.stypy_call_varargs = varargs
    wrappedPath.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'wrappedPath', ['path'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'wrappedPath', localization, ['path'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'wrappedPath(...)' code ##################

    
    # Obtaining the type of the subscript
    int_35 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 16), 'int')
    slice_36 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 49, 11), int_35, None, None)
    # Getting the type of 'path' (line 49)
    path_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 11), 'path')
    # Obtaining the member '__getitem__' of a type (line 49)
    getitem___38 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 11), path_37, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 49)
    subscript_call_result_39 = invoke(stypy.reporting.localization.Localization(__file__, 49, 11), getitem___38, slice_36)
    
    
    # Obtaining an instance of the builtin type 'list' (line 49)
    list_40 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 49)
    # Adding element type (line 49)
    
    # Obtaining the type of the subscript
    int_41 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 28), 'int')
    # Getting the type of 'path' (line 49)
    path_42 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 23), 'path')
    # Obtaining the member '__getitem__' of a type (line 49)
    getitem___43 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 23), path_42, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 49)
    subscript_call_result_44 = invoke(stypy.reporting.localization.Localization(__file__, 49, 23), getitem___43, int_41)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 22), list_40, subscript_call_result_44)
    
    # Applying the binary operator '+' (line 49)
    result_add_45 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 11), '+', subscript_call_result_39, list_40)
    
    # Assigning a type to the variable 'stypy_return_type' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'stypy_return_type', result_add_45)
    
    # ################# End of 'wrappedPath(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'wrappedPath' in the type store
    # Getting the type of 'stypy_return_type' (line 48)
    stypy_return_type_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_46)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'wrappedPath'
    return stypy_return_type_46

# Assigning a type to the variable 'wrappedPath' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'wrappedPath', wrappedPath)

@norecursion
def pathLength(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'pathLength'
    module_type_store = module_type_store.open_function_context('pathLength', 52, 0, False)
    
    # Passed parameters checking function
    pathLength.stypy_localization = localization
    pathLength.stypy_type_of_self = None
    pathLength.stypy_type_store = module_type_store
    pathLength.stypy_function_name = 'pathLength'
    pathLength.stypy_param_names_list = ['cities', 'path']
    pathLength.stypy_varargs_param_name = None
    pathLength.stypy_kwargs_param_name = None
    pathLength.stypy_call_defaults = defaults
    pathLength.stypy_call_varargs = varargs
    pathLength.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'pathLength', ['cities', 'path'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'pathLength', localization, ['cities', 'path'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'pathLength(...)' code ##################

    
    # Assigning a Call to a Name (line 53):
    
    # Call to zip(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'path' (line 53)
    path_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 16), 'path', False)
    
    # Call to wrappedPath(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'path' (line 53)
    path_50 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 34), 'path', False)
    # Processing the call keyword arguments (line 53)
    kwargs_51 = {}
    # Getting the type of 'wrappedPath' (line 53)
    wrappedPath_49 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 22), 'wrappedPath', False)
    # Calling wrappedPath(args, kwargs) (line 53)
    wrappedPath_call_result_52 = invoke(stypy.reporting.localization.Localization(__file__, 53, 22), wrappedPath_49, *[path_50], **kwargs_51)
    
    # Processing the call keyword arguments (line 53)
    kwargs_53 = {}
    # Getting the type of 'zip' (line 53)
    zip_47 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'zip', False)
    # Calling zip(args, kwargs) (line 53)
    zip_call_result_54 = invoke(stypy.reporting.localization.Localization(__file__, 53, 12), zip_47, *[path_48, wrappedPath_call_result_52], **kwargs_53)
    
    # Assigning a type to the variable 'pairs' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'pairs', zip_call_result_54)
    
    # Call to sum(...): (line 54)
    # Processing the call arguments (line 54)
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'pairs' (line 54)
    pairs_63 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 42), 'pairs', False)
    comprehension_64 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 16), pairs_63)
    # Assigning a type to the variable 'r' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'r', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 16), comprehension_64))
    # Assigning a type to the variable 'c' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'c', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 16), comprehension_64))
    
    # Obtaining the type of the subscript
    # Getting the type of 'c' (line 54)
    c_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 26), 'c', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'r' (line 54)
    r_57 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 23), 'r', False)
    # Getting the type of 'cities' (line 54)
    cities_58 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'cities', False)
    # Obtaining the member '__getitem__' of a type (line 54)
    getitem___59 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 16), cities_58, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 54)
    subscript_call_result_60 = invoke(stypy.reporting.localization.Localization(__file__, 54, 16), getitem___59, r_57)
    
    # Obtaining the member '__getitem__' of a type (line 54)
    getitem___61 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 16), subscript_call_result_60, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 54)
    subscript_call_result_62 = invoke(stypy.reporting.localization.Localization(__file__, 54, 16), getitem___61, c_56)
    
    list_65 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 16), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 16), list_65, subscript_call_result_62)
    # Processing the call keyword arguments (line 54)
    kwargs_66 = {}
    # Getting the type of 'sum' (line 54)
    sum_55 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 11), 'sum', False)
    # Calling sum(args, kwargs) (line 54)
    sum_call_result_67 = invoke(stypy.reporting.localization.Localization(__file__, 54, 11), sum_55, *[list_65], **kwargs_66)
    
    # Assigning a type to the variable 'stypy_return_type' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'stypy_return_type', sum_call_result_67)
    
    # ################# End of 'pathLength(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'pathLength' in the type store
    # Getting the type of 'stypy_return_type' (line 52)
    stypy_return_type_68 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_68)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'pathLength'
    return stypy_return_type_68

# Assigning a type to the variable 'pathLength' (line 52)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 0), 'pathLength', pathLength)

@norecursion
def updatePher(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'updatePher'
    module_type_store = module_type_store.open_function_context('updatePher', 58, 0, False)
    
    # Passed parameters checking function
    updatePher.stypy_localization = localization
    updatePher.stypy_type_of_self = None
    updatePher.stypy_type_store = module_type_store
    updatePher.stypy_function_name = 'updatePher'
    updatePher.stypy_param_names_list = ['pher', 'path', 'boost']
    updatePher.stypy_varargs_param_name = None
    updatePher.stypy_kwargs_param_name = None
    updatePher.stypy_call_defaults = defaults
    updatePher.stypy_call_varargs = varargs
    updatePher.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'updatePher', ['pher', 'path', 'boost'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'updatePher', localization, ['pher', 'path', 'boost'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'updatePher(...)' code ##################

    
    # Assigning a Call to a Name (line 59):
    
    # Call to zip(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'path' (line 59)
    path_70 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'path', False)
    
    # Call to wrappedPath(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'path' (line 59)
    path_72 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 34), 'path', False)
    # Processing the call keyword arguments (line 59)
    kwargs_73 = {}
    # Getting the type of 'wrappedPath' (line 59)
    wrappedPath_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 22), 'wrappedPath', False)
    # Calling wrappedPath(args, kwargs) (line 59)
    wrappedPath_call_result_74 = invoke(stypy.reporting.localization.Localization(__file__, 59, 22), wrappedPath_71, *[path_72], **kwargs_73)
    
    # Processing the call keyword arguments (line 59)
    kwargs_75 = {}
    # Getting the type of 'zip' (line 59)
    zip_69 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'zip', False)
    # Calling zip(args, kwargs) (line 59)
    zip_call_result_76 = invoke(stypy.reporting.localization.Localization(__file__, 59, 12), zip_69, *[path_70, wrappedPath_call_result_74], **kwargs_75)
    
    # Assigning a type to the variable 'pairs' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'pairs', zip_call_result_76)
    
    # Getting the type of 'pairs' (line 60)
    pairs_77 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 17), 'pairs')
    # Testing the type of a for loop iterable (line 60)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 60, 4), pairs_77)
    # Getting the type of the for loop variable (line 60)
    for_loop_var_78 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 60, 4), pairs_77)
    # Assigning a type to the variable 'r' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'r', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 4), for_loop_var_78))
    # Assigning a type to the variable 'c' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'c', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 4), for_loop_var_78))
    # SSA begins for a for statement (line 60)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Subscript (line 61):
    
    # Obtaining the type of the subscript
    # Getting the type of 'c' (line 61)
    c_79 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 29), 'c')
    
    # Obtaining the type of the subscript
    # Getting the type of 'r' (line 61)
    r_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 26), 'r')
    # Getting the type of 'pher' (line 61)
    pher_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 21), 'pher')
    # Obtaining the member '__getitem__' of a type (line 61)
    getitem___82 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 21), pher_81, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 61)
    subscript_call_result_83 = invoke(stypy.reporting.localization.Localization(__file__, 61, 21), getitem___82, r_80)
    
    # Obtaining the member '__getitem__' of a type (line 61)
    getitem___84 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 21), subscript_call_result_83, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 61)
    subscript_call_result_85 = invoke(stypy.reporting.localization.Localization(__file__, 61, 21), getitem___84, c_79)
    
    # Getting the type of 'boost' (line 61)
    boost_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 34), 'boost')
    # Applying the binary operator '+' (line 61)
    result_add_87 = python_operator(stypy.reporting.localization.Localization(__file__, 61, 21), '+', subscript_call_result_85, boost_86)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'r' (line 61)
    r_88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 13), 'r')
    # Getting the type of 'pher' (line 61)
    pher_89 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'pher')
    # Obtaining the member '__getitem__' of a type (line 61)
    getitem___90 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), pher_89, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 61)
    subscript_call_result_91 = invoke(stypy.reporting.localization.Localization(__file__, 61, 8), getitem___90, r_88)
    
    # Getting the type of 'c' (line 61)
    c_92 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 16), 'c')
    # Storing an element on a container (line 61)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 8), subscript_call_result_91, (c_92, result_add_87))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'updatePher(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'updatePher' in the type store
    # Getting the type of 'stypy_return_type' (line 58)
    stypy_return_type_93 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_93)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'updatePher'
    return stypy_return_type_93

# Assigning a type to the variable 'updatePher' (line 58)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'updatePher', updatePher)

@norecursion
def evaporatePher(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'evaporatePher'
    module_type_store = module_type_store.open_function_context('evaporatePher', 64, 0, False)
    
    # Passed parameters checking function
    evaporatePher.stypy_localization = localization
    evaporatePher.stypy_type_of_self = None
    evaporatePher.stypy_type_store = module_type_store
    evaporatePher.stypy_function_name = 'evaporatePher'
    evaporatePher.stypy_param_names_list = ['pher', 'maxIter', 'boost']
    evaporatePher.stypy_varargs_param_name = None
    evaporatePher.stypy_kwargs_param_name = None
    evaporatePher.stypy_call_defaults = defaults
    evaporatePher.stypy_call_varargs = varargs
    evaporatePher.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'evaporatePher', ['pher', 'maxIter', 'boost'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'evaporatePher', localization, ['pher', 'maxIter', 'boost'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'evaporatePher(...)' code ##################

    
    # Assigning a BinOp to a Name (line 65):
    # Getting the type of 'boost' (line 65)
    boost_94 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 11), 'boost')
    
    # Call to float(...): (line 65)
    # Processing the call arguments (line 65)
    # Getting the type of 'maxIter' (line 65)
    maxIter_96 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 25), 'maxIter', False)
    # Processing the call keyword arguments (line 65)
    kwargs_97 = {}
    # Getting the type of 'float' (line 65)
    float_95 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 19), 'float', False)
    # Calling float(args, kwargs) (line 65)
    float_call_result_98 = invoke(stypy.reporting.localization.Localization(__file__, 65, 19), float_95, *[maxIter_96], **kwargs_97)
    
    # Applying the binary operator 'div' (line 65)
    result_div_99 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 11), 'div', boost_94, float_call_result_98)
    
    # Assigning a type to the variable 'decr' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'decr', result_div_99)
    
    
    # Call to range(...): (line 66)
    # Processing the call arguments (line 66)
    
    # Call to len(...): (line 66)
    # Processing the call arguments (line 66)
    # Getting the type of 'pher' (line 66)
    pher_102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 23), 'pher', False)
    # Processing the call keyword arguments (line 66)
    kwargs_103 = {}
    # Getting the type of 'len' (line 66)
    len_101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 19), 'len', False)
    # Calling len(args, kwargs) (line 66)
    len_call_result_104 = invoke(stypy.reporting.localization.Localization(__file__, 66, 19), len_101, *[pher_102], **kwargs_103)
    
    # Processing the call keyword arguments (line 66)
    kwargs_105 = {}
    # Getting the type of 'range' (line 66)
    range_100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 13), 'range', False)
    # Calling range(args, kwargs) (line 66)
    range_call_result_106 = invoke(stypy.reporting.localization.Localization(__file__, 66, 13), range_100, *[len_call_result_104], **kwargs_105)
    
    # Testing the type of a for loop iterable (line 66)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 66, 4), range_call_result_106)
    # Getting the type of the for loop variable (line 66)
    for_loop_var_107 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 66, 4), range_call_result_106)
    # Assigning a type to the variable 'r' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'r', for_loop_var_107)
    # SSA begins for a for statement (line 66)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to range(...): (line 67)
    # Processing the call arguments (line 67)
    
    # Call to len(...): (line 67)
    # Processing the call arguments (line 67)
    
    # Obtaining the type of the subscript
    # Getting the type of 'r' (line 67)
    r_110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 32), 'r', False)
    # Getting the type of 'pher' (line 67)
    pher_111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 27), 'pher', False)
    # Obtaining the member '__getitem__' of a type (line 67)
    getitem___112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 27), pher_111, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 67)
    subscript_call_result_113 = invoke(stypy.reporting.localization.Localization(__file__, 67, 27), getitem___112, r_110)
    
    # Processing the call keyword arguments (line 67)
    kwargs_114 = {}
    # Getting the type of 'len' (line 67)
    len_109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 23), 'len', False)
    # Calling len(args, kwargs) (line 67)
    len_call_result_115 = invoke(stypy.reporting.localization.Localization(__file__, 67, 23), len_109, *[subscript_call_result_113], **kwargs_114)
    
    # Processing the call keyword arguments (line 67)
    kwargs_116 = {}
    # Getting the type of 'range' (line 67)
    range_108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 17), 'range', False)
    # Calling range(args, kwargs) (line 67)
    range_call_result_117 = invoke(stypy.reporting.localization.Localization(__file__, 67, 17), range_108, *[len_call_result_115], **kwargs_116)
    
    # Testing the type of a for loop iterable (line 67)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 67, 8), range_call_result_117)
    # Getting the type of the for loop variable (line 67)
    for_loop_var_118 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 67, 8), range_call_result_117)
    # Assigning a type to the variable 'c' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'c', for_loop_var_118)
    # SSA begins for a for statement (line 67)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'c' (line 68)
    c_119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 23), 'c')
    
    # Obtaining the type of the subscript
    # Getting the type of 'r' (line 68)
    r_120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 20), 'r')
    # Getting the type of 'pher' (line 68)
    pher_121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 15), 'pher')
    # Obtaining the member '__getitem__' of a type (line 68)
    getitem___122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 15), pher_121, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 68)
    subscript_call_result_123 = invoke(stypy.reporting.localization.Localization(__file__, 68, 15), getitem___122, r_120)
    
    # Obtaining the member '__getitem__' of a type (line 68)
    getitem___124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 15), subscript_call_result_123, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 68)
    subscript_call_result_125 = invoke(stypy.reporting.localization.Localization(__file__, 68, 15), getitem___124, c_119)
    
    # Getting the type of 'decr' (line 68)
    decr_126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 28), 'decr')
    # Applying the binary operator '>' (line 68)
    result_gt_127 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 15), '>', subscript_call_result_125, decr_126)
    
    # Testing the type of an if condition (line 68)
    if_condition_128 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 68, 12), result_gt_127)
    # Assigning a type to the variable 'if_condition_128' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'if_condition_128', if_condition_128)
    # SSA begins for if statement (line 68)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 69):
    
    # Obtaining the type of the subscript
    # Getting the type of 'c' (line 69)
    c_129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 37), 'c')
    
    # Obtaining the type of the subscript
    # Getting the type of 'r' (line 69)
    r_130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 34), 'r')
    # Getting the type of 'pher' (line 69)
    pher_131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 29), 'pher')
    # Obtaining the member '__getitem__' of a type (line 69)
    getitem___132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 29), pher_131, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 69)
    subscript_call_result_133 = invoke(stypy.reporting.localization.Localization(__file__, 69, 29), getitem___132, r_130)
    
    # Obtaining the member '__getitem__' of a type (line 69)
    getitem___134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 29), subscript_call_result_133, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 69)
    subscript_call_result_135 = invoke(stypy.reporting.localization.Localization(__file__, 69, 29), getitem___134, c_129)
    
    # Getting the type of 'decr' (line 69)
    decr_136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 42), 'decr')
    # Applying the binary operator '-' (line 69)
    result_sub_137 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 29), '-', subscript_call_result_135, decr_136)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'r' (line 69)
    r_138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 21), 'r')
    # Getting the type of 'pher' (line 69)
    pher_139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 16), 'pher')
    # Obtaining the member '__getitem__' of a type (line 69)
    getitem___140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 16), pher_139, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 69)
    subscript_call_result_141 = invoke(stypy.reporting.localization.Localization(__file__, 69, 16), getitem___140, r_138)
    
    # Getting the type of 'c' (line 69)
    c_142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 24), 'c')
    # Storing an element on a container (line 69)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 16), subscript_call_result_141, (c_142, result_sub_137))
    # SSA branch for the else part of an if statement (line 68)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Num to a Subscript (line 71):
    float_143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 29), 'float')
    
    # Obtaining the type of the subscript
    # Getting the type of 'r' (line 71)
    r_144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 21), 'r')
    # Getting the type of 'pher' (line 71)
    pher_145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 16), 'pher')
    # Obtaining the member '__getitem__' of a type (line 71)
    getitem___146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 16), pher_145, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 71)
    subscript_call_result_147 = invoke(stypy.reporting.localization.Localization(__file__, 71, 16), getitem___146, r_144)
    
    # Getting the type of 'c' (line 71)
    c_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 24), 'c')
    # Storing an element on a container (line 71)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 16), subscript_call_result_147, (c_148, float_143))
    # SSA join for if statement (line 68)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'evaporatePher(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'evaporatePher' in the type store
    # Getting the type of 'stypy_return_type' (line 64)
    stypy_return_type_149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_149)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'evaporatePher'
    return stypy_return_type_149

# Assigning a type to the variable 'evaporatePher' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'evaporatePher', evaporatePher)

@norecursion
def doSumWeight(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'doSumWeight'
    module_type_store = module_type_store.open_function_context('doSumWeight', 75, 0, False)
    
    # Passed parameters checking function
    doSumWeight.stypy_localization = localization
    doSumWeight.stypy_type_of_self = None
    doSumWeight.stypy_type_store = module_type_store
    doSumWeight.stypy_function_name = 'doSumWeight'
    doSumWeight.stypy_param_names_list = ['cities', 'pher', 'used', 'current']
    doSumWeight.stypy_varargs_param_name = None
    doSumWeight.stypy_kwargs_param_name = None
    doSumWeight.stypy_call_defaults = defaults
    doSumWeight.stypy_call_varargs = varargs
    doSumWeight.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'doSumWeight', ['cities', 'pher', 'used', 'current'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'doSumWeight', localization, ['cities', 'pher', 'used', 'current'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'doSumWeight(...)' code ##################

    
    # Assigning a Num to a Name (line 76):
    float_150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 19), 'float')
    # Assigning a type to the variable 'runningTotal' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'runningTotal', float_150)
    
    
    # Call to range(...): (line 77)
    # Processing the call arguments (line 77)
    
    # Call to len(...): (line 77)
    # Processing the call arguments (line 77)
    # Getting the type of 'cities' (line 77)
    cities_153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 26), 'cities', False)
    # Processing the call keyword arguments (line 77)
    kwargs_154 = {}
    # Getting the type of 'len' (line 77)
    len_152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 22), 'len', False)
    # Calling len(args, kwargs) (line 77)
    len_call_result_155 = invoke(stypy.reporting.localization.Localization(__file__, 77, 22), len_152, *[cities_153], **kwargs_154)
    
    # Processing the call keyword arguments (line 77)
    kwargs_156 = {}
    # Getting the type of 'range' (line 77)
    range_151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 16), 'range', False)
    # Calling range(args, kwargs) (line 77)
    range_call_result_157 = invoke(stypy.reporting.localization.Localization(__file__, 77, 16), range_151, *[len_call_result_155], **kwargs_156)
    
    # Testing the type of a for loop iterable (line 77)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 77, 4), range_call_result_157)
    # Getting the type of the for loop variable (line 77)
    for_loop_var_158 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 77, 4), range_call_result_157)
    # Assigning a type to the variable 'city' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'city', for_loop_var_158)
    # SSA begins for a for statement (line 77)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    
    # Call to has_key(...): (line 78)
    # Processing the call arguments (line 78)
    # Getting the type of 'city' (line 78)
    city_161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 28), 'city', False)
    # Processing the call keyword arguments (line 78)
    kwargs_162 = {}
    # Getting the type of 'used' (line 78)
    used_159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 15), 'used', False)
    # Obtaining the member 'has_key' of a type (line 78)
    has_key_160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 15), used_159, 'has_key')
    # Calling has_key(args, kwargs) (line 78)
    has_key_call_result_163 = invoke(stypy.reporting.localization.Localization(__file__, 78, 15), has_key_160, *[city_161], **kwargs_162)
    
    # Applying the 'not' unary operator (line 78)
    result_not__164 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 11), 'not', has_key_call_result_163)
    
    # Testing the type of an if condition (line 78)
    if_condition_165 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 78, 8), result_not__164)
    # Assigning a type to the variable 'if_condition_165' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'if_condition_165', if_condition_165)
    # SSA begins for if statement (line 78)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 79):
    # Getting the type of 'runningTotal' (line 79)
    runningTotal_166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 28), 'runningTotal')
    
    # Obtaining the type of the subscript
    # Getting the type of 'city' (line 80)
    city_167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 44), 'city')
    
    # Obtaining the type of the subscript
    # Getting the type of 'current' (line 80)
    current_168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 35), 'current')
    # Getting the type of 'cities' (line 80)
    cities_169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 28), 'cities')
    # Obtaining the member '__getitem__' of a type (line 80)
    getitem___170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 28), cities_169, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 80)
    subscript_call_result_171 = invoke(stypy.reporting.localization.Localization(__file__, 80, 28), getitem___170, current_168)
    
    # Obtaining the member '__getitem__' of a type (line 80)
    getitem___172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 28), subscript_call_result_171, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 80)
    subscript_call_result_173 = invoke(stypy.reporting.localization.Localization(__file__, 80, 28), getitem___172, city_167)
    
    float_174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 53), 'float')
    
    # Obtaining the type of the subscript
    # Getting the type of 'city' (line 80)
    city_175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 73), 'city')
    
    # Obtaining the type of the subscript
    # Getting the type of 'current' (line 80)
    current_176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 64), 'current')
    # Getting the type of 'pher' (line 80)
    pher_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 59), 'pher')
    # Obtaining the member '__getitem__' of a type (line 80)
    getitem___178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 59), pher_177, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 80)
    subscript_call_result_179 = invoke(stypy.reporting.localization.Localization(__file__, 80, 59), getitem___178, current_176)
    
    # Obtaining the member '__getitem__' of a type (line 80)
    getitem___180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 59), subscript_call_result_179, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 80)
    subscript_call_result_181 = invoke(stypy.reporting.localization.Localization(__file__, 80, 59), getitem___180, city_175)
    
    # Applying the binary operator '+' (line 80)
    result_add_182 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 53), '+', float_174, subscript_call_result_181)
    
    # Applying the binary operator '*' (line 80)
    result_mul_183 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 28), '*', subscript_call_result_173, result_add_182)
    
    # Applying the binary operator '+' (line 79)
    result_add_184 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 28), '+', runningTotal_166, result_mul_183)
    
    # Assigning a type to the variable 'runningTotal' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'runningTotal', result_add_184)
    # SSA join for if statement (line 78)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'runningTotal' (line 81)
    runningTotal_185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 11), 'runningTotal')
    # Assigning a type to the variable 'stypy_return_type' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy_return_type', runningTotal_185)
    
    # ################# End of 'doSumWeight(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'doSumWeight' in the type store
    # Getting the type of 'stypy_return_type' (line 75)
    stypy_return_type_186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_186)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'doSumWeight'
    return stypy_return_type_186

# Assigning a type to the variable 'doSumWeight' (line 75)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 0), 'doSumWeight', doSumWeight)

@norecursion
def findSumWeight(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'findSumWeight'
    module_type_store = module_type_store.open_function_context('findSumWeight', 85, 0, False)
    
    # Passed parameters checking function
    findSumWeight.stypy_localization = localization
    findSumWeight.stypy_type_of_self = None
    findSumWeight.stypy_type_store = module_type_store
    findSumWeight.stypy_function_name = 'findSumWeight'
    findSumWeight.stypy_param_names_list = ['cities', 'pher', 'used', 'current', 'soughtTotal']
    findSumWeight.stypy_varargs_param_name = None
    findSumWeight.stypy_kwargs_param_name = None
    findSumWeight.stypy_call_defaults = defaults
    findSumWeight.stypy_call_varargs = varargs
    findSumWeight.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'findSumWeight', ['cities', 'pher', 'used', 'current', 'soughtTotal'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'findSumWeight', localization, ['cities', 'pher', 'used', 'current', 'soughtTotal'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'findSumWeight(...)' code ##################

    
    # Assigning a Num to a Name (line 86):
    float_187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 19), 'float')
    # Assigning a type to the variable 'runningTotal' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'runningTotal', float_187)
    
    # Assigning a Num to a Name (line 87):
    int_188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 11), 'int')
    # Assigning a type to the variable 'next' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'next', int_188)
    
    
    # Call to range(...): (line 88)
    # Processing the call arguments (line 88)
    
    # Call to len(...): (line 88)
    # Processing the call arguments (line 88)
    # Getting the type of 'cities' (line 88)
    cities_191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 26), 'cities', False)
    # Processing the call keyword arguments (line 88)
    kwargs_192 = {}
    # Getting the type of 'len' (line 88)
    len_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 22), 'len', False)
    # Calling len(args, kwargs) (line 88)
    len_call_result_193 = invoke(stypy.reporting.localization.Localization(__file__, 88, 22), len_190, *[cities_191], **kwargs_192)
    
    # Processing the call keyword arguments (line 88)
    kwargs_194 = {}
    # Getting the type of 'range' (line 88)
    range_189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 16), 'range', False)
    # Calling range(args, kwargs) (line 88)
    range_call_result_195 = invoke(stypy.reporting.localization.Localization(__file__, 88, 16), range_189, *[len_call_result_193], **kwargs_194)
    
    # Testing the type of a for loop iterable (line 88)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 88, 4), range_call_result_195)
    # Getting the type of the for loop variable (line 88)
    for_loop_var_196 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 88, 4), range_call_result_195)
    # Assigning a type to the variable 'city' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'city', for_loop_var_196)
    # SSA begins for a for statement (line 88)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'runningTotal' (line 89)
    runningTotal_197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 11), 'runningTotal')
    # Getting the type of 'soughtTotal' (line 89)
    soughtTotal_198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 27), 'soughtTotal')
    # Applying the binary operator '>=' (line 89)
    result_ge_199 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 11), '>=', runningTotal_197, soughtTotal_198)
    
    # Testing the type of an if condition (line 89)
    if_condition_200 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 8), result_ge_199)
    # Assigning a type to the variable 'if_condition_200' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'if_condition_200', if_condition_200)
    # SSA begins for if statement (line 89)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 89)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to has_key(...): (line 91)
    # Processing the call arguments (line 91)
    # Getting the type of 'city' (line 91)
    city_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 28), 'city', False)
    # Processing the call keyword arguments (line 91)
    kwargs_204 = {}
    # Getting the type of 'used' (line 91)
    used_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 15), 'used', False)
    # Obtaining the member 'has_key' of a type (line 91)
    has_key_202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 15), used_201, 'has_key')
    # Calling has_key(args, kwargs) (line 91)
    has_key_call_result_205 = invoke(stypy.reporting.localization.Localization(__file__, 91, 15), has_key_202, *[city_203], **kwargs_204)
    
    # Applying the 'not' unary operator (line 91)
    result_not__206 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 11), 'not', has_key_call_result_205)
    
    # Testing the type of an if condition (line 91)
    if_condition_207 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 91, 8), result_not__206)
    # Assigning a type to the variable 'if_condition_207' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'if_condition_207', if_condition_207)
    # SSA begins for if statement (line 91)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 92):
    # Getting the type of 'runningTotal' (line 92)
    runningTotal_208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 28), 'runningTotal')
    
    # Obtaining the type of the subscript
    # Getting the type of 'city' (line 93)
    city_209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 44), 'city')
    
    # Obtaining the type of the subscript
    # Getting the type of 'current' (line 93)
    current_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 35), 'current')
    # Getting the type of 'cities' (line 93)
    cities_211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 28), 'cities')
    # Obtaining the member '__getitem__' of a type (line 93)
    getitem___212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 28), cities_211, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 93)
    subscript_call_result_213 = invoke(stypy.reporting.localization.Localization(__file__, 93, 28), getitem___212, current_210)
    
    # Obtaining the member '__getitem__' of a type (line 93)
    getitem___214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 28), subscript_call_result_213, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 93)
    subscript_call_result_215 = invoke(stypy.reporting.localization.Localization(__file__, 93, 28), getitem___214, city_209)
    
    float_216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 53), 'float')
    
    # Obtaining the type of the subscript
    # Getting the type of 'city' (line 93)
    city_217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 73), 'city')
    
    # Obtaining the type of the subscript
    # Getting the type of 'current' (line 93)
    current_218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 64), 'current')
    # Getting the type of 'pher' (line 93)
    pher_219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 59), 'pher')
    # Obtaining the member '__getitem__' of a type (line 93)
    getitem___220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 59), pher_219, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 93)
    subscript_call_result_221 = invoke(stypy.reporting.localization.Localization(__file__, 93, 59), getitem___220, current_218)
    
    # Obtaining the member '__getitem__' of a type (line 93)
    getitem___222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 59), subscript_call_result_221, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 93)
    subscript_call_result_223 = invoke(stypy.reporting.localization.Localization(__file__, 93, 59), getitem___222, city_217)
    
    # Applying the binary operator '+' (line 93)
    result_add_224 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 53), '+', float_216, subscript_call_result_223)
    
    # Applying the binary operator '*' (line 93)
    result_mul_225 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 28), '*', subscript_call_result_215, result_add_224)
    
    # Applying the binary operator '+' (line 92)
    result_add_226 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 28), '+', runningTotal_208, result_mul_225)
    
    # Assigning a type to the variable 'runningTotal' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'runningTotal', result_add_226)
    
    # Assigning a Name to a Name (line 94):
    # Getting the type of 'city' (line 94)
    city_227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 19), 'city')
    # Assigning a type to the variable 'next' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'next', city_227)
    # SSA join for if statement (line 91)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'next' (line 95)
    next_228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 11), 'next')
    # Assigning a type to the variable 'stypy_return_type' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'stypy_return_type', next_228)
    
    # ################# End of 'findSumWeight(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'findSumWeight' in the type store
    # Getting the type of 'stypy_return_type' (line 85)
    stypy_return_type_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_229)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'findSumWeight'
    return stypy_return_type_229

# Assigning a type to the variable 'findSumWeight' (line 85)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'findSumWeight', findSumWeight)

@norecursion
def genPath(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'genPath'
    module_type_store = module_type_store.open_function_context('genPath', 98, 0, False)
    
    # Passed parameters checking function
    genPath.stypy_localization = localization
    genPath.stypy_type_of_self = None
    genPath.stypy_type_store = module_type_store
    genPath.stypy_function_name = 'genPath'
    genPath.stypy_param_names_list = ['cities', 'pher']
    genPath.stypy_varargs_param_name = None
    genPath.stypy_kwargs_param_name = None
    genPath.stypy_call_defaults = defaults
    genPath.stypy_call_varargs = varargs
    genPath.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'genPath', ['cities', 'pher'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'genPath', localization, ['cities', 'pher'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'genPath(...)' code ##################

    
    # Assigning a Call to a Name (line 99):
    
    # Call to randint(...): (line 99)
    # Processing the call arguments (line 99)
    int_232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 29), 'int')
    
    # Call to len(...): (line 99)
    # Processing the call arguments (line 99)
    # Getting the type of 'cities' (line 99)
    cities_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 36), 'cities', False)
    # Processing the call keyword arguments (line 99)
    kwargs_235 = {}
    # Getting the type of 'len' (line 99)
    len_233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 32), 'len', False)
    # Calling len(args, kwargs) (line 99)
    len_call_result_236 = invoke(stypy.reporting.localization.Localization(__file__, 99, 32), len_233, *[cities_234], **kwargs_235)
    
    int_237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 44), 'int')
    # Applying the binary operator '-' (line 99)
    result_sub_238 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 32), '-', len_call_result_236, int_237)
    
    # Processing the call keyword arguments (line 99)
    kwargs_239 = {}
    # Getting the type of 'random' (line 99)
    random_230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 14), 'random', False)
    # Obtaining the member 'randint' of a type (line 99)
    randint_231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 14), random_230, 'randint')
    # Calling randint(args, kwargs) (line 99)
    randint_call_result_240 = invoke(stypy.reporting.localization.Localization(__file__, 99, 14), randint_231, *[int_232, result_sub_238], **kwargs_239)
    
    # Assigning a type to the variable 'current' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'current', randint_call_result_240)
    
    # Assigning a List to a Name (line 100):
    
    # Obtaining an instance of the builtin type 'list' (line 100)
    list_241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 100)
    # Adding element type (line 100)
    # Getting the type of 'current' (line 100)
    current_242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'current')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 11), list_241, current_242)
    
    # Assigning a type to the variable 'path' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'path', list_241)
    
    # Assigning a Dict to a Name (line 101):
    
    # Obtaining an instance of the builtin type 'dict' (line 101)
    dict_243 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 11), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 101)
    # Adding element type (key, value) (line 101)
    # Getting the type of 'current' (line 101)
    current_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 12), 'current')
    int_245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 20), 'int')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 11), dict_243, (current_244, int_245))
    
    # Assigning a type to the variable 'used' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'used', dict_243)
    
    
    
    # Call to len(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 'used' (line 102)
    used_247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 14), 'used', False)
    # Processing the call keyword arguments (line 102)
    kwargs_248 = {}
    # Getting the type of 'len' (line 102)
    len_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 10), 'len', False)
    # Calling len(args, kwargs) (line 102)
    len_call_result_249 = invoke(stypy.reporting.localization.Localization(__file__, 102, 10), len_246, *[used_247], **kwargs_248)
    
    
    # Call to len(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 'cities' (line 102)
    cities_251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 26), 'cities', False)
    # Processing the call keyword arguments (line 102)
    kwargs_252 = {}
    # Getting the type of 'len' (line 102)
    len_250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 22), 'len', False)
    # Calling len(args, kwargs) (line 102)
    len_call_result_253 = invoke(stypy.reporting.localization.Localization(__file__, 102, 22), len_250, *[cities_251], **kwargs_252)
    
    # Applying the binary operator '<' (line 102)
    result_lt_254 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 10), '<', len_call_result_249, len_call_result_253)
    
    # Testing the type of an if condition (line 102)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 102, 4), result_lt_254)
    # SSA begins for while statement (line 102)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Name (line 103):
    
    # Call to doSumWeight(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'cities' (line 103)
    cities_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 32), 'cities', False)
    # Getting the type of 'pher' (line 103)
    pher_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 40), 'pher', False)
    # Getting the type of 'used' (line 103)
    used_258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 46), 'used', False)
    # Getting the type of 'current' (line 103)
    current_259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 52), 'current', False)
    # Processing the call keyword arguments (line 103)
    kwargs_260 = {}
    # Getting the type of 'doSumWeight' (line 103)
    doSumWeight_255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 20), 'doSumWeight', False)
    # Calling doSumWeight(args, kwargs) (line 103)
    doSumWeight_call_result_261 = invoke(stypy.reporting.localization.Localization(__file__, 103, 20), doSumWeight_255, *[cities_256, pher_257, used_258, current_259], **kwargs_260)
    
    # Assigning a type to the variable 'sumWeight' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'sumWeight', doSumWeight_call_result_261)
    
    # Assigning a BinOp to a Name (line 104):
    
    # Call to random(...): (line 104)
    # Processing the call keyword arguments (line 104)
    kwargs_264 = {}
    # Getting the type of 'random' (line 104)
    random_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 19), 'random', False)
    # Obtaining the member 'random' of a type (line 104)
    random_263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 19), random_262, 'random')
    # Calling random(args, kwargs) (line 104)
    random_call_result_265 = invoke(stypy.reporting.localization.Localization(__file__, 104, 19), random_263, *[], **kwargs_264)
    
    # Getting the type of 'sumWeight' (line 104)
    sumWeight_266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 37), 'sumWeight')
    # Applying the binary operator '*' (line 104)
    result_mul_267 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 19), '*', random_call_result_265, sumWeight_266)
    
    # Assigning a type to the variable 'rndValue' (line 104)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'rndValue', result_mul_267)
    
    # Assigning a Call to a Name (line 105):
    
    # Call to findSumWeight(...): (line 105)
    # Processing the call arguments (line 105)
    # Getting the type of 'cities' (line 105)
    cities_269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 32), 'cities', False)
    # Getting the type of 'pher' (line 105)
    pher_270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 40), 'pher', False)
    # Getting the type of 'used' (line 105)
    used_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 46), 'used', False)
    # Getting the type of 'current' (line 105)
    current_272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 52), 'current', False)
    # Getting the type of 'rndValue' (line 105)
    rndValue_273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 61), 'rndValue', False)
    # Processing the call keyword arguments (line 105)
    kwargs_274 = {}
    # Getting the type of 'findSumWeight' (line 105)
    findSumWeight_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 18), 'findSumWeight', False)
    # Calling findSumWeight(args, kwargs) (line 105)
    findSumWeight_call_result_275 = invoke(stypy.reporting.localization.Localization(__file__, 105, 18), findSumWeight_268, *[cities_269, pher_270, used_271, current_272, rndValue_273], **kwargs_274)
    
    # Assigning a type to the variable 'current' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'current', findSumWeight_call_result_275)
    
    # Call to append(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'current' (line 106)
    current_278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 20), 'current', False)
    # Processing the call keyword arguments (line 106)
    kwargs_279 = {}
    # Getting the type of 'path' (line 106)
    path_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'path', False)
    # Obtaining the member 'append' of a type (line 106)
    append_277 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 8), path_276, 'append')
    # Calling append(args, kwargs) (line 106)
    append_call_result_280 = invoke(stypy.reporting.localization.Localization(__file__, 106, 8), append_277, *[current_278], **kwargs_279)
    
    
    # Assigning a Num to a Subscript (line 107):
    int_281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 24), 'int')
    # Getting the type of 'used' (line 107)
    used_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 8), 'used')
    # Getting the type of 'current' (line 107)
    current_283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 13), 'current')
    # Storing an element on a container (line 107)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 8), used_282, (current_283, int_281))
    # SSA join for while statement (line 102)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'path' (line 108)
    path_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 11), 'path')
    # Assigning a type to the variable 'stypy_return_type' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'stypy_return_type', path_284)
    
    # ################# End of 'genPath(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'genPath' in the type store
    # Getting the type of 'stypy_return_type' (line 98)
    stypy_return_type_285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_285)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'genPath'
    return stypy_return_type_285

# Assigning a type to the variable 'genPath' (line 98)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 0), 'genPath', genPath)

@norecursion
def bestPath(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'bestPath'
    module_type_store = module_type_store.open_function_context('bestPath', 111, 0, False)
    
    # Passed parameters checking function
    bestPath.stypy_localization = localization
    bestPath.stypy_type_of_self = None
    bestPath.stypy_type_store = module_type_store
    bestPath.stypy_function_name = 'bestPath'
    bestPath.stypy_param_names_list = ['cities', 'seed', 'maxIter', 'boost']
    bestPath.stypy_varargs_param_name = None
    bestPath.stypy_kwargs_param_name = None
    bestPath.stypy_call_defaults = defaults
    bestPath.stypy_call_varargs = varargs
    bestPath.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'bestPath', ['cities', 'seed', 'maxIter', 'boost'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'bestPath', localization, ['cities', 'seed', 'maxIter', 'boost'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'bestPath(...)' code ##################

    
    # Assigning a Call to a Name (line 112):
    
    # Call to randomMatrix(...): (line 112)
    # Processing the call arguments (line 112)
    
    # Call to len(...): (line 112)
    # Processing the call arguments (line 112)
    # Getting the type of 'cities' (line 112)
    cities_288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 28), 'cities', False)
    # Processing the call keyword arguments (line 112)
    kwargs_289 = {}
    # Getting the type of 'len' (line 112)
    len_287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 24), 'len', False)
    # Calling len(args, kwargs) (line 112)
    len_call_result_290 = invoke(stypy.reporting.localization.Localization(__file__, 112, 24), len_287, *[cities_288], **kwargs_289)
    
    int_291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 37), 'int')
    int_292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 40), 'int')
    # Processing the call keyword arguments (line 112)
    kwargs_293 = {}
    # Getting the type of 'randomMatrix' (line 112)
    randomMatrix_286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 11), 'randomMatrix', False)
    # Calling randomMatrix(args, kwargs) (line 112)
    randomMatrix_call_result_294 = invoke(stypy.reporting.localization.Localization(__file__, 112, 11), randomMatrix_286, *[len_call_result_290, int_291, int_292], **kwargs_293)
    
    # Assigning a type to the variable 'pher' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'pher', randomMatrix_call_result_294)
    
    # Call to seed(...): (line 113)
    # Processing the call arguments (line 113)
    # Getting the type of 'seed' (line 113)
    seed_297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 16), 'seed', False)
    # Processing the call keyword arguments (line 113)
    kwargs_298 = {}
    # Getting the type of 'random' (line 113)
    random_295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'random', False)
    # Obtaining the member 'seed' of a type (line 113)
    seed_296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 4), random_295, 'seed')
    # Calling seed(args, kwargs) (line 113)
    seed_call_result_299 = invoke(stypy.reporting.localization.Localization(__file__, 113, 4), seed_296, *[seed_297], **kwargs_298)
    
    
    # Assigning a Num to a Name (line 114):
    float_300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 14), 'float')
    # Assigning a type to the variable 'bestLen' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'bestLen', float_300)
    
    # Assigning a List to a Name (line 115):
    
    # Obtaining an instance of the builtin type 'list' (line 115)
    list_301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 15), 'list')
    # Adding type elements to the builtin type 'list' instance (line 115)
    
    # Assigning a type to the variable 'bestPath' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'bestPath', list_301)
    
    
    # Call to range(...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 'maxIter' (line 116)
    maxIter_303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 22), 'maxIter', False)
    # Processing the call keyword arguments (line 116)
    kwargs_304 = {}
    # Getting the type of 'range' (line 116)
    range_302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 16), 'range', False)
    # Calling range(args, kwargs) (line 116)
    range_call_result_305 = invoke(stypy.reporting.localization.Localization(__file__, 116, 16), range_302, *[maxIter_303], **kwargs_304)
    
    # Testing the type of a for loop iterable (line 116)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 116, 4), range_call_result_305)
    # Getting the type of the for loop variable (line 116)
    for_loop_var_306 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 116, 4), range_call_result_305)
    # Assigning a type to the variable 'iter' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'iter', for_loop_var_306)
    # SSA begins for a for statement (line 116)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 117):
    
    # Call to genPath(...): (line 117)
    # Processing the call arguments (line 117)
    # Getting the type of 'cities' (line 117)
    cities_308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 23), 'cities', False)
    # Getting the type of 'pher' (line 117)
    pher_309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 31), 'pher', False)
    # Processing the call keyword arguments (line 117)
    kwargs_310 = {}
    # Getting the type of 'genPath' (line 117)
    genPath_307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 15), 'genPath', False)
    # Calling genPath(args, kwargs) (line 117)
    genPath_call_result_311 = invoke(stypy.reporting.localization.Localization(__file__, 117, 15), genPath_307, *[cities_308, pher_309], **kwargs_310)
    
    # Assigning a type to the variable 'path' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'path', genPath_call_result_311)
    
    # Assigning a Call to a Name (line 118):
    
    # Call to pathLength(...): (line 118)
    # Processing the call arguments (line 118)
    # Getting the type of 'cities' (line 118)
    cities_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 29), 'cities', False)
    # Getting the type of 'path' (line 118)
    path_314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 37), 'path', False)
    # Processing the call keyword arguments (line 118)
    kwargs_315 = {}
    # Getting the type of 'pathLength' (line 118)
    pathLength_312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 18), 'pathLength', False)
    # Calling pathLength(args, kwargs) (line 118)
    pathLength_call_result_316 = invoke(stypy.reporting.localization.Localization(__file__, 118, 18), pathLength_312, *[cities_313, path_314], **kwargs_315)
    
    # Assigning a type to the variable 'pathLen' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'pathLen', pathLength_call_result_316)
    
    
    # Getting the type of 'pathLen' (line 119)
    pathLen_317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 11), 'pathLen')
    # Getting the type of 'bestLen' (line 119)
    bestLen_318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 21), 'bestLen')
    # Applying the binary operator '>' (line 119)
    result_gt_319 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 11), '>', pathLen_317, bestLen_318)
    
    # Testing the type of an if condition (line 119)
    if_condition_320 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 119, 8), result_gt_319)
    # Assigning a type to the variable 'if_condition_320' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'if_condition_320', if_condition_320)
    # SSA begins for if statement (line 119)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to updatePher(...): (line 121)
    # Processing the call arguments (line 121)
    # Getting the type of 'pher' (line 121)
    pher_322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 23), 'pher', False)
    # Getting the type of 'path' (line 121)
    path_323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 29), 'path', False)
    # Getting the type of 'boost' (line 121)
    boost_324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 35), 'boost', False)
    # Processing the call keyword arguments (line 121)
    kwargs_325 = {}
    # Getting the type of 'updatePher' (line 121)
    updatePher_321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'updatePher', False)
    # Calling updatePher(args, kwargs) (line 121)
    updatePher_call_result_326 = invoke(stypy.reporting.localization.Localization(__file__, 121, 12), updatePher_321, *[pher_322, path_323, boost_324], **kwargs_325)
    
    
    # Assigning a Name to a Name (line 122):
    # Getting the type of 'pathLen' (line 122)
    pathLen_327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 22), 'pathLen')
    # Assigning a type to the variable 'bestLen' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'bestLen', pathLen_327)
    
    # Assigning a Name to a Name (line 123):
    # Getting the type of 'path' (line 123)
    path_328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 23), 'path')
    # Assigning a type to the variable 'bestPath' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'bestPath', path_328)
    # SSA join for if statement (line 119)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to evaporatePher(...): (line 124)
    # Processing the call arguments (line 124)
    # Getting the type of 'pher' (line 124)
    pher_330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 22), 'pher', False)
    # Getting the type of 'maxIter' (line 124)
    maxIter_331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 28), 'maxIter', False)
    # Getting the type of 'boost' (line 124)
    boost_332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 37), 'boost', False)
    # Processing the call keyword arguments (line 124)
    kwargs_333 = {}
    # Getting the type of 'evaporatePher' (line 124)
    evaporatePher_329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'evaporatePher', False)
    # Calling evaporatePher(args, kwargs) (line 124)
    evaporatePher_call_result_334 = invoke(stypy.reporting.localization.Localization(__file__, 124, 8), evaporatePher_329, *[pher_330, maxIter_331, boost_332], **kwargs_333)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'bestPath' (line 125)
    bestPath_335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 11), 'bestPath')
    # Assigning a type to the variable 'stypy_return_type' (line 125)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'stypy_return_type', bestPath_335)
    
    # ################# End of 'bestPath(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'bestPath' in the type store
    # Getting the type of 'stypy_return_type' (line 111)
    stypy_return_type_336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_336)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'bestPath'
    return stypy_return_type_336

# Assigning a type to the variable 'bestPath' (line 111)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 0), 'bestPath', bestPath)

@norecursion
def main(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'main'
    module_type_store = module_type_store.open_function_context('main', 127, 0, False)
    
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

    
    # Assigning a Num to a Name (line 128):
    int_337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 11), 'int')
    # Assigning a type to the variable 'seed' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'seed', int_337)
    
    # Assigning a Num to a Name (line 129):
    int_338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 12), 'int')
    # Assigning a type to the variable 'boost' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'boost', int_338)
    
    # Assigning a Num to a Name (line 130):
    int_339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 11), 'int')
    # Assigning a type to the variable 'iter' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'iter', int_339)
    
    # Assigning a Num to a Name (line 131):
    int_340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 16), 'int')
    # Assigning a type to the variable 'numCities' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'numCities', int_340)
    
    # Assigning a Num to a Name (line 132):
    int_341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 18), 'int')
    # Assigning a type to the variable 'maxDistance' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'maxDistance', int_341)
    
    # Assigning a Num to a Name (line 133):
    int_342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 23), 'int')
    # Assigning a type to the variable 'cityDistanceSeed' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'cityDistanceSeed', int_342)
    
    # Assigning a Call to a Name (line 135):
    
    # Call to randomMatrix(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'numCities' (line 135)
    numCities_344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 26), 'numCities', False)
    # Getting the type of 'maxDistance' (line 135)
    maxDistance_345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 37), 'maxDistance', False)
    # Getting the type of 'cityDistanceSeed' (line 135)
    cityDistanceSeed_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 50), 'cityDistanceSeed', False)
    # Processing the call keyword arguments (line 135)
    kwargs_347 = {}
    # Getting the type of 'randomMatrix' (line 135)
    randomMatrix_343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 13), 'randomMatrix', False)
    # Calling randomMatrix(args, kwargs) (line 135)
    randomMatrix_call_result_348 = invoke(stypy.reporting.localization.Localization(__file__, 135, 13), randomMatrix_343, *[numCities_344, maxDistance_345, cityDistanceSeed_346], **kwargs_347)
    
    # Assigning a type to the variable 'cities' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'cities', randomMatrix_call_result_348)
    
    # Assigning a Call to a Name (line 136):
    
    # Call to bestPath(...): (line 136)
    # Processing the call arguments (line 136)
    # Getting the type of 'cities' (line 136)
    cities_350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 20), 'cities', False)
    # Getting the type of 'seed' (line 136)
    seed_351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 28), 'seed', False)
    # Getting the type of 'iter' (line 136)
    iter_352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 34), 'iter', False)
    # Getting the type of 'boost' (line 136)
    boost_353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 40), 'boost', False)
    # Processing the call keyword arguments (line 136)
    kwargs_354 = {}
    # Getting the type of 'bestPath' (line 136)
    bestPath_349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 11), 'bestPath', False)
    # Calling bestPath(args, kwargs) (line 136)
    bestPath_call_result_355 = invoke(stypy.reporting.localization.Localization(__file__, 136, 11), bestPath_349, *[cities_350, seed_351, iter_352, boost_353], **kwargs_354)
    
    # Assigning a type to the variable 'path' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'path', bestPath_call_result_355)
    
    # Call to pathLength(...): (line 139)
    # Processing the call arguments (line 139)
    # Getting the type of 'cities' (line 139)
    cities_357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 15), 'cities', False)
    # Getting the type of 'path' (line 139)
    path_358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 23), 'path', False)
    # Processing the call keyword arguments (line 139)
    kwargs_359 = {}
    # Getting the type of 'pathLength' (line 139)
    pathLength_356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'pathLength', False)
    # Calling pathLength(args, kwargs) (line 139)
    pathLength_call_result_360 = invoke(stypy.reporting.localization.Localization(__file__, 139, 4), pathLength_356, *[cities_357, path_358], **kwargs_359)
    
    
    # ################# End of 'main(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'main' in the type store
    # Getting the type of 'stypy_return_type' (line 127)
    stypy_return_type_361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_361)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'main'
    return stypy_return_type_361

# Assigning a type to the variable 'main' (line 127)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 0), 'main', main)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 141, 0, False)
    
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

    
    # Call to main(...): (line 142)
    # Processing the call keyword arguments (line 142)
    kwargs_363 = {}
    # Getting the type of 'main' (line 142)
    main_362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'main', False)
    # Calling main(args, kwargs) (line 142)
    main_call_result_364 = invoke(stypy.reporting.localization.Localization(__file__, 142, 4), main_362, *[], **kwargs_363)
    
    # Getting the type of 'True' (line 143)
    True_365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'stypy_return_type', True_365)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 141)
    stypy_return_type_366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_366)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_366

# Assigning a type to the variable 'run' (line 141)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), 'run', run)

# Call to run(...): (line 145)
# Processing the call keyword arguments (line 145)
kwargs_368 = {}
# Getting the type of 'run' (line 145)
run_367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 0), 'run', False)
# Calling run(args, kwargs) (line 145)
run_call_result_369 = invoke(stypy.reporting.localization.Localization(__file__, 145, 0), run_367, *[], **kwargs_368)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
