
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: bidirectional dijkstra/search algorithm, mostly copied from NetworkX:
3: 
4: http://networkx.lanl.gov/
5: 
6: NetworkX is free software; you can redistribute it and/or modify it under the terms of the LGPL (GNU Lesser General Public License) as published by the Free Software Foundation; either version 2.1 of the License, or (at your option) any later version. Please see the license for more information.
7: 
8: '''
9: 
10: import heapq, time, sys, random
11: random.seed(1)
12: #print sys.version
13: 
14: class Graph:
15:     def __init__(self):
16:         self.vertices = {}
17: 
18:     def add_edge(self, a, b, weight):
19:         for id_ in (a, b):
20:             if id_ not in self.vertices:
21:                 self.vertices[id_] = Vertex(id_)
22:         va, vb = self.vertices[a], self.vertices[b]
23:         va.neighs.append((vb, weight))
24:         vb.neighs.append((va, weight))
25: 
26: class Vertex:
27:     def __init__(self, id_):
28:         self.id_ = id_
29:         self.neighs = []
30:     def __repr__(self):
31:         return repr(self.id_)
32: 
33: def bidirectional_dijkstra(G, source_id, target_id):
34:     source, target = G.vertices[source_id], G.vertices[target_id]
35:     if source == target: return (0.0, [source])
36:     #Init:   Forward             Backward
37:     dists =  [{},                {}]# dictionary of final distances
38:     paths =  [{source:[source]}, {target:[target]}] # dictionary of paths
39:     fringe = [[],                []] #heap of (distance, node) tuples for extracting next node to expand
40:     seen =   [{source:0.0},        {target:0.0} ]#dictionary of distances to nodes seen
41:     #initialize fringe heap
42:     heapq.heappush(fringe[0], (0.0, source))
43:     heapq.heappush(fringe[1], (0.0, target))
44:     #variables to hold shortest discovered path
45:     #finaldist = 1e30000
46:     finalpath = []
47:     dir = 1
48:     while fringe[0] and fringe[1]:
49:         # choose direction
50:         # dir == 0 is forward direction and dir == 1 is back
51:         dir = 1-dir
52:         # extract closest to expand
53:         (dist, v) = heapq.heappop(fringe[dir])
54:         if v in dists[dir]:
55:             # Shortest path to v has already been found
56:             continue
57:         # update distance
58:         dists[dir][v] = dist #equal to seen[dir][v]
59:         if v in dists[1-dir]:
60:             # if we have scanned v in both directions we are done
61:             # we have now discovered the shortest path
62:             return (finaldist,finalpath)
63:         for w, weight in v.neighs:
64:             vwLength = dists[dir][v] + weight
65:             if w in dists[dir]:
66:                 pass
67:             elif w not in seen[dir] or vwLength < seen[dir][w]:
68:                 # relaxing
69:                 seen[dir][w] = vwLength
70:                 heapq.heappush(fringe[dir], (vwLength,w))
71:                 paths[dir][w] = paths[dir][v]+[w]
72:                 if w in seen[0] and w in seen[1]:
73:                     #see if this path is better than than the already
74:                     #discovered shortest path
75:                     totaldist = seen[0][w] + seen[1][w]
76:                     if finalpath == [] or finaldist > totaldist:
77:                         finaldist = totaldist
78:                         revpath = paths[1][w][:]
79:                         revpath.reverse()
80:                         finalpath = paths[0][w] + revpath[1:]
81:     return None
82: 
83: def make_graph(n):
84:     G = Graph()
85:     dirs = [(-1,0), (1,0), (0,1), (0,-1)]
86:     for u in range(n):
87:         for v in range(n):
88:             for dir in dirs:
89:                 x, y = u+dir[0], v+dir[1]
90:                 if 0 <= x < n and 0 <= y < n:
91:                     G.add_edge((u,v), (x, y), random.randint(1,3))
92:     return G
93: 
94: def run():
95:     n = 300
96:     #if len(sys.argv) > 1:
97:     #    n = int(sys.argv[1])
98:     t0 = time.time()
99:     G = make_graph(n)
100:     #print 't0 %.2f' % (time.time()-t0)
101:     t1 = time.time()
102:     wt, nodes = bidirectional_dijkstra(G, (0,0), (n-1,n-1))
103:     #print 'wt', wt
104:     #print 't1 %.2f' % (time.time()-t1)
105:     return True
106: 
107: run()

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, (-1)), 'str', '\nbidirectional dijkstra/search algorithm, mostly copied from NetworkX:\n\nhttp://networkx.lanl.gov/\n\nNetworkX is free software; you can redistribute it and/or modify it under the terms of the LGPL (GNU Lesser General Public License) as published by the Free Software Foundation; either version 2.1 of the License, or (at your option) any later version. Please see the license for more information.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# Multiple import statement. import heapq (1/4) (line 10)
import heapq

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'heapq', heapq, module_type_store)
# Multiple import statement. import time (2/4) (line 10)
import time

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'time', time, module_type_store)
# Multiple import statement. import sys (3/4) (line 10)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'sys', sys, module_type_store)
# Multiple import statement. import random (4/4) (line 10)
import random

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'random', random, module_type_store)


# Call to seed(...): (line 11)
# Processing the call arguments (line 11)
int_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 12), 'int')
# Processing the call keyword arguments (line 11)
kwargs_15 = {}
# Getting the type of 'random' (line 11)
random_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'random', False)
# Obtaining the member 'seed' of a type (line 11)
seed_13 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 0), random_12, 'seed')
# Calling seed(args, kwargs) (line 11)
seed_call_result_16 = invoke(stypy.reporting.localization.Localization(__file__, 11, 0), seed_13, *[int_14], **kwargs_15)

# Declaration of the 'Graph' class

class Graph:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 15, 4, False)
        # Assigning a type to the variable 'self' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Graph.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Dict to a Attribute (line 16):
        
        # Assigning a Dict to a Attribute (line 16):
        
        # Obtaining an instance of the builtin type 'dict' (line 16)
        dict_17 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 24), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 16)
        
        # Getting the type of 'self' (line 16)
        self_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'self')
        # Setting the type of the member 'vertices' of a type (line 16)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), self_18, 'vertices', dict_17)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def add_edge(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_edge'
        module_type_store = module_type_store.open_function_context('add_edge', 18, 4, False)
        # Assigning a type to the variable 'self' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Graph.add_edge.__dict__.__setitem__('stypy_localization', localization)
        Graph.add_edge.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Graph.add_edge.__dict__.__setitem__('stypy_type_store', module_type_store)
        Graph.add_edge.__dict__.__setitem__('stypy_function_name', 'Graph.add_edge')
        Graph.add_edge.__dict__.__setitem__('stypy_param_names_list', ['a', 'b', 'weight'])
        Graph.add_edge.__dict__.__setitem__('stypy_varargs_param_name', None)
        Graph.add_edge.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Graph.add_edge.__dict__.__setitem__('stypy_call_defaults', defaults)
        Graph.add_edge.__dict__.__setitem__('stypy_call_varargs', varargs)
        Graph.add_edge.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Graph.add_edge.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Graph.add_edge', ['a', 'b', 'weight'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_edge', localization, ['a', 'b', 'weight'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_edge(...)' code ##################

        
        
        # Obtaining an instance of the builtin type 'tuple' (line 19)
        tuple_19 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 19)
        # Adding element type (line 19)
        # Getting the type of 'a' (line 19)
        a_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 20), 'a')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 20), tuple_19, a_20)
        # Adding element type (line 19)
        # Getting the type of 'b' (line 19)
        b_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 23), 'b')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 20), tuple_19, b_21)
        
        # Testing the type of a for loop iterable (line 19)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 19, 8), tuple_19)
        # Getting the type of the for loop variable (line 19)
        for_loop_var_22 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 19, 8), tuple_19)
        # Assigning a type to the variable 'id_' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'id_', for_loop_var_22)
        # SSA begins for a for statement (line 19)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'id_' (line 20)
        id__23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 15), 'id_')
        # Getting the type of 'self' (line 20)
        self_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 26), 'self')
        # Obtaining the member 'vertices' of a type (line 20)
        vertices_25 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 26), self_24, 'vertices')
        # Applying the binary operator 'notin' (line 20)
        result_contains_26 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 15), 'notin', id__23, vertices_25)
        
        # Testing the type of an if condition (line 20)
        if_condition_27 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 20, 12), result_contains_26)
        # Assigning a type to the variable 'if_condition_27' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'if_condition_27', if_condition_27)
        # SSA begins for if statement (line 20)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Subscript (line 21):
        
        # Assigning a Call to a Subscript (line 21):
        
        # Call to Vertex(...): (line 21)
        # Processing the call arguments (line 21)
        # Getting the type of 'id_' (line 21)
        id__29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 44), 'id_', False)
        # Processing the call keyword arguments (line 21)
        kwargs_30 = {}
        # Getting the type of 'Vertex' (line 21)
        Vertex_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 37), 'Vertex', False)
        # Calling Vertex(args, kwargs) (line 21)
        Vertex_call_result_31 = invoke(stypy.reporting.localization.Localization(__file__, 21, 37), Vertex_28, *[id__29], **kwargs_30)
        
        # Getting the type of 'self' (line 21)
        self_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'self')
        # Obtaining the member 'vertices' of a type (line 21)
        vertices_33 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 16), self_32, 'vertices')
        # Getting the type of 'id_' (line 21)
        id__34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 30), 'id_')
        # Storing an element on a container (line 21)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 16), vertices_33, (id__34, Vertex_call_result_31))
        # SSA join for if statement (line 20)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Tuple to a Tuple (line 22):
        
        # Assigning a Subscript to a Name (line 22):
        
        # Obtaining the type of the subscript
        # Getting the type of 'a' (line 22)
        a_35 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 31), 'a')
        # Getting the type of 'self' (line 22)
        self_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 17), 'self')
        # Obtaining the member 'vertices' of a type (line 22)
        vertices_37 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 17), self_36, 'vertices')
        # Obtaining the member '__getitem__' of a type (line 22)
        getitem___38 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 17), vertices_37, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 22)
        subscript_call_result_39 = invoke(stypy.reporting.localization.Localization(__file__, 22, 17), getitem___38, a_35)
        
        # Assigning a type to the variable 'tuple_assignment_1' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'tuple_assignment_1', subscript_call_result_39)
        
        # Assigning a Subscript to a Name (line 22):
        
        # Obtaining the type of the subscript
        # Getting the type of 'b' (line 22)
        b_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 49), 'b')
        # Getting the type of 'self' (line 22)
        self_41 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 35), 'self')
        # Obtaining the member 'vertices' of a type (line 22)
        vertices_42 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 35), self_41, 'vertices')
        # Obtaining the member '__getitem__' of a type (line 22)
        getitem___43 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 35), vertices_42, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 22)
        subscript_call_result_44 = invoke(stypy.reporting.localization.Localization(__file__, 22, 35), getitem___43, b_40)
        
        # Assigning a type to the variable 'tuple_assignment_2' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'tuple_assignment_2', subscript_call_result_44)
        
        # Assigning a Name to a Name (line 22):
        # Getting the type of 'tuple_assignment_1' (line 22)
        tuple_assignment_1_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'tuple_assignment_1')
        # Assigning a type to the variable 'va' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'va', tuple_assignment_1_45)
        
        # Assigning a Name to a Name (line 22):
        # Getting the type of 'tuple_assignment_2' (line 22)
        tuple_assignment_2_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'tuple_assignment_2')
        # Assigning a type to the variable 'vb' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'vb', tuple_assignment_2_46)
        
        # Call to append(...): (line 23)
        # Processing the call arguments (line 23)
        
        # Obtaining an instance of the builtin type 'tuple' (line 23)
        tuple_50 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 23)
        # Adding element type (line 23)
        # Getting the type of 'vb' (line 23)
        vb_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 26), 'vb', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 26), tuple_50, vb_51)
        # Adding element type (line 23)
        # Getting the type of 'weight' (line 23)
        weight_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 30), 'weight', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 26), tuple_50, weight_52)
        
        # Processing the call keyword arguments (line 23)
        kwargs_53 = {}
        # Getting the type of 'va' (line 23)
        va_47 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'va', False)
        # Obtaining the member 'neighs' of a type (line 23)
        neighs_48 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), va_47, 'neighs')
        # Obtaining the member 'append' of a type (line 23)
        append_49 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), neighs_48, 'append')
        # Calling append(args, kwargs) (line 23)
        append_call_result_54 = invoke(stypy.reporting.localization.Localization(__file__, 23, 8), append_49, *[tuple_50], **kwargs_53)
        
        
        # Call to append(...): (line 24)
        # Processing the call arguments (line 24)
        
        # Obtaining an instance of the builtin type 'tuple' (line 24)
        tuple_58 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 24)
        # Adding element type (line 24)
        # Getting the type of 'va' (line 24)
        va_59 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 26), 'va', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 26), tuple_58, va_59)
        # Adding element type (line 24)
        # Getting the type of 'weight' (line 24)
        weight_60 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 30), 'weight', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 26), tuple_58, weight_60)
        
        # Processing the call keyword arguments (line 24)
        kwargs_61 = {}
        # Getting the type of 'vb' (line 24)
        vb_55 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'vb', False)
        # Obtaining the member 'neighs' of a type (line 24)
        neighs_56 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), vb_55, 'neighs')
        # Obtaining the member 'append' of a type (line 24)
        append_57 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), neighs_56, 'append')
        # Calling append(args, kwargs) (line 24)
        append_call_result_62 = invoke(stypy.reporting.localization.Localization(__file__, 24, 8), append_57, *[tuple_58], **kwargs_61)
        
        
        # ################# End of 'add_edge(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_edge' in the type store
        # Getting the type of 'stypy_return_type' (line 18)
        stypy_return_type_63 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_63)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_edge'
        return stypy_return_type_63


# Assigning a type to the variable 'Graph' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'Graph', Graph)
# Declaration of the 'Vertex' class

class Vertex:

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 27, 4, False)
        # Assigning a type to the variable 'self' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vertex.__init__', ['id_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['id_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 28):
        
        # Assigning a Name to a Attribute (line 28):
        # Getting the type of 'id_' (line 28)
        id__64 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 19), 'id_')
        # Getting the type of 'self' (line 28)
        self_65 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'self')
        # Setting the type of the member 'id_' of a type (line 28)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), self_65, 'id_', id__64)
        
        # Assigning a List to a Attribute (line 29):
        
        # Assigning a List to a Attribute (line 29):
        
        # Obtaining an instance of the builtin type 'list' (line 29)
        list_66 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 29)
        
        # Getting the type of 'self' (line 29)
        self_67 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'self')
        # Setting the type of the member 'neighs' of a type (line 29)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), self_67, 'neighs', list_66)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 30, 4, False)
        # Assigning a type to the variable 'self' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Vertex.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        Vertex.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Vertex.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Vertex.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'Vertex.stypy__repr__')
        Vertex.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        Vertex.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Vertex.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Vertex.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Vertex.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Vertex.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Vertex.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vertex.stypy__repr__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__repr__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__repr__(...)' code ##################

        
        # Call to repr(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'self' (line 31)
        self_69 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 20), 'self', False)
        # Obtaining the member 'id_' of a type (line 31)
        id__70 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 20), self_69, 'id_')
        # Processing the call keyword arguments (line 31)
        kwargs_71 = {}
        # Getting the type of 'repr' (line 31)
        repr_68 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 15), 'repr', False)
        # Calling repr(args, kwargs) (line 31)
        repr_call_result_72 = invoke(stypy.reporting.localization.Localization(__file__, 31, 15), repr_68, *[id__70], **kwargs_71)
        
        # Assigning a type to the variable 'stypy_return_type' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'stypy_return_type', repr_call_result_72)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 30)
        stypy_return_type_73 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_73)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_73


# Assigning a type to the variable 'Vertex' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'Vertex', Vertex)

@norecursion
def bidirectional_dijkstra(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'bidirectional_dijkstra'
    module_type_store = module_type_store.open_function_context('bidirectional_dijkstra', 33, 0, False)
    
    # Passed parameters checking function
    bidirectional_dijkstra.stypy_localization = localization
    bidirectional_dijkstra.stypy_type_of_self = None
    bidirectional_dijkstra.stypy_type_store = module_type_store
    bidirectional_dijkstra.stypy_function_name = 'bidirectional_dijkstra'
    bidirectional_dijkstra.stypy_param_names_list = ['G', 'source_id', 'target_id']
    bidirectional_dijkstra.stypy_varargs_param_name = None
    bidirectional_dijkstra.stypy_kwargs_param_name = None
    bidirectional_dijkstra.stypy_call_defaults = defaults
    bidirectional_dijkstra.stypy_call_varargs = varargs
    bidirectional_dijkstra.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'bidirectional_dijkstra', ['G', 'source_id', 'target_id'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'bidirectional_dijkstra', localization, ['G', 'source_id', 'target_id'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'bidirectional_dijkstra(...)' code ##################

    
    # Assigning a Tuple to a Tuple (line 34):
    
    # Assigning a Subscript to a Name (line 34):
    
    # Obtaining the type of the subscript
    # Getting the type of 'source_id' (line 34)
    source_id_74 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 32), 'source_id')
    # Getting the type of 'G' (line 34)
    G_75 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 21), 'G')
    # Obtaining the member 'vertices' of a type (line 34)
    vertices_76 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 21), G_75, 'vertices')
    # Obtaining the member '__getitem__' of a type (line 34)
    getitem___77 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 21), vertices_76, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 34)
    subscript_call_result_78 = invoke(stypy.reporting.localization.Localization(__file__, 34, 21), getitem___77, source_id_74)
    
    # Assigning a type to the variable 'tuple_assignment_3' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'tuple_assignment_3', subscript_call_result_78)
    
    # Assigning a Subscript to a Name (line 34):
    
    # Obtaining the type of the subscript
    # Getting the type of 'target_id' (line 34)
    target_id_79 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 55), 'target_id')
    # Getting the type of 'G' (line 34)
    G_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 44), 'G')
    # Obtaining the member 'vertices' of a type (line 34)
    vertices_81 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 44), G_80, 'vertices')
    # Obtaining the member '__getitem__' of a type (line 34)
    getitem___82 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 44), vertices_81, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 34)
    subscript_call_result_83 = invoke(stypy.reporting.localization.Localization(__file__, 34, 44), getitem___82, target_id_79)
    
    # Assigning a type to the variable 'tuple_assignment_4' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'tuple_assignment_4', subscript_call_result_83)
    
    # Assigning a Name to a Name (line 34):
    # Getting the type of 'tuple_assignment_3' (line 34)
    tuple_assignment_3_84 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'tuple_assignment_3')
    # Assigning a type to the variable 'source' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'source', tuple_assignment_3_84)
    
    # Assigning a Name to a Name (line 34):
    # Getting the type of 'tuple_assignment_4' (line 34)
    tuple_assignment_4_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'tuple_assignment_4')
    # Assigning a type to the variable 'target' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'target', tuple_assignment_4_85)
    
    
    # Getting the type of 'source' (line 35)
    source_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 7), 'source')
    # Getting the type of 'target' (line 35)
    target_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 17), 'target')
    # Applying the binary operator '==' (line 35)
    result_eq_88 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 7), '==', source_86, target_87)
    
    # Testing the type of an if condition (line 35)
    if_condition_89 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 35, 4), result_eq_88)
    # Assigning a type to the variable 'if_condition_89' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'if_condition_89', if_condition_89)
    # SSA begins for if statement (line 35)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 35)
    tuple_90 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 33), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 35)
    # Adding element type (line 35)
    float_91 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 33), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 33), tuple_90, float_91)
    # Adding element type (line 35)
    
    # Obtaining an instance of the builtin type 'list' (line 35)
    list_92 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 38), 'list')
    # Adding type elements to the builtin type 'list' instance (line 35)
    # Adding element type (line 35)
    # Getting the type of 'source' (line 35)
    source_93 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 39), 'source')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 38), list_92, source_93)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 33), tuple_90, list_92)
    
    # Assigning a type to the variable 'stypy_return_type' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 25), 'stypy_return_type', tuple_90)
    # SSA join for if statement (line 35)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 37):
    
    # Assigning a List to a Name (line 37):
    
    # Obtaining an instance of the builtin type 'list' (line 37)
    list_94 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 37)
    # Adding element type (line 37)
    
    # Obtaining an instance of the builtin type 'dict' (line 37)
    dict_95 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 14), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 37)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 13), list_94, dict_95)
    # Adding element type (line 37)
    
    # Obtaining an instance of the builtin type 'dict' (line 37)
    dict_96 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 33), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 37)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 13), list_94, dict_96)
    
    # Assigning a type to the variable 'dists' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'dists', list_94)
    
    # Assigning a List to a Name (line 38):
    
    # Assigning a List to a Name (line 38):
    
    # Obtaining an instance of the builtin type 'list' (line 38)
    list_97 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 38)
    # Adding element type (line 38)
    
    # Obtaining an instance of the builtin type 'dict' (line 38)
    dict_98 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 14), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 38)
    # Adding element type (key, value) (line 38)
    # Getting the type of 'source' (line 38)
    source_99 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 15), 'source')
    
    # Obtaining an instance of the builtin type 'list' (line 38)
    list_100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 38)
    # Adding element type (line 38)
    # Getting the type of 'source' (line 38)
    source_101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 23), 'source')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 22), list_100, source_101)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 14), dict_98, (source_99, list_100))
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 13), list_97, dict_98)
    # Adding element type (line 38)
    
    # Obtaining an instance of the builtin type 'dict' (line 38)
    dict_102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 33), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 38)
    # Adding element type (key, value) (line 38)
    # Getting the type of 'target' (line 38)
    target_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 34), 'target')
    
    # Obtaining an instance of the builtin type 'list' (line 38)
    list_104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 41), 'list')
    # Adding type elements to the builtin type 'list' instance (line 38)
    # Adding element type (line 38)
    # Getting the type of 'target' (line 38)
    target_105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 42), 'target')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 41), list_104, target_105)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 33), dict_102, (target_103, list_104))
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 13), list_97, dict_102)
    
    # Assigning a type to the variable 'paths' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'paths', list_97)
    
    # Assigning a List to a Name (line 39):
    
    # Assigning a List to a Name (line 39):
    
    # Obtaining an instance of the builtin type 'list' (line 39)
    list_106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 39)
    # Adding element type (line 39)
    
    # Obtaining an instance of the builtin type 'list' (line 39)
    list_107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 39)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 13), list_106, list_107)
    # Adding element type (line 39)
    
    # Obtaining an instance of the builtin type 'list' (line 39)
    list_108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 39)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 13), list_106, list_108)
    
    # Assigning a type to the variable 'fringe' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'fringe', list_106)
    
    # Assigning a List to a Name (line 40):
    
    # Assigning a List to a Name (line 40):
    
    # Obtaining an instance of the builtin type 'list' (line 40)
    list_109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 40)
    # Adding element type (line 40)
    
    # Obtaining an instance of the builtin type 'dict' (line 40)
    dict_110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 14), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 40)
    # Adding element type (key, value) (line 40)
    # Getting the type of 'source' (line 40)
    source_111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 15), 'source')
    float_112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 22), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 14), dict_110, (source_111, float_112))
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 13), list_109, dict_110)
    # Adding element type (line 40)
    
    # Obtaining an instance of the builtin type 'dict' (line 40)
    dict_113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 35), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 40)
    # Adding element type (key, value) (line 40)
    # Getting the type of 'target' (line 40)
    target_114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 36), 'target')
    float_115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 43), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 35), dict_113, (target_114, float_115))
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 13), list_109, dict_113)
    
    # Assigning a type to the variable 'seen' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'seen', list_109)
    
    # Call to heappush(...): (line 42)
    # Processing the call arguments (line 42)
    
    # Obtaining the type of the subscript
    int_118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 26), 'int')
    # Getting the type of 'fringe' (line 42)
    fringe_119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 19), 'fringe', False)
    # Obtaining the member '__getitem__' of a type (line 42)
    getitem___120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 19), fringe_119, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
    subscript_call_result_121 = invoke(stypy.reporting.localization.Localization(__file__, 42, 19), getitem___120, int_118)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 42)
    tuple_122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 42)
    # Adding element type (line 42)
    float_123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 31), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 31), tuple_122, float_123)
    # Adding element type (line 42)
    # Getting the type of 'source' (line 42)
    source_124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 36), 'source', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 31), tuple_122, source_124)
    
    # Processing the call keyword arguments (line 42)
    kwargs_125 = {}
    # Getting the type of 'heapq' (line 42)
    heapq_116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'heapq', False)
    # Obtaining the member 'heappush' of a type (line 42)
    heappush_117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 4), heapq_116, 'heappush')
    # Calling heappush(args, kwargs) (line 42)
    heappush_call_result_126 = invoke(stypy.reporting.localization.Localization(__file__, 42, 4), heappush_117, *[subscript_call_result_121, tuple_122], **kwargs_125)
    
    
    # Call to heappush(...): (line 43)
    # Processing the call arguments (line 43)
    
    # Obtaining the type of the subscript
    int_129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 26), 'int')
    # Getting the type of 'fringe' (line 43)
    fringe_130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 19), 'fringe', False)
    # Obtaining the member '__getitem__' of a type (line 43)
    getitem___131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 19), fringe_130, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 43)
    subscript_call_result_132 = invoke(stypy.reporting.localization.Localization(__file__, 43, 19), getitem___131, int_129)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 43)
    tuple_133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 43)
    # Adding element type (line 43)
    float_134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 31), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 31), tuple_133, float_134)
    # Adding element type (line 43)
    # Getting the type of 'target' (line 43)
    target_135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 36), 'target', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 31), tuple_133, target_135)
    
    # Processing the call keyword arguments (line 43)
    kwargs_136 = {}
    # Getting the type of 'heapq' (line 43)
    heapq_127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'heapq', False)
    # Obtaining the member 'heappush' of a type (line 43)
    heappush_128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 4), heapq_127, 'heappush')
    # Calling heappush(args, kwargs) (line 43)
    heappush_call_result_137 = invoke(stypy.reporting.localization.Localization(__file__, 43, 4), heappush_128, *[subscript_call_result_132, tuple_133], **kwargs_136)
    
    
    # Assigning a List to a Name (line 46):
    
    # Assigning a List to a Name (line 46):
    
    # Obtaining an instance of the builtin type 'list' (line 46)
    list_138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 46)
    
    # Assigning a type to the variable 'finalpath' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'finalpath', list_138)
    
    # Assigning a Num to a Name (line 47):
    
    # Assigning a Num to a Name (line 47):
    int_139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 10), 'int')
    # Assigning a type to the variable 'dir' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'dir', int_139)
    
    
    # Evaluating a boolean operation
    
    # Obtaining the type of the subscript
    int_140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 17), 'int')
    # Getting the type of 'fringe' (line 48)
    fringe_141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 10), 'fringe')
    # Obtaining the member '__getitem__' of a type (line 48)
    getitem___142 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 10), fringe_141, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 48)
    subscript_call_result_143 = invoke(stypy.reporting.localization.Localization(__file__, 48, 10), getitem___142, int_140)
    
    
    # Obtaining the type of the subscript
    int_144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 31), 'int')
    # Getting the type of 'fringe' (line 48)
    fringe_145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 24), 'fringe')
    # Obtaining the member '__getitem__' of a type (line 48)
    getitem___146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 24), fringe_145, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 48)
    subscript_call_result_147 = invoke(stypy.reporting.localization.Localization(__file__, 48, 24), getitem___146, int_144)
    
    # Applying the binary operator 'and' (line 48)
    result_and_keyword_148 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 10), 'and', subscript_call_result_143, subscript_call_result_147)
    
    # Testing the type of an if condition (line 48)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 48, 4), result_and_keyword_148)
    # SSA begins for while statement (line 48)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a BinOp to a Name (line 51):
    
    # Assigning a BinOp to a Name (line 51):
    int_149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 14), 'int')
    # Getting the type of 'dir' (line 51)
    dir_150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'dir')
    # Applying the binary operator '-' (line 51)
    result_sub_151 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 14), '-', int_149, dir_150)
    
    # Assigning a type to the variable 'dir' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'dir', result_sub_151)
    
    # Assigning a Call to a Tuple (line 53):
    
    # Assigning a Subscript to a Name (line 53):
    
    # Obtaining the type of the subscript
    int_152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 8), 'int')
    
    # Call to heappop(...): (line 53)
    # Processing the call arguments (line 53)
    
    # Obtaining the type of the subscript
    # Getting the type of 'dir' (line 53)
    dir_155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 41), 'dir', False)
    # Getting the type of 'fringe' (line 53)
    fringe_156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 34), 'fringe', False)
    # Obtaining the member '__getitem__' of a type (line 53)
    getitem___157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 34), fringe_156, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 53)
    subscript_call_result_158 = invoke(stypy.reporting.localization.Localization(__file__, 53, 34), getitem___157, dir_155)
    
    # Processing the call keyword arguments (line 53)
    kwargs_159 = {}
    # Getting the type of 'heapq' (line 53)
    heapq_153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 20), 'heapq', False)
    # Obtaining the member 'heappop' of a type (line 53)
    heappop_154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 20), heapq_153, 'heappop')
    # Calling heappop(args, kwargs) (line 53)
    heappop_call_result_160 = invoke(stypy.reporting.localization.Localization(__file__, 53, 20), heappop_154, *[subscript_call_result_158], **kwargs_159)
    
    # Obtaining the member '__getitem__' of a type (line 53)
    getitem___161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), heappop_call_result_160, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 53)
    subscript_call_result_162 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), getitem___161, int_152)
    
    # Assigning a type to the variable 'tuple_var_assignment_5' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'tuple_var_assignment_5', subscript_call_result_162)
    
    # Assigning a Subscript to a Name (line 53):
    
    # Obtaining the type of the subscript
    int_163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 8), 'int')
    
    # Call to heappop(...): (line 53)
    # Processing the call arguments (line 53)
    
    # Obtaining the type of the subscript
    # Getting the type of 'dir' (line 53)
    dir_166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 41), 'dir', False)
    # Getting the type of 'fringe' (line 53)
    fringe_167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 34), 'fringe', False)
    # Obtaining the member '__getitem__' of a type (line 53)
    getitem___168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 34), fringe_167, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 53)
    subscript_call_result_169 = invoke(stypy.reporting.localization.Localization(__file__, 53, 34), getitem___168, dir_166)
    
    # Processing the call keyword arguments (line 53)
    kwargs_170 = {}
    # Getting the type of 'heapq' (line 53)
    heapq_164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 20), 'heapq', False)
    # Obtaining the member 'heappop' of a type (line 53)
    heappop_165 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 20), heapq_164, 'heappop')
    # Calling heappop(args, kwargs) (line 53)
    heappop_call_result_171 = invoke(stypy.reporting.localization.Localization(__file__, 53, 20), heappop_165, *[subscript_call_result_169], **kwargs_170)
    
    # Obtaining the member '__getitem__' of a type (line 53)
    getitem___172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), heappop_call_result_171, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 53)
    subscript_call_result_173 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), getitem___172, int_163)
    
    # Assigning a type to the variable 'tuple_var_assignment_6' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'tuple_var_assignment_6', subscript_call_result_173)
    
    # Assigning a Name to a Name (line 53):
    # Getting the type of 'tuple_var_assignment_5' (line 53)
    tuple_var_assignment_5_174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'tuple_var_assignment_5')
    # Assigning a type to the variable 'dist' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 9), 'dist', tuple_var_assignment_5_174)
    
    # Assigning a Name to a Name (line 53):
    # Getting the type of 'tuple_var_assignment_6' (line 53)
    tuple_var_assignment_6_175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'tuple_var_assignment_6')
    # Assigning a type to the variable 'v' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 15), 'v', tuple_var_assignment_6_175)
    
    
    # Getting the type of 'v' (line 54)
    v_176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 11), 'v')
    
    # Obtaining the type of the subscript
    # Getting the type of 'dir' (line 54)
    dir_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 22), 'dir')
    # Getting the type of 'dists' (line 54)
    dists_178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'dists')
    # Obtaining the member '__getitem__' of a type (line 54)
    getitem___179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 16), dists_178, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 54)
    subscript_call_result_180 = invoke(stypy.reporting.localization.Localization(__file__, 54, 16), getitem___179, dir_177)
    
    # Applying the binary operator 'in' (line 54)
    result_contains_181 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 11), 'in', v_176, subscript_call_result_180)
    
    # Testing the type of an if condition (line 54)
    if_condition_182 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 54, 8), result_contains_181)
    # Assigning a type to the variable 'if_condition_182' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'if_condition_182', if_condition_182)
    # SSA begins for if statement (line 54)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 54)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Subscript (line 58):
    
    # Assigning a Name to a Subscript (line 58):
    # Getting the type of 'dist' (line 58)
    dist_183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 24), 'dist')
    
    # Obtaining the type of the subscript
    # Getting the type of 'dir' (line 58)
    dir_184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 14), 'dir')
    # Getting the type of 'dists' (line 58)
    dists_185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'dists')
    # Obtaining the member '__getitem__' of a type (line 58)
    getitem___186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), dists_185, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 58)
    subscript_call_result_187 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), getitem___186, dir_184)
    
    # Getting the type of 'v' (line 58)
    v_188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 19), 'v')
    # Storing an element on a container (line 58)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 8), subscript_call_result_187, (v_188, dist_183))
    
    
    # Getting the type of 'v' (line 59)
    v_189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 11), 'v')
    
    # Obtaining the type of the subscript
    int_190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 22), 'int')
    # Getting the type of 'dir' (line 59)
    dir_191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 24), 'dir')
    # Applying the binary operator '-' (line 59)
    result_sub_192 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 22), '-', int_190, dir_191)
    
    # Getting the type of 'dists' (line 59)
    dists_193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'dists')
    # Obtaining the member '__getitem__' of a type (line 59)
    getitem___194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 16), dists_193, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 59)
    subscript_call_result_195 = invoke(stypy.reporting.localization.Localization(__file__, 59, 16), getitem___194, result_sub_192)
    
    # Applying the binary operator 'in' (line 59)
    result_contains_196 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 11), 'in', v_189, subscript_call_result_195)
    
    # Testing the type of an if condition (line 59)
    if_condition_197 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 59, 8), result_contains_196)
    # Assigning a type to the variable 'if_condition_197' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'if_condition_197', if_condition_197)
    # SSA begins for if statement (line 59)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 62)
    tuple_198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 62)
    # Adding element type (line 62)
    # Getting the type of 'finaldist' (line 62)
    finaldist_199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 20), 'finaldist')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 20), tuple_198, finaldist_199)
    # Adding element type (line 62)
    # Getting the type of 'finalpath' (line 62)
    finalpath_200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 30), 'finalpath')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 20), tuple_198, finalpath_200)
    
    # Assigning a type to the variable 'stypy_return_type' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'stypy_return_type', tuple_198)
    # SSA join for if statement (line 59)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'v' (line 63)
    v_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 25), 'v')
    # Obtaining the member 'neighs' of a type (line 63)
    neighs_202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 25), v_201, 'neighs')
    # Testing the type of a for loop iterable (line 63)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 63, 8), neighs_202)
    # Getting the type of the for loop variable (line 63)
    for_loop_var_203 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 63, 8), neighs_202)
    # Assigning a type to the variable 'w' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'w', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 8), for_loop_var_203))
    # Assigning a type to the variable 'weight' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'weight', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 8), for_loop_var_203))
    # SSA begins for a for statement (line 63)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a BinOp to a Name (line 64):
    
    # Assigning a BinOp to a Name (line 64):
    
    # Obtaining the type of the subscript
    # Getting the type of 'v' (line 64)
    v_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 34), 'v')
    
    # Obtaining the type of the subscript
    # Getting the type of 'dir' (line 64)
    dir_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 29), 'dir')
    # Getting the type of 'dists' (line 64)
    dists_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 23), 'dists')
    # Obtaining the member '__getitem__' of a type (line 64)
    getitem___207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 23), dists_206, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 64)
    subscript_call_result_208 = invoke(stypy.reporting.localization.Localization(__file__, 64, 23), getitem___207, dir_205)
    
    # Obtaining the member '__getitem__' of a type (line 64)
    getitem___209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 23), subscript_call_result_208, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 64)
    subscript_call_result_210 = invoke(stypy.reporting.localization.Localization(__file__, 64, 23), getitem___209, v_204)
    
    # Getting the type of 'weight' (line 64)
    weight_211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 39), 'weight')
    # Applying the binary operator '+' (line 64)
    result_add_212 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 23), '+', subscript_call_result_210, weight_211)
    
    # Assigning a type to the variable 'vwLength' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'vwLength', result_add_212)
    
    
    # Getting the type of 'w' (line 65)
    w_213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 15), 'w')
    
    # Obtaining the type of the subscript
    # Getting the type of 'dir' (line 65)
    dir_214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 26), 'dir')
    # Getting the type of 'dists' (line 65)
    dists_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 20), 'dists')
    # Obtaining the member '__getitem__' of a type (line 65)
    getitem___216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 20), dists_215, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 65)
    subscript_call_result_217 = invoke(stypy.reporting.localization.Localization(__file__, 65, 20), getitem___216, dir_214)
    
    # Applying the binary operator 'in' (line 65)
    result_contains_218 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 15), 'in', w_213, subscript_call_result_217)
    
    # Testing the type of an if condition (line 65)
    if_condition_219 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 65, 12), result_contains_218)
    # Assigning a type to the variable 'if_condition_219' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'if_condition_219', if_condition_219)
    # SSA begins for if statement (line 65)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    pass
    # SSA branch for the else part of an if statement (line 65)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'w' (line 67)
    w_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 17), 'w')
    
    # Obtaining the type of the subscript
    # Getting the type of 'dir' (line 67)
    dir_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 31), 'dir')
    # Getting the type of 'seen' (line 67)
    seen_222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 26), 'seen')
    # Obtaining the member '__getitem__' of a type (line 67)
    getitem___223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 26), seen_222, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 67)
    subscript_call_result_224 = invoke(stypy.reporting.localization.Localization(__file__, 67, 26), getitem___223, dir_221)
    
    # Applying the binary operator 'notin' (line 67)
    result_contains_225 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 17), 'notin', w_220, subscript_call_result_224)
    
    
    # Getting the type of 'vwLength' (line 67)
    vwLength_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 39), 'vwLength')
    
    # Obtaining the type of the subscript
    # Getting the type of 'w' (line 67)
    w_227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 60), 'w')
    
    # Obtaining the type of the subscript
    # Getting the type of 'dir' (line 67)
    dir_228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 55), 'dir')
    # Getting the type of 'seen' (line 67)
    seen_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 50), 'seen')
    # Obtaining the member '__getitem__' of a type (line 67)
    getitem___230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 50), seen_229, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 67)
    subscript_call_result_231 = invoke(stypy.reporting.localization.Localization(__file__, 67, 50), getitem___230, dir_228)
    
    # Obtaining the member '__getitem__' of a type (line 67)
    getitem___232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 50), subscript_call_result_231, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 67)
    subscript_call_result_233 = invoke(stypy.reporting.localization.Localization(__file__, 67, 50), getitem___232, w_227)
    
    # Applying the binary operator '<' (line 67)
    result_lt_234 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 39), '<', vwLength_226, subscript_call_result_233)
    
    # Applying the binary operator 'or' (line 67)
    result_or_keyword_235 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 17), 'or', result_contains_225, result_lt_234)
    
    # Testing the type of an if condition (line 67)
    if_condition_236 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 17), result_or_keyword_235)
    # Assigning a type to the variable 'if_condition_236' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 17), 'if_condition_236', if_condition_236)
    # SSA begins for if statement (line 67)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 69):
    
    # Assigning a Name to a Subscript (line 69):
    # Getting the type of 'vwLength' (line 69)
    vwLength_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 31), 'vwLength')
    
    # Obtaining the type of the subscript
    # Getting the type of 'dir' (line 69)
    dir_238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 21), 'dir')
    # Getting the type of 'seen' (line 69)
    seen_239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 16), 'seen')
    # Obtaining the member '__getitem__' of a type (line 69)
    getitem___240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 16), seen_239, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 69)
    subscript_call_result_241 = invoke(stypy.reporting.localization.Localization(__file__, 69, 16), getitem___240, dir_238)
    
    # Getting the type of 'w' (line 69)
    w_242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 26), 'w')
    # Storing an element on a container (line 69)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 16), subscript_call_result_241, (w_242, vwLength_237))
    
    # Call to heappush(...): (line 70)
    # Processing the call arguments (line 70)
    
    # Obtaining the type of the subscript
    # Getting the type of 'dir' (line 70)
    dir_245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 38), 'dir', False)
    # Getting the type of 'fringe' (line 70)
    fringe_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 31), 'fringe', False)
    # Obtaining the member '__getitem__' of a type (line 70)
    getitem___247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 31), fringe_246, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 70)
    subscript_call_result_248 = invoke(stypy.reporting.localization.Localization(__file__, 70, 31), getitem___247, dir_245)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 70)
    tuple_249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 45), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 70)
    # Adding element type (line 70)
    # Getting the type of 'vwLength' (line 70)
    vwLength_250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 45), 'vwLength', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 45), tuple_249, vwLength_250)
    # Adding element type (line 70)
    # Getting the type of 'w' (line 70)
    w_251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 54), 'w', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 45), tuple_249, w_251)
    
    # Processing the call keyword arguments (line 70)
    kwargs_252 = {}
    # Getting the type of 'heapq' (line 70)
    heapq_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'heapq', False)
    # Obtaining the member 'heappush' of a type (line 70)
    heappush_244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 16), heapq_243, 'heappush')
    # Calling heappush(args, kwargs) (line 70)
    heappush_call_result_253 = invoke(stypy.reporting.localization.Localization(__file__, 70, 16), heappush_244, *[subscript_call_result_248, tuple_249], **kwargs_252)
    
    
    # Assigning a BinOp to a Subscript (line 71):
    
    # Assigning a BinOp to a Subscript (line 71):
    
    # Obtaining the type of the subscript
    # Getting the type of 'v' (line 71)
    v_254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 43), 'v')
    
    # Obtaining the type of the subscript
    # Getting the type of 'dir' (line 71)
    dir_255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 38), 'dir')
    # Getting the type of 'paths' (line 71)
    paths_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 32), 'paths')
    # Obtaining the member '__getitem__' of a type (line 71)
    getitem___257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 32), paths_256, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 71)
    subscript_call_result_258 = invoke(stypy.reporting.localization.Localization(__file__, 71, 32), getitem___257, dir_255)
    
    # Obtaining the member '__getitem__' of a type (line 71)
    getitem___259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 32), subscript_call_result_258, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 71)
    subscript_call_result_260 = invoke(stypy.reporting.localization.Localization(__file__, 71, 32), getitem___259, v_254)
    
    
    # Obtaining an instance of the builtin type 'list' (line 71)
    list_261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 46), 'list')
    # Adding type elements to the builtin type 'list' instance (line 71)
    # Adding element type (line 71)
    # Getting the type of 'w' (line 71)
    w_262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 47), 'w')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 46), list_261, w_262)
    
    # Applying the binary operator '+' (line 71)
    result_add_263 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 32), '+', subscript_call_result_260, list_261)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'dir' (line 71)
    dir_264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 22), 'dir')
    # Getting the type of 'paths' (line 71)
    paths_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 16), 'paths')
    # Obtaining the member '__getitem__' of a type (line 71)
    getitem___266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 16), paths_265, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 71)
    subscript_call_result_267 = invoke(stypy.reporting.localization.Localization(__file__, 71, 16), getitem___266, dir_264)
    
    # Getting the type of 'w' (line 71)
    w_268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 27), 'w')
    # Storing an element on a container (line 71)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 16), subscript_call_result_267, (w_268, result_add_263))
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'w' (line 72)
    w_269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 19), 'w')
    
    # Obtaining the type of the subscript
    int_270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 29), 'int')
    # Getting the type of 'seen' (line 72)
    seen_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 24), 'seen')
    # Obtaining the member '__getitem__' of a type (line 72)
    getitem___272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 24), seen_271, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 72)
    subscript_call_result_273 = invoke(stypy.reporting.localization.Localization(__file__, 72, 24), getitem___272, int_270)
    
    # Applying the binary operator 'in' (line 72)
    result_contains_274 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 19), 'in', w_269, subscript_call_result_273)
    
    
    # Getting the type of 'w' (line 72)
    w_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 36), 'w')
    
    # Obtaining the type of the subscript
    int_276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 46), 'int')
    # Getting the type of 'seen' (line 72)
    seen_277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 41), 'seen')
    # Obtaining the member '__getitem__' of a type (line 72)
    getitem___278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 41), seen_277, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 72)
    subscript_call_result_279 = invoke(stypy.reporting.localization.Localization(__file__, 72, 41), getitem___278, int_276)
    
    # Applying the binary operator 'in' (line 72)
    result_contains_280 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 36), 'in', w_275, subscript_call_result_279)
    
    # Applying the binary operator 'and' (line 72)
    result_and_keyword_281 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 19), 'and', result_contains_274, result_contains_280)
    
    # Testing the type of an if condition (line 72)
    if_condition_282 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 16), result_and_keyword_281)
    # Assigning a type to the variable 'if_condition_282' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 16), 'if_condition_282', if_condition_282)
    # SSA begins for if statement (line 72)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 75):
    
    # Assigning a BinOp to a Name (line 75):
    
    # Obtaining the type of the subscript
    # Getting the type of 'w' (line 75)
    w_283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 40), 'w')
    
    # Obtaining the type of the subscript
    int_284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 37), 'int')
    # Getting the type of 'seen' (line 75)
    seen_285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 32), 'seen')
    # Obtaining the member '__getitem__' of a type (line 75)
    getitem___286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 32), seen_285, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 75)
    subscript_call_result_287 = invoke(stypy.reporting.localization.Localization(__file__, 75, 32), getitem___286, int_284)
    
    # Obtaining the member '__getitem__' of a type (line 75)
    getitem___288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 32), subscript_call_result_287, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 75)
    subscript_call_result_289 = invoke(stypy.reporting.localization.Localization(__file__, 75, 32), getitem___288, w_283)
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'w' (line 75)
    w_290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 53), 'w')
    
    # Obtaining the type of the subscript
    int_291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 50), 'int')
    # Getting the type of 'seen' (line 75)
    seen_292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 45), 'seen')
    # Obtaining the member '__getitem__' of a type (line 75)
    getitem___293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 45), seen_292, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 75)
    subscript_call_result_294 = invoke(stypy.reporting.localization.Localization(__file__, 75, 45), getitem___293, int_291)
    
    # Obtaining the member '__getitem__' of a type (line 75)
    getitem___295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 45), subscript_call_result_294, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 75)
    subscript_call_result_296 = invoke(stypy.reporting.localization.Localization(__file__, 75, 45), getitem___295, w_290)
    
    # Applying the binary operator '+' (line 75)
    result_add_297 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 32), '+', subscript_call_result_289, subscript_call_result_296)
    
    # Assigning a type to the variable 'totaldist' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 20), 'totaldist', result_add_297)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'finalpath' (line 76)
    finalpath_298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 23), 'finalpath')
    
    # Obtaining an instance of the builtin type 'list' (line 76)
    list_299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 36), 'list')
    # Adding type elements to the builtin type 'list' instance (line 76)
    
    # Applying the binary operator '==' (line 76)
    result_eq_300 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 23), '==', finalpath_298, list_299)
    
    
    # Getting the type of 'finaldist' (line 76)
    finaldist_301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 42), 'finaldist')
    # Getting the type of 'totaldist' (line 76)
    totaldist_302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 54), 'totaldist')
    # Applying the binary operator '>' (line 76)
    result_gt_303 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 42), '>', finaldist_301, totaldist_302)
    
    # Applying the binary operator 'or' (line 76)
    result_or_keyword_304 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 23), 'or', result_eq_300, result_gt_303)
    
    # Testing the type of an if condition (line 76)
    if_condition_305 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 20), result_or_keyword_304)
    # Assigning a type to the variable 'if_condition_305' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 20), 'if_condition_305', if_condition_305)
    # SSA begins for if statement (line 76)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 77):
    
    # Assigning a Name to a Name (line 77):
    # Getting the type of 'totaldist' (line 77)
    totaldist_306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 36), 'totaldist')
    # Assigning a type to the variable 'finaldist' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 24), 'finaldist', totaldist_306)
    
    # Assigning a Subscript to a Name (line 78):
    
    # Assigning a Subscript to a Name (line 78):
    
    # Obtaining the type of the subscript
    slice_307 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 78, 34), None, None, None)
    
    # Obtaining the type of the subscript
    # Getting the type of 'w' (line 78)
    w_308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 43), 'w')
    
    # Obtaining the type of the subscript
    int_309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 40), 'int')
    # Getting the type of 'paths' (line 78)
    paths_310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 34), 'paths')
    # Obtaining the member '__getitem__' of a type (line 78)
    getitem___311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 34), paths_310, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 78)
    subscript_call_result_312 = invoke(stypy.reporting.localization.Localization(__file__, 78, 34), getitem___311, int_309)
    
    # Obtaining the member '__getitem__' of a type (line 78)
    getitem___313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 34), subscript_call_result_312, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 78)
    subscript_call_result_314 = invoke(stypy.reporting.localization.Localization(__file__, 78, 34), getitem___313, w_308)
    
    # Obtaining the member '__getitem__' of a type (line 78)
    getitem___315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 34), subscript_call_result_314, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 78)
    subscript_call_result_316 = invoke(stypy.reporting.localization.Localization(__file__, 78, 34), getitem___315, slice_307)
    
    # Assigning a type to the variable 'revpath' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 24), 'revpath', subscript_call_result_316)
    
    # Call to reverse(...): (line 79)
    # Processing the call keyword arguments (line 79)
    kwargs_319 = {}
    # Getting the type of 'revpath' (line 79)
    revpath_317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 24), 'revpath', False)
    # Obtaining the member 'reverse' of a type (line 79)
    reverse_318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 24), revpath_317, 'reverse')
    # Calling reverse(args, kwargs) (line 79)
    reverse_call_result_320 = invoke(stypy.reporting.localization.Localization(__file__, 79, 24), reverse_318, *[], **kwargs_319)
    
    
    # Assigning a BinOp to a Name (line 80):
    
    # Assigning a BinOp to a Name (line 80):
    
    # Obtaining the type of the subscript
    # Getting the type of 'w' (line 80)
    w_321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 45), 'w')
    
    # Obtaining the type of the subscript
    int_322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 42), 'int')
    # Getting the type of 'paths' (line 80)
    paths_323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 36), 'paths')
    # Obtaining the member '__getitem__' of a type (line 80)
    getitem___324 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 36), paths_323, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 80)
    subscript_call_result_325 = invoke(stypy.reporting.localization.Localization(__file__, 80, 36), getitem___324, int_322)
    
    # Obtaining the member '__getitem__' of a type (line 80)
    getitem___326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 36), subscript_call_result_325, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 80)
    subscript_call_result_327 = invoke(stypy.reporting.localization.Localization(__file__, 80, 36), getitem___326, w_321)
    
    
    # Obtaining the type of the subscript
    int_328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 58), 'int')
    slice_329 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 80, 50), int_328, None, None)
    # Getting the type of 'revpath' (line 80)
    revpath_330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 50), 'revpath')
    # Obtaining the member '__getitem__' of a type (line 80)
    getitem___331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 50), revpath_330, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 80)
    subscript_call_result_332 = invoke(stypy.reporting.localization.Localization(__file__, 80, 50), getitem___331, slice_329)
    
    # Applying the binary operator '+' (line 80)
    result_add_333 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 36), '+', subscript_call_result_327, subscript_call_result_332)
    
    # Assigning a type to the variable 'finalpath' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 24), 'finalpath', result_add_333)
    # SSA join for if statement (line 76)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 72)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 67)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 65)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 48)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'None' (line 81)
    None_334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 11), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy_return_type', None_334)
    
    # ################# End of 'bidirectional_dijkstra(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'bidirectional_dijkstra' in the type store
    # Getting the type of 'stypy_return_type' (line 33)
    stypy_return_type_335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_335)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'bidirectional_dijkstra'
    return stypy_return_type_335

# Assigning a type to the variable 'bidirectional_dijkstra' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'bidirectional_dijkstra', bidirectional_dijkstra)

@norecursion
def make_graph(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'make_graph'
    module_type_store = module_type_store.open_function_context('make_graph', 83, 0, False)
    
    # Passed parameters checking function
    make_graph.stypy_localization = localization
    make_graph.stypy_type_of_self = None
    make_graph.stypy_type_store = module_type_store
    make_graph.stypy_function_name = 'make_graph'
    make_graph.stypy_param_names_list = ['n']
    make_graph.stypy_varargs_param_name = None
    make_graph.stypy_kwargs_param_name = None
    make_graph.stypy_call_defaults = defaults
    make_graph.stypy_call_varargs = varargs
    make_graph.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'make_graph', ['n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'make_graph', localization, ['n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'make_graph(...)' code ##################

    
    # Assigning a Call to a Name (line 84):
    
    # Assigning a Call to a Name (line 84):
    
    # Call to Graph(...): (line 84)
    # Processing the call keyword arguments (line 84)
    kwargs_337 = {}
    # Getting the type of 'Graph' (line 84)
    Graph_336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'Graph', False)
    # Calling Graph(args, kwargs) (line 84)
    Graph_call_result_338 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), Graph_336, *[], **kwargs_337)
    
    # Assigning a type to the variable 'G' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'G', Graph_call_result_338)
    
    # Assigning a List to a Name (line 85):
    
    # Assigning a List to a Name (line 85):
    
    # Obtaining an instance of the builtin type 'list' (line 85)
    list_339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 85)
    # Adding element type (line 85)
    
    # Obtaining an instance of the builtin type 'tuple' (line 85)
    tuple_340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 85)
    # Adding element type (line 85)
    int_341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 13), tuple_340, int_341)
    # Adding element type (line 85)
    int_342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 13), tuple_340, int_342)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 11), list_339, tuple_340)
    # Adding element type (line 85)
    
    # Obtaining an instance of the builtin type 'tuple' (line 85)
    tuple_343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 85)
    # Adding element type (line 85)
    int_344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 21), tuple_343, int_344)
    # Adding element type (line 85)
    int_345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 21), tuple_343, int_345)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 11), list_339, tuple_343)
    # Adding element type (line 85)
    
    # Obtaining an instance of the builtin type 'tuple' (line 85)
    tuple_346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 85)
    # Adding element type (line 85)
    int_347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 28), tuple_346, int_347)
    # Adding element type (line 85)
    int_348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 28), tuple_346, int_348)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 11), list_339, tuple_346)
    # Adding element type (line 85)
    
    # Obtaining an instance of the builtin type 'tuple' (line 85)
    tuple_349 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 85)
    # Adding element type (line 85)
    int_350 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 35), tuple_349, int_350)
    # Adding element type (line 85)
    int_351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 35), tuple_349, int_351)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 11), list_339, tuple_349)
    
    # Assigning a type to the variable 'dirs' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'dirs', list_339)
    
    
    # Call to range(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'n' (line 86)
    n_353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 19), 'n', False)
    # Processing the call keyword arguments (line 86)
    kwargs_354 = {}
    # Getting the type of 'range' (line 86)
    range_352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 13), 'range', False)
    # Calling range(args, kwargs) (line 86)
    range_call_result_355 = invoke(stypy.reporting.localization.Localization(__file__, 86, 13), range_352, *[n_353], **kwargs_354)
    
    # Testing the type of a for loop iterable (line 86)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 86, 4), range_call_result_355)
    # Getting the type of the for loop variable (line 86)
    for_loop_var_356 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 86, 4), range_call_result_355)
    # Assigning a type to the variable 'u' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'u', for_loop_var_356)
    # SSA begins for a for statement (line 86)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to range(...): (line 87)
    # Processing the call arguments (line 87)
    # Getting the type of 'n' (line 87)
    n_358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 23), 'n', False)
    # Processing the call keyword arguments (line 87)
    kwargs_359 = {}
    # Getting the type of 'range' (line 87)
    range_357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 17), 'range', False)
    # Calling range(args, kwargs) (line 87)
    range_call_result_360 = invoke(stypy.reporting.localization.Localization(__file__, 87, 17), range_357, *[n_358], **kwargs_359)
    
    # Testing the type of a for loop iterable (line 87)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 87, 8), range_call_result_360)
    # Getting the type of the for loop variable (line 87)
    for_loop_var_361 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 87, 8), range_call_result_360)
    # Assigning a type to the variable 'v' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'v', for_loop_var_361)
    # SSA begins for a for statement (line 87)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'dirs' (line 88)
    dirs_362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 23), 'dirs')
    # Testing the type of a for loop iterable (line 88)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 88, 12), dirs_362)
    # Getting the type of the for loop variable (line 88)
    for_loop_var_363 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 88, 12), dirs_362)
    # Assigning a type to the variable 'dir' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'dir', for_loop_var_363)
    # SSA begins for a for statement (line 88)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Tuple to a Tuple (line 89):
    
    # Assigning a BinOp to a Name (line 89):
    # Getting the type of 'u' (line 89)
    u_364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 23), 'u')
    
    # Obtaining the type of the subscript
    int_365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 29), 'int')
    # Getting the type of 'dir' (line 89)
    dir_366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 25), 'dir')
    # Obtaining the member '__getitem__' of a type (line 89)
    getitem___367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 25), dir_366, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 89)
    subscript_call_result_368 = invoke(stypy.reporting.localization.Localization(__file__, 89, 25), getitem___367, int_365)
    
    # Applying the binary operator '+' (line 89)
    result_add_369 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 23), '+', u_364, subscript_call_result_368)
    
    # Assigning a type to the variable 'tuple_assignment_7' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 16), 'tuple_assignment_7', result_add_369)
    
    # Assigning a BinOp to a Name (line 89):
    # Getting the type of 'v' (line 89)
    v_370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 33), 'v')
    
    # Obtaining the type of the subscript
    int_371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 39), 'int')
    # Getting the type of 'dir' (line 89)
    dir_372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 35), 'dir')
    # Obtaining the member '__getitem__' of a type (line 89)
    getitem___373 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 35), dir_372, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 89)
    subscript_call_result_374 = invoke(stypy.reporting.localization.Localization(__file__, 89, 35), getitem___373, int_371)
    
    # Applying the binary operator '+' (line 89)
    result_add_375 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 33), '+', v_370, subscript_call_result_374)
    
    # Assigning a type to the variable 'tuple_assignment_8' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 16), 'tuple_assignment_8', result_add_375)
    
    # Assigning a Name to a Name (line 89):
    # Getting the type of 'tuple_assignment_7' (line 89)
    tuple_assignment_7_376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 16), 'tuple_assignment_7')
    # Assigning a type to the variable 'x' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 16), 'x', tuple_assignment_7_376)
    
    # Assigning a Name to a Name (line 89):
    # Getting the type of 'tuple_assignment_8' (line 89)
    tuple_assignment_8_377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 16), 'tuple_assignment_8')
    # Assigning a type to the variable 'y' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 19), 'y', tuple_assignment_8_377)
    
    
    # Evaluating a boolean operation
    
    int_378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 19), 'int')
    # Getting the type of 'x' (line 90)
    x_379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 24), 'x')
    # Applying the binary operator '<=' (line 90)
    result_le_380 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 19), '<=', int_378, x_379)
    # Getting the type of 'n' (line 90)
    n_381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 28), 'n')
    # Applying the binary operator '<' (line 90)
    result_lt_382 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 19), '<', x_379, n_381)
    # Applying the binary operator '&' (line 90)
    result_and__383 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 19), '&', result_le_380, result_lt_382)
    
    
    int_384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 34), 'int')
    # Getting the type of 'y' (line 90)
    y_385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 39), 'y')
    # Applying the binary operator '<=' (line 90)
    result_le_386 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 34), '<=', int_384, y_385)
    # Getting the type of 'n' (line 90)
    n_387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 43), 'n')
    # Applying the binary operator '<' (line 90)
    result_lt_388 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 34), '<', y_385, n_387)
    # Applying the binary operator '&' (line 90)
    result_and__389 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 34), '&', result_le_386, result_lt_388)
    
    # Applying the binary operator 'and' (line 90)
    result_and_keyword_390 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 19), 'and', result_and__383, result_and__389)
    
    # Testing the type of an if condition (line 90)
    if_condition_391 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 90, 16), result_and_keyword_390)
    # Assigning a type to the variable 'if_condition_391' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 16), 'if_condition_391', if_condition_391)
    # SSA begins for if statement (line 90)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to add_edge(...): (line 91)
    # Processing the call arguments (line 91)
    
    # Obtaining an instance of the builtin type 'tuple' (line 91)
    tuple_394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 91)
    # Adding element type (line 91)
    # Getting the type of 'u' (line 91)
    u_395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 32), 'u', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 32), tuple_394, u_395)
    # Adding element type (line 91)
    # Getting the type of 'v' (line 91)
    v_396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 34), 'v', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 32), tuple_394, v_396)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 91)
    tuple_397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 39), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 91)
    # Adding element type (line 91)
    # Getting the type of 'x' (line 91)
    x_398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 39), 'x', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 39), tuple_397, x_398)
    # Adding element type (line 91)
    # Getting the type of 'y' (line 91)
    y_399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 42), 'y', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 39), tuple_397, y_399)
    
    
    # Call to randint(...): (line 91)
    # Processing the call arguments (line 91)
    int_402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 61), 'int')
    int_403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 63), 'int')
    # Processing the call keyword arguments (line 91)
    kwargs_404 = {}
    # Getting the type of 'random' (line 91)
    random_400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 46), 'random', False)
    # Obtaining the member 'randint' of a type (line 91)
    randint_401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 46), random_400, 'randint')
    # Calling randint(args, kwargs) (line 91)
    randint_call_result_405 = invoke(stypy.reporting.localization.Localization(__file__, 91, 46), randint_401, *[int_402, int_403], **kwargs_404)
    
    # Processing the call keyword arguments (line 91)
    kwargs_406 = {}
    # Getting the type of 'G' (line 91)
    G_392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'G', False)
    # Obtaining the member 'add_edge' of a type (line 91)
    add_edge_393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 20), G_392, 'add_edge')
    # Calling add_edge(args, kwargs) (line 91)
    add_edge_call_result_407 = invoke(stypy.reporting.localization.Localization(__file__, 91, 20), add_edge_393, *[tuple_394, tuple_397, randint_call_result_405], **kwargs_406)
    
    # SSA join for if statement (line 90)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'G' (line 92)
    G_408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 11), 'G')
    # Assigning a type to the variable 'stypy_return_type' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'stypy_return_type', G_408)
    
    # ################# End of 'make_graph(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'make_graph' in the type store
    # Getting the type of 'stypy_return_type' (line 83)
    stypy_return_type_409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_409)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'make_graph'
    return stypy_return_type_409

# Assigning a type to the variable 'make_graph' (line 83)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 0), 'make_graph', make_graph)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 94, 0, False)
    
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

    
    # Assigning a Num to a Name (line 95):
    
    # Assigning a Num to a Name (line 95):
    int_410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 8), 'int')
    # Assigning a type to the variable 'n' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'n', int_410)
    
    # Assigning a Call to a Name (line 98):
    
    # Assigning a Call to a Name (line 98):
    
    # Call to time(...): (line 98)
    # Processing the call keyword arguments (line 98)
    kwargs_413 = {}
    # Getting the type of 'time' (line 98)
    time_411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 9), 'time', False)
    # Obtaining the member 'time' of a type (line 98)
    time_412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 9), time_411, 'time')
    # Calling time(args, kwargs) (line 98)
    time_call_result_414 = invoke(stypy.reporting.localization.Localization(__file__, 98, 9), time_412, *[], **kwargs_413)
    
    # Assigning a type to the variable 't0' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 't0', time_call_result_414)
    
    # Assigning a Call to a Name (line 99):
    
    # Assigning a Call to a Name (line 99):
    
    # Call to make_graph(...): (line 99)
    # Processing the call arguments (line 99)
    # Getting the type of 'n' (line 99)
    n_416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 19), 'n', False)
    # Processing the call keyword arguments (line 99)
    kwargs_417 = {}
    # Getting the type of 'make_graph' (line 99)
    make_graph_415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'make_graph', False)
    # Calling make_graph(args, kwargs) (line 99)
    make_graph_call_result_418 = invoke(stypy.reporting.localization.Localization(__file__, 99, 8), make_graph_415, *[n_416], **kwargs_417)
    
    # Assigning a type to the variable 'G' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'G', make_graph_call_result_418)
    
    # Assigning a Call to a Name (line 101):
    
    # Assigning a Call to a Name (line 101):
    
    # Call to time(...): (line 101)
    # Processing the call keyword arguments (line 101)
    kwargs_421 = {}
    # Getting the type of 'time' (line 101)
    time_419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 9), 'time', False)
    # Obtaining the member 'time' of a type (line 101)
    time_420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 9), time_419, 'time')
    # Calling time(args, kwargs) (line 101)
    time_call_result_422 = invoke(stypy.reporting.localization.Localization(__file__, 101, 9), time_420, *[], **kwargs_421)
    
    # Assigning a type to the variable 't1' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 't1', time_call_result_422)
    
    # Assigning a Call to a Tuple (line 102):
    
    # Assigning a Subscript to a Name (line 102):
    
    # Obtaining the type of the subscript
    int_423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 4), 'int')
    
    # Call to bidirectional_dijkstra(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 'G' (line 102)
    G_425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 39), 'G', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 102)
    tuple_426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 102)
    # Adding element type (line 102)
    int_427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 43), tuple_426, int_427)
    # Adding element type (line 102)
    int_428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 43), tuple_426, int_428)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 102)
    tuple_429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 102)
    # Adding element type (line 102)
    # Getting the type of 'n' (line 102)
    n_430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 50), 'n', False)
    int_431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 52), 'int')
    # Applying the binary operator '-' (line 102)
    result_sub_432 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 50), '-', n_430, int_431)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 50), tuple_429, result_sub_432)
    # Adding element type (line 102)
    # Getting the type of 'n' (line 102)
    n_433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 54), 'n', False)
    int_434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 56), 'int')
    # Applying the binary operator '-' (line 102)
    result_sub_435 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 54), '-', n_433, int_434)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 50), tuple_429, result_sub_435)
    
    # Processing the call keyword arguments (line 102)
    kwargs_436 = {}
    # Getting the type of 'bidirectional_dijkstra' (line 102)
    bidirectional_dijkstra_424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'bidirectional_dijkstra', False)
    # Calling bidirectional_dijkstra(args, kwargs) (line 102)
    bidirectional_dijkstra_call_result_437 = invoke(stypy.reporting.localization.Localization(__file__, 102, 16), bidirectional_dijkstra_424, *[G_425, tuple_426, tuple_429], **kwargs_436)
    
    # Obtaining the member '__getitem__' of a type (line 102)
    getitem___438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 4), bidirectional_dijkstra_call_result_437, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 102)
    subscript_call_result_439 = invoke(stypy.reporting.localization.Localization(__file__, 102, 4), getitem___438, int_423)
    
    # Assigning a type to the variable 'tuple_var_assignment_9' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'tuple_var_assignment_9', subscript_call_result_439)
    
    # Assigning a Subscript to a Name (line 102):
    
    # Obtaining the type of the subscript
    int_440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 4), 'int')
    
    # Call to bidirectional_dijkstra(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 'G' (line 102)
    G_442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 39), 'G', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 102)
    tuple_443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 102)
    # Adding element type (line 102)
    int_444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 43), tuple_443, int_444)
    # Adding element type (line 102)
    int_445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 43), tuple_443, int_445)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 102)
    tuple_446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 102)
    # Adding element type (line 102)
    # Getting the type of 'n' (line 102)
    n_447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 50), 'n', False)
    int_448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 52), 'int')
    # Applying the binary operator '-' (line 102)
    result_sub_449 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 50), '-', n_447, int_448)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 50), tuple_446, result_sub_449)
    # Adding element type (line 102)
    # Getting the type of 'n' (line 102)
    n_450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 54), 'n', False)
    int_451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 56), 'int')
    # Applying the binary operator '-' (line 102)
    result_sub_452 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 54), '-', n_450, int_451)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 50), tuple_446, result_sub_452)
    
    # Processing the call keyword arguments (line 102)
    kwargs_453 = {}
    # Getting the type of 'bidirectional_dijkstra' (line 102)
    bidirectional_dijkstra_441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'bidirectional_dijkstra', False)
    # Calling bidirectional_dijkstra(args, kwargs) (line 102)
    bidirectional_dijkstra_call_result_454 = invoke(stypy.reporting.localization.Localization(__file__, 102, 16), bidirectional_dijkstra_441, *[G_442, tuple_443, tuple_446], **kwargs_453)
    
    # Obtaining the member '__getitem__' of a type (line 102)
    getitem___455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 4), bidirectional_dijkstra_call_result_454, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 102)
    subscript_call_result_456 = invoke(stypy.reporting.localization.Localization(__file__, 102, 4), getitem___455, int_440)
    
    # Assigning a type to the variable 'tuple_var_assignment_10' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'tuple_var_assignment_10', subscript_call_result_456)
    
    # Assigning a Name to a Name (line 102):
    # Getting the type of 'tuple_var_assignment_9' (line 102)
    tuple_var_assignment_9_457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'tuple_var_assignment_9')
    # Assigning a type to the variable 'wt' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'wt', tuple_var_assignment_9_457)
    
    # Assigning a Name to a Name (line 102):
    # Getting the type of 'tuple_var_assignment_10' (line 102)
    tuple_var_assignment_10_458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'tuple_var_assignment_10')
    # Assigning a type to the variable 'nodes' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'nodes', tuple_var_assignment_10_458)
    # Getting the type of 'True' (line 105)
    True_459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type', True_459)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 94)
    stypy_return_type_460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_460)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_460

# Assigning a type to the variable 'run' (line 94)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'run', run)

# Call to run(...): (line 107)
# Processing the call keyword arguments (line 107)
kwargs_462 = {}
# Getting the type of 'run' (line 107)
run_461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 0), 'run', False)
# Calling run(args, kwargs) (line 107)
run_call_result_463 = invoke(stypy.reporting.localization.Localization(__file__, 107, 0), run_461, *[], **kwargs_462)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
