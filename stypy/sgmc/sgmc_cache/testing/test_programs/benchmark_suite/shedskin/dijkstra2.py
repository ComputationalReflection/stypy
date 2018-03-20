
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

str_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, (-1)), 'str', '\nbidirectional dijkstra/search algorithm, mostly copied from NetworkX:\n\nhttp://networkx.lanl.gov/\n\nNetworkX is free software; you can redistribute it and/or modify it under the terms of the LGPL (GNU Lesser General Public License) as published by the Free Software Foundation; either version 2.1 of the License, or (at your option) any later version. Please see the license for more information.\n\n')
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
int_16 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 12), 'int')
# Processing the call keyword arguments (line 11)
kwargs_17 = {}
# Getting the type of 'random' (line 11)
random_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'random', False)
# Obtaining the member 'seed' of a type (line 11)
seed_15 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 0), random_14, 'seed')
# Calling seed(args, kwargs) (line 11)
seed_call_result_18 = invoke(stypy.reporting.localization.Localization(__file__, 11, 0), seed_15, *[int_16], **kwargs_17)

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
        dict_19 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 24), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 16)
        
        # Getting the type of 'self' (line 16)
        self_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'self')
        # Setting the type of the member 'vertices' of a type (line 16)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), self_20, 'vertices', dict_19)
        
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
        tuple_21 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 20), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 19)
        # Adding element type (line 19)
        # Getting the type of 'a' (line 19)
        a_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 20), 'a')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 20), tuple_21, a_22)
        # Adding element type (line 19)
        # Getting the type of 'b' (line 19)
        b_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 23), 'b')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 20), tuple_21, b_23)
        
        # Assigning a type to the variable 'tuple_21' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'tuple_21', tuple_21)
        # Testing if the for loop is going to be iterated (line 19)
        # Testing the type of a for loop iterable (line 19)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 19, 8), tuple_21)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 19, 8), tuple_21):
            # Getting the type of the for loop variable (line 19)
            for_loop_var_24 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 19, 8), tuple_21)
            # Assigning a type to the variable 'id_' (line 19)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'id_', for_loop_var_24)
            # SSA begins for a for statement (line 19)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'id_' (line 20)
            id__25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 15), 'id_')
            # Getting the type of 'self' (line 20)
            self_26 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 26), 'self')
            # Obtaining the member 'vertices' of a type (line 20)
            vertices_27 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 26), self_26, 'vertices')
            # Applying the binary operator 'notin' (line 20)
            result_contains_28 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 15), 'notin', id__25, vertices_27)
            
            # Testing if the type of an if condition is none (line 20)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 20, 12), result_contains_28):
                pass
            else:
                
                # Testing the type of an if condition (line 20)
                if_condition_29 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 20, 12), result_contains_28)
                # Assigning a type to the variable 'if_condition_29' (line 20)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'if_condition_29', if_condition_29)
                # SSA begins for if statement (line 20)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Subscript (line 21):
                
                # Assigning a Call to a Subscript (line 21):
                
                # Call to Vertex(...): (line 21)
                # Processing the call arguments (line 21)
                # Getting the type of 'id_' (line 21)
                id__31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 44), 'id_', False)
                # Processing the call keyword arguments (line 21)
                kwargs_32 = {}
                # Getting the type of 'Vertex' (line 21)
                Vertex_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 37), 'Vertex', False)
                # Calling Vertex(args, kwargs) (line 21)
                Vertex_call_result_33 = invoke(stypy.reporting.localization.Localization(__file__, 21, 37), Vertex_30, *[id__31], **kwargs_32)
                
                # Getting the type of 'self' (line 21)
                self_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'self')
                # Obtaining the member 'vertices' of a type (line 21)
                vertices_35 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 16), self_34, 'vertices')
                # Getting the type of 'id_' (line 21)
                id__36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 30), 'id_')
                # Storing an element on a container (line 21)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 16), vertices_35, (id__36, Vertex_call_result_33))
                # SSA join for if statement (line 20)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Tuple to a Tuple (line 22):
        
        # Assigning a Subscript to a Name (line 22):
        
        # Obtaining the type of the subscript
        # Getting the type of 'a' (line 22)
        a_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 31), 'a')
        # Getting the type of 'self' (line 22)
        self_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 17), 'self')
        # Obtaining the member 'vertices' of a type (line 22)
        vertices_39 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 17), self_38, 'vertices')
        # Obtaining the member '__getitem__' of a type (line 22)
        getitem___40 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 17), vertices_39, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 22)
        subscript_call_result_41 = invoke(stypy.reporting.localization.Localization(__file__, 22, 17), getitem___40, a_37)
        
        # Assigning a type to the variable 'tuple_assignment_1' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'tuple_assignment_1', subscript_call_result_41)
        
        # Assigning a Subscript to a Name (line 22):
        
        # Obtaining the type of the subscript
        # Getting the type of 'b' (line 22)
        b_42 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 49), 'b')
        # Getting the type of 'self' (line 22)
        self_43 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 35), 'self')
        # Obtaining the member 'vertices' of a type (line 22)
        vertices_44 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 35), self_43, 'vertices')
        # Obtaining the member '__getitem__' of a type (line 22)
        getitem___45 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 35), vertices_44, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 22)
        subscript_call_result_46 = invoke(stypy.reporting.localization.Localization(__file__, 22, 35), getitem___45, b_42)
        
        # Assigning a type to the variable 'tuple_assignment_2' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'tuple_assignment_2', subscript_call_result_46)
        
        # Assigning a Name to a Name (line 22):
        # Getting the type of 'tuple_assignment_1' (line 22)
        tuple_assignment_1_47 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'tuple_assignment_1')
        # Assigning a type to the variable 'va' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'va', tuple_assignment_1_47)
        
        # Assigning a Name to a Name (line 22):
        # Getting the type of 'tuple_assignment_2' (line 22)
        tuple_assignment_2_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'tuple_assignment_2')
        # Assigning a type to the variable 'vb' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 12), 'vb', tuple_assignment_2_48)
        
        # Call to append(...): (line 23)
        # Processing the call arguments (line 23)
        
        # Obtaining an instance of the builtin type 'tuple' (line 23)
        tuple_52 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 23)
        # Adding element type (line 23)
        # Getting the type of 'vb' (line 23)
        vb_53 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 26), 'vb', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 26), tuple_52, vb_53)
        # Adding element type (line 23)
        # Getting the type of 'weight' (line 23)
        weight_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 30), 'weight', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 26), tuple_52, weight_54)
        
        # Processing the call keyword arguments (line 23)
        kwargs_55 = {}
        # Getting the type of 'va' (line 23)
        va_49 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'va', False)
        # Obtaining the member 'neighs' of a type (line 23)
        neighs_50 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), va_49, 'neighs')
        # Obtaining the member 'append' of a type (line 23)
        append_51 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), neighs_50, 'append')
        # Calling append(args, kwargs) (line 23)
        append_call_result_56 = invoke(stypy.reporting.localization.Localization(__file__, 23, 8), append_51, *[tuple_52], **kwargs_55)
        
        
        # Call to append(...): (line 24)
        # Processing the call arguments (line 24)
        
        # Obtaining an instance of the builtin type 'tuple' (line 24)
        tuple_60 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 24)
        # Adding element type (line 24)
        # Getting the type of 'va' (line 24)
        va_61 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 26), 'va', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 26), tuple_60, va_61)
        # Adding element type (line 24)
        # Getting the type of 'weight' (line 24)
        weight_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 30), 'weight', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 26), tuple_60, weight_62)
        
        # Processing the call keyword arguments (line 24)
        kwargs_63 = {}
        # Getting the type of 'vb' (line 24)
        vb_57 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'vb', False)
        # Obtaining the member 'neighs' of a type (line 24)
        neighs_58 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), vb_57, 'neighs')
        # Obtaining the member 'append' of a type (line 24)
        append_59 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), neighs_58, 'append')
        # Calling append(args, kwargs) (line 24)
        append_call_result_64 = invoke(stypy.reporting.localization.Localization(__file__, 24, 8), append_59, *[tuple_60], **kwargs_63)
        
        
        # ################# End of 'add_edge(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_edge' in the type store
        # Getting the type of 'stypy_return_type' (line 18)
        stypy_return_type_65 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_65)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_edge'
        return stypy_return_type_65


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
        id__66 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 19), 'id_')
        # Getting the type of 'self' (line 28)
        self_67 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'self')
        # Setting the type of the member 'id_' of a type (line 28)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), self_67, 'id_', id__66)
        
        # Assigning a List to a Attribute (line 29):
        
        # Assigning a List to a Attribute (line 29):
        
        # Obtaining an instance of the builtin type 'list' (line 29)
        list_68 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 29)
        
        # Getting the type of 'self' (line 29)
        self_69 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'self')
        # Setting the type of the member 'neighs' of a type (line 29)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), self_69, 'neighs', list_68)
        
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
        self_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 20), 'self', False)
        # Obtaining the member 'id_' of a type (line 31)
        id__72 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 20), self_71, 'id_')
        # Processing the call keyword arguments (line 31)
        kwargs_73 = {}
        # Getting the type of 'repr' (line 31)
        repr_70 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 15), 'repr', False)
        # Calling repr(args, kwargs) (line 31)
        repr_call_result_74 = invoke(stypy.reporting.localization.Localization(__file__, 31, 15), repr_70, *[id__72], **kwargs_73)
        
        # Assigning a type to the variable 'stypy_return_type' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'stypy_return_type', repr_call_result_74)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 30)
        stypy_return_type_75 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_75)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_75


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
    source_id_76 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 32), 'source_id')
    # Getting the type of 'G' (line 34)
    G_77 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 21), 'G')
    # Obtaining the member 'vertices' of a type (line 34)
    vertices_78 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 21), G_77, 'vertices')
    # Obtaining the member '__getitem__' of a type (line 34)
    getitem___79 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 21), vertices_78, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 34)
    subscript_call_result_80 = invoke(stypy.reporting.localization.Localization(__file__, 34, 21), getitem___79, source_id_76)
    
    # Assigning a type to the variable 'tuple_assignment_3' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'tuple_assignment_3', subscript_call_result_80)
    
    # Assigning a Subscript to a Name (line 34):
    
    # Obtaining the type of the subscript
    # Getting the type of 'target_id' (line 34)
    target_id_81 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 55), 'target_id')
    # Getting the type of 'G' (line 34)
    G_82 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 44), 'G')
    # Obtaining the member 'vertices' of a type (line 34)
    vertices_83 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 44), G_82, 'vertices')
    # Obtaining the member '__getitem__' of a type (line 34)
    getitem___84 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 44), vertices_83, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 34)
    subscript_call_result_85 = invoke(stypy.reporting.localization.Localization(__file__, 34, 44), getitem___84, target_id_81)
    
    # Assigning a type to the variable 'tuple_assignment_4' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'tuple_assignment_4', subscript_call_result_85)
    
    # Assigning a Name to a Name (line 34):
    # Getting the type of 'tuple_assignment_3' (line 34)
    tuple_assignment_3_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'tuple_assignment_3')
    # Assigning a type to the variable 'source' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'source', tuple_assignment_3_86)
    
    # Assigning a Name to a Name (line 34):
    # Getting the type of 'tuple_assignment_4' (line 34)
    tuple_assignment_4_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'tuple_assignment_4')
    # Assigning a type to the variable 'target' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'target', tuple_assignment_4_87)
    
    # Getting the type of 'source' (line 35)
    source_88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 7), 'source')
    # Getting the type of 'target' (line 35)
    target_89 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 17), 'target')
    # Applying the binary operator '==' (line 35)
    result_eq_90 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 7), '==', source_88, target_89)
    
    # Testing if the type of an if condition is none (line 35)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 35, 4), result_eq_90):
        pass
    else:
        
        # Testing the type of an if condition (line 35)
        if_condition_91 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 35, 4), result_eq_90)
        # Assigning a type to the variable 'if_condition_91' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'if_condition_91', if_condition_91)
        # SSA begins for if statement (line 35)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Obtaining an instance of the builtin type 'tuple' (line 35)
        tuple_92 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 33), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 35)
        # Adding element type (line 35)
        float_93 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 33), 'float')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 33), tuple_92, float_93)
        # Adding element type (line 35)
        
        # Obtaining an instance of the builtin type 'list' (line 35)
        list_94 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 35)
        # Adding element type (line 35)
        # Getting the type of 'source' (line 35)
        source_95 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 39), 'source')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 38), list_94, source_95)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 33), tuple_92, list_94)
        
        # Assigning a type to the variable 'stypy_return_type' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 25), 'stypy_return_type', tuple_92)
        # SSA join for if statement (line 35)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a List to a Name (line 37):
    
    # Assigning a List to a Name (line 37):
    
    # Obtaining an instance of the builtin type 'list' (line 37)
    list_96 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 37)
    # Adding element type (line 37)
    
    # Obtaining an instance of the builtin type 'dict' (line 37)
    dict_97 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 14), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 37)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 13), list_96, dict_97)
    # Adding element type (line 37)
    
    # Obtaining an instance of the builtin type 'dict' (line 37)
    dict_98 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 33), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 37)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 13), list_96, dict_98)
    
    # Assigning a type to the variable 'dists' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'dists', list_96)
    
    # Assigning a List to a Name (line 38):
    
    # Assigning a List to a Name (line 38):
    
    # Obtaining an instance of the builtin type 'list' (line 38)
    list_99 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 38)
    # Adding element type (line 38)
    
    # Obtaining an instance of the builtin type 'dict' (line 38)
    dict_100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 14), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 38)
    # Adding element type (key, value) (line 38)
    # Getting the type of 'source' (line 38)
    source_101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 15), 'source')
    
    # Obtaining an instance of the builtin type 'list' (line 38)
    list_102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 38)
    # Adding element type (line 38)
    # Getting the type of 'source' (line 38)
    source_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 23), 'source')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 22), list_102, source_103)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 14), dict_100, (source_101, list_102))
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 13), list_99, dict_100)
    # Adding element type (line 38)
    
    # Obtaining an instance of the builtin type 'dict' (line 38)
    dict_104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 33), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 38)
    # Adding element type (key, value) (line 38)
    # Getting the type of 'target' (line 38)
    target_105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 34), 'target')
    
    # Obtaining an instance of the builtin type 'list' (line 38)
    list_106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 41), 'list')
    # Adding type elements to the builtin type 'list' instance (line 38)
    # Adding element type (line 38)
    # Getting the type of 'target' (line 38)
    target_107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 42), 'target')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 41), list_106, target_107)
    
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 33), dict_104, (target_105, list_106))
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 13), list_99, dict_104)
    
    # Assigning a type to the variable 'paths' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'paths', list_99)
    
    # Assigning a List to a Name (line 39):
    
    # Assigning a List to a Name (line 39):
    
    # Obtaining an instance of the builtin type 'list' (line 39)
    list_108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 39)
    # Adding element type (line 39)
    
    # Obtaining an instance of the builtin type 'list' (line 39)
    list_109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 39)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 13), list_108, list_109)
    # Adding element type (line 39)
    
    # Obtaining an instance of the builtin type 'list' (line 39)
    list_110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 33), 'list')
    # Adding type elements to the builtin type 'list' instance (line 39)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 13), list_108, list_110)
    
    # Assigning a type to the variable 'fringe' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'fringe', list_108)
    
    # Assigning a List to a Name (line 40):
    
    # Assigning a List to a Name (line 40):
    
    # Obtaining an instance of the builtin type 'list' (line 40)
    list_111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 40)
    # Adding element type (line 40)
    
    # Obtaining an instance of the builtin type 'dict' (line 40)
    dict_112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 14), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 40)
    # Adding element type (key, value) (line 40)
    # Getting the type of 'source' (line 40)
    source_113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 15), 'source')
    float_114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 22), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 14), dict_112, (source_113, float_114))
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 13), list_111, dict_112)
    # Adding element type (line 40)
    
    # Obtaining an instance of the builtin type 'dict' (line 40)
    dict_115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 35), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 40)
    # Adding element type (key, value) (line 40)
    # Getting the type of 'target' (line 40)
    target_116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 36), 'target')
    float_117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 43), 'float')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 35), dict_115, (target_116, float_117))
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 13), list_111, dict_115)
    
    # Assigning a type to the variable 'seen' (line 40)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'seen', list_111)
    
    # Call to heappush(...): (line 42)
    # Processing the call arguments (line 42)
    
    # Obtaining the type of the subscript
    int_120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 26), 'int')
    # Getting the type of 'fringe' (line 42)
    fringe_121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 19), 'fringe', False)
    # Obtaining the member '__getitem__' of a type (line 42)
    getitem___122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 19), fringe_121, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 42)
    subscript_call_result_123 = invoke(stypy.reporting.localization.Localization(__file__, 42, 19), getitem___122, int_120)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 42)
    tuple_124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 42)
    # Adding element type (line 42)
    float_125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 31), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 31), tuple_124, float_125)
    # Adding element type (line 42)
    # Getting the type of 'source' (line 42)
    source_126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 36), 'source', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 31), tuple_124, source_126)
    
    # Processing the call keyword arguments (line 42)
    kwargs_127 = {}
    # Getting the type of 'heapq' (line 42)
    heapq_118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'heapq', False)
    # Obtaining the member 'heappush' of a type (line 42)
    heappush_119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 4), heapq_118, 'heappush')
    # Calling heappush(args, kwargs) (line 42)
    heappush_call_result_128 = invoke(stypy.reporting.localization.Localization(__file__, 42, 4), heappush_119, *[subscript_call_result_123, tuple_124], **kwargs_127)
    
    
    # Call to heappush(...): (line 43)
    # Processing the call arguments (line 43)
    
    # Obtaining the type of the subscript
    int_131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 26), 'int')
    # Getting the type of 'fringe' (line 43)
    fringe_132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 19), 'fringe', False)
    # Obtaining the member '__getitem__' of a type (line 43)
    getitem___133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 19), fringe_132, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 43)
    subscript_call_result_134 = invoke(stypy.reporting.localization.Localization(__file__, 43, 19), getitem___133, int_131)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 43)
    tuple_135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 31), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 43)
    # Adding element type (line 43)
    float_136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 31), 'float')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 31), tuple_135, float_136)
    # Adding element type (line 43)
    # Getting the type of 'target' (line 43)
    target_137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 36), 'target', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 31), tuple_135, target_137)
    
    # Processing the call keyword arguments (line 43)
    kwargs_138 = {}
    # Getting the type of 'heapq' (line 43)
    heapq_129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'heapq', False)
    # Obtaining the member 'heappush' of a type (line 43)
    heappush_130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 4), heapq_129, 'heappush')
    # Calling heappush(args, kwargs) (line 43)
    heappush_call_result_139 = invoke(stypy.reporting.localization.Localization(__file__, 43, 4), heappush_130, *[subscript_call_result_134, tuple_135], **kwargs_138)
    
    
    # Assigning a List to a Name (line 46):
    
    # Assigning a List to a Name (line 46):
    
    # Obtaining an instance of the builtin type 'list' (line 46)
    list_140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 46)
    
    # Assigning a type to the variable 'finalpath' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'finalpath', list_140)
    
    # Assigning a Num to a Name (line 47):
    
    # Assigning a Num to a Name (line 47):
    int_141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 10), 'int')
    # Assigning a type to the variable 'dir' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'dir', int_141)
    
    
    # Evaluating a boolean operation
    
    # Obtaining the type of the subscript
    int_142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 17), 'int')
    # Getting the type of 'fringe' (line 48)
    fringe_143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 10), 'fringe')
    # Obtaining the member '__getitem__' of a type (line 48)
    getitem___144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 10), fringe_143, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 48)
    subscript_call_result_145 = invoke(stypy.reporting.localization.Localization(__file__, 48, 10), getitem___144, int_142)
    
    
    # Obtaining the type of the subscript
    int_146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 31), 'int')
    # Getting the type of 'fringe' (line 48)
    fringe_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 24), 'fringe')
    # Obtaining the member '__getitem__' of a type (line 48)
    getitem___148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 24), fringe_147, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 48)
    subscript_call_result_149 = invoke(stypy.reporting.localization.Localization(__file__, 48, 24), getitem___148, int_146)
    
    # Applying the binary operator 'and' (line 48)
    result_and_keyword_150 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 10), 'and', subscript_call_result_145, subscript_call_result_149)
    
    # Assigning a type to the variable 'result_and_keyword_150' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'result_and_keyword_150', result_and_keyword_150)
    # Testing if the while is going to be iterated (line 48)
    # Testing the type of an if condition (line 48)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 48, 4), result_and_keyword_150)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 48, 4), result_and_keyword_150):
        # SSA begins for while statement (line 48)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a BinOp to a Name (line 51):
        
        # Assigning a BinOp to a Name (line 51):
        int_151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 14), 'int')
        # Getting the type of 'dir' (line 51)
        dir_152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'dir')
        # Applying the binary operator '-' (line 51)
        result_sub_153 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 14), '-', int_151, dir_152)
        
        # Assigning a type to the variable 'dir' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'dir', result_sub_153)
        
        # Assigning a Call to a Tuple (line 53):
        
        # Assigning a Call to a Name:
        
        # Call to heappop(...): (line 53)
        # Processing the call arguments (line 53)
        
        # Obtaining the type of the subscript
        # Getting the type of 'dir' (line 53)
        dir_156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 41), 'dir', False)
        # Getting the type of 'fringe' (line 53)
        fringe_157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 34), 'fringe', False)
        # Obtaining the member '__getitem__' of a type (line 53)
        getitem___158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 34), fringe_157, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 53)
        subscript_call_result_159 = invoke(stypy.reporting.localization.Localization(__file__, 53, 34), getitem___158, dir_156)
        
        # Processing the call keyword arguments (line 53)
        kwargs_160 = {}
        # Getting the type of 'heapq' (line 53)
        heapq_154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 20), 'heapq', False)
        # Obtaining the member 'heappop' of a type (line 53)
        heappop_155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 20), heapq_154, 'heappop')
        # Calling heappop(args, kwargs) (line 53)
        heappop_call_result_161 = invoke(stypy.reporting.localization.Localization(__file__, 53, 20), heappop_155, *[subscript_call_result_159], **kwargs_160)
        
        # Assigning a type to the variable 'call_assignment_5' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'call_assignment_5', heappop_call_result_161)
        
        # Assigning a Call to a Name (line 53):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_5' (line 53)
        call_assignment_5_162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'call_assignment_5', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_163 = stypy_get_value_from_tuple(call_assignment_5_162, 2, 0)
        
        # Assigning a type to the variable 'call_assignment_6' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'call_assignment_6', stypy_get_value_from_tuple_call_result_163)
        
        # Assigning a Name to a Name (line 53):
        # Getting the type of 'call_assignment_6' (line 53)
        call_assignment_6_164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'call_assignment_6')
        # Assigning a type to the variable 'dist' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 9), 'dist', call_assignment_6_164)
        
        # Assigning a Call to a Name (line 53):
        
        # Call to stypy_get_value_from_tuple(...):
        # Processing the call arguments
        # Getting the type of 'call_assignment_5' (line 53)
        call_assignment_5_165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'call_assignment_5', False)
        # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
        stypy_get_value_from_tuple_call_result_166 = stypy_get_value_from_tuple(call_assignment_5_165, 2, 1)
        
        # Assigning a type to the variable 'call_assignment_7' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'call_assignment_7', stypy_get_value_from_tuple_call_result_166)
        
        # Assigning a Name to a Name (line 53):
        # Getting the type of 'call_assignment_7' (line 53)
        call_assignment_7_167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'call_assignment_7')
        # Assigning a type to the variable 'v' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 15), 'v', call_assignment_7_167)
        
        # Getting the type of 'v' (line 54)
        v_168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 11), 'v')
        
        # Obtaining the type of the subscript
        # Getting the type of 'dir' (line 54)
        dir_169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 22), 'dir')
        # Getting the type of 'dists' (line 54)
        dists_170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'dists')
        # Obtaining the member '__getitem__' of a type (line 54)
        getitem___171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 16), dists_170, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 54)
        subscript_call_result_172 = invoke(stypy.reporting.localization.Localization(__file__, 54, 16), getitem___171, dir_169)
        
        # Applying the binary operator 'in' (line 54)
        result_contains_173 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 11), 'in', v_168, subscript_call_result_172)
        
        # Testing if the type of an if condition is none (line 54)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 54, 8), result_contains_173):
            pass
        else:
            
            # Testing the type of an if condition (line 54)
            if_condition_174 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 54, 8), result_contains_173)
            # Assigning a type to the variable 'if_condition_174' (line 54)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'if_condition_174', if_condition_174)
            # SSA begins for if statement (line 54)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 54)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Name to a Subscript (line 58):
        
        # Assigning a Name to a Subscript (line 58):
        # Getting the type of 'dist' (line 58)
        dist_175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 24), 'dist')
        
        # Obtaining the type of the subscript
        # Getting the type of 'dir' (line 58)
        dir_176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 14), 'dir')
        # Getting the type of 'dists' (line 58)
        dists_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'dists')
        # Obtaining the member '__getitem__' of a type (line 58)
        getitem___178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), dists_177, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 58)
        subscript_call_result_179 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), getitem___178, dir_176)
        
        # Getting the type of 'v' (line 58)
        v_180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 19), 'v')
        # Storing an element on a container (line 58)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 8), subscript_call_result_179, (v_180, dist_175))
        
        # Getting the type of 'v' (line 59)
        v_181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 11), 'v')
        
        # Obtaining the type of the subscript
        int_182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 22), 'int')
        # Getting the type of 'dir' (line 59)
        dir_183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 24), 'dir')
        # Applying the binary operator '-' (line 59)
        result_sub_184 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 22), '-', int_182, dir_183)
        
        # Getting the type of 'dists' (line 59)
        dists_185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 16), 'dists')
        # Obtaining the member '__getitem__' of a type (line 59)
        getitem___186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 16), dists_185, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 59)
        subscript_call_result_187 = invoke(stypy.reporting.localization.Localization(__file__, 59, 16), getitem___186, result_sub_184)
        
        # Applying the binary operator 'in' (line 59)
        result_contains_188 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 11), 'in', v_181, subscript_call_result_187)
        
        # Testing if the type of an if condition is none (line 59)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 59, 8), result_contains_188):
            pass
        else:
            
            # Testing the type of an if condition (line 59)
            if_condition_189 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 59, 8), result_contains_188)
            # Assigning a type to the variable 'if_condition_189' (line 59)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'if_condition_189', if_condition_189)
            # SSA begins for if statement (line 59)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Obtaining an instance of the builtin type 'tuple' (line 62)
            tuple_190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 20), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 62)
            # Adding element type (line 62)
            # Getting the type of 'finaldist' (line 62)
            finaldist_191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 20), 'finaldist')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 20), tuple_190, finaldist_191)
            # Adding element type (line 62)
            # Getting the type of 'finalpath' (line 62)
            finalpath_192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 30), 'finalpath')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 20), tuple_190, finalpath_192)
            
            # Assigning a type to the variable 'stypy_return_type' (line 62)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'stypy_return_type', tuple_190)
            # SSA join for if statement (line 59)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'v' (line 63)
        v_193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 25), 'v')
        # Obtaining the member 'neighs' of a type (line 63)
        neighs_194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 25), v_193, 'neighs')
        # Assigning a type to the variable 'neighs_194' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'neighs_194', neighs_194)
        # Testing if the for loop is going to be iterated (line 63)
        # Testing the type of a for loop iterable (line 63)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 63, 8), neighs_194)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 63, 8), neighs_194):
            # Getting the type of the for loop variable (line 63)
            for_loop_var_195 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 63, 8), neighs_194)
            # Assigning a type to the variable 'w' (line 63)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'w', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 8), for_loop_var_195, 2, 0))
            # Assigning a type to the variable 'weight' (line 63)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'weight', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 8), for_loop_var_195, 2, 1))
            # SSA begins for a for statement (line 63)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a BinOp to a Name (line 64):
            
            # Assigning a BinOp to a Name (line 64):
            
            # Obtaining the type of the subscript
            # Getting the type of 'v' (line 64)
            v_196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 34), 'v')
            
            # Obtaining the type of the subscript
            # Getting the type of 'dir' (line 64)
            dir_197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 29), 'dir')
            # Getting the type of 'dists' (line 64)
            dists_198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 23), 'dists')
            # Obtaining the member '__getitem__' of a type (line 64)
            getitem___199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 23), dists_198, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 64)
            subscript_call_result_200 = invoke(stypy.reporting.localization.Localization(__file__, 64, 23), getitem___199, dir_197)
            
            # Obtaining the member '__getitem__' of a type (line 64)
            getitem___201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 23), subscript_call_result_200, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 64)
            subscript_call_result_202 = invoke(stypy.reporting.localization.Localization(__file__, 64, 23), getitem___201, v_196)
            
            # Getting the type of 'weight' (line 64)
            weight_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 39), 'weight')
            # Applying the binary operator '+' (line 64)
            result_add_204 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 23), '+', subscript_call_result_202, weight_203)
            
            # Assigning a type to the variable 'vwLength' (line 64)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'vwLength', result_add_204)
            
            # Getting the type of 'w' (line 65)
            w_205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 15), 'w')
            
            # Obtaining the type of the subscript
            # Getting the type of 'dir' (line 65)
            dir_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 26), 'dir')
            # Getting the type of 'dists' (line 65)
            dists_207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 20), 'dists')
            # Obtaining the member '__getitem__' of a type (line 65)
            getitem___208 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 20), dists_207, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 65)
            subscript_call_result_209 = invoke(stypy.reporting.localization.Localization(__file__, 65, 20), getitem___208, dir_206)
            
            # Applying the binary operator 'in' (line 65)
            result_contains_210 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 15), 'in', w_205, subscript_call_result_209)
            
            # Testing if the type of an if condition is none (line 65)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 65, 12), result_contains_210):
                
                # Evaluating a boolean operation
                
                # Getting the type of 'w' (line 67)
                w_212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 17), 'w')
                
                # Obtaining the type of the subscript
                # Getting the type of 'dir' (line 67)
                dir_213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 31), 'dir')
                # Getting the type of 'seen' (line 67)
                seen_214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 26), 'seen')
                # Obtaining the member '__getitem__' of a type (line 67)
                getitem___215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 26), seen_214, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 67)
                subscript_call_result_216 = invoke(stypy.reporting.localization.Localization(__file__, 67, 26), getitem___215, dir_213)
                
                # Applying the binary operator 'notin' (line 67)
                result_contains_217 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 17), 'notin', w_212, subscript_call_result_216)
                
                
                # Getting the type of 'vwLength' (line 67)
                vwLength_218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 39), 'vwLength')
                
                # Obtaining the type of the subscript
                # Getting the type of 'w' (line 67)
                w_219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 60), 'w')
                
                # Obtaining the type of the subscript
                # Getting the type of 'dir' (line 67)
                dir_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 55), 'dir')
                # Getting the type of 'seen' (line 67)
                seen_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 50), 'seen')
                # Obtaining the member '__getitem__' of a type (line 67)
                getitem___222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 50), seen_221, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 67)
                subscript_call_result_223 = invoke(stypy.reporting.localization.Localization(__file__, 67, 50), getitem___222, dir_220)
                
                # Obtaining the member '__getitem__' of a type (line 67)
                getitem___224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 50), subscript_call_result_223, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 67)
                subscript_call_result_225 = invoke(stypy.reporting.localization.Localization(__file__, 67, 50), getitem___224, w_219)
                
                # Applying the binary operator '<' (line 67)
                result_lt_226 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 39), '<', vwLength_218, subscript_call_result_225)
                
                # Applying the binary operator 'or' (line 67)
                result_or_keyword_227 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 17), 'or', result_contains_217, result_lt_226)
                
                # Testing if the type of an if condition is none (line 67)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 67, 17), result_or_keyword_227):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 67)
                    if_condition_228 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 17), result_or_keyword_227)
                    # Assigning a type to the variable 'if_condition_228' (line 67)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 17), 'if_condition_228', if_condition_228)
                    # SSA begins for if statement (line 67)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Subscript (line 69):
                    
                    # Assigning a Name to a Subscript (line 69):
                    # Getting the type of 'vwLength' (line 69)
                    vwLength_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 31), 'vwLength')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'dir' (line 69)
                    dir_230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 21), 'dir')
                    # Getting the type of 'seen' (line 69)
                    seen_231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 16), 'seen')
                    # Obtaining the member '__getitem__' of a type (line 69)
                    getitem___232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 16), seen_231, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 69)
                    subscript_call_result_233 = invoke(stypy.reporting.localization.Localization(__file__, 69, 16), getitem___232, dir_230)
                    
                    # Getting the type of 'w' (line 69)
                    w_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 26), 'w')
                    # Storing an element on a container (line 69)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 16), subscript_call_result_233, (w_234, vwLength_229))
                    
                    # Call to heappush(...): (line 70)
                    # Processing the call arguments (line 70)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'dir' (line 70)
                    dir_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 38), 'dir', False)
                    # Getting the type of 'fringe' (line 70)
                    fringe_238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 31), 'fringe', False)
                    # Obtaining the member '__getitem__' of a type (line 70)
                    getitem___239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 31), fringe_238, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 70)
                    subscript_call_result_240 = invoke(stypy.reporting.localization.Localization(__file__, 70, 31), getitem___239, dir_237)
                    
                    
                    # Obtaining an instance of the builtin type 'tuple' (line 70)
                    tuple_241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 45), 'tuple')
                    # Adding type elements to the builtin type 'tuple' instance (line 70)
                    # Adding element type (line 70)
                    # Getting the type of 'vwLength' (line 70)
                    vwLength_242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 45), 'vwLength', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 45), tuple_241, vwLength_242)
                    # Adding element type (line 70)
                    # Getting the type of 'w' (line 70)
                    w_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 54), 'w', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 45), tuple_241, w_243)
                    
                    # Processing the call keyword arguments (line 70)
                    kwargs_244 = {}
                    # Getting the type of 'heapq' (line 70)
                    heapq_235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'heapq', False)
                    # Obtaining the member 'heappush' of a type (line 70)
                    heappush_236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 16), heapq_235, 'heappush')
                    # Calling heappush(args, kwargs) (line 70)
                    heappush_call_result_245 = invoke(stypy.reporting.localization.Localization(__file__, 70, 16), heappush_236, *[subscript_call_result_240, tuple_241], **kwargs_244)
                    
                    
                    # Assigning a BinOp to a Subscript (line 71):
                    
                    # Assigning a BinOp to a Subscript (line 71):
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'v' (line 71)
                    v_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 43), 'v')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'dir' (line 71)
                    dir_247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 38), 'dir')
                    # Getting the type of 'paths' (line 71)
                    paths_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 32), 'paths')
                    # Obtaining the member '__getitem__' of a type (line 71)
                    getitem___249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 32), paths_248, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 71)
                    subscript_call_result_250 = invoke(stypy.reporting.localization.Localization(__file__, 71, 32), getitem___249, dir_247)
                    
                    # Obtaining the member '__getitem__' of a type (line 71)
                    getitem___251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 32), subscript_call_result_250, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 71)
                    subscript_call_result_252 = invoke(stypy.reporting.localization.Localization(__file__, 71, 32), getitem___251, v_246)
                    
                    
                    # Obtaining an instance of the builtin type 'list' (line 71)
                    list_253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 46), 'list')
                    # Adding type elements to the builtin type 'list' instance (line 71)
                    # Adding element type (line 71)
                    # Getting the type of 'w' (line 71)
                    w_254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 47), 'w')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 46), list_253, w_254)
                    
                    # Applying the binary operator '+' (line 71)
                    result_add_255 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 32), '+', subscript_call_result_252, list_253)
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'dir' (line 71)
                    dir_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 22), 'dir')
                    # Getting the type of 'paths' (line 71)
                    paths_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 16), 'paths')
                    # Obtaining the member '__getitem__' of a type (line 71)
                    getitem___258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 16), paths_257, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 71)
                    subscript_call_result_259 = invoke(stypy.reporting.localization.Localization(__file__, 71, 16), getitem___258, dir_256)
                    
                    # Getting the type of 'w' (line 71)
                    w_260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 27), 'w')
                    # Storing an element on a container (line 71)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 16), subscript_call_result_259, (w_260, result_add_255))
                    
                    # Evaluating a boolean operation
                    
                    # Getting the type of 'w' (line 72)
                    w_261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 19), 'w')
                    
                    # Obtaining the type of the subscript
                    int_262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 29), 'int')
                    # Getting the type of 'seen' (line 72)
                    seen_263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 24), 'seen')
                    # Obtaining the member '__getitem__' of a type (line 72)
                    getitem___264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 24), seen_263, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 72)
                    subscript_call_result_265 = invoke(stypy.reporting.localization.Localization(__file__, 72, 24), getitem___264, int_262)
                    
                    # Applying the binary operator 'in' (line 72)
                    result_contains_266 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 19), 'in', w_261, subscript_call_result_265)
                    
                    
                    # Getting the type of 'w' (line 72)
                    w_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 36), 'w')
                    
                    # Obtaining the type of the subscript
                    int_268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 46), 'int')
                    # Getting the type of 'seen' (line 72)
                    seen_269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 41), 'seen')
                    # Obtaining the member '__getitem__' of a type (line 72)
                    getitem___270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 41), seen_269, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 72)
                    subscript_call_result_271 = invoke(stypy.reporting.localization.Localization(__file__, 72, 41), getitem___270, int_268)
                    
                    # Applying the binary operator 'in' (line 72)
                    result_contains_272 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 36), 'in', w_267, subscript_call_result_271)
                    
                    # Applying the binary operator 'and' (line 72)
                    result_and_keyword_273 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 19), 'and', result_contains_266, result_contains_272)
                    
                    # Testing if the type of an if condition is none (line 72)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 72, 16), result_and_keyword_273):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 72)
                        if_condition_274 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 16), result_and_keyword_273)
                        # Assigning a type to the variable 'if_condition_274' (line 72)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 16), 'if_condition_274', if_condition_274)
                        # SSA begins for if statement (line 72)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a BinOp to a Name (line 75):
                        
                        # Assigning a BinOp to a Name (line 75):
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'w' (line 75)
                        w_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 40), 'w')
                        
                        # Obtaining the type of the subscript
                        int_276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 37), 'int')
                        # Getting the type of 'seen' (line 75)
                        seen_277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 32), 'seen')
                        # Obtaining the member '__getitem__' of a type (line 75)
                        getitem___278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 32), seen_277, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 75)
                        subscript_call_result_279 = invoke(stypy.reporting.localization.Localization(__file__, 75, 32), getitem___278, int_276)
                        
                        # Obtaining the member '__getitem__' of a type (line 75)
                        getitem___280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 32), subscript_call_result_279, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 75)
                        subscript_call_result_281 = invoke(stypy.reporting.localization.Localization(__file__, 75, 32), getitem___280, w_275)
                        
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'w' (line 75)
                        w_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 53), 'w')
                        
                        # Obtaining the type of the subscript
                        int_283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 50), 'int')
                        # Getting the type of 'seen' (line 75)
                        seen_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 45), 'seen')
                        # Obtaining the member '__getitem__' of a type (line 75)
                        getitem___285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 45), seen_284, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 75)
                        subscript_call_result_286 = invoke(stypy.reporting.localization.Localization(__file__, 75, 45), getitem___285, int_283)
                        
                        # Obtaining the member '__getitem__' of a type (line 75)
                        getitem___287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 45), subscript_call_result_286, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 75)
                        subscript_call_result_288 = invoke(stypy.reporting.localization.Localization(__file__, 75, 45), getitem___287, w_282)
                        
                        # Applying the binary operator '+' (line 75)
                        result_add_289 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 32), '+', subscript_call_result_281, subscript_call_result_288)
                        
                        # Assigning a type to the variable 'totaldist' (line 75)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 20), 'totaldist', result_add_289)
                        
                        # Evaluating a boolean operation
                        
                        # Getting the type of 'finalpath' (line 76)
                        finalpath_290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 23), 'finalpath')
                        
                        # Obtaining an instance of the builtin type 'list' (line 76)
                        list_291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 36), 'list')
                        # Adding type elements to the builtin type 'list' instance (line 76)
                        
                        # Applying the binary operator '==' (line 76)
                        result_eq_292 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 23), '==', finalpath_290, list_291)
                        
                        
                        # Getting the type of 'finaldist' (line 76)
                        finaldist_293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 42), 'finaldist')
                        # Getting the type of 'totaldist' (line 76)
                        totaldist_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 54), 'totaldist')
                        # Applying the binary operator '>' (line 76)
                        result_gt_295 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 42), '>', finaldist_293, totaldist_294)
                        
                        # Applying the binary operator 'or' (line 76)
                        result_or_keyword_296 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 23), 'or', result_eq_292, result_gt_295)
                        
                        # Testing if the type of an if condition is none (line 76)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 76, 20), result_or_keyword_296):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 76)
                            if_condition_297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 20), result_or_keyword_296)
                            # Assigning a type to the variable 'if_condition_297' (line 76)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 20), 'if_condition_297', if_condition_297)
                            # SSA begins for if statement (line 76)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a Name to a Name (line 77):
                            
                            # Assigning a Name to a Name (line 77):
                            # Getting the type of 'totaldist' (line 77)
                            totaldist_298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 36), 'totaldist')
                            # Assigning a type to the variable 'finaldist' (line 77)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 24), 'finaldist', totaldist_298)
                            
                            # Assigning a Subscript to a Name (line 78):
                            
                            # Assigning a Subscript to a Name (line 78):
                            
                            # Obtaining the type of the subscript
                            slice_299 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 78, 34), None, None, None)
                            
                            # Obtaining the type of the subscript
                            # Getting the type of 'w' (line 78)
                            w_300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 43), 'w')
                            
                            # Obtaining the type of the subscript
                            int_301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 40), 'int')
                            # Getting the type of 'paths' (line 78)
                            paths_302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 34), 'paths')
                            # Obtaining the member '__getitem__' of a type (line 78)
                            getitem___303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 34), paths_302, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                            subscript_call_result_304 = invoke(stypy.reporting.localization.Localization(__file__, 78, 34), getitem___303, int_301)
                            
                            # Obtaining the member '__getitem__' of a type (line 78)
                            getitem___305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 34), subscript_call_result_304, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                            subscript_call_result_306 = invoke(stypy.reporting.localization.Localization(__file__, 78, 34), getitem___305, w_300)
                            
                            # Obtaining the member '__getitem__' of a type (line 78)
                            getitem___307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 34), subscript_call_result_306, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                            subscript_call_result_308 = invoke(stypy.reporting.localization.Localization(__file__, 78, 34), getitem___307, slice_299)
                            
                            # Assigning a type to the variable 'revpath' (line 78)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 24), 'revpath', subscript_call_result_308)
                            
                            # Call to reverse(...): (line 79)
                            # Processing the call keyword arguments (line 79)
                            kwargs_311 = {}
                            # Getting the type of 'revpath' (line 79)
                            revpath_309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 24), 'revpath', False)
                            # Obtaining the member 'reverse' of a type (line 79)
                            reverse_310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 24), revpath_309, 'reverse')
                            # Calling reverse(args, kwargs) (line 79)
                            reverse_call_result_312 = invoke(stypy.reporting.localization.Localization(__file__, 79, 24), reverse_310, *[], **kwargs_311)
                            
                            
                            # Assigning a BinOp to a Name (line 80):
                            
                            # Assigning a BinOp to a Name (line 80):
                            
                            # Obtaining the type of the subscript
                            # Getting the type of 'w' (line 80)
                            w_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 45), 'w')
                            
                            # Obtaining the type of the subscript
                            int_314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 42), 'int')
                            # Getting the type of 'paths' (line 80)
                            paths_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 36), 'paths')
                            # Obtaining the member '__getitem__' of a type (line 80)
                            getitem___316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 36), paths_315, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 80)
                            subscript_call_result_317 = invoke(stypy.reporting.localization.Localization(__file__, 80, 36), getitem___316, int_314)
                            
                            # Obtaining the member '__getitem__' of a type (line 80)
                            getitem___318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 36), subscript_call_result_317, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 80)
                            subscript_call_result_319 = invoke(stypy.reporting.localization.Localization(__file__, 80, 36), getitem___318, w_313)
                            
                            
                            # Obtaining the type of the subscript
                            int_320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 58), 'int')
                            slice_321 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 80, 50), int_320, None, None)
                            # Getting the type of 'revpath' (line 80)
                            revpath_322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 50), 'revpath')
                            # Obtaining the member '__getitem__' of a type (line 80)
                            getitem___323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 50), revpath_322, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 80)
                            subscript_call_result_324 = invoke(stypy.reporting.localization.Localization(__file__, 80, 50), getitem___323, slice_321)
                            
                            # Applying the binary operator '+' (line 80)
                            result_add_325 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 36), '+', subscript_call_result_319, subscript_call_result_324)
                            
                            # Assigning a type to the variable 'finalpath' (line 80)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 24), 'finalpath', result_add_325)
                            # SSA join for if statement (line 76)
                            module_type_store = module_type_store.join_ssa_context()
                            

                        # SSA join for if statement (line 72)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 67)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 65)
                if_condition_211 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 65, 12), result_contains_210)
                # Assigning a type to the variable 'if_condition_211' (line 65)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'if_condition_211', if_condition_211)
                # SSA begins for if statement (line 65)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                pass
                # SSA branch for the else part of an if statement (line 65)
                module_type_store.open_ssa_branch('else')
                
                # Evaluating a boolean operation
                
                # Getting the type of 'w' (line 67)
                w_212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 17), 'w')
                
                # Obtaining the type of the subscript
                # Getting the type of 'dir' (line 67)
                dir_213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 31), 'dir')
                # Getting the type of 'seen' (line 67)
                seen_214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 26), 'seen')
                # Obtaining the member '__getitem__' of a type (line 67)
                getitem___215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 26), seen_214, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 67)
                subscript_call_result_216 = invoke(stypy.reporting.localization.Localization(__file__, 67, 26), getitem___215, dir_213)
                
                # Applying the binary operator 'notin' (line 67)
                result_contains_217 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 17), 'notin', w_212, subscript_call_result_216)
                
                
                # Getting the type of 'vwLength' (line 67)
                vwLength_218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 39), 'vwLength')
                
                # Obtaining the type of the subscript
                # Getting the type of 'w' (line 67)
                w_219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 60), 'w')
                
                # Obtaining the type of the subscript
                # Getting the type of 'dir' (line 67)
                dir_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 55), 'dir')
                # Getting the type of 'seen' (line 67)
                seen_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 50), 'seen')
                # Obtaining the member '__getitem__' of a type (line 67)
                getitem___222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 50), seen_221, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 67)
                subscript_call_result_223 = invoke(stypy.reporting.localization.Localization(__file__, 67, 50), getitem___222, dir_220)
                
                # Obtaining the member '__getitem__' of a type (line 67)
                getitem___224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 50), subscript_call_result_223, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 67)
                subscript_call_result_225 = invoke(stypy.reporting.localization.Localization(__file__, 67, 50), getitem___224, w_219)
                
                # Applying the binary operator '<' (line 67)
                result_lt_226 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 39), '<', vwLength_218, subscript_call_result_225)
                
                # Applying the binary operator 'or' (line 67)
                result_or_keyword_227 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 17), 'or', result_contains_217, result_lt_226)
                
                # Testing if the type of an if condition is none (line 67)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 67, 17), result_or_keyword_227):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 67)
                    if_condition_228 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 17), result_or_keyword_227)
                    # Assigning a type to the variable 'if_condition_228' (line 67)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 17), 'if_condition_228', if_condition_228)
                    # SSA begins for if statement (line 67)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Subscript (line 69):
                    
                    # Assigning a Name to a Subscript (line 69):
                    # Getting the type of 'vwLength' (line 69)
                    vwLength_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 31), 'vwLength')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'dir' (line 69)
                    dir_230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 21), 'dir')
                    # Getting the type of 'seen' (line 69)
                    seen_231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 16), 'seen')
                    # Obtaining the member '__getitem__' of a type (line 69)
                    getitem___232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 16), seen_231, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 69)
                    subscript_call_result_233 = invoke(stypy.reporting.localization.Localization(__file__, 69, 16), getitem___232, dir_230)
                    
                    # Getting the type of 'w' (line 69)
                    w_234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 26), 'w')
                    # Storing an element on a container (line 69)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 16), subscript_call_result_233, (w_234, vwLength_229))
                    
                    # Call to heappush(...): (line 70)
                    # Processing the call arguments (line 70)
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'dir' (line 70)
                    dir_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 38), 'dir', False)
                    # Getting the type of 'fringe' (line 70)
                    fringe_238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 31), 'fringe', False)
                    # Obtaining the member '__getitem__' of a type (line 70)
                    getitem___239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 31), fringe_238, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 70)
                    subscript_call_result_240 = invoke(stypy.reporting.localization.Localization(__file__, 70, 31), getitem___239, dir_237)
                    
                    
                    # Obtaining an instance of the builtin type 'tuple' (line 70)
                    tuple_241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 45), 'tuple')
                    # Adding type elements to the builtin type 'tuple' instance (line 70)
                    # Adding element type (line 70)
                    # Getting the type of 'vwLength' (line 70)
                    vwLength_242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 45), 'vwLength', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 45), tuple_241, vwLength_242)
                    # Adding element type (line 70)
                    # Getting the type of 'w' (line 70)
                    w_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 54), 'w', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 45), tuple_241, w_243)
                    
                    # Processing the call keyword arguments (line 70)
                    kwargs_244 = {}
                    # Getting the type of 'heapq' (line 70)
                    heapq_235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'heapq', False)
                    # Obtaining the member 'heappush' of a type (line 70)
                    heappush_236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 16), heapq_235, 'heappush')
                    # Calling heappush(args, kwargs) (line 70)
                    heappush_call_result_245 = invoke(stypy.reporting.localization.Localization(__file__, 70, 16), heappush_236, *[subscript_call_result_240, tuple_241], **kwargs_244)
                    
                    
                    # Assigning a BinOp to a Subscript (line 71):
                    
                    # Assigning a BinOp to a Subscript (line 71):
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'v' (line 71)
                    v_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 43), 'v')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'dir' (line 71)
                    dir_247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 38), 'dir')
                    # Getting the type of 'paths' (line 71)
                    paths_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 32), 'paths')
                    # Obtaining the member '__getitem__' of a type (line 71)
                    getitem___249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 32), paths_248, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 71)
                    subscript_call_result_250 = invoke(stypy.reporting.localization.Localization(__file__, 71, 32), getitem___249, dir_247)
                    
                    # Obtaining the member '__getitem__' of a type (line 71)
                    getitem___251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 32), subscript_call_result_250, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 71)
                    subscript_call_result_252 = invoke(stypy.reporting.localization.Localization(__file__, 71, 32), getitem___251, v_246)
                    
                    
                    # Obtaining an instance of the builtin type 'list' (line 71)
                    list_253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 46), 'list')
                    # Adding type elements to the builtin type 'list' instance (line 71)
                    # Adding element type (line 71)
                    # Getting the type of 'w' (line 71)
                    w_254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 47), 'w')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 46), list_253, w_254)
                    
                    # Applying the binary operator '+' (line 71)
                    result_add_255 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 32), '+', subscript_call_result_252, list_253)
                    
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'dir' (line 71)
                    dir_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 22), 'dir')
                    # Getting the type of 'paths' (line 71)
                    paths_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 16), 'paths')
                    # Obtaining the member '__getitem__' of a type (line 71)
                    getitem___258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 16), paths_257, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 71)
                    subscript_call_result_259 = invoke(stypy.reporting.localization.Localization(__file__, 71, 16), getitem___258, dir_256)
                    
                    # Getting the type of 'w' (line 71)
                    w_260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 27), 'w')
                    # Storing an element on a container (line 71)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 16), subscript_call_result_259, (w_260, result_add_255))
                    
                    # Evaluating a boolean operation
                    
                    # Getting the type of 'w' (line 72)
                    w_261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 19), 'w')
                    
                    # Obtaining the type of the subscript
                    int_262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 29), 'int')
                    # Getting the type of 'seen' (line 72)
                    seen_263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 24), 'seen')
                    # Obtaining the member '__getitem__' of a type (line 72)
                    getitem___264 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 24), seen_263, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 72)
                    subscript_call_result_265 = invoke(stypy.reporting.localization.Localization(__file__, 72, 24), getitem___264, int_262)
                    
                    # Applying the binary operator 'in' (line 72)
                    result_contains_266 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 19), 'in', w_261, subscript_call_result_265)
                    
                    
                    # Getting the type of 'w' (line 72)
                    w_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 36), 'w')
                    
                    # Obtaining the type of the subscript
                    int_268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 46), 'int')
                    # Getting the type of 'seen' (line 72)
                    seen_269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 41), 'seen')
                    # Obtaining the member '__getitem__' of a type (line 72)
                    getitem___270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 41), seen_269, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 72)
                    subscript_call_result_271 = invoke(stypy.reporting.localization.Localization(__file__, 72, 41), getitem___270, int_268)
                    
                    # Applying the binary operator 'in' (line 72)
                    result_contains_272 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 36), 'in', w_267, subscript_call_result_271)
                    
                    # Applying the binary operator 'and' (line 72)
                    result_and_keyword_273 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 19), 'and', result_contains_266, result_contains_272)
                    
                    # Testing if the type of an if condition is none (line 72)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 72, 16), result_and_keyword_273):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 72)
                        if_condition_274 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 16), result_and_keyword_273)
                        # Assigning a type to the variable 'if_condition_274' (line 72)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 16), 'if_condition_274', if_condition_274)
                        # SSA begins for if statement (line 72)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Assigning a BinOp to a Name (line 75):
                        
                        # Assigning a BinOp to a Name (line 75):
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'w' (line 75)
                        w_275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 40), 'w')
                        
                        # Obtaining the type of the subscript
                        int_276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 37), 'int')
                        # Getting the type of 'seen' (line 75)
                        seen_277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 32), 'seen')
                        # Obtaining the member '__getitem__' of a type (line 75)
                        getitem___278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 32), seen_277, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 75)
                        subscript_call_result_279 = invoke(stypy.reporting.localization.Localization(__file__, 75, 32), getitem___278, int_276)
                        
                        # Obtaining the member '__getitem__' of a type (line 75)
                        getitem___280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 32), subscript_call_result_279, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 75)
                        subscript_call_result_281 = invoke(stypy.reporting.localization.Localization(__file__, 75, 32), getitem___280, w_275)
                        
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'w' (line 75)
                        w_282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 53), 'w')
                        
                        # Obtaining the type of the subscript
                        int_283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 50), 'int')
                        # Getting the type of 'seen' (line 75)
                        seen_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 45), 'seen')
                        # Obtaining the member '__getitem__' of a type (line 75)
                        getitem___285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 45), seen_284, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 75)
                        subscript_call_result_286 = invoke(stypy.reporting.localization.Localization(__file__, 75, 45), getitem___285, int_283)
                        
                        # Obtaining the member '__getitem__' of a type (line 75)
                        getitem___287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 45), subscript_call_result_286, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 75)
                        subscript_call_result_288 = invoke(stypy.reporting.localization.Localization(__file__, 75, 45), getitem___287, w_282)
                        
                        # Applying the binary operator '+' (line 75)
                        result_add_289 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 32), '+', subscript_call_result_281, subscript_call_result_288)
                        
                        # Assigning a type to the variable 'totaldist' (line 75)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 20), 'totaldist', result_add_289)
                        
                        # Evaluating a boolean operation
                        
                        # Getting the type of 'finalpath' (line 76)
                        finalpath_290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 23), 'finalpath')
                        
                        # Obtaining an instance of the builtin type 'list' (line 76)
                        list_291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 36), 'list')
                        # Adding type elements to the builtin type 'list' instance (line 76)
                        
                        # Applying the binary operator '==' (line 76)
                        result_eq_292 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 23), '==', finalpath_290, list_291)
                        
                        
                        # Getting the type of 'finaldist' (line 76)
                        finaldist_293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 42), 'finaldist')
                        # Getting the type of 'totaldist' (line 76)
                        totaldist_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 54), 'totaldist')
                        # Applying the binary operator '>' (line 76)
                        result_gt_295 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 42), '>', finaldist_293, totaldist_294)
                        
                        # Applying the binary operator 'or' (line 76)
                        result_or_keyword_296 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 23), 'or', result_eq_292, result_gt_295)
                        
                        # Testing if the type of an if condition is none (line 76)

                        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 76, 20), result_or_keyword_296):
                            pass
                        else:
                            
                            # Testing the type of an if condition (line 76)
                            if_condition_297 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 20), result_or_keyword_296)
                            # Assigning a type to the variable 'if_condition_297' (line 76)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 20), 'if_condition_297', if_condition_297)
                            # SSA begins for if statement (line 76)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                            
                            # Assigning a Name to a Name (line 77):
                            
                            # Assigning a Name to a Name (line 77):
                            # Getting the type of 'totaldist' (line 77)
                            totaldist_298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 36), 'totaldist')
                            # Assigning a type to the variable 'finaldist' (line 77)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 24), 'finaldist', totaldist_298)
                            
                            # Assigning a Subscript to a Name (line 78):
                            
                            # Assigning a Subscript to a Name (line 78):
                            
                            # Obtaining the type of the subscript
                            slice_299 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 78, 34), None, None, None)
                            
                            # Obtaining the type of the subscript
                            # Getting the type of 'w' (line 78)
                            w_300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 43), 'w')
                            
                            # Obtaining the type of the subscript
                            int_301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 40), 'int')
                            # Getting the type of 'paths' (line 78)
                            paths_302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 34), 'paths')
                            # Obtaining the member '__getitem__' of a type (line 78)
                            getitem___303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 34), paths_302, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                            subscript_call_result_304 = invoke(stypy.reporting.localization.Localization(__file__, 78, 34), getitem___303, int_301)
                            
                            # Obtaining the member '__getitem__' of a type (line 78)
                            getitem___305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 34), subscript_call_result_304, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                            subscript_call_result_306 = invoke(stypy.reporting.localization.Localization(__file__, 78, 34), getitem___305, w_300)
                            
                            # Obtaining the member '__getitem__' of a type (line 78)
                            getitem___307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 34), subscript_call_result_306, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 78)
                            subscript_call_result_308 = invoke(stypy.reporting.localization.Localization(__file__, 78, 34), getitem___307, slice_299)
                            
                            # Assigning a type to the variable 'revpath' (line 78)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 24), 'revpath', subscript_call_result_308)
                            
                            # Call to reverse(...): (line 79)
                            # Processing the call keyword arguments (line 79)
                            kwargs_311 = {}
                            # Getting the type of 'revpath' (line 79)
                            revpath_309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 24), 'revpath', False)
                            # Obtaining the member 'reverse' of a type (line 79)
                            reverse_310 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 24), revpath_309, 'reverse')
                            # Calling reverse(args, kwargs) (line 79)
                            reverse_call_result_312 = invoke(stypy.reporting.localization.Localization(__file__, 79, 24), reverse_310, *[], **kwargs_311)
                            
                            
                            # Assigning a BinOp to a Name (line 80):
                            
                            # Assigning a BinOp to a Name (line 80):
                            
                            # Obtaining the type of the subscript
                            # Getting the type of 'w' (line 80)
                            w_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 45), 'w')
                            
                            # Obtaining the type of the subscript
                            int_314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 42), 'int')
                            # Getting the type of 'paths' (line 80)
                            paths_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 36), 'paths')
                            # Obtaining the member '__getitem__' of a type (line 80)
                            getitem___316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 36), paths_315, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 80)
                            subscript_call_result_317 = invoke(stypy.reporting.localization.Localization(__file__, 80, 36), getitem___316, int_314)
                            
                            # Obtaining the member '__getitem__' of a type (line 80)
                            getitem___318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 36), subscript_call_result_317, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 80)
                            subscript_call_result_319 = invoke(stypy.reporting.localization.Localization(__file__, 80, 36), getitem___318, w_313)
                            
                            
                            # Obtaining the type of the subscript
                            int_320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 58), 'int')
                            slice_321 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 80, 50), int_320, None, None)
                            # Getting the type of 'revpath' (line 80)
                            revpath_322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 50), 'revpath')
                            # Obtaining the member '__getitem__' of a type (line 80)
                            getitem___323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 50), revpath_322, '__getitem__')
                            # Calling the subscript (__getitem__) to obtain the elements type (line 80)
                            subscript_call_result_324 = invoke(stypy.reporting.localization.Localization(__file__, 80, 50), getitem___323, slice_321)
                            
                            # Applying the binary operator '+' (line 80)
                            result_add_325 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 36), '+', subscript_call_result_319, subscript_call_result_324)
                            
                            # Assigning a type to the variable 'finalpath' (line 80)
                            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 24), 'finalpath', result_add_325)
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
    None_326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 11), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 81)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy_return_type', None_326)
    
    # ################# End of 'bidirectional_dijkstra(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'bidirectional_dijkstra' in the type store
    # Getting the type of 'stypy_return_type' (line 33)
    stypy_return_type_327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_327)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'bidirectional_dijkstra'
    return stypy_return_type_327

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
    kwargs_329 = {}
    # Getting the type of 'Graph' (line 84)
    Graph_328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'Graph', False)
    # Calling Graph(args, kwargs) (line 84)
    Graph_call_result_330 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), Graph_328, *[], **kwargs_329)
    
    # Assigning a type to the variable 'G' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'G', Graph_call_result_330)
    
    # Assigning a List to a Name (line 85):
    
    # Assigning a List to a Name (line 85):
    
    # Obtaining an instance of the builtin type 'list' (line 85)
    list_331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 11), 'list')
    # Adding type elements to the builtin type 'list' instance (line 85)
    # Adding element type (line 85)
    
    # Obtaining an instance of the builtin type 'tuple' (line 85)
    tuple_332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 85)
    # Adding element type (line 85)
    int_333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 13), tuple_332, int_333)
    # Adding element type (line 85)
    int_334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 13), tuple_332, int_334)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 11), list_331, tuple_332)
    # Adding element type (line 85)
    
    # Obtaining an instance of the builtin type 'tuple' (line 85)
    tuple_335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 21), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 85)
    # Adding element type (line 85)
    int_336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 21), tuple_335, int_336)
    # Adding element type (line 85)
    int_337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 23), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 21), tuple_335, int_337)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 11), list_331, tuple_335)
    # Adding element type (line 85)
    
    # Obtaining an instance of the builtin type 'tuple' (line 85)
    tuple_338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 28), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 85)
    # Adding element type (line 85)
    int_339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 28), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 28), tuple_338, int_339)
    # Adding element type (line 85)
    int_340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 30), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 28), tuple_338, int_340)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 11), list_331, tuple_338)
    # Adding element type (line 85)
    
    # Obtaining an instance of the builtin type 'tuple' (line 85)
    tuple_341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 85)
    # Adding element type (line 85)
    int_342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 35), tuple_341, int_342)
    # Adding element type (line 85)
    int_343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 37), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 35), tuple_341, int_343)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 11), list_331, tuple_341)
    
    # Assigning a type to the variable 'dirs' (line 85)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'dirs', list_331)
    
    
    # Call to range(...): (line 86)
    # Processing the call arguments (line 86)
    # Getting the type of 'n' (line 86)
    n_345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 19), 'n', False)
    # Processing the call keyword arguments (line 86)
    kwargs_346 = {}
    # Getting the type of 'range' (line 86)
    range_344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 13), 'range', False)
    # Calling range(args, kwargs) (line 86)
    range_call_result_347 = invoke(stypy.reporting.localization.Localization(__file__, 86, 13), range_344, *[n_345], **kwargs_346)
    
    # Assigning a type to the variable 'range_call_result_347' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'range_call_result_347', range_call_result_347)
    # Testing if the for loop is going to be iterated (line 86)
    # Testing the type of a for loop iterable (line 86)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 86, 4), range_call_result_347)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 86, 4), range_call_result_347):
        # Getting the type of the for loop variable (line 86)
        for_loop_var_348 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 86, 4), range_call_result_347)
        # Assigning a type to the variable 'u' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'u', for_loop_var_348)
        # SSA begins for a for statement (line 86)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to range(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'n' (line 87)
        n_350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 23), 'n', False)
        # Processing the call keyword arguments (line 87)
        kwargs_351 = {}
        # Getting the type of 'range' (line 87)
        range_349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 17), 'range', False)
        # Calling range(args, kwargs) (line 87)
        range_call_result_352 = invoke(stypy.reporting.localization.Localization(__file__, 87, 17), range_349, *[n_350], **kwargs_351)
        
        # Assigning a type to the variable 'range_call_result_352' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'range_call_result_352', range_call_result_352)
        # Testing if the for loop is going to be iterated (line 87)
        # Testing the type of a for loop iterable (line 87)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 87, 8), range_call_result_352)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 87, 8), range_call_result_352):
            # Getting the type of the for loop variable (line 87)
            for_loop_var_353 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 87, 8), range_call_result_352)
            # Assigning a type to the variable 'v' (line 87)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'v', for_loop_var_353)
            # SSA begins for a for statement (line 87)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'dirs' (line 88)
            dirs_354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 23), 'dirs')
            # Assigning a type to the variable 'dirs_354' (line 88)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'dirs_354', dirs_354)
            # Testing if the for loop is going to be iterated (line 88)
            # Testing the type of a for loop iterable (line 88)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 88, 12), dirs_354)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 88, 12), dirs_354):
                # Getting the type of the for loop variable (line 88)
                for_loop_var_355 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 88, 12), dirs_354)
                # Assigning a type to the variable 'dir' (line 88)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'dir', for_loop_var_355)
                # SSA begins for a for statement (line 88)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Tuple to a Tuple (line 89):
                
                # Assigning a BinOp to a Name (line 89):
                # Getting the type of 'u' (line 89)
                u_356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 23), 'u')
                
                # Obtaining the type of the subscript
                int_357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 29), 'int')
                # Getting the type of 'dir' (line 89)
                dir_358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 25), 'dir')
                # Obtaining the member '__getitem__' of a type (line 89)
                getitem___359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 25), dir_358, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 89)
                subscript_call_result_360 = invoke(stypy.reporting.localization.Localization(__file__, 89, 25), getitem___359, int_357)
                
                # Applying the binary operator '+' (line 89)
                result_add_361 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 23), '+', u_356, subscript_call_result_360)
                
                # Assigning a type to the variable 'tuple_assignment_8' (line 89)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 16), 'tuple_assignment_8', result_add_361)
                
                # Assigning a BinOp to a Name (line 89):
                # Getting the type of 'v' (line 89)
                v_362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 33), 'v')
                
                # Obtaining the type of the subscript
                int_363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 39), 'int')
                # Getting the type of 'dir' (line 89)
                dir_364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 35), 'dir')
                # Obtaining the member '__getitem__' of a type (line 89)
                getitem___365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 35), dir_364, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 89)
                subscript_call_result_366 = invoke(stypy.reporting.localization.Localization(__file__, 89, 35), getitem___365, int_363)
                
                # Applying the binary operator '+' (line 89)
                result_add_367 = python_operator(stypy.reporting.localization.Localization(__file__, 89, 33), '+', v_362, subscript_call_result_366)
                
                # Assigning a type to the variable 'tuple_assignment_9' (line 89)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 16), 'tuple_assignment_9', result_add_367)
                
                # Assigning a Name to a Name (line 89):
                # Getting the type of 'tuple_assignment_8' (line 89)
                tuple_assignment_8_368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 16), 'tuple_assignment_8')
                # Assigning a type to the variable 'x' (line 89)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 16), 'x', tuple_assignment_8_368)
                
                # Assigning a Name to a Name (line 89):
                # Getting the type of 'tuple_assignment_9' (line 89)
                tuple_assignment_9_369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 16), 'tuple_assignment_9')
                # Assigning a type to the variable 'y' (line 89)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 19), 'y', tuple_assignment_9_369)
                
                # Evaluating a boolean operation
                
                int_370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 19), 'int')
                # Getting the type of 'x' (line 90)
                x_371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 24), 'x')
                # Applying the binary operator '<=' (line 90)
                result_le_372 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 19), '<=', int_370, x_371)
                # Getting the type of 'n' (line 90)
                n_373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 28), 'n')
                # Applying the binary operator '<' (line 90)
                result_lt_374 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 19), '<', x_371, n_373)
                # Applying the binary operator '&' (line 90)
                result_and__375 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 19), '&', result_le_372, result_lt_374)
                
                
                int_376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 34), 'int')
                # Getting the type of 'y' (line 90)
                y_377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 39), 'y')
                # Applying the binary operator '<=' (line 90)
                result_le_378 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 34), '<=', int_376, y_377)
                # Getting the type of 'n' (line 90)
                n_379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 43), 'n')
                # Applying the binary operator '<' (line 90)
                result_lt_380 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 34), '<', y_377, n_379)
                # Applying the binary operator '&' (line 90)
                result_and__381 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 34), '&', result_le_378, result_lt_380)
                
                # Applying the binary operator 'and' (line 90)
                result_and_keyword_382 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 19), 'and', result_and__375, result_and__381)
                
                # Testing if the type of an if condition is none (line 90)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 90, 16), result_and_keyword_382):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 90)
                    if_condition_383 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 90, 16), result_and_keyword_382)
                    # Assigning a type to the variable 'if_condition_383' (line 90)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 16), 'if_condition_383', if_condition_383)
                    # SSA begins for if statement (line 90)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Call to add_edge(...): (line 91)
                    # Processing the call arguments (line 91)
                    
                    # Obtaining an instance of the builtin type 'tuple' (line 91)
                    tuple_386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 32), 'tuple')
                    # Adding type elements to the builtin type 'tuple' instance (line 91)
                    # Adding element type (line 91)
                    # Getting the type of 'u' (line 91)
                    u_387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 32), 'u', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 32), tuple_386, u_387)
                    # Adding element type (line 91)
                    # Getting the type of 'v' (line 91)
                    v_388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 34), 'v', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 32), tuple_386, v_388)
                    
                    
                    # Obtaining an instance of the builtin type 'tuple' (line 91)
                    tuple_389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 39), 'tuple')
                    # Adding type elements to the builtin type 'tuple' instance (line 91)
                    # Adding element type (line 91)
                    # Getting the type of 'x' (line 91)
                    x_390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 39), 'x', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 39), tuple_389, x_390)
                    # Adding element type (line 91)
                    # Getting the type of 'y' (line 91)
                    y_391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 42), 'y', False)
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 39), tuple_389, y_391)
                    
                    
                    # Call to randint(...): (line 91)
                    # Processing the call arguments (line 91)
                    int_394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 61), 'int')
                    int_395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 63), 'int')
                    # Processing the call keyword arguments (line 91)
                    kwargs_396 = {}
                    # Getting the type of 'random' (line 91)
                    random_392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 46), 'random', False)
                    # Obtaining the member 'randint' of a type (line 91)
                    randint_393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 46), random_392, 'randint')
                    # Calling randint(args, kwargs) (line 91)
                    randint_call_result_397 = invoke(stypy.reporting.localization.Localization(__file__, 91, 46), randint_393, *[int_394, int_395], **kwargs_396)
                    
                    # Processing the call keyword arguments (line 91)
                    kwargs_398 = {}
                    # Getting the type of 'G' (line 91)
                    G_384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'G', False)
                    # Obtaining the member 'add_edge' of a type (line 91)
                    add_edge_385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 20), G_384, 'add_edge')
                    # Calling add_edge(args, kwargs) (line 91)
                    add_edge_call_result_399 = invoke(stypy.reporting.localization.Localization(__file__, 91, 20), add_edge_385, *[tuple_386, tuple_389, randint_call_result_397], **kwargs_398)
                    
                    # SSA join for if statement (line 90)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'G' (line 92)
    G_400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 11), 'G')
    # Assigning a type to the variable 'stypy_return_type' (line 92)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'stypy_return_type', G_400)
    
    # ################# End of 'make_graph(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'make_graph' in the type store
    # Getting the type of 'stypy_return_type' (line 83)
    stypy_return_type_401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_401)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'make_graph'
    return stypy_return_type_401

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
    int_402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 8), 'int')
    # Assigning a type to the variable 'n' (line 95)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'n', int_402)
    
    # Assigning a Call to a Name (line 98):
    
    # Assigning a Call to a Name (line 98):
    
    # Call to time(...): (line 98)
    # Processing the call keyword arguments (line 98)
    kwargs_405 = {}
    # Getting the type of 'time' (line 98)
    time_403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 9), 'time', False)
    # Obtaining the member 'time' of a type (line 98)
    time_404 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 9), time_403, 'time')
    # Calling time(args, kwargs) (line 98)
    time_call_result_406 = invoke(stypy.reporting.localization.Localization(__file__, 98, 9), time_404, *[], **kwargs_405)
    
    # Assigning a type to the variable 't0' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 't0', time_call_result_406)
    
    # Assigning a Call to a Name (line 99):
    
    # Assigning a Call to a Name (line 99):
    
    # Call to make_graph(...): (line 99)
    # Processing the call arguments (line 99)
    # Getting the type of 'n' (line 99)
    n_408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 19), 'n', False)
    # Processing the call keyword arguments (line 99)
    kwargs_409 = {}
    # Getting the type of 'make_graph' (line 99)
    make_graph_407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'make_graph', False)
    # Calling make_graph(args, kwargs) (line 99)
    make_graph_call_result_410 = invoke(stypy.reporting.localization.Localization(__file__, 99, 8), make_graph_407, *[n_408], **kwargs_409)
    
    # Assigning a type to the variable 'G' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'G', make_graph_call_result_410)
    
    # Assigning a Call to a Name (line 101):
    
    # Assigning a Call to a Name (line 101):
    
    # Call to time(...): (line 101)
    # Processing the call keyword arguments (line 101)
    kwargs_413 = {}
    # Getting the type of 'time' (line 101)
    time_411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 9), 'time', False)
    # Obtaining the member 'time' of a type (line 101)
    time_412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 9), time_411, 'time')
    # Calling time(args, kwargs) (line 101)
    time_call_result_414 = invoke(stypy.reporting.localization.Localization(__file__, 101, 9), time_412, *[], **kwargs_413)
    
    # Assigning a type to the variable 't1' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 't1', time_call_result_414)
    
    # Assigning a Call to a Tuple (line 102):
    
    # Assigning a Call to a Name:
    
    # Call to bidirectional_dijkstra(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 'G' (line 102)
    G_416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 39), 'G', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 102)
    tuple_417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 43), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 102)
    # Adding element type (line 102)
    int_418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 43), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 43), tuple_417, int_418)
    # Adding element type (line 102)
    int_419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 45), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 43), tuple_417, int_419)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 102)
    tuple_420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 102)
    # Adding element type (line 102)
    # Getting the type of 'n' (line 102)
    n_421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 50), 'n', False)
    int_422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 52), 'int')
    # Applying the binary operator '-' (line 102)
    result_sub_423 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 50), '-', n_421, int_422)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 50), tuple_420, result_sub_423)
    # Adding element type (line 102)
    # Getting the type of 'n' (line 102)
    n_424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 54), 'n', False)
    int_425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 56), 'int')
    # Applying the binary operator '-' (line 102)
    result_sub_426 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 54), '-', n_424, int_425)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 50), tuple_420, result_sub_426)
    
    # Processing the call keyword arguments (line 102)
    kwargs_427 = {}
    # Getting the type of 'bidirectional_dijkstra' (line 102)
    bidirectional_dijkstra_415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'bidirectional_dijkstra', False)
    # Calling bidirectional_dijkstra(args, kwargs) (line 102)
    bidirectional_dijkstra_call_result_428 = invoke(stypy.reporting.localization.Localization(__file__, 102, 16), bidirectional_dijkstra_415, *[G_416, tuple_417, tuple_420], **kwargs_427)
    
    # Assigning a type to the variable 'call_assignment_10' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'call_assignment_10', bidirectional_dijkstra_call_result_428)
    
    # Assigning a Call to a Name (line 102):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_10' (line 102)
    call_assignment_10_429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'call_assignment_10', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_430 = stypy_get_value_from_tuple(call_assignment_10_429, 2, 0)
    
    # Assigning a type to the variable 'call_assignment_11' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'call_assignment_11', stypy_get_value_from_tuple_call_result_430)
    
    # Assigning a Name to a Name (line 102):
    # Getting the type of 'call_assignment_11' (line 102)
    call_assignment_11_431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'call_assignment_11')
    # Assigning a type to the variable 'wt' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'wt', call_assignment_11_431)
    
    # Assigning a Call to a Name (line 102):
    
    # Call to stypy_get_value_from_tuple(...):
    # Processing the call arguments
    # Getting the type of 'call_assignment_10' (line 102)
    call_assignment_10_432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'call_assignment_10', False)
    # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
    stypy_get_value_from_tuple_call_result_433 = stypy_get_value_from_tuple(call_assignment_10_432, 2, 1)
    
    # Assigning a type to the variable 'call_assignment_12' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'call_assignment_12', stypy_get_value_from_tuple_call_result_433)
    
    # Assigning a Name to a Name (line 102):
    # Getting the type of 'call_assignment_12' (line 102)
    call_assignment_12_434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'call_assignment_12')
    # Assigning a type to the variable 'nodes' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'nodes', call_assignment_12_434)
    # Getting the type of 'True' (line 105)
    True_435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type', True_435)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 94)
    stypy_return_type_436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_436)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_436

# Assigning a type to the variable 'run' (line 94)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 0), 'run', run)

# Call to run(...): (line 107)
# Processing the call keyword arguments (line 107)
kwargs_438 = {}
# Getting the type of 'run' (line 107)
run_437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 0), 'run', False)
# Calling run(args, kwargs) (line 107)
run_call_result_439 = invoke(stypy.reporting.localization.Localization(__file__, 107, 0), run_437, *[], **kwargs_438)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
