
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # dijkstra shortest-distance algorithm
2: #
3: # (C) 2006 Gustavo J.A.M. Carneiro, licensed under the GPL version 2
4: 
5: import random
6: 
7: random.seed()
8: 
9: G = None
10: 
11: class Vertex(object):
12:     __slots__ = ('name',)
13:     def __init__(self, name):
14:         self.name = name
15: 
16:     def __str__(self):
17:         return self.name
18: 
19:     def __repr__(self):
20:         return self.name
21: 
22: 
23: class Edge(object):
24:     __slots__ = ('u', 'v', 'd')
25:     def __init__(self, u, v, d):
26:         assert isinstance(u, Vertex) # source vertex
27:         assert isinstance(v, Vertex) # destination vertex
28:         assert isinstance(d, float) # distance, or cost
29:         self.u = u
30:         self.v = v
31:         self.d = d
32:     def __str__(self):
33:         return "[%s --%3.2f--> %s]" % (self.u.name, self.d, self.v.name)
34:     def __repr__(self):
35:         return str(self)
36: 
37: class Graph(object):
38:     def __init__(self):
39:         V = []
40: 
41:         for n in xrange(100):
42:             V.append(Vertex(str(n + 1)))
43: 
44:         E = []
45:         for n in xrange(10*len(V)):
46:             u = V[random.randint(0, len(V) - 1)]
47:             while True:
48:                 v = V[random.randint(0, len(V) - 1)]
49:                 if v is not u:
50:                     break
51:             E.append(Edge(u, v, random.uniform(10, 100)))
52: 
53:         self.V = V
54:         self.E = E
55: 
56:     def distance(self, s, S):
57:         for edge in [e for e in G.E if e.u == s and e.v == S[0]]:
58:             d = edge.d
59:             break
60:         else:
61:             raise AssertionError
62: 
63:         for u, v in zip(S[:-1], S[1:]):
64:             for edge in [e for e in G.E if e.u == u and e.v == v]:
65:                 d += edge.d
66:                 break
67:             else:
68:                 raise AssertionError
69:         return d
70: 
71: def Extract_Min(Q, d):
72:     m = None
73:     md = 1e50
74:     for u in Q:
75:         if m is None:
76:             m = u
77:             md = d[u]
78:         else:
79:             if d[u] < md:
80:                 md = d[u]
81:                 m = u
82:     Q.remove(m)
83:     return m
84: 
85: def dijkstra(G, t, s):
86:     d = {}
87:     previous = {}
88:     for v in G.V:
89:         d[v] = 1e50 # infinity
90:         previous[v] = None
91:     del v
92:     d[s] = 0
93:     S = []
94:     Q = list(G.V)
95: 
96: 
97:     while Q:
98:         u = Extract_Min(Q, d)
99:         if u == t:
100:             break
101:         S.append(u)
102:         for edge in [e for e in G.E if e.u == u]:
103:             if d[u] + edge.d < d[edge.v]:
104:                 d[edge.v] = d[u] + edge.d
105:                 previous[edge.v] = u
106: 
107:     S = []
108:     u = t
109:     while previous[u] is not None:
110:         S.insert(0, u)
111:         u = previous[u]
112:     return S
113: 
114: def run():
115:     global G
116:     
117:     for n in xrange(100):
118:         G = Graph()
119:         s = G.V[random.randint(0, len(G.V) - 1)]
120:         while True:
121:             t = G.V[random.randint(0, len(G.V) - 1)]
122:             if t is not s:
123:                 break
124:         S = dijkstra(G, t, s)
125:         if S:
126:             #print "dijkstra %s ---> %s: " % (s, t), S, G.distance(s, S)
127:             G.distance(s, S)
128:             for inter in S[:-1]:
129:                 S1 = dijkstra(G, t, inter)
130:                 #print "\t => dijkstra %s ---> %s: " % (inter, t), S1, G.distance(inter, S1)
131:                 G.distance(inter, S1)
132:                 if S1 != S[ (len(S) - len(S1)) : ]:
133:                     pass#print "************ ALARM! **************"
134:     return True
135: 
136: run()

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import random' statement (line 5)
import random

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'random', random, module_type_store)


# Call to seed(...): (line 7)
# Processing the call keyword arguments (line 7)
kwargs_3 = {}
# Getting the type of 'random' (line 7)
random_1 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'random', False)
# Obtaining the member 'seed' of a type (line 7)
seed_2 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 7, 0), random_1, 'seed')
# Calling seed(args, kwargs) (line 7)
seed_call_result_4 = invoke(stypy.reporting.localization.Localization(__file__, 7, 0), seed_2, *[], **kwargs_3)


# Assigning a Name to a Name (line 9):
# Getting the type of 'None' (line 9)
None_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'None')
# Assigning a type to the variable 'G' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'G', None_5)
# Declaration of the 'Vertex' class

class Vertex(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 13, 4, False)
        # Assigning a type to the variable 'self' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vertex.__init__', ['name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 14):
        # Getting the type of 'name' (line 14)
        name_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 20), 'name')
        # Getting the type of 'self' (line 14)
        self_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'self')
        # Setting the type of the member 'name' of a type (line 14)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 8), self_7, 'name', name_6)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 16, 4, False)
        # Assigning a type to the variable 'self' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Vertex.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        Vertex.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Vertex.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Vertex.stypy__str__.__dict__.__setitem__('stypy_function_name', 'Vertex.stypy__str__')
        Vertex.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        Vertex.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Vertex.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Vertex.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Vertex.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Vertex.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Vertex.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Vertex.stypy__str__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__str__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__str__(...)' code ##################

        # Getting the type of 'self' (line 17)
        self_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 15), 'self')
        # Obtaining the member 'name' of a type (line 17)
        name_9 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 15), self_8, 'name')
        # Assigning a type to the variable 'stypy_return_type' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'stypy_return_type', name_9)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 16)
        stypy_return_type_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_10)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_10


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 19, 4, False)
        # Assigning a type to the variable 'self' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'self', type_of_self)
        
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

        # Getting the type of 'self' (line 20)
        self_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 15), 'self')
        # Obtaining the member 'name' of a type (line 20)
        name_12 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 15), self_11, 'name')
        # Assigning a type to the variable 'stypy_return_type' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'stypy_return_type', name_12)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 19)
        stypy_return_type_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_13


# Assigning a type to the variable 'Vertex' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'Vertex', Vertex)

# Assigning a Tuple to a Name (line 12):

# Obtaining an instance of the builtin type 'tuple' (line 12)
tuple_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 12)
# Adding element type (line 12)
str_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 17), 'str', 'name')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 17), tuple_14, str_15)

# Getting the type of 'Vertex'
Vertex_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Vertex')
# Setting the type of the member '__slots__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Vertex_16, '__slots__', tuple_14)
# Declaration of the 'Edge' class

class Edge(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 25, 4, False)
        # Assigning a type to the variable 'self' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Edge.__init__', ['u', 'v', 'd'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['u', 'v', 'd'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        # Evaluating assert statement condition
        
        # Call to isinstance(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'u' (line 26)
        u_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 26), 'u', False)
        # Getting the type of 'Vertex' (line 26)
        Vertex_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 29), 'Vertex', False)
        # Processing the call keyword arguments (line 26)
        kwargs_20 = {}
        # Getting the type of 'isinstance' (line 26)
        isinstance_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 26)
        isinstance_call_result_21 = invoke(stypy.reporting.localization.Localization(__file__, 26, 15), isinstance_17, *[u_18, Vertex_19], **kwargs_20)
        
        assert_22 = isinstance_call_result_21
        # Assigning a type to the variable 'assert_22' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'assert_22', isinstance_call_result_21)
        # Evaluating assert statement condition
        
        # Call to isinstance(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'v' (line 27)
        v_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 26), 'v', False)
        # Getting the type of 'Vertex' (line 27)
        Vertex_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 29), 'Vertex', False)
        # Processing the call keyword arguments (line 27)
        kwargs_26 = {}
        # Getting the type of 'isinstance' (line 27)
        isinstance_23 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 27)
        isinstance_call_result_27 = invoke(stypy.reporting.localization.Localization(__file__, 27, 15), isinstance_23, *[v_24, Vertex_25], **kwargs_26)
        
        assert_28 = isinstance_call_result_27
        # Assigning a type to the variable 'assert_28' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'assert_28', isinstance_call_result_27)
        # Evaluating assert statement condition
        
        # Call to isinstance(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'd' (line 28)
        d_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 26), 'd', False)
        # Getting the type of 'float' (line 28)
        float_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 29), 'float', False)
        # Processing the call keyword arguments (line 28)
        kwargs_32 = {}
        # Getting the type of 'isinstance' (line 28)
        isinstance_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 28)
        isinstance_call_result_33 = invoke(stypy.reporting.localization.Localization(__file__, 28, 15), isinstance_29, *[d_30, float_31], **kwargs_32)
        
        assert_34 = isinstance_call_result_33
        # Assigning a type to the variable 'assert_34' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'assert_34', isinstance_call_result_33)
        
        # Assigning a Name to a Attribute (line 29):
        # Getting the type of 'u' (line 29)
        u_35 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 17), 'u')
        # Getting the type of 'self' (line 29)
        self_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'self')
        # Setting the type of the member 'u' of a type (line 29)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), self_36, 'u', u_35)
        
        # Assigning a Name to a Attribute (line 30):
        # Getting the type of 'v' (line 30)
        v_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 17), 'v')
        # Getting the type of 'self' (line 30)
        self_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'self')
        # Setting the type of the member 'v' of a type (line 30)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), self_38, 'v', v_37)
        
        # Assigning a Name to a Attribute (line 31):
        # Getting the type of 'd' (line 31)
        d_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 17), 'd')
        # Getting the type of 'self' (line 31)
        self_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'self')
        # Setting the type of the member 'd' of a type (line 31)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), self_40, 'd', d_39)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 32, 4, False)
        # Assigning a type to the variable 'self' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Edge.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        Edge.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Edge.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Edge.stypy__str__.__dict__.__setitem__('stypy_function_name', 'Edge.stypy__str__')
        Edge.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        Edge.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Edge.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Edge.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Edge.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Edge.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Edge.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Edge.stypy__str__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__str__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__str__(...)' code ##################

        str_41 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 15), 'str', '[%s --%3.2f--> %s]')
        
        # Obtaining an instance of the builtin type 'tuple' (line 33)
        tuple_42 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 33)
        # Adding element type (line 33)
        # Getting the type of 'self' (line 33)
        self_43 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 39), 'self')
        # Obtaining the member 'u' of a type (line 33)
        u_44 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 39), self_43, 'u')
        # Obtaining the member 'name' of a type (line 33)
        name_45 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 39), u_44, 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 39), tuple_42, name_45)
        # Adding element type (line 33)
        # Getting the type of 'self' (line 33)
        self_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 52), 'self')
        # Obtaining the member 'd' of a type (line 33)
        d_47 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 52), self_46, 'd')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 39), tuple_42, d_47)
        # Adding element type (line 33)
        # Getting the type of 'self' (line 33)
        self_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 60), 'self')
        # Obtaining the member 'v' of a type (line 33)
        v_49 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 60), self_48, 'v')
        # Obtaining the member 'name' of a type (line 33)
        name_50 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 60), v_49, 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 39), tuple_42, name_50)
        
        # Applying the binary operator '%' (line 33)
        result_mod_51 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 15), '%', str_41, tuple_42)
        
        # Assigning a type to the variable 'stypy_return_type' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'stypy_return_type', result_mod_51)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 32)
        stypy_return_type_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_52)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_52


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 34, 4, False)
        # Assigning a type to the variable 'self' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Edge.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        Edge.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Edge.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Edge.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'Edge.stypy__repr__')
        Edge.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        Edge.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Edge.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Edge.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Edge.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Edge.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Edge.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Edge.stypy__repr__', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to str(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'self' (line 35)
        self_54 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 19), 'self', False)
        # Processing the call keyword arguments (line 35)
        kwargs_55 = {}
        # Getting the type of 'str' (line 35)
        str_53 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 15), 'str', False)
        # Calling str(args, kwargs) (line 35)
        str_call_result_56 = invoke(stypy.reporting.localization.Localization(__file__, 35, 15), str_53, *[self_54], **kwargs_55)
        
        # Assigning a type to the variable 'stypy_return_type' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'stypy_return_type', str_call_result_56)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 34)
        stypy_return_type_57 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_57)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_57


# Assigning a type to the variable 'Edge' (line 23)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 0), 'Edge', Edge)

# Assigning a Tuple to a Name (line 24):

# Obtaining an instance of the builtin type 'tuple' (line 24)
tuple_58 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 24)
# Adding element type (line 24)
str_59 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 17), 'str', 'u')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 17), tuple_58, str_59)
# Adding element type (line 24)
str_60 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 22), 'str', 'v')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 17), tuple_58, str_60)
# Adding element type (line 24)
str_61 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 27), 'str', 'd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 17), tuple_58, str_61)

# Getting the type of 'Edge'
Edge_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Edge')
# Setting the type of the member '__slots__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Edge_62, '__slots__', tuple_58)
# Declaration of the 'Graph' class

class Graph(object, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 38, 4, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'self', type_of_self)
        
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

        
        # Assigning a List to a Name (line 39):
        
        # Obtaining an instance of the builtin type 'list' (line 39)
        list_63 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 39)
        
        # Assigning a type to the variable 'V' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'V', list_63)
        
        
        # Call to xrange(...): (line 41)
        # Processing the call arguments (line 41)
        int_65 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 24), 'int')
        # Processing the call keyword arguments (line 41)
        kwargs_66 = {}
        # Getting the type of 'xrange' (line 41)
        xrange_64 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 41)
        xrange_call_result_67 = invoke(stypy.reporting.localization.Localization(__file__, 41, 17), xrange_64, *[int_65], **kwargs_66)
        
        # Assigning a type to the variable 'xrange_call_result_67' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'xrange_call_result_67', xrange_call_result_67)
        # Testing if the for loop is going to be iterated (line 41)
        # Testing the type of a for loop iterable (line 41)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 41, 8), xrange_call_result_67)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 41, 8), xrange_call_result_67):
            # Getting the type of the for loop variable (line 41)
            for_loop_var_68 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 41, 8), xrange_call_result_67)
            # Assigning a type to the variable 'n' (line 41)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'n', for_loop_var_68)
            # SSA begins for a for statement (line 41)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to append(...): (line 42)
            # Processing the call arguments (line 42)
            
            # Call to Vertex(...): (line 42)
            # Processing the call arguments (line 42)
            
            # Call to str(...): (line 42)
            # Processing the call arguments (line 42)
            # Getting the type of 'n' (line 42)
            n_73 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 32), 'n', False)
            int_74 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 36), 'int')
            # Applying the binary operator '+' (line 42)
            result_add_75 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 32), '+', n_73, int_74)
            
            # Processing the call keyword arguments (line 42)
            kwargs_76 = {}
            # Getting the type of 'str' (line 42)
            str_72 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 28), 'str', False)
            # Calling str(args, kwargs) (line 42)
            str_call_result_77 = invoke(stypy.reporting.localization.Localization(__file__, 42, 28), str_72, *[result_add_75], **kwargs_76)
            
            # Processing the call keyword arguments (line 42)
            kwargs_78 = {}
            # Getting the type of 'Vertex' (line 42)
            Vertex_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 21), 'Vertex', False)
            # Calling Vertex(args, kwargs) (line 42)
            Vertex_call_result_79 = invoke(stypy.reporting.localization.Localization(__file__, 42, 21), Vertex_71, *[str_call_result_77], **kwargs_78)
            
            # Processing the call keyword arguments (line 42)
            kwargs_80 = {}
            # Getting the type of 'V' (line 42)
            V_69 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'V', False)
            # Obtaining the member 'append' of a type (line 42)
            append_70 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 12), V_69, 'append')
            # Calling append(args, kwargs) (line 42)
            append_call_result_81 = invoke(stypy.reporting.localization.Localization(__file__, 42, 12), append_70, *[Vertex_call_result_79], **kwargs_80)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a List to a Name (line 44):
        
        # Obtaining an instance of the builtin type 'list' (line 44)
        list_82 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 12), 'list')
        # Adding type elements to the builtin type 'list' instance (line 44)
        
        # Assigning a type to the variable 'E' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'E', list_82)
        
        
        # Call to xrange(...): (line 45)
        # Processing the call arguments (line 45)
        int_84 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 24), 'int')
        
        # Call to len(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'V' (line 45)
        V_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 31), 'V', False)
        # Processing the call keyword arguments (line 45)
        kwargs_87 = {}
        # Getting the type of 'len' (line 45)
        len_85 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 27), 'len', False)
        # Calling len(args, kwargs) (line 45)
        len_call_result_88 = invoke(stypy.reporting.localization.Localization(__file__, 45, 27), len_85, *[V_86], **kwargs_87)
        
        # Applying the binary operator '*' (line 45)
        result_mul_89 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 24), '*', int_84, len_call_result_88)
        
        # Processing the call keyword arguments (line 45)
        kwargs_90 = {}
        # Getting the type of 'xrange' (line 45)
        xrange_83 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 17), 'xrange', False)
        # Calling xrange(args, kwargs) (line 45)
        xrange_call_result_91 = invoke(stypy.reporting.localization.Localization(__file__, 45, 17), xrange_83, *[result_mul_89], **kwargs_90)
        
        # Assigning a type to the variable 'xrange_call_result_91' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'xrange_call_result_91', xrange_call_result_91)
        # Testing if the for loop is going to be iterated (line 45)
        # Testing the type of a for loop iterable (line 45)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 45, 8), xrange_call_result_91)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 45, 8), xrange_call_result_91):
            # Getting the type of the for loop variable (line 45)
            for_loop_var_92 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 45, 8), xrange_call_result_91)
            # Assigning a type to the variable 'n' (line 45)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'n', for_loop_var_92)
            # SSA begins for a for statement (line 45)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Subscript to a Name (line 46):
            
            # Obtaining the type of the subscript
            
            # Call to randint(...): (line 46)
            # Processing the call arguments (line 46)
            int_95 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 33), 'int')
            
            # Call to len(...): (line 46)
            # Processing the call arguments (line 46)
            # Getting the type of 'V' (line 46)
            V_97 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 40), 'V', False)
            # Processing the call keyword arguments (line 46)
            kwargs_98 = {}
            # Getting the type of 'len' (line 46)
            len_96 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 36), 'len', False)
            # Calling len(args, kwargs) (line 46)
            len_call_result_99 = invoke(stypy.reporting.localization.Localization(__file__, 46, 36), len_96, *[V_97], **kwargs_98)
            
            int_100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 45), 'int')
            # Applying the binary operator '-' (line 46)
            result_sub_101 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 36), '-', len_call_result_99, int_100)
            
            # Processing the call keyword arguments (line 46)
            kwargs_102 = {}
            # Getting the type of 'random' (line 46)
            random_93 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 18), 'random', False)
            # Obtaining the member 'randint' of a type (line 46)
            randint_94 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 18), random_93, 'randint')
            # Calling randint(args, kwargs) (line 46)
            randint_call_result_103 = invoke(stypy.reporting.localization.Localization(__file__, 46, 18), randint_94, *[int_95, result_sub_101], **kwargs_102)
            
            # Getting the type of 'V' (line 46)
            V_104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 16), 'V')
            # Obtaining the member '__getitem__' of a type (line 46)
            getitem___105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 16), V_104, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 46)
            subscript_call_result_106 = invoke(stypy.reporting.localization.Localization(__file__, 46, 16), getitem___105, randint_call_result_103)
            
            # Assigning a type to the variable 'u' (line 46)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'u', subscript_call_result_106)
            
            # Getting the type of 'True' (line 47)
            True_107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 18), 'True')
            # Assigning a type to the variable 'True_107' (line 47)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'True_107', True_107)
            # Testing if the while is going to be iterated (line 47)
            # Testing the type of an if condition (line 47)
            is_suitable_condition(stypy.reporting.localization.Localization(__file__, 47, 12), True_107)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 47, 12), True_107):
                
                # Assigning a Subscript to a Name (line 48):
                
                # Obtaining the type of the subscript
                
                # Call to randint(...): (line 48)
                # Processing the call arguments (line 48)
                int_110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 37), 'int')
                
                # Call to len(...): (line 48)
                # Processing the call arguments (line 48)
                # Getting the type of 'V' (line 48)
                V_112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 44), 'V', False)
                # Processing the call keyword arguments (line 48)
                kwargs_113 = {}
                # Getting the type of 'len' (line 48)
                len_111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 40), 'len', False)
                # Calling len(args, kwargs) (line 48)
                len_call_result_114 = invoke(stypy.reporting.localization.Localization(__file__, 48, 40), len_111, *[V_112], **kwargs_113)
                
                int_115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 49), 'int')
                # Applying the binary operator '-' (line 48)
                result_sub_116 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 40), '-', len_call_result_114, int_115)
                
                # Processing the call keyword arguments (line 48)
                kwargs_117 = {}
                # Getting the type of 'random' (line 48)
                random_108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 22), 'random', False)
                # Obtaining the member 'randint' of a type (line 48)
                randint_109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 22), random_108, 'randint')
                # Calling randint(args, kwargs) (line 48)
                randint_call_result_118 = invoke(stypy.reporting.localization.Localization(__file__, 48, 22), randint_109, *[int_110, result_sub_116], **kwargs_117)
                
                # Getting the type of 'V' (line 48)
                V_119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 20), 'V')
                # Obtaining the member '__getitem__' of a type (line 48)
                getitem___120 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 20), V_119, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 48)
                subscript_call_result_121 = invoke(stypy.reporting.localization.Localization(__file__, 48, 20), getitem___120, randint_call_result_118)
                
                # Assigning a type to the variable 'v' (line 48)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 16), 'v', subscript_call_result_121)
                
                # Getting the type of 'v' (line 49)
                v_122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 19), 'v')
                # Getting the type of 'u' (line 49)
                u_123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 28), 'u')
                # Applying the binary operator 'isnot' (line 49)
                result_is_not_124 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 19), 'isnot', v_122, u_123)
                
                # Testing if the type of an if condition is none (line 49)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 49, 16), result_is_not_124):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 49)
                    if_condition_125 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 16), result_is_not_124)
                    # Assigning a type to the variable 'if_condition_125' (line 49)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'if_condition_125', if_condition_125)
                    # SSA begins for if statement (line 49)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    # SSA join for if statement (line 49)
                    module_type_store = module_type_store.join_ssa_context()
                    


            
            
            # Call to append(...): (line 51)
            # Processing the call arguments (line 51)
            
            # Call to Edge(...): (line 51)
            # Processing the call arguments (line 51)
            # Getting the type of 'u' (line 51)
            u_129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 26), 'u', False)
            # Getting the type of 'v' (line 51)
            v_130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 29), 'v', False)
            
            # Call to uniform(...): (line 51)
            # Processing the call arguments (line 51)
            int_133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 47), 'int')
            int_134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 51), 'int')
            # Processing the call keyword arguments (line 51)
            kwargs_135 = {}
            # Getting the type of 'random' (line 51)
            random_131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 32), 'random', False)
            # Obtaining the member 'uniform' of a type (line 51)
            uniform_132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 32), random_131, 'uniform')
            # Calling uniform(args, kwargs) (line 51)
            uniform_call_result_136 = invoke(stypy.reporting.localization.Localization(__file__, 51, 32), uniform_132, *[int_133, int_134], **kwargs_135)
            
            # Processing the call keyword arguments (line 51)
            kwargs_137 = {}
            # Getting the type of 'Edge' (line 51)
            Edge_128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 21), 'Edge', False)
            # Calling Edge(args, kwargs) (line 51)
            Edge_call_result_138 = invoke(stypy.reporting.localization.Localization(__file__, 51, 21), Edge_128, *[u_129, v_130, uniform_call_result_136], **kwargs_137)
            
            # Processing the call keyword arguments (line 51)
            kwargs_139 = {}
            # Getting the type of 'E' (line 51)
            E_126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'E', False)
            # Obtaining the member 'append' of a type (line 51)
            append_127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 12), E_126, 'append')
            # Calling append(args, kwargs) (line 51)
            append_call_result_140 = invoke(stypy.reporting.localization.Localization(__file__, 51, 12), append_127, *[Edge_call_result_138], **kwargs_139)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # Assigning a Name to a Attribute (line 53):
        # Getting the type of 'V' (line 53)
        V_141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 17), 'V')
        # Getting the type of 'self' (line 53)
        self_142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'self')
        # Setting the type of the member 'V' of a type (line 53)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), self_142, 'V', V_141)
        
        # Assigning a Name to a Attribute (line 54):
        # Getting the type of 'E' (line 54)
        E_143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 17), 'E')
        # Getting the type of 'self' (line 54)
        self_144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'self')
        # Setting the type of the member 'E' of a type (line 54)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), self_144, 'E', E_143)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def distance(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'distance'
        module_type_store = module_type_store.open_function_context('distance', 56, 4, False)
        # Assigning a type to the variable 'self' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Graph.distance.__dict__.__setitem__('stypy_localization', localization)
        Graph.distance.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Graph.distance.__dict__.__setitem__('stypy_type_store', module_type_store)
        Graph.distance.__dict__.__setitem__('stypy_function_name', 'Graph.distance')
        Graph.distance.__dict__.__setitem__('stypy_param_names_list', ['s', 'S'])
        Graph.distance.__dict__.__setitem__('stypy_varargs_param_name', None)
        Graph.distance.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Graph.distance.__dict__.__setitem__('stypy_call_defaults', defaults)
        Graph.distance.__dict__.__setitem__('stypy_call_varargs', varargs)
        Graph.distance.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Graph.distance.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Graph.distance', ['s', 'S'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'distance', localization, ['s', 'S'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'distance(...)' code ##################

        
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'G' (line 57)
        G_158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 32), 'G')
        # Obtaining the member 'E' of a type (line 57)
        E_159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 32), G_158, 'E')
        comprehension_160 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 21), E_159)
        # Assigning a type to the variable 'e' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 21), 'e', comprehension_160)
        
        # Evaluating a boolean operation
        
        # Getting the type of 'e' (line 57)
        e_146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 39), 'e')
        # Obtaining the member 'u' of a type (line 57)
        u_147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 39), e_146, 'u')
        # Getting the type of 's' (line 57)
        s_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 46), 's')
        # Applying the binary operator '==' (line 57)
        result_eq_149 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 39), '==', u_147, s_148)
        
        
        # Getting the type of 'e' (line 57)
        e_150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 52), 'e')
        # Obtaining the member 'v' of a type (line 57)
        v_151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 52), e_150, 'v')
        
        # Obtaining the type of the subscript
        int_152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 61), 'int')
        # Getting the type of 'S' (line 57)
        S_153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 59), 'S')
        # Obtaining the member '__getitem__' of a type (line 57)
        getitem___154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 59), S_153, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 57)
        subscript_call_result_155 = invoke(stypy.reporting.localization.Localization(__file__, 57, 59), getitem___154, int_152)
        
        # Applying the binary operator '==' (line 57)
        result_eq_156 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 52), '==', v_151, subscript_call_result_155)
        
        # Applying the binary operator 'and' (line 57)
        result_and_keyword_157 = python_operator(stypy.reporting.localization.Localization(__file__, 57, 39), 'and', result_eq_149, result_eq_156)
        
        # Getting the type of 'e' (line 57)
        e_145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 21), 'e')
        list_161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 21), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 21), list_161, e_145)
        # Assigning a type to the variable 'list_161' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'list_161', list_161)
        # Testing if the for loop is going to be iterated (line 57)
        # Testing the type of a for loop iterable (line 57)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 57, 8), list_161)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 57, 8), list_161):
            # Getting the type of the for loop variable (line 57)
            for_loop_var_162 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 57, 8), list_161)
            # Assigning a type to the variable 'edge' (line 57)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'edge', for_loop_var_162)
            # SSA begins for a for statement (line 57)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Attribute to a Name (line 58):
            # Getting the type of 'edge' (line 58)
            edge_163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 16), 'edge')
            # Obtaining the member 'd' of a type (line 58)
            d_164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 16), edge_163, 'd')
            # Assigning a type to the variable 'd' (line 58)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'd', d_164)
            # SSA branch for the else part of a for statement (line 57)
            module_type_store.open_ssa_branch('for loop else')
            # Getting the type of 'AssertionError' (line 61)
            AssertionError_165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 18), 'AssertionError')
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 61, 12), AssertionError_165, 'raise parameter', BaseException)
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()
        else:
            # Getting the type of 'AssertionError' (line 61)
            AssertionError_165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 18), 'AssertionError')
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 61, 12), AssertionError_165, 'raise parameter', BaseException)

        
        
        
        # Call to zip(...): (line 63)
        # Processing the call arguments (line 63)
        
        # Obtaining the type of the subscript
        int_167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 27), 'int')
        slice_168 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 63, 24), None, int_167, None)
        # Getting the type of 'S' (line 63)
        S_169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 24), 'S', False)
        # Obtaining the member '__getitem__' of a type (line 63)
        getitem___170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 24), S_169, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 63)
        subscript_call_result_171 = invoke(stypy.reporting.localization.Localization(__file__, 63, 24), getitem___170, slice_168)
        
        
        # Obtaining the type of the subscript
        int_172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 34), 'int')
        slice_173 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 63, 32), int_172, None, None)
        # Getting the type of 'S' (line 63)
        S_174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 32), 'S', False)
        # Obtaining the member '__getitem__' of a type (line 63)
        getitem___175 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 32), S_174, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 63)
        subscript_call_result_176 = invoke(stypy.reporting.localization.Localization(__file__, 63, 32), getitem___175, slice_173)
        
        # Processing the call keyword arguments (line 63)
        kwargs_177 = {}
        # Getting the type of 'zip' (line 63)
        zip_166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 20), 'zip', False)
        # Calling zip(args, kwargs) (line 63)
        zip_call_result_178 = invoke(stypy.reporting.localization.Localization(__file__, 63, 20), zip_166, *[subscript_call_result_171, subscript_call_result_176], **kwargs_177)
        
        # Assigning a type to the variable 'zip_call_result_178' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'zip_call_result_178', zip_call_result_178)
        # Testing if the for loop is going to be iterated (line 63)
        # Testing the type of a for loop iterable (line 63)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 63, 8), zip_call_result_178)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 63, 8), zip_call_result_178):
            # Getting the type of the for loop variable (line 63)
            for_loop_var_179 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 63, 8), zip_call_result_178)
            # Assigning a type to the variable 'u' (line 63)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'u', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 8), for_loop_var_179, 2, 0))
            # Assigning a type to the variable 'v' (line 63)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 8), for_loop_var_179, 2, 1))
            # SSA begins for a for statement (line 63)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Calculating list comprehension
            # Calculating comprehension expression
            # Getting the type of 'G' (line 64)
            G_190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 36), 'G')
            # Obtaining the member 'E' of a type (line 64)
            E_191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 36), G_190, 'E')
            comprehension_192 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 25), E_191)
            # Assigning a type to the variable 'e' (line 64)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 25), 'e', comprehension_192)
            
            # Evaluating a boolean operation
            
            # Getting the type of 'e' (line 64)
            e_181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 43), 'e')
            # Obtaining the member 'u' of a type (line 64)
            u_182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 43), e_181, 'u')
            # Getting the type of 'u' (line 64)
            u_183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 50), 'u')
            # Applying the binary operator '==' (line 64)
            result_eq_184 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 43), '==', u_182, u_183)
            
            
            # Getting the type of 'e' (line 64)
            e_185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 56), 'e')
            # Obtaining the member 'v' of a type (line 64)
            v_186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 56), e_185, 'v')
            # Getting the type of 'v' (line 64)
            v_187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 63), 'v')
            # Applying the binary operator '==' (line 64)
            result_eq_188 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 56), '==', v_186, v_187)
            
            # Applying the binary operator 'and' (line 64)
            result_and_keyword_189 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 43), 'and', result_eq_184, result_eq_188)
            
            # Getting the type of 'e' (line 64)
            e_180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 25), 'e')
            list_193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 25), 'list')
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 25), list_193, e_180)
            # Assigning a type to the variable 'list_193' (line 64)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'list_193', list_193)
            # Testing if the for loop is going to be iterated (line 64)
            # Testing the type of a for loop iterable (line 64)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 64, 12), list_193)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 64, 12), list_193):
                # Getting the type of the for loop variable (line 64)
                for_loop_var_194 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 64, 12), list_193)
                # Assigning a type to the variable 'edge' (line 64)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'edge', for_loop_var_194)
                # SSA begins for a for statement (line 64)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Getting the type of 'd' (line 65)
                d_195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 16), 'd')
                # Getting the type of 'edge' (line 65)
                edge_196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 21), 'edge')
                # Obtaining the member 'd' of a type (line 65)
                d_197 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 21), edge_196, 'd')
                # Applying the binary operator '+=' (line 65)
                result_iadd_198 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 16), '+=', d_195, d_197)
                # Assigning a type to the variable 'd' (line 65)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 16), 'd', result_iadd_198)
                
                # SSA branch for the else part of a for statement (line 64)
                module_type_store.open_ssa_branch('for loop else')
                # Getting the type of 'AssertionError' (line 68)
                AssertionError_199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 22), 'AssertionError')
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 68, 16), AssertionError_199, 'raise parameter', BaseException)
                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()
            else:
                # Getting the type of 'AssertionError' (line 68)
                AssertionError_199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 22), 'AssertionError')
                ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 68, 16), AssertionError_199, 'raise parameter', BaseException)

            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'd' (line 69)
        d_200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 15), 'd')
        # Assigning a type to the variable 'stypy_return_type' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'stypy_return_type', d_200)
        
        # ################# End of 'distance(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'distance' in the type store
        # Getting the type of 'stypy_return_type' (line 56)
        stypy_return_type_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_201)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'distance'
        return stypy_return_type_201


# Assigning a type to the variable 'Graph' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'Graph', Graph)

@norecursion
def Extract_Min(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'Extract_Min'
    module_type_store = module_type_store.open_function_context('Extract_Min', 71, 0, False)
    
    # Passed parameters checking function
    Extract_Min.stypy_localization = localization
    Extract_Min.stypy_type_of_self = None
    Extract_Min.stypy_type_store = module_type_store
    Extract_Min.stypy_function_name = 'Extract_Min'
    Extract_Min.stypy_param_names_list = ['Q', 'd']
    Extract_Min.stypy_varargs_param_name = None
    Extract_Min.stypy_kwargs_param_name = None
    Extract_Min.stypy_call_defaults = defaults
    Extract_Min.stypy_call_varargs = varargs
    Extract_Min.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'Extract_Min', ['Q', 'd'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'Extract_Min', localization, ['Q', 'd'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'Extract_Min(...)' code ##################

    
    # Assigning a Name to a Name (line 72):
    # Getting the type of 'None' (line 72)
    None_202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'None')
    # Assigning a type to the variable 'm' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'm', None_202)
    
    # Assigning a Num to a Name (line 73):
    float_203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 9), 'float')
    # Assigning a type to the variable 'md' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'md', float_203)
    
    # Getting the type of 'Q' (line 74)
    Q_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 13), 'Q')
    # Assigning a type to the variable 'Q_204' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'Q_204', Q_204)
    # Testing if the for loop is going to be iterated (line 74)
    # Testing the type of a for loop iterable (line 74)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 74, 4), Q_204)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 74, 4), Q_204):
        # Getting the type of the for loop variable (line 74)
        for_loop_var_205 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 74, 4), Q_204)
        # Assigning a type to the variable 'u' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'u', for_loop_var_205)
        # SSA begins for a for statement (line 74)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Type idiom detected: calculating its left and rigth part (line 75)
        # Getting the type of 'm' (line 75)
        m_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 11), 'm')
        # Getting the type of 'None' (line 75)
        None_207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 16), 'None')
        
        (may_be_208, more_types_in_union_209) = may_be_none(m_206, None_207)

        if may_be_208:

            if more_types_in_union_209:
                # Runtime conditional SSA (line 75)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Name (line 76):
            # Getting the type of 'u' (line 76)
            u_210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'u')
            # Assigning a type to the variable 'm' (line 76)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'm', u_210)
            
            # Assigning a Subscript to a Name (line 77):
            
            # Obtaining the type of the subscript
            # Getting the type of 'u' (line 77)
            u_211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 19), 'u')
            # Getting the type of 'd' (line 77)
            d_212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 17), 'd')
            # Obtaining the member '__getitem__' of a type (line 77)
            getitem___213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 17), d_212, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 77)
            subscript_call_result_214 = invoke(stypy.reporting.localization.Localization(__file__, 77, 17), getitem___213, u_211)
            
            # Assigning a type to the variable 'md' (line 77)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'md', subscript_call_result_214)

            if more_types_in_union_209:
                # Runtime conditional SSA for else branch (line 75)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_208) or more_types_in_union_209):
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'u' (line 79)
            u_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 17), 'u')
            # Getting the type of 'd' (line 79)
            d_216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 15), 'd')
            # Obtaining the member '__getitem__' of a type (line 79)
            getitem___217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 15), d_216, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 79)
            subscript_call_result_218 = invoke(stypy.reporting.localization.Localization(__file__, 79, 15), getitem___217, u_215)
            
            # Getting the type of 'md' (line 79)
            md_219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 22), 'md')
            # Applying the binary operator '<' (line 79)
            result_lt_220 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 15), '<', subscript_call_result_218, md_219)
            
            # Testing if the type of an if condition is none (line 79)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 79, 12), result_lt_220):
                pass
            else:
                
                # Testing the type of an if condition (line 79)
                if_condition_221 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 12), result_lt_220)
                # Assigning a type to the variable 'if_condition_221' (line 79)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'if_condition_221', if_condition_221)
                # SSA begins for if statement (line 79)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Subscript to a Name (line 80):
                
                # Obtaining the type of the subscript
                # Getting the type of 'u' (line 80)
                u_222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 23), 'u')
                # Getting the type of 'd' (line 80)
                d_223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 21), 'd')
                # Obtaining the member '__getitem__' of a type (line 80)
                getitem___224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 21), d_223, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 80)
                subscript_call_result_225 = invoke(stypy.reporting.localization.Localization(__file__, 80, 21), getitem___224, u_222)
                
                # Assigning a type to the variable 'md' (line 80)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 16), 'md', subscript_call_result_225)
                
                # Assigning a Name to a Name (line 81):
                # Getting the type of 'u' (line 81)
                u_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 20), 'u')
                # Assigning a type to the variable 'm' (line 81)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'm', u_226)
                # SSA join for if statement (line 79)
                module_type_store = module_type_store.join_ssa_context()
                


            if (may_be_208 and more_types_in_union_209):
                # SSA join for if statement (line 75)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Call to remove(...): (line 82)
    # Processing the call arguments (line 82)
    # Getting the type of 'm' (line 82)
    m_229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 13), 'm', False)
    # Processing the call keyword arguments (line 82)
    kwargs_230 = {}
    # Getting the type of 'Q' (line 82)
    Q_227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'Q', False)
    # Obtaining the member 'remove' of a type (line 82)
    remove_228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 4), Q_227, 'remove')
    # Calling remove(args, kwargs) (line 82)
    remove_call_result_231 = invoke(stypy.reporting.localization.Localization(__file__, 82, 4), remove_228, *[m_229], **kwargs_230)
    
    # Getting the type of 'm' (line 83)
    m_232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 11), 'm')
    # Assigning a type to the variable 'stypy_return_type' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'stypy_return_type', m_232)
    
    # ################# End of 'Extract_Min(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'Extract_Min' in the type store
    # Getting the type of 'stypy_return_type' (line 71)
    stypy_return_type_233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_233)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'Extract_Min'
    return stypy_return_type_233

# Assigning a type to the variable 'Extract_Min' (line 71)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'Extract_Min', Extract_Min)

@norecursion
def dijkstra(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'dijkstra'
    module_type_store = module_type_store.open_function_context('dijkstra', 85, 0, False)
    
    # Passed parameters checking function
    dijkstra.stypy_localization = localization
    dijkstra.stypy_type_of_self = None
    dijkstra.stypy_type_store = module_type_store
    dijkstra.stypy_function_name = 'dijkstra'
    dijkstra.stypy_param_names_list = ['G', 't', 's']
    dijkstra.stypy_varargs_param_name = None
    dijkstra.stypy_kwargs_param_name = None
    dijkstra.stypy_call_defaults = defaults
    dijkstra.stypy_call_varargs = varargs
    dijkstra.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'dijkstra', ['G', 't', 's'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'dijkstra', localization, ['G', 't', 's'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'dijkstra(...)' code ##################

    
    # Assigning a Dict to a Name (line 86):
    
    # Obtaining an instance of the builtin type 'dict' (line 86)
    dict_234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 8), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 86)
    
    # Assigning a type to the variable 'd' (line 86)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'd', dict_234)
    
    # Assigning a Dict to a Name (line 87):
    
    # Obtaining an instance of the builtin type 'dict' (line 87)
    dict_235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 15), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 87)
    
    # Assigning a type to the variable 'previous' (line 87)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'previous', dict_235)
    
    # Getting the type of 'G' (line 88)
    G_236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 13), 'G')
    # Obtaining the member 'V' of a type (line 88)
    V_237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 13), G_236, 'V')
    # Assigning a type to the variable 'V_237' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'V_237', V_237)
    # Testing if the for loop is going to be iterated (line 88)
    # Testing the type of a for loop iterable (line 88)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 88, 4), V_237)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 88, 4), V_237):
        # Getting the type of the for loop variable (line 88)
        for_loop_var_238 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 88, 4), V_237)
        # Assigning a type to the variable 'v' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'v', for_loop_var_238)
        # SSA begins for a for statement (line 88)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Num to a Subscript (line 89):
        float_239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 15), 'float')
        # Getting the type of 'd' (line 89)
        d_240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'd')
        # Getting the type of 'v' (line 89)
        v_241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 10), 'v')
        # Storing an element on a container (line 89)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 8), d_240, (v_241, float_239))
        
        # Assigning a Name to a Subscript (line 90):
        # Getting the type of 'None' (line 90)
        None_242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 22), 'None')
        # Getting the type of 'previous' (line 90)
        previous_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'previous')
        # Getting the type of 'v' (line 90)
        v_244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 17), 'v')
        # Storing an element on a container (line 90)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 8), previous_243, (v_244, None_242))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Deleting a member
    module_type_store.del_member(stypy.reporting.localization.Localization(__file__, 91, 4), module_type_store, 'v')
    
    # Assigning a Num to a Subscript (line 92):
    int_245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 11), 'int')
    # Getting the type of 'd' (line 92)
    d_246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'd')
    # Getting the type of 's' (line 92)
    s_247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 6), 's')
    # Storing an element on a container (line 92)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 4), d_246, (s_247, int_245))
    
    # Assigning a List to a Name (line 93):
    
    # Obtaining an instance of the builtin type 'list' (line 93)
    list_248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 93)
    
    # Assigning a type to the variable 'S' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'S', list_248)
    
    # Assigning a Call to a Name (line 94):
    
    # Call to list(...): (line 94)
    # Processing the call arguments (line 94)
    # Getting the type of 'G' (line 94)
    G_250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 13), 'G', False)
    # Obtaining the member 'V' of a type (line 94)
    V_251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 13), G_250, 'V')
    # Processing the call keyword arguments (line 94)
    kwargs_252 = {}
    # Getting the type of 'list' (line 94)
    list_249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'list', False)
    # Calling list(args, kwargs) (line 94)
    list_call_result_253 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), list_249, *[V_251], **kwargs_252)
    
    # Assigning a type to the variable 'Q' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'Q', list_call_result_253)
    
    # Getting the type of 'Q' (line 97)
    Q_254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 10), 'Q')
    # Assigning a type to the variable 'Q_254' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'Q_254', Q_254)
    # Testing if the while is going to be iterated (line 97)
    # Testing the type of an if condition (line 97)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 4), Q_254)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 97, 4), Q_254):
        # SSA begins for while statement (line 97)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Name (line 98):
        
        # Call to Extract_Min(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'Q' (line 98)
        Q_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 24), 'Q', False)
        # Getting the type of 'd' (line 98)
        d_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 27), 'd', False)
        # Processing the call keyword arguments (line 98)
        kwargs_258 = {}
        # Getting the type of 'Extract_Min' (line 98)
        Extract_Min_255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'Extract_Min', False)
        # Calling Extract_Min(args, kwargs) (line 98)
        Extract_Min_call_result_259 = invoke(stypy.reporting.localization.Localization(__file__, 98, 12), Extract_Min_255, *[Q_256, d_257], **kwargs_258)
        
        # Assigning a type to the variable 'u' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'u', Extract_Min_call_result_259)
        
        # Getting the type of 'u' (line 99)
        u_260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 11), 'u')
        # Getting the type of 't' (line 99)
        t_261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 't')
        # Applying the binary operator '==' (line 99)
        result_eq_262 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 11), '==', u_260, t_261)
        
        # Testing if the type of an if condition is none (line 99)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 99, 8), result_eq_262):
            pass
        else:
            
            # Testing the type of an if condition (line 99)
            if_condition_263 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 99, 8), result_eq_262)
            # Assigning a type to the variable 'if_condition_263' (line 99)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'if_condition_263', if_condition_263)
            # SSA begins for if statement (line 99)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # SSA join for if statement (line 99)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to append(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'u' (line 101)
        u_266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 17), 'u', False)
        # Processing the call keyword arguments (line 101)
        kwargs_267 = {}
        # Getting the type of 'S' (line 101)
        S_264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'S', False)
        # Obtaining the member 'append' of a type (line 101)
        append_265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), S_264, 'append')
        # Calling append(args, kwargs) (line 101)
        append_call_result_268 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), append_265, *[u_266], **kwargs_267)
        
        
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'G' (line 102)
        G_274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 32), 'G')
        # Obtaining the member 'E' of a type (line 102)
        E_275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 32), G_274, 'E')
        comprehension_276 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 21), E_275)
        # Assigning a type to the variable 'e' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 21), 'e', comprehension_276)
        
        # Getting the type of 'e' (line 102)
        e_270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 39), 'e')
        # Obtaining the member 'u' of a type (line 102)
        u_271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 39), e_270, 'u')
        # Getting the type of 'u' (line 102)
        u_272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 46), 'u')
        # Applying the binary operator '==' (line 102)
        result_eq_273 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 39), '==', u_271, u_272)
        
        # Getting the type of 'e' (line 102)
        e_269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 21), 'e')
        list_277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 21), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 21), list_277, e_269)
        # Assigning a type to the variable 'list_277' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'list_277', list_277)
        # Testing if the for loop is going to be iterated (line 102)
        # Testing the type of a for loop iterable (line 102)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 102, 8), list_277)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 102, 8), list_277):
            # Getting the type of the for loop variable (line 102)
            for_loop_var_278 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 102, 8), list_277)
            # Assigning a type to the variable 'edge' (line 102)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'edge', for_loop_var_278)
            # SSA begins for a for statement (line 102)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'u' (line 103)
            u_279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 17), 'u')
            # Getting the type of 'd' (line 103)
            d_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 15), 'd')
            # Obtaining the member '__getitem__' of a type (line 103)
            getitem___281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 15), d_280, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 103)
            subscript_call_result_282 = invoke(stypy.reporting.localization.Localization(__file__, 103, 15), getitem___281, u_279)
            
            # Getting the type of 'edge' (line 103)
            edge_283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 22), 'edge')
            # Obtaining the member 'd' of a type (line 103)
            d_284 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 22), edge_283, 'd')
            # Applying the binary operator '+' (line 103)
            result_add_285 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 15), '+', subscript_call_result_282, d_284)
            
            
            # Obtaining the type of the subscript
            # Getting the type of 'edge' (line 103)
            edge_286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 33), 'edge')
            # Obtaining the member 'v' of a type (line 103)
            v_287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 33), edge_286, 'v')
            # Getting the type of 'd' (line 103)
            d_288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 31), 'd')
            # Obtaining the member '__getitem__' of a type (line 103)
            getitem___289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 31), d_288, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 103)
            subscript_call_result_290 = invoke(stypy.reporting.localization.Localization(__file__, 103, 31), getitem___289, v_287)
            
            # Applying the binary operator '<' (line 103)
            result_lt_291 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 15), '<', result_add_285, subscript_call_result_290)
            
            # Testing if the type of an if condition is none (line 103)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 103, 12), result_lt_291):
                pass
            else:
                
                # Testing the type of an if condition (line 103)
                if_condition_292 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 12), result_lt_291)
                # Assigning a type to the variable 'if_condition_292' (line 103)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'if_condition_292', if_condition_292)
                # SSA begins for if statement (line 103)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a BinOp to a Subscript (line 104):
                
                # Obtaining the type of the subscript
                # Getting the type of 'u' (line 104)
                u_293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 30), 'u')
                # Getting the type of 'd' (line 104)
                d_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 28), 'd')
                # Obtaining the member '__getitem__' of a type (line 104)
                getitem___295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 28), d_294, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 104)
                subscript_call_result_296 = invoke(stypy.reporting.localization.Localization(__file__, 104, 28), getitem___295, u_293)
                
                # Getting the type of 'edge' (line 104)
                edge_297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 35), 'edge')
                # Obtaining the member 'd' of a type (line 104)
                d_298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 35), edge_297, 'd')
                # Applying the binary operator '+' (line 104)
                result_add_299 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 28), '+', subscript_call_result_296, d_298)
                
                # Getting the type of 'd' (line 104)
                d_300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'd')
                # Getting the type of 'edge' (line 104)
                edge_301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 18), 'edge')
                # Obtaining the member 'v' of a type (line 104)
                v_302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 18), edge_301, 'v')
                # Storing an element on a container (line 104)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 16), d_300, (v_302, result_add_299))
                
                # Assigning a Name to a Subscript (line 105):
                # Getting the type of 'u' (line 105)
                u_303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 35), 'u')
                # Getting the type of 'previous' (line 105)
                previous_304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'previous')
                # Getting the type of 'edge' (line 105)
                edge_305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 25), 'edge')
                # Obtaining the member 'v' of a type (line 105)
                v_306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 25), edge_305, 'v')
                # Storing an element on a container (line 105)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 16), previous_304, (v_306, u_303))
                # SSA join for if statement (line 103)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # SSA join for while statement (line 97)
        module_type_store = module_type_store.join_ssa_context()

    
    
    # Assigning a List to a Name (line 107):
    
    # Obtaining an instance of the builtin type 'list' (line 107)
    list_307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 8), 'list')
    # Adding type elements to the builtin type 'list' instance (line 107)
    
    # Assigning a type to the variable 'S' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'S', list_307)
    
    # Assigning a Name to a Name (line 108):
    # Getting the type of 't' (line 108)
    t_308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 't')
    # Assigning a type to the variable 'u' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'u', t_308)
    
    
    
    # Obtaining the type of the subscript
    # Getting the type of 'u' (line 109)
    u_309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 19), 'u')
    # Getting the type of 'previous' (line 109)
    previous_310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 10), 'previous')
    # Obtaining the member '__getitem__' of a type (line 109)
    getitem___311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 10), previous_310, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 109)
    subscript_call_result_312 = invoke(stypy.reporting.localization.Localization(__file__, 109, 10), getitem___311, u_309)
    
    # Getting the type of 'None' (line 109)
    None_313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 29), 'None')
    # Applying the binary operator 'isnot' (line 109)
    result_is_not_314 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 10), 'isnot', subscript_call_result_312, None_313)
    
    # Assigning a type to the variable 'result_is_not_314' (line 109)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'result_is_not_314', result_is_not_314)
    # Testing if the while is going to be iterated (line 109)
    # Testing the type of an if condition (line 109)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 109, 4), result_is_not_314)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 109, 4), result_is_not_314):
        # SSA begins for while statement (line 109)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Call to insert(...): (line 110)
        # Processing the call arguments (line 110)
        int_317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 17), 'int')
        # Getting the type of 'u' (line 110)
        u_318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 20), 'u', False)
        # Processing the call keyword arguments (line 110)
        kwargs_319 = {}
        # Getting the type of 'S' (line 110)
        S_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'S', False)
        # Obtaining the member 'insert' of a type (line 110)
        insert_316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 8), S_315, 'insert')
        # Calling insert(args, kwargs) (line 110)
        insert_call_result_320 = invoke(stypy.reporting.localization.Localization(__file__, 110, 8), insert_316, *[int_317, u_318], **kwargs_319)
        
        
        # Assigning a Subscript to a Name (line 111):
        
        # Obtaining the type of the subscript
        # Getting the type of 'u' (line 111)
        u_321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 21), 'u')
        # Getting the type of 'previous' (line 111)
        previous_322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'previous')
        # Obtaining the member '__getitem__' of a type (line 111)
        getitem___323 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 12), previous_322, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 111)
        subscript_call_result_324 = invoke(stypy.reporting.localization.Localization(__file__, 111, 12), getitem___323, u_321)
        
        # Assigning a type to the variable 'u' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'u', subscript_call_result_324)
        # SSA join for while statement (line 109)
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'S' (line 112)
    S_325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 11), 'S')
    # Assigning a type to the variable 'stypy_return_type' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'stypy_return_type', S_325)
    
    # ################# End of 'dijkstra(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'dijkstra' in the type store
    # Getting the type of 'stypy_return_type' (line 85)
    stypy_return_type_326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_326)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'dijkstra'
    return stypy_return_type_326

# Assigning a type to the variable 'dijkstra' (line 85)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'dijkstra', dijkstra)

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

    # Marking variables as global (line 115)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 115, 4), 'G')
    
    
    # Call to xrange(...): (line 117)
    # Processing the call arguments (line 117)
    int_328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 20), 'int')
    # Processing the call keyword arguments (line 117)
    kwargs_329 = {}
    # Getting the type of 'xrange' (line 117)
    xrange_327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 13), 'xrange', False)
    # Calling xrange(args, kwargs) (line 117)
    xrange_call_result_330 = invoke(stypy.reporting.localization.Localization(__file__, 117, 13), xrange_327, *[int_328], **kwargs_329)
    
    # Assigning a type to the variable 'xrange_call_result_330' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'xrange_call_result_330', xrange_call_result_330)
    # Testing if the for loop is going to be iterated (line 117)
    # Testing the type of a for loop iterable (line 117)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 117, 4), xrange_call_result_330)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 117, 4), xrange_call_result_330):
        # Getting the type of the for loop variable (line 117)
        for_loop_var_331 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 117, 4), xrange_call_result_330)
        # Assigning a type to the variable 'n' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'n', for_loop_var_331)
        # SSA begins for a for statement (line 117)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 118):
        
        # Call to Graph(...): (line 118)
        # Processing the call keyword arguments (line 118)
        kwargs_333 = {}
        # Getting the type of 'Graph' (line 118)
        Graph_332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'Graph', False)
        # Calling Graph(args, kwargs) (line 118)
        Graph_call_result_334 = invoke(stypy.reporting.localization.Localization(__file__, 118, 12), Graph_332, *[], **kwargs_333)
        
        # Assigning a type to the variable 'G' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'G', Graph_call_result_334)
        
        # Assigning a Subscript to a Name (line 119):
        
        # Obtaining the type of the subscript
        
        # Call to randint(...): (line 119)
        # Processing the call arguments (line 119)
        int_337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 31), 'int')
        
        # Call to len(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'G' (line 119)
        G_339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 38), 'G', False)
        # Obtaining the member 'V' of a type (line 119)
        V_340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 38), G_339, 'V')
        # Processing the call keyword arguments (line 119)
        kwargs_341 = {}
        # Getting the type of 'len' (line 119)
        len_338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 34), 'len', False)
        # Calling len(args, kwargs) (line 119)
        len_call_result_342 = invoke(stypy.reporting.localization.Localization(__file__, 119, 34), len_338, *[V_340], **kwargs_341)
        
        int_343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 45), 'int')
        # Applying the binary operator '-' (line 119)
        result_sub_344 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 34), '-', len_call_result_342, int_343)
        
        # Processing the call keyword arguments (line 119)
        kwargs_345 = {}
        # Getting the type of 'random' (line 119)
        random_335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 16), 'random', False)
        # Obtaining the member 'randint' of a type (line 119)
        randint_336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 16), random_335, 'randint')
        # Calling randint(args, kwargs) (line 119)
        randint_call_result_346 = invoke(stypy.reporting.localization.Localization(__file__, 119, 16), randint_336, *[int_337, result_sub_344], **kwargs_345)
        
        # Getting the type of 'G' (line 119)
        G_347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'G')
        # Obtaining the member 'V' of a type (line 119)
        V_348 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 12), G_347, 'V')
        # Obtaining the member '__getitem__' of a type (line 119)
        getitem___349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 12), V_348, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 119)
        subscript_call_result_350 = invoke(stypy.reporting.localization.Localization(__file__, 119, 12), getitem___349, randint_call_result_346)
        
        # Assigning a type to the variable 's' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 's', subscript_call_result_350)
        
        # Getting the type of 'True' (line 120)
        True_351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 14), 'True')
        # Assigning a type to the variable 'True_351' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'True_351', True_351)
        # Testing if the while is going to be iterated (line 120)
        # Testing the type of an if condition (line 120)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 120, 8), True_351)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 120, 8), True_351):
            
            # Assigning a Subscript to a Name (line 121):
            
            # Obtaining the type of the subscript
            
            # Call to randint(...): (line 121)
            # Processing the call arguments (line 121)
            int_354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 35), 'int')
            
            # Call to len(...): (line 121)
            # Processing the call arguments (line 121)
            # Getting the type of 'G' (line 121)
            G_356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 42), 'G', False)
            # Obtaining the member 'V' of a type (line 121)
            V_357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 42), G_356, 'V')
            # Processing the call keyword arguments (line 121)
            kwargs_358 = {}
            # Getting the type of 'len' (line 121)
            len_355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 38), 'len', False)
            # Calling len(args, kwargs) (line 121)
            len_call_result_359 = invoke(stypy.reporting.localization.Localization(__file__, 121, 38), len_355, *[V_357], **kwargs_358)
            
            int_360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 49), 'int')
            # Applying the binary operator '-' (line 121)
            result_sub_361 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 38), '-', len_call_result_359, int_360)
            
            # Processing the call keyword arguments (line 121)
            kwargs_362 = {}
            # Getting the type of 'random' (line 121)
            random_352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 20), 'random', False)
            # Obtaining the member 'randint' of a type (line 121)
            randint_353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 20), random_352, 'randint')
            # Calling randint(args, kwargs) (line 121)
            randint_call_result_363 = invoke(stypy.reporting.localization.Localization(__file__, 121, 20), randint_353, *[int_354, result_sub_361], **kwargs_362)
            
            # Getting the type of 'G' (line 121)
            G_364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 16), 'G')
            # Obtaining the member 'V' of a type (line 121)
            V_365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 16), G_364, 'V')
            # Obtaining the member '__getitem__' of a type (line 121)
            getitem___366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 16), V_365, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 121)
            subscript_call_result_367 = invoke(stypy.reporting.localization.Localization(__file__, 121, 16), getitem___366, randint_call_result_363)
            
            # Assigning a type to the variable 't' (line 121)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 't', subscript_call_result_367)
            
            # Getting the type of 't' (line 122)
            t_368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 15), 't')
            # Getting the type of 's' (line 122)
            s_369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 24), 's')
            # Applying the binary operator 'isnot' (line 122)
            result_is_not_370 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 15), 'isnot', t_368, s_369)
            
            # Testing if the type of an if condition is none (line 122)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 122, 12), result_is_not_370):
                pass
            else:
                
                # Testing the type of an if condition (line 122)
                if_condition_371 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 122, 12), result_is_not_370)
                # Assigning a type to the variable 'if_condition_371' (line 122)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'if_condition_371', if_condition_371)
                # SSA begins for if statement (line 122)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # SSA join for if statement (line 122)
                module_type_store = module_type_store.join_ssa_context()
                


        
        
        # Assigning a Call to a Name (line 124):
        
        # Call to dijkstra(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'G' (line 124)
        G_373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 21), 'G', False)
        # Getting the type of 't' (line 124)
        t_374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 24), 't', False)
        # Getting the type of 's' (line 124)
        s_375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 27), 's', False)
        # Processing the call keyword arguments (line 124)
        kwargs_376 = {}
        # Getting the type of 'dijkstra' (line 124)
        dijkstra_372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'dijkstra', False)
        # Calling dijkstra(args, kwargs) (line 124)
        dijkstra_call_result_377 = invoke(stypy.reporting.localization.Localization(__file__, 124, 12), dijkstra_372, *[G_373, t_374, s_375], **kwargs_376)
        
        # Assigning a type to the variable 'S' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'S', dijkstra_call_result_377)
        # Getting the type of 'S' (line 125)
        S_378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 11), 'S')
        # Testing if the type of an if condition is none (line 125)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 125, 8), S_378):
            pass
        else:
            
            # Testing the type of an if condition (line 125)
            if_condition_379 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 8), S_378)
            # Assigning a type to the variable 'if_condition_379' (line 125)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'if_condition_379', if_condition_379)
            # SSA begins for if statement (line 125)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to distance(...): (line 127)
            # Processing the call arguments (line 127)
            # Getting the type of 's' (line 127)
            s_382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 23), 's', False)
            # Getting the type of 'S' (line 127)
            S_383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 26), 'S', False)
            # Processing the call keyword arguments (line 127)
            kwargs_384 = {}
            # Getting the type of 'G' (line 127)
            G_380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'G', False)
            # Obtaining the member 'distance' of a type (line 127)
            distance_381 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 12), G_380, 'distance')
            # Calling distance(args, kwargs) (line 127)
            distance_call_result_385 = invoke(stypy.reporting.localization.Localization(__file__, 127, 12), distance_381, *[s_382, S_383], **kwargs_384)
            
            
            
            # Obtaining the type of the subscript
            int_386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 28), 'int')
            slice_387 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 128, 25), None, int_386, None)
            # Getting the type of 'S' (line 128)
            S_388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 25), 'S')
            # Obtaining the member '__getitem__' of a type (line 128)
            getitem___389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 25), S_388, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 128)
            subscript_call_result_390 = invoke(stypy.reporting.localization.Localization(__file__, 128, 25), getitem___389, slice_387)
            
            # Assigning a type to the variable 'subscript_call_result_390' (line 128)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'subscript_call_result_390', subscript_call_result_390)
            # Testing if the for loop is going to be iterated (line 128)
            # Testing the type of a for loop iterable (line 128)
            is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 128, 12), subscript_call_result_390)

            if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 128, 12), subscript_call_result_390):
                # Getting the type of the for loop variable (line 128)
                for_loop_var_391 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 128, 12), subscript_call_result_390)
                # Assigning a type to the variable 'inter' (line 128)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'inter', for_loop_var_391)
                # SSA begins for a for statement (line 128)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
                
                # Assigning a Call to a Name (line 129):
                
                # Call to dijkstra(...): (line 129)
                # Processing the call arguments (line 129)
                # Getting the type of 'G' (line 129)
                G_393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 30), 'G', False)
                # Getting the type of 't' (line 129)
                t_394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 33), 't', False)
                # Getting the type of 'inter' (line 129)
                inter_395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 36), 'inter', False)
                # Processing the call keyword arguments (line 129)
                kwargs_396 = {}
                # Getting the type of 'dijkstra' (line 129)
                dijkstra_392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 21), 'dijkstra', False)
                # Calling dijkstra(args, kwargs) (line 129)
                dijkstra_call_result_397 = invoke(stypy.reporting.localization.Localization(__file__, 129, 21), dijkstra_392, *[G_393, t_394, inter_395], **kwargs_396)
                
                # Assigning a type to the variable 'S1' (line 129)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'S1', dijkstra_call_result_397)
                
                # Call to distance(...): (line 131)
                # Processing the call arguments (line 131)
                # Getting the type of 'inter' (line 131)
                inter_400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 27), 'inter', False)
                # Getting the type of 'S1' (line 131)
                S1_401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 34), 'S1', False)
                # Processing the call keyword arguments (line 131)
                kwargs_402 = {}
                # Getting the type of 'G' (line 131)
                G_398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 16), 'G', False)
                # Obtaining the member 'distance' of a type (line 131)
                distance_399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 16), G_398, 'distance')
                # Calling distance(args, kwargs) (line 131)
                distance_call_result_403 = invoke(stypy.reporting.localization.Localization(__file__, 131, 16), distance_399, *[inter_400, S1_401], **kwargs_402)
                
                
                # Getting the type of 'S1' (line 132)
                S1_404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 19), 'S1')
                
                # Obtaining the type of the subscript
                
                # Call to len(...): (line 132)
                # Processing the call arguments (line 132)
                # Getting the type of 'S' (line 132)
                S_406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 33), 'S', False)
                # Processing the call keyword arguments (line 132)
                kwargs_407 = {}
                # Getting the type of 'len' (line 132)
                len_405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 29), 'len', False)
                # Calling len(args, kwargs) (line 132)
                len_call_result_408 = invoke(stypy.reporting.localization.Localization(__file__, 132, 29), len_405, *[S_406], **kwargs_407)
                
                
                # Call to len(...): (line 132)
                # Processing the call arguments (line 132)
                # Getting the type of 'S1' (line 132)
                S1_410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 42), 'S1', False)
                # Processing the call keyword arguments (line 132)
                kwargs_411 = {}
                # Getting the type of 'len' (line 132)
                len_409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 38), 'len', False)
                # Calling len(args, kwargs) (line 132)
                len_call_result_412 = invoke(stypy.reporting.localization.Localization(__file__, 132, 38), len_409, *[S1_410], **kwargs_411)
                
                # Applying the binary operator '-' (line 132)
                result_sub_413 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 29), '-', len_call_result_408, len_call_result_412)
                
                slice_414 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 132, 25), result_sub_413, None, None)
                # Getting the type of 'S' (line 132)
                S_415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 25), 'S')
                # Obtaining the member '__getitem__' of a type (line 132)
                getitem___416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 25), S_415, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 132)
                subscript_call_result_417 = invoke(stypy.reporting.localization.Localization(__file__, 132, 25), getitem___416, slice_414)
                
                # Applying the binary operator '!=' (line 132)
                result_ne_418 = python_operator(stypy.reporting.localization.Localization(__file__, 132, 19), '!=', S1_404, subscript_call_result_417)
                
                # Testing if the type of an if condition is none (line 132)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 132, 16), result_ne_418):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 132)
                    if_condition_419 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 132, 16), result_ne_418)
                    # Assigning a type to the variable 'if_condition_419' (line 132)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 16), 'if_condition_419', if_condition_419)
                    # SSA begins for if statement (line 132)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    pass
                    # SSA join for if statement (line 132)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for a for statement
                module_type_store = module_type_store.join_ssa_context()

            
            # SSA join for if statement (line 125)
            module_type_store = module_type_store.join_ssa_context()
            

        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'True' (line 134)
    True_420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 11), 'True')
    # Assigning a type to the variable 'stypy_return_type' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'stypy_return_type', True_420)
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 114)
    stypy_return_type_421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_421)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_421

# Assigning a type to the variable 'run' (line 114)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 0), 'run', run)

# Call to run(...): (line 136)
# Processing the call keyword arguments (line 136)
kwargs_423 = {}
# Getting the type of 'run' (line 136)
run_422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 0), 'run', False)
# Calling run(args, kwargs) (line 136)
run_call_result_424 = invoke(stypy.reporting.localization.Localization(__file__, 136, 0), run_422, *[], **kwargs_423)


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
