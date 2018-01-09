
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import absolute_import
2: 
3: import functools
4: from collections import namedtuple
5: from threading import RLock
6: 
7: _CacheInfo = namedtuple("CacheInfo", ["hits", "misses", "maxsize", "currsize"])
8: 
9: 
10: @functools.wraps(functools.update_wrapper)
11: def update_wrapper(wrapper,
12:                    wrapped,
13:                    assigned = functools.WRAPPER_ASSIGNMENTS,
14:                    updated = functools.WRAPPER_UPDATES):
15:     '''
16:     Patch two bugs in functools.update_wrapper.
17:     '''
18:     # workaround for http://bugs.python.org/issue3445
19:     assigned = tuple(attr for attr in assigned if hasattr(wrapped, attr))
20:     wrapper = functools.update_wrapper(wrapper, wrapped, assigned, updated)
21:     # workaround for https://bugs.python.org/issue17482
22:     wrapper.__wrapped__ = wrapped
23:     return wrapper
24: 
25: 
26: class _HashedSeq(list):
27:     __slots__ = 'hashvalue'
28: 
29:     def __init__(self, tup, hash=hash):
30:         self[:] = tup
31:         self.hashvalue = hash(tup)
32: 
33:     def __hash__(self):
34:         return self.hashvalue
35: 
36: 
37: def _make_key(args, kwds, typed,
38:               kwd_mark=(object(),),
39:               fasttypes=set([int, str, frozenset, type(None)]),
40:               sorted=sorted, tuple=tuple, type=type, len=len):
41:     'Make a cache key from optionally typed positional and keyword arguments'
42:     key = args
43:     if kwds:
44:         sorted_items = sorted(kwds.items())
45:         key += kwd_mark
46:         for item in sorted_items:
47:             key += item
48:     if typed:
49:         key += tuple(type(v) for v in args)
50:         if kwds:
51:             key += tuple(type(v) for k, v in sorted_items)
52:     elif len(key) == 1 and type(key[0]) in fasttypes:
53:         return key[0]
54:     return _HashedSeq(key)
55: 
56: 
57: def lru_cache(maxsize=100, typed=False):
58:     '''Least-recently-used cache decorator.
59: 
60:     If *maxsize* is set to None, the LRU features are disabled and the cache
61:     can grow without bound.
62: 
63:     If *typed* is True, arguments of different types will be cached separately.
64:     For example, f(3.0) and f(3) will be treated as distinct calls with
65:     distinct results.
66: 
67:     Arguments to the cached function must be hashable.
68: 
69:     View the cache statistics named tuple (hits, misses, maxsize, currsize) with
70:     f.cache_info().  Clear the cache and statistics with f.cache_clear().
71:     Access the underlying function with f.__wrapped__.
72: 
73:     See:  http://en.wikipedia.org/wiki/Cache_algorithms#Least_Recently_Used
74: 
75:     '''
76: 
77:     # Users should only access the lru_cache through its public API:
78:     #       cache_info, cache_clear, and f.__wrapped__
79:     # The internals of the lru_cache are encapsulated for thread safety and
80:     # to allow the implementation to change (including a possible C version).
81: 
82:     def decorating_function(user_function):
83: 
84:         cache = dict()
85:         stats = [0, 0]                  # make statistics updateable non-locally
86:         HITS, MISSES = 0, 1             # names for the stats fields
87:         make_key = _make_key
88:         cache_get = cache.get           # bound method to lookup key or return None
89:         _len = len                      # localize the global len() function
90:         lock = RLock()                  # because linkedlist updates aren't threadsafe
91:         root = []                       # root of the circular doubly linked list
92:         root[:] = [root, root, None, None]      # initialize by pointing to self
93:         nonlocal_root = [root]                  # make updateable non-locally
94:         PREV, NEXT, KEY, RESULT = 0, 1, 2, 3    # names for the link fields
95: 
96:         if maxsize == 0:
97: 
98:             def wrapper(*args, **kwds):
99:                 # no caching, just do a statistics update after a successful call
100:                 result = user_function(*args, **kwds)
101:                 stats[MISSES] += 1
102:                 return result
103: 
104:         elif maxsize is None:
105: 
106:             def wrapper(*args, **kwds):
107:                 # simple caching without ordering or size limit
108:                 key = make_key(args, kwds, typed)
109:                 result = cache_get(key, root)   # root used here as a unique not-found sentinel
110:                 if result is not root:
111:                     stats[HITS] += 1
112:                     return result
113:                 result = user_function(*args, **kwds)
114:                 cache[key] = result
115:                 stats[MISSES] += 1
116:                 return result
117: 
118:         else:
119: 
120:             def wrapper(*args, **kwds):
121:                 # size limited caching that tracks accesses by recency
122:                 key = make_key(args, kwds, typed) if kwds or typed else args
123:                 with lock:
124:                     link = cache_get(key)
125:                     if link is not None:
126:                         # record recent use of the key by moving it to the front of the list
127:                         root, = nonlocal_root
128:                         link_prev, link_next, key, result = link
129:                         link_prev[NEXT] = link_next
130:                         link_next[PREV] = link_prev
131:                         last = root[PREV]
132:                         last[NEXT] = root[PREV] = link
133:                         link[PREV] = last
134:                         link[NEXT] = root
135:                         stats[HITS] += 1
136:                         return result
137:                 result = user_function(*args, **kwds)
138:                 with lock:
139:                     root, = nonlocal_root
140:                     if key in cache:
141:                         # getting here means that this same key was added to the
142:                         # cache while the lock was released.  since the link
143:                         # update is already done, we need only return the
144:                         # computed result and update the count of misses.
145:                         pass
146:                     elif _len(cache) >= maxsize:
147:                         # use the old root to store the new key and result
148:                         oldroot = root
149:                         oldroot[KEY] = key
150:                         oldroot[RESULT] = result
151:                         # empty the oldest link and make it the new root
152:                         root = nonlocal_root[0] = oldroot[NEXT]
153:                         oldkey = root[KEY]
154:                         root[KEY] = root[RESULT] = None
155:                         # now update the cache dictionary for the new links
156:                         del cache[oldkey]
157:                         cache[key] = oldroot
158:                     else:
159:                         # put result in a new link at the front of the list
160:                         last = root[PREV]
161:                         link = [last, root, key, result]
162:                         last[NEXT] = root[PREV] = cache[key] = link
163:                     stats[MISSES] += 1
164:                 return result
165: 
166:         def cache_info():
167:             '''Report cache statistics'''
168:             with lock:
169:                 return _CacheInfo(stats[HITS], stats[MISSES], maxsize, len(cache))
170: 
171:         def cache_clear():
172:             '''Clear the cache and cache statistics'''
173:             with lock:
174:                 cache.clear()
175:                 root = nonlocal_root[0]
176:                 root[:] = [root, root, None, None]
177:                 stats[:] = [0, 0]
178: 
179:         wrapper.__wrapped__ = user_function
180:         wrapper.cache_info = cache_info
181:         wrapper.cache_clear = cache_clear
182:         return update_wrapper(wrapper, user_function)
183: 
184:     return decorating_function
185: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import functools' statement (line 3)
import functools

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'functools', functools, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from collections import namedtuple' statement (line 4)
try:
    from collections import namedtuple

except:
    namedtuple = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'collections', None, module_type_store, ['namedtuple'], [namedtuple])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from threading import RLock' statement (line 5)
try:
    from threading import RLock

except:
    RLock = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'threading', None, module_type_store, ['RLock'], [RLock])


# Assigning a Call to a Name (line 7):

# Assigning a Call to a Name (line 7):

# Call to namedtuple(...): (line 7)
# Processing the call arguments (line 7)
str_308170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 24), 'str', 'CacheInfo')

# Obtaining an instance of the builtin type 'list' (line 7)
list_308171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 37), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_308172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 38), 'str', 'hits')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 37), list_308171, str_308172)
# Adding element type (line 7)
str_308173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 46), 'str', 'misses')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 37), list_308171, str_308173)
# Adding element type (line 7)
str_308174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 56), 'str', 'maxsize')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 37), list_308171, str_308174)
# Adding element type (line 7)
str_308175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 67), 'str', 'currsize')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 37), list_308171, str_308175)

# Processing the call keyword arguments (line 7)
kwargs_308176 = {}
# Getting the type of 'namedtuple' (line 7)
namedtuple_308169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 13), 'namedtuple', False)
# Calling namedtuple(args, kwargs) (line 7)
namedtuple_call_result_308177 = invoke(stypy.reporting.localization.Localization(__file__, 7, 13), namedtuple_308169, *[str_308170, list_308171], **kwargs_308176)

# Assigning a type to the variable '_CacheInfo' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '_CacheInfo', namedtuple_call_result_308177)

@norecursion
def update_wrapper(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'functools' (line 13)
    functools_308178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 30), 'functools')
    # Obtaining the member 'WRAPPER_ASSIGNMENTS' of a type (line 13)
    WRAPPER_ASSIGNMENTS_308179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 30), functools_308178, 'WRAPPER_ASSIGNMENTS')
    # Getting the type of 'functools' (line 14)
    functools_308180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 29), 'functools')
    # Obtaining the member 'WRAPPER_UPDATES' of a type (line 14)
    WRAPPER_UPDATES_308181 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 29), functools_308180, 'WRAPPER_UPDATES')
    defaults = [WRAPPER_ASSIGNMENTS_308179, WRAPPER_UPDATES_308181]
    # Create a new context for function 'update_wrapper'
    module_type_store = module_type_store.open_function_context('update_wrapper', 10, 0, False)
    
    # Passed parameters checking function
    update_wrapper.stypy_localization = localization
    update_wrapper.stypy_type_of_self = None
    update_wrapper.stypy_type_store = module_type_store
    update_wrapper.stypy_function_name = 'update_wrapper'
    update_wrapper.stypy_param_names_list = ['wrapper', 'wrapped', 'assigned', 'updated']
    update_wrapper.stypy_varargs_param_name = None
    update_wrapper.stypy_kwargs_param_name = None
    update_wrapper.stypy_call_defaults = defaults
    update_wrapper.stypy_call_varargs = varargs
    update_wrapper.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'update_wrapper', ['wrapper', 'wrapped', 'assigned', 'updated'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'update_wrapper', localization, ['wrapper', 'wrapped', 'assigned', 'updated'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'update_wrapper(...)' code ##################

    str_308182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, (-1)), 'str', '\n    Patch two bugs in functools.update_wrapper.\n    ')
    
    # Assigning a Call to a Name (line 19):
    
    # Assigning a Call to a Name (line 19):
    
    # Call to tuple(...): (line 19)
    # Processing the call arguments (line 19)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 19, 21, True)
    # Calculating comprehension expression
    # Getting the type of 'assigned' (line 19)
    assigned_308190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 38), 'assigned', False)
    comprehension_308191 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 21), assigned_308190)
    # Assigning a type to the variable 'attr' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 21), 'attr', comprehension_308191)
    
    # Call to hasattr(...): (line 19)
    # Processing the call arguments (line 19)
    # Getting the type of 'wrapped' (line 19)
    wrapped_308186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 58), 'wrapped', False)
    # Getting the type of 'attr' (line 19)
    attr_308187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 67), 'attr', False)
    # Processing the call keyword arguments (line 19)
    kwargs_308188 = {}
    # Getting the type of 'hasattr' (line 19)
    hasattr_308185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 50), 'hasattr', False)
    # Calling hasattr(args, kwargs) (line 19)
    hasattr_call_result_308189 = invoke(stypy.reporting.localization.Localization(__file__, 19, 50), hasattr_308185, *[wrapped_308186, attr_308187], **kwargs_308188)
    
    # Getting the type of 'attr' (line 19)
    attr_308184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 21), 'attr', False)
    list_308192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 21), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 21), list_308192, attr_308184)
    # Processing the call keyword arguments (line 19)
    kwargs_308193 = {}
    # Getting the type of 'tuple' (line 19)
    tuple_308183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 15), 'tuple', False)
    # Calling tuple(args, kwargs) (line 19)
    tuple_call_result_308194 = invoke(stypy.reporting.localization.Localization(__file__, 19, 15), tuple_308183, *[list_308192], **kwargs_308193)
    
    # Assigning a type to the variable 'assigned' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'assigned', tuple_call_result_308194)
    
    # Assigning a Call to a Name (line 20):
    
    # Assigning a Call to a Name (line 20):
    
    # Call to update_wrapper(...): (line 20)
    # Processing the call arguments (line 20)
    # Getting the type of 'wrapper' (line 20)
    wrapper_308197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 39), 'wrapper', False)
    # Getting the type of 'wrapped' (line 20)
    wrapped_308198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 48), 'wrapped', False)
    # Getting the type of 'assigned' (line 20)
    assigned_308199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 57), 'assigned', False)
    # Getting the type of 'updated' (line 20)
    updated_308200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 67), 'updated', False)
    # Processing the call keyword arguments (line 20)
    kwargs_308201 = {}
    # Getting the type of 'functools' (line 20)
    functools_308195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 14), 'functools', False)
    # Obtaining the member 'update_wrapper' of a type (line 20)
    update_wrapper_308196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 14), functools_308195, 'update_wrapper')
    # Calling update_wrapper(args, kwargs) (line 20)
    update_wrapper_call_result_308202 = invoke(stypy.reporting.localization.Localization(__file__, 20, 14), update_wrapper_308196, *[wrapper_308197, wrapped_308198, assigned_308199, updated_308200], **kwargs_308201)
    
    # Assigning a type to the variable 'wrapper' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'wrapper', update_wrapper_call_result_308202)
    
    # Assigning a Name to a Attribute (line 22):
    
    # Assigning a Name to a Attribute (line 22):
    # Getting the type of 'wrapped' (line 22)
    wrapped_308203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 26), 'wrapped')
    # Getting the type of 'wrapper' (line 22)
    wrapper_308204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'wrapper')
    # Setting the type of the member '__wrapped__' of a type (line 22)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 4), wrapper_308204, '__wrapped__', wrapped_308203)
    # Getting the type of 'wrapper' (line 23)
    wrapper_308205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 11), 'wrapper')
    # Assigning a type to the variable 'stypy_return_type' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'stypy_return_type', wrapper_308205)
    
    # ################# End of 'update_wrapper(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'update_wrapper' in the type store
    # Getting the type of 'stypy_return_type' (line 10)
    stypy_return_type_308206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_308206)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'update_wrapper'
    return stypy_return_type_308206

# Assigning a type to the variable 'update_wrapper' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'update_wrapper', update_wrapper)
# Declaration of the '_HashedSeq' class
# Getting the type of 'list' (line 26)
list_308207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 17), 'list')

class _HashedSeq(list_308207, ):
    
    # Assigning a Str to a Name (line 27):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'hash' (line 29)
        hash_308208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 33), 'hash')
        defaults = [hash_308208]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 29, 4, False)
        # Assigning a type to the variable 'self' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_HashedSeq.__init__', ['tup', 'hash'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['tup', 'hash'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Subscript (line 30):
        
        # Assigning a Name to a Subscript (line 30):
        # Getting the type of 'tup' (line 30)
        tup_308209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 18), 'tup')
        # Getting the type of 'self' (line 30)
        self_308210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'self')
        slice_308211 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 30, 8), None, None, None)
        # Storing an element on a container (line 30)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 8), self_308210, (slice_308211, tup_308209))
        
        # Assigning a Call to a Attribute (line 31):
        
        # Assigning a Call to a Attribute (line 31):
        
        # Call to hash(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'tup' (line 31)
        tup_308213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 30), 'tup', False)
        # Processing the call keyword arguments (line 31)
        kwargs_308214 = {}
        # Getting the type of 'hash' (line 31)
        hash_308212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 25), 'hash', False)
        # Calling hash(args, kwargs) (line 31)
        hash_call_result_308215 = invoke(stypy.reporting.localization.Localization(__file__, 31, 25), hash_308212, *[tup_308213], **kwargs_308214)
        
        # Getting the type of 'self' (line 31)
        self_308216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'self')
        # Setting the type of the member 'hashvalue' of a type (line 31)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), self_308216, 'hashvalue', hash_call_result_308215)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__hash__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__hash__'
        module_type_store = module_type_store.open_function_context('__hash__', 33, 4, False)
        # Assigning a type to the variable 'self' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        _HashedSeq.stypy__hash__.__dict__.__setitem__('stypy_localization', localization)
        _HashedSeq.stypy__hash__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        _HashedSeq.stypy__hash__.__dict__.__setitem__('stypy_type_store', module_type_store)
        _HashedSeq.stypy__hash__.__dict__.__setitem__('stypy_function_name', '_HashedSeq.stypy__hash__')
        _HashedSeq.stypy__hash__.__dict__.__setitem__('stypy_param_names_list', [])
        _HashedSeq.stypy__hash__.__dict__.__setitem__('stypy_varargs_param_name', None)
        _HashedSeq.stypy__hash__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        _HashedSeq.stypy__hash__.__dict__.__setitem__('stypy_call_defaults', defaults)
        _HashedSeq.stypy__hash__.__dict__.__setitem__('stypy_call_varargs', varargs)
        _HashedSeq.stypy__hash__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        _HashedSeq.stypy__hash__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, '_HashedSeq.stypy__hash__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__hash__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__hash__(...)' code ##################

        # Getting the type of 'self' (line 34)
        self_308217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 15), 'self')
        # Obtaining the member 'hashvalue' of a type (line 34)
        hashvalue_308218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 15), self_308217, 'hashvalue')
        # Assigning a type to the variable 'stypy_return_type' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'stypy_return_type', hashvalue_308218)
        
        # ################# End of '__hash__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__hash__' in the type store
        # Getting the type of 'stypy_return_type' (line 33)
        stypy_return_type_308219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_308219)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__hash__'
        return stypy_return_type_308219


# Assigning a type to the variable '_HashedSeq' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), '_HashedSeq', _HashedSeq)

# Assigning a Str to a Name (line 27):
str_308220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 16), 'str', 'hashvalue')
# Getting the type of '_HashedSeq'
_HashedSeq_308221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), '_HashedSeq')
# Setting the type of the member '__slots__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), _HashedSeq_308221, '__slots__', str_308220)

@norecursion
def _make_key(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'tuple' (line 38)
    tuple_308222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 38)
    # Adding element type (line 38)
    
    # Call to object(...): (line 38)
    # Processing the call keyword arguments (line 38)
    kwargs_308224 = {}
    # Getting the type of 'object' (line 38)
    object_308223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 24), 'object', False)
    # Calling object(args, kwargs) (line 38)
    object_call_result_308225 = invoke(stypy.reporting.localization.Localization(__file__, 38, 24), object_308223, *[], **kwargs_308224)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 24), tuple_308222, object_call_result_308225)
    
    
    # Call to set(...): (line 39)
    # Processing the call arguments (line 39)
    
    # Obtaining an instance of the builtin type 'list' (line 39)
    list_308227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 28), 'list')
    # Adding type elements to the builtin type 'list' instance (line 39)
    # Adding element type (line 39)
    # Getting the type of 'int' (line 39)
    int_308228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 29), 'int', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 28), list_308227, int_308228)
    # Adding element type (line 39)
    # Getting the type of 'str' (line 39)
    str_308229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 34), 'str', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 28), list_308227, str_308229)
    # Adding element type (line 39)
    # Getting the type of 'frozenset' (line 39)
    frozenset_308230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 39), 'frozenset', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 28), list_308227, frozenset_308230)
    # Adding element type (line 39)
    
    # Call to type(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'None' (line 39)
    None_308232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 55), 'None', False)
    # Processing the call keyword arguments (line 39)
    kwargs_308233 = {}
    # Getting the type of 'type' (line 39)
    type_308231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 50), 'type', False)
    # Calling type(args, kwargs) (line 39)
    type_call_result_308234 = invoke(stypy.reporting.localization.Localization(__file__, 39, 50), type_308231, *[None_308232], **kwargs_308233)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 28), list_308227, type_call_result_308234)
    
    # Processing the call keyword arguments (line 39)
    kwargs_308235 = {}
    # Getting the type of 'set' (line 39)
    set_308226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 24), 'set', False)
    # Calling set(args, kwargs) (line 39)
    set_call_result_308236 = invoke(stypy.reporting.localization.Localization(__file__, 39, 24), set_308226, *[list_308227], **kwargs_308235)
    
    # Getting the type of 'sorted' (line 40)
    sorted_308237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 21), 'sorted')
    # Getting the type of 'tuple' (line 40)
    tuple_308238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 35), 'tuple')
    # Getting the type of 'type' (line 40)
    type_308239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 47), 'type')
    # Getting the type of 'len' (line 40)
    len_308240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 57), 'len')
    defaults = [tuple_308222, set_call_result_308236, sorted_308237, tuple_308238, type_308239, len_308240]
    # Create a new context for function '_make_key'
    module_type_store = module_type_store.open_function_context('_make_key', 37, 0, False)
    
    # Passed parameters checking function
    _make_key.stypy_localization = localization
    _make_key.stypy_type_of_self = None
    _make_key.stypy_type_store = module_type_store
    _make_key.stypy_function_name = '_make_key'
    _make_key.stypy_param_names_list = ['args', 'kwds', 'typed', 'kwd_mark', 'fasttypes', 'sorted', 'tuple', 'type', 'len']
    _make_key.stypy_varargs_param_name = None
    _make_key.stypy_kwargs_param_name = None
    _make_key.stypy_call_defaults = defaults
    _make_key.stypy_call_varargs = varargs
    _make_key.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_make_key', ['args', 'kwds', 'typed', 'kwd_mark', 'fasttypes', 'sorted', 'tuple', 'type', 'len'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_make_key', localization, ['args', 'kwds', 'typed', 'kwd_mark', 'fasttypes', 'sorted', 'tuple', 'type', 'len'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_make_key(...)' code ##################

    str_308241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 4), 'str', 'Make a cache key from optionally typed positional and keyword arguments')
    
    # Assigning a Name to a Name (line 42):
    
    # Assigning a Name to a Name (line 42):
    # Getting the type of 'args' (line 42)
    args_308242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 10), 'args')
    # Assigning a type to the variable 'key' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'key', args_308242)
    
    # Getting the type of 'kwds' (line 43)
    kwds_308243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 7), 'kwds')
    # Testing the type of an if condition (line 43)
    if_condition_308244 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 43, 4), kwds_308243)
    # Assigning a type to the variable 'if_condition_308244' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'if_condition_308244', if_condition_308244)
    # SSA begins for if statement (line 43)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 44):
    
    # Assigning a Call to a Name (line 44):
    
    # Call to sorted(...): (line 44)
    # Processing the call arguments (line 44)
    
    # Call to items(...): (line 44)
    # Processing the call keyword arguments (line 44)
    kwargs_308248 = {}
    # Getting the type of 'kwds' (line 44)
    kwds_308246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 30), 'kwds', False)
    # Obtaining the member 'items' of a type (line 44)
    items_308247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 30), kwds_308246, 'items')
    # Calling items(args, kwargs) (line 44)
    items_call_result_308249 = invoke(stypy.reporting.localization.Localization(__file__, 44, 30), items_308247, *[], **kwargs_308248)
    
    # Processing the call keyword arguments (line 44)
    kwargs_308250 = {}
    # Getting the type of 'sorted' (line 44)
    sorted_308245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 23), 'sorted', False)
    # Calling sorted(args, kwargs) (line 44)
    sorted_call_result_308251 = invoke(stypy.reporting.localization.Localization(__file__, 44, 23), sorted_308245, *[items_call_result_308249], **kwargs_308250)
    
    # Assigning a type to the variable 'sorted_items' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'sorted_items', sorted_call_result_308251)
    
    # Getting the type of 'key' (line 45)
    key_308252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'key')
    # Getting the type of 'kwd_mark' (line 45)
    kwd_mark_308253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'kwd_mark')
    # Applying the binary operator '+=' (line 45)
    result_iadd_308254 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 8), '+=', key_308252, kwd_mark_308253)
    # Assigning a type to the variable 'key' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'key', result_iadd_308254)
    
    
    # Getting the type of 'sorted_items' (line 46)
    sorted_items_308255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 20), 'sorted_items')
    # Testing the type of a for loop iterable (line 46)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 46, 8), sorted_items_308255)
    # Getting the type of the for loop variable (line 46)
    for_loop_var_308256 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 46, 8), sorted_items_308255)
    # Assigning a type to the variable 'item' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'item', for_loop_var_308256)
    # SSA begins for a for statement (line 46)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Getting the type of 'key' (line 47)
    key_308257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'key')
    # Getting the type of 'item' (line 47)
    item_308258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 19), 'item')
    # Applying the binary operator '+=' (line 47)
    result_iadd_308259 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 12), '+=', key_308257, item_308258)
    # Assigning a type to the variable 'key' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 12), 'key', result_iadd_308259)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 43)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'typed' (line 48)
    typed_308260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 7), 'typed')
    # Testing the type of an if condition (line 48)
    if_condition_308261 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 48, 4), typed_308260)
    # Assigning a type to the variable 'if_condition_308261' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'if_condition_308261', if_condition_308261)
    # SSA begins for if statement (line 48)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'key' (line 49)
    key_308262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'key')
    
    # Call to tuple(...): (line 49)
    # Processing the call arguments (line 49)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 49, 21, True)
    # Calculating comprehension expression
    # Getting the type of 'args' (line 49)
    args_308268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 38), 'args', False)
    comprehension_308269 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 21), args_308268)
    # Assigning a type to the variable 'v' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 21), 'v', comprehension_308269)
    
    # Call to type(...): (line 49)
    # Processing the call arguments (line 49)
    # Getting the type of 'v' (line 49)
    v_308265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 26), 'v', False)
    # Processing the call keyword arguments (line 49)
    kwargs_308266 = {}
    # Getting the type of 'type' (line 49)
    type_308264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 21), 'type', False)
    # Calling type(args, kwargs) (line 49)
    type_call_result_308267 = invoke(stypy.reporting.localization.Localization(__file__, 49, 21), type_308264, *[v_308265], **kwargs_308266)
    
    list_308270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 21), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 21), list_308270, type_call_result_308267)
    # Processing the call keyword arguments (line 49)
    kwargs_308271 = {}
    # Getting the type of 'tuple' (line 49)
    tuple_308263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 15), 'tuple', False)
    # Calling tuple(args, kwargs) (line 49)
    tuple_call_result_308272 = invoke(stypy.reporting.localization.Localization(__file__, 49, 15), tuple_308263, *[list_308270], **kwargs_308271)
    
    # Applying the binary operator '+=' (line 49)
    result_iadd_308273 = python_operator(stypy.reporting.localization.Localization(__file__, 49, 8), '+=', key_308262, tuple_call_result_308272)
    # Assigning a type to the variable 'key' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'key', result_iadd_308273)
    
    
    # Getting the type of 'kwds' (line 50)
    kwds_308274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 11), 'kwds')
    # Testing the type of an if condition (line 50)
    if_condition_308275 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 8), kwds_308274)
    # Assigning a type to the variable 'if_condition_308275' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'if_condition_308275', if_condition_308275)
    # SSA begins for if statement (line 50)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'key' (line 51)
    key_308276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'key')
    
    # Call to tuple(...): (line 51)
    # Processing the call arguments (line 51)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 51, 25, True)
    # Calculating comprehension expression
    # Getting the type of 'sorted_items' (line 51)
    sorted_items_308282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 45), 'sorted_items', False)
    comprehension_308283 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 25), sorted_items_308282)
    # Assigning a type to the variable 'k' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 25), 'k', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 25), comprehension_308283))
    # Assigning a type to the variable 'v' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 25), 'v', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 25), comprehension_308283))
    
    # Call to type(...): (line 51)
    # Processing the call arguments (line 51)
    # Getting the type of 'v' (line 51)
    v_308279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 30), 'v', False)
    # Processing the call keyword arguments (line 51)
    kwargs_308280 = {}
    # Getting the type of 'type' (line 51)
    type_308278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 25), 'type', False)
    # Calling type(args, kwargs) (line 51)
    type_call_result_308281 = invoke(stypy.reporting.localization.Localization(__file__, 51, 25), type_308278, *[v_308279], **kwargs_308280)
    
    list_308284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 25), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 25), list_308284, type_call_result_308281)
    # Processing the call keyword arguments (line 51)
    kwargs_308285 = {}
    # Getting the type of 'tuple' (line 51)
    tuple_308277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 19), 'tuple', False)
    # Calling tuple(args, kwargs) (line 51)
    tuple_call_result_308286 = invoke(stypy.reporting.localization.Localization(__file__, 51, 19), tuple_308277, *[list_308284], **kwargs_308285)
    
    # Applying the binary operator '+=' (line 51)
    result_iadd_308287 = python_operator(stypy.reporting.localization.Localization(__file__, 51, 12), '+=', key_308276, tuple_call_result_308286)
    # Assigning a type to the variable 'key' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'key', result_iadd_308287)
    
    # SSA join for if statement (line 50)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 48)
    module_type_store.open_ssa_branch('else')
    
    
    # Evaluating a boolean operation
    
    
    # Call to len(...): (line 52)
    # Processing the call arguments (line 52)
    # Getting the type of 'key' (line 52)
    key_308289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 13), 'key', False)
    # Processing the call keyword arguments (line 52)
    kwargs_308290 = {}
    # Getting the type of 'len' (line 52)
    len_308288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 9), 'len', False)
    # Calling len(args, kwargs) (line 52)
    len_call_result_308291 = invoke(stypy.reporting.localization.Localization(__file__, 52, 9), len_308288, *[key_308289], **kwargs_308290)
    
    int_308292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 21), 'int')
    # Applying the binary operator '==' (line 52)
    result_eq_308293 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 9), '==', len_call_result_308291, int_308292)
    
    
    
    # Call to type(...): (line 52)
    # Processing the call arguments (line 52)
    
    # Obtaining the type of the subscript
    int_308295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 36), 'int')
    # Getting the type of 'key' (line 52)
    key_308296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 32), 'key', False)
    # Obtaining the member '__getitem__' of a type (line 52)
    getitem___308297 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 32), key_308296, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 52)
    subscript_call_result_308298 = invoke(stypy.reporting.localization.Localization(__file__, 52, 32), getitem___308297, int_308295)
    
    # Processing the call keyword arguments (line 52)
    kwargs_308299 = {}
    # Getting the type of 'type' (line 52)
    type_308294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 27), 'type', False)
    # Calling type(args, kwargs) (line 52)
    type_call_result_308300 = invoke(stypy.reporting.localization.Localization(__file__, 52, 27), type_308294, *[subscript_call_result_308298], **kwargs_308299)
    
    # Getting the type of 'fasttypes' (line 52)
    fasttypes_308301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 43), 'fasttypes')
    # Applying the binary operator 'in' (line 52)
    result_contains_308302 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 27), 'in', type_call_result_308300, fasttypes_308301)
    
    # Applying the binary operator 'and' (line 52)
    result_and_keyword_308303 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 9), 'and', result_eq_308293, result_contains_308302)
    
    # Testing the type of an if condition (line 52)
    if_condition_308304 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 52, 9), result_and_keyword_308303)
    # Assigning a type to the variable 'if_condition_308304' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 9), 'if_condition_308304', if_condition_308304)
    # SSA begins for if statement (line 52)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining the type of the subscript
    int_308305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 19), 'int')
    # Getting the type of 'key' (line 53)
    key_308306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 15), 'key')
    # Obtaining the member '__getitem__' of a type (line 53)
    getitem___308307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 15), key_308306, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 53)
    subscript_call_result_308308 = invoke(stypy.reporting.localization.Localization(__file__, 53, 15), getitem___308307, int_308305)
    
    # Assigning a type to the variable 'stypy_return_type' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'stypy_return_type', subscript_call_result_308308)
    # SSA join for if statement (line 52)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 48)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _HashedSeq(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 'key' (line 54)
    key_308310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 22), 'key', False)
    # Processing the call keyword arguments (line 54)
    kwargs_308311 = {}
    # Getting the type of '_HashedSeq' (line 54)
    _HashedSeq_308309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 11), '_HashedSeq', False)
    # Calling _HashedSeq(args, kwargs) (line 54)
    _HashedSeq_call_result_308312 = invoke(stypy.reporting.localization.Localization(__file__, 54, 11), _HashedSeq_308309, *[key_308310], **kwargs_308311)
    
    # Assigning a type to the variable 'stypy_return_type' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'stypy_return_type', _HashedSeq_call_result_308312)
    
    # ################# End of '_make_key(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_make_key' in the type store
    # Getting the type of 'stypy_return_type' (line 37)
    stypy_return_type_308313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_308313)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_make_key'
    return stypy_return_type_308313

# Assigning a type to the variable '_make_key' (line 37)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 0), '_make_key', _make_key)

@norecursion
def lru_cache(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_308314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 22), 'int')
    # Getting the type of 'False' (line 57)
    False_308315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 33), 'False')
    defaults = [int_308314, False_308315]
    # Create a new context for function 'lru_cache'
    module_type_store = module_type_store.open_function_context('lru_cache', 57, 0, False)
    
    # Passed parameters checking function
    lru_cache.stypy_localization = localization
    lru_cache.stypy_type_of_self = None
    lru_cache.stypy_type_store = module_type_store
    lru_cache.stypy_function_name = 'lru_cache'
    lru_cache.stypy_param_names_list = ['maxsize', 'typed']
    lru_cache.stypy_varargs_param_name = None
    lru_cache.stypy_kwargs_param_name = None
    lru_cache.stypy_call_defaults = defaults
    lru_cache.stypy_call_varargs = varargs
    lru_cache.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'lru_cache', ['maxsize', 'typed'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'lru_cache', localization, ['maxsize', 'typed'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'lru_cache(...)' code ##################

    str_308316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, (-1)), 'str', 'Least-recently-used cache decorator.\n\n    If *maxsize* is set to None, the LRU features are disabled and the cache\n    can grow without bound.\n\n    If *typed* is True, arguments of different types will be cached separately.\n    For example, f(3.0) and f(3) will be treated as distinct calls with\n    distinct results.\n\n    Arguments to the cached function must be hashable.\n\n    View the cache statistics named tuple (hits, misses, maxsize, currsize) with\n    f.cache_info().  Clear the cache and statistics with f.cache_clear().\n    Access the underlying function with f.__wrapped__.\n\n    See:  http://en.wikipedia.org/wiki/Cache_algorithms#Least_Recently_Used\n\n    ')

    @norecursion
    def decorating_function(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'decorating_function'
        module_type_store = module_type_store.open_function_context('decorating_function', 82, 4, False)
        
        # Passed parameters checking function
        decorating_function.stypy_localization = localization
        decorating_function.stypy_type_of_self = None
        decorating_function.stypy_type_store = module_type_store
        decorating_function.stypy_function_name = 'decorating_function'
        decorating_function.stypy_param_names_list = ['user_function']
        decorating_function.stypy_varargs_param_name = None
        decorating_function.stypy_kwargs_param_name = None
        decorating_function.stypy_call_defaults = defaults
        decorating_function.stypy_call_varargs = varargs
        decorating_function.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'decorating_function', ['user_function'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'decorating_function', localization, ['user_function'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'decorating_function(...)' code ##################

        
        # Assigning a Call to a Name (line 84):
        
        # Assigning a Call to a Name (line 84):
        
        # Call to dict(...): (line 84)
        # Processing the call keyword arguments (line 84)
        kwargs_308318 = {}
        # Getting the type of 'dict' (line 84)
        dict_308317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 16), 'dict', False)
        # Calling dict(args, kwargs) (line 84)
        dict_call_result_308319 = invoke(stypy.reporting.localization.Localization(__file__, 84, 16), dict_308317, *[], **kwargs_308318)
        
        # Assigning a type to the variable 'cache' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'cache', dict_call_result_308319)
        
        # Assigning a List to a Name (line 85):
        
        # Assigning a List to a Name (line 85):
        
        # Obtaining an instance of the builtin type 'list' (line 85)
        list_308320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 85)
        # Adding element type (line 85)
        int_308321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 16), list_308320, int_308321)
        # Adding element type (line 85)
        int_308322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 20), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 16), list_308320, int_308322)
        
        # Assigning a type to the variable 'stats' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'stats', list_308320)
        
        # Assigning a Tuple to a Tuple (line 86):
        
        # Assigning a Num to a Name (line 86):
        int_308323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 23), 'int')
        # Assigning a type to the variable 'tuple_assignment_308157' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'tuple_assignment_308157', int_308323)
        
        # Assigning a Num to a Name (line 86):
        int_308324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 26), 'int')
        # Assigning a type to the variable 'tuple_assignment_308158' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'tuple_assignment_308158', int_308324)
        
        # Assigning a Name to a Name (line 86):
        # Getting the type of 'tuple_assignment_308157' (line 86)
        tuple_assignment_308157_308325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'tuple_assignment_308157')
        # Assigning a type to the variable 'HITS' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'HITS', tuple_assignment_308157_308325)
        
        # Assigning a Name to a Name (line 86):
        # Getting the type of 'tuple_assignment_308158' (line 86)
        tuple_assignment_308158_308326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'tuple_assignment_308158')
        # Assigning a type to the variable 'MISSES' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 14), 'MISSES', tuple_assignment_308158_308326)
        
        # Assigning a Name to a Name (line 87):
        
        # Assigning a Name to a Name (line 87):
        # Getting the type of '_make_key' (line 87)
        _make_key_308327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 19), '_make_key')
        # Assigning a type to the variable 'make_key' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'make_key', _make_key_308327)
        
        # Assigning a Attribute to a Name (line 88):
        
        # Assigning a Attribute to a Name (line 88):
        # Getting the type of 'cache' (line 88)
        cache_308328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 20), 'cache')
        # Obtaining the member 'get' of a type (line 88)
        get_308329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 20), cache_308328, 'get')
        # Assigning a type to the variable 'cache_get' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'cache_get', get_308329)
        
        # Assigning a Name to a Name (line 89):
        
        # Assigning a Name to a Name (line 89):
        # Getting the type of 'len' (line 89)
        len_308330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 15), 'len')
        # Assigning a type to the variable '_len' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), '_len', len_308330)
        
        # Assigning a Call to a Name (line 90):
        
        # Assigning a Call to a Name (line 90):
        
        # Call to RLock(...): (line 90)
        # Processing the call keyword arguments (line 90)
        kwargs_308332 = {}
        # Getting the type of 'RLock' (line 90)
        RLock_308331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 15), 'RLock', False)
        # Calling RLock(args, kwargs) (line 90)
        RLock_call_result_308333 = invoke(stypy.reporting.localization.Localization(__file__, 90, 15), RLock_308331, *[], **kwargs_308332)
        
        # Assigning a type to the variable 'lock' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'lock', RLock_call_result_308333)
        
        # Assigning a List to a Name (line 91):
        
        # Assigning a List to a Name (line 91):
        
        # Obtaining an instance of the builtin type 'list' (line 91)
        list_308334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 91)
        
        # Assigning a type to the variable 'root' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'root', list_308334)
        
        # Assigning a List to a Subscript (line 92):
        
        # Assigning a List to a Subscript (line 92):
        
        # Obtaining an instance of the builtin type 'list' (line 92)
        list_308335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 92)
        # Adding element type (line 92)
        # Getting the type of 'root' (line 92)
        root_308336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 19), 'root')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 18), list_308335, root_308336)
        # Adding element type (line 92)
        # Getting the type of 'root' (line 92)
        root_308337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 25), 'root')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 18), list_308335, root_308337)
        # Adding element type (line 92)
        # Getting the type of 'None' (line 92)
        None_308338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 31), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 18), list_308335, None_308338)
        # Adding element type (line 92)
        # Getting the type of 'None' (line 92)
        None_308339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 37), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 18), list_308335, None_308339)
        
        # Getting the type of 'root' (line 92)
        root_308340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'root')
        slice_308341 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 92, 8), None, None, None)
        # Storing an element on a container (line 92)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 8), root_308340, (slice_308341, list_308335))
        
        # Assigning a List to a Name (line 93):
        
        # Assigning a List to a Name (line 93):
        
        # Obtaining an instance of the builtin type 'list' (line 93)
        list_308342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 93)
        # Adding element type (line 93)
        # Getting the type of 'root' (line 93)
        root_308343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 25), 'root')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 24), list_308342, root_308343)
        
        # Assigning a type to the variable 'nonlocal_root' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'nonlocal_root', list_308342)
        
        # Assigning a Tuple to a Tuple (line 94):
        
        # Assigning a Num to a Name (line 94):
        int_308344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 34), 'int')
        # Assigning a type to the variable 'tuple_assignment_308159' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'tuple_assignment_308159', int_308344)
        
        # Assigning a Num to a Name (line 94):
        int_308345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 37), 'int')
        # Assigning a type to the variable 'tuple_assignment_308160' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'tuple_assignment_308160', int_308345)
        
        # Assigning a Num to a Name (line 94):
        int_308346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 40), 'int')
        # Assigning a type to the variable 'tuple_assignment_308161' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'tuple_assignment_308161', int_308346)
        
        # Assigning a Num to a Name (line 94):
        int_308347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 43), 'int')
        # Assigning a type to the variable 'tuple_assignment_308162' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'tuple_assignment_308162', int_308347)
        
        # Assigning a Name to a Name (line 94):
        # Getting the type of 'tuple_assignment_308159' (line 94)
        tuple_assignment_308159_308348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'tuple_assignment_308159')
        # Assigning a type to the variable 'PREV' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'PREV', tuple_assignment_308159_308348)
        
        # Assigning a Name to a Name (line 94):
        # Getting the type of 'tuple_assignment_308160' (line 94)
        tuple_assignment_308160_308349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'tuple_assignment_308160')
        # Assigning a type to the variable 'NEXT' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 14), 'NEXT', tuple_assignment_308160_308349)
        
        # Assigning a Name to a Name (line 94):
        # Getting the type of 'tuple_assignment_308161' (line 94)
        tuple_assignment_308161_308350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'tuple_assignment_308161')
        # Assigning a type to the variable 'KEY' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 20), 'KEY', tuple_assignment_308161_308350)
        
        # Assigning a Name to a Name (line 94):
        # Getting the type of 'tuple_assignment_308162' (line 94)
        tuple_assignment_308162_308351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'tuple_assignment_308162')
        # Assigning a type to the variable 'RESULT' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 25), 'RESULT', tuple_assignment_308162_308351)
        
        
        # Getting the type of 'maxsize' (line 96)
        maxsize_308352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 11), 'maxsize')
        int_308353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 22), 'int')
        # Applying the binary operator '==' (line 96)
        result_eq_308354 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 11), '==', maxsize_308352, int_308353)
        
        # Testing the type of an if condition (line 96)
        if_condition_308355 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 8), result_eq_308354)
        # Assigning a type to the variable 'if_condition_308355' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'if_condition_308355', if_condition_308355)
        # SSA begins for if statement (line 96)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

        @norecursion
        def wrapper(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'wrapper'
            module_type_store = module_type_store.open_function_context('wrapper', 98, 12, False)
            
            # Passed parameters checking function
            wrapper.stypy_localization = localization
            wrapper.stypy_type_of_self = None
            wrapper.stypy_type_store = module_type_store
            wrapper.stypy_function_name = 'wrapper'
            wrapper.stypy_param_names_list = []
            wrapper.stypy_varargs_param_name = 'args'
            wrapper.stypy_kwargs_param_name = 'kwds'
            wrapper.stypy_call_defaults = defaults
            wrapper.stypy_call_varargs = varargs
            wrapper.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'wrapper', [], 'args', 'kwds', defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'wrapper', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'wrapper(...)' code ##################

            
            # Assigning a Call to a Name (line 100):
            
            # Assigning a Call to a Name (line 100):
            
            # Call to user_function(...): (line 100)
            # Getting the type of 'args' (line 100)
            args_308357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 40), 'args', False)
            # Processing the call keyword arguments (line 100)
            # Getting the type of 'kwds' (line 100)
            kwds_308358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 48), 'kwds', False)
            kwargs_308359 = {'kwds_308358': kwds_308358}
            # Getting the type of 'user_function' (line 100)
            user_function_308356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 25), 'user_function', False)
            # Calling user_function(args, kwargs) (line 100)
            user_function_call_result_308360 = invoke(stypy.reporting.localization.Localization(__file__, 100, 25), user_function_308356, *[args_308357], **kwargs_308359)
            
            # Assigning a type to the variable 'result' (line 100)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 16), 'result', user_function_call_result_308360)
            
            # Getting the type of 'stats' (line 101)
            stats_308361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 16), 'stats')
            
            # Obtaining the type of the subscript
            # Getting the type of 'MISSES' (line 101)
            MISSES_308362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 22), 'MISSES')
            # Getting the type of 'stats' (line 101)
            stats_308363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 16), 'stats')
            # Obtaining the member '__getitem__' of a type (line 101)
            getitem___308364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 16), stats_308363, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 101)
            subscript_call_result_308365 = invoke(stypy.reporting.localization.Localization(__file__, 101, 16), getitem___308364, MISSES_308362)
            
            int_308366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 33), 'int')
            # Applying the binary operator '+=' (line 101)
            result_iadd_308367 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 16), '+=', subscript_call_result_308365, int_308366)
            # Getting the type of 'stats' (line 101)
            stats_308368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 16), 'stats')
            # Getting the type of 'MISSES' (line 101)
            MISSES_308369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 22), 'MISSES')
            # Storing an element on a container (line 101)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 16), stats_308368, (MISSES_308369, result_iadd_308367))
            
            # Getting the type of 'result' (line 102)
            result_308370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 23), 'result')
            # Assigning a type to the variable 'stypy_return_type' (line 102)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'stypy_return_type', result_308370)
            
            # ################# End of 'wrapper(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'wrapper' in the type store
            # Getting the type of 'stypy_return_type' (line 98)
            stypy_return_type_308371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_308371)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'wrapper'
            return stypy_return_type_308371

        # Assigning a type to the variable 'wrapper' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'wrapper', wrapper)
        # SSA branch for the else part of an if statement (line 96)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 104)
        # Getting the type of 'maxsize' (line 104)
        maxsize_308372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 13), 'maxsize')
        # Getting the type of 'None' (line 104)
        None_308373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 24), 'None')
        
        (may_be_308374, more_types_in_union_308375) = may_be_none(maxsize_308372, None_308373)

        if may_be_308374:

            if more_types_in_union_308375:
                # Runtime conditional SSA (line 104)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store


            @norecursion
            def wrapper(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'wrapper'
                module_type_store = module_type_store.open_function_context('wrapper', 106, 12, False)
                
                # Passed parameters checking function
                wrapper.stypy_localization = localization
                wrapper.stypy_type_of_self = None
                wrapper.stypy_type_store = module_type_store
                wrapper.stypy_function_name = 'wrapper'
                wrapper.stypy_param_names_list = []
                wrapper.stypy_varargs_param_name = 'args'
                wrapper.stypy_kwargs_param_name = 'kwds'
                wrapper.stypy_call_defaults = defaults
                wrapper.stypy_call_varargs = varargs
                wrapper.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, 'wrapper', [], 'args', 'kwds', defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'wrapper', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'wrapper(...)' code ##################

                
                # Assigning a Call to a Name (line 108):
                
                # Assigning a Call to a Name (line 108):
                
                # Call to make_key(...): (line 108)
                # Processing the call arguments (line 108)
                # Getting the type of 'args' (line 108)
                args_308377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 31), 'args', False)
                # Getting the type of 'kwds' (line 108)
                kwds_308378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 37), 'kwds', False)
                # Getting the type of 'typed' (line 108)
                typed_308379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 43), 'typed', False)
                # Processing the call keyword arguments (line 108)
                kwargs_308380 = {}
                # Getting the type of 'make_key' (line 108)
                make_key_308376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 22), 'make_key', False)
                # Calling make_key(args, kwargs) (line 108)
                make_key_call_result_308381 = invoke(stypy.reporting.localization.Localization(__file__, 108, 22), make_key_308376, *[args_308377, kwds_308378, typed_308379], **kwargs_308380)
                
                # Assigning a type to the variable 'key' (line 108)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 16), 'key', make_key_call_result_308381)
                
                # Assigning a Call to a Name (line 109):
                
                # Assigning a Call to a Name (line 109):
                
                # Call to cache_get(...): (line 109)
                # Processing the call arguments (line 109)
                # Getting the type of 'key' (line 109)
                key_308383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 35), 'key', False)
                # Getting the type of 'root' (line 109)
                root_308384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 40), 'root', False)
                # Processing the call keyword arguments (line 109)
                kwargs_308385 = {}
                # Getting the type of 'cache_get' (line 109)
                cache_get_308382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 25), 'cache_get', False)
                # Calling cache_get(args, kwargs) (line 109)
                cache_get_call_result_308386 = invoke(stypy.reporting.localization.Localization(__file__, 109, 25), cache_get_308382, *[key_308383, root_308384], **kwargs_308385)
                
                # Assigning a type to the variable 'result' (line 109)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), 'result', cache_get_call_result_308386)
                
                
                # Getting the type of 'result' (line 110)
                result_308387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 19), 'result')
                # Getting the type of 'root' (line 110)
                root_308388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 33), 'root')
                # Applying the binary operator 'isnot' (line 110)
                result_is_not_308389 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 19), 'isnot', result_308387, root_308388)
                
                # Testing the type of an if condition (line 110)
                if_condition_308390 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 110, 16), result_is_not_308389)
                # Assigning a type to the variable 'if_condition_308390' (line 110)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'if_condition_308390', if_condition_308390)
                # SSA begins for if statement (line 110)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Getting the type of 'stats' (line 111)
                stats_308391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 20), 'stats')
                
                # Obtaining the type of the subscript
                # Getting the type of 'HITS' (line 111)
                HITS_308392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 26), 'HITS')
                # Getting the type of 'stats' (line 111)
                stats_308393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 20), 'stats')
                # Obtaining the member '__getitem__' of a type (line 111)
                getitem___308394 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 20), stats_308393, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 111)
                subscript_call_result_308395 = invoke(stypy.reporting.localization.Localization(__file__, 111, 20), getitem___308394, HITS_308392)
                
                int_308396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 35), 'int')
                # Applying the binary operator '+=' (line 111)
                result_iadd_308397 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 20), '+=', subscript_call_result_308395, int_308396)
                # Getting the type of 'stats' (line 111)
                stats_308398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 20), 'stats')
                # Getting the type of 'HITS' (line 111)
                HITS_308399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 26), 'HITS')
                # Storing an element on a container (line 111)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 20), stats_308398, (HITS_308399, result_iadd_308397))
                
                # Getting the type of 'result' (line 112)
                result_308400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 27), 'result')
                # Assigning a type to the variable 'stypy_return_type' (line 112)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 20), 'stypy_return_type', result_308400)
                # SSA join for if statement (line 110)
                module_type_store = module_type_store.join_ssa_context()
                
                
                # Assigning a Call to a Name (line 113):
                
                # Assigning a Call to a Name (line 113):
                
                # Call to user_function(...): (line 113)
                # Getting the type of 'args' (line 113)
                args_308402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 40), 'args', False)
                # Processing the call keyword arguments (line 113)
                # Getting the type of 'kwds' (line 113)
                kwds_308403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 48), 'kwds', False)
                kwargs_308404 = {'kwds_308403': kwds_308403}
                # Getting the type of 'user_function' (line 113)
                user_function_308401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 25), 'user_function', False)
                # Calling user_function(args, kwargs) (line 113)
                user_function_call_result_308405 = invoke(stypy.reporting.localization.Localization(__file__, 113, 25), user_function_308401, *[args_308402], **kwargs_308404)
                
                # Assigning a type to the variable 'result' (line 113)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 16), 'result', user_function_call_result_308405)
                
                # Assigning a Name to a Subscript (line 114):
                
                # Assigning a Name to a Subscript (line 114):
                # Getting the type of 'result' (line 114)
                result_308406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 29), 'result')
                # Getting the type of 'cache' (line 114)
                cache_308407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 16), 'cache')
                # Getting the type of 'key' (line 114)
                key_308408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 22), 'key')
                # Storing an element on a container (line 114)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 16), cache_308407, (key_308408, result_308406))
                
                # Getting the type of 'stats' (line 115)
                stats_308409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 16), 'stats')
                
                # Obtaining the type of the subscript
                # Getting the type of 'MISSES' (line 115)
                MISSES_308410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 22), 'MISSES')
                # Getting the type of 'stats' (line 115)
                stats_308411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 16), 'stats')
                # Obtaining the member '__getitem__' of a type (line 115)
                getitem___308412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 16), stats_308411, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 115)
                subscript_call_result_308413 = invoke(stypy.reporting.localization.Localization(__file__, 115, 16), getitem___308412, MISSES_308410)
                
                int_308414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 33), 'int')
                # Applying the binary operator '+=' (line 115)
                result_iadd_308415 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 16), '+=', subscript_call_result_308413, int_308414)
                # Getting the type of 'stats' (line 115)
                stats_308416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 16), 'stats')
                # Getting the type of 'MISSES' (line 115)
                MISSES_308417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 22), 'MISSES')
                # Storing an element on a container (line 115)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 16), stats_308416, (MISSES_308417, result_iadd_308415))
                
                # Getting the type of 'result' (line 116)
                result_308418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 23), 'result')
                # Assigning a type to the variable 'stypy_return_type' (line 116)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 16), 'stypy_return_type', result_308418)
                
                # ################# End of 'wrapper(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'wrapper' in the type store
                # Getting the type of 'stypy_return_type' (line 106)
                stypy_return_type_308419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_308419)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'wrapper'
                return stypy_return_type_308419

            # Assigning a type to the variable 'wrapper' (line 106)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'wrapper', wrapper)

            if more_types_in_union_308375:
                # Runtime conditional SSA for else branch (line 104)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_308374) or more_types_in_union_308375):

            @norecursion
            def wrapper(localization, *varargs, **kwargs):
                global module_type_store
                # Assign values to the parameters with defaults
                defaults = []
                # Create a new context for function 'wrapper'
                module_type_store = module_type_store.open_function_context('wrapper', 120, 12, False)
                
                # Passed parameters checking function
                wrapper.stypy_localization = localization
                wrapper.stypy_type_of_self = None
                wrapper.stypy_type_store = module_type_store
                wrapper.stypy_function_name = 'wrapper'
                wrapper.stypy_param_names_list = []
                wrapper.stypy_varargs_param_name = 'args'
                wrapper.stypy_kwargs_param_name = 'kwds'
                wrapper.stypy_call_defaults = defaults
                wrapper.stypy_call_varargs = varargs
                wrapper.stypy_call_kwargs = kwargs
                arguments = process_argument_values(localization, None, module_type_store, 'wrapper', [], 'args', 'kwds', defaults, varargs, kwargs)

                if is_error_type(arguments):
                    # Destroy the current context
                    module_type_store = module_type_store.close_function_context()
                    return arguments

                # Initialize method data
                init_call_information(module_type_store, 'wrapper', localization, [], arguments)
                
                # Default return type storage variable (SSA)
                # Assigning a type to the variable 'stypy_return_type'
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
                
                
                # ################# Begin of 'wrapper(...)' code ##################

                
                # Assigning a IfExp to a Name (line 122):
                
                # Assigning a IfExp to a Name (line 122):
                
                
                # Evaluating a boolean operation
                # Getting the type of 'kwds' (line 122)
                kwds_308420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 53), 'kwds')
                # Getting the type of 'typed' (line 122)
                typed_308421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 61), 'typed')
                # Applying the binary operator 'or' (line 122)
                result_or_keyword_308422 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 53), 'or', kwds_308420, typed_308421)
                
                # Testing the type of an if expression (line 122)
                is_suitable_condition(stypy.reporting.localization.Localization(__file__, 122, 22), result_or_keyword_308422)
                # SSA begins for if expression (line 122)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
                
                # Call to make_key(...): (line 122)
                # Processing the call arguments (line 122)
                # Getting the type of 'args' (line 122)
                args_308424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 31), 'args', False)
                # Getting the type of 'kwds' (line 122)
                kwds_308425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 37), 'kwds', False)
                # Getting the type of 'typed' (line 122)
                typed_308426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 43), 'typed', False)
                # Processing the call keyword arguments (line 122)
                kwargs_308427 = {}
                # Getting the type of 'make_key' (line 122)
                make_key_308423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 22), 'make_key', False)
                # Calling make_key(args, kwargs) (line 122)
                make_key_call_result_308428 = invoke(stypy.reporting.localization.Localization(__file__, 122, 22), make_key_308423, *[args_308424, kwds_308425, typed_308426], **kwargs_308427)
                
                # SSA branch for the else part of an if expression (line 122)
                module_type_store.open_ssa_branch('if expression else')
                # Getting the type of 'args' (line 122)
                args_308429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 72), 'args')
                # SSA join for if expression (line 122)
                module_type_store = module_type_store.join_ssa_context()
                if_exp_308430 = union_type.UnionType.add(make_key_call_result_308428, args_308429)
                
                # Assigning a type to the variable 'key' (line 122)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 16), 'key', if_exp_308430)
                # Getting the type of 'lock' (line 123)
                lock_308431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 21), 'lock')
                with_308432 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 123, 21), lock_308431, 'with parameter', '__enter__', '__exit__')

                if with_308432:
                    # Calling the __enter__ method to initiate a with section
                    # Obtaining the member '__enter__' of a type (line 123)
                    enter___308433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 21), lock_308431, '__enter__')
                    with_enter_308434 = invoke(stypy.reporting.localization.Localization(__file__, 123, 21), enter___308433)
                    
                    # Assigning a Call to a Name (line 124):
                    
                    # Assigning a Call to a Name (line 124):
                    
                    # Call to cache_get(...): (line 124)
                    # Processing the call arguments (line 124)
                    # Getting the type of 'key' (line 124)
                    key_308436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 37), 'key', False)
                    # Processing the call keyword arguments (line 124)
                    kwargs_308437 = {}
                    # Getting the type of 'cache_get' (line 124)
                    cache_get_308435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 27), 'cache_get', False)
                    # Calling cache_get(args, kwargs) (line 124)
                    cache_get_call_result_308438 = invoke(stypy.reporting.localization.Localization(__file__, 124, 27), cache_get_308435, *[key_308436], **kwargs_308437)
                    
                    # Assigning a type to the variable 'link' (line 124)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'link', cache_get_call_result_308438)
                    
                    # Type idiom detected: calculating its left and rigth part (line 125)
                    # Getting the type of 'link' (line 125)
                    link_308439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 20), 'link')
                    # Getting the type of 'None' (line 125)
                    None_308440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 35), 'None')
                    
                    (may_be_308441, more_types_in_union_308442) = may_not_be_none(link_308439, None_308440)

                    if may_be_308441:

                        if more_types_in_union_308442:
                            # Runtime conditional SSA (line 125)
                            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                        else:
                            module_type_store = module_type_store

                        
                        # Assigning a Name to a Tuple (line 127):
                        
                        # Assigning a Subscript to a Name (line 127):
                        
                        # Obtaining the type of the subscript
                        int_308443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 24), 'int')
                        # Getting the type of 'nonlocal_root' (line 127)
                        nonlocal_root_308444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 32), 'nonlocal_root')
                        # Obtaining the member '__getitem__' of a type (line 127)
                        getitem___308445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 24), nonlocal_root_308444, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 127)
                        subscript_call_result_308446 = invoke(stypy.reporting.localization.Localization(__file__, 127, 24), getitem___308445, int_308443)
                        
                        # Assigning a type to the variable 'tuple_var_assignment_308163' (line 127)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 24), 'tuple_var_assignment_308163', subscript_call_result_308446)
                        
                        # Assigning a Name to a Name (line 127):
                        # Getting the type of 'tuple_var_assignment_308163' (line 127)
                        tuple_var_assignment_308163_308447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 24), 'tuple_var_assignment_308163')
                        # Assigning a type to the variable 'root' (line 127)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 24), 'root', tuple_var_assignment_308163_308447)
                        
                        # Assigning a Name to a Tuple (line 128):
                        
                        # Assigning a Subscript to a Name (line 128):
                        
                        # Obtaining the type of the subscript
                        int_308448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 24), 'int')
                        # Getting the type of 'link' (line 128)
                        link_308449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 60), 'link')
                        # Obtaining the member '__getitem__' of a type (line 128)
                        getitem___308450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 24), link_308449, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
                        subscript_call_result_308451 = invoke(stypy.reporting.localization.Localization(__file__, 128, 24), getitem___308450, int_308448)
                        
                        # Assigning a type to the variable 'tuple_var_assignment_308164' (line 128)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 24), 'tuple_var_assignment_308164', subscript_call_result_308451)
                        
                        # Assigning a Subscript to a Name (line 128):
                        
                        # Obtaining the type of the subscript
                        int_308452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 24), 'int')
                        # Getting the type of 'link' (line 128)
                        link_308453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 60), 'link')
                        # Obtaining the member '__getitem__' of a type (line 128)
                        getitem___308454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 24), link_308453, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
                        subscript_call_result_308455 = invoke(stypy.reporting.localization.Localization(__file__, 128, 24), getitem___308454, int_308452)
                        
                        # Assigning a type to the variable 'tuple_var_assignment_308165' (line 128)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 24), 'tuple_var_assignment_308165', subscript_call_result_308455)
                        
                        # Assigning a Subscript to a Name (line 128):
                        
                        # Obtaining the type of the subscript
                        int_308456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 24), 'int')
                        # Getting the type of 'link' (line 128)
                        link_308457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 60), 'link')
                        # Obtaining the member '__getitem__' of a type (line 128)
                        getitem___308458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 24), link_308457, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
                        subscript_call_result_308459 = invoke(stypy.reporting.localization.Localization(__file__, 128, 24), getitem___308458, int_308456)
                        
                        # Assigning a type to the variable 'tuple_var_assignment_308166' (line 128)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 24), 'tuple_var_assignment_308166', subscript_call_result_308459)
                        
                        # Assigning a Subscript to a Name (line 128):
                        
                        # Obtaining the type of the subscript
                        int_308460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 24), 'int')
                        # Getting the type of 'link' (line 128)
                        link_308461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 60), 'link')
                        # Obtaining the member '__getitem__' of a type (line 128)
                        getitem___308462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 24), link_308461, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 128)
                        subscript_call_result_308463 = invoke(stypy.reporting.localization.Localization(__file__, 128, 24), getitem___308462, int_308460)
                        
                        # Assigning a type to the variable 'tuple_var_assignment_308167' (line 128)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 24), 'tuple_var_assignment_308167', subscript_call_result_308463)
                        
                        # Assigning a Name to a Name (line 128):
                        # Getting the type of 'tuple_var_assignment_308164' (line 128)
                        tuple_var_assignment_308164_308464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 24), 'tuple_var_assignment_308164')
                        # Assigning a type to the variable 'link_prev' (line 128)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 24), 'link_prev', tuple_var_assignment_308164_308464)
                        
                        # Assigning a Name to a Name (line 128):
                        # Getting the type of 'tuple_var_assignment_308165' (line 128)
                        tuple_var_assignment_308165_308465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 24), 'tuple_var_assignment_308165')
                        # Assigning a type to the variable 'link_next' (line 128)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 35), 'link_next', tuple_var_assignment_308165_308465)
                        
                        # Assigning a Name to a Name (line 128):
                        # Getting the type of 'tuple_var_assignment_308166' (line 128)
                        tuple_var_assignment_308166_308466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 24), 'tuple_var_assignment_308166')
                        # Assigning a type to the variable 'key' (line 128)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 46), 'key', tuple_var_assignment_308166_308466)
                        
                        # Assigning a Name to a Name (line 128):
                        # Getting the type of 'tuple_var_assignment_308167' (line 128)
                        tuple_var_assignment_308167_308467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 24), 'tuple_var_assignment_308167')
                        # Assigning a type to the variable 'result' (line 128)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 51), 'result', tuple_var_assignment_308167_308467)
                        
                        # Assigning a Name to a Subscript (line 129):
                        
                        # Assigning a Name to a Subscript (line 129):
                        # Getting the type of 'link_next' (line 129)
                        link_next_308468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 42), 'link_next')
                        # Getting the type of 'link_prev' (line 129)
                        link_prev_308469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 24), 'link_prev')
                        # Getting the type of 'NEXT' (line 129)
                        NEXT_308470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 34), 'NEXT')
                        # Storing an element on a container (line 129)
                        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 24), link_prev_308469, (NEXT_308470, link_next_308468))
                        
                        # Assigning a Name to a Subscript (line 130):
                        
                        # Assigning a Name to a Subscript (line 130):
                        # Getting the type of 'link_prev' (line 130)
                        link_prev_308471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 42), 'link_prev')
                        # Getting the type of 'link_next' (line 130)
                        link_next_308472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 24), 'link_next')
                        # Getting the type of 'PREV' (line 130)
                        PREV_308473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 34), 'PREV')
                        # Storing an element on a container (line 130)
                        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 24), link_next_308472, (PREV_308473, link_prev_308471))
                        
                        # Assigning a Subscript to a Name (line 131):
                        
                        # Assigning a Subscript to a Name (line 131):
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'PREV' (line 131)
                        PREV_308474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 36), 'PREV')
                        # Getting the type of 'root' (line 131)
                        root_308475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 31), 'root')
                        # Obtaining the member '__getitem__' of a type (line 131)
                        getitem___308476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 31), root_308475, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 131)
                        subscript_call_result_308477 = invoke(stypy.reporting.localization.Localization(__file__, 131, 31), getitem___308476, PREV_308474)
                        
                        # Assigning a type to the variable 'last' (line 131)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 24), 'last', subscript_call_result_308477)
                        
                        # Multiple assignment of 2 elements.
                        
                        # Assigning a Name to a Subscript (line 132):
                        # Getting the type of 'link' (line 132)
                        link_308478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 50), 'link')
                        # Getting the type of 'root' (line 132)
                        root_308479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 37), 'root')
                        # Getting the type of 'PREV' (line 132)
                        PREV_308480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 42), 'PREV')
                        # Storing an element on a container (line 132)
                        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 37), root_308479, (PREV_308480, link_308478))
                        
                        # Assigning a Subscript to a Subscript (line 132):
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'PREV' (line 132)
                        PREV_308481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 42), 'PREV')
                        # Getting the type of 'root' (line 132)
                        root_308482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 37), 'root')
                        # Obtaining the member '__getitem__' of a type (line 132)
                        getitem___308483 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 37), root_308482, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 132)
                        subscript_call_result_308484 = invoke(stypy.reporting.localization.Localization(__file__, 132, 37), getitem___308483, PREV_308481)
                        
                        # Getting the type of 'last' (line 132)
                        last_308485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 24), 'last')
                        # Getting the type of 'NEXT' (line 132)
                        NEXT_308486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 29), 'NEXT')
                        # Storing an element on a container (line 132)
                        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 132, 24), last_308485, (NEXT_308486, subscript_call_result_308484))
                        
                        # Assigning a Name to a Subscript (line 133):
                        
                        # Assigning a Name to a Subscript (line 133):
                        # Getting the type of 'last' (line 133)
                        last_308487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 37), 'last')
                        # Getting the type of 'link' (line 133)
                        link_308488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 24), 'link')
                        # Getting the type of 'PREV' (line 133)
                        PREV_308489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 29), 'PREV')
                        # Storing an element on a container (line 133)
                        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 24), link_308488, (PREV_308489, last_308487))
                        
                        # Assigning a Name to a Subscript (line 134):
                        
                        # Assigning a Name to a Subscript (line 134):
                        # Getting the type of 'root' (line 134)
                        root_308490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 37), 'root')
                        # Getting the type of 'link' (line 134)
                        link_308491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 24), 'link')
                        # Getting the type of 'NEXT' (line 134)
                        NEXT_308492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 29), 'NEXT')
                        # Storing an element on a container (line 134)
                        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 134, 24), link_308491, (NEXT_308492, root_308490))
                        
                        # Getting the type of 'stats' (line 135)
                        stats_308493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 24), 'stats')
                        
                        # Obtaining the type of the subscript
                        # Getting the type of 'HITS' (line 135)
                        HITS_308494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 30), 'HITS')
                        # Getting the type of 'stats' (line 135)
                        stats_308495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 24), 'stats')
                        # Obtaining the member '__getitem__' of a type (line 135)
                        getitem___308496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 24), stats_308495, '__getitem__')
                        # Calling the subscript (__getitem__) to obtain the elements type (line 135)
                        subscript_call_result_308497 = invoke(stypy.reporting.localization.Localization(__file__, 135, 24), getitem___308496, HITS_308494)
                        
                        int_308498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 39), 'int')
                        # Applying the binary operator '+=' (line 135)
                        result_iadd_308499 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 24), '+=', subscript_call_result_308497, int_308498)
                        # Getting the type of 'stats' (line 135)
                        stats_308500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 24), 'stats')
                        # Getting the type of 'HITS' (line 135)
                        HITS_308501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 30), 'HITS')
                        # Storing an element on a container (line 135)
                        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 24), stats_308500, (HITS_308501, result_iadd_308499))
                        
                        # Getting the type of 'result' (line 136)
                        result_308502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 31), 'result')
                        # Assigning a type to the variable 'stypy_return_type' (line 136)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 24), 'stypy_return_type', result_308502)

                        if more_types_in_union_308442:
                            # SSA join for if statement (line 125)
                            module_type_store = module_type_store.join_ssa_context()


                    
                    # Calling the __exit__ method to finish a with section
                    # Obtaining the member '__exit__' of a type (line 123)
                    exit___308503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 21), lock_308431, '__exit__')
                    with_exit_308504 = invoke(stypy.reporting.localization.Localization(__file__, 123, 21), exit___308503, None, None, None)

                
                # Assigning a Call to a Name (line 137):
                
                # Assigning a Call to a Name (line 137):
                
                # Call to user_function(...): (line 137)
                # Getting the type of 'args' (line 137)
                args_308506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 40), 'args', False)
                # Processing the call keyword arguments (line 137)
                # Getting the type of 'kwds' (line 137)
                kwds_308507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 48), 'kwds', False)
                kwargs_308508 = {'kwds_308507': kwds_308507}
                # Getting the type of 'user_function' (line 137)
                user_function_308505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 25), 'user_function', False)
                # Calling user_function(args, kwargs) (line 137)
                user_function_call_result_308509 = invoke(stypy.reporting.localization.Localization(__file__, 137, 25), user_function_308505, *[args_308506], **kwargs_308508)
                
                # Assigning a type to the variable 'result' (line 137)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 16), 'result', user_function_call_result_308509)
                # Getting the type of 'lock' (line 138)
                lock_308510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 21), 'lock')
                with_308511 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 138, 21), lock_308510, 'with parameter', '__enter__', '__exit__')

                if with_308511:
                    # Calling the __enter__ method to initiate a with section
                    # Obtaining the member '__enter__' of a type (line 138)
                    enter___308512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 21), lock_308510, '__enter__')
                    with_enter_308513 = invoke(stypy.reporting.localization.Localization(__file__, 138, 21), enter___308512)
                    
                    # Assigning a Name to a Tuple (line 139):
                    
                    # Assigning a Subscript to a Name (line 139):
                    
                    # Obtaining the type of the subscript
                    int_308514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 20), 'int')
                    # Getting the type of 'nonlocal_root' (line 139)
                    nonlocal_root_308515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 28), 'nonlocal_root')
                    # Obtaining the member '__getitem__' of a type (line 139)
                    getitem___308516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 20), nonlocal_root_308515, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 139)
                    subscript_call_result_308517 = invoke(stypy.reporting.localization.Localization(__file__, 139, 20), getitem___308516, int_308514)
                    
                    # Assigning a type to the variable 'tuple_var_assignment_308168' (line 139)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 20), 'tuple_var_assignment_308168', subscript_call_result_308517)
                    
                    # Assigning a Name to a Name (line 139):
                    # Getting the type of 'tuple_var_assignment_308168' (line 139)
                    tuple_var_assignment_308168_308518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 20), 'tuple_var_assignment_308168')
                    # Assigning a type to the variable 'root' (line 139)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 20), 'root', tuple_var_assignment_308168_308518)
                    
                    
                    # Getting the type of 'key' (line 140)
                    key_308519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 23), 'key')
                    # Getting the type of 'cache' (line 140)
                    cache_308520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 30), 'cache')
                    # Applying the binary operator 'in' (line 140)
                    result_contains_308521 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 23), 'in', key_308519, cache_308520)
                    
                    # Testing the type of an if condition (line 140)
                    if_condition_308522 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 140, 20), result_contains_308521)
                    # Assigning a type to the variable 'if_condition_308522' (line 140)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 20), 'if_condition_308522', if_condition_308522)
                    # SSA begins for if statement (line 140)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    pass
                    # SSA branch for the else part of an if statement (line 140)
                    module_type_store.open_ssa_branch('else')
                    
                    
                    
                    # Call to _len(...): (line 146)
                    # Processing the call arguments (line 146)
                    # Getting the type of 'cache' (line 146)
                    cache_308524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 30), 'cache', False)
                    # Processing the call keyword arguments (line 146)
                    kwargs_308525 = {}
                    # Getting the type of '_len' (line 146)
                    _len_308523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 25), '_len', False)
                    # Calling _len(args, kwargs) (line 146)
                    _len_call_result_308526 = invoke(stypy.reporting.localization.Localization(__file__, 146, 25), _len_308523, *[cache_308524], **kwargs_308525)
                    
                    # Getting the type of 'maxsize' (line 146)
                    maxsize_308527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 40), 'maxsize')
                    # Applying the binary operator '>=' (line 146)
                    result_ge_308528 = python_operator(stypy.reporting.localization.Localization(__file__, 146, 25), '>=', _len_call_result_308526, maxsize_308527)
                    
                    # Testing the type of an if condition (line 146)
                    if_condition_308529 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 146, 25), result_ge_308528)
                    # Assigning a type to the variable 'if_condition_308529' (line 146)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 25), 'if_condition_308529', if_condition_308529)
                    # SSA begins for if statement (line 146)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Name to a Name (line 148):
                    
                    # Assigning a Name to a Name (line 148):
                    # Getting the type of 'root' (line 148)
                    root_308530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 34), 'root')
                    # Assigning a type to the variable 'oldroot' (line 148)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 24), 'oldroot', root_308530)
                    
                    # Assigning a Name to a Subscript (line 149):
                    
                    # Assigning a Name to a Subscript (line 149):
                    # Getting the type of 'key' (line 149)
                    key_308531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 39), 'key')
                    # Getting the type of 'oldroot' (line 149)
                    oldroot_308532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 24), 'oldroot')
                    # Getting the type of 'KEY' (line 149)
                    KEY_308533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 32), 'KEY')
                    # Storing an element on a container (line 149)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 149, 24), oldroot_308532, (KEY_308533, key_308531))
                    
                    # Assigning a Name to a Subscript (line 150):
                    
                    # Assigning a Name to a Subscript (line 150):
                    # Getting the type of 'result' (line 150)
                    result_308534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 42), 'result')
                    # Getting the type of 'oldroot' (line 150)
                    oldroot_308535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 24), 'oldroot')
                    # Getting the type of 'RESULT' (line 150)
                    RESULT_308536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 32), 'RESULT')
                    # Storing an element on a container (line 150)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 150, 24), oldroot_308535, (RESULT_308536, result_308534))
                    
                    # Multiple assignment of 2 elements.
                    
                    # Assigning a Subscript to a Subscript (line 152):
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'NEXT' (line 152)
                    NEXT_308537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 58), 'NEXT')
                    # Getting the type of 'oldroot' (line 152)
                    oldroot_308538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 50), 'oldroot')
                    # Obtaining the member '__getitem__' of a type (line 152)
                    getitem___308539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 50), oldroot_308538, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 152)
                    subscript_call_result_308540 = invoke(stypy.reporting.localization.Localization(__file__, 152, 50), getitem___308539, NEXT_308537)
                    
                    # Getting the type of 'nonlocal_root' (line 152)
                    nonlocal_root_308541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 31), 'nonlocal_root')
                    int_308542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 45), 'int')
                    # Storing an element on a container (line 152)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 31), nonlocal_root_308541, (int_308542, subscript_call_result_308540))
                    
                    # Assigning a Subscript to a Name (line 152):
                    
                    # Obtaining the type of the subscript
                    int_308543 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 45), 'int')
                    # Getting the type of 'nonlocal_root' (line 152)
                    nonlocal_root_308544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 31), 'nonlocal_root')
                    # Obtaining the member '__getitem__' of a type (line 152)
                    getitem___308545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 31), nonlocal_root_308544, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 152)
                    subscript_call_result_308546 = invoke(stypy.reporting.localization.Localization(__file__, 152, 31), getitem___308545, int_308543)
                    
                    # Assigning a type to the variable 'root' (line 152)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 24), 'root', subscript_call_result_308546)
                    
                    # Assigning a Subscript to a Name (line 153):
                    
                    # Assigning a Subscript to a Name (line 153):
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'KEY' (line 153)
                    KEY_308547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 38), 'KEY')
                    # Getting the type of 'root' (line 153)
                    root_308548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 33), 'root')
                    # Obtaining the member '__getitem__' of a type (line 153)
                    getitem___308549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 33), root_308548, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 153)
                    subscript_call_result_308550 = invoke(stypy.reporting.localization.Localization(__file__, 153, 33), getitem___308549, KEY_308547)
                    
                    # Assigning a type to the variable 'oldkey' (line 153)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 24), 'oldkey', subscript_call_result_308550)
                    
                    # Multiple assignment of 2 elements.
                    
                    # Assigning a Name to a Subscript (line 154):
                    # Getting the type of 'None' (line 154)
                    None_308551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 51), 'None')
                    # Getting the type of 'root' (line 154)
                    root_308552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 36), 'root')
                    # Getting the type of 'RESULT' (line 154)
                    RESULT_308553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 41), 'RESULT')
                    # Storing an element on a container (line 154)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 36), root_308552, (RESULT_308553, None_308551))
                    
                    # Assigning a Subscript to a Subscript (line 154):
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'RESULT' (line 154)
                    RESULT_308554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 41), 'RESULT')
                    # Getting the type of 'root' (line 154)
                    root_308555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 36), 'root')
                    # Obtaining the member '__getitem__' of a type (line 154)
                    getitem___308556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 36), root_308555, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 154)
                    subscript_call_result_308557 = invoke(stypy.reporting.localization.Localization(__file__, 154, 36), getitem___308556, RESULT_308554)
                    
                    # Getting the type of 'root' (line 154)
                    root_308558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 24), 'root')
                    # Getting the type of 'KEY' (line 154)
                    KEY_308559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 29), 'KEY')
                    # Storing an element on a container (line 154)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 24), root_308558, (KEY_308559, subscript_call_result_308557))
                    # Deleting a member
                    # Getting the type of 'cache' (line 156)
                    cache_308560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 28), 'cache')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'oldkey' (line 156)
                    oldkey_308561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 34), 'oldkey')
                    # Getting the type of 'cache' (line 156)
                    cache_308562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 28), 'cache')
                    # Obtaining the member '__getitem__' of a type (line 156)
                    getitem___308563 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 28), cache_308562, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 156)
                    subscript_call_result_308564 = invoke(stypy.reporting.localization.Localization(__file__, 156, 28), getitem___308563, oldkey_308561)
                    
                    del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 24), cache_308560, subscript_call_result_308564)
                    
                    # Assigning a Name to a Subscript (line 157):
                    
                    # Assigning a Name to a Subscript (line 157):
                    # Getting the type of 'oldroot' (line 157)
                    oldroot_308565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 37), 'oldroot')
                    # Getting the type of 'cache' (line 157)
                    cache_308566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 24), 'cache')
                    # Getting the type of 'key' (line 157)
                    key_308567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 30), 'key')
                    # Storing an element on a container (line 157)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 24), cache_308566, (key_308567, oldroot_308565))
                    # SSA branch for the else part of an if statement (line 146)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Subscript to a Name (line 160):
                    
                    # Assigning a Subscript to a Name (line 160):
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'PREV' (line 160)
                    PREV_308568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 36), 'PREV')
                    # Getting the type of 'root' (line 160)
                    root_308569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 31), 'root')
                    # Obtaining the member '__getitem__' of a type (line 160)
                    getitem___308570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 31), root_308569, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 160)
                    subscript_call_result_308571 = invoke(stypy.reporting.localization.Localization(__file__, 160, 31), getitem___308570, PREV_308568)
                    
                    # Assigning a type to the variable 'last' (line 160)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 24), 'last', subscript_call_result_308571)
                    
                    # Assigning a List to a Name (line 161):
                    
                    # Assigning a List to a Name (line 161):
                    
                    # Obtaining an instance of the builtin type 'list' (line 161)
                    list_308572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 31), 'list')
                    # Adding type elements to the builtin type 'list' instance (line 161)
                    # Adding element type (line 161)
                    # Getting the type of 'last' (line 161)
                    last_308573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 32), 'last')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 31), list_308572, last_308573)
                    # Adding element type (line 161)
                    # Getting the type of 'root' (line 161)
                    root_308574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 38), 'root')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 31), list_308572, root_308574)
                    # Adding element type (line 161)
                    # Getting the type of 'key' (line 161)
                    key_308575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 44), 'key')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 31), list_308572, key_308575)
                    # Adding element type (line 161)
                    # Getting the type of 'result' (line 161)
                    result_308576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 49), 'result')
                    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 31), list_308572, result_308576)
                    
                    # Assigning a type to the variable 'link' (line 161)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 24), 'link', list_308572)
                    
                    # Multiple assignment of 3 elements.
                    
                    # Assigning a Name to a Subscript (line 162):
                    # Getting the type of 'link' (line 162)
                    link_308577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 63), 'link')
                    # Getting the type of 'cache' (line 162)
                    cache_308578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 50), 'cache')
                    # Getting the type of 'key' (line 162)
                    key_308579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 56), 'key')
                    # Storing an element on a container (line 162)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 50), cache_308578, (key_308579, link_308577))
                    
                    # Assigning a Subscript to a Subscript (line 162):
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'key' (line 162)
                    key_308580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 56), 'key')
                    # Getting the type of 'cache' (line 162)
                    cache_308581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 50), 'cache')
                    # Obtaining the member '__getitem__' of a type (line 162)
                    getitem___308582 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 50), cache_308581, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 162)
                    subscript_call_result_308583 = invoke(stypy.reporting.localization.Localization(__file__, 162, 50), getitem___308582, key_308580)
                    
                    # Getting the type of 'root' (line 162)
                    root_308584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 37), 'root')
                    # Getting the type of 'PREV' (line 162)
                    PREV_308585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 42), 'PREV')
                    # Storing an element on a container (line 162)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 37), root_308584, (PREV_308585, subscript_call_result_308583))
                    
                    # Assigning a Subscript to a Subscript (line 162):
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'PREV' (line 162)
                    PREV_308586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 42), 'PREV')
                    # Getting the type of 'root' (line 162)
                    root_308587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 37), 'root')
                    # Obtaining the member '__getitem__' of a type (line 162)
                    getitem___308588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 37), root_308587, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 162)
                    subscript_call_result_308589 = invoke(stypy.reporting.localization.Localization(__file__, 162, 37), getitem___308588, PREV_308586)
                    
                    # Getting the type of 'last' (line 162)
                    last_308590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 24), 'last')
                    # Getting the type of 'NEXT' (line 162)
                    NEXT_308591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 29), 'NEXT')
                    # Storing an element on a container (line 162)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 24), last_308590, (NEXT_308591, subscript_call_result_308589))
                    # SSA join for if statement (line 146)
                    module_type_store = module_type_store.join_ssa_context()
                    
                    # SSA join for if statement (line 140)
                    module_type_store = module_type_store.join_ssa_context()
                    
                    
                    # Getting the type of 'stats' (line 163)
                    stats_308592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 20), 'stats')
                    
                    # Obtaining the type of the subscript
                    # Getting the type of 'MISSES' (line 163)
                    MISSES_308593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 26), 'MISSES')
                    # Getting the type of 'stats' (line 163)
                    stats_308594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 20), 'stats')
                    # Obtaining the member '__getitem__' of a type (line 163)
                    getitem___308595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 20), stats_308594, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 163)
                    subscript_call_result_308596 = invoke(stypy.reporting.localization.Localization(__file__, 163, 20), getitem___308595, MISSES_308593)
                    
                    int_308597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 37), 'int')
                    # Applying the binary operator '+=' (line 163)
                    result_iadd_308598 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 20), '+=', subscript_call_result_308596, int_308597)
                    # Getting the type of 'stats' (line 163)
                    stats_308599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 20), 'stats')
                    # Getting the type of 'MISSES' (line 163)
                    MISSES_308600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 26), 'MISSES')
                    # Storing an element on a container (line 163)
                    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 20), stats_308599, (MISSES_308600, result_iadd_308598))
                    
                    # Calling the __exit__ method to finish a with section
                    # Obtaining the member '__exit__' of a type (line 138)
                    exit___308601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 21), lock_308510, '__exit__')
                    with_exit_308602 = invoke(stypy.reporting.localization.Localization(__file__, 138, 21), exit___308601, None, None, None)

                # Getting the type of 'result' (line 164)
                result_308603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 23), 'result')
                # Assigning a type to the variable 'stypy_return_type' (line 164)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 16), 'stypy_return_type', result_308603)
                
                # ################# End of 'wrapper(...)' code ##################

                # Teardown call information
                teardown_call_information(localization, arguments)
                
                # Storing the return type of function 'wrapper' in the type store
                # Getting the type of 'stypy_return_type' (line 120)
                stypy_return_type_308604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'stypy_return_type')
                module_type_store.store_return_type_of_current_context(stypy_return_type_308604)
                
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                
                # Return type of the function 'wrapper'
                return stypy_return_type_308604

            # Assigning a type to the variable 'wrapper' (line 120)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'wrapper', wrapper)

            if (may_be_308374 and more_types_in_union_308375):
                # SSA join for if statement (line 104)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 96)
        module_type_store = module_type_store.join_ssa_context()
        

        @norecursion
        def cache_info(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'cache_info'
            module_type_store = module_type_store.open_function_context('cache_info', 166, 8, False)
            
            # Passed parameters checking function
            cache_info.stypy_localization = localization
            cache_info.stypy_type_of_self = None
            cache_info.stypy_type_store = module_type_store
            cache_info.stypy_function_name = 'cache_info'
            cache_info.stypy_param_names_list = []
            cache_info.stypy_varargs_param_name = None
            cache_info.stypy_kwargs_param_name = None
            cache_info.stypy_call_defaults = defaults
            cache_info.stypy_call_varargs = varargs
            cache_info.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'cache_info', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'cache_info', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'cache_info(...)' code ##################

            str_308605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 12), 'str', 'Report cache statistics')
            # Getting the type of 'lock' (line 168)
            lock_308606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 17), 'lock')
            with_308607 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 168, 17), lock_308606, 'with parameter', '__enter__', '__exit__')

            if with_308607:
                # Calling the __enter__ method to initiate a with section
                # Obtaining the member '__enter__' of a type (line 168)
                enter___308608 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 17), lock_308606, '__enter__')
                with_enter_308609 = invoke(stypy.reporting.localization.Localization(__file__, 168, 17), enter___308608)
                
                # Call to _CacheInfo(...): (line 169)
                # Processing the call arguments (line 169)
                
                # Obtaining the type of the subscript
                # Getting the type of 'HITS' (line 169)
                HITS_308611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 40), 'HITS', False)
                # Getting the type of 'stats' (line 169)
                stats_308612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 34), 'stats', False)
                # Obtaining the member '__getitem__' of a type (line 169)
                getitem___308613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 34), stats_308612, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 169)
                subscript_call_result_308614 = invoke(stypy.reporting.localization.Localization(__file__, 169, 34), getitem___308613, HITS_308611)
                
                
                # Obtaining the type of the subscript
                # Getting the type of 'MISSES' (line 169)
                MISSES_308615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 53), 'MISSES', False)
                # Getting the type of 'stats' (line 169)
                stats_308616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 47), 'stats', False)
                # Obtaining the member '__getitem__' of a type (line 169)
                getitem___308617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 47), stats_308616, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 169)
                subscript_call_result_308618 = invoke(stypy.reporting.localization.Localization(__file__, 169, 47), getitem___308617, MISSES_308615)
                
                # Getting the type of 'maxsize' (line 169)
                maxsize_308619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 62), 'maxsize', False)
                
                # Call to len(...): (line 169)
                # Processing the call arguments (line 169)
                # Getting the type of 'cache' (line 169)
                cache_308621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 75), 'cache', False)
                # Processing the call keyword arguments (line 169)
                kwargs_308622 = {}
                # Getting the type of 'len' (line 169)
                len_308620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 71), 'len', False)
                # Calling len(args, kwargs) (line 169)
                len_call_result_308623 = invoke(stypy.reporting.localization.Localization(__file__, 169, 71), len_308620, *[cache_308621], **kwargs_308622)
                
                # Processing the call keyword arguments (line 169)
                kwargs_308624 = {}
                # Getting the type of '_CacheInfo' (line 169)
                _CacheInfo_308610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 23), '_CacheInfo', False)
                # Calling _CacheInfo(args, kwargs) (line 169)
                _CacheInfo_call_result_308625 = invoke(stypy.reporting.localization.Localization(__file__, 169, 23), _CacheInfo_308610, *[subscript_call_result_308614, subscript_call_result_308618, maxsize_308619, len_call_result_308623], **kwargs_308624)
                
                # Assigning a type to the variable 'stypy_return_type' (line 169)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 16), 'stypy_return_type', _CacheInfo_call_result_308625)
                # Calling the __exit__ method to finish a with section
                # Obtaining the member '__exit__' of a type (line 168)
                exit___308626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 17), lock_308606, '__exit__')
                with_exit_308627 = invoke(stypy.reporting.localization.Localization(__file__, 168, 17), exit___308626, None, None, None)

            
            # ################# End of 'cache_info(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'cache_info' in the type store
            # Getting the type of 'stypy_return_type' (line 166)
            stypy_return_type_308628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_308628)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'cache_info'
            return stypy_return_type_308628

        # Assigning a type to the variable 'cache_info' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'cache_info', cache_info)

        @norecursion
        def cache_clear(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function 'cache_clear'
            module_type_store = module_type_store.open_function_context('cache_clear', 171, 8, False)
            
            # Passed parameters checking function
            cache_clear.stypy_localization = localization
            cache_clear.stypy_type_of_self = None
            cache_clear.stypy_type_store = module_type_store
            cache_clear.stypy_function_name = 'cache_clear'
            cache_clear.stypy_param_names_list = []
            cache_clear.stypy_varargs_param_name = None
            cache_clear.stypy_kwargs_param_name = None
            cache_clear.stypy_call_defaults = defaults
            cache_clear.stypy_call_varargs = varargs
            cache_clear.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, 'cache_clear', [], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, 'cache_clear', localization, [], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of 'cache_clear(...)' code ##################

            str_308629 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 12), 'str', 'Clear the cache and cache statistics')
            # Getting the type of 'lock' (line 173)
            lock_308630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 17), 'lock')
            with_308631 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 173, 17), lock_308630, 'with parameter', '__enter__', '__exit__')

            if with_308631:
                # Calling the __enter__ method to initiate a with section
                # Obtaining the member '__enter__' of a type (line 173)
                enter___308632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 17), lock_308630, '__enter__')
                with_enter_308633 = invoke(stypy.reporting.localization.Localization(__file__, 173, 17), enter___308632)
                
                # Call to clear(...): (line 174)
                # Processing the call keyword arguments (line 174)
                kwargs_308636 = {}
                # Getting the type of 'cache' (line 174)
                cache_308634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 16), 'cache', False)
                # Obtaining the member 'clear' of a type (line 174)
                clear_308635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 16), cache_308634, 'clear')
                # Calling clear(args, kwargs) (line 174)
                clear_call_result_308637 = invoke(stypy.reporting.localization.Localization(__file__, 174, 16), clear_308635, *[], **kwargs_308636)
                
                
                # Assigning a Subscript to a Name (line 175):
                
                # Assigning a Subscript to a Name (line 175):
                
                # Obtaining the type of the subscript
                int_308638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 37), 'int')
                # Getting the type of 'nonlocal_root' (line 175)
                nonlocal_root_308639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 23), 'nonlocal_root')
                # Obtaining the member '__getitem__' of a type (line 175)
                getitem___308640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 23), nonlocal_root_308639, '__getitem__')
                # Calling the subscript (__getitem__) to obtain the elements type (line 175)
                subscript_call_result_308641 = invoke(stypy.reporting.localization.Localization(__file__, 175, 23), getitem___308640, int_308638)
                
                # Assigning a type to the variable 'root' (line 175)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 16), 'root', subscript_call_result_308641)
                
                # Assigning a List to a Subscript (line 176):
                
                # Assigning a List to a Subscript (line 176):
                
                # Obtaining an instance of the builtin type 'list' (line 176)
                list_308642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 26), 'list')
                # Adding type elements to the builtin type 'list' instance (line 176)
                # Adding element type (line 176)
                # Getting the type of 'root' (line 176)
                root_308643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 27), 'root')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 26), list_308642, root_308643)
                # Adding element type (line 176)
                # Getting the type of 'root' (line 176)
                root_308644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 33), 'root')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 26), list_308642, root_308644)
                # Adding element type (line 176)
                # Getting the type of 'None' (line 176)
                None_308645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 39), 'None')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 26), list_308642, None_308645)
                # Adding element type (line 176)
                # Getting the type of 'None' (line 176)
                None_308646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 45), 'None')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 26), list_308642, None_308646)
                
                # Getting the type of 'root' (line 176)
                root_308647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 16), 'root')
                slice_308648 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 176, 16), None, None, None)
                # Storing an element on a container (line 176)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 16), root_308647, (slice_308648, list_308642))
                
                # Assigning a List to a Subscript (line 177):
                
                # Assigning a List to a Subscript (line 177):
                
                # Obtaining an instance of the builtin type 'list' (line 177)
                list_308649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 27), 'list')
                # Adding type elements to the builtin type 'list' instance (line 177)
                # Adding element type (line 177)
                int_308650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 28), 'int')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 27), list_308649, int_308650)
                # Adding element type (line 177)
                int_308651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 31), 'int')
                add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 27), list_308649, int_308651)
                
                # Getting the type of 'stats' (line 177)
                stats_308652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 16), 'stats')
                slice_308653 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 177, 16), None, None, None)
                # Storing an element on a container (line 177)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 16), stats_308652, (slice_308653, list_308649))
                # Calling the __exit__ method to finish a with section
                # Obtaining the member '__exit__' of a type (line 173)
                exit___308654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 17), lock_308630, '__exit__')
                with_exit_308655 = invoke(stypy.reporting.localization.Localization(__file__, 173, 17), exit___308654, None, None, None)

            
            # ################# End of 'cache_clear(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function 'cache_clear' in the type store
            # Getting the type of 'stypy_return_type' (line 171)
            stypy_return_type_308656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_308656)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function 'cache_clear'
            return stypy_return_type_308656

        # Assigning a type to the variable 'cache_clear' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'cache_clear', cache_clear)
        
        # Assigning a Name to a Attribute (line 179):
        
        # Assigning a Name to a Attribute (line 179):
        # Getting the type of 'user_function' (line 179)
        user_function_308657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 30), 'user_function')
        # Getting the type of 'wrapper' (line 179)
        wrapper_308658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'wrapper')
        # Setting the type of the member '__wrapped__' of a type (line 179)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 8), wrapper_308658, '__wrapped__', user_function_308657)
        
        # Assigning a Name to a Attribute (line 180):
        
        # Assigning a Name to a Attribute (line 180):
        # Getting the type of 'cache_info' (line 180)
        cache_info_308659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 29), 'cache_info')
        # Getting the type of 'wrapper' (line 180)
        wrapper_308660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 8), 'wrapper')
        # Setting the type of the member 'cache_info' of a type (line 180)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 8), wrapper_308660, 'cache_info', cache_info_308659)
        
        # Assigning a Name to a Attribute (line 181):
        
        # Assigning a Name to a Attribute (line 181):
        # Getting the type of 'cache_clear' (line 181)
        cache_clear_308661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 30), 'cache_clear')
        # Getting the type of 'wrapper' (line 181)
        wrapper_308662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 8), 'wrapper')
        # Setting the type of the member 'cache_clear' of a type (line 181)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 8), wrapper_308662, 'cache_clear', cache_clear_308661)
        
        # Call to update_wrapper(...): (line 182)
        # Processing the call arguments (line 182)
        # Getting the type of 'wrapper' (line 182)
        wrapper_308664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 30), 'wrapper', False)
        # Getting the type of 'user_function' (line 182)
        user_function_308665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 39), 'user_function', False)
        # Processing the call keyword arguments (line 182)
        kwargs_308666 = {}
        # Getting the type of 'update_wrapper' (line 182)
        update_wrapper_308663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 15), 'update_wrapper', False)
        # Calling update_wrapper(args, kwargs) (line 182)
        update_wrapper_call_result_308667 = invoke(stypy.reporting.localization.Localization(__file__, 182, 15), update_wrapper_308663, *[wrapper_308664, user_function_308665], **kwargs_308666)
        
        # Assigning a type to the variable 'stypy_return_type' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'stypy_return_type', update_wrapper_call_result_308667)
        
        # ################# End of 'decorating_function(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'decorating_function' in the type store
        # Getting the type of 'stypy_return_type' (line 82)
        stypy_return_type_308668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_308668)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'decorating_function'
        return stypy_return_type_308668

    # Assigning a type to the variable 'decorating_function' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'decorating_function', decorating_function)
    # Getting the type of 'decorating_function' (line 184)
    decorating_function_308669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 11), 'decorating_function')
    # Assigning a type to the variable 'stypy_return_type' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 4), 'stypy_return_type', decorating_function_308669)
    
    # ################# End of 'lru_cache(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'lru_cache' in the type store
    # Getting the type of 'stypy_return_type' (line 57)
    stypy_return_type_308670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_308670)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'lru_cache'
    return stypy_return_type_308670

# Assigning a type to the variable 'lru_cache' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'lru_cache', lru_cache)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
