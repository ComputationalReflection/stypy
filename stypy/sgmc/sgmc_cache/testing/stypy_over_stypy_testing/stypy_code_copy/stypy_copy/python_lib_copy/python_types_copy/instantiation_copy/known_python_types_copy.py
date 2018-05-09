
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import types
2: import inspect
3: import os
4: import operator
5: 
6: '''
7: File to store known Python language types (not defined in library modules) that are not listed in the types module.
8: Surprisingly, the types module omits several types that Python uses when dealing with commonly used structures, such
9: as iterators and member descriptors. As we need these types to specify type rules, type instances and, in general,
10: working with them, we created this file to list these types and its instances.
11: '''
12: 
13: # Native types that will use a simplified str representation (only its name) on stypy output
14: simple_python_types = [
15:     None,
16:     types.NoneType,
17:     type,
18:     bool,
19:     int,
20:     long,
21:     float,
22:     complex,
23:     str,
24:     unicode,
25: ]
26: 
27: '''
28: Internal python elements to generate various sample type values
29: '''
30: 
31: 
32: class Foo:
33:     qux = 3
34: 
35:     def bar(self):
36:         pass
37: 
38: 
39: def func():
40:     pass
41: 
42: 
43: def firstn(n):
44:     num = 0
45:     while num < n:
46:         yield num
47:         num += 1
48: 
49: 
50: foo = Foo()
51: #
52: #
53: # # ####################################
54: #
55: # '''
56: # This is the table that stypy uses to associate Python types with its textual name and a sample value.
57: # This table is used for generating code automatically: type rules and data type wrappers that checks
58: # members type.
59: #
60: # Is is a <type object>: <tuple> dictionary.
61: #
62: # The tuple contains a type name (to put into Python generated code) and a sample value of this type.
63: # Type names are used to represent types into generated Python code. Note that not all Python type names can
64: # be directly used into type declarations, as some contain the "-" character, not valid as part of a type
65: # name in Python syntax. Sample values are only used when a call is done with the FakeParamValues call handler.
66: # '''
67: known_python_type_typename_samplevalues = {
68:     types.NoneType: ("types.NoneType", None),
69:     type: ("type", type(int)),
70:     bool: ("bool", True),
71:     int: ("int", 1),
72:     long: ("long", 1L),
73:     float: ("float", 1.0),
74:     complex: ("complex", 1j),
75:     str: ("str", "foo"),
76:     unicode: ("unicode", u"foo"),
77:     tuple: ("tuple", (1, 2)),
78:     list: ("list", [1, 2, 3, 4, 5]),
79:     dict: ("dict", {"a": 1, "b": 2}),
80:     set: ("set", {4, 5}),
81:     type(func): ("types.FunctionType", func),
82:     types.LambdaType: ("types.LambdaType", lambda x: x),
83:     types.GeneratorType: ("types.GeneratorType", firstn(4)),
84:     types.CodeType: ("types.CodeType", func.func_code),
85:     types.BuiltinFunctionType: ("types.BuiltinFunctionType", len),
86:     types.ModuleType: ("types.ModuleType", inspect),
87:     file: ("file", file(os.path.dirname(os.path.realpath(__file__)) + "/foo.txt", "w")),
88:     xrange: ("xrange", xrange(1, 3)),
89:     types.SliceType: ("slice", slice(4, 3, 2)),
90:     buffer: ("buffer", buffer("r")),
91:     types.DictProxyType: ("types.DictProxyType", int.__dict__),
92:     types.ClassType: ("types.ClassType", Foo),
93:     types.InstanceType: ("types.InstanceType", foo),
94:     types.MethodType: ("types.MethodType", foo.bar),
95:     iter: ("iter", iter("abc")),
96: 
97:     bytearray: ("bytearray", bytearray("test")),
98:     classmethod: ("classmethod", classmethod(Foo.bar)),
99:     enumerate: ("enumerate", enumerate([1, 2, 3])),
100:     frozenset: ("frozenset", frozenset([1, 2, 3])),
101:     memoryview: ("memoryview", memoryview(buffer("foo"))),
102:     object: ("object", object()),
103:     property: ("property", property(Foo.qux)),
104:     staticmethod: ("staticmethod", staticmethod(Foo.bar)),
105:     super: ("super", super(type(Foo))),
106:     reversed: ("reversed", reversed((1, 2))),
107: 
108:     type(iter((1, 2, 3, 4, 5))): ("ExtraTypeDefinitions.tupleiterator", (1, 2, 3, 4, 5).__iter__()),
109:     type(iter(xrange(1))): ("ExtraTypeDefinitions.rangeiterator", iter(xrange(1))),
110:     type(iter([1, 2])): ("ExtraTypeDefinitions.listiterator", [1, 2, 3, 4, 5].__iter__()),
111:     type(iter(type(int), 0.1)): ("ExtraTypeDefinitions.callable_iterator", iter(type(int), 0.1)),
112:     type(iter(reversed([1, 2]))): ("ExtraTypeDefinitions.listreverseiterator", iter(reversed([1, 2]))),
113:     type(operator.methodcaller(0)): ("ExtraTypeDefinitions.methodcaller", operator.methodcaller(0)),
114:     type(operator.itemgetter(0)): ("ExtraTypeDefinitions.itemgetter", operator.itemgetter(0)),
115:     type(operator.attrgetter(0)): ("ExtraTypeDefinitions.attrgetter", operator.attrgetter(0)),
116:     type(iter(bytearray("test"))): ("ExtraTypeDefinitions.bytearray_iterator", iter(bytearray("test"))),
117: 
118:     type({"a": 1, "b": 2}.viewitems()): ("ExtraTypeDefinitions.dict_items", {"a": 1, "b": 2}.viewitems()),
119:     type({"a": 1, "b": 2}.viewkeys()): ("ExtraTypeDefinitions.dict_keys", {"a": 1, "b": 2}.viewkeys()),
120:     type({"a": 1, "b": 2}.viewvalues()): ("ExtraTypeDefinitions.dict_values", {"a": 1, "b": 2}.viewvalues()),
121: 
122:     type(iter({"a": 1, "b": 2})): ("ExtraTypeDefinitions.dictionary_keyiterator", iter({"a": 1, "b": 2})),
123:     type({"a": 1, "b": 2}.iteritems()): ("ExtraTypeDefinitions.dictionary_itemiterator", {"a": 1}.iteritems()),
124:     type({"a": 1, "b": 2}.itervalues()): ("ExtraTypeDefinitions.dictionary_valueiterator", {"a": 1}.itervalues()),
125: 
126:     type(iter(bytearray("test"))): ("ExtraTypeDefinitions.bytearray_iterator", iter(bytearray("test"))),
127: 
128:     type(ArithmeticError.message): ("ExtraTypeDefinitions.getset_descriptor", ArithmeticError.message),
129:     type(IOError.errno): ("ExtraTypeDefinitions.member_descriptor", IOError.errno),
130:     type(u"foo"._formatter_parser()): ("ExtraTypeDefinitions.formatteriterator", type(u"foo"._formatter_parser()))
131: }
132: 
133: 
134: class ExtraTypeDefinitions:
135:     '''
136:     Additional (not included) type definitions to those defined in the types Python module. This class is needed
137:     to have an usable type object to refer to when generating Python code
138:     '''
139:     SetType = set
140:     iterator = type(iter(""))
141: 
142:     setiterator = type(iter(set()))
143:     tupleiterator = type(iter(tuple()))
144:     rangeiterator = type(iter(xrange(1)))
145:     listiterator = type(iter(list()))
146:     callable_iterator = type(iter(type(int), 0.1))
147:     listreverseiterator = type(iter(reversed(list())))
148:     methodcaller = type(operator.methodcaller(0))
149:     itemgetter = type(operator.itemgetter(0))
150:     attrgetter = type(operator.attrgetter(0))
151: 
152:     dict_items = type(dict({"a": 1, "b": 2}).viewitems())
153:     dict_keys = type(dict({"a": 1, "b": 2}).viewkeys())
154:     dict_values = type(dict({"a": 1, "b": 2}).viewvalues())
155: 
156:     dictionary_keyiterator = type(iter(dict({"a": 1, "b": 2})))
157:     dictionary_itemiterator = type(dict({"a": 1, "b": 2}).iteritems())
158:     dictionary_valueiterator = type(dict({"a": 1, "b": 2}).itervalues())
159:     bytearray_iterator = type(iter(bytearray("test")))
160: 
161:     # Extra builtins without instance counterparts
162:     getset_descriptor = type(ArithmeticError.message)
163:     member_descriptor = type(IOError.errno)
164:     formatteriterator = type(u"foo"._formatter_parser())
165: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import types' statement (line 1)
import types

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'types', types, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import inspect' statement (line 2)
import inspect

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'inspect', inspect, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import operator' statement (line 4)
import operator

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'operator', operator, module_type_store)

str_8588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, (-1)), 'str', '\nFile to store known Python language types (not defined in library modules) that are not listed in the types module.\nSurprisingly, the types module omits several types that Python uses when dealing with commonly used structures, such\nas iterators and member descriptors. As we need these types to specify type rules, type instances and, in general,\nworking with them, we created this file to list these types and its instances.\n')

# Assigning a List to a Name (line 14):

# Obtaining an instance of the builtin type 'list' (line 14)
list_8589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 14)
# Adding element type (line 14)
# Getting the type of 'None' (line 15)
None_8590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 22), list_8589, None_8590)
# Adding element type (line 14)
# Getting the type of 'types' (line 16)
types_8591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'types')
# Obtaining the member 'NoneType' of a type (line 16)
NoneType_8592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 4), types_8591, 'NoneType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 22), list_8589, NoneType_8592)
# Adding element type (line 14)
# Getting the type of 'type' (line 17)
type_8593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'type')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 22), list_8589, type_8593)
# Adding element type (line 14)
# Getting the type of 'bool' (line 18)
bool_8594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'bool')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 22), list_8589, bool_8594)
# Adding element type (line 14)
# Getting the type of 'int' (line 19)
int_8595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 22), list_8589, int_8595)
# Adding element type (line 14)
# Getting the type of 'long' (line 20)
long_8596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 22), list_8589, long_8596)
# Adding element type (line 14)
# Getting the type of 'float' (line 21)
float_8597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 22), list_8589, float_8597)
# Adding element type (line 14)
# Getting the type of 'complex' (line 22)
complex_8598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'complex')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 22), list_8589, complex_8598)
# Adding element type (line 14)
# Getting the type of 'str' (line 23)
str_8599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'str')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 22), list_8589, str_8599)
# Adding element type (line 14)
# Getting the type of 'unicode' (line 24)
unicode_8600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'unicode')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 22), list_8589, unicode_8600)

# Assigning a type to the variable 'simple_python_types' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'simple_python_types', list_8589)
str_8601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, (-1)), 'str', '\nInternal python elements to generate various sample type values\n')
# Declaration of the 'Foo' class

class Foo:

    @norecursion
    def bar(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'bar'
        module_type_store = module_type_store.open_function_context('bar', 35, 4, False)
        # Assigning a type to the variable 'self' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Foo.bar.__dict__.__setitem__('stypy_localization', localization)
        Foo.bar.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Foo.bar.__dict__.__setitem__('stypy_type_store', module_type_store)
        Foo.bar.__dict__.__setitem__('stypy_function_name', 'Foo.bar')
        Foo.bar.__dict__.__setitem__('stypy_param_names_list', [])
        Foo.bar.__dict__.__setitem__('stypy_varargs_param_name', None)
        Foo.bar.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Foo.bar.__dict__.__setitem__('stypy_call_defaults', defaults)
        Foo.bar.__dict__.__setitem__('stypy_call_varargs', varargs)
        Foo.bar.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Foo.bar.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.bar', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'bar', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'bar(...)' code ##################

        pass
        
        # ################# End of 'bar(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'bar' in the type store
        # Getting the type of 'stypy_return_type' (line 35)
        stypy_return_type_8602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8602)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'bar'
        return stypy_return_type_8602


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 32, 0, False)
        # Assigning a type to the variable 'self' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Foo.__init__', [], None, None, defaults, varargs, kwargs)

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

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Foo' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 0), 'Foo', Foo)

# Assigning a Num to a Name (line 33):
int_8603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 10), 'int')
# Getting the type of 'Foo'
Foo_8604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Foo')
# Setting the type of the member 'qux' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Foo_8604, 'qux', int_8603)

@norecursion
def func(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'func'
    module_type_store = module_type_store.open_function_context('func', 39, 0, False)
    
    # Passed parameters checking function
    func.stypy_localization = localization
    func.stypy_type_of_self = None
    func.stypy_type_store = module_type_store
    func.stypy_function_name = 'func'
    func.stypy_param_names_list = []
    func.stypy_varargs_param_name = None
    func.stypy_kwargs_param_name = None
    func.stypy_call_defaults = defaults
    func.stypy_call_varargs = varargs
    func.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'func', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'func', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'func(...)' code ##################

    pass
    
    # ################# End of 'func(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'func' in the type store
    # Getting the type of 'stypy_return_type' (line 39)
    stypy_return_type_8605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8605)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'func'
    return stypy_return_type_8605

# Assigning a type to the variable 'func' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'func', func)

@norecursion
def firstn(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'firstn'
    module_type_store = module_type_store.open_function_context('firstn', 43, 0, False)
    
    # Passed parameters checking function
    firstn.stypy_localization = localization
    firstn.stypy_type_of_self = None
    firstn.stypy_type_store = module_type_store
    firstn.stypy_function_name = 'firstn'
    firstn.stypy_param_names_list = ['n']
    firstn.stypy_varargs_param_name = None
    firstn.stypy_kwargs_param_name = None
    firstn.stypy_call_defaults = defaults
    firstn.stypy_call_varargs = varargs
    firstn.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'firstn', ['n'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'firstn', localization, ['n'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'firstn(...)' code ##################

    
    # Assigning a Num to a Name (line 44):
    int_8606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 10), 'int')
    # Assigning a type to the variable 'num' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'num', int_8606)
    
    
    # Getting the type of 'num' (line 45)
    num_8607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 10), 'num')
    # Getting the type of 'n' (line 45)
    n_8608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 16), 'n')
    # Applying the binary operator '<' (line 45)
    result_lt_8609 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 10), '<', num_8607, n_8608)
    
    # Assigning a type to the variable 'result_lt_8609' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'result_lt_8609', result_lt_8609)
    # Testing if the while is going to be iterated (line 45)
    # Testing the type of an if condition (line 45)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 45, 4), result_lt_8609)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 45, 4), result_lt_8609):
        # SSA begins for while statement (line 45)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        # Creating a generator
        # Getting the type of 'num' (line 46)
        num_8610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 14), 'num')
        GeneratorType_8611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 8), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 8), GeneratorType_8611, num_8610)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'stypy_return_type', GeneratorType_8611)
        
        # Getting the type of 'num' (line 47)
        num_8612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'num')
        int_8613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 15), 'int')
        # Applying the binary operator '+=' (line 47)
        result_iadd_8614 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 8), '+=', num_8612, int_8613)
        # Assigning a type to the variable 'num' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'num', result_iadd_8614)
        
        # SSA join for while statement (line 45)
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'firstn(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'firstn' in the type store
    # Getting the type of 'stypy_return_type' (line 43)
    stypy_return_type_8615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8615)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'firstn'
    return stypy_return_type_8615

# Assigning a type to the variable 'firstn' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'firstn', firstn)

# Assigning a Call to a Name (line 50):

# Call to Foo(...): (line 50)
# Processing the call keyword arguments (line 50)
kwargs_8617 = {}
# Getting the type of 'Foo' (line 50)
Foo_8616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 6), 'Foo', False)
# Calling Foo(args, kwargs) (line 50)
Foo_call_result_8618 = invoke(stypy.reporting.localization.Localization(__file__, 50, 6), Foo_8616, *[], **kwargs_8617)

# Assigning a type to the variable 'foo' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'foo', Foo_call_result_8618)

# Assigning a Dict to a Name (line 67):

# Obtaining an instance of the builtin type 'dict' (line 67)
dict_8619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 42), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 67)
# Adding element type (key, value) (line 67)
# Getting the type of 'types' (line 68)
types_8620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'types')
# Obtaining the member 'NoneType' of a type (line 68)
NoneType_8621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 4), types_8620, 'NoneType')

# Obtaining an instance of the builtin type 'tuple' (line 68)
tuple_8622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 68)
# Adding element type (line 68)
str_8623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 21), 'str', 'types.NoneType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 21), tuple_8622, str_8623)
# Adding element type (line 68)
# Getting the type of 'None' (line 68)
None_8624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 39), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 21), tuple_8622, None_8624)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (NoneType_8621, tuple_8622))
# Adding element type (key, value) (line 67)
# Getting the type of 'type' (line 69)
type_8625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'type')

# Obtaining an instance of the builtin type 'tuple' (line 69)
tuple_8626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 11), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 69)
# Adding element type (line 69)
str_8627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 11), 'str', 'type')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 11), tuple_8626, str_8627)
# Adding element type (line 69)

# Call to type(...): (line 69)
# Processing the call arguments (line 69)
# Getting the type of 'int' (line 69)
int_8629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 24), 'int', False)
# Processing the call keyword arguments (line 69)
kwargs_8630 = {}
# Getting the type of 'type' (line 69)
type_8628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 19), 'type', False)
# Calling type(args, kwargs) (line 69)
type_call_result_8631 = invoke(stypy.reporting.localization.Localization(__file__, 69, 19), type_8628, *[int_8629], **kwargs_8630)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 11), tuple_8626, type_call_result_8631)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (type_8625, tuple_8626))
# Adding element type (key, value) (line 67)
# Getting the type of 'bool' (line 70)
bool_8632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'bool')

# Obtaining an instance of the builtin type 'tuple' (line 70)
tuple_8633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 11), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 70)
# Adding element type (line 70)
str_8634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 11), 'str', 'bool')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 11), tuple_8633, str_8634)
# Adding element type (line 70)
# Getting the type of 'True' (line 70)
True_8635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 19), 'True')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 11), tuple_8633, True_8635)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (bool_8632, tuple_8633))
# Adding element type (key, value) (line 67)
# Getting the type of 'int' (line 71)
int_8636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'int')

# Obtaining an instance of the builtin type 'tuple' (line 71)
tuple_8637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 10), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 71)
# Adding element type (line 71)
str_8638 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 10), 'str', 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 10), tuple_8637, str_8638)
# Adding element type (line 71)
int_8639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 10), tuple_8637, int_8639)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (int_8636, tuple_8637))
# Adding element type (key, value) (line 67)
# Getting the type of 'long' (line 72)
long_8640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'long')

# Obtaining an instance of the builtin type 'tuple' (line 72)
tuple_8641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 11), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 72)
# Adding element type (line 72)
str_8642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 11), 'str', 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 11), tuple_8641, str_8642)
# Adding element type (line 72)
long_8643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 19), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 11), tuple_8641, long_8643)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (long_8640, tuple_8641))
# Adding element type (key, value) (line 67)
# Getting the type of 'float' (line 73)
float_8644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'float')

# Obtaining an instance of the builtin type 'tuple' (line 73)
tuple_8645 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 12), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 73)
# Adding element type (line 73)
str_8646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 12), 'str', 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 12), tuple_8645, str_8646)
# Adding element type (line 73)
float_8647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 21), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 12), tuple_8645, float_8647)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (float_8644, tuple_8645))
# Adding element type (key, value) (line 67)
# Getting the type of 'complex' (line 74)
complex_8648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'complex')

# Obtaining an instance of the builtin type 'tuple' (line 74)
tuple_8649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 14), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 74)
# Adding element type (line 74)
str_8650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 14), 'str', 'complex')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 14), tuple_8649, str_8650)
# Adding element type (line 74)
complex_8651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 25), 'complex')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 14), tuple_8649, complex_8651)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (complex_8648, tuple_8649))
# Adding element type (key, value) (line 67)
# Getting the type of 'str' (line 75)
str_8652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'str')

# Obtaining an instance of the builtin type 'tuple' (line 75)
tuple_8653 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 10), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 75)
# Adding element type (line 75)
str_8654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 10), 'str', 'str')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 10), tuple_8653, str_8654)
# Adding element type (line 75)
str_8655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 17), 'str', 'foo')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 10), tuple_8653, str_8655)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (str_8652, tuple_8653))
# Adding element type (key, value) (line 67)
# Getting the type of 'unicode' (line 76)
unicode_8656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'unicode')

# Obtaining an instance of the builtin type 'tuple' (line 76)
tuple_8657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 14), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 76)
# Adding element type (line 76)
str_8658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 14), 'str', 'unicode')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 14), tuple_8657, str_8658)
# Adding element type (line 76)
unicode_8659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 25), 'unicode', u'foo')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 14), tuple_8657, unicode_8659)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (unicode_8656, tuple_8657))
# Adding element type (key, value) (line 67)
# Getting the type of 'tuple' (line 77)
tuple_8660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'tuple')

# Obtaining an instance of the builtin type 'tuple' (line 77)
tuple_8661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 12), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 77)
# Adding element type (line 77)
str_8662 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 12), 'str', 'tuple')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 12), tuple_8661, str_8662)
# Adding element type (line 77)

# Obtaining an instance of the builtin type 'tuple' (line 77)
tuple_8663 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 22), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 77)
# Adding element type (line 77)
int_8664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 22), tuple_8663, int_8664)
# Adding element type (line 77)
int_8665 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 25), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 22), tuple_8663, int_8665)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 12), tuple_8661, tuple_8663)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (tuple_8660, tuple_8661))
# Adding element type (key, value) (line 67)
# Getting the type of 'list' (line 78)
list_8666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'list')

# Obtaining an instance of the builtin type 'tuple' (line 78)
tuple_8667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 11), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 78)
# Adding element type (line 78)
str_8668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 11), 'str', 'list')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 11), tuple_8667, str_8668)
# Adding element type (line 78)

# Obtaining an instance of the builtin type 'list' (line 78)
list_8669 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 78)
# Adding element type (line 78)
int_8670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 19), list_8669, int_8670)
# Adding element type (line 78)
int_8671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 19), list_8669, int_8671)
# Adding element type (line 78)
int_8672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 19), list_8669, int_8672)
# Adding element type (line 78)
int_8673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 19), list_8669, int_8673)
# Adding element type (line 78)
int_8674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 19), list_8669, int_8674)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 11), tuple_8667, list_8669)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (list_8666, tuple_8667))
# Adding element type (key, value) (line 67)
# Getting the type of 'dict' (line 79)
dict_8675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'dict')

# Obtaining an instance of the builtin type 'tuple' (line 79)
tuple_8676 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 11), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 79)
# Adding element type (line 79)
str_8677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 11), 'str', 'dict')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 11), tuple_8676, str_8677)
# Adding element type (line 79)

# Obtaining an instance of the builtin type 'dict' (line 79)
dict_8678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 19), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 79)
# Adding element type (key, value) (line 79)
str_8679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 20), 'str', 'a')
int_8680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 25), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 19), dict_8678, (str_8679, int_8680))
# Adding element type (key, value) (line 79)
str_8681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 28), 'str', 'b')
int_8682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 33), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 19), dict_8678, (str_8681, int_8682))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 11), tuple_8676, dict_8678)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (dict_8675, tuple_8676))
# Adding element type (key, value) (line 67)
# Getting the type of 'set' (line 80)
set_8683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'set')

# Obtaining an instance of the builtin type 'tuple' (line 80)
tuple_8684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 10), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 80)
# Adding element type (line 80)
str_8685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 10), 'str', 'set')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 10), tuple_8684, str_8685)
# Adding element type (line 80)

# Obtaining an instance of the builtin type 'set' (line 80)
set_8686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 17), 'set')
# Adding type elements to the builtin type 'set' instance (line 80)
# Adding element type (line 80)
int_8687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 17), set_8686, int_8687)
# Adding element type (line 80)
int_8688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 17), set_8686, int_8688)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 10), tuple_8684, set_8686)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (set_8683, tuple_8684))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 81)
# Processing the call arguments (line 81)
# Getting the type of 'func' (line 81)
func_8690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 9), 'func', False)
# Processing the call keyword arguments (line 81)
kwargs_8691 = {}
# Getting the type of 'type' (line 81)
type_8689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'type', False)
# Calling type(args, kwargs) (line 81)
type_call_result_8692 = invoke(stypy.reporting.localization.Localization(__file__, 81, 4), type_8689, *[func_8690], **kwargs_8691)


# Obtaining an instance of the builtin type 'tuple' (line 81)
tuple_8693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 81)
# Adding element type (line 81)
str_8694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 17), 'str', 'types.FunctionType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 17), tuple_8693, str_8694)
# Adding element type (line 81)
# Getting the type of 'func' (line 81)
func_8695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 39), 'func')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 17), tuple_8693, func_8695)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (type_call_result_8692, tuple_8693))
# Adding element type (key, value) (line 67)
# Getting the type of 'types' (line 82)
types_8696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'types')
# Obtaining the member 'LambdaType' of a type (line 82)
LambdaType_8697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 4), types_8696, 'LambdaType')

# Obtaining an instance of the builtin type 'tuple' (line 82)
tuple_8698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 23), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 82)
# Adding element type (line 82)
str_8699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 23), 'str', 'types.LambdaType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 23), tuple_8698, str_8699)
# Adding element type (line 82)

@norecursion
def _stypy_temp_lambda_17(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_17'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_17', 82, 43, True)
    # Passed parameters checking function
    _stypy_temp_lambda_17.stypy_localization = localization
    _stypy_temp_lambda_17.stypy_type_of_self = None
    _stypy_temp_lambda_17.stypy_type_store = module_type_store
    _stypy_temp_lambda_17.stypy_function_name = '_stypy_temp_lambda_17'
    _stypy_temp_lambda_17.stypy_param_names_list = ['x']
    _stypy_temp_lambda_17.stypy_varargs_param_name = None
    _stypy_temp_lambda_17.stypy_kwargs_param_name = None
    _stypy_temp_lambda_17.stypy_call_defaults = defaults
    _stypy_temp_lambda_17.stypy_call_varargs = varargs
    _stypy_temp_lambda_17.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_17', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_17', ['x'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    # Getting the type of 'x' (line 82)
    x_8700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 53), 'x')
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 43), 'stypy_return_type', x_8700)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_17' in the type store
    # Getting the type of 'stypy_return_type' (line 82)
    stypy_return_type_8701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 43), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8701)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_17'
    return stypy_return_type_8701

# Assigning a type to the variable '_stypy_temp_lambda_17' (line 82)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 43), '_stypy_temp_lambda_17', _stypy_temp_lambda_17)
# Getting the type of '_stypy_temp_lambda_17' (line 82)
_stypy_temp_lambda_17_8702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 43), '_stypy_temp_lambda_17')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 23), tuple_8698, _stypy_temp_lambda_17_8702)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (LambdaType_8697, tuple_8698))
# Adding element type (key, value) (line 67)
# Getting the type of 'types' (line 83)
types_8703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'types')
# Obtaining the member 'GeneratorType' of a type (line 83)
GeneratorType_8704 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 4), types_8703, 'GeneratorType')

# Obtaining an instance of the builtin type 'tuple' (line 83)
tuple_8705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 26), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 83)
# Adding element type (line 83)
str_8706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 26), 'str', 'types.GeneratorType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 26), tuple_8705, str_8706)
# Adding element type (line 83)

# Call to firstn(...): (line 83)
# Processing the call arguments (line 83)
int_8708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 56), 'int')
# Processing the call keyword arguments (line 83)
kwargs_8709 = {}
# Getting the type of 'firstn' (line 83)
firstn_8707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 49), 'firstn', False)
# Calling firstn(args, kwargs) (line 83)
firstn_call_result_8710 = invoke(stypy.reporting.localization.Localization(__file__, 83, 49), firstn_8707, *[int_8708], **kwargs_8709)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 26), tuple_8705, firstn_call_result_8710)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (GeneratorType_8704, tuple_8705))
# Adding element type (key, value) (line 67)
# Getting the type of 'types' (line 84)
types_8711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'types')
# Obtaining the member 'CodeType' of a type (line 84)
CodeType_8712 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 4), types_8711, 'CodeType')

# Obtaining an instance of the builtin type 'tuple' (line 84)
tuple_8713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 84)
# Adding element type (line 84)
str_8714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 21), 'str', 'types.CodeType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 21), tuple_8713, str_8714)
# Adding element type (line 84)
# Getting the type of 'func' (line 84)
func_8715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 39), 'func')
# Obtaining the member 'func_code' of a type (line 84)
func_code_8716 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 39), func_8715, 'func_code')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 21), tuple_8713, func_code_8716)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (CodeType_8712, tuple_8713))
# Adding element type (key, value) (line 67)
# Getting the type of 'types' (line 85)
types_8717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'types')
# Obtaining the member 'BuiltinFunctionType' of a type (line 85)
BuiltinFunctionType_8718 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 4), types_8717, 'BuiltinFunctionType')

# Obtaining an instance of the builtin type 'tuple' (line 85)
tuple_8719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 32), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 85)
# Adding element type (line 85)
str_8720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 32), 'str', 'types.BuiltinFunctionType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 32), tuple_8719, str_8720)
# Adding element type (line 85)
# Getting the type of 'len' (line 85)
len_8721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 61), 'len')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 32), tuple_8719, len_8721)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (BuiltinFunctionType_8718, tuple_8719))
# Adding element type (key, value) (line 67)
# Getting the type of 'types' (line 86)
types_8722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'types')
# Obtaining the member 'ModuleType' of a type (line 86)
ModuleType_8723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 4), types_8722, 'ModuleType')

# Obtaining an instance of the builtin type 'tuple' (line 86)
tuple_8724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 23), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 86)
# Adding element type (line 86)
str_8725 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 23), 'str', 'types.ModuleType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 23), tuple_8724, str_8725)
# Adding element type (line 86)
# Getting the type of 'inspect' (line 86)
inspect_8726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 43), 'inspect')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 23), tuple_8724, inspect_8726)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (ModuleType_8723, tuple_8724))
# Adding element type (key, value) (line 67)
# Getting the type of 'file' (line 87)
file_8727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'file')

# Obtaining an instance of the builtin type 'tuple' (line 87)
tuple_8728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 11), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 87)
# Adding element type (line 87)
str_8729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 11), 'str', 'file')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 11), tuple_8728, str_8729)
# Adding element type (line 87)

# Call to file(...): (line 87)
# Processing the call arguments (line 87)

# Call to dirname(...): (line 87)
# Processing the call arguments (line 87)

# Call to realpath(...): (line 87)
# Processing the call arguments (line 87)
# Getting the type of '__file__' (line 87)
file___8737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 57), '__file__', False)
# Processing the call keyword arguments (line 87)
kwargs_8738 = {}
# Getting the type of 'os' (line 87)
os_8734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 40), 'os', False)
# Obtaining the member 'path' of a type (line 87)
path_8735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 40), os_8734, 'path')
# Obtaining the member 'realpath' of a type (line 87)
realpath_8736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 40), path_8735, 'realpath')
# Calling realpath(args, kwargs) (line 87)
realpath_call_result_8739 = invoke(stypy.reporting.localization.Localization(__file__, 87, 40), realpath_8736, *[file___8737], **kwargs_8738)

# Processing the call keyword arguments (line 87)
kwargs_8740 = {}
# Getting the type of 'os' (line 87)
os_8731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 24), 'os', False)
# Obtaining the member 'path' of a type (line 87)
path_8732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 24), os_8731, 'path')
# Obtaining the member 'dirname' of a type (line 87)
dirname_8733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 24), path_8732, 'dirname')
# Calling dirname(args, kwargs) (line 87)
dirname_call_result_8741 = invoke(stypy.reporting.localization.Localization(__file__, 87, 24), dirname_8733, *[realpath_call_result_8739], **kwargs_8740)

str_8742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 70), 'str', '/foo.txt')
# Applying the binary operator '+' (line 87)
result_add_8743 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 24), '+', dirname_call_result_8741, str_8742)

str_8744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 82), 'str', 'w')
# Processing the call keyword arguments (line 87)
kwargs_8745 = {}
# Getting the type of 'file' (line 87)
file_8730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 19), 'file', False)
# Calling file(args, kwargs) (line 87)
file_call_result_8746 = invoke(stypy.reporting.localization.Localization(__file__, 87, 19), file_8730, *[result_add_8743, str_8744], **kwargs_8745)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 11), tuple_8728, file_call_result_8746)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (file_8727, tuple_8728))
# Adding element type (key, value) (line 67)
# Getting the type of 'xrange' (line 88)
xrange_8747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'xrange')

# Obtaining an instance of the builtin type 'tuple' (line 88)
tuple_8748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 88)
# Adding element type (line 88)
str_8749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 13), 'str', 'xrange')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 13), tuple_8748, str_8749)
# Adding element type (line 88)

# Call to xrange(...): (line 88)
# Processing the call arguments (line 88)
int_8751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 30), 'int')
int_8752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 33), 'int')
# Processing the call keyword arguments (line 88)
kwargs_8753 = {}
# Getting the type of 'xrange' (line 88)
xrange_8750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 23), 'xrange', False)
# Calling xrange(args, kwargs) (line 88)
xrange_call_result_8754 = invoke(stypy.reporting.localization.Localization(__file__, 88, 23), xrange_8750, *[int_8751, int_8752], **kwargs_8753)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 13), tuple_8748, xrange_call_result_8754)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (xrange_8747, tuple_8748))
# Adding element type (key, value) (line 67)
# Getting the type of 'types' (line 89)
types_8755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'types')
# Obtaining the member 'SliceType' of a type (line 89)
SliceType_8756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 4), types_8755, 'SliceType')

# Obtaining an instance of the builtin type 'tuple' (line 89)
tuple_8757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 22), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 89)
# Adding element type (line 89)
str_8758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 22), 'str', 'slice')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 22), tuple_8757, str_8758)
# Adding element type (line 89)

# Call to slice(...): (line 89)
# Processing the call arguments (line 89)
int_8760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 37), 'int')
int_8761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 40), 'int')
int_8762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 43), 'int')
# Processing the call keyword arguments (line 89)
kwargs_8763 = {}
# Getting the type of 'slice' (line 89)
slice_8759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 31), 'slice', False)
# Calling slice(args, kwargs) (line 89)
slice_call_result_8764 = invoke(stypy.reporting.localization.Localization(__file__, 89, 31), slice_8759, *[int_8760, int_8761, int_8762], **kwargs_8763)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 22), tuple_8757, slice_call_result_8764)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (SliceType_8756, tuple_8757))
# Adding element type (key, value) (line 67)
# Getting the type of 'buffer' (line 90)
buffer_8765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'buffer')

# Obtaining an instance of the builtin type 'tuple' (line 90)
tuple_8766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 90)
# Adding element type (line 90)
str_8767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 13), 'str', 'buffer')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 13), tuple_8766, str_8767)
# Adding element type (line 90)

# Call to buffer(...): (line 90)
# Processing the call arguments (line 90)
str_8769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 30), 'str', 'r')
# Processing the call keyword arguments (line 90)
kwargs_8770 = {}
# Getting the type of 'buffer' (line 90)
buffer_8768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 23), 'buffer', False)
# Calling buffer(args, kwargs) (line 90)
buffer_call_result_8771 = invoke(stypy.reporting.localization.Localization(__file__, 90, 23), buffer_8768, *[str_8769], **kwargs_8770)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 13), tuple_8766, buffer_call_result_8771)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (buffer_8765, tuple_8766))
# Adding element type (key, value) (line 67)
# Getting the type of 'types' (line 91)
types_8772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'types')
# Obtaining the member 'DictProxyType' of a type (line 91)
DictProxyType_8773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 4), types_8772, 'DictProxyType')

# Obtaining an instance of the builtin type 'tuple' (line 91)
tuple_8774 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 26), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 91)
# Adding element type (line 91)
str_8775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 26), 'str', 'types.DictProxyType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 26), tuple_8774, str_8775)
# Adding element type (line 91)
# Getting the type of 'int' (line 91)
int_8776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 49), 'int')
# Obtaining the member '__dict__' of a type (line 91)
dict___8777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 49), int_8776, '__dict__')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 26), tuple_8774, dict___8777)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (DictProxyType_8773, tuple_8774))
# Adding element type (key, value) (line 67)
# Getting the type of 'types' (line 92)
types_8778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'types')
# Obtaining the member 'ClassType' of a type (line 92)
ClassType_8779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 4), types_8778, 'ClassType')

# Obtaining an instance of the builtin type 'tuple' (line 92)
tuple_8780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 22), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 92)
# Adding element type (line 92)
str_8781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 22), 'str', 'types.ClassType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 22), tuple_8780, str_8781)
# Adding element type (line 92)
# Getting the type of 'Foo' (line 92)
Foo_8782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 41), 'Foo')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 22), tuple_8780, Foo_8782)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (ClassType_8779, tuple_8780))
# Adding element type (key, value) (line 67)
# Getting the type of 'types' (line 93)
types_8783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'types')
# Obtaining the member 'InstanceType' of a type (line 93)
InstanceType_8784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 4), types_8783, 'InstanceType')

# Obtaining an instance of the builtin type 'tuple' (line 93)
tuple_8785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 25), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 93)
# Adding element type (line 93)
str_8786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 25), 'str', 'types.InstanceType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 25), tuple_8785, str_8786)
# Adding element type (line 93)
# Getting the type of 'foo' (line 93)
foo_8787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 47), 'foo')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 25), tuple_8785, foo_8787)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (InstanceType_8784, tuple_8785))
# Adding element type (key, value) (line 67)
# Getting the type of 'types' (line 94)
types_8788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'types')
# Obtaining the member 'MethodType' of a type (line 94)
MethodType_8789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 4), types_8788, 'MethodType')

# Obtaining an instance of the builtin type 'tuple' (line 94)
tuple_8790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 23), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 94)
# Adding element type (line 94)
str_8791 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 23), 'str', 'types.MethodType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 23), tuple_8790, str_8791)
# Adding element type (line 94)
# Getting the type of 'foo' (line 94)
foo_8792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 43), 'foo')
# Obtaining the member 'bar' of a type (line 94)
bar_8793 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 43), foo_8792, 'bar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 23), tuple_8790, bar_8793)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (MethodType_8789, tuple_8790))
# Adding element type (key, value) (line 67)
# Getting the type of 'iter' (line 95)
iter_8794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'iter')

# Obtaining an instance of the builtin type 'tuple' (line 95)
tuple_8795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 11), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 95)
# Adding element type (line 95)
str_8796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 11), 'str', 'iter')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 11), tuple_8795, str_8796)
# Adding element type (line 95)

# Call to iter(...): (line 95)
# Processing the call arguments (line 95)
str_8798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 24), 'str', 'abc')
# Processing the call keyword arguments (line 95)
kwargs_8799 = {}
# Getting the type of 'iter' (line 95)
iter_8797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 19), 'iter', False)
# Calling iter(args, kwargs) (line 95)
iter_call_result_8800 = invoke(stypy.reporting.localization.Localization(__file__, 95, 19), iter_8797, *[str_8798], **kwargs_8799)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 11), tuple_8795, iter_call_result_8800)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (iter_8794, tuple_8795))
# Adding element type (key, value) (line 67)
# Getting the type of 'bytearray' (line 97)
bytearray_8801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'bytearray')

# Obtaining an instance of the builtin type 'tuple' (line 97)
tuple_8802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 97)
# Adding element type (line 97)
str_8803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 16), 'str', 'bytearray')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 16), tuple_8802, str_8803)
# Adding element type (line 97)

# Call to bytearray(...): (line 97)
# Processing the call arguments (line 97)
str_8805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 39), 'str', 'test')
# Processing the call keyword arguments (line 97)
kwargs_8806 = {}
# Getting the type of 'bytearray' (line 97)
bytearray_8804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 29), 'bytearray', False)
# Calling bytearray(args, kwargs) (line 97)
bytearray_call_result_8807 = invoke(stypy.reporting.localization.Localization(__file__, 97, 29), bytearray_8804, *[str_8805], **kwargs_8806)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 16), tuple_8802, bytearray_call_result_8807)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (bytearray_8801, tuple_8802))
# Adding element type (key, value) (line 67)
# Getting the type of 'classmethod' (line 98)
classmethod_8808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'classmethod')

# Obtaining an instance of the builtin type 'tuple' (line 98)
tuple_8809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 98)
# Adding element type (line 98)
str_8810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 18), 'str', 'classmethod')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 18), tuple_8809, str_8810)
# Adding element type (line 98)

# Call to classmethod(...): (line 98)
# Processing the call arguments (line 98)
# Getting the type of 'Foo' (line 98)
Foo_8812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 45), 'Foo', False)
# Obtaining the member 'bar' of a type (line 98)
bar_8813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 45), Foo_8812, 'bar')
# Processing the call keyword arguments (line 98)
kwargs_8814 = {}
# Getting the type of 'classmethod' (line 98)
classmethod_8811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 33), 'classmethod', False)
# Calling classmethod(args, kwargs) (line 98)
classmethod_call_result_8815 = invoke(stypy.reporting.localization.Localization(__file__, 98, 33), classmethod_8811, *[bar_8813], **kwargs_8814)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 18), tuple_8809, classmethod_call_result_8815)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (classmethod_8808, tuple_8809))
# Adding element type (key, value) (line 67)
# Getting the type of 'enumerate' (line 99)
enumerate_8816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'enumerate')

# Obtaining an instance of the builtin type 'tuple' (line 99)
tuple_8817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 99)
# Adding element type (line 99)
str_8818 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 16), 'str', 'enumerate')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 16), tuple_8817, str_8818)
# Adding element type (line 99)

# Call to enumerate(...): (line 99)
# Processing the call arguments (line 99)

# Obtaining an instance of the builtin type 'list' (line 99)
list_8820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 39), 'list')
# Adding type elements to the builtin type 'list' instance (line 99)
# Adding element type (line 99)
int_8821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 40), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 39), list_8820, int_8821)
# Adding element type (line 99)
int_8822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 43), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 39), list_8820, int_8822)
# Adding element type (line 99)
int_8823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 46), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 39), list_8820, int_8823)

# Processing the call keyword arguments (line 99)
kwargs_8824 = {}
# Getting the type of 'enumerate' (line 99)
enumerate_8819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 29), 'enumerate', False)
# Calling enumerate(args, kwargs) (line 99)
enumerate_call_result_8825 = invoke(stypy.reporting.localization.Localization(__file__, 99, 29), enumerate_8819, *[list_8820], **kwargs_8824)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 16), tuple_8817, enumerate_call_result_8825)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (enumerate_8816, tuple_8817))
# Adding element type (key, value) (line 67)
# Getting the type of 'frozenset' (line 100)
frozenset_8826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'frozenset')

# Obtaining an instance of the builtin type 'tuple' (line 100)
tuple_8827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 100)
# Adding element type (line 100)
str_8828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 16), 'str', 'frozenset')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 16), tuple_8827, str_8828)
# Adding element type (line 100)

# Call to frozenset(...): (line 100)
# Processing the call arguments (line 100)

# Obtaining an instance of the builtin type 'list' (line 100)
list_8830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 39), 'list')
# Adding type elements to the builtin type 'list' instance (line 100)
# Adding element type (line 100)
int_8831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 40), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 39), list_8830, int_8831)
# Adding element type (line 100)
int_8832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 43), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 39), list_8830, int_8832)
# Adding element type (line 100)
int_8833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 46), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 39), list_8830, int_8833)

# Processing the call keyword arguments (line 100)
kwargs_8834 = {}
# Getting the type of 'frozenset' (line 100)
frozenset_8829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 29), 'frozenset', False)
# Calling frozenset(args, kwargs) (line 100)
frozenset_call_result_8835 = invoke(stypy.reporting.localization.Localization(__file__, 100, 29), frozenset_8829, *[list_8830], **kwargs_8834)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 16), tuple_8827, frozenset_call_result_8835)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (frozenset_8826, tuple_8827))
# Adding element type (key, value) (line 67)
# Getting the type of 'memoryview' (line 101)
memoryview_8836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'memoryview')

# Obtaining an instance of the builtin type 'tuple' (line 101)
tuple_8837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 101)
# Adding element type (line 101)
str_8838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 17), 'str', 'memoryview')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 17), tuple_8837, str_8838)
# Adding element type (line 101)

# Call to memoryview(...): (line 101)
# Processing the call arguments (line 101)

# Call to buffer(...): (line 101)
# Processing the call arguments (line 101)
str_8841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 49), 'str', 'foo')
# Processing the call keyword arguments (line 101)
kwargs_8842 = {}
# Getting the type of 'buffer' (line 101)
buffer_8840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 42), 'buffer', False)
# Calling buffer(args, kwargs) (line 101)
buffer_call_result_8843 = invoke(stypy.reporting.localization.Localization(__file__, 101, 42), buffer_8840, *[str_8841], **kwargs_8842)

# Processing the call keyword arguments (line 101)
kwargs_8844 = {}
# Getting the type of 'memoryview' (line 101)
memoryview_8839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 31), 'memoryview', False)
# Calling memoryview(args, kwargs) (line 101)
memoryview_call_result_8845 = invoke(stypy.reporting.localization.Localization(__file__, 101, 31), memoryview_8839, *[buffer_call_result_8843], **kwargs_8844)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 17), tuple_8837, memoryview_call_result_8845)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (memoryview_8836, tuple_8837))
# Adding element type (key, value) (line 67)
# Getting the type of 'object' (line 102)
object_8846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'object')

# Obtaining an instance of the builtin type 'tuple' (line 102)
tuple_8847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 102)
# Adding element type (line 102)
str_8848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 13), 'str', 'object')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 13), tuple_8847, str_8848)
# Adding element type (line 102)

# Call to object(...): (line 102)
# Processing the call keyword arguments (line 102)
kwargs_8850 = {}
# Getting the type of 'object' (line 102)
object_8849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 23), 'object', False)
# Calling object(args, kwargs) (line 102)
object_call_result_8851 = invoke(stypy.reporting.localization.Localization(__file__, 102, 23), object_8849, *[], **kwargs_8850)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 13), tuple_8847, object_call_result_8851)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (object_8846, tuple_8847))
# Adding element type (key, value) (line 67)
# Getting the type of 'property' (line 103)
property_8852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'property')

# Obtaining an instance of the builtin type 'tuple' (line 103)
tuple_8853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 103)
# Adding element type (line 103)
str_8854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 15), 'str', 'property')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 15), tuple_8853, str_8854)
# Adding element type (line 103)

# Call to property(...): (line 103)
# Processing the call arguments (line 103)
# Getting the type of 'Foo' (line 103)
Foo_8856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 36), 'Foo', False)
# Obtaining the member 'qux' of a type (line 103)
qux_8857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 36), Foo_8856, 'qux')
# Processing the call keyword arguments (line 103)
kwargs_8858 = {}
# Getting the type of 'property' (line 103)
property_8855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 27), 'property', False)
# Calling property(args, kwargs) (line 103)
property_call_result_8859 = invoke(stypy.reporting.localization.Localization(__file__, 103, 27), property_8855, *[qux_8857], **kwargs_8858)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 15), tuple_8853, property_call_result_8859)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (property_8852, tuple_8853))
# Adding element type (key, value) (line 67)
# Getting the type of 'staticmethod' (line 104)
staticmethod_8860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'staticmethod')

# Obtaining an instance of the builtin type 'tuple' (line 104)
tuple_8861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 19), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 104)
# Adding element type (line 104)
str_8862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 19), 'str', 'staticmethod')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 19), tuple_8861, str_8862)
# Adding element type (line 104)

# Call to staticmethod(...): (line 104)
# Processing the call arguments (line 104)
# Getting the type of 'Foo' (line 104)
Foo_8864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 48), 'Foo', False)
# Obtaining the member 'bar' of a type (line 104)
bar_8865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 48), Foo_8864, 'bar')
# Processing the call keyword arguments (line 104)
kwargs_8866 = {}
# Getting the type of 'staticmethod' (line 104)
staticmethod_8863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 35), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 104)
staticmethod_call_result_8867 = invoke(stypy.reporting.localization.Localization(__file__, 104, 35), staticmethod_8863, *[bar_8865], **kwargs_8866)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 19), tuple_8861, staticmethod_call_result_8867)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (staticmethod_8860, tuple_8861))
# Adding element type (key, value) (line 67)
# Getting the type of 'super' (line 105)
super_8868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'super')

# Obtaining an instance of the builtin type 'tuple' (line 105)
tuple_8869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 12), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 105)
# Adding element type (line 105)
str_8870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 12), 'str', 'super')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 12), tuple_8869, str_8870)
# Adding element type (line 105)

# Call to super(...): (line 105)
# Processing the call arguments (line 105)

# Call to type(...): (line 105)
# Processing the call arguments (line 105)
# Getting the type of 'Foo' (line 105)
Foo_8873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 32), 'Foo', False)
# Processing the call keyword arguments (line 105)
kwargs_8874 = {}
# Getting the type of 'type' (line 105)
type_8872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 27), 'type', False)
# Calling type(args, kwargs) (line 105)
type_call_result_8875 = invoke(stypy.reporting.localization.Localization(__file__, 105, 27), type_8872, *[Foo_8873], **kwargs_8874)

# Processing the call keyword arguments (line 105)
kwargs_8876 = {}
# Getting the type of 'super' (line 105)
super_8871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 21), 'super', False)
# Calling super(args, kwargs) (line 105)
super_call_result_8877 = invoke(stypy.reporting.localization.Localization(__file__, 105, 21), super_8871, *[type_call_result_8875], **kwargs_8876)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 12), tuple_8869, super_call_result_8877)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (super_8868, tuple_8869))
# Adding element type (key, value) (line 67)
# Getting the type of 'reversed' (line 106)
reversed_8878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'reversed')

# Obtaining an instance of the builtin type 'tuple' (line 106)
tuple_8879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 106)
# Adding element type (line 106)
str_8880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 15), 'str', 'reversed')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 15), tuple_8879, str_8880)
# Adding element type (line 106)

# Call to reversed(...): (line 106)
# Processing the call arguments (line 106)

# Obtaining an instance of the builtin type 'tuple' (line 106)
tuple_8882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 37), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 106)
# Adding element type (line 106)
int_8883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 37), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 37), tuple_8882, int_8883)
# Adding element type (line 106)
int_8884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 40), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 37), tuple_8882, int_8884)

# Processing the call keyword arguments (line 106)
kwargs_8885 = {}
# Getting the type of 'reversed' (line 106)
reversed_8881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 27), 'reversed', False)
# Calling reversed(args, kwargs) (line 106)
reversed_call_result_8886 = invoke(stypy.reporting.localization.Localization(__file__, 106, 27), reversed_8881, *[tuple_8882], **kwargs_8885)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 15), tuple_8879, reversed_call_result_8886)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (reversed_8878, tuple_8879))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 108)
# Processing the call arguments (line 108)

# Call to iter(...): (line 108)
# Processing the call arguments (line 108)

# Obtaining an instance of the builtin type 'tuple' (line 108)
tuple_8889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 108)
# Adding element type (line 108)
int_8890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 15), tuple_8889, int_8890)
# Adding element type (line 108)
int_8891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 15), tuple_8889, int_8891)
# Adding element type (line 108)
int_8892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 15), tuple_8889, int_8892)
# Adding element type (line 108)
int_8893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 15), tuple_8889, int_8893)
# Adding element type (line 108)
int_8894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 15), tuple_8889, int_8894)

# Processing the call keyword arguments (line 108)
kwargs_8895 = {}
# Getting the type of 'iter' (line 108)
iter_8888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 9), 'iter', False)
# Calling iter(args, kwargs) (line 108)
iter_call_result_8896 = invoke(stypy.reporting.localization.Localization(__file__, 108, 9), iter_8888, *[tuple_8889], **kwargs_8895)

# Processing the call keyword arguments (line 108)
kwargs_8897 = {}
# Getting the type of 'type' (line 108)
type_8887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'type', False)
# Calling type(args, kwargs) (line 108)
type_call_result_8898 = invoke(stypy.reporting.localization.Localization(__file__, 108, 4), type_8887, *[iter_call_result_8896], **kwargs_8897)


# Obtaining an instance of the builtin type 'tuple' (line 108)
tuple_8899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 34), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 108)
# Adding element type (line 108)
str_8900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 34), 'str', 'ExtraTypeDefinitions.tupleiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 34), tuple_8899, str_8900)
# Adding element type (line 108)

# Call to __iter__(...): (line 108)
# Processing the call keyword arguments (line 108)
kwargs_8908 = {}

# Obtaining an instance of the builtin type 'tuple' (line 108)
tuple_8901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 73), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 108)
# Adding element type (line 108)
int_8902 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 73), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 73), tuple_8901, int_8902)
# Adding element type (line 108)
int_8903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 76), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 73), tuple_8901, int_8903)
# Adding element type (line 108)
int_8904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 79), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 73), tuple_8901, int_8904)
# Adding element type (line 108)
int_8905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 82), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 73), tuple_8901, int_8905)
# Adding element type (line 108)
int_8906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 85), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 73), tuple_8901, int_8906)

# Obtaining the member '__iter__' of a type (line 108)
iter___8907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 73), tuple_8901, '__iter__')
# Calling __iter__(args, kwargs) (line 108)
iter___call_result_8909 = invoke(stypy.reporting.localization.Localization(__file__, 108, 73), iter___8907, *[], **kwargs_8908)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 34), tuple_8899, iter___call_result_8909)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (type_call_result_8898, tuple_8899))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 109)
# Processing the call arguments (line 109)

# Call to iter(...): (line 109)
# Processing the call arguments (line 109)

# Call to xrange(...): (line 109)
# Processing the call arguments (line 109)
int_8913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 21), 'int')
# Processing the call keyword arguments (line 109)
kwargs_8914 = {}
# Getting the type of 'xrange' (line 109)
xrange_8912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 14), 'xrange', False)
# Calling xrange(args, kwargs) (line 109)
xrange_call_result_8915 = invoke(stypy.reporting.localization.Localization(__file__, 109, 14), xrange_8912, *[int_8913], **kwargs_8914)

# Processing the call keyword arguments (line 109)
kwargs_8916 = {}
# Getting the type of 'iter' (line 109)
iter_8911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 9), 'iter', False)
# Calling iter(args, kwargs) (line 109)
iter_call_result_8917 = invoke(stypy.reporting.localization.Localization(__file__, 109, 9), iter_8911, *[xrange_call_result_8915], **kwargs_8916)

# Processing the call keyword arguments (line 109)
kwargs_8918 = {}
# Getting the type of 'type' (line 109)
type_8910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'type', False)
# Calling type(args, kwargs) (line 109)
type_call_result_8919 = invoke(stypy.reporting.localization.Localization(__file__, 109, 4), type_8910, *[iter_call_result_8917], **kwargs_8918)


# Obtaining an instance of the builtin type 'tuple' (line 109)
tuple_8920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 28), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 109)
# Adding element type (line 109)
str_8921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 28), 'str', 'ExtraTypeDefinitions.rangeiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 28), tuple_8920, str_8921)
# Adding element type (line 109)

# Call to iter(...): (line 109)
# Processing the call arguments (line 109)

# Call to xrange(...): (line 109)
# Processing the call arguments (line 109)
int_8924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 78), 'int')
# Processing the call keyword arguments (line 109)
kwargs_8925 = {}
# Getting the type of 'xrange' (line 109)
xrange_8923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 71), 'xrange', False)
# Calling xrange(args, kwargs) (line 109)
xrange_call_result_8926 = invoke(stypy.reporting.localization.Localization(__file__, 109, 71), xrange_8923, *[int_8924], **kwargs_8925)

# Processing the call keyword arguments (line 109)
kwargs_8927 = {}
# Getting the type of 'iter' (line 109)
iter_8922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 66), 'iter', False)
# Calling iter(args, kwargs) (line 109)
iter_call_result_8928 = invoke(stypy.reporting.localization.Localization(__file__, 109, 66), iter_8922, *[xrange_call_result_8926], **kwargs_8927)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 28), tuple_8920, iter_call_result_8928)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (type_call_result_8919, tuple_8920))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 110)
# Processing the call arguments (line 110)

# Call to iter(...): (line 110)
# Processing the call arguments (line 110)

# Obtaining an instance of the builtin type 'list' (line 110)
list_8931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 110)
# Adding element type (line 110)
int_8932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 14), list_8931, int_8932)
# Adding element type (line 110)
int_8933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 14), list_8931, int_8933)

# Processing the call keyword arguments (line 110)
kwargs_8934 = {}
# Getting the type of 'iter' (line 110)
iter_8930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 9), 'iter', False)
# Calling iter(args, kwargs) (line 110)
iter_call_result_8935 = invoke(stypy.reporting.localization.Localization(__file__, 110, 9), iter_8930, *[list_8931], **kwargs_8934)

# Processing the call keyword arguments (line 110)
kwargs_8936 = {}
# Getting the type of 'type' (line 110)
type_8929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'type', False)
# Calling type(args, kwargs) (line 110)
type_call_result_8937 = invoke(stypy.reporting.localization.Localization(__file__, 110, 4), type_8929, *[iter_call_result_8935], **kwargs_8936)


# Obtaining an instance of the builtin type 'tuple' (line 110)
tuple_8938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 25), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 110)
# Adding element type (line 110)
str_8939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 25), 'str', 'ExtraTypeDefinitions.listiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 25), tuple_8938, str_8939)
# Adding element type (line 110)

# Call to __iter__(...): (line 110)
# Processing the call keyword arguments (line 110)
kwargs_8947 = {}

# Obtaining an instance of the builtin type 'list' (line 110)
list_8940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 62), 'list')
# Adding type elements to the builtin type 'list' instance (line 110)
# Adding element type (line 110)
int_8941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 63), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 62), list_8940, int_8941)
# Adding element type (line 110)
int_8942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 66), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 62), list_8940, int_8942)
# Adding element type (line 110)
int_8943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 69), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 62), list_8940, int_8943)
# Adding element type (line 110)
int_8944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 72), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 62), list_8940, int_8944)
# Adding element type (line 110)
int_8945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 75), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 62), list_8940, int_8945)

# Obtaining the member '__iter__' of a type (line 110)
iter___8946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 62), list_8940, '__iter__')
# Calling __iter__(args, kwargs) (line 110)
iter___call_result_8948 = invoke(stypy.reporting.localization.Localization(__file__, 110, 62), iter___8946, *[], **kwargs_8947)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 25), tuple_8938, iter___call_result_8948)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (type_call_result_8937, tuple_8938))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 111)
# Processing the call arguments (line 111)

# Call to iter(...): (line 111)
# Processing the call arguments (line 111)

# Call to type(...): (line 111)
# Processing the call arguments (line 111)
# Getting the type of 'int' (line 111)
int_8952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'int', False)
# Processing the call keyword arguments (line 111)
kwargs_8953 = {}
# Getting the type of 'type' (line 111)
type_8951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 14), 'type', False)
# Calling type(args, kwargs) (line 111)
type_call_result_8954 = invoke(stypy.reporting.localization.Localization(__file__, 111, 14), type_8951, *[int_8952], **kwargs_8953)

float_8955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 25), 'float')
# Processing the call keyword arguments (line 111)
kwargs_8956 = {}
# Getting the type of 'iter' (line 111)
iter_8950 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 9), 'iter', False)
# Calling iter(args, kwargs) (line 111)
iter_call_result_8957 = invoke(stypy.reporting.localization.Localization(__file__, 111, 9), iter_8950, *[type_call_result_8954, float_8955], **kwargs_8956)

# Processing the call keyword arguments (line 111)
kwargs_8958 = {}
# Getting the type of 'type' (line 111)
type_8949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'type', False)
# Calling type(args, kwargs) (line 111)
type_call_result_8959 = invoke(stypy.reporting.localization.Localization(__file__, 111, 4), type_8949, *[iter_call_result_8957], **kwargs_8958)


# Obtaining an instance of the builtin type 'tuple' (line 111)
tuple_8960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 33), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 111)
# Adding element type (line 111)
str_8961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 33), 'str', 'ExtraTypeDefinitions.callable_iterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 33), tuple_8960, str_8961)
# Adding element type (line 111)

# Call to iter(...): (line 111)
# Processing the call arguments (line 111)

# Call to type(...): (line 111)
# Processing the call arguments (line 111)
# Getting the type of 'int' (line 111)
int_8964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 85), 'int', False)
# Processing the call keyword arguments (line 111)
kwargs_8965 = {}
# Getting the type of 'type' (line 111)
type_8963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 80), 'type', False)
# Calling type(args, kwargs) (line 111)
type_call_result_8966 = invoke(stypy.reporting.localization.Localization(__file__, 111, 80), type_8963, *[int_8964], **kwargs_8965)

float_8967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 91), 'float')
# Processing the call keyword arguments (line 111)
kwargs_8968 = {}
# Getting the type of 'iter' (line 111)
iter_8962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 75), 'iter', False)
# Calling iter(args, kwargs) (line 111)
iter_call_result_8969 = invoke(stypy.reporting.localization.Localization(__file__, 111, 75), iter_8962, *[type_call_result_8966, float_8967], **kwargs_8968)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 33), tuple_8960, iter_call_result_8969)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (type_call_result_8959, tuple_8960))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 112)
# Processing the call arguments (line 112)

# Call to iter(...): (line 112)
# Processing the call arguments (line 112)

# Call to reversed(...): (line 112)
# Processing the call arguments (line 112)

# Obtaining an instance of the builtin type 'list' (line 112)
list_8973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 112)
# Adding element type (line 112)
int_8974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 23), list_8973, int_8974)
# Adding element type (line 112)
int_8975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 23), list_8973, int_8975)

# Processing the call keyword arguments (line 112)
kwargs_8976 = {}
# Getting the type of 'reversed' (line 112)
reversed_8972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 14), 'reversed', False)
# Calling reversed(args, kwargs) (line 112)
reversed_call_result_8977 = invoke(stypy.reporting.localization.Localization(__file__, 112, 14), reversed_8972, *[list_8973], **kwargs_8976)

# Processing the call keyword arguments (line 112)
kwargs_8978 = {}
# Getting the type of 'iter' (line 112)
iter_8971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 9), 'iter', False)
# Calling iter(args, kwargs) (line 112)
iter_call_result_8979 = invoke(stypy.reporting.localization.Localization(__file__, 112, 9), iter_8971, *[reversed_call_result_8977], **kwargs_8978)

# Processing the call keyword arguments (line 112)
kwargs_8980 = {}
# Getting the type of 'type' (line 112)
type_8970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'type', False)
# Calling type(args, kwargs) (line 112)
type_call_result_8981 = invoke(stypy.reporting.localization.Localization(__file__, 112, 4), type_8970, *[iter_call_result_8979], **kwargs_8980)


# Obtaining an instance of the builtin type 'tuple' (line 112)
tuple_8982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 35), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 112)
# Adding element type (line 112)
str_8983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 35), 'str', 'ExtraTypeDefinitions.listreverseiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 35), tuple_8982, str_8983)
# Adding element type (line 112)

# Call to iter(...): (line 112)
# Processing the call arguments (line 112)

# Call to reversed(...): (line 112)
# Processing the call arguments (line 112)

# Obtaining an instance of the builtin type 'list' (line 112)
list_8986 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 93), 'list')
# Adding type elements to the builtin type 'list' instance (line 112)
# Adding element type (line 112)
int_8987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 94), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 93), list_8986, int_8987)
# Adding element type (line 112)
int_8988 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 97), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 93), list_8986, int_8988)

# Processing the call keyword arguments (line 112)
kwargs_8989 = {}
# Getting the type of 'reversed' (line 112)
reversed_8985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 84), 'reversed', False)
# Calling reversed(args, kwargs) (line 112)
reversed_call_result_8990 = invoke(stypy.reporting.localization.Localization(__file__, 112, 84), reversed_8985, *[list_8986], **kwargs_8989)

# Processing the call keyword arguments (line 112)
kwargs_8991 = {}
# Getting the type of 'iter' (line 112)
iter_8984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 79), 'iter', False)
# Calling iter(args, kwargs) (line 112)
iter_call_result_8992 = invoke(stypy.reporting.localization.Localization(__file__, 112, 79), iter_8984, *[reversed_call_result_8990], **kwargs_8991)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 35), tuple_8982, iter_call_result_8992)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (type_call_result_8981, tuple_8982))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 113)
# Processing the call arguments (line 113)

# Call to methodcaller(...): (line 113)
# Processing the call arguments (line 113)
int_8996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 31), 'int')
# Processing the call keyword arguments (line 113)
kwargs_8997 = {}
# Getting the type of 'operator' (line 113)
operator_8994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 9), 'operator', False)
# Obtaining the member 'methodcaller' of a type (line 113)
methodcaller_8995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 9), operator_8994, 'methodcaller')
# Calling methodcaller(args, kwargs) (line 113)
methodcaller_call_result_8998 = invoke(stypy.reporting.localization.Localization(__file__, 113, 9), methodcaller_8995, *[int_8996], **kwargs_8997)

# Processing the call keyword arguments (line 113)
kwargs_8999 = {}
# Getting the type of 'type' (line 113)
type_8993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'type', False)
# Calling type(args, kwargs) (line 113)
type_call_result_9000 = invoke(stypy.reporting.localization.Localization(__file__, 113, 4), type_8993, *[methodcaller_call_result_8998], **kwargs_8999)


# Obtaining an instance of the builtin type 'tuple' (line 113)
tuple_9001 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 37), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 113)
# Adding element type (line 113)
str_9002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 37), 'str', 'ExtraTypeDefinitions.methodcaller')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 37), tuple_9001, str_9002)
# Adding element type (line 113)

# Call to methodcaller(...): (line 113)
# Processing the call arguments (line 113)
int_9005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 96), 'int')
# Processing the call keyword arguments (line 113)
kwargs_9006 = {}
# Getting the type of 'operator' (line 113)
operator_9003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 74), 'operator', False)
# Obtaining the member 'methodcaller' of a type (line 113)
methodcaller_9004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 74), operator_9003, 'methodcaller')
# Calling methodcaller(args, kwargs) (line 113)
methodcaller_call_result_9007 = invoke(stypy.reporting.localization.Localization(__file__, 113, 74), methodcaller_9004, *[int_9005], **kwargs_9006)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 37), tuple_9001, methodcaller_call_result_9007)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (type_call_result_9000, tuple_9001))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 114)
# Processing the call arguments (line 114)

# Call to itemgetter(...): (line 114)
# Processing the call arguments (line 114)
int_9011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 29), 'int')
# Processing the call keyword arguments (line 114)
kwargs_9012 = {}
# Getting the type of 'operator' (line 114)
operator_9009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 9), 'operator', False)
# Obtaining the member 'itemgetter' of a type (line 114)
itemgetter_9010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 9), operator_9009, 'itemgetter')
# Calling itemgetter(args, kwargs) (line 114)
itemgetter_call_result_9013 = invoke(stypy.reporting.localization.Localization(__file__, 114, 9), itemgetter_9010, *[int_9011], **kwargs_9012)

# Processing the call keyword arguments (line 114)
kwargs_9014 = {}
# Getting the type of 'type' (line 114)
type_9008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'type', False)
# Calling type(args, kwargs) (line 114)
type_call_result_9015 = invoke(stypy.reporting.localization.Localization(__file__, 114, 4), type_9008, *[itemgetter_call_result_9013], **kwargs_9014)


# Obtaining an instance of the builtin type 'tuple' (line 114)
tuple_9016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 35), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 114)
# Adding element type (line 114)
str_9017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 35), 'str', 'ExtraTypeDefinitions.itemgetter')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 35), tuple_9016, str_9017)
# Adding element type (line 114)

# Call to itemgetter(...): (line 114)
# Processing the call arguments (line 114)
int_9020 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 90), 'int')
# Processing the call keyword arguments (line 114)
kwargs_9021 = {}
# Getting the type of 'operator' (line 114)
operator_9018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 70), 'operator', False)
# Obtaining the member 'itemgetter' of a type (line 114)
itemgetter_9019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 70), operator_9018, 'itemgetter')
# Calling itemgetter(args, kwargs) (line 114)
itemgetter_call_result_9022 = invoke(stypy.reporting.localization.Localization(__file__, 114, 70), itemgetter_9019, *[int_9020], **kwargs_9021)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 35), tuple_9016, itemgetter_call_result_9022)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (type_call_result_9015, tuple_9016))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 115)
# Processing the call arguments (line 115)

# Call to attrgetter(...): (line 115)
# Processing the call arguments (line 115)
int_9026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 29), 'int')
# Processing the call keyword arguments (line 115)
kwargs_9027 = {}
# Getting the type of 'operator' (line 115)
operator_9024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 9), 'operator', False)
# Obtaining the member 'attrgetter' of a type (line 115)
attrgetter_9025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 9), operator_9024, 'attrgetter')
# Calling attrgetter(args, kwargs) (line 115)
attrgetter_call_result_9028 = invoke(stypy.reporting.localization.Localization(__file__, 115, 9), attrgetter_9025, *[int_9026], **kwargs_9027)

# Processing the call keyword arguments (line 115)
kwargs_9029 = {}
# Getting the type of 'type' (line 115)
type_9023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'type', False)
# Calling type(args, kwargs) (line 115)
type_call_result_9030 = invoke(stypy.reporting.localization.Localization(__file__, 115, 4), type_9023, *[attrgetter_call_result_9028], **kwargs_9029)


# Obtaining an instance of the builtin type 'tuple' (line 115)
tuple_9031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 35), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 115)
# Adding element type (line 115)
str_9032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 35), 'str', 'ExtraTypeDefinitions.attrgetter')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 35), tuple_9031, str_9032)
# Adding element type (line 115)

# Call to attrgetter(...): (line 115)
# Processing the call arguments (line 115)
int_9035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 90), 'int')
# Processing the call keyword arguments (line 115)
kwargs_9036 = {}
# Getting the type of 'operator' (line 115)
operator_9033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 70), 'operator', False)
# Obtaining the member 'attrgetter' of a type (line 115)
attrgetter_9034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 70), operator_9033, 'attrgetter')
# Calling attrgetter(args, kwargs) (line 115)
attrgetter_call_result_9037 = invoke(stypy.reporting.localization.Localization(__file__, 115, 70), attrgetter_9034, *[int_9035], **kwargs_9036)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 35), tuple_9031, attrgetter_call_result_9037)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (type_call_result_9030, tuple_9031))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 116)
# Processing the call arguments (line 116)

# Call to iter(...): (line 116)
# Processing the call arguments (line 116)

# Call to bytearray(...): (line 116)
# Processing the call arguments (line 116)
str_9041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 24), 'str', 'test')
# Processing the call keyword arguments (line 116)
kwargs_9042 = {}
# Getting the type of 'bytearray' (line 116)
bytearray_9040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 14), 'bytearray', False)
# Calling bytearray(args, kwargs) (line 116)
bytearray_call_result_9043 = invoke(stypy.reporting.localization.Localization(__file__, 116, 14), bytearray_9040, *[str_9041], **kwargs_9042)

# Processing the call keyword arguments (line 116)
kwargs_9044 = {}
# Getting the type of 'iter' (line 116)
iter_9039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 9), 'iter', False)
# Calling iter(args, kwargs) (line 116)
iter_call_result_9045 = invoke(stypy.reporting.localization.Localization(__file__, 116, 9), iter_9039, *[bytearray_call_result_9043], **kwargs_9044)

# Processing the call keyword arguments (line 116)
kwargs_9046 = {}
# Getting the type of 'type' (line 116)
type_9038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'type', False)
# Calling type(args, kwargs) (line 116)
type_call_result_9047 = invoke(stypy.reporting.localization.Localization(__file__, 116, 4), type_9038, *[iter_call_result_9045], **kwargs_9046)


# Obtaining an instance of the builtin type 'tuple' (line 116)
tuple_9048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 36), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 116)
# Adding element type (line 116)
str_9049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 36), 'str', 'ExtraTypeDefinitions.bytearray_iterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 36), tuple_9048, str_9049)
# Adding element type (line 116)

# Call to iter(...): (line 116)
# Processing the call arguments (line 116)

# Call to bytearray(...): (line 116)
# Processing the call arguments (line 116)
str_9052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 94), 'str', 'test')
# Processing the call keyword arguments (line 116)
kwargs_9053 = {}
# Getting the type of 'bytearray' (line 116)
bytearray_9051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 84), 'bytearray', False)
# Calling bytearray(args, kwargs) (line 116)
bytearray_call_result_9054 = invoke(stypy.reporting.localization.Localization(__file__, 116, 84), bytearray_9051, *[str_9052], **kwargs_9053)

# Processing the call keyword arguments (line 116)
kwargs_9055 = {}
# Getting the type of 'iter' (line 116)
iter_9050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 79), 'iter', False)
# Calling iter(args, kwargs) (line 116)
iter_call_result_9056 = invoke(stypy.reporting.localization.Localization(__file__, 116, 79), iter_9050, *[bytearray_call_result_9054], **kwargs_9055)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 36), tuple_9048, iter_call_result_9056)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (type_call_result_9047, tuple_9048))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 118)
# Processing the call arguments (line 118)

# Call to viewitems(...): (line 118)
# Processing the call keyword arguments (line 118)
kwargs_9064 = {}

# Obtaining an instance of the builtin type 'dict' (line 118)
dict_9058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 9), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 118)
# Adding element type (key, value) (line 118)
str_9059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 10), 'str', 'a')
int_9060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 15), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 9), dict_9058, (str_9059, int_9060))
# Adding element type (key, value) (line 118)
str_9061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 18), 'str', 'b')
int_9062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 23), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 9), dict_9058, (str_9061, int_9062))

# Obtaining the member 'viewitems' of a type (line 118)
viewitems_9063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 9), dict_9058, 'viewitems')
# Calling viewitems(args, kwargs) (line 118)
viewitems_call_result_9065 = invoke(stypy.reporting.localization.Localization(__file__, 118, 9), viewitems_9063, *[], **kwargs_9064)

# Processing the call keyword arguments (line 118)
kwargs_9066 = {}
# Getting the type of 'type' (line 118)
type_9057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'type', False)
# Calling type(args, kwargs) (line 118)
type_call_result_9067 = invoke(stypy.reporting.localization.Localization(__file__, 118, 4), type_9057, *[viewitems_call_result_9065], **kwargs_9066)


# Obtaining an instance of the builtin type 'tuple' (line 118)
tuple_9068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 41), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 118)
# Adding element type (line 118)
str_9069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 41), 'str', 'ExtraTypeDefinitions.dict_items')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 41), tuple_9068, str_9069)
# Adding element type (line 118)

# Call to viewitems(...): (line 118)
# Processing the call keyword arguments (line 118)
kwargs_9076 = {}

# Obtaining an instance of the builtin type 'dict' (line 118)
dict_9070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 76), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 118)
# Adding element type (key, value) (line 118)
str_9071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 77), 'str', 'a')
int_9072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 82), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 76), dict_9070, (str_9071, int_9072))
# Adding element type (key, value) (line 118)
str_9073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 85), 'str', 'b')
int_9074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 90), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 76), dict_9070, (str_9073, int_9074))

# Obtaining the member 'viewitems' of a type (line 118)
viewitems_9075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 76), dict_9070, 'viewitems')
# Calling viewitems(args, kwargs) (line 118)
viewitems_call_result_9077 = invoke(stypy.reporting.localization.Localization(__file__, 118, 76), viewitems_9075, *[], **kwargs_9076)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 41), tuple_9068, viewitems_call_result_9077)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (type_call_result_9067, tuple_9068))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 119)
# Processing the call arguments (line 119)

# Call to viewkeys(...): (line 119)
# Processing the call keyword arguments (line 119)
kwargs_9085 = {}

# Obtaining an instance of the builtin type 'dict' (line 119)
dict_9079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 9), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 119)
# Adding element type (key, value) (line 119)
str_9080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 10), 'str', 'a')
int_9081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 15), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 9), dict_9079, (str_9080, int_9081))
# Adding element type (key, value) (line 119)
str_9082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 18), 'str', 'b')
int_9083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 23), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 9), dict_9079, (str_9082, int_9083))

# Obtaining the member 'viewkeys' of a type (line 119)
viewkeys_9084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 9), dict_9079, 'viewkeys')
# Calling viewkeys(args, kwargs) (line 119)
viewkeys_call_result_9086 = invoke(stypy.reporting.localization.Localization(__file__, 119, 9), viewkeys_9084, *[], **kwargs_9085)

# Processing the call keyword arguments (line 119)
kwargs_9087 = {}
# Getting the type of 'type' (line 119)
type_9078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'type', False)
# Calling type(args, kwargs) (line 119)
type_call_result_9088 = invoke(stypy.reporting.localization.Localization(__file__, 119, 4), type_9078, *[viewkeys_call_result_9086], **kwargs_9087)


# Obtaining an instance of the builtin type 'tuple' (line 119)
tuple_9089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 40), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 119)
# Adding element type (line 119)
str_9090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 40), 'str', 'ExtraTypeDefinitions.dict_keys')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 40), tuple_9089, str_9090)
# Adding element type (line 119)

# Call to viewkeys(...): (line 119)
# Processing the call keyword arguments (line 119)
kwargs_9097 = {}

# Obtaining an instance of the builtin type 'dict' (line 119)
dict_9091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 74), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 119)
# Adding element type (key, value) (line 119)
str_9092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 75), 'str', 'a')
int_9093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 80), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 74), dict_9091, (str_9092, int_9093))
# Adding element type (key, value) (line 119)
str_9094 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 83), 'str', 'b')
int_9095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 88), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 74), dict_9091, (str_9094, int_9095))

# Obtaining the member 'viewkeys' of a type (line 119)
viewkeys_9096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 74), dict_9091, 'viewkeys')
# Calling viewkeys(args, kwargs) (line 119)
viewkeys_call_result_9098 = invoke(stypy.reporting.localization.Localization(__file__, 119, 74), viewkeys_9096, *[], **kwargs_9097)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 40), tuple_9089, viewkeys_call_result_9098)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (type_call_result_9088, tuple_9089))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 120)
# Processing the call arguments (line 120)

# Call to viewvalues(...): (line 120)
# Processing the call keyword arguments (line 120)
kwargs_9106 = {}

# Obtaining an instance of the builtin type 'dict' (line 120)
dict_9100 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 9), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 120)
# Adding element type (key, value) (line 120)
str_9101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 10), 'str', 'a')
int_9102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 15), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 9), dict_9100, (str_9101, int_9102))
# Adding element type (key, value) (line 120)
str_9103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 18), 'str', 'b')
int_9104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 23), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 9), dict_9100, (str_9103, int_9104))

# Obtaining the member 'viewvalues' of a type (line 120)
viewvalues_9105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 9), dict_9100, 'viewvalues')
# Calling viewvalues(args, kwargs) (line 120)
viewvalues_call_result_9107 = invoke(stypy.reporting.localization.Localization(__file__, 120, 9), viewvalues_9105, *[], **kwargs_9106)

# Processing the call keyword arguments (line 120)
kwargs_9108 = {}
# Getting the type of 'type' (line 120)
type_9099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'type', False)
# Calling type(args, kwargs) (line 120)
type_call_result_9109 = invoke(stypy.reporting.localization.Localization(__file__, 120, 4), type_9099, *[viewvalues_call_result_9107], **kwargs_9108)


# Obtaining an instance of the builtin type 'tuple' (line 120)
tuple_9110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 42), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 120)
# Adding element type (line 120)
str_9111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 42), 'str', 'ExtraTypeDefinitions.dict_values')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 42), tuple_9110, str_9111)
# Adding element type (line 120)

# Call to viewvalues(...): (line 120)
# Processing the call keyword arguments (line 120)
kwargs_9118 = {}

# Obtaining an instance of the builtin type 'dict' (line 120)
dict_9112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 78), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 120)
# Adding element type (key, value) (line 120)
str_9113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 79), 'str', 'a')
int_9114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 84), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 78), dict_9112, (str_9113, int_9114))
# Adding element type (key, value) (line 120)
str_9115 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 87), 'str', 'b')
int_9116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 92), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 78), dict_9112, (str_9115, int_9116))

# Obtaining the member 'viewvalues' of a type (line 120)
viewvalues_9117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 78), dict_9112, 'viewvalues')
# Calling viewvalues(args, kwargs) (line 120)
viewvalues_call_result_9119 = invoke(stypy.reporting.localization.Localization(__file__, 120, 78), viewvalues_9117, *[], **kwargs_9118)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 42), tuple_9110, viewvalues_call_result_9119)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (type_call_result_9109, tuple_9110))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 122)
# Processing the call arguments (line 122)

# Call to iter(...): (line 122)
# Processing the call arguments (line 122)

# Obtaining an instance of the builtin type 'dict' (line 122)
dict_9122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 14), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 122)
# Adding element type (key, value) (line 122)
str_9123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 15), 'str', 'a')
int_9124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 20), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 14), dict_9122, (str_9123, int_9124))
# Adding element type (key, value) (line 122)
str_9125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 23), 'str', 'b')
int_9126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 28), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 14), dict_9122, (str_9125, int_9126))

# Processing the call keyword arguments (line 122)
kwargs_9127 = {}
# Getting the type of 'iter' (line 122)
iter_9121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 9), 'iter', False)
# Calling iter(args, kwargs) (line 122)
iter_call_result_9128 = invoke(stypy.reporting.localization.Localization(__file__, 122, 9), iter_9121, *[dict_9122], **kwargs_9127)

# Processing the call keyword arguments (line 122)
kwargs_9129 = {}
# Getting the type of 'type' (line 122)
type_9120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'type', False)
# Calling type(args, kwargs) (line 122)
type_call_result_9130 = invoke(stypy.reporting.localization.Localization(__file__, 122, 4), type_9120, *[iter_call_result_9128], **kwargs_9129)


# Obtaining an instance of the builtin type 'tuple' (line 122)
tuple_9131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 35), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 122)
# Adding element type (line 122)
str_9132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 35), 'str', 'ExtraTypeDefinitions.dictionary_keyiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 35), tuple_9131, str_9132)
# Adding element type (line 122)

# Call to iter(...): (line 122)
# Processing the call arguments (line 122)

# Obtaining an instance of the builtin type 'dict' (line 122)
dict_9134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 87), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 122)
# Adding element type (key, value) (line 122)
str_9135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 88), 'str', 'a')
int_9136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 93), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 87), dict_9134, (str_9135, int_9136))
# Adding element type (key, value) (line 122)
str_9137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 96), 'str', 'b')
int_9138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 101), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 87), dict_9134, (str_9137, int_9138))

# Processing the call keyword arguments (line 122)
kwargs_9139 = {}
# Getting the type of 'iter' (line 122)
iter_9133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 82), 'iter', False)
# Calling iter(args, kwargs) (line 122)
iter_call_result_9140 = invoke(stypy.reporting.localization.Localization(__file__, 122, 82), iter_9133, *[dict_9134], **kwargs_9139)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 35), tuple_9131, iter_call_result_9140)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (type_call_result_9130, tuple_9131))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 123)
# Processing the call arguments (line 123)

# Call to iteritems(...): (line 123)
# Processing the call keyword arguments (line 123)
kwargs_9148 = {}

# Obtaining an instance of the builtin type 'dict' (line 123)
dict_9142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 9), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 123)
# Adding element type (key, value) (line 123)
str_9143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 10), 'str', 'a')
int_9144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 15), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 9), dict_9142, (str_9143, int_9144))
# Adding element type (key, value) (line 123)
str_9145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 18), 'str', 'b')
int_9146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 23), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 9), dict_9142, (str_9145, int_9146))

# Obtaining the member 'iteritems' of a type (line 123)
iteritems_9147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 9), dict_9142, 'iteritems')
# Calling iteritems(args, kwargs) (line 123)
iteritems_call_result_9149 = invoke(stypy.reporting.localization.Localization(__file__, 123, 9), iteritems_9147, *[], **kwargs_9148)

# Processing the call keyword arguments (line 123)
kwargs_9150 = {}
# Getting the type of 'type' (line 123)
type_9141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'type', False)
# Calling type(args, kwargs) (line 123)
type_call_result_9151 = invoke(stypy.reporting.localization.Localization(__file__, 123, 4), type_9141, *[iteritems_call_result_9149], **kwargs_9150)


# Obtaining an instance of the builtin type 'tuple' (line 123)
tuple_9152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 41), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 123)
# Adding element type (line 123)
str_9153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 41), 'str', 'ExtraTypeDefinitions.dictionary_itemiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 41), tuple_9152, str_9153)
# Adding element type (line 123)

# Call to iteritems(...): (line 123)
# Processing the call keyword arguments (line 123)
kwargs_9158 = {}

# Obtaining an instance of the builtin type 'dict' (line 123)
dict_9154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 89), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 123)
# Adding element type (key, value) (line 123)
str_9155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 90), 'str', 'a')
int_9156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 95), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 89), dict_9154, (str_9155, int_9156))

# Obtaining the member 'iteritems' of a type (line 123)
iteritems_9157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 89), dict_9154, 'iteritems')
# Calling iteritems(args, kwargs) (line 123)
iteritems_call_result_9159 = invoke(stypy.reporting.localization.Localization(__file__, 123, 89), iteritems_9157, *[], **kwargs_9158)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 41), tuple_9152, iteritems_call_result_9159)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (type_call_result_9151, tuple_9152))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 124)
# Processing the call arguments (line 124)

# Call to itervalues(...): (line 124)
# Processing the call keyword arguments (line 124)
kwargs_9167 = {}

# Obtaining an instance of the builtin type 'dict' (line 124)
dict_9161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 9), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 124)
# Adding element type (key, value) (line 124)
str_9162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 10), 'str', 'a')
int_9163 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 15), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 9), dict_9161, (str_9162, int_9163))
# Adding element type (key, value) (line 124)
str_9164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 18), 'str', 'b')
int_9165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 23), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 9), dict_9161, (str_9164, int_9165))

# Obtaining the member 'itervalues' of a type (line 124)
itervalues_9166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 9), dict_9161, 'itervalues')
# Calling itervalues(args, kwargs) (line 124)
itervalues_call_result_9168 = invoke(stypy.reporting.localization.Localization(__file__, 124, 9), itervalues_9166, *[], **kwargs_9167)

# Processing the call keyword arguments (line 124)
kwargs_9169 = {}
# Getting the type of 'type' (line 124)
type_9160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'type', False)
# Calling type(args, kwargs) (line 124)
type_call_result_9170 = invoke(stypy.reporting.localization.Localization(__file__, 124, 4), type_9160, *[itervalues_call_result_9168], **kwargs_9169)


# Obtaining an instance of the builtin type 'tuple' (line 124)
tuple_9171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 42), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 124)
# Adding element type (line 124)
str_9172 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 42), 'str', 'ExtraTypeDefinitions.dictionary_valueiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 42), tuple_9171, str_9172)
# Adding element type (line 124)

# Call to itervalues(...): (line 124)
# Processing the call keyword arguments (line 124)
kwargs_9177 = {}

# Obtaining an instance of the builtin type 'dict' (line 124)
dict_9173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 91), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 124)
# Adding element type (key, value) (line 124)
str_9174 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 92), 'str', 'a')
int_9175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 97), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 91), dict_9173, (str_9174, int_9175))

# Obtaining the member 'itervalues' of a type (line 124)
itervalues_9176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 91), dict_9173, 'itervalues')
# Calling itervalues(args, kwargs) (line 124)
itervalues_call_result_9178 = invoke(stypy.reporting.localization.Localization(__file__, 124, 91), itervalues_9176, *[], **kwargs_9177)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 42), tuple_9171, itervalues_call_result_9178)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (type_call_result_9170, tuple_9171))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 126)
# Processing the call arguments (line 126)

# Call to iter(...): (line 126)
# Processing the call arguments (line 126)

# Call to bytearray(...): (line 126)
# Processing the call arguments (line 126)
str_9182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 24), 'str', 'test')
# Processing the call keyword arguments (line 126)
kwargs_9183 = {}
# Getting the type of 'bytearray' (line 126)
bytearray_9181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 14), 'bytearray', False)
# Calling bytearray(args, kwargs) (line 126)
bytearray_call_result_9184 = invoke(stypy.reporting.localization.Localization(__file__, 126, 14), bytearray_9181, *[str_9182], **kwargs_9183)

# Processing the call keyword arguments (line 126)
kwargs_9185 = {}
# Getting the type of 'iter' (line 126)
iter_9180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 9), 'iter', False)
# Calling iter(args, kwargs) (line 126)
iter_call_result_9186 = invoke(stypy.reporting.localization.Localization(__file__, 126, 9), iter_9180, *[bytearray_call_result_9184], **kwargs_9185)

# Processing the call keyword arguments (line 126)
kwargs_9187 = {}
# Getting the type of 'type' (line 126)
type_9179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'type', False)
# Calling type(args, kwargs) (line 126)
type_call_result_9188 = invoke(stypy.reporting.localization.Localization(__file__, 126, 4), type_9179, *[iter_call_result_9186], **kwargs_9187)


# Obtaining an instance of the builtin type 'tuple' (line 126)
tuple_9189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 36), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 126)
# Adding element type (line 126)
str_9190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 36), 'str', 'ExtraTypeDefinitions.bytearray_iterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 36), tuple_9189, str_9190)
# Adding element type (line 126)

# Call to iter(...): (line 126)
# Processing the call arguments (line 126)

# Call to bytearray(...): (line 126)
# Processing the call arguments (line 126)
str_9193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 94), 'str', 'test')
# Processing the call keyword arguments (line 126)
kwargs_9194 = {}
# Getting the type of 'bytearray' (line 126)
bytearray_9192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 84), 'bytearray', False)
# Calling bytearray(args, kwargs) (line 126)
bytearray_call_result_9195 = invoke(stypy.reporting.localization.Localization(__file__, 126, 84), bytearray_9192, *[str_9193], **kwargs_9194)

# Processing the call keyword arguments (line 126)
kwargs_9196 = {}
# Getting the type of 'iter' (line 126)
iter_9191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 79), 'iter', False)
# Calling iter(args, kwargs) (line 126)
iter_call_result_9197 = invoke(stypy.reporting.localization.Localization(__file__, 126, 79), iter_9191, *[bytearray_call_result_9195], **kwargs_9196)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 36), tuple_9189, iter_call_result_9197)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (type_call_result_9188, tuple_9189))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 128)
# Processing the call arguments (line 128)
# Getting the type of 'ArithmeticError' (line 128)
ArithmeticError_9199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 9), 'ArithmeticError', False)
# Obtaining the member 'message' of a type (line 128)
message_9200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 9), ArithmeticError_9199, 'message')
# Processing the call keyword arguments (line 128)
kwargs_9201 = {}
# Getting the type of 'type' (line 128)
type_9198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'type', False)
# Calling type(args, kwargs) (line 128)
type_call_result_9202 = invoke(stypy.reporting.localization.Localization(__file__, 128, 4), type_9198, *[message_9200], **kwargs_9201)


# Obtaining an instance of the builtin type 'tuple' (line 128)
tuple_9203 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 36), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 128)
# Adding element type (line 128)
str_9204 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 36), 'str', 'ExtraTypeDefinitions.getset_descriptor')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 36), tuple_9203, str_9204)
# Adding element type (line 128)
# Getting the type of 'ArithmeticError' (line 128)
ArithmeticError_9205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 78), 'ArithmeticError')
# Obtaining the member 'message' of a type (line 128)
message_9206 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 78), ArithmeticError_9205, 'message')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 36), tuple_9203, message_9206)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (type_call_result_9202, tuple_9203))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 129)
# Processing the call arguments (line 129)
# Getting the type of 'IOError' (line 129)
IOError_9208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 9), 'IOError', False)
# Obtaining the member 'errno' of a type (line 129)
errno_9209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 9), IOError_9208, 'errno')
# Processing the call keyword arguments (line 129)
kwargs_9210 = {}
# Getting the type of 'type' (line 129)
type_9207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'type', False)
# Calling type(args, kwargs) (line 129)
type_call_result_9211 = invoke(stypy.reporting.localization.Localization(__file__, 129, 4), type_9207, *[errno_9209], **kwargs_9210)


# Obtaining an instance of the builtin type 'tuple' (line 129)
tuple_9212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 26), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 129)
# Adding element type (line 129)
str_9213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 26), 'str', 'ExtraTypeDefinitions.member_descriptor')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 26), tuple_9212, str_9213)
# Adding element type (line 129)
# Getting the type of 'IOError' (line 129)
IOError_9214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 68), 'IOError')
# Obtaining the member 'errno' of a type (line 129)
errno_9215 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 68), IOError_9214, 'errno')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 26), tuple_9212, errno_9215)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (type_call_result_9211, tuple_9212))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 130)
# Processing the call arguments (line 130)

# Call to _formatter_parser(...): (line 130)
# Processing the call keyword arguments (line 130)
kwargs_9219 = {}
unicode_9217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 9), 'unicode', u'foo')
# Obtaining the member '_formatter_parser' of a type (line 130)
_formatter_parser_9218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 9), unicode_9217, '_formatter_parser')
# Calling _formatter_parser(args, kwargs) (line 130)
_formatter_parser_call_result_9220 = invoke(stypy.reporting.localization.Localization(__file__, 130, 9), _formatter_parser_9218, *[], **kwargs_9219)

# Processing the call keyword arguments (line 130)
kwargs_9221 = {}
# Getting the type of 'type' (line 130)
type_9216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'type', False)
# Calling type(args, kwargs) (line 130)
type_call_result_9222 = invoke(stypy.reporting.localization.Localization(__file__, 130, 4), type_9216, *[_formatter_parser_call_result_9220], **kwargs_9221)


# Obtaining an instance of the builtin type 'tuple' (line 130)
tuple_9223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 39), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 130)
# Adding element type (line 130)
str_9224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 39), 'str', 'ExtraTypeDefinitions.formatteriterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 39), tuple_9223, str_9224)
# Adding element type (line 130)

# Call to type(...): (line 130)
# Processing the call arguments (line 130)

# Call to _formatter_parser(...): (line 130)
# Processing the call keyword arguments (line 130)
kwargs_9228 = {}
unicode_9226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 86), 'unicode', u'foo')
# Obtaining the member '_formatter_parser' of a type (line 130)
_formatter_parser_9227 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 86), unicode_9226, '_formatter_parser')
# Calling _formatter_parser(args, kwargs) (line 130)
_formatter_parser_call_result_9229 = invoke(stypy.reporting.localization.Localization(__file__, 130, 86), _formatter_parser_9227, *[], **kwargs_9228)

# Processing the call keyword arguments (line 130)
kwargs_9230 = {}
# Getting the type of 'type' (line 130)
type_9225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 81), 'type', False)
# Calling type(args, kwargs) (line 130)
type_call_result_9231 = invoke(stypy.reporting.localization.Localization(__file__, 130, 81), type_9225, *[_formatter_parser_call_result_9229], **kwargs_9230)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 39), tuple_9223, type_call_result_9231)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8619, (type_call_result_9222, tuple_9223))

# Assigning a type to the variable 'known_python_type_typename_samplevalues' (line 67)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'known_python_type_typename_samplevalues', dict_8619)
# Declaration of the 'ExtraTypeDefinitions' class

class ExtraTypeDefinitions:
    str_9232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, (-1)), 'str', '\n    Additional (not included) type definitions to those defined in the types Python module. This class is needed\n    to have an usable type object to refer to when generating Python code\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 134, 0, False)
        # Assigning a type to the variable 'self' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'ExtraTypeDefinitions.__init__', [], None, None, defaults, varargs, kwargs)

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

        pass
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'ExtraTypeDefinitions' (line 134)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 0), 'ExtraTypeDefinitions', ExtraTypeDefinitions)

# Assigning a Name to a Name (line 139):
# Getting the type of 'set' (line 139)
set_9233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 14), 'set')
# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'SetType' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9234, 'SetType', set_9233)

# Assigning a Call to a Name (line 140):

# Call to type(...): (line 140)
# Processing the call arguments (line 140)

# Call to iter(...): (line 140)
# Processing the call arguments (line 140)
str_9237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 25), 'str', '')
# Processing the call keyword arguments (line 140)
kwargs_9238 = {}
# Getting the type of 'iter' (line 140)
iter_9236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 20), 'iter', False)
# Calling iter(args, kwargs) (line 140)
iter_call_result_9239 = invoke(stypy.reporting.localization.Localization(__file__, 140, 20), iter_9236, *[str_9237], **kwargs_9238)

# Processing the call keyword arguments (line 140)
kwargs_9240 = {}
# Getting the type of 'type' (line 140)
type_9235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 15), 'type', False)
# Calling type(args, kwargs) (line 140)
type_call_result_9241 = invoke(stypy.reporting.localization.Localization(__file__, 140, 15), type_9235, *[iter_call_result_9239], **kwargs_9240)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'iterator' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9242, 'iterator', type_call_result_9241)

# Assigning a Call to a Name (line 142):

# Call to type(...): (line 142)
# Processing the call arguments (line 142)

# Call to iter(...): (line 142)
# Processing the call arguments (line 142)

# Call to set(...): (line 142)
# Processing the call keyword arguments (line 142)
kwargs_9246 = {}
# Getting the type of 'set' (line 142)
set_9245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 28), 'set', False)
# Calling set(args, kwargs) (line 142)
set_call_result_9247 = invoke(stypy.reporting.localization.Localization(__file__, 142, 28), set_9245, *[], **kwargs_9246)

# Processing the call keyword arguments (line 142)
kwargs_9248 = {}
# Getting the type of 'iter' (line 142)
iter_9244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 23), 'iter', False)
# Calling iter(args, kwargs) (line 142)
iter_call_result_9249 = invoke(stypy.reporting.localization.Localization(__file__, 142, 23), iter_9244, *[set_call_result_9247], **kwargs_9248)

# Processing the call keyword arguments (line 142)
kwargs_9250 = {}
# Getting the type of 'type' (line 142)
type_9243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 18), 'type', False)
# Calling type(args, kwargs) (line 142)
type_call_result_9251 = invoke(stypy.reporting.localization.Localization(__file__, 142, 18), type_9243, *[iter_call_result_9249], **kwargs_9250)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'setiterator' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9252, 'setiterator', type_call_result_9251)

# Assigning a Call to a Name (line 143):

# Call to type(...): (line 143)
# Processing the call arguments (line 143)

# Call to iter(...): (line 143)
# Processing the call arguments (line 143)

# Call to tuple(...): (line 143)
# Processing the call keyword arguments (line 143)
kwargs_9256 = {}
# Getting the type of 'tuple' (line 143)
tuple_9255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 30), 'tuple', False)
# Calling tuple(args, kwargs) (line 143)
tuple_call_result_9257 = invoke(stypy.reporting.localization.Localization(__file__, 143, 30), tuple_9255, *[], **kwargs_9256)

# Processing the call keyword arguments (line 143)
kwargs_9258 = {}
# Getting the type of 'iter' (line 143)
iter_9254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 25), 'iter', False)
# Calling iter(args, kwargs) (line 143)
iter_call_result_9259 = invoke(stypy.reporting.localization.Localization(__file__, 143, 25), iter_9254, *[tuple_call_result_9257], **kwargs_9258)

# Processing the call keyword arguments (line 143)
kwargs_9260 = {}
# Getting the type of 'type' (line 143)
type_9253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 20), 'type', False)
# Calling type(args, kwargs) (line 143)
type_call_result_9261 = invoke(stypy.reporting.localization.Localization(__file__, 143, 20), type_9253, *[iter_call_result_9259], **kwargs_9260)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'tupleiterator' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9262, 'tupleiterator', type_call_result_9261)

# Assigning a Call to a Name (line 144):

# Call to type(...): (line 144)
# Processing the call arguments (line 144)

# Call to iter(...): (line 144)
# Processing the call arguments (line 144)

# Call to xrange(...): (line 144)
# Processing the call arguments (line 144)
int_9266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 37), 'int')
# Processing the call keyword arguments (line 144)
kwargs_9267 = {}
# Getting the type of 'xrange' (line 144)
xrange_9265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 30), 'xrange', False)
# Calling xrange(args, kwargs) (line 144)
xrange_call_result_9268 = invoke(stypy.reporting.localization.Localization(__file__, 144, 30), xrange_9265, *[int_9266], **kwargs_9267)

# Processing the call keyword arguments (line 144)
kwargs_9269 = {}
# Getting the type of 'iter' (line 144)
iter_9264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 25), 'iter', False)
# Calling iter(args, kwargs) (line 144)
iter_call_result_9270 = invoke(stypy.reporting.localization.Localization(__file__, 144, 25), iter_9264, *[xrange_call_result_9268], **kwargs_9269)

# Processing the call keyword arguments (line 144)
kwargs_9271 = {}
# Getting the type of 'type' (line 144)
type_9263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 20), 'type', False)
# Calling type(args, kwargs) (line 144)
type_call_result_9272 = invoke(stypy.reporting.localization.Localization(__file__, 144, 20), type_9263, *[iter_call_result_9270], **kwargs_9271)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'rangeiterator' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9273, 'rangeiterator', type_call_result_9272)

# Assigning a Call to a Name (line 145):

# Call to type(...): (line 145)
# Processing the call arguments (line 145)

# Call to iter(...): (line 145)
# Processing the call arguments (line 145)

# Call to list(...): (line 145)
# Processing the call keyword arguments (line 145)
kwargs_9277 = {}
# Getting the type of 'list' (line 145)
list_9276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 29), 'list', False)
# Calling list(args, kwargs) (line 145)
list_call_result_9278 = invoke(stypy.reporting.localization.Localization(__file__, 145, 29), list_9276, *[], **kwargs_9277)

# Processing the call keyword arguments (line 145)
kwargs_9279 = {}
# Getting the type of 'iter' (line 145)
iter_9275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 24), 'iter', False)
# Calling iter(args, kwargs) (line 145)
iter_call_result_9280 = invoke(stypy.reporting.localization.Localization(__file__, 145, 24), iter_9275, *[list_call_result_9278], **kwargs_9279)

# Processing the call keyword arguments (line 145)
kwargs_9281 = {}
# Getting the type of 'type' (line 145)
type_9274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 19), 'type', False)
# Calling type(args, kwargs) (line 145)
type_call_result_9282 = invoke(stypy.reporting.localization.Localization(__file__, 145, 19), type_9274, *[iter_call_result_9280], **kwargs_9281)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'listiterator' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9283, 'listiterator', type_call_result_9282)

# Assigning a Call to a Name (line 146):

# Call to type(...): (line 146)
# Processing the call arguments (line 146)

# Call to iter(...): (line 146)
# Processing the call arguments (line 146)

# Call to type(...): (line 146)
# Processing the call arguments (line 146)
# Getting the type of 'int' (line 146)
int_9287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 39), 'int', False)
# Processing the call keyword arguments (line 146)
kwargs_9288 = {}
# Getting the type of 'type' (line 146)
type_9286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 34), 'type', False)
# Calling type(args, kwargs) (line 146)
type_call_result_9289 = invoke(stypy.reporting.localization.Localization(__file__, 146, 34), type_9286, *[int_9287], **kwargs_9288)

float_9290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 45), 'float')
# Processing the call keyword arguments (line 146)
kwargs_9291 = {}
# Getting the type of 'iter' (line 146)
iter_9285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 29), 'iter', False)
# Calling iter(args, kwargs) (line 146)
iter_call_result_9292 = invoke(stypy.reporting.localization.Localization(__file__, 146, 29), iter_9285, *[type_call_result_9289, float_9290], **kwargs_9291)

# Processing the call keyword arguments (line 146)
kwargs_9293 = {}
# Getting the type of 'type' (line 146)
type_9284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 24), 'type', False)
# Calling type(args, kwargs) (line 146)
type_call_result_9294 = invoke(stypy.reporting.localization.Localization(__file__, 146, 24), type_9284, *[iter_call_result_9292], **kwargs_9293)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'callable_iterator' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9295, 'callable_iterator', type_call_result_9294)

# Assigning a Call to a Name (line 147):

# Call to type(...): (line 147)
# Processing the call arguments (line 147)

# Call to iter(...): (line 147)
# Processing the call arguments (line 147)

# Call to reversed(...): (line 147)
# Processing the call arguments (line 147)

# Call to list(...): (line 147)
# Processing the call keyword arguments (line 147)
kwargs_9300 = {}
# Getting the type of 'list' (line 147)
list_9299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 45), 'list', False)
# Calling list(args, kwargs) (line 147)
list_call_result_9301 = invoke(stypy.reporting.localization.Localization(__file__, 147, 45), list_9299, *[], **kwargs_9300)

# Processing the call keyword arguments (line 147)
kwargs_9302 = {}
# Getting the type of 'reversed' (line 147)
reversed_9298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 36), 'reversed', False)
# Calling reversed(args, kwargs) (line 147)
reversed_call_result_9303 = invoke(stypy.reporting.localization.Localization(__file__, 147, 36), reversed_9298, *[list_call_result_9301], **kwargs_9302)

# Processing the call keyword arguments (line 147)
kwargs_9304 = {}
# Getting the type of 'iter' (line 147)
iter_9297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 31), 'iter', False)
# Calling iter(args, kwargs) (line 147)
iter_call_result_9305 = invoke(stypy.reporting.localization.Localization(__file__, 147, 31), iter_9297, *[reversed_call_result_9303], **kwargs_9304)

# Processing the call keyword arguments (line 147)
kwargs_9306 = {}
# Getting the type of 'type' (line 147)
type_9296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 26), 'type', False)
# Calling type(args, kwargs) (line 147)
type_call_result_9307 = invoke(stypy.reporting.localization.Localization(__file__, 147, 26), type_9296, *[iter_call_result_9305], **kwargs_9306)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'listreverseiterator' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9308, 'listreverseiterator', type_call_result_9307)

# Assigning a Call to a Name (line 148):

# Call to type(...): (line 148)
# Processing the call arguments (line 148)

# Call to methodcaller(...): (line 148)
# Processing the call arguments (line 148)
int_9312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 46), 'int')
# Processing the call keyword arguments (line 148)
kwargs_9313 = {}
# Getting the type of 'operator' (line 148)
operator_9310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 24), 'operator', False)
# Obtaining the member 'methodcaller' of a type (line 148)
methodcaller_9311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 24), operator_9310, 'methodcaller')
# Calling methodcaller(args, kwargs) (line 148)
methodcaller_call_result_9314 = invoke(stypy.reporting.localization.Localization(__file__, 148, 24), methodcaller_9311, *[int_9312], **kwargs_9313)

# Processing the call keyword arguments (line 148)
kwargs_9315 = {}
# Getting the type of 'type' (line 148)
type_9309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 19), 'type', False)
# Calling type(args, kwargs) (line 148)
type_call_result_9316 = invoke(stypy.reporting.localization.Localization(__file__, 148, 19), type_9309, *[methodcaller_call_result_9314], **kwargs_9315)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'methodcaller' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9317, 'methodcaller', type_call_result_9316)

# Assigning a Call to a Name (line 149):

# Call to type(...): (line 149)
# Processing the call arguments (line 149)

# Call to itemgetter(...): (line 149)
# Processing the call arguments (line 149)
int_9321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 42), 'int')
# Processing the call keyword arguments (line 149)
kwargs_9322 = {}
# Getting the type of 'operator' (line 149)
operator_9319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 22), 'operator', False)
# Obtaining the member 'itemgetter' of a type (line 149)
itemgetter_9320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 22), operator_9319, 'itemgetter')
# Calling itemgetter(args, kwargs) (line 149)
itemgetter_call_result_9323 = invoke(stypy.reporting.localization.Localization(__file__, 149, 22), itemgetter_9320, *[int_9321], **kwargs_9322)

# Processing the call keyword arguments (line 149)
kwargs_9324 = {}
# Getting the type of 'type' (line 149)
type_9318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 17), 'type', False)
# Calling type(args, kwargs) (line 149)
type_call_result_9325 = invoke(stypy.reporting.localization.Localization(__file__, 149, 17), type_9318, *[itemgetter_call_result_9323], **kwargs_9324)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'itemgetter' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9326, 'itemgetter', type_call_result_9325)

# Assigning a Call to a Name (line 150):

# Call to type(...): (line 150)
# Processing the call arguments (line 150)

# Call to attrgetter(...): (line 150)
# Processing the call arguments (line 150)
int_9330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 42), 'int')
# Processing the call keyword arguments (line 150)
kwargs_9331 = {}
# Getting the type of 'operator' (line 150)
operator_9328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 22), 'operator', False)
# Obtaining the member 'attrgetter' of a type (line 150)
attrgetter_9329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 22), operator_9328, 'attrgetter')
# Calling attrgetter(args, kwargs) (line 150)
attrgetter_call_result_9332 = invoke(stypy.reporting.localization.Localization(__file__, 150, 22), attrgetter_9329, *[int_9330], **kwargs_9331)

# Processing the call keyword arguments (line 150)
kwargs_9333 = {}
# Getting the type of 'type' (line 150)
type_9327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 17), 'type', False)
# Calling type(args, kwargs) (line 150)
type_call_result_9334 = invoke(stypy.reporting.localization.Localization(__file__, 150, 17), type_9327, *[attrgetter_call_result_9332], **kwargs_9333)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'attrgetter' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9335, 'attrgetter', type_call_result_9334)

# Assigning a Call to a Name (line 152):

# Call to type(...): (line 152)
# Processing the call arguments (line 152)

# Call to viewitems(...): (line 152)
# Processing the call keyword arguments (line 152)
kwargs_9346 = {}

# Call to dict(...): (line 152)
# Processing the call arguments (line 152)

# Obtaining an instance of the builtin type 'dict' (line 152)
dict_9338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 27), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 152)
# Adding element type (key, value) (line 152)
str_9339 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 28), 'str', 'a')
int_9340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 33), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 27), dict_9338, (str_9339, int_9340))
# Adding element type (key, value) (line 152)
str_9341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 36), 'str', 'b')
int_9342 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 41), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 27), dict_9338, (str_9341, int_9342))

# Processing the call keyword arguments (line 152)
kwargs_9343 = {}
# Getting the type of 'dict' (line 152)
dict_9337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 22), 'dict', False)
# Calling dict(args, kwargs) (line 152)
dict_call_result_9344 = invoke(stypy.reporting.localization.Localization(__file__, 152, 22), dict_9337, *[dict_9338], **kwargs_9343)

# Obtaining the member 'viewitems' of a type (line 152)
viewitems_9345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 22), dict_call_result_9344, 'viewitems')
# Calling viewitems(args, kwargs) (line 152)
viewitems_call_result_9347 = invoke(stypy.reporting.localization.Localization(__file__, 152, 22), viewitems_9345, *[], **kwargs_9346)

# Processing the call keyword arguments (line 152)
kwargs_9348 = {}
# Getting the type of 'type' (line 152)
type_9336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 17), 'type', False)
# Calling type(args, kwargs) (line 152)
type_call_result_9349 = invoke(stypy.reporting.localization.Localization(__file__, 152, 17), type_9336, *[viewitems_call_result_9347], **kwargs_9348)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'dict_items' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9350, 'dict_items', type_call_result_9349)

# Assigning a Call to a Name (line 153):

# Call to type(...): (line 153)
# Processing the call arguments (line 153)

# Call to viewkeys(...): (line 153)
# Processing the call keyword arguments (line 153)
kwargs_9361 = {}

# Call to dict(...): (line 153)
# Processing the call arguments (line 153)

# Obtaining an instance of the builtin type 'dict' (line 153)
dict_9353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 26), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 153)
# Adding element type (key, value) (line 153)
str_9354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 27), 'str', 'a')
int_9355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 32), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 26), dict_9353, (str_9354, int_9355))
# Adding element type (key, value) (line 153)
str_9356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 35), 'str', 'b')
int_9357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 40), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 26), dict_9353, (str_9356, int_9357))

# Processing the call keyword arguments (line 153)
kwargs_9358 = {}
# Getting the type of 'dict' (line 153)
dict_9352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 21), 'dict', False)
# Calling dict(args, kwargs) (line 153)
dict_call_result_9359 = invoke(stypy.reporting.localization.Localization(__file__, 153, 21), dict_9352, *[dict_9353], **kwargs_9358)

# Obtaining the member 'viewkeys' of a type (line 153)
viewkeys_9360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 21), dict_call_result_9359, 'viewkeys')
# Calling viewkeys(args, kwargs) (line 153)
viewkeys_call_result_9362 = invoke(stypy.reporting.localization.Localization(__file__, 153, 21), viewkeys_9360, *[], **kwargs_9361)

# Processing the call keyword arguments (line 153)
kwargs_9363 = {}
# Getting the type of 'type' (line 153)
type_9351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 'type', False)
# Calling type(args, kwargs) (line 153)
type_call_result_9364 = invoke(stypy.reporting.localization.Localization(__file__, 153, 16), type_9351, *[viewkeys_call_result_9362], **kwargs_9363)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'dict_keys' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9365, 'dict_keys', type_call_result_9364)

# Assigning a Call to a Name (line 154):

# Call to type(...): (line 154)
# Processing the call arguments (line 154)

# Call to viewvalues(...): (line 154)
# Processing the call keyword arguments (line 154)
kwargs_9376 = {}

# Call to dict(...): (line 154)
# Processing the call arguments (line 154)

# Obtaining an instance of the builtin type 'dict' (line 154)
dict_9368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 28), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 154)
# Adding element type (key, value) (line 154)
str_9369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 29), 'str', 'a')
int_9370 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 34), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 28), dict_9368, (str_9369, int_9370))
# Adding element type (key, value) (line 154)
str_9371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 37), 'str', 'b')
int_9372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 42), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 28), dict_9368, (str_9371, int_9372))

# Processing the call keyword arguments (line 154)
kwargs_9373 = {}
# Getting the type of 'dict' (line 154)
dict_9367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 23), 'dict', False)
# Calling dict(args, kwargs) (line 154)
dict_call_result_9374 = invoke(stypy.reporting.localization.Localization(__file__, 154, 23), dict_9367, *[dict_9368], **kwargs_9373)

# Obtaining the member 'viewvalues' of a type (line 154)
viewvalues_9375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 23), dict_call_result_9374, 'viewvalues')
# Calling viewvalues(args, kwargs) (line 154)
viewvalues_call_result_9377 = invoke(stypy.reporting.localization.Localization(__file__, 154, 23), viewvalues_9375, *[], **kwargs_9376)

# Processing the call keyword arguments (line 154)
kwargs_9378 = {}
# Getting the type of 'type' (line 154)
type_9366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 18), 'type', False)
# Calling type(args, kwargs) (line 154)
type_call_result_9379 = invoke(stypy.reporting.localization.Localization(__file__, 154, 18), type_9366, *[viewvalues_call_result_9377], **kwargs_9378)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'dict_values' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9380, 'dict_values', type_call_result_9379)

# Assigning a Call to a Name (line 156):

# Call to type(...): (line 156)
# Processing the call arguments (line 156)

# Call to iter(...): (line 156)
# Processing the call arguments (line 156)

# Call to dict(...): (line 156)
# Processing the call arguments (line 156)

# Obtaining an instance of the builtin type 'dict' (line 156)
dict_9384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 44), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 156)
# Adding element type (key, value) (line 156)
str_9385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 45), 'str', 'a')
int_9386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 50), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 44), dict_9384, (str_9385, int_9386))
# Adding element type (key, value) (line 156)
str_9387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 53), 'str', 'b')
int_9388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 58), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 44), dict_9384, (str_9387, int_9388))

# Processing the call keyword arguments (line 156)
kwargs_9389 = {}
# Getting the type of 'dict' (line 156)
dict_9383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 39), 'dict', False)
# Calling dict(args, kwargs) (line 156)
dict_call_result_9390 = invoke(stypy.reporting.localization.Localization(__file__, 156, 39), dict_9383, *[dict_9384], **kwargs_9389)

# Processing the call keyword arguments (line 156)
kwargs_9391 = {}
# Getting the type of 'iter' (line 156)
iter_9382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 34), 'iter', False)
# Calling iter(args, kwargs) (line 156)
iter_call_result_9392 = invoke(stypy.reporting.localization.Localization(__file__, 156, 34), iter_9382, *[dict_call_result_9390], **kwargs_9391)

# Processing the call keyword arguments (line 156)
kwargs_9393 = {}
# Getting the type of 'type' (line 156)
type_9381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 29), 'type', False)
# Calling type(args, kwargs) (line 156)
type_call_result_9394 = invoke(stypy.reporting.localization.Localization(__file__, 156, 29), type_9381, *[iter_call_result_9392], **kwargs_9393)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'dictionary_keyiterator' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9395, 'dictionary_keyiterator', type_call_result_9394)

# Assigning a Call to a Name (line 157):

# Call to type(...): (line 157)
# Processing the call arguments (line 157)

# Call to iteritems(...): (line 157)
# Processing the call keyword arguments (line 157)
kwargs_9406 = {}

# Call to dict(...): (line 157)
# Processing the call arguments (line 157)

# Obtaining an instance of the builtin type 'dict' (line 157)
dict_9398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 40), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 157)
# Adding element type (key, value) (line 157)
str_9399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 41), 'str', 'a')
int_9400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 46), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 40), dict_9398, (str_9399, int_9400))
# Adding element type (key, value) (line 157)
str_9401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 49), 'str', 'b')
int_9402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 54), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 40), dict_9398, (str_9401, int_9402))

# Processing the call keyword arguments (line 157)
kwargs_9403 = {}
# Getting the type of 'dict' (line 157)
dict_9397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 35), 'dict', False)
# Calling dict(args, kwargs) (line 157)
dict_call_result_9404 = invoke(stypy.reporting.localization.Localization(__file__, 157, 35), dict_9397, *[dict_9398], **kwargs_9403)

# Obtaining the member 'iteritems' of a type (line 157)
iteritems_9405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 35), dict_call_result_9404, 'iteritems')
# Calling iteritems(args, kwargs) (line 157)
iteritems_call_result_9407 = invoke(stypy.reporting.localization.Localization(__file__, 157, 35), iteritems_9405, *[], **kwargs_9406)

# Processing the call keyword arguments (line 157)
kwargs_9408 = {}
# Getting the type of 'type' (line 157)
type_9396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 30), 'type', False)
# Calling type(args, kwargs) (line 157)
type_call_result_9409 = invoke(stypy.reporting.localization.Localization(__file__, 157, 30), type_9396, *[iteritems_call_result_9407], **kwargs_9408)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'dictionary_itemiterator' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9410, 'dictionary_itemiterator', type_call_result_9409)

# Assigning a Call to a Name (line 158):

# Call to type(...): (line 158)
# Processing the call arguments (line 158)

# Call to itervalues(...): (line 158)
# Processing the call keyword arguments (line 158)
kwargs_9421 = {}

# Call to dict(...): (line 158)
# Processing the call arguments (line 158)

# Obtaining an instance of the builtin type 'dict' (line 158)
dict_9413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 41), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 158)
# Adding element type (key, value) (line 158)
str_9414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 42), 'str', 'a')
int_9415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 47), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 41), dict_9413, (str_9414, int_9415))
# Adding element type (key, value) (line 158)
str_9416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 50), 'str', 'b')
int_9417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 55), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 41), dict_9413, (str_9416, int_9417))

# Processing the call keyword arguments (line 158)
kwargs_9418 = {}
# Getting the type of 'dict' (line 158)
dict_9412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 36), 'dict', False)
# Calling dict(args, kwargs) (line 158)
dict_call_result_9419 = invoke(stypy.reporting.localization.Localization(__file__, 158, 36), dict_9412, *[dict_9413], **kwargs_9418)

# Obtaining the member 'itervalues' of a type (line 158)
itervalues_9420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 36), dict_call_result_9419, 'itervalues')
# Calling itervalues(args, kwargs) (line 158)
itervalues_call_result_9422 = invoke(stypy.reporting.localization.Localization(__file__, 158, 36), itervalues_9420, *[], **kwargs_9421)

# Processing the call keyword arguments (line 158)
kwargs_9423 = {}
# Getting the type of 'type' (line 158)
type_9411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 31), 'type', False)
# Calling type(args, kwargs) (line 158)
type_call_result_9424 = invoke(stypy.reporting.localization.Localization(__file__, 158, 31), type_9411, *[itervalues_call_result_9422], **kwargs_9423)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'dictionary_valueiterator' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9425, 'dictionary_valueiterator', type_call_result_9424)

# Assigning a Call to a Name (line 159):

# Call to type(...): (line 159)
# Processing the call arguments (line 159)

# Call to iter(...): (line 159)
# Processing the call arguments (line 159)

# Call to bytearray(...): (line 159)
# Processing the call arguments (line 159)
str_9429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 45), 'str', 'test')
# Processing the call keyword arguments (line 159)
kwargs_9430 = {}
# Getting the type of 'bytearray' (line 159)
bytearray_9428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 35), 'bytearray', False)
# Calling bytearray(args, kwargs) (line 159)
bytearray_call_result_9431 = invoke(stypy.reporting.localization.Localization(__file__, 159, 35), bytearray_9428, *[str_9429], **kwargs_9430)

# Processing the call keyword arguments (line 159)
kwargs_9432 = {}
# Getting the type of 'iter' (line 159)
iter_9427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 30), 'iter', False)
# Calling iter(args, kwargs) (line 159)
iter_call_result_9433 = invoke(stypy.reporting.localization.Localization(__file__, 159, 30), iter_9427, *[bytearray_call_result_9431], **kwargs_9432)

# Processing the call keyword arguments (line 159)
kwargs_9434 = {}
# Getting the type of 'type' (line 159)
type_9426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 25), 'type', False)
# Calling type(args, kwargs) (line 159)
type_call_result_9435 = invoke(stypy.reporting.localization.Localization(__file__, 159, 25), type_9426, *[iter_call_result_9433], **kwargs_9434)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'bytearray_iterator' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9436, 'bytearray_iterator', type_call_result_9435)

# Assigning a Call to a Name (line 162):

# Call to type(...): (line 162)
# Processing the call arguments (line 162)
# Getting the type of 'ArithmeticError' (line 162)
ArithmeticError_9438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 29), 'ArithmeticError', False)
# Obtaining the member 'message' of a type (line 162)
message_9439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 29), ArithmeticError_9438, 'message')
# Processing the call keyword arguments (line 162)
kwargs_9440 = {}
# Getting the type of 'type' (line 162)
type_9437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 24), 'type', False)
# Calling type(args, kwargs) (line 162)
type_call_result_9441 = invoke(stypy.reporting.localization.Localization(__file__, 162, 24), type_9437, *[message_9439], **kwargs_9440)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'getset_descriptor' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9442, 'getset_descriptor', type_call_result_9441)

# Assigning a Call to a Name (line 163):

# Call to type(...): (line 163)
# Processing the call arguments (line 163)
# Getting the type of 'IOError' (line 163)
IOError_9444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 29), 'IOError', False)
# Obtaining the member 'errno' of a type (line 163)
errno_9445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 29), IOError_9444, 'errno')
# Processing the call keyword arguments (line 163)
kwargs_9446 = {}
# Getting the type of 'type' (line 163)
type_9443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 24), 'type', False)
# Calling type(args, kwargs) (line 163)
type_call_result_9447 = invoke(stypy.reporting.localization.Localization(__file__, 163, 24), type_9443, *[errno_9445], **kwargs_9446)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'member_descriptor' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9448, 'member_descriptor', type_call_result_9447)

# Assigning a Call to a Name (line 164):

# Call to type(...): (line 164)
# Processing the call arguments (line 164)

# Call to _formatter_parser(...): (line 164)
# Processing the call keyword arguments (line 164)
kwargs_9452 = {}
unicode_9450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 29), 'unicode', u'foo')
# Obtaining the member '_formatter_parser' of a type (line 164)
_formatter_parser_9451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 29), unicode_9450, '_formatter_parser')
# Calling _formatter_parser(args, kwargs) (line 164)
_formatter_parser_call_result_9453 = invoke(stypy.reporting.localization.Localization(__file__, 164, 29), _formatter_parser_9451, *[], **kwargs_9452)

# Processing the call keyword arguments (line 164)
kwargs_9454 = {}
# Getting the type of 'type' (line 164)
type_9449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 24), 'type', False)
# Calling type(args, kwargs) (line 164)
type_call_result_9455 = invoke(stypy.reporting.localization.Localization(__file__, 164, 24), type_9449, *[_formatter_parser_call_result_9453], **kwargs_9454)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'formatteriterator' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9456, 'formatteriterator', type_call_result_9455)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
