
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

str_8874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, (-1)), 'str', '\nFile to store known Python language types (not defined in library modules) that are not listed in the types module.\nSurprisingly, the types module omits several types that Python uses when dealing with commonly used structures, such\nas iterators and member descriptors. As we need these types to specify type rules, type instances and, in general,\nworking with them, we created this file to list these types and its instances.\n')

# Assigning a List to a Name (line 14):

# Obtaining an instance of the builtin type 'list' (line 14)
list_8875 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 14)
# Adding element type (line 14)
# Getting the type of 'None' (line 15)
None_8876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 22), list_8875, None_8876)
# Adding element type (line 14)
# Getting the type of 'types' (line 16)
types_8877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'types')
# Obtaining the member 'NoneType' of a type (line 16)
NoneType_8878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 4), types_8877, 'NoneType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 22), list_8875, NoneType_8878)
# Adding element type (line 14)
# Getting the type of 'type' (line 17)
type_8879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'type')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 22), list_8875, type_8879)
# Adding element type (line 14)
# Getting the type of 'bool' (line 18)
bool_8880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'bool')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 22), list_8875, bool_8880)
# Adding element type (line 14)
# Getting the type of 'int' (line 19)
int_8881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 22), list_8875, int_8881)
# Adding element type (line 14)
# Getting the type of 'long' (line 20)
long_8882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 22), list_8875, long_8882)
# Adding element type (line 14)
# Getting the type of 'float' (line 21)
float_8883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 22), list_8875, float_8883)
# Adding element type (line 14)
# Getting the type of 'complex' (line 22)
complex_8884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'complex')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 22), list_8875, complex_8884)
# Adding element type (line 14)
# Getting the type of 'str' (line 23)
str_8885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'str')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 22), list_8875, str_8885)
# Adding element type (line 14)
# Getting the type of 'unicode' (line 24)
unicode_8886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'unicode')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 22), list_8875, unicode_8886)

# Assigning a type to the variable 'simple_python_types' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'simple_python_types', list_8875)
str_8887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, (-1)), 'str', '\nInternal python elements to generate various sample type values\n')
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
        stypy_return_type_8888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8888)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'bar'
        return stypy_return_type_8888


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
int_8889 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 10), 'int')
# Getting the type of 'Foo'
Foo_8890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Foo')
# Setting the type of the member 'qux' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Foo_8890, 'qux', int_8889)

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
    stypy_return_type_8891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8891)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'func'
    return stypy_return_type_8891

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
    int_8892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 10), 'int')
    # Assigning a type to the variable 'num' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'num', int_8892)
    
    
    # Getting the type of 'num' (line 45)
    num_8893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 10), 'num')
    # Getting the type of 'n' (line 45)
    n_8894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 16), 'n')
    # Applying the binary operator '<' (line 45)
    result_lt_8895 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 10), '<', num_8893, n_8894)
    
    # Assigning a type to the variable 'result_lt_8895' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'result_lt_8895', result_lt_8895)
    # Testing if the while is going to be iterated (line 45)
    # Testing the type of an if condition (line 45)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 45, 4), result_lt_8895)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 45, 4), result_lt_8895):
        # SSA begins for while statement (line 45)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        # Creating a generator
        # Getting the type of 'num' (line 46)
        num_8896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 14), 'num')
        GeneratorType_8897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 8), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 8), GeneratorType_8897, num_8896)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'stypy_return_type', GeneratorType_8897)
        
        # Getting the type of 'num' (line 47)
        num_8898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'num')
        int_8899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 15), 'int')
        # Applying the binary operator '+=' (line 47)
        result_iadd_8900 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 8), '+=', num_8898, int_8899)
        # Assigning a type to the variable 'num' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'num', result_iadd_8900)
        
        # SSA join for while statement (line 45)
        module_type_store = module_type_store.join_ssa_context()

    
    
    # ################# End of 'firstn(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'firstn' in the type store
    # Getting the type of 'stypy_return_type' (line 43)
    stypy_return_type_8901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8901)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'firstn'
    return stypy_return_type_8901

# Assigning a type to the variable 'firstn' (line 43)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 0), 'firstn', firstn)

# Assigning a Call to a Name (line 50):

# Call to Foo(...): (line 50)
# Processing the call keyword arguments (line 50)
kwargs_8903 = {}
# Getting the type of 'Foo' (line 50)
Foo_8902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 6), 'Foo', False)
# Calling Foo(args, kwargs) (line 50)
Foo_call_result_8904 = invoke(stypy.reporting.localization.Localization(__file__, 50, 6), Foo_8902, *[], **kwargs_8903)

# Assigning a type to the variable 'foo' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'foo', Foo_call_result_8904)

# Assigning a Dict to a Name (line 67):

# Obtaining an instance of the builtin type 'dict' (line 67)
dict_8905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 42), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 67)
# Adding element type (key, value) (line 67)
# Getting the type of 'types' (line 68)
types_8906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'types')
# Obtaining the member 'NoneType' of a type (line 68)
NoneType_8907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 4), types_8906, 'NoneType')

# Obtaining an instance of the builtin type 'tuple' (line 68)
tuple_8908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 68)
# Adding element type (line 68)
str_8909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 21), 'str', 'types.NoneType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 21), tuple_8908, str_8909)
# Adding element type (line 68)
# Getting the type of 'None' (line 68)
None_8910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 39), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 21), tuple_8908, None_8910)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (NoneType_8907, tuple_8908))
# Adding element type (key, value) (line 67)
# Getting the type of 'type' (line 69)
type_8911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'type')

# Obtaining an instance of the builtin type 'tuple' (line 69)
tuple_8912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 11), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 69)
# Adding element type (line 69)
str_8913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 11), 'str', 'type')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 11), tuple_8912, str_8913)
# Adding element type (line 69)

# Call to type(...): (line 69)
# Processing the call arguments (line 69)
# Getting the type of 'int' (line 69)
int_8915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 24), 'int', False)
# Processing the call keyword arguments (line 69)
kwargs_8916 = {}
# Getting the type of 'type' (line 69)
type_8914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 19), 'type', False)
# Calling type(args, kwargs) (line 69)
type_call_result_8917 = invoke(stypy.reporting.localization.Localization(__file__, 69, 19), type_8914, *[int_8915], **kwargs_8916)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 11), tuple_8912, type_call_result_8917)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (type_8911, tuple_8912))
# Adding element type (key, value) (line 67)
# Getting the type of 'bool' (line 70)
bool_8918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'bool')

# Obtaining an instance of the builtin type 'tuple' (line 70)
tuple_8919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 11), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 70)
# Adding element type (line 70)
str_8920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 11), 'str', 'bool')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 11), tuple_8919, str_8920)
# Adding element type (line 70)
# Getting the type of 'True' (line 70)
True_8921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 19), 'True')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 11), tuple_8919, True_8921)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (bool_8918, tuple_8919))
# Adding element type (key, value) (line 67)
# Getting the type of 'int' (line 71)
int_8922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'int')

# Obtaining an instance of the builtin type 'tuple' (line 71)
tuple_8923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 10), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 71)
# Adding element type (line 71)
str_8924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 10), 'str', 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 10), tuple_8923, str_8924)
# Adding element type (line 71)
int_8925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 17), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 71, 10), tuple_8923, int_8925)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (int_8922, tuple_8923))
# Adding element type (key, value) (line 67)
# Getting the type of 'long' (line 72)
long_8926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'long')

# Obtaining an instance of the builtin type 'tuple' (line 72)
tuple_8927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 11), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 72)
# Adding element type (line 72)
str_8928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 11), 'str', 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 11), tuple_8927, str_8928)
# Adding element type (line 72)
long_8929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 19), 'long')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 11), tuple_8927, long_8929)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (long_8926, tuple_8927))
# Adding element type (key, value) (line 67)
# Getting the type of 'float' (line 73)
float_8930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'float')

# Obtaining an instance of the builtin type 'tuple' (line 73)
tuple_8931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 12), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 73)
# Adding element type (line 73)
str_8932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 12), 'str', 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 12), tuple_8931, str_8932)
# Adding element type (line 73)
float_8933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 21), 'float')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 12), tuple_8931, float_8933)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (float_8930, tuple_8931))
# Adding element type (key, value) (line 67)
# Getting the type of 'complex' (line 74)
complex_8934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'complex')

# Obtaining an instance of the builtin type 'tuple' (line 74)
tuple_8935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 14), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 74)
# Adding element type (line 74)
str_8936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 14), 'str', 'complex')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 14), tuple_8935, str_8936)
# Adding element type (line 74)
complex_8937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 25), 'complex')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 14), tuple_8935, complex_8937)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (complex_8934, tuple_8935))
# Adding element type (key, value) (line 67)
# Getting the type of 'str' (line 75)
str_8938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'str')

# Obtaining an instance of the builtin type 'tuple' (line 75)
tuple_8939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 10), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 75)
# Adding element type (line 75)
str_8940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 10), 'str', 'str')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 10), tuple_8939, str_8940)
# Adding element type (line 75)
str_8941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 17), 'str', 'foo')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 10), tuple_8939, str_8941)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (str_8938, tuple_8939))
# Adding element type (key, value) (line 67)
# Getting the type of 'unicode' (line 76)
unicode_8942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'unicode')

# Obtaining an instance of the builtin type 'tuple' (line 76)
tuple_8943 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 14), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 76)
# Adding element type (line 76)
str_8944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 14), 'str', 'unicode')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 14), tuple_8943, str_8944)
# Adding element type (line 76)
unicode_8945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 25), 'unicode', u'foo')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 14), tuple_8943, unicode_8945)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (unicode_8942, tuple_8943))
# Adding element type (key, value) (line 67)
# Getting the type of 'tuple' (line 77)
tuple_8946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'tuple')

# Obtaining an instance of the builtin type 'tuple' (line 77)
tuple_8947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 12), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 77)
# Adding element type (line 77)
str_8948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 12), 'str', 'tuple')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 12), tuple_8947, str_8948)
# Adding element type (line 77)

# Obtaining an instance of the builtin type 'tuple' (line 77)
tuple_8949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 22), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 77)
# Adding element type (line 77)
int_8950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 22), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 22), tuple_8949, int_8950)
# Adding element type (line 77)
int_8951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 25), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 22), tuple_8949, int_8951)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 12), tuple_8947, tuple_8949)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (tuple_8946, tuple_8947))
# Adding element type (key, value) (line 67)
# Getting the type of 'list' (line 78)
list_8952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'list')

# Obtaining an instance of the builtin type 'tuple' (line 78)
tuple_8953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 11), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 78)
# Adding element type (line 78)
str_8954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 11), 'str', 'list')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 11), tuple_8953, str_8954)
# Adding element type (line 78)

# Obtaining an instance of the builtin type 'list' (line 78)
list_8955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 78)
# Adding element type (line 78)
int_8956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 20), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 19), list_8955, int_8956)
# Adding element type (line 78)
int_8957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 23), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 19), list_8955, int_8957)
# Adding element type (line 78)
int_8958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 26), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 19), list_8955, int_8958)
# Adding element type (line 78)
int_8959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 29), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 19), list_8955, int_8959)
# Adding element type (line 78)
int_8960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 32), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 19), list_8955, int_8960)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 11), tuple_8953, list_8955)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (list_8952, tuple_8953))
# Adding element type (key, value) (line 67)
# Getting the type of 'dict' (line 79)
dict_8961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'dict')

# Obtaining an instance of the builtin type 'tuple' (line 79)
tuple_8962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 11), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 79)
# Adding element type (line 79)
str_8963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 11), 'str', 'dict')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 11), tuple_8962, str_8963)
# Adding element type (line 79)

# Obtaining an instance of the builtin type 'dict' (line 79)
dict_8964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 19), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 79)
# Adding element type (key, value) (line 79)
str_8965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 20), 'str', 'a')
int_8966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 25), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 19), dict_8964, (str_8965, int_8966))
# Adding element type (key, value) (line 79)
str_8967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 28), 'str', 'b')
int_8968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 33), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 19), dict_8964, (str_8967, int_8968))

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 11), tuple_8962, dict_8964)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (dict_8961, tuple_8962))
# Adding element type (key, value) (line 67)
# Getting the type of 'set' (line 80)
set_8969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'set')

# Obtaining an instance of the builtin type 'tuple' (line 80)
tuple_8970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 10), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 80)
# Adding element type (line 80)
str_8971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 10), 'str', 'set')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 10), tuple_8970, str_8971)
# Adding element type (line 80)

# Obtaining an instance of the builtin type 'set' (line 80)
set_8972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 17), 'set')
# Adding type elements to the builtin type 'set' instance (line 80)
# Adding element type (line 80)
int_8973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 17), set_8972, int_8973)
# Adding element type (line 80)
int_8974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 17), set_8972, int_8974)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 10), tuple_8970, set_8972)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (set_8969, tuple_8970))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 81)
# Processing the call arguments (line 81)
# Getting the type of 'func' (line 81)
func_8976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 9), 'func', False)
# Processing the call keyword arguments (line 81)
kwargs_8977 = {}
# Getting the type of 'type' (line 81)
type_8975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'type', False)
# Calling type(args, kwargs) (line 81)
type_call_result_8978 = invoke(stypy.reporting.localization.Localization(__file__, 81, 4), type_8975, *[func_8976], **kwargs_8977)


# Obtaining an instance of the builtin type 'tuple' (line 81)
tuple_8979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 81)
# Adding element type (line 81)
str_8980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 17), 'str', 'types.FunctionType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 17), tuple_8979, str_8980)
# Adding element type (line 81)
# Getting the type of 'func' (line 81)
func_8981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 39), 'func')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 17), tuple_8979, func_8981)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (type_call_result_8978, tuple_8979))
# Adding element type (key, value) (line 67)
# Getting the type of 'types' (line 82)
types_8982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'types')
# Obtaining the member 'LambdaType' of a type (line 82)
LambdaType_8983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 4), types_8982, 'LambdaType')

# Obtaining an instance of the builtin type 'tuple' (line 82)
tuple_8984 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 23), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 82)
# Adding element type (line 82)
str_8985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 23), 'str', 'types.LambdaType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 23), tuple_8984, str_8985)
# Adding element type (line 82)

@norecursion
def _stypy_temp_lambda_18(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_18'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_18', 82, 43, True)
    # Passed parameters checking function
    _stypy_temp_lambda_18.stypy_localization = localization
    _stypy_temp_lambda_18.stypy_type_of_self = None
    _stypy_temp_lambda_18.stypy_type_store = module_type_store
    _stypy_temp_lambda_18.stypy_function_name = '_stypy_temp_lambda_18'
    _stypy_temp_lambda_18.stypy_param_names_list = ['x']
    _stypy_temp_lambda_18.stypy_varargs_param_name = None
    _stypy_temp_lambda_18.stypy_kwargs_param_name = None
    _stypy_temp_lambda_18.stypy_call_defaults = defaults
    _stypy_temp_lambda_18.stypy_call_varargs = varargs
    _stypy_temp_lambda_18.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_18', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_18', ['x'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    # Getting the type of 'x' (line 82)
    x_8986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 53), 'x')
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 43), 'stypy_return_type', x_8986)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_18' in the type store
    # Getting the type of 'stypy_return_type' (line 82)
    stypy_return_type_8987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 43), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_8987)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_18'
    return stypy_return_type_8987

# Assigning a type to the variable '_stypy_temp_lambda_18' (line 82)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 43), '_stypy_temp_lambda_18', _stypy_temp_lambda_18)
# Getting the type of '_stypy_temp_lambda_18' (line 82)
_stypy_temp_lambda_18_8988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 43), '_stypy_temp_lambda_18')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 82, 23), tuple_8984, _stypy_temp_lambda_18_8988)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (LambdaType_8983, tuple_8984))
# Adding element type (key, value) (line 67)
# Getting the type of 'types' (line 83)
types_8989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'types')
# Obtaining the member 'GeneratorType' of a type (line 83)
GeneratorType_8990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 4), types_8989, 'GeneratorType')

# Obtaining an instance of the builtin type 'tuple' (line 83)
tuple_8991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 26), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 83)
# Adding element type (line 83)
str_8992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 26), 'str', 'types.GeneratorType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 26), tuple_8991, str_8992)
# Adding element type (line 83)

# Call to firstn(...): (line 83)
# Processing the call arguments (line 83)
int_8994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 56), 'int')
# Processing the call keyword arguments (line 83)
kwargs_8995 = {}
# Getting the type of 'firstn' (line 83)
firstn_8993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 49), 'firstn', False)
# Calling firstn(args, kwargs) (line 83)
firstn_call_result_8996 = invoke(stypy.reporting.localization.Localization(__file__, 83, 49), firstn_8993, *[int_8994], **kwargs_8995)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 83, 26), tuple_8991, firstn_call_result_8996)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (GeneratorType_8990, tuple_8991))
# Adding element type (key, value) (line 67)
# Getting the type of 'types' (line 84)
types_8997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'types')
# Obtaining the member 'CodeType' of a type (line 84)
CodeType_8998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 4), types_8997, 'CodeType')

# Obtaining an instance of the builtin type 'tuple' (line 84)
tuple_8999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 84)
# Adding element type (line 84)
str_9000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 84, 21), 'str', 'types.CodeType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 21), tuple_8999, str_9000)
# Adding element type (line 84)
# Getting the type of 'func' (line 84)
func_9001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 39), 'func')
# Obtaining the member 'func_code' of a type (line 84)
func_code_9002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 39), func_9001, 'func_code')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 84, 21), tuple_8999, func_code_9002)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (CodeType_8998, tuple_8999))
# Adding element type (key, value) (line 67)
# Getting the type of 'types' (line 85)
types_9003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'types')
# Obtaining the member 'BuiltinFunctionType' of a type (line 85)
BuiltinFunctionType_9004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 4), types_9003, 'BuiltinFunctionType')

# Obtaining an instance of the builtin type 'tuple' (line 85)
tuple_9005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 32), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 85)
# Adding element type (line 85)
str_9006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 32), 'str', 'types.BuiltinFunctionType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 32), tuple_9005, str_9006)
# Adding element type (line 85)
# Getting the type of 'len' (line 85)
len_9007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 61), 'len')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 85, 32), tuple_9005, len_9007)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (BuiltinFunctionType_9004, tuple_9005))
# Adding element type (key, value) (line 67)
# Getting the type of 'types' (line 86)
types_9008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'types')
# Obtaining the member 'ModuleType' of a type (line 86)
ModuleType_9009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 4), types_9008, 'ModuleType')

# Obtaining an instance of the builtin type 'tuple' (line 86)
tuple_9010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 23), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 86)
# Adding element type (line 86)
str_9011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 23), 'str', 'types.ModuleType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 23), tuple_9010, str_9011)
# Adding element type (line 86)
# Getting the type of 'inspect' (line 86)
inspect_9012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 43), 'inspect')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 86, 23), tuple_9010, inspect_9012)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (ModuleType_9009, tuple_9010))
# Adding element type (key, value) (line 67)
# Getting the type of 'file' (line 87)
file_9013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'file')

# Obtaining an instance of the builtin type 'tuple' (line 87)
tuple_9014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 11), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 87)
# Adding element type (line 87)
str_9015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 11), 'str', 'file')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 11), tuple_9014, str_9015)
# Adding element type (line 87)

# Call to file(...): (line 87)
# Processing the call arguments (line 87)

# Call to dirname(...): (line 87)
# Processing the call arguments (line 87)

# Call to realpath(...): (line 87)
# Processing the call arguments (line 87)
# Getting the type of '__file__' (line 87)
file___9023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 57), '__file__', False)
# Processing the call keyword arguments (line 87)
kwargs_9024 = {}
# Getting the type of 'os' (line 87)
os_9020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 40), 'os', False)
# Obtaining the member 'path' of a type (line 87)
path_9021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 40), os_9020, 'path')
# Obtaining the member 'realpath' of a type (line 87)
realpath_9022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 40), path_9021, 'realpath')
# Calling realpath(args, kwargs) (line 87)
realpath_call_result_9025 = invoke(stypy.reporting.localization.Localization(__file__, 87, 40), realpath_9022, *[file___9023], **kwargs_9024)

# Processing the call keyword arguments (line 87)
kwargs_9026 = {}
# Getting the type of 'os' (line 87)
os_9017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 24), 'os', False)
# Obtaining the member 'path' of a type (line 87)
path_9018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 24), os_9017, 'path')
# Obtaining the member 'dirname' of a type (line 87)
dirname_9019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 24), path_9018, 'dirname')
# Calling dirname(args, kwargs) (line 87)
dirname_call_result_9027 = invoke(stypy.reporting.localization.Localization(__file__, 87, 24), dirname_9019, *[realpath_call_result_9025], **kwargs_9026)

str_9028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 70), 'str', '/foo.txt')
# Applying the binary operator '+' (line 87)
result_add_9029 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 24), '+', dirname_call_result_9027, str_9028)

str_9030 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 82), 'str', 'w')
# Processing the call keyword arguments (line 87)
kwargs_9031 = {}
# Getting the type of 'file' (line 87)
file_9016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 19), 'file', False)
# Calling file(args, kwargs) (line 87)
file_call_result_9032 = invoke(stypy.reporting.localization.Localization(__file__, 87, 19), file_9016, *[result_add_9029, str_9030], **kwargs_9031)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 87, 11), tuple_9014, file_call_result_9032)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (file_9013, tuple_9014))
# Adding element type (key, value) (line 67)
# Getting the type of 'xrange' (line 88)
xrange_9033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'xrange')

# Obtaining an instance of the builtin type 'tuple' (line 88)
tuple_9034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 88)
# Adding element type (line 88)
str_9035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 13), 'str', 'xrange')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 13), tuple_9034, str_9035)
# Adding element type (line 88)

# Call to xrange(...): (line 88)
# Processing the call arguments (line 88)
int_9037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 30), 'int')
int_9038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 33), 'int')
# Processing the call keyword arguments (line 88)
kwargs_9039 = {}
# Getting the type of 'xrange' (line 88)
xrange_9036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 23), 'xrange', False)
# Calling xrange(args, kwargs) (line 88)
xrange_call_result_9040 = invoke(stypy.reporting.localization.Localization(__file__, 88, 23), xrange_9036, *[int_9037, int_9038], **kwargs_9039)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 13), tuple_9034, xrange_call_result_9040)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (xrange_9033, tuple_9034))
# Adding element type (key, value) (line 67)
# Getting the type of 'types' (line 89)
types_9041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'types')
# Obtaining the member 'SliceType' of a type (line 89)
SliceType_9042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 4), types_9041, 'SliceType')

# Obtaining an instance of the builtin type 'tuple' (line 89)
tuple_9043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 22), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 89)
# Adding element type (line 89)
str_9044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 22), 'str', 'slice')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 22), tuple_9043, str_9044)
# Adding element type (line 89)

# Call to slice(...): (line 89)
# Processing the call arguments (line 89)
int_9046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 37), 'int')
int_9047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 40), 'int')
int_9048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 43), 'int')
# Processing the call keyword arguments (line 89)
kwargs_9049 = {}
# Getting the type of 'slice' (line 89)
slice_9045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 31), 'slice', False)
# Calling slice(args, kwargs) (line 89)
slice_call_result_9050 = invoke(stypy.reporting.localization.Localization(__file__, 89, 31), slice_9045, *[int_9046, int_9047, int_9048], **kwargs_9049)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 22), tuple_9043, slice_call_result_9050)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (SliceType_9042, tuple_9043))
# Adding element type (key, value) (line 67)
# Getting the type of 'buffer' (line 90)
buffer_9051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'buffer')

# Obtaining an instance of the builtin type 'tuple' (line 90)
tuple_9052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 90)
# Adding element type (line 90)
str_9053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 13), 'str', 'buffer')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 13), tuple_9052, str_9053)
# Adding element type (line 90)

# Call to buffer(...): (line 90)
# Processing the call arguments (line 90)
str_9055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 30), 'str', 'r')
# Processing the call keyword arguments (line 90)
kwargs_9056 = {}
# Getting the type of 'buffer' (line 90)
buffer_9054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 23), 'buffer', False)
# Calling buffer(args, kwargs) (line 90)
buffer_call_result_9057 = invoke(stypy.reporting.localization.Localization(__file__, 90, 23), buffer_9054, *[str_9055], **kwargs_9056)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 90, 13), tuple_9052, buffer_call_result_9057)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (buffer_9051, tuple_9052))
# Adding element type (key, value) (line 67)
# Getting the type of 'types' (line 91)
types_9058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'types')
# Obtaining the member 'DictProxyType' of a type (line 91)
DictProxyType_9059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 4), types_9058, 'DictProxyType')

# Obtaining an instance of the builtin type 'tuple' (line 91)
tuple_9060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 26), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 91)
# Adding element type (line 91)
str_9061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 26), 'str', 'types.DictProxyType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 26), tuple_9060, str_9061)
# Adding element type (line 91)
# Getting the type of 'int' (line 91)
int_9062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 49), 'int')
# Obtaining the member '__dict__' of a type (line 91)
dict___9063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 49), int_9062, '__dict__')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 91, 26), tuple_9060, dict___9063)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (DictProxyType_9059, tuple_9060))
# Adding element type (key, value) (line 67)
# Getting the type of 'types' (line 92)
types_9064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'types')
# Obtaining the member 'ClassType' of a type (line 92)
ClassType_9065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 4), types_9064, 'ClassType')

# Obtaining an instance of the builtin type 'tuple' (line 92)
tuple_9066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 22), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 92)
# Adding element type (line 92)
str_9067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 22), 'str', 'types.ClassType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 22), tuple_9066, str_9067)
# Adding element type (line 92)
# Getting the type of 'Foo' (line 92)
Foo_9068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 41), 'Foo')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 22), tuple_9066, Foo_9068)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (ClassType_9065, tuple_9066))
# Adding element type (key, value) (line 67)
# Getting the type of 'types' (line 93)
types_9069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'types')
# Obtaining the member 'InstanceType' of a type (line 93)
InstanceType_9070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 4), types_9069, 'InstanceType')

# Obtaining an instance of the builtin type 'tuple' (line 93)
tuple_9071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 25), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 93)
# Adding element type (line 93)
str_9072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 25), 'str', 'types.InstanceType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 25), tuple_9071, str_9072)
# Adding element type (line 93)
# Getting the type of 'foo' (line 93)
foo_9073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 47), 'foo')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 25), tuple_9071, foo_9073)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (InstanceType_9070, tuple_9071))
# Adding element type (key, value) (line 67)
# Getting the type of 'types' (line 94)
types_9074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'types')
# Obtaining the member 'MethodType' of a type (line 94)
MethodType_9075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 4), types_9074, 'MethodType')

# Obtaining an instance of the builtin type 'tuple' (line 94)
tuple_9076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 23), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 94)
# Adding element type (line 94)
str_9077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 23), 'str', 'types.MethodType')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 23), tuple_9076, str_9077)
# Adding element type (line 94)
# Getting the type of 'foo' (line 94)
foo_9078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 43), 'foo')
# Obtaining the member 'bar' of a type (line 94)
bar_9079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 43), foo_9078, 'bar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 23), tuple_9076, bar_9079)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (MethodType_9075, tuple_9076))
# Adding element type (key, value) (line 67)
# Getting the type of 'iter' (line 95)
iter_9080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'iter')

# Obtaining an instance of the builtin type 'tuple' (line 95)
tuple_9081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 11), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 95)
# Adding element type (line 95)
str_9082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 11), 'str', 'iter')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 11), tuple_9081, str_9082)
# Adding element type (line 95)

# Call to iter(...): (line 95)
# Processing the call arguments (line 95)
str_9084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 24), 'str', 'abc')
# Processing the call keyword arguments (line 95)
kwargs_9085 = {}
# Getting the type of 'iter' (line 95)
iter_9083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 19), 'iter', False)
# Calling iter(args, kwargs) (line 95)
iter_call_result_9086 = invoke(stypy.reporting.localization.Localization(__file__, 95, 19), iter_9083, *[str_9084], **kwargs_9085)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 11), tuple_9081, iter_call_result_9086)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (iter_9080, tuple_9081))
# Adding element type (key, value) (line 67)
# Getting the type of 'bytearray' (line 97)
bytearray_9087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'bytearray')

# Obtaining an instance of the builtin type 'tuple' (line 97)
tuple_9088 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 97)
# Adding element type (line 97)
str_9089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 16), 'str', 'bytearray')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 16), tuple_9088, str_9089)
# Adding element type (line 97)

# Call to bytearray(...): (line 97)
# Processing the call arguments (line 97)
str_9091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 39), 'str', 'test')
# Processing the call keyword arguments (line 97)
kwargs_9092 = {}
# Getting the type of 'bytearray' (line 97)
bytearray_9090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 29), 'bytearray', False)
# Calling bytearray(args, kwargs) (line 97)
bytearray_call_result_9093 = invoke(stypy.reporting.localization.Localization(__file__, 97, 29), bytearray_9090, *[str_9091], **kwargs_9092)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 16), tuple_9088, bytearray_call_result_9093)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (bytearray_9087, tuple_9088))
# Adding element type (key, value) (line 67)
# Getting the type of 'classmethod' (line 98)
classmethod_9094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'classmethod')

# Obtaining an instance of the builtin type 'tuple' (line 98)
tuple_9095 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 98)
# Adding element type (line 98)
str_9096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 18), 'str', 'classmethod')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 18), tuple_9095, str_9096)
# Adding element type (line 98)

# Call to classmethod(...): (line 98)
# Processing the call arguments (line 98)
# Getting the type of 'Foo' (line 98)
Foo_9098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 45), 'Foo', False)
# Obtaining the member 'bar' of a type (line 98)
bar_9099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 45), Foo_9098, 'bar')
# Processing the call keyword arguments (line 98)
kwargs_9100 = {}
# Getting the type of 'classmethod' (line 98)
classmethod_9097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 33), 'classmethod', False)
# Calling classmethod(args, kwargs) (line 98)
classmethod_call_result_9101 = invoke(stypy.reporting.localization.Localization(__file__, 98, 33), classmethod_9097, *[bar_9099], **kwargs_9100)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 98, 18), tuple_9095, classmethod_call_result_9101)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (classmethod_9094, tuple_9095))
# Adding element type (key, value) (line 67)
# Getting the type of 'enumerate' (line 99)
enumerate_9102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'enumerate')

# Obtaining an instance of the builtin type 'tuple' (line 99)
tuple_9103 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 99)
# Adding element type (line 99)
str_9104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 16), 'str', 'enumerate')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 16), tuple_9103, str_9104)
# Adding element type (line 99)

# Call to enumerate(...): (line 99)
# Processing the call arguments (line 99)

# Obtaining an instance of the builtin type 'list' (line 99)
list_9106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 39), 'list')
# Adding type elements to the builtin type 'list' instance (line 99)
# Adding element type (line 99)
int_9107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 40), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 39), list_9106, int_9107)
# Adding element type (line 99)
int_9108 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 43), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 39), list_9106, int_9108)
# Adding element type (line 99)
int_9109 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 46), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 39), list_9106, int_9109)

# Processing the call keyword arguments (line 99)
kwargs_9110 = {}
# Getting the type of 'enumerate' (line 99)
enumerate_9105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 29), 'enumerate', False)
# Calling enumerate(args, kwargs) (line 99)
enumerate_call_result_9111 = invoke(stypy.reporting.localization.Localization(__file__, 99, 29), enumerate_9105, *[list_9106], **kwargs_9110)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 16), tuple_9103, enumerate_call_result_9111)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (enumerate_9102, tuple_9103))
# Adding element type (key, value) (line 67)
# Getting the type of 'frozenset' (line 100)
frozenset_9112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'frozenset')

# Obtaining an instance of the builtin type 'tuple' (line 100)
tuple_9113 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 100)
# Adding element type (line 100)
str_9114 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 16), 'str', 'frozenset')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 16), tuple_9113, str_9114)
# Adding element type (line 100)

# Call to frozenset(...): (line 100)
# Processing the call arguments (line 100)

# Obtaining an instance of the builtin type 'list' (line 100)
list_9116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 39), 'list')
# Adding type elements to the builtin type 'list' instance (line 100)
# Adding element type (line 100)
int_9117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 40), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 39), list_9116, int_9117)
# Adding element type (line 100)
int_9118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 43), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 39), list_9116, int_9118)
# Adding element type (line 100)
int_9119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 46), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 39), list_9116, int_9119)

# Processing the call keyword arguments (line 100)
kwargs_9120 = {}
# Getting the type of 'frozenset' (line 100)
frozenset_9115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 29), 'frozenset', False)
# Calling frozenset(args, kwargs) (line 100)
frozenset_call_result_9121 = invoke(stypy.reporting.localization.Localization(__file__, 100, 29), frozenset_9115, *[list_9116], **kwargs_9120)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 16), tuple_9113, frozenset_call_result_9121)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (frozenset_9112, tuple_9113))
# Adding element type (key, value) (line 67)
# Getting the type of 'memoryview' (line 101)
memoryview_9122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'memoryview')

# Obtaining an instance of the builtin type 'tuple' (line 101)
tuple_9123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 17), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 101)
# Adding element type (line 101)
str_9124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 17), 'str', 'memoryview')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 17), tuple_9123, str_9124)
# Adding element type (line 101)

# Call to memoryview(...): (line 101)
# Processing the call arguments (line 101)

# Call to buffer(...): (line 101)
# Processing the call arguments (line 101)
str_9127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 49), 'str', 'foo')
# Processing the call keyword arguments (line 101)
kwargs_9128 = {}
# Getting the type of 'buffer' (line 101)
buffer_9126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 42), 'buffer', False)
# Calling buffer(args, kwargs) (line 101)
buffer_call_result_9129 = invoke(stypy.reporting.localization.Localization(__file__, 101, 42), buffer_9126, *[str_9127], **kwargs_9128)

# Processing the call keyword arguments (line 101)
kwargs_9130 = {}
# Getting the type of 'memoryview' (line 101)
memoryview_9125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 31), 'memoryview', False)
# Calling memoryview(args, kwargs) (line 101)
memoryview_call_result_9131 = invoke(stypy.reporting.localization.Localization(__file__, 101, 31), memoryview_9125, *[buffer_call_result_9129], **kwargs_9130)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 17), tuple_9123, memoryview_call_result_9131)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (memoryview_9122, tuple_9123))
# Adding element type (key, value) (line 67)
# Getting the type of 'object' (line 102)
object_9132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'object')

# Obtaining an instance of the builtin type 'tuple' (line 102)
tuple_9133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 13), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 102)
# Adding element type (line 102)
str_9134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 13), 'str', 'object')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 13), tuple_9133, str_9134)
# Adding element type (line 102)

# Call to object(...): (line 102)
# Processing the call keyword arguments (line 102)
kwargs_9136 = {}
# Getting the type of 'object' (line 102)
object_9135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 23), 'object', False)
# Calling object(args, kwargs) (line 102)
object_call_result_9137 = invoke(stypy.reporting.localization.Localization(__file__, 102, 23), object_9135, *[], **kwargs_9136)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 13), tuple_9133, object_call_result_9137)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (object_9132, tuple_9133))
# Adding element type (key, value) (line 67)
# Getting the type of 'property' (line 103)
property_9138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'property')

# Obtaining an instance of the builtin type 'tuple' (line 103)
tuple_9139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 103)
# Adding element type (line 103)
str_9140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 15), 'str', 'property')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 15), tuple_9139, str_9140)
# Adding element type (line 103)

# Call to property(...): (line 103)
# Processing the call arguments (line 103)
# Getting the type of 'Foo' (line 103)
Foo_9142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 36), 'Foo', False)
# Obtaining the member 'qux' of a type (line 103)
qux_9143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 36), Foo_9142, 'qux')
# Processing the call keyword arguments (line 103)
kwargs_9144 = {}
# Getting the type of 'property' (line 103)
property_9141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 27), 'property', False)
# Calling property(args, kwargs) (line 103)
property_call_result_9145 = invoke(stypy.reporting.localization.Localization(__file__, 103, 27), property_9141, *[qux_9143], **kwargs_9144)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 103, 15), tuple_9139, property_call_result_9145)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (property_9138, tuple_9139))
# Adding element type (key, value) (line 67)
# Getting the type of 'staticmethod' (line 104)
staticmethod_9146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'staticmethod')

# Obtaining an instance of the builtin type 'tuple' (line 104)
tuple_9147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 19), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 104)
# Adding element type (line 104)
str_9148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 19), 'str', 'staticmethod')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 19), tuple_9147, str_9148)
# Adding element type (line 104)

# Call to staticmethod(...): (line 104)
# Processing the call arguments (line 104)
# Getting the type of 'Foo' (line 104)
Foo_9150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 48), 'Foo', False)
# Obtaining the member 'bar' of a type (line 104)
bar_9151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 48), Foo_9150, 'bar')
# Processing the call keyword arguments (line 104)
kwargs_9152 = {}
# Getting the type of 'staticmethod' (line 104)
staticmethod_9149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 35), 'staticmethod', False)
# Calling staticmethod(args, kwargs) (line 104)
staticmethod_call_result_9153 = invoke(stypy.reporting.localization.Localization(__file__, 104, 35), staticmethod_9149, *[bar_9151], **kwargs_9152)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 19), tuple_9147, staticmethod_call_result_9153)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (staticmethod_9146, tuple_9147))
# Adding element type (key, value) (line 67)
# Getting the type of 'super' (line 105)
super_9154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'super')

# Obtaining an instance of the builtin type 'tuple' (line 105)
tuple_9155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 12), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 105)
# Adding element type (line 105)
str_9156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 12), 'str', 'super')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 12), tuple_9155, str_9156)
# Adding element type (line 105)

# Call to super(...): (line 105)
# Processing the call arguments (line 105)

# Call to type(...): (line 105)
# Processing the call arguments (line 105)
# Getting the type of 'Foo' (line 105)
Foo_9159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 32), 'Foo', False)
# Processing the call keyword arguments (line 105)
kwargs_9160 = {}
# Getting the type of 'type' (line 105)
type_9158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 27), 'type', False)
# Calling type(args, kwargs) (line 105)
type_call_result_9161 = invoke(stypy.reporting.localization.Localization(__file__, 105, 27), type_9158, *[Foo_9159], **kwargs_9160)

# Processing the call keyword arguments (line 105)
kwargs_9162 = {}
# Getting the type of 'super' (line 105)
super_9157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 21), 'super', False)
# Calling super(args, kwargs) (line 105)
super_call_result_9163 = invoke(stypy.reporting.localization.Localization(__file__, 105, 21), super_9157, *[type_call_result_9161], **kwargs_9162)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 12), tuple_9155, super_call_result_9163)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (super_9154, tuple_9155))
# Adding element type (key, value) (line 67)
# Getting the type of 'reversed' (line 106)
reversed_9164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'reversed')

# Obtaining an instance of the builtin type 'tuple' (line 106)
tuple_9165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 106)
# Adding element type (line 106)
str_9166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 15), 'str', 'reversed')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 15), tuple_9165, str_9166)
# Adding element type (line 106)

# Call to reversed(...): (line 106)
# Processing the call arguments (line 106)

# Obtaining an instance of the builtin type 'tuple' (line 106)
tuple_9168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 37), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 106)
# Adding element type (line 106)
int_9169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 37), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 37), tuple_9168, int_9169)
# Adding element type (line 106)
int_9170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 40), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 37), tuple_9168, int_9170)

# Processing the call keyword arguments (line 106)
kwargs_9171 = {}
# Getting the type of 'reversed' (line 106)
reversed_9167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 27), 'reversed', False)
# Calling reversed(args, kwargs) (line 106)
reversed_call_result_9172 = invoke(stypy.reporting.localization.Localization(__file__, 106, 27), reversed_9167, *[tuple_9168], **kwargs_9171)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 15), tuple_9165, reversed_call_result_9172)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (reversed_9164, tuple_9165))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 108)
# Processing the call arguments (line 108)

# Call to iter(...): (line 108)
# Processing the call arguments (line 108)

# Obtaining an instance of the builtin type 'tuple' (line 108)
tuple_9175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 15), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 108)
# Adding element type (line 108)
int_9176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 15), tuple_9175, int_9176)
# Adding element type (line 108)
int_9177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 15), tuple_9175, int_9177)
# Adding element type (line 108)
int_9178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 21), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 15), tuple_9175, int_9178)
# Adding element type (line 108)
int_9179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 15), tuple_9175, int_9179)
# Adding element type (line 108)
int_9180 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 15), tuple_9175, int_9180)

# Processing the call keyword arguments (line 108)
kwargs_9181 = {}
# Getting the type of 'iter' (line 108)
iter_9174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 9), 'iter', False)
# Calling iter(args, kwargs) (line 108)
iter_call_result_9182 = invoke(stypy.reporting.localization.Localization(__file__, 108, 9), iter_9174, *[tuple_9175], **kwargs_9181)

# Processing the call keyword arguments (line 108)
kwargs_9183 = {}
# Getting the type of 'type' (line 108)
type_9173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'type', False)
# Calling type(args, kwargs) (line 108)
type_call_result_9184 = invoke(stypy.reporting.localization.Localization(__file__, 108, 4), type_9173, *[iter_call_result_9182], **kwargs_9183)


# Obtaining an instance of the builtin type 'tuple' (line 108)
tuple_9185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 34), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 108)
# Adding element type (line 108)
str_9186 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 34), 'str', 'ExtraTypeDefinitions.tupleiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 34), tuple_9185, str_9186)
# Adding element type (line 108)

# Call to __iter__(...): (line 108)
# Processing the call keyword arguments (line 108)
kwargs_9194 = {}

# Obtaining an instance of the builtin type 'tuple' (line 108)
tuple_9187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 73), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 108)
# Adding element type (line 108)
int_9188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 73), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 73), tuple_9187, int_9188)
# Adding element type (line 108)
int_9189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 76), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 73), tuple_9187, int_9189)
# Adding element type (line 108)
int_9190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 79), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 73), tuple_9187, int_9190)
# Adding element type (line 108)
int_9191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 82), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 73), tuple_9187, int_9191)
# Adding element type (line 108)
int_9192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 85), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 73), tuple_9187, int_9192)

# Obtaining the member '__iter__' of a type (line 108)
iter___9193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 73), tuple_9187, '__iter__')
# Calling __iter__(args, kwargs) (line 108)
iter___call_result_9195 = invoke(stypy.reporting.localization.Localization(__file__, 108, 73), iter___9193, *[], **kwargs_9194)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 34), tuple_9185, iter___call_result_9195)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (type_call_result_9184, tuple_9185))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 109)
# Processing the call arguments (line 109)

# Call to iter(...): (line 109)
# Processing the call arguments (line 109)

# Call to xrange(...): (line 109)
# Processing the call arguments (line 109)
int_9199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 21), 'int')
# Processing the call keyword arguments (line 109)
kwargs_9200 = {}
# Getting the type of 'xrange' (line 109)
xrange_9198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 14), 'xrange', False)
# Calling xrange(args, kwargs) (line 109)
xrange_call_result_9201 = invoke(stypy.reporting.localization.Localization(__file__, 109, 14), xrange_9198, *[int_9199], **kwargs_9200)

# Processing the call keyword arguments (line 109)
kwargs_9202 = {}
# Getting the type of 'iter' (line 109)
iter_9197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 9), 'iter', False)
# Calling iter(args, kwargs) (line 109)
iter_call_result_9203 = invoke(stypy.reporting.localization.Localization(__file__, 109, 9), iter_9197, *[xrange_call_result_9201], **kwargs_9202)

# Processing the call keyword arguments (line 109)
kwargs_9204 = {}
# Getting the type of 'type' (line 109)
type_9196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'type', False)
# Calling type(args, kwargs) (line 109)
type_call_result_9205 = invoke(stypy.reporting.localization.Localization(__file__, 109, 4), type_9196, *[iter_call_result_9203], **kwargs_9204)


# Obtaining an instance of the builtin type 'tuple' (line 109)
tuple_9206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 28), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 109)
# Adding element type (line 109)
str_9207 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 28), 'str', 'ExtraTypeDefinitions.rangeiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 28), tuple_9206, str_9207)
# Adding element type (line 109)

# Call to iter(...): (line 109)
# Processing the call arguments (line 109)

# Call to xrange(...): (line 109)
# Processing the call arguments (line 109)
int_9210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 78), 'int')
# Processing the call keyword arguments (line 109)
kwargs_9211 = {}
# Getting the type of 'xrange' (line 109)
xrange_9209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 71), 'xrange', False)
# Calling xrange(args, kwargs) (line 109)
xrange_call_result_9212 = invoke(stypy.reporting.localization.Localization(__file__, 109, 71), xrange_9209, *[int_9210], **kwargs_9211)

# Processing the call keyword arguments (line 109)
kwargs_9213 = {}
# Getting the type of 'iter' (line 109)
iter_9208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 66), 'iter', False)
# Calling iter(args, kwargs) (line 109)
iter_call_result_9214 = invoke(stypy.reporting.localization.Localization(__file__, 109, 66), iter_9208, *[xrange_call_result_9212], **kwargs_9213)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 109, 28), tuple_9206, iter_call_result_9214)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (type_call_result_9205, tuple_9206))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 110)
# Processing the call arguments (line 110)

# Call to iter(...): (line 110)
# Processing the call arguments (line 110)

# Obtaining an instance of the builtin type 'list' (line 110)
list_9217 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 14), 'list')
# Adding type elements to the builtin type 'list' instance (line 110)
# Adding element type (line 110)
int_9218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 15), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 14), list_9217, int_9218)
# Adding element type (line 110)
int_9219 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 18), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 14), list_9217, int_9219)

# Processing the call keyword arguments (line 110)
kwargs_9220 = {}
# Getting the type of 'iter' (line 110)
iter_9216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 9), 'iter', False)
# Calling iter(args, kwargs) (line 110)
iter_call_result_9221 = invoke(stypy.reporting.localization.Localization(__file__, 110, 9), iter_9216, *[list_9217], **kwargs_9220)

# Processing the call keyword arguments (line 110)
kwargs_9222 = {}
# Getting the type of 'type' (line 110)
type_9215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'type', False)
# Calling type(args, kwargs) (line 110)
type_call_result_9223 = invoke(stypy.reporting.localization.Localization(__file__, 110, 4), type_9215, *[iter_call_result_9221], **kwargs_9222)


# Obtaining an instance of the builtin type 'tuple' (line 110)
tuple_9224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 25), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 110)
# Adding element type (line 110)
str_9225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 25), 'str', 'ExtraTypeDefinitions.listiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 25), tuple_9224, str_9225)
# Adding element type (line 110)

# Call to __iter__(...): (line 110)
# Processing the call keyword arguments (line 110)
kwargs_9233 = {}

# Obtaining an instance of the builtin type 'list' (line 110)
list_9226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 62), 'list')
# Adding type elements to the builtin type 'list' instance (line 110)
# Adding element type (line 110)
int_9227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 63), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 62), list_9226, int_9227)
# Adding element type (line 110)
int_9228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 66), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 62), list_9226, int_9228)
# Adding element type (line 110)
int_9229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 69), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 62), list_9226, int_9229)
# Adding element type (line 110)
int_9230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 72), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 62), list_9226, int_9230)
# Adding element type (line 110)
int_9231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 75), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 62), list_9226, int_9231)

# Obtaining the member '__iter__' of a type (line 110)
iter___9232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 62), list_9226, '__iter__')
# Calling __iter__(args, kwargs) (line 110)
iter___call_result_9234 = invoke(stypy.reporting.localization.Localization(__file__, 110, 62), iter___9232, *[], **kwargs_9233)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 25), tuple_9224, iter___call_result_9234)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (type_call_result_9223, tuple_9224))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 111)
# Processing the call arguments (line 111)

# Call to iter(...): (line 111)
# Processing the call arguments (line 111)

# Call to type(...): (line 111)
# Processing the call arguments (line 111)
# Getting the type of 'int' (line 111)
int_9238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 19), 'int', False)
# Processing the call keyword arguments (line 111)
kwargs_9239 = {}
# Getting the type of 'type' (line 111)
type_9237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 14), 'type', False)
# Calling type(args, kwargs) (line 111)
type_call_result_9240 = invoke(stypy.reporting.localization.Localization(__file__, 111, 14), type_9237, *[int_9238], **kwargs_9239)

float_9241 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 25), 'float')
# Processing the call keyword arguments (line 111)
kwargs_9242 = {}
# Getting the type of 'iter' (line 111)
iter_9236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 9), 'iter', False)
# Calling iter(args, kwargs) (line 111)
iter_call_result_9243 = invoke(stypy.reporting.localization.Localization(__file__, 111, 9), iter_9236, *[type_call_result_9240, float_9241], **kwargs_9242)

# Processing the call keyword arguments (line 111)
kwargs_9244 = {}
# Getting the type of 'type' (line 111)
type_9235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'type', False)
# Calling type(args, kwargs) (line 111)
type_call_result_9245 = invoke(stypy.reporting.localization.Localization(__file__, 111, 4), type_9235, *[iter_call_result_9243], **kwargs_9244)


# Obtaining an instance of the builtin type 'tuple' (line 111)
tuple_9246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 33), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 111)
# Adding element type (line 111)
str_9247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 33), 'str', 'ExtraTypeDefinitions.callable_iterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 33), tuple_9246, str_9247)
# Adding element type (line 111)

# Call to iter(...): (line 111)
# Processing the call arguments (line 111)

# Call to type(...): (line 111)
# Processing the call arguments (line 111)
# Getting the type of 'int' (line 111)
int_9250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 85), 'int', False)
# Processing the call keyword arguments (line 111)
kwargs_9251 = {}
# Getting the type of 'type' (line 111)
type_9249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 80), 'type', False)
# Calling type(args, kwargs) (line 111)
type_call_result_9252 = invoke(stypy.reporting.localization.Localization(__file__, 111, 80), type_9249, *[int_9250], **kwargs_9251)

float_9253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 91), 'float')
# Processing the call keyword arguments (line 111)
kwargs_9254 = {}
# Getting the type of 'iter' (line 111)
iter_9248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 75), 'iter', False)
# Calling iter(args, kwargs) (line 111)
iter_call_result_9255 = invoke(stypy.reporting.localization.Localization(__file__, 111, 75), iter_9248, *[type_call_result_9252, float_9253], **kwargs_9254)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 111, 33), tuple_9246, iter_call_result_9255)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (type_call_result_9245, tuple_9246))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 112)
# Processing the call arguments (line 112)

# Call to iter(...): (line 112)
# Processing the call arguments (line 112)

# Call to reversed(...): (line 112)
# Processing the call arguments (line 112)

# Obtaining an instance of the builtin type 'list' (line 112)
list_9259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 23), 'list')
# Adding type elements to the builtin type 'list' instance (line 112)
# Adding element type (line 112)
int_9260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 24), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 23), list_9259, int_9260)
# Adding element type (line 112)
int_9261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 27), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 23), list_9259, int_9261)

# Processing the call keyword arguments (line 112)
kwargs_9262 = {}
# Getting the type of 'reversed' (line 112)
reversed_9258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 14), 'reversed', False)
# Calling reversed(args, kwargs) (line 112)
reversed_call_result_9263 = invoke(stypy.reporting.localization.Localization(__file__, 112, 14), reversed_9258, *[list_9259], **kwargs_9262)

# Processing the call keyword arguments (line 112)
kwargs_9264 = {}
# Getting the type of 'iter' (line 112)
iter_9257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 9), 'iter', False)
# Calling iter(args, kwargs) (line 112)
iter_call_result_9265 = invoke(stypy.reporting.localization.Localization(__file__, 112, 9), iter_9257, *[reversed_call_result_9263], **kwargs_9264)

# Processing the call keyword arguments (line 112)
kwargs_9266 = {}
# Getting the type of 'type' (line 112)
type_9256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'type', False)
# Calling type(args, kwargs) (line 112)
type_call_result_9267 = invoke(stypy.reporting.localization.Localization(__file__, 112, 4), type_9256, *[iter_call_result_9265], **kwargs_9266)


# Obtaining an instance of the builtin type 'tuple' (line 112)
tuple_9268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 35), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 112)
# Adding element type (line 112)
str_9269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 35), 'str', 'ExtraTypeDefinitions.listreverseiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 35), tuple_9268, str_9269)
# Adding element type (line 112)

# Call to iter(...): (line 112)
# Processing the call arguments (line 112)

# Call to reversed(...): (line 112)
# Processing the call arguments (line 112)

# Obtaining an instance of the builtin type 'list' (line 112)
list_9272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 93), 'list')
# Adding type elements to the builtin type 'list' instance (line 112)
# Adding element type (line 112)
int_9273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 94), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 93), list_9272, int_9273)
# Adding element type (line 112)
int_9274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 97), 'int')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 93), list_9272, int_9274)

# Processing the call keyword arguments (line 112)
kwargs_9275 = {}
# Getting the type of 'reversed' (line 112)
reversed_9271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 84), 'reversed', False)
# Calling reversed(args, kwargs) (line 112)
reversed_call_result_9276 = invoke(stypy.reporting.localization.Localization(__file__, 112, 84), reversed_9271, *[list_9272], **kwargs_9275)

# Processing the call keyword arguments (line 112)
kwargs_9277 = {}
# Getting the type of 'iter' (line 112)
iter_9270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 79), 'iter', False)
# Calling iter(args, kwargs) (line 112)
iter_call_result_9278 = invoke(stypy.reporting.localization.Localization(__file__, 112, 79), iter_9270, *[reversed_call_result_9276], **kwargs_9277)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 35), tuple_9268, iter_call_result_9278)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (type_call_result_9267, tuple_9268))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 113)
# Processing the call arguments (line 113)

# Call to methodcaller(...): (line 113)
# Processing the call arguments (line 113)
int_9282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 31), 'int')
# Processing the call keyword arguments (line 113)
kwargs_9283 = {}
# Getting the type of 'operator' (line 113)
operator_9280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 9), 'operator', False)
# Obtaining the member 'methodcaller' of a type (line 113)
methodcaller_9281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 9), operator_9280, 'methodcaller')
# Calling methodcaller(args, kwargs) (line 113)
methodcaller_call_result_9284 = invoke(stypy.reporting.localization.Localization(__file__, 113, 9), methodcaller_9281, *[int_9282], **kwargs_9283)

# Processing the call keyword arguments (line 113)
kwargs_9285 = {}
# Getting the type of 'type' (line 113)
type_9279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'type', False)
# Calling type(args, kwargs) (line 113)
type_call_result_9286 = invoke(stypy.reporting.localization.Localization(__file__, 113, 4), type_9279, *[methodcaller_call_result_9284], **kwargs_9285)


# Obtaining an instance of the builtin type 'tuple' (line 113)
tuple_9287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 37), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 113)
# Adding element type (line 113)
str_9288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 37), 'str', 'ExtraTypeDefinitions.methodcaller')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 37), tuple_9287, str_9288)
# Adding element type (line 113)

# Call to methodcaller(...): (line 113)
# Processing the call arguments (line 113)
int_9291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 96), 'int')
# Processing the call keyword arguments (line 113)
kwargs_9292 = {}
# Getting the type of 'operator' (line 113)
operator_9289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 74), 'operator', False)
# Obtaining the member 'methodcaller' of a type (line 113)
methodcaller_9290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 74), operator_9289, 'methodcaller')
# Calling methodcaller(args, kwargs) (line 113)
methodcaller_call_result_9293 = invoke(stypy.reporting.localization.Localization(__file__, 113, 74), methodcaller_9290, *[int_9291], **kwargs_9292)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 37), tuple_9287, methodcaller_call_result_9293)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (type_call_result_9286, tuple_9287))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 114)
# Processing the call arguments (line 114)

# Call to itemgetter(...): (line 114)
# Processing the call arguments (line 114)
int_9297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 29), 'int')
# Processing the call keyword arguments (line 114)
kwargs_9298 = {}
# Getting the type of 'operator' (line 114)
operator_9295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 9), 'operator', False)
# Obtaining the member 'itemgetter' of a type (line 114)
itemgetter_9296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 9), operator_9295, 'itemgetter')
# Calling itemgetter(args, kwargs) (line 114)
itemgetter_call_result_9299 = invoke(stypy.reporting.localization.Localization(__file__, 114, 9), itemgetter_9296, *[int_9297], **kwargs_9298)

# Processing the call keyword arguments (line 114)
kwargs_9300 = {}
# Getting the type of 'type' (line 114)
type_9294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'type', False)
# Calling type(args, kwargs) (line 114)
type_call_result_9301 = invoke(stypy.reporting.localization.Localization(__file__, 114, 4), type_9294, *[itemgetter_call_result_9299], **kwargs_9300)


# Obtaining an instance of the builtin type 'tuple' (line 114)
tuple_9302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 35), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 114)
# Adding element type (line 114)
str_9303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 35), 'str', 'ExtraTypeDefinitions.itemgetter')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 35), tuple_9302, str_9303)
# Adding element type (line 114)

# Call to itemgetter(...): (line 114)
# Processing the call arguments (line 114)
int_9306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 90), 'int')
# Processing the call keyword arguments (line 114)
kwargs_9307 = {}
# Getting the type of 'operator' (line 114)
operator_9304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 70), 'operator', False)
# Obtaining the member 'itemgetter' of a type (line 114)
itemgetter_9305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 70), operator_9304, 'itemgetter')
# Calling itemgetter(args, kwargs) (line 114)
itemgetter_call_result_9308 = invoke(stypy.reporting.localization.Localization(__file__, 114, 70), itemgetter_9305, *[int_9306], **kwargs_9307)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 35), tuple_9302, itemgetter_call_result_9308)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (type_call_result_9301, tuple_9302))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 115)
# Processing the call arguments (line 115)

# Call to attrgetter(...): (line 115)
# Processing the call arguments (line 115)
int_9312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 29), 'int')
# Processing the call keyword arguments (line 115)
kwargs_9313 = {}
# Getting the type of 'operator' (line 115)
operator_9310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 9), 'operator', False)
# Obtaining the member 'attrgetter' of a type (line 115)
attrgetter_9311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 9), operator_9310, 'attrgetter')
# Calling attrgetter(args, kwargs) (line 115)
attrgetter_call_result_9314 = invoke(stypy.reporting.localization.Localization(__file__, 115, 9), attrgetter_9311, *[int_9312], **kwargs_9313)

# Processing the call keyword arguments (line 115)
kwargs_9315 = {}
# Getting the type of 'type' (line 115)
type_9309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'type', False)
# Calling type(args, kwargs) (line 115)
type_call_result_9316 = invoke(stypy.reporting.localization.Localization(__file__, 115, 4), type_9309, *[attrgetter_call_result_9314], **kwargs_9315)


# Obtaining an instance of the builtin type 'tuple' (line 115)
tuple_9317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 35), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 115)
# Adding element type (line 115)
str_9318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 35), 'str', 'ExtraTypeDefinitions.attrgetter')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 35), tuple_9317, str_9318)
# Adding element type (line 115)

# Call to attrgetter(...): (line 115)
# Processing the call arguments (line 115)
int_9321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 90), 'int')
# Processing the call keyword arguments (line 115)
kwargs_9322 = {}
# Getting the type of 'operator' (line 115)
operator_9319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 70), 'operator', False)
# Obtaining the member 'attrgetter' of a type (line 115)
attrgetter_9320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 70), operator_9319, 'attrgetter')
# Calling attrgetter(args, kwargs) (line 115)
attrgetter_call_result_9323 = invoke(stypy.reporting.localization.Localization(__file__, 115, 70), attrgetter_9320, *[int_9321], **kwargs_9322)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 35), tuple_9317, attrgetter_call_result_9323)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (type_call_result_9316, tuple_9317))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 116)
# Processing the call arguments (line 116)

# Call to iter(...): (line 116)
# Processing the call arguments (line 116)

# Call to bytearray(...): (line 116)
# Processing the call arguments (line 116)
str_9327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 24), 'str', 'test')
# Processing the call keyword arguments (line 116)
kwargs_9328 = {}
# Getting the type of 'bytearray' (line 116)
bytearray_9326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 14), 'bytearray', False)
# Calling bytearray(args, kwargs) (line 116)
bytearray_call_result_9329 = invoke(stypy.reporting.localization.Localization(__file__, 116, 14), bytearray_9326, *[str_9327], **kwargs_9328)

# Processing the call keyword arguments (line 116)
kwargs_9330 = {}
# Getting the type of 'iter' (line 116)
iter_9325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 9), 'iter', False)
# Calling iter(args, kwargs) (line 116)
iter_call_result_9331 = invoke(stypy.reporting.localization.Localization(__file__, 116, 9), iter_9325, *[bytearray_call_result_9329], **kwargs_9330)

# Processing the call keyword arguments (line 116)
kwargs_9332 = {}
# Getting the type of 'type' (line 116)
type_9324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 4), 'type', False)
# Calling type(args, kwargs) (line 116)
type_call_result_9333 = invoke(stypy.reporting.localization.Localization(__file__, 116, 4), type_9324, *[iter_call_result_9331], **kwargs_9332)


# Obtaining an instance of the builtin type 'tuple' (line 116)
tuple_9334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 36), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 116)
# Adding element type (line 116)
str_9335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 36), 'str', 'ExtraTypeDefinitions.bytearray_iterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 36), tuple_9334, str_9335)
# Adding element type (line 116)

# Call to iter(...): (line 116)
# Processing the call arguments (line 116)

# Call to bytearray(...): (line 116)
# Processing the call arguments (line 116)
str_9338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 94), 'str', 'test')
# Processing the call keyword arguments (line 116)
kwargs_9339 = {}
# Getting the type of 'bytearray' (line 116)
bytearray_9337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 84), 'bytearray', False)
# Calling bytearray(args, kwargs) (line 116)
bytearray_call_result_9340 = invoke(stypy.reporting.localization.Localization(__file__, 116, 84), bytearray_9337, *[str_9338], **kwargs_9339)

# Processing the call keyword arguments (line 116)
kwargs_9341 = {}
# Getting the type of 'iter' (line 116)
iter_9336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 79), 'iter', False)
# Calling iter(args, kwargs) (line 116)
iter_call_result_9342 = invoke(stypy.reporting.localization.Localization(__file__, 116, 79), iter_9336, *[bytearray_call_result_9340], **kwargs_9341)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 36), tuple_9334, iter_call_result_9342)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (type_call_result_9333, tuple_9334))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 118)
# Processing the call arguments (line 118)

# Call to viewitems(...): (line 118)
# Processing the call keyword arguments (line 118)
kwargs_9350 = {}

# Obtaining an instance of the builtin type 'dict' (line 118)
dict_9344 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 9), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 118)
# Adding element type (key, value) (line 118)
str_9345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 10), 'str', 'a')
int_9346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 15), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 9), dict_9344, (str_9345, int_9346))
# Adding element type (key, value) (line 118)
str_9347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 18), 'str', 'b')
int_9348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 23), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 9), dict_9344, (str_9347, int_9348))

# Obtaining the member 'viewitems' of a type (line 118)
viewitems_9349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 9), dict_9344, 'viewitems')
# Calling viewitems(args, kwargs) (line 118)
viewitems_call_result_9351 = invoke(stypy.reporting.localization.Localization(__file__, 118, 9), viewitems_9349, *[], **kwargs_9350)

# Processing the call keyword arguments (line 118)
kwargs_9352 = {}
# Getting the type of 'type' (line 118)
type_9343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'type', False)
# Calling type(args, kwargs) (line 118)
type_call_result_9353 = invoke(stypy.reporting.localization.Localization(__file__, 118, 4), type_9343, *[viewitems_call_result_9351], **kwargs_9352)


# Obtaining an instance of the builtin type 'tuple' (line 118)
tuple_9354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 41), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 118)
# Adding element type (line 118)
str_9355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 41), 'str', 'ExtraTypeDefinitions.dict_items')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 41), tuple_9354, str_9355)
# Adding element type (line 118)

# Call to viewitems(...): (line 118)
# Processing the call keyword arguments (line 118)
kwargs_9362 = {}

# Obtaining an instance of the builtin type 'dict' (line 118)
dict_9356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 76), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 118)
# Adding element type (key, value) (line 118)
str_9357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 77), 'str', 'a')
int_9358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 82), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 76), dict_9356, (str_9357, int_9358))
# Adding element type (key, value) (line 118)
str_9359 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 85), 'str', 'b')
int_9360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 90), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 76), dict_9356, (str_9359, int_9360))

# Obtaining the member 'viewitems' of a type (line 118)
viewitems_9361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 76), dict_9356, 'viewitems')
# Calling viewitems(args, kwargs) (line 118)
viewitems_call_result_9363 = invoke(stypy.reporting.localization.Localization(__file__, 118, 76), viewitems_9361, *[], **kwargs_9362)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 41), tuple_9354, viewitems_call_result_9363)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (type_call_result_9353, tuple_9354))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 119)
# Processing the call arguments (line 119)

# Call to viewkeys(...): (line 119)
# Processing the call keyword arguments (line 119)
kwargs_9371 = {}

# Obtaining an instance of the builtin type 'dict' (line 119)
dict_9365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 9), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 119)
# Adding element type (key, value) (line 119)
str_9366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 10), 'str', 'a')
int_9367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 15), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 9), dict_9365, (str_9366, int_9367))
# Adding element type (key, value) (line 119)
str_9368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 18), 'str', 'b')
int_9369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 23), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 9), dict_9365, (str_9368, int_9369))

# Obtaining the member 'viewkeys' of a type (line 119)
viewkeys_9370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 9), dict_9365, 'viewkeys')
# Calling viewkeys(args, kwargs) (line 119)
viewkeys_call_result_9372 = invoke(stypy.reporting.localization.Localization(__file__, 119, 9), viewkeys_9370, *[], **kwargs_9371)

# Processing the call keyword arguments (line 119)
kwargs_9373 = {}
# Getting the type of 'type' (line 119)
type_9364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'type', False)
# Calling type(args, kwargs) (line 119)
type_call_result_9374 = invoke(stypy.reporting.localization.Localization(__file__, 119, 4), type_9364, *[viewkeys_call_result_9372], **kwargs_9373)


# Obtaining an instance of the builtin type 'tuple' (line 119)
tuple_9375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 40), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 119)
# Adding element type (line 119)
str_9376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 40), 'str', 'ExtraTypeDefinitions.dict_keys')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 40), tuple_9375, str_9376)
# Adding element type (line 119)

# Call to viewkeys(...): (line 119)
# Processing the call keyword arguments (line 119)
kwargs_9383 = {}

# Obtaining an instance of the builtin type 'dict' (line 119)
dict_9377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 74), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 119)
# Adding element type (key, value) (line 119)
str_9378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 75), 'str', 'a')
int_9379 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 80), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 74), dict_9377, (str_9378, int_9379))
# Adding element type (key, value) (line 119)
str_9380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 83), 'str', 'b')
int_9381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 88), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 74), dict_9377, (str_9380, int_9381))

# Obtaining the member 'viewkeys' of a type (line 119)
viewkeys_9382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 74), dict_9377, 'viewkeys')
# Calling viewkeys(args, kwargs) (line 119)
viewkeys_call_result_9384 = invoke(stypy.reporting.localization.Localization(__file__, 119, 74), viewkeys_9382, *[], **kwargs_9383)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 119, 40), tuple_9375, viewkeys_call_result_9384)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (type_call_result_9374, tuple_9375))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 120)
# Processing the call arguments (line 120)

# Call to viewvalues(...): (line 120)
# Processing the call keyword arguments (line 120)
kwargs_9392 = {}

# Obtaining an instance of the builtin type 'dict' (line 120)
dict_9386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 9), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 120)
# Adding element type (key, value) (line 120)
str_9387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 10), 'str', 'a')
int_9388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 15), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 9), dict_9386, (str_9387, int_9388))
# Adding element type (key, value) (line 120)
str_9389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 18), 'str', 'b')
int_9390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 23), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 9), dict_9386, (str_9389, int_9390))

# Obtaining the member 'viewvalues' of a type (line 120)
viewvalues_9391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 9), dict_9386, 'viewvalues')
# Calling viewvalues(args, kwargs) (line 120)
viewvalues_call_result_9393 = invoke(stypy.reporting.localization.Localization(__file__, 120, 9), viewvalues_9391, *[], **kwargs_9392)

# Processing the call keyword arguments (line 120)
kwargs_9394 = {}
# Getting the type of 'type' (line 120)
type_9385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'type', False)
# Calling type(args, kwargs) (line 120)
type_call_result_9395 = invoke(stypy.reporting.localization.Localization(__file__, 120, 4), type_9385, *[viewvalues_call_result_9393], **kwargs_9394)


# Obtaining an instance of the builtin type 'tuple' (line 120)
tuple_9396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 42), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 120)
# Adding element type (line 120)
str_9397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 42), 'str', 'ExtraTypeDefinitions.dict_values')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 42), tuple_9396, str_9397)
# Adding element type (line 120)

# Call to viewvalues(...): (line 120)
# Processing the call keyword arguments (line 120)
kwargs_9404 = {}

# Obtaining an instance of the builtin type 'dict' (line 120)
dict_9398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 78), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 120)
# Adding element type (key, value) (line 120)
str_9399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 79), 'str', 'a')
int_9400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 84), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 78), dict_9398, (str_9399, int_9400))
# Adding element type (key, value) (line 120)
str_9401 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 87), 'str', 'b')
int_9402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 92), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 78), dict_9398, (str_9401, int_9402))

# Obtaining the member 'viewvalues' of a type (line 120)
viewvalues_9403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 78), dict_9398, 'viewvalues')
# Calling viewvalues(args, kwargs) (line 120)
viewvalues_call_result_9405 = invoke(stypy.reporting.localization.Localization(__file__, 120, 78), viewvalues_9403, *[], **kwargs_9404)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 42), tuple_9396, viewvalues_call_result_9405)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (type_call_result_9395, tuple_9396))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 122)
# Processing the call arguments (line 122)

# Call to iter(...): (line 122)
# Processing the call arguments (line 122)

# Obtaining an instance of the builtin type 'dict' (line 122)
dict_9408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 14), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 122)
# Adding element type (key, value) (line 122)
str_9409 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 15), 'str', 'a')
int_9410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 20), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 14), dict_9408, (str_9409, int_9410))
# Adding element type (key, value) (line 122)
str_9411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 23), 'str', 'b')
int_9412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 28), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 14), dict_9408, (str_9411, int_9412))

# Processing the call keyword arguments (line 122)
kwargs_9413 = {}
# Getting the type of 'iter' (line 122)
iter_9407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 9), 'iter', False)
# Calling iter(args, kwargs) (line 122)
iter_call_result_9414 = invoke(stypy.reporting.localization.Localization(__file__, 122, 9), iter_9407, *[dict_9408], **kwargs_9413)

# Processing the call keyword arguments (line 122)
kwargs_9415 = {}
# Getting the type of 'type' (line 122)
type_9406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'type', False)
# Calling type(args, kwargs) (line 122)
type_call_result_9416 = invoke(stypy.reporting.localization.Localization(__file__, 122, 4), type_9406, *[iter_call_result_9414], **kwargs_9415)


# Obtaining an instance of the builtin type 'tuple' (line 122)
tuple_9417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 35), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 122)
# Adding element type (line 122)
str_9418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 35), 'str', 'ExtraTypeDefinitions.dictionary_keyiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 35), tuple_9417, str_9418)
# Adding element type (line 122)

# Call to iter(...): (line 122)
# Processing the call arguments (line 122)

# Obtaining an instance of the builtin type 'dict' (line 122)
dict_9420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 87), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 122)
# Adding element type (key, value) (line 122)
str_9421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 88), 'str', 'a')
int_9422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 93), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 87), dict_9420, (str_9421, int_9422))
# Adding element type (key, value) (line 122)
str_9423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 96), 'str', 'b')
int_9424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 101), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 87), dict_9420, (str_9423, int_9424))

# Processing the call keyword arguments (line 122)
kwargs_9425 = {}
# Getting the type of 'iter' (line 122)
iter_9419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 82), 'iter', False)
# Calling iter(args, kwargs) (line 122)
iter_call_result_9426 = invoke(stypy.reporting.localization.Localization(__file__, 122, 82), iter_9419, *[dict_9420], **kwargs_9425)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 35), tuple_9417, iter_call_result_9426)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (type_call_result_9416, tuple_9417))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 123)
# Processing the call arguments (line 123)

# Call to iteritems(...): (line 123)
# Processing the call keyword arguments (line 123)
kwargs_9434 = {}

# Obtaining an instance of the builtin type 'dict' (line 123)
dict_9428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 9), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 123)
# Adding element type (key, value) (line 123)
str_9429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 10), 'str', 'a')
int_9430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 15), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 9), dict_9428, (str_9429, int_9430))
# Adding element type (key, value) (line 123)
str_9431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 18), 'str', 'b')
int_9432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 23), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 9), dict_9428, (str_9431, int_9432))

# Obtaining the member 'iteritems' of a type (line 123)
iteritems_9433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 9), dict_9428, 'iteritems')
# Calling iteritems(args, kwargs) (line 123)
iteritems_call_result_9435 = invoke(stypy.reporting.localization.Localization(__file__, 123, 9), iteritems_9433, *[], **kwargs_9434)

# Processing the call keyword arguments (line 123)
kwargs_9436 = {}
# Getting the type of 'type' (line 123)
type_9427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'type', False)
# Calling type(args, kwargs) (line 123)
type_call_result_9437 = invoke(stypy.reporting.localization.Localization(__file__, 123, 4), type_9427, *[iteritems_call_result_9435], **kwargs_9436)


# Obtaining an instance of the builtin type 'tuple' (line 123)
tuple_9438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 41), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 123)
# Adding element type (line 123)
str_9439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 41), 'str', 'ExtraTypeDefinitions.dictionary_itemiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 41), tuple_9438, str_9439)
# Adding element type (line 123)

# Call to iteritems(...): (line 123)
# Processing the call keyword arguments (line 123)
kwargs_9444 = {}

# Obtaining an instance of the builtin type 'dict' (line 123)
dict_9440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 89), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 123)
# Adding element type (key, value) (line 123)
str_9441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 90), 'str', 'a')
int_9442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 95), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 89), dict_9440, (str_9441, int_9442))

# Obtaining the member 'iteritems' of a type (line 123)
iteritems_9443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 89), dict_9440, 'iteritems')
# Calling iteritems(args, kwargs) (line 123)
iteritems_call_result_9445 = invoke(stypy.reporting.localization.Localization(__file__, 123, 89), iteritems_9443, *[], **kwargs_9444)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 41), tuple_9438, iteritems_call_result_9445)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (type_call_result_9437, tuple_9438))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 124)
# Processing the call arguments (line 124)

# Call to itervalues(...): (line 124)
# Processing the call keyword arguments (line 124)
kwargs_9453 = {}

# Obtaining an instance of the builtin type 'dict' (line 124)
dict_9447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 9), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 124)
# Adding element type (key, value) (line 124)
str_9448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 10), 'str', 'a')
int_9449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 15), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 9), dict_9447, (str_9448, int_9449))
# Adding element type (key, value) (line 124)
str_9450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 18), 'str', 'b')
int_9451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 23), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 9), dict_9447, (str_9450, int_9451))

# Obtaining the member 'itervalues' of a type (line 124)
itervalues_9452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 9), dict_9447, 'itervalues')
# Calling itervalues(args, kwargs) (line 124)
itervalues_call_result_9454 = invoke(stypy.reporting.localization.Localization(__file__, 124, 9), itervalues_9452, *[], **kwargs_9453)

# Processing the call keyword arguments (line 124)
kwargs_9455 = {}
# Getting the type of 'type' (line 124)
type_9446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'type', False)
# Calling type(args, kwargs) (line 124)
type_call_result_9456 = invoke(stypy.reporting.localization.Localization(__file__, 124, 4), type_9446, *[itervalues_call_result_9454], **kwargs_9455)


# Obtaining an instance of the builtin type 'tuple' (line 124)
tuple_9457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 42), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 124)
# Adding element type (line 124)
str_9458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 42), 'str', 'ExtraTypeDefinitions.dictionary_valueiterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 42), tuple_9457, str_9458)
# Adding element type (line 124)

# Call to itervalues(...): (line 124)
# Processing the call keyword arguments (line 124)
kwargs_9463 = {}

# Obtaining an instance of the builtin type 'dict' (line 124)
dict_9459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 91), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 124)
# Adding element type (key, value) (line 124)
str_9460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 92), 'str', 'a')
int_9461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 97), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 91), dict_9459, (str_9460, int_9461))

# Obtaining the member 'itervalues' of a type (line 124)
itervalues_9462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 91), dict_9459, 'itervalues')
# Calling itervalues(args, kwargs) (line 124)
itervalues_call_result_9464 = invoke(stypy.reporting.localization.Localization(__file__, 124, 91), itervalues_9462, *[], **kwargs_9463)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 42), tuple_9457, itervalues_call_result_9464)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (type_call_result_9456, tuple_9457))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 126)
# Processing the call arguments (line 126)

# Call to iter(...): (line 126)
# Processing the call arguments (line 126)

# Call to bytearray(...): (line 126)
# Processing the call arguments (line 126)
str_9468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 24), 'str', 'test')
# Processing the call keyword arguments (line 126)
kwargs_9469 = {}
# Getting the type of 'bytearray' (line 126)
bytearray_9467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 14), 'bytearray', False)
# Calling bytearray(args, kwargs) (line 126)
bytearray_call_result_9470 = invoke(stypy.reporting.localization.Localization(__file__, 126, 14), bytearray_9467, *[str_9468], **kwargs_9469)

# Processing the call keyword arguments (line 126)
kwargs_9471 = {}
# Getting the type of 'iter' (line 126)
iter_9466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 9), 'iter', False)
# Calling iter(args, kwargs) (line 126)
iter_call_result_9472 = invoke(stypy.reporting.localization.Localization(__file__, 126, 9), iter_9466, *[bytearray_call_result_9470], **kwargs_9471)

# Processing the call keyword arguments (line 126)
kwargs_9473 = {}
# Getting the type of 'type' (line 126)
type_9465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'type', False)
# Calling type(args, kwargs) (line 126)
type_call_result_9474 = invoke(stypy.reporting.localization.Localization(__file__, 126, 4), type_9465, *[iter_call_result_9472], **kwargs_9473)


# Obtaining an instance of the builtin type 'tuple' (line 126)
tuple_9475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 36), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 126)
# Adding element type (line 126)
str_9476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 36), 'str', 'ExtraTypeDefinitions.bytearray_iterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 36), tuple_9475, str_9476)
# Adding element type (line 126)

# Call to iter(...): (line 126)
# Processing the call arguments (line 126)

# Call to bytearray(...): (line 126)
# Processing the call arguments (line 126)
str_9479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 94), 'str', 'test')
# Processing the call keyword arguments (line 126)
kwargs_9480 = {}
# Getting the type of 'bytearray' (line 126)
bytearray_9478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 84), 'bytearray', False)
# Calling bytearray(args, kwargs) (line 126)
bytearray_call_result_9481 = invoke(stypy.reporting.localization.Localization(__file__, 126, 84), bytearray_9478, *[str_9479], **kwargs_9480)

# Processing the call keyword arguments (line 126)
kwargs_9482 = {}
# Getting the type of 'iter' (line 126)
iter_9477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 79), 'iter', False)
# Calling iter(args, kwargs) (line 126)
iter_call_result_9483 = invoke(stypy.reporting.localization.Localization(__file__, 126, 79), iter_9477, *[bytearray_call_result_9481], **kwargs_9482)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 36), tuple_9475, iter_call_result_9483)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (type_call_result_9474, tuple_9475))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 128)
# Processing the call arguments (line 128)
# Getting the type of 'ArithmeticError' (line 128)
ArithmeticError_9485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 9), 'ArithmeticError', False)
# Obtaining the member 'message' of a type (line 128)
message_9486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 9), ArithmeticError_9485, 'message')
# Processing the call keyword arguments (line 128)
kwargs_9487 = {}
# Getting the type of 'type' (line 128)
type_9484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'type', False)
# Calling type(args, kwargs) (line 128)
type_call_result_9488 = invoke(stypy.reporting.localization.Localization(__file__, 128, 4), type_9484, *[message_9486], **kwargs_9487)


# Obtaining an instance of the builtin type 'tuple' (line 128)
tuple_9489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 36), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 128)
# Adding element type (line 128)
str_9490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 36), 'str', 'ExtraTypeDefinitions.getset_descriptor')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 36), tuple_9489, str_9490)
# Adding element type (line 128)
# Getting the type of 'ArithmeticError' (line 128)
ArithmeticError_9491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 78), 'ArithmeticError')
# Obtaining the member 'message' of a type (line 128)
message_9492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 78), ArithmeticError_9491, 'message')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 36), tuple_9489, message_9492)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (type_call_result_9488, tuple_9489))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 129)
# Processing the call arguments (line 129)
# Getting the type of 'IOError' (line 129)
IOError_9494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 9), 'IOError', False)
# Obtaining the member 'errno' of a type (line 129)
errno_9495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 9), IOError_9494, 'errno')
# Processing the call keyword arguments (line 129)
kwargs_9496 = {}
# Getting the type of 'type' (line 129)
type_9493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'type', False)
# Calling type(args, kwargs) (line 129)
type_call_result_9497 = invoke(stypy.reporting.localization.Localization(__file__, 129, 4), type_9493, *[errno_9495], **kwargs_9496)


# Obtaining an instance of the builtin type 'tuple' (line 129)
tuple_9498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 26), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 129)
# Adding element type (line 129)
str_9499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 26), 'str', 'ExtraTypeDefinitions.member_descriptor')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 26), tuple_9498, str_9499)
# Adding element type (line 129)
# Getting the type of 'IOError' (line 129)
IOError_9500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 68), 'IOError')
# Obtaining the member 'errno' of a type (line 129)
errno_9501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 68), IOError_9500, 'errno')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 26), tuple_9498, errno_9501)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (type_call_result_9497, tuple_9498))
# Adding element type (key, value) (line 67)

# Call to type(...): (line 130)
# Processing the call arguments (line 130)

# Call to _formatter_parser(...): (line 130)
# Processing the call keyword arguments (line 130)
kwargs_9505 = {}
unicode_9503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 9), 'unicode', u'foo')
# Obtaining the member '_formatter_parser' of a type (line 130)
_formatter_parser_9504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 9), unicode_9503, '_formatter_parser')
# Calling _formatter_parser(args, kwargs) (line 130)
_formatter_parser_call_result_9506 = invoke(stypy.reporting.localization.Localization(__file__, 130, 9), _formatter_parser_9504, *[], **kwargs_9505)

# Processing the call keyword arguments (line 130)
kwargs_9507 = {}
# Getting the type of 'type' (line 130)
type_9502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'type', False)
# Calling type(args, kwargs) (line 130)
type_call_result_9508 = invoke(stypy.reporting.localization.Localization(__file__, 130, 4), type_9502, *[_formatter_parser_call_result_9506], **kwargs_9507)


# Obtaining an instance of the builtin type 'tuple' (line 130)
tuple_9509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 39), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 130)
# Adding element type (line 130)
str_9510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 39), 'str', 'ExtraTypeDefinitions.formatteriterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 39), tuple_9509, str_9510)
# Adding element type (line 130)

# Call to type(...): (line 130)
# Processing the call arguments (line 130)

# Call to _formatter_parser(...): (line 130)
# Processing the call keyword arguments (line 130)
kwargs_9514 = {}
unicode_9512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 86), 'unicode', u'foo')
# Obtaining the member '_formatter_parser' of a type (line 130)
_formatter_parser_9513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 86), unicode_9512, '_formatter_parser')
# Calling _formatter_parser(args, kwargs) (line 130)
_formatter_parser_call_result_9515 = invoke(stypy.reporting.localization.Localization(__file__, 130, 86), _formatter_parser_9513, *[], **kwargs_9514)

# Processing the call keyword arguments (line 130)
kwargs_9516 = {}
# Getting the type of 'type' (line 130)
type_9511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 81), 'type', False)
# Calling type(args, kwargs) (line 130)
type_call_result_9517 = invoke(stypy.reporting.localization.Localization(__file__, 130, 81), type_9511, *[_formatter_parser_call_result_9515], **kwargs_9516)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 130, 39), tuple_9509, type_call_result_9517)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 67, 42), dict_8905, (type_call_result_9508, tuple_9509))

# Assigning a type to the variable 'known_python_type_typename_samplevalues' (line 67)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'known_python_type_typename_samplevalues', dict_8905)
# Declaration of the 'ExtraTypeDefinitions' class

class ExtraTypeDefinitions:
    str_9518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, (-1)), 'str', '\n    Additional (not included) type definitions to those defined in the types Python module. This class is needed\n    to have an usable type object to refer to when generating Python code\n    ')

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
set_9519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 14), 'set')
# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'SetType' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9520, 'SetType', set_9519)

# Assigning a Call to a Name (line 140):

# Call to type(...): (line 140)
# Processing the call arguments (line 140)

# Call to iter(...): (line 140)
# Processing the call arguments (line 140)
str_9523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 25), 'str', '')
# Processing the call keyword arguments (line 140)
kwargs_9524 = {}
# Getting the type of 'iter' (line 140)
iter_9522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 20), 'iter', False)
# Calling iter(args, kwargs) (line 140)
iter_call_result_9525 = invoke(stypy.reporting.localization.Localization(__file__, 140, 20), iter_9522, *[str_9523], **kwargs_9524)

# Processing the call keyword arguments (line 140)
kwargs_9526 = {}
# Getting the type of 'type' (line 140)
type_9521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 15), 'type', False)
# Calling type(args, kwargs) (line 140)
type_call_result_9527 = invoke(stypy.reporting.localization.Localization(__file__, 140, 15), type_9521, *[iter_call_result_9525], **kwargs_9526)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'iterator' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9528, 'iterator', type_call_result_9527)

# Assigning a Call to a Name (line 142):

# Call to type(...): (line 142)
# Processing the call arguments (line 142)

# Call to iter(...): (line 142)
# Processing the call arguments (line 142)

# Call to set(...): (line 142)
# Processing the call keyword arguments (line 142)
kwargs_9532 = {}
# Getting the type of 'set' (line 142)
set_9531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 28), 'set', False)
# Calling set(args, kwargs) (line 142)
set_call_result_9533 = invoke(stypy.reporting.localization.Localization(__file__, 142, 28), set_9531, *[], **kwargs_9532)

# Processing the call keyword arguments (line 142)
kwargs_9534 = {}
# Getting the type of 'iter' (line 142)
iter_9530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 23), 'iter', False)
# Calling iter(args, kwargs) (line 142)
iter_call_result_9535 = invoke(stypy.reporting.localization.Localization(__file__, 142, 23), iter_9530, *[set_call_result_9533], **kwargs_9534)

# Processing the call keyword arguments (line 142)
kwargs_9536 = {}
# Getting the type of 'type' (line 142)
type_9529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 18), 'type', False)
# Calling type(args, kwargs) (line 142)
type_call_result_9537 = invoke(stypy.reporting.localization.Localization(__file__, 142, 18), type_9529, *[iter_call_result_9535], **kwargs_9536)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'setiterator' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9538, 'setiterator', type_call_result_9537)

# Assigning a Call to a Name (line 143):

# Call to type(...): (line 143)
# Processing the call arguments (line 143)

# Call to iter(...): (line 143)
# Processing the call arguments (line 143)

# Call to tuple(...): (line 143)
# Processing the call keyword arguments (line 143)
kwargs_9542 = {}
# Getting the type of 'tuple' (line 143)
tuple_9541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 30), 'tuple', False)
# Calling tuple(args, kwargs) (line 143)
tuple_call_result_9543 = invoke(stypy.reporting.localization.Localization(__file__, 143, 30), tuple_9541, *[], **kwargs_9542)

# Processing the call keyword arguments (line 143)
kwargs_9544 = {}
# Getting the type of 'iter' (line 143)
iter_9540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 25), 'iter', False)
# Calling iter(args, kwargs) (line 143)
iter_call_result_9545 = invoke(stypy.reporting.localization.Localization(__file__, 143, 25), iter_9540, *[tuple_call_result_9543], **kwargs_9544)

# Processing the call keyword arguments (line 143)
kwargs_9546 = {}
# Getting the type of 'type' (line 143)
type_9539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 20), 'type', False)
# Calling type(args, kwargs) (line 143)
type_call_result_9547 = invoke(stypy.reporting.localization.Localization(__file__, 143, 20), type_9539, *[iter_call_result_9545], **kwargs_9546)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'tupleiterator' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9548, 'tupleiterator', type_call_result_9547)

# Assigning a Call to a Name (line 144):

# Call to type(...): (line 144)
# Processing the call arguments (line 144)

# Call to iter(...): (line 144)
# Processing the call arguments (line 144)

# Call to xrange(...): (line 144)
# Processing the call arguments (line 144)
int_9552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 37), 'int')
# Processing the call keyword arguments (line 144)
kwargs_9553 = {}
# Getting the type of 'xrange' (line 144)
xrange_9551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 30), 'xrange', False)
# Calling xrange(args, kwargs) (line 144)
xrange_call_result_9554 = invoke(stypy.reporting.localization.Localization(__file__, 144, 30), xrange_9551, *[int_9552], **kwargs_9553)

# Processing the call keyword arguments (line 144)
kwargs_9555 = {}
# Getting the type of 'iter' (line 144)
iter_9550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 25), 'iter', False)
# Calling iter(args, kwargs) (line 144)
iter_call_result_9556 = invoke(stypy.reporting.localization.Localization(__file__, 144, 25), iter_9550, *[xrange_call_result_9554], **kwargs_9555)

# Processing the call keyword arguments (line 144)
kwargs_9557 = {}
# Getting the type of 'type' (line 144)
type_9549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 20), 'type', False)
# Calling type(args, kwargs) (line 144)
type_call_result_9558 = invoke(stypy.reporting.localization.Localization(__file__, 144, 20), type_9549, *[iter_call_result_9556], **kwargs_9557)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'rangeiterator' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9559, 'rangeiterator', type_call_result_9558)

# Assigning a Call to a Name (line 145):

# Call to type(...): (line 145)
# Processing the call arguments (line 145)

# Call to iter(...): (line 145)
# Processing the call arguments (line 145)

# Call to list(...): (line 145)
# Processing the call keyword arguments (line 145)
kwargs_9563 = {}
# Getting the type of 'list' (line 145)
list_9562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 29), 'list', False)
# Calling list(args, kwargs) (line 145)
list_call_result_9564 = invoke(stypy.reporting.localization.Localization(__file__, 145, 29), list_9562, *[], **kwargs_9563)

# Processing the call keyword arguments (line 145)
kwargs_9565 = {}
# Getting the type of 'iter' (line 145)
iter_9561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 24), 'iter', False)
# Calling iter(args, kwargs) (line 145)
iter_call_result_9566 = invoke(stypy.reporting.localization.Localization(__file__, 145, 24), iter_9561, *[list_call_result_9564], **kwargs_9565)

# Processing the call keyword arguments (line 145)
kwargs_9567 = {}
# Getting the type of 'type' (line 145)
type_9560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 19), 'type', False)
# Calling type(args, kwargs) (line 145)
type_call_result_9568 = invoke(stypy.reporting.localization.Localization(__file__, 145, 19), type_9560, *[iter_call_result_9566], **kwargs_9567)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'listiterator' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9569, 'listiterator', type_call_result_9568)

# Assigning a Call to a Name (line 146):

# Call to type(...): (line 146)
# Processing the call arguments (line 146)

# Call to iter(...): (line 146)
# Processing the call arguments (line 146)

# Call to type(...): (line 146)
# Processing the call arguments (line 146)
# Getting the type of 'int' (line 146)
int_9573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 39), 'int', False)
# Processing the call keyword arguments (line 146)
kwargs_9574 = {}
# Getting the type of 'type' (line 146)
type_9572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 34), 'type', False)
# Calling type(args, kwargs) (line 146)
type_call_result_9575 = invoke(stypy.reporting.localization.Localization(__file__, 146, 34), type_9572, *[int_9573], **kwargs_9574)

float_9576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 45), 'float')
# Processing the call keyword arguments (line 146)
kwargs_9577 = {}
# Getting the type of 'iter' (line 146)
iter_9571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 29), 'iter', False)
# Calling iter(args, kwargs) (line 146)
iter_call_result_9578 = invoke(stypy.reporting.localization.Localization(__file__, 146, 29), iter_9571, *[type_call_result_9575, float_9576], **kwargs_9577)

# Processing the call keyword arguments (line 146)
kwargs_9579 = {}
# Getting the type of 'type' (line 146)
type_9570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 24), 'type', False)
# Calling type(args, kwargs) (line 146)
type_call_result_9580 = invoke(stypy.reporting.localization.Localization(__file__, 146, 24), type_9570, *[iter_call_result_9578], **kwargs_9579)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'callable_iterator' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9581, 'callable_iterator', type_call_result_9580)

# Assigning a Call to a Name (line 147):

# Call to type(...): (line 147)
# Processing the call arguments (line 147)

# Call to iter(...): (line 147)
# Processing the call arguments (line 147)

# Call to reversed(...): (line 147)
# Processing the call arguments (line 147)

# Call to list(...): (line 147)
# Processing the call keyword arguments (line 147)
kwargs_9586 = {}
# Getting the type of 'list' (line 147)
list_9585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 45), 'list', False)
# Calling list(args, kwargs) (line 147)
list_call_result_9587 = invoke(stypy.reporting.localization.Localization(__file__, 147, 45), list_9585, *[], **kwargs_9586)

# Processing the call keyword arguments (line 147)
kwargs_9588 = {}
# Getting the type of 'reversed' (line 147)
reversed_9584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 36), 'reversed', False)
# Calling reversed(args, kwargs) (line 147)
reversed_call_result_9589 = invoke(stypy.reporting.localization.Localization(__file__, 147, 36), reversed_9584, *[list_call_result_9587], **kwargs_9588)

# Processing the call keyword arguments (line 147)
kwargs_9590 = {}
# Getting the type of 'iter' (line 147)
iter_9583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 31), 'iter', False)
# Calling iter(args, kwargs) (line 147)
iter_call_result_9591 = invoke(stypy.reporting.localization.Localization(__file__, 147, 31), iter_9583, *[reversed_call_result_9589], **kwargs_9590)

# Processing the call keyword arguments (line 147)
kwargs_9592 = {}
# Getting the type of 'type' (line 147)
type_9582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 26), 'type', False)
# Calling type(args, kwargs) (line 147)
type_call_result_9593 = invoke(stypy.reporting.localization.Localization(__file__, 147, 26), type_9582, *[iter_call_result_9591], **kwargs_9592)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'listreverseiterator' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9594, 'listreverseiterator', type_call_result_9593)

# Assigning a Call to a Name (line 148):

# Call to type(...): (line 148)
# Processing the call arguments (line 148)

# Call to methodcaller(...): (line 148)
# Processing the call arguments (line 148)
int_9598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 46), 'int')
# Processing the call keyword arguments (line 148)
kwargs_9599 = {}
# Getting the type of 'operator' (line 148)
operator_9596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 24), 'operator', False)
# Obtaining the member 'methodcaller' of a type (line 148)
methodcaller_9597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 24), operator_9596, 'methodcaller')
# Calling methodcaller(args, kwargs) (line 148)
methodcaller_call_result_9600 = invoke(stypy.reporting.localization.Localization(__file__, 148, 24), methodcaller_9597, *[int_9598], **kwargs_9599)

# Processing the call keyword arguments (line 148)
kwargs_9601 = {}
# Getting the type of 'type' (line 148)
type_9595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 19), 'type', False)
# Calling type(args, kwargs) (line 148)
type_call_result_9602 = invoke(stypy.reporting.localization.Localization(__file__, 148, 19), type_9595, *[methodcaller_call_result_9600], **kwargs_9601)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'methodcaller' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9603, 'methodcaller', type_call_result_9602)

# Assigning a Call to a Name (line 149):

# Call to type(...): (line 149)
# Processing the call arguments (line 149)

# Call to itemgetter(...): (line 149)
# Processing the call arguments (line 149)
int_9607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 42), 'int')
# Processing the call keyword arguments (line 149)
kwargs_9608 = {}
# Getting the type of 'operator' (line 149)
operator_9605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 22), 'operator', False)
# Obtaining the member 'itemgetter' of a type (line 149)
itemgetter_9606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 22), operator_9605, 'itemgetter')
# Calling itemgetter(args, kwargs) (line 149)
itemgetter_call_result_9609 = invoke(stypy.reporting.localization.Localization(__file__, 149, 22), itemgetter_9606, *[int_9607], **kwargs_9608)

# Processing the call keyword arguments (line 149)
kwargs_9610 = {}
# Getting the type of 'type' (line 149)
type_9604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 17), 'type', False)
# Calling type(args, kwargs) (line 149)
type_call_result_9611 = invoke(stypy.reporting.localization.Localization(__file__, 149, 17), type_9604, *[itemgetter_call_result_9609], **kwargs_9610)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'itemgetter' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9612, 'itemgetter', type_call_result_9611)

# Assigning a Call to a Name (line 150):

# Call to type(...): (line 150)
# Processing the call arguments (line 150)

# Call to attrgetter(...): (line 150)
# Processing the call arguments (line 150)
int_9616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 42), 'int')
# Processing the call keyword arguments (line 150)
kwargs_9617 = {}
# Getting the type of 'operator' (line 150)
operator_9614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 22), 'operator', False)
# Obtaining the member 'attrgetter' of a type (line 150)
attrgetter_9615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 22), operator_9614, 'attrgetter')
# Calling attrgetter(args, kwargs) (line 150)
attrgetter_call_result_9618 = invoke(stypy.reporting.localization.Localization(__file__, 150, 22), attrgetter_9615, *[int_9616], **kwargs_9617)

# Processing the call keyword arguments (line 150)
kwargs_9619 = {}
# Getting the type of 'type' (line 150)
type_9613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 17), 'type', False)
# Calling type(args, kwargs) (line 150)
type_call_result_9620 = invoke(stypy.reporting.localization.Localization(__file__, 150, 17), type_9613, *[attrgetter_call_result_9618], **kwargs_9619)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'attrgetter' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9621, 'attrgetter', type_call_result_9620)

# Assigning a Call to a Name (line 152):

# Call to type(...): (line 152)
# Processing the call arguments (line 152)

# Call to viewitems(...): (line 152)
# Processing the call keyword arguments (line 152)
kwargs_9632 = {}

# Call to dict(...): (line 152)
# Processing the call arguments (line 152)

# Obtaining an instance of the builtin type 'dict' (line 152)
dict_9624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 27), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 152)
# Adding element type (key, value) (line 152)
str_9625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 28), 'str', 'a')
int_9626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 33), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 27), dict_9624, (str_9625, int_9626))
# Adding element type (key, value) (line 152)
str_9627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 36), 'str', 'b')
int_9628 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 41), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 27), dict_9624, (str_9627, int_9628))

# Processing the call keyword arguments (line 152)
kwargs_9629 = {}
# Getting the type of 'dict' (line 152)
dict_9623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 22), 'dict', False)
# Calling dict(args, kwargs) (line 152)
dict_call_result_9630 = invoke(stypy.reporting.localization.Localization(__file__, 152, 22), dict_9623, *[dict_9624], **kwargs_9629)

# Obtaining the member 'viewitems' of a type (line 152)
viewitems_9631 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 22), dict_call_result_9630, 'viewitems')
# Calling viewitems(args, kwargs) (line 152)
viewitems_call_result_9633 = invoke(stypy.reporting.localization.Localization(__file__, 152, 22), viewitems_9631, *[], **kwargs_9632)

# Processing the call keyword arguments (line 152)
kwargs_9634 = {}
# Getting the type of 'type' (line 152)
type_9622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 17), 'type', False)
# Calling type(args, kwargs) (line 152)
type_call_result_9635 = invoke(stypy.reporting.localization.Localization(__file__, 152, 17), type_9622, *[viewitems_call_result_9633], **kwargs_9634)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'dict_items' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9636, 'dict_items', type_call_result_9635)

# Assigning a Call to a Name (line 153):

# Call to type(...): (line 153)
# Processing the call arguments (line 153)

# Call to viewkeys(...): (line 153)
# Processing the call keyword arguments (line 153)
kwargs_9647 = {}

# Call to dict(...): (line 153)
# Processing the call arguments (line 153)

# Obtaining an instance of the builtin type 'dict' (line 153)
dict_9639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 26), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 153)
# Adding element type (key, value) (line 153)
str_9640 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 27), 'str', 'a')
int_9641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 32), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 26), dict_9639, (str_9640, int_9641))
# Adding element type (key, value) (line 153)
str_9642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 35), 'str', 'b')
int_9643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 40), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 153, 26), dict_9639, (str_9642, int_9643))

# Processing the call keyword arguments (line 153)
kwargs_9644 = {}
# Getting the type of 'dict' (line 153)
dict_9638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 21), 'dict', False)
# Calling dict(args, kwargs) (line 153)
dict_call_result_9645 = invoke(stypy.reporting.localization.Localization(__file__, 153, 21), dict_9638, *[dict_9639], **kwargs_9644)

# Obtaining the member 'viewkeys' of a type (line 153)
viewkeys_9646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 21), dict_call_result_9645, 'viewkeys')
# Calling viewkeys(args, kwargs) (line 153)
viewkeys_call_result_9648 = invoke(stypy.reporting.localization.Localization(__file__, 153, 21), viewkeys_9646, *[], **kwargs_9647)

# Processing the call keyword arguments (line 153)
kwargs_9649 = {}
# Getting the type of 'type' (line 153)
type_9637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 'type', False)
# Calling type(args, kwargs) (line 153)
type_call_result_9650 = invoke(stypy.reporting.localization.Localization(__file__, 153, 16), type_9637, *[viewkeys_call_result_9648], **kwargs_9649)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'dict_keys' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9651, 'dict_keys', type_call_result_9650)

# Assigning a Call to a Name (line 154):

# Call to type(...): (line 154)
# Processing the call arguments (line 154)

# Call to viewvalues(...): (line 154)
# Processing the call keyword arguments (line 154)
kwargs_9662 = {}

# Call to dict(...): (line 154)
# Processing the call arguments (line 154)

# Obtaining an instance of the builtin type 'dict' (line 154)
dict_9654 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 28), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 154)
# Adding element type (key, value) (line 154)
str_9655 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 29), 'str', 'a')
int_9656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 34), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 28), dict_9654, (str_9655, int_9656))
# Adding element type (key, value) (line 154)
str_9657 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 37), 'str', 'b')
int_9658 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 42), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 28), dict_9654, (str_9657, int_9658))

# Processing the call keyword arguments (line 154)
kwargs_9659 = {}
# Getting the type of 'dict' (line 154)
dict_9653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 23), 'dict', False)
# Calling dict(args, kwargs) (line 154)
dict_call_result_9660 = invoke(stypy.reporting.localization.Localization(__file__, 154, 23), dict_9653, *[dict_9654], **kwargs_9659)

# Obtaining the member 'viewvalues' of a type (line 154)
viewvalues_9661 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 23), dict_call_result_9660, 'viewvalues')
# Calling viewvalues(args, kwargs) (line 154)
viewvalues_call_result_9663 = invoke(stypy.reporting.localization.Localization(__file__, 154, 23), viewvalues_9661, *[], **kwargs_9662)

# Processing the call keyword arguments (line 154)
kwargs_9664 = {}
# Getting the type of 'type' (line 154)
type_9652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 18), 'type', False)
# Calling type(args, kwargs) (line 154)
type_call_result_9665 = invoke(stypy.reporting.localization.Localization(__file__, 154, 18), type_9652, *[viewvalues_call_result_9663], **kwargs_9664)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'dict_values' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9666, 'dict_values', type_call_result_9665)

# Assigning a Call to a Name (line 156):

# Call to type(...): (line 156)
# Processing the call arguments (line 156)

# Call to iter(...): (line 156)
# Processing the call arguments (line 156)

# Call to dict(...): (line 156)
# Processing the call arguments (line 156)

# Obtaining an instance of the builtin type 'dict' (line 156)
dict_9670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 44), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 156)
# Adding element type (key, value) (line 156)
str_9671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 45), 'str', 'a')
int_9672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 50), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 44), dict_9670, (str_9671, int_9672))
# Adding element type (key, value) (line 156)
str_9673 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 53), 'str', 'b')
int_9674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 58), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 44), dict_9670, (str_9673, int_9674))

# Processing the call keyword arguments (line 156)
kwargs_9675 = {}
# Getting the type of 'dict' (line 156)
dict_9669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 39), 'dict', False)
# Calling dict(args, kwargs) (line 156)
dict_call_result_9676 = invoke(stypy.reporting.localization.Localization(__file__, 156, 39), dict_9669, *[dict_9670], **kwargs_9675)

# Processing the call keyword arguments (line 156)
kwargs_9677 = {}
# Getting the type of 'iter' (line 156)
iter_9668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 34), 'iter', False)
# Calling iter(args, kwargs) (line 156)
iter_call_result_9678 = invoke(stypy.reporting.localization.Localization(__file__, 156, 34), iter_9668, *[dict_call_result_9676], **kwargs_9677)

# Processing the call keyword arguments (line 156)
kwargs_9679 = {}
# Getting the type of 'type' (line 156)
type_9667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 29), 'type', False)
# Calling type(args, kwargs) (line 156)
type_call_result_9680 = invoke(stypy.reporting.localization.Localization(__file__, 156, 29), type_9667, *[iter_call_result_9678], **kwargs_9679)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'dictionary_keyiterator' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9681, 'dictionary_keyiterator', type_call_result_9680)

# Assigning a Call to a Name (line 157):

# Call to type(...): (line 157)
# Processing the call arguments (line 157)

# Call to iteritems(...): (line 157)
# Processing the call keyword arguments (line 157)
kwargs_9692 = {}

# Call to dict(...): (line 157)
# Processing the call arguments (line 157)

# Obtaining an instance of the builtin type 'dict' (line 157)
dict_9684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 40), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 157)
# Adding element type (key, value) (line 157)
str_9685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 41), 'str', 'a')
int_9686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 46), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 40), dict_9684, (str_9685, int_9686))
# Adding element type (key, value) (line 157)
str_9687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 49), 'str', 'b')
int_9688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 54), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 157, 40), dict_9684, (str_9687, int_9688))

# Processing the call keyword arguments (line 157)
kwargs_9689 = {}
# Getting the type of 'dict' (line 157)
dict_9683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 35), 'dict', False)
# Calling dict(args, kwargs) (line 157)
dict_call_result_9690 = invoke(stypy.reporting.localization.Localization(__file__, 157, 35), dict_9683, *[dict_9684], **kwargs_9689)

# Obtaining the member 'iteritems' of a type (line 157)
iteritems_9691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 35), dict_call_result_9690, 'iteritems')
# Calling iteritems(args, kwargs) (line 157)
iteritems_call_result_9693 = invoke(stypy.reporting.localization.Localization(__file__, 157, 35), iteritems_9691, *[], **kwargs_9692)

# Processing the call keyword arguments (line 157)
kwargs_9694 = {}
# Getting the type of 'type' (line 157)
type_9682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 30), 'type', False)
# Calling type(args, kwargs) (line 157)
type_call_result_9695 = invoke(stypy.reporting.localization.Localization(__file__, 157, 30), type_9682, *[iteritems_call_result_9693], **kwargs_9694)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'dictionary_itemiterator' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9696, 'dictionary_itemiterator', type_call_result_9695)

# Assigning a Call to a Name (line 158):

# Call to type(...): (line 158)
# Processing the call arguments (line 158)

# Call to itervalues(...): (line 158)
# Processing the call keyword arguments (line 158)
kwargs_9707 = {}

# Call to dict(...): (line 158)
# Processing the call arguments (line 158)

# Obtaining an instance of the builtin type 'dict' (line 158)
dict_9699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 41), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 158)
# Adding element type (key, value) (line 158)
str_9700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 42), 'str', 'a')
int_9701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 47), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 41), dict_9699, (str_9700, int_9701))
# Adding element type (key, value) (line 158)
str_9702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 50), 'str', 'b')
int_9703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 55), 'int')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 41), dict_9699, (str_9702, int_9703))

# Processing the call keyword arguments (line 158)
kwargs_9704 = {}
# Getting the type of 'dict' (line 158)
dict_9698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 36), 'dict', False)
# Calling dict(args, kwargs) (line 158)
dict_call_result_9705 = invoke(stypy.reporting.localization.Localization(__file__, 158, 36), dict_9698, *[dict_9699], **kwargs_9704)

# Obtaining the member 'itervalues' of a type (line 158)
itervalues_9706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 36), dict_call_result_9705, 'itervalues')
# Calling itervalues(args, kwargs) (line 158)
itervalues_call_result_9708 = invoke(stypy.reporting.localization.Localization(__file__, 158, 36), itervalues_9706, *[], **kwargs_9707)

# Processing the call keyword arguments (line 158)
kwargs_9709 = {}
# Getting the type of 'type' (line 158)
type_9697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 31), 'type', False)
# Calling type(args, kwargs) (line 158)
type_call_result_9710 = invoke(stypy.reporting.localization.Localization(__file__, 158, 31), type_9697, *[itervalues_call_result_9708], **kwargs_9709)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'dictionary_valueiterator' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9711, 'dictionary_valueiterator', type_call_result_9710)

# Assigning a Call to a Name (line 159):

# Call to type(...): (line 159)
# Processing the call arguments (line 159)

# Call to iter(...): (line 159)
# Processing the call arguments (line 159)

# Call to bytearray(...): (line 159)
# Processing the call arguments (line 159)
str_9715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 45), 'str', 'test')
# Processing the call keyword arguments (line 159)
kwargs_9716 = {}
# Getting the type of 'bytearray' (line 159)
bytearray_9714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 35), 'bytearray', False)
# Calling bytearray(args, kwargs) (line 159)
bytearray_call_result_9717 = invoke(stypy.reporting.localization.Localization(__file__, 159, 35), bytearray_9714, *[str_9715], **kwargs_9716)

# Processing the call keyword arguments (line 159)
kwargs_9718 = {}
# Getting the type of 'iter' (line 159)
iter_9713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 30), 'iter', False)
# Calling iter(args, kwargs) (line 159)
iter_call_result_9719 = invoke(stypy.reporting.localization.Localization(__file__, 159, 30), iter_9713, *[bytearray_call_result_9717], **kwargs_9718)

# Processing the call keyword arguments (line 159)
kwargs_9720 = {}
# Getting the type of 'type' (line 159)
type_9712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 25), 'type', False)
# Calling type(args, kwargs) (line 159)
type_call_result_9721 = invoke(stypy.reporting.localization.Localization(__file__, 159, 25), type_9712, *[iter_call_result_9719], **kwargs_9720)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'bytearray_iterator' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9722, 'bytearray_iterator', type_call_result_9721)

# Assigning a Call to a Name (line 162):

# Call to type(...): (line 162)
# Processing the call arguments (line 162)
# Getting the type of 'ArithmeticError' (line 162)
ArithmeticError_9724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 29), 'ArithmeticError', False)
# Obtaining the member 'message' of a type (line 162)
message_9725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 29), ArithmeticError_9724, 'message')
# Processing the call keyword arguments (line 162)
kwargs_9726 = {}
# Getting the type of 'type' (line 162)
type_9723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 24), 'type', False)
# Calling type(args, kwargs) (line 162)
type_call_result_9727 = invoke(stypy.reporting.localization.Localization(__file__, 162, 24), type_9723, *[message_9725], **kwargs_9726)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'getset_descriptor' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9728, 'getset_descriptor', type_call_result_9727)

# Assigning a Call to a Name (line 163):

# Call to type(...): (line 163)
# Processing the call arguments (line 163)
# Getting the type of 'IOError' (line 163)
IOError_9730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 29), 'IOError', False)
# Obtaining the member 'errno' of a type (line 163)
errno_9731 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 29), IOError_9730, 'errno')
# Processing the call keyword arguments (line 163)
kwargs_9732 = {}
# Getting the type of 'type' (line 163)
type_9729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 24), 'type', False)
# Calling type(args, kwargs) (line 163)
type_call_result_9733 = invoke(stypy.reporting.localization.Localization(__file__, 163, 24), type_9729, *[errno_9731], **kwargs_9732)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'member_descriptor' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9734, 'member_descriptor', type_call_result_9733)

# Assigning a Call to a Name (line 164):

# Call to type(...): (line 164)
# Processing the call arguments (line 164)

# Call to _formatter_parser(...): (line 164)
# Processing the call keyword arguments (line 164)
kwargs_9738 = {}
unicode_9736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 29), 'unicode', u'foo')
# Obtaining the member '_formatter_parser' of a type (line 164)
_formatter_parser_9737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 29), unicode_9736, '_formatter_parser')
# Calling _formatter_parser(args, kwargs) (line 164)
_formatter_parser_call_result_9739 = invoke(stypy.reporting.localization.Localization(__file__, 164, 29), _formatter_parser_9737, *[], **kwargs_9738)

# Processing the call keyword arguments (line 164)
kwargs_9740 = {}
# Getting the type of 'type' (line 164)
type_9735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 24), 'type', False)
# Calling type(args, kwargs) (line 164)
type_call_result_9741 = invoke(stypy.reporting.localization.Localization(__file__, 164, 24), type_9735, *[_formatter_parser_call_result_9739], **kwargs_9740)

# Getting the type of 'ExtraTypeDefinitions'
ExtraTypeDefinitions_9742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'ExtraTypeDefinitions')
# Setting the type of the member 'formatteriterator' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), ExtraTypeDefinitions_9742, 'formatteriterator', type_call_result_9741)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
