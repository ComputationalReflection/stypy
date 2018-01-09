
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: class UndefinedType():
2:     '''
3:     The type of an undefined variable
4:     '''
5: 
6:     def __str__(self):
7:         return 'Undefined'
8: 
9:     def __repr__(self):
10:         return self.__str__()
11: 
12:     def __eq__(self, other):
13:         return isinstance(other, UndefinedType)
14: 
15: 
16: class DynamicType():
17:     '''
18:     Any type (type cannot be statically calculated)
19:     '''
20: 
21:     def __init__(self, *members):
22:         TypeGroup.__init__(self, [])
23:         self.members = members
24: 
25:     def __eq__(self, type_):
26:         return True
27: 
28: 
29: class BaseTypeGroup(object):
30:     '''
31:     All type groups inherit from this class
32:     '''
33: 
34:     def __str__(self):
35:         return self.__repr__()
36: 
37: 
38: class TypeGroup(BaseTypeGroup):
39:     '''
40:     A TypeGroup is an entity used in the rule files to group several Python types over a named identity. Type groups
41:     are collections of types that have something in common, and Python functions and methods usually admits any of them
42:     as a parameter when one of them is valid. For example, if a Python library function works with an int as the first
43:     parameter, we can also use bool and long as the first parameter without runtime errors. This is for exameple a
44:     TypeGroup that will be called Integer
45: 
46:     Not all type groups are defined by collections of Python concrete types. Other groups identify Python objects with
47:     a common member or structure (Iterable, Overloads__str__ identify any Python object that is iterable and any Python
48:     object that has defined the __str__ method properly) or even class relationships (SubtypeOf type group only matches
49:     with classes that are a subtype of the one specified.
50: 
51:     Type groups are the workhorse of the type rule specification mechanism and have a great expressiveness and
52:     flexibility to specify admitted types in any Python callable entity.
53: 
54:     Type groups are created in the file type_groups.py
55:     '''
56: 
57:     def __init__(self, grouped_types):
58:         '''
59:         Create a new type group that represent the list of types passed as a parameter
60:         :param grouped_types: List of types that are included inside this type group
61:         :return:
62:         '''
63:         self.grouped_types = grouped_types
64: 
65:     def __contains__(self, type_):
66:         '''
67:         Test if this type group contains the specified type (in operator)
68:         :param type_: Type to test
69:         :return: bool
70:         '''
71:         # if hasattr(type_, "get_python_type"):
72:         #     return type_.get_python_type() in self.grouped_types
73:         #
74:         # return type_ in self.grouped_types
75:         try:
76:             return type_.get_python_type() in self.grouped_types
77:         except:
78:             return type_ in self.grouped_types
79: 
80:     def __eq__(self, type_):
81:         '''
82:         Test if this type group contains the specified type (== operator)
83:         :param type_: Type to test
84:         :return: bool
85:         '''
86:         # if hasattr(type_, "get_python_type"):
87:         #     return type_.get_python_type() in self.grouped_types
88:         # return type_ in self.grouped_types
89:         try:
90:             cond1 = type(type_) in self.grouped_types
91: 
92:             return cond1
93:         except:
94:             return type_ in self.grouped_types
95: 
96:     def __cmp__(self, type_):
97:         '''
98:         Test if this type group contains the specified type (compatarion operators)
99:         :param type_: Type to test
100:         :return: bool
101:         '''
102:         # if hasattr(type_, "get_python_type"):
103:         #     return type_.get_python_type() in self.grouped_types
104:         #
105:         # return type_ in self.grouped_types
106:         try:
107:             # return type_.get_python_type() in self.grouped_types
108:             cond1 = type(type_) in self.grouped_types
109: 
110:             return cond1
111:         except:
112:             return type_ in self.grouped_types
113: 
114:     def __gt__(self, other):
115:         '''
116:         Type group sorting. A type group is less than other type group if contains less types or the types contained
117:         in the type group are all contained in the other one. Otherwise, is greater than the other type group.
118:         :param other: Another type group
119:         :return: bool
120:         '''
121:         if len(self.grouped_types) < len(other.grouped_types):
122:             return False
123: 
124:         for type_ in self.grouped_types:
125:             if type_ not in other.grouped_types:
126:                 return False
127: 
128:         return True
129: 
130:     def __lt__(self, other):
131:         '''
132:         Type group sorting. A type group is less than other type group if contains less types or the types contained
133:         in the type group are all contained in the other one. Otherwise, is greater than the other type group.
134:         :param other: Another type group
135:         :return: bool
136:         '''
137:         if len(self.grouped_types) > len(other.grouped_types):
138:             return False
139: 
140:         for type_ in self.grouped_types:
141:             if type_ not in other.grouped_types:
142:                 return False
143: 
144:         return True
145: 
146:     def __repr__(self):
147:         '''
148:         Textual representation of the type group
149:         :return: str
150:         '''
151:         # ret_str = type(self).__name__  + "("
152:         # for type_ in self.grouped_types:
153:         #     if hasattr(type_, '__name__'):
154:         #         ret_str += type_.__name__ + ", "
155:         #     else:
156:         #         ret_str += str(type_) + ", "
157:         #
158:         # ret_str = ret_str[:-2]
159:         # ret_str+=")"
160: 
161:         ret_str = type(self).__name__
162:         return ret_str
163: 
164: 
165: class DependentType:
166:     '''
167:     A DependentType is a special base class that indicates that a type group has to be called to obtain the real
168:     type it represent. Call is done using the parameters that are trying to match the rule. For example, imagine that
169:     we call the + operator with an object that defines the __add__ method and another type to add to. With an object
170:     that defines an __add__ method we don't really know what will be the result of calling __add__ over this object
171:     with the second parameter, so the __add__ method has to be called (well, in fact, the type inference equivalent
172:     version of the __add__ method will be called) to obtain the real return type.
173: 
174:     Dependent types are a powerful mechanism to calculate the return type of operations that depend on calls to
175:     certain object members or even to detect incorrect definitions of members in objects (__int__ method defined in
176:     object that do not return int, for example).
177:     '''
178: 
179:     def __init__(self, report_errors=False):
180:         '''
181:         Build a Dependent type instance
182:         :param report_errors: Flag to indicate if errors found when calling this type will be reported or not (in that
183:         case other code will do it)
184:         '''
185:         self.report_errors = report_errors
186:         self.call_arity = 0
187: 
188:     def __call__(self, *call_args, **call_kwargs):
189:         '''
190:         Call the dependent type. Empty in this implementation, concrete calls must be defined in subclasses
191:         '''
192:         pass
193: 
194: 
195: class HasMember(TypeGroup, DependentType):
196:     '''
197:         Type of any object that has a member with the specified arity, and that can be called with the corresponding
198:         params.
199:     '''
200: 
201:     def __init__(self, member, expected_return_type, call_arity=0, report_errors=False):
202:         DependentType.__init__(self, report_errors)
203:         TypeGroup.__init__(self, [])
204:         self.member = member
205:         self.expected_return_type = expected_return_type
206:         self.member_obj = None
207:         self.call_arity = call_arity
208: 
209:     def format_arity(self):
210:         str_ = "("
211:         for i in range(self.call_arity):
212:             str_ += "parameter" + str(i) + ", "
213: 
214:         if self.call_arity > 0:
215:             str_ = str_[:-2]
216: 
217:         return str_ + ")"
218: 
219:     def __eq__(self, type_):
220:         self.member_obj = type_.get_type_of_member(None, self.member)
221:         if isinstance(self.member_obj, TypeError):
222:             if not self.report_errors:
223:                 TypeError.remove_error_msg(self.member_obj)
224:             return False
225: 
226:         return True
227: 
228:     def __call__(self, localization, *call_args, **call_kwargs):
229:         if callable(self.member_obj.get_python_type()):
230:             # Call the member
231:             equivalent_type = self.member_obj.invoke(localization, *call_args, **call_kwargs)
232: 
233:             # Call was impossible: Invokation error has to be removed because we provide a general one later
234:             if isinstance(equivalent_type, TypeError):
235:                 if not self.report_errors:
236:                     TypeError.remove_error_msg(equivalent_type)
237:                 self.member_obj = None
238:                 return False, equivalent_type
239: 
240:             # Call was possible, but the expected return type cannot be predetermined (we have to recheck it later)
241:             if isinstance(self.expected_return_type, UndefinedType):
242:                 self.member_obj = None
243:                 return True, equivalent_type
244: 
245:             # Call was possible, but the expected return type is Any)
246:             if self.expected_return_type is DynamicType:
247:                 self.member_obj = None
248:                 return True, equivalent_type
249: 
250:             # Call was possible, so we check if the predetermined return type is the same that the one that is returned
251:             if not issubclass(equivalent_type.get_python_type(), self.expected_return_type):
252:                 self.member_obj = None
253:                 return False, equivalent_type
254:             else:
255:                 return True, equivalent_type
256: 
257:         self.member_obj = None
258:         return True, None
259: 
260:     def __repr__(self):
261:         ret_str = "Instance defining "
262:         ret_str += str(self.member)
263:         ret_str += self.format_arity()
264:         return ret_str
265: 
266: 
267: CastsToInt = HasMember("__int__", int, 0)
268: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

# Declaration of the 'UndefinedType' class

class UndefinedType:
    str_1887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', '\n    The type of an undefined variable\n    ')

    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 6, 4, False)
        # Assigning a type to the variable 'self' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UndefinedType.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        UndefinedType.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UndefinedType.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        UndefinedType.stypy__str__.__dict__.__setitem__('stypy_function_name', 'UndefinedType.__str__')
        UndefinedType.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        UndefinedType.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        UndefinedType.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UndefinedType.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        UndefinedType.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        UndefinedType.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UndefinedType.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UndefinedType.__str__', [], None, None, defaults, varargs, kwargs)

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

        str_1888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 15), 'str', 'Undefined')
        # Assigning a type to the variable 'stypy_return_type' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 8), 'stypy_return_type', str_1888)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 6)
        stypy_return_type_1889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1889)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_1889


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 9, 4, False)
        # Assigning a type to the variable 'self' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UndefinedType.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        UndefinedType.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UndefinedType.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        UndefinedType.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'UndefinedType.__repr__')
        UndefinedType.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        UndefinedType.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        UndefinedType.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UndefinedType.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        UndefinedType.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        UndefinedType.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UndefinedType.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UndefinedType.__repr__', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to __str__(...): (line 10)
        # Processing the call keyword arguments (line 10)
        kwargs_1892 = {}
        # Getting the type of 'self' (line 10)
        self_1890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 15), 'self', False)
        # Obtaining the member '__str__' of a type (line 10)
        str___1891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 15), self_1890, '__str__')
        # Calling __str__(args, kwargs) (line 10)
        str___call_result_1893 = invoke(stypy.reporting.localization.Localization(__file__, 10, 15), str___1891, *[], **kwargs_1892)
        
        # Assigning a type to the variable 'stypy_return_type' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'stypy_return_type', str___call_result_1893)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 9)
        stypy_return_type_1894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1894)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_1894


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 12, 4, False)
        # Assigning a type to the variable 'self' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UndefinedType.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        UndefinedType.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UndefinedType.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        UndefinedType.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'UndefinedType.__eq__')
        UndefinedType.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        UndefinedType.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        UndefinedType.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UndefinedType.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        UndefinedType.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        UndefinedType.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UndefinedType.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UndefinedType.__eq__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__eq__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__eq__(...)' code ##################

        
        # Call to isinstance(...): (line 13)
        # Processing the call arguments (line 13)
        # Getting the type of 'other' (line 13)
        other_1896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 26), 'other', False)
        # Getting the type of 'UndefinedType' (line 13)
        UndefinedType_1897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 33), 'UndefinedType', False)
        # Processing the call keyword arguments (line 13)
        kwargs_1898 = {}
        # Getting the type of 'isinstance' (line 13)
        isinstance_1895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 13)
        isinstance_call_result_1899 = invoke(stypy.reporting.localization.Localization(__file__, 13, 15), isinstance_1895, *[other_1896, UndefinedType_1897], **kwargs_1898)
        
        # Assigning a type to the variable 'stypy_return_type' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'stypy_return_type', isinstance_call_result_1899)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 12)
        stypy_return_type_1900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1900)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_1900


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 1, 0, False)
        # Assigning a type to the variable 'self' (line 2)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UndefinedType.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'UndefinedType' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'UndefinedType', UndefinedType)
# Declaration of the 'DynamicType' class

class DynamicType:
    str_1901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, (-1)), 'str', '\n    Any type (type cannot be statically calculated)\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 21, 4, False)
        # Assigning a type to the variable 'self' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DynamicType.__init__', [], 'members', None, defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 22)
        # Processing the call arguments (line 22)
        # Getting the type of 'self' (line 22)
        self_1904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 27), 'self', False)
        
        # Obtaining an instance of the builtin type 'list' (line 22)
        list_1905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 22)
        
        # Processing the call keyword arguments (line 22)
        kwargs_1906 = {}
        # Getting the type of 'TypeGroup' (line 22)
        TypeGroup_1902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'TypeGroup', False)
        # Obtaining the member '__init__' of a type (line 22)
        init___1903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), TypeGroup_1902, '__init__')
        # Calling __init__(args, kwargs) (line 22)
        init___call_result_1907 = invoke(stypy.reporting.localization.Localization(__file__, 22, 8), init___1903, *[self_1904, list_1905], **kwargs_1906)
        
        
        # Assigning a Name to a Attribute (line 23):
        # Getting the type of 'members' (line 23)
        members_1908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 23), 'members')
        # Getting the type of 'self' (line 23)
        self_1909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'self')
        # Setting the type of the member 'members' of a type (line 23)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), self_1909, 'members', members_1908)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 25, 4, False)
        # Assigning a type to the variable 'self' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DynamicType.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        DynamicType.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DynamicType.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        DynamicType.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'DynamicType.__eq__')
        DynamicType.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['type_'])
        DynamicType.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        DynamicType.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DynamicType.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        DynamicType.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        DynamicType.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DynamicType.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DynamicType.__eq__', ['type_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__eq__', localization, ['type_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__eq__(...)' code ##################

        # Getting the type of 'True' (line 26)
        True_1910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'stypy_return_type', True_1910)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 25)
        stypy_return_type_1911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1911)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_1911


# Assigning a type to the variable 'DynamicType' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'DynamicType', DynamicType)
# Declaration of the 'BaseTypeGroup' class

class BaseTypeGroup(object, ):
    str_1912 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, (-1)), 'str', '\n    All type groups inherit from this class\n    ')

    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 34, 4, False)
        # Assigning a type to the variable 'self' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseTypeGroup.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        BaseTypeGroup.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseTypeGroup.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseTypeGroup.stypy__str__.__dict__.__setitem__('stypy_function_name', 'BaseTypeGroup.__str__')
        BaseTypeGroup.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        BaseTypeGroup.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseTypeGroup.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseTypeGroup.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseTypeGroup.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseTypeGroup.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseTypeGroup.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseTypeGroup.__str__', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to __repr__(...): (line 35)
        # Processing the call keyword arguments (line 35)
        kwargs_1915 = {}
        # Getting the type of 'self' (line 35)
        self_1913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 15), 'self', False)
        # Obtaining the member '__repr__' of a type (line 35)
        repr___1914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 15), self_1913, '__repr__')
        # Calling __repr__(args, kwargs) (line 35)
        repr___call_result_1916 = invoke(stypy.reporting.localization.Localization(__file__, 35, 15), repr___1914, *[], **kwargs_1915)
        
        # Assigning a type to the variable 'stypy_return_type' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'stypy_return_type', repr___call_result_1916)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 34)
        stypy_return_type_1917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1917)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_1917


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 29, 0, False)
        # Assigning a type to the variable 'self' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseTypeGroup.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'BaseTypeGroup' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'BaseTypeGroup', BaseTypeGroup)
# Declaration of the 'TypeGroup' class
# Getting the type of 'BaseTypeGroup' (line 38)
BaseTypeGroup_1918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'BaseTypeGroup')

class TypeGroup(BaseTypeGroup_1918, ):
    str_1919 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, (-1)), 'str', '\n    A TypeGroup is an entity used in the rule files to group several Python types over a named identity. Type groups\n    are collections of types that have something in common, and Python functions and methods usually admits any of them\n    as a parameter when one of them is valid. For example, if a Python library function works with an int as the first\n    parameter, we can also use bool and long as the first parameter without runtime errors. This is for exameple a\n    TypeGroup that will be called Integer\n\n    Not all type groups are defined by collections of Python concrete types. Other groups identify Python objects with\n    a common member or structure (Iterable, Overloads__str__ identify any Python object that is iterable and any Python\n    object that has defined the __str__ method properly) or even class relationships (SubtypeOf type group only matches\n    with classes that are a subtype of the one specified.\n\n    Type groups are the workhorse of the type rule specification mechanism and have a great expressiveness and\n    flexibility to specify admitted types in any Python callable entity.\n\n    Type groups are created in the file type_groups.py\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 57, 4, False)
        # Assigning a type to the variable 'self' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeGroup.__init__', ['grouped_types'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['grouped_types'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_1920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, (-1)), 'str', '\n        Create a new type group that represent the list of types passed as a parameter\n        :param grouped_types: List of types that are included inside this type group\n        :return:\n        ')
        
        # Assigning a Name to a Attribute (line 63):
        # Getting the type of 'grouped_types' (line 63)
        grouped_types_1921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 29), 'grouped_types')
        # Getting the type of 'self' (line 63)
        self_1922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'self')
        # Setting the type of the member 'grouped_types' of a type (line 63)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), self_1922, 'grouped_types', grouped_types_1921)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __contains__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__contains__'
        module_type_store = module_type_store.open_function_context('__contains__', 65, 4, False)
        # Assigning a type to the variable 'self' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeGroup.__contains__.__dict__.__setitem__('stypy_localization', localization)
        TypeGroup.__contains__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeGroup.__contains__.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeGroup.__contains__.__dict__.__setitem__('stypy_function_name', 'TypeGroup.__contains__')
        TypeGroup.__contains__.__dict__.__setitem__('stypy_param_names_list', ['type_'])
        TypeGroup.__contains__.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeGroup.__contains__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeGroup.__contains__.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeGroup.__contains__.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeGroup.__contains__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeGroup.__contains__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeGroup.__contains__', ['type_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__contains__', localization, ['type_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__contains__(...)' code ##################

        str_1923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, (-1)), 'str', '\n        Test if this type group contains the specified type (in operator)\n        :param type_: Type to test\n        :return: bool\n        ')
        
        
        # SSA begins for try-except statement (line 75)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        
        # Call to get_python_type(...): (line 76)
        # Processing the call keyword arguments (line 76)
        kwargs_1926 = {}
        # Getting the type of 'type_' (line 76)
        type__1924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 19), 'type_', False)
        # Obtaining the member 'get_python_type' of a type (line 76)
        get_python_type_1925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 19), type__1924, 'get_python_type')
        # Calling get_python_type(args, kwargs) (line 76)
        get_python_type_call_result_1927 = invoke(stypy.reporting.localization.Localization(__file__, 76, 19), get_python_type_1925, *[], **kwargs_1926)
        
        # Getting the type of 'self' (line 76)
        self_1928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 46), 'self')
        # Obtaining the member 'grouped_types' of a type (line 76)
        grouped_types_1929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 46), self_1928, 'grouped_types')
        # Applying the binary operator 'in' (line 76)
        result_contains_1930 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 19), 'in', get_python_type_call_result_1927, grouped_types_1929)
        
        # Assigning a type to the variable 'stypy_return_type' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'stypy_return_type', result_contains_1930)
        # SSA branch for the except part of a try statement (line 75)
        # SSA branch for the except '<any exception>' branch of a try statement (line 75)
        module_type_store.open_ssa_branch('except')
        
        # Getting the type of 'type_' (line 78)
        type__1931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 19), 'type_')
        # Getting the type of 'self' (line 78)
        self_1932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 28), 'self')
        # Obtaining the member 'grouped_types' of a type (line 78)
        grouped_types_1933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 28), self_1932, 'grouped_types')
        # Applying the binary operator 'in' (line 78)
        result_contains_1934 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 19), 'in', type__1931, grouped_types_1933)
        
        # Assigning a type to the variable 'stypy_return_type' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'stypy_return_type', result_contains_1934)
        # SSA join for try-except statement (line 75)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__contains__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__contains__' in the type store
        # Getting the type of 'stypy_return_type' (line 65)
        stypy_return_type_1935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1935)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__contains__'
        return stypy_return_type_1935


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 80, 4, False)
        # Assigning a type to the variable 'self' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeGroup.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        TypeGroup.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeGroup.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeGroup.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'TypeGroup.__eq__')
        TypeGroup.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['type_'])
        TypeGroup.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeGroup.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeGroup.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeGroup.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeGroup.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeGroup.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeGroup.__eq__', ['type_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__eq__', localization, ['type_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__eq__(...)' code ##################

        str_1936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, (-1)), 'str', '\n        Test if this type group contains the specified type (== operator)\n        :param type_: Type to test\n        :return: bool\n        ')
        
        
        # SSA begins for try-except statement (line 89)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Compare to a Name (line 90):
        
        
        # Call to type(...): (line 90)
        # Processing the call arguments (line 90)
        # Getting the type of 'type_' (line 90)
        type__1938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 25), 'type_', False)
        # Processing the call keyword arguments (line 90)
        kwargs_1939 = {}
        # Getting the type of 'type' (line 90)
        type_1937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 20), 'type', False)
        # Calling type(args, kwargs) (line 90)
        type_call_result_1940 = invoke(stypy.reporting.localization.Localization(__file__, 90, 20), type_1937, *[type__1938], **kwargs_1939)
        
        # Getting the type of 'self' (line 90)
        self_1941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 35), 'self')
        # Obtaining the member 'grouped_types' of a type (line 90)
        grouped_types_1942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 35), self_1941, 'grouped_types')
        # Applying the binary operator 'in' (line 90)
        result_contains_1943 = python_operator(stypy.reporting.localization.Localization(__file__, 90, 20), 'in', type_call_result_1940, grouped_types_1942)
        
        # Assigning a type to the variable 'cond1' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'cond1', result_contains_1943)
        # Getting the type of 'cond1' (line 92)
        cond1_1944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 19), 'cond1')
        # Assigning a type to the variable 'stypy_return_type' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 12), 'stypy_return_type', cond1_1944)
        # SSA branch for the except part of a try statement (line 89)
        # SSA branch for the except '<any exception>' branch of a try statement (line 89)
        module_type_store.open_ssa_branch('except')
        
        # Getting the type of 'type_' (line 94)
        type__1945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 19), 'type_')
        # Getting the type of 'self' (line 94)
        self_1946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 28), 'self')
        # Obtaining the member 'grouped_types' of a type (line 94)
        grouped_types_1947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 28), self_1946, 'grouped_types')
        # Applying the binary operator 'in' (line 94)
        result_contains_1948 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 19), 'in', type__1945, grouped_types_1947)
        
        # Assigning a type to the variable 'stypy_return_type' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'stypy_return_type', result_contains_1948)
        # SSA join for try-except statement (line 89)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 80)
        stypy_return_type_1949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1949)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_1949


    @norecursion
    def stypy__cmp__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__cmp__'
        module_type_store = module_type_store.open_function_context('__cmp__', 96, 4, False)
        # Assigning a type to the variable 'self' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeGroup.stypy__cmp__.__dict__.__setitem__('stypy_localization', localization)
        TypeGroup.stypy__cmp__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeGroup.stypy__cmp__.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeGroup.stypy__cmp__.__dict__.__setitem__('stypy_function_name', 'TypeGroup.__cmp__')
        TypeGroup.stypy__cmp__.__dict__.__setitem__('stypy_param_names_list', ['type_'])
        TypeGroup.stypy__cmp__.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeGroup.stypy__cmp__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeGroup.stypy__cmp__.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeGroup.stypy__cmp__.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeGroup.stypy__cmp__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeGroup.stypy__cmp__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeGroup.__cmp__', ['type_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__cmp__', localization, ['type_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__cmp__(...)' code ##################

        str_1950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, (-1)), 'str', '\n        Test if this type group contains the specified type (compatarion operators)\n        :param type_: Type to test\n        :return: bool\n        ')
        
        
        # SSA begins for try-except statement (line 106)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Compare to a Name (line 108):
        
        
        # Call to type(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'type_' (line 108)
        type__1952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 25), 'type_', False)
        # Processing the call keyword arguments (line 108)
        kwargs_1953 = {}
        # Getting the type of 'type' (line 108)
        type_1951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 20), 'type', False)
        # Calling type(args, kwargs) (line 108)
        type_call_result_1954 = invoke(stypy.reporting.localization.Localization(__file__, 108, 20), type_1951, *[type__1952], **kwargs_1953)
        
        # Getting the type of 'self' (line 108)
        self_1955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 35), 'self')
        # Obtaining the member 'grouped_types' of a type (line 108)
        grouped_types_1956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 35), self_1955, 'grouped_types')
        # Applying the binary operator 'in' (line 108)
        result_contains_1957 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 20), 'in', type_call_result_1954, grouped_types_1956)
        
        # Assigning a type to the variable 'cond1' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'cond1', result_contains_1957)
        # Getting the type of 'cond1' (line 110)
        cond1_1958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 19), 'cond1')
        # Assigning a type to the variable 'stypy_return_type' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'stypy_return_type', cond1_1958)
        # SSA branch for the except part of a try statement (line 106)
        # SSA branch for the except '<any exception>' branch of a try statement (line 106)
        module_type_store.open_ssa_branch('except')
        
        # Getting the type of 'type_' (line 112)
        type__1959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 19), 'type_')
        # Getting the type of 'self' (line 112)
        self_1960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 28), 'self')
        # Obtaining the member 'grouped_types' of a type (line 112)
        grouped_types_1961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 28), self_1960, 'grouped_types')
        # Applying the binary operator 'in' (line 112)
        result_contains_1962 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 19), 'in', type__1959, grouped_types_1961)
        
        # Assigning a type to the variable 'stypy_return_type' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'stypy_return_type', result_contains_1962)
        # SSA join for try-except statement (line 106)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__cmp__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__cmp__' in the type store
        # Getting the type of 'stypy_return_type' (line 96)
        stypy_return_type_1963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1963)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__cmp__'
        return stypy_return_type_1963


    @norecursion
    def __gt__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__gt__'
        module_type_store = module_type_store.open_function_context('__gt__', 114, 4, False)
        # Assigning a type to the variable 'self' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeGroup.__gt__.__dict__.__setitem__('stypy_localization', localization)
        TypeGroup.__gt__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeGroup.__gt__.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeGroup.__gt__.__dict__.__setitem__('stypy_function_name', 'TypeGroup.__gt__')
        TypeGroup.__gt__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        TypeGroup.__gt__.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeGroup.__gt__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeGroup.__gt__.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeGroup.__gt__.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeGroup.__gt__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeGroup.__gt__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeGroup.__gt__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__gt__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__gt__(...)' code ##################

        str_1964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, (-1)), 'str', '\n        Type group sorting. A type group is less than other type group if contains less types or the types contained\n        in the type group are all contained in the other one. Otherwise, is greater than the other type group.\n        :param other: Another type group\n        :return: bool\n        ')
        
        
        
        # Call to len(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'self' (line 121)
        self_1966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 15), 'self', False)
        # Obtaining the member 'grouped_types' of a type (line 121)
        grouped_types_1967 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 15), self_1966, 'grouped_types')
        # Processing the call keyword arguments (line 121)
        kwargs_1968 = {}
        # Getting the type of 'len' (line 121)
        len_1965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 11), 'len', False)
        # Calling len(args, kwargs) (line 121)
        len_call_result_1969 = invoke(stypy.reporting.localization.Localization(__file__, 121, 11), len_1965, *[grouped_types_1967], **kwargs_1968)
        
        
        # Call to len(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'other' (line 121)
        other_1971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 41), 'other', False)
        # Obtaining the member 'grouped_types' of a type (line 121)
        grouped_types_1972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 41), other_1971, 'grouped_types')
        # Processing the call keyword arguments (line 121)
        kwargs_1973 = {}
        # Getting the type of 'len' (line 121)
        len_1970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 37), 'len', False)
        # Calling len(args, kwargs) (line 121)
        len_call_result_1974 = invoke(stypy.reporting.localization.Localization(__file__, 121, 37), len_1970, *[grouped_types_1972], **kwargs_1973)
        
        # Applying the binary operator '<' (line 121)
        result_lt_1975 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 11), '<', len_call_result_1969, len_call_result_1974)
        
        # Testing the type of an if condition (line 121)
        if_condition_1976 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 121, 8), result_lt_1975)
        # Assigning a type to the variable 'if_condition_1976' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'if_condition_1976', if_condition_1976)
        # SSA begins for if statement (line 121)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 122)
        False_1977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 19), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'stypy_return_type', False_1977)
        # SSA join for if statement (line 121)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 124)
        self_1978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 21), 'self')
        # Obtaining the member 'grouped_types' of a type (line 124)
        grouped_types_1979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 21), self_1978, 'grouped_types')
        # Testing the type of a for loop iterable (line 124)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 124, 8), grouped_types_1979)
        # Getting the type of the for loop variable (line 124)
        for_loop_var_1980 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 124, 8), grouped_types_1979)
        # Assigning a type to the variable 'type_' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'type_', for_loop_var_1980)
        # SSA begins for a for statement (line 124)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'type_' (line 125)
        type__1981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 15), 'type_')
        # Getting the type of 'other' (line 125)
        other_1982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 28), 'other')
        # Obtaining the member 'grouped_types' of a type (line 125)
        grouped_types_1983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 28), other_1982, 'grouped_types')
        # Applying the binary operator 'notin' (line 125)
        result_contains_1984 = python_operator(stypy.reporting.localization.Localization(__file__, 125, 15), 'notin', type__1981, grouped_types_1983)
        
        # Testing the type of an if condition (line 125)
        if_condition_1985 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 125, 12), result_contains_1984)
        # Assigning a type to the variable 'if_condition_1985' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'if_condition_1985', if_condition_1985)
        # SSA begins for if statement (line 125)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 126)
        False_1986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 23), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'stypy_return_type', False_1986)
        # SSA join for if statement (line 125)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'True' (line 128)
        True_1987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'stypy_return_type', True_1987)
        
        # ################# End of '__gt__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__gt__' in the type store
        # Getting the type of 'stypy_return_type' (line 114)
        stypy_return_type_1988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1988)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__gt__'
        return stypy_return_type_1988


    @norecursion
    def __lt__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__lt__'
        module_type_store = module_type_store.open_function_context('__lt__', 130, 4, False)
        # Assigning a type to the variable 'self' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeGroup.__lt__.__dict__.__setitem__('stypy_localization', localization)
        TypeGroup.__lt__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeGroup.__lt__.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeGroup.__lt__.__dict__.__setitem__('stypy_function_name', 'TypeGroup.__lt__')
        TypeGroup.__lt__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        TypeGroup.__lt__.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeGroup.__lt__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeGroup.__lt__.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeGroup.__lt__.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeGroup.__lt__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeGroup.__lt__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeGroup.__lt__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__lt__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__lt__(...)' code ##################

        str_1989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, (-1)), 'str', '\n        Type group sorting. A type group is less than other type group if contains less types or the types contained\n        in the type group are all contained in the other one. Otherwise, is greater than the other type group.\n        :param other: Another type group\n        :return: bool\n        ')
        
        
        
        # Call to len(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'self' (line 137)
        self_1991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 15), 'self', False)
        # Obtaining the member 'grouped_types' of a type (line 137)
        grouped_types_1992 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 15), self_1991, 'grouped_types')
        # Processing the call keyword arguments (line 137)
        kwargs_1993 = {}
        # Getting the type of 'len' (line 137)
        len_1990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 11), 'len', False)
        # Calling len(args, kwargs) (line 137)
        len_call_result_1994 = invoke(stypy.reporting.localization.Localization(__file__, 137, 11), len_1990, *[grouped_types_1992], **kwargs_1993)
        
        
        # Call to len(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'other' (line 137)
        other_1996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 41), 'other', False)
        # Obtaining the member 'grouped_types' of a type (line 137)
        grouped_types_1997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 41), other_1996, 'grouped_types')
        # Processing the call keyword arguments (line 137)
        kwargs_1998 = {}
        # Getting the type of 'len' (line 137)
        len_1995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 37), 'len', False)
        # Calling len(args, kwargs) (line 137)
        len_call_result_1999 = invoke(stypy.reporting.localization.Localization(__file__, 137, 37), len_1995, *[grouped_types_1997], **kwargs_1998)
        
        # Applying the binary operator '>' (line 137)
        result_gt_2000 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 11), '>', len_call_result_1994, len_call_result_1999)
        
        # Testing the type of an if condition (line 137)
        if_condition_2001 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 137, 8), result_gt_2000)
        # Assigning a type to the variable 'if_condition_2001' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'if_condition_2001', if_condition_2001)
        # SSA begins for if statement (line 137)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 138)
        False_2002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 19), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'stypy_return_type', False_2002)
        # SSA join for if statement (line 137)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 140)
        self_2003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 21), 'self')
        # Obtaining the member 'grouped_types' of a type (line 140)
        grouped_types_2004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 21), self_2003, 'grouped_types')
        # Testing the type of a for loop iterable (line 140)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 140, 8), grouped_types_2004)
        # Getting the type of the for loop variable (line 140)
        for_loop_var_2005 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 140, 8), grouped_types_2004)
        # Assigning a type to the variable 'type_' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'type_', for_loop_var_2005)
        # SSA begins for a for statement (line 140)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'type_' (line 141)
        type__2006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 15), 'type_')
        # Getting the type of 'other' (line 141)
        other_2007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 28), 'other')
        # Obtaining the member 'grouped_types' of a type (line 141)
        grouped_types_2008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 28), other_2007, 'grouped_types')
        # Applying the binary operator 'notin' (line 141)
        result_contains_2009 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 15), 'notin', type__2006, grouped_types_2008)
        
        # Testing the type of an if condition (line 141)
        if_condition_2010 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 141, 12), result_contains_2009)
        # Assigning a type to the variable 'if_condition_2010' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 12), 'if_condition_2010', if_condition_2010)
        # SSA begins for if statement (line 141)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'False' (line 142)
        False_2011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 23), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 16), 'stypy_return_type', False_2011)
        # SSA join for if statement (line 141)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'True' (line 144)
        True_2012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'stypy_return_type', True_2012)
        
        # ################# End of '__lt__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__lt__' in the type store
        # Getting the type of 'stypy_return_type' (line 130)
        stypy_return_type_2013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2013)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__lt__'
        return stypy_return_type_2013


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 146, 4, False)
        # Assigning a type to the variable 'self' (line 147)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeGroup.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        TypeGroup.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeGroup.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeGroup.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'TypeGroup.__repr__')
        TypeGroup.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        TypeGroup.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeGroup.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeGroup.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeGroup.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeGroup.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeGroup.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeGroup.__repr__', [], None, None, defaults, varargs, kwargs)

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

        str_2014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, (-1)), 'str', '\n        Textual representation of the type group\n        :return: str\n        ')
        
        # Assigning a Attribute to a Name (line 161):
        
        # Call to type(...): (line 161)
        # Processing the call arguments (line 161)
        # Getting the type of 'self' (line 161)
        self_2016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 23), 'self', False)
        # Processing the call keyword arguments (line 161)
        kwargs_2017 = {}
        # Getting the type of 'type' (line 161)
        type_2015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 18), 'type', False)
        # Calling type(args, kwargs) (line 161)
        type_call_result_2018 = invoke(stypy.reporting.localization.Localization(__file__, 161, 18), type_2015, *[self_2016], **kwargs_2017)
        
        # Obtaining the member '__name__' of a type (line 161)
        name___2019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 18), type_call_result_2018, '__name__')
        # Assigning a type to the variable 'ret_str' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'ret_str', name___2019)
        # Getting the type of 'ret_str' (line 162)
        ret_str_2020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 15), 'ret_str')
        # Assigning a type to the variable 'stypy_return_type' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'stypy_return_type', ret_str_2020)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 146)
        stypy_return_type_2021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2021)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_2021


# Assigning a type to the variable 'TypeGroup' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'TypeGroup', TypeGroup)
# Declaration of the 'DependentType' class

class DependentType:
    str_2022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, (-1)), 'str', "\n    A DependentType is a special base class that indicates that a type group has to be called to obtain the real\n    type it represent. Call is done using the parameters that are trying to match the rule. For example, imagine that\n    we call the + operator with an object that defines the __add__ method and another type to add to. With an object\n    that defines an __add__ method we don't really know what will be the result of calling __add__ over this object\n    with the second parameter, so the __add__ method has to be called (well, in fact, the type inference equivalent\n    version of the __add__ method will be called) to obtain the real return type.\n\n    Dependent types are a powerful mechanism to calculate the return type of operations that depend on calls to\n    certain object members or even to detect incorrect definitions of members in objects (__int__ method defined in\n    object that do not return int, for example).\n    ")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'False' (line 179)
        False_2023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 37), 'False')
        defaults = [False_2023]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 179, 4, False)
        # Assigning a type to the variable 'self' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DependentType.__init__', ['report_errors'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['report_errors'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_2024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, (-1)), 'str', '\n        Build a Dependent type instance\n        :param report_errors: Flag to indicate if errors found when calling this type will be reported or not (in that\n        case other code will do it)\n        ')
        
        # Assigning a Name to a Attribute (line 185):
        # Getting the type of 'report_errors' (line 185)
        report_errors_2025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 29), 'report_errors')
        # Getting the type of 'self' (line 185)
        self_2026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'self')
        # Setting the type of the member 'report_errors' of a type (line 185)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 8), self_2026, 'report_errors', report_errors_2025)
        
        # Assigning a Num to a Attribute (line 186):
        int_2027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 186, 26), 'int')
        # Getting the type of 'self' (line 186)
        self_2028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'self')
        # Setting the type of the member 'call_arity' of a type (line 186)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 8), self_2028, 'call_arity', int_2027)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 188, 4, False)
        # Assigning a type to the variable 'self' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DependentType.__call__.__dict__.__setitem__('stypy_localization', localization)
        DependentType.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DependentType.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        DependentType.__call__.__dict__.__setitem__('stypy_function_name', 'DependentType.__call__')
        DependentType.__call__.__dict__.__setitem__('stypy_param_names_list', [])
        DependentType.__call__.__dict__.__setitem__('stypy_varargs_param_name', 'call_args')
        DependentType.__call__.__dict__.__setitem__('stypy_kwargs_param_name', 'call_kwargs')
        DependentType.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        DependentType.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        DependentType.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DependentType.__call__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DependentType.__call__', [], 'call_args', 'call_kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        str_2029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, (-1)), 'str', '\n        Call the dependent type. Empty in this implementation, concrete calls must be defined in subclasses\n        ')
        pass
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 188)
        stypy_return_type_2030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2030)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_2030


# Assigning a type to the variable 'DependentType' (line 165)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 0), 'DependentType', DependentType)
# Declaration of the 'HasMember' class
# Getting the type of 'TypeGroup' (line 195)
TypeGroup_2031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 16), 'TypeGroup')
# Getting the type of 'DependentType' (line 195)
DependentType_2032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 27), 'DependentType')

class HasMember(TypeGroup_2031, DependentType_2032, ):
    str_2033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, (-1)), 'str', '\n        Type of any object that has a member with the specified arity, and that can be called with the corresponding\n        params.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_2034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 64), 'int')
        # Getting the type of 'False' (line 201)
        False_2035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 81), 'False')
        defaults = [int_2034, False_2035]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 201, 4, False)
        # Assigning a type to the variable 'self' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HasMember.__init__', ['member', 'expected_return_type', 'call_arity', 'report_errors'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['member', 'expected_return_type', 'call_arity', 'report_errors'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'self' (line 202)
        self_2038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 31), 'self', False)
        # Getting the type of 'report_errors' (line 202)
        report_errors_2039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 37), 'report_errors', False)
        # Processing the call keyword arguments (line 202)
        kwargs_2040 = {}
        # Getting the type of 'DependentType' (line 202)
        DependentType_2036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'DependentType', False)
        # Obtaining the member '__init__' of a type (line 202)
        init___2037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 8), DependentType_2036, '__init__')
        # Calling __init__(args, kwargs) (line 202)
        init___call_result_2041 = invoke(stypy.reporting.localization.Localization(__file__, 202, 8), init___2037, *[self_2038, report_errors_2039], **kwargs_2040)
        
        
        # Call to __init__(...): (line 203)
        # Processing the call arguments (line 203)
        # Getting the type of 'self' (line 203)
        self_2044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 27), 'self', False)
        
        # Obtaining an instance of the builtin type 'list' (line 203)
        list_2045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 33), 'list')
        # Adding type elements to the builtin type 'list' instance (line 203)
        
        # Processing the call keyword arguments (line 203)
        kwargs_2046 = {}
        # Getting the type of 'TypeGroup' (line 203)
        TypeGroup_2042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'TypeGroup', False)
        # Obtaining the member '__init__' of a type (line 203)
        init___2043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 8), TypeGroup_2042, '__init__')
        # Calling __init__(args, kwargs) (line 203)
        init___call_result_2047 = invoke(stypy.reporting.localization.Localization(__file__, 203, 8), init___2043, *[self_2044, list_2045], **kwargs_2046)
        
        
        # Assigning a Name to a Attribute (line 204):
        # Getting the type of 'member' (line 204)
        member_2048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 22), 'member')
        # Getting the type of 'self' (line 204)
        self_2049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 8), 'self')
        # Setting the type of the member 'member' of a type (line 204)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 8), self_2049, 'member', member_2048)
        
        # Assigning a Name to a Attribute (line 205):
        # Getting the type of 'expected_return_type' (line 205)
        expected_return_type_2050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 36), 'expected_return_type')
        # Getting the type of 'self' (line 205)
        self_2051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'self')
        # Setting the type of the member 'expected_return_type' of a type (line 205)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 8), self_2051, 'expected_return_type', expected_return_type_2050)
        
        # Assigning a Name to a Attribute (line 206):
        # Getting the type of 'None' (line 206)
        None_2052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 26), 'None')
        # Getting the type of 'self' (line 206)
        self_2053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'self')
        # Setting the type of the member 'member_obj' of a type (line 206)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 8), self_2053, 'member_obj', None_2052)
        
        # Assigning a Name to a Attribute (line 207):
        # Getting the type of 'call_arity' (line 207)
        call_arity_2054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 26), 'call_arity')
        # Getting the type of 'self' (line 207)
        self_2055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 8), 'self')
        # Setting the type of the member 'call_arity' of a type (line 207)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 8), self_2055, 'call_arity', call_arity_2054)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def format_arity(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'format_arity'
        module_type_store = module_type_store.open_function_context('format_arity', 209, 4, False)
        # Assigning a type to the variable 'self' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HasMember.format_arity.__dict__.__setitem__('stypy_localization', localization)
        HasMember.format_arity.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HasMember.format_arity.__dict__.__setitem__('stypy_type_store', module_type_store)
        HasMember.format_arity.__dict__.__setitem__('stypy_function_name', 'HasMember.format_arity')
        HasMember.format_arity.__dict__.__setitem__('stypy_param_names_list', [])
        HasMember.format_arity.__dict__.__setitem__('stypy_varargs_param_name', None)
        HasMember.format_arity.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HasMember.format_arity.__dict__.__setitem__('stypy_call_defaults', defaults)
        HasMember.format_arity.__dict__.__setitem__('stypy_call_varargs', varargs)
        HasMember.format_arity.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HasMember.format_arity.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HasMember.format_arity', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'format_arity', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'format_arity(...)' code ##################

        
        # Assigning a Str to a Name (line 210):
        str_2056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 15), 'str', '(')
        # Assigning a type to the variable 'str_' (line 210)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 8), 'str_', str_2056)
        
        
        # Call to range(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'self' (line 211)
        self_2058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 23), 'self', False)
        # Obtaining the member 'call_arity' of a type (line 211)
        call_arity_2059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 23), self_2058, 'call_arity')
        # Processing the call keyword arguments (line 211)
        kwargs_2060 = {}
        # Getting the type of 'range' (line 211)
        range_2057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 17), 'range', False)
        # Calling range(args, kwargs) (line 211)
        range_call_result_2061 = invoke(stypy.reporting.localization.Localization(__file__, 211, 17), range_2057, *[call_arity_2059], **kwargs_2060)
        
        # Testing the type of a for loop iterable (line 211)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 211, 8), range_call_result_2061)
        # Getting the type of the for loop variable (line 211)
        for_loop_var_2062 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 211, 8), range_call_result_2061)
        # Assigning a type to the variable 'i' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'i', for_loop_var_2062)
        # SSA begins for a for statement (line 211)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'str_' (line 212)
        str__2063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'str_')
        str_2064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 20), 'str', 'parameter')
        
        # Call to str(...): (line 212)
        # Processing the call arguments (line 212)
        # Getting the type of 'i' (line 212)
        i_2066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 38), 'i', False)
        # Processing the call keyword arguments (line 212)
        kwargs_2067 = {}
        # Getting the type of 'str' (line 212)
        str_2065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 34), 'str', False)
        # Calling str(args, kwargs) (line 212)
        str_call_result_2068 = invoke(stypy.reporting.localization.Localization(__file__, 212, 34), str_2065, *[i_2066], **kwargs_2067)
        
        # Applying the binary operator '+' (line 212)
        result_add_2069 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 20), '+', str_2064, str_call_result_2068)
        
        str_2070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 43), 'str', ', ')
        # Applying the binary operator '+' (line 212)
        result_add_2071 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 41), '+', result_add_2069, str_2070)
        
        # Applying the binary operator '+=' (line 212)
        result_iadd_2072 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 12), '+=', str__2063, result_add_2071)
        # Assigning a type to the variable 'str_' (line 212)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'str_', result_iadd_2072)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 214)
        self_2073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 11), 'self')
        # Obtaining the member 'call_arity' of a type (line 214)
        call_arity_2074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 11), self_2073, 'call_arity')
        int_2075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 29), 'int')
        # Applying the binary operator '>' (line 214)
        result_gt_2076 = python_operator(stypy.reporting.localization.Localization(__file__, 214, 11), '>', call_arity_2074, int_2075)
        
        # Testing the type of an if condition (line 214)
        if_condition_2077 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 214, 8), result_gt_2076)
        # Assigning a type to the variable 'if_condition_2077' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'if_condition_2077', if_condition_2077)
        # SSA begins for if statement (line 214)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 215):
        
        # Obtaining the type of the subscript
        int_2078 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 25), 'int')
        slice_2079 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 215, 19), None, int_2078, None)
        # Getting the type of 'str_' (line 215)
        str__2080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 19), 'str_')
        # Obtaining the member '__getitem__' of a type (line 215)
        getitem___2081 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 19), str__2080, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 215)
        subscript_call_result_2082 = invoke(stypy.reporting.localization.Localization(__file__, 215, 19), getitem___2081, slice_2079)
        
        # Assigning a type to the variable 'str_' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'str_', subscript_call_result_2082)
        # SSA join for if statement (line 214)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'str_' (line 217)
        str__2083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 15), 'str_')
        str_2084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 22), 'str', ')')
        # Applying the binary operator '+' (line 217)
        result_add_2085 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 15), '+', str__2083, str_2084)
        
        # Assigning a type to the variable 'stypy_return_type' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'stypy_return_type', result_add_2085)
        
        # ################# End of 'format_arity(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'format_arity' in the type store
        # Getting the type of 'stypy_return_type' (line 209)
        stypy_return_type_2086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2086)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'format_arity'
        return stypy_return_type_2086


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 219, 4, False)
        # Assigning a type to the variable 'self' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HasMember.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        HasMember.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HasMember.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        HasMember.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'HasMember.__eq__')
        HasMember.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['type_'])
        HasMember.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        HasMember.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HasMember.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        HasMember.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        HasMember.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HasMember.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HasMember.__eq__', ['type_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__eq__', localization, ['type_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__eq__(...)' code ##################

        
        # Assigning a Call to a Attribute (line 220):
        
        # Call to get_type_of_member(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'None' (line 220)
        None_2089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 51), 'None', False)
        # Getting the type of 'self' (line 220)
        self_2090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 57), 'self', False)
        # Obtaining the member 'member' of a type (line 220)
        member_2091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 57), self_2090, 'member')
        # Processing the call keyword arguments (line 220)
        kwargs_2092 = {}
        # Getting the type of 'type_' (line 220)
        type__2087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 26), 'type_', False)
        # Obtaining the member 'get_type_of_member' of a type (line 220)
        get_type_of_member_2088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 26), type__2087, 'get_type_of_member')
        # Calling get_type_of_member(args, kwargs) (line 220)
        get_type_of_member_call_result_2093 = invoke(stypy.reporting.localization.Localization(__file__, 220, 26), get_type_of_member_2088, *[None_2089, member_2091], **kwargs_2092)
        
        # Getting the type of 'self' (line 220)
        self_2094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'self')
        # Setting the type of the member 'member_obj' of a type (line 220)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 8), self_2094, 'member_obj', get_type_of_member_call_result_2093)
        
        # Type idiom detected: calculating its left and rigth part (line 221)
        # Getting the type of 'TypeError' (line 221)
        TypeError_2095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 39), 'TypeError')
        # Getting the type of 'self' (line 221)
        self_2096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 22), 'self')
        # Obtaining the member 'member_obj' of a type (line 221)
        member_obj_2097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 22), self_2096, 'member_obj')
        
        (may_be_2098, more_types_in_union_2099) = may_be_subtype(TypeError_2095, member_obj_2097)

        if may_be_2098:

            if more_types_in_union_2099:
                # Runtime conditional SSA (line 221)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'self' (line 221)
            self_2100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 8), 'self')
            # Obtaining the member 'member_obj' of a type (line 221)
            member_obj_2101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), self_2100, 'member_obj')
            # Setting the type of the member 'member_obj' of a type (line 221)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 8), self_2100, 'member_obj', remove_not_subtype_from_union(member_obj_2097, TypeError))
            
            
            # Getting the type of 'self' (line 222)
            self_2102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 19), 'self')
            # Obtaining the member 'report_errors' of a type (line 222)
            report_errors_2103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 19), self_2102, 'report_errors')
            # Applying the 'not' unary operator (line 222)
            result_not__2104 = python_operator(stypy.reporting.localization.Localization(__file__, 222, 15), 'not', report_errors_2103)
            
            # Testing the type of an if condition (line 222)
            if_condition_2105 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 222, 12), result_not__2104)
            # Assigning a type to the variable 'if_condition_2105' (line 222)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 12), 'if_condition_2105', if_condition_2105)
            # SSA begins for if statement (line 222)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to remove_error_msg(...): (line 223)
            # Processing the call arguments (line 223)
            # Getting the type of 'self' (line 223)
            self_2108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 43), 'self', False)
            # Obtaining the member 'member_obj' of a type (line 223)
            member_obj_2109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 43), self_2108, 'member_obj')
            # Processing the call keyword arguments (line 223)
            kwargs_2110 = {}
            # Getting the type of 'TypeError' (line 223)
            TypeError_2106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 16), 'TypeError', False)
            # Obtaining the member 'remove_error_msg' of a type (line 223)
            remove_error_msg_2107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 16), TypeError_2106, 'remove_error_msg')
            # Calling remove_error_msg(args, kwargs) (line 223)
            remove_error_msg_call_result_2111 = invoke(stypy.reporting.localization.Localization(__file__, 223, 16), remove_error_msg_2107, *[member_obj_2109], **kwargs_2110)
            
            # SSA join for if statement (line 222)
            module_type_store = module_type_store.join_ssa_context()
            
            # Getting the type of 'False' (line 224)
            False_2112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 19), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 224)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), 'stypy_return_type', False_2112)

            if more_types_in_union_2099:
                # SSA join for if statement (line 221)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'True' (line 226)
        True_2113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'stypy_return_type', True_2113)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 219)
        stypy_return_type_2114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2114)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_2114


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 228, 4, False)
        # Assigning a type to the variable 'self' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HasMember.__call__.__dict__.__setitem__('stypy_localization', localization)
        HasMember.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HasMember.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        HasMember.__call__.__dict__.__setitem__('stypy_function_name', 'HasMember.__call__')
        HasMember.__call__.__dict__.__setitem__('stypy_param_names_list', ['localization'])
        HasMember.__call__.__dict__.__setitem__('stypy_varargs_param_name', 'call_args')
        HasMember.__call__.__dict__.__setitem__('stypy_kwargs_param_name', 'call_kwargs')
        HasMember.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        HasMember.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        HasMember.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HasMember.__call__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HasMember.__call__', ['localization'], 'call_args', 'call_kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['localization'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        
        
        # Call to callable(...): (line 229)
        # Processing the call arguments (line 229)
        
        # Call to get_python_type(...): (line 229)
        # Processing the call keyword arguments (line 229)
        kwargs_2119 = {}
        # Getting the type of 'self' (line 229)
        self_2116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 20), 'self', False)
        # Obtaining the member 'member_obj' of a type (line 229)
        member_obj_2117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 20), self_2116, 'member_obj')
        # Obtaining the member 'get_python_type' of a type (line 229)
        get_python_type_2118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 20), member_obj_2117, 'get_python_type')
        # Calling get_python_type(args, kwargs) (line 229)
        get_python_type_call_result_2120 = invoke(stypy.reporting.localization.Localization(__file__, 229, 20), get_python_type_2118, *[], **kwargs_2119)
        
        # Processing the call keyword arguments (line 229)
        kwargs_2121 = {}
        # Getting the type of 'callable' (line 229)
        callable_2115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 11), 'callable', False)
        # Calling callable(args, kwargs) (line 229)
        callable_call_result_2122 = invoke(stypy.reporting.localization.Localization(__file__, 229, 11), callable_2115, *[get_python_type_call_result_2120], **kwargs_2121)
        
        # Testing the type of an if condition (line 229)
        if_condition_2123 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 229, 8), callable_call_result_2122)
        # Assigning a type to the variable 'if_condition_2123' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'if_condition_2123', if_condition_2123)
        # SSA begins for if statement (line 229)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 231):
        
        # Call to invoke(...): (line 231)
        # Processing the call arguments (line 231)
        # Getting the type of 'localization' (line 231)
        localization_2127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 53), 'localization', False)
        # Getting the type of 'call_args' (line 231)
        call_args_2128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 68), 'call_args', False)
        # Processing the call keyword arguments (line 231)
        # Getting the type of 'call_kwargs' (line 231)
        call_kwargs_2129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 81), 'call_kwargs', False)
        kwargs_2130 = {'call_kwargs_2129': call_kwargs_2129}
        # Getting the type of 'self' (line 231)
        self_2124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 30), 'self', False)
        # Obtaining the member 'member_obj' of a type (line 231)
        member_obj_2125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 30), self_2124, 'member_obj')
        # Obtaining the member 'invoke' of a type (line 231)
        invoke_2126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 231, 30), member_obj_2125, 'invoke')
        # Calling invoke(args, kwargs) (line 231)
        invoke_call_result_2131 = invoke(stypy.reporting.localization.Localization(__file__, 231, 30), invoke_2126, *[localization_2127, call_args_2128], **kwargs_2130)
        
        # Assigning a type to the variable 'equivalent_type' (line 231)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 12), 'equivalent_type', invoke_call_result_2131)
        
        # Type idiom detected: calculating its left and rigth part (line 234)
        # Getting the type of 'TypeError' (line 234)
        TypeError_2132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 43), 'TypeError')
        # Getting the type of 'equivalent_type' (line 234)
        equivalent_type_2133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 26), 'equivalent_type')
        
        (may_be_2134, more_types_in_union_2135) = may_be_subtype(TypeError_2132, equivalent_type_2133)

        if may_be_2134:

            if more_types_in_union_2135:
                # Runtime conditional SSA (line 234)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'equivalent_type' (line 234)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 12), 'equivalent_type', remove_not_subtype_from_union(equivalent_type_2133, TypeError))
            
            
            # Getting the type of 'self' (line 235)
            self_2136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 235, 23), 'self')
            # Obtaining the member 'report_errors' of a type (line 235)
            report_errors_2137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 235, 23), self_2136, 'report_errors')
            # Applying the 'not' unary operator (line 235)
            result_not__2138 = python_operator(stypy.reporting.localization.Localization(__file__, 235, 19), 'not', report_errors_2137)
            
            # Testing the type of an if condition (line 235)
            if_condition_2139 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 235, 16), result_not__2138)
            # Assigning a type to the variable 'if_condition_2139' (line 235)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 16), 'if_condition_2139', if_condition_2139)
            # SSA begins for if statement (line 235)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to remove_error_msg(...): (line 236)
            # Processing the call arguments (line 236)
            # Getting the type of 'equivalent_type' (line 236)
            equivalent_type_2142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 47), 'equivalent_type', False)
            # Processing the call keyword arguments (line 236)
            kwargs_2143 = {}
            # Getting the type of 'TypeError' (line 236)
            TypeError_2140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 20), 'TypeError', False)
            # Obtaining the member 'remove_error_msg' of a type (line 236)
            remove_error_msg_2141 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 20), TypeError_2140, 'remove_error_msg')
            # Calling remove_error_msg(args, kwargs) (line 236)
            remove_error_msg_call_result_2144 = invoke(stypy.reporting.localization.Localization(__file__, 236, 20), remove_error_msg_2141, *[equivalent_type_2142], **kwargs_2143)
            
            # SSA join for if statement (line 235)
            module_type_store = module_type_store.join_ssa_context()
            
            
            # Assigning a Name to a Attribute (line 237):
            # Getting the type of 'None' (line 237)
            None_2145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 34), 'None')
            # Getting the type of 'self' (line 237)
            self_2146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 16), 'self')
            # Setting the type of the member 'member_obj' of a type (line 237)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 16), self_2146, 'member_obj', None_2145)
            
            # Obtaining an instance of the builtin type 'tuple' (line 238)
            tuple_2147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 23), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 238)
            # Adding element type (line 238)
            # Getting the type of 'False' (line 238)
            False_2148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 23), 'False')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 23), tuple_2147, False_2148)
            # Adding element type (line 238)
            # Getting the type of 'equivalent_type' (line 238)
            equivalent_type_2149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 30), 'equivalent_type')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 238, 23), tuple_2147, equivalent_type_2149)
            
            # Assigning a type to the variable 'stypy_return_type' (line 238)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 238, 16), 'stypy_return_type', tuple_2147)

            if more_types_in_union_2135:
                # SSA join for if statement (line 234)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Call to isinstance(...): (line 241)
        # Processing the call arguments (line 241)
        # Getting the type of 'self' (line 241)
        self_2151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 26), 'self', False)
        # Obtaining the member 'expected_return_type' of a type (line 241)
        expected_return_type_2152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 26), self_2151, 'expected_return_type')
        # Getting the type of 'UndefinedType' (line 241)
        UndefinedType_2153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 53), 'UndefinedType', False)
        # Processing the call keyword arguments (line 241)
        kwargs_2154 = {}
        # Getting the type of 'isinstance' (line 241)
        isinstance_2150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 15), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 241)
        isinstance_call_result_2155 = invoke(stypy.reporting.localization.Localization(__file__, 241, 15), isinstance_2150, *[expected_return_type_2152, UndefinedType_2153], **kwargs_2154)
        
        # Testing the type of an if condition (line 241)
        if_condition_2156 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 241, 12), isinstance_call_result_2155)
        # Assigning a type to the variable 'if_condition_2156' (line 241)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'if_condition_2156', if_condition_2156)
        # SSA begins for if statement (line 241)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 242):
        # Getting the type of 'None' (line 242)
        None_2157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 34), 'None')
        # Getting the type of 'self' (line 242)
        self_2158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 16), 'self')
        # Setting the type of the member 'member_obj' of a type (line 242)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 16), self_2158, 'member_obj', None_2157)
        
        # Obtaining an instance of the builtin type 'tuple' (line 243)
        tuple_2159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 243)
        # Adding element type (line 243)
        # Getting the type of 'True' (line 243)
        True_2160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 23), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 23), tuple_2159, True_2160)
        # Adding element type (line 243)
        # Getting the type of 'equivalent_type' (line 243)
        equivalent_type_2161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 29), 'equivalent_type')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 243, 23), tuple_2159, equivalent_type_2161)
        
        # Assigning a type to the variable 'stypy_return_type' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 16), 'stypy_return_type', tuple_2159)
        # SSA join for if statement (line 241)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 246)
        self_2162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 15), 'self')
        # Obtaining the member 'expected_return_type' of a type (line 246)
        expected_return_type_2163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 246, 15), self_2162, 'expected_return_type')
        # Getting the type of 'DynamicType' (line 246)
        DynamicType_2164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 44), 'DynamicType')
        # Applying the binary operator 'is' (line 246)
        result_is__2165 = python_operator(stypy.reporting.localization.Localization(__file__, 246, 15), 'is', expected_return_type_2163, DynamicType_2164)
        
        # Testing the type of an if condition (line 246)
        if_condition_2166 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 246, 12), result_is__2165)
        # Assigning a type to the variable 'if_condition_2166' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 12), 'if_condition_2166', if_condition_2166)
        # SSA begins for if statement (line 246)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 247):
        # Getting the type of 'None' (line 247)
        None_2167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 34), 'None')
        # Getting the type of 'self' (line 247)
        self_2168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 16), 'self')
        # Setting the type of the member 'member_obj' of a type (line 247)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 16), self_2168, 'member_obj', None_2167)
        
        # Obtaining an instance of the builtin type 'tuple' (line 248)
        tuple_2169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 248, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 248)
        # Adding element type (line 248)
        # Getting the type of 'True' (line 248)
        True_2170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 23), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 23), tuple_2169, True_2170)
        # Adding element type (line 248)
        # Getting the type of 'equivalent_type' (line 248)
        equivalent_type_2171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 29), 'equivalent_type')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 248, 23), tuple_2169, equivalent_type_2171)
        
        # Assigning a type to the variable 'stypy_return_type' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 16), 'stypy_return_type', tuple_2169)
        # SSA join for if statement (line 246)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to issubclass(...): (line 251)
        # Processing the call arguments (line 251)
        
        # Call to get_python_type(...): (line 251)
        # Processing the call keyword arguments (line 251)
        kwargs_2175 = {}
        # Getting the type of 'equivalent_type' (line 251)
        equivalent_type_2173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 30), 'equivalent_type', False)
        # Obtaining the member 'get_python_type' of a type (line 251)
        get_python_type_2174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 30), equivalent_type_2173, 'get_python_type')
        # Calling get_python_type(args, kwargs) (line 251)
        get_python_type_call_result_2176 = invoke(stypy.reporting.localization.Localization(__file__, 251, 30), get_python_type_2174, *[], **kwargs_2175)
        
        # Getting the type of 'self' (line 251)
        self_2177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 65), 'self', False)
        # Obtaining the member 'expected_return_type' of a type (line 251)
        expected_return_type_2178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 65), self_2177, 'expected_return_type')
        # Processing the call keyword arguments (line 251)
        kwargs_2179 = {}
        # Getting the type of 'issubclass' (line 251)
        issubclass_2172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 19), 'issubclass', False)
        # Calling issubclass(args, kwargs) (line 251)
        issubclass_call_result_2180 = invoke(stypy.reporting.localization.Localization(__file__, 251, 19), issubclass_2172, *[get_python_type_call_result_2176, expected_return_type_2178], **kwargs_2179)
        
        # Applying the 'not' unary operator (line 251)
        result_not__2181 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 15), 'not', issubclass_call_result_2180)
        
        # Testing the type of an if condition (line 251)
        if_condition_2182 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 251, 12), result_not__2181)
        # Assigning a type to the variable 'if_condition_2182' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 12), 'if_condition_2182', if_condition_2182)
        # SSA begins for if statement (line 251)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 252):
        # Getting the type of 'None' (line 252)
        None_2183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 34), 'None')
        # Getting the type of 'self' (line 252)
        self_2184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 16), 'self')
        # Setting the type of the member 'member_obj' of a type (line 252)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 16), self_2184, 'member_obj', None_2183)
        
        # Obtaining an instance of the builtin type 'tuple' (line 253)
        tuple_2185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 253)
        # Adding element type (line 253)
        # Getting the type of 'False' (line 253)
        False_2186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 23), 'False')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 23), tuple_2185, False_2186)
        # Adding element type (line 253)
        # Getting the type of 'equivalent_type' (line 253)
        equivalent_type_2187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 30), 'equivalent_type')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 253, 23), tuple_2185, equivalent_type_2187)
        
        # Assigning a type to the variable 'stypy_return_type' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 16), 'stypy_return_type', tuple_2185)
        # SSA branch for the else part of an if statement (line 251)
        module_type_store.open_ssa_branch('else')
        
        # Obtaining an instance of the builtin type 'tuple' (line 255)
        tuple_2188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 255, 23), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 255)
        # Adding element type (line 255)
        # Getting the type of 'True' (line 255)
        True_2189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 23), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 23), tuple_2188, True_2189)
        # Adding element type (line 255)
        # Getting the type of 'equivalent_type' (line 255)
        equivalent_type_2190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 255, 29), 'equivalent_type')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 255, 23), tuple_2188, equivalent_type_2190)
        
        # Assigning a type to the variable 'stypy_return_type' (line 255)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 255, 16), 'stypy_return_type', tuple_2188)
        # SSA join for if statement (line 251)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 229)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Attribute (line 257):
        # Getting the type of 'None' (line 257)
        None_2191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 26), 'None')
        # Getting the type of 'self' (line 257)
        self_2192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'self')
        # Setting the type of the member 'member_obj' of a type (line 257)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 8), self_2192, 'member_obj', None_2191)
        
        # Obtaining an instance of the builtin type 'tuple' (line 258)
        tuple_2193 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 258)
        # Adding element type (line 258)
        # Getting the type of 'True' (line 258)
        True_2194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 15), 'True')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 15), tuple_2193, True_2194)
        # Adding element type (line 258)
        # Getting the type of 'None' (line 258)
        None_2195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 21), 'None')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 258, 15), tuple_2193, None_2195)
        
        # Assigning a type to the variable 'stypy_return_type' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'stypy_return_type', tuple_2193)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 228)
        stypy_return_type_2196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2196)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_2196


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 260, 4, False)
        # Assigning a type to the variable 'self' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HasMember.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        HasMember.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HasMember.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        HasMember.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'HasMember.__repr__')
        HasMember.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        HasMember.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        HasMember.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HasMember.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        HasMember.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        HasMember.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HasMember.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HasMember.__repr__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Str to a Name (line 261):
        str_2197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 261, 18), 'str', 'Instance defining ')
        # Assigning a type to the variable 'ret_str' (line 261)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 261, 8), 'ret_str', str_2197)
        
        # Getting the type of 'ret_str' (line 262)
        ret_str_2198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'ret_str')
        
        # Call to str(...): (line 262)
        # Processing the call arguments (line 262)
        # Getting the type of 'self' (line 262)
        self_2200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 23), 'self', False)
        # Obtaining the member 'member' of a type (line 262)
        member_2201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 23), self_2200, 'member')
        # Processing the call keyword arguments (line 262)
        kwargs_2202 = {}
        # Getting the type of 'str' (line 262)
        str_2199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 19), 'str', False)
        # Calling str(args, kwargs) (line 262)
        str_call_result_2203 = invoke(stypy.reporting.localization.Localization(__file__, 262, 19), str_2199, *[member_2201], **kwargs_2202)
        
        # Applying the binary operator '+=' (line 262)
        result_iadd_2204 = python_operator(stypy.reporting.localization.Localization(__file__, 262, 8), '+=', ret_str_2198, str_call_result_2203)
        # Assigning a type to the variable 'ret_str' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 8), 'ret_str', result_iadd_2204)
        
        
        # Getting the type of 'ret_str' (line 263)
        ret_str_2205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'ret_str')
        
        # Call to format_arity(...): (line 263)
        # Processing the call keyword arguments (line 263)
        kwargs_2208 = {}
        # Getting the type of 'self' (line 263)
        self_2206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 263, 19), 'self', False)
        # Obtaining the member 'format_arity' of a type (line 263)
        format_arity_2207 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 263, 19), self_2206, 'format_arity')
        # Calling format_arity(args, kwargs) (line 263)
        format_arity_call_result_2209 = invoke(stypy.reporting.localization.Localization(__file__, 263, 19), format_arity_2207, *[], **kwargs_2208)
        
        # Applying the binary operator '+=' (line 263)
        result_iadd_2210 = python_operator(stypy.reporting.localization.Localization(__file__, 263, 8), '+=', ret_str_2205, format_arity_call_result_2209)
        # Assigning a type to the variable 'ret_str' (line 263)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 263, 8), 'ret_str', result_iadd_2210)
        
        # Getting the type of 'ret_str' (line 264)
        ret_str_2211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 15), 'ret_str')
        # Assigning a type to the variable 'stypy_return_type' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 8), 'stypy_return_type', ret_str_2211)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 260)
        stypy_return_type_2212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_2212)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_2212


# Assigning a type to the variable 'HasMember' (line 195)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 0), 'HasMember', HasMember)

# Assigning a Call to a Name (line 267):

# Call to HasMember(...): (line 267)
# Processing the call arguments (line 267)
str_2214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 23), 'str', '__int__')
# Getting the type of 'int' (line 267)
int_2215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 34), 'int', False)
int_2216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 267, 39), 'int')
# Processing the call keyword arguments (line 267)
kwargs_2217 = {}
# Getting the type of 'HasMember' (line 267)
HasMember_2213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 267, 13), 'HasMember', False)
# Calling HasMember(args, kwargs) (line 267)
HasMember_call_result_2218 = invoke(stypy.reporting.localization.Localization(__file__, 267, 13), HasMember_2213, *[str_2214, int_2215, int_2216], **kwargs_2217)

# Assigning a type to the variable 'CastsToInt' (line 267)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 267, 0), 'CastsToInt', HasMember_call_result_2218)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
