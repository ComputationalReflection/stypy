
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import abc
2: 
3: 
4: class Type:
5:     '''
6:     Abstract class that represent the various types we manage in the type-inference equivalent program. It is the
7:     base type of several key classes in stypy:
8: 
9:     - TypeInferenceProxy: That enables us to infer the type of any member of any Python entity that it represents
10:     - TypeError: A special type to model errors found when type checking program.
11:     - UnionType: A special type to model the union of various possible types, obtained as a result of applying the SSA
12:     algorithm.
13: 
14:     So therefore the basic purpose of the Type subclasses is to represent (and therefore, store) Python types. Methods
15:     of this class are created to work with these represented types.
16:     '''
17:     # This is an abstract class
18:     __metaclass__ = abc.ABCMeta
19: 
20:     # Some Type derived classes may contain elements. This is the property that is used to store these elements.
21:     contained_elements_property_name = "contained_elements_type"
22: 
23:     # Equality between types is a very frequently used operation. This list of properties are key to determine
24:     # equality or inequality on most objects. Therefore these properties are the first to be looked upon to determine
25:     # equality of Types and its subclasses, in order to save performance.
26:     special_properties_for_equality = [
27:         "__name__", "im_class", "im_self", "__module__", "__objclass__"
28:     ]
29: 
30:     def __str__(self):
31:         '''
32:         str representation of the class
33:         :return:
34:         '''
35:         return self.__repr__()
36: 
37:     # ################## STORED PYTHON ENTITY (CLASS, METHOD...) AND PYTHON TYPE/INSTANCE OF THE ENTITY ###############
38: 
39:     @abc.abstractmethod
40:     def get_python_entity(self):
41:         '''
42:         Returns the Python entity (function, method, class, object, module...) represented by this Type.
43:         :return: A Python entity
44:         '''
45:         return
46: 
47:     @abc.abstractmethod
48:     def get_python_type(self):
49:         '''
50:         Returns the Python type of the Python entity (function, method, class, object, module...) represented by this
51:         Type. It is almost equal to get_python_entity except for class instances. The Python type for any class instance
52:         is types.InstanceType, not the type of the class.
53:         :return: A Python type
54:         '''
55:         return
56: 
57:     @abc.abstractmethod
58:     def get_instance(self):
59:         '''
60:         If this Type represent an instance of a class, return this instance.
61:         :return:
62:         '''
63:         return
64: 
65:     def has_type_instance_value(self):
66:         '''
67:         Returns if this Type has a value for the "type_instance" property
68:         :return: bool
69:         '''
70:         return hasattr(self, "type_instance")
71: 
72:     def is_type_instance(self):
73:         '''
74:         For the Python type represented by this object, this method is used to distinguish between a type name and a
75:         the type of the element represented by this type. For example, if this element represent the type of the
76:         value '3' (int) is_type_instance returns true. If, however, this element represents the type 'int', the method
77:          returns False. It also returns false for types that do not have instances (functions, modules...)
78:         :return:
79:         '''
80:         if not hasattr(self, "type_instance"):
81:             return False
82:         return self.type_instance
83: 
84:     def set_type_instance(self, value):
85:         '''
86:         Change the type instance value
87:         :param value:
88:         :return:
89:         '''
90:         self.type_instance = value
91: 
92:     # ############################## MEMBER TYPE GET / SET ###############################
93: 
94:     @abc.abstractmethod
95:     def get_type_of_member(self, localization, member_name):
96:         '''
97:         Gets the type of a member of the stored type
98:         :param localization: Caller information
99:         :param member_name: Name of the member
100:         :return: Type of the member
101:         '''
102:         return
103: 
104:     @abc.abstractmethod
105:     def set_type_of_member(self, localization, member_name, member_type):
106:         '''
107:         Set the type of a member of the represented object. If the member do not exist, it is created with the passed
108:         name and types (except iif the represented object do not support reflection, in that case a TypeError is
109:         returned)
110:         :param localization: Caller information
111:         :param member_name: Name of the member
112:         :param member_type: Type of the member
113:         :return: None
114:         '''
115:         return
116: 
117:     # ############################## MEMBER INVOKATION ###############################
118: 
119:     @abc.abstractmethod
120:     def invoke(self, localization, *args, **kwargs):
121:         '''
122:         Invoke the represented object if this is a callable one (function/method/lambda function/class (instance
123:         construction is modelled by invoking the class with appropriate constructor parameters).
124:         :param localization: Caller information
125:         :param args: Arguments of the call
126:         :param kwargs: Keyword arguments of the call
127:         :return: The type that the performed call returned
128:         '''
129:         return
130: 
131:     # ############################## STRUCTURAL REFLECTION ###############################
132: 
133:     @abc.abstractmethod
134:     def delete_member(self, localization, member):
135:         '''
136:         Removes a member by its name, provided the represented object support structural reflection
137:         :param localization: Caller information
138:         :param member: Name of the member to delete
139:         :return: None
140:         '''
141:         return
142: 
143:     @abc.abstractmethod
144:     def supports_structural_reflection(self):
145:         '''
146:         Checks whether the represented object support structural reflection or not
147:         :return: bool
148:         '''
149:         return
150: 
151:     @abc.abstractmethod
152:     def change_type(self, localization, new_type):
153:         '''
154:         Changes the type of the represented object to new_type, should the represented type support structural
155:         reflection
156:         :param localization: Caller information
157:         :param new_type: Type to change the object to
158:         :return: None
159:         '''
160:         return
161: 
162:     @abc.abstractmethod
163:     def change_base_types(self, localization, new_types):
164:         '''
165:         Changes the supertype of the represented object to the ones in new_types, should the represented type support
166:         structural reflection
167:         :param localization: Caller information
168:         :param new_types: Types to assign as new supertypes of the object
169:         :return: None
170:         '''
171:         return
172: 
173:     @abc.abstractmethod
174:     def add_base_types(self, localization, new_types):
175:         '''
176:         Adds to the supertypes of the represented object the ones in new_types, should the represented type support
177:         structural reflection
178:         :param localization: Caller information
179:         :param new_types: Types to add to the supertypes of the object
180:         :return: None
181:         '''
182:         return
183: 
184:     # ############################## TYPE CLONING ###############################
185: 
186:     @abc.abstractmethod
187:     def clone(self):
188:         '''
189:         Make a deep copy of the represented object
190:         :return:
191:         '''
192:         return
193: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import abc' statement (line 1)
import abc

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'abc', abc, module_type_store)

# Declaration of the 'Type' class

class Type:
    str_8804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, (-1)), 'str', '\n    Abstract class that represent the various types we manage in the type-inference equivalent program. It is the\n    base type of several key classes in stypy:\n\n    - TypeInferenceProxy: That enables us to infer the type of any member of any Python entity that it represents\n    - TypeError: A special type to model errors found when type checking program.\n    - UnionType: A special type to model the union of various possible types, obtained as a result of applying the SSA\n    algorithm.\n\n    So therefore the basic purpose of the Type subclasses is to represent (and therefore, store) Python types. Methods\n    of this class are created to work with these represented types.\n    ')

    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 30, 4, False)
        # Assigning a type to the variable 'self' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Type.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        Type.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Type.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Type.stypy__str__.__dict__.__setitem__('stypy_function_name', 'Type.stypy__str__')
        Type.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        Type.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Type.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Type.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Type.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Type.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Type.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Type.stypy__str__', [], None, None, defaults, varargs, kwargs)

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

        str_8805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, (-1)), 'str', '\n        str representation of the class\n        :return:\n        ')
        
        # Call to __repr__(...): (line 35)
        # Processing the call keyword arguments (line 35)
        kwargs_8808 = {}
        # Getting the type of 'self' (line 35)
        self_8806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 15), 'self', False)
        # Obtaining the member '__repr__' of a type (line 35)
        repr___8807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 15), self_8806, '__repr__')
        # Calling __repr__(args, kwargs) (line 35)
        repr___call_result_8809 = invoke(stypy.reporting.localization.Localization(__file__, 35, 15), repr___8807, *[], **kwargs_8808)
        
        # Assigning a type to the variable 'stypy_return_type' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'stypy_return_type', repr___call_result_8809)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 30)
        stypy_return_type_8810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8810)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_8810


    @norecursion
    def get_python_entity(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_python_entity'
        module_type_store = module_type_store.open_function_context('get_python_entity', 39, 4, False)
        # Assigning a type to the variable 'self' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Type.get_python_entity.__dict__.__setitem__('stypy_localization', localization)
        Type.get_python_entity.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Type.get_python_entity.__dict__.__setitem__('stypy_type_store', module_type_store)
        Type.get_python_entity.__dict__.__setitem__('stypy_function_name', 'Type.get_python_entity')
        Type.get_python_entity.__dict__.__setitem__('stypy_param_names_list', [])
        Type.get_python_entity.__dict__.__setitem__('stypy_varargs_param_name', None)
        Type.get_python_entity.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Type.get_python_entity.__dict__.__setitem__('stypy_call_defaults', defaults)
        Type.get_python_entity.__dict__.__setitem__('stypy_call_varargs', varargs)
        Type.get_python_entity.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Type.get_python_entity.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Type.get_python_entity', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_python_entity', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_python_entity(...)' code ##################

        str_8811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, (-1)), 'str', '\n        Returns the Python entity (function, method, class, object, module...) represented by this Type.\n        :return: A Python entity\n        ')
        # Assigning a type to the variable 'stypy_return_type' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'stypy_return_type', types.NoneType)
        
        # ################# End of 'get_python_entity(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_python_entity' in the type store
        # Getting the type of 'stypy_return_type' (line 39)
        stypy_return_type_8812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8812)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_python_entity'
        return stypy_return_type_8812


    @norecursion
    def get_python_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_python_type'
        module_type_store = module_type_store.open_function_context('get_python_type', 47, 4, False)
        # Assigning a type to the variable 'self' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Type.get_python_type.__dict__.__setitem__('stypy_localization', localization)
        Type.get_python_type.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Type.get_python_type.__dict__.__setitem__('stypy_type_store', module_type_store)
        Type.get_python_type.__dict__.__setitem__('stypy_function_name', 'Type.get_python_type')
        Type.get_python_type.__dict__.__setitem__('stypy_param_names_list', [])
        Type.get_python_type.__dict__.__setitem__('stypy_varargs_param_name', None)
        Type.get_python_type.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Type.get_python_type.__dict__.__setitem__('stypy_call_defaults', defaults)
        Type.get_python_type.__dict__.__setitem__('stypy_call_varargs', varargs)
        Type.get_python_type.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Type.get_python_type.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Type.get_python_type', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_python_type', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_python_type(...)' code ##################

        str_8813 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, (-1)), 'str', '\n        Returns the Python type of the Python entity (function, method, class, object, module...) represented by this\n        Type. It is almost equal to get_python_entity except for class instances. The Python type for any class instance\n        is types.InstanceType, not the type of the class.\n        :return: A Python type\n        ')
        # Assigning a type to the variable 'stypy_return_type' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'stypy_return_type', types.NoneType)
        
        # ################# End of 'get_python_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_python_type' in the type store
        # Getting the type of 'stypy_return_type' (line 47)
        stypy_return_type_8814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8814)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_python_type'
        return stypy_return_type_8814


    @norecursion
    def get_instance(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_instance'
        module_type_store = module_type_store.open_function_context('get_instance', 57, 4, False)
        # Assigning a type to the variable 'self' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Type.get_instance.__dict__.__setitem__('stypy_localization', localization)
        Type.get_instance.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Type.get_instance.__dict__.__setitem__('stypy_type_store', module_type_store)
        Type.get_instance.__dict__.__setitem__('stypy_function_name', 'Type.get_instance')
        Type.get_instance.__dict__.__setitem__('stypy_param_names_list', [])
        Type.get_instance.__dict__.__setitem__('stypy_varargs_param_name', None)
        Type.get_instance.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Type.get_instance.__dict__.__setitem__('stypy_call_defaults', defaults)
        Type.get_instance.__dict__.__setitem__('stypy_call_varargs', varargs)
        Type.get_instance.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Type.get_instance.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Type.get_instance', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_instance', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_instance(...)' code ##################

        str_8815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, (-1)), 'str', '\n        If this Type represent an instance of a class, return this instance.\n        :return:\n        ')
        # Assigning a type to the variable 'stypy_return_type' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'stypy_return_type', types.NoneType)
        
        # ################# End of 'get_instance(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_instance' in the type store
        # Getting the type of 'stypy_return_type' (line 57)
        stypy_return_type_8816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8816)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_instance'
        return stypy_return_type_8816


    @norecursion
    def has_type_instance_value(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'has_type_instance_value'
        module_type_store = module_type_store.open_function_context('has_type_instance_value', 65, 4, False)
        # Assigning a type to the variable 'self' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Type.has_type_instance_value.__dict__.__setitem__('stypy_localization', localization)
        Type.has_type_instance_value.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Type.has_type_instance_value.__dict__.__setitem__('stypy_type_store', module_type_store)
        Type.has_type_instance_value.__dict__.__setitem__('stypy_function_name', 'Type.has_type_instance_value')
        Type.has_type_instance_value.__dict__.__setitem__('stypy_param_names_list', [])
        Type.has_type_instance_value.__dict__.__setitem__('stypy_varargs_param_name', None)
        Type.has_type_instance_value.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Type.has_type_instance_value.__dict__.__setitem__('stypy_call_defaults', defaults)
        Type.has_type_instance_value.__dict__.__setitem__('stypy_call_varargs', varargs)
        Type.has_type_instance_value.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Type.has_type_instance_value.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Type.has_type_instance_value', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'has_type_instance_value', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'has_type_instance_value(...)' code ##################

        str_8817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, (-1)), 'str', '\n        Returns if this Type has a value for the "type_instance" property\n        :return: bool\n        ')
        
        # Call to hasattr(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'self' (line 70)
        self_8819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 23), 'self', False)
        str_8820 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 29), 'str', 'type_instance')
        # Processing the call keyword arguments (line 70)
        kwargs_8821 = {}
        # Getting the type of 'hasattr' (line 70)
        hasattr_8818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 15), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 70)
        hasattr_call_result_8822 = invoke(stypy.reporting.localization.Localization(__file__, 70, 15), hasattr_8818, *[self_8819, str_8820], **kwargs_8821)
        
        # Assigning a type to the variable 'stypy_return_type' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'stypy_return_type', hasattr_call_result_8822)
        
        # ################# End of 'has_type_instance_value(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'has_type_instance_value' in the type store
        # Getting the type of 'stypy_return_type' (line 65)
        stypy_return_type_8823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8823)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'has_type_instance_value'
        return stypy_return_type_8823


    @norecursion
    def is_type_instance(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'is_type_instance'
        module_type_store = module_type_store.open_function_context('is_type_instance', 72, 4, False)
        # Assigning a type to the variable 'self' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Type.is_type_instance.__dict__.__setitem__('stypy_localization', localization)
        Type.is_type_instance.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Type.is_type_instance.__dict__.__setitem__('stypy_type_store', module_type_store)
        Type.is_type_instance.__dict__.__setitem__('stypy_function_name', 'Type.is_type_instance')
        Type.is_type_instance.__dict__.__setitem__('stypy_param_names_list', [])
        Type.is_type_instance.__dict__.__setitem__('stypy_varargs_param_name', None)
        Type.is_type_instance.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Type.is_type_instance.__dict__.__setitem__('stypy_call_defaults', defaults)
        Type.is_type_instance.__dict__.__setitem__('stypy_call_varargs', varargs)
        Type.is_type_instance.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Type.is_type_instance.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Type.is_type_instance', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'is_type_instance', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'is_type_instance(...)' code ##################

        str_8824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, (-1)), 'str', "\n        For the Python type represented by this object, this method is used to distinguish between a type name and a\n        the type of the element represented by this type. For example, if this element represent the type of the\n        value '3' (int) is_type_instance returns true. If, however, this element represents the type 'int', the method\n         returns False. It also returns false for types that do not have instances (functions, modules...)\n        :return:\n        ")
        
        # Type idiom detected: calculating its left and rigth part (line 80)
        str_8825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 29), 'str', 'type_instance')
        # Getting the type of 'self' (line 80)
        self_8826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 23), 'self')
        
        (may_be_8827, more_types_in_union_8828) = may_not_provide_member(str_8825, self_8826)

        if may_be_8827:

            if more_types_in_union_8828:
                # Runtime conditional SSA (line 80)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'self' (line 80)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'self', remove_member_provider_from_union(self_8826, 'type_instance'))
            # Getting the type of 'False' (line 81)
            False_8829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 19), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 81)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 12), 'stypy_return_type', False_8829)

            if more_types_in_union_8828:
                # SSA join for if statement (line 80)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'self' (line 82)
        self_8830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), 'self')
        # Obtaining the member 'type_instance' of a type (line 82)
        type_instance_8831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 15), self_8830, 'type_instance')
        # Assigning a type to the variable 'stypy_return_type' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'stypy_return_type', type_instance_8831)
        
        # ################# End of 'is_type_instance(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'is_type_instance' in the type store
        # Getting the type of 'stypy_return_type' (line 72)
        stypy_return_type_8832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8832)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'is_type_instance'
        return stypy_return_type_8832


    @norecursion
    def set_type_instance(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_type_instance'
        module_type_store = module_type_store.open_function_context('set_type_instance', 84, 4, False)
        # Assigning a type to the variable 'self' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Type.set_type_instance.__dict__.__setitem__('stypy_localization', localization)
        Type.set_type_instance.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Type.set_type_instance.__dict__.__setitem__('stypy_type_store', module_type_store)
        Type.set_type_instance.__dict__.__setitem__('stypy_function_name', 'Type.set_type_instance')
        Type.set_type_instance.__dict__.__setitem__('stypy_param_names_list', ['value'])
        Type.set_type_instance.__dict__.__setitem__('stypy_varargs_param_name', None)
        Type.set_type_instance.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Type.set_type_instance.__dict__.__setitem__('stypy_call_defaults', defaults)
        Type.set_type_instance.__dict__.__setitem__('stypy_call_varargs', varargs)
        Type.set_type_instance.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Type.set_type_instance.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Type.set_type_instance', ['value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_type_instance', localization, ['value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_type_instance(...)' code ##################

        str_8833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, (-1)), 'str', '\n        Change the type instance value\n        :param value:\n        :return:\n        ')
        
        # Assigning a Name to a Attribute (line 90):
        # Getting the type of 'value' (line 90)
        value_8834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 29), 'value')
        # Getting the type of 'self' (line 90)
        self_8835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'self')
        # Setting the type of the member 'type_instance' of a type (line 90)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), self_8835, 'type_instance', value_8834)
        
        # ################# End of 'set_type_instance(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_type_instance' in the type store
        # Getting the type of 'stypy_return_type' (line 84)
        stypy_return_type_8836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8836)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_type_instance'
        return stypy_return_type_8836


    @norecursion
    def get_type_of_member(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_type_of_member'
        module_type_store = module_type_store.open_function_context('get_type_of_member', 94, 4, False)
        # Assigning a type to the variable 'self' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Type.get_type_of_member.__dict__.__setitem__('stypy_localization', localization)
        Type.get_type_of_member.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Type.get_type_of_member.__dict__.__setitem__('stypy_type_store', module_type_store)
        Type.get_type_of_member.__dict__.__setitem__('stypy_function_name', 'Type.get_type_of_member')
        Type.get_type_of_member.__dict__.__setitem__('stypy_param_names_list', ['localization', 'member_name'])
        Type.get_type_of_member.__dict__.__setitem__('stypy_varargs_param_name', None)
        Type.get_type_of_member.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Type.get_type_of_member.__dict__.__setitem__('stypy_call_defaults', defaults)
        Type.get_type_of_member.__dict__.__setitem__('stypy_call_varargs', varargs)
        Type.get_type_of_member.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Type.get_type_of_member.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Type.get_type_of_member', ['localization', 'member_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_type_of_member', localization, ['localization', 'member_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_type_of_member(...)' code ##################

        str_8837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, (-1)), 'str', '\n        Gets the type of a member of the stored type\n        :param localization: Caller information\n        :param member_name: Name of the member\n        :return: Type of the member\n        ')
        # Assigning a type to the variable 'stypy_return_type' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'stypy_return_type', types.NoneType)
        
        # ################# End of 'get_type_of_member(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_type_of_member' in the type store
        # Getting the type of 'stypy_return_type' (line 94)
        stypy_return_type_8838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8838)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_type_of_member'
        return stypy_return_type_8838


    @norecursion
    def set_type_of_member(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_type_of_member'
        module_type_store = module_type_store.open_function_context('set_type_of_member', 104, 4, False)
        # Assigning a type to the variable 'self' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Type.set_type_of_member.__dict__.__setitem__('stypy_localization', localization)
        Type.set_type_of_member.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Type.set_type_of_member.__dict__.__setitem__('stypy_type_store', module_type_store)
        Type.set_type_of_member.__dict__.__setitem__('stypy_function_name', 'Type.set_type_of_member')
        Type.set_type_of_member.__dict__.__setitem__('stypy_param_names_list', ['localization', 'member_name', 'member_type'])
        Type.set_type_of_member.__dict__.__setitem__('stypy_varargs_param_name', None)
        Type.set_type_of_member.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Type.set_type_of_member.__dict__.__setitem__('stypy_call_defaults', defaults)
        Type.set_type_of_member.__dict__.__setitem__('stypy_call_varargs', varargs)
        Type.set_type_of_member.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Type.set_type_of_member.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Type.set_type_of_member', ['localization', 'member_name', 'member_type'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_type_of_member', localization, ['localization', 'member_name', 'member_type'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_type_of_member(...)' code ##################

        str_8839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, (-1)), 'str', '\n        Set the type of a member of the represented object. If the member do not exist, it is created with the passed\n        name and types (except iif the represented object do not support reflection, in that case a TypeError is\n        returned)\n        :param localization: Caller information\n        :param member_name: Name of the member\n        :param member_type: Type of the member\n        :return: None\n        ')
        # Assigning a type to the variable 'stypy_return_type' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'stypy_return_type', types.NoneType)
        
        # ################# End of 'set_type_of_member(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_type_of_member' in the type store
        # Getting the type of 'stypy_return_type' (line 104)
        stypy_return_type_8840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8840)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_type_of_member'
        return stypy_return_type_8840


    @norecursion
    def invoke(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'invoke'
        module_type_store = module_type_store.open_function_context('invoke', 119, 4, False)
        # Assigning a type to the variable 'self' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Type.invoke.__dict__.__setitem__('stypy_localization', localization)
        Type.invoke.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Type.invoke.__dict__.__setitem__('stypy_type_store', module_type_store)
        Type.invoke.__dict__.__setitem__('stypy_function_name', 'Type.invoke')
        Type.invoke.__dict__.__setitem__('stypy_param_names_list', ['localization'])
        Type.invoke.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        Type.invoke.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        Type.invoke.__dict__.__setitem__('stypy_call_defaults', defaults)
        Type.invoke.__dict__.__setitem__('stypy_call_varargs', varargs)
        Type.invoke.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Type.invoke.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Type.invoke', ['localization'], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'invoke', localization, ['localization'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'invoke(...)' code ##################

        str_8841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, (-1)), 'str', '\n        Invoke the represented object if this is a callable one (function/method/lambda function/class (instance\n        construction is modelled by invoking the class with appropriate constructor parameters).\n        :param localization: Caller information\n        :param args: Arguments of the call\n        :param kwargs: Keyword arguments of the call\n        :return: The type that the performed call returned\n        ')
        # Assigning a type to the variable 'stypy_return_type' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'stypy_return_type', types.NoneType)
        
        # ################# End of 'invoke(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'invoke' in the type store
        # Getting the type of 'stypy_return_type' (line 119)
        stypy_return_type_8842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8842)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'invoke'
        return stypy_return_type_8842


    @norecursion
    def delete_member(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'delete_member'
        module_type_store = module_type_store.open_function_context('delete_member', 133, 4, False)
        # Assigning a type to the variable 'self' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Type.delete_member.__dict__.__setitem__('stypy_localization', localization)
        Type.delete_member.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Type.delete_member.__dict__.__setitem__('stypy_type_store', module_type_store)
        Type.delete_member.__dict__.__setitem__('stypy_function_name', 'Type.delete_member')
        Type.delete_member.__dict__.__setitem__('stypy_param_names_list', ['localization', 'member'])
        Type.delete_member.__dict__.__setitem__('stypy_varargs_param_name', None)
        Type.delete_member.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Type.delete_member.__dict__.__setitem__('stypy_call_defaults', defaults)
        Type.delete_member.__dict__.__setitem__('stypy_call_varargs', varargs)
        Type.delete_member.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Type.delete_member.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Type.delete_member', ['localization', 'member'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'delete_member', localization, ['localization', 'member'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'delete_member(...)' code ##################

        str_8843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, (-1)), 'str', '\n        Removes a member by its name, provided the represented object support structural reflection\n        :param localization: Caller information\n        :param member: Name of the member to delete\n        :return: None\n        ')
        # Assigning a type to the variable 'stypy_return_type' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'stypy_return_type', types.NoneType)
        
        # ################# End of 'delete_member(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'delete_member' in the type store
        # Getting the type of 'stypy_return_type' (line 133)
        stypy_return_type_8844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8844)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'delete_member'
        return stypy_return_type_8844


    @norecursion
    def supports_structural_reflection(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'supports_structural_reflection'
        module_type_store = module_type_store.open_function_context('supports_structural_reflection', 143, 4, False)
        # Assigning a type to the variable 'self' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Type.supports_structural_reflection.__dict__.__setitem__('stypy_localization', localization)
        Type.supports_structural_reflection.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Type.supports_structural_reflection.__dict__.__setitem__('stypy_type_store', module_type_store)
        Type.supports_structural_reflection.__dict__.__setitem__('stypy_function_name', 'Type.supports_structural_reflection')
        Type.supports_structural_reflection.__dict__.__setitem__('stypy_param_names_list', [])
        Type.supports_structural_reflection.__dict__.__setitem__('stypy_varargs_param_name', None)
        Type.supports_structural_reflection.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Type.supports_structural_reflection.__dict__.__setitem__('stypy_call_defaults', defaults)
        Type.supports_structural_reflection.__dict__.__setitem__('stypy_call_varargs', varargs)
        Type.supports_structural_reflection.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Type.supports_structural_reflection.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Type.supports_structural_reflection', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'supports_structural_reflection', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'supports_structural_reflection(...)' code ##################

        str_8845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, (-1)), 'str', '\n        Checks whether the represented object support structural reflection or not\n        :return: bool\n        ')
        # Assigning a type to the variable 'stypy_return_type' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'stypy_return_type', types.NoneType)
        
        # ################# End of 'supports_structural_reflection(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'supports_structural_reflection' in the type store
        # Getting the type of 'stypy_return_type' (line 143)
        stypy_return_type_8846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8846)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'supports_structural_reflection'
        return stypy_return_type_8846


    @norecursion
    def change_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'change_type'
        module_type_store = module_type_store.open_function_context('change_type', 151, 4, False)
        # Assigning a type to the variable 'self' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Type.change_type.__dict__.__setitem__('stypy_localization', localization)
        Type.change_type.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Type.change_type.__dict__.__setitem__('stypy_type_store', module_type_store)
        Type.change_type.__dict__.__setitem__('stypy_function_name', 'Type.change_type')
        Type.change_type.__dict__.__setitem__('stypy_param_names_list', ['localization', 'new_type'])
        Type.change_type.__dict__.__setitem__('stypy_varargs_param_name', None)
        Type.change_type.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Type.change_type.__dict__.__setitem__('stypy_call_defaults', defaults)
        Type.change_type.__dict__.__setitem__('stypy_call_varargs', varargs)
        Type.change_type.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Type.change_type.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Type.change_type', ['localization', 'new_type'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'change_type', localization, ['localization', 'new_type'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'change_type(...)' code ##################

        str_8847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, (-1)), 'str', '\n        Changes the type of the represented object to new_type, should the represented type support structural\n        reflection\n        :param localization: Caller information\n        :param new_type: Type to change the object to\n        :return: None\n        ')
        # Assigning a type to the variable 'stypy_return_type' (line 160)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'stypy_return_type', types.NoneType)
        
        # ################# End of 'change_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'change_type' in the type store
        # Getting the type of 'stypy_return_type' (line 151)
        stypy_return_type_8848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8848)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'change_type'
        return stypy_return_type_8848


    @norecursion
    def change_base_types(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'change_base_types'
        module_type_store = module_type_store.open_function_context('change_base_types', 162, 4, False)
        # Assigning a type to the variable 'self' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Type.change_base_types.__dict__.__setitem__('stypy_localization', localization)
        Type.change_base_types.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Type.change_base_types.__dict__.__setitem__('stypy_type_store', module_type_store)
        Type.change_base_types.__dict__.__setitem__('stypy_function_name', 'Type.change_base_types')
        Type.change_base_types.__dict__.__setitem__('stypy_param_names_list', ['localization', 'new_types'])
        Type.change_base_types.__dict__.__setitem__('stypy_varargs_param_name', None)
        Type.change_base_types.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Type.change_base_types.__dict__.__setitem__('stypy_call_defaults', defaults)
        Type.change_base_types.__dict__.__setitem__('stypy_call_varargs', varargs)
        Type.change_base_types.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Type.change_base_types.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Type.change_base_types', ['localization', 'new_types'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'change_base_types', localization, ['localization', 'new_types'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'change_base_types(...)' code ##################

        str_8849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, (-1)), 'str', '\n        Changes the supertype of the represented object to the ones in new_types, should the represented type support\n        structural reflection\n        :param localization: Caller information\n        :param new_types: Types to assign as new supertypes of the object\n        :return: None\n        ')
        # Assigning a type to the variable 'stypy_return_type' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'stypy_return_type', types.NoneType)
        
        # ################# End of 'change_base_types(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'change_base_types' in the type store
        # Getting the type of 'stypy_return_type' (line 162)
        stypy_return_type_8850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8850)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'change_base_types'
        return stypy_return_type_8850


    @norecursion
    def add_base_types(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_base_types'
        module_type_store = module_type_store.open_function_context('add_base_types', 173, 4, False)
        # Assigning a type to the variable 'self' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Type.add_base_types.__dict__.__setitem__('stypy_localization', localization)
        Type.add_base_types.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Type.add_base_types.__dict__.__setitem__('stypy_type_store', module_type_store)
        Type.add_base_types.__dict__.__setitem__('stypy_function_name', 'Type.add_base_types')
        Type.add_base_types.__dict__.__setitem__('stypy_param_names_list', ['localization', 'new_types'])
        Type.add_base_types.__dict__.__setitem__('stypy_varargs_param_name', None)
        Type.add_base_types.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Type.add_base_types.__dict__.__setitem__('stypy_call_defaults', defaults)
        Type.add_base_types.__dict__.__setitem__('stypy_call_varargs', varargs)
        Type.add_base_types.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Type.add_base_types.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Type.add_base_types', ['localization', 'new_types'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_base_types', localization, ['localization', 'new_types'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_base_types(...)' code ##################

        str_8851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, (-1)), 'str', '\n        Adds to the supertypes of the represented object the ones in new_types, should the represented type support\n        structural reflection\n        :param localization: Caller information\n        :param new_types: Types to add to the supertypes of the object\n        :return: None\n        ')
        # Assigning a type to the variable 'stypy_return_type' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'stypy_return_type', types.NoneType)
        
        # ################# End of 'add_base_types(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_base_types' in the type store
        # Getting the type of 'stypy_return_type' (line 173)
        stypy_return_type_8852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8852)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_base_types'
        return stypy_return_type_8852


    @norecursion
    def clone(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'clone'
        module_type_store = module_type_store.open_function_context('clone', 186, 4, False)
        # Assigning a type to the variable 'self' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Type.clone.__dict__.__setitem__('stypy_localization', localization)
        Type.clone.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Type.clone.__dict__.__setitem__('stypy_type_store', module_type_store)
        Type.clone.__dict__.__setitem__('stypy_function_name', 'Type.clone')
        Type.clone.__dict__.__setitem__('stypy_param_names_list', [])
        Type.clone.__dict__.__setitem__('stypy_varargs_param_name', None)
        Type.clone.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Type.clone.__dict__.__setitem__('stypy_call_defaults', defaults)
        Type.clone.__dict__.__setitem__('stypy_call_varargs', varargs)
        Type.clone.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Type.clone.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Type.clone', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'clone', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'clone(...)' code ##################

        str_8853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, (-1)), 'str', '\n        Make a deep copy of the represented object\n        :return:\n        ')
        # Assigning a type to the variable 'stypy_return_type' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'stypy_return_type', types.NoneType)
        
        # ################# End of 'clone(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'clone' in the type store
        # Getting the type of 'stypy_return_type' (line 186)
        stypy_return_type_8854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8854)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'clone'
        return stypy_return_type_8854


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 4, 0, False)
        # Assigning a type to the variable 'self' (line 5)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Type.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Type' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'Type', Type)

# Assigning a Attribute to a Name (line 18):
# Getting the type of 'abc' (line 18)
abc_8855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 20), 'abc')
# Obtaining the member 'ABCMeta' of a type (line 18)
ABCMeta_8856 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 20), abc_8855, 'ABCMeta')
# Getting the type of 'Type'
Type_8857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Type')
# Setting the type of the member '__metaclass__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Type_8857, '__metaclass__', ABCMeta_8856)

# Assigning a Str to a Name (line 21):
str_8858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 39), 'str', 'contained_elements_type')
# Getting the type of 'Type'
Type_8859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Type')
# Setting the type of the member 'contained_elements_property_name' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Type_8859, 'contained_elements_property_name', str_8858)

# Assigning a List to a Name (line 26):

# Obtaining an instance of the builtin type 'list' (line 26)
list_8860 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 38), 'list')
# Adding type elements to the builtin type 'list' instance (line 26)
# Adding element type (line 26)
str_8861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 8), 'str', '__name__')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 38), list_8860, str_8861)
# Adding element type (line 26)
str_8862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 20), 'str', 'im_class')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 38), list_8860, str_8862)
# Adding element type (line 26)
str_8863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 32), 'str', 'im_self')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 38), list_8860, str_8863)
# Adding element type (line 26)
str_8864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 43), 'str', '__module__')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 38), list_8860, str_8864)
# Adding element type (line 26)
str_8865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 57), 'str', '__objclass__')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 38), list_8860, str_8865)

# Getting the type of 'Type'
Type_8866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'Type')
# Setting the type of the member 'special_properties_for_equality' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), Type_8866, 'special_properties_for_equality', list_8860)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
