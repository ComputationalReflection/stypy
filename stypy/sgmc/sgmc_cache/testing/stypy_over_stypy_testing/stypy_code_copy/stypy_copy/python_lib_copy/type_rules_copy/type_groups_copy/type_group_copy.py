
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: class BaseTypeGroup(object):
2:     '''
3:     All type groups inherit from this class
4:     '''
5:     def __str__(self):
6:         return self.__repr__()
7: 
8: 
9: class TypeGroup(BaseTypeGroup):
10:     '''
11:     A TypeGroup is an entity used in the rule files to group several Python types over a named identity. Type groups
12:     are collections of types that have something in common, and Python functions and methods usually admits any of them
13:     as a parameter when one of them is valid. For example, if a Python library function works with an int as the first
14:     parameter, we can also use bool and long as the first parameter without runtime errors. This is for exameple a
15:     TypeGroup that will be called Integer
16: 
17:     Not all type groups are defined by collections of Python concrete types. Other groups identify Python objects with
18:     a common member or structure (Iterable, Overloads__str__ identify any Python object that is iterable and any Python
19:     object that has defined the __str__ method properly) or even class relationships (SubtypeOf type group only matches
20:     with classes that are a subtype of the one specified.
21: 
22:     Type groups are the workhorse of the type rule specification mechanism and have a great expressiveness and
23:     flexibility to specify admitted types in any Python callable entity.
24: 
25:     Type groups are created in the file type_groups.py
26:     '''
27:     def __init__(self, grouped_types):
28:         '''
29:         Create a new type group that represent the list of types passed as a parameter
30:         :param grouped_types: List of types that are included inside this type group
31:         :return:
32:         '''
33:         self.grouped_types = grouped_types
34: 
35:     def __contains__(self, type_):
36:         '''
37:         Test if this type group contains the specified type (in operator)
38:         :param type_: Type to test
39:         :return: bool
40:         '''
41:         # if hasattr(type_, "get_python_type"):
42:         #     return type_.get_python_type() in self.grouped_types
43:         #
44:         # return type_ in self.grouped_types
45:         try:
46:             return type_.get_python_type() in self.grouped_types
47:         except:
48:             return type_ in self.grouped_types
49: 
50:     def __eq__(self, type_):
51:         '''
52:         Test if this type group contains the specified type (== operator)
53:         :param type_: Type to test
54:         :return: bool
55:         '''
56:         # if hasattr(type_, "get_python_type"):
57:         #     return type_.get_python_type() in self.grouped_types
58:         # return type_ in self.grouped_types
59:         try:
60:             cond1 = type_.get_python_type() in self.grouped_types
61:             cond2 = type_.is_type_instance()
62: 
63:             return cond1 and cond2
64:         except:
65:             return type_ in self.grouped_types
66: 
67:     def __cmp__(self, type_):
68:         '''
69:         Test if this type group contains the specified type (compatarion operators)
70:         :param type_: Type to test
71:         :return: bool
72:         '''
73:         # if hasattr(type_, "get_python_type"):
74:         #     return type_.get_python_type() in self.grouped_types
75:         #
76:         # return type_ in self.grouped_types
77:         try:
78:             # return type_.get_python_type() in self.grouped_types
79:             cond1 = type_.get_python_type() in self.grouped_types
80:             cond2 = type_.is_type_instance()
81: 
82:             return cond1 and cond2
83:         except:
84:             return type_ in self.grouped_types
85: 
86:     def __gt__(self, other):
87:         '''
88:         Type group sorting. A type group is less than other type group if contains less types or the types contained
89:         in the type group are all contained in the other one. Otherwise, is greater than the other type group.
90:         :param other: Another type group
91:         :return: bool
92:         '''
93:         if len(self.grouped_types) < len(other.grouped_types):
94:             return False
95: 
96:         for type_ in self.grouped_types:
97:             if type_ not in other.grouped_types:
98:                 return False
99: 
100:         return True
101: 
102:     def __lt__(self, other):
103:         '''
104:         Type group sorting. A type group is less than other type group if contains less types or the types contained
105:         in the type group are all contained in the other one. Otherwise, is greater than the other type group.
106:         :param other: Another type group
107:         :return: bool
108:         '''
109:         if len(self.grouped_types) > len(other.grouped_types):
110:             return False
111: 
112:         for type_ in self.grouped_types:
113:             if type_ not in other.grouped_types:
114:                 return False
115: 
116:         return True
117: 
118:     def __repr__(self):
119:         '''
120:         Textual representation of the type group
121:         :return: str
122:         '''
123:         # ret_str = type(self).__name__  + "("
124:         # for type_ in self.grouped_types:
125:         #     if hasattr(type_, '__name__'):
126:         #         ret_str += type_.__name__ + ", "
127:         #     else:
128:         #         ret_str += str(type_) + ", "
129:         #
130:         # ret_str = ret_str[:-2]
131:         # ret_str+=")"
132: 
133:         ret_str = type(self).__name__
134:         return ret_str
135: 
136: 
137: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

# Declaration of the 'BaseTypeGroup' class

class BaseTypeGroup(object, ):
    str_19846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', '\n    All type groups inherit from this class\n    ')

    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 5, 4, False)
        # Assigning a type to the variable 'self' (line 6)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        BaseTypeGroup.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        BaseTypeGroup.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        BaseTypeGroup.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        BaseTypeGroup.stypy__str__.__dict__.__setitem__('stypy_function_name', 'BaseTypeGroup.stypy__str__')
        BaseTypeGroup.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        BaseTypeGroup.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        BaseTypeGroup.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        BaseTypeGroup.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        BaseTypeGroup.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        BaseTypeGroup.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        BaseTypeGroup.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'BaseTypeGroup.stypy__str__', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to __repr__(...): (line 6)
        # Processing the call keyword arguments (line 6)
        kwargs_19849 = {}
        # Getting the type of 'self' (line 6)
        self_19847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 15), 'self', False)
        # Obtaining the member '__repr__' of a type (line 6)
        repr___19848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 15), self_19847, '__repr__')
        # Calling __repr__(args, kwargs) (line 6)
        repr___call_result_19850 = invoke(stypy.reporting.localization.Localization(__file__, 6, 15), repr___19848, *[], **kwargs_19849)
        
        # Assigning a type to the variable 'stypy_return_type' (line 6)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 8), 'stypy_return_type', repr___call_result_19850)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 5)
        stypy_return_type_19851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19851)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_19851


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


# Assigning a type to the variable 'BaseTypeGroup' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'BaseTypeGroup', BaseTypeGroup)
# Declaration of the 'TypeGroup' class
# Getting the type of 'BaseTypeGroup' (line 9)
BaseTypeGroup_19852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 16), 'BaseTypeGroup')

class TypeGroup(BaseTypeGroup_19852, ):
    str_19853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, (-1)), 'str', '\n    A TypeGroup is an entity used in the rule files to group several Python types over a named identity. Type groups\n    are collections of types that have something in common, and Python functions and methods usually admits any of them\n    as a parameter when one of them is valid. For example, if a Python library function works with an int as the first\n    parameter, we can also use bool and long as the first parameter without runtime errors. This is for exameple a\n    TypeGroup that will be called Integer\n\n    Not all type groups are defined by collections of Python concrete types. Other groups identify Python objects with\n    a common member or structure (Iterable, Overloads__str__ identify any Python object that is iterable and any Python\n    object that has defined the __str__ method properly) or even class relationships (SubtypeOf type group only matches\n    with classes that are a subtype of the one specified.\n\n    Type groups are the workhorse of the type rule specification mechanism and have a great expressiveness and\n    flexibility to specify admitted types in any Python callable entity.\n\n    Type groups are created in the file type_groups.py\n    ')

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

        str_19854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, (-1)), 'str', '\n        Create a new type group that represent the list of types passed as a parameter\n        :param grouped_types: List of types that are included inside this type group\n        :return:\n        ')
        
        # Assigning a Name to a Attribute (line 33):
        # Getting the type of 'grouped_types' (line 33)
        grouped_types_19855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 29), 'grouped_types')
        # Getting the type of 'self' (line 33)
        self_19856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'self')
        # Setting the type of the member 'grouped_types' of a type (line 33)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), self_19856, 'grouped_types', grouped_types_19855)
        
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
        module_type_store = module_type_store.open_function_context('__contains__', 35, 4, False)
        # Assigning a type to the variable 'self' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'self', type_of_self)
        
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

        str_19857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, (-1)), 'str', '\n        Test if this type group contains the specified type (in operator)\n        :param type_: Type to test\n        :return: bool\n        ')
        
        
        # SSA begins for try-except statement (line 45)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        
        # Call to get_python_type(...): (line 46)
        # Processing the call keyword arguments (line 46)
        kwargs_19860 = {}
        # Getting the type of 'type_' (line 46)
        type__19858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 19), 'type_', False)
        # Obtaining the member 'get_python_type' of a type (line 46)
        get_python_type_19859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 19), type__19858, 'get_python_type')
        # Calling get_python_type(args, kwargs) (line 46)
        get_python_type_call_result_19861 = invoke(stypy.reporting.localization.Localization(__file__, 46, 19), get_python_type_19859, *[], **kwargs_19860)
        
        # Getting the type of 'self' (line 46)
        self_19862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 46), 'self')
        # Obtaining the member 'grouped_types' of a type (line 46)
        grouped_types_19863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 46), self_19862, 'grouped_types')
        # Applying the binary operator 'in' (line 46)
        result_contains_19864 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 19), 'in', get_python_type_call_result_19861, grouped_types_19863)
        
        # Assigning a type to the variable 'stypy_return_type' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'stypy_return_type', result_contains_19864)
        # SSA branch for the except part of a try statement (line 45)
        # SSA branch for the except '<any exception>' branch of a try statement (line 45)
        module_type_store.open_ssa_branch('except')
        
        # Getting the type of 'type_' (line 48)
        type__19865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 19), 'type_')
        # Getting the type of 'self' (line 48)
        self_19866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 28), 'self')
        # Obtaining the member 'grouped_types' of a type (line 48)
        grouped_types_19867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 28), self_19866, 'grouped_types')
        # Applying the binary operator 'in' (line 48)
        result_contains_19868 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 19), 'in', type__19865, grouped_types_19867)
        
        # Assigning a type to the variable 'stypy_return_type' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'stypy_return_type', result_contains_19868)
        # SSA join for try-except statement (line 45)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__contains__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__contains__' in the type store
        # Getting the type of 'stypy_return_type' (line 35)
        stypy_return_type_19869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19869)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__contains__'
        return stypy_return_type_19869


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 50, 4, False)
        # Assigning a type to the variable 'self' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeGroup.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        TypeGroup.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeGroup.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeGroup.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'TypeGroup.stypy__eq__')
        TypeGroup.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['type_'])
        TypeGroup.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeGroup.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeGroup.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeGroup.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeGroup.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeGroup.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeGroup.stypy__eq__', ['type_'], None, None, defaults, varargs, kwargs)

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

        str_19870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, (-1)), 'str', '\n        Test if this type group contains the specified type (== operator)\n        :param type_: Type to test\n        :return: bool\n        ')
        
        
        # SSA begins for try-except statement (line 59)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Compare to a Name (line 60):
        
        
        # Call to get_python_type(...): (line 60)
        # Processing the call keyword arguments (line 60)
        kwargs_19873 = {}
        # Getting the type of 'type_' (line 60)
        type__19871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 20), 'type_', False)
        # Obtaining the member 'get_python_type' of a type (line 60)
        get_python_type_19872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 20), type__19871, 'get_python_type')
        # Calling get_python_type(args, kwargs) (line 60)
        get_python_type_call_result_19874 = invoke(stypy.reporting.localization.Localization(__file__, 60, 20), get_python_type_19872, *[], **kwargs_19873)
        
        # Getting the type of 'self' (line 60)
        self_19875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 47), 'self')
        # Obtaining the member 'grouped_types' of a type (line 60)
        grouped_types_19876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 47), self_19875, 'grouped_types')
        # Applying the binary operator 'in' (line 60)
        result_contains_19877 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 20), 'in', get_python_type_call_result_19874, grouped_types_19876)
        
        # Assigning a type to the variable 'cond1' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'cond1', result_contains_19877)
        
        # Assigning a Call to a Name (line 61):
        
        # Call to is_type_instance(...): (line 61)
        # Processing the call keyword arguments (line 61)
        kwargs_19880 = {}
        # Getting the type of 'type_' (line 61)
        type__19878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 20), 'type_', False)
        # Obtaining the member 'is_type_instance' of a type (line 61)
        is_type_instance_19879 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 20), type__19878, 'is_type_instance')
        # Calling is_type_instance(args, kwargs) (line 61)
        is_type_instance_call_result_19881 = invoke(stypy.reporting.localization.Localization(__file__, 61, 20), is_type_instance_19879, *[], **kwargs_19880)
        
        # Assigning a type to the variable 'cond2' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'cond2', is_type_instance_call_result_19881)
        
        # Evaluating a boolean operation
        # Getting the type of 'cond1' (line 63)
        cond1_19882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 19), 'cond1')
        # Getting the type of 'cond2' (line 63)
        cond2_19883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 29), 'cond2')
        # Applying the binary operator 'and' (line 63)
        result_and_keyword_19884 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 19), 'and', cond1_19882, cond2_19883)
        
        # Assigning a type to the variable 'stypy_return_type' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'stypy_return_type', result_and_keyword_19884)
        # SSA branch for the except part of a try statement (line 59)
        # SSA branch for the except '<any exception>' branch of a try statement (line 59)
        module_type_store.open_ssa_branch('except')
        
        # Getting the type of 'type_' (line 65)
        type__19885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 19), 'type_')
        # Getting the type of 'self' (line 65)
        self_19886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 28), 'self')
        # Obtaining the member 'grouped_types' of a type (line 65)
        grouped_types_19887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 28), self_19886, 'grouped_types')
        # Applying the binary operator 'in' (line 65)
        result_contains_19888 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 19), 'in', type__19885, grouped_types_19887)
        
        # Assigning a type to the variable 'stypy_return_type' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'stypy_return_type', result_contains_19888)
        # SSA join for try-except statement (line 59)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 50)
        stypy_return_type_19889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19889)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_19889


    @norecursion
    def stypy__cmp__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__cmp__'
        module_type_store = module_type_store.open_function_context('__cmp__', 67, 4, False)
        # Assigning a type to the variable 'self' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeGroup.stypy__cmp__.__dict__.__setitem__('stypy_localization', localization)
        TypeGroup.stypy__cmp__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeGroup.stypy__cmp__.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeGroup.stypy__cmp__.__dict__.__setitem__('stypy_function_name', 'TypeGroup.stypy__cmp__')
        TypeGroup.stypy__cmp__.__dict__.__setitem__('stypy_param_names_list', ['type_'])
        TypeGroup.stypy__cmp__.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeGroup.stypy__cmp__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeGroup.stypy__cmp__.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeGroup.stypy__cmp__.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeGroup.stypy__cmp__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeGroup.stypy__cmp__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeGroup.stypy__cmp__', ['type_'], None, None, defaults, varargs, kwargs)

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

        str_19890 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, (-1)), 'str', '\n        Test if this type group contains the specified type (compatarion operators)\n        :param type_: Type to test\n        :return: bool\n        ')
        
        
        # SSA begins for try-except statement (line 77)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Compare to a Name (line 79):
        
        
        # Call to get_python_type(...): (line 79)
        # Processing the call keyword arguments (line 79)
        kwargs_19893 = {}
        # Getting the type of 'type_' (line 79)
        type__19891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 20), 'type_', False)
        # Obtaining the member 'get_python_type' of a type (line 79)
        get_python_type_19892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 20), type__19891, 'get_python_type')
        # Calling get_python_type(args, kwargs) (line 79)
        get_python_type_call_result_19894 = invoke(stypy.reporting.localization.Localization(__file__, 79, 20), get_python_type_19892, *[], **kwargs_19893)
        
        # Getting the type of 'self' (line 79)
        self_19895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 47), 'self')
        # Obtaining the member 'grouped_types' of a type (line 79)
        grouped_types_19896 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 47), self_19895, 'grouped_types')
        # Applying the binary operator 'in' (line 79)
        result_contains_19897 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 20), 'in', get_python_type_call_result_19894, grouped_types_19896)
        
        # Assigning a type to the variable 'cond1' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 12), 'cond1', result_contains_19897)
        
        # Assigning a Call to a Name (line 80):
        
        # Call to is_type_instance(...): (line 80)
        # Processing the call keyword arguments (line 80)
        kwargs_19900 = {}
        # Getting the type of 'type_' (line 80)
        type__19898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 20), 'type_', False)
        # Obtaining the member 'is_type_instance' of a type (line 80)
        is_type_instance_19899 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 80, 20), type__19898, 'is_type_instance')
        # Calling is_type_instance(args, kwargs) (line 80)
        is_type_instance_call_result_19901 = invoke(stypy.reporting.localization.Localization(__file__, 80, 20), is_type_instance_19899, *[], **kwargs_19900)
        
        # Assigning a type to the variable 'cond2' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 12), 'cond2', is_type_instance_call_result_19901)
        
        # Evaluating a boolean operation
        # Getting the type of 'cond1' (line 82)
        cond1_19902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 19), 'cond1')
        # Getting the type of 'cond2' (line 82)
        cond2_19903 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 29), 'cond2')
        # Applying the binary operator 'and' (line 82)
        result_and_keyword_19904 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 19), 'and', cond1_19902, cond2_19903)
        
        # Assigning a type to the variable 'stypy_return_type' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'stypy_return_type', result_and_keyword_19904)
        # SSA branch for the except part of a try statement (line 77)
        # SSA branch for the except '<any exception>' branch of a try statement (line 77)
        module_type_store.open_ssa_branch('except')
        
        # Getting the type of 'type_' (line 84)
        type__19905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 19), 'type_')
        # Getting the type of 'self' (line 84)
        self_19906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 28), 'self')
        # Obtaining the member 'grouped_types' of a type (line 84)
        grouped_types_19907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 28), self_19906, 'grouped_types')
        # Applying the binary operator 'in' (line 84)
        result_contains_19908 = python_operator(stypy.reporting.localization.Localization(__file__, 84, 19), 'in', type__19905, grouped_types_19907)
        
        # Assigning a type to the variable 'stypy_return_type' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 12), 'stypy_return_type', result_contains_19908)
        # SSA join for try-except statement (line 77)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__cmp__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__cmp__' in the type store
        # Getting the type of 'stypy_return_type' (line 67)
        stypy_return_type_19909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19909)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__cmp__'
        return stypy_return_type_19909


    @norecursion
    def __gt__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__gt__'
        module_type_store = module_type_store.open_function_context('__gt__', 86, 4, False)
        # Assigning a type to the variable 'self' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'self', type_of_self)
        
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

        str_19910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, (-1)), 'str', '\n        Type group sorting. A type group is less than other type group if contains less types or the types contained\n        in the type group are all contained in the other one. Otherwise, is greater than the other type group.\n        :param other: Another type group\n        :return: bool\n        ')
        
        
        # Call to len(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'self' (line 93)
        self_19912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 15), 'self', False)
        # Obtaining the member 'grouped_types' of a type (line 93)
        grouped_types_19913 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 15), self_19912, 'grouped_types')
        # Processing the call keyword arguments (line 93)
        kwargs_19914 = {}
        # Getting the type of 'len' (line 93)
        len_19911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 11), 'len', False)
        # Calling len(args, kwargs) (line 93)
        len_call_result_19915 = invoke(stypy.reporting.localization.Localization(__file__, 93, 11), len_19911, *[grouped_types_19913], **kwargs_19914)
        
        
        # Call to len(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'other' (line 93)
        other_19917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 41), 'other', False)
        # Obtaining the member 'grouped_types' of a type (line 93)
        grouped_types_19918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 41), other_19917, 'grouped_types')
        # Processing the call keyword arguments (line 93)
        kwargs_19919 = {}
        # Getting the type of 'len' (line 93)
        len_19916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 37), 'len', False)
        # Calling len(args, kwargs) (line 93)
        len_call_result_19920 = invoke(stypy.reporting.localization.Localization(__file__, 93, 37), len_19916, *[grouped_types_19918], **kwargs_19919)
        
        # Applying the binary operator '<' (line 93)
        result_lt_19921 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 11), '<', len_call_result_19915, len_call_result_19920)
        
        # Testing if the type of an if condition is none (line 93)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 93, 8), result_lt_19921):
            pass
        else:
            
            # Testing the type of an if condition (line 93)
            if_condition_19922 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 93, 8), result_lt_19921)
            # Assigning a type to the variable 'if_condition_19922' (line 93)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'if_condition_19922', if_condition_19922)
            # SSA begins for if statement (line 93)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'False' (line 94)
            False_19923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 19), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 94)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'stypy_return_type', False_19923)
            # SSA join for if statement (line 93)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'self' (line 96)
        self_19924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 21), 'self')
        # Obtaining the member 'grouped_types' of a type (line 96)
        grouped_types_19925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 21), self_19924, 'grouped_types')
        # Assigning a type to the variable 'grouped_types_19925' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'grouped_types_19925', grouped_types_19925)
        # Testing if the for loop is going to be iterated (line 96)
        # Testing the type of a for loop iterable (line 96)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 96, 8), grouped_types_19925)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 96, 8), grouped_types_19925):
            # Getting the type of the for loop variable (line 96)
            for_loop_var_19926 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 96, 8), grouped_types_19925)
            # Assigning a type to the variable 'type_' (line 96)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'type_', for_loop_var_19926)
            # SSA begins for a for statement (line 96)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'type_' (line 97)
            type__19927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 15), 'type_')
            # Getting the type of 'other' (line 97)
            other_19928 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 28), 'other')
            # Obtaining the member 'grouped_types' of a type (line 97)
            grouped_types_19929 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 28), other_19928, 'grouped_types')
            # Applying the binary operator 'notin' (line 97)
            result_contains_19930 = python_operator(stypy.reporting.localization.Localization(__file__, 97, 15), 'notin', type__19927, grouped_types_19929)
            
            # Testing if the type of an if condition is none (line 97)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 97, 12), result_contains_19930):
                pass
            else:
                
                # Testing the type of an if condition (line 97)
                if_condition_19931 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 12), result_contains_19930)
                # Assigning a type to the variable 'if_condition_19931' (line 97)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'if_condition_19931', if_condition_19931)
                # SSA begins for if statement (line 97)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'False' (line 98)
                False_19932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 23), 'False')
                # Assigning a type to the variable 'stypy_return_type' (line 98)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 16), 'stypy_return_type', False_19932)
                # SSA join for if statement (line 97)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'True' (line 100)
        True_19933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'stypy_return_type', True_19933)
        
        # ################# End of '__gt__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__gt__' in the type store
        # Getting the type of 'stypy_return_type' (line 86)
        stypy_return_type_19934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19934)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__gt__'
        return stypy_return_type_19934


    @norecursion
    def __lt__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__lt__'
        module_type_store = module_type_store.open_function_context('__lt__', 102, 4, False)
        # Assigning a type to the variable 'self' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'self', type_of_self)
        
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

        str_19935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, (-1)), 'str', '\n        Type group sorting. A type group is less than other type group if contains less types or the types contained\n        in the type group are all contained in the other one. Otherwise, is greater than the other type group.\n        :param other: Another type group\n        :return: bool\n        ')
        
        
        # Call to len(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'self' (line 109)
        self_19937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 15), 'self', False)
        # Obtaining the member 'grouped_types' of a type (line 109)
        grouped_types_19938 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 15), self_19937, 'grouped_types')
        # Processing the call keyword arguments (line 109)
        kwargs_19939 = {}
        # Getting the type of 'len' (line 109)
        len_19936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 11), 'len', False)
        # Calling len(args, kwargs) (line 109)
        len_call_result_19940 = invoke(stypy.reporting.localization.Localization(__file__, 109, 11), len_19936, *[grouped_types_19938], **kwargs_19939)
        
        
        # Call to len(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'other' (line 109)
        other_19942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 41), 'other', False)
        # Obtaining the member 'grouped_types' of a type (line 109)
        grouped_types_19943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 41), other_19942, 'grouped_types')
        # Processing the call keyword arguments (line 109)
        kwargs_19944 = {}
        # Getting the type of 'len' (line 109)
        len_19941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 37), 'len', False)
        # Calling len(args, kwargs) (line 109)
        len_call_result_19945 = invoke(stypy.reporting.localization.Localization(__file__, 109, 37), len_19941, *[grouped_types_19943], **kwargs_19944)
        
        # Applying the binary operator '>' (line 109)
        result_gt_19946 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 11), '>', len_call_result_19940, len_call_result_19945)
        
        # Testing if the type of an if condition is none (line 109)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 109, 8), result_gt_19946):
            pass
        else:
            
            # Testing the type of an if condition (line 109)
            if_condition_19947 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 109, 8), result_gt_19946)
            # Assigning a type to the variable 'if_condition_19947' (line 109)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'if_condition_19947', if_condition_19947)
            # SSA begins for if statement (line 109)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'False' (line 110)
            False_19948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 19), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 110)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'stypy_return_type', False_19948)
            # SSA join for if statement (line 109)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'self' (line 112)
        self_19949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 21), 'self')
        # Obtaining the member 'grouped_types' of a type (line 112)
        grouped_types_19950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 21), self_19949, 'grouped_types')
        # Assigning a type to the variable 'grouped_types_19950' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'grouped_types_19950', grouped_types_19950)
        # Testing if the for loop is going to be iterated (line 112)
        # Testing the type of a for loop iterable (line 112)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 112, 8), grouped_types_19950)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 112, 8), grouped_types_19950):
            # Getting the type of the for loop variable (line 112)
            for_loop_var_19951 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 112, 8), grouped_types_19950)
            # Assigning a type to the variable 'type_' (line 112)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'type_', for_loop_var_19951)
            # SSA begins for a for statement (line 112)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Getting the type of 'type_' (line 113)
            type__19952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 15), 'type_')
            # Getting the type of 'other' (line 113)
            other_19953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 28), 'other')
            # Obtaining the member 'grouped_types' of a type (line 113)
            grouped_types_19954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 28), other_19953, 'grouped_types')
            # Applying the binary operator 'notin' (line 113)
            result_contains_19955 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 15), 'notin', type__19952, grouped_types_19954)
            
            # Testing if the type of an if condition is none (line 113)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 113, 12), result_contains_19955):
                pass
            else:
                
                # Testing the type of an if condition (line 113)
                if_condition_19956 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 113, 12), result_contains_19955)
                # Assigning a type to the variable 'if_condition_19956' (line 113)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'if_condition_19956', if_condition_19956)
                # SSA begins for if statement (line 113)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'False' (line 114)
                False_19957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 23), 'False')
                # Assigning a type to the variable 'stypy_return_type' (line 114)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 16), 'stypy_return_type', False_19957)
                # SSA join for if statement (line 113)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'True' (line 116)
        True_19958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'stypy_return_type', True_19958)
        
        # ################# End of '__lt__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__lt__' in the type store
        # Getting the type of 'stypy_return_type' (line 102)
        stypy_return_type_19959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19959)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__lt__'
        return stypy_return_type_19959


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 118, 4, False)
        # Assigning a type to the variable 'self' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeGroup.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        TypeGroup.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeGroup.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeGroup.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'TypeGroup.stypy__repr__')
        TypeGroup.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        TypeGroup.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeGroup.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeGroup.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeGroup.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeGroup.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeGroup.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeGroup.stypy__repr__', [], None, None, defaults, varargs, kwargs)

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

        str_19960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, (-1)), 'str', '\n        Textual representation of the type group\n        :return: str\n        ')
        
        # Assigning a Attribute to a Name (line 133):
        
        # Call to type(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'self' (line 133)
        self_19962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 23), 'self', False)
        # Processing the call keyword arguments (line 133)
        kwargs_19963 = {}
        # Getting the type of 'type' (line 133)
        type_19961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 18), 'type', False)
        # Calling type(args, kwargs) (line 133)
        type_call_result_19964 = invoke(stypy.reporting.localization.Localization(__file__, 133, 18), type_19961, *[self_19962], **kwargs_19963)
        
        # Obtaining the member '__name__' of a type (line 133)
        name___19965 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 18), type_call_result_19964, '__name__')
        # Assigning a type to the variable 'ret_str' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'ret_str', name___19965)
        # Getting the type of 'ret_str' (line 134)
        ret_str_19966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 15), 'ret_str')
        # Assigning a type to the variable 'stypy_return_type' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'stypy_return_type', ret_str_19966)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 118)
        stypy_return_type_19967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_19967)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_19967


# Assigning a type to the variable 'TypeGroup' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'TypeGroup', TypeGroup)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
