
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from type_groups_copy import *
2: 
3: 
4: class RuleGroupGenerator:
5:     '''
6:     This class is used to generate type group instances from lists of types defined in the TypeGroup object of the
7:     type_groups.py file, using the TypeGroup class as a canvas class to generate them.
8:     '''
9:     rule_group_cache = dict()
10: 
11:     def create_rule_group_class(self, class_name):
12:         '''
13:         Creates a new class named as class_name, with all the members of the TypeGroup class
14:         :param class_name: Name of the new class
15:         :return: A new class, structurally identical to the TypeGroup class. TypeGroup class with the same name can
16:         only be created once. If we try to create one that has been already created, the created one is returned instead
17:         '''
18:         if class_name in self.rule_group_cache.keys():
19:             return self.rule_group_cache[class_name]
20: 
21:         group_class = type(class_name, TypeGroup.__bases__, dict(TypeGroup.__dict__))
22:         instance = group_class(getattr(TypeGroups, class_name))
23:         self.rule_group_cache[class_name] = instance
24: 
25:         return instance
26: 
27:     def create_rule_group_class_list(self, classes_name):
28:         '''
29:         Mass-creation of rule group classes calling the previous method
30:         :param classes_name: List of class names
31:         :return: List of classes
32:         '''
33:         instances = []
34:         for class_name in classes_name:
35:             instance = self.create_rule_group_class(class_name)
36:             instances.append(instance)
37: 
38:         return instances
39: 
40:     def __init__(self):
41:         # self.rule_group_compliance_dict = dict()
42:         # for rule in TypeGroups.get_rule_groups():
43:         #     self.rule_group_compliance_dict[rule] = [False] * eval("len(TypeGroups.{0})".format(rule))
44:         self.added_types = []
45:         self.unclassified_types = []
46: 
47:     #TODO: Delete?
48:     # def add_type(self, type_):
49:     #     if not type_ in self.added_types:
50:     #         self.added_types.append(type_)
51:     #
52:     #     added = False
53:     #     for rule_group in TypeGroups.get_rule_groups():
54:     #         type_list = getattr(TypeGroups, rule_group)
55:     #         if type_ in type_list:
56:     #             for i in range(len(type_list)):
57:     #                 if type_ == type_list[i]:
58:     #                     self.rule_group_compliance_dict[rule_group][i] = True
59:     #                     added = True
60:     #
61:     #     if not added:
62:     #         if not type_ in self.unclassified_types:
63:     #             self.unclassified_types.append(type_)
64:     #
65:     # def get_rule_group(self):
66:     #     ret_rule_group = None
67:     #     added = False
68:     #
69:     #     for (marked_rule_group, marks) in self.rule_group_compliance_dict.items():
70:     #         true_marks = len(filter(lambda x: x == True, marks))
71:     #         if len(getattr(TypeGroups, marked_rule_group)) == true_marks:
72:     #             if ret_rule_group is None:
73:     #                 ret_rule_group = [marked_rule_group]
74:     #             else:
75:     #                 for i in range(len(ret_rule_group)):
76:     #                     if getattr(TypeGroups, marked_rule_group) > getattr(TypeGroups, ret_rule_group[i]):
77:     #                         ret_rule_group[i] = marked_rule_group
78:     #                         added = True
79:     #                     if getattr(TypeGroups, marked_rule_group) < getattr(TypeGroups, ret_rule_group[i]):
80:     #                         added = True
81:     #                 if not added:
82:     #                     ret_rule_group.append(marked_rule_group)
83:     #
84:     #     if ret_rule_group is not None:
85:     #         ret_list = self.create_rule_group_class_list(ret_rule_group)
86:     #         return ret_list + self.unclassified_types
87:     #     else:
88:     #         if len(self.unclassified_types) == 0:
89:     #             return None
90:     #         else:
91:     #             return self.unclassified_types
92:     #
93:     # def is_type_in_rule_group(self, rule_group, type_):
94:     #     return type_ in getattr(TypeGroups, rule_group)
95: 
96: '''
97: TypeGroups composed by collections of types
98: '''
99: RealNumber = RuleGroupGenerator().create_rule_group_class("RealNumber")
100: Number = RuleGroupGenerator().create_rule_group_class("Number")
101: Integer = RuleGroupGenerator().create_rule_group_class("Integer")
102: Str = RuleGroupGenerator().create_rule_group_class("Str")
103: IterableDataStructure = RuleGroupGenerator().create_rule_group_class("IterableDataStructure")
104: IterableObject = RuleGroupGenerator().create_rule_group_class("IterableObject")
105: ByteSequence = RuleGroupGenerator().create_rule_group_class("ByteSequence")
106: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from type_groups_copy import ' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/type_rules_copy/type_groups_copy/')
import_19968 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'type_groups_copy')

if (type(import_19968) is not StypyTypeError):

    if (import_19968 != 'pyd_module'):
        __import__(import_19968)
        sys_modules_19969 = sys.modules[import_19968]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'type_groups_copy', sys_modules_19969.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_19969, sys_modules_19969.module_type_store, module_type_store)
    else:
        from type_groups_copy import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'type_groups_copy', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'type_groups_copy' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'type_groups_copy', import_19968)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/type_rules_copy/type_groups_copy/')

# Declaration of the 'RuleGroupGenerator' class

class RuleGroupGenerator:
    str_19970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, (-1)), 'str', '\n    This class is used to generate type group instances from lists of types defined in the TypeGroup object of the\n    type_groups.py file, using the TypeGroup class as a canvas class to generate them.\n    ')

    @norecursion
    def create_rule_group_class(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'create_rule_group_class'
        module_type_store = module_type_store.open_function_context('create_rule_group_class', 11, 4, False)
        # Assigning a type to the variable 'self' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RuleGroupGenerator.create_rule_group_class.__dict__.__setitem__('stypy_localization', localization)
        RuleGroupGenerator.create_rule_group_class.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RuleGroupGenerator.create_rule_group_class.__dict__.__setitem__('stypy_type_store', module_type_store)
        RuleGroupGenerator.create_rule_group_class.__dict__.__setitem__('stypy_function_name', 'RuleGroupGenerator.create_rule_group_class')
        RuleGroupGenerator.create_rule_group_class.__dict__.__setitem__('stypy_param_names_list', ['class_name'])
        RuleGroupGenerator.create_rule_group_class.__dict__.__setitem__('stypy_varargs_param_name', None)
        RuleGroupGenerator.create_rule_group_class.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RuleGroupGenerator.create_rule_group_class.__dict__.__setitem__('stypy_call_defaults', defaults)
        RuleGroupGenerator.create_rule_group_class.__dict__.__setitem__('stypy_call_varargs', varargs)
        RuleGroupGenerator.create_rule_group_class.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RuleGroupGenerator.create_rule_group_class.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RuleGroupGenerator.create_rule_group_class', ['class_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'create_rule_group_class', localization, ['class_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'create_rule_group_class(...)' code ##################

        str_19971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, (-1)), 'str', '\n        Creates a new class named as class_name, with all the members of the TypeGroup class\n        :param class_name: Name of the new class\n        :return: A new class, structurally identical to the TypeGroup class. TypeGroup class with the same name can\n        only be created once. If we try to create one that has been already created, the created one is returned instead\n        ')
        
        # Getting the type of 'class_name' (line 18)
        class_name_19972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 11), 'class_name')
        
        # Call to keys(...): (line 18)
        # Processing the call keyword arguments (line 18)
        kwargs_19976 = {}
        # Getting the type of 'self' (line 18)
        self_19973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 25), 'self', False)
        # Obtaining the member 'rule_group_cache' of a type (line 18)
        rule_group_cache_19974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 25), self_19973, 'rule_group_cache')
        # Obtaining the member 'keys' of a type (line 18)
        keys_19975 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 25), rule_group_cache_19974, 'keys')
        # Calling keys(args, kwargs) (line 18)
        keys_call_result_19977 = invoke(stypy.reporting.localization.Localization(__file__, 18, 25), keys_19975, *[], **kwargs_19976)
        
        # Applying the binary operator 'in' (line 18)
        result_contains_19978 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 11), 'in', class_name_19972, keys_call_result_19977)
        
        # Testing if the type of an if condition is none (line 18)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 18, 8), result_contains_19978):
            pass
        else:
            
            # Testing the type of an if condition (line 18)
            if_condition_19979 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 18, 8), result_contains_19978)
            # Assigning a type to the variable 'if_condition_19979' (line 18)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'if_condition_19979', if_condition_19979)
            # SSA begins for if statement (line 18)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Obtaining the type of the subscript
            # Getting the type of 'class_name' (line 19)
            class_name_19980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 41), 'class_name')
            # Getting the type of 'self' (line 19)
            self_19981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 19), 'self')
            # Obtaining the member 'rule_group_cache' of a type (line 19)
            rule_group_cache_19982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 19), self_19981, 'rule_group_cache')
            # Obtaining the member '__getitem__' of a type (line 19)
            getitem___19983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 19), rule_group_cache_19982, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 19)
            subscript_call_result_19984 = invoke(stypy.reporting.localization.Localization(__file__, 19, 19), getitem___19983, class_name_19980)
            
            # Assigning a type to the variable 'stypy_return_type' (line 19)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'stypy_return_type', subscript_call_result_19984)
            # SSA join for if statement (line 18)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 21):
        
        # Call to type(...): (line 21)
        # Processing the call arguments (line 21)
        # Getting the type of 'class_name' (line 21)
        class_name_19986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 27), 'class_name', False)
        # Getting the type of 'TypeGroup' (line 21)
        TypeGroup_19987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 39), 'TypeGroup', False)
        # Obtaining the member '__bases__' of a type (line 21)
        bases___19988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 39), TypeGroup_19987, '__bases__')
        
        # Call to dict(...): (line 21)
        # Processing the call arguments (line 21)
        # Getting the type of 'TypeGroup' (line 21)
        TypeGroup_19990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 65), 'TypeGroup', False)
        # Obtaining the member '__dict__' of a type (line 21)
        dict___19991 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 65), TypeGroup_19990, '__dict__')
        # Processing the call keyword arguments (line 21)
        kwargs_19992 = {}
        # Getting the type of 'dict' (line 21)
        dict_19989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 60), 'dict', False)
        # Calling dict(args, kwargs) (line 21)
        dict_call_result_19993 = invoke(stypy.reporting.localization.Localization(__file__, 21, 60), dict_19989, *[dict___19991], **kwargs_19992)
        
        # Processing the call keyword arguments (line 21)
        kwargs_19994 = {}
        # Getting the type of 'type' (line 21)
        type_19985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 22), 'type', False)
        # Calling type(args, kwargs) (line 21)
        type_call_result_19995 = invoke(stypy.reporting.localization.Localization(__file__, 21, 22), type_19985, *[class_name_19986, bases___19988, dict_call_result_19993], **kwargs_19994)
        
        # Assigning a type to the variable 'group_class' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'group_class', type_call_result_19995)
        
        # Assigning a Call to a Name (line 22):
        
        # Call to group_class(...): (line 22)
        # Processing the call arguments (line 22)
        
        # Call to getattr(...): (line 22)
        # Processing the call arguments (line 22)
        # Getting the type of 'TypeGroups' (line 22)
        TypeGroups_19998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 39), 'TypeGroups', False)
        # Getting the type of 'class_name' (line 22)
        class_name_19999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 51), 'class_name', False)
        # Processing the call keyword arguments (line 22)
        kwargs_20000 = {}
        # Getting the type of 'getattr' (line 22)
        getattr_19997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 31), 'getattr', False)
        # Calling getattr(args, kwargs) (line 22)
        getattr_call_result_20001 = invoke(stypy.reporting.localization.Localization(__file__, 22, 31), getattr_19997, *[TypeGroups_19998, class_name_19999], **kwargs_20000)
        
        # Processing the call keyword arguments (line 22)
        kwargs_20002 = {}
        # Getting the type of 'group_class' (line 22)
        group_class_19996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 19), 'group_class', False)
        # Calling group_class(args, kwargs) (line 22)
        group_class_call_result_20003 = invoke(stypy.reporting.localization.Localization(__file__, 22, 19), group_class_19996, *[getattr_call_result_20001], **kwargs_20002)
        
        # Assigning a type to the variable 'instance' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'instance', group_class_call_result_20003)
        
        # Assigning a Name to a Subscript (line 23):
        # Getting the type of 'instance' (line 23)
        instance_20004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 44), 'instance')
        # Getting the type of 'self' (line 23)
        self_20005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'self')
        # Obtaining the member 'rule_group_cache' of a type (line 23)
        rule_group_cache_20006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), self_20005, 'rule_group_cache')
        # Getting the type of 'class_name' (line 23)
        class_name_20007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 30), 'class_name')
        # Storing an element on a container (line 23)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 8), rule_group_cache_20006, (class_name_20007, instance_20004))
        # Getting the type of 'instance' (line 25)
        instance_20008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'instance')
        # Assigning a type to the variable 'stypy_return_type' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'stypy_return_type', instance_20008)
        
        # ################# End of 'create_rule_group_class(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_rule_group_class' in the type store
        # Getting the type of 'stypy_return_type' (line 11)
        stypy_return_type_20009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20009)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_rule_group_class'
        return stypy_return_type_20009


    @norecursion
    def create_rule_group_class_list(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'create_rule_group_class_list'
        module_type_store = module_type_store.open_function_context('create_rule_group_class_list', 27, 4, False)
        # Assigning a type to the variable 'self' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RuleGroupGenerator.create_rule_group_class_list.__dict__.__setitem__('stypy_localization', localization)
        RuleGroupGenerator.create_rule_group_class_list.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RuleGroupGenerator.create_rule_group_class_list.__dict__.__setitem__('stypy_type_store', module_type_store)
        RuleGroupGenerator.create_rule_group_class_list.__dict__.__setitem__('stypy_function_name', 'RuleGroupGenerator.create_rule_group_class_list')
        RuleGroupGenerator.create_rule_group_class_list.__dict__.__setitem__('stypy_param_names_list', ['classes_name'])
        RuleGroupGenerator.create_rule_group_class_list.__dict__.__setitem__('stypy_varargs_param_name', None)
        RuleGroupGenerator.create_rule_group_class_list.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RuleGroupGenerator.create_rule_group_class_list.__dict__.__setitem__('stypy_call_defaults', defaults)
        RuleGroupGenerator.create_rule_group_class_list.__dict__.__setitem__('stypy_call_varargs', varargs)
        RuleGroupGenerator.create_rule_group_class_list.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RuleGroupGenerator.create_rule_group_class_list.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RuleGroupGenerator.create_rule_group_class_list', ['classes_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'create_rule_group_class_list', localization, ['classes_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'create_rule_group_class_list(...)' code ##################

        str_20010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, (-1)), 'str', '\n        Mass-creation of rule group classes calling the previous method\n        :param classes_name: List of class names\n        :return: List of classes\n        ')
        
        # Assigning a List to a Name (line 33):
        
        # Obtaining an instance of the builtin type 'list' (line 33)
        list_20011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 33)
        
        # Assigning a type to the variable 'instances' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'instances', list_20011)
        
        # Getting the type of 'classes_name' (line 34)
        classes_name_20012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 26), 'classes_name')
        # Assigning a type to the variable 'classes_name_20012' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'classes_name_20012', classes_name_20012)
        # Testing if the for loop is going to be iterated (line 34)
        # Testing the type of a for loop iterable (line 34)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 34, 8), classes_name_20012)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 34, 8), classes_name_20012):
            # Getting the type of the for loop variable (line 34)
            for_loop_var_20013 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 34, 8), classes_name_20012)
            # Assigning a type to the variable 'class_name' (line 34)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'class_name', for_loop_var_20013)
            # SSA begins for a for statement (line 34)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 35):
            
            # Call to create_rule_group_class(...): (line 35)
            # Processing the call arguments (line 35)
            # Getting the type of 'class_name' (line 35)
            class_name_20016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 52), 'class_name', False)
            # Processing the call keyword arguments (line 35)
            kwargs_20017 = {}
            # Getting the type of 'self' (line 35)
            self_20014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 23), 'self', False)
            # Obtaining the member 'create_rule_group_class' of a type (line 35)
            create_rule_group_class_20015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 23), self_20014, 'create_rule_group_class')
            # Calling create_rule_group_class(args, kwargs) (line 35)
            create_rule_group_class_call_result_20018 = invoke(stypy.reporting.localization.Localization(__file__, 35, 23), create_rule_group_class_20015, *[class_name_20016], **kwargs_20017)
            
            # Assigning a type to the variable 'instance' (line 35)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'instance', create_rule_group_class_call_result_20018)
            
            # Call to append(...): (line 36)
            # Processing the call arguments (line 36)
            # Getting the type of 'instance' (line 36)
            instance_20021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 29), 'instance', False)
            # Processing the call keyword arguments (line 36)
            kwargs_20022 = {}
            # Getting the type of 'instances' (line 36)
            instances_20019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'instances', False)
            # Obtaining the member 'append' of a type (line 36)
            append_20020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 12), instances_20019, 'append')
            # Calling append(args, kwargs) (line 36)
            append_call_result_20023 = invoke(stypy.reporting.localization.Localization(__file__, 36, 12), append_20020, *[instance_20021], **kwargs_20022)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'instances' (line 38)
        instances_20024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 15), 'instances')
        # Assigning a type to the variable 'stypy_return_type' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'stypy_return_type', instances_20024)
        
        # ################# End of 'create_rule_group_class_list(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_rule_group_class_list' in the type store
        # Getting the type of 'stypy_return_type' (line 27)
        stypy_return_type_20025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_20025)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_rule_group_class_list'
        return stypy_return_type_20025


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 40, 4, False)
        # Assigning a type to the variable 'self' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RuleGroupGenerator.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a List to a Attribute (line 44):
        
        # Obtaining an instance of the builtin type 'list' (line 44)
        list_20026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 44)
        
        # Getting the type of 'self' (line 44)
        self_20027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'self')
        # Setting the type of the member 'added_types' of a type (line 44)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), self_20027, 'added_types', list_20026)
        
        # Assigning a List to a Attribute (line 45):
        
        # Obtaining an instance of the builtin type 'list' (line 45)
        list_20028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 45)
        
        # Getting the type of 'self' (line 45)
        self_20029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'self')
        # Setting the type of the member 'unclassified_types' of a type (line 45)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), self_20029, 'unclassified_types', list_20028)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'RuleGroupGenerator' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'RuleGroupGenerator', RuleGroupGenerator)

# Assigning a Call to a Name (line 9):

# Call to dict(...): (line 9)
# Processing the call keyword arguments (line 9)
kwargs_20031 = {}
# Getting the type of 'dict' (line 9)
dict_20030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 23), 'dict', False)
# Calling dict(args, kwargs) (line 9)
dict_call_result_20032 = invoke(stypy.reporting.localization.Localization(__file__, 9, 23), dict_20030, *[], **kwargs_20031)

# Getting the type of 'RuleGroupGenerator'
RuleGroupGenerator_20033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'RuleGroupGenerator')
# Setting the type of the member 'rule_group_cache' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), RuleGroupGenerator_20033, 'rule_group_cache', dict_call_result_20032)
str_20034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, (-1)), 'str', '\nTypeGroups composed by collections of types\n')

# Assigning a Call to a Name (line 99):

# Call to create_rule_group_class(...): (line 99)
# Processing the call arguments (line 99)
str_20039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 58), 'str', 'RealNumber')
# Processing the call keyword arguments (line 99)
kwargs_20040 = {}

# Call to RuleGroupGenerator(...): (line 99)
# Processing the call keyword arguments (line 99)
kwargs_20036 = {}
# Getting the type of 'RuleGroupGenerator' (line 99)
RuleGroupGenerator_20035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 13), 'RuleGroupGenerator', False)
# Calling RuleGroupGenerator(args, kwargs) (line 99)
RuleGroupGenerator_call_result_20037 = invoke(stypy.reporting.localization.Localization(__file__, 99, 13), RuleGroupGenerator_20035, *[], **kwargs_20036)

# Obtaining the member 'create_rule_group_class' of a type (line 99)
create_rule_group_class_20038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 13), RuleGroupGenerator_call_result_20037, 'create_rule_group_class')
# Calling create_rule_group_class(args, kwargs) (line 99)
create_rule_group_class_call_result_20041 = invoke(stypy.reporting.localization.Localization(__file__, 99, 13), create_rule_group_class_20038, *[str_20039], **kwargs_20040)

# Assigning a type to the variable 'RealNumber' (line 99)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 0), 'RealNumber', create_rule_group_class_call_result_20041)

# Assigning a Call to a Name (line 100):

# Call to create_rule_group_class(...): (line 100)
# Processing the call arguments (line 100)
str_20046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 54), 'str', 'Number')
# Processing the call keyword arguments (line 100)
kwargs_20047 = {}

# Call to RuleGroupGenerator(...): (line 100)
# Processing the call keyword arguments (line 100)
kwargs_20043 = {}
# Getting the type of 'RuleGroupGenerator' (line 100)
RuleGroupGenerator_20042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 9), 'RuleGroupGenerator', False)
# Calling RuleGroupGenerator(args, kwargs) (line 100)
RuleGroupGenerator_call_result_20044 = invoke(stypy.reporting.localization.Localization(__file__, 100, 9), RuleGroupGenerator_20042, *[], **kwargs_20043)

# Obtaining the member 'create_rule_group_class' of a type (line 100)
create_rule_group_class_20045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 9), RuleGroupGenerator_call_result_20044, 'create_rule_group_class')
# Calling create_rule_group_class(args, kwargs) (line 100)
create_rule_group_class_call_result_20048 = invoke(stypy.reporting.localization.Localization(__file__, 100, 9), create_rule_group_class_20045, *[str_20046], **kwargs_20047)

# Assigning a type to the variable 'Number' (line 100)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 0), 'Number', create_rule_group_class_call_result_20048)

# Assigning a Call to a Name (line 101):

# Call to create_rule_group_class(...): (line 101)
# Processing the call arguments (line 101)
str_20053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 55), 'str', 'Integer')
# Processing the call keyword arguments (line 101)
kwargs_20054 = {}

# Call to RuleGroupGenerator(...): (line 101)
# Processing the call keyword arguments (line 101)
kwargs_20050 = {}
# Getting the type of 'RuleGroupGenerator' (line 101)
RuleGroupGenerator_20049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 10), 'RuleGroupGenerator', False)
# Calling RuleGroupGenerator(args, kwargs) (line 101)
RuleGroupGenerator_call_result_20051 = invoke(stypy.reporting.localization.Localization(__file__, 101, 10), RuleGroupGenerator_20049, *[], **kwargs_20050)

# Obtaining the member 'create_rule_group_class' of a type (line 101)
create_rule_group_class_20052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 10), RuleGroupGenerator_call_result_20051, 'create_rule_group_class')
# Calling create_rule_group_class(args, kwargs) (line 101)
create_rule_group_class_call_result_20055 = invoke(stypy.reporting.localization.Localization(__file__, 101, 10), create_rule_group_class_20052, *[str_20053], **kwargs_20054)

# Assigning a type to the variable 'Integer' (line 101)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 0), 'Integer', create_rule_group_class_call_result_20055)

# Assigning a Call to a Name (line 102):

# Call to create_rule_group_class(...): (line 102)
# Processing the call arguments (line 102)
str_20060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 51), 'str', 'Str')
# Processing the call keyword arguments (line 102)
kwargs_20061 = {}

# Call to RuleGroupGenerator(...): (line 102)
# Processing the call keyword arguments (line 102)
kwargs_20057 = {}
# Getting the type of 'RuleGroupGenerator' (line 102)
RuleGroupGenerator_20056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 6), 'RuleGroupGenerator', False)
# Calling RuleGroupGenerator(args, kwargs) (line 102)
RuleGroupGenerator_call_result_20058 = invoke(stypy.reporting.localization.Localization(__file__, 102, 6), RuleGroupGenerator_20056, *[], **kwargs_20057)

# Obtaining the member 'create_rule_group_class' of a type (line 102)
create_rule_group_class_20059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 6), RuleGroupGenerator_call_result_20058, 'create_rule_group_class')
# Calling create_rule_group_class(args, kwargs) (line 102)
create_rule_group_class_call_result_20062 = invoke(stypy.reporting.localization.Localization(__file__, 102, 6), create_rule_group_class_20059, *[str_20060], **kwargs_20061)

# Assigning a type to the variable 'Str' (line 102)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 0), 'Str', create_rule_group_class_call_result_20062)

# Assigning a Call to a Name (line 103):

# Call to create_rule_group_class(...): (line 103)
# Processing the call arguments (line 103)
str_20067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 69), 'str', 'IterableDataStructure')
# Processing the call keyword arguments (line 103)
kwargs_20068 = {}

# Call to RuleGroupGenerator(...): (line 103)
# Processing the call keyword arguments (line 103)
kwargs_20064 = {}
# Getting the type of 'RuleGroupGenerator' (line 103)
RuleGroupGenerator_20063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 24), 'RuleGroupGenerator', False)
# Calling RuleGroupGenerator(args, kwargs) (line 103)
RuleGroupGenerator_call_result_20065 = invoke(stypy.reporting.localization.Localization(__file__, 103, 24), RuleGroupGenerator_20063, *[], **kwargs_20064)

# Obtaining the member 'create_rule_group_class' of a type (line 103)
create_rule_group_class_20066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 24), RuleGroupGenerator_call_result_20065, 'create_rule_group_class')
# Calling create_rule_group_class(args, kwargs) (line 103)
create_rule_group_class_call_result_20069 = invoke(stypy.reporting.localization.Localization(__file__, 103, 24), create_rule_group_class_20066, *[str_20067], **kwargs_20068)

# Assigning a type to the variable 'IterableDataStructure' (line 103)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 0), 'IterableDataStructure', create_rule_group_class_call_result_20069)

# Assigning a Call to a Name (line 104):

# Call to create_rule_group_class(...): (line 104)
# Processing the call arguments (line 104)
str_20074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 62), 'str', 'IterableObject')
# Processing the call keyword arguments (line 104)
kwargs_20075 = {}

# Call to RuleGroupGenerator(...): (line 104)
# Processing the call keyword arguments (line 104)
kwargs_20071 = {}
# Getting the type of 'RuleGroupGenerator' (line 104)
RuleGroupGenerator_20070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 17), 'RuleGroupGenerator', False)
# Calling RuleGroupGenerator(args, kwargs) (line 104)
RuleGroupGenerator_call_result_20072 = invoke(stypy.reporting.localization.Localization(__file__, 104, 17), RuleGroupGenerator_20070, *[], **kwargs_20071)

# Obtaining the member 'create_rule_group_class' of a type (line 104)
create_rule_group_class_20073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 17), RuleGroupGenerator_call_result_20072, 'create_rule_group_class')
# Calling create_rule_group_class(args, kwargs) (line 104)
create_rule_group_class_call_result_20076 = invoke(stypy.reporting.localization.Localization(__file__, 104, 17), create_rule_group_class_20073, *[str_20074], **kwargs_20075)

# Assigning a type to the variable 'IterableObject' (line 104)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'IterableObject', create_rule_group_class_call_result_20076)

# Assigning a Call to a Name (line 105):

# Call to create_rule_group_class(...): (line 105)
# Processing the call arguments (line 105)
str_20081 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 60), 'str', 'ByteSequence')
# Processing the call keyword arguments (line 105)
kwargs_20082 = {}

# Call to RuleGroupGenerator(...): (line 105)
# Processing the call keyword arguments (line 105)
kwargs_20078 = {}
# Getting the type of 'RuleGroupGenerator' (line 105)
RuleGroupGenerator_20077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 15), 'RuleGroupGenerator', False)
# Calling RuleGroupGenerator(args, kwargs) (line 105)
RuleGroupGenerator_call_result_20079 = invoke(stypy.reporting.localization.Localization(__file__, 105, 15), RuleGroupGenerator_20077, *[], **kwargs_20078)

# Obtaining the member 'create_rule_group_class' of a type (line 105)
create_rule_group_class_20080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 15), RuleGroupGenerator_call_result_20079, 'create_rule_group_class')
# Calling create_rule_group_class(args, kwargs) (line 105)
create_rule_group_class_call_result_20083 = invoke(stypy.reporting.localization.Localization(__file__, 105, 15), create_rule_group_class_20080, *[str_20081], **kwargs_20082)

# Assigning a type to the variable 'ByteSequence' (line 105)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), 'ByteSequence', create_rule_group_class_call_result_20083)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
