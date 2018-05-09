
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: class TypeAnnotationRecord:
2:     '''
3:     Class to annotate the types of variables defined in a Python source file. This is used as a record for generating
4:      Python type annotated programs
5:     '''
6:     annotations_per_file = dict()
7: 
8:     @staticmethod
9:     def is_type_changing_method(method_name):
10:         '''
11:         Determines if this method name is bound to a Python method that we know it changes the state of the classes
12:         that define it, to enable type annotations within these methods.
13:         :param method_name: str
14:         :return: bool
15:         '''
16:         return method_name in ["__setitem__",
17:                                "__add__",
18:                                "__setslice__",
19:                                "add",
20:                                "append"]
21: 
22:     # TODO: Remove?
23:     # def __init__(self, function_name):
24:     #     self.function_name = function_name
25:     #     self.annotation_dict = dict()
26: 
27:     # def annotate_type(self, line, col_offset, var_name, type_):
28:     #     if line not in self.annotation_dict.keys():
29:     #         self.annotation_dict[line] = list()
30:     #
31:     #     anottation_tuple = (var_name, type_, col_offset)
32:     #     if not anottation_tuple in self.annotation_dict[line]:
33:     #         self.annotation_dict[line].append(anottation_tuple)
34:     #
35:     # def get_annotations_for_line(self, line):
36:     #     try:
37:     #         return self.annotation_dict[line]
38:     #     except:
39:     #         return None
40:     #
41:     # def clone(self):
42:     #     clone = TypeAnnotationRecord(self.function_name)
43:     #     for (key, value) in self.annotation_dict.items():
44:     #         clone.annotation_dict[key] = value
45:     #
46:     #     return clone
47: 
48:     @staticmethod
49:     def get_instance_for_file(file_name):
50:         '''
51:         Get an instance of this class for the specified file_name. As there can be only one type annotator per file,
52:         this is needed to reuse existing type annotators.
53:         :param file_name: str
54:         :return: TypeAnnotationRecord object
55:         '''
56:         if file_name not in TypeAnnotationRecord.annotations_per_file.keys():
57:             TypeAnnotationRecord.annotations_per_file[file_name] = TypeAnnotationRecord()
58: 
59:         return TypeAnnotationRecord.annotations_per_file[file_name]
60: 
61:     def __init__(self):
62:         '''
63:         Creates a TypeAnnotationRecord object
64:         :return:
65:         '''
66:         self.annotation_dict = dict()
67: 
68:     def annotate_type(self, line, col_offset, var_name, type_):
69:         '''
70:         Annotates a variable type information, including its position
71:         :param line: Source line
72:         :param col_offset: Column inside the source line
73:         :param var_name: Variable name
74:         :param type_: Variable type
75:         :return:
76:         '''
77:         if line not in self.annotation_dict.keys():
78:             self.annotation_dict[line] = list()
79: 
80:         annotation_tuple = (var_name, type_, col_offset)
81:         if annotation_tuple not in self.annotation_dict[line]:
82:             self.annotation_dict[line].append(annotation_tuple)
83: 
84:     def get_annotations_for_line(self, line):
85:         '''
86:         Get all annotations registered for a certain line
87:         :param line: Line number
88:         :return: Annotation list
89:         '''
90:         try:
91:             return self.annotation_dict[line]
92:         except:
93:             return None
94: 
95:     def reset(self):
96:         '''
97:         Remove all type annotations
98:         :return:
99:         '''
100:         self.annotation_dict = dict()
101: 
102:     @staticmethod
103:     def clear_annotations():
104:         '''
105:         Remove all type annotations for all files
106:         :return:
107:         '''
108:         for a in TypeAnnotationRecord.annotations_per_file.values():
109:             a.reset()
110: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

# Declaration of the 'TypeAnnotationRecord' class

class TypeAnnotationRecord:
    str_13612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, (-1)), 'str', '\n    Class to annotate the types of variables defined in a Python source file. This is used as a record for generating\n     Python type annotated programs\n    ')

    @staticmethod
    @norecursion
    def is_type_changing_method(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'is_type_changing_method'
        module_type_store = module_type_store.open_function_context('is_type_changing_method', 8, 4, False)
        
        # Passed parameters checking function
        TypeAnnotationRecord.is_type_changing_method.__dict__.__setitem__('stypy_localization', localization)
        TypeAnnotationRecord.is_type_changing_method.__dict__.__setitem__('stypy_type_of_self', None)
        TypeAnnotationRecord.is_type_changing_method.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeAnnotationRecord.is_type_changing_method.__dict__.__setitem__('stypy_function_name', 'is_type_changing_method')
        TypeAnnotationRecord.is_type_changing_method.__dict__.__setitem__('stypy_param_names_list', ['method_name'])
        TypeAnnotationRecord.is_type_changing_method.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeAnnotationRecord.is_type_changing_method.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeAnnotationRecord.is_type_changing_method.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeAnnotationRecord.is_type_changing_method.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeAnnotationRecord.is_type_changing_method.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeAnnotationRecord.is_type_changing_method.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, None, module_type_store, 'is_type_changing_method', ['method_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'is_type_changing_method', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'is_type_changing_method(...)' code ##################

        str_13613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, (-1)), 'str', '\n        Determines if this method name is bound to a Python method that we know it changes the state of the classes\n        that define it, to enable type annotations within these methods.\n        :param method_name: str\n        :return: bool\n        ')
        
        # Getting the type of 'method_name' (line 16)
        method_name_13614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 15), 'method_name')
        
        # Obtaining an instance of the builtin type 'list' (line 16)
        list_13615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 30), 'list')
        # Adding type elements to the builtin type 'list' instance (line 16)
        # Adding element type (line 16)
        str_13616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 31), 'str', '__setitem__')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 30), list_13615, str_13616)
        # Adding element type (line 16)
        str_13617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 31), 'str', '__add__')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 30), list_13615, str_13617)
        # Adding element type (line 16)
        str_13618 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 31), 'str', '__setslice__')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 30), list_13615, str_13618)
        # Adding element type (line 16)
        str_13619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 31), 'str', 'add')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 30), list_13615, str_13619)
        # Adding element type (line 16)
        str_13620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 31), 'str', 'append')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 30), list_13615, str_13620)
        
        # Applying the binary operator 'in' (line 16)
        result_contains_13621 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 15), 'in', method_name_13614, list_13615)
        
        # Assigning a type to the variable 'stypy_return_type' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'stypy_return_type', result_contains_13621)
        
        # ################# End of 'is_type_changing_method(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'is_type_changing_method' in the type store
        # Getting the type of 'stypy_return_type' (line 8)
        stypy_return_type_13622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13622)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'is_type_changing_method'
        return stypy_return_type_13622


    @staticmethod
    @norecursion
    def get_instance_for_file(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_instance_for_file'
        module_type_store = module_type_store.open_function_context('get_instance_for_file', 48, 4, False)
        
        # Passed parameters checking function
        TypeAnnotationRecord.get_instance_for_file.__dict__.__setitem__('stypy_localization', localization)
        TypeAnnotationRecord.get_instance_for_file.__dict__.__setitem__('stypy_type_of_self', None)
        TypeAnnotationRecord.get_instance_for_file.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeAnnotationRecord.get_instance_for_file.__dict__.__setitem__('stypy_function_name', 'get_instance_for_file')
        TypeAnnotationRecord.get_instance_for_file.__dict__.__setitem__('stypy_param_names_list', ['file_name'])
        TypeAnnotationRecord.get_instance_for_file.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeAnnotationRecord.get_instance_for_file.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeAnnotationRecord.get_instance_for_file.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeAnnotationRecord.get_instance_for_file.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeAnnotationRecord.get_instance_for_file.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeAnnotationRecord.get_instance_for_file.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, None, module_type_store, 'get_instance_for_file', ['file_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_instance_for_file', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_instance_for_file(...)' code ##################

        str_13623 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, (-1)), 'str', '\n        Get an instance of this class for the specified file_name. As there can be only one type annotator per file,\n        this is needed to reuse existing type annotators.\n        :param file_name: str\n        :return: TypeAnnotationRecord object\n        ')
        
        # Getting the type of 'file_name' (line 56)
        file_name_13624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 11), 'file_name')
        
        # Call to keys(...): (line 56)
        # Processing the call keyword arguments (line 56)
        kwargs_13628 = {}
        # Getting the type of 'TypeAnnotationRecord' (line 56)
        TypeAnnotationRecord_13625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 28), 'TypeAnnotationRecord', False)
        # Obtaining the member 'annotations_per_file' of a type (line 56)
        annotations_per_file_13626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 28), TypeAnnotationRecord_13625, 'annotations_per_file')
        # Obtaining the member 'keys' of a type (line 56)
        keys_13627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 28), annotations_per_file_13626, 'keys')
        # Calling keys(args, kwargs) (line 56)
        keys_call_result_13629 = invoke(stypy.reporting.localization.Localization(__file__, 56, 28), keys_13627, *[], **kwargs_13628)
        
        # Applying the binary operator 'notin' (line 56)
        result_contains_13630 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 11), 'notin', file_name_13624, keys_call_result_13629)
        
        # Testing if the type of an if condition is none (line 56)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 56, 8), result_contains_13630):
            pass
        else:
            
            # Testing the type of an if condition (line 56)
            if_condition_13631 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 56, 8), result_contains_13630)
            # Assigning a type to the variable 'if_condition_13631' (line 56)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'if_condition_13631', if_condition_13631)
            # SSA begins for if statement (line 56)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Subscript (line 57):
            
            # Call to TypeAnnotationRecord(...): (line 57)
            # Processing the call keyword arguments (line 57)
            kwargs_13633 = {}
            # Getting the type of 'TypeAnnotationRecord' (line 57)
            TypeAnnotationRecord_13632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 67), 'TypeAnnotationRecord', False)
            # Calling TypeAnnotationRecord(args, kwargs) (line 57)
            TypeAnnotationRecord_call_result_13634 = invoke(stypy.reporting.localization.Localization(__file__, 57, 67), TypeAnnotationRecord_13632, *[], **kwargs_13633)
            
            # Getting the type of 'TypeAnnotationRecord' (line 57)
            TypeAnnotationRecord_13635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'TypeAnnotationRecord')
            # Obtaining the member 'annotations_per_file' of a type (line 57)
            annotations_per_file_13636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 12), TypeAnnotationRecord_13635, 'annotations_per_file')
            # Getting the type of 'file_name' (line 57)
            file_name_13637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 54), 'file_name')
            # Storing an element on a container (line 57)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 57, 12), annotations_per_file_13636, (file_name_13637, TypeAnnotationRecord_call_result_13634))
            # SSA join for if statement (line 56)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Obtaining the type of the subscript
        # Getting the type of 'file_name' (line 59)
        file_name_13638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 57), 'file_name')
        # Getting the type of 'TypeAnnotationRecord' (line 59)
        TypeAnnotationRecord_13639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 15), 'TypeAnnotationRecord')
        # Obtaining the member 'annotations_per_file' of a type (line 59)
        annotations_per_file_13640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 15), TypeAnnotationRecord_13639, 'annotations_per_file')
        # Obtaining the member '__getitem__' of a type (line 59)
        getitem___13641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 15), annotations_per_file_13640, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 59)
        subscript_call_result_13642 = invoke(stypy.reporting.localization.Localization(__file__, 59, 15), getitem___13641, file_name_13638)
        
        # Assigning a type to the variable 'stypy_return_type' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'stypy_return_type', subscript_call_result_13642)
        
        # ################# End of 'get_instance_for_file(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_instance_for_file' in the type store
        # Getting the type of 'stypy_return_type' (line 48)
        stypy_return_type_13643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13643)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_instance_for_file'
        return stypy_return_type_13643


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 61, 4, False)
        # Assigning a type to the variable 'self' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeAnnotationRecord.__init__', [], None, None, defaults, varargs, kwargs)

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

        str_13644 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, (-1)), 'str', '\n        Creates a TypeAnnotationRecord object\n        :return:\n        ')
        
        # Assigning a Call to a Attribute (line 66):
        
        # Call to dict(...): (line 66)
        # Processing the call keyword arguments (line 66)
        kwargs_13646 = {}
        # Getting the type of 'dict' (line 66)
        dict_13645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 31), 'dict', False)
        # Calling dict(args, kwargs) (line 66)
        dict_call_result_13647 = invoke(stypy.reporting.localization.Localization(__file__, 66, 31), dict_13645, *[], **kwargs_13646)
        
        # Getting the type of 'self' (line 66)
        self_13648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'self')
        # Setting the type of the member 'annotation_dict' of a type (line 66)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), self_13648, 'annotation_dict', dict_call_result_13647)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def annotate_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'annotate_type'
        module_type_store = module_type_store.open_function_context('annotate_type', 68, 4, False)
        # Assigning a type to the variable 'self' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeAnnotationRecord.annotate_type.__dict__.__setitem__('stypy_localization', localization)
        TypeAnnotationRecord.annotate_type.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeAnnotationRecord.annotate_type.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeAnnotationRecord.annotate_type.__dict__.__setitem__('stypy_function_name', 'TypeAnnotationRecord.annotate_type')
        TypeAnnotationRecord.annotate_type.__dict__.__setitem__('stypy_param_names_list', ['line', 'col_offset', 'var_name', 'type_'])
        TypeAnnotationRecord.annotate_type.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeAnnotationRecord.annotate_type.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeAnnotationRecord.annotate_type.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeAnnotationRecord.annotate_type.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeAnnotationRecord.annotate_type.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeAnnotationRecord.annotate_type.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeAnnotationRecord.annotate_type', ['line', 'col_offset', 'var_name', 'type_'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'annotate_type', localization, ['line', 'col_offset', 'var_name', 'type_'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'annotate_type(...)' code ##################

        str_13649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, (-1)), 'str', '\n        Annotates a variable type information, including its position\n        :param line: Source line\n        :param col_offset: Column inside the source line\n        :param var_name: Variable name\n        :param type_: Variable type\n        :return:\n        ')
        
        # Getting the type of 'line' (line 77)
        line_13650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 11), 'line')
        
        # Call to keys(...): (line 77)
        # Processing the call keyword arguments (line 77)
        kwargs_13654 = {}
        # Getting the type of 'self' (line 77)
        self_13651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 23), 'self', False)
        # Obtaining the member 'annotation_dict' of a type (line 77)
        annotation_dict_13652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 23), self_13651, 'annotation_dict')
        # Obtaining the member 'keys' of a type (line 77)
        keys_13653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 23), annotation_dict_13652, 'keys')
        # Calling keys(args, kwargs) (line 77)
        keys_call_result_13655 = invoke(stypy.reporting.localization.Localization(__file__, 77, 23), keys_13653, *[], **kwargs_13654)
        
        # Applying the binary operator 'notin' (line 77)
        result_contains_13656 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 11), 'notin', line_13650, keys_call_result_13655)
        
        # Testing if the type of an if condition is none (line 77)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 77, 8), result_contains_13656):
            pass
        else:
            
            # Testing the type of an if condition (line 77)
            if_condition_13657 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 77, 8), result_contains_13656)
            # Assigning a type to the variable 'if_condition_13657' (line 77)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'if_condition_13657', if_condition_13657)
            # SSA begins for if statement (line 77)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Subscript (line 78):
            
            # Call to list(...): (line 78)
            # Processing the call keyword arguments (line 78)
            kwargs_13659 = {}
            # Getting the type of 'list' (line 78)
            list_13658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 41), 'list', False)
            # Calling list(args, kwargs) (line 78)
            list_call_result_13660 = invoke(stypy.reporting.localization.Localization(__file__, 78, 41), list_13658, *[], **kwargs_13659)
            
            # Getting the type of 'self' (line 78)
            self_13661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'self')
            # Obtaining the member 'annotation_dict' of a type (line 78)
            annotation_dict_13662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 12), self_13661, 'annotation_dict')
            # Getting the type of 'line' (line 78)
            line_13663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 33), 'line')
            # Storing an element on a container (line 78)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 12), annotation_dict_13662, (line_13663, list_call_result_13660))
            # SSA join for if statement (line 77)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Tuple to a Name (line 80):
        
        # Obtaining an instance of the builtin type 'tuple' (line 80)
        tuple_13664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 28), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 80)
        # Adding element type (line 80)
        # Getting the type of 'var_name' (line 80)
        var_name_13665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 28), 'var_name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 28), tuple_13664, var_name_13665)
        # Adding element type (line 80)
        # Getting the type of 'type_' (line 80)
        type__13666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 38), 'type_')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 28), tuple_13664, type__13666)
        # Adding element type (line 80)
        # Getting the type of 'col_offset' (line 80)
        col_offset_13667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 45), 'col_offset')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 28), tuple_13664, col_offset_13667)
        
        # Assigning a type to the variable 'annotation_tuple' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'annotation_tuple', tuple_13664)
        
        # Getting the type of 'annotation_tuple' (line 81)
        annotation_tuple_13668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 11), 'annotation_tuple')
        
        # Obtaining the type of the subscript
        # Getting the type of 'line' (line 81)
        line_13669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 56), 'line')
        # Getting the type of 'self' (line 81)
        self_13670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 35), 'self')
        # Obtaining the member 'annotation_dict' of a type (line 81)
        annotation_dict_13671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 35), self_13670, 'annotation_dict')
        # Obtaining the member '__getitem__' of a type (line 81)
        getitem___13672 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 35), annotation_dict_13671, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 81)
        subscript_call_result_13673 = invoke(stypy.reporting.localization.Localization(__file__, 81, 35), getitem___13672, line_13669)
        
        # Applying the binary operator 'notin' (line 81)
        result_contains_13674 = python_operator(stypy.reporting.localization.Localization(__file__, 81, 11), 'notin', annotation_tuple_13668, subscript_call_result_13673)
        
        # Testing if the type of an if condition is none (line 81)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 81, 8), result_contains_13674):
            pass
        else:
            
            # Testing the type of an if condition (line 81)
            if_condition_13675 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 81, 8), result_contains_13674)
            # Assigning a type to the variable 'if_condition_13675' (line 81)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 8), 'if_condition_13675', if_condition_13675)
            # SSA begins for if statement (line 81)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 82)
            # Processing the call arguments (line 82)
            # Getting the type of 'annotation_tuple' (line 82)
            annotation_tuple_13682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 46), 'annotation_tuple', False)
            # Processing the call keyword arguments (line 82)
            kwargs_13683 = {}
            
            # Obtaining the type of the subscript
            # Getting the type of 'line' (line 82)
            line_13676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 33), 'line', False)
            # Getting the type of 'self' (line 82)
            self_13677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'self', False)
            # Obtaining the member 'annotation_dict' of a type (line 82)
            annotation_dict_13678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 12), self_13677, 'annotation_dict')
            # Obtaining the member '__getitem__' of a type (line 82)
            getitem___13679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 12), annotation_dict_13678, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 82)
            subscript_call_result_13680 = invoke(stypy.reporting.localization.Localization(__file__, 82, 12), getitem___13679, line_13676)
            
            # Obtaining the member 'append' of a type (line 82)
            append_13681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 12), subscript_call_result_13680, 'append')
            # Calling append(args, kwargs) (line 82)
            append_call_result_13684 = invoke(stypy.reporting.localization.Localization(__file__, 82, 12), append_13681, *[annotation_tuple_13682], **kwargs_13683)
            
            # SSA join for if statement (line 81)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # ################# End of 'annotate_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'annotate_type' in the type store
        # Getting the type of 'stypy_return_type' (line 68)
        stypy_return_type_13685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13685)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'annotate_type'
        return stypy_return_type_13685


    @norecursion
    def get_annotations_for_line(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_annotations_for_line'
        module_type_store = module_type_store.open_function_context('get_annotations_for_line', 84, 4, False)
        # Assigning a type to the variable 'self' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeAnnotationRecord.get_annotations_for_line.__dict__.__setitem__('stypy_localization', localization)
        TypeAnnotationRecord.get_annotations_for_line.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeAnnotationRecord.get_annotations_for_line.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeAnnotationRecord.get_annotations_for_line.__dict__.__setitem__('stypy_function_name', 'TypeAnnotationRecord.get_annotations_for_line')
        TypeAnnotationRecord.get_annotations_for_line.__dict__.__setitem__('stypy_param_names_list', ['line'])
        TypeAnnotationRecord.get_annotations_for_line.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeAnnotationRecord.get_annotations_for_line.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeAnnotationRecord.get_annotations_for_line.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeAnnotationRecord.get_annotations_for_line.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeAnnotationRecord.get_annotations_for_line.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeAnnotationRecord.get_annotations_for_line.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeAnnotationRecord.get_annotations_for_line', ['line'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_annotations_for_line', localization, ['line'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_annotations_for_line(...)' code ##################

        str_13686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, (-1)), 'str', '\n        Get all annotations registered for a certain line\n        :param line: Line number\n        :return: Annotation list\n        ')
        
        
        # SSA begins for try-except statement (line 90)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Obtaining the type of the subscript
        # Getting the type of 'line' (line 91)
        line_13687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 40), 'line')
        # Getting the type of 'self' (line 91)
        self_13688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 19), 'self')
        # Obtaining the member 'annotation_dict' of a type (line 91)
        annotation_dict_13689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 19), self_13688, 'annotation_dict')
        # Obtaining the member '__getitem__' of a type (line 91)
        getitem___13690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 19), annotation_dict_13689, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 91)
        subscript_call_result_13691 = invoke(stypy.reporting.localization.Localization(__file__, 91, 19), getitem___13690, line_13687)
        
        # Assigning a type to the variable 'stypy_return_type' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 12), 'stypy_return_type', subscript_call_result_13691)
        # SSA branch for the except part of a try statement (line 90)
        # SSA branch for the except '<any exception>' branch of a try statement (line 90)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'None' (line 93)
        None_13692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'stypy_return_type', None_13692)
        # SSA join for try-except statement (line 90)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'get_annotations_for_line(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_annotations_for_line' in the type store
        # Getting the type of 'stypy_return_type' (line 84)
        stypy_return_type_13693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13693)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_annotations_for_line'
        return stypy_return_type_13693


    @norecursion
    def reset(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'reset'
        module_type_store = module_type_store.open_function_context('reset', 95, 4, False)
        # Assigning a type to the variable 'self' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeAnnotationRecord.reset.__dict__.__setitem__('stypy_localization', localization)
        TypeAnnotationRecord.reset.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeAnnotationRecord.reset.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeAnnotationRecord.reset.__dict__.__setitem__('stypy_function_name', 'TypeAnnotationRecord.reset')
        TypeAnnotationRecord.reset.__dict__.__setitem__('stypy_param_names_list', [])
        TypeAnnotationRecord.reset.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeAnnotationRecord.reset.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeAnnotationRecord.reset.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeAnnotationRecord.reset.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeAnnotationRecord.reset.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeAnnotationRecord.reset.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeAnnotationRecord.reset', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'reset', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'reset(...)' code ##################

        str_13694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, (-1)), 'str', '\n        Remove all type annotations\n        :return:\n        ')
        
        # Assigning a Call to a Attribute (line 100):
        
        # Call to dict(...): (line 100)
        # Processing the call keyword arguments (line 100)
        kwargs_13696 = {}
        # Getting the type of 'dict' (line 100)
        dict_13695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 31), 'dict', False)
        # Calling dict(args, kwargs) (line 100)
        dict_call_result_13697 = invoke(stypy.reporting.localization.Localization(__file__, 100, 31), dict_13695, *[], **kwargs_13696)
        
        # Getting the type of 'self' (line 100)
        self_13698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'self')
        # Setting the type of the member 'annotation_dict' of a type (line 100)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), self_13698, 'annotation_dict', dict_call_result_13697)
        
        # ################# End of 'reset(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'reset' in the type store
        # Getting the type of 'stypy_return_type' (line 95)
        stypy_return_type_13699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13699)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'reset'
        return stypy_return_type_13699


    @staticmethod
    @norecursion
    def clear_annotations(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'clear_annotations'
        module_type_store = module_type_store.open_function_context('clear_annotations', 102, 4, False)
        
        # Passed parameters checking function
        TypeAnnotationRecord.clear_annotations.__dict__.__setitem__('stypy_localization', localization)
        TypeAnnotationRecord.clear_annotations.__dict__.__setitem__('stypy_type_of_self', None)
        TypeAnnotationRecord.clear_annotations.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeAnnotationRecord.clear_annotations.__dict__.__setitem__('stypy_function_name', 'clear_annotations')
        TypeAnnotationRecord.clear_annotations.__dict__.__setitem__('stypy_param_names_list', [])
        TypeAnnotationRecord.clear_annotations.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeAnnotationRecord.clear_annotations.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeAnnotationRecord.clear_annotations.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeAnnotationRecord.clear_annotations.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeAnnotationRecord.clear_annotations.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeAnnotationRecord.clear_annotations.__dict__.__setitem__('stypy_declared_arg_number', 0)
        arguments = process_argument_values(localization, None, module_type_store, 'clear_annotations', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'clear_annotations', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'clear_annotations(...)' code ##################

        str_13700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, (-1)), 'str', '\n        Remove all type annotations for all files\n        :return:\n        ')
        
        
        # Call to values(...): (line 108)
        # Processing the call keyword arguments (line 108)
        kwargs_13704 = {}
        # Getting the type of 'TypeAnnotationRecord' (line 108)
        TypeAnnotationRecord_13701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 17), 'TypeAnnotationRecord', False)
        # Obtaining the member 'annotations_per_file' of a type (line 108)
        annotations_per_file_13702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 17), TypeAnnotationRecord_13701, 'annotations_per_file')
        # Obtaining the member 'values' of a type (line 108)
        values_13703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 17), annotations_per_file_13702, 'values')
        # Calling values(args, kwargs) (line 108)
        values_call_result_13705 = invoke(stypy.reporting.localization.Localization(__file__, 108, 17), values_13703, *[], **kwargs_13704)
        
        # Assigning a type to the variable 'values_call_result_13705' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'values_call_result_13705', values_call_result_13705)
        # Testing if the for loop is going to be iterated (line 108)
        # Testing the type of a for loop iterable (line 108)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 108, 8), values_call_result_13705)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 108, 8), values_call_result_13705):
            # Getting the type of the for loop variable (line 108)
            for_loop_var_13706 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 108, 8), values_call_result_13705)
            # Assigning a type to the variable 'a' (line 108)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'a', for_loop_var_13706)
            # SSA begins for a for statement (line 108)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Call to reset(...): (line 109)
            # Processing the call keyword arguments (line 109)
            kwargs_13709 = {}
            # Getting the type of 'a' (line 109)
            a_13707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 12), 'a', False)
            # Obtaining the member 'reset' of a type (line 109)
            reset_13708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 12), a_13707, 'reset')
            # Calling reset(args, kwargs) (line 109)
            reset_call_result_13710 = invoke(stypy.reporting.localization.Localization(__file__, 109, 12), reset_13708, *[], **kwargs_13709)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        
        # ################# End of 'clear_annotations(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'clear_annotations' in the type store
        # Getting the type of 'stypy_return_type' (line 102)
        stypy_return_type_13711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_13711)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'clear_annotations'
        return stypy_return_type_13711


# Assigning a type to the variable 'TypeAnnotationRecord' (line 1)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'TypeAnnotationRecord', TypeAnnotationRecord)

# Assigning a Call to a Name (line 6):

# Call to dict(...): (line 6)
# Processing the call keyword arguments (line 6)
kwargs_13713 = {}
# Getting the type of 'dict' (line 6)
dict_13712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 27), 'dict', False)
# Calling dict(args, kwargs) (line 6)
dict_call_result_13714 = invoke(stypy.reporting.localization.Localization(__file__, 6, 27), dict_13712, *[], **kwargs_13713)

# Getting the type of 'TypeAnnotationRecord'
TypeAnnotationRecord_13715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'TypeAnnotationRecord')
# Setting the type of the member 'annotations_per_file' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), TypeAnnotationRecord_13715, 'annotations_per_file', dict_call_result_13714)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
