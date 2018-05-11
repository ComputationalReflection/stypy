
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import ast
2: import sys
3: import cStringIO
4: 
5: 
6: class PrintVisitor(ast.NodeVisitor):
7:     '''
8:     Pretty-prints an AST node and its children with the following features:
9: 
10:     - Indentation levels
11:     - Non-private properties of each AST Node and their values.
12:     - Highlights line numbers and column offsets in those nodes in which these data is available.
13:     - Highlights representations of some special properties (such as context)
14: 
15:     This visitor is mainly used for debugging purposes, as printing an AST do not have any practical utility to the
16:     stypy end user
17:     '''
18: 
19:     def __init__(self, ast_node, admitted_types=list(), max_output_level=0, indent='    ', output=sys.stdout):
20:         '''
21:         Walk AST tree and print each matched node contents.
22:             ast_node: Root node
23:             admitted_types: Process only node types in the list
24:             max_output_level: limit output to the given depth (0 (default) prints all levels)
25:             indent: Chars to print within indentation levels
26:             output: where to print the ast (stdout as default)
27:         '''
28:         self.current_depth = 0
29:         self.admitted_types = admitted_types
30:         self.max_output_level = max_output_level
31:         self.output = output
32:         self.indent = indent
33:         self.last_line_no = 0
34:         self.last_col_offset = 0
35:         self.visit(ast_node)  # Go!
36: 
37:     def visit(self, node):
38:         nodetype = type(node)
39: 
40:         if len(self.admitted_types) == 0 or nodetype in self.admitted_types:
41:             self.__print_node(node)
42: 
43:         self.current_depth += 1
44: 
45:         if self.max_output_level == 0 or self.current_depth <= self.max_output_level:
46:             self.generic_visit(node)
47: 
48:         self.current_depth -= 1
49: 
50:     def __print_node(self, node):
51:         node_class_name_txt = node.__class__.__name__
52: 
53:         if not hasattr(node, "col_offset"):
54:             col = self.last_col_offset
55:         else:
56:             self.last_col_offset = node.col_offset
57:             col = node.col_offset
58: 
59:         col_offset = ", Column: " + str(col) + "]"
60: 
61:         if not hasattr(node, "lineno"):
62:             line = self.last_line_no
63:         else:
64:             self.last_line_no = node.lineno
65:             line = node.lineno
66: 
67:         lineno = "[Line: " + str(line)
68: 
69:         node_printable_properties = filter(lambda prop: not (prop.startswith("_") or
70:                                                              (prop == "lineno") or (prop == "col_offset")), dir(node))
71: 
72:         printable_properties_txt = reduce(lambda x, y: x + ", " + y + ": " + str(getattr(node, y)),
73:                                           node_printable_properties, "")
74: 
75:         if not (printable_properties_txt == ""):
76:             printable_properties_txt = ": " + printable_properties_txt
77:         else:
78:             node_class_name_txt = "(" + node_class_name_txt + ")"
79: 
80:         txt = " " + node_class_name_txt + printable_properties_txt
81: 
82:         self.output.write(lineno + col_offset + self.indent * self.current_depth + txt + "\n")
83: 
84: 
85: '''
86: Interface with the PrintVisitor from external modules
87: '''
88: 
89: 
90: def print_ast(root_node):
91:     '''
92:     Prints an AST (or any AST node and its children)
93:     :param root_node: Base node to print
94:     :return:
95:     '''
96:     PrintVisitor(root_node)
97: 
98: 
99: def dump_ast(root_node):
100:     '''
101:     Prints an AST (or any AST node and its children) to a string
102:     :param root_node: Base node to print
103:     :return: str
104:     '''
105:     str_output = cStringIO.StringIO()
106:     PrintVisitor(root_node, output=str_output)
107:     txt = str_output.getvalue()
108:     str_output.close()
109: 
110:     return txt
111: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import ast' statement (line 1)
import ast

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'ast', ast, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import sys' statement (line 2)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import cStringIO' statement (line 3)
import cStringIO

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'cStringIO', cStringIO, module_type_store)

# Declaration of the 'PrintVisitor' class
# Getting the type of 'ast' (line 6)
ast_21130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 19), 'ast')
# Obtaining the member 'NodeVisitor' of a type (line 6)
NodeVisitor_21131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 19), ast_21130, 'NodeVisitor')

class PrintVisitor(NodeVisitor_21131, ):
    str_21132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, (-1)), 'str', '\n    Pretty-prints an AST node and its children with the following features:\n\n    - Indentation levels\n    - Non-private properties of each AST Node and their values.\n    - Highlights line numbers and column offsets in those nodes in which these data is available.\n    - Highlights representations of some special properties (such as context)\n\n    This visitor is mainly used for debugging purposes, as printing an AST do not have any practical utility to the\n    stypy end user\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        
        # Call to list(...): (line 19)
        # Processing the call keyword arguments (line 19)
        kwargs_21134 = {}
        # Getting the type of 'list' (line 19)
        list_21133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 48), 'list', False)
        # Calling list(args, kwargs) (line 19)
        list_call_result_21135 = invoke(stypy.reporting.localization.Localization(__file__, 19, 48), list_21133, *[], **kwargs_21134)
        
        int_21136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 73), 'int')
        str_21137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 83), 'str', '    ')
        # Getting the type of 'sys' (line 19)
        sys_21138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 98), 'sys')
        # Obtaining the member 'stdout' of a type (line 19)
        stdout_21139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 98), sys_21138, 'stdout')
        defaults = [list_call_result_21135, int_21136, str_21137, stdout_21139]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 19, 4, False)
        # Assigning a type to the variable 'self' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PrintVisitor.__init__', ['ast_node', 'admitted_types', 'max_output_level', 'indent', 'output'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['ast_node', 'admitted_types', 'max_output_level', 'indent', 'output'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_21140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, (-1)), 'str', '\n        Walk AST tree and print each matched node contents.\n            ast_node: Root node\n            admitted_types: Process only node types in the list\n            max_output_level: limit output to the given depth (0 (default) prints all levels)\n            indent: Chars to print within indentation levels\n            output: where to print the ast (stdout as default)\n        ')
        
        # Assigning a Num to a Attribute (line 28):
        int_21141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 29), 'int')
        # Getting the type of 'self' (line 28)
        self_21142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'self')
        # Setting the type of the member 'current_depth' of a type (line 28)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), self_21142, 'current_depth', int_21141)
        
        # Assigning a Name to a Attribute (line 29):
        # Getting the type of 'admitted_types' (line 29)
        admitted_types_21143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 30), 'admitted_types')
        # Getting the type of 'self' (line 29)
        self_21144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'self')
        # Setting the type of the member 'admitted_types' of a type (line 29)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), self_21144, 'admitted_types', admitted_types_21143)
        
        # Assigning a Name to a Attribute (line 30):
        # Getting the type of 'max_output_level' (line 30)
        max_output_level_21145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 32), 'max_output_level')
        # Getting the type of 'self' (line 30)
        self_21146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'self')
        # Setting the type of the member 'max_output_level' of a type (line 30)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), self_21146, 'max_output_level', max_output_level_21145)
        
        # Assigning a Name to a Attribute (line 31):
        # Getting the type of 'output' (line 31)
        output_21147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 22), 'output')
        # Getting the type of 'self' (line 31)
        self_21148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'self')
        # Setting the type of the member 'output' of a type (line 31)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), self_21148, 'output', output_21147)
        
        # Assigning a Name to a Attribute (line 32):
        # Getting the type of 'indent' (line 32)
        indent_21149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 22), 'indent')
        # Getting the type of 'self' (line 32)
        self_21150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'self')
        # Setting the type of the member 'indent' of a type (line 32)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 8), self_21150, 'indent', indent_21149)
        
        # Assigning a Num to a Attribute (line 33):
        int_21151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 28), 'int')
        # Getting the type of 'self' (line 33)
        self_21152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'self')
        # Setting the type of the member 'last_line_no' of a type (line 33)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), self_21152, 'last_line_no', int_21151)
        
        # Assigning a Num to a Attribute (line 34):
        int_21153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 31), 'int')
        # Getting the type of 'self' (line 34)
        self_21154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self')
        # Setting the type of the member 'last_col_offset' of a type (line 34)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_21154, 'last_col_offset', int_21153)
        
        # Call to visit(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'ast_node' (line 35)
        ast_node_21157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 19), 'ast_node', False)
        # Processing the call keyword arguments (line 35)
        kwargs_21158 = {}
        # Getting the type of 'self' (line 35)
        self_21155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 35)
        visit_21156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), self_21155, 'visit')
        # Calling visit(args, kwargs) (line 35)
        visit_call_result_21159 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), visit_21156, *[ast_node_21157], **kwargs_21158)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def visit(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'visit'
        module_type_store = module_type_store.open_function_context('visit', 37, 4, False)
        # Assigning a type to the variable 'self' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PrintVisitor.visit.__dict__.__setitem__('stypy_localization', localization)
        PrintVisitor.visit.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PrintVisitor.visit.__dict__.__setitem__('stypy_type_store', module_type_store)
        PrintVisitor.visit.__dict__.__setitem__('stypy_function_name', 'PrintVisitor.visit')
        PrintVisitor.visit.__dict__.__setitem__('stypy_param_names_list', ['node'])
        PrintVisitor.visit.__dict__.__setitem__('stypy_varargs_param_name', None)
        PrintVisitor.visit.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PrintVisitor.visit.__dict__.__setitem__('stypy_call_defaults', defaults)
        PrintVisitor.visit.__dict__.__setitem__('stypy_call_varargs', varargs)
        PrintVisitor.visit.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PrintVisitor.visit.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PrintVisitor.visit', ['node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'visit', localization, ['node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'visit(...)' code ##################

        
        # Assigning a Call to a Name (line 38):
        
        # Call to type(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'node' (line 38)
        node_21161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 24), 'node', False)
        # Processing the call keyword arguments (line 38)
        kwargs_21162 = {}
        # Getting the type of 'type' (line 38)
        type_21160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 19), 'type', False)
        # Calling type(args, kwargs) (line 38)
        type_call_result_21163 = invoke(stypy.reporting.localization.Localization(__file__, 38, 19), type_21160, *[node_21161], **kwargs_21162)
        
        # Assigning a type to the variable 'nodetype' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'nodetype', type_call_result_21163)
        
        # Evaluating a boolean operation
        
        
        # Call to len(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'self' (line 40)
        self_21165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 15), 'self', False)
        # Obtaining the member 'admitted_types' of a type (line 40)
        admitted_types_21166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 15), self_21165, 'admitted_types')
        # Processing the call keyword arguments (line 40)
        kwargs_21167 = {}
        # Getting the type of 'len' (line 40)
        len_21164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 11), 'len', False)
        # Calling len(args, kwargs) (line 40)
        len_call_result_21168 = invoke(stypy.reporting.localization.Localization(__file__, 40, 11), len_21164, *[admitted_types_21166], **kwargs_21167)
        
        int_21169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 39), 'int')
        # Applying the binary operator '==' (line 40)
        result_eq_21170 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 11), '==', len_call_result_21168, int_21169)
        
        
        # Getting the type of 'nodetype' (line 40)
        nodetype_21171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 44), 'nodetype')
        # Getting the type of 'self' (line 40)
        self_21172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 56), 'self')
        # Obtaining the member 'admitted_types' of a type (line 40)
        admitted_types_21173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 56), self_21172, 'admitted_types')
        # Applying the binary operator 'in' (line 40)
        result_contains_21174 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 44), 'in', nodetype_21171, admitted_types_21173)
        
        # Applying the binary operator 'or' (line 40)
        result_or_keyword_21175 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 11), 'or', result_eq_21170, result_contains_21174)
        
        # Testing if the type of an if condition is none (line 40)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 40, 8), result_or_keyword_21175):
            pass
        else:
            
            # Testing the type of an if condition (line 40)
            if_condition_21176 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 40, 8), result_or_keyword_21175)
            # Assigning a type to the variable 'if_condition_21176' (line 40)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'if_condition_21176', if_condition_21176)
            # SSA begins for if statement (line 40)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to __print_node(...): (line 41)
            # Processing the call arguments (line 41)
            # Getting the type of 'node' (line 41)
            node_21179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 30), 'node', False)
            # Processing the call keyword arguments (line 41)
            kwargs_21180 = {}
            # Getting the type of 'self' (line 41)
            self_21177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'self', False)
            # Obtaining the member '__print_node' of a type (line 41)
            print_node_21178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 12), self_21177, '__print_node')
            # Calling __print_node(args, kwargs) (line 41)
            print_node_call_result_21181 = invoke(stypy.reporting.localization.Localization(__file__, 41, 12), print_node_21178, *[node_21179], **kwargs_21180)
            
            # SSA join for if statement (line 40)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'self' (line 43)
        self_21182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'self')
        # Obtaining the member 'current_depth' of a type (line 43)
        current_depth_21183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), self_21182, 'current_depth')
        int_21184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 30), 'int')
        # Applying the binary operator '+=' (line 43)
        result_iadd_21185 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 8), '+=', current_depth_21183, int_21184)
        # Getting the type of 'self' (line 43)
        self_21186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'self')
        # Setting the type of the member 'current_depth' of a type (line 43)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), self_21186, 'current_depth', result_iadd_21185)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 45)
        self_21187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 11), 'self')
        # Obtaining the member 'max_output_level' of a type (line 45)
        max_output_level_21188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 11), self_21187, 'max_output_level')
        int_21189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 36), 'int')
        # Applying the binary operator '==' (line 45)
        result_eq_21190 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 11), '==', max_output_level_21188, int_21189)
        
        
        # Getting the type of 'self' (line 45)
        self_21191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 41), 'self')
        # Obtaining the member 'current_depth' of a type (line 45)
        current_depth_21192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 41), self_21191, 'current_depth')
        # Getting the type of 'self' (line 45)
        self_21193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 63), 'self')
        # Obtaining the member 'max_output_level' of a type (line 45)
        max_output_level_21194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 63), self_21193, 'max_output_level')
        # Applying the binary operator '<=' (line 45)
        result_le_21195 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 41), '<=', current_depth_21192, max_output_level_21194)
        
        # Applying the binary operator 'or' (line 45)
        result_or_keyword_21196 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 11), 'or', result_eq_21190, result_le_21195)
        
        # Testing if the type of an if condition is none (line 45)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 45, 8), result_or_keyword_21196):
            pass
        else:
            
            # Testing the type of an if condition (line 45)
            if_condition_21197 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 45, 8), result_or_keyword_21196)
            # Assigning a type to the variable 'if_condition_21197' (line 45)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'if_condition_21197', if_condition_21197)
            # SSA begins for if statement (line 45)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to generic_visit(...): (line 46)
            # Processing the call arguments (line 46)
            # Getting the type of 'node' (line 46)
            node_21200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 31), 'node', False)
            # Processing the call keyword arguments (line 46)
            kwargs_21201 = {}
            # Getting the type of 'self' (line 46)
            self_21198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'self', False)
            # Obtaining the member 'generic_visit' of a type (line 46)
            generic_visit_21199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 12), self_21198, 'generic_visit')
            # Calling generic_visit(args, kwargs) (line 46)
            generic_visit_call_result_21202 = invoke(stypy.reporting.localization.Localization(__file__, 46, 12), generic_visit_21199, *[node_21200], **kwargs_21201)
            
            # SSA join for if statement (line 45)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'self' (line 48)
        self_21203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'self')
        # Obtaining the member 'current_depth' of a type (line 48)
        current_depth_21204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), self_21203, 'current_depth')
        int_21205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 30), 'int')
        # Applying the binary operator '-=' (line 48)
        result_isub_21206 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 8), '-=', current_depth_21204, int_21205)
        # Getting the type of 'self' (line 48)
        self_21207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'self')
        # Setting the type of the member 'current_depth' of a type (line 48)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), self_21207, 'current_depth', result_isub_21206)
        
        
        # ################# End of 'visit(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit' in the type store
        # Getting the type of 'stypy_return_type' (line 37)
        stypy_return_type_21208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21208)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit'
        return stypy_return_type_21208


    @norecursion
    def __print_node(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__print_node'
        module_type_store = module_type_store.open_function_context('__print_node', 50, 4, False)
        # Assigning a type to the variable 'self' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PrintVisitor.__print_node.__dict__.__setitem__('stypy_localization', localization)
        PrintVisitor.__print_node.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PrintVisitor.__print_node.__dict__.__setitem__('stypy_type_store', module_type_store)
        PrintVisitor.__print_node.__dict__.__setitem__('stypy_function_name', 'PrintVisitor.__print_node')
        PrintVisitor.__print_node.__dict__.__setitem__('stypy_param_names_list', ['node'])
        PrintVisitor.__print_node.__dict__.__setitem__('stypy_varargs_param_name', None)
        PrintVisitor.__print_node.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PrintVisitor.__print_node.__dict__.__setitem__('stypy_call_defaults', defaults)
        PrintVisitor.__print_node.__dict__.__setitem__('stypy_call_varargs', varargs)
        PrintVisitor.__print_node.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PrintVisitor.__print_node.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PrintVisitor.__print_node', ['node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__print_node', localization, ['node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__print_node(...)' code ##################

        
        # Assigning a Attribute to a Name (line 51):
        # Getting the type of 'node' (line 51)
        node_21209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 30), 'node')
        # Obtaining the member '__class__' of a type (line 51)
        class___21210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 30), node_21209, '__class__')
        # Obtaining the member '__name__' of a type (line 51)
        name___21211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 30), class___21210, '__name__')
        # Assigning a type to the variable 'node_class_name_txt' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'node_class_name_txt', name___21211)
        
        # Type idiom detected: calculating its left and rigth part (line 53)
        str_21212 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 29), 'str', 'col_offset')
        # Getting the type of 'node' (line 53)
        node_21213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 23), 'node')
        
        (may_be_21214, more_types_in_union_21215) = may_not_provide_member(str_21212, node_21213)

        if may_be_21214:

            if more_types_in_union_21215:
                # Runtime conditional SSA (line 53)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'node' (line 53)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'node', remove_member_provider_from_union(node_21213, 'col_offset'))
            
            # Assigning a Attribute to a Name (line 54):
            # Getting the type of 'self' (line 54)
            self_21216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 18), 'self')
            # Obtaining the member 'last_col_offset' of a type (line 54)
            last_col_offset_21217 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 18), self_21216, 'last_col_offset')
            # Assigning a type to the variable 'col' (line 54)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'col', last_col_offset_21217)

            if more_types_in_union_21215:
                # Runtime conditional SSA for else branch (line 53)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_21214) or more_types_in_union_21215):
            # Assigning a type to the variable 'node' (line 53)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'node', remove_not_member_provider_from_union(node_21213, 'col_offset'))
            
            # Assigning a Attribute to a Attribute (line 56):
            # Getting the type of 'node' (line 56)
            node_21218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 35), 'node')
            # Obtaining the member 'col_offset' of a type (line 56)
            col_offset_21219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 35), node_21218, 'col_offset')
            # Getting the type of 'self' (line 56)
            self_21220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'self')
            # Setting the type of the member 'last_col_offset' of a type (line 56)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 12), self_21220, 'last_col_offset', col_offset_21219)
            
            # Assigning a Attribute to a Name (line 57):
            # Getting the type of 'node' (line 57)
            node_21221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 18), 'node')
            # Obtaining the member 'col_offset' of a type (line 57)
            col_offset_21222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 18), node_21221, 'col_offset')
            # Assigning a type to the variable 'col' (line 57)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'col', col_offset_21222)

            if (may_be_21214 and more_types_in_union_21215):
                # SSA join for if statement (line 53)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a BinOp to a Name (line 59):
        str_21223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 21), 'str', ', Column: ')
        
        # Call to str(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'col' (line 59)
        col_21225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 40), 'col', False)
        # Processing the call keyword arguments (line 59)
        kwargs_21226 = {}
        # Getting the type of 'str' (line 59)
        str_21224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 36), 'str', False)
        # Calling str(args, kwargs) (line 59)
        str_call_result_21227 = invoke(stypy.reporting.localization.Localization(__file__, 59, 36), str_21224, *[col_21225], **kwargs_21226)
        
        # Applying the binary operator '+' (line 59)
        result_add_21228 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 21), '+', str_21223, str_call_result_21227)
        
        str_21229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 47), 'str', ']')
        # Applying the binary operator '+' (line 59)
        result_add_21230 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 45), '+', result_add_21228, str_21229)
        
        # Assigning a type to the variable 'col_offset' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'col_offset', result_add_21230)
        
        # Type idiom detected: calculating its left and rigth part (line 61)
        str_21231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 29), 'str', 'lineno')
        # Getting the type of 'node' (line 61)
        node_21232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 23), 'node')
        
        (may_be_21233, more_types_in_union_21234) = may_not_provide_member(str_21231, node_21232)

        if may_be_21233:

            if more_types_in_union_21234:
                # Runtime conditional SSA (line 61)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'node' (line 61)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'node', remove_member_provider_from_union(node_21232, 'lineno'))
            
            # Assigning a Attribute to a Name (line 62):
            # Getting the type of 'self' (line 62)
            self_21235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 19), 'self')
            # Obtaining the member 'last_line_no' of a type (line 62)
            last_line_no_21236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 19), self_21235, 'last_line_no')
            # Assigning a type to the variable 'line' (line 62)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'line', last_line_no_21236)

            if more_types_in_union_21234:
                # Runtime conditional SSA for else branch (line 61)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_21233) or more_types_in_union_21234):
            # Assigning a type to the variable 'node' (line 61)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'node', remove_not_member_provider_from_union(node_21232, 'lineno'))
            
            # Assigning a Attribute to a Attribute (line 64):
            # Getting the type of 'node' (line 64)
            node_21237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 32), 'node')
            # Obtaining the member 'lineno' of a type (line 64)
            lineno_21238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 32), node_21237, 'lineno')
            # Getting the type of 'self' (line 64)
            self_21239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'self')
            # Setting the type of the member 'last_line_no' of a type (line 64)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 12), self_21239, 'last_line_no', lineno_21238)
            
            # Assigning a Attribute to a Name (line 65):
            # Getting the type of 'node' (line 65)
            node_21240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 19), 'node')
            # Obtaining the member 'lineno' of a type (line 65)
            lineno_21241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 19), node_21240, 'lineno')
            # Assigning a type to the variable 'line' (line 65)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'line', lineno_21241)

            if (may_be_21233 and more_types_in_union_21234):
                # SSA join for if statement (line 61)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a BinOp to a Name (line 67):
        str_21242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 17), 'str', '[Line: ')
        
        # Call to str(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'line' (line 67)
        line_21244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 33), 'line', False)
        # Processing the call keyword arguments (line 67)
        kwargs_21245 = {}
        # Getting the type of 'str' (line 67)
        str_21243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 29), 'str', False)
        # Calling str(args, kwargs) (line 67)
        str_call_result_21246 = invoke(stypy.reporting.localization.Localization(__file__, 67, 29), str_21243, *[line_21244], **kwargs_21245)
        
        # Applying the binary operator '+' (line 67)
        result_add_21247 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 17), '+', str_21242, str_call_result_21246)
        
        # Assigning a type to the variable 'lineno' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'lineno', result_add_21247)
        
        # Assigning a Call to a Name (line 69):
        
        # Call to filter(...): (line 69)
        # Processing the call arguments (line 69)

        @norecursion
        def _stypy_temp_lambda_39(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_39'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_39', 69, 43, True)
            # Passed parameters checking function
            _stypy_temp_lambda_39.stypy_localization = localization
            _stypy_temp_lambda_39.stypy_type_of_self = None
            _stypy_temp_lambda_39.stypy_type_store = module_type_store
            _stypy_temp_lambda_39.stypy_function_name = '_stypy_temp_lambda_39'
            _stypy_temp_lambda_39.stypy_param_names_list = ['prop']
            _stypy_temp_lambda_39.stypy_varargs_param_name = None
            _stypy_temp_lambda_39.stypy_kwargs_param_name = None
            _stypy_temp_lambda_39.stypy_call_defaults = defaults
            _stypy_temp_lambda_39.stypy_call_varargs = varargs
            _stypy_temp_lambda_39.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_39', ['prop'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_39', ['prop'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            
            # Evaluating a boolean operation
            
            # Call to startswith(...): (line 69)
            # Processing the call arguments (line 69)
            str_21251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 77), 'str', '_')
            # Processing the call keyword arguments (line 69)
            kwargs_21252 = {}
            # Getting the type of 'prop' (line 69)
            prop_21249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 61), 'prop', False)
            # Obtaining the member 'startswith' of a type (line 69)
            startswith_21250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 61), prop_21249, 'startswith')
            # Calling startswith(args, kwargs) (line 69)
            startswith_call_result_21253 = invoke(stypy.reporting.localization.Localization(__file__, 69, 61), startswith_21250, *[str_21251], **kwargs_21252)
            
            
            # Getting the type of 'prop' (line 70)
            prop_21254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 62), 'prop', False)
            str_21255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 70), 'str', 'lineno')
            # Applying the binary operator '==' (line 70)
            result_eq_21256 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 62), '==', prop_21254, str_21255)
            
            # Applying the binary operator 'or' (line 69)
            result_or_keyword_21257 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 61), 'or', startswith_call_result_21253, result_eq_21256)
            
            # Getting the type of 'prop' (line 70)
            prop_21258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 84), 'prop', False)
            str_21259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 92), 'str', 'col_offset')
            # Applying the binary operator '==' (line 70)
            result_eq_21260 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 84), '==', prop_21258, str_21259)
            
            # Applying the binary operator 'or' (line 69)
            result_or_keyword_21261 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 61), 'or', result_or_keyword_21257, result_eq_21260)
            
            # Applying the 'not' unary operator (line 69)
            result_not__21262 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 56), 'not', result_or_keyword_21261)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 69)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 43), 'stypy_return_type', result_not__21262)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_39' in the type store
            # Getting the type of 'stypy_return_type' (line 69)
            stypy_return_type_21263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 43), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_21263)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_39'
            return stypy_return_type_21263

        # Assigning a type to the variable '_stypy_temp_lambda_39' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 43), '_stypy_temp_lambda_39', _stypy_temp_lambda_39)
        # Getting the type of '_stypy_temp_lambda_39' (line 69)
        _stypy_temp_lambda_39_21264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 43), '_stypy_temp_lambda_39')
        
        # Call to dir(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'node' (line 70)
        node_21266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 112), 'node', False)
        # Processing the call keyword arguments (line 70)
        kwargs_21267 = {}
        # Getting the type of 'dir' (line 70)
        dir_21265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 108), 'dir', False)
        # Calling dir(args, kwargs) (line 70)
        dir_call_result_21268 = invoke(stypy.reporting.localization.Localization(__file__, 70, 108), dir_21265, *[node_21266], **kwargs_21267)
        
        # Processing the call keyword arguments (line 69)
        kwargs_21269 = {}
        # Getting the type of 'filter' (line 69)
        filter_21248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 36), 'filter', False)
        # Calling filter(args, kwargs) (line 69)
        filter_call_result_21270 = invoke(stypy.reporting.localization.Localization(__file__, 69, 36), filter_21248, *[_stypy_temp_lambda_39_21264, dir_call_result_21268], **kwargs_21269)
        
        # Assigning a type to the variable 'node_printable_properties' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'node_printable_properties', filter_call_result_21270)
        
        # Assigning a Call to a Name (line 72):
        
        # Call to reduce(...): (line 72)
        # Processing the call arguments (line 72)

        @norecursion
        def _stypy_temp_lambda_40(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_40'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_40', 72, 42, True)
            # Passed parameters checking function
            _stypy_temp_lambda_40.stypy_localization = localization
            _stypy_temp_lambda_40.stypy_type_of_self = None
            _stypy_temp_lambda_40.stypy_type_store = module_type_store
            _stypy_temp_lambda_40.stypy_function_name = '_stypy_temp_lambda_40'
            _stypy_temp_lambda_40.stypy_param_names_list = ['x', 'y']
            _stypy_temp_lambda_40.stypy_varargs_param_name = None
            _stypy_temp_lambda_40.stypy_kwargs_param_name = None
            _stypy_temp_lambda_40.stypy_call_defaults = defaults
            _stypy_temp_lambda_40.stypy_call_varargs = varargs
            _stypy_temp_lambda_40.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_40', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_40', ['x', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'x' (line 72)
            x_21272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 55), 'x', False)
            str_21273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 59), 'str', ', ')
            # Applying the binary operator '+' (line 72)
            result_add_21274 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 55), '+', x_21272, str_21273)
            
            # Getting the type of 'y' (line 72)
            y_21275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 66), 'y', False)
            # Applying the binary operator '+' (line 72)
            result_add_21276 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 64), '+', result_add_21274, y_21275)
            
            str_21277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 70), 'str', ': ')
            # Applying the binary operator '+' (line 72)
            result_add_21278 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 68), '+', result_add_21276, str_21277)
            
            
            # Call to str(...): (line 72)
            # Processing the call arguments (line 72)
            
            # Call to getattr(...): (line 72)
            # Processing the call arguments (line 72)
            # Getting the type of 'node' (line 72)
            node_21281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 89), 'node', False)
            # Getting the type of 'y' (line 72)
            y_21282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 95), 'y', False)
            # Processing the call keyword arguments (line 72)
            kwargs_21283 = {}
            # Getting the type of 'getattr' (line 72)
            getattr_21280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 81), 'getattr', False)
            # Calling getattr(args, kwargs) (line 72)
            getattr_call_result_21284 = invoke(stypy.reporting.localization.Localization(__file__, 72, 81), getattr_21280, *[node_21281, y_21282], **kwargs_21283)
            
            # Processing the call keyword arguments (line 72)
            kwargs_21285 = {}
            # Getting the type of 'str' (line 72)
            str_21279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 77), 'str', False)
            # Calling str(args, kwargs) (line 72)
            str_call_result_21286 = invoke(stypy.reporting.localization.Localization(__file__, 72, 77), str_21279, *[getattr_call_result_21284], **kwargs_21285)
            
            # Applying the binary operator '+' (line 72)
            result_add_21287 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 75), '+', result_add_21278, str_call_result_21286)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 72)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 42), 'stypy_return_type', result_add_21287)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_40' in the type store
            # Getting the type of 'stypy_return_type' (line 72)
            stypy_return_type_21288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 42), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_21288)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_40'
            return stypy_return_type_21288

        # Assigning a type to the variable '_stypy_temp_lambda_40' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 42), '_stypy_temp_lambda_40', _stypy_temp_lambda_40)
        # Getting the type of '_stypy_temp_lambda_40' (line 72)
        _stypy_temp_lambda_40_21289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 42), '_stypy_temp_lambda_40')
        # Getting the type of 'node_printable_properties' (line 73)
        node_printable_properties_21290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 42), 'node_printable_properties', False)
        str_21291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 69), 'str', '')
        # Processing the call keyword arguments (line 72)
        kwargs_21292 = {}
        # Getting the type of 'reduce' (line 72)
        reduce_21271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 35), 'reduce', False)
        # Calling reduce(args, kwargs) (line 72)
        reduce_call_result_21293 = invoke(stypy.reporting.localization.Localization(__file__, 72, 35), reduce_21271, *[_stypy_temp_lambda_40_21289, node_printable_properties_21290, str_21291], **kwargs_21292)
        
        # Assigning a type to the variable 'printable_properties_txt' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'printable_properties_txt', reduce_call_result_21293)
        
        
        # Getting the type of 'printable_properties_txt' (line 75)
        printable_properties_txt_21294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 16), 'printable_properties_txt')
        str_21295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 44), 'str', '')
        # Applying the binary operator '==' (line 75)
        result_eq_21296 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 16), '==', printable_properties_txt_21294, str_21295)
        
        # Applying the 'not' unary operator (line 75)
        result_not__21297 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 11), 'not', result_eq_21296)
        
        # Testing if the type of an if condition is none (line 75)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 75, 8), result_not__21297):
            
            # Assigning a BinOp to a Name (line 78):
            str_21302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 34), 'str', '(')
            # Getting the type of 'node_class_name_txt' (line 78)
            node_class_name_txt_21303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 40), 'node_class_name_txt')
            # Applying the binary operator '+' (line 78)
            result_add_21304 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 34), '+', str_21302, node_class_name_txt_21303)
            
            str_21305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 62), 'str', ')')
            # Applying the binary operator '+' (line 78)
            result_add_21306 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 60), '+', result_add_21304, str_21305)
            
            # Assigning a type to the variable 'node_class_name_txt' (line 78)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'node_class_name_txt', result_add_21306)
        else:
            
            # Testing the type of an if condition (line 75)
            if_condition_21298 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 8), result_not__21297)
            # Assigning a type to the variable 'if_condition_21298' (line 75)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'if_condition_21298', if_condition_21298)
            # SSA begins for if statement (line 75)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 76):
            str_21299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 39), 'str', ': ')
            # Getting the type of 'printable_properties_txt' (line 76)
            printable_properties_txt_21300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 46), 'printable_properties_txt')
            # Applying the binary operator '+' (line 76)
            result_add_21301 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 39), '+', str_21299, printable_properties_txt_21300)
            
            # Assigning a type to the variable 'printable_properties_txt' (line 76)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'printable_properties_txt', result_add_21301)
            # SSA branch for the else part of an if statement (line 75)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a BinOp to a Name (line 78):
            str_21302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 34), 'str', '(')
            # Getting the type of 'node_class_name_txt' (line 78)
            node_class_name_txt_21303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 40), 'node_class_name_txt')
            # Applying the binary operator '+' (line 78)
            result_add_21304 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 34), '+', str_21302, node_class_name_txt_21303)
            
            str_21305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 62), 'str', ')')
            # Applying the binary operator '+' (line 78)
            result_add_21306 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 60), '+', result_add_21304, str_21305)
            
            # Assigning a type to the variable 'node_class_name_txt' (line 78)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'node_class_name_txt', result_add_21306)
            # SSA join for if statement (line 75)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a BinOp to a Name (line 80):
        str_21307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 14), 'str', ' ')
        # Getting the type of 'node_class_name_txt' (line 80)
        node_class_name_txt_21308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 20), 'node_class_name_txt')
        # Applying the binary operator '+' (line 80)
        result_add_21309 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 14), '+', str_21307, node_class_name_txt_21308)
        
        # Getting the type of 'printable_properties_txt' (line 80)
        printable_properties_txt_21310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 42), 'printable_properties_txt')
        # Applying the binary operator '+' (line 80)
        result_add_21311 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 40), '+', result_add_21309, printable_properties_txt_21310)
        
        # Assigning a type to the variable 'txt' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'txt', result_add_21311)
        
        # Call to write(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'lineno' (line 82)
        lineno_21315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 26), 'lineno', False)
        # Getting the type of 'col_offset' (line 82)
        col_offset_21316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 35), 'col_offset', False)
        # Applying the binary operator '+' (line 82)
        result_add_21317 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 26), '+', lineno_21315, col_offset_21316)
        
        # Getting the type of 'self' (line 82)
        self_21318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 48), 'self', False)
        # Obtaining the member 'indent' of a type (line 82)
        indent_21319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 48), self_21318, 'indent')
        # Getting the type of 'self' (line 82)
        self_21320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 62), 'self', False)
        # Obtaining the member 'current_depth' of a type (line 82)
        current_depth_21321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 62), self_21320, 'current_depth')
        # Applying the binary operator '*' (line 82)
        result_mul_21322 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 48), '*', indent_21319, current_depth_21321)
        
        # Applying the binary operator '+' (line 82)
        result_add_21323 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 46), '+', result_add_21317, result_mul_21322)
        
        # Getting the type of 'txt' (line 82)
        txt_21324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 83), 'txt', False)
        # Applying the binary operator '+' (line 82)
        result_add_21325 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 81), '+', result_add_21323, txt_21324)
        
        str_21326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 89), 'str', '\n')
        # Applying the binary operator '+' (line 82)
        result_add_21327 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 87), '+', result_add_21325, str_21326)
        
        # Processing the call keyword arguments (line 82)
        kwargs_21328 = {}
        # Getting the type of 'self' (line 82)
        self_21312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'self', False)
        # Obtaining the member 'output' of a type (line 82)
        output_21313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), self_21312, 'output')
        # Obtaining the member 'write' of a type (line 82)
        write_21314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), output_21313, 'write')
        # Calling write(args, kwargs) (line 82)
        write_call_result_21329 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), write_21314, *[result_add_21327], **kwargs_21328)
        
        
        # ################# End of '__print_node(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__print_node' in the type store
        # Getting the type of 'stypy_return_type' (line 50)
        stypy_return_type_21330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21330)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__print_node'
        return stypy_return_type_21330


# Assigning a type to the variable 'PrintVisitor' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'PrintVisitor', PrintVisitor)
str_21331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, (-1)), 'str', '\nInterface with the PrintVisitor from external modules\n')

@norecursion
def print_ast(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'print_ast'
    module_type_store = module_type_store.open_function_context('print_ast', 90, 0, False)
    
    # Passed parameters checking function
    print_ast.stypy_localization = localization
    print_ast.stypy_type_of_self = None
    print_ast.stypy_type_store = module_type_store
    print_ast.stypy_function_name = 'print_ast'
    print_ast.stypy_param_names_list = ['root_node']
    print_ast.stypy_varargs_param_name = None
    print_ast.stypy_kwargs_param_name = None
    print_ast.stypy_call_defaults = defaults
    print_ast.stypy_call_varargs = varargs
    print_ast.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'print_ast', ['root_node'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'print_ast', localization, ['root_node'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'print_ast(...)' code ##################

    str_21332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, (-1)), 'str', '\n    Prints an AST (or any AST node and its children)\n    :param root_node: Base node to print\n    :return:\n    ')
    
    # Call to PrintVisitor(...): (line 96)
    # Processing the call arguments (line 96)
    # Getting the type of 'root_node' (line 96)
    root_node_21334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 17), 'root_node', False)
    # Processing the call keyword arguments (line 96)
    kwargs_21335 = {}
    # Getting the type of 'PrintVisitor' (line 96)
    PrintVisitor_21333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'PrintVisitor', False)
    # Calling PrintVisitor(args, kwargs) (line 96)
    PrintVisitor_call_result_21336 = invoke(stypy.reporting.localization.Localization(__file__, 96, 4), PrintVisitor_21333, *[root_node_21334], **kwargs_21335)
    
    
    # ################# End of 'print_ast(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'print_ast' in the type store
    # Getting the type of 'stypy_return_type' (line 90)
    stypy_return_type_21337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_21337)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'print_ast'
    return stypy_return_type_21337

# Assigning a type to the variable 'print_ast' (line 90)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 0), 'print_ast', print_ast)

@norecursion
def dump_ast(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'dump_ast'
    module_type_store = module_type_store.open_function_context('dump_ast', 99, 0, False)
    
    # Passed parameters checking function
    dump_ast.stypy_localization = localization
    dump_ast.stypy_type_of_self = None
    dump_ast.stypy_type_store = module_type_store
    dump_ast.stypy_function_name = 'dump_ast'
    dump_ast.stypy_param_names_list = ['root_node']
    dump_ast.stypy_varargs_param_name = None
    dump_ast.stypy_kwargs_param_name = None
    dump_ast.stypy_call_defaults = defaults
    dump_ast.stypy_call_varargs = varargs
    dump_ast.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'dump_ast', ['root_node'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'dump_ast', localization, ['root_node'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'dump_ast(...)' code ##################

    str_21338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, (-1)), 'str', '\n    Prints an AST (or any AST node and its children) to a string\n    :param root_node: Base node to print\n    :return: str\n    ')
    
    # Assigning a Call to a Name (line 105):
    
    # Call to StringIO(...): (line 105)
    # Processing the call keyword arguments (line 105)
    kwargs_21341 = {}
    # Getting the type of 'cStringIO' (line 105)
    cStringIO_21339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 17), 'cStringIO', False)
    # Obtaining the member 'StringIO' of a type (line 105)
    StringIO_21340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 17), cStringIO_21339, 'StringIO')
    # Calling StringIO(args, kwargs) (line 105)
    StringIO_call_result_21342 = invoke(stypy.reporting.localization.Localization(__file__, 105, 17), StringIO_21340, *[], **kwargs_21341)
    
    # Assigning a type to the variable 'str_output' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'str_output', StringIO_call_result_21342)
    
    # Call to PrintVisitor(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'root_node' (line 106)
    root_node_21344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 17), 'root_node', False)
    # Processing the call keyword arguments (line 106)
    # Getting the type of 'str_output' (line 106)
    str_output_21345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 35), 'str_output', False)
    keyword_21346 = str_output_21345
    kwargs_21347 = {'output': keyword_21346}
    # Getting the type of 'PrintVisitor' (line 106)
    PrintVisitor_21343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'PrintVisitor', False)
    # Calling PrintVisitor(args, kwargs) (line 106)
    PrintVisitor_call_result_21348 = invoke(stypy.reporting.localization.Localization(__file__, 106, 4), PrintVisitor_21343, *[root_node_21344], **kwargs_21347)
    
    
    # Assigning a Call to a Name (line 107):
    
    # Call to getvalue(...): (line 107)
    # Processing the call keyword arguments (line 107)
    kwargs_21351 = {}
    # Getting the type of 'str_output' (line 107)
    str_output_21349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 10), 'str_output', False)
    # Obtaining the member 'getvalue' of a type (line 107)
    getvalue_21350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 10), str_output_21349, 'getvalue')
    # Calling getvalue(args, kwargs) (line 107)
    getvalue_call_result_21352 = invoke(stypy.reporting.localization.Localization(__file__, 107, 10), getvalue_21350, *[], **kwargs_21351)
    
    # Assigning a type to the variable 'txt' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'txt', getvalue_call_result_21352)
    
    # Call to close(...): (line 108)
    # Processing the call keyword arguments (line 108)
    kwargs_21355 = {}
    # Getting the type of 'str_output' (line 108)
    str_output_21353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'str_output', False)
    # Obtaining the member 'close' of a type (line 108)
    close_21354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 4), str_output_21353, 'close')
    # Calling close(args, kwargs) (line 108)
    close_call_result_21356 = invoke(stypy.reporting.localization.Localization(__file__, 108, 4), close_21354, *[], **kwargs_21355)
    
    # Getting the type of 'txt' (line 110)
    txt_21357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 11), 'txt')
    # Assigning a type to the variable 'stypy_return_type' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'stypy_return_type', txt_21357)
    
    # ################# End of 'dump_ast(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'dump_ast' in the type store
    # Getting the type of 'stypy_return_type' (line 99)
    stypy_return_type_21358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_21358)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'dump_ast'
    return stypy_return_type_21358

# Assigning a type to the variable 'dump_ast' (line 99)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 0), 'dump_ast', dump_ast)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
