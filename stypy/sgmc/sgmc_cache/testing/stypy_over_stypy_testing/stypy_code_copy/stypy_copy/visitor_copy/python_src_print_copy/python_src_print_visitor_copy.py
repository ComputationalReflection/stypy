
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
ast_4549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 19), 'ast')
# Obtaining the member 'NodeVisitor' of a type (line 6)
NodeVisitor_4550 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 6, 19), ast_4549, 'NodeVisitor')

class PrintVisitor(NodeVisitor_4550, ):
    str_4551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, (-1)), 'str', '\n    Pretty-prints an AST node and its children with the following features:\n\n    - Indentation levels\n    - Non-private properties of each AST Node and their values.\n    - Highlights line numbers and column offsets in those nodes in which these data is available.\n    - Highlights representations of some special properties (such as context)\n\n    This visitor is mainly used for debugging purposes, as printing an AST do not have any practical utility to the\n    stypy end user\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        
        # Call to list(...): (line 19)
        # Processing the call keyword arguments (line 19)
        kwargs_4553 = {}
        # Getting the type of 'list' (line 19)
        list_4552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 48), 'list', False)
        # Calling list(args, kwargs) (line 19)
        list_call_result_4554 = invoke(stypy.reporting.localization.Localization(__file__, 19, 48), list_4552, *[], **kwargs_4553)
        
        int_4555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 73), 'int')
        str_4556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 83), 'str', '    ')
        # Getting the type of 'sys' (line 19)
        sys_4557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 98), 'sys')
        # Obtaining the member 'stdout' of a type (line 19)
        stdout_4558 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 98), sys_4557, 'stdout')
        defaults = [list_call_result_4554, int_4555, str_4556, stdout_4558]
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

        str_4559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, (-1)), 'str', '\n        Walk AST tree and print each matched node contents.\n            ast_node: Root node\n            admitted_types: Process only node types in the list\n            max_output_level: limit output to the given depth (0 (default) prints all levels)\n            indent: Chars to print within indentation levels\n            output: where to print the ast (stdout as default)\n        ')
        
        # Assigning a Num to a Attribute (line 28):
        int_4560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 29), 'int')
        # Getting the type of 'self' (line 28)
        self_4561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'self')
        # Setting the type of the member 'current_depth' of a type (line 28)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), self_4561, 'current_depth', int_4560)
        
        # Assigning a Name to a Attribute (line 29):
        # Getting the type of 'admitted_types' (line 29)
        admitted_types_4562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 30), 'admitted_types')
        # Getting the type of 'self' (line 29)
        self_4563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'self')
        # Setting the type of the member 'admitted_types' of a type (line 29)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), self_4563, 'admitted_types', admitted_types_4562)
        
        # Assigning a Name to a Attribute (line 30):
        # Getting the type of 'max_output_level' (line 30)
        max_output_level_4564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 32), 'max_output_level')
        # Getting the type of 'self' (line 30)
        self_4565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'self')
        # Setting the type of the member 'max_output_level' of a type (line 30)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), self_4565, 'max_output_level', max_output_level_4564)
        
        # Assigning a Name to a Attribute (line 31):
        # Getting the type of 'output' (line 31)
        output_4566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 22), 'output')
        # Getting the type of 'self' (line 31)
        self_4567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'self')
        # Setting the type of the member 'output' of a type (line 31)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), self_4567, 'output', output_4566)
        
        # Assigning a Name to a Attribute (line 32):
        # Getting the type of 'indent' (line 32)
        indent_4568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 22), 'indent')
        # Getting the type of 'self' (line 32)
        self_4569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'self')
        # Setting the type of the member 'indent' of a type (line 32)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 8), self_4569, 'indent', indent_4568)
        
        # Assigning a Num to a Attribute (line 33):
        int_4570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 28), 'int')
        # Getting the type of 'self' (line 33)
        self_4571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'self')
        # Setting the type of the member 'last_line_no' of a type (line 33)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), self_4571, 'last_line_no', int_4570)
        
        # Assigning a Num to a Attribute (line 34):
        int_4572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 31), 'int')
        # Getting the type of 'self' (line 34)
        self_4573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self')
        # Setting the type of the member 'last_col_offset' of a type (line 34)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_4573, 'last_col_offset', int_4572)
        
        # Call to visit(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'ast_node' (line 35)
        ast_node_4576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 19), 'ast_node', False)
        # Processing the call keyword arguments (line 35)
        kwargs_4577 = {}
        # Getting the type of 'self' (line 35)
        self_4574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'self', False)
        # Obtaining the member 'visit' of a type (line 35)
        visit_4575 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), self_4574, 'visit')
        # Calling visit(args, kwargs) (line 35)
        visit_call_result_4578 = invoke(stypy.reporting.localization.Localization(__file__, 35, 8), visit_4575, *[ast_node_4576], **kwargs_4577)
        
        
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
        node_4580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 24), 'node', False)
        # Processing the call keyword arguments (line 38)
        kwargs_4581 = {}
        # Getting the type of 'type' (line 38)
        type_4579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 19), 'type', False)
        # Calling type(args, kwargs) (line 38)
        type_call_result_4582 = invoke(stypy.reporting.localization.Localization(__file__, 38, 19), type_4579, *[node_4580], **kwargs_4581)
        
        # Assigning a type to the variable 'nodetype' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'nodetype', type_call_result_4582)
        
        # Evaluating a boolean operation
        
        
        # Call to len(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'self' (line 40)
        self_4584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 15), 'self', False)
        # Obtaining the member 'admitted_types' of a type (line 40)
        admitted_types_4585 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 15), self_4584, 'admitted_types')
        # Processing the call keyword arguments (line 40)
        kwargs_4586 = {}
        # Getting the type of 'len' (line 40)
        len_4583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 11), 'len', False)
        # Calling len(args, kwargs) (line 40)
        len_call_result_4587 = invoke(stypy.reporting.localization.Localization(__file__, 40, 11), len_4583, *[admitted_types_4585], **kwargs_4586)
        
        int_4588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 39), 'int')
        # Applying the binary operator '==' (line 40)
        result_eq_4589 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 11), '==', len_call_result_4587, int_4588)
        
        
        # Getting the type of 'nodetype' (line 40)
        nodetype_4590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 44), 'nodetype')
        # Getting the type of 'self' (line 40)
        self_4591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 56), 'self')
        # Obtaining the member 'admitted_types' of a type (line 40)
        admitted_types_4592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 56), self_4591, 'admitted_types')
        # Applying the binary operator 'in' (line 40)
        result_contains_4593 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 44), 'in', nodetype_4590, admitted_types_4592)
        
        # Applying the binary operator 'or' (line 40)
        result_or_keyword_4594 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 11), 'or', result_eq_4589, result_contains_4593)
        
        # Testing if the type of an if condition is none (line 40)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 40, 8), result_or_keyword_4594):
            pass
        else:
            
            # Testing the type of an if condition (line 40)
            if_condition_4595 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 40, 8), result_or_keyword_4594)
            # Assigning a type to the variable 'if_condition_4595' (line 40)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'if_condition_4595', if_condition_4595)
            # SSA begins for if statement (line 40)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to __print_node(...): (line 41)
            # Processing the call arguments (line 41)
            # Getting the type of 'node' (line 41)
            node_4598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 30), 'node', False)
            # Processing the call keyword arguments (line 41)
            kwargs_4599 = {}
            # Getting the type of 'self' (line 41)
            self_4596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 12), 'self', False)
            # Obtaining the member '__print_node' of a type (line 41)
            print_node_4597 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 12), self_4596, '__print_node')
            # Calling __print_node(args, kwargs) (line 41)
            print_node_call_result_4600 = invoke(stypy.reporting.localization.Localization(__file__, 41, 12), print_node_4597, *[node_4598], **kwargs_4599)
            
            # SSA join for if statement (line 40)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'self' (line 43)
        self_4601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'self')
        # Obtaining the member 'current_depth' of a type (line 43)
        current_depth_4602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), self_4601, 'current_depth')
        int_4603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 30), 'int')
        # Applying the binary operator '+=' (line 43)
        result_iadd_4604 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 8), '+=', current_depth_4602, int_4603)
        # Getting the type of 'self' (line 43)
        self_4605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'self')
        # Setting the type of the member 'current_depth' of a type (line 43)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), self_4605, 'current_depth', result_iadd_4604)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 45)
        self_4606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 11), 'self')
        # Obtaining the member 'max_output_level' of a type (line 45)
        max_output_level_4607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 11), self_4606, 'max_output_level')
        int_4608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 36), 'int')
        # Applying the binary operator '==' (line 45)
        result_eq_4609 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 11), '==', max_output_level_4607, int_4608)
        
        
        # Getting the type of 'self' (line 45)
        self_4610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 41), 'self')
        # Obtaining the member 'current_depth' of a type (line 45)
        current_depth_4611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 41), self_4610, 'current_depth')
        # Getting the type of 'self' (line 45)
        self_4612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 63), 'self')
        # Obtaining the member 'max_output_level' of a type (line 45)
        max_output_level_4613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 63), self_4612, 'max_output_level')
        # Applying the binary operator '<=' (line 45)
        result_le_4614 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 41), '<=', current_depth_4611, max_output_level_4613)
        
        # Applying the binary operator 'or' (line 45)
        result_or_keyword_4615 = python_operator(stypy.reporting.localization.Localization(__file__, 45, 11), 'or', result_eq_4609, result_le_4614)
        
        # Testing if the type of an if condition is none (line 45)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 45, 8), result_or_keyword_4615):
            pass
        else:
            
            # Testing the type of an if condition (line 45)
            if_condition_4616 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 45, 8), result_or_keyword_4615)
            # Assigning a type to the variable 'if_condition_4616' (line 45)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'if_condition_4616', if_condition_4616)
            # SSA begins for if statement (line 45)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to generic_visit(...): (line 46)
            # Processing the call arguments (line 46)
            # Getting the type of 'node' (line 46)
            node_4619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 31), 'node', False)
            # Processing the call keyword arguments (line 46)
            kwargs_4620 = {}
            # Getting the type of 'self' (line 46)
            self_4617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 12), 'self', False)
            # Obtaining the member 'generic_visit' of a type (line 46)
            generic_visit_4618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 12), self_4617, 'generic_visit')
            # Calling generic_visit(args, kwargs) (line 46)
            generic_visit_call_result_4621 = invoke(stypy.reporting.localization.Localization(__file__, 46, 12), generic_visit_4618, *[node_4619], **kwargs_4620)
            
            # SSA join for if statement (line 45)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'self' (line 48)
        self_4622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'self')
        # Obtaining the member 'current_depth' of a type (line 48)
        current_depth_4623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), self_4622, 'current_depth')
        int_4624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 30), 'int')
        # Applying the binary operator '-=' (line 48)
        result_isub_4625 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 8), '-=', current_depth_4623, int_4624)
        # Getting the type of 'self' (line 48)
        self_4626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'self')
        # Setting the type of the member 'current_depth' of a type (line 48)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), self_4626, 'current_depth', result_isub_4625)
        
        
        # ################# End of 'visit(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'visit' in the type store
        # Getting the type of 'stypy_return_type' (line 37)
        stypy_return_type_4627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4627)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'visit'
        return stypy_return_type_4627


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
        node_4628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 30), 'node')
        # Obtaining the member '__class__' of a type (line 51)
        class___4629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 30), node_4628, '__class__')
        # Obtaining the member '__name__' of a type (line 51)
        name___4630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 30), class___4629, '__name__')
        # Assigning a type to the variable 'node_class_name_txt' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'node_class_name_txt', name___4630)
        
        # Type idiom detected: calculating its left and rigth part (line 53)
        str_4631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 29), 'str', 'col_offset')
        # Getting the type of 'node' (line 53)
        node_4632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 23), 'node')
        
        (may_be_4633, more_types_in_union_4634) = may_not_provide_member(str_4631, node_4632)

        if may_be_4633:

            if more_types_in_union_4634:
                # Runtime conditional SSA (line 53)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'node' (line 53)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'node', remove_member_provider_from_union(node_4632, 'col_offset'))
            
            # Assigning a Attribute to a Name (line 54):
            # Getting the type of 'self' (line 54)
            self_4635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 18), 'self')
            # Obtaining the member 'last_col_offset' of a type (line 54)
            last_col_offset_4636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 18), self_4635, 'last_col_offset')
            # Assigning a type to the variable 'col' (line 54)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'col', last_col_offset_4636)

            if more_types_in_union_4634:
                # Runtime conditional SSA for else branch (line 53)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_4633) or more_types_in_union_4634):
            # Assigning a type to the variable 'node' (line 53)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'node', remove_not_member_provider_from_union(node_4632, 'col_offset'))
            
            # Assigning a Attribute to a Attribute (line 56):
            # Getting the type of 'node' (line 56)
            node_4637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 35), 'node')
            # Obtaining the member 'col_offset' of a type (line 56)
            col_offset_4638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 35), node_4637, 'col_offset')
            # Getting the type of 'self' (line 56)
            self_4639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'self')
            # Setting the type of the member 'last_col_offset' of a type (line 56)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 12), self_4639, 'last_col_offset', col_offset_4638)
            
            # Assigning a Attribute to a Name (line 57):
            # Getting the type of 'node' (line 57)
            node_4640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 18), 'node')
            # Obtaining the member 'col_offset' of a type (line 57)
            col_offset_4641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 18), node_4640, 'col_offset')
            # Assigning a type to the variable 'col' (line 57)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'col', col_offset_4641)

            if (may_be_4633 and more_types_in_union_4634):
                # SSA join for if statement (line 53)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a BinOp to a Name (line 59):
        str_4642 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 21), 'str', ', Column: ')
        
        # Call to str(...): (line 59)
        # Processing the call arguments (line 59)
        # Getting the type of 'col' (line 59)
        col_4644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 40), 'col', False)
        # Processing the call keyword arguments (line 59)
        kwargs_4645 = {}
        # Getting the type of 'str' (line 59)
        str_4643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 36), 'str', False)
        # Calling str(args, kwargs) (line 59)
        str_call_result_4646 = invoke(stypy.reporting.localization.Localization(__file__, 59, 36), str_4643, *[col_4644], **kwargs_4645)
        
        # Applying the binary operator '+' (line 59)
        result_add_4647 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 21), '+', str_4642, str_call_result_4646)
        
        str_4648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 47), 'str', ']')
        # Applying the binary operator '+' (line 59)
        result_add_4649 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 45), '+', result_add_4647, str_4648)
        
        # Assigning a type to the variable 'col_offset' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'col_offset', result_add_4649)
        
        # Type idiom detected: calculating its left and rigth part (line 61)
        str_4650 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 29), 'str', 'lineno')
        # Getting the type of 'node' (line 61)
        node_4651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 23), 'node')
        
        (may_be_4652, more_types_in_union_4653) = may_not_provide_member(str_4650, node_4651)

        if may_be_4652:

            if more_types_in_union_4653:
                # Runtime conditional SSA (line 61)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'node' (line 61)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'node', remove_member_provider_from_union(node_4651, 'lineno'))
            
            # Assigning a Attribute to a Name (line 62):
            # Getting the type of 'self' (line 62)
            self_4654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 19), 'self')
            # Obtaining the member 'last_line_no' of a type (line 62)
            last_line_no_4655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 19), self_4654, 'last_line_no')
            # Assigning a type to the variable 'line' (line 62)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'line', last_line_no_4655)

            if more_types_in_union_4653:
                # Runtime conditional SSA for else branch (line 61)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_4652) or more_types_in_union_4653):
            # Assigning a type to the variable 'node' (line 61)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'node', remove_not_member_provider_from_union(node_4651, 'lineno'))
            
            # Assigning a Attribute to a Attribute (line 64):
            # Getting the type of 'node' (line 64)
            node_4656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 32), 'node')
            # Obtaining the member 'lineno' of a type (line 64)
            lineno_4657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 32), node_4656, 'lineno')
            # Getting the type of 'self' (line 64)
            self_4658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 12), 'self')
            # Setting the type of the member 'last_line_no' of a type (line 64)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 12), self_4658, 'last_line_no', lineno_4657)
            
            # Assigning a Attribute to a Name (line 65):
            # Getting the type of 'node' (line 65)
            node_4659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 19), 'node')
            # Obtaining the member 'lineno' of a type (line 65)
            lineno_4660 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 19), node_4659, 'lineno')
            # Assigning a type to the variable 'line' (line 65)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'line', lineno_4660)

            if (may_be_4652 and more_types_in_union_4653):
                # SSA join for if statement (line 61)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a BinOp to a Name (line 67):
        str_4661 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 17), 'str', '[Line: ')
        
        # Call to str(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'line' (line 67)
        line_4663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 33), 'line', False)
        # Processing the call keyword arguments (line 67)
        kwargs_4664 = {}
        # Getting the type of 'str' (line 67)
        str_4662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 29), 'str', False)
        # Calling str(args, kwargs) (line 67)
        str_call_result_4665 = invoke(stypy.reporting.localization.Localization(__file__, 67, 29), str_4662, *[line_4663], **kwargs_4664)
        
        # Applying the binary operator '+' (line 67)
        result_add_4666 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 17), '+', str_4661, str_call_result_4665)
        
        # Assigning a type to the variable 'lineno' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'lineno', result_add_4666)
        
        # Assigning a Call to a Name (line 69):
        
        # Call to filter(...): (line 69)
        # Processing the call arguments (line 69)

        @norecursion
        def _stypy_temp_lambda_15(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_15'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_15', 69, 43, True)
            # Passed parameters checking function
            _stypy_temp_lambda_15.stypy_localization = localization
            _stypy_temp_lambda_15.stypy_type_of_self = None
            _stypy_temp_lambda_15.stypy_type_store = module_type_store
            _stypy_temp_lambda_15.stypy_function_name = '_stypy_temp_lambda_15'
            _stypy_temp_lambda_15.stypy_param_names_list = ['prop']
            _stypy_temp_lambda_15.stypy_varargs_param_name = None
            _stypy_temp_lambda_15.stypy_kwargs_param_name = None
            _stypy_temp_lambda_15.stypy_call_defaults = defaults
            _stypy_temp_lambda_15.stypy_call_varargs = varargs
            _stypy_temp_lambda_15.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_15', ['prop'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_15', ['prop'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            
            # Evaluating a boolean operation
            
            # Call to startswith(...): (line 69)
            # Processing the call arguments (line 69)
            str_4670 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 77), 'str', '_')
            # Processing the call keyword arguments (line 69)
            kwargs_4671 = {}
            # Getting the type of 'prop' (line 69)
            prop_4668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 61), 'prop', False)
            # Obtaining the member 'startswith' of a type (line 69)
            startswith_4669 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 61), prop_4668, 'startswith')
            # Calling startswith(args, kwargs) (line 69)
            startswith_call_result_4672 = invoke(stypy.reporting.localization.Localization(__file__, 69, 61), startswith_4669, *[str_4670], **kwargs_4671)
            
            
            # Getting the type of 'prop' (line 70)
            prop_4673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 62), 'prop', False)
            str_4674 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 70), 'str', 'lineno')
            # Applying the binary operator '==' (line 70)
            result_eq_4675 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 62), '==', prop_4673, str_4674)
            
            # Applying the binary operator 'or' (line 69)
            result_or_keyword_4676 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 61), 'or', startswith_call_result_4672, result_eq_4675)
            
            # Getting the type of 'prop' (line 70)
            prop_4677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 84), 'prop', False)
            str_4678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 92), 'str', 'col_offset')
            # Applying the binary operator '==' (line 70)
            result_eq_4679 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 84), '==', prop_4677, str_4678)
            
            # Applying the binary operator 'or' (line 69)
            result_or_keyword_4680 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 61), 'or', result_or_keyword_4676, result_eq_4679)
            
            # Applying the 'not' unary operator (line 69)
            result_not__4681 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 56), 'not', result_or_keyword_4680)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 69)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 43), 'stypy_return_type', result_not__4681)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_15' in the type store
            # Getting the type of 'stypy_return_type' (line 69)
            stypy_return_type_4682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 43), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_4682)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_15'
            return stypy_return_type_4682

        # Assigning a type to the variable '_stypy_temp_lambda_15' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 43), '_stypy_temp_lambda_15', _stypy_temp_lambda_15)
        # Getting the type of '_stypy_temp_lambda_15' (line 69)
        _stypy_temp_lambda_15_4683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 43), '_stypy_temp_lambda_15')
        
        # Call to dir(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'node' (line 70)
        node_4685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 112), 'node', False)
        # Processing the call keyword arguments (line 70)
        kwargs_4686 = {}
        # Getting the type of 'dir' (line 70)
        dir_4684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 108), 'dir', False)
        # Calling dir(args, kwargs) (line 70)
        dir_call_result_4687 = invoke(stypy.reporting.localization.Localization(__file__, 70, 108), dir_4684, *[node_4685], **kwargs_4686)
        
        # Processing the call keyword arguments (line 69)
        kwargs_4688 = {}
        # Getting the type of 'filter' (line 69)
        filter_4667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 36), 'filter', False)
        # Calling filter(args, kwargs) (line 69)
        filter_call_result_4689 = invoke(stypy.reporting.localization.Localization(__file__, 69, 36), filter_4667, *[_stypy_temp_lambda_15_4683, dir_call_result_4687], **kwargs_4688)
        
        # Assigning a type to the variable 'node_printable_properties' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'node_printable_properties', filter_call_result_4689)
        
        # Assigning a Call to a Name (line 72):
        
        # Call to reduce(...): (line 72)
        # Processing the call arguments (line 72)

        @norecursion
        def _stypy_temp_lambda_16(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_16'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_16', 72, 42, True)
            # Passed parameters checking function
            _stypy_temp_lambda_16.stypy_localization = localization
            _stypy_temp_lambda_16.stypy_type_of_self = None
            _stypy_temp_lambda_16.stypy_type_store = module_type_store
            _stypy_temp_lambda_16.stypy_function_name = '_stypy_temp_lambda_16'
            _stypy_temp_lambda_16.stypy_param_names_list = ['x', 'y']
            _stypy_temp_lambda_16.stypy_varargs_param_name = None
            _stypy_temp_lambda_16.stypy_kwargs_param_name = None
            _stypy_temp_lambda_16.stypy_call_defaults = defaults
            _stypy_temp_lambda_16.stypy_call_varargs = varargs
            _stypy_temp_lambda_16.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_16', ['x', 'y'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_16', ['x', 'y'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'x' (line 72)
            x_4691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 55), 'x', False)
            str_4692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 59), 'str', ', ')
            # Applying the binary operator '+' (line 72)
            result_add_4693 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 55), '+', x_4691, str_4692)
            
            # Getting the type of 'y' (line 72)
            y_4694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 66), 'y', False)
            # Applying the binary operator '+' (line 72)
            result_add_4695 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 64), '+', result_add_4693, y_4694)
            
            str_4696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 70), 'str', ': ')
            # Applying the binary operator '+' (line 72)
            result_add_4697 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 68), '+', result_add_4695, str_4696)
            
            
            # Call to str(...): (line 72)
            # Processing the call arguments (line 72)
            
            # Call to getattr(...): (line 72)
            # Processing the call arguments (line 72)
            # Getting the type of 'node' (line 72)
            node_4700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 89), 'node', False)
            # Getting the type of 'y' (line 72)
            y_4701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 95), 'y', False)
            # Processing the call keyword arguments (line 72)
            kwargs_4702 = {}
            # Getting the type of 'getattr' (line 72)
            getattr_4699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 81), 'getattr', False)
            # Calling getattr(args, kwargs) (line 72)
            getattr_call_result_4703 = invoke(stypy.reporting.localization.Localization(__file__, 72, 81), getattr_4699, *[node_4700, y_4701], **kwargs_4702)
            
            # Processing the call keyword arguments (line 72)
            kwargs_4704 = {}
            # Getting the type of 'str' (line 72)
            str_4698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 77), 'str', False)
            # Calling str(args, kwargs) (line 72)
            str_call_result_4705 = invoke(stypy.reporting.localization.Localization(__file__, 72, 77), str_4698, *[getattr_call_result_4703], **kwargs_4704)
            
            # Applying the binary operator '+' (line 72)
            result_add_4706 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 75), '+', result_add_4697, str_call_result_4705)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 72)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 42), 'stypy_return_type', result_add_4706)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_16' in the type store
            # Getting the type of 'stypy_return_type' (line 72)
            stypy_return_type_4707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 42), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_4707)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_16'
            return stypy_return_type_4707

        # Assigning a type to the variable '_stypy_temp_lambda_16' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 42), '_stypy_temp_lambda_16', _stypy_temp_lambda_16)
        # Getting the type of '_stypy_temp_lambda_16' (line 72)
        _stypy_temp_lambda_16_4708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 42), '_stypy_temp_lambda_16')
        # Getting the type of 'node_printable_properties' (line 73)
        node_printable_properties_4709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 42), 'node_printable_properties', False)
        str_4710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 69), 'str', '')
        # Processing the call keyword arguments (line 72)
        kwargs_4711 = {}
        # Getting the type of 'reduce' (line 72)
        reduce_4690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 35), 'reduce', False)
        # Calling reduce(args, kwargs) (line 72)
        reduce_call_result_4712 = invoke(stypy.reporting.localization.Localization(__file__, 72, 35), reduce_4690, *[_stypy_temp_lambda_16_4708, node_printable_properties_4709, str_4710], **kwargs_4711)
        
        # Assigning a type to the variable 'printable_properties_txt' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'printable_properties_txt', reduce_call_result_4712)
        
        
        # Getting the type of 'printable_properties_txt' (line 75)
        printable_properties_txt_4713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 16), 'printable_properties_txt')
        str_4714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 44), 'str', '')
        # Applying the binary operator '==' (line 75)
        result_eq_4715 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 16), '==', printable_properties_txt_4713, str_4714)
        
        # Applying the 'not' unary operator (line 75)
        result_not__4716 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 11), 'not', result_eq_4715)
        
        # Testing if the type of an if condition is none (line 75)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 75, 8), result_not__4716):
            
            # Assigning a BinOp to a Name (line 78):
            str_4721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 34), 'str', '(')
            # Getting the type of 'node_class_name_txt' (line 78)
            node_class_name_txt_4722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 40), 'node_class_name_txt')
            # Applying the binary operator '+' (line 78)
            result_add_4723 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 34), '+', str_4721, node_class_name_txt_4722)
            
            str_4724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 62), 'str', ')')
            # Applying the binary operator '+' (line 78)
            result_add_4725 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 60), '+', result_add_4723, str_4724)
            
            # Assigning a type to the variable 'node_class_name_txt' (line 78)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'node_class_name_txt', result_add_4725)
        else:
            
            # Testing the type of an if condition (line 75)
            if_condition_4717 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 8), result_not__4716)
            # Assigning a type to the variable 'if_condition_4717' (line 75)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'if_condition_4717', if_condition_4717)
            # SSA begins for if statement (line 75)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 76):
            str_4718 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 39), 'str', ': ')
            # Getting the type of 'printable_properties_txt' (line 76)
            printable_properties_txt_4719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 46), 'printable_properties_txt')
            # Applying the binary operator '+' (line 76)
            result_add_4720 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 39), '+', str_4718, printable_properties_txt_4719)
            
            # Assigning a type to the variable 'printable_properties_txt' (line 76)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'printable_properties_txt', result_add_4720)
            # SSA branch for the else part of an if statement (line 75)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a BinOp to a Name (line 78):
            str_4721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 34), 'str', '(')
            # Getting the type of 'node_class_name_txt' (line 78)
            node_class_name_txt_4722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 40), 'node_class_name_txt')
            # Applying the binary operator '+' (line 78)
            result_add_4723 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 34), '+', str_4721, node_class_name_txt_4722)
            
            str_4724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 62), 'str', ')')
            # Applying the binary operator '+' (line 78)
            result_add_4725 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 60), '+', result_add_4723, str_4724)
            
            # Assigning a type to the variable 'node_class_name_txt' (line 78)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 12), 'node_class_name_txt', result_add_4725)
            # SSA join for if statement (line 75)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a BinOp to a Name (line 80):
        str_4726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 14), 'str', ' ')
        # Getting the type of 'node_class_name_txt' (line 80)
        node_class_name_txt_4727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 20), 'node_class_name_txt')
        # Applying the binary operator '+' (line 80)
        result_add_4728 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 14), '+', str_4726, node_class_name_txt_4727)
        
        # Getting the type of 'printable_properties_txt' (line 80)
        printable_properties_txt_4729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 42), 'printable_properties_txt')
        # Applying the binary operator '+' (line 80)
        result_add_4730 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 40), '+', result_add_4728, printable_properties_txt_4729)
        
        # Assigning a type to the variable 'txt' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'txt', result_add_4730)
        
        # Call to write(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'lineno' (line 82)
        lineno_4734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 26), 'lineno', False)
        # Getting the type of 'col_offset' (line 82)
        col_offset_4735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 35), 'col_offset', False)
        # Applying the binary operator '+' (line 82)
        result_add_4736 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 26), '+', lineno_4734, col_offset_4735)
        
        # Getting the type of 'self' (line 82)
        self_4737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 48), 'self', False)
        # Obtaining the member 'indent' of a type (line 82)
        indent_4738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 48), self_4737, 'indent')
        # Getting the type of 'self' (line 82)
        self_4739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 62), 'self', False)
        # Obtaining the member 'current_depth' of a type (line 82)
        current_depth_4740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 62), self_4739, 'current_depth')
        # Applying the binary operator '*' (line 82)
        result_mul_4741 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 48), '*', indent_4738, current_depth_4740)
        
        # Applying the binary operator '+' (line 82)
        result_add_4742 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 46), '+', result_add_4736, result_mul_4741)
        
        # Getting the type of 'txt' (line 82)
        txt_4743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 83), 'txt', False)
        # Applying the binary operator '+' (line 82)
        result_add_4744 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 81), '+', result_add_4742, txt_4743)
        
        str_4745 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 89), 'str', '\n')
        # Applying the binary operator '+' (line 82)
        result_add_4746 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 87), '+', result_add_4744, str_4745)
        
        # Processing the call keyword arguments (line 82)
        kwargs_4747 = {}
        # Getting the type of 'self' (line 82)
        self_4731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'self', False)
        # Obtaining the member 'output' of a type (line 82)
        output_4732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), self_4731, 'output')
        # Obtaining the member 'write' of a type (line 82)
        write_4733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 8), output_4732, 'write')
        # Calling write(args, kwargs) (line 82)
        write_call_result_4748 = invoke(stypy.reporting.localization.Localization(__file__, 82, 8), write_4733, *[result_add_4746], **kwargs_4747)
        
        
        # ################# End of '__print_node(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__print_node' in the type store
        # Getting the type of 'stypy_return_type' (line 50)
        stypy_return_type_4749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_4749)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__print_node'
        return stypy_return_type_4749


# Assigning a type to the variable 'PrintVisitor' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'PrintVisitor', PrintVisitor)
str_4750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, (-1)), 'str', '\nInterface with the PrintVisitor from external modules\n')

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

    str_4751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, (-1)), 'str', '\n    Prints an AST (or any AST node and its children)\n    :param root_node: Base node to print\n    :return:\n    ')
    
    # Call to PrintVisitor(...): (line 96)
    # Processing the call arguments (line 96)
    # Getting the type of 'root_node' (line 96)
    root_node_4753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 17), 'root_node', False)
    # Processing the call keyword arguments (line 96)
    kwargs_4754 = {}
    # Getting the type of 'PrintVisitor' (line 96)
    PrintVisitor_4752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'PrintVisitor', False)
    # Calling PrintVisitor(args, kwargs) (line 96)
    PrintVisitor_call_result_4755 = invoke(stypy.reporting.localization.Localization(__file__, 96, 4), PrintVisitor_4752, *[root_node_4753], **kwargs_4754)
    
    
    # ################# End of 'print_ast(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'print_ast' in the type store
    # Getting the type of 'stypy_return_type' (line 90)
    stypy_return_type_4756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4756)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'print_ast'
    return stypy_return_type_4756

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

    str_4757 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, (-1)), 'str', '\n    Prints an AST (or any AST node and its children) to a string\n    :param root_node: Base node to print\n    :return: str\n    ')
    
    # Assigning a Call to a Name (line 105):
    
    # Call to StringIO(...): (line 105)
    # Processing the call keyword arguments (line 105)
    kwargs_4760 = {}
    # Getting the type of 'cStringIO' (line 105)
    cStringIO_4758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 17), 'cStringIO', False)
    # Obtaining the member 'StringIO' of a type (line 105)
    StringIO_4759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 17), cStringIO_4758, 'StringIO')
    # Calling StringIO(args, kwargs) (line 105)
    StringIO_call_result_4761 = invoke(stypy.reporting.localization.Localization(__file__, 105, 17), StringIO_4759, *[], **kwargs_4760)
    
    # Assigning a type to the variable 'str_output' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'str_output', StringIO_call_result_4761)
    
    # Call to PrintVisitor(...): (line 106)
    # Processing the call arguments (line 106)
    # Getting the type of 'root_node' (line 106)
    root_node_4763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 17), 'root_node', False)
    # Processing the call keyword arguments (line 106)
    # Getting the type of 'str_output' (line 106)
    str_output_4764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 35), 'str_output', False)
    keyword_4765 = str_output_4764
    kwargs_4766 = {'output': keyword_4765}
    # Getting the type of 'PrintVisitor' (line 106)
    PrintVisitor_4762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'PrintVisitor', False)
    # Calling PrintVisitor(args, kwargs) (line 106)
    PrintVisitor_call_result_4767 = invoke(stypy.reporting.localization.Localization(__file__, 106, 4), PrintVisitor_4762, *[root_node_4763], **kwargs_4766)
    
    
    # Assigning a Call to a Name (line 107):
    
    # Call to getvalue(...): (line 107)
    # Processing the call keyword arguments (line 107)
    kwargs_4770 = {}
    # Getting the type of 'str_output' (line 107)
    str_output_4768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 10), 'str_output', False)
    # Obtaining the member 'getvalue' of a type (line 107)
    getvalue_4769 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 10), str_output_4768, 'getvalue')
    # Calling getvalue(args, kwargs) (line 107)
    getvalue_call_result_4771 = invoke(stypy.reporting.localization.Localization(__file__, 107, 10), getvalue_4769, *[], **kwargs_4770)
    
    # Assigning a type to the variable 'txt' (line 107)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'txt', getvalue_call_result_4771)
    
    # Call to close(...): (line 108)
    # Processing the call keyword arguments (line 108)
    kwargs_4774 = {}
    # Getting the type of 'str_output' (line 108)
    str_output_4772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'str_output', False)
    # Obtaining the member 'close' of a type (line 108)
    close_4773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 4), str_output_4772, 'close')
    # Calling close(args, kwargs) (line 108)
    close_call_result_4775 = invoke(stypy.reporting.localization.Localization(__file__, 108, 4), close_4773, *[], **kwargs_4774)
    
    # Getting the type of 'txt' (line 110)
    txt_4776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 11), 'txt')
    # Assigning a type to the variable 'stypy_return_type' (line 110)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'stypy_return_type', txt_4776)
    
    # ################# End of 'dump_ast(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'dump_ast' in the type store
    # Getting the type of 'stypy_return_type' (line 99)
    stypy_return_type_4777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_4777)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'dump_ast'
    return stypy_return_type_4777

# Assigning a type to the variable 'dump_ast' (line 99)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 0), 'dump_ast', dump_ast)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
