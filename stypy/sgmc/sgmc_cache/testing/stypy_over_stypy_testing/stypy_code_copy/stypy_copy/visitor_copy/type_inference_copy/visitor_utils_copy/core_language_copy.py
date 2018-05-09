
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import ast
2: 
3: import data_structures_copy
4: 
5: '''
6: Helper functions to create AST Nodes for basic Python language elements
7: '''
8: 
9: 
10: # ############################################# BASIC LANGUAGE ELEMENTS ##############################################
11: 
12: 
13: def create_attribute(owner_name, att_name, context=ast.Load(), line=0, column=0):
14:     '''
15:     Creates an ast.Attribute node using the provided parameters to fill its field values
16: 
17:     :param owner_name: (str) owner name of the attribute (for instance, if the attribute is obj.method,
18:     the owner is "obj")
19:     :param att_name: (str) Name of the attribute ("method" in the previous example)
20:     :param context: ast.Load (default) or ast.Store)
21:     :param line: Line number (optional)
22:     :param column: Column offset (optional)
23:     :return: An AST Attribute node.
24:     '''
25:     attribute = ast.Attribute()
26:     attribute.attr = att_name
27:     attribute.ctx = context
28:     attribute.lineno = line
29:     attribute.col_offset = column
30: 
31:     if isinstance(owner_name, str):
32:         attribute_name = ast.Name()
33:         attribute_name.ctx = ast.Load()
34:         attribute_name.id = owner_name
35:         attribute_name.lineno = line
36:         attribute_name.col_offset = column
37: 
38:         attribute.value = attribute_name
39:     else:
40:         attribute.value = owner_name
41: 
42:     return attribute
43: 
44: 
45: def create_Name(var_name, right_hand_side=True, line=0, column=0):
46:     '''
47:     Creates an ast.Name node using the provided parameters to fill its field values
48: 
49:     :param var_name: (str) value of the name
50:     :param right_hand_side: ast.Load (default) or ast.Store
51:     :param line: Line number (optional)
52:     :param column: Column offset (optional)
53:     :return: An AST Name node.
54:     '''
55:     name = ast.Name()
56:     name.id = var_name
57:     name.lineno = line
58:     name.col_offset = column
59: 
60:     if right_hand_side:
61:         name.ctx = ast.Load()
62:     else:
63:         name.ctx = ast.Store()
64: 
65:     return name
66: 
67: 
68: def create_Assign(left_hand_side, right_hand_side):
69:     '''
70:     Creates an Assign AST Node, with its left and right hand side
71:     :param left_hand_side: Left hand side of the assignment (AST Node)
72:     :param right_hand_side: Right hand side of the assignment (AST Node)
73:     :return: AST Assign node
74:     '''
75:     right_hand_side.ctx = ast.Load()
76:     left_hand_side.ctx = ast.Store()
77:     return ast.Assign(targets=[left_hand_side], value=right_hand_side)
78: 
79: 
80: def create_str(s, line=0, col=0):
81:     '''
82:     Creates an AST Str node with the passed contents
83:     :param s: Content of the AST Node
84:     :param line: Line
85:     :param col: Column
86:     :return: An AST Str
87:     '''
88:     str_ = ast.Str()
89: 
90:     str_.s = s
91:     str_.lineno = line
92:     str_.col_offset = col
93: 
94:     return str_
95: 
96: 
97: def create_alias(name, alias="", asname=None):
98:     '''
99:     Creates an AST Alias node
100: 
101:     :param name: Name of the aliased variable
102:     :param alias: Alias name
103:     :param asname: Name to put if the alias uses the "as" keyword
104:     :return: An AST alias node
105:     '''
106:     alias_node = ast.alias()
107: 
108:     alias_node.alias = alias
109:     alias_node.asname = asname
110:     alias_node.name = name
111: 
112:     return alias_node
113: 
114: 
115: def create_importfrom(module, names, level=0, line=0, column=0):
116:     '''
117:     Creates an AST ImportFrom node
118: 
119:     :param module: Module to import
120:     :param names: Members of the module to import
121:     :param level: Level of the import
122:     :param line: Line
123:     :param column: Column
124:     :return: An AST ImportFrom node
125:     '''
126:     importfrom = ast.ImportFrom()
127:     importfrom.level = level
128:     importfrom.module = module
129: 
130:     if data_structures_copy.is_iterable(names):
131:         importfrom.names = names
132:     else:
133:         importfrom.names = [names]
134: 
135:     importfrom.lineno = line
136:     importfrom.col_offset = column
137: 
138:     return importfrom
139: 
140: 
141: def create_num(n, lineno=0, col_offset=0):
142:     '''
143:     Create an AST Num node
144: 
145:     :param n: Value
146:     :param lineno: line
147:     :param col_offset: column
148:     :return: An AST Num node
149:     '''
150:     num = ast.Num()
151:     num.n = n
152:     num.lineno = lineno
153:     num.col_offset = col_offset
154: 
155:     return num
156: 
157: 
158: def create_type_tuple(*elems):
159:     '''
160:     Creates an AST Tuple node
161: 
162:     :param elems: ELements of the tuple
163:     :return: AST Tuple node
164:     '''
165:     tuple = ast.Tuple()
166: 
167:     tuple.elts = []
168:     for elem in elems:
169:         tuple.elts.append(elem)
170: 
171:     return tuple
172: 

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

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import data_structures_copy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')
import_15314 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'data_structures_copy')

if (type(import_15314) is not StypyTypeError):

    if (import_15314 != 'pyd_module'):
        __import__(import_15314)
        sys_modules_15315 = sys.modules[import_15314]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'data_structures_copy', sys_modules_15315.module_type_store, module_type_store)
    else:
        import data_structures_copy

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'data_structures_copy', data_structures_copy, module_type_store)

else:
    # Assigning a type to the variable 'data_structures_copy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'data_structures_copy', import_15314)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')

str_15316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, (-1)), 'str', '\nHelper functions to create AST Nodes for basic Python language elements\n')

@norecursion
def create_attribute(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Call to Load(...): (line 13)
    # Processing the call keyword arguments (line 13)
    kwargs_15319 = {}
    # Getting the type of 'ast' (line 13)
    ast_15317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 51), 'ast', False)
    # Obtaining the member 'Load' of a type (line 13)
    Load_15318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 51), ast_15317, 'Load')
    # Calling Load(args, kwargs) (line 13)
    Load_call_result_15320 = invoke(stypy.reporting.localization.Localization(__file__, 13, 51), Load_15318, *[], **kwargs_15319)
    
    int_15321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 68), 'int')
    int_15322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 78), 'int')
    defaults = [Load_call_result_15320, int_15321, int_15322]
    # Create a new context for function 'create_attribute'
    module_type_store = module_type_store.open_function_context('create_attribute', 13, 0, False)
    
    # Passed parameters checking function
    create_attribute.stypy_localization = localization
    create_attribute.stypy_type_of_self = None
    create_attribute.stypy_type_store = module_type_store
    create_attribute.stypy_function_name = 'create_attribute'
    create_attribute.stypy_param_names_list = ['owner_name', 'att_name', 'context', 'line', 'column']
    create_attribute.stypy_varargs_param_name = None
    create_attribute.stypy_kwargs_param_name = None
    create_attribute.stypy_call_defaults = defaults
    create_attribute.stypy_call_varargs = varargs
    create_attribute.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_attribute', ['owner_name', 'att_name', 'context', 'line', 'column'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_attribute', localization, ['owner_name', 'att_name', 'context', 'line', 'column'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_attribute(...)' code ##################

    str_15323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, (-1)), 'str', '\n    Creates an ast.Attribute node using the provided parameters to fill its field values\n\n    :param owner_name: (str) owner name of the attribute (for instance, if the attribute is obj.method,\n    the owner is "obj")\n    :param att_name: (str) Name of the attribute ("method" in the previous example)\n    :param context: ast.Load (default) or ast.Store)\n    :param line: Line number (optional)\n    :param column: Column offset (optional)\n    :return: An AST Attribute node.\n    ')
    
    # Assigning a Call to a Name (line 25):
    
    # Call to Attribute(...): (line 25)
    # Processing the call keyword arguments (line 25)
    kwargs_15326 = {}
    # Getting the type of 'ast' (line 25)
    ast_15324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 16), 'ast', False)
    # Obtaining the member 'Attribute' of a type (line 25)
    Attribute_15325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 16), ast_15324, 'Attribute')
    # Calling Attribute(args, kwargs) (line 25)
    Attribute_call_result_15327 = invoke(stypy.reporting.localization.Localization(__file__, 25, 16), Attribute_15325, *[], **kwargs_15326)
    
    # Assigning a type to the variable 'attribute' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'attribute', Attribute_call_result_15327)
    
    # Assigning a Name to a Attribute (line 26):
    # Getting the type of 'att_name' (line 26)
    att_name_15328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 21), 'att_name')
    # Getting the type of 'attribute' (line 26)
    attribute_15329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'attribute')
    # Setting the type of the member 'attr' of a type (line 26)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 4), attribute_15329, 'attr', att_name_15328)
    
    # Assigning a Name to a Attribute (line 27):
    # Getting the type of 'context' (line 27)
    context_15330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 20), 'context')
    # Getting the type of 'attribute' (line 27)
    attribute_15331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'attribute')
    # Setting the type of the member 'ctx' of a type (line 27)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 4), attribute_15331, 'ctx', context_15330)
    
    # Assigning a Name to a Attribute (line 28):
    # Getting the type of 'line' (line 28)
    line_15332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 23), 'line')
    # Getting the type of 'attribute' (line 28)
    attribute_15333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'attribute')
    # Setting the type of the member 'lineno' of a type (line 28)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 4), attribute_15333, 'lineno', line_15332)
    
    # Assigning a Name to a Attribute (line 29):
    # Getting the type of 'column' (line 29)
    column_15334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 27), 'column')
    # Getting the type of 'attribute' (line 29)
    attribute_15335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'attribute')
    # Setting the type of the member 'col_offset' of a type (line 29)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 4), attribute_15335, 'col_offset', column_15334)
    
    # Type idiom detected: calculating its left and rigth part (line 31)
    # Getting the type of 'str' (line 31)
    str_15336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 30), 'str')
    # Getting the type of 'owner_name' (line 31)
    owner_name_15337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 18), 'owner_name')
    
    (may_be_15338, more_types_in_union_15339) = may_be_subtype(str_15336, owner_name_15337)

    if may_be_15338:

        if more_types_in_union_15339:
            # Runtime conditional SSA (line 31)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'owner_name' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'owner_name', remove_not_subtype_from_union(owner_name_15337, str))
        
        # Assigning a Call to a Name (line 32):
        
        # Call to Name(...): (line 32)
        # Processing the call keyword arguments (line 32)
        kwargs_15342 = {}
        # Getting the type of 'ast' (line 32)
        ast_15340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 25), 'ast', False)
        # Obtaining the member 'Name' of a type (line 32)
        Name_15341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 25), ast_15340, 'Name')
        # Calling Name(args, kwargs) (line 32)
        Name_call_result_15343 = invoke(stypy.reporting.localization.Localization(__file__, 32, 25), Name_15341, *[], **kwargs_15342)
        
        # Assigning a type to the variable 'attribute_name' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'attribute_name', Name_call_result_15343)
        
        # Assigning a Call to a Attribute (line 33):
        
        # Call to Load(...): (line 33)
        # Processing the call keyword arguments (line 33)
        kwargs_15346 = {}
        # Getting the type of 'ast' (line 33)
        ast_15344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 29), 'ast', False)
        # Obtaining the member 'Load' of a type (line 33)
        Load_15345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 29), ast_15344, 'Load')
        # Calling Load(args, kwargs) (line 33)
        Load_call_result_15347 = invoke(stypy.reporting.localization.Localization(__file__, 33, 29), Load_15345, *[], **kwargs_15346)
        
        # Getting the type of 'attribute_name' (line 33)
        attribute_name_15348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'attribute_name')
        # Setting the type of the member 'ctx' of a type (line 33)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), attribute_name_15348, 'ctx', Load_call_result_15347)
        
        # Assigning a Name to a Attribute (line 34):
        # Getting the type of 'owner_name' (line 34)
        owner_name_15349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 28), 'owner_name')
        # Getting the type of 'attribute_name' (line 34)
        attribute_name_15350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'attribute_name')
        # Setting the type of the member 'id' of a type (line 34)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), attribute_name_15350, 'id', owner_name_15349)
        
        # Assigning a Name to a Attribute (line 35):
        # Getting the type of 'line' (line 35)
        line_15351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 32), 'line')
        # Getting the type of 'attribute_name' (line 35)
        attribute_name_15352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'attribute_name')
        # Setting the type of the member 'lineno' of a type (line 35)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), attribute_name_15352, 'lineno', line_15351)
        
        # Assigning a Name to a Attribute (line 36):
        # Getting the type of 'column' (line 36)
        column_15353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 36), 'column')
        # Getting the type of 'attribute_name' (line 36)
        attribute_name_15354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'attribute_name')
        # Setting the type of the member 'col_offset' of a type (line 36)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), attribute_name_15354, 'col_offset', column_15353)
        
        # Assigning a Name to a Attribute (line 38):
        # Getting the type of 'attribute_name' (line 38)
        attribute_name_15355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 26), 'attribute_name')
        # Getting the type of 'attribute' (line 38)
        attribute_15356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'attribute')
        # Setting the type of the member 'value' of a type (line 38)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), attribute_15356, 'value', attribute_name_15355)

        if more_types_in_union_15339:
            # Runtime conditional SSA for else branch (line 31)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_15338) or more_types_in_union_15339):
        # Assigning a type to the variable 'owner_name' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'owner_name', remove_subtype_from_union(owner_name_15337, str))
        
        # Assigning a Name to a Attribute (line 40):
        # Getting the type of 'owner_name' (line 40)
        owner_name_15357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 26), 'owner_name')
        # Getting the type of 'attribute' (line 40)
        attribute_15358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'attribute')
        # Setting the type of the member 'value' of a type (line 40)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), attribute_15358, 'value', owner_name_15357)

        if (may_be_15338 and more_types_in_union_15339):
            # SSA join for if statement (line 31)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'attribute' (line 42)
    attribute_15359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 11), 'attribute')
    # Assigning a type to the variable 'stypy_return_type' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'stypy_return_type', attribute_15359)
    
    # ################# End of 'create_attribute(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_attribute' in the type store
    # Getting the type of 'stypy_return_type' (line 13)
    stypy_return_type_15360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15360)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_attribute'
    return stypy_return_type_15360

# Assigning a type to the variable 'create_attribute' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'create_attribute', create_attribute)

@norecursion
def create_Name(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 45)
    True_15361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 42), 'True')
    int_15362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 53), 'int')
    int_15363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 63), 'int')
    defaults = [True_15361, int_15362, int_15363]
    # Create a new context for function 'create_Name'
    module_type_store = module_type_store.open_function_context('create_Name', 45, 0, False)
    
    # Passed parameters checking function
    create_Name.stypy_localization = localization
    create_Name.stypy_type_of_self = None
    create_Name.stypy_type_store = module_type_store
    create_Name.stypy_function_name = 'create_Name'
    create_Name.stypy_param_names_list = ['var_name', 'right_hand_side', 'line', 'column']
    create_Name.stypy_varargs_param_name = None
    create_Name.stypy_kwargs_param_name = None
    create_Name.stypy_call_defaults = defaults
    create_Name.stypy_call_varargs = varargs
    create_Name.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_Name', ['var_name', 'right_hand_side', 'line', 'column'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_Name', localization, ['var_name', 'right_hand_side', 'line', 'column'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_Name(...)' code ##################

    str_15364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, (-1)), 'str', '\n    Creates an ast.Name node using the provided parameters to fill its field values\n\n    :param var_name: (str) value of the name\n    :param right_hand_side: ast.Load (default) or ast.Store\n    :param line: Line number (optional)\n    :param column: Column offset (optional)\n    :return: An AST Name node.\n    ')
    
    # Assigning a Call to a Name (line 55):
    
    # Call to Name(...): (line 55)
    # Processing the call keyword arguments (line 55)
    kwargs_15367 = {}
    # Getting the type of 'ast' (line 55)
    ast_15365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 11), 'ast', False)
    # Obtaining the member 'Name' of a type (line 55)
    Name_15366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 11), ast_15365, 'Name')
    # Calling Name(args, kwargs) (line 55)
    Name_call_result_15368 = invoke(stypy.reporting.localization.Localization(__file__, 55, 11), Name_15366, *[], **kwargs_15367)
    
    # Assigning a type to the variable 'name' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'name', Name_call_result_15368)
    
    # Assigning a Name to a Attribute (line 56):
    # Getting the type of 'var_name' (line 56)
    var_name_15369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 14), 'var_name')
    # Getting the type of 'name' (line 56)
    name_15370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'name')
    # Setting the type of the member 'id' of a type (line 56)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 4), name_15370, 'id', var_name_15369)
    
    # Assigning a Name to a Attribute (line 57):
    # Getting the type of 'line' (line 57)
    line_15371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 18), 'line')
    # Getting the type of 'name' (line 57)
    name_15372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'name')
    # Setting the type of the member 'lineno' of a type (line 57)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 4), name_15372, 'lineno', line_15371)
    
    # Assigning a Name to a Attribute (line 58):
    # Getting the type of 'column' (line 58)
    column_15373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 22), 'column')
    # Getting the type of 'name' (line 58)
    name_15374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'name')
    # Setting the type of the member 'col_offset' of a type (line 58)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 4), name_15374, 'col_offset', column_15373)
    # Getting the type of 'right_hand_side' (line 60)
    right_hand_side_15375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 7), 'right_hand_side')
    # Testing if the type of an if condition is none (line 60)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 60, 4), right_hand_side_15375):
        
        # Assigning a Call to a Attribute (line 63):
        
        # Call to Store(...): (line 63)
        # Processing the call keyword arguments (line 63)
        kwargs_15384 = {}
        # Getting the type of 'ast' (line 63)
        ast_15382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 19), 'ast', False)
        # Obtaining the member 'Store' of a type (line 63)
        Store_15383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 19), ast_15382, 'Store')
        # Calling Store(args, kwargs) (line 63)
        Store_call_result_15385 = invoke(stypy.reporting.localization.Localization(__file__, 63, 19), Store_15383, *[], **kwargs_15384)
        
        # Getting the type of 'name' (line 63)
        name_15386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'name')
        # Setting the type of the member 'ctx' of a type (line 63)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), name_15386, 'ctx', Store_call_result_15385)
    else:
        
        # Testing the type of an if condition (line 60)
        if_condition_15376 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 60, 4), right_hand_side_15375)
        # Assigning a type to the variable 'if_condition_15376' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'if_condition_15376', if_condition_15376)
        # SSA begins for if statement (line 60)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 61):
        
        # Call to Load(...): (line 61)
        # Processing the call keyword arguments (line 61)
        kwargs_15379 = {}
        # Getting the type of 'ast' (line 61)
        ast_15377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 19), 'ast', False)
        # Obtaining the member 'Load' of a type (line 61)
        Load_15378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 19), ast_15377, 'Load')
        # Calling Load(args, kwargs) (line 61)
        Load_call_result_15380 = invoke(stypy.reporting.localization.Localization(__file__, 61, 19), Load_15378, *[], **kwargs_15379)
        
        # Getting the type of 'name' (line 61)
        name_15381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'name')
        # Setting the type of the member 'ctx' of a type (line 61)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), name_15381, 'ctx', Load_call_result_15380)
        # SSA branch for the else part of an if statement (line 60)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Attribute (line 63):
        
        # Call to Store(...): (line 63)
        # Processing the call keyword arguments (line 63)
        kwargs_15384 = {}
        # Getting the type of 'ast' (line 63)
        ast_15382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 19), 'ast', False)
        # Obtaining the member 'Store' of a type (line 63)
        Store_15383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 19), ast_15382, 'Store')
        # Calling Store(args, kwargs) (line 63)
        Store_call_result_15385 = invoke(stypy.reporting.localization.Localization(__file__, 63, 19), Store_15383, *[], **kwargs_15384)
        
        # Getting the type of 'name' (line 63)
        name_15386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'name')
        # Setting the type of the member 'ctx' of a type (line 63)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), name_15386, 'ctx', Store_call_result_15385)
        # SSA join for if statement (line 60)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'name' (line 65)
    name_15387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 11), 'name')
    # Assigning a type to the variable 'stypy_return_type' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'stypy_return_type', name_15387)
    
    # ################# End of 'create_Name(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_Name' in the type store
    # Getting the type of 'stypy_return_type' (line 45)
    stypy_return_type_15388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15388)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_Name'
    return stypy_return_type_15388

# Assigning a type to the variable 'create_Name' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'create_Name', create_Name)

@norecursion
def create_Assign(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_Assign'
    module_type_store = module_type_store.open_function_context('create_Assign', 68, 0, False)
    
    # Passed parameters checking function
    create_Assign.stypy_localization = localization
    create_Assign.stypy_type_of_self = None
    create_Assign.stypy_type_store = module_type_store
    create_Assign.stypy_function_name = 'create_Assign'
    create_Assign.stypy_param_names_list = ['left_hand_side', 'right_hand_side']
    create_Assign.stypy_varargs_param_name = None
    create_Assign.stypy_kwargs_param_name = None
    create_Assign.stypy_call_defaults = defaults
    create_Assign.stypy_call_varargs = varargs
    create_Assign.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_Assign', ['left_hand_side', 'right_hand_side'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_Assign', localization, ['left_hand_side', 'right_hand_side'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_Assign(...)' code ##################

    str_15389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, (-1)), 'str', '\n    Creates an Assign AST Node, with its left and right hand side\n    :param left_hand_side: Left hand side of the assignment (AST Node)\n    :param right_hand_side: Right hand side of the assignment (AST Node)\n    :return: AST Assign node\n    ')
    
    # Assigning a Call to a Attribute (line 75):
    
    # Call to Load(...): (line 75)
    # Processing the call keyword arguments (line 75)
    kwargs_15392 = {}
    # Getting the type of 'ast' (line 75)
    ast_15390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 26), 'ast', False)
    # Obtaining the member 'Load' of a type (line 75)
    Load_15391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 26), ast_15390, 'Load')
    # Calling Load(args, kwargs) (line 75)
    Load_call_result_15393 = invoke(stypy.reporting.localization.Localization(__file__, 75, 26), Load_15391, *[], **kwargs_15392)
    
    # Getting the type of 'right_hand_side' (line 75)
    right_hand_side_15394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'right_hand_side')
    # Setting the type of the member 'ctx' of a type (line 75)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 4), right_hand_side_15394, 'ctx', Load_call_result_15393)
    
    # Assigning a Call to a Attribute (line 76):
    
    # Call to Store(...): (line 76)
    # Processing the call keyword arguments (line 76)
    kwargs_15397 = {}
    # Getting the type of 'ast' (line 76)
    ast_15395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 25), 'ast', False)
    # Obtaining the member 'Store' of a type (line 76)
    Store_15396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 25), ast_15395, 'Store')
    # Calling Store(args, kwargs) (line 76)
    Store_call_result_15398 = invoke(stypy.reporting.localization.Localization(__file__, 76, 25), Store_15396, *[], **kwargs_15397)
    
    # Getting the type of 'left_hand_side' (line 76)
    left_hand_side_15399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'left_hand_side')
    # Setting the type of the member 'ctx' of a type (line 76)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 4), left_hand_side_15399, 'ctx', Store_call_result_15398)
    
    # Call to Assign(...): (line 77)
    # Processing the call keyword arguments (line 77)
    
    # Obtaining an instance of the builtin type 'list' (line 77)
    list_15402 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 77)
    # Adding element type (line 77)
    # Getting the type of 'left_hand_side' (line 77)
    left_hand_side_15403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 31), 'left_hand_side', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 30), list_15402, left_hand_side_15403)
    
    keyword_15404 = list_15402
    # Getting the type of 'right_hand_side' (line 77)
    right_hand_side_15405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 54), 'right_hand_side', False)
    keyword_15406 = right_hand_side_15405
    kwargs_15407 = {'targets': keyword_15404, 'value': keyword_15406}
    # Getting the type of 'ast' (line 77)
    ast_15400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 11), 'ast', False)
    # Obtaining the member 'Assign' of a type (line 77)
    Assign_15401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 11), ast_15400, 'Assign')
    # Calling Assign(args, kwargs) (line 77)
    Assign_call_result_15408 = invoke(stypy.reporting.localization.Localization(__file__, 77, 11), Assign_15401, *[], **kwargs_15407)
    
    # Assigning a type to the variable 'stypy_return_type' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'stypy_return_type', Assign_call_result_15408)
    
    # ################# End of 'create_Assign(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_Assign' in the type store
    # Getting the type of 'stypy_return_type' (line 68)
    stypy_return_type_15409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15409)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_Assign'
    return stypy_return_type_15409

# Assigning a type to the variable 'create_Assign' (line 68)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'create_Assign', create_Assign)

@norecursion
def create_str(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_15410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 23), 'int')
    int_15411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 30), 'int')
    defaults = [int_15410, int_15411]
    # Create a new context for function 'create_str'
    module_type_store = module_type_store.open_function_context('create_str', 80, 0, False)
    
    # Passed parameters checking function
    create_str.stypy_localization = localization
    create_str.stypy_type_of_self = None
    create_str.stypy_type_store = module_type_store
    create_str.stypy_function_name = 'create_str'
    create_str.stypy_param_names_list = ['s', 'line', 'col']
    create_str.stypy_varargs_param_name = None
    create_str.stypy_kwargs_param_name = None
    create_str.stypy_call_defaults = defaults
    create_str.stypy_call_varargs = varargs
    create_str.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_str', ['s', 'line', 'col'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_str', localization, ['s', 'line', 'col'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_str(...)' code ##################

    str_15412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, (-1)), 'str', '\n    Creates an AST Str node with the passed contents\n    :param s: Content of the AST Node\n    :param line: Line\n    :param col: Column\n    :return: An AST Str\n    ')
    
    # Assigning a Call to a Name (line 88):
    
    # Call to Str(...): (line 88)
    # Processing the call keyword arguments (line 88)
    kwargs_15415 = {}
    # Getting the type of 'ast' (line 88)
    ast_15413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 11), 'ast', False)
    # Obtaining the member 'Str' of a type (line 88)
    Str_15414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 11), ast_15413, 'Str')
    # Calling Str(args, kwargs) (line 88)
    Str_call_result_15416 = invoke(stypy.reporting.localization.Localization(__file__, 88, 11), Str_15414, *[], **kwargs_15415)
    
    # Assigning a type to the variable 'str_' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'str_', Str_call_result_15416)
    
    # Assigning a Name to a Attribute (line 90):
    # Getting the type of 's' (line 90)
    s_15417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 13), 's')
    # Getting the type of 'str_' (line 90)
    str__15418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'str_')
    # Setting the type of the member 's' of a type (line 90)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 4), str__15418, 's', s_15417)
    
    # Assigning a Name to a Attribute (line 91):
    # Getting the type of 'line' (line 91)
    line_15419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 18), 'line')
    # Getting the type of 'str_' (line 91)
    str__15420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'str_')
    # Setting the type of the member 'lineno' of a type (line 91)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 4), str__15420, 'lineno', line_15419)
    
    # Assigning a Name to a Attribute (line 92):
    # Getting the type of 'col' (line 92)
    col_15421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 22), 'col')
    # Getting the type of 'str_' (line 92)
    str__15422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'str_')
    # Setting the type of the member 'col_offset' of a type (line 92)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 4), str__15422, 'col_offset', col_15421)
    # Getting the type of 'str_' (line 94)
    str__15423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 11), 'str_')
    # Assigning a type to the variable 'stypy_return_type' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'stypy_return_type', str__15423)
    
    # ################# End of 'create_str(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_str' in the type store
    # Getting the type of 'stypy_return_type' (line 80)
    stypy_return_type_15424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15424)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_str'
    return stypy_return_type_15424

# Assigning a type to the variable 'create_str' (line 80)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'create_str', create_str)

@norecursion
def create_alias(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_15425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 29), 'str', '')
    # Getting the type of 'None' (line 97)
    None_15426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 40), 'None')
    defaults = [str_15425, None_15426]
    # Create a new context for function 'create_alias'
    module_type_store = module_type_store.open_function_context('create_alias', 97, 0, False)
    
    # Passed parameters checking function
    create_alias.stypy_localization = localization
    create_alias.stypy_type_of_self = None
    create_alias.stypy_type_store = module_type_store
    create_alias.stypy_function_name = 'create_alias'
    create_alias.stypy_param_names_list = ['name', 'alias', 'asname']
    create_alias.stypy_varargs_param_name = None
    create_alias.stypy_kwargs_param_name = None
    create_alias.stypy_call_defaults = defaults
    create_alias.stypy_call_varargs = varargs
    create_alias.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_alias', ['name', 'alias', 'asname'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_alias', localization, ['name', 'alias', 'asname'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_alias(...)' code ##################

    str_15427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, (-1)), 'str', '\n    Creates an AST Alias node\n\n    :param name: Name of the aliased variable\n    :param alias: Alias name\n    :param asname: Name to put if the alias uses the "as" keyword\n    :return: An AST alias node\n    ')
    
    # Assigning a Call to a Name (line 106):
    
    # Call to alias(...): (line 106)
    # Processing the call keyword arguments (line 106)
    kwargs_15430 = {}
    # Getting the type of 'ast' (line 106)
    ast_15428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 17), 'ast', False)
    # Obtaining the member 'alias' of a type (line 106)
    alias_15429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 17), ast_15428, 'alias')
    # Calling alias(args, kwargs) (line 106)
    alias_call_result_15431 = invoke(stypy.reporting.localization.Localization(__file__, 106, 17), alias_15429, *[], **kwargs_15430)
    
    # Assigning a type to the variable 'alias_node' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'alias_node', alias_call_result_15431)
    
    # Assigning a Name to a Attribute (line 108):
    # Getting the type of 'alias' (line 108)
    alias_15432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 23), 'alias')
    # Getting the type of 'alias_node' (line 108)
    alias_node_15433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'alias_node')
    # Setting the type of the member 'alias' of a type (line 108)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 4), alias_node_15433, 'alias', alias_15432)
    
    # Assigning a Name to a Attribute (line 109):
    # Getting the type of 'asname' (line 109)
    asname_15434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 24), 'asname')
    # Getting the type of 'alias_node' (line 109)
    alias_node_15435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'alias_node')
    # Setting the type of the member 'asname' of a type (line 109)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 4), alias_node_15435, 'asname', asname_15434)
    
    # Assigning a Name to a Attribute (line 110):
    # Getting the type of 'name' (line 110)
    name_15436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 22), 'name')
    # Getting the type of 'alias_node' (line 110)
    alias_node_15437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'alias_node')
    # Setting the type of the member 'name' of a type (line 110)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 4), alias_node_15437, 'name', name_15436)
    # Getting the type of 'alias_node' (line 112)
    alias_node_15438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 11), 'alias_node')
    # Assigning a type to the variable 'stypy_return_type' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'stypy_return_type', alias_node_15438)
    
    # ################# End of 'create_alias(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_alias' in the type store
    # Getting the type of 'stypy_return_type' (line 97)
    stypy_return_type_15439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15439)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_alias'
    return stypy_return_type_15439

# Assigning a type to the variable 'create_alias' (line 97)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 0), 'create_alias', create_alias)

@norecursion
def create_importfrom(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_15440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 43), 'int')
    int_15441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 51), 'int')
    int_15442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 61), 'int')
    defaults = [int_15440, int_15441, int_15442]
    # Create a new context for function 'create_importfrom'
    module_type_store = module_type_store.open_function_context('create_importfrom', 115, 0, False)
    
    # Passed parameters checking function
    create_importfrom.stypy_localization = localization
    create_importfrom.stypy_type_of_self = None
    create_importfrom.stypy_type_store = module_type_store
    create_importfrom.stypy_function_name = 'create_importfrom'
    create_importfrom.stypy_param_names_list = ['module', 'names', 'level', 'line', 'column']
    create_importfrom.stypy_varargs_param_name = None
    create_importfrom.stypy_kwargs_param_name = None
    create_importfrom.stypy_call_defaults = defaults
    create_importfrom.stypy_call_varargs = varargs
    create_importfrom.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_importfrom', ['module', 'names', 'level', 'line', 'column'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_importfrom', localization, ['module', 'names', 'level', 'line', 'column'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_importfrom(...)' code ##################

    str_15443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, (-1)), 'str', '\n    Creates an AST ImportFrom node\n\n    :param module: Module to import\n    :param names: Members of the module to import\n    :param level: Level of the import\n    :param line: Line\n    :param column: Column\n    :return: An AST ImportFrom node\n    ')
    
    # Assigning a Call to a Name (line 126):
    
    # Call to ImportFrom(...): (line 126)
    # Processing the call keyword arguments (line 126)
    kwargs_15446 = {}
    # Getting the type of 'ast' (line 126)
    ast_15444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 17), 'ast', False)
    # Obtaining the member 'ImportFrom' of a type (line 126)
    ImportFrom_15445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 17), ast_15444, 'ImportFrom')
    # Calling ImportFrom(args, kwargs) (line 126)
    ImportFrom_call_result_15447 = invoke(stypy.reporting.localization.Localization(__file__, 126, 17), ImportFrom_15445, *[], **kwargs_15446)
    
    # Assigning a type to the variable 'importfrom' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'importfrom', ImportFrom_call_result_15447)
    
    # Assigning a Name to a Attribute (line 127):
    # Getting the type of 'level' (line 127)
    level_15448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 23), 'level')
    # Getting the type of 'importfrom' (line 127)
    importfrom_15449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'importfrom')
    # Setting the type of the member 'level' of a type (line 127)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 4), importfrom_15449, 'level', level_15448)
    
    # Assigning a Name to a Attribute (line 128):
    # Getting the type of 'module' (line 128)
    module_15450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 24), 'module')
    # Getting the type of 'importfrom' (line 128)
    importfrom_15451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'importfrom')
    # Setting the type of the member 'module' of a type (line 128)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 4), importfrom_15451, 'module', module_15450)
    
    # Call to is_iterable(...): (line 130)
    # Processing the call arguments (line 130)
    # Getting the type of 'names' (line 130)
    names_15454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 40), 'names', False)
    # Processing the call keyword arguments (line 130)
    kwargs_15455 = {}
    # Getting the type of 'data_structures_copy' (line 130)
    data_structures_copy_15452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 7), 'data_structures_copy', False)
    # Obtaining the member 'is_iterable' of a type (line 130)
    is_iterable_15453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 7), data_structures_copy_15452, 'is_iterable')
    # Calling is_iterable(args, kwargs) (line 130)
    is_iterable_call_result_15456 = invoke(stypy.reporting.localization.Localization(__file__, 130, 7), is_iterable_15453, *[names_15454], **kwargs_15455)
    
    # Testing if the type of an if condition is none (line 130)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 130, 4), is_iterable_call_result_15456):
        
        # Assigning a List to a Attribute (line 133):
        
        # Obtaining an instance of the builtin type 'list' (line 133)
        list_15460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 133)
        # Adding element type (line 133)
        # Getting the type of 'names' (line 133)
        names_15461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 28), 'names')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 27), list_15460, names_15461)
        
        # Getting the type of 'importfrom' (line 133)
        importfrom_15462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'importfrom')
        # Setting the type of the member 'names' of a type (line 133)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 8), importfrom_15462, 'names', list_15460)
    else:
        
        # Testing the type of an if condition (line 130)
        if_condition_15457 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 4), is_iterable_call_result_15456)
        # Assigning a type to the variable 'if_condition_15457' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'if_condition_15457', if_condition_15457)
        # SSA begins for if statement (line 130)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 131):
        # Getting the type of 'names' (line 131)
        names_15458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 27), 'names')
        # Getting the type of 'importfrom' (line 131)
        importfrom_15459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'importfrom')
        # Setting the type of the member 'names' of a type (line 131)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 8), importfrom_15459, 'names', names_15458)
        # SSA branch for the else part of an if statement (line 130)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Attribute (line 133):
        
        # Obtaining an instance of the builtin type 'list' (line 133)
        list_15460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 133)
        # Adding element type (line 133)
        # Getting the type of 'names' (line 133)
        names_15461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 28), 'names')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 27), list_15460, names_15461)
        
        # Getting the type of 'importfrom' (line 133)
        importfrom_15462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'importfrom')
        # Setting the type of the member 'names' of a type (line 133)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 8), importfrom_15462, 'names', list_15460)
        # SSA join for if statement (line 130)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Name to a Attribute (line 135):
    # Getting the type of 'line' (line 135)
    line_15463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 24), 'line')
    # Getting the type of 'importfrom' (line 135)
    importfrom_15464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'importfrom')
    # Setting the type of the member 'lineno' of a type (line 135)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 4), importfrom_15464, 'lineno', line_15463)
    
    # Assigning a Name to a Attribute (line 136):
    # Getting the type of 'column' (line 136)
    column_15465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 28), 'column')
    # Getting the type of 'importfrom' (line 136)
    importfrom_15466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'importfrom')
    # Setting the type of the member 'col_offset' of a type (line 136)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 4), importfrom_15466, 'col_offset', column_15465)
    # Getting the type of 'importfrom' (line 138)
    importfrom_15467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 11), 'importfrom')
    # Assigning a type to the variable 'stypy_return_type' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'stypy_return_type', importfrom_15467)
    
    # ################# End of 'create_importfrom(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_importfrom' in the type store
    # Getting the type of 'stypy_return_type' (line 115)
    stypy_return_type_15468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15468)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_importfrom'
    return stypy_return_type_15468

# Assigning a type to the variable 'create_importfrom' (line 115)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 0), 'create_importfrom', create_importfrom)

@norecursion
def create_num(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_15469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 25), 'int')
    int_15470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 39), 'int')
    defaults = [int_15469, int_15470]
    # Create a new context for function 'create_num'
    module_type_store = module_type_store.open_function_context('create_num', 141, 0, False)
    
    # Passed parameters checking function
    create_num.stypy_localization = localization
    create_num.stypy_type_of_self = None
    create_num.stypy_type_store = module_type_store
    create_num.stypy_function_name = 'create_num'
    create_num.stypy_param_names_list = ['n', 'lineno', 'col_offset']
    create_num.stypy_varargs_param_name = None
    create_num.stypy_kwargs_param_name = None
    create_num.stypy_call_defaults = defaults
    create_num.stypy_call_varargs = varargs
    create_num.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_num', ['n', 'lineno', 'col_offset'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_num', localization, ['n', 'lineno', 'col_offset'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_num(...)' code ##################

    str_15471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, (-1)), 'str', '\n    Create an AST Num node\n\n    :param n: Value\n    :param lineno: line\n    :param col_offset: column\n    :return: An AST Num node\n    ')
    
    # Assigning a Call to a Name (line 150):
    
    # Call to Num(...): (line 150)
    # Processing the call keyword arguments (line 150)
    kwargs_15474 = {}
    # Getting the type of 'ast' (line 150)
    ast_15472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 10), 'ast', False)
    # Obtaining the member 'Num' of a type (line 150)
    Num_15473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 10), ast_15472, 'Num')
    # Calling Num(args, kwargs) (line 150)
    Num_call_result_15475 = invoke(stypy.reporting.localization.Localization(__file__, 150, 10), Num_15473, *[], **kwargs_15474)
    
    # Assigning a type to the variable 'num' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'num', Num_call_result_15475)
    
    # Assigning a Name to a Attribute (line 151):
    # Getting the type of 'n' (line 151)
    n_15476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'n')
    # Getting the type of 'num' (line 151)
    num_15477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'num')
    # Setting the type of the member 'n' of a type (line 151)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 4), num_15477, 'n', n_15476)
    
    # Assigning a Name to a Attribute (line 152):
    # Getting the type of 'lineno' (line 152)
    lineno_15478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 17), 'lineno')
    # Getting the type of 'num' (line 152)
    num_15479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'num')
    # Setting the type of the member 'lineno' of a type (line 152)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 4), num_15479, 'lineno', lineno_15478)
    
    # Assigning a Name to a Attribute (line 153):
    # Getting the type of 'col_offset' (line 153)
    col_offset_15480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 21), 'col_offset')
    # Getting the type of 'num' (line 153)
    num_15481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'num')
    # Setting the type of the member 'col_offset' of a type (line 153)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 4), num_15481, 'col_offset', col_offset_15480)
    # Getting the type of 'num' (line 155)
    num_15482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 11), 'num')
    # Assigning a type to the variable 'stypy_return_type' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'stypy_return_type', num_15482)
    
    # ################# End of 'create_num(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_num' in the type store
    # Getting the type of 'stypy_return_type' (line 141)
    stypy_return_type_15483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15483)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_num'
    return stypy_return_type_15483

# Assigning a type to the variable 'create_num' (line 141)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), 'create_num', create_num)

@norecursion
def create_type_tuple(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'create_type_tuple'
    module_type_store = module_type_store.open_function_context('create_type_tuple', 158, 0, False)
    
    # Passed parameters checking function
    create_type_tuple.stypy_localization = localization
    create_type_tuple.stypy_type_of_self = None
    create_type_tuple.stypy_type_store = module_type_store
    create_type_tuple.stypy_function_name = 'create_type_tuple'
    create_type_tuple.stypy_param_names_list = []
    create_type_tuple.stypy_varargs_param_name = 'elems'
    create_type_tuple.stypy_kwargs_param_name = None
    create_type_tuple.stypy_call_defaults = defaults
    create_type_tuple.stypy_call_varargs = varargs
    create_type_tuple.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_type_tuple', [], 'elems', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_type_tuple', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_type_tuple(...)' code ##################

    str_15484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, (-1)), 'str', '\n    Creates an AST Tuple node\n\n    :param elems: ELements of the tuple\n    :return: AST Tuple node\n    ')
    
    # Assigning a Call to a Name (line 165):
    
    # Call to Tuple(...): (line 165)
    # Processing the call keyword arguments (line 165)
    kwargs_15487 = {}
    # Getting the type of 'ast' (line 165)
    ast_15485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'ast', False)
    # Obtaining the member 'Tuple' of a type (line 165)
    Tuple_15486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 12), ast_15485, 'Tuple')
    # Calling Tuple(args, kwargs) (line 165)
    Tuple_call_result_15488 = invoke(stypy.reporting.localization.Localization(__file__, 165, 12), Tuple_15486, *[], **kwargs_15487)
    
    # Assigning a type to the variable 'tuple' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'tuple', Tuple_call_result_15488)
    
    # Assigning a List to a Attribute (line 167):
    
    # Obtaining an instance of the builtin type 'list' (line 167)
    list_15489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 167)
    
    # Getting the type of 'tuple' (line 167)
    tuple_15490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'tuple')
    # Setting the type of the member 'elts' of a type (line 167)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 4), tuple_15490, 'elts', list_15489)
    
    # Getting the type of 'elems' (line 168)
    elems_15491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 16), 'elems')
    # Assigning a type to the variable 'elems_15491' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'elems_15491', elems_15491)
    # Testing if the for loop is going to be iterated (line 168)
    # Testing the type of a for loop iterable (line 168)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 168, 4), elems_15491)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 168, 4), elems_15491):
        # Getting the type of the for loop variable (line 168)
        for_loop_var_15492 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 168, 4), elems_15491)
        # Assigning a type to the variable 'elem' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'elem', for_loop_var_15492)
        # SSA begins for a for statement (line 168)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'elem' (line 169)
        elem_15496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 26), 'elem', False)
        # Processing the call keyword arguments (line 169)
        kwargs_15497 = {}
        # Getting the type of 'tuple' (line 169)
        tuple_15493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'tuple', False)
        # Obtaining the member 'elts' of a type (line 169)
        elts_15494 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), tuple_15493, 'elts')
        # Obtaining the member 'append' of a type (line 169)
        append_15495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), elts_15494, 'append')
        # Calling append(args, kwargs) (line 169)
        append_call_result_15498 = invoke(stypy.reporting.localization.Localization(__file__, 169, 8), append_15495, *[elem_15496], **kwargs_15497)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'tuple' (line 171)
    tuple_15499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 11), 'tuple')
    # Assigning a type to the variable 'stypy_return_type' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'stypy_return_type', tuple_15499)
    
    # ################# End of 'create_type_tuple(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_type_tuple' in the type store
    # Getting the type of 'stypy_return_type' (line 158)
    stypy_return_type_15500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_15500)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_type_tuple'
    return stypy_return_type_15500

# Assigning a type to the variable 'create_type_tuple' (line 158)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 0), 'create_type_tuple', create_type_tuple)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
