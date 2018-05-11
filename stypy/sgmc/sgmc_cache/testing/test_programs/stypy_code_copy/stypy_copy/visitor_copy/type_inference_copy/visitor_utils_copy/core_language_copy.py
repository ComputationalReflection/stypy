
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
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')
import_30879 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'data_structures_copy')

if (type(import_30879) is not StypyTypeError):

    if (import_30879 != 'pyd_module'):
        __import__(import_30879)
        sys_modules_30880 = sys.modules[import_30879]
        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'data_structures_copy', sys_modules_30880.module_type_store, module_type_store)
    else:
        import data_structures_copy

        import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'data_structures_copy', data_structures_copy, module_type_store)

else:
    # Assigning a type to the variable 'data_structures_copy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'data_structures_copy', import_30879)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/visitor_utils_copy/')

str_30881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, (-1)), 'str', '\nHelper functions to create AST Nodes for basic Python language elements\n')

@norecursion
def create_attribute(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Call to Load(...): (line 13)
    # Processing the call keyword arguments (line 13)
    kwargs_30884 = {}
    # Getting the type of 'ast' (line 13)
    ast_30882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 51), 'ast', False)
    # Obtaining the member 'Load' of a type (line 13)
    Load_30883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 51), ast_30882, 'Load')
    # Calling Load(args, kwargs) (line 13)
    Load_call_result_30885 = invoke(stypy.reporting.localization.Localization(__file__, 13, 51), Load_30883, *[], **kwargs_30884)
    
    int_30886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 68), 'int')
    int_30887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 78), 'int')
    defaults = [Load_call_result_30885, int_30886, int_30887]
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

    str_30888 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, (-1)), 'str', '\n    Creates an ast.Attribute node using the provided parameters to fill its field values\n\n    :param owner_name: (str) owner name of the attribute (for instance, if the attribute is obj.method,\n    the owner is "obj")\n    :param att_name: (str) Name of the attribute ("method" in the previous example)\n    :param context: ast.Load (default) or ast.Store)\n    :param line: Line number (optional)\n    :param column: Column offset (optional)\n    :return: An AST Attribute node.\n    ')
    
    # Assigning a Call to a Name (line 25):
    
    # Call to Attribute(...): (line 25)
    # Processing the call keyword arguments (line 25)
    kwargs_30891 = {}
    # Getting the type of 'ast' (line 25)
    ast_30889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 16), 'ast', False)
    # Obtaining the member 'Attribute' of a type (line 25)
    Attribute_30890 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 16), ast_30889, 'Attribute')
    # Calling Attribute(args, kwargs) (line 25)
    Attribute_call_result_30892 = invoke(stypy.reporting.localization.Localization(__file__, 25, 16), Attribute_30890, *[], **kwargs_30891)
    
    # Assigning a type to the variable 'attribute' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'attribute', Attribute_call_result_30892)
    
    # Assigning a Name to a Attribute (line 26):
    # Getting the type of 'att_name' (line 26)
    att_name_30893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 21), 'att_name')
    # Getting the type of 'attribute' (line 26)
    attribute_30894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'attribute')
    # Setting the type of the member 'attr' of a type (line 26)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 4), attribute_30894, 'attr', att_name_30893)
    
    # Assigning a Name to a Attribute (line 27):
    # Getting the type of 'context' (line 27)
    context_30895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 20), 'context')
    # Getting the type of 'attribute' (line 27)
    attribute_30896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'attribute')
    # Setting the type of the member 'ctx' of a type (line 27)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 4), attribute_30896, 'ctx', context_30895)
    
    # Assigning a Name to a Attribute (line 28):
    # Getting the type of 'line' (line 28)
    line_30897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 23), 'line')
    # Getting the type of 'attribute' (line 28)
    attribute_30898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'attribute')
    # Setting the type of the member 'lineno' of a type (line 28)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 4), attribute_30898, 'lineno', line_30897)
    
    # Assigning a Name to a Attribute (line 29):
    # Getting the type of 'column' (line 29)
    column_30899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 27), 'column')
    # Getting the type of 'attribute' (line 29)
    attribute_30900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'attribute')
    # Setting the type of the member 'col_offset' of a type (line 29)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 4), attribute_30900, 'col_offset', column_30899)
    
    # Type idiom detected: calculating its left and rigth part (line 31)
    # Getting the type of 'str' (line 31)
    str_30901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 30), 'str')
    # Getting the type of 'owner_name' (line 31)
    owner_name_30902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 18), 'owner_name')
    
    (may_be_30903, more_types_in_union_30904) = may_be_subtype(str_30901, owner_name_30902)

    if may_be_30903:

        if more_types_in_union_30904:
            # Runtime conditional SSA (line 31)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'owner_name' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'owner_name', remove_not_subtype_from_union(owner_name_30902, str))
        
        # Assigning a Call to a Name (line 32):
        
        # Call to Name(...): (line 32)
        # Processing the call keyword arguments (line 32)
        kwargs_30907 = {}
        # Getting the type of 'ast' (line 32)
        ast_30905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 25), 'ast', False)
        # Obtaining the member 'Name' of a type (line 32)
        Name_30906 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 25), ast_30905, 'Name')
        # Calling Name(args, kwargs) (line 32)
        Name_call_result_30908 = invoke(stypy.reporting.localization.Localization(__file__, 32, 25), Name_30906, *[], **kwargs_30907)
        
        # Assigning a type to the variable 'attribute_name' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'attribute_name', Name_call_result_30908)
        
        # Assigning a Call to a Attribute (line 33):
        
        # Call to Load(...): (line 33)
        # Processing the call keyword arguments (line 33)
        kwargs_30911 = {}
        # Getting the type of 'ast' (line 33)
        ast_30909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 29), 'ast', False)
        # Obtaining the member 'Load' of a type (line 33)
        Load_30910 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 29), ast_30909, 'Load')
        # Calling Load(args, kwargs) (line 33)
        Load_call_result_30912 = invoke(stypy.reporting.localization.Localization(__file__, 33, 29), Load_30910, *[], **kwargs_30911)
        
        # Getting the type of 'attribute_name' (line 33)
        attribute_name_30913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'attribute_name')
        # Setting the type of the member 'ctx' of a type (line 33)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), attribute_name_30913, 'ctx', Load_call_result_30912)
        
        # Assigning a Name to a Attribute (line 34):
        # Getting the type of 'owner_name' (line 34)
        owner_name_30914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 28), 'owner_name')
        # Getting the type of 'attribute_name' (line 34)
        attribute_name_30915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'attribute_name')
        # Setting the type of the member 'id' of a type (line 34)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), attribute_name_30915, 'id', owner_name_30914)
        
        # Assigning a Name to a Attribute (line 35):
        # Getting the type of 'line' (line 35)
        line_30916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 32), 'line')
        # Getting the type of 'attribute_name' (line 35)
        attribute_name_30917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'attribute_name')
        # Setting the type of the member 'lineno' of a type (line 35)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), attribute_name_30917, 'lineno', line_30916)
        
        # Assigning a Name to a Attribute (line 36):
        # Getting the type of 'column' (line 36)
        column_30918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 36), 'column')
        # Getting the type of 'attribute_name' (line 36)
        attribute_name_30919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'attribute_name')
        # Setting the type of the member 'col_offset' of a type (line 36)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), attribute_name_30919, 'col_offset', column_30918)
        
        # Assigning a Name to a Attribute (line 38):
        # Getting the type of 'attribute_name' (line 38)
        attribute_name_30920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 26), 'attribute_name')
        # Getting the type of 'attribute' (line 38)
        attribute_30921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'attribute')
        # Setting the type of the member 'value' of a type (line 38)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), attribute_30921, 'value', attribute_name_30920)

        if more_types_in_union_30904:
            # Runtime conditional SSA for else branch (line 31)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_30903) or more_types_in_union_30904):
        # Assigning a type to the variable 'owner_name' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'owner_name', remove_subtype_from_union(owner_name_30902, str))
        
        # Assigning a Name to a Attribute (line 40):
        # Getting the type of 'owner_name' (line 40)
        owner_name_30922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 26), 'owner_name')
        # Getting the type of 'attribute' (line 40)
        attribute_30923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'attribute')
        # Setting the type of the member 'value' of a type (line 40)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), attribute_30923, 'value', owner_name_30922)

        if (may_be_30903 and more_types_in_union_30904):
            # SSA join for if statement (line 31)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'attribute' (line 42)
    attribute_30924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 11), 'attribute')
    # Assigning a type to the variable 'stypy_return_type' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'stypy_return_type', attribute_30924)
    
    # ################# End of 'create_attribute(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_attribute' in the type store
    # Getting the type of 'stypy_return_type' (line 13)
    stypy_return_type_30925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30925)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_attribute'
    return stypy_return_type_30925

# Assigning a type to the variable 'create_attribute' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'create_attribute', create_attribute)

@norecursion
def create_Name(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'True' (line 45)
    True_30926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 42), 'True')
    int_30927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 53), 'int')
    int_30928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 63), 'int')
    defaults = [True_30926, int_30927, int_30928]
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

    str_30929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, (-1)), 'str', '\n    Creates an ast.Name node using the provided parameters to fill its field values\n\n    :param var_name: (str) value of the name\n    :param right_hand_side: ast.Load (default) or ast.Store\n    :param line: Line number (optional)\n    :param column: Column offset (optional)\n    :return: An AST Name node.\n    ')
    
    # Assigning a Call to a Name (line 55):
    
    # Call to Name(...): (line 55)
    # Processing the call keyword arguments (line 55)
    kwargs_30932 = {}
    # Getting the type of 'ast' (line 55)
    ast_30930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 11), 'ast', False)
    # Obtaining the member 'Name' of a type (line 55)
    Name_30931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 11), ast_30930, 'Name')
    # Calling Name(args, kwargs) (line 55)
    Name_call_result_30933 = invoke(stypy.reporting.localization.Localization(__file__, 55, 11), Name_30931, *[], **kwargs_30932)
    
    # Assigning a type to the variable 'name' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'name', Name_call_result_30933)
    
    # Assigning a Name to a Attribute (line 56):
    # Getting the type of 'var_name' (line 56)
    var_name_30934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 14), 'var_name')
    # Getting the type of 'name' (line 56)
    name_30935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'name')
    # Setting the type of the member 'id' of a type (line 56)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 4), name_30935, 'id', var_name_30934)
    
    # Assigning a Name to a Attribute (line 57):
    # Getting the type of 'line' (line 57)
    line_30936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 18), 'line')
    # Getting the type of 'name' (line 57)
    name_30937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'name')
    # Setting the type of the member 'lineno' of a type (line 57)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 4), name_30937, 'lineno', line_30936)
    
    # Assigning a Name to a Attribute (line 58):
    # Getting the type of 'column' (line 58)
    column_30938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 22), 'column')
    # Getting the type of 'name' (line 58)
    name_30939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'name')
    # Setting the type of the member 'col_offset' of a type (line 58)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 4), name_30939, 'col_offset', column_30938)
    # Getting the type of 'right_hand_side' (line 60)
    right_hand_side_30940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 7), 'right_hand_side')
    # Testing if the type of an if condition is none (line 60)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 60, 4), right_hand_side_30940):
        
        # Assigning a Call to a Attribute (line 63):
        
        # Call to Store(...): (line 63)
        # Processing the call keyword arguments (line 63)
        kwargs_30949 = {}
        # Getting the type of 'ast' (line 63)
        ast_30947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 19), 'ast', False)
        # Obtaining the member 'Store' of a type (line 63)
        Store_30948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 19), ast_30947, 'Store')
        # Calling Store(args, kwargs) (line 63)
        Store_call_result_30950 = invoke(stypy.reporting.localization.Localization(__file__, 63, 19), Store_30948, *[], **kwargs_30949)
        
        # Getting the type of 'name' (line 63)
        name_30951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'name')
        # Setting the type of the member 'ctx' of a type (line 63)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), name_30951, 'ctx', Store_call_result_30950)
    else:
        
        # Testing the type of an if condition (line 60)
        if_condition_30941 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 60, 4), right_hand_side_30940)
        # Assigning a type to the variable 'if_condition_30941' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'if_condition_30941', if_condition_30941)
        # SSA begins for if statement (line 60)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 61):
        
        # Call to Load(...): (line 61)
        # Processing the call keyword arguments (line 61)
        kwargs_30944 = {}
        # Getting the type of 'ast' (line 61)
        ast_30942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 19), 'ast', False)
        # Obtaining the member 'Load' of a type (line 61)
        Load_30943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 19), ast_30942, 'Load')
        # Calling Load(args, kwargs) (line 61)
        Load_call_result_30945 = invoke(stypy.reporting.localization.Localization(__file__, 61, 19), Load_30943, *[], **kwargs_30944)
        
        # Getting the type of 'name' (line 61)
        name_30946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'name')
        # Setting the type of the member 'ctx' of a type (line 61)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), name_30946, 'ctx', Load_call_result_30945)
        # SSA branch for the else part of an if statement (line 60)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Attribute (line 63):
        
        # Call to Store(...): (line 63)
        # Processing the call keyword arguments (line 63)
        kwargs_30949 = {}
        # Getting the type of 'ast' (line 63)
        ast_30947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 19), 'ast', False)
        # Obtaining the member 'Store' of a type (line 63)
        Store_30948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 19), ast_30947, 'Store')
        # Calling Store(args, kwargs) (line 63)
        Store_call_result_30950 = invoke(stypy.reporting.localization.Localization(__file__, 63, 19), Store_30948, *[], **kwargs_30949)
        
        # Getting the type of 'name' (line 63)
        name_30951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'name')
        # Setting the type of the member 'ctx' of a type (line 63)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), name_30951, 'ctx', Store_call_result_30950)
        # SSA join for if statement (line 60)
        module_type_store = module_type_store.join_ssa_context()
        

    # Getting the type of 'name' (line 65)
    name_30952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 11), 'name')
    # Assigning a type to the variable 'stypy_return_type' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'stypy_return_type', name_30952)
    
    # ################# End of 'create_Name(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_Name' in the type store
    # Getting the type of 'stypy_return_type' (line 45)
    stypy_return_type_30953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30953)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_Name'
    return stypy_return_type_30953

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

    str_30954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, (-1)), 'str', '\n    Creates an Assign AST Node, with its left and right hand side\n    :param left_hand_side: Left hand side of the assignment (AST Node)\n    :param right_hand_side: Right hand side of the assignment (AST Node)\n    :return: AST Assign node\n    ')
    
    # Assigning a Call to a Attribute (line 75):
    
    # Call to Load(...): (line 75)
    # Processing the call keyword arguments (line 75)
    kwargs_30957 = {}
    # Getting the type of 'ast' (line 75)
    ast_30955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 26), 'ast', False)
    # Obtaining the member 'Load' of a type (line 75)
    Load_30956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 26), ast_30955, 'Load')
    # Calling Load(args, kwargs) (line 75)
    Load_call_result_30958 = invoke(stypy.reporting.localization.Localization(__file__, 75, 26), Load_30956, *[], **kwargs_30957)
    
    # Getting the type of 'right_hand_side' (line 75)
    right_hand_side_30959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'right_hand_side')
    # Setting the type of the member 'ctx' of a type (line 75)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 4), right_hand_side_30959, 'ctx', Load_call_result_30958)
    
    # Assigning a Call to a Attribute (line 76):
    
    # Call to Store(...): (line 76)
    # Processing the call keyword arguments (line 76)
    kwargs_30962 = {}
    # Getting the type of 'ast' (line 76)
    ast_30960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 25), 'ast', False)
    # Obtaining the member 'Store' of a type (line 76)
    Store_30961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 25), ast_30960, 'Store')
    # Calling Store(args, kwargs) (line 76)
    Store_call_result_30963 = invoke(stypy.reporting.localization.Localization(__file__, 76, 25), Store_30961, *[], **kwargs_30962)
    
    # Getting the type of 'left_hand_side' (line 76)
    left_hand_side_30964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 4), 'left_hand_side')
    # Setting the type of the member 'ctx' of a type (line 76)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 4), left_hand_side_30964, 'ctx', Store_call_result_30963)
    
    # Call to Assign(...): (line 77)
    # Processing the call keyword arguments (line 77)
    
    # Obtaining an instance of the builtin type 'list' (line 77)
    list_30967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 30), 'list')
    # Adding type elements to the builtin type 'list' instance (line 77)
    # Adding element type (line 77)
    # Getting the type of 'left_hand_side' (line 77)
    left_hand_side_30968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 31), 'left_hand_side', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 30), list_30967, left_hand_side_30968)
    
    keyword_30969 = list_30967
    # Getting the type of 'right_hand_side' (line 77)
    right_hand_side_30970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 54), 'right_hand_side', False)
    keyword_30971 = right_hand_side_30970
    kwargs_30972 = {'targets': keyword_30969, 'value': keyword_30971}
    # Getting the type of 'ast' (line 77)
    ast_30965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 11), 'ast', False)
    # Obtaining the member 'Assign' of a type (line 77)
    Assign_30966 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 11), ast_30965, 'Assign')
    # Calling Assign(args, kwargs) (line 77)
    Assign_call_result_30973 = invoke(stypy.reporting.localization.Localization(__file__, 77, 11), Assign_30966, *[], **kwargs_30972)
    
    # Assigning a type to the variable 'stypy_return_type' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'stypy_return_type', Assign_call_result_30973)
    
    # ################# End of 'create_Assign(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_Assign' in the type store
    # Getting the type of 'stypy_return_type' (line 68)
    stypy_return_type_30974 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30974)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_Assign'
    return stypy_return_type_30974

# Assigning a type to the variable 'create_Assign' (line 68)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 0), 'create_Assign', create_Assign)

@norecursion
def create_str(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_30975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 23), 'int')
    int_30976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 30), 'int')
    defaults = [int_30975, int_30976]
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

    str_30977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, (-1)), 'str', '\n    Creates an AST Str node with the passed contents\n    :param s: Content of the AST Node\n    :param line: Line\n    :param col: Column\n    :return: An AST Str\n    ')
    
    # Assigning a Call to a Name (line 88):
    
    # Call to Str(...): (line 88)
    # Processing the call keyword arguments (line 88)
    kwargs_30980 = {}
    # Getting the type of 'ast' (line 88)
    ast_30978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 11), 'ast', False)
    # Obtaining the member 'Str' of a type (line 88)
    Str_30979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 11), ast_30978, 'Str')
    # Calling Str(args, kwargs) (line 88)
    Str_call_result_30981 = invoke(stypy.reporting.localization.Localization(__file__, 88, 11), Str_30979, *[], **kwargs_30980)
    
    # Assigning a type to the variable 'str_' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'str_', Str_call_result_30981)
    
    # Assigning a Name to a Attribute (line 90):
    # Getting the type of 's' (line 90)
    s_30982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 13), 's')
    # Getting the type of 'str_' (line 90)
    str__30983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'str_')
    # Setting the type of the member 's' of a type (line 90)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 4), str__30983, 's', s_30982)
    
    # Assigning a Name to a Attribute (line 91):
    # Getting the type of 'line' (line 91)
    line_30984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 18), 'line')
    # Getting the type of 'str_' (line 91)
    str__30985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'str_')
    # Setting the type of the member 'lineno' of a type (line 91)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 4), str__30985, 'lineno', line_30984)
    
    # Assigning a Name to a Attribute (line 92):
    # Getting the type of 'col' (line 92)
    col_30986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 22), 'col')
    # Getting the type of 'str_' (line 92)
    str__30987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'str_')
    # Setting the type of the member 'col_offset' of a type (line 92)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 4), str__30987, 'col_offset', col_30986)
    # Getting the type of 'str_' (line 94)
    str__30988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 11), 'str_')
    # Assigning a type to the variable 'stypy_return_type' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'stypy_return_type', str__30988)
    
    # ################# End of 'create_str(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_str' in the type store
    # Getting the type of 'stypy_return_type' (line 80)
    stypy_return_type_30989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_30989)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_str'
    return stypy_return_type_30989

# Assigning a type to the variable 'create_str' (line 80)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'create_str', create_str)

@norecursion
def create_alias(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_30990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 29), 'str', '')
    # Getting the type of 'None' (line 97)
    None_30991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 40), 'None')
    defaults = [str_30990, None_30991]
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

    str_30992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, (-1)), 'str', '\n    Creates an AST Alias node\n\n    :param name: Name of the aliased variable\n    :param alias: Alias name\n    :param asname: Name to put if the alias uses the "as" keyword\n    :return: An AST alias node\n    ')
    
    # Assigning a Call to a Name (line 106):
    
    # Call to alias(...): (line 106)
    # Processing the call keyword arguments (line 106)
    kwargs_30995 = {}
    # Getting the type of 'ast' (line 106)
    ast_30993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 17), 'ast', False)
    # Obtaining the member 'alias' of a type (line 106)
    alias_30994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 17), ast_30993, 'alias')
    # Calling alias(args, kwargs) (line 106)
    alias_call_result_30996 = invoke(stypy.reporting.localization.Localization(__file__, 106, 17), alias_30994, *[], **kwargs_30995)
    
    # Assigning a type to the variable 'alias_node' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'alias_node', alias_call_result_30996)
    
    # Assigning a Name to a Attribute (line 108):
    # Getting the type of 'alias' (line 108)
    alias_30997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 23), 'alias')
    # Getting the type of 'alias_node' (line 108)
    alias_node_30998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'alias_node')
    # Setting the type of the member 'alias' of a type (line 108)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 4), alias_node_30998, 'alias', alias_30997)
    
    # Assigning a Name to a Attribute (line 109):
    # Getting the type of 'asname' (line 109)
    asname_30999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 24), 'asname')
    # Getting the type of 'alias_node' (line 109)
    alias_node_31000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'alias_node')
    # Setting the type of the member 'asname' of a type (line 109)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 4), alias_node_31000, 'asname', asname_30999)
    
    # Assigning a Name to a Attribute (line 110):
    # Getting the type of 'name' (line 110)
    name_31001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 22), 'name')
    # Getting the type of 'alias_node' (line 110)
    alias_node_31002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'alias_node')
    # Setting the type of the member 'name' of a type (line 110)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 4), alias_node_31002, 'name', name_31001)
    # Getting the type of 'alias_node' (line 112)
    alias_node_31003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 11), 'alias_node')
    # Assigning a type to the variable 'stypy_return_type' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'stypy_return_type', alias_node_31003)
    
    # ################# End of 'create_alias(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_alias' in the type store
    # Getting the type of 'stypy_return_type' (line 97)
    stypy_return_type_31004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31004)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_alias'
    return stypy_return_type_31004

# Assigning a type to the variable 'create_alias' (line 97)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 0), 'create_alias', create_alias)

@norecursion
def create_importfrom(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_31005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 43), 'int')
    int_31006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 51), 'int')
    int_31007 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 61), 'int')
    defaults = [int_31005, int_31006, int_31007]
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

    str_31008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, (-1)), 'str', '\n    Creates an AST ImportFrom node\n\n    :param module: Module to import\n    :param names: Members of the module to import\n    :param level: Level of the import\n    :param line: Line\n    :param column: Column\n    :return: An AST ImportFrom node\n    ')
    
    # Assigning a Call to a Name (line 126):
    
    # Call to ImportFrom(...): (line 126)
    # Processing the call keyword arguments (line 126)
    kwargs_31011 = {}
    # Getting the type of 'ast' (line 126)
    ast_31009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 17), 'ast', False)
    # Obtaining the member 'ImportFrom' of a type (line 126)
    ImportFrom_31010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 17), ast_31009, 'ImportFrom')
    # Calling ImportFrom(args, kwargs) (line 126)
    ImportFrom_call_result_31012 = invoke(stypy.reporting.localization.Localization(__file__, 126, 17), ImportFrom_31010, *[], **kwargs_31011)
    
    # Assigning a type to the variable 'importfrom' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'importfrom', ImportFrom_call_result_31012)
    
    # Assigning a Name to a Attribute (line 127):
    # Getting the type of 'level' (line 127)
    level_31013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 23), 'level')
    # Getting the type of 'importfrom' (line 127)
    importfrom_31014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'importfrom')
    # Setting the type of the member 'level' of a type (line 127)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 4), importfrom_31014, 'level', level_31013)
    
    # Assigning a Name to a Attribute (line 128):
    # Getting the type of 'module' (line 128)
    module_31015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 24), 'module')
    # Getting the type of 'importfrom' (line 128)
    importfrom_31016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'importfrom')
    # Setting the type of the member 'module' of a type (line 128)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 4), importfrom_31016, 'module', module_31015)
    
    # Call to is_iterable(...): (line 130)
    # Processing the call arguments (line 130)
    # Getting the type of 'names' (line 130)
    names_31019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 40), 'names', False)
    # Processing the call keyword arguments (line 130)
    kwargs_31020 = {}
    # Getting the type of 'data_structures_copy' (line 130)
    data_structures_copy_31017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 7), 'data_structures_copy', False)
    # Obtaining the member 'is_iterable' of a type (line 130)
    is_iterable_31018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 7), data_structures_copy_31017, 'is_iterable')
    # Calling is_iterable(args, kwargs) (line 130)
    is_iterable_call_result_31021 = invoke(stypy.reporting.localization.Localization(__file__, 130, 7), is_iterable_31018, *[names_31019], **kwargs_31020)
    
    # Testing if the type of an if condition is none (line 130)

    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 130, 4), is_iterable_call_result_31021):
        
        # Assigning a List to a Attribute (line 133):
        
        # Obtaining an instance of the builtin type 'list' (line 133)
        list_31025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 133)
        # Adding element type (line 133)
        # Getting the type of 'names' (line 133)
        names_31026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 28), 'names')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 27), list_31025, names_31026)
        
        # Getting the type of 'importfrom' (line 133)
        importfrom_31027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'importfrom')
        # Setting the type of the member 'names' of a type (line 133)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 8), importfrom_31027, 'names', list_31025)
    else:
        
        # Testing the type of an if condition (line 130)
        if_condition_31022 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 4), is_iterable_call_result_31021)
        # Assigning a type to the variable 'if_condition_31022' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'if_condition_31022', if_condition_31022)
        # SSA begins for if statement (line 130)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Attribute (line 131):
        # Getting the type of 'names' (line 131)
        names_31023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 27), 'names')
        # Getting the type of 'importfrom' (line 131)
        importfrom_31024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'importfrom')
        # Setting the type of the member 'names' of a type (line 131)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 8), importfrom_31024, 'names', names_31023)
        # SSA branch for the else part of an if statement (line 130)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a List to a Attribute (line 133):
        
        # Obtaining an instance of the builtin type 'list' (line 133)
        list_31025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 133)
        # Adding element type (line 133)
        # Getting the type of 'names' (line 133)
        names_31026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 28), 'names')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 133, 27), list_31025, names_31026)
        
        # Getting the type of 'importfrom' (line 133)
        importfrom_31027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'importfrom')
        # Setting the type of the member 'names' of a type (line 133)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 8), importfrom_31027, 'names', list_31025)
        # SSA join for if statement (line 130)
        module_type_store = module_type_store.join_ssa_context()
        

    
    # Assigning a Name to a Attribute (line 135):
    # Getting the type of 'line' (line 135)
    line_31028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 24), 'line')
    # Getting the type of 'importfrom' (line 135)
    importfrom_31029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'importfrom')
    # Setting the type of the member 'lineno' of a type (line 135)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 4), importfrom_31029, 'lineno', line_31028)
    
    # Assigning a Name to a Attribute (line 136):
    # Getting the type of 'column' (line 136)
    column_31030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 28), 'column')
    # Getting the type of 'importfrom' (line 136)
    importfrom_31031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'importfrom')
    # Setting the type of the member 'col_offset' of a type (line 136)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 4), importfrom_31031, 'col_offset', column_31030)
    # Getting the type of 'importfrom' (line 138)
    importfrom_31032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 11), 'importfrom')
    # Assigning a type to the variable 'stypy_return_type' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'stypy_return_type', importfrom_31032)
    
    # ################# End of 'create_importfrom(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_importfrom' in the type store
    # Getting the type of 'stypy_return_type' (line 115)
    stypy_return_type_31033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31033)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_importfrom'
    return stypy_return_type_31033

# Assigning a type to the variable 'create_importfrom' (line 115)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 0), 'create_importfrom', create_importfrom)

@norecursion
def create_num(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_31034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 25), 'int')
    int_31035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 39), 'int')
    defaults = [int_31034, int_31035]
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

    str_31036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, (-1)), 'str', '\n    Create an AST Num node\n\n    :param n: Value\n    :param lineno: line\n    :param col_offset: column\n    :return: An AST Num node\n    ')
    
    # Assigning a Call to a Name (line 150):
    
    # Call to Num(...): (line 150)
    # Processing the call keyword arguments (line 150)
    kwargs_31039 = {}
    # Getting the type of 'ast' (line 150)
    ast_31037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 10), 'ast', False)
    # Obtaining the member 'Num' of a type (line 150)
    Num_31038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 10), ast_31037, 'Num')
    # Calling Num(args, kwargs) (line 150)
    Num_call_result_31040 = invoke(stypy.reporting.localization.Localization(__file__, 150, 10), Num_31038, *[], **kwargs_31039)
    
    # Assigning a type to the variable 'num' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'num', Num_call_result_31040)
    
    # Assigning a Name to a Attribute (line 151):
    # Getting the type of 'n' (line 151)
    n_31041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'n')
    # Getting the type of 'num' (line 151)
    num_31042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'num')
    # Setting the type of the member 'n' of a type (line 151)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 4), num_31042, 'n', n_31041)
    
    # Assigning a Name to a Attribute (line 152):
    # Getting the type of 'lineno' (line 152)
    lineno_31043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 17), 'lineno')
    # Getting the type of 'num' (line 152)
    num_31044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'num')
    # Setting the type of the member 'lineno' of a type (line 152)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 4), num_31044, 'lineno', lineno_31043)
    
    # Assigning a Name to a Attribute (line 153):
    # Getting the type of 'col_offset' (line 153)
    col_offset_31045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 21), 'col_offset')
    # Getting the type of 'num' (line 153)
    num_31046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 4), 'num')
    # Setting the type of the member 'col_offset' of a type (line 153)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 4), num_31046, 'col_offset', col_offset_31045)
    # Getting the type of 'num' (line 155)
    num_31047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 11), 'num')
    # Assigning a type to the variable 'stypy_return_type' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'stypy_return_type', num_31047)
    
    # ################# End of 'create_num(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_num' in the type store
    # Getting the type of 'stypy_return_type' (line 141)
    stypy_return_type_31048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31048)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_num'
    return stypy_return_type_31048

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

    str_31049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, (-1)), 'str', '\n    Creates an AST Tuple node\n\n    :param elems: ELements of the tuple\n    :return: AST Tuple node\n    ')
    
    # Assigning a Call to a Name (line 165):
    
    # Call to Tuple(...): (line 165)
    # Processing the call keyword arguments (line 165)
    kwargs_31052 = {}
    # Getting the type of 'ast' (line 165)
    ast_31050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'ast', False)
    # Obtaining the member 'Tuple' of a type (line 165)
    Tuple_31051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 12), ast_31050, 'Tuple')
    # Calling Tuple(args, kwargs) (line 165)
    Tuple_call_result_31053 = invoke(stypy.reporting.localization.Localization(__file__, 165, 12), Tuple_31051, *[], **kwargs_31052)
    
    # Assigning a type to the variable 'tuple' (line 165)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'tuple', Tuple_call_result_31053)
    
    # Assigning a List to a Attribute (line 167):
    
    # Obtaining an instance of the builtin type 'list' (line 167)
    list_31054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 167)
    
    # Getting the type of 'tuple' (line 167)
    tuple_31055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'tuple')
    # Setting the type of the member 'elts' of a type (line 167)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 4), tuple_31055, 'elts', list_31054)
    
    # Getting the type of 'elems' (line 168)
    elems_31056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 16), 'elems')
    # Assigning a type to the variable 'elems_31056' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'elems_31056', elems_31056)
    # Testing if the for loop is going to be iterated (line 168)
    # Testing the type of a for loop iterable (line 168)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 168, 4), elems_31056)

    if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 168, 4), elems_31056):
        # Getting the type of the for loop variable (line 168)
        for_loop_var_31057 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 168, 4), elems_31056)
        # Assigning a type to the variable 'elem' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'elem', for_loop_var_31057)
        # SSA begins for a for statement (line 168)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'elem' (line 169)
        elem_31061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 26), 'elem', False)
        # Processing the call keyword arguments (line 169)
        kwargs_31062 = {}
        # Getting the type of 'tuple' (line 169)
        tuple_31058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'tuple', False)
        # Obtaining the member 'elts' of a type (line 169)
        elts_31059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), tuple_31058, 'elts')
        # Obtaining the member 'append' of a type (line 169)
        append_31060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), elts_31059, 'append')
        # Calling append(args, kwargs) (line 169)
        append_call_result_31063 = invoke(stypy.reporting.localization.Localization(__file__, 169, 8), append_31060, *[elem_31061], **kwargs_31062)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()

    
    # Getting the type of 'tuple' (line 171)
    tuple_31064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 11), 'tuple')
    # Assigning a type to the variable 'stypy_return_type' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'stypy_return_type', tuple_31064)
    
    # ################# End of 'create_type_tuple(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_type_tuple' in the type store
    # Getting the type of 'stypy_return_type' (line 158)
    stypy_return_type_31065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_31065)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_type_tuple'
    return stypy_return_type_31065

# Assigning a type to the variable 'create_type_tuple' (line 158)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 0), 'create_type_tuple', create_type_tuple)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
