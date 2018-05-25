
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import ast
2: 
3: from ...visitor_copy.type_inference_copy.visitor_utils_copy import stypy_functions_copy
4: import statement_visitor_copy
5: 
6: 
7: 
8: class TypeInferenceGeneratorVisitor(ast.NodeVisitor):
9:     '''
10:     This visitor is responsible of generating type inference code AST Tree from standard Pyhon code contained in a
11:     .py file. It just process the Module node, generating a fixed prefix code nodes, creating and running a
12:     StatementVisitor object and appending a fixed postfix code at the end to form the full AST tree of the type
13:     inference program created from a Python source code file.
14:     '''
15: 
16:     def __init__(self, file_name, original_code=None):
17:         '''
18:         Initialices the visitor.
19:         :param file_name: File name of the source code whose ast will be parsed. This is needed for localization,
20:         needed to report errors precisely.
21:         :param original_code: If present, it includes the original file source code as a comment at the beggining of
22:         the file. This can be useful for debugging purposes.
23:         '''
24:         self.type_inference_ast = ast.Module()
25:         self.file_name = file_name
26:         self.original_code = original_code
27: 
28:     @staticmethod
29:     def get_postfix_src_code(cls):
30:         '''
31:         All generated type inference programs has this code at the end, to capture generated TypeErrors and TypeWarnings
32:         in known variables
33:         :return:
34:         '''
35:         return "\n{0} = stypy.errors.type_error.TypeError.get_error_msgs()\n{1} = stypy.errors.type_warning.TypeWarning.get_warning_msgs()\n" \
36:             .format(stypy_functions_copy.default_type_error_var_name, stypy_functions_copy.default_type_warning_var_name)
37: 
38:     def generic_visit(self, node):
39:         new_stmts = list()
40: 
41:         # Add the source code of the original program as a comment, if provided
42:         # if not self.original_code is None:
43:         new_stmts.append(stypy_functions_copy.create_original_code_comment(self.file_name, self.original_code))
44: 
45:         # Writes the instruction: from stypy import *
46:         new_stmts.append(stypy_functions_copy.create_src_comment("Import the stypy library"))
47:         new_stmts.append(stypy_functions_copy.create_import_stypy())
48: 
49:         # Writes the instruction: type_store = TypeStore(__file__)
50:         new_stmts.append(stypy_functions_copy.create_src_comment("Create the module type store"))
51:         new_stmts.append(stypy_functions_copy.create_type_store())
52: 
53:         new_stmts.append(stypy_functions_copy.create_program_section_src_comment("Begin of the type inference program"))
54: 
55:         # Visit the source code beginning with an Statement visitor
56:         statements_visitor = statement_visitor_copy.StatementVisitor(self.file_name)
57: 
58:         new_stmts.extend(statements_visitor.visit(node, []))
59: 
60:         new_stmts.append(stypy_functions_copy.create_program_section_src_comment("End of the type inference program"))
61:         return ast.Module(new_stmts)
62: 

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

# 'from testing.test_programs.stypy_code_copy.stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy import stypy_functions_copy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/')
import_1 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy')

if (type(import_1) is not StypyTypeError):

    if (import_1 != 'pyd_module'):
        __import__(import_1)
        sys_modules_2 = sys.modules[import_1]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy', sys_modules_2.module_type_store, module_type_store, ['stypy_functions_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_2, sys_modules_2.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy import stypy_functions_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy', None, module_type_store, ['stypy_functions_copy'], [stypy_functions_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy', import_1)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import statement_visitor_copy' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/')
import_3 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'statement_visitor_copy')

if (type(import_3) is not StypyTypeError):

    if (import_3 != 'pyd_module'):
        __import__(import_3)
        sys_modules_4 = sys.modules[import_3]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'statement_visitor_copy', sys_modules_4.module_type_store, module_type_store)
    else:
        import statement_visitor_copy

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'statement_visitor_copy', statement_visitor_copy, module_type_store)

else:
    # Assigning a type to the variable 'statement_visitor_copy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'statement_visitor_copy', import_3)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/')

# Declaration of the 'TypeInferenceGeneratorVisitor' class
# Getting the type of 'ast' (line 8)
ast_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 36), 'ast')
# Obtaining the member 'NodeVisitor' of a type (line 8)
NodeVisitor_6 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 36), ast_5, 'NodeVisitor')

class TypeInferenceGeneratorVisitor(NodeVisitor_6, ):
    str_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, (-1)), 'str', '\n    This visitor is responsible of generating type inference code AST Tree from standard Pyhon code contained in a\n    .py file. It just process the Module node, generating a fixed prefix code nodes, creating and running a\n    StatementVisitor object and appending a fixed postfix code at the end to form the full AST tree of the type\n    inference program created from a Python source code file.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 16)
        None_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 48), 'None')
        defaults = [None_8]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 16, 4, False)
        # Assigning a type to the variable 'self' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceGeneratorVisitor.__init__', ['file_name', 'original_code'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['file_name', 'original_code'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        str_9 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, (-1)), 'str', '\n        Initialices the visitor.\n        :param file_name: File name of the source code whose ast will be parsed. This is needed for localization,\n        needed to report errors precisely.\n        :param original_code: If present, it includes the original file source code as a comment at the beggining of\n        the file. This can be useful for debugging purposes.\n        ')
        
        # Assigning a Call to a Attribute (line 24):
        
        # Call to Module(...): (line 24)
        # Processing the call keyword arguments (line 24)
        kwargs_12 = {}
        # Getting the type of 'ast' (line 24)
        ast_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 34), 'ast', False)
        # Obtaining the member 'Module' of a type (line 24)
        Module_11 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 34), ast_10, 'Module')
        # Calling Module(args, kwargs) (line 24)
        Module_call_result_13 = invoke(stypy.reporting.localization.Localization(__file__, 24, 34), Module_11, *[], **kwargs_12)
        
        # Getting the type of 'self' (line 24)
        self_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'self')
        # Setting the type of the member 'type_inference_ast' of a type (line 24)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), self_14, 'type_inference_ast', Module_call_result_13)
        
        # Assigning a Name to a Attribute (line 25):
        # Getting the type of 'file_name' (line 25)
        file_name_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 25), 'file_name')
        # Getting the type of 'self' (line 25)
        self_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'self')
        # Setting the type of the member 'file_name' of a type (line 25)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), self_16, 'file_name', file_name_15)
        
        # Assigning a Name to a Attribute (line 26):
        # Getting the type of 'original_code' (line 26)
        original_code_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 29), 'original_code')
        # Getting the type of 'self' (line 26)
        self_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'self')
        # Setting the type of the member 'original_code' of a type (line 26)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), self_18, 'original_code', original_code_17)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @staticmethod
    @norecursion
    def get_postfix_src_code(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_postfix_src_code'
        module_type_store = module_type_store.open_function_context('get_postfix_src_code', 28, 4, False)
        
        # Passed parameters checking function
        TypeInferenceGeneratorVisitor.get_postfix_src_code.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceGeneratorVisitor.get_postfix_src_code.__dict__.__setitem__('stypy_type_of_self', None)
        TypeInferenceGeneratorVisitor.get_postfix_src_code.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceGeneratorVisitor.get_postfix_src_code.__dict__.__setitem__('stypy_function_name', 'get_postfix_src_code')
        TypeInferenceGeneratorVisitor.get_postfix_src_code.__dict__.__setitem__('stypy_param_names_list', ['cls'])
        TypeInferenceGeneratorVisitor.get_postfix_src_code.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceGeneratorVisitor.get_postfix_src_code.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceGeneratorVisitor.get_postfix_src_code.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceGeneratorVisitor.get_postfix_src_code.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceGeneratorVisitor.get_postfix_src_code.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceGeneratorVisitor.get_postfix_src_code.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, None, module_type_store, 'get_postfix_src_code', ['cls'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_postfix_src_code', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_postfix_src_code(...)' code ##################

        str_19 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, (-1)), 'str', '\n        All generated type inference programs has this code at the end, to capture generated TypeErrors and TypeWarnings\n        in known variables\n        :return:\n        ')
        
        # Call to format(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'stypy_functions_copy' (line 36)
        stypy_functions_copy_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 20), 'stypy_functions_copy', False)
        # Obtaining the member 'default_type_error_var_name' of a type (line 36)
        default_type_error_var_name_23 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 20), stypy_functions_copy_22, 'default_type_error_var_name')
        # Getting the type of 'stypy_functions_copy' (line 36)
        stypy_functions_copy_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 70), 'stypy_functions_copy', False)
        # Obtaining the member 'default_type_warning_var_name' of a type (line 36)
        default_type_warning_var_name_25 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 70), stypy_functions_copy_24, 'default_type_warning_var_name')
        # Processing the call keyword arguments (line 35)
        kwargs_26 = {}
        str_20 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 15), 'str', '\n{0} = stypy.errors.type_error.TypeError.get_error_msgs()\n{1} = stypy.errors.type_warning.TypeWarning.get_warning_msgs()\n')
        # Obtaining the member 'format' of a type (line 35)
        format_21 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 15), str_20, 'format')
        # Calling format(args, kwargs) (line 35)
        format_call_result_27 = invoke(stypy.reporting.localization.Localization(__file__, 35, 15), format_21, *[default_type_error_var_name_23, default_type_warning_var_name_25], **kwargs_26)
        
        # Assigning a type to the variable 'stypy_return_type' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'stypy_return_type', format_call_result_27)
        
        # ################# End of 'get_postfix_src_code(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_postfix_src_code' in the type store
        # Getting the type of 'stypy_return_type' (line 28)
        stypy_return_type_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_postfix_src_code'
        return stypy_return_type_28


    @norecursion
    def generic_visit(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'generic_visit'
        module_type_store = module_type_store.open_function_context('generic_visit', 38, 4, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceGeneratorVisitor.generic_visit.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceGeneratorVisitor.generic_visit.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceGeneratorVisitor.generic_visit.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceGeneratorVisitor.generic_visit.__dict__.__setitem__('stypy_function_name', 'TypeInferenceGeneratorVisitor.generic_visit')
        TypeInferenceGeneratorVisitor.generic_visit.__dict__.__setitem__('stypy_param_names_list', ['node'])
        TypeInferenceGeneratorVisitor.generic_visit.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceGeneratorVisitor.generic_visit.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceGeneratorVisitor.generic_visit.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceGeneratorVisitor.generic_visit.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceGeneratorVisitor.generic_visit.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceGeneratorVisitor.generic_visit.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceGeneratorVisitor.generic_visit', ['node'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'generic_visit', localization, ['node'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'generic_visit(...)' code ##################

        
        # Assigning a Call to a Name (line 39):
        
        # Call to list(...): (line 39)
        # Processing the call keyword arguments (line 39)
        kwargs_30 = {}
        # Getting the type of 'list' (line 39)
        list_29 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 20), 'list', False)
        # Calling list(args, kwargs) (line 39)
        list_call_result_31 = invoke(stypy.reporting.localization.Localization(__file__, 39, 20), list_29, *[], **kwargs_30)
        
        # Assigning a type to the variable 'new_stmts' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'new_stmts', list_call_result_31)
        
        # Call to append(...): (line 43)
        # Processing the call arguments (line 43)
        
        # Call to create_original_code_comment(...): (line 43)
        # Processing the call arguments (line 43)
        # Getting the type of 'self' (line 43)
        self_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 75), 'self', False)
        # Obtaining the member 'file_name' of a type (line 43)
        file_name_37 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 75), self_36, 'file_name')
        # Getting the type of 'self' (line 43)
        self_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 91), 'self', False)
        # Obtaining the member 'original_code' of a type (line 43)
        original_code_39 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 91), self_38, 'original_code')
        # Processing the call keyword arguments (line 43)
        kwargs_40 = {}
        # Getting the type of 'stypy_functions_copy' (line 43)
        stypy_functions_copy_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 25), 'stypy_functions_copy', False)
        # Obtaining the member 'create_original_code_comment' of a type (line 43)
        create_original_code_comment_35 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 25), stypy_functions_copy_34, 'create_original_code_comment')
        # Calling create_original_code_comment(args, kwargs) (line 43)
        create_original_code_comment_call_result_41 = invoke(stypy.reporting.localization.Localization(__file__, 43, 25), create_original_code_comment_35, *[file_name_37, original_code_39], **kwargs_40)
        
        # Processing the call keyword arguments (line 43)
        kwargs_42 = {}
        # Getting the type of 'new_stmts' (line 43)
        new_stmts_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'new_stmts', False)
        # Obtaining the member 'append' of a type (line 43)
        append_33 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), new_stmts_32, 'append')
        # Calling append(args, kwargs) (line 43)
        append_call_result_43 = invoke(stypy.reporting.localization.Localization(__file__, 43, 8), append_33, *[create_original_code_comment_call_result_41], **kwargs_42)
        
        
        # Call to append(...): (line 46)
        # Processing the call arguments (line 46)
        
        # Call to create_src_comment(...): (line 46)
        # Processing the call arguments (line 46)
        str_48 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 65), 'str', 'Import the stypy library')
        # Processing the call keyword arguments (line 46)
        kwargs_49 = {}
        # Getting the type of 'stypy_functions_copy' (line 46)
        stypy_functions_copy_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 25), 'stypy_functions_copy', False)
        # Obtaining the member 'create_src_comment' of a type (line 46)
        create_src_comment_47 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 25), stypy_functions_copy_46, 'create_src_comment')
        # Calling create_src_comment(args, kwargs) (line 46)
        create_src_comment_call_result_50 = invoke(stypy.reporting.localization.Localization(__file__, 46, 25), create_src_comment_47, *[str_48], **kwargs_49)
        
        # Processing the call keyword arguments (line 46)
        kwargs_51 = {}
        # Getting the type of 'new_stmts' (line 46)
        new_stmts_44 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'new_stmts', False)
        # Obtaining the member 'append' of a type (line 46)
        append_45 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), new_stmts_44, 'append')
        # Calling append(args, kwargs) (line 46)
        append_call_result_52 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), append_45, *[create_src_comment_call_result_50], **kwargs_51)
        
        
        # Call to append(...): (line 47)
        # Processing the call arguments (line 47)
        
        # Call to create_import_stypy(...): (line 47)
        # Processing the call keyword arguments (line 47)
        kwargs_57 = {}
        # Getting the type of 'stypy_functions_copy' (line 47)
        stypy_functions_copy_55 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 25), 'stypy_functions_copy', False)
        # Obtaining the member 'create_import_stypy' of a type (line 47)
        create_import_stypy_56 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 25), stypy_functions_copy_55, 'create_import_stypy')
        # Calling create_import_stypy(args, kwargs) (line 47)
        create_import_stypy_call_result_58 = invoke(stypy.reporting.localization.Localization(__file__, 47, 25), create_import_stypy_56, *[], **kwargs_57)
        
        # Processing the call keyword arguments (line 47)
        kwargs_59 = {}
        # Getting the type of 'new_stmts' (line 47)
        new_stmts_53 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'new_stmts', False)
        # Obtaining the member 'append' of a type (line 47)
        append_54 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), new_stmts_53, 'append')
        # Calling append(args, kwargs) (line 47)
        append_call_result_60 = invoke(stypy.reporting.localization.Localization(__file__, 47, 8), append_54, *[create_import_stypy_call_result_58], **kwargs_59)
        
        
        # Call to append(...): (line 50)
        # Processing the call arguments (line 50)
        
        # Call to create_src_comment(...): (line 50)
        # Processing the call arguments (line 50)
        str_65 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 65), 'str', 'Create the module type store')
        # Processing the call keyword arguments (line 50)
        kwargs_66 = {}
        # Getting the type of 'stypy_functions_copy' (line 50)
        stypy_functions_copy_63 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 25), 'stypy_functions_copy', False)
        # Obtaining the member 'create_src_comment' of a type (line 50)
        create_src_comment_64 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 25), stypy_functions_copy_63, 'create_src_comment')
        # Calling create_src_comment(args, kwargs) (line 50)
        create_src_comment_call_result_67 = invoke(stypy.reporting.localization.Localization(__file__, 50, 25), create_src_comment_64, *[str_65], **kwargs_66)
        
        # Processing the call keyword arguments (line 50)
        kwargs_68 = {}
        # Getting the type of 'new_stmts' (line 50)
        new_stmts_61 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'new_stmts', False)
        # Obtaining the member 'append' of a type (line 50)
        append_62 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 8), new_stmts_61, 'append')
        # Calling append(args, kwargs) (line 50)
        append_call_result_69 = invoke(stypy.reporting.localization.Localization(__file__, 50, 8), append_62, *[create_src_comment_call_result_67], **kwargs_68)
        
        
        # Call to append(...): (line 51)
        # Processing the call arguments (line 51)
        
        # Call to create_type_store(...): (line 51)
        # Processing the call keyword arguments (line 51)
        kwargs_74 = {}
        # Getting the type of 'stypy_functions_copy' (line 51)
        stypy_functions_copy_72 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 25), 'stypy_functions_copy', False)
        # Obtaining the member 'create_type_store' of a type (line 51)
        create_type_store_73 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 25), stypy_functions_copy_72, 'create_type_store')
        # Calling create_type_store(args, kwargs) (line 51)
        create_type_store_call_result_75 = invoke(stypy.reporting.localization.Localization(__file__, 51, 25), create_type_store_73, *[], **kwargs_74)
        
        # Processing the call keyword arguments (line 51)
        kwargs_76 = {}
        # Getting the type of 'new_stmts' (line 51)
        new_stmts_70 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'new_stmts', False)
        # Obtaining the member 'append' of a type (line 51)
        append_71 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), new_stmts_70, 'append')
        # Calling append(args, kwargs) (line 51)
        append_call_result_77 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), append_71, *[create_type_store_call_result_75], **kwargs_76)
        
        
        # Call to append(...): (line 53)
        # Processing the call arguments (line 53)
        
        # Call to create_program_section_src_comment(...): (line 53)
        # Processing the call arguments (line 53)
        str_82 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 81), 'str', 'Begin of the type inference program')
        # Processing the call keyword arguments (line 53)
        kwargs_83 = {}
        # Getting the type of 'stypy_functions_copy' (line 53)
        stypy_functions_copy_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 25), 'stypy_functions_copy', False)
        # Obtaining the member 'create_program_section_src_comment' of a type (line 53)
        create_program_section_src_comment_81 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 25), stypy_functions_copy_80, 'create_program_section_src_comment')
        # Calling create_program_section_src_comment(args, kwargs) (line 53)
        create_program_section_src_comment_call_result_84 = invoke(stypy.reporting.localization.Localization(__file__, 53, 25), create_program_section_src_comment_81, *[str_82], **kwargs_83)
        
        # Processing the call keyword arguments (line 53)
        kwargs_85 = {}
        # Getting the type of 'new_stmts' (line 53)
        new_stmts_78 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'new_stmts', False)
        # Obtaining the member 'append' of a type (line 53)
        append_79 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), new_stmts_78, 'append')
        # Calling append(args, kwargs) (line 53)
        append_call_result_86 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), append_79, *[create_program_section_src_comment_call_result_84], **kwargs_85)
        
        
        # Assigning a Call to a Name (line 56):
        
        # Call to StatementVisitor(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'self' (line 56)
        self_89 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 69), 'self', False)
        # Obtaining the member 'file_name' of a type (line 56)
        file_name_90 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 69), self_89, 'file_name')
        # Processing the call keyword arguments (line 56)
        kwargs_91 = {}
        # Getting the type of 'statement_visitor_copy' (line 56)
        statement_visitor_copy_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 29), 'statement_visitor_copy', False)
        # Obtaining the member 'StatementVisitor' of a type (line 56)
        StatementVisitor_88 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 29), statement_visitor_copy_87, 'StatementVisitor')
        # Calling StatementVisitor(args, kwargs) (line 56)
        StatementVisitor_call_result_92 = invoke(stypy.reporting.localization.Localization(__file__, 56, 29), StatementVisitor_88, *[file_name_90], **kwargs_91)
        
        # Assigning a type to the variable 'statements_visitor' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'statements_visitor', StatementVisitor_call_result_92)
        
        # Call to extend(...): (line 58)
        # Processing the call arguments (line 58)
        
        # Call to visit(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'node' (line 58)
        node_97 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 50), 'node', False)
        
        # Obtaining an instance of the builtin type 'list' (line 58)
        list_98 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 56), 'list')
        # Adding type elements to the builtin type 'list' instance (line 58)
        
        # Processing the call keyword arguments (line 58)
        kwargs_99 = {}
        # Getting the type of 'statements_visitor' (line 58)
        statements_visitor_95 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 25), 'statements_visitor', False)
        # Obtaining the member 'visit' of a type (line 58)
        visit_96 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 25), statements_visitor_95, 'visit')
        # Calling visit(args, kwargs) (line 58)
        visit_call_result_100 = invoke(stypy.reporting.localization.Localization(__file__, 58, 25), visit_96, *[node_97, list_98], **kwargs_99)
        
        # Processing the call keyword arguments (line 58)
        kwargs_101 = {}
        # Getting the type of 'new_stmts' (line 58)
        new_stmts_93 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'new_stmts', False)
        # Obtaining the member 'extend' of a type (line 58)
        extend_94 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), new_stmts_93, 'extend')
        # Calling extend(args, kwargs) (line 58)
        extend_call_result_102 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), extend_94, *[visit_call_result_100], **kwargs_101)
        
        
        # Call to append(...): (line 60)
        # Processing the call arguments (line 60)
        
        # Call to create_program_section_src_comment(...): (line 60)
        # Processing the call arguments (line 60)
        str_107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 81), 'str', 'End of the type inference program')
        # Processing the call keyword arguments (line 60)
        kwargs_108 = {}
        # Getting the type of 'stypy_functions_copy' (line 60)
        stypy_functions_copy_105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 25), 'stypy_functions_copy', False)
        # Obtaining the member 'create_program_section_src_comment' of a type (line 60)
        create_program_section_src_comment_106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 25), stypy_functions_copy_105, 'create_program_section_src_comment')
        # Calling create_program_section_src_comment(args, kwargs) (line 60)
        create_program_section_src_comment_call_result_109 = invoke(stypy.reporting.localization.Localization(__file__, 60, 25), create_program_section_src_comment_106, *[str_107], **kwargs_108)
        
        # Processing the call keyword arguments (line 60)
        kwargs_110 = {}
        # Getting the type of 'new_stmts' (line 60)
        new_stmts_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'new_stmts', False)
        # Obtaining the member 'append' of a type (line 60)
        append_104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), new_stmts_103, 'append')
        # Calling append(args, kwargs) (line 60)
        append_call_result_111 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), append_104, *[create_program_section_src_comment_call_result_109], **kwargs_110)
        
        
        # Call to Module(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'new_stmts' (line 61)
        new_stmts_114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 26), 'new_stmts', False)
        # Processing the call keyword arguments (line 61)
        kwargs_115 = {}
        # Getting the type of 'ast' (line 61)
        ast_112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 15), 'ast', False)
        # Obtaining the member 'Module' of a type (line 61)
        Module_113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 15), ast_112, 'Module')
        # Calling Module(args, kwargs) (line 61)
        Module_call_result_116 = invoke(stypy.reporting.localization.Localization(__file__, 61, 15), Module_113, *[new_stmts_114], **kwargs_115)
        
        # Assigning a type to the variable 'stypy_return_type' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'stypy_return_type', Module_call_result_116)
        
        # ################# End of 'generic_visit(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'generic_visit' in the type store
        # Getting the type of 'stypy_return_type' (line 38)
        stypy_return_type_117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_117)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'generic_visit'
        return stypy_return_type_117


# Assigning a type to the variable 'TypeInferenceGeneratorVisitor' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'TypeInferenceGeneratorVisitor', TypeInferenceGeneratorVisitor)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
