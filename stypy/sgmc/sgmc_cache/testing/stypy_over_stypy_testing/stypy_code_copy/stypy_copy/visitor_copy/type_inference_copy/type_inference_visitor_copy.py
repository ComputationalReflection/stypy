
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import ast
2: 
3: from stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy import stypy_functions_copy
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
28:     @classmethod
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

# 'from stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy import stypy_functions_copy' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/')
import_9281 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy')

if (type(import_9281) is not StypyTypeError):

    if (import_9281 != 'pyd_module'):
        __import__(import_9281)
        sys_modules_9282 = sys.modules[import_9281]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy', sys_modules_9282.module_type_store, module_type_store, ['stypy_functions_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_9282, sys_modules_9282.module_type_store, module_type_store)
    else:
        from stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy import stypy_functions_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy', None, module_type_store, ['stypy_functions_copy'], [stypy_functions_copy])

else:
    # Assigning a type to the variable 'stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy', import_9281)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import statement_visitor_copy' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/')
import_9283 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'statement_visitor_copy')

if (type(import_9283) is not StypyTypeError):

    if (import_9283 != 'pyd_module'):
        __import__(import_9283)
        sys_modules_9284 = sys.modules[import_9283]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'statement_visitor_copy', sys_modules_9284.module_type_store, module_type_store)
    else:
        import statement_visitor_copy

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'statement_visitor_copy', statement_visitor_copy, module_type_store)

else:
    # Assigning a type to the variable 'statement_visitor_copy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'statement_visitor_copy', import_9283)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/visitor_copy/type_inference_copy/')

# Declaration of the 'TypeInferenceGeneratorVisitor' class
# Getting the type of 'ast' (line 8)
ast_9285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 36), 'ast')
# Obtaining the member 'NodeVisitor' of a type (line 8)
NodeVisitor_9286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 36), ast_9285, 'NodeVisitor')

class TypeInferenceGeneratorVisitor(NodeVisitor_9286, ):
    str_9287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, (-1)), 'str', '\n    This visitor is responsible of generating type inference code AST Tree from standard Pyhon code contained in a\n    .py file. It just process the Module node, generating a fixed prefix code nodes, creating and running a\n    StatementVisitor object and appending a fixed postfix code at the end to form the full AST tree of the type\n    inference program created from a Python source code file.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 16)
        None_9288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 48), 'None')
        defaults = [None_9288]
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

        str_9289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, (-1)), 'str', '\n        Initialices the visitor.\n        :param file_name: File name of the source code whose ast will be parsed. This is needed for localization,\n        needed to report errors precisely.\n        :param original_code: If present, it includes the original file source code as a comment at the beggining of\n        the file. This can be useful for debugging purposes.\n        ')
        
        # Assigning a Call to a Attribute (line 24):
        
        # Call to Module(...): (line 24)
        # Processing the call keyword arguments (line 24)
        kwargs_9292 = {}
        # Getting the type of 'ast' (line 24)
        ast_9290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 34), 'ast', False)
        # Obtaining the member 'Module' of a type (line 24)
        Module_9291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 34), ast_9290, 'Module')
        # Calling Module(args, kwargs) (line 24)
        Module_call_result_9293 = invoke(stypy.reporting.localization.Localization(__file__, 24, 34), Module_9291, *[], **kwargs_9292)
        
        # Getting the type of 'self' (line 24)
        self_9294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'self')
        # Setting the type of the member 'type_inference_ast' of a type (line 24)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), self_9294, 'type_inference_ast', Module_call_result_9293)
        
        # Assigning a Name to a Attribute (line 25):
        # Getting the type of 'file_name' (line 25)
        file_name_9295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 25), 'file_name')
        # Getting the type of 'self' (line 25)
        self_9296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'self')
        # Setting the type of the member 'file_name' of a type (line 25)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), self_9296, 'file_name', file_name_9295)
        
        # Assigning a Name to a Attribute (line 26):
        # Getting the type of 'original_code' (line 26)
        original_code_9297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 29), 'original_code')
        # Getting the type of 'self' (line 26)
        self_9298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'self')
        # Setting the type of the member 'original_code' of a type (line 26)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), self_9298, 'original_code', original_code_9297)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def get_postfix_src_code(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_postfix_src_code'
        module_type_store = module_type_store.open_function_context('get_postfix_src_code', 28, 4, False)
        # Assigning a type to the variable 'self' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TypeInferenceGeneratorVisitor.get_postfix_src_code.__dict__.__setitem__('stypy_localization', localization)
        TypeInferenceGeneratorVisitor.get_postfix_src_code.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TypeInferenceGeneratorVisitor.get_postfix_src_code.__dict__.__setitem__('stypy_type_store', module_type_store)
        TypeInferenceGeneratorVisitor.get_postfix_src_code.__dict__.__setitem__('stypy_function_name', 'TypeInferenceGeneratorVisitor.get_postfix_src_code')
        TypeInferenceGeneratorVisitor.get_postfix_src_code.__dict__.__setitem__('stypy_param_names_list', [])
        TypeInferenceGeneratorVisitor.get_postfix_src_code.__dict__.__setitem__('stypy_varargs_param_name', None)
        TypeInferenceGeneratorVisitor.get_postfix_src_code.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TypeInferenceGeneratorVisitor.get_postfix_src_code.__dict__.__setitem__('stypy_call_defaults', defaults)
        TypeInferenceGeneratorVisitor.get_postfix_src_code.__dict__.__setitem__('stypy_call_varargs', varargs)
        TypeInferenceGeneratorVisitor.get_postfix_src_code.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TypeInferenceGeneratorVisitor.get_postfix_src_code.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TypeInferenceGeneratorVisitor.get_postfix_src_code', [], None, None, defaults, varargs, kwargs)

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

        str_9299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, (-1)), 'str', '\n        All generated type inference programs has this code at the end, to capture generated TypeErrors and TypeWarnings\n        in known variables\n        :return:\n        ')
        
        # Call to format(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'stypy_functions_copy' (line 36)
        stypy_functions_copy_9302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 20), 'stypy_functions_copy', False)
        # Obtaining the member 'default_type_error_var_name' of a type (line 36)
        default_type_error_var_name_9303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 20), stypy_functions_copy_9302, 'default_type_error_var_name')
        # Getting the type of 'stypy_functions_copy' (line 36)
        stypy_functions_copy_9304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 70), 'stypy_functions_copy', False)
        # Obtaining the member 'default_type_warning_var_name' of a type (line 36)
        default_type_warning_var_name_9305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 70), stypy_functions_copy_9304, 'default_type_warning_var_name')
        # Processing the call keyword arguments (line 35)
        kwargs_9306 = {}
        str_9300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 15), 'str', '\n{0} = stypy.errors.type_error.TypeError.get_error_msgs()\n{1} = stypy.errors.type_warning.TypeWarning.get_warning_msgs()\n')
        # Obtaining the member 'format' of a type (line 35)
        format_9301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 15), str_9300, 'format')
        # Calling format(args, kwargs) (line 35)
        format_call_result_9307 = invoke(stypy.reporting.localization.Localization(__file__, 35, 15), format_9301, *[default_type_error_var_name_9303, default_type_warning_var_name_9305], **kwargs_9306)
        
        # Assigning a type to the variable 'stypy_return_type' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'stypy_return_type', format_call_result_9307)
        
        # ################# End of 'get_postfix_src_code(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_postfix_src_code' in the type store
        # Getting the type of 'stypy_return_type' (line 28)
        stypy_return_type_9308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_9308)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_postfix_src_code'
        return stypy_return_type_9308


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
        kwargs_9310 = {}
        # Getting the type of 'list' (line 39)
        list_9309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 20), 'list', False)
        # Calling list(args, kwargs) (line 39)
        list_call_result_9311 = invoke(stypy.reporting.localization.Localization(__file__, 39, 20), list_9309, *[], **kwargs_9310)
        
        # Assigning a type to the variable 'new_stmts' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'new_stmts', list_call_result_9311)
        
        # Call to append(...): (line 43)
        # Processing the call arguments (line 43)
        
        # Call to create_original_code_comment(...): (line 43)
        # Processing the call arguments (line 43)
        # Getting the type of 'self' (line 43)
        self_9316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 75), 'self', False)
        # Obtaining the member 'file_name' of a type (line 43)
        file_name_9317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 75), self_9316, 'file_name')
        # Getting the type of 'self' (line 43)
        self_9318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 91), 'self', False)
        # Obtaining the member 'original_code' of a type (line 43)
        original_code_9319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 91), self_9318, 'original_code')
        # Processing the call keyword arguments (line 43)
        kwargs_9320 = {}
        # Getting the type of 'stypy_functions_copy' (line 43)
        stypy_functions_copy_9314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 25), 'stypy_functions_copy', False)
        # Obtaining the member 'create_original_code_comment' of a type (line 43)
        create_original_code_comment_9315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 25), stypy_functions_copy_9314, 'create_original_code_comment')
        # Calling create_original_code_comment(args, kwargs) (line 43)
        create_original_code_comment_call_result_9321 = invoke(stypy.reporting.localization.Localization(__file__, 43, 25), create_original_code_comment_9315, *[file_name_9317, original_code_9319], **kwargs_9320)
        
        # Processing the call keyword arguments (line 43)
        kwargs_9322 = {}
        # Getting the type of 'new_stmts' (line 43)
        new_stmts_9312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'new_stmts', False)
        # Obtaining the member 'append' of a type (line 43)
        append_9313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), new_stmts_9312, 'append')
        # Calling append(args, kwargs) (line 43)
        append_call_result_9323 = invoke(stypy.reporting.localization.Localization(__file__, 43, 8), append_9313, *[create_original_code_comment_call_result_9321], **kwargs_9322)
        
        
        # Call to append(...): (line 46)
        # Processing the call arguments (line 46)
        
        # Call to create_src_comment(...): (line 46)
        # Processing the call arguments (line 46)
        str_9328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 65), 'str', 'Import the stypy library')
        # Processing the call keyword arguments (line 46)
        kwargs_9329 = {}
        # Getting the type of 'stypy_functions_copy' (line 46)
        stypy_functions_copy_9326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 25), 'stypy_functions_copy', False)
        # Obtaining the member 'create_src_comment' of a type (line 46)
        create_src_comment_9327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 25), stypy_functions_copy_9326, 'create_src_comment')
        # Calling create_src_comment(args, kwargs) (line 46)
        create_src_comment_call_result_9330 = invoke(stypy.reporting.localization.Localization(__file__, 46, 25), create_src_comment_9327, *[str_9328], **kwargs_9329)
        
        # Processing the call keyword arguments (line 46)
        kwargs_9331 = {}
        # Getting the type of 'new_stmts' (line 46)
        new_stmts_9324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'new_stmts', False)
        # Obtaining the member 'append' of a type (line 46)
        append_9325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 8), new_stmts_9324, 'append')
        # Calling append(args, kwargs) (line 46)
        append_call_result_9332 = invoke(stypy.reporting.localization.Localization(__file__, 46, 8), append_9325, *[create_src_comment_call_result_9330], **kwargs_9331)
        
        
        # Call to append(...): (line 47)
        # Processing the call arguments (line 47)
        
        # Call to create_import_stypy(...): (line 47)
        # Processing the call keyword arguments (line 47)
        kwargs_9337 = {}
        # Getting the type of 'stypy_functions_copy' (line 47)
        stypy_functions_copy_9335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 25), 'stypy_functions_copy', False)
        # Obtaining the member 'create_import_stypy' of a type (line 47)
        create_import_stypy_9336 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 25), stypy_functions_copy_9335, 'create_import_stypy')
        # Calling create_import_stypy(args, kwargs) (line 47)
        create_import_stypy_call_result_9338 = invoke(stypy.reporting.localization.Localization(__file__, 47, 25), create_import_stypy_9336, *[], **kwargs_9337)
        
        # Processing the call keyword arguments (line 47)
        kwargs_9339 = {}
        # Getting the type of 'new_stmts' (line 47)
        new_stmts_9333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'new_stmts', False)
        # Obtaining the member 'append' of a type (line 47)
        append_9334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), new_stmts_9333, 'append')
        # Calling append(args, kwargs) (line 47)
        append_call_result_9340 = invoke(stypy.reporting.localization.Localization(__file__, 47, 8), append_9334, *[create_import_stypy_call_result_9338], **kwargs_9339)
        
        
        # Call to append(...): (line 50)
        # Processing the call arguments (line 50)
        
        # Call to create_src_comment(...): (line 50)
        # Processing the call arguments (line 50)
        str_9345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 65), 'str', 'Create the module type store')
        # Processing the call keyword arguments (line 50)
        kwargs_9346 = {}
        # Getting the type of 'stypy_functions_copy' (line 50)
        stypy_functions_copy_9343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 25), 'stypy_functions_copy', False)
        # Obtaining the member 'create_src_comment' of a type (line 50)
        create_src_comment_9344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 25), stypy_functions_copy_9343, 'create_src_comment')
        # Calling create_src_comment(args, kwargs) (line 50)
        create_src_comment_call_result_9347 = invoke(stypy.reporting.localization.Localization(__file__, 50, 25), create_src_comment_9344, *[str_9345], **kwargs_9346)
        
        # Processing the call keyword arguments (line 50)
        kwargs_9348 = {}
        # Getting the type of 'new_stmts' (line 50)
        new_stmts_9341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'new_stmts', False)
        # Obtaining the member 'append' of a type (line 50)
        append_9342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 8), new_stmts_9341, 'append')
        # Calling append(args, kwargs) (line 50)
        append_call_result_9349 = invoke(stypy.reporting.localization.Localization(__file__, 50, 8), append_9342, *[create_src_comment_call_result_9347], **kwargs_9348)
        
        
        # Call to append(...): (line 51)
        # Processing the call arguments (line 51)
        
        # Call to create_type_store(...): (line 51)
        # Processing the call keyword arguments (line 51)
        kwargs_9354 = {}
        # Getting the type of 'stypy_functions_copy' (line 51)
        stypy_functions_copy_9352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 25), 'stypy_functions_copy', False)
        # Obtaining the member 'create_type_store' of a type (line 51)
        create_type_store_9353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 25), stypy_functions_copy_9352, 'create_type_store')
        # Calling create_type_store(args, kwargs) (line 51)
        create_type_store_call_result_9355 = invoke(stypy.reporting.localization.Localization(__file__, 51, 25), create_type_store_9353, *[], **kwargs_9354)
        
        # Processing the call keyword arguments (line 51)
        kwargs_9356 = {}
        # Getting the type of 'new_stmts' (line 51)
        new_stmts_9350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'new_stmts', False)
        # Obtaining the member 'append' of a type (line 51)
        append_9351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), new_stmts_9350, 'append')
        # Calling append(args, kwargs) (line 51)
        append_call_result_9357 = invoke(stypy.reporting.localization.Localization(__file__, 51, 8), append_9351, *[create_type_store_call_result_9355], **kwargs_9356)
        
        
        # Call to append(...): (line 53)
        # Processing the call arguments (line 53)
        
        # Call to create_program_section_src_comment(...): (line 53)
        # Processing the call arguments (line 53)
        str_9362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 81), 'str', 'Begin of the type inference program')
        # Processing the call keyword arguments (line 53)
        kwargs_9363 = {}
        # Getting the type of 'stypy_functions_copy' (line 53)
        stypy_functions_copy_9360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 25), 'stypy_functions_copy', False)
        # Obtaining the member 'create_program_section_src_comment' of a type (line 53)
        create_program_section_src_comment_9361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 25), stypy_functions_copy_9360, 'create_program_section_src_comment')
        # Calling create_program_section_src_comment(args, kwargs) (line 53)
        create_program_section_src_comment_call_result_9364 = invoke(stypy.reporting.localization.Localization(__file__, 53, 25), create_program_section_src_comment_9361, *[str_9362], **kwargs_9363)
        
        # Processing the call keyword arguments (line 53)
        kwargs_9365 = {}
        # Getting the type of 'new_stmts' (line 53)
        new_stmts_9358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'new_stmts', False)
        # Obtaining the member 'append' of a type (line 53)
        append_9359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), new_stmts_9358, 'append')
        # Calling append(args, kwargs) (line 53)
        append_call_result_9366 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), append_9359, *[create_program_section_src_comment_call_result_9364], **kwargs_9365)
        
        
        # Assigning a Call to a Name (line 56):
        
        # Call to StatementVisitor(...): (line 56)
        # Processing the call arguments (line 56)
        # Getting the type of 'self' (line 56)
        self_9369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 69), 'self', False)
        # Obtaining the member 'file_name' of a type (line 56)
        file_name_9370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 69), self_9369, 'file_name')
        # Processing the call keyword arguments (line 56)
        kwargs_9371 = {}
        # Getting the type of 'statement_visitor_copy' (line 56)
        statement_visitor_copy_9367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 29), 'statement_visitor_copy', False)
        # Obtaining the member 'StatementVisitor' of a type (line 56)
        StatementVisitor_9368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 29), statement_visitor_copy_9367, 'StatementVisitor')
        # Calling StatementVisitor(args, kwargs) (line 56)
        StatementVisitor_call_result_9372 = invoke(stypy.reporting.localization.Localization(__file__, 56, 29), StatementVisitor_9368, *[file_name_9370], **kwargs_9371)
        
        # Assigning a type to the variable 'statements_visitor' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'statements_visitor', StatementVisitor_call_result_9372)
        
        # Call to extend(...): (line 58)
        # Processing the call arguments (line 58)
        
        # Call to visit(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'node' (line 58)
        node_9377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 50), 'node', False)
        
        # Obtaining an instance of the builtin type 'list' (line 58)
        list_9378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 56), 'list')
        # Adding type elements to the builtin type 'list' instance (line 58)
        
        # Processing the call keyword arguments (line 58)
        kwargs_9379 = {}
        # Getting the type of 'statements_visitor' (line 58)
        statements_visitor_9375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 25), 'statements_visitor', False)
        # Obtaining the member 'visit' of a type (line 58)
        visit_9376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 25), statements_visitor_9375, 'visit')
        # Calling visit(args, kwargs) (line 58)
        visit_call_result_9380 = invoke(stypy.reporting.localization.Localization(__file__, 58, 25), visit_9376, *[node_9377, list_9378], **kwargs_9379)
        
        # Processing the call keyword arguments (line 58)
        kwargs_9381 = {}
        # Getting the type of 'new_stmts' (line 58)
        new_stmts_9373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'new_stmts', False)
        # Obtaining the member 'extend' of a type (line 58)
        extend_9374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), new_stmts_9373, 'extend')
        # Calling extend(args, kwargs) (line 58)
        extend_call_result_9382 = invoke(stypy.reporting.localization.Localization(__file__, 58, 8), extend_9374, *[visit_call_result_9380], **kwargs_9381)
        
        
        # Call to append(...): (line 60)
        # Processing the call arguments (line 60)
        
        # Call to create_program_section_src_comment(...): (line 60)
        # Processing the call arguments (line 60)
        str_9387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 81), 'str', 'End of the type inference program')
        # Processing the call keyword arguments (line 60)
        kwargs_9388 = {}
        # Getting the type of 'stypy_functions_copy' (line 60)
        stypy_functions_copy_9385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 25), 'stypy_functions_copy', False)
        # Obtaining the member 'create_program_section_src_comment' of a type (line 60)
        create_program_section_src_comment_9386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 25), stypy_functions_copy_9385, 'create_program_section_src_comment')
        # Calling create_program_section_src_comment(args, kwargs) (line 60)
        create_program_section_src_comment_call_result_9389 = invoke(stypy.reporting.localization.Localization(__file__, 60, 25), create_program_section_src_comment_9386, *[str_9387], **kwargs_9388)
        
        # Processing the call keyword arguments (line 60)
        kwargs_9390 = {}
        # Getting the type of 'new_stmts' (line 60)
        new_stmts_9383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'new_stmts', False)
        # Obtaining the member 'append' of a type (line 60)
        append_9384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 8), new_stmts_9383, 'append')
        # Calling append(args, kwargs) (line 60)
        append_call_result_9391 = invoke(stypy.reporting.localization.Localization(__file__, 60, 8), append_9384, *[create_program_section_src_comment_call_result_9389], **kwargs_9390)
        
        
        # Call to Module(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'new_stmts' (line 61)
        new_stmts_9394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 26), 'new_stmts', False)
        # Processing the call keyword arguments (line 61)
        kwargs_9395 = {}
        # Getting the type of 'ast' (line 61)
        ast_9392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 15), 'ast', False)
        # Obtaining the member 'Module' of a type (line 61)
        Module_9393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 15), ast_9392, 'Module')
        # Calling Module(args, kwargs) (line 61)
        Module_call_result_9396 = invoke(stypy.reporting.localization.Localization(__file__, 61, 15), Module_9393, *[new_stmts_9394], **kwargs_9395)
        
        # Assigning a type to the variable 'stypy_return_type' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'stypy_return_type', Module_call_result_9396)
        
        # ################# End of 'generic_visit(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'generic_visit' in the type store
        # Getting the type of 'stypy_return_type' (line 38)
        stypy_return_type_9397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_9397)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'generic_visit'
        return stypy_return_type_9397


# Assigning a type to the variable 'TypeInferenceGeneratorVisitor' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'TypeInferenceGeneratorVisitor', TypeInferenceGeneratorVisitor)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
