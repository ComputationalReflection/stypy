from stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy import *
from stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy import core_language_copy, functions_copy, stypy_functions_copy, data_structures_copy

import statement_visitor_copy
import ast

class ValueVisitor(ast.NodeVisitor):
    """
    Visitor for value nodes, that are AST nodes that return tuples of statement lists and temp variable nodes when
    the result of a certain operation represented by the node is stored. If when processing a statement
    node one of its children nodes is a value node, a StatementVisitor is automatically run to process this child node.
    This visitor allows us to generate three address code for the type inference programs
    """

    def __init__(self, filename):
        self.file_name = filename

    def visit(self, node, context):
        """Visit a node."""
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        if visitor == self.generic_visit:
            return visitor(node, context + [method])
        return visitor(node, context)

    def __visit_instruction_body(self, body, context):
        new_stmts = []
        temp = []

        if not isinstance(body, list):
            body = [body]

        # Visit all body instructions
        for stmt in body:
            stmts, temp = self.visit(stmt, context)
            if isinstance(stmts, list):
                new_stmts.extend(stmts)
            else:
                new_stmts.append(stmts)
                # new_stmts = stypy_functions_copy.flatten_lists(new_stmts)

        return new_stmts, temp

    def generic_visit(self, node, context):
        if hasattr(statement_visitor_copy.StatementVisitor, context[-1]):
            return self.visit_statement(node, context[:-1])
        else:
            raise Exception("VALUE VISITOR: " + context[-1] + " is not yet implemented")
            # """Called if no explicit visitor function exists for a node."""
            # for field, value in ast.iter_fields(node):
            # if isinstance(value, list):
            # for item in value:
            # if isinstance(item, ast.AST):
            # self.visit(item, context)
            # elif isinstance(value, ast.AST):
            # self.visit(value, context)

    def visit_statement(self, node, context):
        s_visitor = statement_visitor_copy.StatementVisitor(self.file_name)

        return s_visitor.visit(node, context), []

    # ######################################### ASSIGNMENTS #############################################

    def visit_AugAssign(self, node, context):
        # print inspect.stack()[0][3]

        operator_name = type(node.op).__name__.lower()

        # Obtain target (left part)
        if type(node.target) is ast.Name:
            get_target_stmts, target_var = stypy_functions_copy.create_get_type_of(node.target.id, node.lineno,
                                                                              node.col_offset)
        else:
            get_target_obj_left_stmts, target_left_var = self.visit(node.target.value, context)
            # get_target_obj_left_stmts, target_left_var = stypy_functions_copy.create_get_type_of(node.target.value.id,
            #                                                                                 node.lineno,
            #                                                                                 node.col_offset)

            if type(node.target) is ast.Attribute:
                get_type_of_member, target_var = stypy_functions_copy.create_get_type_of_member(target_left_var,
                                                                                           node.target.attr,
                                                                                           node.lineno,
                                                                                           node.col_offset)
            else:
                instructions, var = self.visit(node.target, context)
                # get_type_of_member, target_var = stypy_functions_copy.create_get_type_of_member(target_left_var,
                #                                                                        var, node.lineno,
                #
                #                                                             node.col_offset)
                target_var = var
                get_type_of_member = instructions  # stypy_functions_copy.flatten_lists(instructions, get_type_of_member)

            get_target_stmts = stypy_functions_copy.flatten_lists(get_target_obj_left_stmts, get_type_of_member)

        # Obtain right part
        right_stmts, temp_op2 = self.visit(node.value, context)

        # Call operator
        assign_stmts, temp_assign = stypy_functions_copy.create_binary_operator(operator_name, target_var, temp_op2,
                                                                           node.lineno, node.col_offset)

        # Set result type
        if type(node.target) is ast.Name:
            augment_assign = stypy_functions_copy.create_set_type_of(node.target.id, temp_assign, node.lineno,
                                                                node.col_offset)
        else:
            get_target_obj_stmts, target_var = self.visit(node.target.value, context)
            # get_target_obj_stmts, target_var = stypy_functions_copy.create_get_type_of(node.target.value.id, node.lineno,
            #                                                                       node.col_offset)

            if type(node.target) is ast.Attribute:
                set_type = stypy_functions_copy.create_set_type_of_member(target_var, node.target.attr, temp_assign,
                                                                     node.lineno,
                                                                     node.col_offset)
            else:
                # temp_value: type to add
                # target_var: container
                # node.slice: subscript to calculate
                slice_stmts, slice_var = self.visit(node.target.slice, context)
                set_type_of_stmts = stypy_functions_copy.create_add_stored_type(target_var, slice_var, temp_assign,
                                                                           node.lineno,
                                                                           node.col_offset)
                set_type = stypy_functions_copy.flatten_lists(slice_stmts, set_type_of_stmts)

            augment_assign = stypy_functions_copy.flatten_lists(get_target_obj_stmts, set_type)

        return stypy_functions_copy.flatten_lists(stypy_functions_copy.create_blank_line(),
                                             get_target_stmts,
                                             right_stmts,
                                             assign_stmts,
                                             augment_assign,
                                             stypy_functions_copy.create_blank_line()), temp_assign

    # ######################################### EXPRESSIONS, NAMES, LITERALS ##########################################

    def visit_Expr(self, node, context):
        # print inspect.stack()[0][3]

        if stypy_functions_copy.is_src_comment(node) or stypy_functions_copy.is_blank_line(node):
            return node, []

        expr = ast.Expr()
        stypy_functions_copy.assign_line_and_column(expr, node)
        # Evaluate the expression value, add the result.
        expr.value, temp_op = self.visit(node.value, context)

        return expr.value, temp_op

    def visit_Attribute(self, node, context):
        # print inspect.stack()[0][3]

        value_stmts, owner_var = self.visit(node.value, context)
        if type(node.ctx) == ast.Load:
            get_member_stmts, member_var = stypy_functions_copy.create_get_type_of_member(owner_var, node.attr, node.lineno,
                                                                                     node.col_offset)

            # To call a method we need all the source information (self object)
            tuple_node = ast.Tuple()
            tuple_node.elts = [member_var, owner_var]

            return stypy_functions_copy.flatten_lists(
                value_stmts,
                get_member_stmts), member_var  # tuple_node
            # else:
            # return stypy_functions_copy.flatten_lists(
            # value_stmts,
            # get_member_stmts), member_var
        else:
            return stypy_functions_copy.flatten_lists(
                value_stmts,
                core_language_copy.create_attribute(owner_var, node.attr, False, node.lineno, node.col_offset)), []

    def visit_Name(self, node, context):
        # print inspect.stack()[0][3]

        if type(node.ctx) == ast.Load:
            return stypy_functions_copy.create_get_type_of(node.id, node.lineno, node.col_offset)
        else:
            return core_language_copy.create_Name(node.id, False, node.lineno, node.col_offset), []

    def visit_Num(self, node, context):
        # print inspect.stack()[0][3]

        localization = stypy_functions_copy.create_localization(node.lineno, node.col_offset)
        get_type_call = core_language_copy.create_Name('get_builtin_type', line=node.lineno, column=node.col_offset)
        get_type_call_param = core_language_copy.create_str(type(node.n).__name__)
        get_type_call_param2 = core_language_copy.create_num(node.n, node.lineno, node.col_offset)
        call = functions_copy.create_call(get_type_call, [localization, get_type_call_param, get_type_call_param2])
        return stypy_functions_copy.create_temp_Assign(call, line=node.lineno, column=node.col_offset)

        # return [], core_language_copy.create_Name(type(node.n).__name__, line=node.lineno, column=node.col_offset)

    def visit_Str(self, node, context):
        # print inspect.stack()[0][3]
        localization = stypy_functions_copy.create_localization(node.lineno, node.col_offset)
        get_type_call = core_language_copy.create_Name('get_builtin_type', line=node.lineno, column=node.col_offset)
        get_type_call_param = core_language_copy.create_str(type(node.s).__name__)
        get_type_call_param2 = core_language_copy.create_str(node.s, node.lineno, node.col_offset)
        call = functions_copy.create_call(get_type_call, [localization, get_type_call_param, get_type_call_param2])
        return stypy_functions_copy.create_temp_Assign(call, line=node.lineno, column=node.col_offset)
        # return [], core_language_copy.create_Name(type(node.s).__name__, line=node.lineno, column=node.col_offset)

    def visit_Index(self, node, context):
        # print inspect.stack()[0][3]
        return self.visit(node.value, context)

    def visit_Subscript(self, node, context):
        # print inspect.stack()[0][3]

        stmts = []

        # Type of accessor between []
        node.slice.lineno = node.lineno
        node.slice.col_offset = node.col_offset
        slice_stmts, slice_var = self.visit(node.slice, context)
        stmts.append(slice_stmts)

        # Obtain the subscripted data structure
        value_stmts, value_var = self.visit(node.value, context)
        stmts.append(value_stmts)

        # Slice type assigment
        slice_assign, slice_var = stypy_functions_copy.create_temp_Assign(slice_var, node.lineno, node.col_offset)
        stmts.append(slice_assign)

        member_stmts, member_var = stypy_functions_copy.create_get_type_of_member(value_var, '__getitem__',
                                                                             node.lineno, node.col_offset)
        stmts.append(member_stmts)

        # Call to the subscript
        comment = stypy_functions_copy.create_src_comment("Calling the subscript to obtain elements type", node.lineno)
        stmts.append(comment)
        localization = stypy_functions_copy.create_localization(node.lineno, node.col_offset)
        call_attribute = core_language_copy.create_attribute(member_var, 'invoke')

        call_to_subscript = functions_copy.create_call(call_attribute, [localization, slice_var])

        elem_stmts, elem_var = stypy_functions_copy.create_temp_Assign(call_to_subscript, node.lineno, node.col_offset)
        stmts.append(elem_stmts)

        return stypy_functions_copy.flatten_lists(stypy_functions_copy.create_blank_line(),
                                             stmts,
                                             stypy_functions_copy.create_blank_line()), elem_var

    def visit_Repr(self, node, context):
        return self.visit(node.value, context)

    # ################################## ARITHMETHIC AND LOGICAL OPERATIONS #########################################

    def visit_BoolOp(self, node, context):
        # print inspect.stack()[0][3]
        bool_op_stmts = [stypy_functions_copy.create_src_comment("Evaluating a boolean operation")]

        # Evaluate left operand
        left_stmts, temp_op1 = self.visit(node.values[0], context)
        if type(temp_op1) is ast.Tuple:
            temp_op1 = temp_op1.elts[0]

        bool_op_stmts.append(left_stmts)

        for i in range(1, len(node.values)):
            # Evaluate right operand
            right_stmts, temp_op2 = self.visit(node.values[i], context)
            if type(temp_op2) is ast.Tuple:
                temp_op2 = temp_op2.elts[0]

            bool_op_stmts.append(right_stmts)

            # Operator symbol
            operator_name = type(node.op).__name__.lower()

            # Call operator
            operator_stmts, result_var = stypy_functions_copy.create_binary_operator(operator_name, temp_op1, temp_op2,
                                                                                node.lineno, node.col_offset)
            bool_op_stmts.append(operator_stmts)
            temp_op1 = result_var  # For the next iteration

        return stypy_functions_copy.flatten_lists(stypy_functions_copy.create_blank_line(),
                                             bool_op_stmts,
                                             stypy_functions_copy.create_blank_line()
                                             ), result_var

    def visit_BinOp(self, node, context):
        # print inspect.stack()[0][3]

        # Evaluate left operand
        left_stmts, temp_op1 = self.visit(node.left, context)

        if type(temp_op1) is ast.Tuple:
            temp_op1 = temp_op1.elts[0]

        # Evaluate right operand
        right_stmts, temp_op2 = self.visit(node.right, context)

        if type(temp_op2) is ast.Tuple:
            temp_op2 = temp_op2.elts[0]

        # Operator symbol
        operator_name = type(node.op).__name__.lower()

        # Call operator
        operator_stmts, result_var = stypy_functions_copy.create_binary_operator(operator_name, temp_op1, temp_op2,
                                                                            node.lineno, node.col_offset)

        return stypy_functions_copy.flatten_lists(stypy_functions_copy.create_blank_line(),
                                             left_stmts,
                                             right_stmts,
                                             operator_stmts,
                                             stypy_functions_copy.create_blank_line()
                                             ), result_var

    def visit_UnaryOp(self, node, context):
        # print inspect.stack()[0][3]

        # Evaluate the operand
        right_stmts, temp_op = self.visit(node.operand, context)
        operator_name = type(node.op).__name__.lower()

        operator_stmts, result_var = stypy_functions_copy.create_unary_operator(operator_name, temp_op,
                                                                           node.lineno, node.col_offset)

        return stypy_functions_copy.flatten_lists(stypy_functions_copy.create_blank_line(),
                                             right_stmts,
                                             operator_stmts,
                                             stypy_functions_copy.create_blank_line()
                                             ), result_var

    def visit_Compare(self, node, context):
        # Multiple comparison syntax is supported in Python:
        # http://www.java2s.com/Tutorial/Python/0080__Operator
        # /multiplecomparisonscanbemadeonthesamelineevaluatedinlefttorightorder.htm

        # print inspect.stack()[0][3]

        all_assign_stmts = []
        final_op_result = None

        # Evaluate left part
        left_stmts, left_op = self.visit(node.left, context)
        if type(left_op) is ast.Tuple:
            left_op = left_op.elts[0]
        all_assign_stmts.append(left_stmts)

        for i in range(len(node.comparators)):
            # Evaluate each one of the right parts
            right_stmts, rigth_op = self.visit(node.comparators[i], context)
            if type(rigth_op) is ast.Tuple:
                rigth_op = rigth_op.elts[0]

            all_assign_stmts.append(right_stmts)

            # Build logical operator and add its statements
            operator_name = type(node.ops[i]).__name__.lower()
            operator_stmts, temp_op_result = stypy_functions_copy.create_binary_operator(operator_name, left_op, rigth_op,
                                                                                    node.lineno, node.col_offset)
            all_assign_stmts.append(operator_stmts)

            # If there is more than one comparison, the new left part is the right part recently evaluated
            # The result of the previous operation is also stored to perform the 'and'
            left_stmts, left_op = right_stmts, rigth_op

            if i >= 1:
                # If more than one comparison exist, they are related with the previous one by using an 'and' logical
                # operator, according to Python documentation.
                operator_stmts, final_op_result = stypy_functions_copy.create_binary_operator(
                    'and_',
                    final_op_result,
                    temp_op_result,
                    node.lineno,
                    node.col_offset)

                all_assign_stmts.append(operator_stmts)
            else:
                final_op_result = temp_op_result

            previous_operation_result = temp_op_result

        return stypy_functions_copy.flatten_lists(stypy_functions_copy.create_blank_line(),
                                             all_assign_stmts,
                                             stypy_functions_copy.create_blank_line()), final_op_result

    # ######################################### functions_copy #############################################

    def visit_Call(self, node, context):
        # print inspect.stack()[0][3]

        context.append(node)
        # Localization of the function call
        localization = stypy_functions_copy.create_localization(node.lineno, node.col_offset)
        call_stmts = []
        arguments = []
        keyword_arguments = {}

        name_to_call = stypy_functions_copy.get_descritive_element_name(node.func)

        # Obtain the function to be called
        call_stmts.append(stypy_functions_copy.create_src_comment("Calling '{0}'".format(name_to_call), node.lineno))
        context.append(node)

        get_type_of_stmts, function_to_call = self.visit(node.func, context)
        context.remove(node)

        if len(node.args) > 0:
            call_stmts.append(stypy_functions_copy.create_src_comment("Processing call arguments", node.lineno))
        # First call parameters are built from standard parameters plus var args (args + starargs)
        # Evaluate arguments of the call
        for arg in node.args:
            stmts, temp = self.visit(arg, context)
            call_stmts.append(stmts)
            arguments.append(temp)

        # Evaluate arguments of the call
        if not node.starargs is None:
            #for arg in node.starargs:
            #stmts, temp = self.visit(arg, context)
            stmts, temp = self.visit(node.starargs, context)
            call_stmts.append(stmts)
            arguments.append(temp)

        call_stmts.append(stypy_functions_copy.create_src_comment("Processing call keyword arguments", node.lineno))
        # Second call parameters are built from the keywords and keyword args (keywords + kwargs)
        # Evaluate the keyword arguments of the call
        for arg in node.keywords:
            stmts, temp = self.visit(arg, context)
            call_stmts.append(stmts)
            keyword_arguments[arg.arg] = temp

        # Evaluate arguments of the call
        if not node.kwargs is None:
            #for arg in node.kwargs:
            #stmts, temp = self.visit(arg, context)
            stmts, temp = self.visit(node.kwargs, context)
            call_stmts.append(stmts)
            keyword_arguments[temp.id] = temp
            #keyword_arguments[node.kwargs.id] = temp

        dict_arguments = data_structures_copy.create_keyword_dict(keyword_arguments)
        dict_assign_stmts, dict_assign = stypy_functions_copy.create_temp_Assign(dict_arguments, node.lineno,
                                                                            node.col_offset)
        call_stmts.append(dict_assign_stmts)

        call_stmts.append(stypy_functions_copy.create_set_unreferenced_var_check(False))
        call_stmts.append(get_type_of_stmts)
        call_stmts.append(stypy_functions_copy.create_set_unreferenced_var_check(True))

        call_stmts.append(stypy_functions_copy.create_src_comment("Performing the call", node.lineno))
        call = create_call_to_type_inference_code(function_to_call, localization, keywords=[], kwargs=dict_assign,
                                                  starargs=arguments,
                                                  line=node.lineno, column=node.col_offset)

        assign_stmts, temp_assign = stypy_functions_copy.create_temp_Assign(call, node.lineno, node.col_offset)
        call_stmts.append(assign_stmts)

        context.remove(node)
        return stypy_functions_copy.flatten_lists(
            stypy_functions_copy.create_blank_line(),
            call_stmts,
            stypy_functions_copy.create_blank_line(),
        ), temp_assign

    def visit_keyword(self, node, context):
        # print inspect.stack()[0][3]

        get_value_stmts, temp_value = self.visit(node.value, context)
        assign_keyword_value, keyword_value = stypy_functions_copy.create_temp_Assign(temp_value, context[-1].lineno,
                                                                                 context[-1].col_offset)
        return stypy_functions_copy.flatten_lists(get_value_stmts, assign_keyword_value), keyword_value

    # ########################################## HIGHER ORDER functions_copy ##########################################

    def visit_Lambda(self, node, context):
        # Function declaration localization
        decl_localization = core_language_copy.create_Name('localization', False, line=node.lineno, column=node.col_offset)

        # The 'norecursion' decorator, mandatory in every stypy code generation to enable the type inference program
        # to not to hang on recursive calls.
        decorator_list = core_language_copy.create_Name('norecursion', line=node.lineno - 1, column=node.col_offset)

        defaults_types = []
        for elem in node.args.defaults:
            stmts, type_ = self.visit(elem, context)
            defaults_types.append(type_)

        node.name = stypy_functions_copy.new_temp_lambda_str()

        # Create and add the function definition header
        function_def = functions_copy.create_function_def(node.name,
                                                     decl_localization,
                                                     decorator_list,
                                                     [],
                                                     line=node.lineno,
                                                     column=node.col_offset)

        # Defaults are stored in a variable at the beginning of the function
        function_def.body.append(stypy_functions_copy.create_src_comment("Assign values to the parameters with defaults"))
        function_def.body.append(core_language_copy.create_Assign(core_language_copy.create_Name("defaults"),
                                                             data_structures_copy.create_list(defaults_types)))

        # Generate code from setting a new context in the type store
        function_def.body.append(stypy_functions_copy.create_src_comment
                                 ("Create a new context for function '{0}'".format(node.name)))
        context_set = functions_copy.create_context_set(node.name, node.lineno,
                                                   node.col_offset)
        function_def.body.append(context_set)

        # Generate code for arity checking of the function arguments and assigning them to suitable local variables
        function_def.body.append(stypy_functions_copy.create_src_comment("Passed parameters checking function"))
        f_preamble = functions_copy.create_arg_number_test(node)
        function_def.body.append(f_preamble)

        # Generate code for create a new stack push for error reporting.
        function_def.body.append(stypy_functions_copy.create_src_comment("Stacktrace push for error reporting"))
        declared_arguments = functions_copy.obtain_arg_list(node.args, functions_copy.is_method(context))
        stack_push = functions_copy.create_stacktrace_push(node.name, declared_arguments)
        function_def.body.append(stack_push)

        # Initialize the variable where the return of the function will be stored.
        # This is needed due to a single return statement must exist within a function in order to not to conflict with
        # the SSA algorithm
        function_def.body.append(stypy_functions_copy.create_src_comment("Default return type storage variable (SSA)"))
        function_def.body.append(stypy_functions_copy.create_default_return_variable())

        function_def.body.append(stypy_functions_copy.create_blank_line())
        function_def.body.append(
            stypy_functions_copy.create_src_comment("Begin of the function '{0}' code".format(node.name)))

        # Lambdas only have a single instruction body
        stmts, temp = self.visit(node.body, context)
        function_def.body.append(stmts)
        function_def.body = stypy_functions_copy.flatten_lists(function_def.body)

        # Return value to the default return value name
        return_comment = stypy_functions_copy.create_src_comment("Assigning the return type of the lambda function")
        function_def.body.append(return_comment)
        default_function_ret_var_name = core_language_copy.create_Name(stypy_functions_copy.default_function_ret_var_name)
        ret_assign = core_language_copy.create_Assign(default_function_ret_var_name, temp)
        function_def.body.append(ret_assign)

        function_def.body.append(
            stypy_functions_copy.create_end_block_src_comment("End of the function '{0}' code".format(node.name)))

        # Pop the error reporting stack trace
        function_def.body.append(stypy_functions_copy.create_src_comment("Stacktrace pop (error reporting)"))
        stack_pop = functions_copy.create_stacktrace_pop()
        function_def.body.append(stack_pop)

        # Generate code for pop the function context.
        function_def.body.append(stypy_functions_copy.create_src_comment
                                 ("Destroy the context of function '{0}'".format(node.name)))
        context_unset = functions_copy.create_context_unset()
        function_def.body.append(context_unset)

        # Finally, return the return value (contained in a predefined var name)
        # in the single return statement of each function.
        function_def.body.append(stypy_functions_copy.create_src_comment("Return type of the function"))
        function_def.body.append(stypy_functions_copy.create_return_from_function(node.lineno, node.col_offset))

        # Register the function type outside the function context
        # register_expr = stypy_functions_copy.create_set_type_of(node.name, core_language_copy.create_Name(node.name),
        # node.lineno, node.col_offset)
        register_expr = functions_copy.create_type_for_lambda_function(node.name, node.name,
                                                                  node.lineno, node.col_offset)
        get_lambda_stmt, lambda_var = stypy_functions_copy.create_get_type_of(node.name, node.lineno, node.col_offset)
        # TODO: Delete name at the end?
        return stypy_functions_copy.flatten_lists(function_def, register_expr, get_lambda_stmt), lambda_var

    # ########################################### DATA STRUCTURES #####################################################

    def visit_List(self, node, context):
        # print inspect.stack()[0][3]

        list_stmts = []

        localization = stypy_functions_copy.create_localization(node.lineno, node.col_offset)

        # Get a list instance
        call_comment = stypy_functions_copy.create_src_comment("Obtaining an instance of the builtin type 'list'",
                                                          node.lineno)
        list_stmts.append(call_comment)
        get_list_func = core_language_copy.create_Name("get_builtin_type")
        param1 = core_language_copy.create_str('list')
        get_list_call = functions_copy.create_call(get_list_func, [localization, param1])
        get_list_stmts, list_var = stypy_functions_copy.create_temp_Assign(get_list_call, node.lineno, node.col_offset)
        list_stmts.append(get_list_stmts)

        comment = stypy_functions_copy.create_src_comment("Adding type elements to the builtin type 'list' instance",
                                                     node.lineno)
        list_stmts.append(comment)

        # Prepare the add type function
        add_type_func = core_language_copy.create_attribute(list_var.id, "add_type")
        # Iterate over all elements of the list and add them
        for element in node.elts:
            comment = stypy_functions_copy.create_src_comment("Adding element type", node.lineno)
            list_stmts.append(comment)
            element_stmts, list_element = self.visit(element, context)
            list_stmts.append(stypy_functions_copy.flatten_lists(element_stmts))

            add_type_to_list_call = functions_copy.create_call_expression(add_type_func, [localization, list_element])
            list_stmts.append(add_type_to_list_call)

        return stypy_functions_copy.flatten_lists(stypy_functions_copy.create_blank_line(),
                                             list_stmts,
                                             stypy_functions_copy.create_blank_line()), list_var

    def visit_Tuple(self, node, context):
        # print inspect.stack()[0][3]

        tuple_stmts = []

        localization = stypy_functions_copy.create_localization(node.lineno, node.col_offset)

        # Get a list instance
        call_comment = stypy_functions_copy.create_src_comment("Obtaining an instance of the builtin type 'tuple'",
                                                          node.lineno)
        tuple_stmts.append(call_comment)
        get_tuple_func = core_language_copy.create_Name("get_builtin_type")
        param1 = core_language_copy.create_str('tuple')
        get_tuple_call = functions_copy.create_call(get_tuple_func, [localization, param1])
        get_tuple_stmts, list_var = stypy_functions_copy.create_temp_Assign(get_tuple_call, node.lineno, node.col_offset)
        tuple_stmts.append(get_tuple_stmts)

        comment = stypy_functions_copy.create_src_comment("Adding type elements to the builtin type 'tuple' instance",
                                                     node.lineno)
        tuple_stmts.append(comment)

        # Prepare the add type function
        add_type_func = core_language_copy.create_attribute(list_var.id, "add_type")
        # Iterate over all elements of the list and add them
        for element in node.elts:
            comment = stypy_functions_copy.create_src_comment("Adding element type", node.lineno)
            tuple_stmts.append(comment)
            element_stmts, tuple_element = self.visit(element, context)
            tuple_stmts.append(stypy_functions_copy.flatten_lists(element_stmts))

            add_type_to_list_call = functions_copy.create_call_expression(add_type_func, [localization, tuple_element])
            tuple_stmts.append(add_type_to_list_call)

        return stypy_functions_copy.flatten_lists(stypy_functions_copy.create_blank_line(),
                                             tuple_stmts,
                                             stypy_functions_copy.create_blank_line()), list_var

        # # print inspect.stack()[0][3]
        #
        # tuple_stmts = []
        # tuple_elmts = []
        #
        # if type(node.ctx) == ast.Store:
        # for element in node.elts:
        # elem_to_store, temp = self.visit(element, context)
        # tuple_elmts.append(elem_to_store)
        #
        # node.elts = tuple_elmts
        #
        # return stypy_functions_copy.flatten_lists(tuple_stmts, node), []
        # else:
        #     for element in node.elts:
        #         elem_stmts, temp = self.visit(element, context)
        #         tuple_stmts.append(elem_stmts)
        #         tuple_elmts.append(temp)
        #
        #     node.elts = tuple_elmts
        #     tuple_assign, tuple_var = stypy_functions_copy.create_temp_Assign(node, node.lineno, node.col_offset)
        #
        #     return stypy_functions_copy.flatten_lists(tuple_stmts, tuple_assign), tuple_var

    def visit_Slice(self, node, context):
        # print inspect.stack()[0][3]
        localization = stypy_functions_copy.create_localization(node.lineno, node.col_offset)

        if node.lower is not None:
            lower_inst, lower_var = self.visit(node.lower, context)
        else:
            lower_inst, lower_var = [], core_language_copy.create_Name("None")

        if node.upper is not None:
            upper_inst, upper_var = self.visit(node.upper, context)
        else:
            upper_inst, upper_var = [], core_language_copy.create_Name("None")

        if node.step is not None:
            step_inst, step_var = self.visit(node.upper, context)
        else:
            step_inst, step_var = [], core_language_copy.create_Name("None")

        get_slice_func = core_language_copy.create_Name("ensure_slice_bounds")
        get_list_call = functions_copy.create_call(get_slice_func, [localization, lower_var, upper_var, step_var])
        assign_stmts, assign_var = stypy_functions_copy.create_temp_Assign(get_list_call, 0, 0)
        return stypy_functions_copy.flatten_lists(lower_inst,
                                             upper_inst,
                                             step_inst,
                                             assign_stmts), assign_var

    def visit_ListComp(self, node, context):
        src_comment = stypy_functions_copy.create_src_comment("Calculating list comprehension")
        localization = stypy_functions_copy.create_localization(node.lineno, node.col_offset)

        # context_set = functions_copy.create_context_set("list comprehension expression")

        elt_stmts, elt_var = self.visit(node.elt, context)

        generator_stmts = []
        generator_vars = []
        for gen in node.generators:
            stypy_functions_copy.assign_line_and_column(gen, node)
            temp_stmts, temp_var = self.visit(gen, context)
            generator_stmts.append(temp_stmts)
            generator_vars.append(generator_vars)

        get_type_call = core_language_copy.create_Name('get_builtin_type', line=node.lineno, column=node.col_offset)
        get_type_call_param = core_language_copy.create_str('list')
        call = functions_copy.create_call(get_type_call, [localization, get_type_call_param])
        ret_list_stmts, ret_list_var = stypy_functions_copy.create_temp_Assign(call, line=node.lineno,
                                                                          column=node.col_offset)

        invoke_set_element = core_language_copy.create_attribute(ret_list_var, 'set_elements_type')

        # context_unset = functions_copy.create_context_unset()
        false_ = core_language_copy.create_Name('False')
        call_to_set_element = functions_copy.create_call_expression(invoke_set_element, [localization, elt_var, false_])

        return stypy_functions_copy.flatten_lists(src_comment, generator_stmts, elt_stmts, ret_list_stmts,
                                             call_to_set_element), ret_list_var

    def visit_comprehension(self, node, context):
        src_comment = stypy_functions_copy.create_src_comment("Calculating comprehension expression")
        if_stmts = []
        if_vars = []
        for if_ in node.ifs:
            temp_stmts, temp_var = self.visit(if_, context)
            if_stmts.append(temp_stmts)
            if_vars.append(temp_var)

        iter_stmts, iter_var = self.visit(node.iter, context)
        get_elements_call = core_language_copy.create_attribute(iter_var, 'get_elements_type', line=node.lineno,
                                                           column=node.col_offset)
        iter_elems_call = functions_copy.create_call(get_elements_call, [], line=node.lineno,
                                                column=node.col_offset)
        iter_elems_stmts, iter_elems_var = stypy_functions_copy.create_temp_Assign(iter_elems_call, line=node.lineno,
                                                                              column=node.col_offset)

        target_var, temp = self.visit(node.target, context)
        if isinstance(target_var, ast.Name):
            set_type_of_target_stmts = stypy_functions_copy.create_set_type_of(target_var.id, iter_elems_var, lineno=node.lineno,
                                                                      col_offset=node.col_offset)
        else:
            set_type_of_target_stmts = stypy_functions_copy.create_set_type_of(temp.id, iter_elems_var, lineno=node.lineno,
                                                                      col_offset=node.col_offset)
            set_type_of_target_stmts = target_var + set_type_of_target_stmts

        # localization = stypy_functions_copy.create_localization(node.lineno, node.col_offset)

        return stypy_functions_copy.flatten_lists(src_comment, iter_stmts, iter_elems_stmts,
                                             set_type_of_target_stmts, if_stmts), target_var

    def visit_Set(self, node, context):
        set_stmts = []

        localization = stypy_functions_copy.create_localization(node.lineno, node.col_offset)

        # Get a set instance
        call_comment = stypy_functions_copy.create_src_comment("Obtaining an instance of the builtin type 'set'",
                                                          node.lineno)
        set_stmts.append(call_comment)
        get_sett_func = core_language_copy.create_Name("get_builtin_type")
        param1 = core_language_copy.create_str('set')
        get_set_call = functions_copy.create_call(get_sett_func, [localization, param1])
        get_set_stmts, set_var = stypy_functions_copy.create_temp_Assign(get_set_call, node.lineno, node.col_offset)
        set_stmts.append(get_set_stmts)

        comment = stypy_functions_copy.create_src_comment("Adding type elements to the builtin type 'set' instance",
                                                     node.lineno)
        set_stmts.append(comment)

        # Prepare the add type function
        add_type_func = core_language_copy.create_attribute(set_var.id, "add_type")
        # Iterate over all elements of the list and add them
        for element in node.elts:
            comment = stypy_functions_copy.create_src_comment("Adding element type", node.lineno)
            set_stmts.append(comment)
            element_stmts, set_element = self.visit(element, context)
            set_stmts.append(stypy_functions_copy.flatten_lists(element_stmts))

            add_type_to_set_call = functions_copy.create_call_expression(add_type_func, [localization, set_element])
            set_stmts.append(add_type_to_set_call)

        return stypy_functions_copy.flatten_lists(stypy_functions_copy.create_blank_line(),
                                             set_stmts,
                                             stypy_functions_copy.create_blank_line()), set_var

    def visit_SetComp(self, node, context):
        src_comment = stypy_functions_copy.create_src_comment("Calculating set comprehension")
        localization = stypy_functions_copy.create_localization(node.lineno, node.col_offset)

        context_set = functions_copy.create_context_set("set comprehension expression", node.lineno, node.col_offset)

        elt_stmts, elt_var = self.visit(node.elt, context)

        generator_stmts = []
        generator_vars = []
        for gen in node.generators:
            stypy_functions_copy.assign_line_and_column(gen, node)
            temp_stmts, temp_var = self.visit(gen, context)
            generator_stmts.append(temp_stmts)
            generator_vars.append(generator_vars)

        get_type_call = core_language_copy.create_Name('get_builtin_type', line=node.lineno, column=node.col_offset)
        get_type_call_param = core_language_copy.create_str('set')
        call = functions_copy.create_call(get_type_call, [localization, get_type_call_param])
        ret_set_stmts, ret_set_var = stypy_functions_copy.create_temp_Assign(call, line=node.lineno,
                                                                        column=node.col_offset)

        invoke_set_element = core_language_copy.create_attribute(ret_set_var, 'set_elements_type')

        context_unset = functions_copy.create_context_unset()
        false_ = core_language_copy.create_Name('False')
        call_to_set_element = functions_copy.create_call_expression(invoke_set_element, [localization, elt_var, false_])

        return stypy_functions_copy.flatten_lists(src_comment, context_set, generator_stmts, elt_stmts, ret_set_stmts,
                                             context_unset, call_to_set_element), ret_set_var

    def visit_Dict(self, node, context):
        set_stmts = []

        localization = stypy_functions_copy.create_localization(node.lineno, node.col_offset)

        # Get a dict instance
        call_comment = stypy_functions_copy.create_src_comment("Obtaining an instance of the builtin type 'dict'",
                                                          node.lineno)
        set_stmts.append(call_comment)
        get_sett_func = core_language_copy.create_Name("get_builtin_type")
        param1 = core_language_copy.create_str('dict')
        get_set_call = functions_copy.create_call(get_sett_func, [localization, param1])
        get_set_stmts, set_var = stypy_functions_copy.create_temp_Assign(get_set_call, node.lineno, node.col_offset)
        set_stmts.append(get_set_stmts)

        comment = stypy_functions_copy.create_src_comment("Adding type elements to the builtin type 'dict' instance",
                                                     node.lineno)
        set_stmts.append(comment)

        # Prepare the add type function
        add_type_func = core_language_copy.create_attribute(set_var.id, "add_key_and_value_type")
        # Iterate over all elements of the dict and add them
        for key, value in zip(node.keys, node.values):
            comment = stypy_functions_copy.create_src_comment("Adding element type (key, value)", node.lineno)
            set_stmts.append(comment)
            key_stmts, key_var = self.visit(key, context)
            value_stmts, value_var = self.visit(value, context)

            tuple_node = ast.Tuple()
            tuple_node.elts = [key_var, value_var]

            # set_stmts.append(stypy_functions_copy.flatten_lists(key_stmts))
            # set_stmts.append(stypy_functions_copy.flatten_lists(value_stmts))
            set_stmts.append(key_stmts)
            set_stmts.append(value_stmts)

            add_type_to_set_call = functions_copy.create_call_expression(add_type_func, [localization, tuple_node])
            set_stmts.append(add_type_to_set_call)

        return stypy_functions_copy.flatten_lists(stypy_functions_copy.create_blank_line(),
                                             set_stmts,
                                             stypy_functions_copy.create_blank_line()), set_var

    def visit_DictComp(self, node, context):
        src_comment = stypy_functions_copy.create_src_comment("Calculating dict comprehension")
        localization = stypy_functions_copy.create_localization(node.lineno, node.col_offset)

        context_set = functions_copy.create_context_set("dict comprehension expression", node.lineno,
                                                   node.col_offset)

        key_stmts, key_var = self.visit(node.key, context)
        value_stmts, value_var = self.visit(node.value, context)

        generator_stmts = []
        generator_vars = []
        for gen in node.generators:
            stypy_functions_copy.assign_line_and_column(gen, node)
            temp_stmts, temp_var = self.visit(gen, context)
            generator_stmts.append(temp_stmts)
            generator_vars.append(generator_vars)

        get_type_call = core_language_copy.create_Name('get_builtin_type', line=node.lineno, column=node.col_offset)
        get_type_call_param = core_language_copy.create_str('dict')
        call = functions_copy.create_call(get_type_call, [localization, get_type_call_param])
        ret_set_stmts, ret_set_var = stypy_functions_copy.create_temp_Assign(call, line=node.lineno,
                                                                        column=node.col_offset)

        invoke_set_element = core_language_copy.create_attribute(ret_set_var, 'add_key_and_value_type')

        context_unset = functions_copy.create_context_unset()
        tuple_node = ast.Tuple()
        tuple_node.elts = [key_var, value_var]
        false_ = core_language_copy.create_Name('False')
        call_to_set_element = functions_copy.create_call_expression(invoke_set_element, [localization, tuple_node, false_])

        return stypy_functions_copy.flatten_lists(src_comment, context_set, generator_stmts, key_stmts, value_stmts,
                                             ret_set_stmts,
                                             context_unset, call_to_set_element), ret_set_var

    def visit_GeneratorExp(self, node, context):
        src_comment = stypy_functions_copy.create_src_comment("Calculating generator expression")
        localization = stypy_functions_copy.create_localization(node.lineno, node.col_offset)

        context_set = functions_copy.create_context_set("list comprehension expression", node.lineno, node.col_offset)

        elt_stmts, elt_var = self.visit(node.elt, context)

        generator_stmts = []
        generator_vars = []
        for gen in node.generators:
            stypy_functions_copy.assign_line_and_column(gen, node)
            temp_stmts, temp_var = self.visit(gen, context)
            generator_stmts.append(temp_stmts)
            generator_vars.append(generator_vars)

        get_type_call = core_language_copy.create_Name('get_builtin_type', line=node.lineno, column=node.col_offset)
        get_type_call_param = core_language_copy.create_str('list')
        call = functions_copy.create_call(get_type_call, [localization, get_type_call_param])
        ret_list_stmts, ret_list_var = stypy_functions_copy.create_temp_Assign(call, line=node.lineno,
                                                                          column=node.col_offset)

        invoke_set_element = core_language_copy.create_attribute(ret_list_var, 'set_elements_type')

        context_unset = functions_copy.create_context_unset()
        call_to_set_element = functions_copy.create_call_expression(invoke_set_element, [localization, elt_var])

        return stypy_functions_copy.flatten_lists(src_comment, context_set, generator_stmts, elt_stmts, ret_list_stmts,
                                             context_unset, call_to_set_element), ret_list_var

    def visit_IfExp(self, node, context):
        if_stmt_body = []
        if_stmt_orelse = []

        begin_if_comment = stypy_functions_copy.create_blank_line()

        # Process conditional expression of the if
        condition_stmt, if_stmt_test = self.visit(node.test, context)

        # Test the type of the if condition
        condition_comment = stypy_functions_copy.create_src_comment("Testing the type of an if expression", node.lineno)
        attribute = core_language_copy.create_Name("is_suitable_condition")
        localization = stypy_functions_copy.create_localization(node.lineno, node.col_offset)
        test_condition_call = functions_copy.create_call_expression(attribute, [localization, if_stmt_test])

        # Begin the if body
        if_stmt_body.append(stypy_functions_copy.create_src_comment("SSA begins for if expression", node.lineno))
        clone_stmt1, type_store_var1 = stypy_functions_copy.create_clone_type_store()
        if_stmt_body.append(clone_stmt1)

        # Process if body sentences
        body_stmts, body_value = self.__visit_instruction_body(node.body, context)
        if_stmt_body.extend(body_stmts)

        # Process else branch
        if_stmt_body.append(
            stypy_functions_copy.create_src_comment("SSA branch for the else part of an if expression", node.lineno))
        clone_stmt2, type_store_var2 = stypy_functions_copy.create_clone_type_store()
        if_stmt_body.append(clone_stmt2)

        set_type_store1 = stypy_functions_copy.create_set_type_store(type_store_var1)
        if_stmt_body.append(set_type_store1)

        # Process else body sentences
        orelse_stmts, orelse_value = self.__visit_instruction_body(node.orelse, context)
        if_stmt_orelse.extend(orelse_stmts)

        # Join if
        if_stmt_orelse.append(stypy_functions_copy.create_src_comment("SSA join for if expression", node.lineno))
        clone_stmt3, type_store_var3 = stypy_functions_copy.create_clone_type_store()
        if_stmt_orelse.append(clone_stmt3)
        join_stmt, final_type_store = stypy_functions_copy.create_join_type_store("ssa_join_with_else_branch",
                                                                             [type_store_var1, type_store_var2,
                                                                              type_store_var3])
        if_stmt_orelse.append(join_stmt)

        # Calculate the final type store
        set_type_store2 = stypy_functions_copy.create_set_type_store(final_type_store)
        if_stmt_orelse.append(set_type_store2)

        # Unify all if body statements
        all_if_stmts = if_stmt_body + if_stmt_orelse

        # Unify all if sentences
        if_stmt = stypy_functions_copy.flatten_lists(condition_comment, test_condition_call, all_if_stmts)

        union_type = core_language_copy.create_attribute("union_type", "UnionType")
        union_add = core_language_copy.create_attribute(union_type, "add")

        union_call = functions_copy.create_call(union_add, [body_value, orelse_value])

        ret_assign_stmts, ret_var = stypy_functions_copy.create_temp_Assign(union_call, line=node.lineno,
                                                                       column=node.col_offset)

        end_if_comment = stypy_functions_copy.create_blank_line()

        return stypy_functions_copy.flatten_lists(begin_if_comment,
                                             condition_stmt,
                                             if_stmt,
                                             ret_assign_stmts,
                                             end_if_comment), ret_var

    def visit_Ellipsis(self, node, context):
        return [], []

    def visit_ExtSlice(self, node, context):
        # localization = stypy_functions_copy.create_localization(node.lineno, node.col_offset)
        stmts = []
        dim_vars = []
        for dim in node.dims:
            stypy_functions_copy.assign_line_and_column(dim, node)
            dim_stmts, dim_var = self.visit(dim, context)
            stmts.append(dim_stmts)
            dim_vars.append(dim_var)

        tuple_node = ast.Tuple()
        tuple_node.elts = dim_vars

        return stypy_functions_copy.flatten_lists(stmts), tuple_node

    def visit_alias(self, node, context):
        # localization = stypy_functions_copy.create_localization(node.lineno, node.col_offset)
        return_name = core_language_copy.create_str(node.name)
        alias_method_call = []

        if not node.asname is None:
            alias_name = core_language_copy.create_str(node.asname)
            alias_method_call = stypy_functions_copy.create_add_alias(alias_name, return_name, node.lineno, node.col_offset)

        return stypy_functions_copy.flatten_lists(alias_method_call), return_name
