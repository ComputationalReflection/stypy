import os

from stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy import *
from stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy import core_language_copy, functions_copy, stypy_functions_copy, data_structures_copy, \
    conditional_statements_copy, idioms_copy
import value_visitor_copy
import ast

class StatementVisitor(ast.NodeVisitor):
    """
    Visitor for statement nodes, that are AST nodes that return only statement lists. If when processing a statement
    node one of its children nodes is a value node, a ValueVisitor is automatically run to process this child node.
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

    def generic_visit(self, node, context):
        if hasattr(value_visitor_copy.ValueVisitor, context[-1]):
            return self.visit_value(node, context[:-1])
        else:
            raise Exception("SENTENCE VISITOR: " + context[-1] + " is not yet implemented")

    def __visit_instruction_body(self, body, context):
        new_stmts = []

        # Visit all body instructions
        v_visitor = value_visitor_copy.ValueVisitor(self.file_name)
        for stmt in body:
            stmts, temp = v_visitor.visit(stmt, context)
            if isinstance(stmts, list):
                new_stmts.extend(stmts)
            else:
                new_stmts.append(stmts)
                # new_stmts = stypy_functions_copy.flatten_lists(new_stmts)

        return new_stmts

    def visit_value(self, node, context):
        v_visitor = value_visitor_copy.ValueVisitor(self.file_name)

        result, temp = v_visitor.visit(node, context)

        return result, temp

    # ######################################### MAIN MODULE #############################################

    def visit_Module(self, node, context):
        # print inspect.stack()[0][3]

        context.append(node)
        stmts = self.__visit_instruction_body(node.body, context)
        context.remove(node)

        return stmts

    # ######################################### functions_copy #############################################

    def visit_FunctionDef(self, node, context):
        # print inspect.stack()[0][3]

        self.building_function_name = node.name

        # Function declaration localization
        decl_localization = core_language_copy.create_Name('localization', False, line=node.lineno, column=node.col_offset)

        # Decorators are declarative properties of methods or functions_copy that give problems if directly ported to
        # type-inference equivalent code. For now, we only translate the @staticmethod decorator.
        decorator_list = []
        if len(node.decorator_list) > 0:
            for dec in node.decorator_list:
                if isinstance(dec, ast.Name):
                    if dec.id == "staticmethod":
                        decorator_list.append(dec)

        # The 'norecursion' decorator, mandatory in every stypy code generation to enable the type inference program
        # to not to hang on recursive calls.
        decorator_list.append(core_language_copy.create_Name('norecursion', line=node.lineno - 1, column=node.col_offset))

        defaults_types = []
        defaults_stmts = []

        for elem in node.args.defaults:
            stmts, type_ = self.visit(elem, context)
            defaults_types.append(type_)
            defaults_stmts.append(stmts)

        # Create and add the function definition header
        function_def = functions_copy.create_function_def(node.name,
                                                     decl_localization,
                                                     decorator_list,
                                                     context,
                                                     line=node.lineno,
                                                     column=node.col_offset)

        # Defaults are stored in a variable at the beginning of the function
        function_def.body.append(stypy_functions_copy.create_src_comment("Assign values to the parameters with defaults"))
        function_def.body.append(stypy_functions_copy.create_set_unreferenced_var_check(False))
        function_def.body.append(defaults_stmts)
        function_def.body.append(stypy_functions_copy.create_set_unreferenced_var_check(True))
        function_def.body.append(core_language_copy.create_Assign(core_language_copy.create_Name("defaults"),
                                                             data_structures_copy.create_list(defaults_types)))

        # Generate code from setting a new context in the type store
        function_def.body.append(stypy_functions_copy.create_src_comment
                                 ("Create a new context for function '{0}'".format(node.name)))
        context_set = functions_copy.create_context_set(node.name, node.lineno,
                                                   node.col_offset)
        function_def.body.append(context_set)

        if functions_copy.is_method(context):
            function_def.body.append(
                stypy_functions_copy.create_set_type_of('self', core_language_copy.create_Name('type_of_self'), node.lineno + 1,
                                                   node.col_offset))

        # Generate code for arity checking of the function arguments and assigning them to suitable local variables
        function_def.body.append(stypy_functions_copy.create_src_comment("Passed parameters checking function"))

        # Generate argument number test
        f_preamble = functions_copy.create_arg_number_test(node, context)

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

        context.append(node)
        # Visit the function body
        function_def.body.extend(self.__visit_instruction_body(node.body, context))
        context.remove(node)

        function_def.body.append(
            stypy_functions_copy.create_end_block_src_comment("End of the function '{0}' code".format(node.name)))

        # Pop the error reporting stack trace
        function_def.body.append(stypy_functions_copy.create_src_comment("Stacktrace pop (error reporting)"))
        stack_pop = functions_copy.create_stacktrace_pop()
        function_def.body.append(stack_pop)

        # Generate code for pop the function context.
        function_def.body.append(stypy_functions_copy.create_src_comment
                                 ("Destroy the context of function '{0}'".format(node.name)))

        function_def.body.append(stypy_functions_copy.create_store_return_from_function(node.lineno, node.col_offset))
        context_unset = functions_copy.create_context_unset()
        function_def.body.append(context_unset)

        # Finally, return the return value (contained in a predefined var name)
        # in the single return statement of each function.
        if not functions_copy.is_constructor(node):
            function_def.body.append(stypy_functions_copy.create_src_comment("Return type of the function"))
            function_def.body.append(stypy_functions_copy.create_return_from_function(node.lineno, 0))

        if not functions_copy.is_method(context):
            # Register the function type outside the function context
            register_expr = stypy_functions_copy.create_set_type_of(node.name, core_language_copy.create_Name(node.name),
                                                               node.lineno, node.col_offset)
        else:
            register_expr = []

        return stypy_functions_copy.flatten_lists(function_def, register_expr)

    def visit_Return(self, node, context):
        # print inspect.stack()[0][3]

        union_type = core_language_copy.create_attribute("union_type", "UnionType")
        union_add = core_language_copy.create_attribute(union_type, "add")
        default_function_ret_var = core_language_copy.create_Name(stypy_functions_copy.default_function_ret_var_name)
        if node.value is not None:
            get_value_stmts, value_var = self.visit_value(node.value, context)
        else:
            get_value_stmts, value_var = [], core_language_copy.create_Name("types.NoneType")

        union_call = functions_copy.create_call(union_add, [default_function_ret_var, value_var])

        # default_function_ret_var = core_language_copy.create_Name(stypy_functions_copy.default_function_ret_var_name)
        # ret_assign = core_language_copy.create_Assign(default_function_ret_var, union_call)
        ret_assign = stypy_functions_copy.assign_as_return_type(union_call)
        return stypy_functions_copy.flatten_lists(get_value_stmts, ret_assign)

    # ################################################ CLASSES ##########################################

    def visit_ClassDef(self, node, context):
        # print inspect.stack()[0][3]

        self.building_class_name = node.name

        # Writes the instruction: from stypy import *
        initial_comment = stypy_functions_copy.create_src_comment("Declaration of the '{0}' class".format(node.name))

        # Obtaining the type of the superclasses of this class (if they exist)
        superclass_inst = []
        new_bases = []
        for class_name in node.bases:
            superclass_, type_var = self.visit(class_name,
                                               context)  # stypy_functions_copy.create_get_type_of(class_name.id, node.lineno, node.col_offset)
            # superclass_type, superclass_var = stypy_functions_copy.create_temp_Assign(superclass_, node.lineno,
            #                                                                      node.col_offset)
            # superclass_inst.append(superclass_type)
            # new_bases.append(superclass_var)
            superclass_inst.append(superclass_)
            new_bases.append(type_var)

        node.bases = []

        # Remove class decorators (it may interfere in type inference generated code)
        node.decorator_list = []

        context.append(node)
        # Visit all body instructions
        node.body = self.__visit_instruction_body(node.body, context)

        context.remove(node)
        # Register the function type outside the function context
        register_expr = stypy_functions_copy.create_set_type_of(node.name, core_language_copy.create_Name(node.name),
                                                           node.lineno, node.col_offset)

        # Base type assignment
        if len(new_bases) > 0:
            class_obj_inst, class_var = stypy_functions_copy.create_get_type_of(node.name,
                                                                           node.lineno, node.col_offset)
            change_base_types_att = core_language_copy.create_attribute(class_var, 'change_base_types')
            bases_tuple = core_language_copy.create_type_tuple(*new_bases)
            localization = stypy_functions_copy.create_localization(node.lineno, node.col_offset)
            change_bases_call = functions_copy.create_call_expression(change_base_types_att, [localization, bases_tuple])
            change_bases = [class_obj_inst, change_bases_call]
        else:
            change_bases = []

        return stypy_functions_copy.flatten_lists(initial_comment, superclass_inst, node,
                                             stypy_functions_copy.create_blank_line(), register_expr, change_bases)

    # ################################################ ASSIGNMENTS ###################################################

    def visit_Assign(self, node, context):
        # print inspect.stack()[0][3]

        get_value_stmts, temp_value = self.visit_value(node.value, context)

        if type(temp_value) is ast.Tuple:
            temp_value = temp_value.elts[0]

        if isinstance(node.targets[0], ast.Name):
            name = node.targets[0]
            set_type_of_stmts = stypy_functions_copy.create_set_type_of(name.id, temp_value, name.lineno,
                                                                   name.col_offset)
        else:
            get_target_stmts, target_var = self.visit(node.targets[0].value, context)
            if type(target_var) is ast.Tuple:
                target_var = target_var.elts[0]

            if isinstance(node.targets[0], ast.Subscript):
                # temp_value: type to add
                # target_var: container
                # node.slice: subscript to calculate
                slice_stmts, slice_var = self.visit(node.targets[0].slice, context)
                set_type_of_stmts = stypy_functions_copy.create_add_stored_type(target_var, slice_var, temp_value,
                                                                           node.targets[0].lineno,
                                                                           node.targets[0].col_offset)
                set_type_of_stmts = stypy_functions_copy.flatten_lists(get_target_stmts, slice_stmts, set_type_of_stmts)

            else:
                set_type_of_stmts = stypy_functions_copy.create_set_type_of_member(target_var, node.targets[0].attr,
                                                                              temp_value,
                                                                              node.targets[0].lineno,
                                                                              node.targets[0].col_offset)

                set_type_of_stmts = stypy_functions_copy.flatten_lists(get_target_stmts, set_type_of_stmts)

        return stypy_functions_copy.flatten_lists(get_value_stmts, set_type_of_stmts)

    # ################################################ PASS NODES #####################################################

    def visit_Pass(self, node, context):
        return node

    # ################################################ IF #####################################################

    def visit_If(self, node, context):
        # print inspect.stack()[0][3]

        if_stmt_body = []
        if_stmt_orelse = []

        begin_if_comment = stypy_functions_copy.create_blank_line()

        is_an_idiom, left_stmts_tuple, rigth_stmts_tuple, idiom_name = idioms_copy.is_recognized_idiom(node.test, self,
                                                                                                  context)

        if is_an_idiom:
            condition_comment = stypy_functions_copy.create_src_comment(
                "Type idiom detected: calculating its left and rigth part", node.lineno)
            # Statements for the left part of the idiom
            if_stmt_body += left_stmts_tuple[0]
            # Statements for the right part of the idiom
            if_stmt_body += rigth_stmts_tuple[0]

            test_condition_call = []
            condition_stmt = []
            if_stmt_body.append(stypy_functions_copy.create_blank_line())
        else:
            # Process conditional expression of the if
            condition_stmt, if_stmt_test = self.visit(node.test, context)

            # Test the type of the if condition
            condition_comment = stypy_functions_copy.create_src_comment("Testing the type of an if condition", node.lineno)
            attribute = core_language_copy.create_Name("is_suitable_condition")
            localization = stypy_functions_copy.create_localization(node.lineno, node.col_offset)
            test_condition_call = functions_copy.create_call_expression(attribute, [localization, if_stmt_test])

        more_types_temp_var = None
        if is_an_idiom:
            idiom_if_func = core_language_copy.create_Name(idioms_copy.get_recognized_idiom_function(idiom_name), node.lineno,
                                                      node.col_offset)
            call_to_idiom_if_func = functions_copy.create_call(idiom_if_func, [left_stmts_tuple[1], rigth_stmts_tuple[1]])

            may_be_type_temp_var = stypy_functions_copy.new_temp_Name(False, idioms_copy.may_be_var_name, node.lineno,
                                                                 node.col_offset)
            more_types_temp_var = stypy_functions_copy.new_temp_Name(False, idioms_copy.more_types_var_name, node.lineno,
                                                                node.col_offset)
            if_func_ret_type_tuple = core_language_copy.create_type_tuple(may_be_type_temp_var, more_types_temp_var)
            may_be_type_assignment = core_language_copy.create_Assign(if_func_ret_type_tuple, call_to_idiom_if_func)
            if_stmt_body.append(stypy_functions_copy.flatten_lists(may_be_type_assignment))

            # Create first idiom if
            idiom_first_if_body = []
            if_may_be = conditional_statements_copy.create_if(may_be_type_temp_var, idiom_first_if_body)
            if_stmt_body.append(if_may_be)

            # Create second idiom if
            idiom_more_types_body = []
            if_more_types = conditional_statements_copy.create_if(more_types_temp_var, idiom_more_types_body)

            # Begin the second if body
            idiom_more_types_body.append(stypy_functions_copy.create_src_comment("Runtime conditional SSA", node.lineno))
            clone_stmt1, type_store_var1 = stypy_functions_copy.create_clone_type_store()
            idiom_more_types_body.append(clone_stmt1)

            # Needed to remove the "used before assignment" warning in the generated body: Basically we store the
            # existing type store into the temp var name instead of cloning it.
            may_be_type_assignment = core_language_copy.create_Assign(type_store_var1, core_language_copy.create_Name(
                stypy_functions_copy.default_module_type_store_var_name))
            if_more_types.orelse = [may_be_type_assignment]

            idiom_first_if_body.append(if_more_types)  # Second if goes inside first one

            # Set the type of the condition var according to the identified idiom

            set_type = idioms_copy.set_type_of_idiom_var(idiom_name, "if", node.test, rigth_stmts_tuple[1], node.lineno,
                                                    node.col_offset)
            if not len(set_type) == 0:
                idiom_first_if_body.append(set_type)

            body_stmts_location = idiom_first_if_body
        else:
            # Begin the if body
            if_stmt_body.append(stypy_functions_copy.create_src_comment("SSA begins for if statement", node.lineno))
            clone_stmt1, type_store_var1 = stypy_functions_copy.create_clone_type_store()
            if_stmt_body.append(clone_stmt1)
            body_stmts_location = if_stmt_body

        # Process if body sentences
        body_stmts_location.extend(self.__visit_instruction_body(node.body, context))

        # Process else branch
        if len(node.orelse) > 0:
            if is_an_idiom:
                idiom_more_types_body_2 = []
                if_more_types_2 = conditional_statements_copy.create_if(more_types_temp_var, idiom_more_types_body_2)
                # Begin the third if body
                idiom_more_types_body_2.append(
                    stypy_functions_copy.create_src_comment("Runtime conditional SSA for else branch", node.lineno))
                clone_stmt2, type_store_var2 = stypy_functions_copy.create_clone_type_store()
                idiom_more_types_body_2.append(clone_stmt2)
                body_stmts_location.append(if_more_types_2)
                body_stmts_location = if_stmt_body  # Return to the parent sentence trunk of the function

                # Process else with idioms_copy part:
                # Create first idiom if
                idiom_first_if_body_else = []

                or_cond = ast.BoolOp()
                or_cond.op = ast.Or()
                not_cond = ast.UnaryOp()
                not_cond.op = ast.Not()
                not_cond.operand = may_be_type_temp_var
                or_cond.values = [not_cond, more_types_temp_var]

                if_may_be = conditional_statements_copy.create_if(or_cond, idiom_first_if_body_else)
                body_stmts_location.append(if_may_be)

                # Create second idiom if
                idiom_more_types_body_else = []

                and_cond = ast.BoolOp()
                and_cond.op = ast.And()
                and_cond.values = [may_be_type_temp_var, more_types_temp_var]

                if_more_types = conditional_statements_copy.create_if(and_cond, idiom_more_types_body_else)
                # Begin the second if body
                idiom_more_types_body_else.append(
                    stypy_functions_copy.create_src_comment("Runtime conditional SSA", node.lineno))
                set_type_store1 = stypy_functions_copy.create_set_type_store(type_store_var1)
                idiom_more_types_body_else.append(set_type_store1)

                idiom_first_if_body_else.append(if_more_types)  # Second if goes inside first one

                # Set the type of the condition var according to the identified idiom in the else branch
                set_type = idioms_copy.set_type_of_idiom_var(idiom_name, "else", node.test, rigth_stmts_tuple[1],
                                                        node.lineno,
                                                        node.col_offset)

                idiom_first_if_body_else.append(set_type)
                body_stmts_location = idiom_first_if_body_else
            else:
                if_stmt_body.append(
                    stypy_functions_copy.create_src_comment("SSA branch for the else part of an if statement", node.lineno))
                clone_stmt2, type_store_var2 = stypy_functions_copy.create_clone_type_store()
                if_stmt_body.append(clone_stmt2)

                set_type_store1 = stypy_functions_copy.create_set_type_store(type_store_var1)
                if_stmt_body.append(set_type_store1)
                body_stmts_location = if_stmt_body

            # Process else body sentences
            body_stmts_location.extend(self.__visit_instruction_body(node.orelse, context))

        else:
            clone_stmt2, type_store_var2 = stypy_functions_copy.create_clone_type_store()
            body_stmts_location.append(clone_stmt2)
            # No else means that the after-if type store is the one that is currently on use and the
            # pre-if type store is the one cloned before. Else type store is None
            # type_store_var2 = core_language_copy.create_Name(stypy_functions_copy.default_module_type_store_var_name)
            type_store_var3 = core_language_copy.create_Name('None')

        if is_an_idiom:
            if len(node.orelse) == 0:
                idiom_more_types_body_final = []
                if_more_types_final = conditional_statements_copy.create_if(more_types_temp_var, idiom_more_types_body_final)
                body_stmts_location.append(if_more_types_final)
                final_stmts = idiom_more_types_body_final
            else:
                idiom_more_types_body_final = []

                and_cond = ast.BoolOp()
                and_cond.op = ast.And()
                and_cond.values = [may_be_type_temp_var, more_types_temp_var]

                if_more_types_final = conditional_statements_copy.create_if(and_cond, idiom_more_types_body_final)
                body_stmts_location.append(if_more_types_final)
                clone_stmt3, type_store_var3 = stypy_functions_copy.create_clone_type_store()
                idiom_more_types_body_final.append(clone_stmt3)

                final_stmts = idiom_more_types_body_final
        else:
            if len(node.orelse) > 0:
                clone_stmt3, type_store_var3 = stypy_functions_copy.create_clone_type_store()
                if_stmt_body.append(clone_stmt3)
            final_stmts = if_stmt_body


        # Join if
        final_stmts.append(stypy_functions_copy.create_src_comment("SSA join for if statement", node.lineno))
        join_stmt, final_type_store = stypy_functions_copy.create_join_type_store("ssa_join_with_else_branch",
                                                                             [type_store_var1, type_store_var2,
                                                                              type_store_var3])
        final_stmts.append(join_stmt)

        # Calculate the final type store
        set_type_store2 = stypy_functions_copy.create_set_type_store(final_type_store)
        final_stmts.append(set_type_store2)

        # Unify all if body statements
        all_if_stmts = if_stmt_body

        # Unify all if sentences
        if_stmt = stypy_functions_copy.flatten_lists(condition_comment, test_condition_call, all_if_stmts)

        end_if_comment = stypy_functions_copy.create_blank_line()

        return stypy_functions_copy.flatten_lists(begin_if_comment,
                                             condition_stmt,
                                             if_stmt,
                                             end_if_comment)

    # ################################################ FOR #####################################################

    def visit_For(self, node, context):
        # print inspect.stack()[0][3]

        for_stmt_body = []
        for_stmt_orelse = []

        begin_for_comment = stypy_functions_copy.create_blank_line()

        # Process for test
        iter_stmt, for_stmt_test = self.visit(node.iter, context)

        # Check if the for statement is suitable for iteration
        condition_comment = stypy_functions_copy.create_src_comment("Testing the type of a for loop iterable", node.lineno)
        loop_test_fname = core_language_copy.create_Name("is_suitable_for_loop_condition")
        localization = stypy_functions_copy.create_localization(node.lineno, node.col_offset)
        condition_call = functions_copy.create_call_expression(loop_test_fname, [localization, for_stmt_test])

        # Get the type of the loop iteration variable and assign it
        get_target_comment = stypy_functions_copy.create_src_comment("Getting the type of the for loop variable",
                                                                node.lineno)
        for_stmt_body.append(get_target_comment)
        loop_target_fname = core_language_copy.create_Name("get_type_of_for_loop_variable")
        target_assign_call = functions_copy.create_call(loop_target_fname, [localization, for_stmt_test])
        target_assign, target_assign_var = stypy_functions_copy.create_temp_Assign(target_assign_call, node.lineno,
                                                                              node.col_offset)
        for_stmt_body.append(target_assign)

        if isinstance(node.target, ast.Tuple):
            get_elements_call = core_language_copy.create_attribute(target_assign_var.id, "get_elements_type")
            assign_target_type = []
            for elt in node.target.elts:
                call_to_elements = functions_copy.create_call(get_elements_call, [])
                type_set = stypy_functions_copy.create_set_type_of(elt.id, call_to_elements, node.lineno,
                                                              node.col_offset)
                assign_target_type.append(type_set)
        else:
            assign_target_type = stypy_functions_copy.create_set_type_of(node.target.id, target_assign_var, node.lineno,
                                                                    node.col_offset)

        for_stmt_body.append(assign_target_type)

        # For body
        for_stmt_body.append(stypy_functions_copy.create_src_comment("SSA begins for a for statement", node.lineno))
        clone_stmt1, type_store_var1 = stypy_functions_copy.create_clone_type_store()
        for_stmt_body.append(clone_stmt1)

        # Process for body statements
        for_stmt_body.extend(self.__visit_instruction_body(node.body, context))

        for_stmt_body.append(
            stypy_functions_copy.create_src_comment("SSA branch for the else part of a for statement", node.lineno))
        clone_stmt2, type_store_var2 = stypy_functions_copy.create_clone_type_store()
        for_stmt_body.append(clone_stmt2)

        # Else part of a for statement
        set_type_store1 = stypy_functions_copy.create_set_type_store(type_store_var1)
        for_stmt_body.append(set_type_store1)

        for_stmt_orelse.extend(self.__visit_instruction_body(node.orelse, context))

        # Join and finish for
        for_stmt_orelse.append(stypy_functions_copy.create_src_comment("SSA join for a for statement"))
        clone_stmt3, type_store_var3 = stypy_functions_copy.create_clone_type_store()
        for_stmt_orelse.append(clone_stmt3)

        join_stmt, final_type_store = stypy_functions_copy.create_join_type_store("ssa_join_with_else_branch",
                                                                             [type_store_var1, type_store_var2,
                                                                              type_store_var3])
        for_stmt_orelse.append(join_stmt)

        # Assign final type store
        set_type_store2 = stypy_functions_copy.create_set_type_store(final_type_store)
        for_stmt_orelse.append(set_type_store2)

        for_stmts = for_stmt_body + for_stmt_orelse
        for_stmt = stypy_functions_copy.flatten_lists(condition_comment, condition_call, for_stmts)

        end_for_comment = stypy_functions_copy.create_blank_line()

        return stypy_functions_copy.flatten_lists(begin_for_comment,
                                             iter_stmt,
                                             for_stmt,
                                             end_for_comment)

    # ################################################ WHILE #####################################################

    def visit_While(self, node, context):
        # print inspect.stack()[0][3]

        while_stmt_body = []
        while_stmt_orelse = []

        begin_while_comment = stypy_functions_copy.create_blank_line()

        # Process the condition of the while statement
        condition_stmt, while_stmt_test = self.visit(node.test, context)

        # Test the type of the while condition
        condition_comment = stypy_functions_copy.create_src_comment("Testing the type of an if condition", node.lineno)
        attribute = core_language_copy.create_Name("is_suitable_condition")
        localization = stypy_functions_copy.create_localization(node.lineno, node.col_offset)
        condition_call = functions_copy.create_call_expression(attribute, [localization, while_stmt_test])

        # Process the body of the while statement
        while_stmt_body.append(stypy_functions_copy.create_src_comment("SSA begins for while statement", node.lineno))
        clone_stmt1, type_store_var1 = stypy_functions_copy.create_clone_type_store()
        while_stmt_body.append(clone_stmt1)

        # While body
        while_stmt_body.extend(self.__visit_instruction_body(node.body, context))

        # Else part of the while statements
        while_stmt_body.append(
            stypy_functions_copy.create_src_comment("SSA branch for the else part of a while statement", node.lineno))
        clone_stmt2, type_store_var2 = stypy_functions_copy.create_clone_type_store()
        while_stmt_body.append(clone_stmt2)

        set_type_store1 = stypy_functions_copy.create_set_type_store(type_store_var1)
        while_stmt_body.append(set_type_store1)

        # While else part
        while_stmt_orelse.extend(self.__visit_instruction_body(node.orelse, context))

        # Join type stores and finish while
        while_stmt_orelse.append(stypy_functions_copy.create_src_comment("SSA join for while statement", node.lineno))
        clone_stmt3, type_store_var3 = stypy_functions_copy.create_clone_type_store()
        while_stmt_orelse.append(clone_stmt3)

        join_stmt, final_type_store = stypy_functions_copy.create_join_type_store("ssa_join_with_else_branch",
                                                                             [type_store_var1, type_store_var2,
                                                                              type_store_var3])
        while_stmt_orelse.append(join_stmt)

        # Final type store
        set_type_store2 = stypy_functions_copy.create_set_type_store(final_type_store)
        while_stmt_orelse.append(set_type_store2)

        all_while_stmts = while_stmt_body + while_stmt_orelse
        while_stmt = stypy_functions_copy.flatten_lists(condition_comment, condition_call, all_while_stmts)

        end_while_comment = stypy_functions_copy.create_blank_line()
        return stypy_functions_copy.flatten_lists(begin_while_comment,
                                             condition_stmt,
                                             while_stmt,
                                             end_while_comment)

    # ################################################ EXCEPTIONS #####################################################

    def visit_TryExcept(self, node, context):
        # print inspect.stack()[0][3]

        try_except_stmts = []
        handler_type_stores = []
        begin_except_comment = stypy_functions_copy.create_blank_line()

        # Begin the exception body
        try_except_stmts.append(stypy_functions_copy.create_src_comment("SSA begins for try-except statement", node.lineno))
        clone_stmt1, pre_try_type_store = stypy_functions_copy.create_clone_type_store()
        try_except_stmts.append(clone_stmt1)

        # Process exception body sentences
        try_except_stmts.extend(self.__visit_instruction_body(node.body, context))

        try_except_stmts.append(
            stypy_functions_copy.create_src_comment("SSA branch for the except part of a try statement", node.lineno))
        clone_stmt2, post_try_type_store = stypy_functions_copy.create_clone_type_store()
        try_except_stmts.append(clone_stmt2)

        # Process all except handlers
        for handler in node.handlers:
            if handler.type is None:
                except_handler = "<any exception>"
            else:
                except_handler = handler.type.id

            try_except_stmts.append(stypy_functions_copy.create_src_comment(
                "SSA branch for the except '{0}' branch of a try statement".format(except_handler), node.lineno))

            set_type_store_handler = stypy_functions_copy.create_set_type_store(pre_try_type_store)
            try_except_stmts.append(set_type_store_handler)

            if not handler.type is None and not handler.name is None:
                try_except_stmts.append(stypy_functions_copy.create_src_comment("Storing handler type"))
                handler_type_stmts, handle_type_var = self.visit(handler.type, context)
                handler_name_assign = stypy_functions_copy.create_set_type_of(handler.name.id, handle_type_var,
                                                                         handler.lineno,
                                                                         handler.col_offset)
                try_except_stmts.append(handler_type_stmts)
                try_except_stmts.append(handler_name_assign)

            # Process except handler body sentences
            try_except_stmts.extend(self.__visit_instruction_body(handler.body, context))

            clone_stmt_handler, type_store_except_handler = stypy_functions_copy.create_clone_type_store()
            try_except_stmts.append(clone_stmt_handler)
            handler_type_stores.append(type_store_except_handler)

        # Else branch
        try_except_stmts.append(stypy_functions_copy.create_src_comment(
            "SSA branch for the else branch of a try statement", node.lineno))
        set_type_store_handler = stypy_functions_copy.create_set_type_store(pre_try_type_store)
        try_except_stmts.append(set_type_store_handler)

        # Process except handler body sentences
        if len(node.orelse) > 0:
            try_except_stmts.extend(self.__visit_instruction_body(node.orelse, context))

            clone_stmt_handler, type_store_else_handler = stypy_functions_copy.create_clone_type_store()
            try_except_stmts.append(clone_stmt_handler)
            handler_type_stores.append(type_store_else_handler)

        # Join try
        try_except_stmts.append(stypy_functions_copy.create_src_comment("SSA join for try-except statement", node.lineno))
        join_stmt, final_type_store = stypy_functions_copy.create_join_type_store("join_exception_block",
                                                                             [pre_try_type_store, post_try_type_store,
                                                                              core_language_copy.create_Name(
                                                                                  "None")] + handler_type_stores)
        try_except_stmts.append(join_stmt)

        # Calculate the final type store
        set_type_store_final = stypy_functions_copy.create_set_type_store(final_type_store)
        try_except_stmts.append(set_type_store_final)

        # Unify all if sentences
        all_try_except_stmt = stypy_functions_copy.flatten_lists(begin_except_comment, try_except_stmts)

        end_except_comment = stypy_functions_copy.create_blank_line()

        return stypy_functions_copy.flatten_lists(begin_except_comment,
                                             all_try_except_stmt,
                                             end_except_comment)

    def visit_TryFinally(self, node, context):
        # print inspect.stack()[0][3]
        try_finally_stmts = []

        initial_comment = stypy_functions_copy.create_src_comment("Try-finally block", node.lineno)
        try_finally_stmts.append(initial_comment)

        # Process exception body sentences
        try_finally_stmts.extend(self.__visit_instruction_body(node.body, context))

        try_finally_stmts.append(stypy_functions_copy.create_blank_line())
        finally_comment = stypy_functions_copy.create_src_comment("finally branch of the try-finally block",
                                                             node.lineno)
        try_finally_stmts.append(finally_comment)

        # Process exception body sentences
        try_finally_stmts.extend(self.__visit_instruction_body(node.finalbody, context))

        return stypy_functions_copy.flatten_lists(stypy_functions_copy.create_blank_line(),
                                             try_finally_stmts,
                                             stypy_functions_copy.create_blank_line())

    # ################################################ IMPORTS ##########################################

    def visit_Import(self, node, context):
        # print inspect.stack()[0][3]
        localization = stypy_functions_copy.create_localization(node.lineno, node.col_offset)
        call_stmts = []

        import_func = core_language_copy.create_Name("import_elements_from_external_module")

        dir_name = os.path.dirname(self.file_name)
        alias_stmt = []
        for alias in node.names:
            call_stmts.append(
                stypy_functions_copy.create_src_comment("Importing '{0}' module".format(alias.name), node.lineno))
            stypy_functions_copy.assign_line_and_column(alias, node)
            alias_stmt, alias_var = self.visit(alias, context)
            call = functions_copy.create_call_expression(import_func, [localization,
                                                                 # core_language_copy.create_str(dir_name),
                                                                  alias_var,  # core_language_copy.create_str(alias.name),
                                                                  core_language_copy.create_Name("type_store")])
            call_stmts.append(call)

        return stypy_functions_copy.flatten_lists(call_stmts, alias_stmt,
                                             stypy_functions_copy.create_blank_line())

    def visit_ImportFrom(self, node, context):
        # print inspect.stack()[0][3]
        localization = stypy_functions_copy.create_localization(node.lineno, node.col_offset)
        call_stmts = []

        call_stmts.append(
            stypy_functions_copy.create_src_comment("Importing from '{0}' module".format(node.module), node.lineno))
        import_func = core_language_copy.create_Name("import_elements_from_external_module")

        dir_name = os.path.dirname(self.file_name)
        elements = []
        alias_stmt = []
        for alias in node.names:
            stypy_functions_copy.assign_line_and_column(alias, node)
            alias_stmt, alias_var = self.visit(alias, context)
            elements.append(alias_var)  # core_language_copy.create_str(alias.name))

        starargs = data_structures_copy.create_list(elements)
        call = functions_copy.create_call_expression(import_func, [localization,
                                                              #core_language_copy.create_str(dir_name),
                                                              core_language_copy.create_str(node.module),
                                                              core_language_copy.create_Name("type_store")],
                                                starargs=starargs)
        call_stmts.append(call)

        return stypy_functions_copy.flatten_lists(call_stmts, alias_stmt,
                                             stypy_functions_copy.create_blank_line())

    # #################################################### PRINT ######################################################

    def visit_Print(self, node, context):
        # print inspect.stack()[0][3]

        print_stmts = self.__visit_instruction_body(node.values, context)

        return stypy_functions_copy.flatten_lists(print_stmts)

    # #################################################### GLOBAL #####################################################

    def visit_Global(self, node, context):
        # print inspect.stack()[0][3]
        global_stmts = [stypy_functions_copy.create_src_comment("Marking variables as global", node.lineno)]
        mark_function = core_language_copy.create_attribute('type_store', 'mark_as_global', node.lineno, node.col_offset)
        localization = stypy_functions_copy.create_localization(node.lineno, node.col_offset)
        for name in node.names:
            mark_call = functions_copy.create_call_expression(mark_function, [localization, core_language_copy.create_str(name)])
            global_stmts.append(mark_call)

        return stypy_functions_copy.flatten_lists(global_stmts)

    # ######################################### BREAK AND CONTINUE ####################################################

    def visit_Break(self, node, context):
        # print inspect.stack()[0][3]
        return []

    def visit_Continue(self, node, context):
        # print inspect.stack()[0][3]
        return []

    # ######################################### DELETE #######################################################

    def visit_Delete(self, node, context):
        # print inspect.stack()[0][3]
        stmts = []

        comment = stypy_functions_copy.create_src_comment("Deleting a member")
        for target in node.targets:
            # Is an attribute of a class or module
            if type(target) is ast.Attribute:
                parent = target.value.id
                attr = core_language_copy.create_str(target.attr)
            else:
                # Is a name (__builtins__)
                parent = '__builtins__'
                attr = target

            parent_stmts, parent_var = stypy_functions_copy.create_get_type_of(parent, node.lineno, node.col_offset)

            delete_func = core_language_copy.create_attribute(parent_var, 'delete_member',
                                                         line=node.lineno,
                                                         column=node.col_offset)

            localization = stypy_functions_copy.create_localization(node.lineno, node.col_offset)
            delete_member_call = functions_copy.create_call_expression(delete_func,
                                                                  [localization, attr])

            stmts += stypy_functions_copy.flatten_lists(comment, parent_stmts, delete_member_call)

        return stmts

    # ######################################### ASSERT #######################################################

    def visit_Assert(self, node, context):
        comment = stypy_functions_copy.create_src_comment("Evaluating assert statement condition")
        stmts, var = self.visit(node.test, context)
        return stypy_functions_copy.flatten_lists(comment, stmts)

    def visit_Exec(self, node, context):
        src_comment = stypy_functions_copy.create_src_comment("Dynamic code evaluation using an exec statement")
        localization = stypy_functions_copy.create_localization(node.lineno, node.col_offset)
        type_check_func = core_language_copy.create_Name('ensure_var_of_types',
                                                    line=node.lineno,
                                                    column=node.col_offset)

        body_stmts, body_var = self.visit(node.body, context)

        description = core_language_copy.create_str("exec parameter")
        string_type = core_language_copy.create_str("StringType")
        file_type = core_language_copy.create_str("FileType")
        code_type = core_language_copy.create_str("CodeType")
        check_params_call = functions_copy.create_call_expression(type_check_func,
                                                             [localization, body_var, description,
                                                              string_type,
                                                              file_type,
                                                              code_type])

        unsupported_feature_call = stypy_functions_copy.create_unsupported_feature_call(localization, "exec",
                                                                                   "Dynamic code evaluation using exec is not yet supported by stypy",
                                                                                   node.lineno, node.col_offset)

        return stypy_functions_copy.flatten_lists(src_comment, body_stmts, check_params_call, unsupported_feature_call)

    def visit_Yield(self, node, context):
        src_comment = stypy_functions_copy.create_src_comment("Creating a generator")
        localization = stypy_functions_copy.create_localization(node.lineno, node.col_offset)
        generator_stored_type_stmts, generator_stored_type_var = self.visit(node.value, context)

        types_module = core_language_copy.create_str("types")
        generator_type = core_language_copy.create_str("GeneratorType")
        get_type_func = core_language_copy.create_Name("get_python_api_type")
        get_generator_call = functions_copy.create_call(get_type_func,
                                                   [localization, types_module, generator_type])

        generator_stmts, generator_type_var = stypy_functions_copy.create_temp_Assign(get_generator_call, node.lineno,
                                                                                 node.col_offset)

        # Assign to the generator its stored type
        store_type_func = core_language_copy.create_attribute(generator_type_var, 'set_elements_type',
                                                         line=node.lineno,
                                                         column=node.col_offset)

        store_type_call = functions_copy.create_call_expression(store_type_func, [localization, generator_stored_type_var])

        ret_assign = stypy_functions_copy.assign_as_return_type(generator_type_var)

        return stypy_functions_copy.flatten_lists(src_comment, generator_stored_type_stmts, generator_stmts, store_type_call,
                                             ret_assign)

    def visit_Raise(self, node, context):
        localization = stypy_functions_copy.create_localization(node.lineno, node.col_offset)

        type_stmts, type_var = self.visit(node.type, context)

        type_check_func = core_language_copy.create_Name('ensure_var_of_types',
                                                    line=node.lineno,
                                                    column=node.col_offset)

        description = core_language_copy.create_str("raise parameter")
        admitted_type = core_language_copy.create_Name('Exception')

        check_params_call = functions_copy.create_call_expression(type_check_func,
                                                             [localization, type_var, description,
                                                              admitted_type])

        return stypy_functions_copy.flatten_lists(type_stmts, check_params_call)

    def visit_With(self, node, context):
        localization = stypy_functions_copy.create_localization(node.lineno, node.col_offset)
        type_check_func = core_language_copy.create_Name('ensure_var_has_members',
                                                    line=node.lineno,
                                                    column=node.col_offset)

        context_stmts, context_var = self.visit(node.context_expr, context)

        description = core_language_copy.create_str("with parameter")

        enter = core_language_copy.create_str("__enter__")
        exit = core_language_copy.create_str("__exit__")
        check_params_call = functions_copy.create_call(type_check_func,
                                                  [localization, context_var, description,
                                                   enter,
                                                   exit,
                                                   ])

        call_test_stmts, call_test_var = stypy_functions_copy.create_temp_Assign(check_params_call, line=node.lineno,
                                                                            column=node.col_offset)
        # Call __enter__
        enter_comment = stypy_functions_copy.create_src_comment("Calling the __enter__ method to initiate a with section")
        enter_method, enter_var = stypy_functions_copy.create_get_type_of_member(context_var, '__enter__', node.lineno,
                                                                            node.col_offset)
        enter_method_invoke = core_language_copy.create_attribute(enter_var, 'invoke', node.lineno, node.col_offset)
        enter_method_call, call_var = stypy_functions_copy.create_temp_Assign(
            functions_copy.create_call(enter_method_invoke, [localization]), node.lineno, node.col_offset)
        body_stmts = [enter_comment, enter_method, enter_method_call]
        if node.optional_vars is not None:
            assing_to_var = stypy_functions_copy.create_set_type_of(node.optional_vars.id, call_var, node.lineno,
                                                               node.col_offset)
            body_stmts += [assing_to_var]

        body_stmts += self.__visit_instruction_body(node.body, context)

        # Call __exit__
        exit_comment = stypy_functions_copy.create_src_comment("Calling the __exit__ method to finish a with section")
        exit_method, exit_var = stypy_functions_copy.create_get_type_of_member(context_var, '__exit__', node.lineno,
                                                                          node.col_offset)
        exit_method_invoke = core_language_copy.create_attribute(exit_var, 'invoke', node.lineno, node.col_offset)
        none_type = core_language_copy.create_Name('None')
        exit_method_call, call_var = stypy_functions_copy.create_temp_Assign(functions_copy.create_call(exit_method_invoke,
                                                                                              [localization,
                                                                                               none_type,
                                                                                               none_type,
                                                                                               none_type]),
                                                                        node.lineno, node.col_offset)

        body_stmts += [exit_comment, exit_method, exit_method_call]

        with_if = conditional_statements_copy.create_if(call_test_var, body_stmts)

        return stypy_functions_copy.flatten_lists(context_stmts, call_test_stmts, with_if)

    # Integrated into visit_FunctionDef
    def visit_arguments(self, node, context):
        return node
