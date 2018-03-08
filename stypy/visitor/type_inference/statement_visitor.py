#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy

import value_visitor
from stypy.module_imports.python_library_modules import is_python_library_module
from stypy.type_inference_programs.aux_functions import *
from stypy.visitor.type_inference.visitor_utils import core_language, functions, stypy_functions, data_structures, \
    conditional_statements, idioms


class StatementVisitor(ast.NodeVisitor):
    prepend_statements = []

    """
    Visitor for statement nodes, that are AST nodes that return only statement lists. If when processing a statement
    node one of its children nodes is a value node, a ValueVisitor is automatically run to process this child node.
    """

    def __init__(self, filename):
        self.file_name = filename

    def visit(self, node, context):
        """
        Visits a node. This method redirects to the concrete visitor method depending on the type of the node
        :param node: Node to be visited
        :param context: Context of the visit
        :return:
        """
        method = 'visit_' + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        if visitor == self.generic_visit:
            return visitor(node, context + [method])
        return visitor(node, context)

    def generic_visit(self, node, context):
        """
        Default implementation of the visit if none of the visit methods of this class fits with the node type. It
        should call the ValueVisitor class to visit those nodes that are not in this visitor. If not, it is considered
        an error, as there are no suitable visitor to the node. Currently, this should not happen, as all posible
        visitors should be implemented.
        :param node:
        :param context:
        :return:
        """
        if hasattr(value_visitor.ValueVisitor, context[-1]):
            return self.visit_value(node, context[:-1])
        else:
            raise Exception("STATEMENT VISITOR: " + context[-1] + " is not yet implemented")

    def __visit_instruction_body(self, body, context):
        """
        Visit all the instructions of an instruction list, calling the appropriate visitor methods
        :param body:
        :param context:
        :return:
        """
        new_stmts = []

        # Visit all body instructions
        v_visitor = value_visitor.ValueVisitor(self.file_name)

        for stmt in body:
            stmts, temp = v_visitor.visit(stmt, context)
            if isinstance(stmts, list):
                new_stmts.extend(stmts)
            else:
                new_stmts.append(stmts)

        return new_stmts

    def visit_value(self, node, context):
        """
        Call the value visitor
        :param node:
        :param context:
        :return:
        """
        v_visitor = value_visitor.ValueVisitor(self.file_name)

        result, temp = v_visitor.visit(node, context)

        return result, temp

    # ######################################### MAIN MODULE #############################################

    def visit_Module(self, node, context):
        """
        Visit a Module node
        :param node:
        :param context:
        :return:
        """

        context.append(node)
        stmts = self.__visit_instruction_body(node.body, context)
        context.remove(node)

        return stmts

    # ######################################### FUNCTIONS #############################################

    def visit_FunctionDef(self, node, context):
        """
        Visit a FunctionDef node
        :param node:
        :param context:
        :return:
        """
        self.building_function_name = node.name

        # Function declaration localization
        decl_localization = core_language.create_Name('localization', False, line=node.lineno, column=node.col_offset)

        # Decorators are declarative properties of methods or functions that give problems if directly ported to
        # type-inference equivalent code. For now, we only translate the @staticmethod decorator.
        decorator_list = []
        if len(node.decorator_list) > 0:
            for dec in node.decorator_list:
                if isinstance(dec, ast.Name):
                    if dec.id == "staticmethod":
                        decorator_list.append(dec)

        # The 'norecursion' decorator, mandatory in every stypy code generation to enable the type inference program
        # to not to hang on recursive calls.
        decorator_list.append(core_language.create_Name('norecursion', line=node.lineno - 1, column=node.col_offset))

        defaults_types = []
        defaults_stmts = []

        for elem in node.args.defaults:
            stmts, type_ = self.visit(elem, context)
            defaults_types.append(type_)
            defaults_stmts.append(stmts)

        # Create and add the function definition header
        function_def = functions.create_function_def(node.name,
                                                     decl_localization,
                                                     decorator_list,
                                                     context,
                                                     line=node.lineno,
                                                     column=node.col_offset)

        # Defaults are stored in a variable at the beginning of the function
        function_def.body.append(stypy_functions.create_src_comment("Assign values to the parameters with defaults"))
        function_def.body.append(stypy_functions.create_set_unreferenced_var_check(False))
        function_def.body.append(defaults_stmts)
        function_def.body.append(stypy_functions.create_set_unreferenced_var_check(True))
        function_def.body.append(core_language.create_Assign(core_language.create_Name("defaults"),
                                                             data_structures.create_list(defaults_types)))

        # Generate code from setting a new context in the type store
        function_def.body.append(stypy_functions.create_src_comment
                                 ("Create a new context for function '{0}'".format(node.name)))
        context_set = functions.create_context_set(node.name, node.lineno,
                                                   node.col_offset, False)
        function_def.body.append(context_set)

        if functions.is_method(context, decorator_list):
            function_def.body.append(
                stypy_functions.create_set_type_of('self', core_language.create_Name('type_of_self'),
                                                   node.lineno + 1,
                                                   node.col_offset))

        # Generate code for arity checking of the function arguments and assigning them to suitable local variables
        function_def.body.append(stypy_functions.create_blank_line())
        function_def.body.append(stypy_functions.create_src_comment("Passed parameters checking function"))

        # Generate argument number test
        f_preamble = functions.create_arg_number_test(node, decorator_list, context)

        function_def.body.append(f_preamble)

        # Generate code for create a new stack push for error reporting and initialize other data.
        function_def.body.append(stypy_functions.create_src_comment("Initialize method data"))
        declared_arguments = functions.obtain_arg_list(node.args, functions.is_method(context))

        init_f_name = core_language.create_Name("init_call_information")
        arguments_var = core_language.create_Name("arguments")
        mts_var = core_language.create_Name("module_type_store")
        loc_var = core_language.create_Name("localization")
        init_f_call = functions.create_call(init_f_name, [mts_var, core_language.create_str(node.name),
                                                          loc_var, declared_arguments, arguments_var])
        init_f_call_expr = ast.Expr()
        init_f_call_expr.value = init_f_call
        function_def.body.append(init_f_call_expr)

        # Initialize the variable where the return of the function will be stored.
        # This is needed due to a single return statement must exist within a function in order to not to conflict with
        # the SSA algorithm
        function_def.body.append(stypy_functions.create_blank_line())
        function_def.body.append(stypy_functions.create_src_comment("Default return type storage variable (SSA)"))
        function_def.body.append(stypy_functions.create_default_return_variable())

        function_def.body.append(stypy_functions.create_blank_line())
        function_def.body.append(
            stypy_functions.create_program_section_src_comment("Begin of '{0}(...)' code".format(node.name)))

        context.append(node)
        # Visit the function body
        function_def.body.extend(self.__visit_instruction_body(node.body, context))
        context.remove(node)

        function_def.body.append(
            stypy_functions.create_program_section_src_comment("End of '{0}(...)' code".format(node.name)))

        # Pop the error reporting stack trace
        init_f_name = core_language.create_Name("teardown_call_information")
        arguments_var = core_language.create_Name("arguments")
        loc_var = core_language.create_Name("localization")
        teardown_f_call = functions.create_call_expression(init_f_name, [loc_var, arguments_var])

        function_def.body.append(stypy_functions.create_src_comment("Teardown call information"))
        function_def.body.append(teardown_f_call)

        # Finally, return the return value (contained in a predefined var name)
        # in the single return statement of each function.
        if not functions.is_constructor(node):
            function_def.body.append(
                stypy_functions.create_store_return_from_function(node.name, node.lineno, node.col_offset))
        else:
            function_def.body.append(functions.create_context_unset())

        if not functions.is_method(context):
            # Register the function type outside the function context
            register_expr = stypy_functions.create_set_type_of(node.name, core_language.create_Name(
                functions.get_function_name_in_ti_files(node.name)),
                                                               node.lineno, node.col_offset)
        else:
            register_expr = []

        return stypy_functions.flatten_lists(function_def, register_expr)

    def visit_Return(self, node, context):
        """
        Visit a Return node
        :param node:
        :param context:
        :return:
        """
        default_function_ret_var = core_language.create_Name(stypy_functions.default_function_ret_var_name)
        if node.value is not None:
            get_value_stmts, value_var = self.visit_value(node.value, context)
        else:
            get_value_stmts, value_var = [], core_language.create_NoneType()

        ret_store = stypy_functions.create_set_type_of(default_function_ret_var.id, value_var, node.lineno,
                                                       node.col_offset)

        return stypy_functions.flatten_lists(get_value_stmts, ret_store)  # ret_assign, ret_store)

    # ################################################ CLASSES ##########################################

    def visit_ClassDef(self, node, context):
        """
        Visit a ClassDef node
        :param node:
        :param context:
        :return:
        """
        # Store the name of the class that is currently in the building process
        self.building_class_name = node.name

        # Writes the instruction: from stypy import *
        initial_comment = stypy_functions.create_src_comment("Declaration of the '{0}' class".format(node.name))

        # Obtaining the type of the superclasses of this class (if they exist)
        superclass_inst = []
        new_bases = []
        if len(node.bases) == 1 and type(node.bases[0]) is ast.Name and node.bases[0].id == 'object':
            # Plain new-style class
            new_bases.append(core_language.create_Name('object'))
        else:
            for class_name in node.bases:
                superclass_, type_var = self.visit(class_name,
                                                   context)
                superclass_inst.append(superclass_)
                new_bases.append(type_var)

        node.bases = new_bases  # []

        # Remove class decorators (it may interfere in type inference generated code)
        decorator_instrs = []
        decorator_vars = []

        if len(node.decorator_list) > 0:
            for name in node.decorator_list:
                if type(name) is ast.Name:
                    decorator_class_obj_inst, decorator_class_var = stypy_functions.create_get_type_of(name.id,
                                                                                                       node.lineno,
                                                                                                       node.col_offset)
                    decorator_instrs += decorator_class_obj_inst
                    decorator_vars.append(decorator_class_var)
                else:
                    # TODO: Decorators are not supported in this stypy version (just @staticmethod)
                    print (
                        "StatementVisitor: visit_ClassDef -> Skipped the processing of a decorator of type " + str(
                            name))

            node.decorator_list = []

        context.append(node)
        # Visit all body instructions
        node.body = self.__visit_instruction_body(node.body, context)

        context.remove(node)

        # Register the class type outside the class context
        if len(decorator_vars) > 0:
            # Register the original class:
            register_original_class_instr = stypy_functions.create_set_type_of(node.name,
                                                                               core_language.create_Name(node.name),
                                                                               node.lineno, node.col_offset)

            # Wrap the original class in all decorators
            final_type_var = None
            decorator_application_instr = []
            for var in decorator_vars:
                get_original_class_instr, original_class_var = stypy_functions.create_get_type_of(node.name,
                                                                                                  node.lineno,
                                                                                                  node.col_offset)

                var_invokation = core_language.create_Name('invoke')

                final_type_var_inst, final_type_var = stypy_functions.create_temp_Assign(
                    functions.create_call(var_invokation, [stypy_functions.create_localization(node.lineno,
                                                                                               node.col_offset),
                                                           var, original_class_var]), node.lineno,
                    node.col_offset, descriptive_var_name='class')
                decorator_application_instr = stypy_functions.flatten_lists(get_original_class_instr,
                                                                            decorator_application_instr,
                                                                            final_type_var_inst)

            register_expr = stypy_functions.flatten_lists(stypy_functions.create_set_type_of(node.name, final_type_var,
                                                                                             node.lineno,
                                                                                             node.col_offset))
            register_expr = stypy_functions.flatten_lists(register_original_class_instr, decorator_application_instr,
                                                          register_expr)
        else:
            register_expr = stypy_functions.create_set_type_of(node.name, core_language.create_Name(node.name),
                                                               node.lineno, node.col_offset)

        return stypy_functions.flatten_lists(initial_comment, decorator_instrs, superclass_inst, node,
                                             stypy_functions.create_blank_line(), register_expr)

    # ################################################ ASSIGNMENTS ###################################################

    def visit_Assign(self, node, context):
        """
        Visit an Assign node
        :param node:
        :param context:
        :return:
        """

        type_store_invokation = []
        get_value_stmts, temp_value = self.visit_value(node.value, context)

        if type(temp_value) is ast.Tuple:
            temp_value = temp_value.elts[0]

        if isinstance(node.targets[0], ast.Name):
            name = node.targets[0]
            if name.id == '__all__':  # Assigning public members to the type store if __all__ is present
                if isinstance(node.value, ast.List):
                    public_members = map(lambda elem: core_language.create_str(elem.s), node.value.elts)
                    type_store_name = core_language.create_Name(stypy_functions.default_module_type_store_var_name)
                    type_store_call = core_language.create_attribute(type_store_name, 'set_exportable_members')
                    type_store_invokation = [functions.create_call_expression(type_store_call, [
                        data_structures.create_list(public_members)])]
                    type_store_invokation = [node] + [type_store_invokation]

            set_type_of_stmts = stypy_functions.create_set_type_of(name.id, temp_value,
                                                                   name.lineno,
                                                                   name.col_offset)
        else:
            get_target_stmts, target_var = self.visit(node.targets[0].value, context)
            if type(target_var) is ast.Tuple:
                target_var = target_var.elts[0]

            if isinstance(node.targets[0], ast.Subscript):
                # temp_value: type to add
                # target_var: container
                # node.slice: subscript to calculate
                node.targets[0].slice.lineno = node.targets[0].lineno
                node.targets[0].slice.col_offset = node.targets[0].col_offset
                slice_stmts, slice_var = self.visit(node.targets[0].slice, context)
                set_type_of_stmts = stypy_functions.create_add_stored_type(target_var, slice_var, temp_value,
                                                                           node.targets[0].lineno,
                                                                           node.targets[0].col_offset)
                set_type_of_stmts = stypy_functions.flatten_lists(get_target_stmts, slice_stmts, set_type_of_stmts)
            else:
                set_type_of_stmts = stypy_functions.create_set_type_of_member(target_var, node.targets[0].attr,
                                                                              temp_value,
                                                                              node.targets[0].lineno,
                                                                              node.targets[0].col_offset)

                set_type_of_stmts = stypy_functions.flatten_lists(get_target_stmts, set_type_of_stmts)

        instructions = stypy_functions.flatten_lists(type_store_invokation, get_value_stmts)
        return instructions + set_type_of_stmts

    # ################################################ PASS NODES #####################################################

    def visit_Pass(self, node, context):
        """
        Visit a Pass node
        :param node:
        :param context:
        :return:
        """

        return node

    # ################################################ IF #####################################################

    def visit_If(self, node, context):
        """
        Visit an If node
        :param node:
        :param context:
        :return:
        """

        if_stmt_body = []
        ev_none_test_condition = None
        else_stmts = []
        ev_none_condition_comment = []

        begin_if_comment = stypy_functions.create_blank_line()

        is_an_idiom, left_stmts_tuple, rigth_stmts_tuple, idiom_name = idioms.is_recognized_idiom(node.test, self,
                                                                                                  context)

        # Process if __name__ == '__main__' statements to enable its calling
        if core_language.is_main(node):
            node.body = self.__visit_instruction_body(node.body, context)
            return node

        if is_an_idiom:
            condition_comment = stypy_functions.create_src_comment(
                "Type idiom detected: calculating its left and rigth part", node.lineno)
            # Statements for the left part of the idiom
            if not isinstance(left_stmts_tuple[0], list):
                left = [left_stmts_tuple[0]]
            else:
                left = left_stmts_tuple[0]

            if_stmt_body += left
            # Statements for the right part of the idiom
            if_stmt_body += rigth_stmts_tuple[0]

            test_condition_call = []
            condition_stmt = []
            if_stmt_body.append(stypy_functions.create_blank_line())
        else:
            # Process conditional expression of the if
            condition_stmt, if_stmt_test = self.visit(node.test, context)

            # Test the type of the if condition
            condition_comment = stypy_functions.create_src_comment("Testing the type of an if condition", node.lineno)
            attribute = core_language.create_Name("is_suitable_condition")
            localization = stypy_functions.create_localization(node.lineno, node.col_offset)
            # test_condition_call = functions.create_call_expression(attribute, [localization, if_stmt_test])
            test_condition = functions.create_call(attribute, [localization, if_stmt_test])
            condition_assign, assign_var = stypy_functions.create_temp_Assign(test_condition, node.lineno,
                                                                              node.col_offset, "if_condition")
            condition_type = stypy_functions.create_set_type_of(assign_var.id, assign_var, node.lineno, node.col_offset)
            test_condition_call = [condition_assign, condition_type]

            # Test if the condition dynamically evaluates to None
            ev_none_condition_comment = stypy_functions.create_src_comment("Testing if the type of an if condition is none", node.lineno)
            ev_none_attribute = core_language.create_Name("evaluates_to_none")

            ev_none_test_condition = functions.create_call(ev_none_attribute, [localization, if_stmt_test])

        more_types_temp_var = None
        if is_an_idiom:
            idiom_if_func = core_language.create_Name(idioms.get_recognized_idiom_function(idiom_name), node.lineno,
                                                      node.col_offset)
            call_to_idiom_if_func = functions.create_call(idiom_if_func, [left_stmts_tuple[1], rigth_stmts_tuple[1]])

            may_be_type_temp_var = stypy_functions.new_temp_Name(False, idioms.may_be_var_name, node.lineno,
                                                                 node.col_offset)
            more_types_temp_var = stypy_functions.new_temp_Name(False, idioms.more_types_var_name, node.lineno,
                                                                node.col_offset)
            if_func_ret_type_tuple = core_language.create_type_tuple(may_be_type_temp_var, more_types_temp_var)
            may_be_type_assignment = core_language.create_Assign(if_func_ret_type_tuple, call_to_idiom_if_func)
            if_stmt_body.append(stypy_functions.flatten_lists(may_be_type_assignment))

            # Create first idiom if
            idiom_first_if_body = []
            if_may_be = conditional_statements.create_if(may_be_type_temp_var, idiom_first_if_body)
            if_stmt_body.append(if_may_be)

            # Create second idiom if
            idiom_more_types_body = []
            if_more_types = conditional_statements.create_if(more_types_temp_var, idiom_more_types_body)

            # Begin the second if body
            idiom_more_types_body.append(stypy_functions.create_src_comment("Runtime conditional SSA", node.lineno))
            clone_stmt1, type_store_var1 = stypy_functions.create_open_ssa_context("idiom if")
            idiom_more_types_body.append(clone_stmt1)

            # Needed to remove the "used before assignment" warning in the generated body: Basically we store the
            # existing type store into the temp var name instead of cloning it.
            may_be_type_assignment = core_language.create_Assign(type_store_var1, core_language.create_Name(
                stypy_functions.default_module_type_store_var_name))
            if_more_types.orelse = [may_be_type_assignment]

            idiom_first_if_body.append(if_more_types)  # Second if goes inside first one

            # Set the type of the condition var according to the identified idiom

            set_type = idioms.set_type_of_idiom_var(idiom_name, "if", node.test, rigth_stmts_tuple[1], node.lineno,
                                                    node.col_offset)
            if not len(set_type) == 0:
                idiom_first_if_body.append(set_type)

            body_stmts_location = idiom_first_if_body
        else:
            # Begin the if body
            if_stmt_body.append(stypy_functions.create_src_comment("SSA begins for if statement", node.lineno))
            clone_stmt1, type_store_var1 = stypy_functions.create_open_ssa_context("if")
            if_stmt_body.append(clone_stmt1)
            body_stmts_location = if_stmt_body

        # Process if body sentences
        body_stmts_location.extend(self.__visit_instruction_body(node.body, context))

        # Process else branch
        if len(node.orelse) > 0:
            if is_an_idiom:
                idiom_more_types_body_2 = []
                if_more_types_2 = conditional_statements.create_if(more_types_temp_var, idiom_more_types_body_2)
                # Begin the third if body
                idiom_more_types_body_2.append(
                    stypy_functions.create_src_comment("Runtime conditional SSA for else branch", node.lineno))
                clone_stmt2 = stypy_functions.create_open_ssa_branch("idiom else")
                idiom_more_types_body_2.append(clone_stmt2)
                body_stmts_location.append(if_more_types_2)
                body_stmts_location = if_stmt_body  # Return to the parent sentence trunk of the function

                # Process else with idioms part:
                # Create first idiom if
                idiom_first_if_body_else = []

                or_cond = ast.BoolOp()
                or_cond.op = ast.Or()
                not_cond = ast.UnaryOp()
                not_cond.op = ast.Not()
                not_cond.operand = may_be_type_temp_var
                or_cond.values = [not_cond, more_types_temp_var]

                if_may_be = conditional_statements.create_if(or_cond, idiom_first_if_body_else)
                body_stmts_location.append(if_may_be)

                # Create second idiom if
                idiom_more_types_body_else = []

                and_cond = ast.BoolOp()
                and_cond.op = ast.And()
                and_cond.values = [may_be_type_temp_var, more_types_temp_var]

                # Set the type of the condition var according to the identified idiom in the else branch
                set_type = idioms.set_type_of_idiom_var(idiom_name, "else", node.test, rigth_stmts_tuple[1],
                                                        node.lineno,
                                                        node.col_offset)

                idiom_first_if_body_else.append(set_type)
                body_stmts_location = idiom_first_if_body_else
            else:
                if_stmt_body.append(
                    stypy_functions.create_src_comment("SSA branch for the else part of an if statement",
                                                       node.lineno))
                clone_stmt2 = stypy_functions.create_open_ssa_branch("else")
                if_stmt_body.append(clone_stmt2)
                body_stmts_location = if_stmt_body

            # Process else body sentences
            else_stmts = self.__visit_instruction_body(node.orelse, context)
            body_stmts_location.extend(else_stmts)

        if is_an_idiom:
            if len(node.orelse) == 0:
                idiom_more_types_body_final = []
                if_more_types_final = conditional_statements.create_if(more_types_temp_var, idiom_more_types_body_final)
                body_stmts_location.append(if_more_types_final)
                final_stmts = idiom_more_types_body_final
            else:
                idiom_more_types_body_final = []

                and_cond = ast.BoolOp()
                and_cond.op = ast.And()
                and_cond.values = [may_be_type_temp_var, more_types_temp_var]

                if_more_types_final = conditional_statements.create_if(and_cond, idiom_more_types_body_final)
                body_stmts_location.append(if_more_types_final)

                final_stmts = idiom_more_types_body_final
        else:
            final_stmts = if_stmt_body

        # Join if
        final_stmts.append(stypy_functions.create_src_comment("SSA join for if statement", node.lineno))
        join_stmt, final_type_store = stypy_functions.create_join_ssa_context()
        final_stmts.append(join_stmt)

        # Unify all if body statements
        all_if_stmts = if_stmt_body

        # Unify all if sentences
        if_stmt = stypy_functions.flatten_lists(condition_comment, test_condition_call, all_if_stmts)

        end_if_comment = stypy_functions.create_blank_line()

        if ev_none_test_condition is None:
            return stypy_functions.flatten_lists(begin_if_comment,
                                             condition_stmt,
                                             if_stmt,
                                             end_if_comment)
        else:
            # This if controls a runtime idiom when an if only executes its else part if its condition dynamically
            # evaluates to None
            if_none = ast.If()

            if len(else_stmts) == 0:
                if_none.body = [ast.Pass()]
            else:
                if_none.body = else_stmts
            if_none.test = ev_none_test_condition

            if_none.orelse = stypy_functions.flatten_lists(begin_if_comment,
                                             if_stmt,
                                             end_if_comment)

            return stypy_functions.flatten_lists(condition_stmt, [ev_none_condition_comment, if_none])

    # ################################################ FOR #####################################################

    def visit_For(self, node, context):
        """
        Visit a For node
        :param node:
        :param context:
        :return:
        """

        for_stmt_body = []
        for_stmt_orelse = []
        else_inst = []

        begin_for_comment = stypy_functions.create_blank_line()

        # Process for test
        iter_stmt, for_stmt_test = self.visit(node.iter, context)

        set_type_of_loop_var = stypy_functions.create_set_type_of(for_stmt_test.id, for_stmt_test, node.lineno,
                                                                              node.col_offset)

        # Check if the for statement is suitable for iteration
        condition_comment = stypy_functions.create_src_comment("Testing the type of a for loop iterable",
                                                               node.lineno)
        loop_test_fname = core_language.create_Name("is_suitable_for_loop_condition")
        localization = stypy_functions.create_localization(node.lineno, node.col_offset)
        condition_call = functions.create_call_expression(loop_test_fname, [localization, for_stmt_test])

        # This if controls if the loop is going to be iterated
        if_node_iteration = ast.If()

        # Check if the for statement is suitable for iteration
        iteration_comment = stypy_functions.create_src_comment("Testing if the for loop is going to be iterated",
                                                               node.lineno)
        loop_it_fname = core_language.create_Name("will_iterate_loop")
        iteration_call = functions.create_call(loop_it_fname, [localization, for_stmt_test])

        if_node_iteration.body = []
        if_node_iteration.test = iteration_call

        if_node_iteration.orelse = []


        # Get the type of the loop iteration variable and assign it
        get_target_comment = stypy_functions.create_src_comment("Getting the type of the for loop variable",
                                                                node.lineno)
        for_stmt_body.append(get_target_comment)
        loop_target_fname = core_language.create_Name("get_type_of_for_loop_variable")
        target_assign_call = functions.create_call(loop_target_fname, [localization, for_stmt_test])
        target_assign, target_assign_var = stypy_functions.create_temp_Assign(target_assign_call, node.lineno,
                                                                              node.col_offset, "for_loop_var")
        for_stmt_body.append(target_assign)

        if isinstance(node.target, ast.Tuple):
            get_elements_call = core_language.create_Name("get_contained_elements_type")
            assign_target_type = []
            cont = 0
            for elt in node.target.elts:
                if isinstance(elt, ast.Name):
                    call_to_elements = functions.create_call(get_elements_call,
                                                             [localization, core_language.create_Name(
                                                                 target_assign_var.id),
                                                              core_language.create_num(len(node.target.elts)),
                                                              core_language.create_num(cont)])
                    type_set = stypy_functions.create_set_type_of(elt.id, call_to_elements, node.lineno,
                                                                  node.col_offset)
                    assign_target_type.append(type_set)
                else:
                    names = stypy_functions.extract_name_node_values(elt)
                    for name in names:
                        call_to_elements = functions.create_call(get_elements_call,
                                                                 [localization, core_language.create_Name(
                                                                     target_assign_var.id),
                                                              core_language.create_num(len(node.target.elts)),
                                                              core_language.create_num(cont)])
                        type_set = stypy_functions.create_set_type_of(name, call_to_elements, node.lineno,
                                                                      node.col_offset)
                        assign_target_type.append(type_set)
                cont+=1
        else:
            if type(node.target) is ast.Name:
                assign_target_type = stypy_functions.create_set_type_of(node.target.id, target_assign_var, node.lineno,
                                                                    node.col_offset)
            else:
                # for iterating over an attribute
                get_target_stmts, target_left_var = self.visit(node.target.value, context)

                # assign_target_type = stypy_functions.create_set_type_of(target_left_var.id, target_assign_var, node.lineno,
                #                                                     node.col_offset)

                set_type_of_member_stmts = stypy_functions.create_set_type_of_member(target_left_var,
                                                                                               node.target.attr,
                                                                                               target_assign_var,
                                                                                               node.lineno,
                                                                                               node.col_offset)

                assign_target_type = get_target_stmts + [set_type_of_member_stmts]

        for_stmt_body.append(assign_target_type)

        # For body
        for_stmt_body.append(stypy_functions.create_src_comment("SSA begins for a for statement", node.lineno))
        clone_stmt1, type_store_var1 = stypy_functions.create_open_ssa_context("for loop")
        for_stmt_body.append(clone_stmt1)

        # Process for body statements
        for_stmt_body.extend(self.__visit_instruction_body(node.body, context))

        if len(node.orelse) > 0:
            for_stmt_orelse.append(
                stypy_functions.create_src_comment("SSA branch for the else part of a for statement", node.lineno))
            clone_stmt2 = stypy_functions.create_open_ssa_branch("for loop else")
            for_stmt_orelse.append(clone_stmt2)

            # Else part of a for statement
            else_inst = self.__visit_instruction_body(node.orelse, context)
            for_stmt_orelse.extend(else_inst)

            # # Join and finish for
            # for_stmt_orelse.append(stypy_functions.create_src_comment("SSA join for a for else statement"))
            # join_stmt, final_type_store = stypy_functions.create_join_ssa_context()
            # for_stmt_orelse.append(join_stmt)

        if len(node.orelse) > 0:
            for_stmts = for_stmt_body + for_stmt_orelse
        else:
            for_stmts = for_stmt_body

        # Join and finish for
        for_stmts.append(stypy_functions.create_src_comment("SSA join for a for statement"))
        join_stmt, final_type_store = stypy_functions.create_join_ssa_context()
        for_stmts.append(join_stmt)

        if_node_iteration.body = [for_stmts]
        if_node_iteration.orelse = else_inst

        for_stmt = stypy_functions.flatten_lists(condition_comment, condition_call, if_node_iteration)

        end_for_comment = stypy_functions.create_blank_line()

        return stypy_functions.flatten_lists(begin_for_comment,
                                             iter_stmt,
                                             set_type_of_loop_var,
                                             iteration_comment,
                                             for_stmt,
                                             end_for_comment)

    # ################################################ WHILE #####################################################

    def visit_While(self, node, context):
        """
        Visit a While node
        :param node:
        :param context:
        :return:
        """

        while_stmt_body = []
        while_stmt_orelse = []
        else_inst = []

        begin_while_comment = stypy_functions.create_blank_line()

        # Process the condition of the while statement
        condition_stmt, while_stmt_test = self.visit(node.test, context)

        set_type_of_loop_var = stypy_functions.create_set_type_of(while_stmt_test.id, while_stmt_test, node.lineno,
                                                                              node.col_offset)

        # Test the type of the while condition
        condition_comment = stypy_functions.create_src_comment("Testing the type of an if condition", node.lineno)
        attribute = core_language.create_Name("is_suitable_condition")
        localization = stypy_functions.create_localization(node.lineno, node.col_offset)
        condition_call = functions.create_call_expression(attribute, [localization, while_stmt_test])

        # This if controls if the loop is going to be iterated
        if_node_iteration = ast.If()

        # Check if the for statement is suitable for iteration
        iteration_comment = stypy_functions.create_src_comment("Testing if the while is going to be iterated",
                                                               node.lineno)
        loop_it_fname = core_language.create_Name("will_iterate_loop")
        iteration_call = functions.create_call(loop_it_fname, [localization, while_stmt_test])

        if_node_iteration.body = []
        if_node_iteration.test = iteration_call

        if_node_iteration.orelse = []


        # Process the body of the while statement
        while_stmt_body.append(stypy_functions.create_src_comment("SSA begins for while statement", node.lineno))
        clone_stmt1, type_store_var1 = stypy_functions.create_open_ssa_context("while loop")
        while_stmt_body.append(clone_stmt1)

        # While body
        while_stmt_body.extend(self.__visit_instruction_body(node.body, context))

        if len(node.orelse) > 0:
            # Else part of the while statements
            while_stmt_orelse.append(
                stypy_functions.create_src_comment("SSA branch for the else part of a while statement",
                                                   node.lineno))
            clone_stmt2 = stypy_functions.create_open_ssa_branch("while loop else")

            while_stmt_orelse.append(clone_stmt2)

            # While else part
            else_inst = self.__visit_instruction_body(node.orelse, context)
            while_stmt_orelse.extend(else_inst)

            # # Join type stores and finish while
            # while_stmt_orelse.append(stypy_functions.create_src_comment("SSA join for while else statement", node.lineno))
            #
            # join_stmt, final_type_store = stypy_functions.create_join_ssa_context()
            # while_stmt_orelse.append(join_stmt)

        if len(node.orelse) > 0:
            all_while_stmts = while_stmt_body + while_stmt_orelse
        else:
            all_while_stmts = while_stmt_body

        # Join type stores and finish while
        all_while_stmts.append(stypy_functions.create_src_comment("SSA join for while statement", node.lineno))

        join_stmt, final_type_store = stypy_functions.create_join_ssa_context()
        all_while_stmts.append(join_stmt)

        if_node_iteration.body = [all_while_stmts]
        if_node_iteration.orelse = else_inst

        while_stmt = stypy_functions.flatten_lists(condition_comment, condition_call, if_node_iteration)

        end_while_comment = stypy_functions.create_blank_line()
        return stypy_functions.flatten_lists(begin_while_comment,
                                             condition_stmt,
                                             set_type_of_loop_var,
                                             iteration_comment,
                                             while_stmt,
                                             end_while_comment)

    # ################################################ EXCEPTIONS #####################################################

    def visit_TryExcept(self, node, context):
        """
        Visit a TryExcept node
        :param node:
        :param context:
        :return:
        """

        try_except_stmts = []
        begin_except_comment = stypy_functions.create_blank_line()

        # Begin the exception body
        try_except_stmts.append(stypy_functions.create_src_comment("SSA begins for try-except statement", node.lineno))
        clone_stmt1, type_store_var1 = stypy_functions.create_open_ssa_context("try-except")
        try_except_stmts.append(clone_stmt1)

        # Process exception body sentences
        try_except_stmts.extend(self.__visit_instruction_body(node.body, context))

        try_except_stmts.append(
            stypy_functions.create_src_comment("SSA branch for the except part of a try statement", node.lineno))

        # Process all except handlers
        for handler in node.handlers:
            if handler.type is None:
                except_handler = "<any exception>"
            else:
                if type(handler.type) is ast.Name:
                    except_handler = handler.type.id
                else:
                    except_handler = type(handler.type).__name__

            try_except_stmts.append(stypy_functions.create_src_comment(
                "SSA branch for the except '{0}' branch of a try statement".format(except_handler), node.lineno))

            if not handler.type is None and not handler.name is None:
                try_except_stmts.append(stypy_functions.create_src_comment("Storing handler type"))
                handler_type_stmts, handle_type_var = self.visit(handler.type, context)
                if not type(handler.name) is ast.Tuple:
                    handler_name_assign = stypy_functions.create_set_type_of(handler.name.id, handle_type_var,
                                                                         handler.lineno,
                                                                         handler.col_offset)
                else:
                    handler_name_assign = []
                    for el in handler.name.elts:
                        handler_name_assign.append(stypy_functions.create_set_type_of(el.id, handle_type_var,
                                                                             handler.lineno,
                                                                             handler.col_offset))

            else:
                handler_type_stmts = []
                handler_name_assign = []
            clone_stmt_handler = stypy_functions.create_open_ssa_branch("except")
            try_except_stmts.append(clone_stmt_handler)

            try_except_stmts.append(handler_type_stmts)
            try_except_stmts.append(handler_name_assign)

            # Process except handler body sentences
            try_except_stmts.extend(self.__visit_instruction_body(handler.body, context))

        # Process except handler body statements
        if len(node.orelse) > 0:
            # Else branch
            try_except_stmts.append(stypy_functions.create_src_comment(
                "SSA branch for the else branch of a try statement", node.lineno))

            clone_stmt_handler = stypy_functions.create_open_ssa_branch("except else")
            try_except_stmts.append(clone_stmt_handler)

            try_except_stmts.extend(self.__visit_instruction_body(node.orelse, context))

        # Join try
        try_except_stmts.append(stypy_functions.create_src_comment("SSA join for try-except statement", node.lineno))
        join_stmt, final_type_store = stypy_functions.create_join_ssa_context()

        try_except_stmts.append(join_stmt)

        # Unify all if sentences
        all_try_except_stmt = stypy_functions.flatten_lists(begin_except_comment, try_except_stmts)

        end_except_comment = stypy_functions.create_blank_line()

        return stypy_functions.flatten_lists(begin_except_comment,
                                             all_try_except_stmt,
                                             end_except_comment)

    def visit_TryFinally(self, node, context):
        """
        Visit a TryFinally node
        :param node:
        :param context:
        :return:
        """

        try_finally_stmts = []

        initial_comment = stypy_functions.create_src_comment("Try-finally block", node.lineno)
        try_finally_stmts.append(initial_comment)

        # Process exception body sentences
        try_finally_stmts.extend(self.__visit_instruction_body(node.body, context))

        try_finally_stmts.append(stypy_functions.create_blank_line())
        finally_comment = stypy_functions.create_src_comment("finally branch of the try-finally block",
                                                             node.lineno)
        try_finally_stmts.append(finally_comment)

        # Process exception body sentences
        try_finally_stmts.extend(self.__visit_instruction_body(node.finalbody, context))

        return stypy_functions.flatten_lists(stypy_functions.create_blank_line(),
                                             try_finally_stmts,
                                             stypy_functions.create_blank_line())

    # ################################################ IMPORTS ##########################################

    def visit_Import(self, node, context):
        """
        Visit an Import node
        :param node:
        :param context:
        :return:
        """

        # Default localization for all statements derived from the import line
        localization = stypy_functions.create_localization(node.lineno, node.col_offset)
        # Name of the function that imports elements
        import_func = core_language.create_Name(stypy_functions.default_import_function)

        # Name of the module type stores
        type_store_name = core_language.create_Name(stypy_functions.default_module_type_store_var_name)

        restore_path_instr = []

        # Call statements
        call_stmts = [
            stypy_functions.update_localization(node.lineno, node.col_offset),
            stypy_functions.create_blank_line()
        ]

        cont = 0
        for module_name in node.names:
            cont += 1

            if len(node.names) > 1:
                # Comment
                call_stmts.append(
                    stypy_functions.create_src_comment("Multiple import statement. import {0} ({1}/{2})".
                                                       format(module_name.name, cont, len(node.names)),
                                                       node.lineno))
            else:
                # Comment
                call_stmts.append(
                    stypy_functions.create_src_comment("'import {0}' statement".format(module_name.name),
                                                       node.lineno))
            # Name of the module to import
            if module_name.asname is None:
                module_to_import_name = core_language.create_str(module_name.name)
            else:
                module_to_import_name = core_language.create_str(module_name.asname)

            if is_python_library_module(module_name.name):
                node_clone = copy.copy(node)
                node_clone.names = [module_name]
                # Import the type inference equivalent of the module.
                call_stmts.append(node_clone)
                # Import elements from the library module
                call = functions.create_call_expression(import_func, [localization,
                                                                      module_to_import_name,
                                                                      core_language.create_Name(
                                                                          module_to_import_name.s),
                                                                      type_store_name,
                                                                      ],
                                                        )
            else:
                # Names of functions and parameters to use in code generation
                fix_path_func = core_language.create_Name("update_path_to_current_file_folder")
                restore_path_func = core_language.create_Name("remove_current_file_folder_from_path")

                parent_module = ""
                import os
                splitted_path = os.path.dirname(self.file_name).split('/')
                if hasattr(node, 'level'):
                    level = node.level - 1
                    if level < 0:
                        level = 0
                else:
                    level = 0

                for i in xrange(len(splitted_path) - level):
                    parent_module += splitted_path[i] + "/"

                # Generate code for putting the processed file directory as the first search path (to find modules
                # placed on the same directory than this one)
                file_name = core_language.create_str(parent_module)
                update_path_instr = functions.create_call_expression(fix_path_func, [file_name])
                call_stmts.append(update_path_instr)

                # Function call to restore the path at the end
                restore_path_instr = functions.create_call_expression(restore_path_func, [file_name])

                # This function will trigger type inference code generation for modules imported from this one
                generate_code_func = core_language.create_Name("generate_type_inference_code_for_module")
                call_expr = functions.create_call(generate_code_func, [localization,
                                                                       core_language.create_str(module_name.name)])
                assign_expr, assign_var = stypy_functions.create_temp_Assign(call_expr, node.lineno, node.col_offset,
                                                                             "import")
                call_stmts.append(assign_expr)

                # This if controls if the type inference code generation was an error
                if_node_error = ast.If()

                # This if distinguish between a non Python library module whose source is available or a native
                # compiled pyd
                if_node_pyd = ast.If()

                # Is this an error test
                if_node_error.body = if_node_pyd
                if_node_error.test = ast.Compare()
                type_call = functions.create_call(core_language.create_Name("type"), [assign_var])

                if_node_error.test.left = type_call
                if_node_error.test.comparators = [core_language.create_Name("StypyTypeError")]
                if_node_error.test.ops = [ast.IsNot()]

                # If the import process is an error, store the error as the value of a member whose name is the own
                # module
                else_instr = stypy_functions.create_set_type_of(module_name.name, assign_var,
                                                                node.lineno,
                                                                node.col_offset)
                if_node_error.orelse = [else_instr]

                # Pyd library processing
                if_node_pyd.test = ast.Compare()
                if_node_pyd.test.left = assign_var
                if_node_pyd.test.comparators = [core_language.create_str("pyd_module")]
                if_node_pyd.test.ops = [ast.NotEq()]

                __import__call = core_language.create_Name("__import__")
                # Import elements from the type inference program created from the module source code
                import_call = functions.create_call_expression(__import__call, [
                    assign_var,
                ])

                # Retrieve loaded module
                subscript = ast.Subscript()
                subscript.lineno = node.lineno
                subscript.col_offset = node.col_offset
                subscript.slice = assign_var

                subscript.value = core_language.create_attribute("sys", "modules")

                sys_modules_stmts, sys_modules_var = stypy_functions.create_temp_Assign(subscript,
                                                                                        node.lineno,
                                                                                        node.col_offset,
                                                                                        "sys_modules")

                # Import elements from the type inference program created from the module source code
                import_from_module_call = \
                    functions.create_call_expression(import_func, [localization,
                                                                   module_to_import_name,
                                                                   core_language.create_attribute(
                                                                       sys_modules_var,
                                                                       stypy_functions.
                                                                           default_module_type_store_var_name),
                                                                   type_store_name,
                                                                   ])

                if_node_pyd.body = stypy_functions.flatten_lists(import_call, sys_modules_stmts,
                                                                 import_from_module_call)

                # It is a pyd? load it normally
                if_node_pyd.orelse = []

                node_clone = copy.copy(node)
                node_clone.names = [module_name]

                # Import the type inference equivalent of the module.
                if_node_pyd.orelse.append(node_clone)
                # Import elements from the library module
                if_node_pyd.orelse.append(functions.create_call_expression(import_func, [localization,
                                                                                         module_to_import_name,
                                                                                         core_language.create_Name(
                                                                                             module_to_import_name.s),
                                                                                         type_store_name,
                                                                                         ],
                                                                           ))
                # Add the outer if
                call = if_node_error

            call_stmts.append(call)

        return stypy_functions.flatten_lists(call_stmts, restore_path_instr,
                                             stypy_functions.create_blank_line())

    def visit_ImportFrom(self, node, context):
        """
        Visit an ImportFrom node
        :param node:
        :param context:
        :return:
        """

        # Default localization for all statements derived from the import line
        localization = stypy_functions.create_localization(node.lineno, node.col_offset)
        # Name of the function that imports elements
        import_func = core_language.create_Name(stypy_functions.default_import_from_function)
        # Name of the module to import
        module_to_import_name = core_language.create_str(node.module)
        # Name of the module type stores
        type_store_name = core_language.create_Name(stypy_functions.default_module_type_store_var_name)

        restore_path_instr = []

        elements = []
        alias_stmt = []
        for alias in node.names:
            stypy_functions.assign_line_and_column(alias, node)
            alias_stmt, alias_var = self.visit(alias, context)
            elements.append(alias_var)

        # List of element names to be imported
        names = data_structures.create_list(elements)
        values_ids = []
        elements_for_comment = ""

        if len(elements) == 1 and elements[0].s == '*':
            values = core_language.create_None()
            elements_for_comment = "*"
        else:
            # List of element values to be imported
            for name in node.names:
                if name.asname is not None:
                    name_to_use = name.asname
                else:
                    name_to_use = name.name
                values_ids.append(core_language.create_Name(name_to_use))
                elements_for_comment += name_to_use + ", "
            values = data_structures.create_list(values_ids)

        # Call statements
        call_stmts = [stypy_functions.update_localization(node.lineno, node.col_offset),
                      stypy_functions.create_blank_line(),
                      # Comment
                      stypy_functions.create_src_comment("'from {0} import {1}' statement".format(node.module,
                                                                                                  elements_for_comment[
                                                                                                  :-2]),
                                                         node.lineno)]

        if is_python_library_module(node.module):
            node.level = 0

            try_node = ast.TryExcept()
            # Import the type inference equivalent of the module.
            try_node.body = [node]

            handler_node = ast.ExceptHandler()
            handler_node.body = []
            if type(values) is ast.Name:
                if not values.id == 'None':
                    handler_node.body.append(core_language.create_Assign(values, core_language.create_Name("UndefinedType")))
                else:
                    handler_node.body.append(ast.Pass())
            else:
                for val in values.elts:
                    handler_node.body.append(core_language.create_Assign(val, core_language.create_Name("UndefinedType")))
            #handler_node.body = [ast.Pass()]
            handler_node.type = None
            handler_node.name = None

            try_node.handlers = [handler_node]
            try_node.orelse = None

            call_stmts.append(try_node)

            # call_stmts.append(node)

            # Import elements from the library module
            call = functions.create_call_expression(import_func, [localization,
                                                                  module_to_import_name,
                                                                  core_language.create_None(),
                                                                  type_store_name,
                                                                  names,
                                                                  values
                                                                  ],
                                                    )
        else:
            # Names of functions and parameters to use in code generation
            fix_path_func = core_language.create_Name("update_path_to_current_file_folder")
            restore_path_func = core_language.create_Name("remove_current_file_folder_from_path")

            parent_module = ""
            import os
            splitted_path = os.path.dirname(self.file_name).split('/')
            level = node.level - 1
            if level < 0:
                level = 0
            for i in xrange(len(splitted_path) - level):
                parent_module += splitted_path[i] + "/"

            # Generate code for putting the processed file directory as the first search path (to find modules placed on
            # the same directory than this one)
            file_name = core_language.create_str(parent_module)
            update_path_instr = functions.create_call_expression(fix_path_func, [file_name])
            call_stmts.append(update_path_instr)

            # Function call to restore the path at the end
            restore_path_instr = functions.create_call_expression(restore_path_func, [file_name])

            # This function will trigger type inference code generation for modules imported from this one
            generate_code_func = core_language.create_Name("generate_type_inference_code_for_module")
            str_module_name = core_language.create_str(node.module)  # Imported module
            call_expr = functions.create_call(generate_code_func, [localization, str_module_name])
            assign_expr, assign_var = stypy_functions.create_temp_Assign(call_expr, node.lineno, node.col_offset,
                                                                         "import")
            call_stmts.append(assign_expr)

            # This if controls if the type inference code generation was an error
            if_node_error = ast.If()
            # This if distinguish between a non Python library module whose source is available or a native compiled pyd
            if_node_pyd = ast.If()

            # Is this an error test
            if_node_error.body = if_node_pyd
            if_node_error.test = ast.Compare()
            type_call = functions.create_call(core_language.create_Name("type"), [assign_var])

            if_node_error.test.left = type_call
            if_node_error.test.comparators = [core_language.create_Name("StypyTypeError")]
            if_node_error.test.ops = [ast.IsNot()]

            # If the import process is an error, store the error as the value of a member whose name is the own module
            else_instr = stypy_functions.create_set_type_of(node.module, assign_var,
                                                            node.lineno,
                                                            node.col_offset)
            if_node_error.orelse = [else_instr]

            # Pyd library processing
            if_node_pyd.test = ast.Compare()
            if_node_pyd.test.left = assign_var
            if_node_pyd.test.comparators = [core_language.create_str("pyd_module")]
            if_node_pyd.test.ops = [ast.NotEq()]

            __import__call = core_language.create_Name("__import__")
            # Import elements from the type inference program created from the module source code
            import_call = functions.create_call_expression(__import__call, [
                assign_var,
            ])

            # Retrieve loaded module
            subscript = ast.Subscript()
            subscript.lineno = node.lineno
            subscript.col_offset = node.col_offset
            subscript.slice = assign_var

            subscript.value = core_language.create_attribute("sys", "modules")

            sys_modules_stmts, sys_modules_var = stypy_functions.create_temp_Assign(subscript,
                                                                                    node.lineno,
                                                                                    node.col_offset,
                                                                                    "sys_modules")

            # Import elements from the type inference program created from the module source code
            import_from_module_call = \
                functions.create_call_expression(import_func, [localization,
                                                               module_to_import_name,
                                                               core_language.create_attribute(
                                                                   sys_modules_var,
                                                                   stypy_functions.
                                                                       default_module_type_store_var_name),
                                                               type_store_name,
                                                               names
                                                               ])

            # Nest modules if applicable
            nest_func = core_language.create_Name("nest_module")
            __file__name = core_language.create_Name("__file__")
            nest_module_call = \
                functions.create_call_expression(nest_func, [localization,
                                                             __file__name,
                                                             sys_modules_var,
                                                             core_language.create_attribute(
                                                                 sys_modules_var,
                                                                 stypy_functions.default_module_type_store_var_name),
                                                             type_store_name,
                                                             ])

            if_node_pyd.body = stypy_functions.flatten_lists(import_call, sys_modules_stmts, import_from_module_call,
                                                             nest_module_call)

            # It is a pyd? load it normally
            if_node_pyd.orelse = []

            node.level = 0
            # Import the type inference equivalent of the module.
            if_node_pyd.orelse.append(node)
            # Import elements from the library module
            if_node_pyd.orelse.append(functions.create_call_expression(import_func, [localization,
                                                                                     module_to_import_name,
                                                                                     core_language.create_None(),
                                                                                     type_store_name,
                                                                                     names,
                                                                                     values
                                                                                     ],
                                                                       ))
            # Add the outer if
            call = if_node_error

        call_stmts.append(call)

        # Future imports must be at the beginning of the file and cannot be processed
        if node.module == '__future__':
            return []
        else:
            return stypy_functions.flatten_lists(call_stmts, alias_stmt, restore_path_instr,
                                                 stypy_functions.create_blank_line())

    # #################################################### PRINT ######################################################

    def visit_Print(self, node, context):
        """
        Visits a Print node
        :param node:
        :param context:
        :return:
        """

        print_stmts = self.__visit_instruction_body(node.values, context)

        return stypy_functions.flatten_lists(print_stmts)

    # #################################################### GLOBAL #####################################################

    def visit_Global(self, node, context):
        """
        Visits a Global node
        :param node:
        :param context:
        :return:
        """
        global_stmts = [stypy_functions.create_src_comment("Marking variables as global", node.lineno)]
        mark_function = core_language.create_attribute(stypy_functions.default_module_type_store_var_name,
                                                       'declare_global', node.lineno, node.col_offset)
        localization = stypy_functions.create_localization(node.lineno, node.col_offset)
        for name in node.names:
            mark_call = functions.create_call_expression(mark_function, [localization, core_language.create_str(name)])
            global_stmts.append(mark_call)

        return stypy_functions.flatten_lists(global_stmts)

    # ######################################### BREAK AND CONTINUE ####################################################

    def visit_Break(self, node, context):
        """
        Visits a Break node
        :param node:
        :param context:
        :return:
        """
        return []

    def visit_Continue(self, node, context):
        """
        Visits a Continue node
        :param node:
        :param context:
        :return:
        """
        return []

    # ######################################### DELETE #######################################################

    def visit_Delete(self, node, context):
        """
        Visits a Delete node
        :param node:
        :param context:
        :return:
        """
        stmts = []

        comment = stypy_functions.create_src_comment("Deleting a member")
        for target in node.targets:
            # Is an attribute of a class or module
            if type(target) is ast.Attribute:
                member_to_delete = core_language.create_str(target.attr)
                if type(target.value) is ast.Name:
                    owner_stmts, owner_var = stypy_functions.create_get_type_of(target.value.id, node.lineno,
                                                                                node.col_offset)
                else:
                    owner_stmts, owner_var = self.visit(target.value, context)
            else:
                if type(target) is ast.Subscript:
                    owner_stmts, owner_var = self.visit(target.value, context)
                    member_to_delete_stmts, member_to_delete = self.visit(target, context)
                    owner_stmts += member_to_delete_stmts
                else:
                    # Is a name
                    member_to_delete = core_language.create_str(target.id)

                    owner_stmts = []
                    owner_var = core_language.create_Name(stypy_functions.default_module_type_store_var_name)

            if type(target) is ast.Subscript:
                delete_func = core_language.create_Name('del_contained_elements_type',
                                                        line=node.lineno,
                                                        column=node.col_offset)
            else:
                delete_func = core_language.create_attribute(stypy_functions.default_module_type_store_var_name,
                                                             'del_member',
                                                             line=node.lineno,
                                                             column=node.col_offset)

            localization = stypy_functions.create_localization(node.lineno, node.col_offset)
            delete_member_call = functions.create_call_expression(delete_func,
                                                                  [localization, owner_var, member_to_delete])

            stmts += stypy_functions.flatten_lists(comment, owner_stmts, delete_member_call)

        return stmts

    # ######################################### ASSERT #######################################################

    def visit_Assert(self, node, context):
        """
        Visits an Assert node
        :param node:
        :param context:
        :return:
        """
        comment = stypy_functions.create_src_comment("Evaluating assert statement condition")
        stmts, var = self.visit(node.test, context)

        target_assign, target_assign_var = stypy_functions.create_temp_Assign(var, node.lineno,
                                                                              node.col_offset, "assert")
        assing_to_var = stypy_functions.create_set_type_of(target_assign_var.id, var, node.lineno,
                                                           node.col_offset)
        return stypy_functions.flatten_lists(comment, stmts, target_assign, [assing_to_var])

    def visit_Exec(self, node, context):
        """
        Visits an Exec node
        :param node:
        :param context:
        :return:
        """
        src_comment = stypy_functions.create_src_comment("Dynamic code evaluation using an exec statement")
        localization = stypy_functions.create_localization(node.lineno, node.col_offset)
        type_check_func = core_language.create_Name('ensure_var_of_types',
                                                    line=node.lineno,
                                                    column=node.col_offset)

        body_stmts, body_var = self.visit(node.body, context)

        description = core_language.create_str("exec parameter")
        string_type = core_language.create_str("StringType")
        file_type = core_language.create_str("FileType")
        code_type = core_language.create_str("CodeType")
        check_params_call = functions.create_call_expression(type_check_func,
                                                             [localization, body_var, description,
                                                              string_type,
                                                              file_type,
                                                              code_type])

        dynamic_warning_func = core_language.create_Name("enable_usage_of_dynamic_types_warning")
        unsupported_feature_call = functions.create_call_expression(dynamic_warning_func,
                                                                    [localization])

        return stypy_functions.flatten_lists(src_comment, body_stmts, check_params_call, unsupported_feature_call)

    def visit_Yield(self, node, context):
        """
        Visits a Yield node
        :param node:
        :param context:
        :return:
        """
        src_comment = stypy_functions.create_src_comment("Creating a generator")
        localization = stypy_functions.create_localization(node.lineno, node.col_offset)
        if node.value is not None:
            generator_stored_type_stmts, generator_stored_type_var = self.visit(node.value, context)
        else:
            generator_stored_type_stmts = []
            generator_stored_type_var = core_language.create_None()

        generator_stmts, generator_type_var = stypy_functions.create_get_type_instance_of("GeneratorType",
                                                                                          node.lineno,
                                                                                          node.col_offset)

        # Assign to the generator its stored type
        store_type_func = core_language.create_Name('set_contained_elements_type',
                                                    line=node.lineno,
                                                    column=node.col_offset)

        store_type_call = functions.create_call_expression(store_type_func, [localization, generator_type_var,
                                                                             generator_stored_type_var])

        ret_assign = stypy_functions.assign_as_return_type(generator_type_var, node.lineno, node.col_offset)

        return stypy_functions.flatten_lists(src_comment, generator_stored_type_stmts, generator_stmts,
                                             store_type_call,
                                             ret_assign)

    def visit_Raise(self, node, context):
        """
        Visits a Raise node
        :param node:
        :param context:
        :return:
        """
        if node.type is None:
            return []

        localization = stypy_functions.create_localization(node.lineno, node.col_offset)

        type_stmts, type_var = self.visit(node.type, context)

        type_check_func = core_language.create_Name('ensure_var_of_types',
                                                    line=node.lineno,
                                                    column=node.col_offset)

        description = core_language.create_str("raise parameter")
        admitted_type = core_language.create_Name('BaseException')

        check_params_call = functions.create_call_expression(type_check_func,
                                                             [localization, type_var, description,
                                                              admitted_type])

        return stypy_functions.flatten_lists(type_stmts, check_params_call)

    def visit_With(self, node, context):
        """
        Visits a With node
        :param node:
        :param context:
        :return:
        """
        localization = stypy_functions.create_localization(node.lineno, node.col_offset)
        type_check_func = core_language.create_Name('ensure_var_has_members',
                                                    line=node.lineno,
                                                    column=node.col_offset)

        context_stmts, context_var = self.visit(node.context_expr, context)

        description = core_language.create_str("with parameter")

        enter = core_language.create_str("__enter__")
        exit = core_language.create_str("__exit__")
        check_params_call = functions.create_call(type_check_func,
                                                  [localization, context_var, description,
                                                   enter,
                                                   exit,
                                                   ])

        call_test_stmts, call_test_var = stypy_functions.create_temp_Assign(check_params_call, line=node.lineno,
                                                                            column=node.col_offset,
                                                                            descriptive_var_name='with')
        # Call __enter__
        enter_comment = stypy_functions.create_src_comment("Calling the __enter__ method to initiate a with section")
        enter_method, enter_var = stypy_functions.create_get_type_of_member(context_var, '__enter__', node.lineno,
                                                                            node.col_offset)
        enter_method_invoke = core_language.create_Name('invoke', node.lineno, node.col_offset)
        enter_method_call, call_var = stypy_functions.create_temp_Assign(
            functions.create_call(enter_method_invoke, [localization, enter_var]), node.lineno, node.col_offset,
            descriptive_var_name='with_enter')

        body_stmts = [enter_comment, enter_method, enter_method_call]
        if node.optional_vars is not None:
            if type(node.optional_vars) is not ast.Tuple:
                assing_to_var = stypy_functions.create_set_type_of(node.optional_vars.id, call_var, node.lineno,
                                                                       node.col_offset)
                body_stmts += [assing_to_var]
            else:
                assing_to_var = []
                for var in node.optional_vars.elts:
                    assing_to_var.append(stypy_functions.create_set_type_of(var.id, call_var, node.lineno,
                                                                       node.col_offset))

                body_stmts += [assing_to_var]

        body_stmts += self.__visit_instruction_body(node.body, context)

        # Call __exit__
        exit_comment = stypy_functions.create_src_comment("Calling the __exit__ method to finish a with section")
        exit_method, exit_var = stypy_functions.create_get_type_of_member(context_var, '__exit__', node.lineno,
                                                                          node.col_offset)
        exit_method_invoke = core_language.create_Name('invoke', node.lineno, node.col_offset)
        none_type = core_language.create_Name('None')
        exit_method_call, call_var = stypy_functions.create_temp_Assign(functions.create_call(exit_method_invoke,
                                                                                              [localization,
                                                                                               exit_var,
                                                                                               none_type,
                                                                                               none_type,
                                                                                               none_type]),
                                                                        node.lineno, node.col_offset,
                                                                        descriptive_var_name='with_exit')

        body_stmts += [exit_comment, exit_method, exit_method_call]

        with_if = conditional_statements.create_if(call_test_var, body_stmts)

        return stypy_functions.flatten_lists(context_stmts, call_test_stmts, with_if)

    def visit_arguments(self, node, context):
        """
        Unused. Integrated into visit_FunctionDef
        :param node:
        :param context:
        :return:
        """
        return node
