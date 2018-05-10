import ast

from stypy.visitor.type_inference.visitor_utils import stypy_functions, core_language, functions


def is_direct_call_to_stypy_interface(node):
    try:
        if type(node.func) is ast.Attribute:
            return node.func.value.id == "stypy_interface"
    except:
        return False

    return False


def visit_Call_get_value_from_tuple(self, node, context):
    """
    Visits a Call node
    :param node:
    :param context:
    :return:
    """

    context.append(node)
    # Localization of the function call
    localization = stypy_functions.create_localization(node.lineno, node.col_offset)
    call_stmts = []
    arguments = []
    keyword_arguments = {}

    name_to_call = stypy_functions.get_descritive_element_name(node.func)

    # Obtain the function to be called
    call_stmts.append(stypy_functions.create_src_comment("Call to {0}(...):".format(name_to_call), node.lineno))

    function_to_call = core_language.create_Name('stypy_get_value_from_tuple')

    if len(node.args) > 0:
        call_stmts.append(stypy_functions.create_src_comment("Processing the call arguments", node.lineno))
    # First call parameters are built from standard parameters plus var args (args + starargs)
    # Evaluate arguments of the call

    stmts, temp = self.visit(node.args[0], context)
    call_stmts.append(stmts)

    arguments.append(temp)
    arguments.append(node.args[1])
    arguments.append(node.args[2])

    call_stmts.append(
        stypy_functions.create_src_comment("Calling {0}(tuple, tuple length, tuple pos)".format(name_to_call), node.lineno))
    call = functions.create_call(function_to_call, arguments, line=node.lineno, column=node.col_offset)

    assign_stmts, temp_assign = stypy_functions.create_temp_Assign(call, node.lineno, node.col_offset,
                                                                   descriptive_var_name="{0}_call_result".format(
                                                                       name_to_call))
    call_stmts.append(assign_stmts)

    return stypy_functions.flatten_lists(
        stypy_functions.create_blank_line(),
        call_stmts,
        stypy_functions.create_blank_line(),
    ), temp_assign


def visit_Call_generic_stypy_interface(self, node, context, fname):
    """
    Visits a Call node
    :param node:
    :param context:
    :return:
    """

    context.append(node)
    # Localization of the function call
    localization = stypy_functions.create_localization(node.lineno, node.col_offset)
    call_stmts = []
    arguments = []
    keyword_arguments = {}

    name_to_call = stypy_functions.get_descritive_element_name(node.func)

    # Obtain the function to be called
    call_stmts.append(stypy_functions.create_src_comment("Call to {0}(...):".format(name_to_call), node.lineno))

    function_to_call = core_language.create_Name(fname)

    if len(node.args) > 0:
        call_stmts.append(stypy_functions.create_src_comment("Processing the call arguments", node.lineno))
    # First call parameters are built from standard parameters plus var args (args + starargs)
    # Evaluate arguments of the call

    stmts, temp = self.visit(node.args[0], context)
    call_stmts.append(stmts)

    arguments.append(temp)
    for arg in node.args[1:]:
        arguments.append(arg)

    call_stmts.append(
        stypy_functions.create_src_comment("Calling {0}".format(name_to_call), node.lineno))
    call = functions.create_call(function_to_call, arguments, line=node.lineno, column=node.col_offset)

    assign_stmts, temp_assign = stypy_functions.create_temp_Assign(call, node.lineno, node.col_offset,
                                                                   descriptive_var_name="{0}_call_result".format(
                                                                       name_to_call))
    call_stmts.append(assign_stmts)

    return stypy_functions.flatten_lists(
        stypy_functions.create_blank_line(),
        call_stmts,
        stypy_functions.create_blank_line(),
    ), temp_assign


def call_to_stypy_interface(visitor, node, context):
    name_to_call = stypy_functions.get_descritive_element_name(node.func)
    if name_to_call == "stypy_get_value_from_tuple":
        return visit_Call_get_value_from_tuple(visitor, node, context)

    return visit_Call_generic_stypy_interface(visitor, node, context, name_to_call)
    #assert "Critical stypy error: Unexpected call to stypy interface from type inference code"
