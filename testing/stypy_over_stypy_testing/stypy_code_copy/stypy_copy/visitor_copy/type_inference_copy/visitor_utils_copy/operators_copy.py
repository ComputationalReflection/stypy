import core_language_copy
import stypy_functions_copy
import functions_copy

"""
Helper functions to generate operator related nodes in the type inference AST
"""

# ##################################### OPERATORS #########################################


def create_binary_operator(op_name, op1, op2, line=0, column=0):
    localization = stypy_functions_copy.create_localization(line, column)

    binop_func = core_language_copy.create_Name("operator")
    binop = functions_copy.create_call(binop_func, [localization, op_name, op1, op2])

    return binop


def create_unary_operator(op_name, op, line=0, column=0):
    localization = stypy_functions_copy.create_localization(line, column)

    unop_func = core_language_copy.create_Name("operator")
    unop = functions_copy.create_call(unop_func, [localization, op_name, op])

    return unop
