import ast

import conditional_statements
import core_language
import data_structures
import stypy_functions
from stypy import types

"""
This file contains helper functions to generate type inference code. 
These functions refer to function-related language elements such as declarations and invokations.
"""


def create_call(func, args, keywords=list(), kwargs=None, starargs=None, line=0, column=0):
    """
    Creates an AST Call node

    :param func: Function name
    :param args: List of arguments
    :param keywords: List of default arguments
    :param kwargs: Dict of keyword arguments
    :param starargs: Variable list of arguments
    :param line: Line
    :param column: Column
    :return: AST Call node
    """
    call = ast.Call()
    call.args = []

    if data_structures.is_iterable(args):
        for arg in args:
            call.args.append(arg)
    else:
        call.args.append(args)

    call.func = func
    call.lineno = line
    call.col_offset = column
    call.keywords = keywords
    call.kwargs = kwargs
    call.starargs = starargs

    return call


def create_call_expression(func, args, keywords=list(), kwargs=None, starargs=None, line=0, column=0):
    """
    Creates an AST Call node that will be enclosed in an expression node. This is used when the call are not a part
    of a longer expression, but the expression itself

    :param func: Function name
    :param args: List of arguments
    :param keywords: List of default arguments
    :param kwargs: Dict of keyword arguments
    :param starargs: Variable list of arguments
    :param line: Line
    :param column: Column
    :return: AST Expr node
    """
    call = create_call(func, args, keywords, kwargs, starargs, line, column)
    call_expression = ast.Expr()
    call_expression.value = call
    call_expression.lineno = line
    call_expression.col_offset = column

    return call_expression


def is_method(context, decorators=[]):
    """
    Determines if an AST Function node represent a method (belongs to an AST ClassDef node)
    :param context:
    :param decorators: Decorators of the method
    :return:
    """
    ismethod = False

    if not len(context) == 0:
        ismethod = isinstance(context[-1], ast.ClassDef)

    is_static = len(filter(lambda name: name.id == "staticmethod", decorators)) > 0
    return ismethod and not is_static


def is_static_method(node):
    """
    Checks if a method is static
    :param node:
    :return:
    """
    if not hasattr(node, "decorator_list"):
        return False
    if len(node.decorator_list) == 0:
        return False
    for dec_name in node.decorator_list:
        if hasattr(dec_name, "id"):
            if dec_name.id == "staticmethod":
                return True
    return False


def is_constructor(node):
    """
    Determines if an AST Function node represent a constructor (its name is __init__)
    :param node: AST Function node or str
    :return: bool
    """
    if type(node) is str:
        return node == "__init__"

    return node.name == "__init__"


def create_function_def(name, localization, decorators, context, line=0, column=0):
    """
    Creates a FunctionDef node, that represent a function declaration. This is used in type inference code, so every
    created function has the following parameters (type_of_self, localization, *varargs, **kwargs) for methods and
    (localization, *varargs, **kwargs) for functions.

    :param name: Name of the function
    :param localization: Localization parameter
    :param decorators: Decorators of the function, mainly the norecursion one
    :param context: Context passed to this method
    :param line: Line
    :param column: Column
    :return: An AST FunctionDef node
    """
    function_def_arguments = ast.arguments()
    function_def_arguments.args = [localization]

    isconstructor = is_constructor(name)
    ismethod = is_method(context, decorators)

    function_def = ast.FunctionDef()
    function_def.lineno = line
    function_def.col_offset = column
    if types.type_inspection.is_special_name_method(name):
        function_def.name = types.type_inspection.convert_special_name_method(name)
    else:
        function_def.name = name

    function_def.args = function_def_arguments

    function_def_arguments.args = []

    if isconstructor or (ismethod and not isconstructor):
        function_def_arguments.args.append(core_language.create_Name('type_of_self'))

    function_def_arguments.args.append(localization)

    function_def_arguments.kwarg = "kwargs"
    function_def_arguments.vararg = "varargs"
    function_def_arguments.defaults = []

    if data_structures.is_iterable(decorators):
        function_def.decorator_list = decorators
    else:
        function_def.decorator_list = [decorators]

    global_ts = ast.Global()
    global_ts.names = [stypy_functions.default_module_type_store_var_name]
    function_def.body = [global_ts]

    return function_def


def create_return(value):
    """
    Creates an AST Return node
    :param value: Value to return
    :return: An AST Return node
    """
    node = ast.Return()
    node.value = value

    return node


def obtain_arg_list(args, ismethod=False, isstaticmethod=False):
    """
    Creates an AST List node with the names of the arguments passed to a function
    :param args: Arguments
    :param ismethod: Whether to count the first argument (self) or not
    :param isstaticmethod: Determines if the method is static
    :return: An AST List
    """
    arg_list = ast.List()

    arg_list.elts = []
    if ismethod and not isstaticmethod:
        arg_list_contents = args.args[1:]
    else:
        arg_list_contents = args.args

    try:
        for arg in arg_list_contents:
            arg_list.elts.append(core_language.create_str(arg.id))
    except Exception as ex:
        print ex

    return arg_list


def create_stacktrace_push(func_name, declared_arguments):
    """
    Creates an AST Node that model the call to the localitazion.set_stack_trace method

    :param func_name: Name of the function that will do the push to the stack trace
    :param declared_arguments: Arguments of the call
    :return: An AST Expr node
    """
    # Code to push a new stack trace to handle errors.
    attribute = core_language.create_attribute("localization", "set_stack_trace")
    arguments_var = core_language.create_Name("arguments")
    stack_push_call = create_call(attribute, [core_language.create_str(func_name), declared_arguments, arguments_var])
    stack_push = ast.Expr()
    stack_push.value = stack_push_call

    return stack_push


def create_stacktrace_pop():
    """
    Creates an AST Node that model the call to the localitazion.unset_stack_trace method

    :return: An AST Expr node
    """
    # Code to pop a stack trace once the function finishes.
    attribute = core_language.create_attribute("localization", "unset_stack_trace")
    stack_pop_call = create_call(attribute, [])
    stack_pop = ast.Expr()
    stack_pop.value = stack_pop_call

    return stack_pop


def create_context_set(func_name, lineno, col_offset, access_parent=True):
    """
    Creates an AST Node that model the call to the type_store.set_context method

    :param func_name: Name of the function that will do the push to the stack trace
    :param lineno: Line
    :param col_offset: Column
    :param access_parent: Value of the "has access to its parent context" parameter
    :return: An AST Expr node
    """
    attribute = core_language.create_attribute(stypy_functions.default_module_type_store_var_name,
                                               "open_function_context")
    context_set_call = create_call(attribute, [core_language.create_str(func_name),
                                               core_language.create_num(lineno),
                                               core_language.create_num(col_offset),
                                               core_language.create_bool(access_parent)])

    left_hand_side = core_language.create_Name(stypy_functions.default_module_type_store_var_name, False)
    assign_statement = ast.Assign([left_hand_side], context_set_call)

    return assign_statement


def create_context_unset():
    """
    Creates an AST Node that model the call to the type_store.unset_context method

    :return: An AST Expr node
    """

    # Generate code for pop the function context.
    comment = stypy_functions.create_src_comment("Destroy the current context")

    # Code to pop a stack trace once the function finishes.
    attribute = core_language.create_attribute(stypy_functions.default_module_type_store_var_name,
                                               "close_function_context")
    context_unset_call = create_call(attribute, [])

    left_hand_side = core_language.create_Name(stypy_functions.default_module_type_store_var_name, False)
    assign_statement = ast.Assign([left_hand_side], context_unset_call)

    return stypy_functions.flatten_lists(comment, assign_statement)


def create_arg_number_test(function_def_node, decorators, context=[]):
    """
    Creates an AST Node that model the call to the process_argument_values method. This method is used to check
    the parameters passed to a function/method in a type inference program

    :param function_def_node: AST Node with the function definition
    :param decorators: Decorators of the function
    :param context: Context passed to the call
    :return: List of AST nodes that perform the call to the mentioned function and make the necessary tests once it
    is called
    """
    args_test_resul = core_language.create_Name('arguments', False)

    # Call to arg test function
    func = core_language.create_Name('process_argument_values')
    # Fixed parameters
    localization_arg = core_language.create_Name('localization')
    type_store_arg = core_language.create_Name(stypy_functions.default_module_type_store_var_name)

    # Declaration data arguments
    # Func name
    if is_method(context, decorators):
        if types.type_inspection.is_special_name_method(function_def_node.name):
            function_name_arg = core_language.create_str(context[-1].name + "." + types.type_inspection.convert_special_name_method(function_def_node.name))
        else:
            function_name_arg = core_language.create_str(context[-1].name + "." + function_def_node.name)
        type_of_self_arg = core_language.create_Name('type_of_self')
    else:
        if types.type_inspection.is_special_name_method(function_def_node.name):
            function_name_arg = core_language.create_str(types.type_inspection.convert_special_name_method(function_def_node.name))
        else:
            function_name_arg = core_language.create_str(function_def_node.name)
        type_of_self_arg = core_language.create_Name('None')

    # Declared param names list
    param_names_list_arg = obtain_arg_list(function_def_node.args, is_method(context),
                                           is_static_method(function_def_node))

    # Declared var args parameter name
    if function_def_node.args.vararg is None:
        declared_varargs = None
    else:
        declared_varargs = function_def_node.args.vararg
    varargs_param_name = core_language.create_str(declared_varargs)
    # Declared kwargs parameter name
    if function_def_node.args.kwarg is None:
        declared_kwargs = None
    else:
        declared_kwargs = function_def_node.args.kwarg
    kwargs_param_name = core_language.create_str(declared_kwargs)

    # Call data arguments
    # Declared defaults list name
    call_defaults = core_language.create_Name('defaults')  # function_def_node.args.defaults

    # Call varargs
    call_varargs = core_language.create_Name('varargs')
    # Call kwargs
    call_kwargs = core_language.create_Name('kwargs')

    # Store call information into the function object to recursion checks
    if is_method(context):
        f_name = function_def_node.name
        if types.type_inspection.is_special_name_method(function_def_node.name):
            f_name = types.type_inspection.convert_special_name_method(f_name)

        f_var_name = core_language.create_attribute(core_language.create_attribute(context[-1].name,
                                                                                   f_name), '__dict__')
    else:
        if types.type_inspection.is_special_name_method(function_def_node.name):
            f_var_name = core_language.create_Name(types.type_inspection.convert_special_name_method(function_def_node.name), False)
        else:
            f_var_name = core_language.create_Name(function_def_node.name, False)

    info_storage_instr = []

    if function_def_node.name is not '__init__':
        if is_method(context):
            subscript_call = create_call_expression(core_language.create_attribute(f_var_name, '__setitem__'),
                                                    [core_language.create_str('stypy_localization'), localization_arg])

            info_storage_instr.append(subscript_call)

            subscript_call = create_call_expression(core_language.create_attribute(f_var_name, '__setitem__'),
                                                    [core_language.create_str('stypy_type_of_self'), type_of_self_arg])

            info_storage_instr.append(subscript_call)

            subscript_call = create_call_expression(core_language.create_attribute(f_var_name, '__setitem__'),
                                                    [core_language.create_str('stypy_type_store'), type_store_arg])

            info_storage_instr.append(subscript_call)

            subscript_call = create_call_expression(core_language.create_attribute(f_var_name, '__setitem__'),
                                                    [core_language.create_str('stypy_function_name'),
                                                     function_name_arg])

            info_storage_instr.append(subscript_call)

            subscript_call = create_call_expression(core_language.create_attribute(f_var_name, '__setitem__'),
                                                    [core_language.create_str('stypy_param_names_list'),
                                                     param_names_list_arg])

            info_storage_instr.append(subscript_call)

            subscript_call = create_call_expression(core_language.create_attribute(f_var_name, '__setitem__'),
                                                    [core_language.create_str('stypy_varargs_param_name'),
                                                     varargs_param_name])

            info_storage_instr.append(subscript_call)

            subscript_call = create_call_expression(core_language.create_attribute(f_var_name, '__setitem__'),
                                                    [core_language.create_str('stypy_kwargs_param_name'),
                                                     kwargs_param_name])

            info_storage_instr.append(subscript_call)

            subscript_call = create_call_expression(core_language.create_attribute(f_var_name, '__setitem__'),
                                                    [core_language.create_str('stypy_call_defaults'), call_defaults])

            info_storage_instr.append(subscript_call)

            subscript_call = create_call_expression(core_language.create_attribute(f_var_name, '__setitem__'),
                                                    [core_language.create_str('stypy_call_varargs'), call_varargs])

            info_storage_instr.append(subscript_call)

            subscript_call = create_call_expression(core_language.create_attribute(f_var_name, '__setitem__'),
                                                    [core_language.create_str('stypy_call_kwargs'), call_kwargs])

            info_storage_instr.append(subscript_call)

            subscript_call = create_call_expression(core_language.create_attribute(f_var_name, '__setitem__'),
                                                    [core_language.create_str('stypy_declared_arg_number'),
                                                     core_language.create_num(len(function_def_node.args.args))])

            info_storage_instr.append(subscript_call)
        else:
            info_storage_instr.append(
                core_language.create_Assign(core_language.create_attribute(f_var_name, 'stypy_localization'),
                                            localization_arg))
            info_storage_instr.append(
                core_language.create_Assign(core_language.create_attribute(f_var_name, 'stypy_type_of_self'),
                                            type_of_self_arg))

            info_storage_instr.append(
                core_language.create_Assign(core_language.create_attribute(f_var_name, 'stypy_type_store'),
                                            type_store_arg))
            info_storage_instr.append(
                core_language.create_Assign(core_language.create_attribute(f_var_name, 'stypy_function_name'),
                                            function_name_arg))
            info_storage_instr.append(core_language.create_Assign(
                core_language.create_attribute(f_var_name, 'stypy_param_names_list'),
                param_names_list_arg))

            info_storage_instr.append(core_language.create_Assign(
                core_language.create_attribute(f_var_name, 'stypy_varargs_param_name'),
                varargs_param_name))
            info_storage_instr.append(core_language.create_Assign(
                core_language.create_attribute(f_var_name, 'stypy_kwargs_param_name'),
                kwargs_param_name))
            info_storage_instr.append(
                core_language.create_Assign(core_language.create_attribute(f_var_name, 'stypy_call_defaults'),
                                            call_defaults))
            info_storage_instr.append(
                core_language.create_Assign(core_language.create_attribute(f_var_name, 'stypy_call_varargs'),
                                            call_varargs))
            info_storage_instr.append(
                core_language.create_Assign(core_language.create_attribute(f_var_name, 'stypy_call_kwargs'),
                                            call_kwargs))
    # Parameter number check call
    call = create_call(func,
                       [localization_arg, type_of_self_arg, type_store_arg, function_name_arg, param_names_list_arg,
                        varargs_param_name, kwargs_param_name, call_defaults, call_varargs, call_kwargs])

    assign = core_language.create_Assign(args_test_resul, call)

    # After parameter number check call
    argument_errors = core_language.create_Name('arguments')
    is_error_type = core_language.create_Name('is_error_type')
    if_test = create_call(is_error_type, argument_errors)

    if is_constructor(function_def_node):
        argument_errors = None  # core_language.create_Name('None')

    body = [create_context_unset(), create_return(argument_errors)]
    if_ = conditional_statements.create_if(if_test, body)

    return info_storage_instr + [assign, if_]


def get_function_name_in_ti_files(name):
    """
    Get the name of the generated function in the type inference files, which is identical to the original function
    name except if it is a Python special function with special semantics (such as __str__). In that case, the function
    name is transformed to avoid triggering these semantics accidentally.
    :param name:
    :return:
    """
    if types.type_inspection.is_special_name_method(name):
        return types.type_inspection.convert_special_name_method(name)
    return name


def create_type_for_lambda_function(function_name, lambda_call, lineno, col_offset):
    """
    Creates a variable to store a lambda function definition

    :param function_name: Name of the lambda function
    :param lambda_call: Lambda function
    :param lineno: Line
    :param col_offset: Column
    :return: Statements to create the lambda function type
    """

    call_arg = core_language.create_Name(lambda_call)

    set_type_stmts = stypy_functions.create_set_type_of(function_name, call_arg, lineno, col_offset)

    # return stypy_functions.flatten_lists(assign, set_type_stmts)
    return stypy_functions.flatten_lists(set_type_stmts)
