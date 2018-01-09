import ast

import core_language_copy
import data_structures_copy
import conditional_statements_copy
import stypy_functions_copy

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

    if data_structures_copy.is_iterable(args):
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


def is_method(context):
    """
    Determines if an AST Function node represent a method (belongs to an AST ClassDef node)
    :param context:
    :return:
    """
    ismethod = False

    if not len(context) == 0:
        ismethod = isinstance(context[-1], ast.ClassDef)

    return ismethod


def is_static_method(node):
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
    ismethod = is_method(context)

    function_def = ast.FunctionDef()
    function_def.lineno = line
    function_def.col_offset = column
    function_def.name = name

    function_def.args = function_def_arguments

    function_def_arguments.args = []

    if isconstructor:
        function_def_arguments.args.append(core_language_copy.create_Name('type_of_self'))

    if ismethod and not isconstructor:
        function_def_arguments.args.append(core_language_copy.create_Name('type_of_self'))

    function_def_arguments.args.append(localization)

    function_def_arguments.kwarg = "kwargs"
    function_def_arguments.vararg = "varargs"
    function_def_arguments.defaults = []

    if data_structures_copy.is_iterable(decorators):
        function_def.decorator_list = decorators
    else:
        function_def.decorator_list = [decorators]

    function_def.body = []

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
    :return: An AST List
    """
    arg_list = ast.List()

    arg_list.elts = []
    if ismethod and not isstaticmethod:
        arg_list_contents = args.args[1:]
    else:
        arg_list_contents = args.args

    for arg in arg_list_contents:
        arg_list.elts.append(core_language_copy.create_str(arg.id))

    return arg_list


def create_stacktrace_push(func_name, declared_arguments):
    """
    Creates an AST Node that model the call to the localitazion.set_stack_trace method

    :param func_name: Name of the function that will do the push to the stack trace
    :param declared_arguments: Arguments of the call
    :return: An AST Expr node
    """
    # Code to push a new stack trace to handle errors.
    attribute = core_language_copy.create_attribute("localization", "set_stack_trace")
    arguments_var = core_language_copy.create_Name("arguments")
    stack_push_call = create_call(attribute, [core_language_copy.create_str(func_name), declared_arguments, arguments_var])
    stack_push = ast.Expr()
    stack_push.value = stack_push_call

    return stack_push


def create_stacktrace_pop():
    """
    Creates an AST Node that model the call to the localitazion.unset_stack_trace method

    :return: An AST Expr node
    """
    # Code to pop a stack trace once the function finishes.
    attribute = core_language_copy.create_attribute("localization", "unset_stack_trace")
    stack_pop_call = create_call(attribute, [])
    stack_pop = ast.Expr()
    stack_pop.value = stack_pop_call

    return stack_pop


def create_context_set(func_name, lineno, col_offset):
    """
    Creates an AST Node that model the call to the type_store.set_context method

    :param func_name: Name of the function that will do the push to the stack trace
    :param lineno: Line
    :param col_offset: Column
    :return: An AST Expr node
    """
    attribute = core_language_copy.create_attribute("type_store", "set_context")
    context_set_call = create_call(attribute, [core_language_copy.create_str(func_name),
                                               core_language_copy.create_num(lineno),
                                               core_language_copy.create_num(col_offset)])
    context_set = ast.Expr()
    context_set.value = context_set_call

    return context_set


def create_context_unset():
    """
    Creates an AST Node that model the call to the type_store.unset_context method

    :return: An AST Expr node
    """
    # Code to pop a stack trace once the function finishes.
    attribute = core_language_copy.create_attribute("type_store", "unset_context")
    context_unset_call = create_call(attribute, [])
    context_unset = ast.Expr()
    context_unset.value = context_unset_call

    return context_unset


def create_arg_number_test(function_def_node, context=[]):
    """
    Creates an AST Node that model the call to the process_argument_values method. This method is used to check
    the parameters passed to a function/method in a type inference program

    :param function_def_node: AST Node with the function definition
    :param context: Context passed to the call
    :return: List of AST nodes that perform the call to the mentioned function and make the necessary tests once it
    is called
    """
    args_test_resul = core_language_copy.create_Name('arguments', False)

    # Call to arg test function
    func = core_language_copy.create_Name('process_argument_values')
    # Fixed parameters
    localization_arg = core_language_copy.create_Name('localization')
    type_store_arg = core_language_copy.create_Name('type_store')

    # Declaration data arguments
    # Func name
    if is_method(context):
        function_name_arg = core_language_copy.create_str(context[-1].name + "." + function_def_node.name)
        type_of_self_arg = core_language_copy.create_Name('type_of_self')
    else:
        function_name_arg = core_language_copy.create_str(function_def_node.name)
        type_of_self_arg = core_language_copy.create_Name('None')

    # Declared param names list
    param_names_list_arg = obtain_arg_list(function_def_node.args, is_method(context),
                                           is_static_method(function_def_node))

    # Declared var args parameter name
    if function_def_node.args.vararg is None:
        declared_varargs = None
    else:
        declared_varargs = function_def_node.args.vararg
    varargs_param_name = core_language_copy.create_str(declared_varargs)
    # Declared kwargs parameter name
    if function_def_node.args.kwarg is None:
        declared_kwargs = None
    else:
        declared_kwargs = function_def_node.args.kwarg
    kwargs_param_name = core_language_copy.create_str(declared_kwargs)

    # Call data arguments
    # Declared defaults list name
    call_defaults = core_language_copy.create_Name('defaults')  # function_def_node.args.defaults

    # Call varargs
    call_varargs = core_language_copy.create_Name('varargs')
    # Call kwargs
    call_kwargs = core_language_copy.create_Name('kwargs')

    # Parameter number check call
    call = create_call(func,
                       [localization_arg, type_of_self_arg, type_store_arg, function_name_arg, param_names_list_arg,
                        varargs_param_name, kwargs_param_name, call_defaults, call_varargs, call_kwargs])

    assign = core_language_copy.create_Assign(args_test_resul, call)

    # After parameter number check call
    argument_errors = core_language_copy.create_Name('arguments')
    is_error_type = core_language_copy.create_Name('is_error_type')
    if_test = create_call(is_error_type, argument_errors)

    if is_constructor(function_def_node):
        argument_errors = None  # core_language.create_Name('None')

    body = [create_context_unset(), create_return(argument_errors)]
    if_ = conditional_statements_copy.create_if(if_test, body)

    return [assign, if_]


def create_type_for_lambda_function(function_name, lambda_call, lineno, col_offset):
    """
    Creates a variable to store a lambda function definition

    :param function_name: Name of the lambda function
    :param lambda_call: Lambda function
    :param lineno: Line
    :param col_offset: Column
    :return: Statements to create the lambda function type
    """
    # TODO: Remove?
    # call_arg = core_language.create_Name(lambda_call)
    # call_func = core_language.create_Name("LambdaFunctionType")
    # call = create_call(call_func, call_arg)
    # assign_target = core_language.create_Name(lambda_call, False)
    # assign = core_language.create_Assign(assign_target, call)

    call_arg = core_language_copy.create_Name(lambda_call)

    set_type_stmts = stypy_functions_copy.create_set_type_of(function_name, call_arg, lineno, col_offset)

    # return stypy_functions.flatten_lists(assign, set_type_stmts)
    return stypy_functions_copy.flatten_lists(set_type_stmts)
