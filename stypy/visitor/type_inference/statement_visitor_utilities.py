import ast


def is_nested_function(context):
    function_found = False
    class_found = False

    for elem in reversed(context):
        if type(elem) is ast.FunctionDef:
            function_found = True
        if type(elem) is ast.ClassDef and function_found:
            class_found = True

    return function_found and class_found


def is_true_constant_in_loop_test(node):
    try:
        if isinstance(node.test, ast.Name):
            return node.test.id == 'True'
    except:
        return False

    return False


def has_a_return_or_a_raise(body):
    for elem in body:
        if type(elem) is ast.Raise or type(elem) is ast.Return:
            return True

    return False