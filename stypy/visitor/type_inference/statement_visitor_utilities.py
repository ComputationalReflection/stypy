import ast

def is_true_constant_in_loop_test(node):
    try:
        if isinstance(node.test, ast.Name):
            return node.test.id == 'True'
    except:
        return False

    return False