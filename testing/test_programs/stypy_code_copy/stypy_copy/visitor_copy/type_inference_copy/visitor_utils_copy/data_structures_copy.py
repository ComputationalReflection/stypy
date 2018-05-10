from collections import Iterable
import ast

import core_language_copy

"""
Helper functions to create data structures related nodes in the type inference program AST tree
"""


# ######################################## COLLECTIONS HANDLING FUNCTIONS #############################################


def is_iterable(obj):
    """
    Determines if the parameter is iterable
    :param obj: Any instance
    :return: Boolean value
    """
    return isinstance(obj, Iterable)


def create_list(contents):
    list_node = ast.List(ctx=ast.Load())
    list_node.elts = contents

    return list_node


def create_keyword_dict(keywords):
    dict_node = ast.Dict(ctx=ast.Load(), keys=[], values=[])

    if keywords is not None:
        for elem in keywords:
            dict_node.keys.append(core_language.create_str(elem))
            dict_node.values.append(keywords[elem])

    return dict_node
