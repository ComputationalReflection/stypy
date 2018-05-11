from ..type_store_copy.typestore_copy import TypeStore
from ..python_lib_copy.python_types_copy.type_inference_copy import union_type_copy
from ..python_lib_copy.python_types_copy.type_inference_copy.undefined_type_copy import UndefinedType

"""
Implementation of the SSA algorithm to calculate types of variables when dealing with branches in source code (ifs,
loops, ...)
"""


# TODO: Remove?
# from stypy.type_store.function_context import FunctionContext

# ############## SSA FOR CLAUSES WITH AN OPTIONAL ELSE BRANCH (IF, FOR, WHILE...) ###############
#
# def __join_annotations(function_context_a, function_context_b):
#     # Curiously, declaring a "global" in one of the branches avoids the potential unreferenced variable error for all
# of them, so
#     # we simply add the variables of both branches.
#     a_annotations = function_context_a.annotation_record
#     if (function_context_b is None):
#         b_annotations = dict()
#     else:
#         b_annotations = function_context_b.annotation_record.annotation_dict
#
#     for (line, annotations) in b_annotations.items():
#         for annotation in annotations:
#             a_annotations.annotate_type(line, annotation[2], annotation[0], annotation[1])
#
#     return a_annotations


def __join_globals(function_context_if, function_context_else):
    """
    Join the global variables placed in two function contexts
    :param function_context_if: Function context
    :param function_context_else: Function context
    :return: The first function context with the globals of both of them
    """
    # Curiously, declaring a "global" in one of the branches avoids the potential unreferenced variable error for all
    # of them, so we simply add the variables of both branches.
    if_globals = function_context_if.global_vars
    if function_context_else is None:
        else_globals = []
    else:
        else_globals = function_context_else.global_vars

    for var in else_globals:
        if var not in if_globals:
            if_globals.append(var)

    return if_globals


def __ssa_join_with_else_function_context(function_context_previous, function_context_if, function_context_else):
    """
    Helper function of the SSA implementation of an if-else structure, used with each function context in the type
    store
    :param function_context_previous: Function context
    :param function_context_if: Function context
    :param function_context_else: Function context
    :return: A dictionary with names of variables and its joined types
    """
    type_dict = {}

    if function_context_else is None:
        function_context_else = []  # Only the if branch is present

    for var_name in function_context_if:
        if var_name in function_context_else:
            # Variable defined in if and else body
            type_dict[var_name] = union_type_copy.UnionType.add(function_context_if[var_name],
                                                           function_context_else[var_name])
        else:
            # Variable defined in if and in the previous context
            if var_name in function_context_previous:
                # Variable defined in if body (the previous type is then considered)
                type_dict[var_name] = union_type_copy.UnionType.add(function_context_previous[var_name],
                                                               function_context_if[var_name])
            else:
                # Variable defined in if body, but did not existed in the previous type store (it could be not defined)
                type_dict[var_name] = union_type_copy.UnionType.add(function_context_if[var_name], UndefinedType())

    for var_name in function_context_else:
        if var_name in function_context_if:
            continue  # Already processed (above)
        else:
            # Variable defined in the else body, but not in the if body
            if var_name in function_context_previous:
                # Variable defined in else (the previous one is considered)
                type_dict[var_name] = union_type_copy.UnionType.add(function_context_previous[var_name],
                                                               function_context_else[var_name])
            else:
                # Variable defined in else body, but did not existed in the previous type store (it could be not
                # defined)
                type_dict[var_name] = union_type_copy.UnionType.add(function_context_else[var_name], UndefinedType())

    # type_store_previous does not need to be iterated because it is included in the if and else stores
    return type_dict


def ssa_join_with_else_branch(type_store_previous, type_store_if, type_store_else):
    """
    Implements the SSA algorithm with the type stores of an if-else structure
    :param type_store_previous: Type store
    :param type_store_if: Function context
    :param type_store_else:
    :return:
    """
    # Join the variables of the previous, the if and the else branches type stores into a single dict
    joined_type_store = TypeStore(type_store_previous.program_name)
    joined_type_store.last_function_contexts = type_store_previous.last_function_contexts
    joined_type_store.context_stack = []  # Need to empty it, as a default function context is created when initializing
    for i in range(len(type_store_previous.context_stack)):
        # Only an if branch?
        if type_store_else is None:
            function_context_else = None
        else:
            function_context_else = type_store_else[i]

        joined_context_dict = __ssa_join_with_else_function_context(type_store_previous[i], type_store_if[i],
                                                                    function_context_else)

        # joined_f_context = FunctionContext(type_store_previous[i].function_name)
        joined_f_context = type_store_previous[i].copy()
        joined_f_context.types_of = joined_context_dict
        joined_f_context.global_vars = __join_globals(type_store_if[i], function_context_else)
        joined_f_context.annotation_record = type_store_if[
            i].annotation_record  # __join_annotations(type_store_if[i], function_context_else[i])

        joined_type_store.context_stack.append(joined_f_context)

    return joined_type_store


# ############## SSA FOR EXCEPTION SENTENCES ###############

def __join_except_branches_function_context(function_context_previous, function_context_new):
    """
    Helper function to join variables of function contexts that belong to different except
    blocks
    :param function_context_previous: Function context
    :param function_context_new: Function context
    :return: A dictionary with names of variables and its joined types
    """
    type_dict = {}

    for var_name in function_context_previous:
        if var_name in function_context_new:
            # Variable defined in both function contexts
            type_dict[var_name] = union_type_copy.UnionType.add(function_context_previous[var_name],
                                                           function_context_new[var_name])
        else:
            # Variable defined in previous but not on new function context
            type_dict[var_name] = function_context_previous[var_name]

    for var_name in function_context_new:
        if var_name in function_context_previous:
            # Variable defined in both function contexts
            type_dict[var_name] = union_type_copy.UnionType.add(function_context_new[var_name],
                                                           function_context_previous[var_name])
        else:
            # Variable defined in new but not on previous function context
            type_dict[var_name] = function_context_new[var_name]

    # type_store_previous does not need to be iterated because it is included in the if and else stores
    return type_dict


def __join_except_branches(type_store_previous, type_store_new):
    """
    SSA algorithm to join type stores of different except branches
    :param type_store_previous: Type store
    :param type_store_new: Type store
    :return:
    """
    joined_type_store = TypeStore(type_store_previous.program_name)
    joined_type_store.last_function_contexts = type_store_previous.last_function_contexts
    joined_type_store.context_stack = []  # Need to empty it, as a default function context is created when initializing
    for i in range(len(type_store_previous.context_stack)):
        joined_context_dict = __join_except_branches_function_context(type_store_previous[i], type_store_new[i])
        # joined_f_context = FunctionContext(type_store_previous[i].function_name)
        joined_f_context = type_store_previous[i].copy()
        joined_f_context.types_of = joined_context_dict
        joined_f_context.global_vars = __join_globals(type_store_previous[i], type_store_new[i])
        joined_f_context.annotation_record = type_store_previous[
            i].annotation_record  # __join_annotations(type_store_previous[i], type_store_new[i])

        joined_type_store.context_stack.append(joined_f_context)

    return joined_type_store


def __join_finally_function_context(function_context_previous, function_context_finally):
    """
    Join the variables of a function context on a finally block with a function context of the joined type store
     of all the except branches in an exception clause
    :param function_context_previous: Function context
    :param function_context_finally: Function context
    :return: A dictionary with names of variables and its joined types
    """
    type_dict = {}

    for var_name in function_context_previous:
        if var_name in function_context_finally:
            # Variable defined in both function contexts
            type_dict[var_name] = function_context_finally[var_name]
        else:
            # Variable defined in previous but not on new function context
            pass

    for var_name in function_context_finally:
        if var_name in function_context_previous:
            # Variable defined in both function contexts
            pass  # Already covered
        else:
            # Variable defined in new but not on previous function context
            type_dict[var_name] = function_context_finally[var_name]

    # type_store_previous does not need to be iterated because it is included in the if and else stores
    return type_dict


def __join_finally_branch(type_store_exception_block, type_store_finally):
    """
    Join the type stores of a finally branch and the joined type store of all except branches in a exception handling
     block
    :param type_store_exception_block: Type store
    :param type_store_finally: Type store
    :return:
    """
    joined_type_store = TypeStore(type_store_exception_block.program_name)
    joined_type_store.last_function_contexts = type_store_exception_block.last_function_contexts
    joined_type_store.context_stack = []  # Need to empty it, as a default function context is created when initializing
    for i in range(len(type_store_exception_block.context_stack)):
        joined_context_dict = __join_finally_function_context(type_store_exception_block[i], type_store_finally[i])
        # joined_f_context = FunctionContext(type_store_exception_block[i].function_name)
        joined_f_context = type_store_exception_block[i].copy()
        joined_f_context.types_of = joined_context_dict
        joined_f_context.global_vars = __join_globals(type_store_exception_block[i], type_store_finally[i])
        joined_f_context.annotation_record = type_store_exception_block[
            i].annotation_record  # __join_annotations(type_store_exception_block[i], type_store_finally[i])

        joined_type_store.context_stack.append(joined_f_context)

    return joined_type_store


def __join_try_except_function_context(function_context_previous, function_context_try, function_context_except):
    """
    Implements the SSA algorithm in try-except blocks, dealing with function contexts.

    :param function_context_previous: Function context
    :param function_context_try: Function context
    :param function_context_except: Function context
    :return: A dictionary with names of variables and its joined types
    """
    type_dict = {}

    for var_name in function_context_try:
        if var_name in function_context_except:
            # Variable defined in if and else body
            type_dict[var_name] = union_type_copy.UnionType.add(function_context_try[var_name],
                                                           function_context_except[var_name])
            if var_name not in function_context_previous:
                type_dict[var_name] = union_type_copy.UnionType.add(type_dict[var_name], UndefinedType())
        else:
            # Variable defined in if but not else body
            if var_name in function_context_previous:
                # Variable defined in if body (the previous type is then considered)
                type_dict[var_name] = union_type_copy.UnionType.add(function_context_previous[var_name],
                                                               function_context_try[var_name])
            else:
                # Variable defined in if body, but did not existed in the previous type store (it could be not defined)
                type_dict[var_name] = union_type_copy.UnionType.add(function_context_try[var_name], UndefinedType())

    for var_name in function_context_except:
        if var_name in function_context_try:
            continue  # Already processed (above)
        else:
            # Variable defined in the else body, but not in the if body
            if var_name in function_context_previous:
                # Variable defined in else (the previous one is considered)
                type_dict[var_name] = union_type_copy.UnionType.add(function_context_previous[var_name],
                                                               function_context_except[var_name])
            else:
                # Variable defined in else body, but did not existed in the previous type store (it could be not
                # defined)
                type_dict[var_name] = union_type_copy.UnionType.add(function_context_except[var_name], UndefinedType())

    # type_store_previous does not need to be iterated because it is included in the if and else stores
    return type_dict


def __join__try_except(type_store_previous, type_store_posttry, type_store_excepts):
    """
    SSA Algotihm implementation for type stores in a try-except block
    :param type_store_previous: Type store
    :param type_store_posttry: Type store
    :param type_store_excepts: Type store
    :return:
    """
    joined_type_store = TypeStore(type_store_previous.program_name)
    joined_type_store.last_function_contexts = type_store_previous.last_function_contexts
    joined_type_store.context_stack = []  # Need to empty it, as a default function context is created when initializing
    for i in range(len(type_store_previous.context_stack)):
        joined_context_dict = __join_try_except_function_context(type_store_previous[i], type_store_posttry[i],
                                                                 type_store_excepts[i])
        # joined_f_context = FunctionContext(type_store_previous[i].function_name)
        joined_f_context = type_store_previous[i].copy()
        joined_f_context.types_of = joined_context_dict
        joined_f_context.global_vars = __join_globals(type_store_posttry[i], type_store_excepts[i])
        joined_f_context.annotation_record = type_store_posttry[
            i].annotation_record  # __join_annotations(type_store_posttry[i], type_store_excepts[i])

        joined_type_store.context_stack.append(joined_f_context)

    return joined_type_store


def join_exception_block(type_store_pretry, type_store_posttry, type_store_finally=None, *type_store_except_branches):
    """
    Implements the SSA algorithm for a full try-except-finally block, calling previous function
    :param type_store_pretry: Type store
    :param type_store_posttry: Type store
    :param type_store_finally: Type store
    :param type_store_except_branches: Type store
    :return:
    """
    # Join the variables of the previous, the if and the else branches type stores into a single dict
    # types_dict = __join_if_else_function_context(type_store_previous, type_store_if, type_store_else)

    joined_type_store = TypeStore(type_store_pretry.program_name)
    joined_type_store.last_function_contexts = type_store_pretry.last_function_contexts
    joined_type_store.context_stack = []  # Need to empty it, as a default function context is created when initializing

    # Process all except branches to leave a single type store. else branch is treated as an additional except branch.
    if len(type_store_except_branches) == 1:
        type_store_excepts = type_store_except_branches[0]
    else:
        cont = 1
        type_store_excepts = type_store_except_branches[0]
        while cont < len(type_store_except_branches):
            type_store_excepts = __join_except_branches(type_store_excepts, type_store_except_branches[cont])
            cont += 1

    # Join the pre exception block type store with the try branch and the union of all the except branches
    joined_context_dict = __join__try_except(type_store_pretry, type_store_posttry,
                                             type_store_excepts)

    # Finally is special because it overwrites the type of already defined variables
    if type_store_finally is not None:
        joined_context_dict = __join_finally_branch(joined_context_dict, type_store_finally)

    return joined_context_dict
