#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stypy.errors.type_error import StypyTypeError
from stypy.reporting.localization import Localization
from stypy.types.undefined_type import UndefinedType
from stypy.types.union_type import UnionType


# ######################################### SSA WITH TYPES ##########################################################

def assign_ssa_finally_branch(branch_rest, branch_finally):
    """
    Assign all the types contained on the dictionary branch2 to the dictionary branch1, overwriting them if the same
    name exist in both branches. This is needed for the implementation of the finally branch, that do not follow the
    same behaviour as the other branches when implementing the SSA algorithm.
    :param branch_rest:
    :param branch_finally:
    :return:
    """
    for name in branch_finally:
        if isinstance(branch_finally[name], UnionType):
            types = branch_finally[name].get_types()
            types_without_undefined = filter(lambda t: t is not UndefinedType, types)
            if len(types_without_undefined) < len(types):  # There were undefined types
                union_without_undefined = UnionType.create_from_type_list(types_without_undefined)
                branch_rest[name] = UnionType.add(branch_rest[name], union_without_undefined)
                continue

        branch_rest[name] = branch_finally[name]
    return branch_rest


def join_ssa_branches(previous_context, branch1, branch2):
    """
    Implementation of the SSA joining algorithm for two SSA branches
    :param previous_context:
    :param branch1:
    :param branch2:
    :return:
    """
    type_dict = dict()

    # NOTE: Explanations in comments are done with an if/else control structure. Other structures are the same.

    # Proceed with the first branch (if)

    # For any variable stored in the first branch (note that we only deal with variables that change its type in the
    # active branch, not all the possible variables accessible within the SSA context. This is also a major speedup
    # over our previous version.
    for var_name in branch1:
        if var_name in branch2:
            # Variable defined in if and else body: Joins both types
            type_dict[var_name] = UnionType.add(branch1[var_name],
                                                branch2[var_name])
        else:
            # Variable defined in if and in the previous context: Joins the previous type and the if one
            if var_name in previous_context:
                type_dict[var_name] = UnionType.add(previous_context[var_name],
                                                    branch1[var_name])
            else:
                # Variable defined in if body, but did not exist in the previous context: Joins the if type with an
                # undefined type, as the if could not be executed
                type_dict[var_name] = UnionType.add(branch1[var_name], UndefinedType)

    # Now proceed with the second branch (else). If no else is present and empty dict is passed, and therefore no
    # processing is done.

    for var_name in branch2:
        if var_name in branch1:
            continue  # Already processed (above)
        else:
            # Variable defined only in the else body and in the previous context: Treat equally to the if branch
            # counterpart
            if var_name in previous_context:
                type_dict[var_name] = UnionType.add(previous_context[var_name],
                                                    branch2[var_name])
            else:
                # Variable defined in else body, but did not existed in the previous context: Same treatment as their
                # if branch counterpart
                type_dict[var_name] = UnionType.add(branch2[var_name], UndefinedType)

    # type_store_previous does not need to be iterated because it is included in the if and else stores
    return type_dict


# ######################################### SSA WITH TYPES OF MEMBERS #################################################

def assign_ssa_finally_branch_types_of_members_for_object(obj, branch_rest, branch_finally):
    """
    Assign all the types contained on the dictionary branch2 to the dictionary branch1, overwriting them if the same
    name exist in both branches. This is needed for the implementation of the finally branch, that do not follow the
    same behaviour as the other branches when implementing the SSA algorithm.
    :param obj:
    :param branch_rest:
    :param branch_finally:
    :return:
    """
    for name in branch_finally:
        if isinstance(branch_finally[name], UnionType):
            types = branch_finally[name].get_types()
            types_without_undefined = filter(lambda t: t is not UndefinedType, types)
            if len(types_without_undefined) < len(types):  # There were undefined types
                union_without_undefined = UnionType.create_from_type_list(types_without_undefined)
                branch_rest[name] = UnionType.add(branch_rest[name], union_without_undefined)
                continue

        # if obj not in branch_rest:
        #     branch_rest[obj] = dict()

        branch_rest[name] = branch_finally[name]
    return branch_rest


def assign_ssa_finally_branch_types_of_members(branch_rest, branch_finally):
    type_dict = dict()

    # NOTE: Explanations in comments are done with an if/else control structure. Other structures are the same.

    # Proceed with the first branch (if)

    # For any variable stored in the first branch (note that we only deal with variables that change its type in the
    # active branch, not all the possible variables accessible within the SSA context. This is also a major speedup
    # over our previous version.
    for obj in branch_rest:
        if obj in branch_finally:
            branch_finally_obj = branch_finally[obj]
        else:
            branch_finally_obj = dict()

        # Variable defined in if and else body: Joins both types
        type_dict[obj] = assign_ssa_finally_branch_types_of_members_for_object(obj, branch_rest[obj],
                                                                               branch_finally_obj)

    # Now proceed with the second branch (else). If no else is present and empty dict is passed, and therefore no
    # processing is done.

    for obj in branch_finally:
        if obj in branch_rest:
            continue  # Already processed (above)
        else:
            # Variable defined only in the else body and in the previous context: Treat equally to the if branch
            # counterpart
            type_dict[obj] = assign_ssa_finally_branch_types_of_members_for_object(obj, dict(),
                                                                                   branch_finally[obj])

    # type_store_previous does not need to be iterated because it is included in the if and else stores
    return type_dict


def __join_ssa_branches_types_of_members_for_object(obj, previous_context, branch1, branch2):
    """
    Implementation of the SSA joining algorithm for two SSA branches, handling types of members
    :param previous_context:
    :param branch1: dict(obj, dict(name, type))
    :param branch2:
    :return:
    """
    type_dict = dict()

    # NOTE: Explanations in comments are done with an if/else control structure. Other structures are the same.

    # Proceed with the first branch (if)

    # For any variable stored in the first branch (note that we only deal with variables that change its type in the
    # active branch, not all the possible variables accessible within the SSA context. This is also a major speedup
    # over our previous version.
    for var_name in branch1:
        if var_name in branch2:
            # Variable defined in if and else body: Joins both types
            type_dict[var_name] = UnionType.add(branch1[var_name],
                                                branch2[var_name])
        else:
            # Variable defined in if and in the previous context: Joins the previous type and the if one
            previous_type = previous_context.get_type_of_member(Localization.get_current(), obj, var_name)
            if not isinstance(previous_type, StypyTypeError):
                type_dict[var_name] = UnionType.add(previous_type,
                                                    branch1[var_name])
            else:
                # Variable defined in if body, but did not exist in the previous context: Joins the if type with an
                # undefined type, as the if could not be executed
                type_dict[var_name] = UnionType.add(branch1[var_name], UndefinedType)
                StypyTypeError.remove_error_msg(previous_type)

    # Now proceed with the second branch (else). If no else is present and empty dict is passed, and therefore no
    # processing is done.

    for var_name in branch2:
        if var_name in branch1:
            continue  # Already processed (above)
        else:
            # Variable defined only in the else body and in the previous context: Treat equally to the if branch
            # counterpart
            previous_type = previous_context.get_type_of_member(Localization.get_current(), obj, var_name)
            if not isinstance(previous_type, StypyTypeError):
                type_dict[var_name] = UnionType.add(previous_type,
                                                    branch2[var_name])
            else:
                # Variable defined in else body, but did not existed in the previous context: Same treatment as their
                # if branch counterpart
                type_dict[var_name] = UnionType.add(branch2[var_name], UndefinedType)
                StypyTypeError.remove_error_msg(previous_type)

    # type_store_previous does not need to be iterated because it is included in the if and else stores
    return type_dict


def join_ssa_branches_types_of_members(previous_context, branch1, branch2):
    """
    Implementation of the SSA joining algorithm for two SSA branches, handling types of members
    :param previous_context:
    :param branch1: dict(obj, dict(name, type))
    :param branch2:
    :return:
    """
    type_dict = dict()

    # NOTE: Explanations in comments are done with an if/else control structure. Other structures are the same.

    # Proceed with the first branch (if)

    # For any variable stored in the first branch (note that we only deal with variables that change its type in the
    # active branch, not all the possible variables accessible within the SSA context. This is also a major speedup
    # over our previous version.
    for obj in branch1:
        if obj in branch2:
            branch2_obj = branch2[obj]
        else:
            branch2_obj = dict()

        # Variable defined in if and else body: Joins both types
        type_dict[obj] = __join_ssa_branches_types_of_members_for_object(obj, previous_context, branch1[obj],
                                                                         branch2_obj)

    # Now proceed with the second branch (else). If no else is present and empty dict is passed, and therefore no
    # processing is done.

    for obj in branch2:
        if obj in branch1:
            continue  # Already processed (above)
        else:
            # Variable defined only in the else body and in the previous context: Treat equally to the if branch
            # counterpart
            type_dict[obj] = __join_ssa_branches_types_of_members_for_object(obj, previous_context, dict(),
                                                                             branch2[obj])

    # type_store_previous does not need to be iterated because it is included in the if and else stores
    return type_dict
