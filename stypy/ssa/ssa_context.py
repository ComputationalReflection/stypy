#!/usr/bin/env python
# -*- coding: utf-8 -*-
import types

from ssa_algorithm import join_ssa_branches, assign_ssa_finally_branch, join_ssa_branches_types_of_members, \
    assign_ssa_finally_branch_types_of_members
from stypy.contexts.context import Context
from stypy.reporting.localization import Localization
from stypy.types.undefined_type import UndefinedType


class SSAContext(Context):
    """
    Optimized version of the SSA algorithm with stypy. Its main advantages over its predecessor are:
    - No type cloning: Major performance boost and more simplicity
    - Orthogonal model for any Python type and type store: Any container of types are treated the same
    - Orthogonal model for one-branch SSA (if/for/while with no else), two-branch SSA (if/for/while with else) and
     unlimited-branch SSA (try-except*).

     * The finally branch is special and will be treated separately.
    """

    def __init__(self, parent_context, context_name=None, on_ssa=False):
        """
        Creates a new context with the specified parent context and a value for telling if this context is inside a
        SSA open context or not.
        :param parent_context:
        :return:
        """
        super(SSAContext, self).__init__(parent_context, context_name)

        # SSA-related information

        # Are we currently on a SSA context?
        self.on_ssa = on_ssa

        # Branches of the current SSA context (if any). Stores dicts of (name, type)
        self.ssa_branches = list()

        # Active branch (for SSA)
        self.current_branch = context_name

        # Stored types of members
        self.types_of_members = dict()

        # Types of members stored on the branches of the current SSA context (if any).
        # Stores dicts of {object, dict(name, type)}
        self.ssa_branches_types_of_members = list()

    def __get_hash(self, obj):
        """
        Obtain an adequate hash code value of the passed object
        :param obj:
        :return:
        """

        # Presence of eq and hash methods means that the instance is hashable either using its current hash method
        # or just using its memory address (if __hash__ method is not available)
        def dynamic_hash(self):
            return id(self)

        if hasattr(obj, '__eq__'):
            try:
                obj.__hash__()
            except:
                try:
                    obj.__hash__ = types.MethodType(dynamic_hash, obj)
                except:
                    pass

        return obj

    def has_member(self, localization, obj, name):
        """
        Gets the type of the specified member name for the object obj, looking on its parent context if not found
        on its own type store
        :param localization:
        :param obj:
        :param name:
        :return:
        """
        try:
            # Lookup inside our types
            self.__get_hash(obj)
            if obj in self.types_of_members:
                obj = self.types_of_members[self.__get_hash(obj)][name]
                return True
        except KeyError:
            # If not found, lookup our parent context (if any) recursively
            if self.parent_context is not None:
                return self.parent_context.has_member(Localization.get_current(), obj, name)
        except Exception as exc:
            try:
                # Lookup inside our types
                if id(obj) in self.types_of_members:
                    obj = self.types_of_members[id(obj)][name]
                    return True
            except KeyError:
                # If not found, lookup our parent context (if any) recursively
                if self.parent_context is not None:
                    return self.parent_context.has_member(Localization.get_current(), obj, name)

        return super(SSAContext, self).has_member(localization, obj, name)

    def get_type_of_member(self, localization, obj, name):
        """
        Gets the type of the specified member name for the object obj, looking on its parent context if not found
        on its own type store
        :param localization:
        :param obj:
        :param name:
        :return:
        """
        try:
            # Lookup inside our types
            return self.types_of_members[self.__get_hash(obj)][name]
        except:
            # If not found, lookup our parent context (if any) recursively
            if self.parent_context is not None:
                return self.parent_context.get_type_of_member(localization, obj, name)

        return super(SSAContext, self).get_type_of_member(localization, obj, name)

    def set_type_of_member(self, localization, obj, name, type_):
        """
        Sets the type of the specified member name for the object obj, looking on its parent context if not found
        on its own type store
        :param localization:
        :param obj:
        :param name:
        :param type_:
        :return:
        """
        if self.on_ssa:
            if self.__get_hash(obj) not in self.types_of_members:
                self.types_of_members[self.__get_hash(obj)] = dict()

            # Lookup inside our types
            self.types_of_members[self.__get_hash(obj)][name] = type_
            return None
        else:
            return super(SSAContext, self).set_type_of_member(localization, obj, name, type_)

    def del_member(self, localization, obj, name):
        """
        Deletes the type of the specified member name for the object obj, looking on its parent context if not found
        on its own type store
        :param localization:
        :param obj:
        :param name:
        :return:
        """
        try:
            if self.on_ssa:
                # Lookup inside our types
                if self.__get_hash(obj) not in self.types_of_members:
                    self.types_of_members[self.__get_hash(obj)] = dict()

                self.types_of_members[self.__get_hash(obj)][name] = UndefinedType
                return None
            else:
                # Lookup inside our types
                del self.types_of_members[self.__get_hash(obj)][name]
                return None
        except KeyError:
            # If not found, lookup our parent context (if any) recursively
            if self.parent_context is not None:
                return self.parent_context.del_member(Localization.get_current(), obj, name)

        return super(SSAContext, self).del_member(localization, obj, name)

    # ################################################ SSA HANDLING METHODS ############################################

    @staticmethod
    def create_ssa_context(parent_context, context_name):
        """
        Open a SSA context for storing variables. This means:
        - A new context with no variables is created as a child of the current one. The new context has no initial
        variables on its own.
        - This open context is the first branch of the SSA algorithm. At least a branch will be present on every SSA
        process, and an unlimited number of branches are possible
        - The new context is responsible of storing the type of the variables that are written while it is active. These
        types take precedence over the types of its parent context (if v is written inside and if, this will be the type
        of v while the if context is active)

        :param parent_context: Parent context
        :param context_name: Name of the open context
        :return:
        """
        return SSAContext(parent_context, context_name, True)

    def open_ssa_context(self, context_name):
        """
        Open a SSA context for storing variables. This means:
        - A new context with no variables is created as a child of the current one. The new context has no initial
        variables on its own.
        - This open context is the first branch of the SSA algorithm. At least a branch will be present on every SSA
        process, and an unlimited number of branches are possible
        - The new context is responsible of storing the type of the variables that are written while it is active. These
        types take precedence over the types of its parent context (if v is written inside and if, this will be the type
        of v while the if context is active)

        :param context_name: Name of the context (if needed)
        :param context_name: Name of the open context
        :return:
        """
        return SSAContext(self, context_name, True)

    def __push_current_branches(self):
        """
        Stores the current SSA branch into a SSA branch list
        :return:
        """
        self.ssa_branches.append((self.current_branch, self.types_of))
        self.ssa_branches_types_of_members.append((self.current_branch, self.types_of_members))

    def open_ssa_branch(self, branch_type):
        """
        A SSA branch is a new code branch that happens while a SSA context is active. It can be an else block, an except
        block or a finally block. Opening a branch means that:
         - The current branch is stored (stack push) for the join operation. This means that all the types written
         inside a branch will be saved and will not be available in other branches, as all the possible scenarios while
         this is used behaves like this.
         - The dict that stores the types is reset. The context behaves as normal.
         :param branch_type: Name of the open branch. Purely cosmetic except on finally branches
        :return:
        """
        self.__push_current_branches()
        self.current_branch = branch_type
        self.types_of = dict()
        self.types_of_members = dict()

        # Try else can access to try defined variables
        if branch_type == "except else":
            for branch in self.ssa_branches:
                if branch[0] == "try-except":
                    self.types_of = branch[1]
            for branch in self.ssa_branches_types_of_members:
                if branch[0] == "try-except":
                    self.types_of_members = branch[1]

    def join_ssa_context(self):
        """
        The join operation picks up all the stored branches and join the types of its variables using the SSA algoritm.
        The output of this join operation is placed back into the parent context that originated the SSA context,
        updating or adding its types if needed. This updated parent context is returned so this join process could be
        used with an arbitrary level of anidation.
        :return:
        """
        # Previous context
        context_previous = self.parent_context

        # Store the active branch as an additional branch
        self.__push_current_branches()

        # Our SSA algorithm only joins up to two branches (if/else). If more than two branches are present, we must
        # join the additional ones with the result of joining the previous ones 2 by 2.
        branch_name1, branch1 = self.ssa_branches[0]
        branch_name1, branch1_types_of_members = self.ssa_branches_types_of_members[0]

        # Basic if-no else case
        if len(self.ssa_branches) == 1:
            # Join types
            branch1 = join_ssa_branches(context_previous, branch1, dict())
            # Join types of object members
            branch1_types_of_members = join_ssa_branches_types_of_members(context_previous, branch1_types_of_members,
                                                                          dict())
        else:
            # For if-else and except cases (potentially multi-branch)
            for i in xrange(1, len(self.ssa_branches)):
                branch_name2, branch2 = self.ssa_branches[i]
                branch_name2, branch2_types_of_members = self.ssa_branches_types_of_members[i]

                # Finally branches overwrite the types of variables present in other branches, as they always execute at
                # the end
                if branch_name2 == "finally":
                    # Assign types
                    branch1 = assign_ssa_finally_branch(branch1, branch2)
                    # Assign types of members
                    branch1_types_of_members = assign_ssa_finally_branch_types_of_members(branch1_types_of_members,
                                                                                          branch2_types_of_members)
                else:
                    # Join types
                    branch1 = join_ssa_branches(context_previous, branch1, branch2)
                    # Join types of object members
                    branch1_types_of_members = join_ssa_branches_types_of_members(context_previous,
                                                                                  branch1_types_of_members,
                                                                                  branch2_types_of_members)

        for var in branch1:
            # Assign new type values to the parent context
            self.parent_context.set_type_of(Localization.get_current(), var, branch1[var])

        for obj in branch1_types_of_members:
            for att_name in branch1_types_of_members[self.__get_hash(obj)]:
                # Assign new type values for object members to the parent context
                self.parent_context.set_type_of_member(Localization.get_current(), obj, att_name,
                                                       branch1_types_of_members[self.__get_hash(obj)][att_name])

        if not isinstance(self.parent_context, SSAContext):
            self.parent_context.on_ssa = False

        self.remove_next_context()
        for key, value in self.aliases.items():
            self.parent_context.aliases[key] = value

        return self.parent_context