#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stypy.type_inference_programs.aux_functions import *
from stypy.visitor.type_inference.visitor_utils import core_language, stypy_functions


class ClassAttributesVisitor(ast.NodeTransformer):
    """
    This desugaring visitor converts class-bound attributes such as:

    class Example:
        att = "hello"
        <rest of the members that are not attributes>

    into this equivalent form:

    class Example:
        <rest of the members that are not attributes>

    Example.att = "hello"

    The first form cannot be properly processed by stypy due to limitations in the way they are transformed into AST
    nodes. The second form is completely processable using the same assignment processing code we already have.

    """

    @staticmethod
    def __extract_attribute_attached_comments(attr, node):
        """
        Extract the comments attached to a class attribute
        :param attr:
        :param node:
        :return:
        """
        attr_index = node.body.index(attr)
        separator_comment = None
        comment = None

        for i in xrange(attr_index):
            if stypy_functions.is_blank_line(node.body[i]):
                separator_comment = node.body[i]
            if stypy_functions.is_src_comment(node.body[i]):
                comment = node.body[i]
                comment.value.id = comment.value.id.replace("# Assignment", "# Class-bound assignment")

        return separator_comment, comment

    def __init__(self):
        """
        Initialize the visitor
        """
        self.transform_names = False
        self.class_attributes = None
        self.class_name = None

    @staticmethod
    def all_lines_are_comments(node_body):
        """
        Determines if all lines in an instruction body are comments
        :param node_body:
        :return:
        """
        for line in node_body:
            if not (stypy_functions.is_blank_line(line) or (stypy_functions.is_src_comment(line))):
                return False

        return True

    def value_is_a_class_attribute(self, attr):
        """
        Determines if a certain attribute value is a class attribute (identifies class attributes)
        :param attr:
        :return:
        """
        attr_name = None
        if type(attr.value) is ast.Name:
            attr_name = attr.value.id
        if type(attr.value) is ast.Call:
            if hasattr(attr.value.func, 'value'):
                if type(attr.value.func.value) is ast.Name:
                    attr_name = attr.value.func.value.id

        if attr_name is None:
            return False

        for attribute in filter(lambda element: isinstance(element, ast.Assign), self.class_attributes):
            if hasattr(attribute.targets[0], 'id'):
                if attribute.targets[0].id == attr_name:
                    return True
        return False

    def name_is_a_class_member(self, name):
        """
        Determines if a certain name is a class member (identifies class methods)
        :param name:
        :return:
        """
        for attribute in filter(lambda element: isinstance(element, ast.Assign), self.class_attributes):
            if hasattr(attribute.targets[0], 'id'):
                if attribute.targets[0].id == name:
                    return attribute

        for method in self.class_method:
            if hasattr(method, 'name'):
                if method.name == name:
                    return method

        return None

    def visit_ClassDef(self, node):
        """
        Transform class definitions to put its attribute initializers outside of the class definition
        :param node:
        :return:
        """
        self.class_attributes = filter(lambda element: isinstance(element, ast.Assign) or
                                                       isinstance(element, ast.If) or
                                                       (isinstance(element, ast.Expr) and
                                                        type(element.value) is ast.Call), node.body)

        self.class_method = filter(lambda element: isinstance(element, ast.FunctionDef), node.body)
        self.class_name = node.name

        attr_stmts = []
        for attr in self.class_attributes:
            separator_comment, comment = self.__extract_attribute_attached_comments(attr, node)
            if separator_comment is not None:
                node.body.remove(separator_comment)
                attr_stmts.append(separator_comment)

            if separator_comment is not None:
                node.body.remove(comment)
                attr_stmts.append(comment)

            node.body.remove(attr)

            if type(attr) is not ast.Assign:
                self.transform_names = True
                attr_stmts.append(self.visit(attr))
                self.transform_names = False
                continue

            if type(attr.targets[0]) is ast.Name:
                temp_class_attr = core_language.create_attribute(node.name, attr.targets[0].id)
            else:
                try:
                    if type(attr.targets[0]) is ast.Subscript:
                        temp_class_attr = core_language.create_nested_attribute(node.name, attr.targets[0].value)
                    else:
                        temp_class_attr = core_language.create_nested_attribute(node.name, attr.targets[0])
                except Exception as ex:
                    print ex

            if not self.value_is_a_class_attribute(attr):
                self.transform_names = True
                value_var = self.visit(attr.value)
                self.transform_names = False
                attr_stmts.append(core_language.create_Assign(temp_class_attr, value_var))
            else:
                if type(attr.value) is ast.Call:
                    func_name = core_language.create_nested_attribute(node.name, attr.value.func)
                    attr.value.func = func_name
                    temp_class_value = attr.value
                else:
                    temp_class_value = core_language.create_nested_attribute(node.name, attr.value)

                attr_stmts.append(core_language.create_Assign(temp_class_attr, temp_class_value))

        # Extracting all attributes from a class may leave the program in an incorrect state if all the members in the
        # class are attributes. An empty class body is an error, we add a pass node in that special case
        if len(node.body) == 0 or ClassAttributesVisitor.all_lines_are_comments(node.body):
            node.body.append(stypy_functions.create_pass_node())

        return stypy_functions.flatten_lists(node, attr_stmts)

    def process_node(self, node):
        """
        Processes call nodes
        :param node:
        :return:
        """
        node.func = self.visit(node.func)
        new_args = []
        for arg in node.args:
            new_args.append(self.visit(arg))
        node.args = new_args

        if hasattr(node, 'starargs'):
            if node.starargs is not None:
                node.starargs = self.visit(node.starargs)

        if hasattr(node, 'kwargs'):
            if node.kwargs is not None:
                node.kwargs = self.visit(node.kwargs)

        return node

    def visit_Call(self, node):
        """
        Processes calls done to initialize attributes
        :param node:
        :return:
        """
        if not self.transform_names:
            return self.process_node(node)

        else:
            if type(node.func) is not ast.Name:
                return self.process_node(node)

            member = self.name_is_a_class_member(node.func.id)
            if member is None:
                return self.process_node(node)
            if type(member) is not ast.FunctionDef:
                return self.process_node(node)

            # Special case: This method is really a function defined in the class with internal usage only.
            # Turn it to static to avoid unbound method call errors
            if not hasattr(member, 'decorator_list'):
                member.decorator_list = []

            found = False
            for dec in member.decorator_list:
                if type(dec) is ast.Name:
                    if dec.id == "staticmethod":
                        found = True

            if not found:
                member.decorator_list = [core_language.create_Name("staticmethod")] + member.decorator_list

            return self.process_node(node)

    def visit_Name(self, node):
        """
        Processes names used to initialize attributes
        :param node:
        :return:
        """
        if not self.transform_names:
            return node

        if self.name_is_a_class_member(node.id) is not None:
            # Param names are not changed
            if hasattr(node, 'context'):
                if type(node.context) is not ast.Param:
                    return node

            return core_language.create_attribute(self.class_name, node.id)

        return node
