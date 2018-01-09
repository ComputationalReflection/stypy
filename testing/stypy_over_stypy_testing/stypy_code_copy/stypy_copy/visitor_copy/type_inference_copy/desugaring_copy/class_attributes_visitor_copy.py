from stypy_copy.code_generation_copy.type_inference_programs_copy.aux_functions_copy import *
from stypy_copy.visitor_copy.type_inference_copy.visitor_utils_copy import core_language_copy, stypy_functions_copy
import ast

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
        attr_index = node.body.index(attr)
        separator_comment = None
        comment = None

        for i in range(attr_index):
            if stypy_functions_copy.is_blank_line(node.body[i]):
                separator_comment = node.body[i]
            if stypy_functions_copy.is_src_comment(node.body[i]):
                comment = node.body[i]
                comment.value.id = comment.value.id.replace("# Assignment", "# Class-bound assignment")

        return separator_comment, comment

    def visit_ClassDef(self, node):
        class_attributes = filter(lambda element: isinstance(element, ast.Assign), node.body)

        attr_stmts = []
        for attr in class_attributes:
            separator_comment, comment = self.__extract_attribute_attached_comments(attr, node)
            if separator_comment is not None:
                node.body.remove(separator_comment)
                attr_stmts.append(separator_comment)

            if separator_comment is not None:
                node.body.remove(comment)
                attr_stmts.append(comment)

            node.body.remove(attr)

            temp_class_attr = core_language_copy.create_attribute(node.name, attr.targets[0].id)
            if len(filter(lambda class_attr: class_attr.targets[0] == attr.value, class_attributes)) == 0:
                attr_stmts.append(core_language_copy.create_Assign(temp_class_attr, attr.value))
            else:
                temp_class_value = core_language_copy.create_attribute(node.name, attr.value.id)
                attr_stmts.append(core_language_copy.create_Assign(temp_class_attr, temp_class_value))

        # Extracting all attributes from a class may leave the program in an incorrect state if all the members in the
        # class are attributes. An empty class body is an error, we add a pass node in that special case
        if len(node.body) == 0:
            node.body.append(stypy_functions_copy.create_pass_node())

        return stypy_functions_copy.flatten_lists(node, attr_stmts)
