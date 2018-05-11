import ast
import collections

from ...visitor_copy.type_inference_copy.visitor_utils_copy import stypy_functions_copy
from ...reporting_copy import print_utils_copy
from ...python_lib_copy.python_types_copy.type_inference_copy import union_type_copy
from ...type_store_copy.type_annotation_record_copy import TypeAnnotationRecord


class TypeAnnotationVisitor(ast.NodeVisitor):
    """
    This visitor is used to generate a version of the original Python source code with type annotations attached
    to each of its lines, for those lines that have a type change for any of its variables recorded. It uses the
    information stored in a TypeAnnotationRecord instance attached to the processed file to extract annotations for
    a certain line, merge them and print them in comments just before the source code line that has the variables
    of the annotations.
    """

    def __init__(self, file_name, type_store):
        file_name = file_name.replace("\\", "/")
        self.file_name = file_name
        self.type_store = type_store
        self.fcontexts = type_store.get_all_processed_function_contexts()

    @staticmethod
    def __mergue_annotations(annotations):
        """
        Picks annotations stored in a list of tuples and merge those belonging to the same variable, creating
        union types if necessary (same variable with more than one type)
        :param annotations:
        :return:
        """
        str_annotation = ""
        vars_dict = dict()
        for tuple_ in annotations:
            if not print_utils_copy.is_private_variable_name(tuple_[0]):
                if tuple_[0] not in vars_dict:
                    vars_dict[tuple_[0]] = tuple_[1]
                else:
                    vars_dict[tuple_[0]] = union_type_copy.UnionType.add(vars_dict[tuple_[0]], tuple_[1])

        for (name, type) in vars_dict.items():
            str_annotation += str(name) + ": " + print_utils_copy.get_type_str(type) + "; "

        if len(str_annotation) > 2:
            str_annotation = str_annotation[:-2]

        return str_annotation

    def __get_type_annotations(self, line):
        """
        Get the type annotations associated with a source code line of the original Python program
        :param line: Line number
        :return: str with the formatted annotations, ready to be written
        """
        str_annotation = ""
        all_annotations = []
        for fcontext in self.fcontexts:
            annotations = TypeAnnotationRecord.get_instance_for_file(self.file_name).get_annotations_for_line(line)
            if annotations is not None:
                all_annotations.extend(annotations)
                # str_annotation += self.__mergue_annotations(annotations)
                # for tuple_ in annotations:
                #     if not print_utils.is_private_variable_name(tuple_[0]):
                #         str_annotation += str(tuple_[0]) + ": " + print_utils.get_type_str(tuple_[1]) + "; "

        # if len(str_annotation) > 2:
        #     str_annotation = str_annotation[:-2]
        str_annotation = self.__mergue_annotations(all_annotations)
        return str_annotation

    def __get_type_annotations_for_function(self, fname, line):
        """
        Gets the annotations belonging to a certain function whose name is fname and is declared in the passed source
        code line, to avoid obtaining the wrong function in case there are multiple functions with the same name.
        This is used to annotate the possible types of the parameters of a function, checking all the calls that this
        function has during the program execution
        :param fname: Function name
        :param line: Source code line
        :return: str with the parameters of the functions and its annotated types
        """
        str_annotation = ""
        for fcontext in self.fcontexts:
            if fcontext.function_name == fname and fcontext.declaration_line == line:
                header_str = fcontext.get_header_str()
                if header_str not in str_annotation:
                    str_annotation += header_str + " /\ "

        if len(str_annotation) > 2:
            str_annotation = str_annotation[:-3]

        return str_annotation

    def __visit_instruction_body(self, body):
        """
        Visits all the instructions of a body, calculating its possible type annotations, turning it AST comment nodes
        and returning a list with the comment node and the original node. This way each source code line with
        annotations will appear in the generated file just below a comment with its annotations.
        :param body: Body of instructions
        :return: list
        """
        new_stmts = []

        annotations = []
        # Visit all body instructions
        for stmt in body:
            stmts = self.visit(stmt)
            if hasattr(stmt, "lineno"):
                annotations = self.__get_type_annotations(stmt.lineno)
                if not annotations == "":
                    annotations = stypy_functions_copy.create_src_comment(annotations)
                    stmts = stypy_functions_copy.flatten_lists(annotations, stmts)

            if isinstance(stmts, list):
                new_stmts.extend(stmts)
            else:
                new_stmts.append(stmts)

        return new_stmts

    """
    The rest of visit_ methods belong to those nodes that may have instruction bodies. These bodies are processed by
    the previous function so any instruction can have its possible type annotations generated. All follow the same
    coding pattern.
    """

    def generic_visit(self, node):
        if hasattr(node, 'body'):
            if isinstance(node.body, collections.Iterable):
                stmts = self.__visit_instruction_body(node.body)
            else:
                stmts = self.__visit_instruction_body([node.body])

            node.body = stmts

        if hasattr(node, 'orelse'):
            if isinstance(node.orelse, collections.Iterable):
                stmts = self.__visit_instruction_body(node.orelse)
            else:
                stmts = self.__visit_instruction_body([node.orelse])

            node.orelse = stmts

        return node

    # ######################################### MAIN MODULE #############################################

    def visit_Module(self, node):
        stmts = self.__visit_instruction_body(node.body)

        node.body = stmts
        return node

    # ######################################### FUNCTIONS #############################################

    def visit_FunctionDef(self, node):
        annotations = self.__get_type_annotations_for_function(node.name, node.lineno)
        stmts = self.__visit_instruction_body(node.body)
        node.body = stmts

        if not annotations == "":
            annotations = stypy_functions_copy.create_src_comment(annotations)
        else:
            annotations = stypy_functions_copy.create_src_comment("<Dead code detected>")

        return stypy_functions_copy.flatten_lists(annotations, node)

    def visit_If(self, node):
        stmts = self.__visit_instruction_body(node.body)
        node.body = stmts

        stmts = self.__visit_instruction_body(node.orelse)
        node.orelse = stmts

        return node

    def visit_While(self, node):
        stmts = self.__visit_instruction_body(node.body)
        node.body = stmts

        stmts = self.__visit_instruction_body(node.orelse)
        node.orelse = stmts

        return node

    def visit_For(self, node):
        stmts = self.__visit_instruction_body(node.body)
        node.body = stmts

        stmts = self.__visit_instruction_body(node.orelse)
        node.orelse = stmts

        return node

    def visit_ClassDef(self, node):
        stmts = self.__visit_instruction_body(node.body)
        node.body = stmts

        return node

    def visit_With(self, node):
        stmts = self.__visit_instruction_body(node.body)
        node.body = stmts

        return node

    def visit_TryExcept(self, node):
        stmts = self.__visit_instruction_body(node.body)
        node.body = stmts

        for handler in node.handlers:
            stmts = self.__visit_instruction_body(handler.body)
            handler.body = stmts

        stmts = self.__visit_instruction_body(node.orelse)
        node.orelse = stmts
        return node

    def visit_TryFinally(self, node):
        stmts = self.__visit_instruction_body(node.body)
        node.body = stmts

        stmts = self.__visit_instruction_body(node.finalbody)
        node.finalbody = stmts

        return node
