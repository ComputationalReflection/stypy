import os

from stypy.sgmc.sgmc_main import SGMC
from stypy.type_inference_programs.aux_functions import *
from stypy.visitor.type_inference.visitor_utils import stypy_functions


class RelativeImportVisitor(ast.NodeTransformer):
    """
    This transformer ensures that relative imports are transformed in to non-relative equivalent forms
    """

    def __init__(self, file_path):
        self.file_path = file_path

    def visit_ImportFrom(self, node):
        assign_node = []
        if hasattr(node, 'level'):
            if node.level > 0:
                module_name = SGMC.get_sgmc_full_module_name(os.path.dirname(self.file_path))

                if module_name.startswith("."):
                    module_name = module_name[1:]

                if module_name.startswith("site_packages."):
                    module_name = module_name[len("site_packages."):]

                if node.level - 1 > 0:
                    parent_module_name = SGMC.get_parent_module_name(module_name, node.level - 1)
                else:
                    parent_module_name = module_name

                if node.module is not None:
                    node.module = parent_module_name + "." + node.module
                else:
                    node.module = parent_module_name

                node.level = 0

        return stypy_functions.flatten_lists(node, assign_node)
