# import ast
# from stypy.visitor.python_src_print.python_src_print_visitor import dump_ast
# from stypy.visitor.type_inference.type_inference_visitor import TypeInferenceGeneratorVisitor
# from stypy.visitor.python_src_generation.python_src_generator_visitor import PythonSrcGeneratorVisitor

# import numpy.core.numerictypes
# import types
#
#
#
# if __name__ == '__main__':
#     try:
#         m = 3
#     except:
#         print (m)
#     else:
#         print ("Else: " + str(m))
#
#
#         # a = "hi "
#     # b = "yo"
#     # eval_str = "a + b"
#     # r = ast.parse(eval_str)
#     # r2 = TypeInferenceGeneratorVisitor("").visit(r)
#     # ast.fix_missing_locations(r2)
#     # #print dump_ast(r2)
#     # ti_code = PythonSrcGeneratorVisitor(r2).generate_code()
#     # print ti_code
#     # exec(ti_code)
#     # print eval("result_add_3")
#
#     # kwargs = {
#     #     'constant_values': [(1, 1.0),  (2, 2.0), (3, 3.0)]
#     # }
#     # pad_width = [('a', 'a'), ('b', 'b'), ('c', 'c')]
#     #
#     # for axis, ((pad_before, pad_after), (before_val, after_val)) in enumerate(
#     #         zip(pad_width, kwargs['constant_values'])):
#     #     print axis
#     #     print pad_before
#     #     print pad_after
#     #     print before_val
#     #     print after_val
#     #import pyd_wrapper
#
# #     template = """
# # import types
# # import sys
# # import {0}
# #
# # class {1}(types.ModuleType):
# #     def __init__(self):
# #         object.__setattr__(self, 'module_obj', {0})
# #
# #     def __getattr__(self, name):
# #         return getattr(self.module_obj, name)
# #
# #     def __setattr__(self, name, value):
# #         return setattr(self.module_obj, name, value)
# #     """
# #
# #     exec(template.format('numpy.core.multiarray', 'multiarray_wrapper'))
# #     x = multiarray_wrapper()
# #
# #
# #     print x.set_typeDict
# #
# #     import pyd_wrapper
# #
# #     x = pyd_wrapper.ModuleWrapper('numpy.core.multiarray')
# #     print dir(x)
# #     print x.__dict__
# #     print x.__class__
#     # actualiza setattr
#     # mete delattr


class A:
    client_class = None

    def __init__(self):
        self.client = self.client_class(self)

a = A()

print dir(a)