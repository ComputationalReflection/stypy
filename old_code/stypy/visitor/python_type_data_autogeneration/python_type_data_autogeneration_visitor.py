import ast

type_test_name = "type_test"


class TypeDataAutoGeneratorVisitor(ast.NodeVisitor):
    """
    This visitor is used to generate a version of the original source code that dynamically captures the types of
    variables in functions, methods and the global scope. This program can be executed and code is inserted in key
    places to capture the value of the variables at the end of the execution of any of the previously mentioned
    elements. This has several limitations:
    - If a variable changes types during the execution of a function, only the last type is captured.
    - If the program has errors at runtime, nothing is captured
    - The technique may fail with certain Python constructs.

    In general, this visitor is only used as a helper for creating Python data files that can be used to unit test
    the type inference code generation modules and classes, not being a part of the end-user functionality of stypy.
     As a tool to facilitate the development of stypy, the code is not polished at the same level as the rest of the
     code, as its only function is to generate an approximation of the types of the variables that a correct execution
     of the tested programs should return. Normally the generated table has to be modified by hand, as this technique
     is not 100% accurate.
    """

    def visit_Module(self, node):
        alias = ast.alias(name="TypeDataFileWriter", asname=None)
        import_ = ast.ImportFrom(level=0,
                                 module="stypy.code_generation.type_inference_programs.checking.type_data_file_writer",
                                 names=[alias])

        name = ast.Name(id="__file__", ctx=ast.Load())
        # attribute_module = ast.Attribute(attr="type_data_file_writer", ctx=ast.Load(),
        #                                   value=ast.Name(id="", ctx=ast.Load()))
        attribute = ast.Name(id="TypeDataFileWriter", ctx=ast.Load())  # , value=attribute_module)
        call = ast.Call(args=[name], func=attribute, keywords=[], kwargs=None, starargs=None)
        assign = ast.Assign(targets=[ast.Name(id=type_test_name, ctx=ast.Store())],
                            value=call)

        node.body.insert(0, assign)
        node.body.insert(0, import_)

        for stmt in node.body:
            self.visit(stmt)

        locals_call = ast.Call(args=[], func=ast.Name(id="globals", ctx=ast.Load()), keywords=[], kwargs=None,
                               starargs=None)
        attribute = ast.Attribute(attr="add_type_dict_for_main_context", ctx=ast.Load(),
                                  value=ast.Name(id=type_test_name, ctx=ast.Load()))

        call = ast.Call(args=[locals_call], func=attribute, keywords=[], kwargs=None, starargs=None)
        expr = ast.Expr(value=call)

        attribute_generate = ast.Attribute(attr="generate_type_data_file", ctx=ast.Load(),
                                           value=ast.Name(id=type_test_name, ctx=ast.Load()))
        call_generate = ast.Call(args=[], func=attribute_generate, keywords=[], kwargs=None, starargs=None)
        expr_final = ast.Expr(value=call_generate, ctx=ast.Load())
        node.body.append(expr)
        node.body.append(expr_final)
        return node

    def visit_FunctionDef(self, node):
        for stmt in node.body:
            self.visit(stmt)

        locals_call = ast.Call(args=[], func=ast.Name(id="locals", ctx=ast.Load()), keywords=[], kwargs=None,
                               starargs=None)
        attribute = ast.Attribute(attr="add_type_dict_for_context", ctx=ast.Load(),
                                  value=ast.Name(id=type_test_name, ctx=ast.Load()))
        call = ast.Call(args=[locals_call], func=attribute, keywords=[], kwargs=None, starargs=None)
        expr = ast.Expr(value=call)
        node.body.append(expr)

        return node

    def visit_Return(self, node):
        self.visit(node.value)

        index = ast.Index(value=ast.Num(n=0))
        locals_call = ast.Call(args=[], func=ast.Name(id="locals", ctx=ast.Load()), keywords=[], kwargs=None,
                               starargs=None)
        attribute = ast.Attribute(attr="add_type_dict_for_context", ctx=ast.Load(),
                                  value=ast.Name(id=type_test_name, ctx=ast.Load()))
        call = ast.Call(args=[locals_call], func=attribute, keywords=[], kwargs=None, starargs=None)
        tuple_ = ast.Tuple(ctx=ast.Load(), elts=[node.value, call])
        subscript = ast.Subscript(ctx=ast.Load(), slice=index, value=tuple_)
        node.value = subscript

        return node
