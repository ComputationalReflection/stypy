# stypy

stypy is a static type checker for Python. It translates each Python program into Python code that type-checks the original program. The generated code replaces each variable with a type variable, evaluating expression types instead of their values.  The generated type checker program detects type errors in different tricky Python idioms, and ensures termination.

Therefore, stypy type-checks Python programs with Python itself. Static type checking is obtained with the dynamic execution of a convergent Python type checker generated for the source program. The generated type checkers take advantage of the rich meta-programming features of the Python programming language: introspection is used to inspect the structures of objects, classes and modules, and the inheritance graph; the AST of the original program is easily obtained, transformed, compiled into the  target  type  checker,  and  executed;  recursion  can  be  detected dynamically with the use of decorators; and the types of variables, functions, classes and modules can evolve throughout the execution of the type checker

For those cases where the type checker cannot generate a type checker program from Python sources (e. g. native functions), a rule system is used, where each function is checked to receive the correct parameters and return the appropriate type.

For the following sample program:

```python
class Counter:
	count = 0
	def inc(self, value):
		self.count += value
		return self.count

obj = Counter()
sum = obj.inc(1) + obj.inc(0.2)
```

stypy produces the following type checker (passing to ```stypy_checker.py``` the program to be analyzed). The generated type checker is simplified for the sake of clarity:

```python
class Counter:
	count = int
	def inc(self, *args)
		if len(args) != 1:
			return TypeError("Wrong number of args")
		self.count = op('+', get_member(self, "count"), args[0])
		return get_member(self, "count")

ts.set("Counter", Counter)
ts.set("obj", ts.get("Counter")())
ts.set("sum", op('+', get_member(ts.get("obj"),"inc")(int),
get_member(ts.get("obj"),"inc")(float))
```

Each variable in the original program (```Counter```, ```count```, ```inc```, ```obj``` and ```sum```) represents a type variable in the generated type checker. Python facilitates this association, since most of the language entities (modules, classes, functions, types. . .) are first-class objects.

The code in the generated type checker evaluates types of expressions instead of their values. Python code is used to infer types instead of representing them as constraints or predicates.  When the inferred type of an expression is ```TypeError```, it means that it is not well typed. After executing the type checker, the collection of instantiated ```TypeError``` instances is consulted to see if the program is well typed.

Types are saved in a type store (```ts```), instead of using the Python ```globals``` and ```locals``` functions. Thereby, when the original program uses an undefined variable, the get method of ```ts``` returns a ```TypeError```. Similarly, the ```get_member``` function makes use of Python metaprogramming to check if an object, class or module provide a member, inspecting the inheritance tree. The generated type checker must never produce a runtime error, since it is aimed at type-checking the source program.

The ```inc``` method in the generated type checker checks the original ```inc``` method in the source program. If the number of parameters is not the expected one, a ```TypeError``` is returned. Otherwise, the type of an addition is computed and assigned to the count field of the particular ```Counter``` object pointed by ```self```. The ```op``` function is part of the type system that the generated type checkers use. That function returns the type of any binary Python operator, passing the types of the operands as parameters.

In our example, the second invocation to ```inc``` changes the type of ```obj.count``` from ```int``` to ```float```. Our type system considers a different type for each single object (e.g., ```obj```). The type of an object is represented with another object holding the types of its members. This consideration is especially useful for type-checking structural intercession operations, since the structure of Python objects, classes and modules can be modified at runtime.

There are cases in which the generated type checker will report ```TypeWarning``` instances instead of```TypeError```. These warnings represent those cases in which the type of a variable may change depending on the program execution flow. In these cases, certain operations might be correct or not depending on the program execution. stypy reports those cases to warn the programmer about potential runtime errors.

The stypy type checker receives two parameters:

* ```--print_ts```: Prints the resulting type store of the generated type checker
* ```--strict```: Treats TypeWarnings as TypeErrors.

stypy is currently in alpha stage (v0.2). It is currently able to process small Python programs, detecting a great variety of type errors. We provide a test battery containing a considerable amount of common type errors to illustrate the type errors stypy detects..

Main features of the new 0.2 release include:

- The implementation of the stypy global module cache (SGMC), that stores each generated type-checker program. The generated files use the same timestamp as the source ones. Therefore, when the source file changes, the differences between timestamps are detected and the type checker file is regenerated. This implies a major performance upgrade for stypy.
- Type checking of the ```numpy```, ```scipy``` and ```matplotlib``` libraries. Full support for ```scipy``` and ```matplotlib``` is currently a work in progress.



# CHANGELOG:

[10/01/2018] Fix for the "for line in file:" idiom to consider file objects as iterables.
[11/01/2018] Fix for the error produced when setting the type of a container when the index is a union type


Copyright (c) [2016-2018] [Francisco Ortín Soler, José Manuel Redondo López, Baltasar García Pérez-Schofield]