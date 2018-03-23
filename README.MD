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




# Changelog:

- [09/01/2018] Release of the new v0.2 alpha version of stypy to the Github public repository
- [10/01/2018] Fix for the "for line in file:" idiom to consider file objects as iterables.
- [11/01/2018] Fix for the error produced when setting the type of a container when the index is a union type
- [12/01/2018] In some particular cases, union types of int and float were reduced to simple float types. This has been fixed.
               When creating comprehensions from strings the inner type of the comprehension (str) was not correctly calculated. This has been fixed.
               Error reports of invalid calls now specify both the actual parameters passed and the correct expected ones. This is specially useful to analyze type warnings.
- [15/01/2018] Added the 'adatron' and 'ac_encode' programs from the shedskin benchmark suite to the test battery. Solved type inference problems when code perform calls to the file.read() and file.write() native methods.
- [16/01/2018] Added the 'amaze' program from the shedskin benchmark suite to the test battery. Type warnings now indicate the operation that caused its triggering. This is useful to identify which operation triggered a warning when a line have multiple ones (such as aritmethic expressions) and several of them may trigger a warning.
- [17/01/2018] Classes inheriting from Exception that do not define a constructor no longer report an error with their argument number. Using raw_input native function no longer returns a dynamic type. Using the method readlines() from the file native type now returns the proper return type.
- [18/01/2018] Classes that inheriting from other classes but not defining a custom constructor now properly call its parent constructor. Tuples of base Python types are considered equal if they contain the same base types, no matter its order. Added the 'ant', 'bh' and 'block' programs from the shedskin benchmark suite to the test battery. The builtin function 'sum' now not only infers types with integers.
- [19/01/2018] Classes that defined the __getitem__ and __setitem__ special methods now are treated like proper containers. Comparisons between types now properly return bool types on all occasions. RecursionTypes (results from invoking a function recursively) are no longer included in union types. Added the 'brainfuck', 'chaos', 'chess', 'dijkstra' and 'dijkstra2' programs from the shedskin benchmark suite to the test battery. Fixed type inference problems when calling the Python randrange function. SSA algorithms no longer fail when joining variables whose type is a union type in the global context. raise parameters now are correctly checked as subclasses of BaseException
- [22/01/2018] Improved error messages when dealing with return types from recursive calls. Dictionaries are able now to store user objects as keys. Corrected the behavior of reporting an undefined type when a dictionary contained NoneType types.
- [23/01/2018] Created a new testing system when programs in the test battery do not require an external file with type information of variables to be considered correct. They are correct if the stypy analysis do not return TypeErrors. This new system is optional, as type data files are the way of precisely checking that the type inference programs are correct. However, this speeds up adding new programs to the test battery. The flag that controls this behavior is called force_type_data_files. Fixed a problem where for loops cannot iterate over class attributes, creating the attribute and its type in the process. Added the 'genetic' and 'genetic2' programs from the shedskin benchmark suite to the test battery. Types stored in a list are now correctly inferred when using the += operator with a tuple
- [24/01/2018] Added the full programs section of shedskin benchmark suite to the test battery (54 programs). Test battery code now accepts a number of user-defined expected errors to consider certain tests correct. max and min builtin functions now correctly process the type of its parameters. Union types of instances of the same type are now properly mergued (an unique instance of the type is generated with the union types of its members)
- [26/01/2018] Fixed type inference for the list.pop method. If position of a container is a type error when writing to it, this type error is now propagated and a new one is not generated. Loops over empty data structures (type of contents if UndefinedType) will not iterate. Therefore, stypy now did not process the loop body in these cases. This removes a lot of false positives.
- [29/01/2018] Fixed code generation for the else statement of the for clauses not closing correctly its SSA contexts. Added while loop to the "loop over empty data structures will not iterate" optimization of the 26/01/2018. If conditions that dynamically evaluates to NoneType only execute the else part. This is a runtime idiom.
- [30/01/2018] Tested the type inference of the shedskin kanoodle, kmeansapp programs. Inequality operators can now be applied to any object.
- [31/01/2018] Added a web page to describe the different stypy test suites. itertools.product object can now be properly processed. When manipulating dictionaries whose keys are union types, now each member of a union type is inserted with its associated value.
- [01/02/2018] Fixed loading the real module code instead of the type-inference equivalent when a module imports a member of a submodule defined in its own directory tree. Fixed problems with the file.next() method. Iterating files directly no longer causes false negatives. Added initial support to infer types of code that uses the Python re library.
- [02/02/2018] Added more support to the Python re module. 34/52 shedskin programs processed correctly (WIP)
- [05/02/2018] Lists that has been populated partially from other lists using slice indexes have now types correctly calculated.
- [07/02/2018] Errors produced in assert expressions or in loops conditions inside an SSA algorithm now are correctly reported as warnings
- [08/02/2018] Added support to type inference with the collections.deque Python class
- [15/02/2018] Comparisons between ints and strings no longer report a type error. else clases in for and while loops now properly generate type inference code.
- [16/02/2018] Fixed rare bug with code generation of for loops with else branches. Fixed assignments of lists when using slices incorrectly calculating the list type.
- [19/02/2018] Attributes that start with __ are now located in its class or instance. Comparisons between ints and str are now properly processed.
- [20/02/2018] Union types with tuples now integrate tuples with the same element types into a single tuple better.
- [21/02/2018] Fixed getting elements from a dictionary inside a comprehension. Solved type inference problems when calling the set.difference Python API method. Fixed type inference problems when calling a overriden __str__ method over user classes. Iteration over text files now properly infers str types. Type is inferred correctly when the % operator is used with strings into format usages.
- [23/02/2018] Fixed incorrectly reporting type warnings as type errors when calling members from a variable that has a union type inside a SSA branch. 48/52 shedskin programs processed.
- [26/02/2018] Fixed type inference problems with random.choice and random.seed functions. Stypy can now limit the amount of reported type errors and warnings to a maximum configurable value if the user decides to do so. It has been fixed to 200 each by default. If the limit is hit, the program warns the user about the incomplete type error reporting. Solved problems when returning type errors when using == with objects of different types. Fixed type inference with defaultdict types. Provided more information when dealing with invalid loop variables.
- [28/02/2018] Improved the algorithm that merges object of the same type into a single instance when dealing with union types to include more cases. This also increased the performance of stypy significantly. Union types no longer accept multiple NoneTypes. 51/52 shedskin programs are now operational.
- [01/03/2018] Fixed type rules for map functions when passing a numeric type as the first parameter. 52/52 shedskin programs correctly processed
- [05/03/2018] Multiple assignments in a for loop condition now correctly assign individual types over iterating variables if the type of the condition is a tuple. This improves type inference in multiple cases.
- [07/03/2018] Improved the type inference of dictionaries when constructed from tuples with two elements.
- [08/03/2018] Fixed a performance problem when desugaring multiple assignments with a function source.
- [09/03/2018] Improved type inference when returning multiple elements from a function. Improved the type inference code when returning tuples from functions. Fixed a type inference problem when comparing numbers and lists.
- [12/03/2018] Fixed a regression bug when obtaining contained types from a union type and when determining of two types are mergeable (only when both types were methods). Fixed type inference errors with the != operator.
- [13/03/2018] Implemented the detection of the While True: idiom, when the body of a while loop do not generate a SSA branch as it always enters the loop.
- [14/03/2018] Incorporated a new type of idiom that identifies if branches with a variable that is tested to be None. None value is removed from the variable possible values in the code that follows the if, provided the if clause includes a raise or a return in its body. This decreases the amount of warnings reported by stypy in certain cases. Improved type inference when assigning multiple values to variables.
- [15/03/2018] Added a warning when a program uses recursive types, indicating that type inference may not be accurate when using recursion. Similar warnings in the same line referring to potential undefined types are now packed into a single one to improve readability
- [16/03/2018] Error and warning reporting have been much improved: Warnings are now grouped by line when causes are similar. A bug with source code lines shown in errors have been solved, so now almost every error shows its origin source code line.
- [19/03/2018] Stack traces of errors and warnings now show the exact line when the function is invoked to provide additional information. These lines are cut to a maximum length to facilitate reading the errors. Fixed a type inference error that infers a tuple with an undefined type when performing a product of a tuple with an integer.
- [20/03/2018] Added support for the heapq module. Fixed a rare problem when inferring types when adding tuples to lists.
- [21/03/2018] Removed some errors that were reported incorrectly due to temporal variables created by stypy within function contexts.
- [22/03/2018] Improved type inference for the collections.defaultdict class


Copyright (c) [2016-2018] [Francisco Ortín Soler, José Manuel Redondo López, Baltasar García Pérez-Schofield]