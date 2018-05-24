class Test:

    def write(self, str):
        print(str)

    def visit(self, obj):
        print obj

    def visit_Dict(self, t):
        self.write("{")

        a = 5
        def write_pair(pair):
            (k, v) = pair
            self.visit(k)
            self.write(": ")
            self.visit(v)
            print(a)

        write_pair((3,4))


t = Test()
t.visit_Dict(None)


