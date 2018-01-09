
# Foo: Foo

class Foo:
    # self: instance
    att = 'sample'
    # met() -> str /\ met() -> TypeError 

    def met(self):
        # self: instance
        return self.att

# f: Foo instance
f = Foo()
# att_predelete: str
att_predelete = f.att
# met_predelete: instancemethod
met_predelete = f.met
# met_result_predelete: str
met_result_predelete = f.met()
# func_predelete: instancemethod
func_predelete = f.met
del Foo.att
# att_postdelete: TypeError
att_postdelete = f.att
# met_result_postdelete: TypeError
met_result_postdelete = f.met()
del Foo.met
# met_postdelete: TypeError
met_postdelete = f.met
# func_result_postdelete: TypeError
func_result_postdelete = func_predelete()