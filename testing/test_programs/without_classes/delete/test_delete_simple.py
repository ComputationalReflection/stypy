

class Foo:
    att = "sample"

    def met(self):
        return self.att



f = Foo()

att_predelete = f.att
met_predelete = f.met
met_result_predelete = f.met()
func_predelete = f.met

del Foo.att

att_postdelete = f.att
met_result_postdelete = f.met()

del Foo.met

met_postdelete = f.met
func_result_postdelete = func_predelete()