"""Module Regular Expression & Parse Walk Functionality"""
import re, os
from antlr4 import InputStream, CommonTokenStream, ParseTreeWalker
from statmtLexer import statmtLexer
from statmtParser import statmtParser
from statmtListener import statmtListener


class InnerCheck(statmtListener):
    """Class Representing Inner Operator Listener"""

    def __init__(self, i, j, k):
        self.param = {"variable": {}, "spatial": {i: "i", j: "j"}, "reduce": {k: "k"}}
        self.info = {}
        self.stack = []
        self.cur_var = -1
        self.types = {
            ("i", "j"): "fmat",
            ("i", "k"): "rmat",
            ("k", "j"): "cmat",
            ("i",): "rvec",
            ("j",): "cvec",
        }
        self.innerOp = ""

    def genInnerOp(self):
        return self.innerOp + "\n"

    def genOpDecl(self):
        lines = []
        row, col, tot = 0, 0, 0
        for var in self.param["variable"].values():
            if var == -1:
                continue
            var_type = self.info[var]
            if var_type[0] == "r":
                lines.append(
                    "Array<Type,Size_Thr_Row> reg_{}=in_reg_row[{}];\n".format(var, row)
                )
                row += 1
            elif var_type[0] == "c":
                lines.append(
                    "Array<Type,Size_Thr_Col> reg_{}=in_reg_col[{}];\n".format(var, col)
                )
                col += 1
            elif var_type[0] == "f":
                lines.append(
                    "Array<Type,Size_Thr_Tot> reg_{}=in_reg_tot[{}];\n".format(var, tot)
                )
                tot += 1
            else:
                raise ValueError("Error Variable")
        return lines

    def genIterDecl(self):
        lines = []
        for var in self.param["variable"].values():
            var_type = self.info[var]
            if var == -1:
                continue
            if var_type[1:] == "vec":
                pass
            elif var_type[0] == "r":
                lines.append("TileLoadRow tile_{};\n".format(var))
            elif var_type[0] == "c":
                lines.append("TileLoadCol tile_{};\n".format(var))
            elif var_type[0] == "t":
                raise ValueError("Wait for implementation")
            else:
                raise ValueError("Error Variable")
        for var in self.param["variable"].values():
            var_type = self.info[var]
            if var == -1:
                continue
            if var_type[1:] == "vec":
                if var_type[0] == "r":
                    lines.append(
                        "SliceLoadVec<Type,Size_Thr_Row,Size_Warp_Row> slice_{};\n".format(
                            var
                        )
                    )
                elif var_type[0] == "c":
                    lines.append(
                        "SliceLoadVec<Type,Size_Thr_Col,Size_Warp_Col> slice_{};\n".format(
                            var
                        )
                    )
                elif var_type[0] == "t":
                    raise ValueError("Wait for implementation")
                else:
                    raise ValueError("Error Variable")
            elif var_type[0] == "r":
                lines.append("SliceLoadRow slice_{};\n".format(var))
            elif var_type[0] == "c":
                lines.append("SliceLoadCol slice_{};\n".format(var))
            elif var_type[0] == "t":
                raise ValueError("Wait for implementation")
            else:
                raise ValueError("Error Variable")
        return lines

    def genMemory(self):
        lines = []
        for var in self.param["variable"].values():
            var_type = self.info[var]
            if var == -1:
                continue
            if var_type[1:] == "vec":
                pass
            elif var_type[0] == "r":
                lines.append(
                    "typename TileLoadRow::shared_memory smem_{};\n".format(var)
                )
            elif var_type[0] == "c":
                lines.append(
                    "typename TileLoadCol::shared_memory smem_{};\n".format(var)
                )
            elif var_type[0] == "t":
                continue
            else:
                raise ValueError("Error Variable")
        return lines

    def genIterInit(self):
        lines = []
        for var in self.param["variable"].values():
            var_type = self.info[var]
            if var == -1:
                continue
            if var_type[1:] == "vec":
                pass
            elif var_type[0] == "r":
                lines.append(
                    "tile_{0}((Type *)param.inputs[{0}],smem.smem_{0},{{param.dim2[{0}].x,param.dim2[{0}].y,param.size_K}},{{id.block_m,id.block_k,id.tot_thr}}),\n".format(
                        var
                    )
                )
            elif var_type[0] == "c":
                lines.append(
                    "tile_{0}((Type *)param.inputs[{0}],smem.smem_{0},{{param.dim2[{0}].x,param.dim2[{0}].y,param.size_K}},{{id.block_n,id.block_k,id.tot_thr}}),\n".format(
                        var
                    )
                )
            elif var_type[0] == "t":
                raise ValueError("Wait for implementation")
            else:
                raise ValueError("Error Variable")
        row, col, tot = 0, 0, 0
        for var in self.param["variable"].values():
            var_type = self.info[var]
            if var == -1:
                continue
            if var_type[1:] == "vec":
                if var_type[0] == "r":
                    lines.append(
                        "slice_{0}(&regs_row[{1}],(Type *)param.inputs[{0}],{{param.dim2[{0}].x}},{{id.block_m,id.warp_m,id.thread_m}}),\n".format(
                            var, row
                        )
                    )
                    row += 1
                elif var_type[0] == "c":
                    lines.append(
                        "slice_{0}(&regs_col[{1}],(Type *)param.inputs[{0}],{{param.dim2[{0}].x}},{{id.block_n,id.warp_n,id.thread_n}}),\n".format(
                            var, col
                        )
                    )
                    col += 1
                elif var_type[0] == "t":
                    raise ValueError("Wait for implementation")
                else:
                    raise ValueError("Error Variable")
            elif var_type[0] == "r":
                lines.append(
                    "slice_{0}(&regs_row[{1}],smem.smem_{0}.tile.storage,{{id.warp_m,id.thread_m}}),\n".format(
                        var, row
                    )
                )
                row += 1
            elif var_type[0] == "c":
                lines.append(
                    "slice_{0}(&regs_col[{1}],smem.smem_{0}.tile.storage,{{id.warp_n,id.thread_n}}),\n".format(
                        var, col
                    )
                )
                col += 1
            elif var_type[0] == "t":
                raise ValueError("Wait for implementation")
            else:
                raise ValueError("Error Variable")
        lines[-1] = lines[-1][:-2] + "\n"
        return lines

    def genIterFunc(self):
        funcs = {"reset_load": [], "tile_load": [], "slice_load": []}
        for var in self.param["variable"].values():
            var_type = self.info[var]
            if var == -1:
                continue
            if var_type[1:] == "vec" or var_type[0] == "t":
                pass
            elif var_type[1:] == "mat":
                funcs["reset_load"].append("slice_{}.reset();\n".format(var))
                funcs["tile_load"].append(
                    "tile_{0}.load();tile_{0}.next();\n".format(var)
                )
                funcs["slice_load"].append(
                    "slice_{0}.load();slice_{0}.next();\n".format(var)
                )
            else:
                raise ValueError("Error Variable")
        return funcs

    def exitInnerOp(self, ctx):
        self.innerOp = ctx.getText()

    def enterArray(self, ctx):
        var_name = ctx.variable().getText()
        if var_name in self.param["spatial"] or var_name in self.param["reduce"]:
            raise ValueError("Error Variable")
        if var_name not in self.param["variable"]:
            self.param["variable"][var_name] = self.cur_var
            self.cur_var += 1
        arr = self.param["variable"][var_name]
        self.stack.append(arr)

    def exitArray(self, ctx):
        self.stack.pop()

    def enterVariable(self, ctx):
        def transfer_name(var_ind):
            if var_ind == -1:  # output
                return "out_reg.storage"
            else:
                return "reg_{}.storage".format(var_ind)

        var_name = ctx.getChild(0).symbol
        if var_name.text in self.param["variable"]:
            var_name.text = transfer_name(self.param["variable"][var_name.text])
        elif var_name.text in self.param["spatial"]:
            var_name.text = self.param["spatial"][var_name.text]
        elif var_name.text in self.param["reduce"]:
            var_name.text = self.param["reduce"][var_name.text]
        else:
            raise ValueError("Not Given")

    def enterVarList(self, ctx):
        var_tuple = tuple()
        var = self.stack[-1]
        if ctx.getChildCount() == 1:
            if var == -1:  # output
                raise ValueError("Wait for implementation")
            var_name = ctx.getText()
            if var_name in self.param["reduce"]:
                raise ValueError("Wait for implementation")
            if var_name not in self.param["spatial"]:
                raise ValueError("Error Variable")
            var_tuple = (self.param["spatial"][var_name],)
        elif ctx.getChildCount() == 3:
            if ctx.getChild(1).getText() != ",":
                raise ValueError("Error Variable")
            vars_ = [ctx.getChild(0).getText(), ctx.getChild(2).getText()]
            for var_name in vars_:
                if var_name in self.param["spatial"]:
                    var_tuple += (self.param["spatial"][var_name],)
                elif var_name in self.param["reduce"]:
                    var_tuple += (self.param["reduce"][var_name],)
                else:
                    raise ValueError("Error Variable")
        else:
            raise ValueError("Error VarList")
        if var_tuple not in self.types:
            raise ValueError("Wait for implementation")
        var_type = self.types[var_tuple]
        if var in self.info:
            if self.info[var] != var_type:
                raise ValueError("Error Variable")
        else:
            self.info[var] = var_type

    def exitVarList(self, ctx):
        var_type = self.info[self.stack[-1]]
        if ctx.getChildCount() == 3:
            if var_type[0] == "f":
                new_text = "{}*Size_Thr_Col+{}".format(
                    ctx.getChild(0).getText(), ctx.getChild(2).getText()
                )
                ctx.removeLastChild()
                ctx.removeLastChild()
                symbol = ctx.getChild(0).getChild(0).symbol
                symbol.text = new_text
            elif var_type[0] == "r":
                ctx.removeLastChild()
                ctx.removeLastChild()
            elif var_type[0] == "c":
                del ctx.children[0]
                del ctx.children[0]


class EpilogCheck(statmtListener):
    """Class Representing Epilogue Operator Listener"""

    def __init__(self, i, j, k):
        self.state = None
        self.variable = None
        self.info = {}
        self.stack = []
        self.param = {"spatial": {i: "i", j: "j"}, "reduce": {k: "k"}}
        self.cur_var = 0
        self.epilogOp = ""

    def genIterDecl(self):
        lines = []
        if self.state is None:
            lines = [
                "using Tile = TileStore<Type>;\n",
                "Tile tile;\n",
                "SliceStore<Type> slice;\n",
            ]
        elif self.state == 0:
            lines = [
                "using Tile = TileStoreSel<Type, comparator>;\n",
                "Tile tile;\n",
                "SliceStoreSel<Type, comparator> slice;\n",
            ]
        elif self.state == 1:
            lines = [
                "using Tile = TileStoreAgg<Type, atomicAdd>;\n",
                "TileLoadVec<int, Size_Thr_Row, Size_Warp_Row, Size_Block_Row> tile_r;\n",
                "TileLoadVec<int, Size_Thr_Col, Size_Warp_Col, Size_Block_Col> tile_c;\n",
                "Tile tile;\n" "SliceStore<Type> slice;\n",
            ]
        elif self.state == 2:
            lines = [
                "using Tile = TileStoreUnlinear<Type, {}, comparator, {}, Count_Thr_Block_Tot>;\n".format(
                    self.variable["algo"], self.variable["size"]
                ),
                "Tile tile;\n" "SliceStore<Type> slice;\n",
            ]
        else:
            raise ValueError("Not implemented")
        return lines

    def genMemory(self):
        lines = ["typename Tile::shared_memory smem;\n"]
        return lines

    def genIterInit(self):
        lines = []
        if self.state is None:
            lines = [
                "tile((Type *)param.work, smem.smem, param.dim2[Num_Inputs], id),\n",
                "slice(smem.smem.tile.storage, (Array<Type, Size_Thr_Col> *)&out_reg, id)\n",
            ]
        elif self.state == 0:
            lines = [
                "tile((Type *)param.work, smem.smem, (int *)param.inputs[Num_Inputs-1], (Array<Type, Size_Thr_Col> *)&out_reg, id.tot_thr),\n",
                "slice(smem.smem.tile.storage, (Array<Type, Size_Thr_Col> *)&out_reg, id.tot_thr)\n",
            ]
        elif self.state == 1:
            lines = [
                "tile_r((int *)param.inputs[Num_Inputs-{0}], smem.smem.ind_x.storage, {{param.dim2[Num_Inputs-{0}].x}}, {{id.block_m, id.warp_m, id.thread_m}}),\n".format(
                    2 - self.variable[self.info["i"]]
                ),
                "tile_c((int *)param.inputs[Num_Inputs-{0}], smem.smem.ind_y.storage, {{param.dim2[Num_Inputs-{0}].x}}, {{id.block_n, id.warp_n, id.thread_n}}),\n".format(
                    2 - self.variable[self.info["j"]]
                ),
                "tile((Type *)param.work, smem.smem, {param.size_p.kRow, param.size_p.kColumn}, param.dim2[Num_Inputs], id),\n",
                "slice(smem.smem.tile.storage, (Array<Type, Size_Thr_Col> *)&out_reg, id)\n",
            ]
        elif self.state == 2:
            lines = [
                "tile(smem.smem, (Type *)param.work, (int *)param.inputs[3], (int *)param.inputs[2]),\n",
                "slice(smem.smem.tile.storage, (Array<Type, Size_Thr_Col> *)&out_reg, id)\n",
            ]
        return lines

    def genCompare(self):
        line = ""
        if self.state == 0:
            line = """MMLT_DEVICE
bool comparator(Type {}) {{
  return {};
}}\n""".format(
                list(self.variable)[0], self.info
            )
        elif self.state == 2:
            line = """MMLT_DEVICE
bool comparator(Type target, Type num) {{
  return {};
}}\n""".format(
                self.info
            )
        return line

    def genIterFunc(self):
        funcs = {"tile_store": [], "slice_store": []}
        funcs["tile_store"] = ["tile.store();\n", "tile.next();\n"]
        funcs["slice_store"] = ["slice.store();\n", "slice.next();\n"]
        return funcs

    def exitEpilogOp(self, ctx):
        self.epilogOp = ctx.getText()

    def enterSelOp(self, ctx):
        self.state = 0
        self.variable = set()
        self.info = ctx.getChild(2).getText()

    def enterAggOp(self, ctx):
        self.state = 1
        self.variable = dict()

    def enterUnlOp(self, ctx):
        self.state = 2
        self.variable = dict()
        self.variable["size"] = int(ctx.getChild(2).getText())
        if self.variable["size"] <= 0:
            raise ValueError("Invalid Size")
        self.variable["algo"] = ctx.variable().getText()
        self.variable["vars"] = set()
        self.info = ctx.expression().getText()

    def exitUnlOp(self, ctx):
        if len(self.variable["vars"]) != 2:
            raise ValueError("Invalid Expression")
        elif (
            "target" not in self.variable["vars"] or "num" not in self.variable["vars"]
        ):
            raise ValueError("Unknown Variable")

    def enterVariable(self, ctx):
        var_name = ctx.getChild(0).symbol
        if self.state == 0:
            if len(self.variable) == 0:
                self.variable.add(var_name.text)
            elif var_name in self.variable:
                pass
            else:
                raise ValueError("More than one variable")
            var_name.text = "param"
        elif self.state == 1:
            if var_name.text in self.variable:
                var_name.text = "ind_{}".format(self.variable[var_name.text])
            elif var_name.text in self.param["spatial"]:
                var_name.text = self.param["spatial"][var_name.text]
            elif var_name.text in self.param["reduce"]:
                var_name.text = self.param["reduce"][var_name.text]
            else:
                raise ValueError("Illegal variable")
        elif self.state == 2:
            if var_name.text != self.variable["algo"]:
                self.variable["vars"].add(var_name.text)

    def enterArray(self, ctx):
        var_name = ctx.variable().getText()
        if var_name in self.param["spatial"] or var_name in self.param["reduce"]:
            raise ValueError("Error Variable")
        if var_name not in self.variable:
            self.variable[var_name] = self.cur_var
            self.cur_var += 1
        if self.cur_var > 2:
            raise ValueError("More than two variable")
        self.stack.append(var_name)

    def exitArray(self, ctx):
        self.stack.pop()

    def enterVarList(self, ctx):
        if self.state == 0:
            raise ValueError("Not supported")
        if ctx.getChildCount() == 1:
            var_name = ctx.getText()
            if var_name in self.param["reduce"]:
                raise ValueError("Wait for implementation")
            if var_name not in self.param["spatial"]:
                raise ValueError("Error Variable")
            if var_name in self.variable:
                raise ValueError("Already has the index")
            self.info[var_name] = self.stack[-1]
        else:
            raise ValueError("Index Error")


class NaturalCheck(statmtListener):
    """Class Representing Natural Operator Listener"""

    def __init__(self, i, j, k):
        self.variable = {}
        self.param = {"spatial": {i: "i", j: "j"}, "reduce": {k: "k"}}
        self.info = {}
        self.stack = []
        self.cur_var = 0
        self.natOp = ""

    def exitNatOp(self, ctx):
        self.natOp = ctx.getText()

    def enterVariable(self, ctx):
        var_name = ctx.getChild(0).symbol
        if var_name.text in self.variable:
            var_name.text = "ind_{}".format(self.variable[var_name.text])
        elif var_name.text in self.param["spatial"]:
            var_name.text = self.param["spatial"][var_name.text]
        elif var_name.text in self.param["reduce"]:
            var_name.text = self.param["reduce"][var_name.text]
        else:
            raise ValueError("Illegal variable")

    def enterArray(self, ctx):
        var_name = ctx.variable().getText()
        if var_name in self.param["spatial"] or var_name in self.param["reduce"]:
            raise ValueError("Error Variable")
        if var_name not in self.variable:
            self.variable[var_name] = self.cur_var
            self.cur_var += 1
        if self.cur_var > 2:
            raise ValueError("More than two variable")
        self.stack.append(var_name)

    def exitArray(self, ctx):
        self.stack.pop()

    def enterVarList(self, ctx):
        if ctx.getChildCount() == 1:
            var_name = ctx.getText()
            if var_name in self.param["reduce"]:
                raise ValueError("Wait for implementation")
            if var_name not in self.param["spatial"]:
                raise ValueError("Error Variable")
            self.info[var_name] = self.stack[-1]
        else:
            raise ValueError("Index Error")


class DeclrSplit(statmtListener):
    """Class Splitting Different Operators"""

    def __init__(self):
        self._params = {}
        self.params_set = set()
        self.natural_tree = None
        self.inner_tree = None
        self.epilog_tree = None

    def enterWhereClause(self, ctx):
        self._params["i"] = ctx.variable(0).getText()
        self.params_set.add(self._params["i"])
        self._params["j"] = ctx.variable(1).getText()
        self.params_set.add(self._params["j"])

    def enterWhenClause(self, ctx):
        self.natural_tree = ctx

    def enterReduceClause(self, ctx):
        self._params["k"] = ctx.variable().getText()
        self.params_set.add(self._params["k"])

    def enterInnerOp(self, ctx):
        self.inner_tree = ctx

    def enterEpilogOp(self, ctx):
        self.epilog_tree = ctx

    @property
    def params(self):
        if len(self.params_set) != 3:
            raise ValueError("Need three different axes")
        else:
            return self._params


class GrowingLibrary:
    """Growing Library that parse declaratives"""

    def __init__(self, declarative, template, generated, out_call, includes):
        self.walker = ParseTreeWalker()
        input_stream = InputStream(declarative)
        lexer = statmtLexer(input_stream)
        stream = CommonTokenStream(lexer)
        parser = statmtParser(stream)
        tree = parser.declr()
        self.split = DeclrSplit()
        self.walker.walk(self.split, tree)
        i, j, k = self.split.params["i"], self.split.params["j"], self.split.params["k"]
        self.inner = self.parse(self.split.inner_tree, InnerCheck(i, j, k))
        self.epilog = self.parse(self.split.epilog_tree, EpilogCheck(i, j, k))
        self.natural = self.parse(self.split.natural_tree, NaturalCheck(i, j, k))
        self.pattern = r"@(\d+):"
        self.template = template
        self.generated = generated
        self.out_call = out_call
        self.includes = includes

    def parse(self, tree, checker):
        if tree is not None:
            self.walker.walk(checker, tree)
        return checker

    def __eq__(self, other):
        cond0 = self.inner.innerOp == other.inner.innerOp
        cond1 = self.epilog.epilogOp == other.epilog.epilogOp
        cond2 = self.natural.natOp == other.natural.natOp
        return cond0 and cond1 and cond2

    def genInner(self):
        with open(
            os.path.join(self.template, "operator.h"), "r", encoding="utf-8"
        ) as template:
            with open(
                os.path.join(self.generated, "operator.h"), "w", encoding="utf-8"
            ) as code:
                for line in template:
                    code.write(line)
                    mtch = re.search(self.pattern, line)
                    if mtch:
                        num = int(mtch.group(1))
                        if num == 0:
                            code.writelines(self.inner.genOpDecl())
                        elif num == 1:
                            code.write(self.inner.genInnerOp())
                        else:
                            raise ValueError("No Such Hole")

    def genIterator(self):
        dict_lines = {}
        with open(
            os.path.join(self.template, "iterator.h"), "r", encoding="utf-8"
        ) as template:
            with open(
                os.path.join(self.generated, "iterator.h"), "w", encoding="utf-8"
            ) as code:
                for line in template:
                    code.write(line)
                    mtch = re.search(self.pattern, line)
                    if mtch:
                        num = int(mtch.group(1))
                        if num == 0:
                            code.writelines(self.inner.genIterDecl())
                        elif num == 1:
                            code.writelines(self.inner.genMemory())
                        elif num == 2:
                            code.writelines(self.inner.genIterInit())
                        elif num == 3:
                            dict_lines = self.inner.genIterFunc()
                            code.writelines(dict_lines["reset_load"])
                        elif num == 4:
                            code.writelines(dict_lines["tile_load"])
                        elif num == 5:
                            code.writelines(dict_lines["slice_load"])
                        elif num == 6:
                            code.writelines(self.epilog.genCompare())
                        elif num == 7:
                            code.writelines(self.epilog.genIterDecl())
                        elif num == 8:
                            code.writelines(self.epilog.genMemory())
                        elif num == 9:
                            code.writelines(self.epilog.genIterInit())
                        elif num == 10:
                            dict_lines = self.epilog.genIterFunc()
                            code.writelines(dict_lines["tile_store"])
                        elif num == 11:
                            code.writelines(dict_lines["slice_store"])
                        elif num == 12:
                            code.writelines(self.includes)
                        else:
                            raise ValueError("No Such Hole")

    def genNumPut(self):
        row, col, tot = 0, 0, 0
        whole = 0
        for k, v in self.inner.info.items():
            if k == -1:
                continue
            else:
                if v[0] == "r":
                    row += 1
                elif v[0] == "c":
                    col += 1
                elif v[0] == "f":
                    tot += 1
                else:
                    raise ValueError("No Such Layout")
                whole = row + col + tot
                if self.epilog.state is None:
                    pass
                elif self.epilog.state == 0:
                    whole += 1
                elif self.epilog.state == 1:
                    whole += 2
                elif self.epilog.state == 2:
                    whole += 2
                else:
                    raise ValueError("No Such State")
        lines = [
            "#define Num_Inputs {}\n".format(whole),
            "#define Num_Inputs_Row {}\n".format(row),
            "#define Num_Inputs_Col {}\n".format(col),
            "#define Num_Inputs_Tot {}\n".format(tot),
        ]
        with open(
            os.path.join(self.template, "macros.h"), "r", encoding="utf-8"
        ) as template:
            with open(
                os.path.join(self.generated, "macros.h"), "w", encoding="utf-8"
            ) as code:
                for line in template:
                    code.write(line)
                    mtch = re.search(self.pattern, line)
                    if mtch:
                        num = int(mtch.group(1))
                        if num == 0:
                            code.writelines(lines)

    def genCaller(self):
        isNat = self.natural.natOp != ""
        out = ""
        vars_ = self.inner.param["variable"]
        layout = self.inner.info
        inputs = [""] * (len(vars_) - 1)
        dims = [""] * len(inputs)
        for k, v in vars_.items():
            if v == -1:
                out = k
                continue
            inputs[v] = k
            lo = layout[v]
            if lo[1:] == "vec":
                if lo[0] == "r":
                    dims[v] = "{M_row, 1}"
                    if isNat:
                        inputs[v] += "+last_i"
                elif lo[0] == "c":
                    dims[v] = "{N_col, 1}"
                    if isNat:
                        inputs[v] += "+last_j"
                else:
                    raise ValueError("Not implemented")
            elif lo[1:] == "mat":
                if lo[0] == "r":
                    dims[v] = "{K_dep, M_row}"
                    if isNat:
                        inputs[v] += "+last_i"
                elif lo[0] == "c":
                    dims[v] = "{K_dep, N_col}"
                    if isNat:
                        inputs[v] += "+last_j"
                else:
                    raise ValueError("Not implemented")
            else:
                raise ValueError("Wrong Layout")
        if self.epilog.state is None:
            dims.append("{M_row, N_col}" if not isNat else "{M, N}")
        elif self.epilog.state == 0:
            inputs.append("gcounter")
            dims.append("{1, 1}")
            dims.append("{M_row, N_col}" if not isNat else "{M, N}")
        elif self.epilog.state == 1:
            lays = self.epilog.info
            local_in = [lays["i"], lays["j"]]
            if isNat:
                local_in = [lays["i"] + "+last_i", lays["j"] + "+last_j"]
            local_di = ["{M_row, 1}", "{N_col, 1}"]
            inputs.extend(local_in)
            dims.extend(local_di)
            dims.append("{L_row, L_col}")
        elif self.epilog.state == 2:
            inputs.append("gcounter" if not isNat else "gcounter+i")
            inputs.append("gmutex" if not isNat else "gmutex+i")
            dims.append("{1, 1}")
            dims.append("{1, 1}")
            dims.append("{M_row, N_col}" if not isNat else "{M, N}")
        else:
            raise ValueError("Not Implemented")
        if not isNat:
            param = "{{ {}, nullptr, {{ {} }}, {{M_row, N_col, K_dep}}, K_dep, {{ {} }}, 1 }}".format(
                out, ",".join(inputs), ",".join(dims)
            )
            return "launch({});".format(param)
        else:
            # *(Result + i) where Result is void **
            param = "{{ *({}+i), nullptr, {{ {} }}, {{M, N, K_dep}}, K_dep, {{ {} }}, 1}}".format(
                out, ",".join(inputs), ",".join(dims)
            )
            ret = """Params p_arr[P_num];
cudaStream_t streams[P_num];
int cur_i = 0, cur_j = 0;
for (int i = 0; i < P_num; i++) {{
	int last_i = cur_i, last_j = cur_j;
	while (cur_i < M_row && {}[cur_i] == i) {{
		cur_i++;
	}}
	while (cur_j < N_col && {}[cur_j] == i) {{
		cur_j++;
	}}
	int M = cur_i-last_i, N = cur_j-last_j;
	Params p = {};
	p_arr[i] = p;
	cudaStreamCreate(streams + i);
}}

for (int i = 0; i < P_num; i++)
	launch(p_arr[i], streams[i]);""".format(
                self.natural.info["i"], self.natural.info["j"], param
            )
            return ret

    def gen_codes(self):
        self.genInner()
        self.genIterator()
        self.genNumPut()
        if len(self.out_call) == 0:
            return self.genCaller()
        else:
            with open(self.out_call, "w", encoding="utf-8") as out:
                out.write(self.genCaller())

    def __repr__(self):
        return self.gen_codes()
