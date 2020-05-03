import itertools
import json
import math
from ast import literal_eval
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from pprint import pformat

import numpy as np
from plumbum import cli

from . import getParts, gradientDescentSolution, readFromTSV
from .sortSolution import sortSolution
from .stateGraph import Algo
from .vis import generateColors, visualizeAll


def algoSolution(task, parts):
	a = Algo(task, parts)
	solutionNodes = a()
	for solutionNode in solutionNodes:
		path = a.getPath(solutionNode)
		print(path)
		yield a.getSolutionFromPath(path)


methods = {
	"sort": sortSolution,
	"graph": algoSolution,
}

from .z3Solution import z3Solution
methods["z3"] = z3Solution


thisDir = Path(__file__).absolute().parent
moduleParentDir = thisDir.parent
testsDir = moduleParentDir / "tests"


class AllocationCLI(cli.Application):
	#method = cli.SwitchAttr(["-m", "--method"], argtype=cli.Set(*methods), argname="method", help="Method of solution.", default="sort")
	method = cli.SwitchAttr(["-m", "--method"], argtype=cli.Set(*methods), argname="method", help="Method of solution.", default="z3")

	#def main(self, input=testsDir / "47,31->51,22,5.tsv"):
	#def main(self, input=testsDir / "wood1.tsv"):
	#def main(self, input=testsDir / "wood.tsv"):
	def main(self, input=testsDir / "toy.tsv"):
		task, parts = readFromTSV(Path(input))
		partsColors = generateColors(parts)

		for solution in methods[self.method](task, parts):
			print("\n")
			visualizeAll(solution, parts, partsColors)


if __name__ == "__main__":
	AllocationCLI.run()
