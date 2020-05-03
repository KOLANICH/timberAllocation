import typing

import z3
from timberAllocation.utils import z3Numpy
import numpy as np

from . import DirPart, DirVectorPart, FloatType, NumT, Part, PartId, Slot, getParts, getSlots, initSolution
from .middleRepr import vectorFactorization

def z3Solution(task: typing.Dict[NumT, int], parts: typing.Dict[PartId, NumT]):
	intial = []
	initialTotal = 0
	for partId, partLen in parts.items():
		intial.append(partLen)
		initialTotal += partLen

	target = []

	targetTotal = 0
	for i, (partLen, count) in enumerate(task.items()):
		for j in range(count):
			partId = PartId((partLen, j))
			target.append(partLen)
			targetTotal += partLen

	unusedLen = initialTotal - targetTotal

	if unusedLen:
		if unusedLen > 0:
			target.append(unusedLen)
		else:
			raise ValueError("The problem is unsolvable: length in the left is greater than the length in the right", unusedLen)

	intial = np.array(intial)
	target = np.array(target)

	it = np.hstack([intial, target])
	print(it, intial.shape[0], target.shape[0])

	middle = z3Numpy.array("middle", (it.shape[0] - 1,), dtype=float)
	alphaI = z3Numpy.array("ai", (middle.shape[0], intial.shape[0]), dtype=bool)
	alphaT = z3Numpy.array("at", (middle.shape[0], target.shape[0]), dtype=bool)
	alpha = z3Numpy.array(np.hstack([alphaI,alphaT]))  # np.hstack always returns np.array
	print(alpha)


	s = z3.Optimize()

	s.add(middle @ alpha == it)
	print((alpha.sum(axis=0)).__class__, (alpha.sum(axis=0) == 1).__class__)
	#s.add(alpha.sum(axis=0))

	joins = alphaT.sum()
	cuts = alphaI.sum()
	s.minimize(cuts)
	s.minimize(joins)

	res = s.check()
	print(res)

