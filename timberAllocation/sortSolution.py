import sys
import typing
from decimal import Decimal
from functools import partial
import heapq

import more_itertools

from . import DirPart, HashableDefaultDict, NumT, Part, PartId, Slot, getParts, getSlots, initSolution

__all__ = ("sortSolution",)


def initialize(task: typing.Dict[Decimal, int], parts: typing.Dict[PartId, NumT]):
	firstTotal = Decimal(0)

	first = []
	for partId, partLen in parts.items():
		first.append(DirPart(partId, partLen, 0))
		firstTotal += partLen

	secondTotal = Decimal(0)

	second = []
	for i, (partLen, count) in enumerate(task.items()):
		for j in range(count):
			second.append(DirPart(PartId((partLen, j)), partLen, 1))
			secondTotal += partLen

	unusedLen = firstTotal - secondTotal

	if unusedLen:
		if unusedLen > 0:
			second.append(DirPart(PartId("U"), unusedLen, 1))
		else:
			raise ValueError("The problem is unsolvable: length in the left is greater than the length in the right", unusedLen)

	return first, second


def annihilateAdjacent(joint, solution, allowSplitting):
	secondConsumed = False
	newJoint = []

	haveStalled = True

	f = heapq.heappop(joint)
	while joint:
		s = heapq.heappop(joint)
		if secondConsumed:
			secondConsumed = False
		else:
			#print("annihilateInequalPairUnparam", f, s, f.dir != s.dir, f.len - s.len)
			if f.dir == s.dir:
				heapq.heappush(newJoint, f)
				secondConsumed = False
			else:
				i, t = (f, s) if f.dir == 0 else (s, f)

				if abs(f.len - s.len) <= sys.float_info.epsilon:
					l = f.len
					haveStalled = False
					secondConsumed = True
				else:
					if t.len > i.len:
						l = i.len
						toAppend = t.split(l)[1]
						haveStalled = False
						heapq.heappush(newJoint, toAppend)
						if t.id._origId[0] != "U":
							solution[t.id._origId[0]][t.id._origId[1]].append((i.id.origId, l))
						secondConsumed = True
					else:  # i.len > t.len
						if allowSplitting:
							l = t.len
							toAppend = i.split(l)[1]
							haveStalled = False
							heapq.heappush(newJoint, toAppend)
							if t.id._origId[0] != "U":
								solution[t.id._origId[0]][t.id._origId[1]].append((i.id.origId, l))
							secondConsumed = True
						else:
							heapq.heappush(newJoint, f)
							secondConsumed = False

			if secondConsumed:
				newJoint.extend(joint)
				return newJoint, False
		f = s

	if not secondConsumed:
		heapq.heappush(newJoint, s)
	return newJoint, haveStalled


def solve(joint):
	solution = initSolution()

	print(joint)

	allowCutting = False

	while joint:
		joint, allowCutting = annihilateAdjacent(joint, solution, allowCutting)
		joint = sorted(joint)
		print(joint)

	yield solution


def sortSolution(task: typing.Dict[Decimal, int], parts: typing.Dict[PartId, NumT]) -> typing.Iterator[HashableDefaultDict]:
	first, second = initialize(task, parts)
	joint = first + second
	joint = sorted(joint)
	return solve(joint)
