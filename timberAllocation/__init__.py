from collections import defaultdict
from defaultlist import defaultlist
from functools import partial
import itertools
import math
import typing
import csv
import ast
from pathlib import Path
from copy import deepcopy
from decimal import Decimal
import numpy as np

import RichConsole

from .utils import TupleHashable, StructuredComparable

#FloatType = Decimal
FloatType = float
NumT = typing.Union[float, int, Decimal, np.float64]
TSVReadT = typing.List[typing.Tuple[NumT]]


def readFromTSV(path: Path) -> typing.Tuple[typing.Mapping, TSVReadT]:
	with path.open("rt", encoding="utf-8") as f:
		lsIter = iter(f)
		firstLine = next(lsIter)
		if firstLine[0] != "#":
			raise ValueError("First line must be a comment specifying the problem")

		firstLine = firstLine[1:].lstrip()
		task = ast.literal_eval(firstLine)

		decimalPrecision = FloatType("0.0001")

		if FloatType is Decimal:
			task = {FloatType(k).quantize(decimalPrecision): v for k, v in task.items()}

		ds = []
		for r in csv.reader(lsIter, dialect=csv.excel_tab):
			rI = r[0]
			try:
				rI = int(rI)
			except ValueError:
				pass

			rNew = [rI]
			for i in range(1, len(r)):
				n = r[i]
				if n:
					try:
						n = int(n)
					except ValueError:
						n = FloatType(n)
					rNew.append(n)

			r = tuple(rNew)
			ds.append(r)
	return task, getParts(ds)


class PartId(TupleHashable):
	__slots__ = ("_origId", "chunkId")

	def _coIter(self, other) -> typing.Iterable[typing.Tuple[typing.Any, typing.Any]]:
		yield "origId", self.origId, other.origId
		yield "chunkId", self.chunkId, other.chunkId

	def __init__(self, origId: typing.Any, chunkId: typing.Any = ()) -> None:
		if not isinstance(origId, tuple):
			origId = (origId,)
		self._origId = origId
		self.chunkId = chunkId

	@property
	def origId(self) -> "PartId":
		return PartId(self._origId)

	def toTuple(self) -> typing.Union[typing.Tuple]:
		return self._origId + (self.chunkId if self.chunkId else ())

	def __repr__(self) -> str:
		return repr(self.toTuple()) if self.chunkId else (repr(self._origId) if len(self._origId) > 1 else repr(self._origId[0]))

	def __add__(self, other: typing.Tuple) -> "PartId":
		res = self.__class__(self._origId, self.chunkId + other)
		return res

	def __lt__(self, other: "PartId") -> bool:
		return self.toTuple() < other.toTuple()

	def __gt__(self, other: "PartId") -> bool:
		return self.toTuple() > other.toTuple()

	def __le__(self, other: "PartId") -> bool:
		return self.toTuple() <= other.toTuple()

	def __ge__(self, other: "PartId") -> bool:
		return self.toTuple() >= other.toTuple()


class IPart:
	__slots__ = ("id",)

	def __init__(self, id: PartId) -> None:
		self.id = id

	def __lt__(self, other: "Part") -> bool:
		return self.len < other.len

	def __gt__(self, other: "Part"):
		return self.len > other.len

	def __ge__(self, other: "Part"):
		return self.len >= other.len

	def __le__(self, other: "Part"):
		return self.len <= other.len

	def toTuple(self):
		return (self.id,)

	def __repr__(self):
		return self.__class__.__name__[0] + "<" + repr(self.id) + ", " + str(self.len) + ">"


class Part(IPart):
	__slots__ = ("id", "len")

	def __init__(self, id: PartId, len: NumT) -> None:
		super().__init__(id)
		self.len = len

	def toTuple(self):
		return super().toTuple() + (self.len,)

	def split(self, splitPoint: NumT) -> typing.Tuple["Part", "Part"]:
		secondChunkLen = self.len - splitPoint
		firstChunk = deepcopy(self)
		firstChunk.id += (0,)
		firstChunk.len = splitPoint

		secondChunk = deepcopy(self)
		secondChunk.id += (1,)
		secondChunk.len = secondChunkLen

		return (firstChunk, secondChunk)


class DirPart(Part):
	__slots__ = ("dir",)

	def __init__(self, id: PartId, len: NumT, dir: int) -> None:
		super().__init__(id, len)
		self.dir = dir

	def __repr__(self):
		return str((RichConsole.groups.Fore.lightRed if self.dir else RichConsole.groups.Fore.lightGreen)(super().__repr__()))


class VectorPart(IPart):
	__slots__ = ("basis", "pieces")

	def __init__(self, id: PartId, piecesVector: "np.ndarray", basis: typing.Optional[np.ndarray] = None) -> None:
		if not isinstance(piecesVector, np.ndarray):
			raise ValueError()
		super().__init__(id)
		self.pieces = piecesVector
		self.basis = basis

	def toTuple(self):
		return super().toTuple() + (self.pieces,)

	def split(self, *chunks: "np.ndarray") -> typing.Iterable["VectorPart"]:
		restChunkPieces = deepcopy(self.pieces)
		for i, piecesToRetain in enumerate(chunks):
			if not isinstance(piecesToRetain, np.ndarray):
				raise ValueError(piecesToRetain)

			restChunkPieces = restChunkPieces & ~piecesToRetain
			chunk = deepcopy(self)
			chunk.id += (i,)
			chunk.pieces = piecesToRetain

			yield chunk

		if restChunkPieces.any():
			chunk = deepcopy(self)
			chunk.id += (i,)
			chunk.pieces = restChunkPieces
			yield chunk

	@property
	def cutted(self) -> bool:
		return self.id != self.id.origId

	@classmethod
	def _subParts(cls, pieces: np.ndarray) -> np.ndarray:
		print(pieces)
		return np.identity(pieces.shape[0])[pieces]

	def subParts(self):
		return self.__class__._subParts(self.pieces)

	@classmethod
	def _getCuttings(cls, pieces: np.ndarray) -> typing.Iterator[typing.Iterable[np.ndarray]]:
		subParts = cls._subParts(pieces)

		for i in range(1, len(subParts) // 2 - 1):  # the rest are the complements
			for firstPieceParts in itertools.combinations(subParts, i):
				firstPieceVec = np.sum(firstPieceParts, axis=0).astype(bool)
				secondPieceVec = pieces & ~firstPieceVec
				yield (firstPieceVec, secondPieceVec)
				for secondPieceCuttings in cls._getCuttings(secondPieceVec):
					yield (firstPieceVec,) + secondPieceCuttings

	def getCuttings(self) -> typing.Iterator[typing.Iterable[np.ndarray]]:
		return self.__class__._getCuttings(self.pieces)
		#for firstPieceVec in self.__class__._getCuttings(self.pieces):
		#	yield self.split(firstPieceVec)

	@property
	def len(self) -> FloatType:
		return self.basis @ self.pieces


class DirVectorPart(VectorPart):
	__slots__ = ("dir",)

	def __init__(self, id: PartId, piecesVector: "np.ndarray", basis: np.ndarray, dir: int) -> None:
		super().__init__(id, piecesVector, basis)
		self.dir = dir

	def __repr__(self):
		return str((RichConsole.groups.Fore.lightRed if self.dir else RichConsole.groups.Fore.lightGreen)(super().__repr__()))


def getParts(ds: TSVReadT) -> typing.Dict[PartId, NumT]:
	parts = {}
	for r in ds:
		rId = r[0]
		r = r[1:]
		if len(r) == 1:
			# parts[rId] = r[0]
			parts[PartId(rId,)] = r[0]
		else:
			for i, rp in enumerate(r):
				parts[PartId((rId, i))] = rp
	return parts


class Slot(TupleHashable):
	__slots__ = ("origL", "inSizeId", "l")

	def __init__(self, inSizeId: int, l: NumT) -> None:
		self.origL = l
		self.inSizeId = inSizeId
		self.l = l

	def toTuple(self) -> typing.Tuple[NumT, int, NumT]:
		return (self.origL, self.inSizeId, self.l)

	def __repr__(self) -> str:
		return repr(self.toTuple())


def getSlots(task: typing.Dict[int, int]) -> typing.Dict[int, Slot]:
	slots = {}
	j = 0
	for slotL, count in task.items():
		for i in range(count):
			slots[j] = Slot(i, slotL)
			j += 1
	return slots


class HashableDefaultDict(TupleHashable, defaultdict):
	__slots__ = ()

	def _toTuple(self) -> typing.Iterator[typing.Tuple]:
		for k, v in self.items():
			yield (k, v.toTuple())

	def toTuple(self) -> typing.Tuple:
		return tuple(self._toTuple())


class HashableDefaultList(TupleHashable, defaultlist):
	__slots__ = ()

	def _toTuple(self) -> typing.Iterator[typing.Tuple]:
		for v in self:
			yield v.toTuple()

	def toTuple(self) -> typing.Tuple:
		return tuple(self._toTuple())


class HashableList(TupleHashable, list):
	__slots__ = ()

	def toTuple(self) -> typing.Tuple:
		return tuple(self)


def initSolution() -> typing.DefaultDict[NumT, typing.List[NumT]]:
	return HashableDefaultDict(partial(HashableDefaultList, HashableList))


def greedySolution(task: typing.Dict[NumT, int], parts: typing.Dict[PartId, NumT]) -> typing.Iterator[HashableDefaultDict]:
	from .greedy import getCandidates, iteration

	opParts = deepcopy(parts)
	slots = getSlots(task)
	solution = initSolution()

	cumDelta = 0

	while slots:
		cumDelta = iteration(cumDelta, slots, slots, opParts, opParts, solution)

	yield solution


def greedyGenerationsSolution(task, parts):
	from .greedy import getCandidates, iteration

	opParts = deepcopy(parts)
	slots = getSlots(task)
	solution = initSolution()

	cumDelta = 0

	garbageSlots = {}
	garbageParts = {}

	while slots:
		cumDelta = iteration(cumDelta, slots, garbageSlots, opParts, garbageParts)

	while garbageSlots:
		cumDelta = iteration(cumDelta, garbageSlots, slots, garbageParts, garbageParts)

	yield solution


def optimizeSolutionUsingGradientDescent(alpha, L, T, parts):
	from .optimization import opt, alphaToSolution
	from matplotlib import pyplot as plt

	plt.matshow(alpha)
	plt.colorbar()
	plt.show()

	res = opt(L, alpha, T)

	plt.matshow(res)
	plt.colorbar()
	plt.show()

	solution = alphaToSolution(L, res, T, parts)
	yield solution


def gradientDescentSolution(task, parts):
	from .optimization import genAlpha, genL, genTarg

	T = genTarg(getSlots(task))
	L = genL(parts)
	alpha = genAlpha(L, T)
	return optimizeSolutionUsingGradientDescent(alpha, L, T, parts)


def hybridSolution(task, parts):
	from .optimization import solutionToAlphaMatrix, genL, genTarg

	sol = next(iter(greedySolution(task, parts)))

	T = genTarg(getSlots(task))
	L = genL(parts)
	alpha = solutionToAlphaMatrix(task, parts, sol)
	return optimizeSolutionUsingGradientDescent(alpha, L, T, parts)
