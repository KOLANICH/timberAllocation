import typing
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from copy import deepcopy
from decimal import Decimal
from heapq import heappop, heappush
from math import inf
from weakref import ref

import numpy as np

import networkx as nx
import networkx.drawing.nx_agraph
import RichConsole

from . import DirPart, DirVectorPart, FloatType, NumT, Part, PartId, Slot, getParts, getSlots, initSolution
from .middleRepr import vectorFactorization
from .utils import RichReprable, RichStringableAndReprable, StructuredComparable, TupleHashable


class Metrics(RichStringableAndReprable, TupleHashable, StructuredComparable):
	__slots__ = ()

	def _coIter(self, other: "Metrics") -> typing.Iterable[typing.Tuple[typing.Any, typing.Any]]:
		for k in self.__class__.__slots__:
			sv = getattr(self, k)
			ov = getattr(other, k)
			yield (k, sv, ov)

	def intize(self) -> None:
		for k in self.__class__.__slots__:
			v = getattr(self, k)
			if isinstance(v, float) and v.is_integer():
				setattr(self, k, int(v))

	def toTuple(self) -> typing.Tuple[typing.Type["State"], typing.Any, NumT]:
		return tuple(getattr(self, k) for k in self.__class__.__slots__)


class PathMetrics(Metrics):
	__slots__ = ("_piecesInInitial", "_piecesInTarget", "lengthToAllocate", "chunksToAnnihilate")

	def __init__(self) -> None:
		self._piecesInInitial = None
		self._piecesInTarget = None
		self.lengthToAllocate = None
		self.chunksToAnnihilate = None

	@property
	def piecesInInitial(self):
		return self._piecesInInitial

	@piecesInInitial.setter
	def piecesInInitial(self, v):
		if v < 0:
			raise ValueError("piecesInInitial", v)
		self._piecesInInitial = v

	@property
	def piecesInTarget(self):
		return self._piecesInTarget

	@piecesInTarget.setter
	def piecesInTarget(self, v):
		if v < 0:
			raise ValueError("piecesInTarget", v)
		self._piecesInTarget = v

	def __rich_str__(self) -> RichConsole.RichStr:
		return "P(" + RichConsole.groups.Fore.lightGreen(str(self.piecesInInitial)) + ", " + RichConsole.groups.Fore.lightRed(str(self.piecesInTarget)) + "), L" + RichConsole.groups.Fore.lightBlue(str(self.lengthToAllocate))


class QualityMetrics(Metrics):
	__slots__ = ("cutsCount", "joinsCount")

	def __init__(self) -> None:
		self.cutsCount = None
		self.joinsCount = None

	@property
	def cutsAndJoinsCount(self) -> int:
		return self.cutsCount + self.joinsCount

	def _coIter(self, other: "QualityMetrics") -> typing.Iterable[typing.Tuple[typing.Any, typing.Any]]:
		yield ("cutsAndJoinsCount", self.cutsAndJoinsCount, other.cutsAndJoinsCount)

	def __rich_str__(self) -> RichConsole.RichStr:
		return "C(" + RichConsole.groups.Fore.lightCyan(str(self.cutsCount)) + "," + RichConsole.groups.Back.lightYellow(RichConsole.groups.Fore.black(str(self.joinsCount))) + ")"


class MetricsSet(RichStringableAndReprable, TupleHashable):
	__slots__ = ("path", "quality")

	def __init__(self) -> None:
		self.path = PathMetrics()
		self.quality = QualityMetrics()

	def toTuple(self) -> typing.Tuple[typing.Type["State"], typing.Any, NumT]:
		return tuple(getattr(self, k) for k in self.__class__.__slots__)

	def __rich_str__(self) -> RichConsole.RichStr:
		return self.path.__rich_str__() + ", " + self.quality.__rich_str__()

	def intize(self) -> None:
		for k in self.__class__.__slots__:
			getattr(self, k).intize()

	def __eq__(self, other: "MetricsSet") -> bool:
		return self.path == other.path

	def __lt__(self, other: "MetricsSet") -> bool:
		return self.path < other.path and self.quality <= other.quality

	def __gt__(self, other: "MetricsSet") -> bool:
		return self.path > other.path or self.quality > other.quality

	def __le__(self, other: "MetricsSet") -> bool:
		return self.path < other.path or (self.path == other.path and self.quality <= other.quality)

	def __ge__(self, other: "MetricsSet") -> bool:
		return self.path > other.path or (self.path == other.path and self.quality >= other.quality)


class Metered:
	__slots__ = ("metrics",)

	def __init__(self) -> None:
		self.metrics = None

	def __lt__(self, other: "Metered") -> bool:
		return self.metrics < other.metrics

	def __gt__(self, other: "Metered") -> bool:
		return self.metrics > other.metrics

	def __ge__(self, other: "Metered") -> bool:
		return self.metrics >= other.metrics

	def __le__(self, other: "Metered") -> bool:
		return self.metrics <= other.metrics


class PartsStorage(TupleHashable):
	"""Stores only pats ids. Takes less memory but it is not always possible to use it"""

	__slots__ = ("sizes",)

	def __init__(self) -> None:
		self.sizes = {}

	def add(self, part: Part) -> None:
		s = self.sizes.get(part.len, None)
		if s is None:
			self.sizes[part.len] = s = set()

		s |= {part.id}

	def remove(self, part: Part) -> None:
		self.sizes[part.len] -= {part.id}
		if not self.sizes[part.len]:
			del self.sizes[part.len]

	@classmethod
	def _getFirstIdUncut(cls, idsOfTheLen):
		for iD in idsOfTheLen:
			if idsOfTheLen == idsOfTheLen.origId:
				return iD

	def get(self, size: NumT, uncut=False):
		idsOfTheLen = self.sizes[size]
		if uncut:
			iD = self.__class__._getFirstIdUncut(idsOfTheLen)
		else:
			iD = next(iter(idsOfTheLen))
		return Part(iD, size)

	def pop(self, size: NumT, uncut: bool = False) -> Part:
		p = self.get(size, uncut)
		self.remove(p)
		return p

	def resize(self, part, newSize):
		self.remove(part)
		self.add(Part(part.id, newSize))

	def parts(self) -> typing.Iterable[Part]:
		for size, sizeGroup in self.sizes.items():
			for iD in sizeGroup:
				yield Part(iD, size)

	def toTuple(self) -> typing.Tuple:
		res = []
		for size, sizeGroup in self.sizes.items():
			res.append((size, tuple(sorted(sizeGroup))))
		return tuple(res)

	def __repr__(self) -> str:
		els = []
		for el in sorted(self.sizes):
			count = len(self.sizes[el])
			elStr = str(el.normalize() if isinstance(el, Decimal) else el)
			if count > 1:
				elStr += "×" + str(count)
			els.append(elStr)

		return "{" + ", ".join(els) + "}"


class ObjectsPartsStorage(PartsStorage):
	"""Stores parts objects explicitly"""

	__slots__ = ()

	def add(self, part: Part) -> None:
		s = self.sizes.get(part.len, None)
		if s is None:
			self.sizes[part.len] = s = set()

		s.add(part)

	def remove(self, part: Part) -> None:
		self.sizes[part.len] -= {part}
		if not self.sizes[part.len]:
			del self.sizes[part.len]

	@classmethod
	def _getFirstUncut(cls, partsOfTheLen: typing.Set[DirVectorPart]) -> typing.Optional[DirVectorPart]:
		for part in partsOfTheLen:
			if not part.cutted:
				return part

	def get(self, size: NumT, uncut: bool = False) -> DirVectorPart:
		partsOfTheLen = self.sizes[size]
		if uncut:
			return self.__class__._getFirstUncut(partsOfTheLen)
		else:
			return next(iter(partsOfTheLen))

	def resize(self, part, newSize):
		raise NotImplementedError

	def parts(self):
		for size, sizeGroup in self.sizes.items():
			for part in sizeGroup:
				yield part


USE_VECTORIZED = True


class SolutionState(Metered, TupleHashable):
	__slots__ = ("piecesInitial", "piecesTarget")
	DONT_REPR = frozenset(("initialPieces", "piecesTarget"))

	def __init__(self, task: typing.Dict[float, int], parts: typing.Dict[PartId, NumT]) -> None:
		super().__init__()
		self.metrics = MetricsSet()
		self.metrics.quality.cutsCount = 0
		self.metrics.quality.joinsCount = 0
		self.metrics.path.lengthToAllocate = 0
		self.metrics.path.piecesInInitial = 0
		self.metrics.path.piecesInTarget = 0

		if USE_VECTORIZED:
			self.piecesInitial = ObjectsPartsStorage()
			self.piecesTarget = ObjectsPartsStorage()
		else:
			self.piecesInitial = PartsStorage()
			self.piecesTarget = PartsStorage()


		self.metrics.path.lengthToAllocate = FloatType(0)

		uniqueParts = defaultdict(list)
		for partId, partLen in parts.items():
			uniqueParts[partLen].append(partId)
			self.metrics.path.lengthToAllocate += partLen
			self.metrics.path.piecesInInitial += 1

		targetTotal = FloatType(0)
		uniqueTargets = defaultdict(list)
		for i, (partLen, count) in enumerate(task.items()):
			for j in range(count):
				partId = PartId((partLen, j))
				uniqueTargets[partLen].append(partId)
				self.metrics.path.piecesInTarget += count
				targetTotal += partLen

		unusedLen = self.metrics.path.lengthToAllocate - targetTotal

		if unusedLen:
			if unusedLen > 0:
				uniqueTargets[unusedLen].append(PartId((inf, 0)))
				self.metrics.path.piecesInTarget += 1
			else:
				raise ValueError("The problem is unsolvable: length in the left is greater than the length in the right", unusedLen, self.metrics.lengthToAllocate, targetTotal, self.piecesInitial, self.piecesTarget)

		if USE_VECTORIZED:
			b, U = vectorFactorization(np.hstack([list(uniqueParts), list(uniqueTargets)]))

			UParts = U[:, : len(uniqueParts)]
			UTargets = U[:, -len(uniqueTargets) :]

			for i, (partLen, partIds) in enumerate(uniqueParts.items()):
				partVec = UParts[:, i]
				for partId in partIds:
					self.piecesInitial.add(DirVectorPart(partId, partVec, b, 0))

			for i, (partLen, partsOfThisLenIds) in enumerate(uniqueTargets.items()):
				partVec = UTargets[:, i]
				for partId in partsOfThisLenIds:
					self.piecesTarget.add(DirVectorPart(partId, partVec, b, 1))

			self.metrics.path.chunksToAnnihilate = np.sum(UParts)
		else:
			for i, (partLen, partIds) in enumerate(uniqueParts.items()):
				for partId in partIds:
					self.piecesInitial.add(DirPart(partId, partLen, 0))

			for i, (partLen, partsOfThisLenIds) in enumerate(uniqueTargets.items()):
				for partId in partsOfThisLenIds:
					self.piecesTarget.add(DirPart(partId, partLen, 1))

			self.metrics.path.chunksToAnnihilate = 0

	def toTuple(self) -> typing.Tuple:
		return (self.piecesInitial, self.piecesTarget)

	def __repr__(self) -> str:
		return repr(self.piecesInitial) + ", " + repr(self.piecesTarget) + ";" + self.metrics.__rich_str__().plain()


class State(RichReprable, Metered):
	__slots__ = ("children", "sizeToConsume", "parent", "metrics")
	nodeAbbrev = None

	def __init__(self, sizeToConsume: typing.Any, parent: typing.Optional["State"]) -> None:
		super().__init__()
		self.metrics = MetricsSet()

		self.sizeToConsume = sizeToConsume
		self.children = None
		self.parent = parent

	@abstractmethod
	def apply(self, solutionState: SolutionState) -> None:
		raise NotImplemented

	def toTuple(self) -> typing.Tuple[typing.Type["State"], typing.Any, NumT]:
		return (self.__class__, self.sizeToConsume,) + tuple(getattr(self, k) for k in self.__class__.__slots__)

	def __eq__(self, other: "State") -> bool:
		return self.toTuple() == other.toTuple()

	def __hash__(self) -> int:
		return hash(self.toTuple())

	def generateUnfilteredChildren(self, solutionState: SolutionState, statesObserved: typing.Set[SolutionState]) -> None:
		self.children = []

		annihilateNodesMade = 0

		#for n in Annihilate.make(self, solutionState):
		for n in AnnihilateAll.make(self, solutionState):
			annihilateNodesMade += 1
			yield n

		if not annihilateNodesMade:
			cutNodesMade = 0
			#for n in Cut.make(self, solutionState):
			for n in CutAndAnnihilateAll.make(self, solutionState):
				yield n

	def generateChildren(self, solutionState: SolutionState, statesObserved: typing.Set[SolutionState]) -> None:
		filteredChildren = []
		for c in self.generateUnfilteredChildren(solutionState, statesObserved):
			ss = deepcopy(solutionState)
			c.apply(ss)
			#if ss.metrics not in statesObserved:
			if ss in statesObserved:
				continue
			filteredChildren.append(c)

		self.children = filteredChildren

	def __repr__(self) -> str:
		return str(self.__rich_repr__())


class Annihilate(State):
	__slots__ = ()
	nodeAbbrev = "A"

	def __init__(self, sizeToConsume: typing.Any, parent: typing.Optional["State"]) -> None:
		super().__init__(sizeToConsume, parent)

	@classmethod
	def updateMetrics(cls, metrics, partLen, piecesConsumedFromEach):
		metrics.quality.cutsCount += 0
		metrics.quality.joinsCount += 0
		metrics.path.piecesInInitial -= piecesConsumedFromEach
		metrics.path.piecesInTarget -= piecesConsumedFromEach
		metrics.path.lengthToAllocate -= partLen
		metrics.intize()

	@classmethod
	def make(cls, parent: State, solutionState: SolutionState) -> typing.Iterable[State]:
		for partLen in set(solutionState.piecesInitial.sizes) & set(solutionState.piecesTarget.sizes):
			newState = cls(partLen, parent)

			newState.metrics = deepcopy(solutionState.metrics)
			cls.updateMetrics(newState.metrics, partLen, 1)

			#solutionStateNew = deepcopy(solutionState)
			#newState.apply(solutionStateNew)
			yield newState

	def apply(self, solutionState: SolutionState) -> None:
		solutionState.piecesTarget.pop(self.sizeToConsume)
		solutionState.piecesInitial.pop(self.sizeToConsume)

		self.updateMetrics(solutionState.metrics, self.sizeToConsume, 1)

	def __rich_repr__(self) -> RichConsole.RichStr:
		return RichConsole.groups.Fore.lightGreen(self.__class__.nodeAbbrev) + RichConsole.groups.Fore.lightGreen("<") + RichConsole.groups.Fore.cyan(repr(self.sizeToConsume)) + ";" + self.metrics.__rich_str__() + RichConsole.groups.Fore.lightGreen(">")


class AnnihilateAll(Annihilate):
	__slots__ = ()
	nodeAbbrev = "Aa"

	def __init__(self, sizeToConsume: typing.Any, parent: typing.Optional["State"]) -> None:
		super().__init__(sizeToConsume, parent)

	@classmethod
	def make(cls, parent: State, solutionState: SolutionState) -> typing.Iterable[State]:
		duplicated = set(solutionState.piecesInitial.sizes) & set(solutionState.piecesTarget.sizes)
		if duplicated:
			sumLen = sum(partLen for partLen in duplicated)

			newState = cls(None, parent)
			newState.metrics = deepcopy(solutionState.metrics)
			try:
				cls.updateMetrics(newState.metrics, sumLen, len(duplicated))
			except BaseException:
				print(solutionState, solutionState.piecesInitial.sizes, newState.metrics)
				raise

			yield newState

	def getNodesToAnnihilate(self, solutionState):
		print(solutionState.piecesInitial.sizes, solutionState.piecesTarget.sizes)
		return set(solutionState.piecesInitial.sizes) & set(solutionState.piecesTarget.sizes)

	def apply(self, solutionState: SolutionState):
		duplicated = self.getNodesToAnnihilate(solutionState)
		sumLen = sum(partLen for partLen in duplicated)

		res = []
		for l in duplicated:
			a = solutionState.piecesInitial.pop(l)
			b = solutionState.piecesTarget.pop(l)
			res.append((a, b))

		self.updateMetrics(solutionState.metrics, sumLen, len(duplicated))

		solutionState.metrics.intize()
		return res

	def __rich_repr__(self) -> RichConsole.RichStr:
		return RichConsole.groups.Fore.lightGreen(self.__class__.nodeAbbrev) + RichConsole.groups.Fore.lightGreen("<") + self.metrics.__rich_str__() + RichConsole.groups.Fore.lightGreen(">")


class CutAndAnnihilateAll(State):
	"""For visualization purposes only for now"""

	__slots__ = ("chunks", "direction")
	nodeAbbrev = "CA"

	@classmethod
	def updateMetricsCut(cls, metrics: MetricsSet, partLen: FloatType, direction: int) -> None:
		if direction:
			metrics.quality.joinsCount += 1
			metrics.path.piecesInTarget += 1
		else:
			metrics.quality.cutsCount += 1
			metrics.path.piecesInInitial += 1
		metrics.intize()

	@classmethod
	def updateMetricsAnnihilate(cls, metrics: MetricsSet, partLen: int, piecesConsumedFromEach: int, annihilatedChunks: int) -> None:
		metrics.quality.cutsCount += 0
		metrics.quality.joinsCount += 0
		metrics.path.piecesInInitial -= piecesConsumedFromEach
		metrics.path.piecesInTarget -= piecesConsumedFromEach
		metrics.path.lengthToAllocate -= partLen
		metrics.path.chunksToAnnihilate -= annihilatedChunks
		metrics.intize()

	@classmethod
	def enumerateVectorizedCuts(cls, parent: State, solutionState: SolutionState, storage: ObjectsPartsStorage, direction: int) -> typing.Iterator[State]:
		for partLenT, partsOfTheLen in storage.sizes.items():
			part = ObjectsPartsStorage._getFirstUncut(partsOfTheLen)
			if part is None:
				continue

			for chunks in part.getCuttings():
				#print("firstPieceVec", firstPieceVec, "second", second)
				newState = cls(part.len, parent, direction, chunks)
				ss1 = deepcopy(solutionState)
				newState.apply(ss1)
				newState.metrics = ss1.metrics
				yield newState

	@classmethod
	def makeVectorized(cls, parent: State, solutionState: SolutionState) -> typing.Iterable[State]:
		for partLenI, iPartsOfSize in solutionState.piecesInitial.sizes.items():
			iPartFirst = next(iter(iPartsOfSize))
			iPartUncut = ObjectsPartsStorage._getFirstUncut(iPartsOfSize)

			for partLenT, tPartsOfSize in solutionState.piecesTarget.sizes.items():
				if partLenI < partLenT:
					direction = 1
					partLen = partLenT
					tPart = ObjectsPartsStorage._getFirstUncut(tPartsOfSize)
					if tPart is None:
						continue
					iPart = iPartFirst
					cutPart = tPart
				elif partLenI > partLenT:
					direction = 0
					partLen = partLenI
					tPart = next(iter(tPartsOfSize))
					iPart = iPartUncut
					cutPart = iPart
				else:
					continue

				if iPart is None:
					continue

				commonPieces = iPart.pieces & tPart.pieces
				if commonPieces.any():
					#for firstPieceVec in DirVectorPart._getCuttings(commonPieces):
					#	#print(firstPieceL, partLen, secondPieceL)

					#	#metric = addedWaste

					#	print("firstPieceVec", firstPieceVec)
					#	newState = cls(partLen, parent, direction, firstPieceVec)
					#	ss1 = deepcopy(solutionState)
					#	newState.apply(ss1)
					#	newState.metrics = ss1.metrics
					#	yield newState


					#print("firstPieceVec", firstPieceVec)
					newState = cls(partLen, parent, direction, (commonPieces, cutPart.pieces & ~commonPieces))
					ss1 = deepcopy(solutionState)
					newState.apply(ss1)
					newState.metrics = ss1.metrics
					yield newState

	@classmethod
	def makeVectorized(cls, parent: State, solutionState: SolutionState) -> typing.Iterable[State]:
		yield from cls.enumerateVectorizedCuts(parent, solutionState, solutionState.piecesInitial, 0)
		yield from cls.enumerateVectorizedCuts(parent, solutionState, solutionState.piecesTarget, 1)

	@classmethod
	def makeNonVectorized(cls, parent: State, solutionState: SolutionState) -> typing.Iterable[State]:
		for partLenI in solutionState.piecesInitial.sizes:
			for partLenT in solutionState.piecesTarget.sizes:
				if partLenI < partLenT:
					direction = 1
				elif partLenI > partLenT:
					direction = 0
				else:
					continue

				firstPieceL = min(partLenI, partLenT)
				partLen = max(partLenI, partLenT)
				secondPieceL = partLen - firstPieceL

				if firstPieceL < secondPieceL:
				#	continue
					firstPieceL, secondPieceL = secondPieceL, firstPieceL

				#print(firstPieceL, partLen, secondPieceL)

				#metric = addedWaste

				newState = cls(partLen, parent, direction, (firstPieceL,))
				ss1 = deepcopy(solutionState)
				newState.apply(ss1)
				newState.metrics = ss1.metrics

				#solutionStateNew = deepcopy(solutionState)
				#newState.apply(solutionStateNew)
				yield newState

	@classmethod
	def make(cls, parent: State, solutionState: SolutionState) -> typing.Iterable[State]:
		if USE_VECTORIZED:
			return cls.makeVectorized(parent, solutionState)
		else:
			return cls.makeNonVectorized(parent, solutionState)

	def __init__(self, sizeToConsume: typing.Any, parent: State, direction: bool, chunks: NumT) -> None:
		super().__init__(sizeToConsume, parent)
		self.chunks = chunks
		self.direction = direction

	def apply(self, solutionState: SolutionState) -> None:
		piecesDb = solutionState.piecesTarget if self.direction else solutionState.piecesInitial
		part = piecesDb.pop(self.sizeToConsume)
		self.updateMetricsCut(solutionState.metrics, part.len, self.direction)

		chunks = part.split(*self.chunks)

		for c in chunks:
			piecesDb.add(c)

		# annihilate step

		sumLen = 0
		res = []
		annihilatedChunks = 0

		duplicated0 = set(solutionState.piecesInitial.sizes) & set(solutionState.piecesTarget.sizes)

		duplicated = duplicated0

		while duplicated:
			sumLen += sum(partLen for partLen in duplicated)

			for l in duplicated:
				a = solutionState.piecesInitial.pop(l)
				b = solutionState.piecesTarget.pop(l)
				if USE_VECTORIZED:
					annihilatedChunks += np.sum(a.pieces)
				res.append((a, b))

			duplicated = set(solutionState.piecesInitial.sizes) & set(solutionState.piecesTarget.sizes)

		self.updateMetricsAnnihilate(solutionState.metrics, sumLen, len(duplicated0), annihilatedChunks)

		solutionState.metrics.intize()
		return res

	def __rich_repr__(self) -> RichConsole.RichStr:
		return RichConsole.groups.Fore.lightGreen(self.__class__.nodeAbbrev) + RichConsole.groups.Fore.lightGreen("<") + RichConsole.groups.Fore.cyan(repr(self.sizeToConsume)) + RichConsole.groups.Fore.lightGreen(" <- " if self.direction else " -> ") + RichConsole.groups.Fore.cyan(repr(self.chunks)) + ";" + self.metrics.__rich_str__() + RichConsole.groups.Fore.lightGreen(">")


class Root(State):
	__slots__ = ()

	def __init__(self) -> None:
		super().__init__(None, None)

	def __rich_repr__(self) -> RichConsole.RichStr:
		return RichConsole.groups.Fore.lightGreen(self.__class__.__name__)

	def apply(self, solutionState: SolutionState) -> None:
		pass


# from matplotlib import pyplot as plt
_debugNx = nx.DiGraph()


class SearchStrategy:
	__slots__ = ("frontier",)

	def __init__(self, initialState):
		raise NotImplementedError

	def push(self, v):
		raise NotImplementedError

	def pop(self):
		raise NotImplementedError

	def prune(self, bestSolutionMetrics):
		raise NotImplementedError

	def __bool__(self) -> bool:
		return bool(self.frontier)

	def __repr__(self):
		return repr(self.frontier)


class BeFS(SearchStrategy):
	__slots__ = ()

	def __init__(self, initialState: Root) -> None:
		self.frontier = [initialState]

	def push(self, v: State) -> None:
		heappush(self.frontier, v)

	def pop(self) -> State:
		return heappop(self.frontier)

	@classmethod
	def findMiddleElement(cls, pivotA, pivotB, predicate):
		while pivotA + 1 < pivotB:
			pivotMiddle = (pivotA + pivotB) // 2
			if predicate(pivotMiddle):
				pivotB = pivotA
			else:
				pivotA = pivotMiddle
		return pivotMiddle

	def prune(self, bestSolutionMetrics):
		self.frontier = sorted(self.frontier, key=lambda el: el.metrics.quality)
		pivotMiddle = self.findMiddleElement(0, len(self.frontier) - 1, (lambda pivotMiddle: self.frontier[pivotMiddle].metrics.quality > bestSolutionMetrics.quality))
		self.frontier = sorted(self.frontier[:pivotMiddle])


class GBeFS(BeFS):
	__slots__ = ()

	def pop(self) -> State:
		res = self.frontier[0]
		self.frontier = []
		return res

	def prune(self, bestSolutionMetrics):
		pass


class BrFS(SearchStrategy):
	__slots__ = ()

	def __init__(self, initialState: Root) -> None:
		self.frontier = deque((initialState,))

	def push(self, v: State) -> None:
		self.frontier.append(v)

	def pop(self) -> State:
		return self.frontier.popleft()

	def prune(self, bestSolutionMetrics):
		pass


class DFS(SearchStrategy):
	__slots__ = ()

	def __init__(self, initialState: Root) -> None:
		self.frontier = deque((initialState,))

	def push(self, v: State) -> None:
		self.frontier.append(v)

	def pop(self) -> State:
		return self.frontier.pop()

	def prune(self, bestSolutionMetrics):
		pass


class Algo:
	__slots__ = ("initialState", "initialSolutionState", "statesObserved", "solutions", "bestSolutionMetrics")

	def __init__(self, task: typing.Dict[int, int], parts: typing.Dict[typing.Any, NumT]) -> None:
		self.initialState = Root()
		self.initialSolutionState = SolutionState(task, parts)
		self.statesObserved = set()
		self.solutions = []
		self.bestSolutionMetrics = MetricsSet()
		for smName in self.bestSolutionMetrics.__class__.__slots__:
			subMetrics = getattr(self.bestSolutionMetrics, smName)
			for k in subMetrics.__class__.__slots__:
				setattr(subMetrics, k, inf)

		# from analysis
		self.bestSolutionMetrics.quality.cutsCount = self.initialSolutionState.metrics.path.piecesInInitial - 1
		self.bestSolutionMetrics.quality.joinsCount = self.initialSolutionState.metrics.path.piecesInTarget - 1

	def getPath(self, node: State) -> typing.Iterable[State]:
		path = []
		while node:
			path.append(node)
			node = node.parent

		path = list(reversed(path[:-1]))  # without root node
		return path

	def getSolutionStateFromPath(self, path: typing.Iterable[State]) -> SolutionState:
		fs = deepcopy(self.initialSolutionState)

		for node in path:
			node.apply(fs)
		return fs

	def getSolutionFromPath(self, path: typing.Iterable[State]):
		solution = initSolution()
		fs = deepcopy(self.initialSolutionState)

		for node in path:
			if isinstance(node, (CutAndAnnihilateAll, Annihilate)):
				for iP, tP in node.apply(fs):
					print(tP.id, tP.id.origId, tP.id._origId)
					origTargetSize = tP.id._origId[0]  # we encode sizes of targets in their ids as the first component
					if origTargetSize == inf:
						continue
					origTargetInSlotId = tP.id._origId[1]
					solution[origTargetSize][origTargetInSlotId].append((iP.id.origId, iP.len))
			else:
				node.apply(fs)
		return solution

	def getSolutionState(self, node: State) -> SolutionState:
		return self.getSolutionStateFromPath(self.getPath(node))

	def goalTest(self, solutionState: SolutionState) -> bool:
		#return not bool(set(solutionState.piecesInitial.sizes) - set(solutionState.piecesTarget.sizes))
		return not bool(solutionState.piecesInitial.sizes)

	def appendSolution(self, mostPromisingNode: State, fs: SolutionState, frontier: list) -> None:
		heappush(self.solutions, mostPromisingNode)
		needPruneFrontier = False
		for smName in self.bestSolutionMetrics.__class__.__slots__:
			subMetrics = getattr(fs.metrics, smName)
			subMetricsBest = getattr(self.bestSolutionMetrics, smName)
			for k in subMetrics.__class__.__slots__:
				bestKnown = getattr(subMetricsBest, k)
				newOne = getattr(subMetrics, k)
				if bestKnown > newOne:
					bestKnown = newOne
					setattr(subMetricsBest, k, bestKnown)
					needPruneFrontier = True

		if needPruneFrontier:
			frontier.prune(self.bestSolutionMetrics)

	def search(self, strategy: SearchStrategy, limit: int, maxSolutions: int, debug: bool = False) -> None:
		frontier = strategy(self.initialState)

		self.statesObserved = set()

		fs = deepcopy(self.initialSolutionState)
		i = 0
		while True:
			if not frontier:
				break
			mostPromisingNode = frontier.pop()
			print(mostPromisingNode)
			fs = self.getSolutionState(mostPromisingNode)

			#self.statesObserved |= {fs.metrics}
			self.statesObserved |= {fs}
			#_debugNx.add_node(mostPromisingNode.__rich_repr__().plain(), i=i)

			parentState = self.getSolutionState(mostPromisingNode.parent)
			_debugNx.add_edge(repr(parentState), repr(fs), i=i)
			#_debugNx.add_edge(parentState.metrics.__rich_str__().plain(), fs.metrics.__rich_str__().plain(), i=i)

			if self.goalTest(fs):
				_debugNx.add_edge(repr(fs), "solved")
				#_debugNx.add_edge(repr(fs.metrics.__rich_str__().plain()), "solved")
				self.appendSolution(mostPromisingNode, fs, frontier)

				if len(self.solutions) >= maxSolutions:
					break

			networkx.drawing.nx_agraph.write_dot(_debugNx, "./graph.dot")

			i += 1

			mostPromisingNode.generateChildren(fs, self.statesObserved)

			for child in mostPromisingNode.children:
				#print("path", child, child.metrics.path, ">", self.bestSolutionMetrics.path, child.metrics.path > self.bestSolutionMetrics.path)
				#print("quality", child, child.metrics.quality, ">", self.bestSolutionMetrics.quality, child.metrics.quality > self.bestSolutionMetrics.quality)
				if child.metrics.quality > self.bestSolutionMetrics.quality:
					continue
				frontier.push(child)

			if i > limit:
				raise Exception("Fuck")

		print("Nodes expanded", i)

		networkx.drawing.nx_agraph.write_dot(_debugNx, "./graph.dot")

	def __call__(self) -> typing.List[State]:
		self.search(DFS, 100, 1)
		#print("DFS", self.solutions)
		self.search(BeFS, 1000, 100, True)
		print("BeFS", self.solutions)
		return self.solutions

	def __repr__(self):
		return self.__class__.__name__ + "(" + ", ".join((k + "=" + repr(getattr(self, k))) for k in self.__class__.__slots__) + ")"
