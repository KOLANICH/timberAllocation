import typing
from abc import ABC, abstractmethod

import RichConsole


class RichReprable(ABC):
	__slots__ = ()

	@abstractmethod
	def __rich_repr__(self) -> RichConsole.RichStr:
		raise NotImplementedError

	def __repr__(self) -> str:
		return str(self.__rich_repr__())


class RichStringable(ABC):
	__slots__ = ()

	@abstractmethod
	def __rich_str__(self) -> RichConsole.RichStr:
		raise NotImplementedError

	def __str__(self) -> str:
		return str(self.__rich_str__())


class RichStringableAndReprable(RichReprable, RichStringable):
	__slots__ = ()

	def __rich_repr__(self) -> RichConsole.RichStr:
		return RichConsole.groups.Fore.lightGreen(self.__class__.__name__) + RichConsole.groups.Fore.lightGreen("<") + self.__rich_str__() + RichConsole.groups.Fore.lightGreen(">")


class TupleHashable(ABC):
	__slots__ = ()

	@abstractmethod
	def toTuple(self) -> typing.Tuple:
		raise NotImplementedError()

	def __eq__(self, other: "TupleHashable") -> bool:
		#if not isinstance(other, tuple):
		ot = other.toTuple()
		#else:
		#	ot = other
		return self.toTuple() == ot

	def __hash__(self) -> int:
		return hash(self.toTuple())


class StructuredComparable(ABC):
	__slots__ = ()

	@abstractmethod
	def _coIter(self, other) -> typing.Iterable[typing.Tuple[str, typing.Any, typing.Any]]:
		raise NotImplementedError

	def _doComparison(self, other: "StructuredComparable", predicate: typing.Callable) -> bool:
		res = True
		for k, sv, ov in self._coIter(other):
			res = res and predicate(sv, ov)
		return res

	def __lt__(self, other: "StructuredComparable") -> bool:
		return self._doComparison(other, lambda sv, ov: sv < ov)

	def __gt__(self, other: "StructuredComparable") -> bool:
		return self._doComparison(other, lambda sv, ov: sv > ov)

	def __le__(self, other: "StructuredComparable") -> bool:
		return self._doComparison(other, lambda sv, ov: sv <= ov)

	def __ge__(self, other: "StructuredComparable") -> bool:
		return self._doComparison(other, lambda sv, ov: sv >= ov)
