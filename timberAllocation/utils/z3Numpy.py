__all__ = ("ndarray", "array")

import sys

import numpy as np
import z3

z3Internal = sys.modules["z3.z3"]

_get_argsBackup = z3Internal._get_args


def _get_argsMonkeyPatched(args):
	try:
		argsNew = []
		for arg in args:
			if isinstance(arg, np.ndarray):
				argsNew.extend(arg.flatten())
			else:
				argsNew.append(arg)
		return _get_argsBackup(argsNew)
	except BaseException:
		return _get_argsBackup(args)


z3Internal._get_args = _get_argsMonkeyPatched


def z3OptimizeDecorate(f):
	def optimizeF(self, *args, **kwargs):
		if isinstance(args[0], np.ndarray):
			for el in args[0].flat:
				f(self, el, *args[1:], **kwargs)

	return optimizeF


z3.Optimize.minimize = z3OptimizeDecorate(z3.Optimize.minimize)
z3.Optimize.maximize = z3OptimizeDecorate(z3.Optimize.maximize)


def generateFiniteLenInts(name, count, tp, ctx):
	tp = np.dtype(tp)
	bitsCount = tp.itemsize * 8
	return [z3.BitVec(name + "__" + str(i), bitsCount) for i in range(count)]


def generateFiniteLenFloats(name, count, tp, ctx):
	tp = np.dtype(tp)
	fpSort = floatTypes[tp.itemsize]
	if isinstance(fpSort, tuple):
		fpSort = z3.FPSort(*fpSort)
	return [z3.FP(name + "__" + str(i), fpSort) for i in range(count)]


typesRemapping = {
	np.bool_: lambda name, count, tp, ctx: z3.BoolVector(name, count, ctx),
	bool: lambda name, count, tp, ctx: z3.BoolVector(name, count, ctx),
	int: lambda name, count, tp, ctx: z3.IntVector(name, count, ctx),
	float: lambda name, count, tp, ctx: z3.RealVector(name, count, ctx),
}

floatTypes = {
	1: (4, 4),
	#2: (5, 11),
	2: z3.FloatHalf(),
	#4: (8, 24),
	4: z3.FloatSingle(),
	#8: (11, 53),
	8: z3.FloatDouble(),
	10: (15, 63),
	#16: (15, 111),
	16: z3.FloatQuadruple(),
	32: (19, 237),
}

supertypesRemapping = {
	np.integer: generateFiniteLenInts,
	np.floating: generateFiniteLenFloats,
}

backRemapSortsToTypes = {
	z3.Z3_BOOL_SORT: bool,
	z3.Z3_INT_SORT: int,
	z3.Z3_REAL_SORT: float,
}


def _getBackRemapSortsToDTypes(x):
	sort = x.sort()
	if isinstance(x, z3.FPRef):
		size = sort.sbits() + sort.ebits()
		return np.dtype("float" + str(size))
	elif isinstance(x, z3.BitVecRef):
		return np.dtype("int" + str(sort.size()))
	else:
		return np.dtype(backRemapSortsToTypes[sort.kind()])


def _makeOurNdArray(underlyingArray, shape, order=None):
	res = ourNdArray(shape=shape, order=order, dtype=object)
	res.flat = np.array(underlyingArray, order=order, subok=True)
	return res


class ndarray(np.ndarray):
	__slots__ = ()

	def _applyElementwise(self, another, pred):
		return _makeOurNdArray([pred(s, a) for (s, a) in np.broadcast(self, another)], self.shape)

	def __eq__(self, another):
		return self._applyElementwise(another, lambda s, a: s == a)

	def __ne__(self, another):
		return self._applyElementwise(another, lambda s, a: s != a)

	def __ge__(self, another):
		return self._applyElementwise(another, lambda s, a: s >= a)

	def __le__(self, another):
		return self._applyElementwise(another, lambda s, a: s <= a)

	def __gt__(self, another):
		return self._applyElementwise(another, lambda s, a: s > a)

	def __lt__(self, another):
		return self._applyElementwise(another, lambda s, a: s < a)

	def __and__(self, another):
		if isinstance(self.flat[0], z3.BoolRef):
			return self._applyElementwise(another, z3.And)
		else:
			return self._applyElementwise(another, lambda s, a: s & a)

	def __or__(self, another):
		if isinstance(self.flat[0], z3.BoolRef):
			return self._applyElementwise(another, z3.Or)
		else:
			return self._applyElementwise(another, lambda s, a: s | a)

	def __xor__(self, another):
		if isinstance(self.flat[0], z3.BoolRef):
			return self._applyElementwise(another, z3.Xor)
		else:
			return self._applyElementwise(another, lambda s, a: s ^ a)

	def __invert__(self, another):
		if isinstance(self.flat[0], z3.BoolRef):
			return self._applyElementwise(None, lambda s, a: z3.Not(s))
		else:
			return self._applyElementwise(another, lambda s, a: ~s)

	def toN(self):
		dt = _getBackRemapSortsToDTypes(self.flat[0])
		res = np.ndarray(shape=self.shape, dtype=dt)
		uintView = res.view(dtype=np.dtype("u" + str(dt.itemsize)))
		for i, elRes in enumerate(self.flat):
			if isinstance(elRes, z3.RatNumRef):
				res.flat[i] = elRes.as_fraction()
			elif isinstance(elRes, z3.z3.FPNumRef):
				uintView.flat[i] = fpNumRefToFloatInt(elRes)
			else:
				res.flat[i] = elRes
		return res


ourNdArray = ndarray


def arrayCreateFromNameAndShape(name, shape, dtype=bool, order=None, ctx=None):
	ctor = typesRemapping.get(dtype, None)
	if ctor is None:
		for superType, ctorCandidate in supertypesRemapping.items():
			if issubclass(dtype, superType):
				ctor = ctorCandidate
				break

	flat = ctor(name, np.product(shape), dtype, ctx=ctx)
	return _makeOurNdArray(flat, shape, order)


def arrayCreateFromExistingArray(*args, **kwargs):
	res = np.array(*args, **kwargs)
	res1 = ourNdArray(shape=res.shape, order=None, dtype=res.dtype)
	res1.flat = res.flat
	del res
	return res1


def array(*args, **kwargs):
	if isinstance(args[0], str) or isinstance(kwargs.get("name"), str):
		return arrayCreateFromNameAndShape(*args, **kwargs)
	else:
		return arrayCreateFromExistingArray(*args, **kwargs)


array.__wraps__ = arrayCreateFromNameAndShape

simplifyBackup = z3.simplify


def newSimplify(a, *arguments, **keywords):
	if isinstance(a, np.ndarray):
		return _makeOurNdArray([simplifyBackup(e, *arguments, **keywords) for e in a.flat], a.shape)
	else:
		simplifyBackup(a, *arguments, **keywords)


newSimplify.__wraps__ = simplifyBackup
z3.simplify = newSimplify


def fpNumRefToFloatInt(a):
	significandBitsWithoutSign = a.sbits() - 1
	signRemoveMask = (1 << significandBitsWithoutSign) - 1
	# sign exponent significand
	return (
		(
			(int(a.isNegative()) << a.ebits())
			|
			a.exponent_as_long(biased=True)
		) << significandBitsWithoutSign
	) | (a.significand_as_long() & signRemoveMask)

if not hasattr(z3.IntNumRef, "__int__"):
	z3.IntNumRef.__int__ = z3.IntNumRef.as_long

if not hasattr(z3.FPNumRef, "__float__"):

	def FPNumRefToNumpyFloat(self):
		dtF = np.dtype("f" + str((self.ebits() + self.sbits()) // 8))
		dtI = np.dtype("u" + str(dtF.itemsize))
		f = np.ndarray((1,), dtype=dtF)
		i = f.view(dtype=dtI)
		i[0] = fpNumRefToFloatInt(self)
		return f[0]

	def numpyFPNumRefToFloat(self):
		return float(self.toNumpyFloat())

	z3.FPNumRef.toNumpyFloat = FPNumRefToNumpyFloat
	z3.FPNumRef.__float__ = numpyFPNumRefToFloat


z3IntValBackup = z3.IntVal


def z3IntValPatched(v, *args, **kwargs):
	if isinstance(v, np.integer):
		v = int(v)
	return z3IntValBackup(v, *args, **kwargs)


z3IntValPatched.__wraps__ = z3IntValBackup
z3Internal.IntVal = z3.IntVal = z3IntValPatched

z3FPValBackup = z3.FPVal


def z3FPValPatched(v, *args, **kwargs):
	if isinstance(v, np.integer):
		v = int(v)
	elif isinstance(v, np.floating):
		v = float(v)
	return z3FPValBackup(v, *args, **kwargs)


z3FPValPatched.__wraps__ = z3FPValBackup
z3Internal.FPVal = z3.FPVal = z3FPValPatched

z3RealValBackup = z3.RealVal


def z3RealValPatched(v, *args, **kwargs):
	if isinstance(v, np.integer):
		v = int(v)
	elif isinstance(v, np.floating):
		v = float(v)
	return z3RealValBackup(v, *args, **kwargs)


z3RealValPatched.__wraps__ = z3RealValBackup
z3Internal.RealVal = z3.RealVal = z3RealValPatched


def Abs(self):
	return z3.If(self >= 0, self, -self)


if not hasattr(z3.ArithRef, "__abs__"):
	z3.ArithRef.__abs__ = Abs

if not hasattr(z3.FPRef, "__abs__"):
	z3.FPRef.__abs__ = Abs

if not hasattr(z3.BoolRef, "__int__"):

	def z3BoolRefToInt(self):
		return z3.If(self, 1, 0)

	z3.BoolRef.__int__ = z3BoolRefToInt

if not hasattr(z3.BoolRef, "__add__"):

	def z3BoolRefAdd(self, another):
		return self.__int__() + another.__int__()

	z3.BoolRef.__add__ = z3BoolRefAdd

ModelRefGetItemBackup = z3.ModelRef.__getitem__


def __getitem__Patched(self, idx):
	if isinstance(idx, np.ndarray):
		dt = _getBackRemapSortsToDTypes(idx.flat[0])
		res = ourNdArray(shape=idx.shape, dtype=object, subok=True)  # our ndarray
		for i, el in enumerate(idx.flat):
			res.flat[i] = ModelRefGetItemBackup(self, el)
		return res
	else:
		return ModelRefGetItemBackup(self, idx)


z3.ModelRef.__getitem__ = __getitem__Patched


_to_float_str_backup = z3Internal._to_float_str


def _to_float_str(*args, **kwargs):
	if isinstance(args[0], np.number):
		args = list(args)
		args[0] = float(args[0])

	return _to_float_str_backup(*args, **kwargs)


z3Internal._to_float_str = _to_float_str


"""
import numpy as np
import z3
from timberAllocation.utils import z3Numpy

x = z3Numpy.array("x", (2, 2), dtype=np.float16)
y = z3Numpy.array("y", (2, 2), dtype=np.float16)

s = z3.Solver()
s.add(x @ y == np.identity(x.shape[0]))
s.add(abs(x) < 10)
s.add(abs(y) < 10)
s.add(abs(x) > 0.1)
s.add(abs(y) > 0.1)

if s.check() == z3.sat:
	mod = s.model()
	xx = mod[x]
	yy = mod[y]
	print(xx, yy, z3.simplify(xx @ yy))
	print(xx.toN(), yy.toN(), xx.toN() @ yy.toN())
"""
