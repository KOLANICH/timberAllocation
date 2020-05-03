import typing

import numpy as np


def getResultVector(initialLengths, finalLengths):
	return np.array(tuple(initialLengths) + tuple(finalLengths))


def getMiddleStateMaxSize(initialLengths, finalLengths):
	return len(initialLengths) + len(finalLengths) - 1


def vectorFactorization(vec: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
	"""I have probably reinvented a wheel. I have searched the Internet and haven't found this kind of factorization.
	Factors a vector vec into the product of an "upper-triangular-like" (mostly upper triangular, but with holes and elements in the lower triangle when necessary) made of 1es matrix `U` and a basis row-vector `b`, so b @ U === vec.
	"""

	vec = np.array(vec)

	dtype = vec[0].__class__  # so works with Decimal too. In this case arrays dtype is `object`.

	basis = np.full(vec.shape[0], dtype(np.nan))
	matrix = np.zeros((basis.shape[0], vec.shape[0]), dtype=bool)

	i = -1

	basisSize = 0
	while (vec > 0).any():
		remainingSizes = sorted(set(vec))
		if len(remainingSizes) >= 2:
			secondLargest, largest = remainingSizes[-2:]
			basisVec = largest - secondLargest
		else:  # 1 size only
			basisVec = remainingSizes[0]
			secondLargest = 0

		basis[i] = basisVec

		for j, s in enumerate(vec):
			if s == secondLargest:
				matrix[matrix.shape[0] + i, j] = 0
			else:
				if s >= basisVec:
					matrix[matrix.shape[0] + i, j] = 1
					vec[j] -= basisVec
				else:
					matrix[matrix.shape[0] + i, j] = 0

		i -= 1
		basisSize += 1

	return basis[-basisSize:], matrix[-basisSize:, :]


def minimalReprToGraph(shared, mat, initialLengths, finalLengths):
	import networkx

	g = networkx.DiGraph()
	for n in initialLengths:
		if isinstance(n, float) and n.is_integer():
			n = int(n)
		g.add_node(n, color="green")
	for n in finalLengths:
		if isinstance(n, float) and n.is_integer():
			n = int(n)
		g.add_node(n, color="red")

	for i, l in enumerate(initialLengths + finalLengths):
		if isinstance(l, float) and l.is_integer():
			l = int(l)
		for j, sL in enumerate(shared):
			sL = float(sL)
			if sL.is_integer():
				sL = int(sL)
			if sL != l and mat[j, i]:
				g.add_edge(l, sL)
	return g
