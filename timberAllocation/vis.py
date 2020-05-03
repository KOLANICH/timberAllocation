import colorsys
import typing
from collections import defaultdict
from decimal import Decimal

from defaultlist import defaultlist
from numpy import float32, float64

import RichConsole
from RichConsole import RichStr

from . import FloatType, NumT, PartId


def makeLogPart(partsColors, zoom: float, chunkL: NumT, label: str, partId: PartId, char: str = "-") -> typing.Union[RichStr, str]:
	size = int(round(chunkL * zoom) - len(label))
	halfSize = int(round(size / 2))
	text = char * halfSize + label + char * (size - halfSize)
	if partId is not None:
		return RichConsole.RGBColor(None, *partsColors[partId.origId])(text)
	else:
		return text


def makeLog(parts: typing.Iterable[RichStr]) -> RichStr:
	return "|" + RichConsole.rsjoin("|", parts) + "|"


def visualizeSolution(solution, partsColors, zoom=FloatType(0.75)) -> None:
	for slotL, combineds in sorted(solution.items(), key=lambda x: x[0], reverse=True):
		totalRealL = FloatType(0)
		for combined in combineds:
			l = []
			for chunk in combined:
				chunkId, chunkL = chunk
				totalRealL += chunkL
				label = repr(chunkId) + "," + str(chunkL)
				l.append(makeLogPart(partsColors, zoom, chunkL, label, chunkId))
			if totalRealL < slotL:
				lack = slotL - totalRealL
				l.append(rgbBlack(makeLogPart(partsColors, zoom, lack, str(lack), None, char=".")))
			print(slotL, makeLog(l))


rgbBlack = RichConsole.RGBColor(0, 0, 0, bg=True)  # simple bg colors don't work in Jupyter


def visualizeCutting(parts, solution, partsColors, zoom=FloatType(0.75)) -> None:
	cutting = defaultdict(list)
	uncut = set(parts)
	for slotL, combineds in solution.items():
		for combined in combineds:
			for chunk in combined:
				chunkL = chunk[1]
				cutting[chunk[0]].append(chunk[1])

	for partId, chunkLs in cutting.items():
		l = []
		for chunkL in chunkLs:
			label = str(chunkL)
			l.append(makeLogPart(partsColors, zoom, chunkL, label, partId))
		waste = parts[partId.origId] - sum(chunkLs)
		if waste:
			l.append(rgbBlack(makeLogPart(partsColors, zoom, waste, str(waste), partId, char="&")))
		print(partId, parts[partId.origId], makeLog(l))
		uncut -= {partId}

	for partId, partL in sorted(((partId, parts[partId.origId]) for partId in uncut), key=lambda x: x[1], reverse=True):
		print(partId, parts[partId.origId], RichConsole.groups.Underline.underline((makeLog((makeLogPart(partsColors, zoom, partL, str(partL), partId, char="="),)))))


def generateColors(parts: typing.Dict[PartId, FloatType]) -> typing.Dict[PartId, typing.Tuple[int, int, int]]:
	step = 1.0 / len(parts)
	partsColors = {}
	for i, p in enumerate(parts):
		partsColors[p.origId] = tuple((int(round(255 * c)) for c in colorsys.hsv_to_rgb(i / 2 * step, 1 - 0.3 * (i % 3 == 0), 0.5 + 0.5 * (i % 2))))
	return partsColors


def visualizeAll(solution, parts, partsColors, zoom=FloatType(0.75)):
	print("\nSolution")
	visualizeSolution(solution, partsColors, zoom)
	print("\nCutting")
	visualizeCutting(parts, solution, partsColors, zoom)
