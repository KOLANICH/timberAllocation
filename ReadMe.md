timberAllocation.py [![Unlicensed work](https://raw.githubusercontent.com/unlicense/unlicense.org/master/static/favicon.png)](https://unlicense.org/)
====================
![GitLab Build Status](https://gitlab.com/KOLANICH/timberAllocation.py/badges/master/pipeline.svg)
[![TravisCI Build Status](https://travis-ci.org/KOLANICH/timberAllocation.py.svg?branch=master)](https://travis-ci.org/KOLANICH/timberAllocation.py)
![GitLab Coverage](https://gitlab.com/KOLANICH/timberAllocation.py/badges/master/coverage.svg)
[![Coveralls Coverage](https://img.shields.io/coveralls/KOLANICH/timberAllocation.py.svg)](https://coveralls.io/r/KOLANICH/timberAllocation.py)
[![Libraries.io Status](https://img.shields.io/librariesio/github/KOLANICH/timberAllocation.py.svg)](https://libraries.io/github/KOLANICH/timberAllocation.py)

A not very working project. And completely unfinished paper for now full of bullshit.

Sometimes one needs to make something not very loaded from timber, but has some wastes spared from previous construction.

Assumptions:
* You have catalogued all the pieces. You have a TSV file, each row of which contains an id of a log and LENGTHS OF ITS PIECES (NOT full length, but lengths of all pieces sequentially, if the log is already made of multiple pieces and can be disassembled into them without cutting), sequentially. You can find a physical log given its id. For example ids are written on logs in large font with a permanent marker or paint (pen, nail scribing and pencil proven to be unreliable).
* You can cut any log into 2 pieces.
* Your construction tolerates pieces that are not pieces, but multiple logs stacked and fixed with nails, boards, metal plates and similar shit.
* All the pieces have the same section.

This set of scripts aims to help in this situation. It tries to compute the optimal points to cut the existing timber. Though doesn't really help a lot: the results are suboptimal, inhuman (all the results were discarded by the human expert on the basis "I don't want to join these pieces") and practically not very usable.

Optimality criteria (in an ideal case):
* The least of wastes is produced. A waste is an unused piece of a log that has been cut.
* The count of logs cut is minimal.
* The count of joins is minimal.


This is similar to both stock cutting problem (in which we cut pieces of identical size into pieces of different sizes in order to minimize waste and joins are not allowed) and ... problem (in which we join predefined different pieces to get ieces of the same size), both of which are known to be NP-hard.


Problem initial formalization as a search in a graph
--------------------------------------------------

We can formulate the problem as a search in the graph of states, where each state is:
* a set of available pieces of timber;
* slots. A slot is an amount of timber yet to be allocated. All the timber in a slot is joined into the same target piece.
* A `waste` variable.

When there are no empty slots, we have found a solution.

There are 2 possible kinds of actions in each state:
* `Cut` a piece into 2 pieces. Removes a piece from the set and adds 2 pieces which sum of lengths is equal to the length of the cut piece. Generates `waste` in amount of a cut piece.
* `Join` a piece and fill a slot with it. Removes a piece and decreases amount in the slot by it. Consumes `waste` in amount of a consumed piece.

The initial state is when there is no waste and the pieces are the initial pieces.
The target state is when there is no empty slots.

If all the cut pieces consumed there is no waste.

This formulation of the task is useless because it is impossible to search in the graph for pieces > 3. But it gives us important insights we can use to reformulate the task.

The insights:
* The problem always have a solution, if there is enough timber provided to occupy all the slots. We can always cut the logs into very (probably infinitily) small equal pieces and then reassemble new logs from them.
* All the pieces resulting from cuts persist untill consumed by joins.
* Thought we can cut any amount, we only need to cut the amounts which are differrences between the logs and the slots, because cuts that don't make us closer to filling a slot are useless. This allows us to search in a discrete space with a finite branching factor in each node!
* `Cut` and `Join` actions are MOSTLY (but not wholly) permuttible (when we say `permuttible` here, we mean that `a2(a1(s)) === a1(a2(s))`)
    * `Join` can always follow a `Cut` producing a taken piece, but not opposite.
    * `Cut` actions are always permuttible.
    * `Join` actions can always be deferred to the end. So we ideally can put all the cuts to the beginning and joins to the end. But doing so is harmful in this formulation because a `Join` eliminates a piece and possibly a slot, reducing branching factor, but a `Cut` adds a piece increasing branching factor.
* Though our target solution requres the software to tell us the ids of pieces to cut and join, we cannot consider pieces as distinguishable in states because it creates redundant states.
* The most importantly, `Join` is an inverse of `Cut` in some sense. The algo can cut a piece and then join the cut pieces into a single one, essentially doing nothing except producing redundant cuts.

All of these produce too many redundant paths, which are problematic to avoid, even if one avoids them, he misses the shortcuts they provide. But the fact that `Cut` and `Join` are inverses of each other and that we can defer `Join`s means we can do a bidirectional search. And in formulation for bidirectional search the problem is much easier!

Problem reformalization 1
-------------------------

Each state is a set of pieces of timber.

There are 2 initial states:
* All the pieces are the initial pieces.
* All the pieces are the target pieces and an additional piece for `unallocated`, if there may be any `unallocated` timber.

There are only 1 possible kind of actions in each state:
* `Cut` a piece into 2 pieces. Removes a piece from the set and adds 2 pieces which sum of lengths is equal to the length of the cut piece.

We run a bidirectional search from these 2 states, and somewhen they meet in the same state. They must meet - as we have seen the solution always exist.

The insights:
* All the pieces resulting from cuts persist until consumed by joins.
* Let `M` is the middle state the searches meet in. It can contain either completely unknown pieces, or it can contain some pieces the same for either of initial states.
* If it contains some pieces from the initial states, since the pieces are indistinguishable and cuts are permuttible, we can claim they are the ones from the initial states. And we should do that, it will not make any cuts and any waste, and our goal is to minimize cuts and waste.
* If an unknown piece is longer than a piece in an initial state, it can be cut into the piece in the initial state and a remainder. Since the cuts are permuttible, this cut can be made from the initial state itself. Since we can only cut and the cut pieces persist unless further cut, this cut must be eventually made, sow we should to it now. The only variability here is the choice of the piece to cut. But the cut has to be done.
* So the optimal middle state minimizing cuts contains each piece from either one of initial states or both, if it is possible.


Problem reformalization 2
-------------------------
Each state is 2 sets of pieces of timber + 2 sets of cuts. Pieces of timber sets are the difference between the middle state and the frontier states of the bidirectional search. Sets of cuts record the path.

Initial state: sets of cuts are empty, sets of pieces are are initialized as in bidirectional search formalization.
Target state: sets of pieces are empty, sets of cuts contains the solution.

There are 2 possible kind of actions in each state:
* `Annihilate` 2 pieces of equal length from the opposite sets, removing them from our consideration - they persist untill the middle state of the bidirectional search.
* `Cut` a piece of length `l0` into 2 pieces of lengths `l1` and `l2`, such as `l1 + l2 = l0` and `l1 >= l2`. Removes the piece `l0` from the set and adds 2 pieces `l1` and `l2`.

Insigths:
* On each step sum of lengths of pieces in the sets is the same.
* `Annihilation`s are always permuttible with each other.
* `Cut`s are still permuttible and still can be postponed. Each `cut` can generate waste (if it is possible to have waste) and by itself is undesireable.
* `Cut`s can only contract lengths of individual pieces.
* Total length in opposite sets is equal and can only contract. So any `annihilation` brings us closer to the solution.
* Each `annihilation` also removes `cut` branches that can generate waste.
* Let `uncut` be amount of wood kept uncut, more precisely a piece is uncut if it is the same in the initial state and the final state. Then `waste = unallocated - uncut`. So to minimize waste we must maximize `uncut`.
* `Annihilate` nodes are idempotent and can always be applied. That allows to apply all annihilation operations possible in the curent state in a single node `AnnihilateAll`.
* If it is possible to make an `annihilation` of pieces or cutting a piece other than an annihilated one, this `cutting` and `annihilation` can be pemutted. It is though useful to do annihilation first.


Theorem: Only the steps having any of initial pieces determine the amount of waste. Whatever we do on the other steps, we cannot decrease amount of waste.
Proof:
	While we have initial pieces, we can increase amount of waste by cutting them. While we don't, we cannot.
Corrolary:
	As soon as we either annihilate or cut all the initial pieces we have fixed the amount of waste.

Let's call a cut `necessary` if we have not solved the problem and we have no pieces to annihilate, so the only action we can do is to cut.
Let's call a cut `final` if after an annihilation has followed this cut, there will be no pieces left.

Theorem: Each necessary nonfinal `cut` in one side can either increment count of `necessary` cuts either by 1 or by 2
Proof:
We have 3 options:
1. If a necessary cut produces a piece of the same size that the one of the existing pieces, we would be able to annihilate them. The resulting piece will either be annihilated somewhen because there must be a solution (this means the cut will has to be done from the opposite side of the search to match it), or will be further cut.
2. If a necessary cut produces no pieces that can be annihilated, we have to cut 2 other pieces to match the sizes of the resulting pieces, as in 1. It is not possible to already have the pieces matching the sizes of any of the resulting pieces, otherwise this either would have been option 1 or it .
3. The final cut must produce the pieces that will be annihilated all, so it increments the count of further needed cuts by 0.

So it makes sense to combine `Annihilate`s to `AnnihilateAll` and `AnnihilateAll` to the previous `Cut`, having only `CutAndAnnihilateAll` node.


Problem reformalization 3
-------------------------
The same as in 2, but

There are only 1 possible kind of actions in each state:
* `AnnihilateAndCut` a piece of length `l0` into 2 pieces of lengths `l1` and `l2`, such as `l1 + l2 = l0` and `l1 >= l2` and either `l1` or `l2` piece is present in the opposite set, so is guaranteedly annihilated. Further we call this operation a `cut`. Removes at least one piece from the opposite set and can also remove pieces from own set.

Insights:
* upper bound on sum of counts of cuts from the both sides is `<count of pieces from both sides> - 2`. Proof: `(count of pieces from initial side - 1) + (count of pieces from target side - 1)` nonfinal cuts consuming 1 piece from either side and 1 final cut consuming both pieces at worst.
* We can also predict the upper bound on count of pieces in the middle state in bidi search: `(count of pieces from initial side - 1) + (count of pieces from target side - 1)` nonfinal cuts add 1 piece to the middle state, and the final cut adds 2, so `count of pieces from initial side + count of pieces from target side - 1`.

We have a computation vs cuts vs waste tradeoff. Selecting the middle state to be:
* near the initial state minimizes count of cuts and possibly waste (in presence of redundant material) at expense of greater count of joins and computation. This can be achieved by generating cuts on initial side only if it is not possible to generate cuts on target side.
* near the target state minimizes count of joins at expense of greater count of cuts and waste (in presence of redundant material). This can be achieved by generating cuts on target side only if it is not possible to generate cuts on initial side.
* the one balancing cuts and joins from both sides.
* the one minimizing search cost at expense of both cuts and joins relatively to the minimal cuts/joins states. If we had a constant branching factor, this would have been the previous one, but count of children in each node is not easily predictable since cuts are subjet to constraints based on pieces sizes and they change.

Depending on the problem, sometimes these variants of the middle state are the same, sometimes they are different.

But we are still plagued with redundant paths because of permuttability, but we cannot prioritize cuts on size because some cuts possible in the state are mutually excluding. So we need to group cuts into the groups


A simple greedy nonoptimal non-search algo
------------------------------------------

A little step away from the solving of the problem with graph search is a simple algo derived from the reformalization 3.

1. We have 2 arrays of parts as described. For example, `parts_I` is array of initial parts, `parts_T` is the array of target parts. Each part has an id, length and a direction from the set `{"I", "T"}` which means the direction of the search and matches the direction of the array.
2. We concatenate them: `parts_I || parts_T`.
3. We sort this array `O(n*log(n))`.
4. We crawl that array for `O(n)` in 2 passes, annihilating adjacent elements of opposite directions.
    a. In the first pass we only annihilate elements that are equal. They must be adjacent since the arrays are sorted.
    a. In the second one we partially annihilate elements that are not equal, leaving only the larger ones.

5. We repeat this until the array is not empty.
6. Then we recover the solution from the set of cuts.

Though it is still desireable to use a search algo since
    a. it can find better solutions
    b. can be reused for the modified problem, where pieces have not only lengths, but also fixed widths, and we want to make multiple rectangular frames of them having known dimensions.


Vector factorization
--------------------

Let's do some linear algebra.

Let we have a row-vector of positive numbers `x`. If we can factor it into a product `b . U === x`, where `b` is a row-vector of minimal size, that will be our basis, `U` is the "upper-triangular-like" (mostly upper triangular, but with holes in the upper triangle and elements in the lower triangle when necessary) matrix, which elements are either 1 or 0, then we can probably have a common representation between the numbers we can operate in.

The first insight, that since the middle state has at most `(count of pieces from initial side + count of pieces from target side) - 1` pieces, `b` is guaranteedly smaller than `x` at least by 1 if 'x' is made of numbers matching both sides of the bidi search. But in general case it is not the case and for vectors of `n` elements in the worst case we will have basis of `n` elements, if all the numbers are lineary independent. In the case of cuts we have `n-1` at worst since they are guaranteedly lineary dependent.

Factoring is straightforward and is just a mixture of Gram-Schmidt orthogonalization, Gaussian elimination and Euclidus algorithm.

1. The largest number should have all its components present.
2. The second largest should have all its components except the one differing it to the largest number present.

So we repeat the following loop until the residues goes to 0. `i` is from `-1` to `-n`:
1. We find the largest unique number `L` and the second largest number `l`. If there is only one number, we set `L` to it and `l` to `0`;
2. We differ them `e[i] = L - l`. This will be our basis number. We assign `0` to the second largest and `1` to the largest for `i`th component.
3. We find all the elements in the vector matching the second largest element and just set their `i`th component to `0` by definition.
4. For other elements greater than the current basis number we subtract it and set `1` for the component, otherwise we set `0`.

Some basis numbers can be duplicated. This is because we allow components to be only 1 and 0 - either taken by part or not.

We can also apply the same procedure to the resulting basis again and multiply the matrices ... and again and again. This will give us a smaller basis, but the matrices will be non-binary. Though visualization of the matrix looks like it may be helpful for unsupervised learning. Needs some additional research.


Problem reformalization 4
-------------------------

OK, now instead of raw lengths of the pieces we have binary vectors. Taking one of the components from the piece doesn't exclude other components. So we can deal with permuttibility by enumerating all the combinations of components that can be taken. This increases counts of children in one node but completely eliminates permutations between nodes of the same part - once a part is cut, we can never cut it again anymore. Permutations between nodes of different parts can be eliminated too - now we always know which chunks we are allowed to cut from a part, independent from the history and side branches.

** Fucking shit - it doesn't help. Probably a bug in the algo. BrFS (breadth-first search) that has worked for the small example has become intractable. GBeFS (greedy best-first search) doesn't achieve the solution. Even if we don't do combinations and mirror the old behavior with cutting part-by-part.**

Solution strategy
-----------------

