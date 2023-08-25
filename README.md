This a relatively simple sudoku solver I have written, largely for myself, but it may be of use to you too!

My main aim is to have it solve sudoku puzzles as a human would, just faster and better at keeping track of possibilities, and tell you exactly what it has done so you can follow along and learn.

It can also act as a helper: the .solve(step=True) option will solve just one more cell, to give you a helping hand.

Encyclopedia of my terms relative to usual terms:
single: a True cell which is only True in one of it's houses. Corresponds to either naked single if house is row-col or hidden single if house is val-row, val-col, val-sqr.

n-tuple: n squared True cells, at the intersections of n houses of n types (e.g the intersection n rows and n vals, or n values in n sqrs). Equivalent to hidden tuples, naked tuple, or a basic fish. My general definition sounds complicated, but imagine rotating an X-wing into the paper - it would become a hidden doublet or a naked doubled, depending on which axis you rotated it about.

Intersection/ommision/locked/Pointing/claiming: Because val-sqr houses share multiple cells with val-row and val-col houses, if all of one house's Trues lie inside other house, the other house cannot have any other Trues. No fancy stuff here, this is the same as the 2d version.

Still to do:
-Actually write the program.
-Think of an acronym for a name