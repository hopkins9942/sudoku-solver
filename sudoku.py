"""
Current thoughts:
when solving, once full possibilities are put in, you only need to look at that
so take solved cells out of poss, and track solved board seperately

self.board tells what is solved (in 3d: val,row,col)
a solved cell makes all cells which share a house with it zero,
its value doesn't matter but I'll set to True for comfort
self.poss tells what is possibile in unsolved houses


The Plan:
A sudoku board is not a 2d board, it is a 9 by 9 by 9 cube of binary values. 
Considering the board like this allows symmetries to become clear - something
I wish to use, because the same function can be used to find both X-wings and
hidden doubles. Additionally it reduces the amount of frankly silly sudoku 
terminology I have to either learn or use.

Throughout this program I will use i, j, and k to refer to rows, columns, and 
numbers respectively. What I mean by a sudoku board being in fact a 9 by 9 by 9
cube of binary values is that this is the most useful representation of the 
board. First consider that a solved cell in the 2d board is just a cell where
only there is only one possibile number - now the board is a 9 by 9 board of
cells with a list of possibilities. This is conveniently represented by a 9 by 
9 by 9 array, where two of the dimentions are the rows and columns of the 
original board, and along the third is a length 9 array of binary values, 0 if 
the index of that value is not a posibility of that cell, 1 if it is. For the
convenience of how numpy represents 3d arrays, I have chosen the first
dimention for this, so if poss[k,i,j] = True, k+1 (+1 due to zero indexing) is
a possibility for the (i,j)th cell on the original board.

In this representation, instead of ensuring there is one of each number in each
house, one ensures that there is a single True in each house, where there are
4 types of house: row-val, col-val, sqr-val, single value versions of 2d 
houses, and row-col, all values in one i,j cell. Solver looks for inconsisties 
which allow Trues to be set to Falses where they cannot be in solution.


"""

import numpy as np



class Puzzle:
    
    #setup
    def sqr(i,j):
        """Gives index of sqr house containing cell ij
        Not made a classmethod as doesn't need to modify class state"""
        return np.ravel_multi_index((i//3, j//3), (3,3))
    
    # house masks - take & of a valMask with another mask
    valMask = [np.tile((np.arange(9)==k).reshape(9,1,1), (1,9,9)) for k in range(9)]
    rowMask = [np.tile((np.arange(9)==i).reshape(1,9,1), (9,1,9)) for i in range(9)]
    colMask = [np.tile((np.arange(9)==j).reshape(1,1,9), (9,9,1)) for j in range(9)]
    sqrMask = [np.tile(
                np.kron((np.arange(9)==s).reshape(1,3,3), np.ones((3,3))).astype(bool),
                (9,1,1)) for s in range(9)]
    #For sqrMask, uses Kronecker product to expand a 3*3 array with one True
    # to a 9*9 with the sqr pattern, then tiles into 9*9*9
    houseMask = []
    # following section didn't work as can't access variables defined in class inside list comprehension, because namespaces cursed
    # for k in range(9):
    #     houseMask += ([valMask[k]&rowMask[i] for i in range(9)] +
    #                   [valMask[k]&colMask[j] for j in range(9)] +
    #                   [valMask[k]&sqrMask[s] for s in range(9)])
    # for i in range(9):
    #     houseMask += [rowMask[i]&colMask[j] for j in range(9)]
    for k in range(9):
        for i in range(9):
            houseMask.append(valMask[k]&rowMask[i])
        for j in range(9):
            houseMask.append(valMask[k]&colMask[j])
        for s in range(9):
            houseMask.append(valMask[k]&sqrMask[s])
    for i in range(9):
        for j in range(9):
            houseMask.append(rowMask[i]&colMask[j])
        
    assert len(houseMask)==324
    # houseMask contains all houses
    # (each row, each col and each sqr) for each val, then each (i,j) cell
    def cellHouses(k,i,j):
        """Returns flat indices of houses that cell kij is member of,
        for indexing houseMask
        Returns indices of val-row, val-col, val-sqr, and row-col houses in 
        this order"""
        return [k*27+i, k*27+9+j, k*27+9+9+Puzzle.sqr(i,j), 243+i*9+j]
    
    
    def __init__(self, board=None, preset='medium1'):
        if board is not None:
            assert isinstance(board,np.ndarray)
            assert board.shape == (9,9)
            assert board.dtype == int
            self.board = board
        elif preset=='medium1':
            self.board = np.array([[0,0,0,0,0,0,4,0,0],
                                   [0,0,9,0,2,4,0,1,0],
                                   [0,0,0,3,0,0,0,0,0],
                                   [6,0,2,4,0,0,0,0,0],
                                   [5,4,0,2,6,0,7,0,0],
                                   [1,0,3,0,5,9,0,0,0],
                                   [0,0,0,0,0,0,0,0,3],
                                   [3,0,1,0,0,0,6,0,5],
                                   [0,0,0,0,0,7,0,9,0]])
        else:
            raise TypeError('Input board, choose preset, or use .enterByLine() method')
            
        self.poss = self.initPoss()
    
    def __repr__(self):
        """Should return string that would yield object when passed to eval()
        """
        return f'Puzzle(\n{self.board})'
    
    def __str__(self):
        """Should return readable string represntation for when puzzle instance
        is printed - I'll do board
        """
        return self.board.__str__()
        
    @classmethod
    def enterByLine(cls):
        board = np.zeros((9,9), dtype=int)
        for i in range(9):
            board[i,:] = [int(n) for n in input(f"Enter {('1st' if i==0 else ('2nd' if i==1 else ('3rd' if i==2 else f'{i+1}th')))} line: ")]
        return cls(board)
    
    def initPoss(self):
        """
        Calculate possibilities just based on solved cells,
        only intended to initialise puzzle, 
        as once initialised all solver methods just work with poss and remove 
        cells from poss directly
        """
        poss = np.full((9,9,9), True)
        for i in range(9):
            for j in range(9):
                if self.board[i,j] != 0:
                    k = self.board[i,j]-1
                    houseIndx = Puzzle.cellHouses(k,i,j)
                    for indx in houseIndx:
                        poss[Puzzle.houseMask[indx]] = False
                    # poss[houseMask[vr]] = False
                    # poss[k,:,j] = False
                    # poss[k,i,:] = False
                    # poss[k, (i//3)*3:(i//3)*3+3, (j//3)*3:(j//3)*3+3] = False
                    poss[k,i,j] = True
        # This is copied from solveCell, solveCell not used because that 
        # requires self.poss to already be defined
        return poss
    
    
    def showPoss(self):
        #Show column by column values that could go in each row
        for j in range(9):
            print('\nColumn ', j+1)
            for i in range(9):
                nums = np.arange(9)[self.poss[:,i,j]] + 1
                print(nums)
        # n.b. single poss values are solved cells
    
    def solveCell(self, k, i, j):
        """
        for removing possibilities from cells that share house when cell ij=k+1
        and updating board
        """
        
        self.board[i,j]=k+1
        houseIndx = Puzzle.cellHouses(k,i,j)
        for indx in houseIndx:
            self.poss[Puzzle.houseMask[indx]] = False
        self.poss[k,i,j] = True
    
    
        
    def solve(self):
        """Main method, solves puzzle.
        Applies increasingly complex methods
        Starts with singles: when one cell in a house is only possiblity,
        it can be solved, removing other possiblitiies from any other house 
        that cell is in. In usual parlance, this is both hidden (only poss in ij)
        and naked (only poss in k) singles
        
        Next do ommission, then tuples/x wing/xywing, then full chains/loops
        """
        
        # Singles
        for h in range(len(self.houseMask)):
            if self.poss[self.houseMask[h]]:
                pass
        # Should track houses that are solved, not just cells - saves iterating over all
    
    