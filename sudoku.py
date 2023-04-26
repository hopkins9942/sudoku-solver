"""
Current thoughts:
when solving, once full possibilities are put in, you only need to look at that
so take solved cells out of poss, and track solved board seperately

self.board tells what is solved (in 3d: val,row,col)
a solved cell makes all cells which share a house with it zero,
its value doesn't matter but I'll set to True for comfort
self.poss tells what is possibile in unsolved houses
"""

import numpy as np



class Puzzle:
    
    # house masks - take & of a valMask with another mask
    valMask = [np.tile((np.arange(9)==m).reshape(9,1,1), (1,9,9)) for m in range(9)]
    rowMask = [np.tile((np.arange(9)==m).reshape(1,9,1), (9,1,9)) for m in range(9)]
    colMask = [np.tile((np.arange(9)==m).reshape(1,1,9), (9,9,1)) for m in range(9)]
    sqrMask = [np.tile(
               np.kron((np.arange(9)==m).reshape(1,3,3), np.ones((3,3))).astype(bool),
               (9,1,1)) for m in range(9)]
    #For sqrMask, uses Kronecker product to expand a 3*3 array with one True
    # to a 9*9 with the sqr pattern, then tiles into 9*9*9
    def sqr(i,j):
        return np.ravel_multi_index((i//3, j//3), (3,3))
    
    def __init__(self, board=None, preset='medium1'):
        """Could also add presets for testing"""
        if board is not None:
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
                    poss[:,i,j] = False
                    poss[k,:,j] = False
                    poss[k,i,:] = False
                    poss[k, (i//3)*3:(i//3)*3+3, (j//3)*3:(j//3)*3+3] = False
                    poss[k,i,j] = True
        # This is copied from solveCell, solveCell not used because that 
        # requires self.poss to already be defined
        return poss
    
    def showPoss(self):
        #Show column by column
        pass
    
    def solveCell(self, k, i, j):
        """
        for removing possibilities due to shared house when cell ij=k+1
        or, when cell ij is determined to be True
        Order of indices is kij as k determines depth (first layer of nesting)
        i is row (2nd layer) and j is column (3rd layer)
        
        Could use house masks, but this is possibly less clear
        self.poss[rowMask[i]&colMask[j]] = False
        self.poss[valMask[k]&rowMask[i]] = False
        self.poss[valMask[k]&colMask[j]] = False
        self.poss[valMask[k]&sqrMask[sqr(i,j)]] = False
        
        """
        
        self.board[i,j]=k+1
        
        self.poss[:,i,j] = False
        self.poss[k,:,j] = False
        self.poss[k,i,:] = False
        self.poss[k, (i//3)*3:(i//3)*3+3, (j//3)*3:(j//3)*3+3] = False
        
        self.poss[k,i,j] = True
        return 0
    
    
        
    def solve(self):
        """Main method, solves puzzle.
        Applies increasingly complex methods
        Starts with singles: when one cell in a house is only possiblity,
        it can be solved, removing other possiblitiies from any other house 
        that cell is in. In usual parlance, this is both hidden (only poss in ij)
        and naked (only poss in k) singles
        """
        
        # Singles 
        
        
    
    