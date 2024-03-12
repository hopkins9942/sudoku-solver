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

h is id of house in Puzzle.houseMask[h] (complete list of houses)
hindx refers to index of h in incomplete list i.e. self.unsolvedHouses[hindx]

all internals (ijksh) are zero indexed,
outputs (row, column, value, square, house) are 1-indexed
exception is error messages which refer to zero indexed h

all this would work for hexidecimal and 4by4 sudoku


New Plan:
    singles and SSI is only needed, so write singles method to be used by SSI
    solveCell should recursively breadth-first solve all resultant singles
    is bredth first strictly needed if it  checks inconsistency each time and can be depth limited
    Also write inconsistency cheker with order argument
    So far not using uniqueness as I cansider it cheating.
    
    Pattern spotting better for fish etc. so implement later after SSI working.
    Do some single-solution puzzles still require uniqueness to solve?
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
        
    assert len(houseMask)==4*81
    # houseMask contains all houses
    # (each row, each col and each sqr) for each val, then each (i,j) cell
    def cellHouses(k,i,j):
        """Returns flat indices of houses that cell kij is member of,
        for indexing houseMask
        Returns indices of val-row, val-col, val-sqr, and row-col houses in 
        this order"""
        return [k*27+i, k*27+9+j, k*27+9+9+Puzzle.sqr(i,j), 243+i*9+j]
    
    def cellHousesMask(k,i,j):
        """Returns mask of all cells linked to cell ijk"""
        hs = Puzzle.cellHouses(k,i,j)
        mask = np.zeros((9,9))
        for h in hs:
            mask = np.logical_or(mask, Puzzle.houseMask[h])
        mask[k,i,j] = False
        return mask
        
        
    # def h2ijk(h):
    #     if h>=81*3:
    #         i = (h-243)//9
    #         j = (h-243)-i*9
    #         return f'row {i+1}, col {j+1}'
    #     else:
    #         k = h//27
    #         t = h-k*27
    #         word = ['row', 'col', 'sqr'][t//9]
    #         num = t-9*(t//9)
    #         return f'number {k+1}, {word} {num+1}'
        
    def h2str(h):
        if h>=81*3:
            i = (h-243)//9
            j = (h-243)-i*9
            return f'row {i+1}, col {j+1}'
        else:
            k = h//27
            t = h-k*27
            word = ['row', 'column', 'square'][t//9]
            num = t-9*(t//9)
            return f'value {k+1}, {word} {num+1}'
            

    def initPossUnsolvedHouses(board):
        """
        Calculate possibilities just based on solved cells,
        only intended to initialise puzzle, 
        as once initialised all solver methods just work with poss and remove 
        cells from poss directly
        Also returns list of unsolved house indices
        
        Going by the puzzles in my app, medium can be solved by singles only,
        hard by singles+intersections.
        """
        poss = np.full((9,9,9), True)
        unsolvedHouses = np.arange(4*81)
        for i in range(9):
            for j in range(9):
                if board[i,j] != 0:
                    k = board[i,j]-1
                    fourhs = Puzzle.cellHouses(k,i,j)
                    for h in fourhs:
                        if np.count_nonzero(unsolvedHouses==h)!=1:
                            # Something wrong, likely invalid board entered
                            raise ValueError(f'Likely error in entered board, h={h}: {Puzzle.h2str(h)}')
                        poss[Puzzle.houseMask[h]] = False
                        unsolvedHouses = np.delete(unsolvedHouses, unsolvedHouses==h)
                    poss[k,i,j] = True
        # Similar to solveCell
        return poss, unsolvedHouses
    
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
        elif preset=='medium2':
            self.board = np.array([[0, 0, 0, 8, 0, 7, 3, 0, 0],
                                   [0, 6, 7, 0, 0, 0, 0, 1, 0],
                                   [0, 0, 3, 4, 0, 0, 0, 0, 2],
                                   [0, 7, 0, 6, 0, 0, 8, 0, 0],
                                   [2, 0, 0, 0, 4, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 9, 0, 3, 0],
                                   [0, 2, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 8, 0, 0, 0, 0, 0, 9],
                                   [0, 3, 4, 0, 9, 0, 0, 2, 0]])
            
        elif preset=='hard1':
            self.board = np.array([[0, 4, 2, 8, 0, 0, 0, 0, 5],
                                   [0, 6, 0, 0, 1, 0, 0, 0, 0],
                                   [0, 0, 1, 0, 0, 0, 0, 3, 9],
                                   [6, 0, 5, 0, 0, 3, 4, 0, 8],
                                   [0, 2, 8, 0, 0, 0, 0, 0, 3],
                                   [9, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 9, 4, 0],
                                   [0, 5, 0, 4, 0, 0, 0, 0, 0],
                                   [0, 0, 4, 0, 0, 1, 0, 8, 2]])
        elif preset=='hard2':
            self.board = np.array([[0, 0, 0, 0, 1, 0, 2, 0, 7],
                                   [1, 0, 8, 2, 0, 0, 9, 5, 0],
                                   [0, 0, 0, 0, 5, 4, 0, 0, 8],
                                   [9, 0, 0, 0, 6, 0, 0, 7, 0],
                                   [2, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 7, 0, 0, 0, 0, 0, 8, 0],
                                   [5, 0, 0, 8, 0, 0, 6, 1, 0],
                                   [0, 0, 2, 0, 0, 1, 0, 0, 0],
                                   [7, 0, 0, 0, 0, 5, 0, 0, 0]])
        elif preset=='hard5':
            self.board = np.array([[0, 0, 8, 0, 0, 0, 3, 0, 0],
                                   [6, 0, 0, 0, 0, 0, 9, 0, 0],
                                   [0, 1, 0, 0, 0, 9, 0, 0, 0],
                                   [0, 0, 0, 0, 6, 7, 8, 5, 0],
                                   [0, 0, 7, 5, 0, 2, 6, 0, 0],
                                   [0, 0, 0, 0, 1, 0, 0, 9, 0],
                                   [0, 0, 9, 2, 0, 0, 0, 8, 0],
                                   [0, 0, 6, 3, 7, 0, 0, 0, 0],
                                   [3, 0, 5, 9, 0, 0, 0, 0, 0]])# can't be solved by intersection
        elif preset=='diabolic1':
            self.board = np.array([[0, 0, 4, 2, 0, 0, 0, 0, 0],
                                   [0, 8, 0, 7, 1, 0, 0, 0, 0],
                                   [0, 2, 0, 0, 0, 6, 0, 0, 0],
                                   [3, 0, 6, 0, 0, 4, 0, 0, 1],
                                   [0, 0, 0, 0, 8, 0, 6, 0, 3],
                                   [0, 0, 0, 6, 0, 3, 7, 9, 0],
                                   [4, 0, 2, 8, 0, 0, 0, 0, 0],
                                   [0, 0, 1, 4, 2, 7, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 7, 0]])
        elif preset=='diabolic2':
            self.board = np.array([[0, 6, 0, 0, 0, 4, 0, 0, 0],
                                   [0, 0, 2, 3, 6, 0, 0, 0, 0],
                                   [9, 0, 0, 7, 0, 0, 0, 0, 0],
                                   [0, 9, 0, 6, 8, 0, 0, 4, 0],
                                   [0, 0, 0, 0, 0, 3, 0, 9, 0],
                                   [0, 5, 0, 0, 0, 0, 0, 7, 0],
                                   [7, 0, 3, 0, 0, 0, 9, 5, 0],
                                   [0, 0, 0, 0, 3, 0, 0, 0, 0],
                                   [5, 0, 0, 4, 0, 9, 2, 6, 0]])
        elif preset=='diabolic3':
            self.board = np.array([[0, 7, 5, 0, 1, 6, 0, 0, 3],
                                   [0, 0, 4, 0, 0, 0, 0, 5, 1],
                                   [8, 0, 0, 0, 0, 7, 0, 0, 0],
                                   [0, 0, 1, 0, 8, 0, 0, 0, 0],
                                   [0, 8, 7, 0, 0, 9, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 5, 9, 0],
                                   [0, 0, 0, 3, 0, 0, 0, 0, 0],
                                   [0, 0, 3, 0, 5, 0, 0, 1, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 7, 4]])#the only diabolic I solved by hand
        else:
            raise TypeError('Input board, choose preset, or use .enterByLine() method')
        
        self.poss, self.unsolvedHouses = Puzzle.initPossUnsolvedHouses(self.board)
        
        
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
    
    def checkPoss(self):
        for h in range(81*4):
            if np.count_nonzero(self.poss[Puzzle.houseMask[h]])<1:
                raise ValueError(f'h={h} ({Puzzle.h2str(h)}) has <1 possible value!')
    
    def showPoss(self):
        #Show column by column values that could go in each row
        for j in range(9):
            print('\nColumn ', j+1)
            for i in range(9):
                nums = np.arange(9)[self.poss[:,i,j]] + 1
                print(nums)
        # n.b. single poss values are solved cells
    
    
    # new methods
    
    def solve(self, step=False):
        """SOlve singles, then do SSI with increasing depth and order
        """
        """Solves puzzle if step=False, solves one cell if step=True.
        
        Plan:
        Apply increasingly complex methods
        Starts with singles, then omission, then looks for doubles, then triples etc.
        increases order of n-tuples
        Then do chains - which may involve arbitrarily setting a cell, then 
        running a simple singles+intersections solve until inconsistency
        This means I need a consisiteency detector - houses with no Trues, or house already removed from unsolvedHouses
        
        output board
        
        subfunctions return 1 if it progress made so should try simpler technique again, 
        -1 if no progress made so try more complex technique,
        and 0 if board solved so can exit!
        
        sort print statements
        """
        return self.solve_singles(step)
        
        # c2f = [self.solve_singles, self.solve_intersection] # which f to use at each complexity level
        # c = 0 # complexity level
        # for sweep in range(100):# could calculate maximum possible number of sweeps based on 81 max numbers to fill, some number max intersections etc
        #     out = c2f[c]()
        #     c -= out
        #     if out==0:
        #         print('Sudoku solved!')
        #         return self.board
        #     elif c==len(c2f):
        #         print('Solver out of ideas - you\'re on your own from here!')
        #         return self.board
            
    
    def solveCell(self, k, i, j, depthToGo=0):
        """
        for removing possibilities from cells that share house when cell ijk 
        is solution and updating board and unsolvedHouses
        
        Could add to so checks if solving cell creates more singles to solve
        
        need to put print staements here
        """
        self.board[i,j]=k+1
        mask = Puzzle.cellHousesMask(k,i,j)
        self.poss[mask] = False
        self.poss[k,i,j] = True
        self.check()
        if depthToGo>=1:
            linkedCells = np.argwhere(self.poss&mask) 
            for n in range(linkedCells.shape[0]):
                self.solveCell(*linkedCells[n,:], depthToGo=depthToGo-1)    
        
        fourhs = Puzzle.cellHouses(k,i,j)
        for h in fourhs:
            self.unsolvedHouses = np.delete(self.unsolvedHouses, self.unsolvedHouses==h)#is unsolvedHouses necessary in new system?
        
        
    def check(order=1):
        """checks for existence of n parallel houses with <n shared houses
        to put poss in"""
        pass
    
    def solve_singles(self, step=False):
        for sweep in range(81 if not step else 1):
            #81 as each sweep should find at least one cell
            # 1 for just one step
            for h in self.unsolvedHouses:
                #note: unsolvedHouses updated as loop occurs, meaning it 
                # doesn't try to solve houses it has already solved earlier in loop
                justHouse = self.poss&Puzzle.houseMask[h] # so keeps 9,9,9 shape
                if np.count_nonzero(justHouse)==1:
                    #single 
                    k,i,j = np.argwhere(justHouse)[0]
                    if h>=81*3:
                        solvetype = '        naked'
                    else:
                        t = h-27*(h//27) # note h//27 = k
                        solvetype = ['   row hidden',
                                      'column hidden',
                                      'square hidden'][t//9]
                    print(f'Found {solvetype} single value {k+1} at row {i+1}, column {j+1}')
                    self.solveCell(k,i,j, depthToGo=9**3 if not step else 1)
                    break
            
            if len(self.unsolvedHouses)==0:
                #solved!
                return 0
            elif h==self.unsolvedHouses[-1]:
                print('No more singles, now searching for an intersection.')
                return -1
        
    def SSI():
        """
        copy puzzle, suppose cell, solve to some depth then look for inconsistencies 
        of some order. Reapeat, increasing depth and order.
        solveCell could do breadth-first and trace chain automatically.
        intersections are found by depth 0 order 1
        X-wings (naked and hidden tuples) found by depth 0 order 2, or depth 1 order 1
        n-fish (n-tuples) including fins found by depth 0 order n-1
        All non-fish examples I've seen can be solved by order 1, any depth
        If all else fails, can do hogh deptha and high order
        Structure: could depth limit, or could just chek inconsistencies when 
        stops finding singles - danger is this could find complicated chains before
        shorter more obvious ones. Other danger is trying every poss at depth 1,
        then depth2, then depth 3, .. recalculating every time.
        It is possible that there's always a low-depth chain to find, alternatively 
        could do depth first, log all chains, and print and use the lowest depth one!
        """
        pass
    
    def solve_intersection(self, step=False):
        """
        Current bug: finds an intersection, eturns to singles, doen't find any
        returns to intersections finds same intersection, repeat
        May be replaceable by SSI"""
        # for sweep in range(81 if not step else 1):
            #may want to go back to singles every time intersection found - easier for user
            
        # plan - sweep all houses h<81*3, find where truths lie all in only one other house
        #put plan in docstring
        for h in self.unsolvedHouses[self.unsolvedHouses<81*3]:
            truth_coords = np.nonzero(self.poss[Puzzle.houseMask[h]])[0]
            if len(truth_coords)>3:
                #intersection impossible
                continue
            
            # below find that if intersection exists, finds otherh
            otherh = -1
            thirds = truth_coords//3 # 000111222
            if np.all(thirds==thirds[0]):
                #all equal, therefore intersection 
                if (h-27*(h//27))//9==0:
                    #row
                    i = h - 27*(h//27)
                    s = 3*(i//3) + thirds[0] # gives index of square
                    otherh = 27*(h//27) + 18 + s
                elif (h-27*(h//27))//9==1:
                    #col
                    j = h - 27*(h//27) - 9
                    s = j//3 + 3*thirds[0]
                    otherh = 27*(h//27) + 18 + s
                elif (h-27*(h//27))//9==2:
                    #sqr-row
                    s = h - 27*(h//27) - 18
                    i = 3*(s//3) + thirds[0]
                    otherh = 27*(h//27) + i
                else:
                    raise ValueError('This line should not be run')
            #sqr-col
            altthirds = truth_coords%3 # 012012012
            if np.all(altthirds==altthirds[0])and((h-27*(h//27))//9==2):
                #all equal and h is square therefore sqr-col intersection
                s = h - 27*(h//27) - 18
                j = 3*(s%3) + altthirds[0]
                otherh = 27*(h//27) + 9 + j
            
            if otherh!=-1:
                #intersection found
                #h is house with all truths in one other house, otherh
                # first checks if anything new gained
                mask = Puzzle.houseMask[otherh]&np.logical_not(Puzzle.houseMask[h])
                if np.count_nonzero(self.poss[mask])==0:
                    #nothing to be gained, possibly found this intersection before
                    continue
                # now removing other possibilities from otherh
                print(f'Intersection found: {Puzzle.h2str(h)} to house {Puzzle.h2str(otherh)} ({h}-{otherh}).')
                self.poss[mask] = False
                return 1
        
        if h == self.unsolvedHouses[self.unsolvedHouses<81*3][-1]:
            #no intersection found
            print('No intersections found.')
            return -1
        
    
    
    
    
    # old methods
    
    # def solveCell(self, k, i, j):
    #     """
    #     for removing possibilities from cells that share house when cell ij=k+1
    #     and updating board
    #     """
    #     self.board[i,j]=k+1
    #     fourhs = Puzzle.cellHouses(k,i,j)
    #     for h in fourhs:
    #         self.poss[Puzzle.houseMask[h]] = False
    #         self.unsolvedHouses = np.delete(self.unsolvedHouses, self.unsolvedHouses==h)
    #     self.poss[k,i,j] = True
    
        
    # def solve(self, step=False, simple=False):
    #     """Solves puzzle if step=False, solves one cell if step=True.
        
    #     Plan:
    #     Apply increasingly complex methods
    #     Starts with singles, then omission, then looks for doubles, then triples etc.
    #     increases order of n-tuples
    #     Then do chains - which may involve arbitrarily setting a cell, then 
    #     running a simple singles+intersections solve until inconsistency
    #     This means I need a consisiteency detector - houses with no Trues, or house already removed from unsolvedHouses
        
    #     output board
        
    #     subfunctions return 1 if it progress made so should try simpler technique again, 
    #     -1 if no progress made so try more complex technique,
    #     and 0 if board solved so can exit!
        
    #     sort print statements
    #     """
        
    #     c2f = [self.solve_singles, self.solve_intersection] # which f to use at each complexity level
    #     c = 0 # complexity level
    #     for sweep in range(100):# could calculate maximum possible number of sweeps based on 81 max numbers to fill, some number max intersections etc
    #         out = c2f[c]()
    #         c -= out
    #         if out==0:
    #             print('Sudoku solved!')
    #             return self.board
    #         elif c==len(c2f):
    #             print('Solver out of ideas - you\'re on your own from here!')
    #             return self.board
            
        
    # def solve_singles(self, step=False):
    #     for sweep in range(81 if not step else 1):
    #         #81 as each sweep should find at least one cell
    #         # 1 for just one step
    #         for h in self.unsolvedHouses:
    #             #note: unsolvedHouses updated as loop occurs, meaning it 
    #             # doesn't try to solve houses it has already solved earlier in loop
    #             justHouse = self.poss&Puzzle.houseMask[h] # so keeps 9,9,9 shape
    #             if np.count_nonzero(justHouse)==1:
    #                 k,i,j = np.argwhere(justHouse)[0]
                    
    #                 if h>=81*3:
    #                     solvetype = '        naked'
    #                 else:
    #                     t = h-27*(h//27) # note h//27 = k
    #                     solvetype = ['   row hidden',
    #                                  'column hidden',
    #                                  'square hidden'][t//9]
    #                 # if h<81:
    #                 #     solvetype = '   row hidden'
    #                 # elif h<81*2:
    #                 #     solvetype = 'column hidden'
    #                 # elif h<81*3:
    #                 #     solvetype = 'square hidden' # ERROR - houses not divided like this
    #                 # else: # equiv elif h<81*4:
    #                 #     solvetype = '        naked'
    #                 print(f'Found {solvetype} single value {k+1} at row {i+1}, column {j+1}')
    #                 self.solveCell(k,i,j)
    #                 break
    #         # reset means will find all hiddens before stars on nakeds
    #         # possibly easier for user
            
    #         if len(self.unsolvedHouses)==0:
    #             return 0
    #         elif h==self.unsolvedHouses[-1]:
    #             print('No more singles, now searching for an intersection.')
    #             return -1
    
    # def solve_breadth_first(self, maxDepth):
    #     """As on tin"""
    #     pass
    
    
    # def SSI(self, order=1):
    #     """Ultimate solving method, cannot fail.
    #     Supposes a cell, then solves singles checking for inconsistencies. 
    #     Solve occurs breadth-first for easy-to-find chains, with maximum depth 
    #     that increases depth steadily. 
    #     Order is the level of inconsistencies searched for, n parallel houses
    #     with <n shared houses to put poss in
    #     After working, set up naming"""
    #     possList = np.argwhere(self.poss) but unsolved
    #     testPuzzle = Puzzle(board=self.board) # check this isn't shallow copy
    #     for n in range(possList.shape[0]):
    #         testPuzzle = Puzzle(board=self.board) # check this isn't shallow copy
    #         testPuzzle.solve(*possList[n,:])
            
    
    # def solve_intersection(self, step=False):
    #     """
    #     Current bug: finds an intersection, eturns to singles, doen't find any
    #     returns to intersections finds same intersection, repeat
    #     May be replaceable by SSI"""
    #     # for sweep in range(81 if not step else 1):
    #         #may want to go back to singles every time intersection found - easier for user
            
    #     # plan - sweep all houses h<81*3, find where truths lie all in only one other house
    #     #put plan in docstring
    #     for h in self.unsolvedHouses[self.unsolvedHouses<81*3]:
    #         truth_coords = np.nonzero(self.poss[Puzzle.houseMask[h]])[0]
    #         if len(truth_coords)>3:
    #             #intersection impossible
    #             continue
            
    #         # below find that if intersection exists, finds otherh
    #         otherh = -1
    #         thirds = truth_coords//3 # 000111222
    #         if np.all(thirds==thirds[0]):
    #             #all equal, therefore intersection 
    #             if (h-27*(h//27))//9==0:
    #                 #row
    #                 i = h - 27*(h//27)
    #                 s = 3*(i//3) + thirds[0] # gives index of square
    #                 otherh = 27*(h//27) + 18 + s
    #             elif (h-27*(h//27))//9==1:
    #                 #col
    #                 j = h - 27*(h//27) - 9
    #                 s = j//3 + 3*thirds[0]
    #                 otherh = 27*(h//27) + 18 + s
    #             elif (h-27*(h//27))//9==2:
    #                 #sqr-row
    #                 s = h - 27*(h//27) - 18
    #                 i = 3*(s//3) + thirds[0]
    #                 otherh = 27*(h//27) + i
    #             else:
    #                 raise ValueError('This line should not be run')
    #         #sqr-col
    #         altthirds = truth_coords%3 # 012012012
    #         if np.all(altthirds==altthirds[0])and((h-27*(h//27))//9==2):
    #             #all equal and h is square therefore sqr-col intersection
    #             s = h - 27*(h//27) - 18
    #             j = 3*(s%3) + altthirds[0]
    #             otherh = 27*(h//27) + 9 + j
            
    #         if otherh!=-1:
    #             #intersection found
    #             #h is house with all truths in one other house, otherh
    #             # first checks if anything new gained
    #             mask = Puzzle.houseMask[otherh]&np.logical_not(Puzzle.houseMask[h])
    #             if np.count_nonzero(self.poss[mask])==0:
    #                 #nothing to be gained, possibly found this intersection before
    #                 continue
    #             # now removing other possibilities from otherh
    #             print(f'Intersection found: {Puzzle.h2str(h)} to house {Puzzle.h2str(otherh)} ({h}-{otherh}).')
    #             self.poss[mask] = False
    #             return 1
        
    #     if h == self.unsolvedHouses[self.unsolvedHouses<81*3][-1]:
    #         #no intersection found
    #         print('No intersections found.')
    #         return -1
        
    
    
    
    
    
    
    
    
    