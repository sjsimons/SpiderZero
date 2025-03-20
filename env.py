import torch
class SolitaireGame():
    def __init__(self,num_stacks=10,device='cpu'):

        self.num_stacks = num_stacks
        self.num_decks = int(num_stacks/5)
        self.num_moves = num_stacks*num_stacks+1
        self.backups = torch.as_tensor([0],dtype=torch.int,device=device)
        self.completed = torch.tensor([0],dtype=torch.int,device=device)
        self.device = device

        # list of tensors for storing true values of cards in each stack
        self.stacks = [torch.tensor([],dtype=torch.int,device=device) for _ in range(num_stacks)]
        # list of tensors for storing whether a card is face up or not
        self.face_up = [torch.tensor([],dtype=torch.int,device=device) for _ in range(num_stacks)]
        # list of tensors for storing backup cards
        self.backup_stacks = [torch.tensor([],dtype=torch.int,device=device) for _ in range(num_stacks)]

    
    def reset(self):
        deck = torch.arange(1, 52+1).repeat(self.num_decks).to(device=self.device,dtype=torch.int)
        deck = deck[torch.randperm(len(deck))]

        for i in range(int(self.num_stacks*0.4)):
            self.stacks[i] = deck[i*6:(i+1)*6]
        
        for i in range(int(self.num_stacks*-0.6),0):
            self.stacks[i] = deck[(i+self.num_stacks)*5:(i+self.num_stacks+1)*5]
        
        for i in range(self.num_stacks):
            self.face_up[i] = torch.zeros(len(self.stacks[i]),dtype=torch.int,device=self.device)
            self.face_up[i][-1] = 1
            self.backup_stacks[i] = deck[(i+self.num_stacks)*5:(i+self.num_stacks+1)*5]
        
        self.backups[0] = 5
        self.completed[0] = 0

    def backup(self):
        self.backups -= 1
        for i in range(self.num_stacks):
            self.stacks[i] = torch.cat([self.stacks[i], self.backup_stacks[i][self.backups]])
            self.face_up[i] = torch.cat([self.face_up[i], torch.tensor([1],device=self.device)])
        
        self.check_completed()
        
    def move(self, source_idx, target_idx):
        target_top_value = 0 if len(self.stacks[target_idx]) == 0 else self.stacks[target_idx][-1]
        n = 1

        if target_top_value == 0:
            for i in range(len(self.stacks[source_idx])):
                if len(self.stacks[source_idx]) == i:
                    n = i
                    break
                elif self.face_up[source_idx][-i-1] == 1 and self.stacks[source_idx][-i-1] % 13 == self.stacks[source_idx][-(i+2)] % 13 - 1 and self.stacks[source_idx][-i-1] % 13 != 0:
                    continue
                else:
                    n = i
                    break
        else:
            while True:
                if (self.stacks[source_idx][-n] % 13) == ((target_top_value - 1) % 13):
                    break
                else:
                    n += 1
                    if n > len(self.stacks[source_idx]):
                        raise Exception("Invalid move")
        
        self.stacks[target_idx] = torch.cat([self.stacks[target_idx], self.stacks[source_idx][-n:]])
        self.stacks[source_idx] = self.stacks[source_idx][:-n]

        self.face_up[target_idx] = torch.cat([self.face_up[target_idx], self.face_up[source_idx][-n:]])
        self.face_up[source_idx] = self.face_up[source_idx][:-n]

        if len(self.stacks[source_idx]) > 0:
            if self.face_up[source_idx][-1] == 0:
                self.face_up[source_idx][-1] = 1
        
        self.check_completed()
    
    def check_completed(self):
        for i in range(self.num_stacks):
            if len(self.stacks[i]) >= 13:
                if torch.all(self.stacks[i][-13:].flip(0) % torch.arange(1,14,device=self.device) == 0):
                    self.completed += 1
                    self.stacks[i] = self.stacks[i][:-13]
                    self.face_up[i] = self.face_up[i][:-13]
                    if self.face_up[i][-1] == 0:
                        self.face_up[i][-1] = 1
    
    def eval(self):
        score=0
        for i in range(len(self.stacks)):
            for j in range(1,len(self.stacks[i])):

                if self.face_up[i][-j] == 1 and self.face_up[i][-j-1] == 1 and self.stacks[i][-j] == self.stacks[i][-j-1] - 1 and self.stacks[i][j] % 13 != 0:
                    score += 1
            
        score += 12*self.completed

        return (score/96)**2
    
    def legal_moves(self):
        moves = []
        for target_idx in range(self.num_stacks):
            # get the value of the top card of the target stack (0 if the stack is empty)
            target_top_value = 0 if len(self.stacks[target_idx]) == 0 else self.stacks[target_idx][-1]
            # if the top of the target stack is an ace then it can't recieve any more cards
            if target_top_value % 13 == 1:
                continue
            # if the target stack is empty then any card can be moved to it
            elif target_top_value == 0:
                for source_idx in range(self.num_stacks):
                    if source_idx == target_idx or len(self.stacks[source_idx]) == 0:
                        continue
                    moves.append(source_idx*self.num_stacks+target_idx)
            for source_idx in range(self.num_stacks):
                # if the source stack is empty then it can't move any cards
                if source_idx == target_idx or len(self.stacks[source_idx]) == 0:
                    continue
                else:
                    n=1
                    while True:
                        # Check if the nth top card of the source stack can be moved to the target stack
                        if (self.stacks[source_idx][-n] % 13) == ((target_top_value - 1) % 13):
                            moves.append(source_idx*self.num_stacks+target_idx)
                        # if there are more cards in the stack
                        elif n < len(self.stacks[source_idx]):
                            # if the next card in the stack continues a sequence
                            if self.face_up[source_idx][-(n+1)] == 1 & self.stacks[source_idx][-n] == (self.stacks[source_idx][-(n+1)]+1):
                                # check the next card in the sequence
                                n += 1
                                continue
                        # otherwise move on to the next possible source stack
                        break
        
        # if there are backup cards remaining and all stacks have at least one card then the stock can be dealt
        if self.backups > 0 & sum([1 for stack in self.stacks if stack.shape[0] == 0]) == 0:
            moves.append(self.num_moves-1)

        return moves
    
    def display(self):
        for i in range(max(len(stack) for stack in self.stacks)):
            for j in range(self.num_stacks):
                if i < len(self.stacks[j]):
                    suit = (self.stacks[j][i]-1) // 13
                    value = (self.stacks[j][i]-1) % 13 + 1
                    if value == 1:
                        value = 'A'
                    elif value == 11:
                        value = 'J'
                    elif value == 12:
                        value = 'Q'
                    elif value == 13:
                        value = 'K'
                    print(f"{value}{'♥♦♣♠'[suit] if self.face_up[j][i] == 1 else 'X'}", end="\t")
                else:
                    print("  ", end="\t")
            print()
        
        print(f"Backups: {self.backups}")
        print(f"Completed: {self.completed}")
        print()


class Environment():
    def __init__(self, num_stacks=10, device='cpu'):
        self.game = SolitaireGame(num_stacks, device)
        self.device = device

    def reset(self):
        self.game.reset()
        return self.state()
    
    def legal_moves(self):
        move_indexes = self.game.legal_moves()
        moves = torch.zeros(self.game.num_moves, dtype=torch.int, device=self.device)
        moves[move_indexes] = 1
        return moves
    
    def state(self):
        stacks = torch.zeros((self.game.num_stacks, max(len(stack) for stack in self.game.stacks)), dtype=torch.int, device=self.device)
        for i in range(len(self.game.stacks)):
            stacks[i,:len(self.game.stacks[i])] = self.game.stacks[i]*self.game.face_up[i]+53*(1-self.game.face_up[i])
        global_features = torch.tensor([self.game.backups, self.game.completed], device=self.device)

        return stacks, global_features

    def step(self, action):
        if action == self.game.num_moves-1:
            self.game.backup()
        else:
            # Move card(s) from source stack to target stack
            source_idx, target_idx = action // self.game.num_stacks, action % self.game.num_stacks           
            self.game.move(source_idx, target_idx)
        
        next_state = self.state()
        reward = self.game.eval()
        done = reward == 1

        return next_state, reward, done
            
            
    
class SmallEnvironment(Environment):
    def __init__(self,num_stacks=5,num_decks=1):
        super().__init__(num_stacks, num_decks)