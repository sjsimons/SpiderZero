from archive.solitaire import *
import torch

def input_move(game: Game):
    while True:

        move = input("Enter your move:  ")

        if move == "exit":
            return "exit"
        
        if move == "list":
            return "list"

        if move == "stock":
            
            if game.backups > 0:
                if sum([1 for stack in game.live_stacks if len(stack) == 1]) == 0:
                    return "stock"

                else:
                    print("All stacks must have at least one card")
                    continue
            
            else: 
                print("No stock cards remaining")
                continue


        if move == "undo":
            print("Not implemented yet")
            continue
        
        if move == "hint":
            print("Not implemented yet")
            continue

        move = [_ for _ in move]

        try:

            source_idx = ord(move[0].lower())-97
            target_idx = ord(move[1].lower())-97

            if len(move) > 2:
                n = int("".join(move[2:]))
            else:
                n = game.live_stacks[source_idx].num_movable_to(game.live_stacks[target_idx])
        
        except: 

            print("Invalid move")
            continue
        
        if game.live_stacks[source_idx].can_move(game.live_stacks[target_idx],n):

            return source_idx,target_idx,n
        
        else:

            print("Invalid move")
            continue


def positional_encoding(seq_len, embedding_dim, scale_factor=105):
    pos = torch.arange(0, seq_len).unsqueeze(1).float()  # Positions 0 to seq_len-1
    i = torch.arange(0, embedding_dim, 2).float()  # Indices for sine/cosine (even indices for sine, odd indices for cosine)

    # Calculate the angle rates for the positional encoding
    angle_rates = 1 / (scale_factor ** (i / embedding_dim))  # Dynamic scaling factor
    angle_rads = pos * angle_rates  # Angle calculation

    # Apply sine to even indices, cosine to odd indices
    pos_encoding = torch.zeros(seq_len, embedding_dim)
    pos_encoding[:, 0::2] = torch.sin(angle_rads)
    pos_encoding[:, 1::2] = torch.cos(angle_rads)

    return pos_encoding

def prepare_batch(stacks, padding_idx):
    """
    convert from list of stacks to Tensor of padded stacks
    batch: list of stacks
    padding_idx: index to pad the stacks with
    """
    max_len = max(len(stack) for stack in stacks)
    padded_stacks = [stack + [padding_idx] * (max_len - len(stack)) for stack in stacks]
    return torch.tensor(padded_stacks, dtype=torch.long)