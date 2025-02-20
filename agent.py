import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import positional_encoding
import random

class StackTransformer(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, num_layers=4, num_cards=54, padding_idx = 0, device='cpu'):
        super().__init__()

        self.embed_dim = embed_dim      # embedding dimension
        self.num_heads = num_heads      # num heads in the transformer
        self.num_layers = num_layers    # num layers in the transformer
        self.num_cards = num_cards      # scale this depending on number of suits
        self.padding_idx = padding_idx  # padding index for the embedding
        self.device = device

        self.base_card_embedding = nn.Embedding(num_cards, embed_dim, padding_idx).to(device)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)
    
    def embed_stacks(self, stacks): 
        """
        stacks: tensor of stacks, shape (batch_size, stack_length)
        """
        batch_size, stack_length = stacks.shape

        # get basic card embeddings
        base_card_embedding = self.base_card_embedding(stacks)

        # add positional encoding
        pos_enc = positional_encoding(stack_length, self.embed_dim)
        pos_enc = pos_enc.unsqueeze(0).repeat(batch_size, 1, 1).to(device=self.device)  # Shape (batch_size, stack_length, embed_dim)

        return base_card_embedding + pos_enc

    def forward(self, stacks):
        """
        stacks: Tensor of card indices (batch_size, max_stack_len)
        """

        # Pad the stacks to the same length and 
        stack_embedding = self.embed_stacks(stacks)

        stack_encoding = self.transformer_encoder(stack_embedding)

        stack_encoding = stack_encoding.mean(dim=1)  # Average over the stack length

        return stack_encoding

class SpiderSolitaireModel(nn.Module):
    """Full model with shared Transformer and policy/value heads."""
    def __init__(self, num_stacks=10, embed_dim=64, num_heads=2, num_layers=2, num_moves=101, device='cpu'):
        super().__init__()
        self.stack_transformer = StackTransformer(embed_dim, num_heads, num_layers, device=device)
        self.to(device)

        # Fully connected network
        self.fc = nn.Sequential(
            nn.Linear((embed_dim * num_stacks)+2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        ).to(device)

        # Policy head (move selection)
        self.policy_head = nn.Linear(128, num_moves).to(device)

        # Value head (game outcome)
        self.value_head = nn.Linear(128, 1).to(device)

    def forward(self, stacks, legal_moves, global_features):
        """
        stacks: (batch_size, num_stacks, stack_size) tensor of card indices.
        legal_moves: (batch_size, num_moves) binary mask of legal moves.
        global_features: (batch_size, 2) tensor of global features.
        """

        # pass through transformer in batches
        b, n, s = stacks.shape
        stacks = stacks.view(b * n, s)
        stack_features = self.stack_transformer(stacks)
        stack_features = stack_features.view(b, n, self.stack_transformer.embed_dim)

        # Flatten stack features
        board_features = stack_features.view(b, -1)  # (batch_size, num_stacks * embed_dim)

        global_features = global_features.float()
        board_features = torch.cat([board_features, global_features], dim=1)

        # Pass through FC network
        x = self.fc(board_features)  # (batch_size, 128)

        # Compute outputs
        policy_logits = self.policy_head(x)  # (batch_size, num_moves)
        value = self.value_head(x)  # (batch_size, 1)

        # Apply legal move masking (softmax only over valid moves)
        policy_logits = policy_logits.masked_fill(legal_moves == 0, -1e9)
        policy_probs = F.softmax(policy_logits, dim=1)

        return policy_probs, value

class Buffer:
    class Buffer:
        def __init__(self, capacity=10000):
            self.capacity = capacity
            self.memory = []
            self.position = 0

        def add(self, state, legal_moves, action, reward, next_state, next_legal_moves, done):
            """Save a transition."""
            if len(self.memory) < self.capacity:
                self.memory.append(None)
            self.memory[self.position] = (state, legal_moves, action, reward, next_state, next_legal_moves, done)
            self.position = (self.position + 1) % self.capacity

        def sample(self, batch_size):
            """Sample a random batch of transitions."""
            batch = random.sample(self.memory, batch_size)
            state, legal_moves, action, reward, next_state, next_legal_moves, done = zip(*batch)
            return [state, legal_moves, action, reward, next_state, next_legal_moves, done]
        
        def clear(self):
            self.memory = []
            self.position = 0

        def __len__(self):
            return len(self.memory)