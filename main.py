from agent import SpiderSolitaireModel, Buffer
from env import Environment
from train import train
import torch
def main():

    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize the agent (SpiderSolitaireModel)
    my_agent = SpiderSolitaireModel(device=device)
    
    # Create the environment
    env = Environment(device=device)
    env.reset()
    
    # Initialize the replay buffer
    buffer = Buffer.Buffer(capacity=10000)
    
    # Start the training process
    train(my_agent, env, buffer)

if __name__ == "__main__":
    main()
