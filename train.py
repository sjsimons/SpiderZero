import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from agent import *
from env import *

def train(agent: SpiderSolitaireModel, env: Environment, buffer: Buffer, batch_size=32, gamma=0.99):
    optimizer = optim.Adam(agent.parameters(), lr=1e-3)

    for episode in range(1000):

        state = env.reset()
        stacks, global_features = state
        legal_moves = env.legal_moves()

        done = False
        episode_rewards = 0
        episode_steps = 0
        states, actions, rewards, next_states, dones, values = [], [], [], [], [], []


        while not done:

            # Get action probabilities and current state value from the policy network
            action_probs, state_value = agent(stacks.unsqueeze(0),legal_moves.unsqueeze(0),global_features.unsqueeze(0))

            # Sample action (use categorical sampling)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()

            states.append(state)
            actions.append(action)
            values.append(state_value)

            # Take a step in the environment
            next_state, reward, done = env.step(action.item())

            next_legal_moves = env.legal_moves()

            # Store experience in the replay buffer
            buffer.add(state, action.item(), legal_moves, reward, next_state, next_legal_moves, done)

            legal_moves = next_legal_moves
            state = next_state
            episode_rewards += reward
            episode_steps += 1

            if done or episode_steps > 100 or torch.sum(legal_moves) == 0:
                if done:
                    print('done!')
                break

        # Once the episode is over, update the policy network using the replay buffer
        if len(buffer.memory) > batch_size:
            experiences = buffer.sample(batch_size)
            states_list, actions, legal_moves, rewards, next_states_list, next_legal_moves, dones = experiences

            # Convert to tensors
            stacks_tensor = torch.zeros((batch_size,env.game.num_stacks,max((stacks.shape[1] for stacks,_ in states_list))),device=env.device,dtype=torch.int)
            global_features_tensor = torch.cat([global_features.unsqueeze(0) for _,global_features in states_list]).to(env.device)
            for i,(stacks,global_features) in enumerate(states_list):
                stacks_tensor[i,:stacks.shape[0],:stacks.shape[1]] = stacks

            next_stacks_tensor = torch.zeros((batch_size,env.game.num_stacks,max((stacks.shape[1] for stacks,_ in next_states_list))),device=env.device,dtype=torch.int)
            next_global_features_tensor = torch.cat([global_features.unsqueeze(0) for _,global_features in next_states_list]).to(env.device)
            for i,(stacks,global_features) in enumerate(next_states_list):
                next_stacks_tensor[i,:stacks.shape[0],:stacks.shape[1]] = stacks
            
            actions_tensor = torch.as_tensor(actions, device=env.device)
            legal_moves_tensor = torch.stack(legal_moves).to(device=env.device)
            next_legal_moves_tensor = torch.stack(next_legal_moves).to(device=env.device)
            rewards_tensor = torch.as_tensor(rewards,dtype=torch.float, device=env.device)

            # Get current and next state values
            action_probs, state_values = agent(stacks_tensor,legal_moves_tensor,global_features_tensor)
            _, next_state_values = agent(next_stacks_tensor,next_legal_moves_tensor,next_global_features_tensor)

            # Compute advantages using TD(λ)
            deltas = rewards_tensor + gamma * next_state_values.squeeze(-1) - state_values.squeeze(-1)
            advantages = deltas 

            # Compute policy loss (Negative log-probability * advantage)
            log_probs = torch.log(action_probs.gather(1, actions_tensor.unsqueeze(-1)))
            policy_loss = -torch.mean(log_probs * advantages)

            # Compute value loss (MSE between predicted value and actual reward)
            value_loss = F.mse_loss(state_values.squeeze(-1), rewards_tensor)

            # **Separate Optimization Steps**
            optimizer.zero_grad()

            # Policy head update
            policy_loss.backward(retain_graph=True)  # Retain graph so value head can still backpropagate
            for param in agent.policy_head.parameters():
                param.grad = param.grad  # Only update policy head parameters

            # Value head update
            value_loss.backward()
            for param in agent.value_head.parameters():
                param.grad = param.grad  # Only update value head parameters

            optimizer.step()

        # Reset buffer
        buffer.clear()

        if episode % 100 == 0:
            print(f"Episode {episode}, Reward: {episode_rewards}, Policy Loss: {policy_loss.item()}, Value Loss: {value_loss.item()}")