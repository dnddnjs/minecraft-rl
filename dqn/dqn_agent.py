from utils.utils import *
from hparams import HyperParams as hp
import torch.nn.functional as F

def train_model(model, target_model, batch, optimizer):
    states = to_tensor(batch.state)
    next_states = to_tensor(batch.next_state)
    actions = to_tensor_long(batch.action)
    rewards = to_tensor(batch.reward)
    masks = to_tensor(batch.mask)

    pred = model(states)
    next_pred = target_model(next_states).detach()
    # one-hot encoding
    one_hot_action = torch.zeros(hp.batch_size, pred.size(-1))
    one_hot_action.scatter_(1, actions.unsqueeze(1), 1)
    pred = torch.sum(pred.mul(one_hot_action), dim=1)

    # Q Learning: get maximum Q value at s' from target model
    target = rewards + masks * hp.gamma * next_pred.max(1)[0]

    optimizer.zero_grad()

    # Smooth L1 Loss function
    loss = F.smooth_l1_loss(pred,target)
    loss.backward()

    # and train
    optimizer.step()