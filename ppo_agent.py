from utils.utils import *
from hparams import HyperParams as hp


def get_gae(rewards, masks, values):
    returns = torch.zeros_like(rewards)
    advants = torch.zeros_like(rewards)

    running_returns = 0
    previous_value = 0
    running_advants = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + hp.gamma * running_returns * masks[t]
        running_tderror = rewards[t] + hp.gamma * previous_value * masks[t] - \
                    values.data[t]
        running_advants = running_tderror + hp.gamma * hp.lamda * \
                          running_advants * masks[t]

        returns[t] = running_returns
        previous_value = values.data[t]
        advants[t] = running_advants

    advants = (advants - advants.mean()) / advants.std()
    return returns, advants


def surrogate_loss(actor, advants, states, old_policy, actions, index):
    mu, std, logstd = actor(states)
    new_policy = log_density(actions, mu, std, logstd)
    old_policy = old_policy[index]

    ratio = torch.exp(new_policy - old_policy)
    surrogate = ratio * advants
    return surrogate, ratio


def train_model(actor, critic, batch, actor_optim, critic_optim):
    states = to_tensor(batch.state)
    actions = to_tensor(batch.action)
    rewards = to_tensor(batch.reward)
    masks = to_tensor(batch.mask)
    values = critic(states)

    # ----------------------------
    # step 1: get returns and GAEs and log probability of old policy
    returns, advants = get_gae(rewards, masks, values)
    mu, std, logstd = actor(states)
    old_policy = log_density(actions, mu, std, logstd)
    old_values = values.clone()

    criterion = torch.nn.MSELoss()
    n = len(states)
    arr = np.arange(n)

    # ----------------------------
    # step 2: get value loss and actor loss and update actor & critic
    for epoch in range(3):
        print('epoch is ' + str(epoch))
        np.random.shuffle(arr)

        for i in range(n // hp.batch_size):
            batch_index = arr[hp.batch_size * i: hp.batch_size * (i + 1)]
            batch_index = to_tensor_long(batch_index)
            inputs = states[batch_index]
            returns_samples = returns.unsqueeze(1)[batch_index]
            advants_samples = advants.unsqueeze(1)[batch_index]
            actions_samples = actions[batch_index]
            oldvalue_samples = old_values[batch_index].detach()

            loss, ratio = surrogate_loss(actor, advants_samples, inputs,
                                         old_policy.detach(), actions_samples,
                                         batch_index)

            values = critic(inputs)
            clipped_values = oldvalue_samples + \
                             torch.clamp(values - oldvalue_samples,
                                         -hp.clip_param,
                                         hp.clip_param)
            critic_loss1 = criterion(clipped_values, returns_samples)
            critic_loss2 = criterion(values, returns_samples)
            critic_loss = torch.max(critic_loss1, critic_loss2).mean()
            clipped_ratio = torch.clamp(ratio,
                                        1.0 - hp.clip_param,
                                        1.0 + hp.clip_param)
            clipped_loss = clipped_ratio * advants_samples
            actor_loss = -torch.min(loss, clipped_loss).mean()

            loss = actor_loss + 0.5 * critic_loss

            critic_optim.zero_grad()
            loss.backward(retain_graph=True)
            critic_optim.step()

            actor_optim.zero_grad()
            loss.backward()
            actor_optim.step()
