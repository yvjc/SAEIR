import numpy as np
import torch
import torch.nn as nn
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algorithms.utils.util import check

class R_MAPPO():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        
        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        if self._use_popart:
            self.value_team_normalizer = self.policy.critic_team.v_out
            self.value_group_normalizer = self.policy.critic_group.v_out
            self.value_idv_normalizer = self.policy.critic_idv.v_out
        elif self._use_valuenorm:
            self.value_team_normalizer = ValueNorm(1).to(self.device)
            self.value_group_normalizer = ValueNorm(1).to(self.device)
            self.value_idv_normalizer = ValueNorm(1).to(self.device)
        else:
            self.value_team_normalizer = None
            self.value_group_normalizer = None
            self.value_idv_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch, value_normalizer):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
        if self._use_popart or self._use_valuenorm:
            value_normalizer.update(return_batch)
            error_clipped = value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_team_batch, rnn_states_critic_group_batch, \
        rnn_states_critic_idv_batch, actions_batch, value_team_preds_batch, value_group_preds_batch, value_idv_preds_batch, \
        return_batch_team, return_batch_group, return_batch_idv, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_team_targ, adv_group_targ, adv_idv_targ, available_actions_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_team_targ = check(adv_team_targ).to(**self.tpdv)
        adv_group_targ = check(adv_group_targ).to(**self.tpdv)
        adv_idv_targ = check(adv_idv_targ).to(**self.tpdv)
        value_team_preds_batch = check(value_team_preds_batch).to(**self.tpdv)
        value_group_preds_batch = check(value_group_preds_batch).to(**self.tpdv)
        value_idv_preds_batch = check(value_idv_preds_batch).to(**self.tpdv)
        return_batch_team = check(return_batch_team).to(**self.tpdv)
        return_batch_group = check(return_batch_group).to(**self.tpdv)
        return_batch_idv = check(return_batch_idv).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # Reshape to do in a single forward pass for all steps
        values_team, values_group, values_idv, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch, 
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_team_batch,
                                                                              rnn_states_critic_group_batch,
                                                                              rnn_states_critic_idv_batch, 
                                                                              actions_batch, 
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        W_sum = adv_idv_targ.exp().sum() + adv_group_targ.exp().sum() + adv_team_targ.exp().sum()
        W_idv = adv_idv_targ.exp().sum() / W_sum
        W_group = adv_group_targ.exp().sum() / W_sum
        W_team = adv_team_targ.exp().sum() / W_sum

        surr1 = imp_weights * (W_team * adv_team_targ + W_group * adv_group_targ + W_idv * adv_idv_targ)
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * (W_team * adv_team_targ + W_group * adv_group_targ + W_idv * adv_idv_targ)

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()

        # critic update
        value_team_loss = self.cal_value_loss(values_team, value_team_preds_batch, return_batch_team, active_masks_batch, self.value_team_normalizer)
        value_group_loss = self.cal_value_loss(values_group, value_group_preds_batch, return_batch_group, active_masks_batch, self.value_group_normalizer)
        value_idv_loss = self.cal_value_loss(values_idv, value_idv_preds_batch, return_batch_idv, active_masks_batch, self.value_idv_normalizer)

        self.policy.critic_team_optimizer.zero_grad()
        self.policy.critic_group_optimizer.zero_grad()
        self.policy.critic_idv_optimizer.zero_grad()

        (value_team_loss * self.value_loss_coef).backward()
        (value_group_loss * self.value_loss_coef).backward()
        (value_idv_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_team_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic_team.parameters(), self.max_grad_norm)
            critic_group_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic_group.parameters(), self.max_grad_norm)
            critic_idv_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic_idv.parameters(), self.max_grad_norm)
        else:
            critic_team_grad_norm = get_gard_norm(self.policy.critic_team.parameters())
            critic_group_grad_norm = get_gard_norm(self.policy.critic_group.parameters())
            critic_idv_grad_norm = get_gard_norm(self.policy.critic_idv.parameters())

        self.policy.critic_team_optimizer.step()
        self.policy.critic_group_optimizer.step()
        self.policy.critic_idv_optimizer.step()

        return value_team_loss, value_group_loss, value_idv_loss, critic_team_grad_norm, critic_group_grad_norm, \
        critic_idv_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if self._use_popart or self._use_valuenorm:
            advantages_team = buffer.returns_team[:-1] - self.value_team_normalizer.denormalize(buffer.value_team_preds[:-1])
            advantages_group = buffer.returns_group[:-1] - self.value_group_normalizer.denormalize(buffer.value_group_preds[:-1])
            advantages_idv = buffer.returns_idv[:-1] - self.value_idv_normalizer.denormalize(buffer.value_idv_preds[:-1])
        else:
            advantages_team = buffer.returns_team[:-1] - buffer.value_team_preds[:-1]
            advantages_group = buffer.returns_group[:-1] - buffer.value_group_preds[:-1]
            advantages_idv = buffer.returns_idv[:-1] - buffer.value_idv_preds[:-1]

        advantages_team_copy = advantages_team.copy()
        advantages_group_copy = advantages_group.copy()
        advantages_idv_copy = advantages_idv.copy()

        advantages_team_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        advantages_group_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        advantages_idv_copy[buffer.active_masks[:-1] == 0.0] = np.nan

        mean_advantages_team = np.nanmean(advantages_team_copy)
        mean_advantages_group = np.nanmean(advantages_group_copy)
        mean_advantages_idv = np.nanmean(advantages_idv_copy)
        std_advantages_team = np.nanstd(advantages_team_copy)
        std_advantages_group = np.nanstd(advantages_group_copy)
        std_advantages_idv = np.nanstd(advantages_idv_copy)

        advantages_team = (advantages_team - mean_advantages_team) / (std_advantages_team + 1e-5)
        advantages_group = (advantages_group - mean_advantages_group) / (std_advantages_group + 1e-5)
        advantages_idv = (advantages_idv - mean_advantages_idv) / (std_advantages_idv + 1e-5)
        

        train_info = {}

        train_info['value_team_loss'] = 0
        train_info['value_group_loss'] = 0
        train_info['value_idv_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_team_grad_norm'] = 0
        train_info['critic_group_grad_norm'] = 0
        train_info['critic_idv_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages_team, advantages_group, advantages_idv, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                pass
                # data_generator = buffer.naive_recurrent_generator(advantages_team, advantages_group, advantages_idv, self.num_mini_batch)
            else:
                pass
                #data_generator = buffer.feed_forward_generator(advantages_team, advantages_group, advantages_idv, self.num_mini_batch)

            for sample in data_generator:

                value_team_loss, value_group_loss, value_idv_loss, critic_team_grad_norm, critic_group_grad_norm, critic_idv_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample, update_actor)

                train_info['value_team_loss'] += value_team_loss.item()
                train_info['value_group_loss'] += value_group_loss.item()
                train_info['value_idv_loss'] += value_idv_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_team_grad_norm'] += critic_team_grad_norm
                train_info['critic_group_grad_norm'] += critic_group_grad_norm
                train_info['critic_idv_grad_norm'] += critic_idv_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic_team.train()
        self.policy.critic_group.train()
        self.policy.critic_idv.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic_team.eval()
        self.policy.critic_group.eval()
        self.policy.critic_group.eval()
