import torch
from onpolicy.algorithms.r_mappo.algorithm.r_actor_MSHG_critic import R_Actor, R_MSHG_Critic
from onpolicy.utils.util import update_linear_schedule


class R_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, agent_number, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space
        self.agent_number = agent_number

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic_team = R_MSHG_Critic(args, self.share_obs_space, self.agent_number, device=self.device, edge_types='team')
        self.critic_group = R_MSHG_Critic(args, self.share_obs_space, self.agent_number, device=self.device, edge_types='group')
        self.critic_idv = R_MSHG_Critic(args, self.share_obs_space, self.agent_number, device=self.device, edge_types='idv')

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_team_optimizer = torch.optim.Adam(self.critic_team.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
        self.critic_group_optimizer = torch.optim.Adam(self.critic_group.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
        self.critic_idv_optimizer = torch.optim.Adam(self.critic_idv.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_team_optimizer, episode, episodes, self.critic_lr)
        update_linear_schedule(self.critic_group_optimizer, episode, episodes, self.critic_lr)
        update_linear_schedule(self.critic_idv_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic_team, rnn_states_critic_group, rnn_states_critic_idv,
                     masks, available_actions=None, deterministic=False):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        actions, action_log_probs, rnn_states_actor, act_logits = self.actor(obs,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 available_actions,
                                                                 deterministic)

        features_team, rnn_states_critic_team = self.critic_team(cent_obs, rnn_states_critic_team, masks)
        features_group, rnn_states_critic_group = self.critic_group(cent_obs, rnn_states_critic_group, masks)
        features_idv, rnn_states_critic_idv = self.critic_idv(cent_obs, rnn_states_critic_idv, masks)
        
        values_team = self.critic_team.v_out(features_team) 
        values_group = self.critic_group.v_out(features_group)
        values_idv = self.critic_idv.v_out(features_idv)
        
        return values_team, values_group, values_idv, actions, action_log_probs, rnn_states_actor, rnn_states_critic_team, rnn_states_critic_group, rnn_states_critic_idv, act_logits

    def get_values(self, cent_obs, obs, rnn_states_actor,rnn_states_critic_team, rnn_states_critic_group, rnn_states_critic_idv, masks, available_actions=None, deterministic=False):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """

        features_team, _ = self.critic_team(cent_obs, rnn_states_critic_team, masks)
        features_group, _ = self.critic_group(cent_obs, rnn_states_critic_group, masks)
        features_idv, _ = self.critic_idv(cent_obs, rnn_states_critic_idv, masks)

        values_team = self.critic_team.v_out(features_team) 
        values_group = self.critic_group.v_out(features_group) 
        values_idv = self.critic_idv.v_out(features_idv)

        return values_team, values_group, values_idv

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic_team, rnn_states_critic_group, rnn_states_critic_idv,
                          action, masks, available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks)

        features_team, _ = self.critic_team(cent_obs, rnn_states_critic_team, masks)
        features_group, _ = self.critic_group(cent_obs, rnn_states_critic_group, masks)
        features_idv, _ = self.critic_idv(cent_obs, rnn_states_critic_idv, masks)

        values_team = self.critic_team.v_out(features_team) 
        values_group = self.critic_group.v_out(features_group) 
        values_idv = self.critic_idv.v_out(features_idv)

        return values_team, values_group, values_idv, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor, act_logits = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)

        return actions, rnn_states_actor
