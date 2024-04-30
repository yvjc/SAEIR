import torch
import torch.nn as nn
from onpolicy.algorithms.utils.util import init, check
from onpolicy.algorithms.utils.cnn import CNNBase
from onpolicy.algorithms.utils.mlp import MLPBase
from onpolicy.algorithms.utils.rnn import RNNLayer
from onpolicy.algorithms.utils.act import ACTLayer
from onpolicy.algorithms.utils.popart import PopArt
from onpolicy.utils.util import get_shape_from_obs_space

from torch.nn import functional as F
import math
from torch.autograd import Variable


class R_Actor(nn.Module):
    """
    Actor network class for MAPPO. Outputs actions given observations.
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param obs_space: (gym.Space) observation space.
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(R_Actor, self).__init__()
        self.hidden_size = args.hidden_size

        self._gain = args.gain
        self._use_orthogonal = args.use_orthogonal
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain)

        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Compute actions from the given inputs.
        :param obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (np.ndarray / torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param deterministic: (bool) whether to sample from action distribution or return the mode.

        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of taken actions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        actions, action_log_probs, act_logits = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states, act_logits

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Compute log probability and entropy of given actions.
        :param obs: (torch.Tensor) observation inputs into network.
        :param action: (torch.Tensor) actions whose entropy and log probability to evaluate.
        :param rnn_states: (torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (torch.Tensor) mask tensor denoting if hidden states should be reinitialized to zeros.
        :param available_actions: (torch.Tensor) denotes which actions are available to agent
                                                              (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            actor_features, rnn_states = self.rnn(actor_features, rnn_states, masks)

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                   action, available_actions,
                                                                   active_masks=
                                                                   active_masks if self._use_policy_active_masks
                                                                   else None)

        return action_log_probs, dist_entropy

class R_MSHG_Critic(nn.Module):
    """
    Critic network class for MAPPO. Outputs value function predictions given centralized input (MAPPO) or
                            local observations (IPPO).
    :param args: (argparse.Namespace) arguments containing relevant model information.
    :param cent_obs_space: (gym.Space) (centralized) observation space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self, args, cent_obs_space, agent_number, edge_types='team', device=torch.device("cpu")):
        super(R_MSHG_Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self._use_naive_recurrent_policy = args.use_naive_recurrent_policy
        self._use_recurrent_policy = args.use_recurrent_policy
        self._recurrent_N = args.recurrent_N
        self._use_popart = args.use_popart
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base(args, cent_obs_shape)

        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            self.rnn = RNNLayer(self.hidden_size, self.hidden_size, self._recurrent_N, self._use_orthogonal)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        if self._use_popart:
            self.v_out = init_(PopArt(self.hidden_size, 1, device=device))
        else:
            self.v_out = init_(nn.Linear(self.hidden_size, 1))
        
        if edge_types == 'team':
            self.scale = agent_number
        elif edge_types == 'group':
            self.scale = args.num_in_group
        elif edge_types == 'idv':
            self.scale = 0
        else:
            ValueError('Unknown edge types!')

        self.agent_number = agent_number
        self.edge_types = edge_types
        self.nmp_layers = args.layer_N
        self.nmp_mlp_start = FC_dict_softmax(args, input_dim = args.hidden_size, output_dim = args.hidden_size, agent_number = self.agent_number, edge_types = self.edge_types)
        self.nmp_mlp_end = FC(input_dim = args.hidden_size * 2, output_dim = args.hidden_size)
        
        attention_mlp = []
        for i in range(self.nmp_layers):
            attention_mlp.append(FC(input_dim = args.hidden_size * 2, output_dim = 1))
        self.attention_mlp = nn.ModuleList(attention_mlp)

        node2edge_start_mlp = []
        for i in range(self.nmp_layers):
            node2edge_start_mlp.append(FC(input_dim = args.hidden_size, output_dim = args.hidden_size))
        self.node2edge_start_mlp = nn.ModuleList(node2edge_start_mlp)

        edge_aggregation_list = []
        for i in range(self.nmp_layers):
            edge_aggregation_list.append(edge_aggregation(args, input_dim = args.hidden_size, output_dim = args.hidden_size, agent_number=self.agent_number, edge_types=self.edge_types))
        self.edge_aggregation_list = nn.ModuleList(edge_aggregation_list)

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """
        Compute actions from the given inputs.
        :param cent_obs: (np.ndarray / torch.Tensor) observation inputs into network.
        :param rnn_states: (np.ndarray / torch.Tensor) if RNN network, hidden states for RNN.
        :param masks: (np.ndarray / torch.Tensor) mask tensor denoting if RNN states should be reinitialized to zeros.

        :return values: (torch.Tensor) value function predictions.
        :return rnn_states: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        if self._use_naive_recurrent_policy or self._use_recurrent_policy:
            critic_features, rnn_states = self.rnn(critic_features, rnn_states, masks)

        feature_len = critic_features.shape[1]
        critic_features = critic_features.view(-1, self.agent_number, feature_len)
        assert(len(critic_features.shape) == 3)
        curr_hidden = critic_features 
        corr_ = F.normalize(critic_features, p = 2, dim = 2)
        corr = torch.matmul(corr_, corr_.permute(0,2,1))

        H = self.init_adj_attention(curr_hidden, corr, scale_factor = self.scale)

        edge_hidden = self.node2edge(curr_hidden, H, idx=0) 
        edge_feat = self.nmp_mlp_start(edge_hidden)                      
        node_feat = curr_hidden
        
        critic_node_feat = self.nmp_mlp_end(self.edge2node(edge_feat,node_feat, H, 0))

        critic_node_feat = critic_node_feat.view(-1, feature_len)

        return critic_node_feat, rnn_states

    def edge2node(self, x, ori, H, idx):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = self.edge_aggregation_list[idx](x,H,ori)
        return incoming/incoming.size(1)

    def node2edge(self, x, H, idx):
        x = self.node2edge_start_mlp[idx](x)
        edge_init = torch.matmul(H,x)
        node_num = x.shape[1]
        edge_num = edge_init.shape[1]
        x_rep = (x[:,:,None,:].transpose(2, 1)).repeat(1, edge_num, 1, 1)
        edge_rep = edge_init[:, :, None, :].repeat(1, 1, node_num, 1)
        node_edge_cat = torch.cat((x_rep, edge_rep), dim=-1)
        attention_weight = self.attention_mlp[idx](node_edge_cat)[:, :, :, 0]
        H_weight = attention_weight * H
        H_weight = F.softmax(H_weight, dim = 2)
        H_weight = H_weight * H
        edges = torch.matmul(H_weight, x)
        return edges
    
    def init_adj_attention(self, feat, feat_corr, scale_factor=2):
        batch = feat.shape[0]
        actor_number = feat.shape[1]
        if scale_factor == actor_number:
            H_matrix = torch.ones(batch,1,actor_number).type_as(feat)
            return H_matrix
        group_size = scale_factor
        if group_size < 1:
            H_matrix_2D = torch.eye(actor_number, actor_number).type_as(feat)
            H_matrix = H_matrix_2D.repeat(batch, 1, 1)
            return H_matrix

        _, indice = torch.topk(feat_corr, dim = 2, k = group_size, largest = True)
        H_matrix = torch.zeros(batch, actor_number, actor_number).type_as(feat)
        H_matrix = H_matrix.scatter(2, indice, 1)

        return H_matrix
    
    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

class FC_dict_softmax(nn.Module):
    def __init__(self, args, input_dim, output_dim, agent_number, edge_types='team'):
        super(FC_dict_softmax, self).__init__()
        if edge_types == 'team':
            edge_type_num = 1
        elif edge_types == 'group':
            edge_type_num = args.num_group
        elif edge_types == 'idv':
            edge_type_num = agent_number
        else:
            ValueError('Unknown edge types!')

        self.FC_distribution = FC(input_dim = input_dim, output_dim = edge_type_num)
        self.FC_factor = FC(input_dim = input_dim, output_dim = 1)
        self.init_FC =FC(input_dim = input_dim, output_dim = input_dim)

    def forward(self, x):
        x = self.init_FC(x)
        distribution = gumbel_softmax(self.FC_distribution(x),tau=1/2, hard=False)
        factor = torch.sigmoid(self.FC_factor(x))
        out = factor * distribution
        return out

class FC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FC, self).__init__()

        self.layer = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()
        self.layernorm = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x)
        x = self.layernorm(x)
        return x
        
class edge_aggregation(nn.Module):
    def __init__(self, args, input_dim, output_dim, agent_number, edge_types='team'):
        super(edge_aggregation, self).__init__()
        if edge_types == 'team':
            self.edge_type_num = 1
        elif edge_types == 'group':
            self.edge_type_num = args.num_group
        elif edge_types == 'idv':
            self.edge_type_num = agent_number
        else:
            ValueError('Unknown edge types!')
        
        self.dict_dim = input_dim
        self.agg_mlp = []
        for i in range(self.edge_type_num):
            self.agg_mlp.append(FC(input_dim = input_dim, output_dim = input_dim))
        self.agg_mlp = nn.ModuleList(self.agg_mlp)

    def forward(self, edge_distribution, H, ori):
        batch = edge_distribution.shape[0]
        edges = edge_distribution.shape[1]
        edge_feature = torch.zeros(batch, edges, ori.shape[-1]).type_as(ori)
        edges = torch.matmul(H, ori)
        for i in range(self.edge_type_num):
            edge_feature += edge_distribution[:, :, i:i+1] * self.agg_mlp[i](edges)

        node_feature = torch.cat((torch.matmul(H.permute(0, 2, 1), edge_feature), ori), dim = -1)
        return node_feature

def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Sample from Gumbel(0, 1)
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Draw a sample from the Gumbel-Softmax distribution
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return my_softmax(y / tau, axis=-1)

def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3
    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes
    Constraints:
    - this implementation only works on batch_size x num_features tensor for now
    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y

def my_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input)
    return soft_max_1d.transpose(axis, 0)
