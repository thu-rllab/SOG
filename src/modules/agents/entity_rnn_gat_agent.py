import torch as th
import torch.nn as nn
import torch.nn.functional as F
from modules.layers import EntityAttentionLayer, EntityPoolingLayer


class EntityAttentionRNNGATAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(EntityAttentionRNNGATAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.attn_embed_dim)
        if args.pooling_type is None:
            self.attn = EntityAttentionLayer(args.attn_embed_dim,
                                             args.attn_embed_dim,
                                             args.attn_embed_dim*(1+args.double_attn), args)
        else:
            self.attn = EntityPoolingLayer(args.attn_embed_dim,
                                           args.attn_embed_dim,
                                           args.attn_embed_dim*(1+args.double_attn),
                                           args.pooling_type,
                                           args)
        self.fc2 = nn.Linear(args.attn_embed_dim*(1+args.double_attn), args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim*2, args.n_actions)
        self.sub_scheduler_mlp1 = nn.Sequential(
                    nn.Linear(args.gat_encoder_out_size*2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, args.gat_encoder_out_size//2),
                    nn.ReLU(),
                    nn.Linear(args.gat_encoder_out_size//2, 2))
        if self.args.second_graph:
            self.sub_scheduler_mlp2 = nn.Sequential(
                        nn.Linear(args.gat_encoder_out_size*2, args.gat_encoder_out_size//2),
                        nn.ReLU(),
                        nn.Linear(args.gat_encoder_out_size//2, args.gat_encoder_out_size//2),
                        nn.ReLU(),
                        nn.Linear(args.gat_encoder_out_size//2, 2))
        
        self.gat_encoder = EntityAttentionLayer(args.rnn_hidden_dim, args.gat_encoder_out_size, args.gat_encoder_out_size, args)
        self.sub_processor1 = EntityAttentionLayer(args.rnn_hidden_dim, args.rnn_hidden_dim, args.rnn_hidden_dim, args)
        if args.second_graph:
            self.sub_processor2 = EntityAttentionLayer(args.rnn_hidden_dim, args.rnn_hidden_dim, args.rnn_hidden_dim, args)  

    def init_hidden(self):
        # make hidden states on same device as model
        self.attn_weights=None
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, ret_attn_logits=None, ret_attn_weights=False):
        entities, obs_mask, entity_mask = inputs
        bs, ts, ne, ed = entities.shape
        entities = entities.reshape(bs * ts, ne, ed)
        obs_mask = obs_mask.reshape(bs * ts, ne, ne)
        entity_mask = entity_mask.reshape(bs * ts, ne)
        agent_mask = entity_mask[:, :self.args.n_agents]
        x1 = F.relu(self.fc1(entities))
        attn_outs = self.attn(x1, pre_mask=obs_mask,
                              post_mask=agent_mask)
        x2 = attn_outs
        for i in range(self.args.repeat_attn):
            x2 = self.attn(x2, pre_mask=obs_mask,
                              post_mask=agent_mask,
                              ret_attn_logits=ret_attn_logits)
            x2 = attn_outs
        x3 = F.relu(self.fc2(x2))
        x3 = x3.reshape(bs, ts, self.args.n_agents, -1)

        h = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        hs = []
        for t in range(ts):
            curr_x3 = x3[:, t].reshape(-1, self.args.rnn_hidden_dim)
            h = self.rnn(curr_x3, h)
            hs.append(h.reshape(bs, self.args.n_agents, self.args.rnn_hidden_dim))
        hs = th.stack(hs, dim=1)  # Concat over time
        hs = hs.reshape(bs*ts, self.args.n_agents, self.args.rnn_hidden_dim)

        full_obs_mask = 1-(1-agent_mask.unsqueeze(1))*(1-agent_mask.unsqueeze(2))
        gat_state =self.gat_encoder(hs, pre_mask=full_obs_mask, post_mask=agent_mask) #bs*ts, na, gat_encoder_out_size
        gat_state_ori = gat_state.clone()
        adj1 = self.sub_scheduler(self.sub_scheduler_mlp1, gat_state, full_obs_mask) #bs*ts, na, na
        msg = F.elu(self.sub_processor1(hs, pre_mask=adj1, post_mask=agent_mask))
        if self.args.second_graph:
            adj2 = self.sub_scheduler(self.sub_scheduler_mlp2, gat_state_ori, full_obs_mask) #bs*ts, na, na
            msg = self.sub_processor2(hs, pre_mask=adj2, post_mask=agent_mask)
        hsm = th.cat([hs, msg], dim=-1)

        q = self.fc3(hsm)
        q = q.reshape(bs, ts, self.args.n_agents, -1)
        q = q.masked_fill(agent_mask.reshape(bs, ts, self.args.n_agents, 1).bool(), 0)

        return q, hs
    
    def sub_scheduler(self, mlp, gat_state, full_obs_mask):
        bsts, na, hid_size = gat_state.size()
        hard_attn_input = th.cat([gat_state.repeat(1, 1, na).view(bsts, na * na, -1), gat_state.repeat(1, na, 1)], dim=2).view(bsts, na, na, 2 * hid_size)
        hard_attn_output = F.gumbel_softmax(mlp(hard_attn_input), hard=True) #bsts,na,na,2
        hard_attn_output = th.narrow(hard_attn_output, 3, 1, 1) #bsts,na,na,1
        adj = (1 - hard_attn_output.squeeze()).masked_fill(full_obs_mask.bool(), 1)
        return adj
        

        


class ImagineEntityAttentionRNNGATAgent(EntityAttentionRNNGATAgent):
    def __init__(self, *args, **kwargs):
        super(ImagineEntityAttentionRNNGATAgent, self).__init__(*args, **kwargs)

    def logical_not(self, inp):
        return 1 - inp

    def logical_or(self, inp1, inp2):
        out = inp1 + inp2
        out[out > 1] = 1
        return out

    def entitymask2attnmask(self, entity_mask):
        bs, ts, ne = entity_mask.shape
        # agent_mask = entity_mask[:, :, :self.args.n_agents]
        in1 = (1 - entity_mask.to(th.float)).reshape(bs * ts, ne, 1)
        in2 = (1 - entity_mask.to(th.float)).reshape(bs * ts, 1, ne)
        attn_mask = 1 - th.bmm(in1, in2)
        return attn_mask.reshape(bs, ts, ne, ne).to(th.uint8)

    def forward(self, inputs, hidden_state, imagine=False, **kwargs):
        if not imagine:
            if kwargs.get('ret_attn_weights', False):
                return super(ImagineEntityAttentionRNNGATAgent, self).forward(inputs, hidden_state, ret_attn_weights=True)
            else:
                return super(ImagineEntityAttentionRNNGATAgent, self).forward(inputs, hidden_state)
        entities, obs_mask, entity_mask = inputs
        bs, ts, ne, ed = entities.shape

        # create random split of entities (once per episode)
        groupA_probs = th.rand(bs, 1, 1, device=entities.device).repeat(1, 1, ne)

        groupA = th.bernoulli(groupA_probs).to(th.uint8)
        groupB = self.logical_not(groupA)
        # mask out entities not present in env
        groupA = self.logical_or(groupA, entity_mask[:, [0]])
        groupB = self.logical_or(groupB, entity_mask[:, [0]])

        # convert entity mask to attention mask
        groupAattnmask = self.entitymask2attnmask(groupA)
        groupBattnmask = self.entitymask2attnmask(groupB)
        # create attention mask for interactions between groups
        interactattnmask = self.logical_or(self.logical_not(groupAattnmask),
                                           self.logical_not(groupBattnmask))
        # get within group attention mask
        withinattnmask = self.logical_not(interactattnmask)

        activeattnmask = self.entitymask2attnmask(entity_mask[:, [0]])
        # get masks to use for mixer (no obs_mask but mask out unused entities)
        Wattnmask_noobs = self.logical_or(withinattnmask, activeattnmask)
        Iattnmask_noobs = self.logical_or(interactattnmask, activeattnmask)
        # mask out agents that aren't observable (also expands time dim due to shape of obs_mask)
        withinattnmask = self.logical_or(withinattnmask, obs_mask)
        interactattnmask = self.logical_or(interactattnmask, obs_mask)

        entities = entities.repeat(3, 1, 1, 1)
        obs_mask = th.cat([obs_mask, withinattnmask, interactattnmask], dim=0)
        entity_mask = entity_mask.repeat(3, 1, 1)

        inputs = (entities, obs_mask, entity_mask)
        hidden_state = hidden_state.repeat(3, 1, 1)
        q, h = super(ImagineEntityAttentionRNNGATAgent, self).forward(inputs, hidden_state)
        return q, h, (Wattnmask_noobs.repeat(1, ts, 1, 1), Iattnmask_noobs.repeat(1, ts, 1, 1))
