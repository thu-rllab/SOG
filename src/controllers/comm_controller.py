from .entity_controller import EntityMAC
import torch as th
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from modules.layers.comm_mixer import AverageMessageEncoder
class CommMAC(EntityMAC):
    def __init__(self, scheme, groups, args):
        assert args.use_msg
        super(CommMAC, self).__init__(scheme, groups, args)
        input_shape = self._get_input_shape(scheme)
        self.message_mixer = AverageMessageEncoder()
        self.message=None

    def decide_header(self, avail_actions, test_mode=False, agent_vis_mask=None):
        # entity, obs_mask, entity_mask = agent_inputs
        if self.args.order_leader:
            neighbor_num = th.sum(th.logical_not(agent_vis_mask), dim=-1) #bs*n
            neighbor_num=neighbor_num.squeeze(1)
            header = th.zeros_like(neighbor_num).bool()
            alive_agent = th.logical_not(avail_actions[:,0,:,0])
            neighbor_num *= alive_agent
            if not self.args.select_by_prob:
                for _ in range(self.args.header_num):
                    _, ind = neighbor_num.max(1)
                    for i in range(neighbor_num.size(0)):
                        if alive_agent[i,ind[i]]:
                            header[i, ind[i]]=True
                        neighbor_num[i, ind[i]]=0
            else:
                for _ in range(self.args.header_num):
                    for i in range(neighbor_num.size(0)):
                        dis = Categorical(probs=neighbor_num[i].float())
                        ind = dis.sample()
                        if alive_agent[i, ind]:
                            header[i,ind]=True
                        neighbor_num[i,ind]=0
            return header.detach()
        else:
            alive_agent = th.logical_not(avail_actions[:,0,:,0]) #bs*n, 0 dead, 1 alive
            bs, _ = alive_agent.shape
            candidate_num = th.sum(alive_agent, axis=1).unsqueeze(1).repeat(1,self.n_agents) #bs*n
            rnd = alive_agent*th.rand([bs, self.n_agents], device=self.args.device)
            generation_alpha = self.args.generation_alpha if test_mode else 1.0
            header = (rnd*candidate_num < self.args.header_num * generation_alpha) * alive_agent
            return header.detach() #bs*n
    
    def decide_group(self, agent_inputs, avail_actions, test_mode=False):
        if self.args.use_comm_sr:
            agent_vis_mask = self.gt_mask[:,:, :self.n_agents, :self.n_agents]
        else:
            entity, obs_mask, entity_mask = agent_inputs
            # header try to dominate all visible agents
            agent_vis_mask = obs_mask[:,:, :self.n_agents, :self.n_agents]
        header = self.decide_header(avail_actions, test_mode=test_mode, agent_vis_mask=agent_vis_mask)
        bs, _=header.shape
        
        #control_message[i,j]=True means i want to dominate j
        control_message = header.unsqueeze(1).unsqueeze(3).repeat(1,1,1,self.n_agents)*th.logical_not(agent_vis_mask)
        #remove message trying to control a header
        control_message *= th.logical_not(header.unsqueeze(1).unsqueeze(2).repeat(1,1,self.n_agents,1)) #bs*1*n*n
        #remain only one master for each agent
        if self.args.random_master: #choose the random seen header as leader
            ind_lst = []
            for i in range(self.n_agents):
                random_ind = th.randperm(self.n_agents, device = self.args.device)
                ind_lst.append(F.one_hot(random_ind[th.max(control_message[:,:,random_ind,i], dim=2)[1]], num_classes=self.n_agents))
                ind = th.cat(ind_lst, dim=1).permute(0,2,1).unsqueeze(1)
        else: #choose first seen header:
            ind = F.one_hot(th.max(control_message, dim=2)[1], self.n_agents).permute(0,1,3,2)
        control_message *= ind.bool()
        #bs*1*n*n, control_message[:,0, i,j]=True means agent i leads agent j.
        return control_message.detach(), header

    def message_comm(self, agent_inputs, avail_actions, t, train_mode=False, test_mode=False, **kwargs):
        entity, obs_mask, entity_mask = agent_inputs
        if train_mode:
            message_personal, msg_dis, msg_dis_inf = self.agent(agent_inputs, self.hidden_states, ret_inf_msg=True, **kwargs)
        else:
            message_personal, _ = self.agent(agent_inputs, self.hidden_states) #bs*n*msg_d  
        message_personal = message_personal.squeeze(1)
        lt = t.stop-t.start

        if lt == 1 and t.start % self.args.msg_T==0: #only needed for interact with env. For training only msg_dis and msg_dis_inf is needed.
            message_matrix, header = self.decide_group(agent_inputs, avail_actions, test_mode=test_mode) #bs*n*n, bs*n
            with th.no_grad():
                message_matrix = message_matrix.squeeze(1)+th.diag_embed(header)
                message_personal_r = message_personal.unsqueeze(1).repeat(1,self.n_agents,1,1)
                message_pass_h = message_matrix.unsqueeze(3)*message_personal_r #bs*n*n, mgp[bs,i,j,:] means msg passed from j to i
                message_header = self.message_mixer(message_pass_h, message_matrix) #bs*n*msg_d
                message_header_r = message_header.unsqueeze(2).repeat(1,1,self.n_agents,1)
                message_pass_a = th.sum(message_matrix.unsqueeze(3)*message_header_r, dim=1) #bs*n*msg_d
                if self.args.no_feedback:
                    receive_matrix = header
                else:
                    receive_matrix = th.max(message_matrix,dim=1)[0] #bs*n, 0 use self message, 1 use received message
                #bs*n*msg_d
                self.message = receive_matrix.unsqueeze(2) * message_pass_a + th.logical_not(receive_matrix).unsqueeze(2) * message_personal
            # self.message=self.message.detach()
        elif not self.args.only_use_head_msg:
            self.message = message_personal.detach()
            # TODO: bug fix. use personal message if communication doesn't needs
            # TODO: bug fix.2 add detach() to remove bp process
        if train_mode:
            return message_personal, self.message, msg_dis, msg_dis_inf
        else:
            return message_personal, self.message #bs*n*msg_d, bs*n*msg_d

    def forward(self, ep_batch, t, test_mode=False, fix_msg=None, train_mode=False, **kwargs):
        if t is None:
            t = slice(0, ep_batch["avail_actions"].shape[1])
            int_t = False
        elif type(t) is int:
            t = slice(t, t + 1)
            int_t = True

        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        if train_mode: #need to first calc Q net to build hidden states, then calc message.
            p_msg = ep_batch["self_message"]
            h_msg = ep_batch["head_message"]
            if kwargs.get('imagine', False):
                agent_outs, self.hidden_states, groups = self.agent(agent_inputs, self.hidden_states, msg = h_msg, **kwargs)
            else:
                agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states, msg = h_msg)
            _, _, msg_dis, msg_dis_inf = self.message_comm(agent_inputs, avail_actions, t, train_mode=True, **kwargs)
        else: #calc message first then send message to Q net.
            if fix_msg is None:
                p_msg, h_msg = self.message_comm(agent_inputs, avail_actions, t, test_mode=test_mode, **kwargs)
            else:
                p_msg = h_msg = fix_msg
            if kwargs.get('imagine', False):
                agent_outs, self.hidden_states, groups = self.agent(agent_inputs, self.hidden_states, msg = h_msg, **kwargs)
            else:
                agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states, msg = h_msg)

        outs = (agent_outs.squeeze(1),) if int_t else (agent_outs,)
        if kwargs.get('imagine', False):
            outs += (groups,)
        if fix_msg is None:
            outs += (p_msg, h_msg)
        if train_mode:
            outs += (msg_dis, msg_dis_inf)
        return outs

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False, ret_agent_outs=False, ret_msg=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs, self_msg, head_msg = self.forward(ep_batch, t_ep, test_mode=test_mode, fix_msg=None)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        if ret_agent_outs:
            return chosen_actions, agent_outputs[bs], self_msg[bs], head_msg[bs]
        if ret_msg:
            return chosen_actions, self_msg[bs], head_msg[bs]
        return chosen_actions

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with entity + observation mask inputs.
        bs = batch.batch_size
        entities = []
        entities.append(batch["entities"][:, t])  # bs, ts, n_entities, vshape
        if self.args.entity_last_action:
            ent_acs = th.zeros(bs, t.stop - t.start, self.args.n_entities,
                               self.args.n_actions, device=batch.device,
                               dtype=batch["entities"].dtype)
            if t.start == 0:
                ent_acs[:, 1:, :self.args.n_agents] = (
                    batch["actions_onehot"][:, slice(0, t.stop - 1)])
            else:
                ent_acs[:, :, :self.args.n_agents] = (
                    batch["actions_onehot"][:, slice(t.start - 1, t.stop - 1)])
            entities.append(ent_acs)
        entities = th.cat(entities, dim=3)
        if self.args.gt_mask_avail:
            return (entities, batch["obs_mask"][:, t], batch["entity_mask"][:, t], batch["gt_mask"][:, t])
        if self.args.use_comm_sr:
            self.gt_mask = batch["gt_mask"][:, t]
        return (entities, batch["obs_mask"][:, t], batch["entity_mask"][:, t])
