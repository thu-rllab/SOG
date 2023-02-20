from .comm_controller import CommMAC
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class DppCommMac(CommMAC):
    def __init__(self, scheme, groups, args):
        super().__init__(scheme, groups, args)
        self.simlarity_func = nn.CosineSimilarity(dim=3)
    @th.no_grad()
    def get_comm_mask(self, agent_mask, z):
        cos_matrix = self.simlarity_func(z.unsqueeze(1),z.unsqueeze(2)) #[bs*ts, na, na]

        #get the 2^n mask, 1 means sample, 0 means not (contrary to the agent_mask)
        mask  = 2 ** th.arange(self.args.n_agents - 1, -1, -1).to(cos_matrix.device)
        x = th.arange(2**self.args.n_agents).to(cos_matrix.device)
        y = x.unsqueeze(-1).bitwise_and(mask).ne(0) #[2^n,n] from [0,0,0,...,0] to [1,1,1,...,1]
        m = th.logical_and(y.unsqueeze(1), y.unsqueeze(-1)) #[2^n, n, n]
        #get the determinant of all subsets
        cofactor_matrix = cos_matrix.unsqueeze(1) * m.unsqueeze(0) #[bs*ts, 2^n, n, n]
        missed_ones = th.diag(th.ones(self.args.n_agents)).unsqueeze(0).to(cos_matrix.device)*th.logical_not(m) #[2^n,n,n]
        cofactor_matrix += missed_ones.unsqueeze(0)
        p = th.clamp(th.linalg.det(cofactor_matrix), min=1e-10) #[bs*ts, 2^n]
        #mask out unavailable agents
        need_fill = th.logical_and(y.unsqueeze(0), agent_mask.unsqueeze(1)).sum(2).ne(0) #[bs*ts, 2^n], True means discard the p
        p = p.masked_fill(th.logical_or(th.logical_or(need_fill, th.isnan(p)), th.isinf(p)), 0)
        p[:,0]=0 #We can't sample 0 agents
        invalid_mask = p.sum(1)==0 #this sample is invalid. sample 0.
        p[:,0] = p[:,0].masked_fill(invalid_mask, 1.0)
        self.p = p
        try:
            cat = Categorical(probs=p)
        except:
            error_dict = {'p':p.detach().cpu().numpy(), 'agent_mask':agent_mask.detach().cpu().numpy(), 'z':z.detach().cpu().numpy()}
            import pickle
            with open("./error_dict.pkl", "wb") as f:
                pickle.dump(error_dict, f)
            print('Saving p, exit.')
            exit(0)
        self.s = cat.sample()
        header_mask = self.s.unsqueeze(-1).bitwise_and(mask).ne(0)
        #return th.logical_or(sample_mask.unsqueeze(1), sample_mask.unsqueeze(-1)), sample_mask
        return header_mask
    @th.no_grad()
    def decide_group(self, agent_inputs, avail_actions, head_feature,test_mode=False):
        entity, obs_mask, entity_mask = agent_inputs
        bs, ts, _ = entity_mask.shape
        #agent_state = entity[:,:,:self.n_agents,:]
        agent_vis_mask = obs_mask[:,:, :self.n_agents, :self.n_agents]
        agent_mask = entity_mask[...,:self.n_agents]
        header = self.get_comm_mask(agent_mask.view(bs*ts, -1),
            head_feature.view(bs*ts,self.n_agents,-1))
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
                # th.max(control_message[:,:,random_ind, i], dim=2)[1] # TODO: duplicate codes
                ind_lst.append(F.one_hot(random_ind[th.max(control_message[:,:,random_ind,i], dim=2)[1]], num_classes=self.n_agents))
                ind = th.cat(ind_lst, dim=1).permute(0,2,1).unsqueeze(1)
        else: #choose first seen header:
            ind = F.one_hot(th.max(control_message, dim=2)[1], self.n_agents).permute(0,1,3,2)
        control_message *= ind.bool()
        #bs*1*n*n, control_message[:,0, i,j]=True means agent i leads agent j.
        return control_message.detach(), header
    def message_comm(self, agent_inputs, avail_actions, t, train_mode=False, test_mode=False, **kwargs):
        # TODO: concate hidden state feature?
        entity, obs_mask, entity_mask = agent_inputs
        if train_mode:
            message_personal, msg_dis, msg_dis_inf, head_feature = self.agent(agent_inputs, self.hidden_states, ret_inf_msg=True, ret_attn_input=True, **kwargs)
        else:
            message_personal, _, head_feature = self.agent(agent_inputs, self.hidden_states, ret_attn_input=True) #bs*n*msg_d  
        message_personal = message_personal.squeeze(1)
        lt = t.stop-t.start

        if lt == 1 and t.start % self.args.msg_T==0: #only needed for interact with env. For training only msg_dis and msg_dis_inf is needed.
            message_matrix, header = self.decide_group(agent_inputs, avail_actions, head_feature, test_mode=test_mode) #bs*n*n, bs*n
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