import torch
import torch.nn.functional as F
from .comm_controller import CommMAC
from modules.agents import REGISTRY as agent_REGISTRY


class RlCommMAC(CommMAC):
    def __init__(self, scheme, groups, args):
        super(RlCommMAC, self).__init__(scheme, groups, args)
        self.elector = agent_REGISTRY["election_agent"](self.input_shape, args)
        self.head_prob = None
        self.selected_head = None
    def decide_header(self, entity, entity_mask):
        # TODO: lack of alive agent mask
        header_dist = self.elector(entity, entity_mask) # B, max_n_agent
        header = header_dist.sample() # 
        head_prob = header_dist.log_prob(header).exp() # B, max_n_agent
        self.head_prob = torch.prod(head_prob, dim=-1) # TODO: check here again and again
        self.selected_head = header
        return header.detach().squeeze(1)
    def decide_group(self, agent_inputs, avail_actions, test_mode=False):
        entity, obs_mask, entity_mask = agent_inputs
        # header try to dominate all visible agents
        agent_vis_mask = obs_mask[:,:, :self.n_agents, :self.n_agents]
        if test_mode:
            with torch.no_grad():
                header = self.decide_header(entity, entity_mask)
        else:
            header = self.decide_header(entity, entity_mask)
        bs, _=header.shape

        #control_message[i,j]=True means i want to dominate j
        control_message = header.unsqueeze(1).unsqueeze(3).repeat(1,1,1,self.n_agents)*torch.logical_not(agent_vis_mask)
        #remove message trying to control a header
        control_message *= torch.logical_not(header.unsqueeze(1).unsqueeze(2).repeat(1,1,self.n_agents,1)) #bs*1*n*n
        #remain only one master for each agent
        if self.args.random_master: #choose the random seen header as leader
            ind_lst = []
            for i in range(self.n_agents):
                random_ind = torch.randperm(self.n_agents, device = self.args.device)
                ind_lst.append(F.one_hot(
                    random_ind[torch.max(control_message[:,:,random_ind,i], dim=2)[1]],
                    num_classes=self.n_agents))
                ind = torch.cat(ind_lst, dim=1).permute(0,2,1).unsqueeze(1)
        else: #choose first seen header:
            ind = F.one_hot(torch.max(control_message, dim=2)[1], self.n_agents).permute(0,1,3,2)
        control_message *= ind.bool()
        #bs*1*n*n, control_message[:,0, i,j]=True means agent i leads agent j.
        return control_message.detach(), header
    def select_actions(self, ep_batch, t_ep, t_env, bs=..., test_mode=False,
                             ret_agent_outs=False, ret_msg=False):
        # set head-related variables as None
        self.head_prob = None
        self.selected_head = None # head-election action
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs, self_msg, head_msg = self.forward(ep_batch, t_ep,
            test_mode=test_mode, fix_msg=None)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs],
            avail_actions[bs], t_env, test_mode=test_mode)
        if ret_agent_outs:
            return chosen_actions, agent_outputs[bs], self_msg[bs], head_msg[bs]
        if ret_msg:
            return chosen_actions, self_msg[bs], head_msg[bs], self.head_prob, self.selected_head
        return chosen_actions
    def elector_parameters(self):
        return self.elector.parameters()
    def save_models(self, path):
        torch.save(self.agent.state_dict(), "{}/agent.th".format(path))
        torch.save(self.elector.state_dict(), "{}/elector.th".format(path))
    def load_models(self, path):
        self.agent.load_state_dict(torch.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))
        self.elector.load_state_dict(torch.load("{}/elector.th".format(path), map_location=lambda storage, loc: storage))
    def cuda(self):
        super().cuda()
        self.elector.cuda()
