import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.flex_qmix import FlexQMixer, LinearFlexQMixer
from modules.mixers.weighted_vdn import WVDNMixer
import torch as th
from torch.optim import RMSprop


class GatQLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.local_q_weight=None
        self.unnorm_local_q_weight = None
        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            elif args.mixer == "flex_qmix":
                assert args.entity_scheme, "FlexQMixer only available with entity scheme"
                self.mixer = FlexQMixer(args)
            elif args.mixer == "lin_flex_qmix":
                assert args.entity_scheme, "FlexQMixer only available with entity scheme"
                self.mixer = LinearFlexQMixer(args)
            elif args.mixer == "wvdn":
                assert args.entity_scheme, "WVDNMixer only available with entity scheme"
                self.mixer = WVDNMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps,
                                 weight_decay=args.weight_decay)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.current_id = 0
        self.T = 0

    def _get_mixer_ins(self, batch, repeat_batch=1):
        if not self.args.entity_scheme:
            return (batch["state"][:, :-1].repeat(repeat_batch, 1, 1),
                    batch["state"][:, 1:])
        else:
            entities = []
            bs, max_t, ne, ed = batch["entities"].shape
            entities.append(batch["entities"])
            if self.args.entity_last_action:
                last_actions = th.zeros(bs, max_t, ne, self.args.n_actions,
                                        device=batch.device,
                                        dtype=batch["entities"].dtype)
                last_actions[:, 1:, :self.args.n_agents] = batch["actions_onehot"][:, :-1]
                entities.append(last_actions)

            entities = th.cat(entities, dim=3)
            return ((entities[:, :-1].repeat(repeat_batch, 1, 1, 1),
                     batch["entity_mask"][:, :-1].repeat(repeat_batch, 1, 1)),
                    (entities[:, 1:],
                     batch["entity_mask"][:, 1:]))

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        with th.no_grad():
            self.mac.train()
            self.mixer.train()
            mix_ins, _ = self._get_mixer_ins(batch)
            self.mac.init_hidden(batch.batch_size)
            mac_out_pos = self.mac.forward(batch, t=None, train_id=self.current_id , force_msg=True)
            chosen_action_qvals_pos = th.gather(mac_out_pos[:, :-1], dim=3, index=actions).squeeze(3)
            global_action_qvals_pos = self.mixer(chosen_action_qvals_pos, mix_ins)
            self.mac.init_hidden(batch.batch_size)
            mac_out_neg = self.mac.forward(batch, t=None, train_id=self.current_id , force_msg=False)
            chosen_action_qvals_neg = th.gather(mac_out_neg[:, :-1], dim=3, index=actions).squeeze(3)
            global_action_qvals_neg = self.mixer(chosen_action_qvals_neg, mix_ins)
            deltaQ = (global_action_qvals_pos - global_action_qvals_neg).detach()
            label = (deltaQ.detach() > self.T).long().squeeze() #[bs, ts]
            bs, ts = label.shape
        self.mac.init_hidden(batch.batch_size)
        # enable things like dropout on mac and mixer, but not target_mac and target_mixer
        self.target_mac.eval()
        self.target_mixer.eval()
        if 'imagine' in self.args.agent:
            all_mac_out, x2_gate, groups = self.mac.forward(batch, t=None, imagine=True, train_mode=True)
            # Pick the Q-Values for the actions taken by each agent
            rep_actions = actions.repeat(3, 1, 1, 1)
            all_chosen_action_qvals = th.gather(all_mac_out[:, :-1], dim=3, index=rep_actions).squeeze(3)  # Remove the last dim
            x2_gate = x2_gate.chunk(3,dim=0)[0]
            mac_out, moW, moI = all_mac_out.chunk(3, dim=0)
            chosen_action_qvals, caqW, caqI = all_chosen_action_qvals.chunk(3, dim=0)
            caq_imagine = th.cat([caqW, caqI], dim=2)
        else:
            mac_out, x2_gate = self.mac.forward(batch, t=None, train_mode=True)
            # Pick the Q-Values for the actions taken by each agent
            chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        self.target_mac.init_hidden(batch.batch_size)
        target_mac_out = self.target_mac.forward(batch, t=None)
        avail_actions_targ = avail_actions
        target_mac_out = target_mac_out[:, 1:]

        # Mask out unavailable actions
        target_mac_out[avail_actions_targ[:, 1:] == 0] = -9999999  # From OG deepmarl

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions_targ == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]
        
        # Mix
        if self.mixer is not None:
            if 'imagine' in self.args.agent:
                mix_ins, targ_mix_ins = self._get_mixer_ins(batch)
                global_action_qvals = self.mixer(chosen_action_qvals,
                                                 mix_ins)
                # don't need last timestep
                groups = [gr[:, :-1] for gr in groups]
                caq_imagine = self.mixer(caq_imagine, mix_ins,
                                             imagine_groups=groups)
            else:
                mix_ins, targ_mix_ins = self._get_mixer_ins(batch)
                global_action_qvals = self.mixer(chosen_action_qvals, mix_ins)

            target_max_qvals = self.target_mixer(target_max_qvals, targ_mix_ins)
        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        # Td-error
        td_error = (global_action_qvals - targets.detach())
        mask = mask.expand_as(td_error)
        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        # Normal L2 loss, take mean over actual data
        qloss = (masked_td_error ** 2).sum() / mask.sum()
        loss=qloss
        if 'imagine' in self.args.agent:
            im_prop = self.args.lmbda
            im_td_error = (caq_imagine - targets.detach())
            im_masked_td_error = im_td_error * mask
            im_loss = (im_masked_td_error ** 2).sum() / mask.sum()
            loss = (1 - im_prop) * qloss + im_prop * im_loss
        
        agent_mask = batch["entity_mask"][:,:-1, :self.args.n_agents]
        agent_mask = mask.squeeze() * (1-agent_mask[:,:,self.current_id])
        while(agent_mask.sum() == 0):
            self.current_id = (self.current_id + 1) % self.args.n_agents
            agent_mask = batch["entity_mask"][:,:-1, :self.args.n_agents]
            agent_mask = mask.squeeze() * (1-agent_mask[:,:,self.current_id])
        x2_gate = x2_gate.reshape(bs, ts+1, self.args.n_agents, 2)[:, :-1, self.current_id] #[bs, ts, 2]
        CE = th.nn.CrossEntropyLoss(reduction='none')
        ce_loss = CE(x2_gate.reshape(-1,2), label.reshape(-1))
        ce_loss = self.args.ce_weight * (ce_loss.reshape(bs, ts) * agent_mask).sum() / agent_mask.sum()
        loss += ce_loss
        mean_deltaQ = (deltaQ * mask).sum() / mask.sum()
        self.T = (1-self.args.beta)*self.T + self.args.beta * mean_deltaQ.detach()
        self.current_id = (self.current_id + 1) % self.args.n_agents
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        try:
            grad_norm=grad_norm.item()
        except:
            pass
        self.optimiser.step()
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("qloss", qloss.item(), t_env)
            self.logger.log_stat("ce_loss", ce_loss.item(), t_env)
            self.logger.log_stat("gating_rate", ((label * agent_mask).sum() / agent_mask.sum()).item(), t_env)
            self.logger.log_stat("min_local_taken_q", chosen_action_qvals.min(-1)[0].mean().item(), t_env)
            self.logger.log_stat("max_local_taken_q", chosen_action_qvals.max(-1)[0].mean().item(), t_env)
            if 'imagine' in self.args.agent:
                self.logger.log_stat("im_loss", im_loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (global_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path, evaluate=False):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if not evaluate:
            if self.mixer is not None:
                self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
