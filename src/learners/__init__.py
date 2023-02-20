from telnetlib import GA
from .q_learner import QLearner
from .msg_q_learner import MsgQLearner
from .copa_q_learner import COPAQLearner
from .gat_q_learner import GatQLearner

REGISTRY = {}
REGISTRY["q_learner"] = QLearner
REGISTRY["msg_q_learner"] = MsgQLearner
REGISTRY["copa_q_learner"] = COPAQLearner
REGISTRY["gat_q_learner"] = GatQLearner

