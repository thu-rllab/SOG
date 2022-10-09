import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from .multi_discrete import MultiDiscrete
from .scenarios import load as sload
from envs.multiagentenv import MultiAgentEnv

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentParticleEnv(gym.Env, MultiAgentEnv):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, 
                scenario_id="resource_collection.py", 
                num_agents = 6,
                num_resource=6, 
                mining_radius=0.2, 
                seed=None, 
                entity_scheme=True,
                sight_range_kind = 1, 
                num_predator = 6, 
                num_prey = 2,
                shared_viewer=True):
        np.random.seed(seed)
        self.entity_scheme = entity_scheme
        scenario = sload(scenario_id).Scenario()
        if scenario_id == "resource_collection.py":
            self.world = scenario.make_world(num_agents=num_agents, 
                                        num_resource=num_resource, 
                                        mining_radius=mining_radius, 
                                        sight_range_kind=sight_range_kind,)
        elif scenario_id == "predator_prey.py":
            # The predator_prey must be reset with args constrain_num!
            self.world = scenario.make_world(num_predator=num_predator, 
                                            num_prey=num_prey,
                                        sight_range_kind=sight_range_kind,)
        self.agents = self.world.policy_agents
        # set required vectorized gym env property
        self.n = len(self.world.policy_agents)
        # scenario callbacks
        self.reset_callback = scenario.reset_world
        self.reward_callback = scenario.reward
        self.observation_callback = None
        self.info_callback = None
        self.done_callback = scenario.done
        self.entity_callback = scenario.get_entity
        self.mask_callback = scenario.get_mask
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = True
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = self.world.discrete_action if hasattr(self.world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.time = 0

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # physical action space
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(self.world.dim_p * 2 + 2)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(self.world.dim_p,), dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # communication action space
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(self.world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(self.world.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(self.world.max_entity_size,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

        self.episode_limit = self.world.episode_limit
        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def step(self, action_n):
        action_n = [int(a) for a in action_n]
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()
        reward = self._get_reward()
        done = self._get_done()
        info = {}
        return reward, done, info

    def reset(self, constrain_num=None, test=False, index=None):
        # reset world
        self.reset_callback(self.world, constrain_num=constrain_num)
        # reset renderer
        self._reset_render()
        # record observations for each agent
        self.agents = self.world.policy_agents
        self.n = len(self.world.policy_agents)
        return self.get_entities(), self.get_masks()
    
    def get_entities(self):
        if self.entity_callback is None:
            return None
        return self.entity_callback(self.world)

    def get_masks(self):
        if self.mask_callback is None:
            return None
        return self.mask_callback(self.world)
    
    def get_env_info(self, args):
        env_info = {"entity_shape": self.world.max_entity_size,
                    "n_actions": 6,
                    "n_agents": self.world.max_n_agents,
                    "n_entities": self.world.max_n_agents + self.world.max_n_enemies+self.world.max_n_entities,
                    "episode_limit": self.world.episode_limit}
        return env_info
    
    def get_avail_actions(self):
        n_agent = len(self.agents)
        un_agent = self.world.max_n_agents - n_agent
        return [[0,1,1,1,1,1] for _ in range(n_agent)]+[[1,0,0,0,0,0] for _ in range(un_agent)]

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self):
        if self.done_callback is None:
            return False
        return self.done_callback(self.world)

    # get reward for a particular agent
    def _get_reward(self):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # process action
        # if isinstance(action_space, MultiDiscrete):
        #     act = []
        #     size = action_space.high - action_space.low + 1
        #     index = 0
        #     for s in size:
        #         act.append(action[index:(index+s)])
        #         index += s
        #     action = act
        # else:
        action = [action]

        if agent.movable:
            # physical action
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # process discrete action
                if action[0] == 2: agent.action.u[0] = -1.0
                if action[0] == 3: agent.action.u[0] = +1.0
                if action[0] == 4: agent.action.u[1] = -1.0
                if action[0] == 5: agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                else:
                    agent.action.u = action[0]
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            # print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from envs.multiagent_particle_env import rendering
                self.viewers[i] = rendering.Viewer(700,700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from envs.multiagent_particle_env import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from envs.multiagent_particle_env import rendering
            # update bounds to center around agent
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

        return results

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x,y]))
        return dx
    
    def close(self):
        print("Env closed.")
    
    def save_replay(self):
        print("Saving replay function not implemented.")

