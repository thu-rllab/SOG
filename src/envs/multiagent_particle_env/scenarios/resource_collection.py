import numpy as np
from envs.multiagent_particle_env.core import World, Agent, Landmark
from envs.multiagent_particle_env.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, 
                num_agents=6, 
                num_resource=6, 
                mining_radius = 0.2,
                sight_range_kind = 1,):
        world = World()
        # add agents
        world.resource_kind=3
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_resource)]
        resource_kind = [0,0,1,1,2,2]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.05
            landmark.mining_radius = mining_radius
            landmark.resource_kind = resource_kind[i]
        home = Landmark()
        home.name = "home"
        home.collide = False
        home.movable = False
        home.color=[0.9,0.9,0.9]
        home.size = 0.1
        home.mining_radius = 0.15

        world.landmarks.append(home)
        # make initial conditions
        world.ability_set = [0.1,0.5,0.9]
        world.max_speed_set = [0.3,0.5,0.7]
        world.sight_range_set = [0.2, 0.5, 1, 0.8]
        world.home = np.array([0.0, 0.0])
        
        world.max_n_agents = num_agents
        world.max_n_enemies = 0
        world.max_n_entities = 7 #entity without agents
        world.max_entity_size = world.max_n_agents + world.max_n_enemies + world.max_n_entities
        world.episode_limit = 145
        world.sight_range_kind = sight_range_kind
        self.reset_world(world, constrain_num=[num_agents])
        
        return world

    def reset_world(self, world, constrain_num=None):
        if constrain_num is not None:
            num_agents = np.random.choice(constrain_num)
            world.agents = [Agent() for i in range(num_agents)]
            for i, agent in enumerate(world.agents):
                agent.name = 'agent %d' % i
                agent.collide = True
                agent.silent = True
                agent.size = 0.05 
                agent.accel = 3.0 
                agent.max_speed = 1.0
                agent.ability = [0.5]*world.resource_kind
                agent.sight_range = world.sight_range_set[world.sight_range_kind]
        # random properties for agents
        for i, agent in enumerate(world.agents):
            ability = np.random.randint(0,len(world.ability_set),world.resource_kind)
            agent.ability = [world.ability_set[i] for i in ability]
            agent.color = agent.ability
            agent.max_speed = world.max_speed_set[np.random.randint(0,len(world.max_speed_set))]
            #partially observable
            agent.sight_range = world.sight_range_set[world.sight_range_kind]
        for i, landmark in enumerate(world.landmarks):
            if i < len(world.landmarks) -1:
                landmark.color = np.array([0.25, 0.25, 0.25])
                landmark.color[landmark.resource_kind] += 0.5
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-0.2, +0.2, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.resource = np.zeros(world.resource_kind)
        for i, landmark in enumerate(world.landmarks):
            if i < len(world.landmarks) -1:
                landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            else: #its home
                landmark.state.p_pos = np.random.uniform(-0.4, +0.4, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        world.time_step = 0
    
    def can_collecting(self, agent, landmark):
        dis = self.dist( agent, landmark)
        return True if dis <= landmark.mining_radius else False
    
    def dist(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dis = np.sqrt(np.sum(np.square(delta_pos)))
        return dis

    def get_entity(self, world):
        all_entities = []
        for agent in world.agents:
            st = np.zeros(world.max_entity_size)
            st[0:2] = agent.state.p_pos
            st[2:4] = agent.state.p_vel
            st[4] = agent.max_speed
            st[5] = agent.sight_range
            st[6:9] = agent.ability
            st[9:12] = [1,0,0] #entity kind is agent
            st[12:15] = agent.state.resource #resource_kind
            all_entities.append(st)
        for _ in range(world.max_n_agents - len(world.agents)):
            all_entities.append(np.zeros(world.max_entity_size))
        for i, landmark in enumerate(world.landmarks):
            st = np.zeros(world.max_entity_size)
            st[0:2] = landmark.state.p_pos
            st[2:4] = [0.0,0.0]
            st[4] = 0.0
            st[5] = 0.0
            st[6:9] = [0.0,0.0,0.0]
            st[12:15] = [0,0,0]
            if i< len(world.landmarks)-1:
                st[9:12] = [0,1,0] #resource
                st[landmark.resource_kind+12] = 1 #resource_kind
            else:
                st[9:12] = [0,0,1] #home
            all_entities.append(st)
        return all_entities

    def get_mask(self, world):
        #mask[i,j] =1 means i can not observe j
        obs_mask = np.ones([world.max_n_agents+world.max_n_entities, world.max_n_agents+world.max_n_entities])
        for i, agent1 in enumerate(world.agents):
            for j, agent2 in enumerate(world.agents):
                if self.dist(agent1, agent2) <= agent1.sight_range:
                    obs_mask[i,j] = 0
            for j, landmark in enumerate(world.landmarks):
                if self.dist(agent1, landmark) <= agent1.sight_range:
                    obs_mask[i, world.max_n_agents+j] = 0
            obs_mask[i, world.max_n_agents+world.max_n_entities-1] = 0 #can observe home at anytime
        entity_mask = np.ones(world.max_n_agents + world.max_n_entities,dtype=np.uint8)
        entity_mask[:len(world.agents)] = 0
        return obs_mask, entity_mask
    
    def done(self, world):
        return True if world.time_step >= world.episode_limit else False

    def reward(self, world):
        r = 0.0
        home = world.landmarks[-1]
        for agent in world.agents:
            if sum(agent.state.resource) >= 1:
                if self.can_collecting(agent, home):
                    agent.state.resource = np.zeros(world.resource_kind)
                    r += 1
            else:
                for i, landmark in enumerate(world.landmarks):
                    if i < len(world.landmarks) - 1: #not home
                        if self.can_collecting(agent, landmark):
                            agent.state.resource[landmark.resource_kind] = 1
                            r += 10 * agent.ability[landmark.resource_kind]
                            break
        return r
        