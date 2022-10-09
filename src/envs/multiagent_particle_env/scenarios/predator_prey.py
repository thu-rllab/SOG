import numpy as np
from envs.multiagent_particle_env.core import World, Agent, Landmark
from envs.multiagent_particle_env.scenario import BaseScenario
from envs.multiagent_particle_env.core import Action
from copy import deepcopy

class Scenario(BaseScenario):
    # The predator_prey must be reset with args constrain_num!
    def make_world(self, 
                num_predator=6, 
                num_prey=2,
                num_landmark=3,
                num_hole=2,
                catching_range = 0.14, #not used
                sight_range_kind = 1,):
        world = World()
        # add agents
        world.num_predator=num_predator
        world.num_prey=num_prey
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmark)]+[Landmark() for i in range(num_hole)]
        world.num_landmark = num_landmark
        world.catching_range = catching_range
        world.num_hole = num_hole
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.movable = False
            landmark.size = 0.05
            landmark.collide = True if i < num_landmark else False
            landmark.catching_range = catching_range

        # make initial conditions
        world.max_speed_pred_set = [0.2, 0.3, 0.5]
        world.max_speed_prey_set = [1,1.2]
        world.sight_range_set = [0.2, 0.5, 1, 0.35]
        world.home = np.array([0.0, 0.0])
        world.max_entity_size = 10
        world.max_n_agents = num_predator
        world.max_n_enemies = num_prey
        world.max_n_entities = num_landmark+num_hole #entity without agents
        world.episode_limit = 145
        world.sight_range_kind = sight_range_kind
        self.reset_world(world, constrain_num=[[num_predator], [num_prey]])
        
        return world

    def random_action(self, agent, world):
        action = Action()
        action.u = np.zeros(world.dim_p)
        action.c = np.zeros(world.dim_c)
        ra = np.random.choice(5)
        if ra == 1: action.u[0] = -1.0
        if ra == 2: action.u[0] = +1.0
        if ra == 3: action.u[1] = -1.0
        if ra == 4: action.u[1] = +1.0
        #if out of bound, then return
        if agent.state.p_pos[0] >=1.1: action.u[0]=-1.0
        if agent.state.p_pos[0] <=-1.1: action.u[0]=1.0
        if agent.state.p_pos[1] >=1.1: action.u[1]=-1.0
        if agent.state.p_pos[1] <=-1.1: action.u[1]=1.0
        if agent.accel is not None:
            sensitivity = agent.accel
        action.u *= sensitivity
        return action

    def reset_world(self, world, constrain_num=None):
        if constrain_num is not None:
            num_predator = np.random.choice(constrain_num[0])
            world.num_predator = num_predator
            agent_pred = [Agent() for i in range(num_predator)]
            for i, agent in enumerate(agent_pred):
                agent.name = 'predator %d' % i
                agent.collide = True
                agent.silent = True
                agent.size = 0.05 
                agent.color = np.array([0.25,0.75,0.25])
                agent.accel = 3.0
                agent.is_pred=True 
                agent.max_speed = np.random.choice(world.max_speed_pred_set)
                agent.sight_range = world.sight_range_set[world.sight_range_kind]
            num_prey = np.random.choice(constrain_num[1])
            agent_prey = [Agent() for i in range(num_prey)]
            for i, agent in enumerate(agent_prey):
                agent.name = 'prey %d' % i
                agent.collide = True
                agent.silent = True
                agent.size = 0.05 
                agent.accel = 4.5 
                agent.color = np.array([0.75,0.25,0.25])
                agent.is_pred = False
                agent.max_speed = np.random.choice(world.max_speed_prey_set)
                agent.sight_range = world.sight_range_set[world.sight_range_kind]
                agent.catching_range = world.catching_range
                agent.action_callback = self.random_action
            world.agents = agent_pred + agent_prey
        # random properties for agents
        for i, agent in enumerate(world.agents):
            speed_set = world.max_speed_pred_set if i < world.num_predator else world.max_speed_prey_set
            agent.max_speed = np.random.choice(speed_set)
            #partially observable
            agent.sight_range = world.sight_range_set[world.sight_range_kind]
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.2,0.2,0.2]) if i < world.num_landmark else np.array([0.7,0.7,0.7])
            if not landmark.collide: #is hole
                landmark.open_tc = 10
        # set random initial states
        for i, agent in enumerate(world.agents):
            if i < world.num_predator:
                agent.state.p_pos = np.random.uniform(-0.8, -0.5, world.dim_p)
            else:
                agent.state.p_pos = np.random.uniform(0.5, 0.8, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-0.9, +0.9, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        world.time_step = 0
        world.prey_remain = num_prey

    def done(self, world):
        if world.time_step >= world.episode_limit: return True
        if world.prey_remain == 0: return True
        return False
    
    def dist(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dis = np.sqrt(np.sum(np.square(delta_pos)))
        return dis
    
    def can_catch(self, agent1, agent2):
        dis = self.dist( agent1, agent2)
        return True if dis <= agent1.catching_range else False
    
    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist <= dist_min else False

    def predator(self, world):
        return [agent for agent in world.agents if agent.is_pred]
    
    def prey(self, world):
        return [agent for agent in world.agents if not agent.is_pred]
    
    def hole(self, world):
        return [landmark for landmark in world.landmarks if not landmark.collide]
    
    def obstacle(self, world):
        return [landmark for landmark in world.landmarks if landmark.collide]
    
    def reward(self, world):
        remove_list = []
        prey = self.prey(world)
        hole = self.hole(world)
        predator = self.predator(world)
        rew = 0
        for landmark in hole:
            if landmark.open_tc > 0: #not open
                landmark.open_tc -= 1
        for pr in prey:
            for i, ho in enumerate(hole):
                if self.is_collision(pr, ho) and ho.open_tc == 0:
                    ind = [j for j in range(len(hole))]
                    ind.remove(i)
                    tar_hole = hole[np.random.choice(ind)]
                    ho.open_tc=10
                    pr.state.p_pos = deepcopy(tar_hole.state.p_pos)+np.random.uniform(-0.02, +0.02, world.dim_p) # move to another hole
            rew -= 0.1 * min([self.dist(pr,pd) for pd in predator])
            close_num = 0
            for pd in predator:
                if self.can_catch(pr, pd):
                    close_num += 1
            if close_num >= 3:
                rew += 10
                remove_list.append(pr)
                world.prey_remain -= 1
        for pr in remove_list:
            world.agents.remove(pr)
        return rew

    def get_entity(self, world):
        all_entities = []
        predator = self.predator(world)
        prey = self.prey(world)
        for agent in predator:
            st = np.zeros(world.max_entity_size)
            st[0:2] = agent.state.p_pos
            st[2:4] = agent.state.p_vel
            st[4] = agent.max_speed
            st[5] = agent.sight_range
            st[6:10] = [1,0,0,0] #entity kind is predator
            all_entities.append(st)
        for _ in range(world.max_n_agents - len(predator)):
            all_entities.append(np.zeros(world.max_entity_size))
        for agent in prey:
            st = np.zeros(world.max_entity_size)
            st[0:2] = agent.state.p_pos
            st[2:4] = agent.state.p_vel
            st[4] = agent.max_speed
            st[5] = agent.sight_range
            st[6:10] = [0,1,0,0] #entity kind is prey
            all_entities.append(st)
        for _ in range(world.max_n_enemies - len(prey)):
            all_entities.append(np.zeros(world.max_entity_size))
        for i, landmark in enumerate(world.landmarks):
            st = np.zeros(world.max_entity_size)
            st[0:2] = landmark.state.p_pos
            st[2:4] = [0.0,0.0]
            st[4] = 0.0
            st[5] = 0.0
            if landmark.collide: # is obstacle
                st[6:10] = [0,0,1,0]
            else: # is hole
                st[6:10] = [0,0,0,1]
            all_entities.append(st)
        return all_entities

    def get_mask(self, world):
        #mask[i,j] =1 means i can not observe j
        predator = self.predator(world)
        prey = self.prey(world)
        obs_mask = np.ones([world.max_n_agents+world.max_n_enemies+world.max_n_entities, \
                world.max_n_agents+world.max_n_enemies+world.max_n_entities])
        for i, agent1 in enumerate(predator):
            for j, agent2 in enumerate(predator):
                if self.dist(agent1, agent2) <= agent1.sight_range:
                    obs_mask[i,j] = 0
            for j, agent2 in enumerate(prey):
                if self.dist(agent1, agent2) <= agent1.sight_range:
                    obs_mask[i,world.max_n_agents+j] = 0
            for j, landmark in enumerate(world.landmarks):
                if self.dist(agent1, landmark) <= agent1.sight_range:
                    obs_mask[i, world.max_n_agents+world.max_n_enemies+j] = 0
        entity_mask = np.ones(world.max_n_agents +world.max_n_enemies+ world.max_n_entities,dtype=np.uint8)
        entity_mask[:len(predator)] = 0
        return obs_mask, entity_mask
        
    
    