import numpy as np
import mesa.space
from mesa import Agent
import math
import random
from agent.epsilon_greedy import EpsilonGreedy


def is_collision(agent, planned_coord, init_coords, radius, speed, planned_angle_of_movement):
    if agent.model.space.out_of_bounds(planned_coord):
        temp_coords = agent.model.space.torus_adj(planned_coord)
    else:
        temp_coords = planned_coord


    neighbor_ids = [i.unique_id for i in agent.model.space.get_neighbors(temp_coords, radius, True)]

    if agent.unique_id in neighbor_ids:
        neighbor_ids.remove(agent.unique_id)

    if len(neighbor_ids) > 0:
        return True

    return False




def compute_destination(angle, coordinates, speed):
    x_incr = speed * round(math.sin(math.radians(angle)), 5)
    y_incr = speed * round(math.cos(math.radians(angle)), 5)

    return (coordinates[0] + x_incr, coordinates[1] + y_incr)

class Skater(Agent):
    
    def __init__(self, unique_id, model):
        
        super().__init__(unique_id, model)
        self.alg = EpsilonGreedy(model.epsilon, model.A, model.init_value)
        self.first_action = random.choice(model.A)
        self.first_step = True
        self.curr_action = self.first_action
        self.stuck_streak = 0
        self.anti_stuck = False
        self.collisions = 0
        self.time_step_count = 0
        self.epsilon = model.epsilon
        self.orig_radius = self.model.radius
        self.radius = self.model.radius
       
        
    def step(self):
        self.collisions = 0 
        if self.first_step:
            planned_angle_of_movement = self.first_action
            self.first_step = False
        else:
            planned_angle_of_movement = self.alg.selectAction()
           

        self.curr_action = planned_angle_of_movement
        self.time_step_count += 1

        if self.time_step_count % 50000 == 0:
            self.alg.epsilon = self.alg.epsilon / 2

        dest_coord = compute_destination(planned_angle_of_movement, self.pos, self.model.speed)

        if is_collision(self, dest_coord, self.pos, self.radius, self.model.speed, planned_angle_of_movement):
            #print("Colll")
            self.alg.update(self.model.A.index(planned_angle_of_movement), self.model.R2)
            self.collisions = 1
            self.stuck_streak += 1
            if self.stuck_streak > 300:
                self.radius = 0

        else:
            self.radius = self.orig_radius
            self.stuck_streak = 0
            self.anti_stuck = False
            self.alg.update(self.model.A.index(planned_angle_of_movement), self.model.R1)
            self.model.space.move_agent(self, dest_coord)
       
        