
import random
import numpy as np

from mesa import Model
from mesa.space import ContinuousSpace
from mesa.time import BaseScheduler
from mesa.datacollection import DataCollector

import matplotlib.pyplot as plt

from agent.skater import Skater






def compute_percentage_angle_0(model):

    percentage_angle_0 = sum([1 for agent in model.schedule.agents if agent.curr_action == model.A[0]]) / len(model.schedule.agents)
    #x = sorted(agent_wealths)
    #N = model.num_agents
    #B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x))
    return percentage_angle_0


def compute_percentage_angle_1(model):

    percentage_angle_1 = sum([1 for agent in model.schedule.agents if agent.curr_action == model.A[1]]) / len(model.schedule.agents)
    #x = sorted(agent_wealths)
    #N = model.num_agents
    #B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x))
    return percentage_angle_1

def compute_percentage_angle_2(model):

    percentage_angle_2 = sum([1 for agent in model.schedule.agents if agent.curr_action == model.A[2]]) / len(model.schedule.agents)
    #x = sorted(agent_wealths)
    #N = model.num_agents
    #B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x))
    return percentage_angle_2

def compute_percentage_angle_3(model):

    percentage_angle_3 = sum([1 for agent in model.schedule.agents if agent.curr_action == model.A[3]]) / len(model.schedule.agents)
    #x = sorted(agent_wealths)
    #N = model.num_agents
    #B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x))
    return percentage_angle_3

def compute_percentage_angle_4(model):

    percentage_angle_4 = sum([1 for agent in model.schedule.agents if agent.curr_action == model.A[4]]) / len(model.schedule.agents)
    #x = sorted(agent_wealths)
    #N = model.num_agents
    #B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x))
    return percentage_angle_4

def compute_percentage_angle_5(model):

    percentage_angle_5 = sum([1 for agent in model.schedule.agents if agent.curr_action == model.A[5]]) / len(model.schedule.agents)
    #x = sorted(agent_wealths)
    #N = model.num_agents
    #B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x))
    return percentage_angle_5

def compute_percentage_angle_6(model):

    percentage_angle_6 = sum([1 for agent in model.schedule.agents if agent.curr_action == model.A[6]]) / len(model.schedule.agents)
    #x = sorted(agent_wealths)
    #N = model.num_agents
    #B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x))
    return percentage_angle_6

def compute_percentage_angle_7(model):

    percentage_angle_7 = sum([1 for agent in model.schedule.agents if agent.curr_action == model.A[7]]) / len(model.schedule.agents)
    #x = sorted(agent_wealths)
    #N = model.num_agents
    #B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x))
    return percentage_angle_7

def compute_percentage_angle_8(model):

    percentage_angle_8 = sum([1 for agent in model.schedule.agents if agent.curr_action == model.A[8]]) / len(model.schedule.agents)
    #x = sorted(agent_wealths)
    #N = model.num_agents
    #B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x))
    return percentage_angle_8

def compute_percentage_angle_9(model):

    percentage_angle_9 = sum([1 for agent in model.schedule.agents if agent.curr_action == model.A[9]]) / len(model.schedule.agents)
    #x = sorted(agent_wealths)
    #N = model.num_agents
    #B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x))
    return percentage_angle_9






def compute_num_of_collisions(model):
    
    collisions = [agent.collisions for agent in model.schedule.agents]
    return sum(collisions)


def compute_mean_reward_angle_0(model):

    
    reward_avg = [agent.alg.avg_values[0] for agent in model.schedule.agents]
    #x = sorted(agent_wealths)
    #N = model.num_agents
    #B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x))
    return sum(reward_avg) / len(reward_avg)

def compute_mean_reward_angle_1(model):

    
    reward_avg = [agent.alg.avg_values[1] for agent in model.schedule.agents]
    #x = sorted(agent_wealths)
    #N = model.num_agents
    #B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x))
    return sum(reward_avg) / len(reward_avg)

def compute_mean_reward_angle_2(model):

    
    reward_avg = [agent.alg.avg_values[2] for agent in model.schedule.agents]
    #x = sorted(agent_wealths)
    #N = model.num_agents
    #B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x))
    return sum(reward_avg) / len(reward_avg)


def compute_mean_reward_angle_3(model):

    
    reward_avg = [agent.alg.avg_values[3] for agent in model.schedule.agents]
    #x = sorted(agent_wealths)
    #N = model.num_agents
    #B = sum( xi * (N-i) for i,xi in enumerate(x) ) / (N*sum(x))
    return sum(reward_avg) / len(reward_avg)


class SkaterModel(Model):

    def __init__(self, N, width, height, speed, k, R1, R2, radius, epsilon, init_value):
        
        self.N = N
        self.A = [int((360 / k)) * i for i in range(int(360/(360 / k)))] # Assuming 360 % k == 0
        self.speed = speed
        self.R1 = R1
        self.R2 = R2
        self.radius = radius
        self.epsilon = epsilon
        self.init_value = init_value

        self.schedule = BaseScheduler(self)
        self.space = ContinuousSpace(width, height, True,
                                     grid_width=10, grid_height=10)
        self.make_agents()
        self.running = True

        

    def make_agents(self):
      
        for i in range(self.N):
            x = random.random() * self.space.x_max
            y = random.random() * self.space.y_max
            pos = (x, y)
            heading = np.random.random(2) * 2 - np.array((1, 1))
            heading /= np.linalg.norm(heading)
            skater = Skater(i, self)
            self.space.place_agent(skater, pos)
            self.schedule.add(skater)


        #self.datacollector = DataCollector(
        #    model_reporters={"0": compute_mean_reward_angle_0, "90": compute_mean_reward_angle_1, "180": compute_mean_reward_angle_2, "270": compute_mean_reward_angle_3})

        #self.datacollector = DataCollector(
        #    model_reporters={"0": compute_percentage_angle_0, "60": compute_percentage_angle_1, "120": compute_percentage_angle_2, "180": compute_percentage_angle_3, "240": compute_percentage_angle_4, "300": compute_percentage_angle_5})#, "perct_angle_6": compute_percentage_angle_6, "perct_angle_7": compute_percentage_angle_7, "perct_angle_8": compute_percentage_angle_8, "perct_angle_9": compute_percentage_angle_9})

        #self.datacollector = DataCollector(
        #    model_reporters={"0": compute_percentage_angle_0, "90": compute_percentage_angle_1, "180": compute_percentage_angle_2, "270": compute_percentage_angle_3})


        

        self.datacollector = DataCollector(
            model_reporters={"Total Collisions": compute_num_of_collisions})
    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()




N = 60
width = 100
height = 100
speed = 1
k = 6
R1 = 10
R2 = -5
radius = 2
epsilon = 0.05
init_value = 0

time_steps_arr = [700000]

# Have a few of these
arg_arrs = [[60, 1, 6, 10, -5, 2, 0.05, 0]]#, [60, 1, 4, 10, -5, 2, 0, 20]]#, [60, 1, 4, 10, -5, 2, 0, 0], [60, 1, 4, 10, -5, 2, 0, 8], [25, 1, 4, 10, -5, 2, 0.05, 0], [25, 1, 4, 10, -5, 2, 0, 0]]
#arg_arrs = [        [60, 1, 6, 10, -5, 2, 0, 0], [60, 1, 6, 10, -5, 2, 0, 8], [25, 1, 6, 10, -5, 2, 0, 0], [25, 1, 6, 10, -5, 2, 0, 8]     ]    #, [60, 1, 6, 10, -5, 2, 0]]

iters = 1

plot_type = "Collisions"

for arg_arr in arg_arrs:
    for time_steps in time_steps_arr:
        for iter_num in range(iters):

            N = arg_arr[0]
            speed = arg_arr[1]
            k = arg_arr[2]
            R1 = arg_arr[3]
            R2 = arg_arr[4]
            radius = arg_arr[5]
            epsilon = arg_arr[6]
            init_value = arg_arr[7]

            print("ITER: {0}".format(iter_num))

            model = SkaterModel(N, width, height, speed, k, R1, R2, radius, epsilon, init_value)
            for i in range(time_steps):
                if i % 10000 == 0:
                    print("{0} out of {1}".format(i, time_steps))
                model.step()

            #collisions = model.datacollector.get_model_vars_dataframe()
            #print(collisions)
            #collisions.plot()


            avg_reward = model.datacollector.get_model_vars_dataframe()
            #print(avg_reward)
            avg_reward.plot()

            plt.xlim(0, time_steps)#time_steps)
            plt.xlabel("Seconds")
            #plt.ylabel("Mean Reward")
            #plt.ylabel("Percentage of Skaters") # that have chosen the skating angle  - ASK SOPHIE
            plt.ylabel(plot_type) # that have chosen the skating angle  - ASK SOPHIE
            plt.savefig('saved_plots/N={0}_delta={1}_k={2}_R1={3}_R2={4}_radius={5}_epsilon={6}_steps={7}_iter={8}_initValue={9}_plot_type={10}.png'.format(N, speed, k, R1, R2, radius, epsilon, time_steps, iter_num, init_value, plot_type))
            #plt.show()



