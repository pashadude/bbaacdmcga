import random
import json
import numpy as np
from PIL import Image
from deap import base, creator, tools, algorithms
#import pygmo as pg


class DeapModifiedImageNSGA2:

    def __init__(self, image_name, classes):
        self.target_image = Image.open(image_name)
        self.pix = self.target_image.load()
        self.current_pixel = [0, 0]
        self.toolbox = base.Toolbox()
        self.ndim = 3
        self.classes = classes

        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        self.toolbox.register("attr_rgb", random.randint(0, 255))
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_rgb, n=3)
        self.toolbox.register("evaluate", self.__get_mc_fitness)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self.__mutate, eta=20.0, indpb=1.0/self.ndim)
        self.toolbox.register("select", tools.selNSGA2)

    def __mutate(self, individual, indpb):
        for i in range(len(individual)):
            individual[i] = random.randint(0, 255) if random.random() <= indpb else individual[i]
        return individual

    def __get_mc_fitness(self, classes):
        return fitness

    def __count_class_fitness(self, classes):
        fitness = 0
        class_count = 0
        for key in self.classes.keys():
            true_class = int(self.classes[key])
            miss_class = abs(true_class - int(classes.get(key)))
            fitness += true_class/(true_class + miss_class)
            class_count += 1
        class_fitness = fitness / class_count
        return class_fitness

class ConvFitnessImageGA: