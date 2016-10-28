#coding=u8
import math

def sigmod(x):
    return 1 / (1 + math.exp(-x))

def logloss(positive_probability, actuality):
    epsilon = 1e-15
    positive_probability = max(positive_probability, epsilon)
    positive_probability = min(positive_probability, 1 - epsilon)
    return actuality * math.log(positive_probability) + (1 - actuality) * math.log(1 - positive_probability)
