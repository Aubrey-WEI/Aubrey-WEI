import random
from deap import base, creator, tools, algorithms
import numpy as np
import pandas as pd

monthly_returns = pd.read_csv('average_monthly_returns_2013.csv').values.flatten()
cov_matrix = pd.read_excel('covariance_matrix_2013.xlsx').values

# 创建适应度函数
def evaluate(individual):
    weights = np.array(individual)
    total = np.sum(weights)
    if total == 0:
        return 0,  # Avoid division by zero
    weights = weights / total  # Normalize weights
    # Ensure monthly_returns and weights are NumPy arrays
    returns = np.array(monthly_returns)
    port_return = np.sum(returns * weights)
    port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = port_return/port_risk if port_risk != 0 else 0
    return sharpe_ratio,  # 目标是最大化夏普比率

def normalize_weights(individual):
    total = sum(individual)
    if total != 0:
        individual = [x / total for x in individual]
    '''for i in range(len(individual)):
        individual[i] = individual[i] / total'''
    return individual,


# 设置遗传算法
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(monthly_returns))
toolbox.register("population", tools.initRepeat, list, creator.Individual)

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)
toolbox.register("normalize", normalize_weights)


# 遗传算法主循环
def run_genetic_algorithm():
    population = toolbox.population(n=100)
    NGEN = 50
    CXPB = 0.7
    MUTPB = 0.2

    for gen in range(NGEN):
        offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
        offspring = list(map(lambda ind: toolbox.normalize(ind)[0], offspring))  # 应用归一化
        fits = list(map(toolbox.evaluate, offspring))
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population[:] = toolbox.select(offspring, k=len(population))
        population = list(map(lambda ind: toolbox.normalize(ind)[0], population))  # 应用归一化


    # 获取最佳个体
    best_ind = tools.selBest(population, 1)[0]
    return best_ind

#produce optimum result
best_ind = run_genetic_algorithm()
print(f"最佳权重分配: {best_ind}")
print(f"最佳适应度: {best_ind.fitness.values[0]}")