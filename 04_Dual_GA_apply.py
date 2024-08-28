import random
from deap import base, creator, tools, algorithms
import numpy as np
import pandas as pd

# 读取用户上传的文件
df_returns = pd.read_csv('dual_average_monthly_returns_2014.csv')
df_cov_matrix = pd.read_excel('Dual_GA_covariance_matrix_2014.xlsx')

# 检查数据形状
print("Shape of average_monthly_returns_2013:", df_returns.shape)
print("Shape of covariance_matrix_2013:", df_cov_matrix.shape)

# 确保数据形状匹配
if df_returns.shape[0] != df_cov_matrix.shape[0] or df_cov_matrix.shape[0] != df_cov_matrix.shape[1]:
    raise ValueError("The dimensions of monthly returns and covariance matrix do not match or are not correct.")

# 填充缺失值
df_returns_filled = df_returns.fillna(method='ffill')
df_cov_matrix_filled = df_cov_matrix.fillna(method='ffill')

# 将处理后的数据展平为一维数组和二维数组
monthly_returns = df_returns_filled.values.flatten()
cov_matrix = df_cov_matrix_filled.values

# 确保协方差矩阵是对称的
cov_matrix = (cov_matrix + cov_matrix.T) / 2

# 打印生成的数据
print("Processed Average Monthly Returns:")
print(monthly_returns)
print("\nProcessed Covariance Matrix:")
print(cov_matrix)

# 创建适应度函数
def evaluate(individual):
    weights = np.array(individual)
    total = np.sum(weights)
    if total == 0:
        return (0.0,),  # 避免除零错误，确保返回元组
    weights = weights / total  # 归一化权重
    returns = np.array(monthly_returns)
    port_return = np.sum(returns * weights)
    port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    if port_risk == 0:
        return (0.0,)
    sharpe_ratio = port_return / port_risk
    if np.isnan(sharpe_ratio):
        return (0.0,)  # 处理 nan 值
    return (sharpe_ratio,)  # 确保返回值为元组


def normalize_weights(individual):
    total = sum(individual)
    if total != 0:
        individual[:] = [max(0, x / total) for x in individual]  # 确保非负权重并归一化
    return individual,


# 设置遗传算法
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 1)  # 使用 0 到 1 之间的随机数
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=len(monthly_returns))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0.5, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)
toolbox.register("normalize", normalize_weights)


# 遗传算法主循环
def run_genetic_algorithm():
    population = toolbox.population(n=20)
    NGEN = 50
    CXPB = 0.7
    MUTPB = 0.2

    print("Initial population:")
    for ind in population:
        toolbox.normalize(ind)  # 初始种群的归一化
        ind.fitness.values = toolbox.evaluate(ind)
        print(f"Individual: {ind}, Fitness: {ind.fitness.values}")

    for gen in range(NGEN):
        print(f"Generation {gen}")
        offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
        for ind in offspring:
            toolbox.normalize(ind)  # 应用归一化
            ind.fitness.values = toolbox.evaluate(ind)  # 确保每次评估后的适应度值是有效的

        print("Offspring after varAnd:")
        for ind in offspring:
            print(f"Individual: {ind}, Fitness: {ind.fitness.values}")

        population[:] = toolbox.select(offspring, k=len(population))

        # 归一化选择后的每个个体
        for ind in population:
            toolbox.normalize(ind)

        best_ind = tools.selBest(population, 1)[0]
        print(f"Best individual in generation {gen}: {best_ind}")
        print(f"Best fitness in generation {gen}: {best_ind.fitness.values}")
        print(f"Sum of best individual weights in generation {gen}: {sum(best_ind)}")

    # 获取最佳个体
    best_ind = tools.selBest(population, 1)[0]
    return best_ind


# 生成最优结果
best_ind = run_genetic_algorithm()
print(f"最佳权重分配: {best_ind}")
print(f"最佳适应度: {best_ind.fitness.values}")
print(f"Sum of best individual weights: {sum(best_ind)}")

df = pd.DataFrame(best_ind, columns=["weights"])
df.to_excel('weights_2014.xlsx', index=False)


# 计算最佳适应度，这个其实和前面那个是一样的
def calculate_sharpe_ratio(weights, returns, cov_matrix):
    weights = np.array(weights)
    total = np.sum(weights)
    if total == 0:
        return 0.0  # 避免除零错误
    weights = weights / total  # 归一化权重
    port_return = np.sum(returns * weights)
    port_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    if port_risk == 0:
        return 0.0
    sharpe_ratio = port_return / port_risk
    return sharpe_ratio, port_risk


# 计算并打印最佳适应度
best_sharpe_ratio = calculate_sharpe_ratio(best_ind, monthly_returns, cov_matrix)
print(f"Calculated Best Sharpe Ratio and Portfolio Risk: {best_sharpe_ratio}") #portfolio risk means portfolio deviation




