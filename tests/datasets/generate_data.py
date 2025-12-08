import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import skewnorm

## Changepoints
# np.random.seed(42)
# n = 2000  # Total number of data points
# beta_0 = np.array([2, 1, 0, 0])
# beta_1 = np.array([0, -2, -1, 0])
# beta_2 = np.array([0, 0, 2, 1])

# # Initialize arrays for X_t and Y_t
# X_t = np.random.multivariate_normal(np.zeros(4), np.eye(4), n)
# epsilon_t = np.random.normal(0, 1, n)
# T = np.arange(n)
# T = T + 1

# # Generate Y_t based on the beta values and changepoints
# Y_t = np.zeros(n)

# # Define changepoints: t=500 and t=1500
# for t in range(n):
#     if t < 500:
#         Y_t[t] = np.dot(X_t[t], beta_0) + epsilon_t[t]
#     elif t < 1500:
#         Y_t[t] = np.dot(X_t[t], beta_1) + epsilon_t[t]
#     else:
#         Y_t[t] = np.dot(X_t[t], beta_2) + epsilon_t[t]

# # Create a dataframe for the data
# data = pd.DataFrame(np.hstack((T.reshape(-1, 1), X_t, Y_t.reshape(-1, 1))), columns=['timestamp', 'X1', 'X2', 'X3', 'X4', 'y'])

# # Save to a CSV file
# csv_file_path = 'changepoint.csv'
# data.to_csv(csv_file_path, index=False)

# print(csv_file_path)




## Distribution drift
np.random.seed(42)
N = 2000
beta_start = np.array([2, 1, 0, 0])
beta_end = np.array([0, 0, 2, 1])

X_t = np.random.multivariate_normal(mean=np.zeros(4), cov=np.eye(4), size=N)
T = np.arange(N)
T = T + 1

beta = np.zeros((N, 4))
for t in range(N):
    a = t / (N - 1)
    beta[t] = beta_start + a * (beta_end - beta_start)

epsilon_t = np.random.normal(0, 1, N)
Y_t = np.array([X_t[i] @ beta[i] + epsilon_t[i] for i in range(N)])

# Create a dataframe for the data
data = pd.DataFrame(np.hstack((T.reshape(-1, 1), X_t, Y_t.reshape(-1, 1))), columns=['timestamp', 'X1', 'X2', 'X3', 'X4', 'y'])

# Save to a CSV file
csv_file_path = 'distribution_drift.csv'
data.to_csv(csv_file_path, index=False)

print(csv_file_path)



## heteroskedastic, heavy-tailed
np.random.seed(42)
n = 2000
beta = np.array([2, 1, 0.5, -0.5])
d = 4

X_t = np.random.multivariate_normal(np.zeros(4), np.eye(4), n)
mu_X = X_t @ beta

E_mu_power3 = np.mean(np.abs(mu_X)**3)

def generate_heteroskedastic_heavy_tailed_errors(mu_X, E_mu_power3):
    n = len(mu_X)
    epsilon = np.zeros(n)
    
    for i in range(n):
        sigma_i = 1 + 2 * (np.abs(mu_X[i])**3) / E_mu_power3
        epsilon[i] = stats.t.rvs(df=2)
    
    return epsilon

epsilon_t = generate_heteroskedastic_heavy_tailed_errors(mu_X, E_mu_power3)

Y_t = mu_X + epsilon_t

T = np.arange(1, n + 1)
column_names = ['timestamp'] + [f'X{i+1}' for i in range(d)] + ['y']
data = pd.DataFrame(
    np.hstack((T.reshape(-1, 1), X_t, Y_t.reshape(-1, 1))),
    columns=column_names
)

csv_file_path = 'heavy-tailed.csv'
data.to_csv(csv_file_path, index=False)

print(csv_file_path)



## variance changepoints
np.random.seed(42)
n = 2000
beta = np.array([2, 1, 0.5, -0.5])

sigma_0 = 1.0   # t < 500: medium variance
sigma_1 = 3.0   # 500 ≤ t < 1000: high variance
sigma_2 = 0.5   # 1000 ≤ t < 1500: low variance

X_t = np.random.multivariate_normal(np.zeros(4), np.eye(4), n)
T = np.arange(n) + 1

Y_t = np.zeros(n)
epsilon_t = np.zeros(n)

for t in range(n):
    if t < 500:
        epsilon_t[t] = np.random.normal(0, sigma_0)
    elif t < 1500:
        epsilon_t[t] = np.random.normal(0, sigma_1)
    else:
        epsilon_t[t] = np.random.normal(0, sigma_2)
    
    Y_t[t] = np.dot(X_t[t], beta) + epsilon_t[t]

data = pd.DataFrame(
    np.hstack((T.reshape(-1, 1), X_t, Y_t.reshape(-1, 1))),
    columns=['timestamp', 'X1', 'X2', 'X3', 'X4', 'y']
)

csv_file_path = 'variance_changepoint.csv'
data.to_csv(csv_file_path, index=False)

print(csv_file_path)



##  Extreme Distribution drift
np.random.seed(42)
N = 2000
beta_start = np.array([20, 10, 1, 1])
beta_end = np.array([1, 1, 20, 10])

X_t = np.random.multivariate_normal(mean=np.zeros(4), cov=np.eye(4), size=N)
T = np.arange(N)
T = T + 1

beta = np.zeros((N, 4))
for t in range(N):
    a = t / (N - 1)
    beta[t] = beta_start + a * (beta_end - beta_start)

epsilon_t = np.random.normal(0, 1, N)
Y_t = np.array([X_t[i] @ beta[i] + epsilon_t[i] for i in range(N)])

# Create a dataframe for the data
data = pd.DataFrame(np.hstack((T.reshape(-1, 1), X_t, Y_t.reshape(-1, 1))), columns=['timestamp', 'X1', 'X2', 'X3', 'X4', 'y'])

# Save to a CSV file
csv_file_path = 'extreme_drift.csv'
data.to_csv(csv_file_path, index=False)

print(csv_file_path)
