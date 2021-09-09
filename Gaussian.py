import numpy as np
import math as m
import plotly.express as plt

with open('data_1.npy', 'rb') as f:
    X = np.load(f)

fig = plt.scatter(x=X[0], y=X[1])
fig.show()

def gaussian_dist (val, mean, std):

    prob =((1/(m.sqrt(2 * m.pi)* std) * m.exp(-((val-mean)**2/(2*std)**2))))

    return prob

x_mean = np.mean(X[0])
y_mean = np.mean(X[1])

mean_vect = np.array([[x_mean], [y_mean]])

x_std = np.std(X[0])
y_std = np.std(X[1])

cov = np.cov(X[0], X[1])

px_list = []
py_list = []

for one in range(0, len(X[0])):
    
    px = gaussian_dist(X[0][one], x_mean, x_std)
    py = gaussian_dist(X[1][one], y_mean, y_std)

    if px*py < .02**2:
        px_list.append(X[0][one])
        py_list.append(X[1][one])


print('')
print('Top 3 Outliers')
for one in range(0, len(px_list)):
    print(one+1, ') [', px_list[one], ', ', py_list[one], ']')

print('')
print('Mean Vector')
print(mean_vect)

print('')
print('Covariance Matrix')
print(cov)
print('')





