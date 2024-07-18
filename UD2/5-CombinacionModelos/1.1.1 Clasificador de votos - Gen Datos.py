from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import mglearn

X,y=make_moons(n_samples=100,noise=0.15)

fig, ax = plt.subplots(figsize=(6,4))

mglearn.discrete_scatter(X[:,0], X[:,1],y,ax=ax)
ax.set_title("Dataset moons")