import numpy as np
from matplotlib import pyplot as plt

# dp=np.random.randint(-10,10,(50,2))
# dp=np.array([[j,0] for j in range(50)])
dp=np.random.uniform(0,1,(1000,2))
print(dp.shape)
plt.scatter(dp[:,0],dp[:,1],c='b')
plt.show()

rot=60
rad_rot=np.deg2rad(rot)
rot_mat=np.array([[np.cos(rad_rot),np.sin(rad_rot)],[-np.sin(rad_rot),np.cos(rad_rot)]])
print(rot_mat)
df_rot=np.dot(dp,rot_mat)
plt.scatter(dp[:,0],dp[:,1],c='b')
plt.scatter(df_rot[:,0],df_rot[:,1],c='r')
plt.show()

