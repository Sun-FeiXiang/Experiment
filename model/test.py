

import numpy as np
b=np.array([[0, 4, 4],[2, 0, 3],[1, 3, 4]])
print(b)
print('count(1)=',np.sum(b==2))
print('count(3)=',np.sum(b==3))
print('count(4)=',np.sum(b==4))

print('count(4)=',np.sum(b[0]==4))

