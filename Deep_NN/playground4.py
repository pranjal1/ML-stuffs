#how to do random searches for hyper parameter

import numpy as np

#say I need to implement random search for learning rate (alpha).
#I need to search from 0.0001 to 1

#implementing uniform random search beteen 0.0001 and 1

avg = [0.0,0.0,0.0,0.0]

for x in range(0,100):
	a = []
	for i in range(0,100):
		a.append(np.random.uniform(0.0001,1))

	cnt = [0.0,0.0,0.0,0.0]

	for i in a:
		if i>=0.0001 and i<0.001:
			cnt[0]+=1
		elif i>=0.001 and i<0.01:
			cnt[1]+=1
		elif i>=0.01 and i<0.1:
			cnt[2]+=1
		elif i>=0.1 and i<1:
			cnt[3]+=1
	for y in range(len(cnt)):
		cnt[y] /= 100
		avg[y] += cnt[y]

print avg
#[0.07%, 0.7800000000000005%, 8.769999999999996%, 90.38000000000001%]
#the number produced are mostly in the last range, this is not good search strategy

#we want the search to take samples from each of the above range

#solution: use logarithmic scale:
avg = [0.0,0.0,0.0,0.0]

for x in range(0,100):
	a = []
	for i in range(0,100):
		r = -4 * np.random.rand() #a float between 0 and -4, 10*log(0.0001) = -4
		a.append(10**r)#val betn 10^0 to 10^-4
	cnt = [0.0,0.0,0.0,0.0]


	for i in a:
		if i>=0.0001 and i<0.001:
			cnt[0]+=1
		elif i>=0.001 and i<0.01:
			cnt[1]+=1
		elif i>=0.01 and i<0.1:
			cnt[2]+=1
		elif i>=0.1 and i<1:
			cnt[3]+=1
	for y in range(len(cnt)):
		cnt[y] /= 100
		avg[y] += cnt[y]
print avg
#[24.819999999999997, 24.75, 25.44000000000001, 24.99]
#we get samples from all the regions








#similarly if we need to sample the value of beta for momentum and we want to take values between 0.9 and 0.999, we use this approach
#(1-beta) => 0.1 to 0.001, log -1 to -3
beta =[]
for x in range(100):
	r = -2 * np.random.rand() - 1 #a float between -1 and -3, 10*log(0.001) = -3 and 10*log(0.1) = -1
	beta.append(1 - 10**r)#val betn 10^-3 and 10^-1
print max(beta),min(beta)


