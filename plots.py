import matplotlib.pyplot as plt
import re

iterVsCost = {}
iterVsAccu = {}
costVsAccu = {}
iterValues = []
costValues = []
accuValues = []
costTicks = []
iterTicks = []
accuTicks = []
costTicksForAccu = []


with open('multicell rnn_result.txt', 'r') as results:
    for i in results:
        if len(i) != 1:
            nums = (re.findall('[0-9]+\.?[0-9]*', i))
            iterVsCost[nums[0]] = nums[1]
            iterVsAccu[nums[0]] = nums[2]
            costVsAccu[nums[1]] = nums[2]

for i in (iterVsCost.items()):
    iterValues.append(int(i[0]))
    costValues.append(float(i[1]))

for i in (iterVsAccu.items()):
    accuValues.append(float(i[1]))

# print(iterValues)

for i in range(10000):
    if i == 0:
        iterTicks.append(i)
    elif i % 500 == 0:
        iterTicks.append(i)
# print(iterTicks)

for i in range(300):
    if i == 0:
        costTicks.append(i)
    elif i % 20 == 0:
        costTicks.append(i)

for i in range(100):
    if i == 0:
        accuTicks.append(i)
    elif i % 1 == 0:
        i = i/100
        accuTicks.append(i)

# print(accuTicks)

for i in range(300):
    if i == 0:
        costTicksForAccu.append(i)
    elif i % 1 == 0:
        costTicksForAccu.append(i)

plt.plot(iterValues, costValues)
plt.xticks(iterTicks)
plt.xlabel('Iterations')
plt.yticks(costTicks)
plt.ylabel('Cost')
fig = plt.gcf()
fig.set_size_inches(10, 5.5)
plt.xlim(0, 10000)
plt.show()

plt.plot(iterValues, accuValues)
plt.xticks(iterTicks)
plt.xlabel('Iterations')
plt.yticks(accuTicks)
plt.ylabel('Accuracy')
fig1 = plt.gcf()
fig1.set_size_inches(10, 5.5)
plt.ylim(0, 0.2)
plt.xlim(0, 10000)
plt.show()

plt.plot(accuValues, costValues, 'go')
plt.xticks(accuTicks)
plt.xlabel('Accuracy')
plt.yticks(costTicks)
plt.ylabel('Cost')
fig2 = plt.gcf()
fig2.set_size_inches(10, 5.5)
plt.xlim(0, .13)
# plt.xlim(0, 0.2)
plt.show()