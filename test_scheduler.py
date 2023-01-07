import torch
from torch import nn
import matplotlib.pyplot as plt

model = nn.Sequential(nn.Linear(2, 3))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

gradual_learning_rates = (
    [0, 1.],
    [2e4, 5e-1],
    [4e4, 3e-1],
    [6e4, 1e-1],
    [8e4, 5e-2],
)
def stepwise_lr_change(step):
    last_lr = gradual_learning_rates[0][-1]
    for step_lr_set in gradual_learning_rates[1:]:
        step_thres, tar_lr = step_lr_set
        if step > step_thres:
            last_lr = tar_lr
    return last_lr

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, stepwise_lr_change)
lrs = []
for epoch in range(100_000):
    scheduler.step()
    lrs.append(scheduler.get_last_lr())

print(lrs)
plt.plot(lrs)
plt.savefig('tmp.png')
