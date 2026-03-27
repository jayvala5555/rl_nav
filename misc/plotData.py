import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("./monitor.csv", skiprows=1)

plt.plot(data["r"])
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.xlabel("Episodes", fontsize=13)
plt.ylabel("Reward", fontsize=13)
# plt.title("Training Reward Curve")

plt.show()