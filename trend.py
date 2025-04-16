import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rc("figure", autolayout=True, figsize=(11, 5))
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)

df = pd.read_csv('data/Location1.csv')
feature = 'Power'
df = df[['Time', feature]]
df['Time'] = pd.to_datetime(df['Time'])
df.set_index('Time', inplace=True)
df = df.resample('W').mean()
print(df)

moving_average = df.rolling(
    window=14,       
    center=True,      # puts the average at the center of the window
    min_periods=7,  # choose about half the window size
).mean()              # compute the mean (could also do median, std, min, max, ...)
print(moving_average)
ax = df.plot(style=".", color="0.5")
moving_average.plot(
    ax=ax, linewidth=3, title="Power - Weekly", legend=False,
);
plt.show()


'''
df = pd.read_csv('data/Location1.csv')
feature = 'Power'
df = df[['Time', feature]]
df['Time'] = pd.to_datetime(df['Time'])
df.set_index('Time', inplace=True)
df.index = pd.to_datetime(df.index)
df = df.to_period("1h")
df = df.resample('D').mean()
print(df)
def plot_periodogram(ts, detrend='linear', ax=None):
    from scipy.signal import periodogram
    fs = pd.Timedelta("365D") / pd.Timedelta("1D")
    freqencies, spectrum = periodogram(
        ts,
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum',
    )
    if ax is None:
        _, ax = plt.subplots()
    ax.step(freqencies, spectrum, color="purple")
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels(
        [
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
        ],
        rotation=30,
    )
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel("Variance")
    ax.set_title("Periodogram")
    return ax
ax = plot_periodogram(df.Power)
plt.show()
'''