import pandas as pd

events = pd.read_csv('data/events.csv')
items = pd.read_csv('data/items.csv')
users = pd.read_csv('data/users.csv')

print(events['watch_seconds'].describe())
# genre