import os
for c in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    os.makedirs(f"train/{c}", exist_ok=True)