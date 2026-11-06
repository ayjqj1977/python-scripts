import sys
import pkgutil

print("这个APP内置的可用库：")
print("-" * 30)
for _, name, _ in pkgutil.iter_modules(sys.path):
    print(name)
