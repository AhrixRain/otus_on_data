import importlib, sys
for name in ("numpy", "torch", "yaml", "h5py", "scipy"):
    try:
        mod = importlib.import_module(name)
        print(f"{name:8s} OK   {getattr(mod, '__version__', '?')}")
    except Exception as error:
        print(f"{name:8s} MISSING ({type(error).__name__})")
print("python", sys.version.split()[0])
