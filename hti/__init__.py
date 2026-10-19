import os

# numpy, torch and rdkit each ship their own OpenMP runtime on Windows; without this the kernel dies on import
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
