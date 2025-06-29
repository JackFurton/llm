from setuptools import setup, Extension
from Cython.Build import cythonize

# Define extensions
extensions = [
    Extension(
        "src.cython_ext.fast_tokenizer",
        ["src/cython_ext/fast_tokenizer.pyx"],
    ),
]

setup(
    name="custom_llm",
    version="0.1.0",
    ext_modules=cythonize(extensions),
)
