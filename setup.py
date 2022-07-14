import os
from setuptools import setup

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

setup(name='miniformer',
      version='0.1.0',
      description='Minimal Transformer re-implementation inspired by minGPT',
      author='Artem Yatsenko',
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages = ['miniformer'],
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
      ],
      install_requires=['numpy', 'requests', 'pillow'],
      python_requires='>=3.8',
      extras_require={
        'gpu': ["pyopencl", "six"],
        'testing': [
            "pytest",
            "torch~=1.11.0",
            "tqdm",
        ],
      },
      include_package_data=True)

