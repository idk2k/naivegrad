import setuptools

setuptools.setup(
    name="naivegrad",
    version="0.1.1",
    author="Dmitriy Srkv.",
    author_email="north.projectamericas@gmail.com",
    description="Naive autograd core with nn features",
    long_description="description",
    url="https://github.com/idk2/naivegrad",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)