import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="RegComp", # Replace with your own username
    version="0.0.1",
    author="Owais Sarwar",
    author_email="osarwar@andrew.cmu.edu",
    description="A tool to compare linear regression methods",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/osarwar/RegComp",
    packages=setuptools.find_packages(),
    install_requires=[
    'pandas',
    'numpy>=1.16.1',
    'scikit-learn>=0.23.2'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # python_requires='>=3.6',
)