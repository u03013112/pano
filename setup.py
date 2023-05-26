from setuptools import setup, find_packages

setup(
    name="u0_stitcher",
    version="0.1.4",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
    ],
    author="u03013112",
    author_email="u03013112@hotmail.com",
    description="stitcher for two fisheye camera",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/u03013112/pano",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
