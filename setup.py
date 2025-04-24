from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()
    # Remove comments and empty lines
    requirements = [line for line in requirements if line and not line.startswith('#')]

setup(
    name="multilingual_emotion_detection",
    version="0.1.0",
    description="A system for detecting emotions in multilingual text (English and Hindi)",
    author="Abhisek",
    author_email="your.email@example.com",
    url="https://github.com/username/multilingual_emotion_detection",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    # Additional data files
    package_data={
        "multilingual_emotion_detection": ["data/*.csv", "models/*.bin"],
    },
)

