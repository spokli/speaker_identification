import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="spid",
    version="0.1",
    description="My Speaker Identification Module",
    long_description=long_description,
    author="Marco Kinkel",
    author_email="hi@marco-kinkel.de",
    url="test@test.de",
    packages=find_packages(exclude=["test"]),
    python_requires=">=3.8, <4",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "pydub",
        "torch",
        "noisereduce",
        "python_speech_features",
        "webrtcvad",  # for voice activity detection
    ],
    # List additional groups of dependencies here (e.g. development
    # dependencies). Users will be able to install these using the "extras"
    # syntax, for example:
    #
    #   $ pip install sampleproject[dev]
    #
    # Similar to `install_requires` above, these must be valid existing
    # projects.
    extras_require={  # Optional
        "dev": [
            "pre-commit",
            "black",
            "pycln",
            "isort",
            "jupyter",
            "jupyterlab",
        ],
        "test": ["coverage"],
    },
)
