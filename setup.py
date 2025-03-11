from setuptools import setup, find_packages

setup(
    name="CAPybara",
    version="0.1",
    packages=find_packages(),
    install_requires=["pyscf", "numpy", "pandas"],  # List dependencies here
    entry_points={
        "console_scripts": [
            "CAPybara=CAPybara.main:main",
        ]
    },
)
