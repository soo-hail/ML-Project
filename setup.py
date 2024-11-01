# Setyp.py Instruction-Manual for our Python-Project, that helps other to install required dependencies automatically, used for creating Python-Packages. Unlike "requirements.txt" it allows to publish project(as package) on "PyPi".
# -e . : Installs our own package in editable mode. Changes in your source code are immediately reflected
# '-e .': super useful when developing a package, creates a 'live-link' to the code, used to test our package 'without reinstalling' package when we make changes in code(package's code). 

from setuptools import find_packages, setup
from typing import List


def get_requirements()->List[str]:
    # Returns List of requirements.
    HYPHON_E_DOT = '-e .'
    
    requirements = []
    with open('requirements.txt') as f:
        requirements = f.readlines()
        
        # Remove whitespace and newline characters
        requirements = [row.replace('\n', '').strip() for row in requirements]
        
        # Remove '-e .' and any empty strings
        requirements = [req for req in requirements if req and req != HYPHON_E_DOT]
            
    return requirements

# Define our-project.
setup(
    # Project(as package) Meta-Data.
    name="mlproject",
    version="0.0.1",
    author="Mohammed Sohial",
    author_email="mohammed01705@gmail.com",
    description="End-to-End ML Project",
    packages=find_packages(), # Gets all the installed packages, so they are included when someone install this package.
    
    install_requires = get_requirements() # All the required packages, they are automatically installed when someone install this package(project).
)

# In short, this setup.py makes it easy for others to install and use your ML project as a package with dependencies automatically managed.