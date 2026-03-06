from setuptools import find_packages, setup
from typing import List

hypn_e_dot = '-e .'

def get_requirements(file_path:str) -> List[str] :

    requirements = []

    with open(file_path) as file_obj :
        requirements = file_obj.readlines()
        requirements = [req.replace('\n', '') for req in requirements]

        if hypn_e_dot in requirements :
            requirements.remove(hypn_e_dot)

        return requirements

setup(
    name = 'src',
    version = '0.0.1',
    author = 'Mihir Parmar',
    author_email= 'connectwithmihir1@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)