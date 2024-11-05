
from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'

'''
you would get the same result in terms of the function’s behavior if you wrote it as 
def get_requirements(file_path: str): without the -> List[str] return type hint. 
The type hint does not affect the execution of the function; it is purely for documentation and 
type checking purposes.
'''
def get_requirements(file_path:str)->List[str]:
    '''
    This function returns the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[line.replace("\n", "") for line in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements

setup(
name='mlproject',
version='0.0.1',
author='Vijay Pai',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')
)