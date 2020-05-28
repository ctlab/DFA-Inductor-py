import os

from setuptools import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))

requires = [
    'python-sat',
    'click'
]

about = {}
with open(os.path.join(here, 'dfainductor', '__about__.py'), 'r', encoding='utf-8') as f:
    exec(f.read(), about)

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()

setup(
    name=about['__title__'],
    version=about['__version__'],
    description=about['__description__'],
    author=about['__author__'],
    author_email=about['__author_email__'],
    license=about['__license__'],
    packages=find_packages(),
    install_requires=requires,
    python_requires=">=3.7",
    entry_points={
        'console_scripts': [
            'dfainductor = dfainductor:cli',
        ]
    }
)