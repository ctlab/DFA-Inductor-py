import os

from setuptools import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))

requires = [
    'python-sat',
    'click'
]

extras = {
    'dev': ['pytest']
}

about = {}
with open(os.path.join(here, 'dfainductor', '__about__.py'), 'r', encoding='utf-8') as f:
    exec(f.read(), about)

with open('README.md', 'r', encoding='utf-8') as f:
    readme = f.read()


setup(
    name=about['__title__'],
    version=about['__version__'],
    author=about['__author__'],
    author_email=about['__author_email__'],
    description=about['__description__'],
    long_description=readme,
    long_description_content_type='text/markdown',
    url="https://github.com/ctlab/DFA-Inductor-py",
    license=about['__license__'],
    packages=find_packages(),
    install_requires=requires,
    extras_require=extras,
    python_requires=">=3.7",
    entry_points={
        'console_scripts': [
            'dfainductor = dfainductor:cli',
        ]
    },
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Natural Language :: Russian',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics'
    ]
)
