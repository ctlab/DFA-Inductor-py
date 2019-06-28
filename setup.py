from setuptools import find_packages, setup

from dfainductor import __version__


def main():
    setup(
        name='dfainductor',
        description='A python tool for solving minDFA problem',
        license='MIT',
        version=__version__,
        author='Ilya Zakirzyanov',
        author_email='ilya.zakirzyanov@gmail.com',
        packages=find_packages(),
        entry_points={
            'console_scripts': [
                'dfainductor = dfainductor:cli',
            ]
        }
    )


if __name__ == '__main__':
    main()
