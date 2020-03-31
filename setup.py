import os
import re
import sys
from setuptools import setup, find_packages


install_requires = ['torch>=1.1.0', 'pytorch_ranger>=0.1.1']

PY35 = (3, 5, 0)


if sys.version_info < PY35:
    raise RuntimeError('torch-optimizer requires Python 3.5.0+')


def _read(f):
    with open(os.path.join(os.path.dirname(__file__), f)) as f_:
        return f_.read().strip()


def _read_version():
    regexp = re.compile(r"^__version__\W*=\W*'([\d.abrc]+)'")
    init_py = os.path.join(
        os.path.dirname(__file__), 'torch_optimizer', '__init__.py'
    )
    with open(init_py) as f:
        for line in f:
            match = regexp.match(line)
            if match is not None:
                return match.group(1)
        raise RuntimeError(
            'Cannot find version in torch_optimizer/__init__.py'
        )


classifiers = [
    'License :: OSI Approved :: Apache Software License',
    'Intended Audience :: Developers',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Operating System :: OS Independent',
    'Development Status :: 3 - Alpha',
]

keywords = [
    'torch-optimizer',
    'pytorch',
    # optimizers
    'accsgd',
    'adabound',
    'adamod',
    'diffgrad',
    'lamb',
    'lookahead',
    'novograd',
    'pid',
    'qhadam',
    'qhm',
    'radam',
    'sgdw',
    'yogi',
    'ranger',
]


setup(
    name='torch-optimizer',
    version=_read_version(),
    description=('pytorch-optimizer'),
    long_description='\n\n'.join((_read('README.rst'), _read('CHANGES.rst'))),
    long_description_content_type='text/x-rst',
    classifiers=classifiers,
    platforms=['POSIX'],
    author='Nikolay Novik',
    author_email='nickolainovik@gmail.com',
    url='https://github.com/jettify/pytorch-optimizer',
    download_url='https://pypi.org/project/torch-optimizer/',
    license='Apache 2',
    packages=find_packages(),
    install_requires=install_requires,
    keywords=keywords,
    zip_safe=True,
    include_package_data=True,
)
