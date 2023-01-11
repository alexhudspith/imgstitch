from setuptools import setup, find_packages
import os.path
import re

here = os.path.join(os.path.abspath(os.path.dirname(__file__)))


def get_descriptions():
    path = os.path.join(here, 'README.md')
    with open(path) as f:
        lines = f.read()
    # Long description is the body text after the Description heading up to the next heading
    # Short description is the first sentence of that, up to the first period (.)
    pattern = re.compile(r'^\s*#+ Description\s*$\s*([^\.]*.?)(.*?)^#',  flags=re.DOTALL | re.MULTILINE)
    m = pattern.search(lines)
    short_desc = m.group(1)
    long_desc = short_desc + m.group(2)
    return short_desc, long_desc


def get_version():
    path = os.path.join(here, 'src/imgstitch', '__init__.py')
    # __version__ = "x.y.z"
    pattern = re.compile(r'^\s*__version__\s*=\s*[\'"](.+?)[\'"]')
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                return m.group(1)

    raise RuntimeError("Unable to find __version__")


VERSION = get_version()
DESCRIPTION, LONG_DESCRIPTION = get_descriptions()


setup(
    name='imgstitch',
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='Alex Hudspith',
    # author_email='',
    # url='',
    python_requires='>=3.10',
    tests_require=['pytest'],
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    include_package_data=True,
    entry_points={
        "console_scripts": {
            "imgstitch = imgstitch.main:main"
        }
    },
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: End Users/Desktop',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Multimedia :: Graphics',
        'License :: OSI Approved :: MIT License'
    ]
)
