from setuptools import setup, find_packages

setup(
    name='final_project',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    description='Your project description',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://your.project.url',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)