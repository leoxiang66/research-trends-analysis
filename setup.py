from setuptools import setup, find_packages


with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = [
'pandas',
'requests-toolkit-stable==0.8.0',
'evaluate==0.2.2',
'kmeans_pytorch==0.3',
'scikit_learn==1.0.2',
'sentence_transformers==2.2.2',
'torch',
'yellowbrick==1.5',
'transformers==4.22.1',
'textdistance==4.5.0',
'datasets==2.5.2',
'ml-leoxiang66',
'KeyBartAdapter==0.1.12'
]

setup(
    name="TrendFlow",
    version=f'0.2.6',
    author="Tao Xiang",
    author_email="tao.xiang@tum.de",
    description="A tool for literature research and analysis",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/leoxiang66/research-trends-analysis",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
  "Programming Language :: Python :: 3.7",
  "License :: OSI Approved :: MIT License",
    ],
)