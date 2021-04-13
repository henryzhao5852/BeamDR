from setuptools import setup

with open('README.md') as f:
    readme = f.read()

setup(
   name='BeamDR',
   version='0.1.0',
   description='Multi-Step Reasoning Over Unstructured Text with Beam Dense Retrieval',
   url='https://github.com/henryzhao5852/BeamDR',
   license="MIT",
   long_description=readme,
   install_requires=[
        'torch==1.2.0'
        'transformers==2.4.1', 
        'pytrec-eval',
        'faiss-cpu',
        'wget'
    ],
)