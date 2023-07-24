from setuptools import setup

setup(name='QAIII',
      version = '0.1',
      description = "document QA package using langchain and chromadb",
      author = 'chih chuan chang',
      author_email = 'ccchang@iii.org.tw',
      install_requires = ['pypdf==3.12.2','langchain==0.0.234','chromadb==0.3.29', 'openai==0.27.8',\
                   'tiktoken==0.4.0'],
      py_modules = ['QAIII','helper','__init__']
      )