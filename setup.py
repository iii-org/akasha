from setuptools import setup

setup(name='akasha',
      version = '0.1',
      description = "document QA package using langchain and chromadb",
      author = 'chih chuan chang',
      author_email = 'ccchang@iii.org.tw',
      install_requires = ['pypdf==3.12.2','langchain==0.0.234','chromadb==0.3.29', 'openai==0.27.8',\
                   'tiktoken==0.4.0','lark==1.1.7','scikit-learn<1.3.0','jieba==0.42.1',\
                   'sentence-transformers==2.2.2', 'llama-cpp-python==0.1.77'],
      packages=['akasha']
      )