from setuptools import setup

setup(name='akasha',
      version = '0.5',
      description = "document QA package using langchain and chromadb",
      author = 'chih chuan chang',
      author_email = 'ccchang@iii.org.tw',
      install_requires = ['pypdf==3.12.2','langchain==0.0.234','chromadb==0.3.29', 'openai==0.27.8',\
                   'tiktoken==0.4.0','lark==1.1.7','scikit-learn<1.3.0','jieba==0.42.1',\
                   'sentence-transformers==2.2.2', 'torch>=2.0.1', 'transformers==4.31.0',\
                        'llama-cpp-python>=0.1.77','auto-gptq==0.3.1','tqdm==4.65.0', "docx2txt==0.8", "rouge==1.0.1",\
                              "rouge-chinese==1.0.3","bert-score==0.3.13", "opencc","click","tokenizers==0.13.3"],
      packages=['akasha', 'akasha.models','cli','akasha.eval','akasha.summary'],
      entry_points={
            'console_scripts': [
                  'akasha = cli.glue:akasha'
                  ]
            }
      )
