import akasha

DEFAULT_MODEL = "openai:gpt-3.5-turbo"
DEFAULT_EMBED = "openai:text-embedding-ada-002"
DEFAULT_MAX_INPUT_TOKENS = 3000
DEFAULT_MAX_OUTPUT_TOKENS = 1024
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_SEARCH_TYPE = "auto"


def create_questionset():

    ev = akasha.eval(question_type="fact",
                     question_style="single_choice",
                     keep_logs=True,
                     verbose=True)
    questions, answers = ev.create_questionset(data_source=["docs/mic"],
                                               question_num=3,
                                               choice_num=4,
                                               output_file_path="123.txt")

    print(questions, answers)


create_questionset()
