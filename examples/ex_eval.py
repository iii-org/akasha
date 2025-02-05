import akasha

DEFAULT_MODEL = "openai:gpt-3.5-turbo"
DEFAULT_EMBED = "openai:text-embedding-ada-002"
DEFAULT_MAX_INPUT_TOKENS = 3000
DEFAULT_MAX_OUTPUT_TOKENS = 1024
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_SEARCH_TYPE = "auto"

# create a eval object with the model and question type, question_style
# question type : fact, irrelevant, summary, compared
# question_style : essay, single_choice
ev = akasha.eval(model="openai:gpt-4o",
                 question_type="fact",
                 question_style="essay",
                 keep_logs=True,
                 verbose=True)

# create a question set with the data source, number of questions, number of choices, and output file path
questions, answers = ev.create_questionset(data_source=["docs/mic"],
                                           question_num=3,
                                           choice_num=4,
                                           output_file_path="cq3.txt")
# save the logs or turn verbose on to see the details
ev.save_logs("log_cq.json")

# you can also create a quesion set with a specific topic using create_topic_questionset
questions, answers = ev.create_topic_questionset(
    data_source=["docs/mic"],
    topic="工業 4.0",
    question_num=3,
    choice_num=4,
    output_file_path="4-0topic.txt")

#assign the quesion set file name, and evaluate the model performance of the question set, it will return the evaluation result and the totken usage.
print(ev.evaluation(questionset_file="cq3.txt", data_source=["docs/mic"]))
