"""Generate a small question set; optionally evaluate it (extra dependencies)."""
from pathlib import Path
import sys

# Support both python path/to/example.py and python -m examples.<module>.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples._common import DATA, configure, parser, print_response, workspace, copy_documents

def main(argv=None):
    cli = parser(__doc__)
    cli.add_argument("--topic", help="Generate questions for a specific topic")
    cli.add_argument("--evaluate", action="store_true", help="Run evaluation; requires the full scoring dependencies")
    args = configure(cli, argv)
    import akasha
    with workspace(args, "eval"):
        source = copy_documents()
        evaluator = akasha.eval(model=args.model, embeddings=args.embeddings,
                                env_file=args.env_file, question_type="fact",
                                question_style="essay", search_type="knn",
                                language="en", keep_logs=True)
        if args.topic:
            questions, answers = evaluator.create_topic_questionset(
                data_source=source, topic=args.topic, question_num=2,
                output_file_path="questions.json")
        else:
            questions, answers = evaluator.create_questionset(
                data_source=source, question_num=2, output_file_path="questions.json")
        if not questions or not answers:
            raise RuntimeError("The model did not produce a usable question set.")
        print("Questions:", questions)
        print("Answers:", answers)
        if args.evaluate:
            print(evaluator.evaluation(questionset_file="questions.json", data_source=source))
        evaluator.save_logs("eval.json")


if __name__ == "__main__":
    main()
