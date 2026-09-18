"""Compute ROUGE locally; opt into LLM or BERT scoring."""
from pathlib import Path
import sys

# Support both python path/to/example.py and python -m examples.<module>.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples._common import DATA, configure, parser, print_response, workspace

def main(argv=None):
    cli = parser(__doc__)
    cli.add_argument("--llm", action="store_true", help="Also score with the configured model")
    cli.add_argument("--bert", action="store_true", help="Requires full dependencies and may download model weights")
    args = configure(cli, argv)
    import akasha.helper as ah
    candidate = "Industry 4.0 connects sensors and production systems."
    reference = "Industry 4.0 connects industrial sensors with manufacturing information systems."
    print("ROUGE:", ah.get_rouge_score(candidate, reference, language="en"))
    if args.llm:
        model = ah.handle_model(args.model, env_file=args.env_file)
        print("LLM score:", ah.get_llm_score(candidate, reference, model))
    if args.bert:
        print("BERT score:", ah.get_bert_score(candidate, reference, language="en"))


if __name__ == "__main__":
    main()
