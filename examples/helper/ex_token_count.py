"""Count tokens and words, convert Chinese, and extract JSON without a model API."""
from pathlib import Path
import sys

# Support both python path/to/example.py and python -m examples.<module>.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples._common import DATA, configure, parser, print_response, workspace

def main(argv=None):
    args = configure(parser(__doc__), argv)
    import akasha.helper as ah
    text = "Industry 4.0 connects industrial sensors and information systems."
    print("Tokens:", ah.myTokenizer.compute_tokens(text, args.model))
    print("Words:", ah.get_doc_length("en", text))
    print("Traditional Chinese:", ah.sim_to_trad("工业数据"))
    print("JSON:", ah.extract_json('Example: {"title": "Industry 4.0", "connected": true}'))


if __name__ == "__main__":
    main()
