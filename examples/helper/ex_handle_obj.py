"""Use model/embedding helpers, batching, streaming and structured output."""
from pathlib import Path
import sys

# Support both python path/to/example.py and python -m examples.<module>.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples._common import DATA, configure, parser, print_response, workspace

def main(argv=None):
    args = configure(parser(__doc__), argv)
    import akasha.helper as ah
    from langchain_core.messages import HumanMessage, SystemMessage
    from pydantic import BaseModel

    model = ah.handle_model(args.model, max_output_tokens=512, env_file=args.env_file)
    prompt = [SystemMessage(content="Answer concisely."),
              HumanMessage(content="What is Industry 4.0?")]
    print(ah.call_model(model, prompt))
    print(ah.call_batch_model(model, ["What is a sensor?", "What is predictive maintenance?"]))
    print_response(ah.call_stream_model(model, "Explain industrial sensors in one sentence."))

    class ModelInfo(BaseModel):
        provider: str
        model_name: str

    print(ah.call_JSON_formatter(model, 'Describe the alias "openai:gpt-4o-mini".', keys=ModelInfo))
    embeddings = ah.handle_embeddings(args.embeddings, env_file=args.env_file)
    vector = embeddings.embed_query("Industry 4.0")
    print("Embedding dimensions:", len(vector))
    print("Embedding name:", ah.handle_model_type(embeddings))


if __name__ == "__main__":
    main()
