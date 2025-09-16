import pytest
import akasha
from typing import Tuple


@pytest.fixture
def base_line():
    eva = akasha.eval(
        embeddings="openai:text-embedding-3-small",
        model="openai:gpt-3.5-turbo",
        verbose=True,
        search_type="bm25",
        chunk_size=500,
        max_input_tokens=2468,
        temperature=0.15,
        keep_logs=True,
    )
    doc_path = "./docs/mic/"
    return eva, doc_path


@pytest.mark.eval
def test_Model_Eval(base_line: Tuple[akasha.eval, str]):
    eva, doc_path = base_line

    assert eva.verbose is True
    assert eva.search_type == "bm25"
    assert eva.chunk_size == 500
    assert eva.max_input_tokens == 2468
    assert eva.temperature == 0.15

    ql, al = eva.create_questionset(
        doc_path, question_type="compared", question_num=2, question_style="essay"
    )
    # com_name = eva.logs[eva.timestamp_list[-1]]["questionset_path"]
    assert len(ql) == len(al)

    ql, al = eva.create_questionset(
        doc_path, question_type="summary", question_num=2, question_style="essay"
    )
    # sum_name = eva.logs[eva.timestamp_list[-1]]["questionset_path"]
    assert len(ql) == len(al)

    ql, al = eva.create_questionset(
        doc_path, question_num=2, question_type="fact", question_style="essay"
    )
    f1_name = eva.logs[eva.timestamp_list[-1]]["questionset_path"]
    assert len(ql) == len(al)

    ql, al = eva.create_questionset(
        doc_path, question_num=2, question_style="single_choice"
    )
    f2_name = eva.logs[eva.timestamp_list[-1]]["questionset_path"]
    assert len(ql) == len(al)

    avg_rouge, avg_llm_score, tokens = eva.evaluation(
        f1_name, doc_path, question_style="essay"
    )
    assert isinstance(avg_rouge, float)
    assert isinstance(avg_llm_score, float)
    assert isinstance(tokens, list)

    assert 0 <= avg_rouge <= 1
    assert 0 <= avg_llm_score <= 1

    cor_rate, tokens = eva.evaluation(
        f2_name, doc_path, question_style="single_choice", prompt_format_type="chat_gpt"
    )
    assert isinstance(cor_rate, float)
    assert isinstance(tokens, list)

    assert 0 <= cor_rate <= 1

    return
