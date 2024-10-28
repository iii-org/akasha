import pytest
import akasha.summary as summary


@pytest.fixture
def base_line():
    sum = summary.Summary(
        verbose=False,
        chunk_overlap=41,
        chunk_size=501,
        max_input_tokens=3020,
        temperature=0.15,
    )
    file_path = "./docs/mic/20230531_智慧製造需求下之邊緣運算與新興通訊發展分析.pdf"
    return sum, file_path


@pytest.mark.summary
def test_Summary(base_line):
    sum, file_path = base_line

    assert sum.verbose == False
    assert sum.chunk_size == 501
    assert sum.chunk_overlap == 41
    assert sum.max_input_tokens == 3020
    assert sum.temperature == 0.15

    text = sum.summarize_file(file_path=file_path,
                              summary_type="map_reduce",
                              summary_len=100)

    assert type(text) == str

    text = sum.summarize_file(
        file_path=file_path,
        summary_type="refine",
        summary_len=100,
        output_file_path="./summarization/summary.txt",
    )
    assert type(text) == str

    return
