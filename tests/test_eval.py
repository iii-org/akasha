import pytest
import akasha.eval as eval



@pytest.fixture
def base_line():
    eva = eval.Model_Eval(verbose=False, search_type="tfidf",chunk_size=500, max_token=3010, temperature=0.15)
    doc_path = "./docs/mic/"
    return eva, doc_path



@pytest.mark.eval
def test_Model_Eval(base_line):
    
    eva, doc_path = base_line

    assert eva.verbose == False
    assert eva.search_type == "tfidf"
    assert eva.chunk_size == 500
    assert eva.max_token == 3010
    assert eva.temperature == 0.15
   
    
    ql, al = eva.auto_create_questionset( doc_path, question_num = 2, question_type="essay")
    f1_name = eva.logs[eva.timestamp_list[-1]]['questionset_path'] 
    assert len(ql) == len(al)
    
    
    ql, al = eva.auto_create_questionset( doc_path, question_num = 2, question_type="single_choice") 
    f2_name = eva.logs[eva.timestamp_list[-1]]['questionset_path']
    assert len(ql) == len(al)
    
    
    avg_bert, avg_rouge, avg_llm_score, tokens = eva.auto_evaluation(f1_name, doc_path, question_type="essay")
    assert isinstance(avg_bert, float)
    assert isinstance(avg_rouge, float)
    assert isinstance(avg_llm_score, float)
    assert isinstance(tokens, int)
    
    assert 0 <= avg_bert <= 1
    assert 0 <= avg_rouge <= 1
    assert 0 <= avg_llm_score <= 1
    
    cor_rate, tokens = eva.auto_evaluation(f2_name, doc_path, question_type="single_choice")
    assert isinstance(cor_rate, float)
    assert isinstance(tokens, int)
    
    assert 0 <= cor_rate <= 1
    
    return