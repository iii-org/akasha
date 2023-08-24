# coding:utf-8
from rouge import Rouge
import rouge_chinese
from bert_score import score
import jieba
import warnings
warnings.filterwarnings("ignore")
jieba.setLogLevel(jieba.logging.INFO)  ## ignore logging jieba model information

def get_bert_score(candidate_str:str, reference_str:str,langugage:str="ch", round_digit=3):


    if langugage == "zh" or "ch":
        P, R, F1 = score([candidate_str], [reference_str], lang="zh", verbose=False)
    else :
        P, R, F1 = score([candidate_str], [reference_str], lang="en", verbose=False)
    # round float into 3 digits behind 0
    F1 = round(float(F1),round_digit)
    
    return F1


def get_rouge_score(candidate_str:str, reference_str:str, language:str="ch", round_digit=3):
    
    
    if language == "zh" or "ch":
        rouge = rouge_chinese.Rouge(metrics=[ 'rouge-l'])
        cand = ' '.join(jieba.cut(candidate_str))
        ref = ' '.join(jieba.cut(reference_str))        
    else :    
        rouge = Rouge(metrics=[ 'rouge-l'])
        cand = ' '.join(list(candidate_str))
        ref = ' '.join(list(reference_str))
        
    F1 = rouge.get_scores(cand, ref)[0]['rouge-l']['f']
    
    F1 = round(F1, round_digit)
    
    return F1



