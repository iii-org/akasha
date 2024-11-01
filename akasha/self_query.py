#from langchain_chroma import Chroma
from langchain_core.documents import Document
#from langchain_openai import OpenAIEmbeddings
from langchain.chains.query_constructor.base import AttributeInfo
from typing import List, Union, Set, Any, Tuple
import json
from pydantic import Field
import operator
from langchain.retrievers.self_query.chroma import ChromaTranslator
#from langchain.chains.query_constructor.ir import StructuredQuery
from langchain.chains.query_constructor.base import StructuredQueryOutputParser
from langchain_core.language_models.base import BaseLanguageModel
import akasha.helper as helper
from akasha.db import dbs, extract_db_by_ids


def query_filter(prompt: str,
                 model_obj: BaseLanguageModel,
                 db: dbs,
                 metadata_field_info: List[AttributeInfo] = [],
                 document_content_description: str = "") -> Tuple[dbs, str]:

    query, fnl_filter, data_source = generate_query_filter(
        model_obj, prompt, metadata_field_info, document_content_description)
    #print(fnl_filter)
    new_docs = [
        DocumentCP(doc, metadata, ids)
        for doc, metadata, ids in zip(db.docs, db.metadatas, db.ids)
    ]
    filtered_docs = filter_docs(new_docs, fnl_filter, data_source)
    filtered_docs_set = set(fd.ids for fd in filtered_docs)

    new_dbs = extract_db_by_ids(db, filtered_docs_set)

    return new_dbs, query


def generate_query_constructor(metadata_info: List[AttributeInfo],
                               document_content_description: str, prompt: str):

    data_source = {
        "content": document_content_description,
        "attributes": {
            attr.name: {
                "description": attr.description,
                "type": attr.type
            }
            for attr in metadata_info
        }
    }
    json_string = json.dumps(data_source, ensure_ascii=False, indent=4)
    st1 = """\
<< Structured Request Schema >>
When responding use a markdown code snippet with a JSON object formatted in the following schema:

```json
{{
    "query": string \\ text string to compare to document contents
    "filter": string \\ logical condition statement for filtering documents
}}
```

The query string should contain only text that is expected to match the contents of documents. Any conditions in the filter should not be mentioned in the query as well.

A logical condition statement is composed of one or more comparison and logical operation statements.

A comparison statement takes the form: `comp(attr, val)`:
- `comp` (eq | ne | gt | gte | lt | lte): comparator
- `attr` (string):  name of attribute to apply the comparison to
- `val` (string): is the comparison value

A logical operation statement takes the form `op(statement1, statement2, ...)`:
- `op` (and | or): logical operator
- `statement1`, `statement2`, ... (comparison statements or logical operation statements): one or more statements to apply the operation to

Make sure that you only use the comparators and logical operators listed above and no others.
Make sure that filters only refer to attributes that exist in the data source.
Make sure that filters only use the attributed names with its function names if there are functions applied on them.
Make sure that filters only use format `YYYY-MM-DD` when handling date data typed values.
Make sure that filters take into account the descriptions of attributes and only make comparisons that are feasible given the type of data being stored.
Make sure that filters are only used as needed. If there are no filters that should be applied return "NO_FILTER" for the filter value.


<< Example 1. >>
Data Source:
```json
{
    "content": "Lyrics of a song",
    "attributes": {
        "artist": {
            "type": "string",
            "description": "Name of the song artist"
            },
        "length": {
            "type": "integer",
            "description": "Length of the song in seconds"
            },
        "genre": {
            "type": "string",
            "description": "The song genre, one of "pop", "rock" or "rap""
            }
        }
}
```

User Query:
What are songs by Taylor Swift or Katy Perry about teenage romance under 3 minutes long in the dance pop genre

Structured Request:
```json
{
    "query": "teenager love",
    "filter": "and(or(eq(\"artist\", \"Taylor Swift\"), eq(\"artist\", \"Katy Perry\")), lt(\"length\", 180))"
    }

```

User Query:
What are songs that were not published on Spotify

Structured Request:
```json
{    
"query": "",
"filter": "NO_FILTER"
}
```


<< Example 3. >>
Data Source:
```json
"""
    st2 = """\
```

User Query:
"""
    st3 = """\
        

Structured Request:

"""

    query_constructor_prompt = st1 + json_string + st2 + prompt + st3
    return query_constructor_prompt, data_source


class DocumentCP(Document):
    ids: Union[int, str] = Field(default="")

    def __init__(self, page_content, metadata, ids):
        super().__init__(page_content)
        self.metadata = metadata
        self.ids = ids

    def __hash__(self):
        return hash(self.page_content)

    def __eq__(self, other):
        return isinstance(
            other, DocumentCP) and self.page_content == other.page_content


def handle_attr(attr: Union[int, float, str], data_source: dict, keyword: str):

    if data_source['attributes'][keyword]['type'] == 'string':
        return str(attr)
    elif data_source['attributes'][keyword]['type'] == 'integer':
        return int(attr)
    elif data_source['attributes'][keyword]['type'] == 'float':
        return float(attr)
    return str(attr)


def find_subset(
    val: Union[int, float, str],
    keyword: str,
    cur_docs: List[DocumentCP],
    data_source: dict,
) -> Set[DocumentCP]:

    # Define a dictionary to map operators to functions
    operator_map = {
        '$eq': operator.eq,
        '$ne': operator.ne,
        '$gt': operator.gt,
        '$gte': operator.ge,
        '$lt': operator.lt,
        '$lte': operator.le
    }

    res = set()
    for op, func in operator_map.items():
        if op in val:
            attr = handle_attr(val[op], data_source, keyword)

            if isinstance(attr, str) and op == '$eq':
                attr = attr.lower()
                for doc in cur_docs:
                    if keyword in doc.metadata and attr in doc.metadata[
                            keyword].lower():
                        res.add(doc)
            else:
                for doc in cur_docs:
                    if keyword in doc.metadata and func(
                            doc.metadata[keyword], attr):
                        res.add(doc)

    return res


def transfer_filter(filters: dict) -> dict:
    """Recursively convert all $and filters to $or filters."""
    if '$and' in filters:
        filters = {'$or': [transfer_filter(f) for f in filters['$and']]}
    else:
        for key, value in filters.items():
            filters[key] = transfer_filter(value)
    return filters


def filter_docs(docs: Document,
                filters: dict,
                data_source: dict,
                loose_filter: bool = False) -> List[DocumentCP]:
    """_summary_

    Args:
        docs (Document): the documents to be filtered
        filters (dict): the filters dictionary to be applied
        data_source (dict): the data type of each attributes from documents metadata
        loose_filter (bool, optional): if True, change $and to $or. Defaults to False.

    Returns:
        List[DocumentCP]: _description_
    """
    if filters is None or len(filters) == 0:
        return docs
    if loose_filter:
        filters = transfer_filter(filters)

    def recur(cur_docs: DocumentCP, filters: dict):
        res = set()
        flag = True
        if '$and' in filters:
            for f in filters['$and']:
                if flag:
                    flag = False
                    res = recur(cur_docs, f)
                else:
                    res = res.intersection(recur(cur_docs, f))
        elif '$or' in filters:
            for f in filters['$or']:
                res = res.union(recur(cur_docs, f))
        else:
            keyword = next(iter(filters))
            val = filters[keyword]

            res = find_subset(val, keyword, cur_docs, data_source)

        return res

    return list(recur(docs, filters['filter']))


def generate_query_filter(
    model_obj: BaseLanguageModel,
    prompt: str,
    metadata_field_info: List[AttributeInfo] = [],
    document_content_description: str = "",
) -> tuple[str, dict, dict[str, Any]]:
    """based on metadata_field_info and document_content_description, generate a query and filter to filter out documents

    Args:
        model_obj (BaseLanguageModel): _description_
        prompt (str): _description_
        metadata_field_info (List[AttributeInfo], optional): _description_. Defaults to [].
        document_content_description (str, optional): _description_. Defaults to "".

    Returns:
        tuple[str, dict, dict[str, Any]]: query, fnl_filter, data_source
    """
    query_constructor_prompt, data_source = generate_query_constructor(
        metadata_field_info, document_content_description, prompt)
    res = helper.call_model(model_obj, query_constructor_prompt)

    vis = ChromaTranslator()
    stq = StructuredQueryOutputParser.from_components().parse(res)

    query, fnl_filter = vis.visit_structured_query(stq)
    return query, fnl_filter, data_source
