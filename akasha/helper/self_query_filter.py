from langchain_core.documents import Document

# from langchain_openai import OpenAIEmbeddings
from langchain.chains.query_constructor.base import AttributeInfo
from typing import List, Union, Set, Any, Tuple, Optional, Callable
import json
from pydantic import Field
import operator
from langchain_core.language_models.base import BaseLanguageModel
import akasha.helper as helper
from akasha.utils.db import dbs, extract_db_by_ids
import re
import logging


def self_query(
    prompt: str,
    model_obj: BaseLanguageModel,
    db: dbs,
    metadata_field_info: List[Union[AttributeInfo, dict]] = [],
    document_content_description: str = "",
    custom_parser: Optional[Callable[[str], Tuple[str, dict]]] = None,
    loose_filter: bool = False,
) -> Tuple[dbs, str, dict]:
    """use prompt to generate query and filter to filter out documents, return the filtered documents dbs and the query and matched fields

    Args:
        prompt (str): user question to generate query and filter
        model_obj (BaseLanguageModel): llm object
        db (dbs): the documents dbs to be filtered
        metadata_field_info (List[Union[AttributeInfo, dict]], optional): the metadata attributes info . Defaults to [].
        document_content_description (str, optional): the desciption of documents. Defaults to "".
        custom_parser (Optional[Callable[[str], Tuple[str, dict]]], optional): custom parser function. Defaults to None.

    Returns:
        Tuple[dbs, str, dict]: the filtered documents dbs, the query(str) and matched fields dict(dict[db.ids, List[str]])
    """
    query, fnl_filter, data_source = generate_query_filter(
        model_obj,
        prompt,
        metadata_field_info,
        document_content_description,
        custom_parser,
    )

    new_docs = [
        DocumentCP(doc, metadata, ids)
        for doc, metadata, ids in zip(db.docs, db.metadatas, db.ids)
    ]
    filtered_docs = filter_docs(new_docs, fnl_filter, data_source, loose_filter)
    filtered_docs_set = set(fd.ids for fd in filtered_docs)
    matched_fields_dict = {fd.ids: fd.matched_fields for fd in filtered_docs}
    new_dbs = extract_db_by_ids(db, filtered_docs_set)
    return new_dbs, query, matched_fields_dict


def check_metadata_info(
    metadata_info: List[Union[AttributeInfo, dict]],
) -> List[AttributeInfo]:
    """check if the metadata_info is in the right format"""

    new_metadata_info = []
    if isinstance(metadata_info, list):
        for i, attr in enumerate(metadata_info):
            if isinstance(attr, dict):
                if "name" in attr and "description" in attr and "type" in attr:
                    new_metadata_info.append(AttributeInfo(**attr))
                else:
                    logging.warning(
                        """The attribute info should be a dictionary with keys "name", "description" and "type"."""
                    )
            elif isinstance(attr, AttributeInfo):
                new_metadata_info.append(attr)
            else:
                logging.warning(
                    """The attribute info should be a dictionary or an instance of AttributeInfo."""
                )
    else:
        logging.warning(
            """The metadata info should be a list of dictionaries or instances of AttributeInfo."""
        )
    return new_metadata_info


def generate_query_constructor(
    metadata_info: List[Union[AttributeInfo, dict]],
    document_content_description: str,
    prompt: str,
) -> tuple[str, dict]:
    """generate a query constructor prompt based on metadata_info and document_content_description"""

    new_metadata_info = check_metadata_info(metadata_info)

    data_source = {
        "content": document_content_description,
        "attributes": {
            attr.name: {"description": attr.description, "type": attr.type}
            for attr in new_metadata_info
        },
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
    "filter": "and(or(eq(artist, Taylor Swift), eq(artist, Katy Perry)), lt(length, 180))"
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
    """Document subclass for comparing equality based on page content only.

    Args:
        Document (_type_): _description_

    Returns:
        _type_: _description_
    """

    ids: Union[int, str] = Field(default="")
    matched_fields: List[str] = Field(default=[])

    def __init__(self, page_content, metadata, ids):
        super().__init__(page_content)
        self.metadata = metadata
        self.ids = ids

    def __hash__(self):
        return hash(self.page_content)

    def __eq__(self, other):
        return isinstance(other, DocumentCP) and self.page_content == other.page_content


def handle_attr(
    attr: Union[int, float, str], data_source: dict, keyword: str
) -> Union[int, float, str]:
    """change attriubte type based on the data_source

    Args:
        attr (Union[int, float, str]): the attribute to be changed
        data_source (dict): the dictionary which has the data type of each attributes from documents metadata
        keyword (str): the attribute key name

    Returns:
        Union[int, float, str]:  changed type of the attribute
    """
    if data_source["attributes"][keyword]["type"] == "string":
        return str(attr)
    elif data_source["attributes"][keyword]["type"] == "integer":
        return int(attr)
    elif data_source["attributes"][keyword]["type"] == "float":
        return float(attr)
    return str(attr)


def find_subset(
    val: Union[int, float, str],
    keyword: str,
    cur_docs: List[DocumentCP],
    data_source: dict,
) -> Set[DocumentCP]:
    """base on the keyword value and operator, find the subset of documents

    Args:
        val (Union[int, float, str]): _description_
        keyword (str): _description_
        cur_docs (List[DocumentCP]): _description_
        data_source (dict): _description_

    Returns:
        Set[DocumentCP]: _description_
    """
    # Define a dictionary to map operators to functions
    operator_map = {
        "$eq": operator.eq,
        "$ne": operator.ne,
        "$gt": operator.gt,
        "$gte": operator.ge,
        "$lt": operator.lt,
        "$lte": operator.le,
    }

    res = set()
    for op, func in operator_map.items():
        if op in val:
            attr = handle_attr(val[op], data_source, keyword)

            if isinstance(attr, str) and op == "$eq":
                attr = attr.lower()
                for doc in cur_docs:
                    if (
                        keyword in doc.metadata
                        and attr in doc.metadata[keyword].lower()
                    ):
                        doc.matched_fields.append(keyword)
                        res.add(doc)
            else:
                for doc in cur_docs:
                    if keyword in doc.metadata and func(doc.metadata[keyword], attr):
                        doc.matched_fields.append(keyword)
                        res.add(doc)

    return res


def transfer_filter(filters: Union[dict, int, float, str]) -> dict:
    """Recursively convert all $and filters to $or filters."""
    if not isinstance(filters, dict):
        return filters
    elif "$and" in filters:
        filters = {"$or": [transfer_filter(f) for f in filters["$and"]]}
    else:
        for key, value in filters.items():
            filters[key] = transfer_filter(value)
    return filters


def filter_docs(
    docs: Document, filters: dict, data_source: dict, loose_filter: bool = False
) -> List[DocumentCP]:
    """filter the documents based on the filters

    Args:
        docs (Document): the documents to be filtered
        filters (dict): the filters dictionary to be applied
        data_source (dict): the data type of each attributes from documents metadata
        loose_filter (bool, optional): if True, change $and to $or. Defaults to False.

    Returns:
        List[DocumentCP]: return the filtered documents
    """
    if filters is None or len(filters) == 0:
        return docs
    if loose_filter:
        filters = transfer_filter(filters)

    def recur(cur_docs: DocumentCP, filters: dict):
        res = set()
        flag = True
        if "$and" in filters:
            for f in filters["$and"]:
                if flag:
                    flag = False
                    res = recur(cur_docs, f)
                else:
                    res = res.intersection(recur(cur_docs, f))
        elif "$or" in filters:
            for f in filters["$or"]:
                res = res.union(recur(cur_docs, f))
        else:
            keyword = next(iter(filters))
            val = filters[keyword]

            res = find_subset(val, keyword, cur_docs, data_source)

        return res

    return list(recur(docs, filters))


def translate(expr):
    """Translate a logical expression to a nested dictionary format."""
    # Pattern to match logical operations (e.g., and(...), or(...))
    logical_op_pattern = re.compile(r"(and|or|AND|OR)\((.*)\)")
    # Pattern to match comparison operations (e.g., eq(公司名稱, a公司))
    comparison_op_pattern = re.compile(r"(eq|ne|gt|gte|lt|lte)\(([^,]+),([^,]+)\)")

    # Check if it's a logical operation
    logical_match = logical_op_pattern.match(expr)
    if logical_match:
        logical_op = logical_match.group(1)  # "and" or "or"
        inner_expr = logical_match.group(2)  # Expressions inside the parentheses

        # Split inner expressions by commas not inside parentheses
        parts = split_expressions(inner_expr)
        # Recursively translate each part
        parsed_parts = [translate(part.strip()) for part in parts]
        return {f"${logical_op}": parsed_parts}

    # Check if it's a comparison operation
    comparison_match = comparison_op_pattern.match(expr)
    if comparison_match:
        comparison_op, field, value = comparison_match.groups()
        field = field.strip()
        value = value.strip()
        return {field: {f"${comparison_op}": value}}

    raise ValueError("Invalid expression format")


def split_expressions(expr):
    # Split expressions by commas outside parentheses
    parts = []
    depth = 0
    current_part = []

    for char in expr:
        if char == "," and depth == 0:
            parts.append("".join(current_part).strip())
            current_part = []
        else:
            current_part.append(char)
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1

    if current_part:
        parts.append("".join(current_part).strip())

    return parts


def translate_output(llm_res: str) -> tuple[str, dict]:
    """translate the output of the language model to query and filter
    the output of llm may look like this:
    ```json
    {
        "query": "產品疑慮",
        "filter": "and(eq(公司名稱, a公司), eq(拜訪年, 2024))"
    }
    ```
    first we extract the query and filter from the llm_res by using helper.extract_json
    then we translate the filter to a dictionary looks like this:
    {'$and': [{'公司名稱': {'$eq': 'a公司'}}, {'拜訪年': {'$eq': '2024'}}]}

    Args:
        llm_res (str): the output of the language model

    Returns:
        tuple[str, dict]: _description_
    """
    query = ""
    filters = {}
    try:
        llm_res = helper.extract_json(llm_res)
        query = llm_res["query"]
        filter = llm_res["filter"].replace('"', "").replace("\\", "").replace("  ", "")

        filters = translate(filter)
        print(filters)
    except Exception as e:
        print(f"Error: {e}")

    return query, filters


def generate_query_filter(
    model_obj: BaseLanguageModel,
    prompt: str,
    metadata_field_info: List[Union[AttributeInfo, dict]] = [],
    document_content_description: str = "",
    custom_parser: Optional[Callable[[str], Tuple[str, dict]]] = None,
) -> tuple[str, dict, dict[str, Any]]:
    """based on metadata_field_info and document_content_description, generate a query and filter to filter out documents

    Args:
        model_obj (BaseLanguageModel): llm object
        prompt (str): the user question to generate the query and filter
        metadata_field_info (List[Union[AttributeInfo, dict]], optional): _description_. Defaults to [].
        document_content_description (str, optional): _description_. Defaults to "".
        if custom_parser is not None, it will be used to parse the output of the model
    Returns:
        tuple[str, dict, dict[str, Any]]: query, fnl_filter, data_source
    """
    query_constructor_prompt, data_source = generate_query_constructor(
        metadata_field_info, document_content_description, prompt
    )
    res = helper.call_model(model_obj, query_constructor_prompt, False)

    if custom_parser is not None:
        query, fnl_filter = custom_parser(res)
    else:
        query, fnl_filter = translate_output(res)

    return query, fnl_filter, data_source
