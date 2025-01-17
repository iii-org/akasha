import pytest
import akasha

EMB_OBJ = akasha.handle_embeddings("openai:text-embedding-ada-002", False, "")
CHUNK_SIZE = 1000
CERTAIN_FILE = "docs/mic/20230224_製造業機廠鏈智慧應用發展態勢.pdf"


@pytest.mark.db
def test_create_db():
    from akasha.utils.db.create_db import create_directory_db, create_single_file_db
    suc, ign = create_directory_db("docs/mic", EMB_OBJ, CHUNK_SIZE)

    assert suc == True
    assert ign == []

    suc = create_single_file_db(CERTAIN_FILE, EMB_OBJ, CHUNK_SIZE)

    assert suc == True

    return


@pytest.mark.db
def test_load_extract_db():

    db, ign = akasha.process_db("docs/mic", EMB_OBJ, CHUNK_SIZE)

    assert len(db.get_ids()) > 0

    assert ign == []

    new_db = akasha.extract_db_by_file(db, [CERTAIN_FILE])

    assert len(db.get_ids()) > len(new_db.get_ids())

    return


@pytest.mark.db
def test_delete_file_db():

    delete_num = akasha.delete_documents_by_file(CERTAIN_FILE, EMB_OBJ,
                                                 CHUNK_SIZE)

    assert delete_num > 0

    return
