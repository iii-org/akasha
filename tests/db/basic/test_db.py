import pytest
import akasha  # noqa: F401
from akasha.helper import handle_embeddings
from tests.support.live import load_test_env, require_keys
from tests.support.paths import DOCUMENTS_ROOT

CHUNK_SIZE = 1000
CERTAIN_FILE = DOCUMENTS_ROOT / "20230224_製造業機廠鏈智慧應用發展態勢.pdf"


@pytest.fixture(scope="module")
def emb_obj():
    require_keys("GEMINI_API_KEY")
    return handle_embeddings(
        "gemini:gemini-embedding-001",
        False,
        load_test_env(),
    )


@pytest.mark.db
@pytest.mark.live
@pytest.mark.requires_api
@pytest.mark.smoke
def test_create_db(emb_obj):
    from akasha.utils.db.create_db import create_directory_db, create_single_file_db

    suc, ign = create_directory_db(DOCUMENTS_ROOT, emb_obj, CHUNK_SIZE)

    assert suc is True
    assert ign == []

    suc = create_single_file_db(CERTAIN_FILE, emb_obj, CHUNK_SIZE)

    assert suc is True

    return


@pytest.mark.db
@pytest.mark.live
@pytest.mark.requires_api
@pytest.mark.smoke
def test_load_extract_db(emb_obj):
    from akasha.utils.db import process_db, extract_db_by_file

    db, ign = process_db(DOCUMENTS_ROOT, emb_obj, CHUNK_SIZE)

    assert len(db.get_ids()) > 0

    assert ign == []

    new_db = extract_db_by_file(db, [CERTAIN_FILE])

    assert len(db.get_ids()) > len(new_db.get_ids())

    return


@pytest.mark.db
@pytest.mark.live
@pytest.mark.requires_api
@pytest.mark.smoke
def test_delete_file_db(emb_obj):
    from akasha.utils.db import delete_documents_by_file

    delete_num = delete_documents_by_file(CERTAIN_FILE, emb_obj, CHUNK_SIZE)

    assert delete_num > 0

    return
