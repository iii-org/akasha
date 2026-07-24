import pytest
import os
import akasha
from pathlib import Path
from dotenv import load_dotenv
import shutil

# 測試用路徑
BASE_PATH = Path(__file__).parent
DATA_PATH = BASE_PATH / "tests_data"
DOCS_PATH = DATA_PATH / "docs"
IMAGES_PATH = DATA_PATH / "images"

# 載入環境變數 (指定從 tests/.env 讀取)
ENV_PATH = BASE_PATH / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# 待測試模型 (由使用者指定)
MODELS = [
    "openai:gpt-4o",
    "google:gemini-2.5-flash", 
]

EMBEDDINGS = [
    "openai:text-embedding-3-small",
    "google:gemini-embedding-001",
]

@pytest.fixture(scope="module")
def check_env():
    """檢查必要的 API Key 是否存在"""
    required_keys = ["OPENAI_API_KEY", "GEMINI_API_KEY", "BRAVE_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    if missing_keys:
        pytest.skip(f"缺少必要的 API Key: {', '.join(missing_keys)}")

@pytest.mark.integration
@pytest.mark.upgrade
@pytest.mark.requires_api
@pytest.mark.parametrize("model_name", MODELS)
def test_api_basic_ask(check_env, model_name):
    """測試最基本的 API 問答是否通暢"""
    ak = akasha.ask(model=model_name, verbose=True, env_file=str(ENV_PATH))
    response = ak("你好，請簡短回答。你是誰？")
    assert isinstance(response, str)
    assert len(response) > 0
    print(f"\n[Model: {model_name}] Basic Ask Response: {response}")

@pytest.mark.integration
@pytest.mark.upgrade
@pytest.mark.requires_api
@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("embed_name", EMBEDDINGS)
def test_rag_smoke(check_env, model_name, embed_name):
    """[Light] RAG 煙霧測試：使用遠端 API 進行文件檢索與回答"""
    if not DOCS_PATH.exists():
        pytest.skip("找不到 tests_data/docs 資料夾")
        
    ak = akasha.RAG(model=model_name, embeddings=embed_name, verbose=True, env_file=str(ENV_PATH))
    # data_source 可以是目錄路徑
    response = ak(data_source=str(DOCS_PATH), prompt="這份文件的主要內容是什麼？")
    
    assert isinstance(response, str)
    assert len(response) > 0
    print(f"\n[Model: {model_name}, Embed: {embed_name}] RAG Response: {response}")

@pytest.mark.integration
@pytest.mark.upgrade
@pytest.mark.requires_api
@pytest.mark.parametrize("model_name", MODELS)
def test_agent_native_tool_calling(check_env, model_name):
    """測試原生 tool calling Agent 讀取 JSON 並處理資訊的能力"""
    json_file = DOCS_PATH / "simple_case.json"
    if not json_file.exists():
        pytest.skip("找不到 simple_case.json 文件")

    # 建立 Agent 並賦予基本工具
    import akasha.agent.agent_tools as at
    tool_list = [at.websearch_tool(search_engine="brave")]
    
    agent = akasha.agents(
        model=model_name,
        tools=tool_list,
        verbose=True,
        env_file=str(ENV_PATH)
    )
    
    # 命令 Agent 結合 JSON 文件內容進行回答
    with open(json_file, 'r', encoding='utf-8') as f:
        json_content = f.read()
        
    prompt = f"請根據以下 JSON 內容，告訴我內容中提到的版本號是多少：\n{json_content}"
    response = agent(prompt)
    
    assert isinstance(response, str)
    assert len(response) > 0
    print(f"\n[Model: {model_name}] Agent Native Tool Task Response: {response}")

@pytest.mark.integration
@pytest.mark.upgrade
@pytest.mark.requires_api
@pytest.mark.parametrize("model_name", MODELS)
def test_vision_card_recognition(check_env, model_name):
    """測試多模態影像辨識（名片辨識）"""
    # 找出 images 資料夾下的圖檔
    image_files = list(IMAGES_PATH.glob("*.jpg")) + list(IMAGES_PATH.glob("*.png"))
    if not image_files:
        pytest.skip("找不到測試影像檔 (jpg/png)")

    target_image = str(image_files[0])
    
    # 使用 akasha.ask 的 vision 功能
    qa = akasha.ask(model=model_name, verbose=True, env_file=str(ENV_PATH))
    prompt = "這是一張名片。請幫我提取這張名片上的姓名、電話與公司名稱（如果有）。"
    
    try:
        response = qa.vision(prompt=prompt, image_path=target_image)
    except AttributeError:
        pytest.skip(f"模型 {model_name} 可能在當前版本不支援 vision 方法")

    assert isinstance(response, str)
    assert len(response) > 0
    print(f"\n[Model: {model_name}] Vision Response: {response}")

@pytest.mark.integration
@pytest.mark.upgrade
@pytest.mark.requires_api
@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("embed_name", EMBEDDINGS)
def test_memory_stability(check_env, model_name, embed_name):
    """測試 Memory 記憶功能 (light 版使用遠端 embedding + Chroma)"""
    mem = akasha.MemoryManager(
        model=model_name,
        embeddings=embed_name,
        memory_name="stability_test",
        verbose=True,
        env_file=str(ENV_PATH)
    )
    
    # 模擬兩輪對話
    mem.add_memory("使用者：我住在台北。", "助手：收到，您住在台北。")
    mem.add_memory("使用者：我喜歡吃拉麵。", "助手：好的，您喜歡吃拉麵。")
    
    # 測試檢索
    query = "請問我住在哪裡？喜歡吃什麼？"
    history = mem.search_memory(query, top_k=2)
    
    # 結合 RAG/Ask 回答
    qa = akasha.ask(model=model_name, env_file=str(ENV_PATH))
    response = qa(query, history_messages=history)
    
    assert "台北" in response or "拉麵" in response
    print(f"\n[Model: {model_name}, Embed: {embed_name}] Memory Check Response: {response}")


@pytest.mark.integration
@pytest.mark.upgrade
@pytest.mark.requires_api
@pytest.mark.smoke
@pytest.mark.skipif(os.name != "nt", reason="Windows absolute-path regression test")
def test_windows_absolute_path_rag_smoke(check_env, tmp_path):
    """驗證 Windows absolute path 可搭配遠端 embedding 與 Chroma 正常運作。"""
    test_file = tmp_path / "windows-absolute-rag.txt"
    test_file.write_text(
        "The capital of France is Paris. The Eiffel Tower is in Paris.",
        encoding="utf-8",
    )

    ak = akasha.RAG(
        model="openai:gpt-4o",
        embeddings="openai:text-embedding-3-small",
        verbose=True,
        env_file=str(ENV_PATH),
    )
    response = ak(data_source=str(test_file), prompt="What is the capital of France?")

    assert "Paris" in response or "巴黎" in response
    print(f"\n[Windows Absolute Path RAG] Response: {response}")


@pytest.mark.integration
@pytest.mark.upgrade
@pytest.mark.requires_api
@pytest.mark.smoke
@pytest.mark.skipif(os.name != "nt", reason="Windows absolute-path regression test")
def test_windows_absolute_path_memory_smoke(check_env, tmp_path):
    """驗證 MemoryManager 可使用 Windows absolute path 作為持久化目錄。"""
    mem_root = tmp_path / "memory-root"
    shutil.rmtree(mem_root, ignore_errors=True)

    mem = akasha.MemoryManager(
        model="openai:gpt-4o",
        embeddings="openai:text-embedding-3-small",
        memory_name="smoke",
        memory_dirname=str(mem_root),
        verbose=True,
        env_file=str(ENV_PATH),
    )
    mem.add_memory("使用者：我住在台北。", "助手：收到，您住在台北。")
    mem.add_memory("使用者：我喜歡吃拉麵。", "助手：好的，您喜歡吃拉麵。")

    history = mem.search_memory("請問我住在哪裡？喜歡吃什麼？", top_k=2)

    assert len(history) >= 2
    assert any("台北" in item for item in history)
    assert any("拉麵" in item for item in history)
    print(f"\n[Windows Absolute Path Memory] History: {history}")
