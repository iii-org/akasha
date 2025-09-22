import akasha


ak = akasha.ask(
    model="openai:gpt-4o", max_input_tokens=8000, keep_logs=True, verbose=True
)


mem = akasha.MemoryManager(
    memory_name="test_memory",
    model="openai:gpt-4o",
    embeddings="openai:text-embedding-3-small",
    verbose=True,
)
mem.add_memory("Hi, 我的名字是小宋", "Hello, 小宋! 很高興認識你。")


prompt = "我的名字是什麼?"
history_msg = mem.search_memory(prompt, top_k=3)

response = ak(
    prompt=prompt,
    history_messages=history_msg,
)
