from llama_cpp import Llama

llm = Llama(
    model_path="/Users/heart/Code/CPP-Project/llama-agent/models/Qwen3.5-4B-UD-Q8_K_XL.gguf",
    n_gpu_layers=-1,
    n_ctx=4096,
    n_threads=8,
    use_mmap=True,
    chat_format="chatml",
    verbose=False
)

print("模型加载完成，开始生成...\n")

prompt = "请用一句话解释 Transformer 的注意力机制。"

# 1) 显式看 tokenize
toks = llm.tokenize(prompt.encode("utf-8"), add_bos=True, special=False)
print("TOKENS_IN:", toks[:20], "len=", len(toks))

# 2) 生成（流式）看 token 输出过程
out = []
for chunk in llm.create_completion(
    prompt=prompt,
    max_tokens=64,
    temperature=0.7,
    top_p=0.9,
    stream=True,
):
    text = chunk["choices"][0]["text"]
    out.append(text)
    print(text, end="", flush=True)

print("\n\nFINAL:", "".join(out))