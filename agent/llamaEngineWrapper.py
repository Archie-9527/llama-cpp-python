from typing import List, Optional, Generator, Dict, Any
import llama_cpp

# 类型别名
Token = int
Tokens = List[int]

# ----------------------------------------------------------------------
# 轻量会话上下文，完全由 Wrapper 管理，上层只持有此对象引用
# ----------------------------------------------------------------------
class SessionContext:
    """
    表示一个推理会话的上下文状态，封装 llama.cpp 序列 ID、当前位置等信息。
    上层代码不应修改其内容，仅通过 Wrapper 接口操作。
    """
    def __init__(self, seq_id: int):
        self.seq_id = seq_id
        self.current_pos = 0          # 当前已解码的 token 总数（序列长度）
        self.system_end_pos = 0       # 系统提示结束位置，用于分支共享
        self.tool_call_start_pos = -1 # 最近一次工具调用开始位置（用于回收）
        self.metadata: Dict[str, Any] = {}


# ----------------------------------------------------------------------
# 引擎抽象层
# ----------------------------------------------------------------------
class LlamaEngineWrapper:
    """
    对 llama.cpp 引擎的完整封装，提供会话管理、生成、内存优化原语。
    内部使用 llama-cpp-python 的低阶 API 或 ctypes 绑定。
    """

    def __init__(self, model_path: str, **kwargs):
        """
        加载模型并初始化引擎。
        Args:
            model_path: GGUF 模型文件路径。
            **kwargs: 传递给 llama.cpp 的额外参数，如 n_gpu_layers, n_ctx 等。
        """
        # 使用 llama-cpp-python 的高级 Llama 对象，同时保留其底层 C 指针以调用序列 API
        self._model = llama_cpp.Llama(
            model_path=model_path,
            n_ctx=kwargs.get("n_ctx", 4096),
            n_gpu_layers=kwargs.get("n_gpu_layers", 0),
            verbose=False,
            # 其他参数...
        )
        # 获取底层 C 结构体指针，用于调用 llama_kv_cache_seq_* 等原生函数
        self._ctx = self._model._ctx  # llama_context pointer (ctypes)
        self._next_seq_id = 0         # 简单的序列 ID 分配器
        self._sessions: Dict[int, SessionContext] = {}

        # 保存模型的相关信息
        self.eos_token_id = self._model.token_eos()
        self.bos_token_id = self._model.token_bos()

    # ---------- Token 化辅助方法 ----------
    def encode(self, text: str) -> Tokens:
        """将文本转换为 token 列表，上层构建上下文时使用。"""
        return self._model.tokenize(text.encode("utf-8"), add_bos=False)

    def decode(self, tokens: Tokens) -> str:
        """将 token 列表解码回文本。"""
        return self._model.detokenize(tokens).decode("utf-8", errors="replace")

    # ---------- 会话管理 ----------
    def create_session(self, system_prompt: str) -> SessionContext:
        """
        创建一个新会话，将系统提示作为共享前缀写入 KV Cache。
        返回 SessionContext 对象供后续操作。
        """
        seq_id = self._next_seq_id
        self._next_seq_id += 1
        ctx = SessionContext(seq_id)

        # 将系统提示 token 化并写入模型
        sys_tokens = self.encode(system_prompt)
        if sys_tokens:
            self._batch_add_tokens(seq_id, ctx, sys_tokens)  # 内部完成 decode 并更新 pos
            ctx.system_end_pos = ctx.current_pos

        self._sessions[seq_id] = ctx
        return ctx

    def fork_session(self, parent: SessionContext, at_pos: Optional[int] = None) -> SessionContext:
        """
        基于父会话创建一个分叉会话，共享 KV Cache 前缀。
        如果 at_pos 为 None，则在父会话的当前位置分叉（共享到此为止的全部前缀）。
        内部使用 llama_kv_cache_seq_cp 实现共享。
        """
        if at_pos is None:
            at_pos = parent.current_pos
        seq_id = self._next_seq_id
        self._next_seq_id += 1
        child = SessionContext(seq_id)

        # 复制父序列的 KV Cache [0, at_pos) 到新序列
        # 底层调用 llama_kv_cache_seq_cp(self._ctx, parent.seq_id, seq_id, 0, at_pos)
        self._kv_cache_seq_cp(parent.seq_id, seq_id, 0, at_pos)
        child.current_pos = at_pos
        child.system_end_pos = parent.system_end_pos

        self._sessions[seq_id] = child
        return child

    def free_session(self, ctx: SessionContext):
        """
        释放会话占用的所有 KV Cache 及相关资源。
        """
        # 删除整个序列的 KV Cache
        self._kv_cache_seq_rm(ctx.seq_id, 0, -1)
        self._sessions.pop(ctx.seq_id, None)

    def append_tokens(self, ctx: SessionContext, tokens: Tokens):
        """
        向会话追加一段 token（例如工具调用结果），立即更新 KV Cache，
        并推进 current_pos。
        """
        self._batch_add_tokens(ctx.seq_id, ctx, tokens)

    # ---------- 生成器（核心推理原语）----------
    def generate(
        self,
        ctx: SessionContext,
        stop_tokens: Optional[List[int]] = None,
        max_new_tokens: int = 512,
    ) -> Generator[Token, None, None]:
        """
        流式生成 token 序列，每生成一个 token 就 yield。
        遇到 stop_tokens 中的任何一个 token 时停止，并 yield 该 stop token 后结束。
        不会在内部修改会话状态，由上层决定如何处理生成的 token。
        """
        if stop_tokens is None:
            stop_tokens = [self.eos_token_id]

        for _ in range(max_new_tokens):
            # 单步推理：以当前序列的最后一个 token 作为输入
            next_token = self._sample_next_token(ctx.seq_id, ctx.current_pos)
            yield next_token

            # 手动将新 token 加入序列，更新 position
            self._append_single_token(ctx.seq_id, ctx, next_token)

            if next_token in stop_tokens:
                break

    # ---------- KV Cache 管理原语（为优化策略暴露）----------
    def mark_region_for_cleanup(
        self, ctx: SessionContext, start_pos: int, end_pos: int, label: str = "tool_call"
    ):
        """
        标记从 start_pos 到 end_pos 的 token 区域，可供后续按需清理。
        比如工具调用段可标记为 "tool_invocation"，等待回收。
        """
        # 记录标记以便上层决策
        ctx.metadata.setdefault("marked_regions", []).append({
            "start": start_pos,
            "end": end_pos,
            "label": label,
        })

    def cleanup_marked_region(self, ctx: SessionContext, label: str):
        """
        清除所有标记为 label 的区域，释放对应的 KV Cache。
        """
        regions = ctx.metadata.get("marked_regions", [])
        for region in regions:
            if region["label"] == label:
                self._kv_cache_seq_rm(ctx.seq_id, region["start"], region["end"])
                # 清理后 current_pos 不变，但 KV 槽位已空出，后续使用需谨慎
        # 移除已清理的标记
        ctx.metadata["marked_regions"] = [
            r for r in regions if r["label"] != label
        ]

    def get_kv_cache_usage(self, ctx: SessionContext) -> int:
        """
        返回当前会话在 KV Cache 中占用的 token 数（近似值）。
        """
        # 可以通过 llama_get_kv_cache_token_count 或基于内部结构计算
        return ctx.current_pos  # 简化处理，实际应考虑碎片

    def get_peak_memory_usage(self) -> int:
        """返回当前引擎进程的显存占用（字节），用于 Benchmark。"""
        # 具体实现依赖 nvml 或 ggml 的内存统计接口
        return 0  # 占位

    # ================================================================
    # 内部辅助方法（封装对 llama-cpp-python 原生 API 的调用）
    # ================================================================
    def _batch_add_tokens(self, seq_id: int, ctx: SessionContext, tokens: Tokens):
        """将多个 token 一次性送入模型，更新 KV Cache。"""
        # 构造 batch：所有 token 同属一个序列，位置从 current_pos 开始递增
        n_tokens = len(tokens)
        batch = llama_cpp.llama_batch_get_one(tokens, n_tokens, seq_id, ctx.current_pos)
        self._model.eval(batch)  # 内部会调用 llama_decode 并更新 KV Cache
        ctx.current_pos += n_tokens

    def _append_single_token(self, seq_id: int, ctx: SessionContext, token: Token):
        """追加单个 token，用于生成过程中。"""
        self._batch_add_tokens(seq_id, ctx, [token])

    def _sample_next_token(self, seq_id: int, current_pos: int) -> Token:
        """
        基于当前序列状态，采样下一个 token。
        实际需要：构建 batch 仅包含当前序列的最后一个 token（位置 current_pos-1），
        进行一次 decode 得到 logits，然后应用采样器。
        """
        # 简化示例：使用 Llama 对象的内置采样（需要先 eval 最后一个 token）
        # 注意：生成过程中，kv cache 在 _append_single_token 中已更新，
        # 这里我们需要的是“预测下一个 token”，因此需从最新状态开始。
        last_token = self._get_last_token(seq_id)  # 从内部获取序列的最后一个 token
        batch = llama_cpp.llama_batch_get_one([last_token], 1, seq_id, current_pos)
        self._model.eval(batch)
        # 从 logits 中采样
        logits = self._model.scores[-1, :]  # 最后一个 token 的 logits
        token = self._model.sample(logits, temperature=0.7, top_p=0.9)
        return token

    def _get_last_token(self, seq_id: int) -> Token:
        """从引擎内部状态获取某个序列的最后一个 token（需根据实际实现调整）。"""
        # 实际需维护一个 session token 历史缓冲区，这里仅示意
        return 0

    def _kv_cache_seq_rm(self, seq_id: int, p0: int, p1: int):
        """
        删除序列 seq_id 中 [p0, p1) 范围的 KV Cache。
        对应 llama_kv_cache_seq_rm(self._ctx, seq_id, p0, p1)。
        """
        # 实际调用：llama_cpp.llama_kv_cache_seq_rm(self._ctx, seq_id, p0, p1)
        pass

    def _kv_cache_seq_cp(self, src_seq: int, dst_seq: int, p0: int, p1: int):
        """
        将源序列 [p0, p1) 的 KV Cache 复制到目标序列。
        对应 llama_kv_cache_seq_cp(self._ctx, src_seq, dst_seq, p0, p1)。
        """
        # 实际调用：llama_cpp.llama_kv_cache_seq_cp(self._ctx, src_seq, dst_seq, p0, p1)
        pass