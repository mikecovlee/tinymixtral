#!/usr/bin/env python3
# Copyright (C) Michael Lee (李登淳) 2026. All rights reserved.
# Open-source under the MIT License. See LICENSE for details.
"""基于 Gradio 的 TinyMixtral 浏览器 Chat UI（流式输出，可调采样参数）。

用法:
    python scripts/chat_ui.py --model ~/tinymixtral-it
    python scripts/chat_ui.py --model mikecovlee/tinymixtral-it --port 8000
然后浏览器打开 http://127.0.0.1:8000
"""

import argparse
import os
import threading

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

STATE = {"model": None, "tokenizer": None, "device": "cpu"}

STOP_MARKERS = ("<|assistant|>", "<|user|>", "<|system|>")


class StopOnMarkers(StoppingCriteria):
    """生成文本中出现角色标记时停止，避免模型自问自答地续写。"""

    def __init__(self, tokenizer, prompt_len, markers):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.markers = markers

    def __call__(self, input_ids, scores, **kwargs):
        text = self.tokenizer.decode(input_ids[0, self.prompt_len:], skip_special_tokens=True)
        return any(m in text for m in self.markers)

CSS = """
html, body { scrollbar-gutter: stable; }
.gradio-container { width: 100% !important; max-width: 1200px !important; margin: 0 auto !important;
  padding: 0 16px !important; overflow-x: hidden !important; }
main.fillable { max-width: 1200px !important; }
#main-row { width: 100% !important; }
#chat-col { flex: 1 1 0 !important; min-width: 0 !important; max-width: 100% !important; overflow: hidden !important; }
#param-col { flex: 0 0 340px !important; min-width: 300px !important; max-width: 360px !important; }
#header { text-align: center; padding: 12px 0 6px; }
#header h1 {
  font-size: 30px; font-weight: 800; margin: 0; letter-spacing: -0.5px; color: #6366f1;
  background: linear-gradient(90deg, #6366f1, #a855f7 55%, #ec4899);
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
#header p { margin: 4px 0 0; font-size: 13px; opacity: .7; }
#chatbot { border-radius: 16px !important; scrollbar-gutter: stable; }
#chatbot .message { font-size: 15px; line-height: 1.6;
  overflow-wrap: anywhere !important; word-break: break-word !important; max-width: 100% !important; }
#input-row { gap: 8px; }
#input-row textarea { min-height: 46px; }
#send-btn, #clear-btn { min-width: 84px; height: 46px; }
.panel-card {
  border-radius: 16px !important; border: 1px solid var(--border-color-primary) !important;
  padding: 16px 18px !important;
}
.panel-card h4 { margin: 2px 0 10px; letter-spacing: .2px; }
footer { display: none !important; }
"""


def _seed_everything(seed):
    if seed is None or int(seed) < 0:
        return
    seed = int(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if STATE["device"] == "mps" and hasattr(torch, "mps"):
        torch.mps.manual_seed(seed)


def generate(messages, temperature, top_p, top_k, repetition_penalty, max_new_tokens, seed):
    """按采样参数流式生成，逐段 yield 累积文本。"""
    tok = STATE["tokenizer"]
    model = STATE["model"]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt", add_special_tokens=False).to(STATE["device"])
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    prompt_len = inputs["input_ids"].shape[1]
    kwargs = dict(
        **inputs,
        max_new_tokens=max(1, int(max_new_tokens or 0)),
        streamer=streamer,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
        repetition_penalty=float(repetition_penalty or 1.0),
        stopping_criteria=StoppingCriteriaList([StopOnMarkers(tok, prompt_len, STOP_MARKERS)]),
    )
    if temperature and float(temperature) > 0:
        kwargs.update(
            do_sample=True,
            temperature=float(temperature),
            top_p=min(max(float(top_p), 1e-4), 1.0),
            top_k=int(top_k) if top_k and int(top_k) > 0 else 0,
        )
    else:
        kwargs.update(do_sample=False)
    errors = {}

    def run():
        try:
            _seed_everything(seed)
            with torch.inference_mode():
                model.generate(**kwargs)
        except Exception as exc:
            errors["error"] = exc
        finally:
            streamer.end()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    acc = ""
    try:
        for chunk in streamer:
            acc += chunk
            cuts = [acc.find(m) for m in STOP_MARKERS if m in acc]
            if cuts:
                yield acc[:min(cuts)].rstrip()
                break
            yield acc
    finally:
        thread.join()
    if "error" in errors:
        raise errors["error"]


def user_submit(message, history):
    if not message or not message.strip():
        return "", history
    return "", history + [{"role": "user", "content": message.strip()}]


def bot(history, temperature, top_p, top_k, repetition_penalty, max_new_tokens, seed, system):
    """把 history 转成对话消息，流式填充最后一条 assistant 回复。"""
    if not history or history[-1]["role"] != "user":
        yield history
        return
    messages = []
    if system and system.strip():
        messages.append({"role": "system", "content": system.strip()})
    messages.extend({"role": t["role"], "content": t["content"]} for t in history)
    history = history + [{"role": "assistant", "content": ""}]
    try:
        for acc in generate(messages, temperature, top_p, top_k,
                            repetition_penalty, max_new_tokens, seed):
            history[-1]["content"] = acc
            yield history
    except Exception as exc:
        history[-1]["content"] = f"[error] {exc}"
        yield history


def build_demo(defaults):
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.violet,
        neutral_hue=gr.themes.colors.slate,
    )
    with gr.Blocks(theme=theme, css=CSS, title="TinyMixtral Chat") as demo:
        gr.HTML(
            '<div id="header"><h1>TinyMixtral Chat</h1>'
            f'<p>{defaults["params"]} · running on {defaults["device"]}</p></div>'
        )
        with gr.Row(equal_height=True, elem_id="main-row"):
            with gr.Column(scale=7, elem_id="chat-col"):
                chatbot = gr.Chatbot(
                    type="messages", elem_id="chatbot", height=520, show_label=False,
                    show_copy_button=True, placeholder="Ask TinyMixtral anything…",
                )
                with gr.Row(elem_id="input-row"):
                    msg = gr.Textbox(
                        placeholder="Type a message…",
                        show_label=False, autofocus=True, container=False,
                        scale=8, lines=1, max_lines=6,
                    )
                    send = gr.Button("Send", variant="primary", elem_id="send-btn", scale=1)
                    clear = gr.Button("Clear", variant="secondary", elem_id="clear-btn", scale=1)
            with gr.Column(scale=3, min_width=320, elem_id="param-col", elem_classes=["panel-card"]):
                gr.Markdown("#### Generation")
                temperature = gr.Slider(0, 1.5, value=defaults["temp"], step=0.05,
                                        label="Temperature (0 = greedy)")
                top_p = gr.Slider(0.05, 1.0, value=defaults["top_p"], step=0.01, label="Top-p")
                top_k = gr.Slider(0, 100, value=defaults["top_k"], step=1,
                                  label="Top-k (0 = off)")
                repetition_penalty = gr.Slider(1.0, 2.0, value=defaults["repetition_penalty"],
                                               step=0.01, label="Repetition penalty")
                max_tokens = gr.Slider(16, 1024, value=defaults["max_tokens"], step=16,
                                       label="Max new tokens")
                seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                gr.Markdown("#### System prompt")
                system = gr.Textbox(value=defaults["system"], lines=4, show_label=False,
                                    placeholder="You are a helpful assistant.")

        controls = [temperature, top_p, top_k, repetition_penalty, max_tokens, seed, system]
        outputs = [msg, chatbot]
        send.click(user_submit, [msg, chatbot], outputs, queue=False).then(
            bot, [chatbot, *controls], chatbot)
        msg.submit(user_submit, [msg, chatbot], outputs, queue=False).then(
            bot, [chatbot, *controls], chatbot)
        clear.click(lambda: [], None, chatbot, queue=False)
    return demo


def main():
    p = argparse.ArgumentParser(description="TinyMixtral Gradio Chat UI")
    p.add_argument("--model", default="~/tinymixtral-it", help="已发布的 HF 目录或 Hub 模型 id")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default=None, help="cuda/mps/cpu（默认自动）")
    p.add_argument("--max-tokens", type=int, default=256, help="默认最大生成长度")
    p.add_argument("--temp", type=float, default=0.7, help="默认采样温度（0 为贪心）")
    p.add_argument("--top-p", type=float, default=0.9, help="默认 nucleus 阈值")
    p.add_argument("--top-k", type=int, default=0, help="默认 top-k（0 为关闭）")
    p.add_argument("--repetition-penalty", type=float, default=1.1, help="重复惩罚")
    p.add_argument("--system", default="", help="默认 system prompt")
    args = p.parse_args()
    if args.max_tokens <= 0 or args.temp < 0 or not 0 < args.top_p <= 1:
        p.error("invalid generation parameters")

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
    model.generation_config.eos_token_id = tokenizer.eos_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval().to(device)
    if device == "cuda":
        model = model.to(torch.bfloat16)
    nM = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {nM:.0f}M params on {device}")

    STATE.update(model=model, tokenizer=tokenizer, device=device)

    demo = build_demo({
        "params": f"{nM:.0f}M params · MoE top-2/4",
        "device": device,
        "temp": args.temp,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "repetition_penalty": args.repetition_penalty,
        "max_tokens": args.max_tokens,
        "system": args.system,
    })
    demo.queue(default_concurrency_limit=1).launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
