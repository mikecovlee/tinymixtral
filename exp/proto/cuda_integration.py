import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch  # noqa: E402

from model.config import TinyMixtralConfig  # noqa: E402
from model.cpt_router import CPTRouter  # noqa: E402
from model.modeling import TinyMixtralForCausalLM  # noqa: E402
from scripts.train_utils import make_adamw, make_cosine_schedule  # noqa: E402


def cfg(**kw):
    base = dict(
        vocab_size=41,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        num_local_experts=4,
        num_experts_per_tok=2,
        expert_intermediate_size=32,
        cpt_projection_dim=16,
        cpt_init_seed=3,
    )
    base.update(kw)
    return TinyMixtralConfig(**base)


dev = "cuda"
torch.manual_seed(5)

# --- 1) eager vs compiled on CUDA: probabilities, gradients, proposal ---
router = CPTRouter(cfg(cpt_state_chunk_size=8), layer_index=0).to(dev)
x = torch.randn(3, 10, 64, device=dev, requires_grad=True)
mask = torch.randint(0, 2, (3, 10), device=dev, dtype=torch.bool)
mask[0, 0] = True

# eager first (bypass compile)
ref = router._forward_impl(x, mask)
ref.probabilities.square().sum().backward()
ref_prob = ref.probabilities.detach().clone()
ref_load = ref.proposal.load_sum.detach().clone()
ref_xg = x.grad.detach().clone()
ref_grads = {}
for name, param in (("projection", router.projection), ("anchors", router.anchors), ("energy", router.energy)):
    ref_grads[name] = param.grad.detach().clone()
router.zero_grad()
x.grad = None

out = router(x, mask)  # compiled path
out.probabilities.square().sum().backward()
print("prob max err:", (out.probabilities.detach() - ref_prob).abs().max().item())
print("load_sum max err:", (out.proposal.load_sum.detach() - ref_load).abs().max().item())
print("x grad max err:", (x.grad - ref_xg).abs().max().item())
for name, param in (("projection", router.projection), ("anchors", router.anchors), ("energy", router.energy)):
    print(f"{name} grad max err:", (param.grad - ref_grads[name]).abs().max().item())

# --- 2) full model training step: autocast bf16 + checkpointing + commit ---
torch.manual_seed(6)
model = TinyMixtralForCausalLM(cfg()).to(dev)
model.gradient_checkpointing_enable()
model.train()
opt = make_adamw(model, lr=1e-3, weight_decay=0.01)
sched = make_cosine_schedule(opt, warmup_steps=2, total_steps=10)

input_ids = torch.randint(0, 41, (2, 16), device=dev)
attention_mask = torch.ones(2, 16, dtype=torch.long, device=dev)
attention_mask[1, 12:] = 0

for step in range(3):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids, labels=input_ids.clone(), attention_mask=attention_mask)
    loss = out["loss"]
    assert torch.isfinite(loss), f"non-finite loss at step {step}"
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert torch.isfinite(grad_norm), f"non-finite grad norm at step {step}"
    opt.step()
    version = model.commit_cpt_transaction(out["cpt_transaction"], optimizer_step=step + 1)
    sched.step()
    opt.zero_grad(set_to_none=True)
    print(f"step {step}: loss={loss.item():.4f} version={version}")

model.validate_persistent_cpt_state()
print("checkpointed bf16 training + commit: OK")

# --- 3) eval mode (no grad) through compiled path ---
model.eval()
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    eval_out = model(input_ids, attention_mask=attention_mask)
assert torch.isfinite(eval_out["logits"]).all()
print("eval path: OK")

# --- 4) save/load roundtrip after compiled training ---
import tempfile  # noqa: E402

with tempfile.TemporaryDirectory() as tmp:
    model.save_pretrained(tmp)
    loaded = TinyMixtralForCausalLM.from_pretrained(tmp).to(dev)
    loaded.eval()
    with torch.no_grad():
        a = model(input_ids)["logits"]
        b = loaded(input_ids)["logits"]
    print("save/load logits max err:", (a - b).abs().max().item())
print("ALL CUDA INTEGRATION CHECKS PASSED")
