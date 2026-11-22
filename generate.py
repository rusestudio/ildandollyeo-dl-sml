import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from gpt_model import GPTConfig, GPTModel

device = "cuda"
ckpt = torch.load(
    "checkpoints/d768L12_lr0.0006_do0_wd0.1_674244st/ckpt.pt", map_location=device
)
config = GPTConfig(**ckpt["gpt_config"])
model = GPTModel(config).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

tokenizer = Tokenizer.from_file("data/tokenizer/tokenizer.json")
eos_id = tokenizer.token_to_id("[EOS]")


@torch.no_grad()
def generate(prompt, max_new_tokens=150, temperature=0.3, top_k=50):
    idx = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=False).ids], device=device
    )
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -config.seq_len :]  # 학습 context 256을 넘지 않게 자르기
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(idx_cond)[:, -1, :]
        logits = logits.float() / temperature
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        if idx_next.item() == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return tokenizer.decode(idx[0].tolist())


prompts = [
    # 설명문 (학습 데이터와 가장 비슷한 스타일)
    "Photosynthesis is the process by which",
    "The main causes of World War I were",
    # 사실 지식
    "The capital of France is",
    "Water boils at a temperature of",
    # 형식 따라가기
    "Here are five tips for studying effectively:\n1.",
    "Q: Why is the sky blue?\nA:",
    # 이야기
    "Once upon a time, in a small village,",
    # 숫자 (한 자리씩 분리한 tokenizer 효과 확인)
    "If you add 25 and 17, the result is",
]

torch.manual_seed(42)
for p in prompts:
    print("=" * 80)
    print(generate(p))


# for t in [0.3, 0.8, 1.2]:
#     print(f"--- temperature {t}")
#     print(generate("Photosynthesis is the process by which", temperature=t))
