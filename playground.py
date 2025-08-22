import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import torch.nn.functional as F
from tqdm import tqdm
CACHE_DIR = "/home/yl535/rds/hpc-work/cache"

# -----------------------
# 配置
# -----------------------
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
BATCH_SIZE = 20
LR = 5e-5
EPOCHS = 1
BETA = 0.1
MAX_PROMPT_LEN = 512
MAX_RESP_LEN = 512
PAD_TO_MULTIPLE_OF = 8  # 便于 tensor core 加速
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# CACHE_DIR = "/home/yl535/rds/hpc-work/cache"
CACHE_DIR = None

# -----------------------
# 加载模型和 tokenizer
# -----------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR, padding_side="right", truncation=True, max_length=MAX_PROMPT_LEN)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR).to(DEVICE)

# reference policy，冻结参数
ref_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR).to(DEVICE)
ref_model.eval()  # 不训练

# -----------------------
# 加载偏好数据集
# 数据集格式: { "prompt": ..., "chosen": ..., "rejected": ... }
# -----------------------
dataset = load_dataset("Dahoas/rm-static", cache_dir=CACHE_DIR)["train"]

# def tokenize_pair(example):
#     prompt_ids = tokenizer(example["prompt"], return_tensors="pt").input_ids
#     chosen_ids = tokenizer(example["chosen"], return_tensors="pt").input_ids
#     rejected_ids = tokenizer(example["rejected"], return_tensors="pt").input_ids
#     return {
#         "prompt_ids": prompt_ids,
#         "chosen_ids": chosen_ids,
#         "rejected_ids": rejected_ids
#     }

# dataset = dataset.map(tokenize_pair)

# -----------------------
# DataLoader
# -----------------------
def apply_chat_template(prompt, response, tokenizer):
    messages = [
        {"role": "user", "content": prompt.strip()},
        {"role": "assistant", "content": response.strip()}
    ]
    msg = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)#.strip()  # There's an automatic /n at the end
    return msg


def collate_fn(batch):
    prompt_chosen_texts = []
    prompt_rejected_texts = []
    prompt_only_texts = []
    prompt_offset_lens = []
    for item in batch:
        prompt_chosen_texts.append(apply_chat_template(item['prompt'], item['chosen'], tokenizer))
        prompt_rejected_texts.append(apply_chat_template(item['prompt'], item['rejected'], tokenizer))
        prompt_only = apply_chat_template(item['prompt'], '', tokenizer)
        prompt_only_len = tokenizer(prompt_only, return_tensors="pt", padding=False, truncation=True).input_ids.shape[1]
        prompt_offset_lens.append(prompt_only_len-2)

    prompt_chosen_encoded = tokenizer(prompt_chosen_texts, return_tensors="pt", padding=True, truncation=True)
    prompt_rejected_encoded = tokenizer(prompt_rejected_texts, return_tensors="pt", padding=True, truncation=True)

    return {
        'prompt_chosen_ids': prompt_chosen_encoded.input_ids,
        'prompt_chosen_attention_mask': prompt_chosen_encoded.attention_mask,
        'prompt_rejected_ids': prompt_rejected_encoded.input_ids,
        'prompt_rejected_attention_mask': prompt_rejected_encoded.attention_mask,
        'prompt_offset_lens': torch.tensor(prompt_offset_lens, dtype=torch.int32)}


loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)



def get_sequence_logprobs(ids, logits, attention_mask, prompt_offset_lens):
    logprobs = F.log_softmax(logits[:, :-1, :].contiguous(), dim=-1)
    logprobs_selected = torch.gather(logprobs, dim=-1, index=ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    logprobs_selected = logprobs_selected * attention_mask[:, :-1]
    seq_logprobs = []
    for i, prompt_offset_len in enumerate(prompt_offset_lens):
        seq_logprob = logprobs_selected[i, prompt_offset_len:].sum()
        seq_logprobs.append(seq_logprob)
    seq_logprobs = torch.stack(seq_logprobs)
    return seq_logprobs



for batch in loader:
    prompt_chosen_batch = batch['prompt_chosen_ids'].to(DEVICE)
    prompt_chosen_batch_attention_mask = batch['prompt_chosen_attention_mask'].to(DEVICE)
    prompt_rejected_batch = batch['prompt_rejected_ids'].to(DEVICE)
    prompt_rejected_batch_attention_mask = batch['prompt_rejected_attention_mask'].to(DEVICE)
    prompt_offset_lens_batch = batch['prompt_offset_lens'].to(DEVICE)
    
    # print(tokenizer.decode(prompt_chosen_batch[1]))
    # print(tokenizer.decode(prompt_chosen_batch[1][prompt_offset_lens_batch[1]:]))

    # 计算损失
    outputs_chosen = model(prompt_chosen_batch)
    output_rejceted = model(prompt_rejected_batch)

    chosen_seq_logprob = get_sequence_logprobs(prompt_chosen_batch, outputs_chosen.logits, prompt_chosen_batch_attention_mask, prompt_offset_lens_batch)
    rejected_seq_logprob = get_sequence_logprobs(prompt_rejected_batch, output_rejceted.logits, prompt_rejected_batch_attention_mask, prompt_offset_lens_batch)

    with torch.no_grad():
        outputs_chosen_ref = ref_model(prompt_chosen_batch)
        output_rejceted_ref = ref_model(prompt_rejected_batch)
        chosen_ref_seq_logprob = get_sequence_logprobs(prompt_chosen_batch, outputs_chosen_ref.logits, prompt_chosen_batch_attention_mask, prompt_offset_lens_batch)
        rejected_ref_seq_logprob = get_sequence_logprobs(prompt_rejected_batch, output_rejceted_ref.logits, prompt_rejected_batch_attention_mask, prompt_offset_lens_batch)

    log_ratio_chosen = chosen_seq_logprob - chosen_ref_seq_logprob
    log_ratio_rejected = rejected_seq_logprob - rejected_ref_seq_logprob
    loss = -torch.log(torch.sigmoid((log_ratio_chosen - log_ratio_rejected)/BETA)).mean()
    print(loss.item())

    assert False


# -----------------------
# 优化器
# -----------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)



# -----------------------
# 训练循环
# -----------------------
for epoch in range(EPOCHS):
    for batch in tqdm(loader):
        optimizer.zero_grad()
        losses = []

        prompt_ids = batch['prompt_ids'].to(DEVICE)
        chosen_ids = batch['chosen_ids'].to(DEVICE)
        rejected_ids = batch['rejected_ids'].to(DEVICE)

        chosen_lo
        # print('-------------------')
        # print(prompt_ids.shape, chosen_ids.shape, rejected_ids.shape)
        # assert False

        # prompt_ids = torch.tensor([item['prompt_ids'] for item in batch]).to(DEVICE)
        # chosen_ids = torch.tensor([item['chosen_ids'] for item in batch]).to(DEVICE)
        # rejected_ids = torch.tensor([item['rejected_ids'] for item in batch]).to(DEVICE)


        for item in batch:
            prompt_ids = torch.tensor(item["prompt_ids"]).to(DEVICE)
            chosen_ids = torch.tensor(item["chosen_ids"]).to(DEVICE)
            rejected_ids = torch.tensor(item["rejected_ids"]).to(DEVICE)

            print('--------------------------------')
            print(prompt_ids.shape, chosen_ids.shape, rejected_ids.shape)
            # 模型 log probs
            chosen_logits = model(input_ids=prompt_ids.unsqueeze(0), labels=chosen_ids.unsqueeze(0)).logits
            rejected_logits = model(input_ids=prompt_ids.unsqueeze(0), labels=rejected_ids.unsqueeze(0)).logits

            # reference log probs
            with torch.no_grad():
                chosen_logits_ref = ref_model(input_ids=prompt_ids.unsqueeze(0), labels=chosen_ids.unsqueeze(0)).logits
                rejected_logits_ref = ref_model(input_ids=prompt_ids.unsqueeze(0), labels=rejected_ids.unsqueeze(0)).logits

            # log_softmax
            chosen_log_probs = F.log_softmax(chosen_logits, dim=-1)
            rejected_log_probs = F.log_softmax(rejected_logits, dim=-1)
            chosen_log_probs_ref = F.log_softmax(chosen_logits_ref, dim=-1)
            rejected_log_probs_ref = F.log_softmax(rejected_logits_ref, dim=-1)

            # 聚合序列 log prob
            chosen_seq_log_prob = chosen_log_probs.gather(2, chosen_ids.unsqueeze(0).unsqueeze(-1)).squeeze(-1).sum()
            rejected_seq_log_prob = rejected_log_probs.gather(2, rejected_ids.unsqueeze(0).unsqueeze(-1)).squeeze(-1).sum()
            chosen_seq_log_prob_ref = chosen_log_probs_ref.gather(2, chosen_ids.unsqueeze(0).unsqueeze(-1)).squeeze(-1).sum()
            rejected_seq_log_prob_ref = rejected_log_probs_ref.gather(2, rejected_ids.unsqueeze(0).unsqueeze(-1)).squeeze(-1).sum()

            # DPO loss
            log_ratio_chosen = chosen_seq_log_prob - chosen_seq_log_prob_ref
            log_ratio_rejected = rejected_seq_log_prob - rejected_seq_log_prob_ref
            loss = -torch.log(torch.sigmoid((log_ratio_chosen - log_ratio_rejected)/BETA))
            losses.append(loss)

        batch_loss = torch.stack(losses).mean()
        batch_loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} Loss: {batch_loss.item():.4f}")
