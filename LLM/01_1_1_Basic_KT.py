import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

model_name = "K-intelligence/Midm-2.0-Mini-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
generation_config = GenerationConfig.from_pretrained(model_name)

prompt = "KT에 대해 소개해줘"

# message for inference
messages = [
    {"role": "system", 
     "content": "Mi:dm(믿:음)은 KT에서 개발한 AI 기반 어시스턴트이다."},
    {"role": "user", "content": prompt}
]

input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,  # 딕셔너리 형태로 반환
    return_tensors="pt"
)

# 2. 딕셔너리 안에 들어있는 모든 텐서들을 한꺼번에 GPU로 보냅니다.
inputs = {k: v.to("cuda") for k, v in input_ids.items()}

output = model.generate(
    **inputs,  # input_ids와 attention_mask가 자동으로 들어갑니다.
    max_new_tokens=128,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id # 패딩 경고 방지
)
print(tokenizer.decode(output[0], skip_special_tokens=True))