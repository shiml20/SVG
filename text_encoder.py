import torch
from transformers import T5EncoderModel, T5Tokenizer

def test_text_encoder(model_path="PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers", 
                     device="cuda", torch_type=torch.float16):
    # 初始化设备
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    try:
        # 加载tokenizer和text encoder
        tokenizer = T5Tokenizer.from_pretrained(model_path, subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained(
            model_path, 
            subfolder="text_encoder",
            torch_dtype=torch_type
        ).to(device)
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        return

    # 验证模型结构
    print("="*50)
    print("模型结构验证:")
    print(f"Tokenizer 类型: {type(tokenizer).__name__}")
    print(f"TextEncoder 类型: {type(text_encoder).__name__}")
    print(f"模型参数数量: {sum(p.numel() for p in text_encoder.parameters()):,}")
    print("模型结构示例:")
    print(text_encoder)

    # 测试前向传播
    test_texts = [
        "a photo of a cat",
        "a photo of a dog",
        "a photo of valley",
    ]
    import json


    # 读取 class_to_idx.json 文件
    with open('/ytech_m2v2_hdd/sml/DiffMoE_research_local/class_to_idx.json', 'r') as file:
        class_to_idx = json.load(file)

    # 读取描述文本的 JSON 文件
    with open('/ytech_m2v2_hdd/sml/DiffMoE_research_local/class_to_name_t2i.json', 'r') as file:
        data = json.load(file)

    # 只保留 class_to_idx 中存在的类的描述文本
    test_texts = [data[key] for key in data if key in class_to_idx]

    # 打印测试文本
    print(test_texts)
    print(len(test_texts))


    # Tokenize输入
    inputs = tokenizer(
        test_texts,
        max_length=120,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    # 执行前向传播
    with torch.no_grad():
        outputs = text_encoder(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask
        )

    # 分析输出
    print("\n" + "="*50)
    print("输出验证:")
    print(f"输入文本数量: {len(test_texts)}")
    print(f"输入IDs形状: {inputs.input_ids.shape}")
    print(f"注意力掩码形状: {inputs.attention_mask.shape}")
    print(f"输出隐藏状态形状: {outputs.last_hidden_state.shape}")
    
    text_feature = outputs.last_hidden_state
    torch.save(text_feature, "text_feature.pt")
    # 输出示例
    print("\n示例输出 (第一个文本的最后5个token的嵌入):")
    # for d in outputs.last_hidden_state[:]:
        # print(torch.mean(d))

    # print(outputs.last_hidden_state[0, -5:, :5])
    # print(outputs.last_hidden_state[1, -5:, :5])
    # print(outputs.last_hidden_state[2, -5:, :5])

    # 清理显存
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # 测试参数设置
    test_config = {
        "model_path": "PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
        "device": "cuda",
        "torch_type": torch.float16  # 测试半精度推理
    }
    
    test_text_encoder(**test_config)