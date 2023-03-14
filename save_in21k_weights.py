import torch
import requests
import src.deit as deit
from transformers import ViTModel

vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
deit_model = deit.__dict__['deit_base_with_pooling']()
save_path = "pretrained/deit/deit_base_patch16_in21k_full.pth"

# -- Function to find corresponding param name
def fetch_nparam(deit_param_name):
    '''
    Fetch corresponding named parameter in ViT given one from DEIT's
    '''
    deit_words = deit_param_name.split('.')
    vit_words = []
    if deit_words[1] == "cls_token":
        # Case 1: cls token
        vit_words = ["embeddings", "cls_token"]    
    elif deit_words[1] == "pos_embed":
        # Case 2: pos embed
        vit_words = ["embeddings", "position_embeddings"]    
    elif deit_words[1] == "patch_embed":
        # Case 3: patch embed
        vit_words = ["embeddings", "patch_embeddings", "projection", deit_words[-1]]    
    elif deit_words[1] == "blocks":
        # Case 4: blocks
        vit_words = ["encoder", "layer", deit_words[2]]        
        if deit_words[3] == "norm1":
            # Case 4.1: norm1
            vit_words += ["layernorm_before", deit_words[-1]]        
        elif deit_words[3] == "attn":
            # Case 4.2: attn 
            vit_words += ["attention"]
            if deit_words[4] == "qkv":
                # Case 4.2.1: qkv
                vit_words += ["attention", "qkv", deit_words[-1]]            
            elif deit_words[4] == "proj":
                # Case 4.2.2: proj
                vit_words += ["output", "dense", deit_words[-1]]        
        elif deit_words[3] == "norm2":
            # Case 4.3: norm2
            vit_words += ["layernorm_after", deit_words[-1]]        
        else:
            # Case 4.4: mlp
            if deit_words[4] == "fc1":
                # Case 4.4.1: fc1
                vit_words += ["intermediate", "dense", deit_words[-1]]
            elif deit_words[4] == "fc2": 
                # Case 4.4.2: fc2
                vit_words += ["output", "dense", deit_words[-1]]
    elif deit_words[1] == "norm":
        # Case 5: norm
        vit_words = ["layernorm", deit_words[-1]]
    elif deit_words[0] == "dense":
        # Case 6: dense
        vit_words = ["pooler", "dense", deit_words[-1]]
    else:
        raise Exception(f"{deit_param_name} is not in DEIT's named parameters.")
        exit(0)
    vit_param_name = ".".join(vit_words)
    return vit_param_name

# -- Transfer ViT's weights to DEIT
vit_params_dict = {}
for vit_param_name, vit_param in vit_model.named_parameters(): 
    if vit_param.requires_grad == False: continue
    vit_params_dict[vit_param_name] = vit_param

deit_params_dict = {}
with torch.no_grad():
    for deit_param_name, deit_param in deit_model.named_parameters(): 
        if deit_param.requires_grad == False: continue
        deit_params_dict[deit_param_name] = deit_param.detach().clone()
        vit_param_name = fetch_nparam(deit_param_name)
        if "qkv" in vit_param_name:
            vit_words = vit_param_name.split('.')
            # query
            query_words = vit_words.copy(); query_words[-2] = "query"
            query_name = ".".join(query_words); query_param = vit_params_dict[query_name]
            # key
            key_words = vit_words.copy(); key_words[-2] = "key"
            key_name = ".".join(key_words); key_param = vit_params_dict[key_name]
            # value
            value_words = vit_words.copy(); value_words[-2] = "value"
            value_name = ".".join(value_words); value_param = vit_params_dict[value_name]
            # qkv
            qkv_param = torch.cat([query_param, key_param, value_param], dim=0)
            deit_param.copy_(qkv_param)
        else:
            vit_param = vit_params_dict[vit_param_name]
            deit_param.copy_(vit_param)

# -- Save state dictionary
torch.save(deit_model.state_dict(), save_path)

# -- Cross-check parameters
try:
    deit_model.load_state_dict(torch.load(save_path), strict=True)
    print("Successfully saved and loaded all ViT's weights in DEIT!")
except Exception:
    print("Unable to load some of ViT's weights in DEIT :(")
    exit(0)