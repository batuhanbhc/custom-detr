# Input/data
img_size = 640
batch_size = 4

# Dataset
num_class = 80

# Shared model dimensions
d_model = 256
num_heads = 8
num_levels = 3

# Hybrid encoder
num_enc_layers = 1
pre_norm = False
expansion = 4.0
ffn_dropout = 0.0
msa_dropout = 0.0

# Query selection
top_k = 300
base_anchor_width = 0.05
base_anchor_height = 0.05

# Deformable decoder
num_decoder_layers = 6
k_list = [4, 6, 4]

decoder_expansion = 4.0
decoder_pre_norm = True
decoder_ffn_dropout = 0.0
decoder_msa_dropout = 0.0