import argparse
import os
import sys
import torch
import glob

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', default=None, type=str, help='Input Megatron Checkpoint folder')
    parser.add_argument('--output_folder', default=None, type=str, help='Output SAT checkpoint folder')
    parser.add_argument('--target_tp', default=1, type=int, help='Target TP degree')
    parser.add_argument('--layers', default=1, type=int, help='layer numbers.')
    args = parser.parse_args()
    print(f'args = {args}')
    return args



def megatron_to_sat(checkpoint_name, target, num_layer):
    # checkpoint_name = '/thudm/workspace/hanyu/SwissArmyTransformer-old/data/global_step10400/iter_0010400/10400/mp_rank_00_model_states.pt'
    sd = torch.load(checkpoint_name, map_location='cpu')
    new_sd = {}
    new_sd['transformer.word_embeddings.weight'] = sd['model']['word_embeddings_for_head']['weight']

    encoder = sd['model']['language_model']['encoder']
    # num_layer = 30
    for i in range(num_layer):
        new_sd['transformer.layers.' + str(i) +'.input_layernorm.weight'] = encoder['layers.' + str(i) + '.input_layernorm.weight']
        new_sd['transformer.layers.' + str(i) +'.input_layernorm.bias'] = encoder['layers.' + str(i) + '.input_layernorm.bias']

        new_sd['transformer.layers.' + str(i) + '.attention.query_key_value.weight'] = encoder['layers.' + str(i) + '.self_attention.query_key_value.weight']
        new_sd['transformer.layers.' + str(i) + '.attention.query_key_value.bias'] = encoder['layers.' + str(i) + '.self_attention.query_key_value.bias']

        new_sd['transformer.layers.' + str(i) + '.attention.dense.weight'] = encoder['layers.' + str(i) + '.self_attention.dense.weight']
        new_sd['transformer.layers.' + str(i) + '.attention.dense.bias'] = encoder['layers.' + str(i) + '.self_attention.dense.bias']

        new_sd['transformer.layers.' + str(i) + '.post_attention_layernorm.weight'] = encoder['layers.' + str(i) + '.post_attention_layernorm.weight']
        new_sd['transformer.layers.' + str(i) + '.post_attention_layernorm.bias'] = encoder['layers.' + str(i) + '.post_attention_layernorm.bias']

        new_sd['transformer.layers.' + str(i) + '.mlp.dense_h_to_4h.weight'] = encoder['layers.' + str(i) + '.mlp.dense_h_to_4h.weight']
        new_sd['transformer.layers.' + str(i) + '.mlp.dense_h_to_4h.bias'] =  encoder['layers.' + str(i) + '.mlp.dense_h_to_4h.bias']

        new_sd['transformer.layers.' + str(i) + '.mlp.dense_4h_to_h.weight'] =  encoder['layers.' + str(i) + '.mlp.dense_4h_to_h.weight']
        new_sd['transformer.layers.' + str(i) + '.mlp.dense_4h_to_h.bias'] =  encoder['layers.' + str(i) + '.mlp.dense_4h_to_h.bias']
        print(f"layer: {i} attention.dense.weight: {encoder['layers.' + str(i) + '.self_attention.dense.weight'].size()}")

    new_sd['transformer.final_layernorm.weight'] = encoder['final_layernorm.weight']
    new_sd['transformer.final_layernorm.bias'] = encoder['final_layernorm.bias']
    new_sd = { 'module': new_sd }
    # target = open('/thudm/workspace/hanyu/SwissArmyTransformer/data/global_step10400/iter_0010400/10400/mp_rank_00_model_states.pt', 'w')
    torch.save(new_sd, target)

# ${INPUT_FOLDER} ${OUTPUT_FOLDER} ${NUM_LAYERS} ${TARGET_TP}"
def main():
    # dir_path = str(sys.argv[1])
    # target_dir = str(sys.argv[2])
    # num_layer=int(sys.argv[3])
    # MP=int(sys.argv[4])

    args = parse_arguments()
    print(args)
    print(f'Converting Megatron checkpoint in {args.input_folder} to SAT checkpoint in {args.output_folder}')

    iter_path = os.path.join(args.input_folder, 'latest_checkpointed_iteration.txt')
    iteration = open(iter_path).read()
    
    print(f"iteration: {iteration} ...")
    
    iter_dir = glob.glob(os.path.join(args.input_folder, "iter*"))[0]  
    new_iter_dir = os.path.join(args.output_folder, 'iter_' + iteration)
    # iter_dir = os.path.join(args.input_folder, 'iter_' + iteration)

    os.makedirs(new_iter_dir, exist_ok=True)
    new_iter_path = os.path.join(new_iter_dir, 'latest')

    print(f"iter_path: {iter_dir}, new_iter_path: {new_iter_dir}")
    os.system('cp {} {}'.format(iter_path, new_iter_path))

    new_model_dir = os.path.join(new_iter_dir, iteration)
    os.makedirs(new_model_dir, exist_ok=True)

    for i in range(args.target_tp):
        model_dir = os.path.join(iter_dir, 'mp_rank_' + f"{i:02d}")
        model_path = os.path.join(model_dir, 'model_optim_rng.pt')
        new_model_path = os.path.join(new_model_dir, 'mp_rank_' + f"{i:02d}" + '_model_states.pt')
        print(f"model_path: {model_path}, new_model_path: {new_model_path}")
        megatron_to_sat(model_path, new_model_path, args.layers)
    
if __name__ == "__main__":
    main()
