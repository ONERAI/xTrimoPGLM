from transformers import AdamW, get_linear_schedule_with_warmup, set_seed, AutoModelForCausalLM, AutoTokenizer,AutoTokenizer, AutoConfig, AutoModelForMaskedLM, AutoModel, AutoModelForSequenceClassification, AutoModelForTokenClassification
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'biomap-research/xtrimopglm-3b-mlm'
tokenizer  = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
topn_list = [1, 2, 5, 10]

# top L/5 ACC
def metric_eval(pred_y, y, inds, ls, lens):
    tests = []
    t_y = []
    rs = []
    for idx in inds:
        row = idx // lens
        col = idx % lens
        if row >= col:
            continue
        if abs(row - col) <= 6:
            continue
        p = pred_y[idx]
        gt = y[idx]
        tests.append((p,gt))
        if len(tests)>=ls:
            break
    cnt = 0
    for p, gt in tests:
        if gt == 1:
            cnt += 1
    return cnt, ls, cnt/ls 

def eval_mode(model, eval_dataloader):
    acc_sum = [0. for _ in topn_list]
    cnt = 0
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            seqs, label = batch
            output = tokenizer(seqs, add_special_tokens=True, return_tensors='pt')
            input_ids = output["input_ids"].to(device)
            attention_mask = output["attention_mask"].cuda().to(device)
            output_logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=torch.tensor(label, dtype=torch.long).to(device))
            tmp_eval_loss = output_logits.loss
            logits = output_logits.logits.view(-1, 2)[...,-1]
            label = label.view(-1)
            logits = logits.view(-1)
            indices = torch.argsort(-logits)
            L = input_ids.size()[-1]
            for j in range(len(topn_list)):
                _, _, acc = metric_eval(logits, label, indices, L//(topn_list[j]), L)
                acc_sum[j] += acc
            cnt += 1
    return acc_sum, cnt




def lora_main(task_modality):

    # step-1: initilize the model
    config = AutoConfig.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    config.task_modality = task_modality
    config.activation_func = torch.nn.functional.relu
    config.inter_hidden_size = [128, 2]
    config.num_labels = 2
    config.bias = True
    config.add_special_tokens=True

    # model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True)


    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, 
        target_modules=['k_proj', 'q_proj', 'v_proj', 'out_proj', 'query_key_value', 'self_attention.dense', 'dense_h_to_4h', 'dense_4h_to_h'],
        inference_mode=False, 
        r=16, 
        lora_alpha=16, 
        lora_dropout=0.0
    )

    model = AutoModelForTokenClassification.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16, trust_remote_code=True)
    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    # step-2: initilize the train test valid dataset
    train_dataset = YourDataset(split ="train")
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=1)

    eval_dataset = YourDataset(split ="valid")
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=1)

    test_dataset = YourDataset(split ="test")
    test_sampler = SequentialSampler(eval_dataset)
    test_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=1)


    # step-3: define the training and validation loop
    seed = 1234
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    num_train_epochs = 15
    global_step = 0
    eval_interval = 1000
    test_interval = 1000



    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.05 * (len(train_dataloader) * num_train_epochs),
        num_training_steps=(len(train_dataloader) * num_train_epochs),
    )

    t_total = len(train_dataloader) * num_train_epochs  # total number of training steps


    # put the model in training mode
    model.train()
    for epoch in range(num_train_epochs):
        for batch in tqdm(train_dataloader, desc="Training"):
            seqs, label = batch
            output = tokenizer(seqs, add_special_tokens=True, return_tensors='pt')
            input_ids = output["input_ids"].to(device)
            attention_mask = output["attention_mask"].cuda().to(device)
            output_logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=torch.tensor(label, dtype=torch.long).to(device))
            loss = output_logits.loss
            if global_step % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Loss after {global_step} steps: {loss.item()} lr : {current_lr}", flush=True)

            # backward pass to get the gradients
            loss.backward()

            # update
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            if global_step > 0 and global_step % eval_interval == 0:
                acc_sum, eval_cnt = eval_mode(model, eval_dataloader)
                f_string = f"validation epoch: {epoch} global_step: {global_step} number: {len(eval_dataloader)} "
                for i in range(len(acc_sum)):
                    f_string += "PrecL/{}={:.3f}/{}={:.6f} ".format(int(topn_list[i]), acc_sum[i], eval_cnt, acc_sum[i]/eval_cnt)
                print(f_string)

            if global_step > 0 and global_step % test_interval == 0:
                acc_sum, eval_cnt = eval_mode(model, test_dataloader)
                f_string = f"test epoch: {epoch} global_step: {global_step} number: {len(test_dataloader)} "
                for i in range(len(acc_sum)):
                    f_string += "PrecL/{}={:.3f}/{}={:.6f} ".format(int(topn_list[i]), acc_sum[i], eval_cnt, acc_sum[i]/eval_cnt)
                print(f_string)



if __name__ == "__main__":
    lora_main('pair')