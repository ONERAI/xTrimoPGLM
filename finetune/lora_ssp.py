from transformers import AdamW, get_linear_schedule_with_warmup, set_seed, AutoModelForCausalLM, AutoTokenizer,AutoTokenizer, AutoConfig, AutoModelForMaskedLM, AutoModel, AutoModelForSequenceClassification, AutoModelForTokenClassification
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import numpy as np
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = 'biomap-research/xtrimopglm-3b-mlm'
tokenizer  = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)

def eval_mode(model, eval_dataloader):
    # put model in evaluation mode


    def calculate_class_accuracies(predict_score, true_labels, class_correct, class_total):
        predicted_labels = predict_score.argmax(dim=-1)
        for true, pred in zip(true_labels, predicted_labels):
            class_total[true.item()] += 1
            if true == pred:
                class_correct[true.item()] += 1

        return class_correct, class_total

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    metric = [[0] for _ in range(3)]
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.inference_mode():
            seqs, label = batch
            output = tokenizer(seqs, add_special_tokens=True, return_tensors='pt')
            input_ids = output["input_ids"].to(device)
            attention_mask = output["attention_mask"].cuda().to(device)
            output_logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=torch.tensor(label, dtype=torch.long).to(device))
            tmp_eval_loss = output_logits.loss
            logits = output_logits.logits.view(-1, 3)
            
            eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1

            ret = calculate_class_accuracies(logits.detach().cpu(), torch.tensor(label, dtype=torch.long).view(-1), class_correct, class_total)



    return class_correct, class_total


def lora_main(task_modality):

    # step-1: initilize the model
    config = AutoConfig.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
    config.task_modality = task_modality
    config.activation_func = torch.nn.functional.relu
    config.inter_hidden_size = [128, 3]
    config.num_labels = 3
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
    train_dataset = YourDataset(split ="Train")
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=1)

    eval_dataset = YourDataset(split ="test")
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=1)


    # step-3: define the training and validation loop
    seed = 1234
    set_seed(seed)
    model.to(device)

    num_train_epochs = 15

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.05 * (len(train_dataloader) * num_train_epochs),
        num_training_steps=(len(train_dataloader) * num_train_epochs),
    )
    global_step = 0
    grad_acc_step = 8
    eval_interval = 1000

    t_total = len(train_dataloader) * num_train_epochs  # total number of training steps


    # eval_mode(model, eval_dataloader)
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
                print(f"epoch: {epoch} loss after {global_step} steps: {loss.item()} lr : {current_lr}", flush=True)
            
            loss = loss/grad_acc_step
            # backward pass to get the gradients
            loss.backward()

            # update
            if global_step > 0 and global_step % grad_acc_step == 0:
                optimizer.step()
                optimizer.zero_grad()
            global_step += 1
            lr_scheduler.step()
            if global_step > 0 and global_step % eval_interval == 0:
                class_correct, class_total = eval_mode(model, eval_dataloader)
                total_correct, total_num = 0, 0
                class_accuracies = {}
                for each in class_total:
                    tmp_correct = class_correct[each]
                    tmp_num = class_total[each]
                    tmp_acc = tmp_correct/ tmp_num if tmp_num > 0 else 0.0
                    class_accuracies[each] = tmp_acc
                    total_correct += tmp_correct
                    total_num += tmp_num
                for cls, accuracy in class_accuracies.items():
                    print(f"Overall accuracy for class {cls}: {accuracy * 100:.2f}%", flush=True)
                print(f"Overall ACC: {total_correct/total_num:.2f}", flush=True)

if __name__ == "__main__":
    lora_main('token')










