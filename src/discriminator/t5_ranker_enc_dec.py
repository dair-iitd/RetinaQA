import torch
from torch import nn
from transformers import T5ForConditionalGeneration


def get_inf_mask(bool_mask):
    return (bool_mask) * -1000000.0

class T5ForCandidateRanking(nn.Module):
    def __init__(self, model_path, target_id):
        super().__init__()
        self.t5_enc_dec = T5ForConditionalGeneration.from_pretrained(model_path,cache_dir = "~/.cache/huggingface/transformers/", local_files_only=False)
        self.loss_fct = torch.nn.CrossEntropyLoss()
        self.target_id = target_id

    def forward(self, model_inputs):

        labels = model_inputs.get("labels",None)
        input_ids = model_inputs.get("model_input")
        attention_masks = model_inputs.get("attention_mask")
        mask = model_inputs.get("mask")

        new_input = {}
        # print(input_ids.shape)
        batch_size = input_ids.size(0)
        sample_size = input_ids.size(1)
        
        if sample_size == 0:
            print("Returning 0 loss !!!")
            return (torch.FloatTensor(0).to("cuda"), {}) if return_outputs else torch.FloatTensor(0).to("cuda")

        new_input["input_ids"] = input_ids.view((batch_size * sample_size,-1))
        new_input["attention_mask"] = attention_masks.view((batch_size * sample_size,-1))
        new_input["decoder_input_ids"] = torch.zeros(len(new_input["input_ids"]), 1).to("cuda").int()

        outputs = self.t5_enc_dec(**new_input, return_dict=True)
        logits = outputs["logits"][:, 0, self.target_id].unsqueeze(1)
        logits = logits.view((batch_size, sample_size))
        logits = logits + get_inf_mask(mask)
        
        loss = None        
        if not labels is None:
            loss = self.loss_fct(logits, labels.view(-1))

        return (loss, outputs)

if __name__ == "__main__":
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("t5-base",cache_dir = "~/.cache/huggingface/transformers/", local_files_only=False)
    t5_cr = T5ForCandidateRanking("t5-base")