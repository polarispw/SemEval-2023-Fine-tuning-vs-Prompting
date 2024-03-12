from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import PreTrainedModel, AutoModel, AutoConfig, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput

from config.args_list import CLSModelArguments


class CLSBert(PreTrainedModel):
    def __init__(self, args: CLSModelArguments):
        super(CLSBert, self).__init__(AutoConfig.from_pretrained(args.model_name_or_path))

        # you can change the attributes init in ModelConfig here before loading the model
        self.name_or_path = args.model_name_or_path
        self.cache_dir = args.cache_dir
        self.max_position_embeddings = args.max_seq_length

        self.num_labels = args.num_labels
        self.problem_type = args.problem_type

        self.base = AutoModel.from_pretrained(self.name_or_path, cache_dir=self.cache_dir)
        self.dropout = nn.Dropout(0.1)
        self.classifiers = nn.ModuleList()
        for i in range(self.num_labels):
            self.classifiers.append(nn.Linear(self.config.hidden_size, 1))

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        # here we sum the last hidden state of all tokens together for cls input
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)  # [batch_size, hidden_size]

        # labels: [batch_size, num_categories]
        use_cl = False
        loss_fct = BCEWithLogitsLoss()
        logits = []

        for i in range(self.num_labels):
            logits.append(self.classifiers[i](pooled_output))  # [batch_size]
        logits = torch.stack(logits, dim=0)  # [num_categories, batch_size]
        logits = logits.view(-1, self.num_labels)  # [batch_size, num_categories]
        loss = loss_fct(logits, labels.float())

        if use_cl:
            batch_size = input_ids.shape[0]
            y_matrix = []
            for i in range(batch_size):
                y_matrix.append([labels[i].T @ labels[j] for j in range(batch_size)])
            y_matrix = torch.tensor(y_matrix).to(logits.device)
            w_matrix = []
            for i in range(batch_size):
                row_sum = torch.sum(y_matrix[i]) + 1e-6
                w_matrix.append([y_matrix[i][j] / row_sum for j in range(batch_size)])
            w_matrix = torch.tensor(w_matrix).to(logits.device)

            sim_matrix = F.cosine_similarity(pooled_output.unsqueeze(1), pooled_output.unsqueeze(0), dim=-1)
            sim_matrix = sim_matrix / 0.05

            for i in range(batch_size):
                sim_matrix[i][i] = 0.0
                loss += -torch.log(torch.sum(w_matrix[i] * torch.exp(sim_matrix[i])) / torch.sum(torch.exp(sim_matrix[i])))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class SIMBert(PreTrainedModel):
    """
    This class is used to build a model that has a BERT-like architecture to do supervised SimCSE pre-training.
    We use base + MLP to get the sentence embedding, inferring from the SimCSE paper.
    """

    def __init__(self, args: CLSModelArguments):
        super(SIMBert, self).__init__(AutoConfig.from_pretrained(args.model_name_or_path))

        # you can change the attributes init in ModelConfig here before loading the model
        self.name_or_path = args.model_name_or_path
        self.cache_dir = args.cache_dir
        self.max_position_embeddings = args.max_seq_length
        self.num_labels = args.num_labels
        self.problem_type = args.problem_type

        self.base = AutoModel.from_pretrained(self.name_or_path, cache_dir=self.cache_dir)

        self.loss = nn.CrossEntropyLoss()
        self.scale = 20.0

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                head_mask: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None, ):
        outputs = self.base(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            output_attentions=output_attentions,
                            output_hidden_states=output_hidden_states,
                            return_dict=return_dict)

        # here we sum the last hidden state of all tokens together as pooled output
        pooled_output = outputs.last_hidden_state.sum(dim=1)

        # labels = torch.tensor([0, 1, 2, 3]).to(pooled_output.device)
        logits, loss = self.simcse_sup_loss(pooled_output, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def simcse_sup_loss(self, output_hidden_states: torch.Tensor, labels: torch.Tensor):
        """
        This function is used to compute the supervised loss for SimCSE.
        We consider the output_hidden_states in a batch with the same label as pos pairs, and others as neg pairs.
        :param output_hidden_states:
        :param labels:
        :return:
        """
        # get pos pairs from input labels
        label2idx = {}
        for idx, label in enumerate(labels):
            if label not in label2idx:
                label2idx[label] = [idx]
            else:
                label2idx[label].append(idx)

        # calculate cosine similarity
        sim = F.cosine_similarity(output_hidden_states.unsqueeze(1), output_hidden_states.unsqueeze(0), dim=-1)
        scaled_sim = sim / self.scale

        pos_probs = torch.where(labels.unsqueeze(1) == labels.unsqueeze(0), scaled_sim, torch.zeros_like(scaled_sim))
        neg_probs = torch.where(labels.unsqueeze(1) != labels.unsqueeze(0), scaled_sim, torch.zeros_like(scaled_sim))
        pos_probs = torch.sub(pos_probs, torch.eye(labels.shape[0], device=scaled_sim.device) / self.scale)

        neg_probs = torch.add(neg_probs, torch.eye(labels.shape[0], device=scaled_sim.device) / self.scale)
        pos_probs = torch.sum(pos_probs, dim=0)
        neg_probs = torch.sum(neg_probs, dim=0)
        pred_probs = torch.stack([pos_probs, neg_probs], dim=0).transpose(0, 1)
        loss = self.loss(pred_probs, torch.zeros_like(labels))
        return pos_probs - neg_probs, loss


if __name__ == "__main__":
    print(torch.cuda.is_available())
    my_model = SIMBert(CLSModelArguments('bert-base-uncased', cache_dir='../.cache')).to('cuda:0')
    my_model.train()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    test_inputs = tokenizer(["Hello, my dog is cute",
                             "Hello, my dog is cute",
                             "Dimension where cosine similarity is computed",
                             "Dimension where cosine similarity is computed"],
                            padding=True,
                            truncation=True,
                            return_tensors="pt").to('cuda')

    loss, sim = my_model(**test_inputs)
    print(loss)
    loss.backward()
    print("finish")

    # l = [[1, 1, 0], [1, 0, 1], [1, 1, 1]]
    # l = torch.tensor(l)
    # batch_size = 3
    # y_matrix = []
    # for i in range(batch_size):
    #     y_matrix.append([l[i].T @ l[j] for j in range(batch_size)])
    # y_matrix = torch.tensor(y_matrix)
    # w_matrix = []
    # for i in range(batch_size):
    #     row_sum = torch.sum(y_matrix[i]) + 1e-6
    #     w_matrix.append([y_matrix[i][j] / row_sum for j in range(batch_size)])
    # w_matrix = torch.tensor(w_matrix)
    # print(y_matrix)
    # print(w_matrix)