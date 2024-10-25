import torch
import torch.nn as nn
from torch import Tensor


class ContrastiveLossWithAttention(nn.Module):
    """
    Contrastive loss with attention between two permutations,
    as described in "Contrastive learning for supervised graph matching"
    by Ratnayaka et al. (2023).
    """
    def __init__(self):
        super(ContrastiveLossWithAttention, self).__init__()
    
    def forward(self, pred_dsmat: Tensor, gt_perm: Tensor, src_ns: Tensor, tgt_ns: Tensor, beta_value, mask=False) -> Tensor:
        
        batch_num = pred_dsmat.shape[0]
      
        pred_dsmat = torch.clamp(pred_dsmat, min=0.0, max=1.0)
        gt_predicted_values = torch.mul(pred_dsmat, gt_perm)
        
        beta = torch.full_like(pred_dsmat, beta_value)
        
        column_gt_values = torch.matmul(torch.ones_like(gt_predicted_values), gt_predicted_values)
        column_gt_values_minus_beta = column_gt_values-beta
        gt_available_columns = torch.matmul(torch.ones_like(gt_predicted_values), gt_perm)

        attention_tgt = torch.mul(torch.ge(pred_dsmat,column_gt_values_minus_beta).float(), gt_available_columns)
        attention_tgt_without_gt = attention_tgt - gt_perm
        attention_tgt_predicted_values_without_gt = torch.mul(attention_tgt_without_gt, pred_dsmat)  
        attention_tgt_negatives_selected = attention_tgt_predicted_values_without_gt
        
        row_gt_values = torch.matmul(gt_predicted_values, torch.ones_like(gt_predicted_values))
        row_gt_values_minus_beta = row_gt_values - beta
        gt_available_rows = torch.matmul(gt_perm, torch.ones_like(gt_predicted_values))

        attention_src = torch.mul(torch.ge(pred_dsmat, row_gt_values_minus_beta).float(), gt_available_rows)
        attention_src_without_gt = attention_src - gt_perm
        attention_src_predicted_values_without_gt = torch.mul(attention_src_without_gt, pred_dsmat)
        attention_src_negatives_selected = attention_src_predicted_values_without_gt

        def calculateLoss(
            gt_perm,
            gt_predicted_values,
            attention_src_negatives_selected,
            attention_tgt_negatives_selected,
            mask=False
        ):
                    
            gt_indices = (gt_perm == 1).nonzero(as_tuple=True)
            corresponding_target_indices = gt_indices[1]
            corresponding_source_indices = gt_indices[0]
            
            attention_src_negatives_selected_squared = torch.square(attention_src_negatives_selected)
            attention_tgt_negatives_selected_squared = torch.square(attention_tgt_negatives_selected)
            gt_predicted_values_squared = torch.square(gt_predicted_values)

            src_negative_sum = torch.sum(attention_src_negatives_selected_squared,1)
            src_positive_sum = torch.sum(gt_predicted_values_squared,1)
            tgt_negative_sum = torch.sum(attention_tgt_negatives_selected_squared,0)

            corresponding_tgt_negative_sum = torch.index_select(tgt_negative_sum, 0, corresponding_target_indices)
            
            if mask is True:
                src_negative_sum = torch.index_select(src_negative_sum, 0, corresponding_source_indices) 
                src_positive_sum = torch.index_select(src_positive_sum, 0, corresponding_source_indices) 

            overall_negative_sum = src_negative_sum + corresponding_tgt_negative_sum 
            
            denominator = 1 + overall_negative_sum
            probability = src_positive_sum/denominator
            elementwise_loss = -0.5 * torch.log(probability)
            
            loss = torch.sum(elementwise_loss)
            return loss
            
        loss = torch.tensor(0.).to(pred_dsmat.device)
        n_sum = torch.zeros_like(loss)

        for b in range(batch_num):
            loss += calculateLoss(
                gt_perm[b, :src_ns[b], :tgt_ns[b]],
                gt_predicted_values[b, :src_ns[b], :tgt_ns[b]],
                attention_src_negatives_selected[b, :src_ns[b], :tgt_ns[b]],
                attention_tgt_negatives_selected[b, :src_ns[b], :tgt_ns[b]],
                mask=mask
            )

            n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)
         
        return loss/n_sum
    

class PaleMappingLoss(object):
    def __init__(self):
        self.loss_fn = self._euclidean_loss

    def loss(self, inputs1, inputs2):
        return self.loss_fn(inputs1, inputs2)

    def _euclidean_loss(self, inputs1, inputs2):
        sub = inputs2 - inputs1
        square_sub = sub**2
        loss = torch.sum(square_sub)        
        return loss
    
    
class PaleEmbeddingLoss(object):
    def __init__(self, loss_fn='xent', neg_sample_weights=1.0):
        """
        Basic class that applies skip-gram-like loss
        (i.e., dot product of node+target and node and negative samples)
        Args:
            bilinear_weights: use a bilinear weight for affinity calculation: u^T A v. If set to
                false, it is assumed that input dimensions are the same and the affinity will be
                based on dot product.
        """
        self.neg_sample_weights = neg_sample_weights
        self.output_dim = 1
        if loss_fn == 'xent':
            self.loss_fn = self._xent_loss
        else:
            print("Not implemented yet.")


    def loss(self, inputs1, inputs2, neg_samples):
        """ negative sampling loss.
        Args:
            neg_samples: tensor of shape [num_neg_samples x input_dim2]. Negative samples for all
            inputs in batch inputs1.
        """
        return self.loss_fn(inputs1, inputs2, neg_samples)

    def affinity(self, inputs1, inputs2):
        """ Affinity score between batch of inputs1 and inputs2.
        Args:
            inputs1: tensor of shape [n_batch_edges x feature_size].
        """
        # shape: [n_batch_edges, input_dim1]
        result = torch.sum(inputs1 * inputs2, dim=1) # shape: (n_batch_edges,)
        return result

    def neg_cost(self, inputs1, neg_samples):
        """ For each input in batch, compute the sum of its affinity to negative samples.

        Returns:
            Tensor of shape [n_batch_edges x num_neg_samples]. For each node, a list of affinities to
                negative samples is computed.
        """
        neg_aff = inputs1.mm(neg_samples.t()) #(n_batch_edges, num_neg_samples)
        return neg_aff


    def sigmoid_cross_entropy_with_logits(self, labels, logits):
        sig_aff = torch.sigmoid(logits)
        loss = labels * -torch.log(sig_aff) + (1 - labels) * -torch.log(1 - sig_aff)
        return loss

    def _xent_loss(self, inputs1, inputs2, neg_samples):
        """
        inputs1: Tensor (512, 256), normalized vector
        inputs2: Tensor (512, 256), normalized vector
        neg_sample: Tensor (20, 256)
        """
        cuda = inputs1.is_cuda
        true_aff = self.affinity(inputs1, inputs2)
        neg_aff = self.neg_cost(inputs1, neg_samples)
        true_labels = torch.ones(true_aff.shape)  # (n_batch_edges,)
        if cuda:
            true_labels = true_labels.cuda()
        true_xent = self.sigmoid_cross_entropy_with_logits(labels=true_labels, logits=true_aff)
        neg_labels = torch.zeros(neg_aff.shape)
        if cuda:
            neg_labels = neg_labels.cuda()
        neg_xent = self.sigmoid_cross_entropy_with_logits(labels=neg_labels, logits=neg_aff)
        loss0 = true_xent.sum()
        loss1 = self.neg_sample_weights * neg_xent.sum()
        loss = loss0 + loss1
        return loss, loss0, loss1