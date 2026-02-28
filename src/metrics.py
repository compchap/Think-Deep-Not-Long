import torch
import torch.nn.functional as F

def calculate_jsd(p_final, p_inter):
    """Equation 2: Measuring the change in internal prediction[cite: 144, 145]."""
    m = 0.5 * (p_final + p_inter)
    # eps prevents log(0) errors
    eps = 1e-10
    term1 = F.kl_div(m.log(), p_final + eps, reduction='batchmean')
    term2 = F.kl_div(m.log(), p_inter + eps, reduction='batchmean')
    return 0.5 * (term1 + term2)

def get_settling_depth(all_hidden_states, lm_head, final_norm, threshold=0.5):
    """
    Finds the first layer where the prediction stabilizes[cite: 113, 155].
    """
    L = len(all_hidden_states) - 1 # Layers are 0-indexed in hidden_states
    
    # Final layer distribution is our 'gold standard' for this token [cite: 108]
    final_h = final_norm(all_hidden_states[-1][:, -1, :])
    p_final = F.softmax(lm_head(final_h), dim=-1)
    
    for l in range(L + 1):
        # Project intermediate hidden state to vocabulary space (Logit Lens) [cite: 105, 106]
        inter_h = final_norm(all_hidden_states[l][:, -1, :])
        p_inter = F.softmax(lm_head(inter_h), dim=-1)
        
        jsd = calculate_jsd(p_final, p_inter)
        if jsd.item() <= threshold: # Threshold g [cite: 113, 115]
            return l + 1 # Return 1-based layer index
            
    return L + 1