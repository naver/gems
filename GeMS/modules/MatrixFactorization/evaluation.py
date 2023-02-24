GeMS
Copyright (C) 2023-present NAVER Corp.
CC BY-NC-SA 4.0

import numpy as np
import torch

def precision(correct_predictions, k):
    num_hit = float(np.sum(correct_predictions))
    return num_hit / k

def recall(correct_predictions, num_relevant):
    num_hit = float(np.sum(correct_predictions))
    return num_hit / num_relevant

def ndcg(correct_predictions, num_relevant, k):
    ideal_correct_predictions = np.zeros([k])
    ideal_correct_predictions[:num_relevant] = 1
    return dcg(correct_predictions, k) / dcg(ideal_correct_predictions, k)

def dcg(correct_predictions, k):
    result = 0.0
    for rank in range(k):
        result += correct_predictions[rank] / np.log2(rank + 2)
    return result

def map(correct_predictions, num_relevant, k):
    result = 0.0
    for rank in range(k):
        result += precision(correct_predictions[:rank + 1], rank + 1) * correct_predictions[rank]
    result /= num_relevant
    return result

def evaluate(correct_predicted_interactions, num_true_interactions, metrics):
    """
    Evaluates a ranking model in terms of precision and recall for the given cutoff values
    Args:
        correct_predicted_interactions: (array<bool>: n_rows * max(cutoffs)) 1 iff prediction matches a true interaction
        num_true_interactions: (array<bool>: n_rows) number of true interactions associated to each row
        metrics: (list<tuple<string,int>>) list of metrics to consider, with tuples made of the metric type and cutoff

    Returns:
        eval_results: dictionary with evaluation results for each metric cumulated over all rows; keys are the metrics
    """
    eval_results = {metric: 0 for metric in metrics}
    for metric in metrics:
        (metric_type, k) = metric # Get the metric type and cutoff e.g. ("precision", 5)
        correct_predictions = correct_predicted_interactions[:, :k]
        for row_id, row_correct_prediction in enumerate(correct_predictions): # Rows represent either users or searches
            if metric_type == "precision":
                eval_results[metric] += precision(row_correct_prediction, k)
            elif metric_type == "recall":
                eval_results[metric] += recall(row_correct_prediction, num_true_interactions[row_id])
            elif metric_type == "ndcg":
                eval_results[metric] += ndcg(row_correct_prediction, num_true_interactions[row_id], k)
            elif metric_type == "map":
                eval_results[metric] += map(row_correct_prediction, num_true_interactions[row_id], k)

    return eval_results

def predict_evaluate(data_loader, model, device_embed, known_interactions, metrics, levels=('user',)):
    max_k = max([metric[1] for metric in metrics])
    eval_results = {level: {metric: 0.0 for metric in metrics} for level in levels}
    num_user = 0
    num_interaction = 0
    for (_, batch) in enumerate(data_loader):
        user_ids = batch['user_ids']
        user_seq_sizes = batch['seq_sizes']
        num_user += len(user_ids)

        # Recover the true interactions from before padding
        true_interactions = [interactions.tolist()[:user_seq_sizes[u]] for (u, interactions)
                             in enumerate(batch['interactions'])]
        batch_num_interaction = sum([user_seq_sizes[u] for u in range(len(user_ids))]).numpy()
        num_interaction += batch_num_interaction

        # Predict the items interacted for each user and mask the items which appeared in known interactions
        predicted_scores = model.predict(user_ids.to(device_embed)).cpu()
        ## Shape of predicted_scores: (batch_size, num_item)
        mask_value = -np.inf
        for i, user in enumerate(user_ids):
            for item in known_interactions[int(user)]:
                predicted_scores[i, item] = mask_value
        predicted_interactions = torch.argsort(predicted_scores, dim=1, descending=True).numpy()
        ## Shape of predicted_interactions: (batch_size, num_item)

        for level in levels:
            if level == 'user': # Evaluation is conducted for every user and averaged over users (macro-level)
                # Identify the correctly predicted interactions
                correct_predicted_interactions = np.zeros([len(user_ids), max_k], dtype=bool)
                ## Shape of correct_predicted_interactions: (batch_size, max_k)
                num_true_interactions = np.zeros([len(user_ids)], dtype=int) # Number of relevant items per user
                ## Shape of num_true_interactions: (batch_size)
                for i, user in enumerate(user_ids):
                    user_true_interactions = set(true_interactions[i])
                    num_true_interactions[i] = len(user_true_interactions)
                    user_predicted_interactions = predicted_interactions[i, :max_k]
                    for j, user_predicted_interaction in enumerate(user_predicted_interactions):
                        if user_predicted_interaction in user_true_interactions:
                            correct_predicted_interactions[i, j] = 1
            elif level == 'interaction': # Evaluation is conducted for every interaction and averaged over interactions (micro-level)
                # Identify the correctly predicted interactions
                correct_predicted_interactions = np.zeros([batch_num_interaction, max_k], dtype=bool)
                ## Shape of correct_predicted_interactions: (num_search, max_k)
                num_true_interactions = np.zeros([batch_num_interaction], dtype=int)  # Number of relevant items per user
                ## Shape of num_true_interactions: (num_search)
                search_id = 0
                for i, user in enumerate(user_ids):
                    user_predicted_interactions = predicted_interactions[i, :max_k]
                    for true_interaction in true_interactions[i]:
                        num_true_interactions[search_id] = 1 # Always one click per interaction
                        for j, user_predicted_interaction in enumerate(user_predicted_interactions):
                            if user_predicted_interaction == true_interaction:
                                correct_predicted_interactions[search_id, j] = 1
                                break # True interaction was found, no need to look further
                        search_id += 1
            else:
                print("Incorrect evaluation level:", level)
                continue

            # Perform the evaluation
            sub_eval_results = evaluate(correct_predicted_interactions, num_true_interactions, metrics)
            eval_results[level] = {metric: eval_results[level][metric] + sub_eval_results[metric] for metric in metrics}

    num = {'user': num_user, 'interaction': num_interaction}
    eval_results = {level: {metric: eval_results[level][metric] / num[level] for metric in metrics} for level in levels}

    return eval_results
