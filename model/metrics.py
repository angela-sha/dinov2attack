import numpy as np 

# Generate metrics for a poison attack
def get_metrics(results_before: dict, 
                results_after: dict, 
                source_label, target_label
                ):
    """
    Results has the following keys:
    results = {
            'source_predictions': [],
            'target_predictions': [],
            'source_confidences': [],
            'target_confidences': []
        }
    """
    source_correct_before = [s for s in results_before['source_predictions'] \
                             if s == source_label]
    source_correct_after = [s for s in results_after['source_predictions'] \
                            if s == source_label]
    
    source_incorrect_after = [c for s, c in zip(results_after['source_predictions'],
                                                results_after['source_confidences'])\
                                                if s != source_label]
    
    target_correct_before = [t for t in results_before['target_predictions'] \
                             if t == target_label]
    target_correct_after = [t for t in results_after['target_predictions'] \
                            if t == target_label]
    
    metrics = {
        'accuracy_source_before': len(source_correct_before)/\
            len(results_before['source_predictions']),
        'accuracy_target_before': len(target_correct_before)/\
            len(results_before['target_predictions']),
        'accuracy_source_after': len(source_correct_after)/\
            len(results_after['source_predictions']),
        'accuracy_target_after': len(target_correct_after)/\
            len(results_after['target_predictions']),
        'attack_avg_confidence': np.mean(source_incorrect_after),    
    }

    print("Printing results...")
    print(results_before)
    print(results_after)
    print(metrics)

    return metrics