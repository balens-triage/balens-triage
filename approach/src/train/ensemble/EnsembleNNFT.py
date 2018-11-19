import logging
import numpy as np
from src.train.ensemble.EnsembleNN import EnsembleNN

logger = logging.getLogger(__name__)


def normalize(tpl_list):
    minim = np.array([it[1] for it in tpl_list]).min()
    minimized = [(it[0], it[1] - minim) for it in tpl_list]

    maximum = np.array([it[1] for it in minimized]).max()
    if maximum == 0:
        raise Exception('maximum of number cannot be 0')
    normalized = [(it[0], it[1] / maximum) for it in minimized]
    return normalized


def flip_perc(tpl_list):
    return [(it[0], 1 - it[1]) for it in tpl_list]


# Extending our nn approach with CF for opening the system and reducing developer load
class EnsembleNNFT(EnsembleNN):
    def __init__(self, namespace, container, *args):
        super().__init__(namespace, container, *args)

        self.alpha = .5

        # First argument is the layer name for the ensemble class
        if len(args) > 1:
            self.alpha = args[1]

        logger.info('alpha: {:.2f}'.format(self.alpha))

    def ensemble_predict(self, bug, full_response=False):
        prediction = self.predict(bug)

        pred_vec, bug_vec = self._bug_to_vector(prediction, bug['vector'])

        probs = self._ensemble.predict([pred_vec, bug_vec])

        k = self.top_k * self.top_k_padding

        top_k_probas = list(zip(self._ensemble_labels, probs[0]))
        top_k_probas = normalize(top_k_probas)

        prediction_times = self._ft_estimator.get_prediction_times(bug['vector'])
        prediction_times = flip_perc(normalize(prediction_times))
        prediction_times = dict(prediction_times)

        acc_results = []

        for ens_prediction in top_k_probas:
            email = prediction[ens_prediction[0]]
            ftime = 0
            if email in prediction_times:
                ftime = prediction_times[email]
            acc_results.append((email, ens_prediction[1], ftime))

        cost_results = [(pred[0], self._get_cost(pred[1], pred[2]), pred[1], pred[2]) for
                        pred in acc_results]
        sorted_cost_results = sorted(cost_results, key=lambda x: x[1],
                                     reverse=True)[:k]

        if full_response:
            return sorted_cost_results
        else:
            return [cost[0] for cost in sorted_cost_results]

    def _get_cost(self, accuracy, ftime):
        return (self.alpha * accuracy + (1 - self.alpha) * ftime) / 2
