import torch

from torch_autoneb.fill import Fill


class FillHighest(Fill):
    def __init__(self, interpolate_count=9, metric="train_data_energy", insert_thres=.1):
        self.interpolate_count = interpolate_count  # this is also used in dg_simple
        self.metric = metric
        self.insert_thres = insert_thres

    def fill(self, chain, insert_count, weights, neb_model, transition_state):
        insert_count = min(insert_count, chain.shape[0] - 1)

        potential = neb_model.potential
        alphas = torch.linspace(0, 1, self.interpolate_count + 2)[1:-1]
        if chain.is_cuda:
            alphas = alphas.cuda(chain.get_device())
        chain_max, chain_min, scores = self.compute_scores(alphas, chain, potential, transition_state)

        # These casts go from numpy -> float
        # scores -= float(chain_min)
        scores /= float(chain_max - chain_min) + 1e-12
        values, order = scores.sort()

        max_values = values[:, -1]
        max_alphas = alphas[order[:, -1]]

        _, gap_order = max_values.sort()
        gaps_to_fill = gap_order[-insert_count:]

        fill = []
        for gap_idx in range(chain.shape[0] - 1):
            if gap_idx in gaps_to_fill and max_values[gap_idx] > self.insert_thres:
                fill.append([max_alphas[gap_idx]])
            else:
                fill.append([])

        return PyTorchNEB.fill_chain(chain, fill, weights)

    def compute_scores(self, alphas, chain, potential, transition_state):
        # Potentially get the information form the previous run
        scores = chain.new(chain.shape[0] - 1, self.interpolate_count).zero_()
        chain_min = 1e12
        chain_max = -1e12
        value_count = (chain.shape[0] - 1) * (len(alphas) + 1) + 1
        accepted_ts_metric = False

        if transition_state is not None:
            user_data = transition_state.user_data
            if user_data is not None and "sub_" + self.metric in user_data:
                sub_data = user_data["sub_" + self.metric]
                if sub_data.shape[0] == value_count:
                    accepted_ts_metric = True
                    score_b = sub_data[0]
                    for i in range(chain.shape[0] - 1):
                        # Start: Can take last value
                        score_a = score_b
                        idx_a = i * (self.interpolate_count + 1)
                        # Stop: Need to compute
                        score_b = sub_data[idx_a + self.interpolate_count + 1]
                        # As a reference, compute the min/max of the chain
                        chain_min = min(chain_min, min(score_a, score_b))
                        chain_max = max(chain_max, max(score_a, score_b))
                        for j, alpha in enumerate(alphas):
                            scores[i, j] = self.score_for_point(alpha, sub_data[idx_a + j + 1], score_a, score_b)

        # Else, compute it
        if not accepted_ts_metric:
            with potential.no_leaking_jobs(), tqdm(range(value_count), "Insert eval") as pbar:
                score_b = potential.getMetricValues(chain[0], [self.metric])[self.metric]
                pbar.update()
                for i in range(chain.shape[0] - 1):
                    # Start: Can take last value
                    a = chain[i]
                    score_a = score_b
                    # Stop: Need to compute
                    b = chain[i + 1]
                    score_b = potential.getMetricValues(b, [self.metric])[self.metric]
                    pbar.update()

                    # As a reference, compute the min/max of the chain
                    chain_min = min(chain_min, min(score_a, score_b))
                    chain_max = max(chain_max, max(score_a, score_b))

                    for j, alpha in enumerate(alphas):
                        coords = (1 - alpha) * a + alpha * b
                        # Compute the score of the point relative to the difference between surrounding path elements
                        job = potential.startJob(coords, [self.metric], None, None)
                        job.result_callback = self.store_metric_in_tensor(scores, (i, j), self.metric, score_a, score_b, alpha, pbar)
                    potential.handleJobResults()
        return chain_max, chain_min, scores

    def store_metric_in_tensor(self, tensor, index, metric, score_a, score_b, alpha, pbar=None):
        """
        Parameters
        ----------
        tensor: torch.FloatTensor
        index: tuple
        metric: str
        score_a: float
        score_b: float
        alpha: float
        pbar: tqdm or None

        Returns
        -------
        callable
        """

        def store(job_result):
            """
            Parameters
            ----------
            job_result: helper.models.pytorch.model_parallel.ModelJobResult
            """
            tensor[index] = self.score_for_point(alpha, job_result.metrics[metric], score_a, score_b)
            if pbar is not None:
                pbar.update()

        return store

    def score_for_point(self, alpha, energy, score_a, score_b):
        return energy - (score_a * (1 - alpha) + score_b * alpha)

    def __repr__(self):
        if self.insert_thres == .1:
            return "highest"
        else:
            return "highest-%f%s" % (self.insert_thres, self.metric if self.metric != "train_data_energy" else "")