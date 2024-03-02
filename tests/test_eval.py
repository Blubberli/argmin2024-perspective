import unittest

from evaluate_diversity import alpha_ndcg, calculateNormalizedDiscountedKLDivergence


class TestAlphaNDCG(unittest.TestCase):

    def test_alpha_ndcg_example(self):
        """
        First, we calculate the relevance score product for each item (considering alpha and redundancy):
        item 1: 1 * 1 = 1 because it is relevant and there is no redundancy
        item 2: 0 because the relevance score is 0 so the product is 0
        item 3: 1 * (1-alpha) = 1 * 0.5 = 0.5 because it is relevant and there is redundancy.
        item 4: 1 * (1-alpha) = 1 * 0.5 = 0.5 because it is relevant and there is redundancy.
        item 5: 0 because the relevance score is 0 so the product is 0.
        Next we take the ranking into account, we want to weigh each score the lower the rank is so we divide
        each score by log2(rank + 1):
        1/(log2(1+1)) + 0/(log2(2+1)) + 0.5/(log2(3+1)) + 0.5/(log2(4+1)) + 0/(log2(5+1)) = 1.715
        For the ideal ranking we would just move the second item to the end. This would result in the following
        relevance score product:
        item 1: 1 * 1 = 1 because it is relevant and there is no redundancy
        item 2: 1*1 = 1 because it is relevant and there is no redundancy
        item 3: 1 * (1-alpha) = 1 * 0.5 = 0.5 because it is relevant and there is redundancy.
        item 4 and 5 are zero.
        the sum would then be:
        1/(log2(1+1)) + 1/(log2(2+1)) + 0.5/(log2(3+1)) + 0/(log2(4+1)) + 0/(log2(5+1)) = 1.881
        Finally we would divide the two scores and get 1.715/1.881 = ~0.911
        :return:
        """
        relevance_scores = [1, 0, 1, 1, 0]
        perspectives = ['A', 'A', 'B', 'A', 'B']
        ground_truth_relevance = [1, 1, 1, 0, 0]
        ground_truth_perspectives = ['A', 'A', 'B', 'B', 'A']
        alpha = 0.5
        alpha_ndcg_score = \
        alpha_ndcg(perspectives_global=ground_truth_perspectives, perspectives_predictions=perspectives,
                   relevance_scores_global=ground_truth_relevance, relevance_scores_predictions=relevance_scores,
                   alpha=alpha, k_range=[5])[5]
        expected_outcome = 0.911
        self.assertAlmostEqual(alpha_ndcg_score, expected_outcome, places=2)

    def test_normalized_kl_divergence(self):
        # Test data setup
        ranked_perspectives = ["A", "A", "B", "B", "C", "C"]
        gold_distribution = {"A": 0.33, "B": 0.33, "C": 0.33}
        cut_off_points = [2, 4, 6]
        k = 6

        # Expected value setup (this should be computed based on the expected behavior of the function)
        expected_rkl_value = 0.2424

        # Calculate normalized KL-divergence
        calculated_rkl_value = \
        calculateNormalizedDiscountedKLDivergence(ranked_perspectives=ranked_perspectives, gold_propotion=0.33,
                                        protected_group="A", cut_off_points=cut_off_points, k=k)

        # Assert equality
        self.assertAlmostEqual(calculated_rkl_value, expected_rkl_value, places=2,
                               msg="The calculated rKL value does not match the expected value.")

    def test_perfect_kl_divergence(self):
        # create a ranked list that has a fair distribution of each group
        ranked_perspectives = ["A", "A", "B", "B", "C", "C"]
        gold_distribution = {"A": 0.33, "B": 0.33, "C": 0.33}
        cut_off_points = [6]
        k = 6
        for category, gold_proportion in gold_distribution.items():
            calculated_rkl_value = calculateNormalizedDiscountedKLDivergence(ranked_perspectives, gold_proportion, category,
                                                                   cut_off_points, k)
            self.assertAlmostEqual(calculated_rkl_value, 0.0, places=4,
                                   msg="The calculated rKL value does not match the expected value.")


if __name__ == '__main__':
    unittest.main()
