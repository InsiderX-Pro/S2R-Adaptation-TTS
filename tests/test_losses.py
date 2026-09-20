import unittest
import torch
import torch.nn.functional as F

from s2r_adaptation.losses import firered_loss, omnivoice_loss, weighted_example_mean


class LossTests(unittest.TestCase):
    def test_example_count_denominator_and_gradients(self):
        losses = torch.tensor([2., 4.], requires_grad=True)
        result = weighted_example_mean(losses, [.1, 1.])
        self.assertAlmostEqual(result.item(), 2.1, places=6)
        result.backward()
        torch.testing.assert_close(losses.grad, torch.tensor([.05, .5]))

    def test_uniform_scaling_does_not_cancel(self):
        values = torch.tensor([2., 4.])
        self.assertEqual(weighted_example_mean(values, [.5, .5]).item(), 1.5)

    def test_firered_weights_flow_and_stop(self):
        flow = torch.tensor([2., 3.], requires_grad=True)
        stop = torch.tensor([4., 5.], requires_grad=True)
        total = firered_loss(flow, stop, [.125, 1.])
        self.assertAlmostEqual(total.item(), 1.9, places=6)
        total.backward()
        torch.testing.assert_close(flow.grad, torch.tensor([.0625, .5]))
        torch.testing.assert_close(stop.grad, torch.tensor([.00625, .05]))

    def test_zero_weights_keep_autograd(self):
        loss = torch.tensor([2., 3.], requires_grad=True)
        value = weighted_example_mean(loss, [0., 0.])
        value.backward()
        torch.testing.assert_close(loss.grad, torch.zeros(2))

    def test_bad_shapes_and_weights(self):
        for weights in [[.1], [[.1, 1.]], [float("nan"), 1], [-.1, 1], [1.1, 1]]:
            with self.assertRaises(ValueError): weighted_example_mean(torch.tensor([1., 2.]), weights)

    def test_omni_variable_lengths_match_separate_examples(self):
        torch.manual_seed(5)
        logits = torch.randn(2, 2, 5, 7, requires_grad=True)
        labels = torch.randint(0, 7, (2, 2, 5))
        labels[0, :, 2:] = -100
        labels[1, :, 0] = -100
        a = omnivoice_loss(logits[:1], labels[:1], [1.], [8, 4])
        b = omnivoice_loss(logits[1:], labels[1:], [1.], [8, 4])
        expected = (.125 * a + b) / 2
        actual = omnivoice_loss(logits, labels, [.125, 1.], [8, 4])
        torch.testing.assert_close(actual, expected)
        expected_grad, = torch.autograd.grad(expected, logits, retain_graph=True)
        actual_grad, = torch.autograd.grad(actual, logits)
        torch.testing.assert_close(actual_grad, expected_grad)

    def test_omni_preserves_native_single_example_objective(self):
        torch.manual_seed(42)
        logits = torch.randn(1, 2, 4, 5)
        labels = torch.tensor([[[0, 1, -100, 3], [1, 2, -100, 4]]])
        tokens = F.cross_entropy(logits.permute(0, 3, 1, 2), labels, reduction="none", ignore_index=-100)
        native = ((tokens.sum((0, 2)) / (labels != -100).sum((0, 2))) * torch.tensor([2/3, 1/3])).sum()
        torch.testing.assert_close(omnivoice_loss(logits, labels, [.5], [8, 4]), native * .5)

    def test_omni_ignored_tokens_have_zero_gradient(self):
        logits = torch.randn(1, 1, 3, 4, requires_grad=True)
        labels = torch.tensor([[[1, -100, -100]]])
        omnivoice_loss(logits, labels, [.1], [1]).backward()
        self.assertEqual(torch.count_nonzero(logits.grad[:, :, 1:]).item(), 0)

    def test_accumulation_matches_full_batch(self):
        x = torch.tensor([2., 3., 4.], requires_grad=True)
        weights = [.125, .5, 1.]
        full = weighted_example_mean(x.square(), weights)
        accum = sum(weighted_example_mean(x[i:i+1].square(), [weights[i]]) / 3 for i in range(3))
        torch.testing.assert_close(full, accum)
        torch.testing.assert_close(torch.autograd.grad(full, x, retain_graph=True)[0],
                                   torch.autograd.grad(accum, x)[0])


if __name__ == "__main__": unittest.main()
