# tests/test_rl_drug_design.py

import unittest
from models.rl_drug_design import train_policy_gradient, PolicyNetwork
from unittest.mock import MagicMock

class TestRLDrugDesign(unittest.TestCase):
    def test_train_policy_gradient(self):
        env = MagicMock()
        env.reset.return_value = [0.0] * 10
        env.step.return_value = ([0.0] * 10, 1.0, False, {})
        env.max_steps = 5
        
        policy_network = PolicyNetwork(state_dim=10, action_dim=3)
        trained_policy = train_policy_gradient(env, policy_network, epochs=10)
        self.assertIsNotNone(trained_policy)

if __name__ == '__main__':
    unittest.main()
