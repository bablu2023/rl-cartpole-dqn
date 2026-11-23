"""keras-rl2 DQN agent.

Importing `rl` switches TensorFlow into an older execution mode, so the
model must be built AFTER this module's imports. `create_agent` does that.
"""
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import BoltzmannQPolicy

try:  # keras-rl2 needs the legacy optimizer when it exists
    from tensorflow.keras.optimizers.legacy import Adam
except ImportError:
    from tensorflow.keras.optimizers import Adam

from src.model import build_model


def create_agent(states, actions, hidden=(24, 24), learning_rate=1e-3):
    """Build the model and a compiled DQN agent around it."""
    model = build_model(states, actions, hidden)

    dqn = DQNAgent(
        model=model,
        nb_actions=actions,
        # picks actions with probability weighted by their Q-values (explores + exploits)
        policy=BoltzmannQPolicy(),
        # replay memory of past (state, action, reward, next_state) experiences
        memory=SequentialMemory(limit=50000, window_length=1),
        nb_steps_warmup=10,        # steps to collect before learning starts
        target_model_update=1e-2,  # slow (soft) update of the target network
    )
    dqn.compile(Adam(learning_rate=learning_rate), metrics=["mae"])
    return dqn, model
