"""
dowham_v1.py — Reference implementation of the DoWhaM intrinsic reward.

This module implements "Don't Do What Doesn't Matter" (DoWhaM), the
action-usefulness intrinsic motivation method of:

    Seurin, Strub, Preux, Pietquin. "Don't Do What Doesn't Matter: Intrinsic
    Motivation with Action Usefulness." IJCAI 2021.

Idea
----
Instead of rewarding *state* novelty (count-based / curiosity methods), DoWhaM
rewards the agent for successfully performing actions that are *rarely
effective*. Most actions change the state almost every time they are used (e.g.
moving forward in open space), whereas a few actions only matter in special
contexts (e.g. ``toggle`` opens a door, ``pickup`` grabs a key). Those rare-but-
effective actions are usually landmarks in the environment's dynamics and are
hard to discover by random exploration, so DoWhaM biases the agent toward the
states where such actions pay off.

Paper formulation
-----------------
For every action ``a`` the method tracks, over the whole history ``H`` of
transitions ``(s_h, a_h, s_{h+1})`` (across all episodes):

    U^H(a) = Σ_h 1{a_h = a}                         # times the action was used     (Eq. 1)
    E^H(a) = Σ_h 1{a_h = a} · 1{s_h ≠ s_{h+1}}      # times it was effective        (Eq. 2)

An action is "effective" when it actually changes the state (``s_t ≠ s_{t+1}``).
The bonus is a continuous approximation of the exponential decay
``exp(-η · E/U)``:

    B(a_t) = ( η^(1 - E^H(a_t)/U^H(a_t)) - 1 ) / ( η - 1 )                          (Eq. 3)

``B`` ranges from 1 when the action has never been effective (E = 0) down to 0
when it is always effective (E = U). Small ``η`` spreads the bonus uniformly
across actions; large ``η`` concentrates it on the rare-but-efficient ones
(the paper uses η = 40).

The final intrinsic reward divides the bonus by the square root of an *episodic*
state count ``N_τ(s_{t+1})`` (reset every episode) so the signal decays as a
state is revisited within an episode, and is only paid when the action was
effective:

    r_i(s_t, a_t, s_{t+1}) = B(a_t) / sqrt(N_τ(s_{t+1}))   if s_t ≠ s_{t+1}         (Eq. 4)
                           = 0                              otherwise

This intrinsic reward is added to the sparse extrinsic reward as
``r = r_e + β · r_i`` (the paper uses scaling β = 0.05, applied by the caller —
see ``trainer.py``).
"""

import numpy as np


class DoWhaMIntrinsicRewardV1:
    """DoWhaM intrinsic reward (Seurin et al., IJCAI 2021).

    Maintains the usage / effectiveness statistics needed to compute the action
    bonus ``B(a)`` and the episodic state count used to discount it. Counts in
    ``usage_counts`` / ``effectiveness_counts`` accumulate across episodes;
    ``state_visit_counts`` is episodic and cleared by :meth:`reset_episode`.

    Typical per-step call sequence (see ``CustomEnv.step`` in ``trainer.py``):
        1. :meth:`update_state_visits` — bump the episodic count of the new state;
        2. :meth:`update_usage` — record that ``action`` was used in ``obs``;
        3. :meth:`update_effectiveness` — record whether it changed the state;
        4. :meth:`calculate_intrinsic_reward` — return the DoWhaM reward.

    Args:
        eta: Ratio decay ``η`` from Eq. 3 (paper value: 40). Larger values
            sharpen the bonus toward rare-but-effective actions.
        tau: Episodic-count normalization exponent (see module docstring); the
            count is raised to ``tau`` before the square root.
    """

    def __init__(self, eta, H, tau):
        self.eta = eta  # Decay parameter
        self.H = H
        self.tau = tau  # State normalization exponent
        self.usage_counts = {}  # Tracks U(a): action usage counts per state
        self.effectiveness_counts = {}  # Tracks E(a): action effectiveness counts per state
        self.state_visit_counts = {}  # Tracks state visit counts per state

    def update_usage(self, obs, action):
        """Increment ``U(a)`` — the usage count for ``action`` in state ``obs``.

        Implements the running accumulation of Eq. 1, conditioned on the state
        hash ``obs``.
        """
        if obs not in self.usage_counts:
            self.usage_counts[obs] = {}
        self.usage_counts[obs][action] = self.usage_counts[obs].get(action, 0) + 1

    def update_effectiveness(self, obs, action, next_obs, state_changed):
        """Increment ``E(a)`` — the effectiveness count — when the state changed.

        Implements the running accumulation of Eq. 2: ``action`` taken in ``obs``
        counts as effective only when ``state_changed`` is True (``s_t ≠
        s_{t+1}``). ``next_obs`` is part of the call signature for symmetry but
        the effectiveness decision is carried by ``state_changed``.
        """
        if obs not in self.effectiveness_counts:
            self.effectiveness_counts[obs] = {}
        if action not in self.effectiveness_counts[obs]:
            self.effectiveness_counts[obs][action] = 0

        if state_changed:
            self.effectiveness_counts[obs][action] += 1

    def calculate_bonus(self, obs, action):
        """Compute the action bonus ``B(a)`` for ``action`` in state ``obs`` (Eq. 3).

        Returns ``(η^(1 - E/U) - 1) / (η - 1)``, which is 1 when the action has
        never been effective (E = 0) and approaches 0 as it becomes reliably
        effective (E → U). The ``U == 1 and E == 1`` branch is a guard for the
        first effective use of an action — Eq. 3 would return 0 there — and
        instead grants the maximum bonus of 1.0 to reward the discovery.

        ``U`` defaults to 1 (avoids division by zero for an unseen action) and
        ``E`` defaults to 0.
        """
        U = self.usage_counts[obs].get(action, 1)
        E = self.effectiveness_counts[obs].get(action, 0)

        # Handle first-time effective actions
        if U == 1 and E == 1:
            return 1.0  # Maximum reward for first effectiveness

        ratio = E / U
        exp_term = self.eta ** (1 - ratio)
        bonus = (exp_term - 1) / (self.eta - 1)
        return bonus

    def update_state_visits(self, current_obs, next_obs):
        """Update the episodic state count ``N_τ`` for the transition's states.

        ``N_τ(s)`` (reset each episode) counts how often a state has been seen so
        far this episode; it is the denominator term in Eq. 4. The next state's
        count is incremented on every step, and the current state is seeded to 1
        the first time it appears.
        """
        if current_obs not in self.state_visit_counts:
            self.state_visit_counts[current_obs] = 1
        if next_obs not in self.state_visit_counts:
            self.state_visit_counts[next_obs] = 0

        self.state_visit_counts[next_obs] += 1

    def calculate_intrinsic_reward(self, obs, action, next_obs, position_changed):
        """Compute the DoWhaM intrinsic reward for a transition (Eq. 4).

        Returns ``B(a) / sqrt(N_τ(s_{t+1})^tau)`` when the state changed, and 0
        otherwise — DoWhaM only pays out for *effective* actions. The bonus
        ``B(a)`` rewards rarely-effective actions while the episodic-count
        denominator makes the signal fade as the resulting state is revisited
        within the episode.

        Args:
            obs: Hash of the state the action was taken in (``s_t``).
            action: The action taken.
            next_obs: Hash of the resulting state (``s_{t+1}``); used for the
                episodic count.
            position_changed: Whether the state actually changed
                (``s_t ≠ s_{t+1}``); the effectiveness gate for the reward.

        Returns:
            float: The intrinsic reward (before the external β scaling).
        """
        # Reward any action that results in a state change
        if not position_changed:
            return 0

        # Normalize the reward by state visit counts
        state_count = self.state_visit_counts.get(next_obs, 1) ** self.tau
        action_bonus = self.calculate_bonus(obs, action)
        reward = action_bonus / np.sqrt(state_count)
        return reward

    def reset_episode(self):
        """Reset the episodic state count at the start of a new episode.

        Only ``state_visit_counts`` (the ``N_τ`` term, reset per episode in the
        paper) is cleared; the cross-episode ``usage_counts`` and
        ``effectiveness_counts`` (``U`` and ``E``) deliberately persist.
        """
        self.state_visit_counts.clear()
