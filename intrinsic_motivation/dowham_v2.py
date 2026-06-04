"""
dowham_v2.py — Area-aware DoWhaM Adaptation (ADA) intrinsic reward.

This module implements ADA (Area-aware DoWhaM Adaptation), the spatial extension
of DoWhaM. Equation references below (Eq. 3.x) follow the ADA derivation in:

    B. EREN, "Effective exploration via intrinsic motivation in reinforcement
    learning," Yüksek lisans tezi, LİSANSÜSTÜ EĞİTİM ENSTİTÜSÜ, İZMİR EKONOMİ
    ÜNİVERSİTESİ, 2026.

ADA keeps the original DoWhaM action-usefulness mechanism (reward actions that
are rarely effective) but adds two **spatial novelty** bonuses so the agent is
also pushed to grow the area it has actually seen and reached. The class is named
``DoWhaMIntrinsicRewardV2`` for backward compatibility; the algorithm it
implements is ADA.

Where DoWhaM tracks action statistics globally and only penalizes revisiting
states, ADA additionally:
  * conditions all action statistics on the observation hash ``h(o_t)`` (the same
    turn is treated differently in front of a door vs. in open space), and
  * maintains spatial-memory sets, built incrementally from partial
    observations, that separate positions the agent has *visited* from positions
    it has merely *seen* (the frontier).

Counters (indexed by observation hash ``h(o_t)`` and action ``a``; Sec. 3.1.2):
    U(h(o_t), a)        usage count            (Eq. 3.1) — update_usage
    E(h(o_t), a)        effectiveness count    (Eq. 3.2) — update_effectiveness
    N_τ(h(o_{t+1}))     episodic state count   (Eq. 3.3) — update_state_visits
``U`` and ``E`` persist across episodes; ``N_τ`` is reset every episode.

Spatial memory (Sec. 3.1.3), built from the egocentric view only:
    P_visited_t   positions the agent has stepped on this episode  (Eq. 3.6)
    P_unseen_t    frontier: positions seen but not yet reached      (Eq. 3.7)
    V_t, V_{t+1}  global coords currently visible before/after the step

Bonuses:
    Action bonus (Eq. 3.8–3.9):
        ρ_t = E(h(o_t), a_t) / U(h(o_t), a_t)
        B   = 1.0                              if U = E = 1   (first effective use)
            = (η^(1 - ρ_t) - 1) / (η - 1)      otherwise
    Expansion bonus (Eq. 3.10–3.11) — reward revealing new map cells:
        S_new = { p ∈ V_{t+1} : p ∉ V_t and p ∉ P_visited_t }
        I_exp = 1{|S_new| > 0} · B · ln(1 + |S_new|)
    Achievement bonus (Eq. 3.12) — reward stepping onto a seen-but-unvisited cell:
        I_ach = 1{ P_{t+1} ∈ P_unseen_t } · B

Final intrinsic reward (Eq. 3.13), only paid when the observation changed:
        r_ADA = (B + I_exp + I_ach) / sqrt(N_τ(h(o_{t+1})))   if h(o_t) ≠ h(o_{t+1})
              = 0                                              otherwise

As in DoWhaM, this is added to the sparse extrinsic reward, scaled by β by the
caller (β = 0.05 in ``trainer.py``).
"""

from collections import deque

import numpy as np


class UniqueDeque(deque):
    """
    A deque that ensures all elements are unique.
    When adding elements, duplicates are not added again.
    """

    def __init__(self, *args, maxlen=None, **kwargs):
        super().__init__(*args, maxlen=maxlen)
        self._set = set(self)  # Internal set to enforce uniqueness

    def append(self, item):
        """Add an item to the right end of the deque if it is not already present."""
        if item not in self._set:
            super().append(item)
            self._set.add(item)

    def appendleft(self, item):
        """Add an item to the left end of the deque if it is not already present."""
        if item not in self._set:
            super().appendleft(item)
            self._set.add(item)

    def extend(self, iterable):
        """Extend the deque by appending elements from the iterable if they are not already present."""
        for item in iterable:
            self.append(item)

    def extendleft(self, iterable):
        """Extend the deque by appending elements to the left from the iterable if they are not already present."""
        for item in iterable:
            self.appendleft(item)

    def remove(self, item):
        """Remove the first occurrence of the item."""
        try:
            super().remove(item)
        except ValueError:
            pass
        try:
            self._set.remove(item)
        except KeyError:
            pass

    def pop(self):
        """Remove and return an element from the right end of the deque."""
        item = super().pop()
        self._set.remove(item)
        return item

    def popleft(self):
        """Remove and return an element from the left end of the deque."""
        item = super().popleft()
        self._set.remove(item)
        return item

    def clear(self):
        """Clear all items from the deque."""
        super().clear()
        self._set.clear()

    def __contains__(self, item):
        """Check if an item is in the deque."""
        return item in self._set


class DoWhaMIntrinsicRewardV2:
    """ADA (Area-aware DoWhaM Adaptation) intrinsic reward — see module docstring.

    Maintains the state-conditioned usage/effectiveness counters, the episodic
    state count, and the visited/unseen position sets needed to compute the
    action, expansion, and achievement bonuses of Eq. 3.13.

    Per-step call sequence (see ``CustomEnv.step`` in ``trainer.py``):
        1. :meth:`update_state_visits` — bump ``N_τ`` for the new state (Eq. 3.3);
        2. :meth:`update_usage` — record action usage ``U`` (Eq. 3.1);
        3. :meth:`update_effectiveness` — record effectiveness ``E`` (Eq. 3.2);
        4. :meth:`calculate_intrinsic_reward` — update the spatial sets and
           return the combined ADA reward (Eq. 3.13).

    Args:
        eta: Decay parameter ``η > 1`` (Eq. 3.9). Larger values concentrate the
            action bonus on rarely-effective actions.
        H: Exponent applied to the effectiveness ratio in the alternative bonus
            form :meth:`_calculate_bonus`.
        tau: Episodic-count normalization exponent applied before the square root
            in the reward denominator.
        randomize_state_transition: Flag reserved for state-transition randomization.
        max_steps: Episode length; sizes the ``recent_transitions`` buffer.
        transition_divisor: Divides ``max_steps`` to set the
            ``recent_transitions`` deque ``maxlen``.
    """

    def __init__(self, eta, H, tau, randomize_state_transition=False, max_steps=200, transition_divisor=1):
        print("DoWhaM V2 Intrinsic Reward Initialized")
        self.action_state = {}
        self.eta = eta
        self.H = H
        self.tau = tau
        self.usage_counts = {}
        self.max_steps = max_steps
        self.effectiveness_counts = {}
        self.state_visit_counts = {}
        self.recent_transitions = UniqueDeque(
            maxlen=max_steps // transition_divisor)  # Track recent state transitions
        self.unseen_positions = set()
        self.visited_positions = set()
        self.randomize_state_transition = randomize_state_transition

    def update_usage(self, obs, action):
        """Increment ``U(h(o_t), a)`` for ``action`` in observation ``obs`` (Eq. 3.1)."""
        if obs not in self.usage_counts:
            self.usage_counts[obs] = {}

        if action not in self.usage_counts[obs]:
            self.usage_counts[obs][action] = 0

        self.usage_counts[obs][action] += 1

    def update_effectiveness(self, obs, action, next_obs, state_changed):
        """Increment ``E(h(o_t), a)`` when the action changed the state (Eq. 3.2).

        ``state_changed`` carries the ``o_t ≠ o_{t+1}`` test.
        """
        if obs not in self.effectiveness_counts:
            self.effectiveness_counts[obs] = {}

        if action not in self.effectiveness_counts[obs]:
            self.effectiveness_counts[obs][action] = 0

        if state_changed:
            self.effectiveness_counts[obs][action] += 1

        # self.update_state_transition(obs, action, state_changed)

    def _calculate_bonus(self, obs, action):
        """Action-bonus variant raising the effectiveness ratio to the power ``H``.

        Computes ``(η^((E/U)^H) - 1) / (η - 1)``, an alternative shaping of the
        action bonus relative to :meth:`calculate_bonus` (Eq. 3.9).
        """
        U = self.usage_counts[obs].get(action, 1)
        E = self.effectiveness_counts[obs].get(action, 0)
        ratio = E / U
        term = ratio ** self.H
        exp_term = self.eta ** term
        bonus = (exp_term - 1) / (self.eta - 1)
        return bonus

    def calculate_bonus(self, obs, action):
        """Compute the action bonus ``B(h(o_t), a)`` (Eq. 3.8–3.9).

        With effectiveness ratio ``ρ = E/U``, returns ``(η^(1 - ρ) - 1)/(η - 1)``
        — 1 when the action is never effective (ρ = 0), decaying to 0 as it
        becomes reliably effective (ρ → 1). The ``U == 1 and E == 1`` branch
        grants the maximum bonus 1.0 on an action's first effective use.
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
        """Update the episodic state count ``N_τ`` for the transition (Eq. 3.3).

        ``N_τ(h(o_{t+1}))`` (reset each episode) is the denominator term of
        Eq. 3.13; it is incremented for the next state on every step, and the
        current state is seeded to 1 the first time it appears.
        """
        if current_obs not in self.state_visit_counts:
            self.state_visit_counts[current_obs] = 1

        if next_obs not in self.state_visit_counts:
            self.state_visit_counts[next_obs] = 0

        self.state_visit_counts[next_obs] += 1

    def update_state_transition(self, obs, action, state_changed):
        """Record the most recent state-change flag for ``(obs, action)``."""
        if obs not in self.action_state:
            self.action_state[obs] = {}

        if action not in self.action_state[obs]:
            self.action_state[obs][action] = False

        self.action_state[obs][action] = state_changed

    def calculate_intrinsic_reward(self, obs, action, next_obs, state_changed, curr_view, next_view, next_pos):
        """Compute the combined ADA intrinsic reward for a transition (Eq. 3.13).

        Updates the spatial-memory sets from the partial view, then combines the
        action bonus ``B`` with the expansion and achievement bonuses and
        normalizes by the episodic state count:

            r_ADA = (B + I_exp + I_ach) / sqrt(N_τ(h(o_{t+1})))   if the state changed
                  = 0                                              otherwise

        Args:
            obs: Hash of the current state ``h(o_t)``.
            action: Action taken.
            next_obs: Hash of the resulting state ``h(o_{t+1})``; indexes ``N_τ``.
            state_changed: Whether ``o_t ≠ o_{t+1}`` (the reward gate).
            curr_view: Global coords visible before the step (``V_t``).
            next_view: Global coords visible after the step (``V_{t+1}``).
            next_pos: Agent's position after the step (``P_{t+1}``).

        Returns:
            float: The ADA intrinsic reward, rounded to 5 decimals (pre-β-scaling).
        """
        # Achievement test (Eq. 3.12): check whether P_{t+1} is on the
        # seen-but-unvisited frontier, before P_{t+1} is added to the visited set.
        was_unseen_position = next_pos in self.unseen_positions

        # P_visited update (Eq. 3.6): the agent has now stepped on next_pos.
        self.visited_positions.add(next_pos)

        # Newly revealed cells S_new (Eq. 3.10): visible now, not visible before,
        # and never visited.
        curr_set = set(curr_view)
        next_set = set(next_view)
        newly_seen_set = next_set - curr_set - self.unseen_positions - self.visited_positions
        newly_seen = list(newly_seen_set)

        # Check if there are newly seen positions
        has_newly_seen = len(newly_seen) > 0

        # P_unseen update (Eq. 3.7): grow the frontier with newly seen cells,
        # then drop anything already visited.
        self.unseen_positions.update(newly_seen if len(newly_seen) != 0 else curr_set)
        self.unseen_positions -= self.visited_positions

        # ADA pays out only for effective actions (h(o_t) != h(o_{t+1})).
        if not state_changed:
            return 0

        # Action bonus B (Eq. 3.9) and the episodic-count denominator term.
        action_bonus = self.calculate_bonus(obs, action)
        state_count = self.state_visit_counts[next_obs] or 1

        total_reward = action_bonus

        # Expansion bonus I_exp (Eq. 3.11): B * ln(1 + |S_new|) when new cells appear.
        if has_newly_seen:
            expansion_bonus = action_bonus * np.log1p(len(newly_seen))  # log(1 + k)
            total_reward += 1.0 * expansion_bonus

        # Achievement bonus I_ach (Eq. 3.12): add B for stepping onto the frontier.
        if was_unseen_position:
            total_reward += 1.0 * action_bonus

        # Normalize by sqrt(N_τ) (Eq. 3.13).
        return round(total_reward / np.sqrt(state_count), 5)

    def reset_episode(self):
        """Reset the per-episode state at the start of a new episode.

        Clears the episodic state count ``N_τ`` and both spatial-memory sets
        (``P_visited`` and ``P_unseen``), matching the episode initialization of
        Algorithm 1 (``N_τ = 0``, ``P_visited_0 = {P_0}``, ``P_unseen_0 = ∅``).
        The cross-episode ``U`` and ``E`` counters are intentionally preserved.
        """
        self.state_visit_counts.clear()
        self.unseen_positions.clear()
        self.visited_positions.clear()
