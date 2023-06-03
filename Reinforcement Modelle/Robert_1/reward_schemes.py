def profitDelta(self):
    reward = (self._share_value - self._last_share_value) / self._last_share_value - \
              (self._current_price - self._last_price) / self._last_price
    return reward


def quadraticDelta(self):
    delta = profitDelta()
    reward = np.sign(delta) * delta**2
    return reward


def profitDeltaActionPenalty(self, action, penalty=0.0025):
    delta = (self._share_value - self._last_share_value) / self._last_share_value - \
             (self._current_price - self._last_price) / self._last_price
    reward = delta - action*penalty
    return reward
