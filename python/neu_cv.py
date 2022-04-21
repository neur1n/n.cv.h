"""****************************************************************************
MIT License

Copyright (c) 2022 Jihang Li (jihangli AT duck DOT com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


Last update: 2022-04-13 18:42
****************************************************************************"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Python imports
import copy
import cv2
import numpy as np

# User imports


#============================================================ NLowPassFilter{{{
class NLowPassFilter(object):
    def __init__(self, alpha = 0.0):
        super(NLowPassFilter, self).__init__()

        self._hatxprev = None
        self._hatx = None

        self.reset(alpha)

    def filter(self, x, alpha=None):
        if alpha is not None and np.all((alpha > 0.0) & (alpha <= 1.0)):
            self._alpha = alpha

        if not self._initialized:
            self._hatxprev = x
            self._initialized = True

        self._hatx = self._alpha * x + (1.0 - self._alpha) * self._hatxprev
        self._hatxprev = self._hatx

        return self._hatx

    def last_value(self):
        return self._hatxprev

    def reset(self, alpha=None):
        if alpha is not None:
            assert np.all((alpha > 0.0) & (alpha <= 1.0)), \
                    "'alpha' ({}) must be in (0.0, 1.0].".format(alpha)
            self._alpha = alpha

        self._initialized = False
#}}}

#============================================================ NOneEuroFilter{{{
class NOneEuroFilter(object):
    def __init__(self, frequency: float, beta: float=0.0, mincutoff: float=1.0, dcutoff: float=1.0):
        super(NOneEuroFilter, self).__init__()

        self._timestamp = None
        self.reset(frequency, beta, mincutoff, dcutoff)

    def filter(self, x, timestamp=None):
        if self._timestamp and timestamp:
            self._frequency = 1.0 / (timestamp - self._timestamp)
        self._timestamp = timestamp

        if not self._initialized:
            dx = 0.0
            self._initialized = True
        else:
            dx = (x - self._x.last_value()) * self._frequency

        edx = self._dx.filter(dx)
        cutoff = self._mincutoff + self._beta * np.absolute(edx)
        return self._x.filter(x, self._get_alpha(cutoff))

    def reset(self, frequency=None, beta=None, mincutoff=None, dcutoff=None):
        self._initialized = False

        if frequency is not None:
            assert frequency >= 0.0, \
                    "'frequency' ({}) must be larger than 0.0.".format(frequency)
            self._frequency = frequency

        if beta is not None:
            assert beta >= 0.0, \
                    "'beta' ({}) must be larger than 0.0.".format(beta)
            self._beta = beta

        if mincutoff is not None:
            assert mincutoff >= 0.0, \
                    "'mincutoff' ({}) must be larger than 0.0.".format(mincutoff)
            self._mincutoff = mincutoff
            self._x = NLowPassFilter(self._get_alpha(mincutoff))

        if dcutoff is not None:
            assert dcutoff >= 0.0, \
                    "'dcutoff' ({}) must be larger than 0.0.".format(dcutoff)
            self._dcutoff = dcutoff
            self._dx = NLowPassFilter(self._get_alpha(dcutoff))

    def _get_alpha(self, cutoff):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau * self._frequency)
#}}}

#============================================================= NKalmanFilter{{{
class NKalmanFilter(object):
    def __init__(self, state_d, measurement_d, control_d = None,
            r_scale = 1.0, q_scale = 1.0):
        super(NKalmanFilter, self).__init__()

        self._kf: cv2.KalmanFilter = None
        self._initialized = False
        self._state = {}  # To mimic deepcopy

        self._state_d = state_d
        self._measurement_d = measurement_d
        self._r_scale = r_scale
        self._q_scale = q_scale

        self.reset(state_d, measurement_d, control_d, r_scale, q_scale)

    def correct(self, measurement: np.ndarray) -> np.ndarray:
        self._kf.correct(measurement)
        return self._kf.statePost[0:self._measurement_d]

    def filter(self, measurement: np.ndarray, control = None) -> np.ndarray:
        if not self._initialized:
            self._kf.statePost = self._measurement_to_state(measurement)
            self._initialized = True
            return measurement[0:self._measurement_d]

        self.predict(control)
        return self.correct(measurement)

    def peek(self, steps: int = 1, control=None) -> np.ndarray:
        assert steps >= 1, "'steps' ({}) must be at least 1.".format(steps)

        self._save_state()

        results = np.zeros((steps, self._measurement_d, 1))

        for i in range(steps):
            p = self._kf.predict(control)
            results[i] = copy.deepcopy(p[0:self._measurement_d])

        self._load_state()

        return results

    def predict(self, control = None) -> np.ndarray:
        self._kf.predict(control)
        return self._kf.statePre[0:self._measurement_d]

    def reset(self, state_d, measurement_d, control_d = None,
            r_scale = None, q_scale = None):
        assert state_d > 0 and measurement_d > 0, \
                "Dimensions of state ({}) and measurement ({}) must be positive." \
                .format(state_d, measurement_d)

        self._initialized = False

        self._kf = cv2.KalmanFilter(state_d, measurement_d, control_d)

        self._kf.measurementMatrix = \
                np.eye(measurement_d, state_d, dtype=np.float32)

        if r_scale is not None:
            self._r_scale = r_scale

        if q_scale is not None:
            self._q_scale = q_scale

        self._kf.measurementNoiseCov = \
                np.eye(measurement_d, dtype=np.float32) * self._r_scale
        self._kf.processNoiseCov = \
                np.eye(self._state_d, self._state_d, dtype=np.float32) * self._q_scale

    def get_control_matrix(self) -> np.ndarray:
        return self._kf.controlMatrix

    def get_measurement_matrix(self) -> np.ndarray:
        return self._kf.measurementMatrix

    def get_measurement_noise_cov(self) -> np.ndarray:
        return self._kf.measurementNoiseCov

    def get_process_noise_cov(self) -> np.ndarray:
        return self._kf.processNoiseCov

    def get_transition_matrix(self) -> np.ndarray:
        return self._kf.transitionMatrix

    def set_control_matrix(self, B: np.ndarray):
        self._kf.controlMatrix = B

    def set_measurement_matrix(self, H: np.ndarray):
        self._kf.measurementMatrix = H

    def set_measurement_noise_cov(self, R: np.ndarray):
        self._kf.measurementNoiseCov = R

    def set_process_noise_cov(self, Q: np.ndarray):
        self._kf.processNoiseCov = Q

    def set_transition_matrix(self, A: np.ndarray):
        self._kf.transitionMatrix = A

    def _load_state(self):
        self._kf.transitionMatrix = self._state["A"]
        self._kf.controlMatrix = self._state["B"]
        self._kf.measurementMatrix = self._state["H"]
        self._kf.measurementNoiseCov = self._state["R"]
        self._kf.processNoiseCov = self._state["Q"]
        self._kf.errorCovPre = self._state["error_cov_pre"]
        self._kf.errorCovPost = self._state["error_cov_post"]
        self._kf.statePre = self._state["state_pre"]
        self._kf.statePost = self._state["state_post"]

    def _measurement_to_state(self, measurement: np.ndarray) -> np.ndarray:
        state = np.zeros(self._kf.statePost.shape, dtype=np.float32)

        if measurement.ndim > 1:
            mr, mc = measurement.shape

            assert mr <= self._state_d and mc <= self._state_d, \
                    "Measurement {}x{} has a bigger size than state ({}x{})." \
                    .format(mr, mc, self._state_d, self._state_d)

            state[0:mr, 0:mc] = measurement
        else:
            mr = measurement.shape[0]
            mc = 1

            assert mr <= self._kf.statePost.shape[0] and mc <= self._kf.statePost.shape[1], \
                    "Measurement {}x{} has a bigger size than state ({})." \
                    .format(mr, mc, self._kf.statePost.shape)

            state[0:mr, 0:mc] = measurement.reshape(mr, mc)

        return state

    def _save_state(self):
        self._state["A"] = self._kf.transitionMatrix
        self._state["B"] = self._kf.controlMatrix
        self._state["H"] = self._kf.measurementMatrix
        self._state["R"] = self._kf.measurementNoiseCov
        self._state["Q"] = self._kf.processNoiseCov
        self._state["error_cov_pre"] = self._kf.errorCovPre
        self._state["error_cov_post"] = self._kf.errorCovPost
        self._state["state_pre"] = self._kf.statePre
        self._state["state_post"] = self._kf.statePost
#}}}
