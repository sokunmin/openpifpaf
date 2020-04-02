import logging
import time

import numpy as np

# pylint: disable=import-error
from ..functional import cumulative_average, scalar_square_add_gauss

LOG = logging.getLogger(__name__)


class PifHr(object):
    v_threshold = 0.1

    def __init__(self, pif_nn):
        self.pif_nn = pif_nn

        self.target_accumulator = None
        self.scales = None
        self.scales_n = None

        self._clipped = None

    @property
    def targets(self):
        if self._clipped is not None:
            return self._clipped  # (#kp, edgeH, edgeW)
        # clip those are > 1.0
        self._clipped = np.minimum(1.0, self.target_accumulator)
        return self._clipped

    def fill(self, pif, stride, min_scale=0.0):
        return self.fill_multiple([pif], stride, min_scale)

    def fill_multiple(self, pifs, stride, min_scale=0.0):
        start = time.perf_counter()

        if self.target_accumulator is None:  # <-
            shape = (  # > (#kp, 4, upH, upW)
                pifs[0].shape[0],  # > #kp
                int((pifs[0].shape[2] - 1) * stride + 1),  # > edgeH = (upH -1) * stride + 1
                int((pifs[0].shape[3] - 1) * stride + 1),  # > edgeW = (upW -1) * stride + 1
            )  # > (#kp, edgeH, edgeW)
            ta = np.zeros(shape, dtype=np.float32)  # (#kp, edgeH, edgeW)
            self.scales = np.zeros(shape, dtype=np.float32)  # (#kp, edgeH, edgeW)
            self.scales_n = np.zeros(shape, dtype=np.float32)  # (#kp, edgeH, edgeW)
        else:
            ta = np.zeros(self.target_accumulator.shape, dtype=np.float32)

        for pif in pifs:  # > #pifs -> (#kp, 4, upH, upW)
            for t, p, scale, n in zip(ta, pif, self.scales, self.scales_n):  # > #kp
                p = p[:, p[0] > self.v_threshold]  # (4, (p[0]:cls > 0.1)=(upH, upW)) -> (4, #pos)
                if min_scale:
                    p = p[:, p[3] > min_scale / stride]

                v, x, y, s = p  # > (cls, reg_x, reg_y, scale)
                x = x * stride
                y = y * stride
                s = s * stride
                # TOCHECK: v / 16 / #pifs
                scalar_square_add_gauss(t, x, y, s, v / self.pif_nn / len(pifs)) # `reg_xy + scales` -> disks of keypoints
                cumulative_average(scale, n, x, y, s, s, v)  # TOCHECK: cumulative average?

        if self.target_accumulator is None:
            self.target_accumulator = ta
        else:
            self.target_accumulator = np.maximum(ta, self.target_accumulator)
        # TOCHECK: target_intensities = disks of keypoints?
        LOG.debug('target_intensities %.3fs', time.perf_counter() - start)
        return self

    def fill_sequence(self, pifs, strides, min_scales): # > (#kp, (cls, reg_x, reg_y, scale), upH, upW)
        if len(pifs) == 10:
            for pif1, pif2, stride, min_scale in zip(pifs[:5], pifs[5:], strides, min_scales):
                self.fill_multiple([pif1, pif2], stride, min_scale=min_scale)
        else:  # <-
            for pif, stride, min_scale in zip(pifs, strides, min_scales):
                self.fill(pif, stride, min_scale=min_scale)

        return self


# def clip(v, minv, maxv):
#     return max(minv, min(maxv, v))
#
#
# def approx_exp(x):  # TOCHECK: compare w/ paper equation (3)?
#     if x > 2.0 or x < -2.0:
#         return 0.0
#     x = 1.0 + x / 8.0  #
#     x *= x
#     x *= x
#     x *= x
#     return x
#
#
# def scalar_square_add_gauss(field, x, y, sigma, v, truncate=2.0):
#     for i in range(x.shape[0]):  # > #pos
#         csigma = sigma[i]  # scale, max=2.0
#         csigma2 = csigma * csigma  # variance
#         cx = x[i]
#         cy = y[i]
#         cv = v[i]
#
#         minx = np.array((clip(cx - truncate * csigma, 0, field.shape[1] - 1)), dtype=np.intp)
#         maxx = np.array((clip(cx + truncate * csigma, minx + 1, field.shape[1])), dtype=np.intp)
#         miny = np.array((clip(cy - truncate * csigma, 0, field.shape[0] - 1)), dtype=np.intp)
#         maxy = np.array((clip(cy + truncate * csigma, miny + 1, field.shape[0])), dtype=np.intp)
#         for xx in range(minx, maxx):
#             deltax2 = (xx - cx)**2  # l2 norm
#             for yy in range(miny, maxy):
#                 deltay2 = (yy - cy)**2 # l2 norm
#                 vv = cv * approx_exp(-0.5 * (deltax2 + deltay2) / csigma2)  # TOCHECK: equation (3), a2 not used?
#                 field[yy, xx] += vv
#
#
# def cumulative_average(cuma, cumw, x, y, width, v, w):
#     for i in range(x.shape[0]):  # > #pos
#         cw = w[i]  # v
#         if cw <= 0.0:
#             continue
#
#         cv = v[i]  # scale
#         cx = x[i]
#         cy = y[i]
#         cwidth = width[i]  # scale
#
#         minx = np.array((clip(cx - cwidth, 0, cuma.shape[1] - 1)), dtype=np.intp)
#         maxx = np.array((clip(cx + cwidth, minx + 1, cuma.shape[1])), dtype=np.intp)
#         miny = np.array((clip(cy - cwidth, 0, cuma.shape[0] - 1)), dtype=np.intp)
#         maxy = np.array((clip(cy + cwidth, miny + 1, cuma.shape[0])), dtype=np.intp)
#         for xx in range(minx, maxx):
#             for yy in range(miny, maxy):
#                 # TOCHECK: (v * scale + scales[yy,xx] * scales_n[yy,xx]) / (scales[yy,xx] + v)
#                 cumw_yx = cumw[yy, xx]
#                 cuma_yx = cuma[yy, xx]
#                 cumwa_yx = cumw_yx * cuma_yx
#                 cwv = cw * cv
#                 cuma[yy, xx] = (cw * cv + cumw[yy, xx] * cuma[yy, xx]) / (cumw[yy, xx] + cw)
#                 # TOCHECK:
#                 cumw[yy, xx] += cw