"""Decoder for pif-paf fields."""

import logging
import time

from openpifpaf.decoder import generator
from openpifpaf.decoder.paf_scored import PafScored
from openpifpaf.decoder.pif_hr import PifHr
from openpifpaf.decoder.pif_seeds import PifSeeds
from openpifpaf.decoder.utils import normalize_pif, normalize_paf

LOG = logging.getLogger(__name__)


class PifPaf(object):
    force_complete = True
    connection_method = 'blend'
    fixed_b = None
    pif_fixed_scale = None
    paf_th = 0.1

    def __init__(self, stride, *,
                 keypoints,
                 skeleton,
                 pif_index=0, paf_index=1,
                 pif_min_scale=0.0,
                 paf_min_distance=0.0,
                 paf_max_distance=None,
                 seed_threshold=0.2,
                 debug_visualizer=None):
        self.strides = stride
        self.pif_indices = pif_index
        self.paf_indices = paf_index
        self.pif_min_scales = pif_min_scale
        self.paf_min_distances = paf_min_distance
        self.paf_max_distances = paf_max_distance
        if not isinstance(self.strides, (list, tuple)):  # <-
            self.strides = [self.strides]  # > [8]
            self.pif_indices = [self.pif_indices]  # > [0]
            self.paf_indices = [self.paf_indices]  # > [1]
        if not isinstance(self.pif_min_scales, (list, tuple)):  # <-
            self.pif_min_scales = [self.pif_min_scales for _ in self.strides]  # > [0.0]
        if not isinstance(self.paf_min_distances, (list, tuple)):  # <-
            self.paf_min_distances = [self.paf_min_distances for _ in self.strides]  # > [0.0]
        if not isinstance(self.paf_max_distances, (list, tuple)):  # <-
            self.paf_max_distances = [self.paf_max_distances for _ in self.strides]  # > [None]
        assert len(self.strides) == len(self.pif_indices)
        assert len(self.strides) == len(self.paf_indices)
        assert len(self.strides) == len(self.pif_min_scales)
        assert len(self.strides) == len(self.paf_min_distances)
        assert len(self.strides) == len(self.paf_max_distances)

        self.keypoints = keypoints  # > names of keypoints
        self.skeleton = skeleton  # > keypoint pairs

        self.seed_threshold = seed_threshold  # > Test: 0.2
        self.debug_visualizer = debug_visualizer

        self.pif_nn = 16
        self.paf_nn = 1 if self.connection_method == 'max' else 35

    def __call__(self, fields, initial_annotations=None):
        start = time.perf_counter()

        if self.debug_visualizer:  # TOCHECK: how to debug visualizer?
            for stride, pif_i in zip(self.strides, self.pif_indices):
                self.debug_visualizer.pif_raw(fields[pif_i], stride)
            for stride, paf_i in zip(self.strides, self.paf_indices):
                self.debug_visualizer.paf_raw(fields[paf_i], stride, reg_components=3)

        # normalize
        normalized_pifs = [normalize_pif(*fields[pif_i], fixed_scale=self.pif_fixed_scale) # > None
                           for pif_i in self.pif_indices]  # > pif head idx
        normalized_pafs = [normalize_paf(*fields[paf_i], fixed_b=self.fixed_b)
                           for paf_i in self.paf_indices]

        # PifHr
        pifhr = PifHr(self.pif_nn)  # `pif_nn`: 16
        pifhr.fill_sequence(normalized_pifs, self.strides, self.pif_min_scales)  # `strides`: [8], `min_scales`: [0.0]

        # PifSeed
        seeds = PifSeeds(pifhr.target_accumulator, self.seed_threshold, # 0.2
                         debug_visualizer=self.debug_visualizer)
        seeds.fill_sequence(normalized_pifs, self.strides, self.pif_min_scales)  # `strides`: [8], `min_scales`: [0.0]

        # PafScored: `pifhr.targets` = `pifhr.target_accumulator` <= 1.0
        paf_scored = PafScored(pifhr.targets, self.skeleton, score_th=self.paf_th)  # `paf_th`: 0.1
        paf_scored.fill_sequence(  # min_dist: 0., max_dist: None
            normalized_pafs, self.strides, self.paf_min_distances, self.paf_max_distances)
        # TOCHECK: `greedy` vs. `dijkstra` vs. ` weighted A*`
        gen = generator.Greedy(
            pifhr, paf_scored, seeds,
            seed_threshold=self.seed_threshold,  # 0.2
            connection_method=self.connection_method,  # blend
            paf_nn=self.paf_nn,  # 35
            paf_th=self.paf_th,  # 0.1
            keypoints=self.keypoints,  # names of keypoints
            skeleton=self.skeleton,  # pairs
            debug_visualizer=self.debug_visualizer,
        )

        annotations = gen.annotations(initial_annotations=initial_annotations) # None -> (#obj, Ann)
        # TOCHECK: force_complete?
        if self.force_complete:  # <-
            # > `pifhr.targets`: (#kp, edgeH, edgeW)
            gen.paf_scored = PafScored(pifhr.targets, self.skeleton, score_th=0.0001)
            gen.paf_scored.fill_sequence(
                normalized_pafs, self.strides, self.paf_min_distances, self.paf_max_distances)
            annotations = gen.complete_annotations(annotations)

        LOG.debug('annotations %d, %.3fs', len(annotations), time.perf_counter() - start)
        return annotations
