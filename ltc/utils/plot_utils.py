from typing import Tuple, List, Any, Dict, Iterable
import random
from colorsys import hsv_to_rgb

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl

import cv2
import numpy as np


def add_subfigure(gt: np.array,
                  pred:np.array,
                  video_info: dict,
                  fig_grid: gridspec.GridSpec,
                  fig: plt.Figure):
    grid = fig_grid.subgridspec(nrows=2, ncols=1)
    ax1 = fig.add_subplot(grid[0])
    prop = mpl.font_manager.FontProperties(family='sans-serif',
                                           size=7.5,
                                           weight="normal",
                                           style="normal")
    ax1.imshow(gt)
    ax1.set_yticks([])
    ax1.set_xticks([])
    ax1.set_ylabel("GT", rotation=45, fontsize=9)
    ax1.set_title(f"{video_info['video_name']}, len={video_info['len']}", fontproperties=prop, pad=0.5)
    ax2 = fig.add_subplot(grid[1])

    ax2.imshow(pred)
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_ylabel("Pred", rotation=45, labelpad=6, fontsize=9)
    fig.subplots_adjust(wspace=0.00, hspace=0.0)


def create_result_fig(results: List):
    num_data = len(results)
    fig = plt.figure(figsize=(12, 27))
    half_num_data = (num_data // 2)
    gs0 = gridspec.GridSpec(nrows=half_num_data,
                            ncols=2,
                            figure=fig,
                            wspace=0.00,
                            hspace=0.34,
                            left=0,
                            right=1.0,
                            bottom=0.00,
                            top=0.95)
    size = 256
    for col, result in enumerate([results[:half_num_data], results[half_num_data:]]) :
        for idx, data in enumerate(result):
            #     size = int(relative_ratio[idx] * 256)

            gt = cv2.resize(data['gt'], dsize=(size, 20), interpolation=cv2.INTER_NEAREST)
            pred = cv2.resize(data['pred'], dsize=(size, 20), interpolation=cv2.INTER_NEAREST)
            add_subfigure(gt, pred, data, gs0[idx, col], fig)

    return fig


def generate_image_for_segmentation(
    labels: List[int],
    lengths: List[int],
    colors: np.ndarray,
    label_name_mapping: Dict[int, str] = None,
    white_label: Iterable[int] = 0,
    split_width: int = 0,
    height: int = 50,
) -> np.ndarray:
    # todo: make the image size fixed and not a function of the number os splits.
    num_splits = 1 + len(labels)
    width = num_splits * split_width + sum(lengths)
    result = np.zeros(shape=(height, width, 3), dtype=np.uint8)
    for i in range(len(labels)):
        start = (i + 1) * split_width + sum(lengths[:i])
        end = start + lengths[i] + 1
        color = colors[labels[i]]

        if labels[i] in white_label:
            color = np.full(3, fill_value=120, dtype=np.uint8)
        result[:, start:end, :] = color

        if label_name_mapping is not None:
            pos_x = start + 2
            pos_y = int(height / 2)
            cv2.putText(
                result,
                label_name_mapping[labels[i]],
                (pos_x, pos_y),
                cv2.FONT_HERSHEY_COMPLEX,
                0.4,
                (0, 0, 0),
            )

    return result


def generate_n_colors(n: int, shuffle: bool = False, random_seed:int = 0) -> np.ndarray:
    assert n > 0

    list_of_colors = []
    sat = 0.5
    val = 0.65
    for i in range(n):
        hue = i / n
        list_of_colors.append([255*x for x in hsv_to_rgb(hue, sat, val)])

    if shuffle:
        random.seed(random_seed)
        random.shuffle(list_of_colors)

    return np.array(list_of_colors)


def generate_distinct_colors(n: int,
                             random_seed: int = 0,
                             exclude_colors=[(0, 0, 0), (1, 1, 1), (0.972, 0.972, 0.972)]) -> np.ndarray:
    from distinctipy import distinctipy, get_rgb256
    from colorsys import hsv_to_rgb, rgb_to_hsv
    colors = distinctipy.get_colors(n,
                                    pastel_factor=0.2,
                                    exclude_colors=exclude_colors,
                                    rng=random_seed)
    sat = 0.55
    colors = [get_rgb256(c) for c in colors]
    hsv_colors = [rgb_to_hsv(*x) for x in colors]
    hsv_colors = [(x[0], sat, x[-1]) for x in hsv_colors]
    rgb_colors = [hsv_to_rgb(*x) for x in hsv_colors]
    return rgb_colors


def summarize_list(the_list: List[Any]) -> Tuple[List[Any], List[int]]:
    """
    Given a list of items, it summarizes them in a way that no two neighboring values are the same.
    It also returns the size of each section.
    e.g. [4, 5, 5, 6] -> [4, 5, 6], [1, 2, 1]
    """
    summary = []
    lens = []
    if len(the_list) > 0:
        current = the_list[0]
        summary.append(current)
        lens.append(1)
        for item in the_list[1:]:
            if item != current:
                current = item
                summary.append(item)
                lens.append(1)
            else:
                lens[-1] += 1
    return summary, lens


def unsummarize_list(labels: List[int], lengths: List[int]) -> List[int]:
    """
    Does the reverse of summarize list. You give it a list of segment labels and their lengths and it returns the full
    labels for the full sequence.
    e.g. ([4, 5, 6], [1, 2, 1]) -> [4, 5, 5, 6]
    """
    assert len(labels) == len(lengths)

    the_sequence = []
    for label, length in zip(labels, lengths):
        the_sequence.extend([label] * length)

    return the_sequence
