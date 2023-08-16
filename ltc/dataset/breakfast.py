from ltc.dataset.video_dataset import VideoDataset
import ltc.utils.logging as logging

logger = logging.get_logger(__name__)


class Breakfast(VideoDataset):
    def __init__(self, cfg, mode):
        super(Breakfast, self).__init__(cfg, mode)
        logger.info("Constructing Breakfast {} dataset with {} videos.".format(mode, self._dataset_size))
