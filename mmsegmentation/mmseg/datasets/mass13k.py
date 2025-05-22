from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class MaSSDataset(BaseSegDataset):
    """ MaSS dataset
    
    In segmentation map annotation for MaSS, 0 stands for background,
    which is not included in 6 categories, ``reduce_zero_label`` is fixed to True
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    METAINFO = dict(
        classes=('background','person','building', 'tree','ground', 'sky','water'),
        palette=[
            [0,0,0],
            [255,0,0], 
            [64,34,64],
            [0,255,0],
            [93,64,0],
            [0,0,255], 
            [0,255,255]
            ])

    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)