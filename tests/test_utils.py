import unittest
import pytest
from pathlib import Path

import torch
import torch.nn as nn
import torchvision

from src import utils


class TestSaveAndLoadCheckpoint(unittest.TestCase):

    def test_save_and_load_checkpoint(self):
        model = torchvision.models.resnet18(pretrained=False)
        utils.save_checkpoint(model, epoch=100, filename='tmp.pth', save_arch=True)

        loaded_model = utils.load_model('tmp.pth')

        torch.testing.assert_allclose(model.conv1.weight, loaded_model.conv1.weight)

        model.conv1.weight = nn.Parameter(torch.zeros_like(model.conv1.weight))
        model = utils.load_checkpoint('tmp.pth', model=model)['model']

        assert (model.conv1.weight != 0).any()

    def tearDown(self):
        Path('tmp.pth').unlink()  # rm


# class TestAverageMeter(unittest.TestCase):
#
#     def test_average_meter(self):
#         raise NotImplementedError
#
#
# class TestCheckDuplicate(unittest.TestCase):
#
#     def test_check_duplicate(self):
#         raise NotImplemented
#
#
# from src.inplace_abn import InPlaceABNSync
# from src.sync_batchnorm import SynchronizedBatchNorm2d
# from src.modeling import resnet_csail
#
# @pytest.mark.parametrize('src_bn, dst_bn', [
#     (nn.BatchNorm2d, InPlaceABNSync),
#     (InPlaceABNSync, nn.BatchNorm2d),
#     (nn.BatchNorm2d, SynchronizedBatchNorm2d),
#     (SynchronizedBatchNorm2d, nn.BatchNorm2d),
#     (InPlaceABNSync, SynchronizedBatchNorm2d),
#     (SynchronizedBatchNorm2d, InPlaceABNSync),
# ])
# def test_replace_bn(src_bn, dst_bn):
#     model = resnet_csail.resnet18(pretrained=True, bn_module=src_bn)
#     w, b = model.bn1.weight, model.bn1.bias
#     utils.replace_bn(model, src_bn, dst_bn)
#
#     cnt_src_bn, cnt_dst_bn = 0, 0
#
#     for name, m in model.named_modules():
#         if name == 'bn1':
#             assert (w == m.weight).all() and (b == m.bias).all()
#         if isinstance(m, src_bn):
#             cnt_src_bn += 1
#         if isinstance(m, dst_bn):
#             cnt_dst_bn += 1
#
#     assert cnt_src_bn == 0 and cnt_dst_bn > 0
