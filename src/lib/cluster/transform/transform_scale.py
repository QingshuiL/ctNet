import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
import json
import os
import warnings
from copy import deepcopy
from math import inf
import random
PIL_RESIZE_MODE = {'bilinear': Image.BILINEAR, 'nearest': Image.NEAREST}

class ScaleMatch(object):
    """
        ScaleMatch face two problem when using:
            1) May generate too small scale, it will lead loss to NaN.
            2) May generate too big scale, it will lead out of memory.

            we find bigger batch size can ease 1) problem.
        there are four way to handle these problem:
            1) clip scale constraint to a specified scale_range
            2) change SM target distribute by scale mean and var
            3) use MonotonicityScaleMatch
            4) use chose scale as warm up scale

    """
    def __init__(self, distribute=None, sizes=None,              # param group 1
                 anno_file=None, bins=100, except_rate=-1.,      # param group 2
                 scale_range=(0., 2.), default_scale=1.0, max_sample_try=5, out_scale_deal='clip', use_log_bins=False,
                 mode='bilinear', debug_no_image_resize=False, debug_close_record=True):
        assert anno_file is not None or (distribute is not None and sizes is not None)
        if anno_file is not None:
            if except_rate < 0:
                except_rate = 1./ bins * 2
            distribute, sizes = ScaleMatch._get_distribute(json.load(open(anno_file))['annotations'], bins,
                                                           except_rate, use_log_bins)
        self.distri_cumsum = np.cumsum(distribute)
        self.sizes = sizes
        self.mode = PIL_RESIZE_MODE[mode]

        self.scale_range = scale_range   # scale_range[1] to avoid out of memory
        self.out_scale_deal = out_scale_deal
        assert out_scale_deal in ['clip', 'use_default_scale']
        self.max_sample_try = max_sample_try
        self.default_scale = default_scale

        self.fail_time = 0
        self.debug_no_image_resize = debug_no_image_resize
        self.debug_record = DebugScaleRecord(debug_close_record)

    @staticmethod
    def _get_distribute(annotations, bins=100, except_rate=0.1, use_log_bins=False, mu_sigma=(-1, -1)):
        """
        except_rate: to except len(annotations)*except_rate/2 abnormal points as head and tial bin
        """
        annos = [anno for anno in annotations if not anno['iscrowd']]                    # ?????? iscrowd ???0???
        if len(annos) > 0 and 'ignore' in annos[0]:                                      # ?????? ignore ???0???
            annos = [anno for anno in annos if not anno['ignore']]
        sizes = np.sqrt(np.array([anno['bbox'][2] * anno['bbox'][3] for anno in annos])) # ??????ann????????????
        sizes = sizes[sizes > 0]

        if mu_sigma[0] > 0 and mu_sigma[1] > 0:
            print('distribute(mu, sigma): ', np.mean(sizes), np.std(sizes), end='->')
            sizes = (sizes - np.mean(sizes)) / np.std(sizes)
            sizes = sizes * mu_sigma[1] + mu_sigma[0]
            print(np.mean(sizes), np.std(sizes))
            sizes = sizes.clip(1)

        if use_log_bins:
            sizes = np.log(sizes)
        sizes = np.sort(sizes)
        N = len(sizes)
        hist_sizes = sizes[int(N * except_rate / 2): int(N * (1 - except_rate / 2))]
        if except_rate > 0:
            c, s = np.histogram(hist_sizes, bins=bins-2)         # ??????98??? c??????????????? s??????????????????
            c = np.array([int(N * except_rate / 2)] + c.tolist() + [N - int(N * (1 - except_rate / 2))])
            s = [sizes[0]] + s.tolist() + [sizes[-1]]
            s = np.array(s)
        else:
            c, s = np.histogram(hist_sizes, bins=bins)
        c = c / len(sizes)
        if use_log_bins:
            s = np.exp(s)
        return c, s

    def _sample_by_distribute(self):
        r = np.random.uniform()
        idx = np.nonzero(r <= self.distri_cumsum + 1e-6)[0][0]
        mins, maxs = self.sizes[idx], self.sizes[idx + 1]
        ir = np.random.uniform()
        return (maxs - mins) * ir + mins

    def default_scale_deal(self, image, target):
        scale = self.default_scale
        # resize bbox mean size to our want size
        size = int(round(scale * image.height)), int(round(scale * image.width))
        target = target.resize((size[1], size[0]))
        image = F.resize(image, size, self.mode)
        return image, target

    def __call__(self, image, target):
        if len(target.bbox) == 0:
            return self.default_scale_deal(image, target)

        # record old target info
        old_target = deepcopy(target)
        old_mode = target.mode

        # cal mean size of image's bbox
        boxes = target.convert('xywh').bbox.cpu().numpy()
        sizes = np.sqrt(boxes[:, 2] * boxes[:, 3])
        sizes = sizes[sizes > 0]
        src_size = np.exp(np.log(sizes).mean())
        # src_size = sizes.mean()

        # sample a size respect to target distribute
        # For memory stable, we set scale range.
        scale = self.default_scale
        for try_i in range(self.max_sample_try):
            dst_size = self._sample_by_distribute()
            _scale = dst_size / src_size
            if self.scale_range[1] > _scale > self.scale_range[0]:
                scale = _scale
                break                                                  # last time forget
        self.debug_record(_scale)
        if self.out_scale_deal == 'clip':
            if _scale >= self.scale_range[1]:
                scale = self.scale_range[1]     # note 1
            elif _scale <= self.scale_range[0]:
                scale = self.scale_range[0]   # note 1

        # resize bbox mean size to our want size
        size = int(round(scale * image.height)), int(round(scale * image.width))
        target = target.convert(old_mode)
        target = target.resize((size[1], size[0]))

        # remove too tiny size to avoid unstable loss
        if len(target.bbox) > 0:
            target = target[(target.bbox[:, 2] - target.bbox[:, 0] + 1) >= 2]
            target = target[(target.bbox[:, 3] - target.bbox[:, 1] + 1) >= 2]

        if len(target.bbox) == 0:
            self.fail_time += 1
            if self.fail_time % 1 == 0:
                warnings.warn("Scale Matching failed for {} times, you may need to change the mean to min. "
                              "dst_size is {}, src_size is {}, sizes is {}".format(self.fail_time, dst_size, src_size, sizes))
            return self.default_scale_deal(image, old_target)
        if not self.debug_no_image_resize:
            image = F.resize(image, size, self.mode)
        return image, target



class MonotonicityScaleMatch(object):
    def __init__(self, 
                 src_anno_file, 
                 dst_anno_file, 
                 bins=100, 
                 except_rate=-1.,
                 scale_range=(0., 2.), 
                 default_scale=1.0, 
                 out_scale_deal='clip',
                 use_log_bins=False, 
                 mode='bilinear', 
                 mu_sigma=(-1, -1),
                 debug_no_image_resize=False, 
                 debug_close_record=False):

        if except_rate < 0:
            except_rate = 1. / bins * 2
        dst_distri, dst_sizes = ScaleMatch._get_distribute(json.load(open(dst_anno_file))['annotations'],
                                                           bins, except_rate, use_log_bins, mu_sigma)
        dst_distri_cumsum = np.cumsum(dst_distri)           # ??????????????????????????????
        src_sizes = MonotonicityScaleMatch.match_distribute(json.load(open(src_anno_file))['annotations'],
                                                            dst_distri_cumsum)
        self.src_sizes = src_sizes
        self.dst_sizes = dst_sizes

        self.default_scale = default_scale
        self.mode = PIL_RESIZE_MODE[mode]
        self.fail_time = 0
        self.scale_range = scale_range   # scale_range[1] to avoid out of memory
        self.out_scale_deal = out_scale_deal
        assert out_scale_deal in ['clip', 'use_default_scale']

        self.debug_no_image_resize = debug_no_image_resize
        self.debug_record = DebugScaleRecord(debug_close_record)

    @staticmethod
    def match_distribute(src_annotations, dst_distri_cumsum):
        annos = [anno for anno in src_annotations if not anno['iscrowd']]
        if len(annos) > 0 and 'ignore' in annos[0]:
            annos = [anno for anno in annos if not anno['ignore']]
        sizes = np.sqrt(np.array([anno['bbox'][2] * anno['bbox'][3] for anno in annos]))
        sizes = sizes[sizes > 0]
        sizes = np.sort(sizes)
        print('-------------mean:',np.mean(sizes))
        N = len(sizes)
        src_sizes = [sizes[0]]
        for p_sum in dst_distri_cumsum:
            src_sizes.append(sizes[min(int(p_sum * N), N-1)])
        if src_sizes[-1] < sizes[-1]:
            src_sizes[-1] = sizes[-1]
        return np.array(src_sizes)

    def _sample_by_distribute(self, src_size):
        bin_i = np.nonzero(src_size <= self.src_sizes[1:] + 1e-6)[0][0]
        dst_bin_d = self.dst_sizes[bin_i + 1] - self.dst_sizes[bin_i]
        src_bin_d = self.src_sizes[bin_i + 1] - self.src_sizes[bin_i]
        dst_size = (src_size - self.src_sizes[bin_i]) / src_bin_d * dst_bin_d + self.dst_sizes[bin_i]
        return dst_size

    def default_scale_deal(self, image, target):
        scale = self.default_scale
        # resize bbox mean size to our want size
        size = int(round(scale * image.height)), int(round(scale * image.width))
        target = target.resize((size[1], size[0]))
        if not self.debug_no_image_resize:
            image = F.resize(image, size, self.mode)
        return image, target

    def __call__(self, image, target):                    # ????????????????????????????????????????????????
        if len(target.bbox) == 0:
            return self.default_scale_deal(image, target)

        # record old target info
        old_target = deepcopy(target)
        old_mode = target.mode

        # cal mean size of image's bbox
        boxes = target.convert('xywh').bbox.cpu().numpy()
        sizes = np.sqrt(boxes[:, 2] * boxes[:, 3])
        sizes = sizes[sizes > 0]
        src_size = np.exp(np.log(sizes).mean())
        # src_size = sizes.mean()

        # sample a size respect to target distribute
        # For memory stable, we set scale range.
        dst_size = self._sample_by_distribute(src_size)
        scale = dst_size / src_size
        self.debug_record(scale)
        if self.out_scale_deal == 'clip':
            if scale >= self.scale_range[1]:
                scale = self.scale_range[1]     # note 1
            elif scale <= self.scale_range[0]:
                scale = self.scale_range[0]   # note 1
        else:
            if scale >= self.scale_range[1] or scale <= self.scale_range[0]:
                scale = self.default_scale

        # resize bbox mean size to our want size
        print('transform scale is:', scale)
        size = int(round(scale * image.height)), int(round(scale * image.width))
        target = target.convert(old_mode)
        target = target.resize((size[1], size[0]))

        # remove too tiny size to avoid unstable loss
        if len(target.bbox) > 0:
            target = target[(target.bbox[:, 2] - target.bbox[:, 0] + 1) >= 2]
            target = target[(target.bbox[:, 3] - target.bbox[:, 1] + 1) >= 2]

        if len(target.bbox) == 0:
            self.fail_time += 1
            if self.fail_time % 1 == 0:
                warnings.warn("Scale Matching failed for {} times, you may need to change the mean to min. "
                              "dst_size is {}, src_size is {}, sizes is {}".format(self.fail_time, dst_size, src_size, sizes))
            return self.default_scale_deal(image, old_target)
        if not self.debug_no_image_resize:
            image = F.resize(image, size, self.mode)
        return image, target



class GaussianScaleMatch(MonotonicityScaleMatch):
    def __init__(self, src_anno_file, mu_sigma, bins=100, except_rate=-1.,
                 scale_range=(0., 2.), default_scale=1.0, out_scale_deal='clip',
                 use_log_bins=False, mode='bilinear', standard_gaussian_sample_file=None,
                 use_size_in_image=True, min_size=0,
                 debug_no_image_resize=False, debug_close_record=False):
        """
        1. GaussianScaleMatch use equal area histogram to split bin, not equal x-distance, so [except_rate] are removed.
        2. _get_gaussain_distribute can get simulate gaussian distribute, can can set [min_size] to constraint it.
        3. use log mean size of objects in each image as src distribute, not log size of each object
        :param src_anno_file:
        :param mu_sigma:
        :param bins:
        :param except_rate:
        :param scale_range:
        :param default_scale:
        :param out_scale_deal:
        :param use_log_bins:
        :param mode:
        :param standard_gaussian_sample_file:
        :param debug_no_image_resize:
        :param debug_close_record:
        """
        assert out_scale_deal in ['clip', 'use_default_scale']
        assert use_log_bins, "GaussianScaleMatch need USE_LOG_BINS set to True."
        assert except_rate <= 0, 'GaussianScaleMatch need except_rate < 0'

        if except_rate < 0:
            except_rate = 1. / bins * 2
        mu, sigma = mu_sigma
        dst_distri, dst_sizes = GaussianScaleMatch._get_gaussain_distribute(mu, sigma, bins, except_rate, use_log_bins,
                                                                            standard_gaussian_sample_file, min_size)
        dst_distri_cumsum = np.cumsum(dst_distri)
        src_sizes = GaussianScaleMatch.match_distribute(json.load(open(src_anno_file))['annotations'],
                                                        dst_distri_cumsum, use_size_in_image)
        self.src_sizes = src_sizes
        self.dst_sizes = dst_sizes

        self.default_scale = default_scale
        self.mode = PIL_RESIZE_MODE[mode]
        self.fail_time = 0
        self.scale_range = scale_range   # scale_range[1] to avoid out of memory
        self.out_scale_deal = out_scale_deal

        self.debug_no_image_resize = debug_no_image_resize
        self.debug_record = DebugScaleRecord(debug_close_record)

    @staticmethod
    def _get_gaussain_distribute(mu, sigma, bins=100, except_rate=0.1, use_log_bins=False,
                                 standard_gaussian_sample_file=None, min_size=0):
        """
        except_rate: to except len(annotations)*except_rate/2 abnormal points as head and tial bin
        """
        x = np.load(standard_gaussian_sample_file)
        sizes = x * sigma + mu
        if min_size >= 0:
            sizes = sizes[sizes > min_size]
        sizes = np.sort(sizes)
        N = len(sizes)
        # hist_sizes = sizes[int(N * except_rate / 2): int(N * (1 - except_rate / 2))]
        # if except_rate > 0:
        #     c, s = np.histogram(hist_sizes, bins=bins-2)
        #     c = np.array([int(N * except_rate / 2)] + c.tolist() + [N - int(N * (1 - except_rate / 2))])
        #     s = [sizes[0]] + s.tolist() + [sizes[-1]]
        #     s = np.array(s)
        # else:
        #     c, s = np.histogram(hist_sizes, bins=bins)
        from math import ceil
        step = int(ceil(N / bins))
        last_c = N - step * (bins - 1)
        s = np.array(sizes[::step].tolist() + [sizes[-1]])
        c = np.array([step] * (bins - 1) + [last_c])

        c = c / len(sizes)
        if use_log_bins:
            s = np.exp(s)
        return c, s

    def _sample_by_distribute(self, src_size):
        bin_i = np.nonzero(src_size <= self.src_sizes[1:] + 1e-6)[0][0]
        dst_bin_d = np.log(self.dst_sizes[bin_i + 1]) - np.log(self.dst_sizes[bin_i])
        src_bin_d = np.log(self.src_sizes[bin_i + 1]) - np.log(self.src_sizes[bin_i])
        dst_size = np.exp(
            (np.log(src_size) - np.log(self.src_sizes[bin_i])) / src_bin_d * dst_bin_d + np.log(self.dst_sizes[bin_i])
        )
        return dst_size

    @staticmethod
    def match_distribute(src_annotations, dst_distri_cumsum, use_size_in_image=True):
        def get_json_sizes(annos):
            annos = [anno for anno in annos if not anno['iscrowd']]
            if len(annos) > 0 and 'ignore' in annos[0]:
                annos = [anno for anno in annos if not anno['ignore']]
            sizes = np.sqrt(np.array([anno['bbox'][2] * anno['bbox'][3] for anno in annos]))
            sizes = sizes[sizes > 0]
            return sizes

        def get_im2annos(annotations):
            im2annos = {}
            for anno in annotations:
                iid = anno['image_id']
                if iid in im2annos:
                    im2annos[iid].append(anno)
                else:
                    im2annos[iid] = [anno]
            return im2annos

        def get_json_sizes_in_image(annotations):
            im2annos = get_im2annos(annotations)
            _sizes = []
            for iid, annos in im2annos.items():
                annos = [anno for anno in annos if not anno['iscrowd']]
                if len(annos) > 0 and 'ignore' in annos[0]:
                    annos = [anno for anno in annos if not anno['ignore']]
                sizes = np.sqrt(np.array([anno['bbox'][2] * anno['bbox'][3] for anno in annos]))
                sizes = sizes[sizes > 0]
                size = np.exp(np.log(sizes).mean())
                _sizes.append(size)
            return _sizes
        if use_size_in_image:
            sizes = get_json_sizes_in_image(src_annotations)
        else:
            sizes = get_json_sizes(src_annotations)
        sizes = np.sort(sizes)
        N = len(sizes)
        src_sizes = [sizes[0]]
        for p_sum in dst_distri_cumsum:
            src_sizes.append(sizes[min(int(p_sum * N), N-1)])
        if src_sizes[-1] < sizes[-1]:
            src_sizes[-1] = sizes[-1]
        return np.array(src_sizes)


class GaussTransfrom(object):
    def __init__(self,             
                 mu=79.5,     
                 sigma=5,
                 scale_range=(0.5, 3.), 
                 default_scale=1.0,  
                 out_scale_deal='clip', 
                 mode='bilinear', 
                 use_size_in_image=False,
                 debug_no_image_resize=False, 
                 debug_close_record=True):
        self.scale_range = scale_range
        self.mu = mu
        self.sigma = sigma
        self.default_scale = default_scale
        self.mode = PIL_RESIZE_MODE[mode]
        self.fail_time = 0
        self.scale_range = scale_range   # scale_range[1] to avoid out of memory
        self.out_scale_deal = out_scale_deal
        self.debug_no_image_resize = debug_no_image_resize
        self.debug_record = DebugScaleRecord(debug_close_record)
    
    def _sample_scale(self, src_size):
        nol_size = np.random.normal(self.mu, self.sigma)
        scale = nol_size / src_size
        return scale, nol_size

    def default_scale_deal(self, image, target):
        scale = self.default_scale
        # resize bbox mean size to our want size
        size = int(round(scale * image.height)), int(round(scale * image.width))
        target = target.resize((size[1], size[0]))
        image = F.resize(image, size, self.mode)
        return image, target, scale

    def __call__(self, image, target):
        # print(len(target.bbox))
        if len(target.bbox) == 0:
            return self.default_scale_deal(image, target)

        # record old target info
        old_target = deepcopy(target)
        old_mode = target.mode

        # cal mean size of image's bbox
        boxes = target.convert('xywh').bbox.cpu().numpy()
        sizes = np.sqrt(boxes[:, 2] * boxes[:, 3])
        sizes = sizes[sizes > 0]
        # src_size = np.exp(np.log(sizes).mean())
        src_size = sizes.mean()

        scale, dst_size = self._sample_scale(src_size)

        self.debug_record(scale)
        if self.out_scale_deal == 'clip':
            if scale >= self.scale_range[1]:
                scale = self.scale_range[1]     # note 1
            elif scale <= self.scale_range[0]:
                scale = self.scale_range[0]   # note 1

        # resize bbox mean size to our want size
        size = int(round(scale * image.height)), int(round(scale * image.width))
        target = target.convert('xyxy')
        target = target.resize((size[1], size[0]))

        # remove too tiny size to avoid unstable loss
        if len(target.bbox) > 0:
                target = target[(target.bbox[:, 2] - target.bbox[:, 0] + 1) >= 2]
                target = target[(target.bbox[:, 3] - target.bbox[:, 1] + 1) >= 2]

        # print(len(target.bbox))
        if len(target.bbox) == 0:
            self.fail_time += 1
            if self.fail_time % 1 == 0:
                warnings.warn("Scale Matching failed for {} times, you may need to change the mean to min. "
                              "dst_size is {}, src_size is {}, sizes is {}".format(self.fail_time, dst_size, src_size, sizes))
            return self.default_scale_deal(image, old_target)

        image = F.resize(image, size, self.mode)
        print('The transform scale:{0} and szie:{1}'.format(scale, size))
        return image, target, scale


class DebugScaleRecord(object):
    def __init__(self, close=False):
        self.debug_record_scales = [inf, -inf]
        self.iters = 0
        self.close = close

    def __call__(self, scale):
        if self.close: return
        self.iters += 1
        last_record_scales = deepcopy(self.debug_record_scales)
        update = False
        if scale > self.debug_record_scales[1] + 1e-2:
            self.debug_record_scales[1] = scale
            update = True
        if scale < self.debug_record_scales[0] - 1e-2:
            self.debug_record_scales[0] = scale
            update = True
        if (update and self.iters > 1000) or self.iters == 1000:
            warnings.warn('update record scale {} -> {}'.format(last_record_scales, self.debug_record_scales))

'''
if __name__ == "__main__":
    root = r'C:\\Users\\10295\\CV\\CenterNet-master\\data'
    vis_dir = os.path.join(root,'visdrone2019')
    coco_dir = os.path.join(root,'coco2017')
    src_anno_file = os.path.join(vis_dir,'annotations','train.json')
    dst_anno_file = os.path.join(coco_dir,'annotations','instances_train2017.json')
    image_dir = os.path.join(vis_dir,'images')
    sm = MonotonicityScaleMatch(
                src_anno_file, 
                dst_anno_file, 
                bins=100, 
                except_rate=-1.,
                scale_range=(0., 2.), 
                default_scale=1.0, 
                out_scale_deal='clip',
                use_log_bins=False, 
                mode='bilinear', 
                mu_sigma=(-1, -1),
                debug_no_image_resize=False, 
                debug_close_record=False)
'''

    

                 
