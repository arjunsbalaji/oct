# default_exp runs

import sys

sys.path.append('/workspace/oct')

from oct.startup import *
from model import CapsNet
import numpy as np
import mlflow
from fastai.vision import *
import mlflow.pytorch as MLPY
from fastai.utils.mem import gpu_mem_get_all

gpu_mem_get_all()

### Configuration Setup

name = 'FCN'

config_dict = loadConfigJSONToDict('configCAPS_APPresnet18.json')
config_dict['LEARNER']['lr']= 0.001
config_dict['LEARNER']['bs'] = 16
config_dict['LEARNER']['epochs'] = 30
config_dict['LEARNER']['runsave_dir'] = '/workspace/oct_ca_seg/runsaves/'
config_dict['MODEL'] = 'FASTAI DEEPLABv3 NO PRETRAIN'
config = DeepConfig(config_dict)

config_dict

metrics = [sens, spec, dice, acc]

def saveConfigRun(dictiontary, run_dir, name):
    with open(run_dir/name, 'w') as file:
        json.dump(dictiontary, file)

## Dataset

cocodata_path = Path('/workspace/oct_ca_seg/COCOdata/')
train_path = cocodata_path/'train/images'
valid_path = cocodata_path/'valid/images'
test_path = cocodata_path/'test/images'

### For complete dataset

fn_get_y = lambda image_name: Path(image_name).parent.parent/('labels/'+Path(image_name).name)
codes = np.loadtxt(cocodata_path/'codes.txt', dtype=str)
tfms = get_transforms()
src = (SegCustomItemList
       .from_folder(cocodata_path, recurse=True, extensions='.jpg')
       .filter_by_func(lambda fname: Path(fname).parent.name == 'images', )
       .split_by_folder('train', 'valid')
       .label_from_func(fn_get_y, classes=codes))
src.transform(tfms, tfm_y=True, size=config.LEARNER.img_size)
data = src.databunch(cocodata_path,
                     bs=config.LEARNER.bs,
                     val_bs=2*config.LEARNER.bs,
                     num_workers = config.LEARNER.num_workers)
stats = [torch.tensor([0.2190, 0.1984, 0.1928]), torch.tensor([0.0645, 0.0473, 0.0434])]
data.normalize(stats);
data.c_in, data.c_out = 3, 2

### For converting Validation set into a mini set to experiment on
'''
fn_get_y = lambda image_name: Path(image_name).parent.parent/('labels/'+Path(image_name).name)
codes = np.loadtxt(cocodata_path/'codes.txt', dtype=str)
tfms = get_transforms()
src = (SegCustomItemList
       .from_folder(test_path, recurse=True, extensions='.jpg')
       .filter_by_func(lambda fname: Path(fname).parent.name == 'images', )
       .split_by_rand_pct(0.9)
       .label_from_func(fn_get_y, classes=codes))
src.transform(tfms, tfm_y=True, size =config.LEARNER.img_size)
data = src.databunch(test_path,
                     bs=config.LEARNER.bs,
                     val_bs=2*config.LEARNER.bs,
                     num_workers = config.LEARNER.num_workers)
stats = [torch.tensor([0.2190, 0.1984, 0.1928]), torch.tensor([0.0645, 0.0473, 0.0434])]
data.normalize(stats);
data.c_in, data.c_out = 3, 2
'''
### Fastai FCN

import torchvision

fcn =torchvision.models.segmentation.fcn_resnet50(pretrained=False, num_classes=2, progress=False).cuda()

class FCN(torch.nn.Module):
    def __init__(self, model_base):
        super(FCN, self).__init__()
        self.model = model_base
        self.name = 'FCN'
    def forward(self, x):
        x = self.model(x)['out']
        return x

FCNetwork = FCN(fcn)

#torchvision.models.segmentation.fcn_resnet50(pretrained=False, num_classes=2, progress=False)

run_dir = config.LEARNER.runsave_dir+'/'+name
#os.mkdir(run_dir)
exp_name = 'fastai_dl3'
mlflow_CB = partial(MLFlowTracker,
                    exp_name=exp_name,
                    uri='file:/workspace/oct_ca_seg/runsaves/fastai_experiments/mlruns/',
                    params=config.config_dict,
                    log_model=True,
                    nb_path="/workspace/oct_ca_seg/oct/02_caps.ipynb")
learner = Learner(data = data,
                  model=FCNetwork,
                  metrics = metrics,
                  callback_fns=mlflow_CB)

with mlflow.start_run():
    learner.fit_one_cycle(1, slice(config.LEARNER.lr), pct_start=0.9)
    MLPY.save_model(learner.model, run_dir+'/model')
    save_all_results(learner, run_dir, exp_name)
    saveConfigRun(config.config_dict, run_dir=Path(run_dir), name = 'configFCN__bs16_epochs30_lr0.001.json')



