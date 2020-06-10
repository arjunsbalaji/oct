import sys

sys.path.append('/workspace/oct_ca_seg/oct/')

from oct.startup import *
from model import CapsNet
import numpy as np
import mlflow
from fastai.vision import *
import mlflow.pytorch as MLPY


try:
    epochs = int(sys.argv[2])
    bs = int(sys.argv[3])
except:
    epochs=1
    bs=1

### Configuration Setup
config_dict = loadConfigJSONToDict('init_config.json')
config_dict['LEARNER']['bs'] = bs
config_dict['LEARNER']['epochs'] = epochs
config_dict['LEARNER']['img_size'] = 256
config_dict['LEARNER']['lr']= 0.0001
config_dict['LEARNER']['num_workers'] = 1
config_dict['MODEL']['dims1'] = 16
config_dict['MODEL']['dims2'] = 32
config_dict['MODEL']['dims3'] = 64
config_dict['MODEL']['maps1'] = 8
config_dict['MODEL']['maps2'] = 12
config_dict['MODEL']['maps3'] = 16
config_dict['MODEL']['f1dims'] = 8
config_dict['MODEL']['f2dims'] = 16
config_dict['MODEL']['f1maps'] = 2
config_dict['MODEL']['f2maps'] = 2 #this should be number of classes. background, lumen => 2 etc.
config = DeepConfig(config_dict)

saveDictToConfigJSON(config.config_dict, 'configUNET.json')

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

### For converting Validation set into a mini set to experiment on

### Runs

metrics = [sens, spec, dice, my_Dice_Loss, acc]

exp_name = 'FASTAI UNET'
mlflow_CB = partial(MLFlowTracker,
                    exp_name=exp_name,
                    uri='file:/workspace/oct_ca_seg/runsaves/fastai_experiments/mlruns/',
                    params=config.config_dict,
                    nb_path="/workspace/oct_ca_seg/oct/02_runs.ipynb")
with mlflow.start_run():
    learner = unet_learner(data,
                           models.resnet18,
                           pretrained=False,
                           y_range=[0,1],
                           metrics=metrics,
                           callback_fns=[mlflow_CB])
    learner.fit_one_cycle(config.LEARNER.epochs, slice(config.LEARNER.lr), pct_start=0.8)
    MLPY.log_model(learner.model, '/workspace/oct_ca_seg/runsaves/fastai_experiments/mlruns/'+exp_name)