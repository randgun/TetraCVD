import yaml
import os
from utils.tools import YamlObject, set_logger, seed_everything
import torch
from exp_tetra import Exp_Tetra
import random
import numpy as np
import logging
from sklearn.metrics import classification_report, confusion_matrix
from utils.metrics import print_metrics

os.environ["TOKENIZERS_PARALLELISM"] = "false"

lucky_seed = 2023
seed_everything(lucky_seed)


with open('./default.yaml', "r") as f:
    args = yaml.safe_load(f)
args = YamlObject(**args)


args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args.__dict__)
print(args.ts_backbone.__dict__)
print(args.language_backbone.__dict__)
print(args.tetra.__dict__)
Exp = Exp_Tetra

acc_vec = np.zeros((args.n_splits, ))
auprc_vec = np.zeros((args.n_splits, ))
auroc_vec = np.zeros((args.n_splits, ))
f1_vec = np.zeros((args.n_splits, ))

macro_auc = np.zeros((args.n_splits, ))
micro_auc = np.zeros((args.n_splits, ))
macro_f1 = np.zeros((args.n_splits, ))
micro_f1 = np.zeros((args.n_splits, ))
p_at_1 = np.zeros((args.n_splits, ))
p_at_3 = np.zeros((args.n_splits, ))
r_at_1 = np.zeros((args.n_splits, ))
r_at_3 = np.zeros((args.n_splits, ))

TEST = not args.is_training
assert args.task in ['multi-class', 'multi-label']

for k in range(args.n_splits):
    print("----------------- {}_th fold -----------------".format(k+1))
    split_path = 'splits/' + str(k+1) + '_fold.npy'
    args.tetra.split_path = split_path
    # setting record of experiments
    setting = '{}_{}_{}_fold_structure_{}_itc_{}_dim_{}_numlayers_{}_toplayer_{}_dropout_{}'.format(
        args.task,
        args.language_backbone.category,
        k+1,
        args.tetra.structure,
        args.tetra.itc,
        args.tetra.hidden_size,
        args.tetra.num_layers,
        args.tetra.num_top_layer,
        args.tetra.drop_rate,
        )
    
    args.ts_backbone.global_structure = torch.ones(args.ts_backbone.d_inp, args.ts_backbone.d_inp)

    exp = Exp(args)  # set experiments
    if args.is_training:
        print('\n>>>>>>>start training : {}>>>>>>>>>>'.format(setting))
        exp.train(setting)

    print('\n>>>>>>>testing : {}<<<<<<<<<<'.format(setting))
    res = exp.test(setting, test=TEST)
    if args.task == 'multi-class':
        acc_test, aupr_test, auc_test = res[1:4]
        print("Acc_test: {0:.4f}, Aupr_test: {1:.4f}, Auc_test: {2:.4f}".format(acc_test, aupr_test, auc_test))
        print('----------------------------------------------------------')
        print('Classification report and confusion matrix\n', classification_report(res[3], res[4], digits=4))
        print(confusion_matrix(res[3], res[4], labels=list(range(args.n_classes))))
        print('----------------------------------------------------------')
    
        acc_vec[k] = acc_test
        auprc_vec[k] = aupr_test
        auroc_vec[k] = auc_test
    else:
        macro_auc[k] = res['auc_macro']
        micro_auc[k] = res['auc_micro']
        macro_f1[k] = res['f1_macro']
        micro_f1[k] = res['f1_micro']
        p_at_1[k] = res['prec_at_1']
        p_at_3[k] = res['prec_at_3']
        r_at_1[k] = res['rec_at_1']
        r_at_3[k] = res['rec_at_3']
        print_metrics(res)
        print('Classification report\n', classification_report(res['y'], res['yhat'], digits=4))
        # print(multilabel_confusion_matrix(res['y'], res['yhat'], labels=list(range(args.n_classes))))
        print('----------------------------------------------------------')


if args.task == 'multi-class':
    # display mean and standard deviation
    mean_acc, std_acc = np.mean(acc_vec), np.std(acc_vec)
    mean_auprc, std_auprc = np.mean(auprc_vec), np.std(auprc_vec)
    mean_auroc, std_auroc = np.mean(auroc_vec), np.std(auroc_vec)
    print('-------------- Mean Results --------------')
    print('Accuracy = %.4f +/- %.4f' % (mean_acc, std_acc))
    print('AUPRC    = %.4f +/- %.4f' % (mean_auprc, std_auprc))
    print('AUROC    = %.4f +/- %.4f' % (mean_auroc, std_auroc))
else:
    mean_macro_auc, std_macro_auc = np.mean(macro_auc), np.std(macro_auc)
    mean_micro_auc, std_micro_auc = np.mean(micro_auc), np.std(micro_auc)
    mean_macro_f1, std_macro_f1 = np.mean(macro_f1), np.std(macro_f1)
    mean_micro_f1, std_micro_f1 = np.mean(micro_f1), np.std(micro_f1)

    mean_p_at_1, std_p_at_1 = np.mean(p_at_1), np.std(p_at_1)
    mean_p_at_3, std_p_at_3 = np.mean(p_at_3), np.std(p_at_3)
    mean_r_at_1, std_r_at_1 = np.mean(r_at_1), np.std(r_at_1)
    mean_r_at_3, std_r_at_3 = np.mean(r_at_3), np.std(r_at_3)
    
    print('-------------- Mean Results --------------')
    print('macro_auc = %.4f +/- %.4f' % (mean_macro_auc, std_macro_auc))
    print('micro_auc = %.4f +/- %.4f' % (mean_micro_auc, std_micro_auc))
    print('macro_f1  = %.4f +/- %.4f' % (mean_macro_f1, std_macro_f1))
    print('micro_f1  = %.4f +/- %.4f' % (mean_micro_f1, std_micro_f1))
    print('p@1  = %.4f +/- %.4f' % (mean_p_at_1, std_p_at_1))
    print('p@3  = %.4f +/- %.4f' % (mean_p_at_3, std_p_at_3))
    print('r@1  = %.4f +/- %.4f' % (mean_r_at_1, std_r_at_1))
    print('r@3  = %.4f +/- %.4f' % (mean_r_at_3, std_r_at_3))
