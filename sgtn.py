import argparse
from engine import *
from models import *
from coco import *
from util import *
import neptune

parser = argparse.ArgumentParser(description='WILDCAT Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset (e.g. data/')
parser.add_argument('--image-size', '-i', default=448, type=int,
                    metavar='N', help='image size (default: 224)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--epoch_step', default=[30], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--device_ids', default=[0], type=int, nargs='+',
                    help='number of epochs to change learning rate')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('-bt', '--batch-size-test', default=None, type=int,
                    metavar='N', help='mini-batch size for test (default: 16)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lrd', '--learning-rate-decay', default=0.1, type=float,
                    metavar='LRD', help='learning rate decay')
parser.add_argument('--lrp', '--learning-rate-pretrained', default=0.1, type=float,
                    metavar='LR', help='learning rate for pre-trained layers')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=0, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--embedding', default='data/coco/coco_glove_word2vec.pkl',
                    type=str, metavar='EMB', help='path to embedding (default: glove)')
parser.add_argument('--embedding-length', default=300, type=int, metavar='EMB',
                    help='embedding length (default: 300)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-n', '--neptune', dest='neptune', action='store_true',
                    help='run with neptune')
parser.add_argument('--neptune-path', default='neptune.txt', type=str, metavar='PATH',
                    help='Neptune API keys (default: neptune.txt)')
parser.add_argument('--adj-dd-threshold', default=0.4, type=float, metavar='ADJ_T',
                    help='Data-driven adj threshod (default: 0.4')
parser.add_argument('--adj-files', default=['data/coco/bert_base_cosine_adj.pkl_emb',
                                            'data/coco/glove_cosine_adj.pkl_emb',
                                            'data/coco/char2vec_cosine_adj.pkl_emb'], type=str, nargs='+',
                    help='Adj files  (default: [bert_base_newest_avg_4layers_cosine_adj.pkl_emb, glove_cosine_adj.pkl_emb, '
                         'char2vec_cosine_adj.pkl_emb])')
parser.add_argument('--annonymize', action='store_true', help='Run with annonymized images')

parser.add_argument('--exp-name', dest='exp_name', default='coco', type=str, metavar='COCO',
                    help='Name of experiment to have different location to save checkpoints')


def main_coco():
    global args, best_prec1, use_gpu
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    torch.backends.cudnn.benchmark = False

    if args.neptune:
        with open(args.neptune_path, 'r') as f:
            nep = f.readlines()
            neptune.init(nep[0].strip(), api_token=nep[1].strip())
            neptune.create_experiment(params=vars(args),
                                      upload_source_files=['sgtn.py', 'coco.py', 'engine.py', 'models.py',
                                                           'util.py'])

    train_dataset = COCO2014(args.data, phase='train', annonymize=args.annonymize, inp_name=args.embedding)
    val_dataset = COCO2014(args.data, phase='val', annonymize=args.annonymize, inp_name=args.embedding)
    num_classes = 80

    print('Embedding:', args.embedding, '(', args.embedding_length, ')')
    print('Adjacency files:', args.adj_files)
    print('Data-driven Adjacency Threshold:', args.adj_dd_threshold)

    model = gcn_resnext50_32x4d_swsl(num_classes=num_classes, t=args.adj_dd_threshold, adj_files=args.adj_files,
                                     in_channel=args.embedding_length)

    # define loss function (criterion)
    criterion = nn.MultiLabelSoftMarginLoss()

    # define optimizer
    optimizer = torch.optim.SGD(model.get_config_optim(args.lr, args.lrp),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    state = {'batch_size': args.batch_size, 'batch_size_test': args.batch_size_test, 'image_size': args.image_size,
             'max_epochs': args.epochs,
             'evaluate': args.evaluate, 'resume': args.resume, 'num_classes': num_classes}
    state['difficult_examples'] = True

    model_path = "checkpoint/coco/%s" % args.exp_name
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    state['save_model_path'] = model_path

    state['workers'] = args.workers
    state['epoch_step'] = args.epoch_step
    state['lr'] = args.lr
    state['lr_decay'] = args.lrd
    state['device_ids'] = args.device_ids
    state['neptune'] = args.neptune
    if args.evaluate:
        state['evaluate'] = True
    engine = GCNMultiLabelMAPEngine(state)
    engine.learning(model, criterion, train_dataset, val_dataset, optimizer)


if __name__ == '__main__':
    main_coco()
