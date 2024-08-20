import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser(description='TODO')

    parser.add_argument('--exp', '-e', type=str, default='/home/xzhang/workplace/VGAECCI-main/data/MERFISH/exp.csv')
    parser.add_argument('--adj', '-a', type=str, default='/home/xzhang/workplace/VGAECCI-main/data/MERFISH/adj.csv')
    parser.add_argument('--verbose', action='store_true', help='Print loss of training process')   
    parser.add_argument('--outdir', '-o', type=str, default='output/', help='Output path')
    parser.add_argument('--outpostfix', '-n', type=str,default='model',help='The postfix of the output file')  
    parser.add_argument('--log_add_number', type=int, default=None, help='Perform log10(x+log_add_number) transform')
    parser.add_argument('--fil_gene', type=int, default=None, help='Remove genes expressed in less than fil_gene cells')
    parser.add_argument('--latent_feature', '-l', default=None, help='')

    # Training options
    parser.add_argument('--test_ratio', '-t', type=int, default=0.1, help='Testing set ratio (>1: edge number, <1: edge ratio; default: 0.1)')    
    parser.add_argument('--iteration', '-i', type=int, default=40, help='Iteration (default: 40)')        
    parser.add_argument('--encode_dim', type=int, nargs=3, default=[125, 125, 150], help='Encoder structure')   
    parser.add_argument('--regularization_dim', type=int, nargs=3, default=[150, 125, 150], help='Adversarial regularization structure') 
    parser.add_argument('--lr1', type=float, default=0.0004, help='TODO')
    parser.add_argument('--lr2', type=float, default=0.0008, help='TODO')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight for L2 loss on latent features')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability)')
    parser.add_argument('--features', type=int, default=1, help='Whether to use features (1) or not (0)')
    parser.add_argument('--seed', type=int, default=1, help='Random seed for repeat results')    
    parser.add_argument('--activation', type=str, default='relu', help="Activation function of hidden units (default: relu)")
    parser.add_argument('--init', type=str, default='glorot_uniform', help="Initialization method for weights (default: glorot_uniform)")
    parser.add_argument('--optimizer', type=str, default='RMSProp', help="Optimization method (default: Adam or RMSProp )")

    # Clustering options
    parser.add_argument('--cluster', action='store_true', help='TODO')
    parser.add_argument('--cluster_num', type=int, default=None, help='TODO')    

    parser.add_argument('--gpu', '-g', type=int, default=0, help='Select gpu device number for training')


    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Select gpu device number  
    import os 
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)  #if you use GPU, you must be sure that there is at least one GPU available in your device
    except:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  #set only using cpu

    # Import modules
    try:
        import tensorflow as tf  #import tf and the rest module after parse_args() to make argparse help show faster
    except ImportError:
        raise ImportError('VGAECCI requires TensorFlow. Please follow instructions')
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from scipy.spatial.distance import pdist, squareform
    import copy
    from vgaecci.io import *
    from vgaecci.plot import *
    from vgaecci.utils import sparse2tuple, packed_data, set_placeholder, set_optimizer, update, ranked_partial
    from vgaecci.models import VGAECCI, Discriminator
    from vgaecci.metrics import linkpred_metrics, select_optimal_threshold
  

    # Set random seed
    seed = args.seed
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)
    #tf.random.set_seed(seed)

    # Import and pack datasets    载入数据    
    exp_df, adj_df = read_dataset(args.exp, args.adj, args.fil_gene, args.log_add_number) 
    
    exp, adj = exp_df.values, adj_df.values


    feas = packed_data(exp, adj, args.test_ratio)                     
    var_placeholders = set_placeholder(feas['adj_train'], args.encode_dim[1])

    # Output some basic information
    cell_num = exp.shape[0]
    gene_num = exp.shape[1]
    predefined_edge_num = np.where(adj==1)[0].shape[0]/2       
    print("======== Parameters ========")
    print('Cell number: {}\nGene number: {}\nPredefined local connection number: {}\niteration: {}'.format(
            cell_num, gene_num, predefined_edge_num, args.iteration))
    print("============================")

    vgaecci = VGAECCI(var_placeholders, feas['num_features'], feas['num_nodes'], feas['features_nonzero'], args.encode_dim[0], args.encode_dim[1], args.encode_dim[2])
    print()
    deeplinc_discriminator = Discriminator(args.encode_dim[1], args.regularization_dim[0], args.regularization_dim[1], args.regularization_dim[2])
    opt = set_optimizer(vgaecci, deeplinc_discriminator, var_placeholders, feas['pos_weight'], feas['norm'], feas['num_nodes'], args.lr1, args.lr2)

################################################################################################################

    saver = tf.compat.v1.train.Saver(max_to_keep=1)

   
    # Initialize session
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    # Metrics list
    train_loss = []
    test_ap = []
    test_roc = []

    # latent_feature = None
    max_test_ap_score = 0

    # Train model
    for epoch in range(args.iteration):

        emb_hidden1_train, emb_hidden2_train, emb_hidden3_train, avg_cost_train = update(vgaecci, opt, sess, feas['adj_norm'], feas['adj_label'], feas['features'], var_placeholders, feas['adj_train'], args.dropout, args.encode_dim[1])      #####33333333
        train_loss.append(avg_cost_train)

        lm_train = linkpred_metrics(feas['test_edges'], feas['test_edges_false'])

        roc_score, ap_score, acc_score, _ = lm_train.get_roc_score(emb_hidden3_train, feas)     
        test_ap.append(ap_score)
        test_roc.append(roc_score)
        


        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost_train), "test_roc=", "{:.5f}".format(roc_score), "test_ap=", "{:.5f}".format(ap_score))
                                                       
        if ap_score > max_test_ap_score:
            max_test_ap_score = ap_score

            saver.save(sess, '/home/xzhang/workplace/VGAECCI-main/VGAE-CCI/model/'+args.outpostfix, global_step=epoch+1)  
        
            np.save("/home/xzhang/workplace/VGAECCI-main/VGAE-CCI/output_numpy/emb_hidden1_"+str(epoch+1)+'.npy', emb_hidden1_train)    
            np.save("/home/xzhang/workplace/VGAECCI-main/VGAE-CCI/output_numpy/emb_hidden2_"+str(epoch+1)+'.npy', emb_hidden2_train)
            np.save("/home/xzhang/workplace/VGAECCI-main/VGAE-CCI/output_numpy/emb_hidden3_"+str(epoch+1)+'.npy', emb_hidden3_train)
            
            latent_feature = copy.deepcopy(emb_hidden3_train)       #2 gai  3


    plot_evaluating_metrics(test_ap, "epoch", "score", ["AUPRC"], "AUPRC")
    plot_evaluating_metrics(test_roc, "epoch", "score", ["AUROC"], "AUROC")
    plot_evaluating_metrics(train_loss, "epoch", "score", ["loss"], "loss")
    write_pickle(feas, 'feas')

################################################################################################################
 
    adj_reconstructed_prob, adj_reconstructed, _, _, all_acc_score, max_acc_score, optimal_threshold = select_optimal_threshold(feas['test_edges'], feas['test_edges_false']).select(latent_feature, feas)
    print("optimal_threshold is " , optimal_threshold)   
    print("max_acc_score is " , max_acc_score)   

    # write_json(all_acc_score, 'acc_diff_threshold_'+args.outpostfix)   
    # write_json({'optimal_threshold':optimal_threshold,'max_acc_score':max_acc_score}, 'threshold_'+args.outpostfix) 
    # write_csv_matrix(adj_reconstructed_prob, 'adj_reconstructed_prob_'+args.outpostfix)  
    # write_csv_matrix(adj_reconstructed, 'adj_reconstructed_'+args.outpostfix)      
 
    




