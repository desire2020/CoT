import numpy as np
import tensorflow as tf
import random
from dataloader import Gen_Data_loader
from generator import Generator
from target_lstm import TARGET_LSTM
import pickle

#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
SEQ_LENGTH = 20 # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 0 # supervise (maximum likelihood estimation) epochs (not recommended)
SEED = 88
BATCH_SIZE = 64
M_DROPOUT_RATE = 0.5 # Dropout rate of M (optional)

#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 200000
positive_file = 'save/real_data.txt'
negative_file = 'save/generator_sample.txt'
eval_file = 'save/eval_file.txt'
generated_num = 10000


def generate_samples(sess, trainable_model, batch_size, generated_num, output_file):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))

    with open(output_file, 'w') as fout:
        for poem in generated_samples:
            buffer = ' '.join([str(x) for x in poem]) + '\n'
            fout.write(buffer)


def target_loss(sess, target_lstm, data_loader):
    # target_loss means the oracle negative log-likelihood tested with the oracle model "target_lstm"
    # For more details, please see the Section 4 in https://arxiv.org/abs/1609.05473
    nll = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(target_lstm.likelihood_loss, {target_lstm.x: batch})
        nll.append(g_loss)

    return np.mean(nll)


def mle_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()

    for it in range(data_loader.num_batch):
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.maximum_likelihood(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)
def jsd_calculate(sess, generator, oracle, sample_window=200):
    real_s = []
    fake_s = []
    jsd = []
    for it in range(sample_window):
        real_s.append(oracle.generate(sess))
        fake_s.append(generator.generate(sess))
    for s in real_s:
        p_g = sess.run(generator.g_prediction, feed_dict={generator.x:s})
        p_p = sess.run(oracle.g_prediction, feed_dict={oracle.x:s})
        p_m = 0.5 * (p_g + p_p)
        log_p_p = np.log(p_p)
        log_p_m = np.log(p_m)
        log_kl_gm = np.mean(np.sum(log_p_p - log_p_m, axis=-1))
        jsd.append(log_kl_gm)
    for s in fake_s:
        p_g = sess.run(generator.g_prediction, feed_dict={generator.x:s})
        p_p = sess.run(oracle.g_prediction, feed_dict={oracle.x:s})
        p_m = 0.5 * (p_g + p_p)
        log_p_g = np.log(p_g)
        log_p_m = np.log(p_m)
        log_kl_gm = np.mean(np.sum(log_p_g - log_p_m, axis=-1))
        jsd.append(log_kl_gm)
    jsd = np.mean(jsd)
    return jsd

def main():
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0

    gen_data_loader = Gen_Data_loader(BATCH_SIZE, SEQ_LENGTH)
    val_data_loader = Gen_Data_loader(BATCH_SIZE, SEQ_LENGTH)
    likelihood_data_loader = Gen_Data_loader(BATCH_SIZE, SEQ_LENGTH) # For testing
    vocab_size = 5000

    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, START_TOKEN)
    target_params = pickle.load(open('save/target_params_py3.pkl', 'rb'))
    target_lstm = TARGET_LSTM(vocab_size, BATCH_SIZE, 32, 32, SEQ_LENGTH, START_TOKEN, target_params) # The oracle model

    mediator = Generator(vocab_size, BATCH_SIZE*2, EMB_DIM*2, HIDDEN_DIM*2, SEQ_LENGTH, START_TOKEN, name="mediator", dropout_rate=M_DROPOUT_RATE, learning_rate=3e-3)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # First, use the oracle model to provide the positive examples, which are sampled from the oracle data distribution
    generate_samples(sess, target_lstm, BATCH_SIZE, generated_num, positive_file)
    gen_data_loader.create_batches(positive_file)
    generate_samples(sess, target_lstm, BATCH_SIZE, generated_num, eval_file)
    val_data_loader.create_batches(eval_file)

    log = open('save/experiment-log.txt', 'w')
    log_nll = open('save/experiment-log-nll.txt', 'w')
    log_jsd = open('save/experiment-log-jsd.txt', 'w')
    #  pre-train generator (default 0 epochs)(not recommended)
    print('Start pre-training...')
    log.write('pre-training...\n')
    for epoch in range(PRE_EPOCH_NUM):
        loss = mle_epoch(sess, generator, gen_data_loader)
        if epoch % 1 == 0:
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
            likelihood_data_loader.create_batches(negative_file)
            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            print('pre-train epoch ', epoch, 'nll_oracle ', test_loss)
            buffer = 'epoch:\t'+ str(epoch) + '\tnll_oracle:\t' + str(test_loss) + '\n'
            log_nll.write(buffer)
        if epoch % 1 == 0:
            test_loss = target_loss(sess, generator, val_data_loader)
            print('pre-train epoch ', epoch, 'nll_test ', test_loss)
            buffer = 'epoch:\t'+ str(epoch) + '\tnll_test:\t' + str(test_loss) + '\n'
            log_nll.write(buffer)

    print('#########################################################################')
    print('Start Cooperative Training...')
    for iter_idx in range(TOTAL_BATCH):
        # Train the generator for one step
        for it in range(1):
            samples = generator.generate(sess)
            rewards = mediator.get_reward(sess, np.concatenate([samples, samples], axis=0))
            feed = {generator.x: samples, generator.rewards: rewards[0:BATCH_SIZE]}
            _ = sess.run(generator.g_updates, feed_dict=feed)
        # Test
        if iter_idx % 100 == 0 or iter_idx == TOTAL_BATCH - 1:
            generate_samples(sess, generator, BATCH_SIZE, generated_num, negative_file)
            likelihood_data_loader.create_batches(negative_file)
            test_loss = target_loss(sess, target_lstm, likelihood_data_loader)
            buffer = 'batch:\t' + str(iter_idx) + '\tnll_oracle:\t' + str(test_loss) + '\n'
            print('batch: ', iter_idx, 'nll_oracle: ', test_loss)
            log_nll.write(buffer)
        if iter_idx % 100 == 0:
            test_loss = target_loss(sess, generator, val_data_loader)
            print('batch:\t', iter_idx, 'nll_test ', test_loss)
            buffer = 'batch:\t'+ str(iter_idx) + '\tnll_test:\t' + str(test_loss) + '\n'
            log_nll.write(buffer)
        # Train the mediator
        for _ in range(1):
            bnll_ = []
            collected_x = []
            ratio = 2
            for it in range(ratio):
                if it % 2 == 0:
                    x_batch = gen_data_loader.next_batch()
                else:
                    x_batch = generator.generate(sess)
                collected_x.append(x_batch)
            collected_x = np.reshape(collected_x, [-1, SEQ_LENGTH])
            np.random.shuffle(collected_x)
            collected_x = np.reshape(collected_x, [-1, BATCH_SIZE*2, SEQ_LENGTH])
            for it in range(1):
                feed = {
                    mediator.x: collected_x[it],
                }
                bnll = sess.run(mediator.likelihood_loss, feed)
                bnll_.append(bnll)
                sess.run(mediator.dropout_on)
                _ = sess.run(mediator.likelihood_updates, feed)
                sess.run(mediator.dropout_off)
            if iter_idx % 10 == 0:
                bnll = np.mean(bnll_)
                print("mediator cooptrain iter#%d, balanced_nll %f" % (iter_idx, bnll))
                log.write("%d\t%f\n" % (iter_idx, bnll))
        if iter_idx % gen_data_loader.num_batch == 0:
            jsd = jsd_calculate(sess, generator, target_lstm)
            print('cooptrain epoch#', iter_idx // gen_data_loader.num_batch, 'jsd ', jsd)
            log_jsd.write("%d\t%f\n" % (iter_idx // gen_data_loader.num_batch, jsd))

    log.close()
    log_nll.close()
    log_jsd.close()

if __name__ == '__main__':
    main()
