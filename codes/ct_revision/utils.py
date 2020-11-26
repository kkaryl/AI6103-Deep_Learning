import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import torch.nn.functional as F

cmap = sns.diverging_palette(262, 10, sep=1, n=16, s=99, l=50, center="dark", as_cmap=True)  # best

import os.path


def check_mnist_dataset_exists(path_data='../../data/'):
    flag_train_data = os.path.isfile(path_data + 'mnist/train_data.pt')
    flag_train_label = os.path.isfile(path_data + 'mnist/train_label.pt')
    flag_test_data = os.path.isfile(path_data + 'mnist/test_data.pt')
    flag_test_label = os.path.isfile(path_data + 'mnist/test_label.pt')
    if flag_train_data == False or flag_train_label == False or flag_test_data == False or flag_test_label == False:
        print('MNIST dataset missing - downloading...')
        import torchvision
        import torchvision.transforms as transforms
        trainset = torchvision.datasets.MNIST(root=path_data + 'mnist/temp', train=True,
                                              download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.MNIST(root=path_data + 'mnist/temp', train=False,
                                             download=True, transform=transforms.ToTensor())
        train_data = torch.Tensor(60000, 28, 28)
        train_label = torch.LongTensor(60000)
        for idx, example in enumerate(trainset):
            train_data[idx] = example[0].squeeze()
            train_label[idx] = example[1]
        torch.save(train_data, path_data + 'mnist/train_data.pt')
        torch.save(train_label, path_data + 'mnist/train_label.pt')
        test_data = torch.Tensor(10000, 28, 28)
        test_label = torch.LongTensor(10000)
        for idx, example in enumerate(testset):
            test_data[idx] = example[0].squeeze()
            test_label[idx] = example[1]
        torch.save(test_data, path_data + 'mnist/test_data.pt')
        torch.save(test_label, path_data + 'mnist/test_label.pt')
    return path_data


def check_fashion_mnist_dataset_exists(path_data='../../data/'):
    flag_train_data = os.path.isfile(path_data + 'fashion-mnist/train_data.pt')
    flag_train_label = os.path.isfile(path_data + 'fashion-mnist/train_label.pt')
    flag_test_data = os.path.isfile(path_data + 'fashion-mnist/test_data.pt')
    flag_test_label = os.path.isfile(path_data + 'fashion-mnist/test_label.pt')
    if flag_train_data == False or flag_train_label == False or flag_test_data == False or flag_test_label == False:
        print('FASHION-MNIST dataset missing - downloading...')
        import torchvision
        import torchvision.transforms as transforms
        trainset = torchvision.datasets.FashionMNIST(root=path_data + 'fashion-mnist/temp', train=True,
                                                     download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.FashionMNIST(root=path_data + 'fashion-mnist/temp', train=False,
                                                    download=True, transform=transforms.ToTensor())
        train_data = torch.Tensor(60000, 28, 28)
        train_label = torch.LongTensor(60000)
        for idx, example in enumerate(trainset):
            train_data[idx] = example[0].squeeze()
            train_label[idx] = example[1]
        torch.save(train_data, path_data + 'fashion-mnist/train_data.pt')
        torch.save(train_label, path_data + 'fashion-mnist/train_label.pt')
        test_data = torch.Tensor(10000, 28, 28)
        test_label = torch.LongTensor(10000)
        for idx, example in enumerate(testset):
            test_data[idx] = example[0].squeeze()
            test_label[idx] = example[1]
        torch.save(test_data, path_data + 'fashion-mnist/test_data.pt')
        torch.save(test_label, path_data + 'fashion-mnist/test_label.pt')
    return path_data


def check_cifar_dataset_exists(path_data='../../data/'):
    flag_train_data = os.path.isfile(path_data + 'cifar/train_data.pt')
    flag_train_label = os.path.isfile(path_data + 'cifar/train_label.pt')
    flag_test_data = os.path.isfile(path_data + 'cifar/test_data.pt')
    flag_test_label = os.path.isfile(path_data + 'cifar/test_label.pt')
    if flag_train_data == False or flag_train_label == False or flag_test_data == False or flag_test_label == False:
        print('CIFAR dataset missing - downloading...')
        import torchvision
        import torchvision.transforms as transforms
        trainset = torchvision.datasets.CIFAR10(root=path_data + 'cifar/temp', train=True,
                                                download=True, transform=transforms.ToTensor())
        testset = torchvision.datasets.CIFAR10(root=path_data + 'cifar/temp', train=False,
                                               download=True, transform=transforms.ToTensor())
        train_data = torch.Tensor(50000, 3, 32, 32)
        train_label = torch.LongTensor(50000)
        for idx, example in enumerate(trainset):
            train_data[idx] = example[0]
            train_label[idx] = example[1]
        torch.save(train_data, path_data + 'cifar/train_data.pt')
        torch.save(train_label, path_data + 'cifar/train_label.pt')
        test_data = torch.Tensor(10000, 3, 32, 32)
        test_label = torch.LongTensor(10000)
        for idx, example in enumerate(testset):
            test_data[idx] = example[0]
            test_label[idx] = example[1]
        torch.save(test_data, path_data + 'cifar/test_data.pt')
        torch.save(test_label, path_data + 'cifar/test_label.pt')
    return path_data


def show(X):
    if X.dim() == 3 and X.size(0) == 3:
        plt.imshow(np.transpose(X.numpy(), (1, 2, 0)))
        plt.show()
    elif X.dim() == 2:
        plt.imshow(X.numpy(), cmap='gray')
        plt.show()
    else:
        print('WRONG TENSOR SIZE')


def show_prob_mnist(p):
    p = p.data.squeeze().numpy()

    ft = 15
    label = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')
    # p=p.data.squeeze().numpy()
    y_pos = np.arange(len(p)) * 1.2
    target = 2
    width = 0.9
    col = 'blue'
    # col='darkgreen'

    plt.rcdefaults()
    fig, ax = plt.subplots()

    # the plot
    ax.barh(y_pos, p, width, align='center', color=col)

    ax.set_xlim([0, 1.3])
    # ax.set_ylim([-0.8, len(p)*1.2-1+0.8])

    # y label
    ax.set_yticks(y_pos)
    ax.set_yticklabels(label, fontsize=ft)
    ax.invert_yaxis()
    # ax.set_xlabel('Performance')
    # ax.set_title('How fast do you want to go today?')

    # x label
    ax.set_xticklabels([])
    ax.set_xticks([])
    # x_pos=np.array([0, 0.25 , 0.5 , 0.75 , 1])
    # ax.set_xticks(x_pos)
    # ax.set_xticklabels( [0, 0.25 , 0.5 , 0.75 , 1] , fontsize=15)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(4)

    for i in range(len(p)):
        str_nb = "{0:.2f}".format(p[i])
        ax.text(p[i] + 0.05, y_pos[i], str_nb,
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transData, color=col, fontsize=ft)

    plt.show()


def show_prob_fashion_mnist(p):
    p = p.data.squeeze().numpy()

    ft = 15
    label = ('T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot')
    # p=p.data.squeeze().numpy()
    y_pos = np.arange(len(p)) * 1.2
    target = 2
    width = 0.9
    col = 'blue'
    # col='darkgreen'

    plt.rcdefaults()
    fig, ax = plt.subplots()

    # the plot
    ax.barh(y_pos, p, width, align='center', color=col)

    ax.set_xlim([0, 1.3])
    # ax.set_ylim([-0.8, len(p)*1.2-1+0.8])

    # y label
    ax.set_yticks(y_pos)
    ax.set_yticklabels(label, fontsize=ft)
    ax.invert_yaxis()
    # ax.set_xlabel('Performance')
    # ax.set_title('How fast do you want to go today?')

    # x label
    ax.set_xticklabels([])
    ax.set_xticks([])
    # x_pos=np.array([0, 0.25 , 0.5 , 0.75 , 1])
    # ax.set_xticks(x_pos)
    # ax.set_xticklabels( [0, 0.25 , 0.5 , 0.75 , 1] , fontsize=15)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_linewidth(4)

    for i in range(len(p)):
        str_nb = "{0:.2f}".format(p[i])
        ax.text(p[i] + 0.05, y_pos[i], str_nb,
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transData, color=col, fontsize=ft)

    plt.show()


def display_scores(sc):
    ft = 10
    ft_label = 12

    bs = sc.size(0)
    nb_class = sc.size(1)

    f, ax = plt.subplots(1, bs)

    if bs == 2:
        f.set_size_inches(8, 8)
        f.subplots_adjust(left=None, bottom=None, right=None, top=None,
                          wspace=None, hspace=0.5)
    else:
        f.set_size_inches(12, 12)
        f.subplots_adjust(left=None, bottom=None, right=None, top=None,
                          wspace=None, hspace=0.25)

    max_score = sc.max().item()
    min_score = sc.min().item()

    label_pos = min_score - 8
    xmin = -5.5
    xmax = 5.5

    label = []
    for i in range(nb_class):
        str_nb = "{0:.0f}".format(i)
        mystr = 'class ' + str_nb
        label.append(mystr)

    y_pos = np.arange(nb_class) * 1.2

    for count in range(bs):

        ax[count].set_title('data point ' + "{0:.0f}".format(count))

        scores = sc[count].numpy()

        width = 0.9
        col = 'darkgreen'

        #    plt.rcdefaults()

        # line in the middle
        ax[count].plot([0, 0], [y_pos[0] - 1, y_pos[-1] + 1], color='k', linewidth=4)

        # the plot
        barlist = ax[count].barh(y_pos, scores, width, align='center', color=col)

        for idx, bar in enumerate(barlist):
            if scores[idx] < 0:
                bar.set_color('r')

        ax[count].set_xlim([xmin, xmax])
        ax[count].invert_yaxis()

        # no y label
        ax[count].set_yticklabels([])
        ax[count].set_yticks([])

        # x label
        ax[count].set_xticklabels([])
        ax[count].set_xticks([])

        ax[count].spines['right'].set_visible(False)
        ax[count].spines['top'].set_visible(False)
        ax[count].spines['bottom'].set_visible(False)
        ax[count].spines['left'].set_visible(False)

        ax[count].set_aspect('equal')

        for i in range(len(scores)):

            str_nb = "{0:.1f}".format(scores[i])
            if scores[i] >= 0:
                ax[count].text(scores[i] + 0.05, y_pos[i], str_nb,
                               horizontalalignment='left', verticalalignment='center',
                               transform=ax[count].transData, color=col, fontsize=ft)
            else:
                ax[count].text(scores[i] - 0.05, y_pos[i], str_nb,
                               horizontalalignment='right', verticalalignment='center',
                               transform=ax[count].transData, color='r', fontsize=ft)

            if count == 0:
                ax[0].text(label_pos, y_pos[i], label[i],
                           horizontalalignment='left', verticalalignment='center',
                           transform=ax[0].transData, color='black', fontsize=ft_label)

    plt.show()


def get_error(scores, labels):
    bs = scores.size(0)
    predicted_labels = scores.argmax(dim=1)
    indicator = (predicted_labels == labels)
    num_matches = indicator.sum()

    return 1 - num_matches.float() / bs


def display_num_param(net):
    nb_param = 0
    for param in net.parameters():
        nb_param += param.numel()
    print('There are {} ({:.2f} million) parameters in this neural network'.format(
        nb_param, nb_param / 1e6)
    )


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data


#############################
# PTB
#############################
def check_ptb_dataset_exists(path_data='../data/', batch_size=20):
    flag_idx2word = os.path.isfile(path_data + 'ptb/idx2word.pt')
    flag_test_data = os.path.isfile(path_data + 'ptb/test_data.pt')
    flag_train_data = os.path.isfile(path_data + 'ptb/train_data.pt')
    flag_word2idx = os.path.isfile(path_data + 'ptb/word2idx.pt')
    if flag_idx2word == False or flag_test_data == False or flag_train_data == False or flag_word2idx == False:
        print('PTB dataset missing - generating...')
        data_folder = 'ptb/data_raw'
        corpus = Corpus(path_data + data_folder)
        # batch_size = 20
        train_data = batchify(corpus.train, batch_size)
        val_data = batchify(corpus.valid, batch_size)
        test_data = batchify(corpus.test, batch_size)
        vocab_size = len(corpus.dictionary)
        torch.save(train_data, path_data + 'ptb/train_data.pt')
        torch.save(test_data, path_data + 'ptb/test_data.pt')
        torch.save(corpus.dictionary.idx2word, path_data + 'ptb/idx2word.pt')
        torch.save(corpus.dictionary.word2idx, path_data + 'ptb/word2idx.pt')
    return path_data


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids


path_data = '../data/'
_ = check_ptb_dataset_exists(path_data)
word2idx = torch.load(path_data + 'ptb/word2idx.pt')
idx2word = torch.load(path_data + 'ptb/idx2word.pt')


def normalize_gradient(net):
    grad_norm_sq = 0

    for p in net.parameters():
        grad_norm_sq += p.grad.data.norm() ** 2

    grad_norm = math.sqrt(grad_norm_sq)

    if grad_norm < 1e-4:
        net.zero_grad()
        print('grad norm close to zero')
    else:
        for p in net.parameters():
            p.grad.data.div_(grad_norm)

    return grad_norm


def sentence2vector(sentence):
    words = sentence.split()
    x = torch.LongTensor(len(words), 1)
    for idx, word in enumerate(words):

        if word not in word2idx:
            print('You entered a word which is not in the vocabulary.')
            print('Make sure that you do not have any capital letters')
        else:
            x[idx, 0] = word2idx[word]
    return x


def show_next_word(scores):
    num_word_display = 30
    prob = F.softmax(scores, dim=2)
    p = prob[-1].squeeze()
    p, word_idx = torch.topk(p, num_word_display)

    for i, idx in enumerate(word_idx):
        percentage = p[i].item() * 100
        word = idx2word[idx.item()]
        print("{:.1f}%\t".format(percentage), word)
