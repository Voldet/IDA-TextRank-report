
import numpy as np
import networkx as nx
import codecs
from segmentation import Segmentation


def input(path='./test/doc/01.txt'):
    from textrank4zh import TextRank4Sentence

    text = codecs.open(path, 'r', 'utf-8').read()
    tr4s = TextRank4Sentence()
    tr4s.analyze(text=text, lower=True, source='all_filters')
    # res = tr4s.get_key_sentences(num=3)
    # for item in res:
    #     print(item.index, item.weight, item.sentence)

    return tr4s


class Abstract(object):

    def __init__(self):
        self.sentences = None
        self.key_sentence = None
        self.word_list = None
        self.text_rank = None

    def read_text(self, text):
        seg = Segmentation()
        self.sentences, self.word_list = seg.segment_sentence(text)

    @staticmethod
    def similarity(word_list1, word_list2):
        if len(word_list1) == 0 or len(word_list2) == 0:
            return 0
        words = list(set(word_list1 + word_list2))
        size = len(words)
        v = [0] * size
        for i in range(size):
            if words[i] in word_list1 and words[i] in word_list2:
                v[i] = 1
        sim = sum(v)
        dom = np.log(len(word_list1)) + np.log((len(word_list2)))
        if sim < 1e-1 or dom < 1e-1:
            return 0
        else:
            return sim / dom

    def weight_matrix(self):
        size = len(self.sentences)
        source = self.word_list
        weight = np.zeros((size, size))
        for i in range(size):
            for j in range(i + 1, size):
                dis = self.similarity(source[i], source[j])
                weight[i][j] = dis
                weight[j][i] = dis
        return weight

    def find_abstract(self):
        rs = []
        textrank = PageRank()
        graph = self.weight_matrix()
        textrank.weight_to_rank(graph)

        # for row in graph:
        #     print('sel', row)
        # factor = {'alpha': 0.85, }
        # nx_graph = nx.from_numpy_matrix(graph)
        # scores = nx.pagerank(nx_graph, **factor)  # this is a dict
        # sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        # rs = list()
        # for index, score in sorted_scores:
        #     rs.append((index, score, self.sentences[index]))

        res = textrank.page_rank(200)
        for k in range(res.shape[0]):
            rs.append([k, res[k], self.sentences[k]])
        rs.sort(key=lambda rs: rs[1], reverse=True)
        # rs is a list, each element is consist of (index, text rank, sentence).
        self.key_sentence = rs
        self.text_rank = textrank

    def visual(self, path='sentences.gexf'):
        """
        to make the ranked result be visualized, sort the key_sentence in order 'name weight'
        the return value could be used by some tools (ex. gephi)
        """
        G = nx.Graph()
        graph = self.text_rank.rank_m
        for i in range(graph.shape[0]):
            for j in range(i, graph.shape[1]):
                if float(graph[i][j]) != 0:
                    G.add_weighted_edges_from([(i, j, float(graph[i][j]))])
        # print(G.number_of_nodes())
        # print(G.number_of_edges())
        nx.write_gexf(G, path)
        pass


class PageRank(object):

    def __init__(self, rank_matrix=None):
        self.rank_m = rank_matrix
        self.factor = 0.85

    def page_rank(self, num, factor=None):
        ini = np.array([1] * self.rank_m.shape[0])
        if factor is None:
            factor = self.factor
        i_f = 1 - factor
        err = ini
        for n in range(num):
            temp = i_f * ini + factor * np.dot(self.rank_m, ini)
            err = temp - ini
            ini = temp
            if np.dot(err, err.T) < 1e-6:
                print('Iteration number: %d' % n)
                s = sum(ini)
                return ini / s
        raise Exception('Does not convergence!', err)

    def weight_to_rank(self, weight_m):
        size = weight_m.shape[0]
        rank_m = np.array(weight_m)
        for i in range(size):
            a = sum(weight_m[:, i])
            if a != 0:
                rank_m[:, i] = weight_m[:, i] / a
        self.rank_m = rank_m


def main():
    path = './test/doc/08.txt'
    # article = input(path)
    # for item in article.key_sentences[0:5]:
    #     print(item)

    ab = Abstract()
    ab.read_text(codecs.open(path, 'r', 'utf-8').read())
    ab.find_abstract()
    ab.visual()
    # f = open('../test/doc/十九大报告top10&tail10.txt', 'a', encoding='utf-8')
    # f.write('tail 10:\n')
    for item in ab.key_sentence[0:5]:
        print(item)
    #     f.write('%s。\n' % item[2])
    # f.write('\n')
    # f.close()


if __name__ == '__main__':
    main()

