import jieba.posseg as seg


default_sentence_delimiters = ['?', '!', ';', '？', '！', '。', '；', '……', '…', '\n']
default_allow_speech_tags = ['an', 'i', 'j', 'l', 'n', 'nr', 'nrfg', 'ns', 'nt', 'nz', 't', 'v', 'vd', 'vn', 'eng']


class Segmentation(object):

    def __init__(self, allow_speech_tags=default_allow_speech_tags
                 , sentence_seg_words=default_sentence_delimiters):
        self.allow_speech_tags = allow_speech_tags
        self.sentence_seg_words = sentence_seg_words

    def segment(self, text, use_speech_tags_filter=False, lower=True, speech_tags=None):
        """
        :parameter
        :param use_speech_tags_filter: if use tags to filt
        :param lower: if lower word, usually do for English
        :param speech_tags: choose the speacial speech list
        :return:
        """
        jr = seg.cut(text)
        if speech_tags is None:
            speech_tags = self.allow_speech_tags

        if use_speech_tags_filter:
            jr_set = [w for w in jr if w.flag in speech_tags]
        else:
            jr_set = [w for w in jr]

        word_list = [w.word.strip() for w in jr_set if w.flag != 'x']
        word_list = [word for word in word_list if len(word) > 0]
        if lower:
            word_list = [word.lower() for word in word_list]

        return word_list

    def segment_sentence(self, text, lower=True):
        """
        :param text:
        :param lower:
        :return: res: set of sentences, word_lists: set word in each sentence
        """
        res = [text]
        for sep in self.sentence_seg_words:
            text, res = res, []
            for seq in text:
                res += seq.split(sep)
        res = [s.strip() for s in res if len(s.strip()) > 0]
        word_lists = []
        for sentence in res:
            word_lists.append(self.segment(sentence, True, lower=lower))
        return res, word_lists

