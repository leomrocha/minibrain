import os
from sortedcontainers import SortedDict
import numpy as np
import pyconll
import pyconll.util
import pickle
import ntpath


# list of languages that won't be used:
# this is because I'll be using only the first 2 segments of UTF-8, so the idea is that
# Maybe blacklist
MAYBE_BLACKLIST = ["Kurmanji", "Urdu", "Indonesian", "Coptic-Scriptorium", "Kazakh",
                   "Marathi", "Tamil", "Thai", "Warlpiri"]
LANG_TOKENS_BLACKLIST = ["Hindi", "Chinese", "Korean", "Tagalog", "Vietnamese",
                         "Telugu", "Uyghur", "Cantonese", "Japanese",
                         "ar_nyuad-ud", "myv_jr-ud", "br_keb-ud"]  # last 2 -> processing errors with pyconll
BLACKLIST = MAYBE_BLACKLIST + LANG_TOKENS_BLACKLIST

# Tags obtained from a full analysis on the UD-treebank v2.4 data

UPOS = {'ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON',
        'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X', '_'}
UPOS_LIST = sorted(list(UPOS))
UPOS_IDX2CHAR = SortedDict(enumerate(UPOS_LIST))
UPOS_CHAR2IDX = SortedDict(zip(UPOS_LIST, range(len(UPOS_LIST))))

# deprel analysis of ud-treebank v2.4 on all the language files
DEPREL = {'_', 'acl', 'acl:adv', 'acl:appos', 'acl:cleft', 'acl:focus', 'acl:inf', 'acl:part', 'acl:poss', 'acl:relcl', 'advcl', 'advcl:appos', 'advcl:arg', 'advcl:cleft', 'advcl:cond', 'advcl:coverb', 'advcl:periph', 'advcl:relcl', 'advcl:sp', 'advcl:svc', 'advcl:tcl', 'advmod', 'advmod:appos', 'advmod:arg', 'advmod:cc', 'advmod:det', 'advmod:df', 'advmod:discourse', 'advmod:emph', 'advmod:locy', 'advmod:mode', 'advmod:neg', 'advmod:obl', 'advmod:periph', 'advmod:que', 'advmod:sentcon', 'advmod:tfrom', 'advmod:tlocy', 'advmod:tmod', 'advmod:to', 'advmod:tto', 'amod', 'amod:advmod', 'amod:att', 'amod:attlvc', 'amod:flat', 'amod:mode', 'amod:obl', 'appos', 'appos:conj', 'appos:nmod', 'aux', 'aux:aglt', 'aux:caus', 'aux:clitic', 'aux:cnd', 'aux:imp', 'aux:mood', 'aux:neg', 'aux:part', 'aux:pass', 'aux:poss', 'aux:q', 'case', 'case:acc', 'case:aspect', 'case:circ', 'case:dec', 'case:det', 'case:gen', 'case:loc', 'case:pred', 'case:pref', 'case:suff', 'case:voc', 'cc', 'cc:nc', 'cc:preconj', 'ccomp', 'ccomp:cleft', 'ccomp:obj', 'ccomp:obl', 'ccomp:pmod', 'ccomp:pred', 'clf', 'compound', 'compound:a', 'compound:affix', 'compound:coll', 'compound:conjv', 'compound:dir', 'compound:ext', 'compound:lvc', 'compound:n', 'compound:nn', 'compound:plur', 'compound:preverb', 'compound:prt', 'compound:quant', 'compound:redup', 'compound:smixut', 'compound:svc', 'compound:v', 'compound:vo', 'compound:vv', 'conj', 'conj:appos', 'conj:coord', 'conj:dicto', 'conj:extend', 'conj:redup', 'conj:svc', 'cop', 'cop:expl', 'cop:locat', 'cop:own', 'csubj', 'csubj:cleft', 'csubj:cop', 'csubj:pass', 'csubj:quasi', 'dep', 'dep:alt', 'dep:iobj', 'dep:obj', 'dep:prt', 'det', 'det:def', 'det:numgov', 'det:nummod', 'det:poss', 'det:predet', 'det:rel', 'discourse', 'discourse:emo', 'discourse:filler', 'discourse:intj', 'discourse:q', 'discourse:sp', 'dislocated', 'dislocated:cleft', 'expl', 'expl:impers', 'expl:pass', 'expl:poss', 'expl:pv', 'fixed', 'fixed:name', 'flat', 'flat:abs', 'flat:foreign', 'flat:name', 'flat:range', 'flat:repeat', 'flat:sibl', 'flat:title', 'flat:vv', 'goeswith', 'iobj', 'iobj:agent', 'iobj:appl', 'iobj:caus', 'list', 'mark', 'mark:adv', 'mark:advb', 'mark:advmod', 'mark:comp', 'mark:obj', 'mark:obl', 'mark:prt', 'mark:q', 'mark:rel', 'mark:relcl', 'nmod', 'nmod:abl', 'nmod:advmod', 'nmod:agent', 'nmod:appos', 'nmod:arg', 'nmod:att', 'nmod:attlvc', 'nmod:cau', 'nmod:clas', 'nmod:cmp', 'nmod:comp', 'nmod:dat', 'nmod:flat', 'nmod:gen', 'nmod:gmod', 'nmod:gobj', 'nmod:gsubj', 'nmod:ins', 'nmod:npmod', 'nmod:obl', 'nmod:obllvc', 'nmod:own', 'nmod:part', 'nmod:pmod', 'nmod:poss', 'nmod:pred', 'nmod:ref', 'nmod:tmod', 'nsubj', 'nsubj:advmod', 'nsubj:appos', 'nsubj:caus', 'nsubj:cop', 'nsubj:expl', 'nsubj:lvc', 'nsubj:nc', 'nsubj:obj', 'nsubj:own', 'nsubj:pass', 'nsubj:periph', 'nsubj:quasi', 'nummod', 'nummod:entity', 'nummod:gov', 'obj', 'obj:advmod', 'obj:advneg', 'obj:agent', 'obj:appl', 'obj:cau', 'obj:caus', 'obj:lvc', 'obj:obl', 'obj:periph', 'obl', 'obl:advmod', 'obl:agent', 'obl:appl', 'obl:arg', 'obl:cau', 'obl:cmpr', 'obl:comp', 'obl:lmod', 'obl:loc', 'obl:mod', 'obl:npmod', 'obl:own', 'obl:patient', 'obl:periph', 'obl:poss', 'obl:prep', 'obl:sentcon', 'obl:tmod', 'obl:x', 'orphan', 'parataxis', 'parataxis:appos', 'parataxis:conj', 'parataxis:deletion', 'parataxis:discourse', 'parataxis:dislocated', 'parataxis:hashtag', 'parataxis:insert', 'parataxis:newsent', 'parataxis:nsubj', 'parataxis:obj', 'parataxis:parenth', 'parataxis:rel', 'parataxis:rep', 'parataxis:restart', 'punct', 'reparandum', 'root', 'vocative', 'vocative:cl', 'vocative:mention', 'xcomp', 'xcomp:adj', 'xcomp:ds', 'xcomp:obj', 'xcomp:pred', 'xcomp:sp', 'xcomp:subj'}
DEPREL_LIST = sorted(list(DEPREL))
DEPREL_IDX2CHAR = SortedDict(enumerate(DEPREL_LIST))
DEPREL_CHAR2IDX = SortedDict(zip(DEPREL_LIST, range(len(DEPREL_LIST))))


# from https://stackoverflow.com/questions/8384737/extract-file-name-from-path-no-matter-what-the-os-path-format
def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)


def get_all_files_recurse(rootdir):
    allfiles = []
    for root, directories, filenames in os.walk(rootdir):
        for filename in filenames:
            allfiles.append(os.path.join(root, filename))
    return allfiles


def filter_conllu_files(conllufiles, blacklist):
    prefiltered_conllu = []
    for f in conllufiles:
        todel = list(filter(lambda bl: bl in f, blacklist))
        if len(todel) == 0:
            prefiltered_conllu.append(f)
    conllu_train = [f for f in prefiltered_conllu if "-train" in f]
    conllu_test = [f for f in prefiltered_conllu if "-test" in f]
    conllu_dev = [f for f in prefiltered_conllu if "-dev" in f]
    return conllu_train, conllu_test, conllu_dev


def seq2charlevel(sequence, deprel=True):
    """
    Creates a matrix with character level description of the tag annotation for the training and testing of
    Language Part of Speech Neural Networks
    :param sequence: conllu (pyconll) sequence
    :param deprel: if True will also extract deprel to the output
    :return: a numpy array of chars where the first column is the text (char by char) and the
    remaining columns are the tags assigned upos and deprel (in that order)
    """
    # convert text to list
    txt = sequence.text
    l_txt = np.array(list(txt))
    # l_tags_upos = np.empty(l_txt.shape, dtype="U5")  # 5 chars is the max length of the upos type (analysis v2.4)
    l_tags_upos = np.full(l_txt.shape, fill_value="_", dtype="U5")  # 5 chars is the max lenght of the upos type
    l_tags_deprel = None
    if deprel:
        # l_tags_deprel = np.empty(l_txt.shape, dtype="U20")  # 20 chars -> max length of deprel field (analysis v2.4)
        l_tags_deprel = np.full(l_txt.shape, fill_value="_", dtype="U20")
    # for each token, go in the string
    index = 0
    # not a good practice accessing private elements, but is the only way as the pyconll API does not handle it nicely
    # TODO make a modification and pull request to the pyconll repo
    for t in sequence._tokens:
        try:
            # find token indices in the remaining of the sequence (this is to do a good tagging)
            tidx = txt[index:].find(t.form)
            tlen = len(t.form)
            idx_start = index + tidx
            idx_end = idx_start + tlen
            # set the flags for each char
            l_tags_upos[idx_start:idx_end] = t.upos
            if deprel:
                l_tags_deprel[idx_start:idx_end] = t.deprel
            # set index to the new absolute text position
            index = idx_end
        except Exception as e:
            print("Token exception: ", e)
            pass
    if deprel:
        ret = np.stack([l_txt, l_tags_upos, l_tags_deprel])
    else:
        ret = np.stack([l_txt, l_tags_upos])
    return ret


def charseq2int(charseq, char2int_codebook, upos2int, deprel2int):
    """

    :param charseq: character sequence in a numpy matrix form where shape = (2,N) or (3,N)
            charseq[0] is the character sequence
            charseq[1] is the upos tag
            charseq[2] is the deprel tag
    :param char2int_codebook: dictionary codebook encoding the chars to int indices
    :param upos2int: dictionary encoding the upos tags to int
    :param deprel2int: dictionary encoding the deprel tags to int
    :return: an output matrix of type int of the same dimensions as the input with the index coding for each row
    """
    assert 2 <= charseq.shape[0] <= 3
    ret = np.empty(shape=charseq.shape, dtype=np.int32)
    # WARNING there is an error in the code file it says 'unk' instead of '<unk>' -> will fix later
    ret[0, :] = np.vectorize(char2int_codebook.get)(charseq[0], char2int_codebook["unk"])
    ret[1, :] = np.vectorize(upos2int.get)(charseq[1], upos2int["_"])  # defaults to '_' which is None for conllu format
    if charseq.shape[0] == 3:
        # defaults to '_' which is None for conllu format
        ret[2, :] = np.vectorize(deprel2int.get)(charseq[2], deprel2int["_"])
    return ret


def conll2seq(conll, char2int_codebook, upos2int, deprel2int, deprel=True, return_chars=False):
    """

    :param conll: the conllu (pyconll) object loaded from file to convert
    :param char2int_codebook: dictionary codebook encoding the chars to int indices
    :param upos2int: dictionary encoding the upos tags to int
    :param deprel2int: dictionary encoding the deprel tags to int
    :param deprel: if deprel field should be processed, default=True
    :param return_chars: if the matrix of chars should be returned, allows using less memory when False
    :return: a pair of lists,
     the first list contains the text sequence with charlevel tagging
     the second list is a list of numpy arrays containing the int encoding of the elements
    """

#   :param charseq: character sequence in a numpy matrix form where shape = (2,N) or (3,N)
#     charseq[0] is the character sequence
#     charseq[1] is the upos tag
#     charseq[2] is the deprel tag
    list_charseq = []
    list_ind_charseq = []
    for seq in conll:
        try:
            charseq = seq2charlevel(seq, deprel=deprel)
            ind_charseq = charseq2int(charseq, char2int_codebook, upos2int, deprel2int)
            list_charseq.append(charseq)
            list_ind_charseq.append(ind_charseq)
        except Exception as e:
            print("Exception during conll2seq ", e)
            pass
    ret = list_ind_charseq
    if return_chars:
        ret = (list_charseq, list_ind_charseq)
    return ret


def process_conllu_file(fname, char2int_codebook, upos2int, deprel2int, deprel=True, save_to=None):
    """
    Processes the input file (must be a conllu file) and outputs the charlevel encoded matrices.
    If save_to is a valid output filename it will save it
    :param fname:
    :param char2int_codebook:
    :param upos2int:
    :param deprel2int:
    :param deprel:
    :param save_to:
    :return: the entire file encoded for charlevel in 2 parallel matrices (chars and int coding)
    """
    # print("processing: {}".format(path_leaf(fname)))
    conll = pyconll.load_from_file(fname)
    ret = conll2seq(conll, char2int_codebook, upos2int, deprel2int, deprel)
    if save_to:
        with open(save_to, "wb") as fout:
            # print("saving data to: {}".format(path_leaf(save_to)))
            pickle.dump(ret, fout, protocol=pickle.HIGHEST_PROTOCOL)
    return ret


def process_all(flist, utf8_char2int_codebook, return_data=False, save_processed=True):
    """
    Processes a list of conllu files (must be valid or will fail)
    Will save the processed char level codes in a pickle file in the same directory as the conllu files
    if return_data is true will keep all data in memory and return it, use with caution.
    Else the return value is an empty list
    :param flist:
    :param utf8_char2int_codebook:
    :param return_data:
    :param save_processed: if the processed data should be saved
    :return:
    """
    # TODO make this process in parallel ;)
    # is worthless to process if not saving and not returning the data
    assert return_data or save_processed
    all_data = []
    # TODO use a command line indicator of % progress in processing
    for fin in flist:
        # the input file MUST be a conllu file
        try:
            fout = None
            if save_processed:
                fout = fin.replace('.conllu', '-charseq_code.pkl')
            data = process_conllu_file(fin, utf8_char2int_codebook, UPOS_CHAR2IDX, DEPREL_CHAR2IDX, True, fout)
            if return_data:
                all_data.append(data)
        except Exception as e:
            print("Exception processing file: {}".format(path_leaf(fin)), e)
    return all_data


# if __name__ == '__main__':
#     # get flist,
#     # load values from
#     process_all()
