import numpy as np
import pickle


# Bit of a whacky hack and for sure not the most efficient one, but it just works for what I want
def prepend_zeros(s, n):
    return '0' * (n - len(s))+s


def get_strbintable(n):
    bl = n.bit_length() - 1  # because n is actually never used
    lines = [' '.join(i for i in prepend_zeros("{0:b}".format(l), bl)) for l in range(n)]
    return lines


def get_npbintable(n):
    bins = np.fromstring('\n'.join(get_strbintable(n)), dtype='int32', sep=' ')
    bins = bins.reshape([n, -1])
    return bins


# this part makes sure to encode as bin
eye4 = np.eye(4)
eye64 = np.eye(64)
eye256 = np.eye(256)

# code for 7 bits, Byte 1 of utf-8
code_b7 = np.append(np.zeros([2**7, 1]), get_npbintable(2**7), axis=1)

# code for 6 bits, Byte 2 to 4 of utf-8 -> this is going to be used later for all the other values
code_b6 = np.append(np.append(np.ones([2**6, 1]), np.zeros([2**6, 1]), axis=1),
                    get_npbintable(2**6), axis=1)

# code for 5 bits, Byte 1 of
code_b5 = np.append(np.append(np.ones([2**5, 2]), np.zeros([2**5, 1]), axis=1),
                    get_npbintable(2**5), axis=1)

# code for 4 bits, Byte 2 to 4 of utf-8 -> this is going to be used later for all the other values
code_b4 = np.append(np.append(np.ones([2**4, 3]), np.zeros([2**4, 1]), axis=1),
                    get_npbintable(2**4), axis=1)

# code for 3 bits, Byte 2 to 4 of utf-8 -> this is going to be used later for all the other values
code_b3 = np.append(np.append(np.ones([2**3, 4]), np.zeros([2**3, 1]), axis=1),
                    get_npbintable(2**3), axis=1)


def encode_utf8(l):
    el = l.encode()
    code = np.zeros(36)  # 32 is the size of the input code + 4 of the extra redundancy
    nbytes = len(el)
    # assert( 0<nbytes && nbytes<=4)
    assert(nbytes<=4)
    bin4 = eye4[nbytes-1]  # this adds redundant knowledge about the  part
    # this is ugly but explicit, for the moment is good enough and I can see what is
    code[:4] = bin4
    if nbytes == 1:
        code[4:12] = code_b7[el[0]& 0b01111111 ]
    elif nbytes == 2:
        code[4:12] = code_b5[el[0] & 0b00011111 ]
        code[12:20] = code_b6[el[1] & 0b00111111]
    elif nbytes == 3:
        code[4:12] = code_b4[el[0] & 0b00001111]
        code[12:20] = code_b6[el[1] & 0b00111111]
        code[20:28] = code_b6[el[2] & 0b00111111]
    elif nbytes == 4:
        code[4:12] = code_b3[el[0] & 0b00000111]
        code[12:20] = code_b6[el[1] & 0b00111111]
        code[20:28] = code_b6[el[2] & 0b00111111]
        code[28:36] = code_b6[el[3] & 0b00111111]
    else:
        raise Exception("Bad input, input has to have 1 to 4 input bytes")
    return code


# TODO I need to find a more efficient way of doing this that could make this as vector or matrix operations instead
def encode_utf8_multihot(c, segments=4):
    e_c = list(c.encode())
#     code = np.zeros(36)  # 32 is the size of the input code + 4 of the extra redundancy
    nbytes = len(e_c)
    # assert( 0<nbytes && nbytes<=4)
    assert(nbytes<=4)
    bin4 = eye4[nbytes-1]  # this adds redundant knowledge about the  part
    # this is ugly but explicit, for the moment is good enough and I can see what is
#     code[:4] = bin4
    # max size of each part of the code
    # I will treat the first byte as always 8 bits, this will make it easier to decode later and adds aditional information
    # this has an extra benefit, when a code is there only certain regions will become 1 giving an extra hint to the network
    # maxsizes = [2**8, 2**6, 2**6, 2**6]
    code = np.zeros(4 + (2**8) + (segments-1)*(2**6))
    masks = [0xff, 0x3f, 0x3f, 0x3f]
    indices = [256+4, 64+256+4, 2*64 + 256+4, 3*64 + 256+4]
    maxsizes = [eye256, eye64, eye64, eye64]
    masks = masks[:segments]
    indices = indices[:segments]
    maxsizes = maxsizes [:segments]
    # print(len(masks), len(indices), masks, indices)

    code[:4] = bin4
    prev_i = 4
    for i, n, e, m in zip(indices[:nbytes], e_c, maxsizes[:nbytes], masks[:nbytes]):
        code[prev_i:i] = e[n & m]  # masking
        prev_i = i
    return code


# masks make values for utf-8 valid, the process is first adding the missing bits for the valid encoding,
# and then subtracting the ones that should not be there
and_mask1 = [0b01111111, 0x00, 0x00, 0x00]  # and mask
# or_mask1 = [0x00, 0x00, 0x00, 0x00]
and_mask2 = [0b11011111, 0b10111111, 0x00, 0x00]
or_mask2 = [0b11000000, 0b10000000, 0x00, 0x00]

and_mask3 = [0b11101111, 0b10111111, 0b10111111, 0x00]
or_mask3 = [0b11100000, 0b10000000, 0b10000000, 0x00]

and_mask4 = [0b11110111, 0b10111111, 0b10111111, 0b10111111]
or_mask4 = [0b11110000, 0b10000000, 0b10000000, 0b10000000]


def create_tables(segments=4):
    assert(0 < segments <= 4)
    # will create a table with all the codings -> this one
    # and dictionaries with the mappings
    code_matrix = []
    code_count = 0
    except_count = 0
    txt2code = {}  # keeps a mapping from txtacter to the code
    code2txt = {}  # keeps a mapping from  the code to the original txtacter
    txt2num = {}  # from character to a coded index number for the table (for use in torch.nn.F.Embedding)
    num2txt = {}  # keeps a mapping from  the index in the table to the original character
    # to encode we need to take in account that there are 4 segments

    def append_code(txt, index):
        multihot = encode_utf8_multihot(txt, segments)
        code_matrix.append(multihot)
        txt2code[txt] = multihot
        code2txt[bytes(multihot)] = txt
        txt2num[txt] = index
        num2txt[index] = txt

    # max number of elements that can be created in the bytes 2 3 and 4 of utf-8 codes
    max_6b = 2 ** 6
    # mask1 = 0b01111111
    # max1 = 0xff & mask1
    # generate all values for the first segment,
    for i in range(2**7):  # is the same as max1
        txt = str(bytes([i]), 'utf-8')
        code_count
        append_code(txt, code_count)
        code_count += 1

    if segments >= 2:
        # generate all values for the second segment,
        # index_offset_2 = 128
        max2_a = 0xff & 0b00011111
        for i in range(max2_a):
            for j in range(max_6b):
                # index =
                try:
                    txt = str(bytes([i | or_mask2[0], j | or_mask2[1]]), 'utf-8')
                    # index = index_offset_2 + i*j
                    append_code(txt, code_count)
                    code_count += 1
                except Exception as e:
                    # print(i, j, i | or_mask2[0], j | or_mask2[1])
                    except_count +=1
                    # raise e
                    pass
    if segments >= 3:
        # generate all values for the third segment,
        # index_offset_3 = index_offset_2 + (max2_a * max_6b)
        max3_a = 0xff & 0b00001111
        for i in range(max3_a):
            for j in range(max_6b):
                for k in range(max_6b):
                    try:
                        txt = str(bytes([i | or_mask3[0], j | or_mask3[1], k | or_mask3[2]]), 'utf-8')
                        # index = index_offset_3 + i*j*k
                        append_code(txt, code_count)
                        code_count += 1
                    except Exception as e:
                        # print(i, j, i | or_mask2[0], j | or_mask2[1])
                        except_count += 1
                        # raise e
                        pass

    if segments == 4:
        # generate all values for the fourth segment,
        # index_offset_4 = index_offset_3 + (max3_a * max_6b)
        max4_a = 0xff & 0b00000111
        for i in range(max4_a):
            for j in range(max_6b):
                for k in range(max_6b):
                    for l in range(max_6b):
                        try:
                            txt = str(bytes([i | or_mask4[0], j | or_mask4[1], k | or_mask4[2], l | or_mask4[3]]), 'utf-8')
                            # index = index_offset_4 + i*j*k*l
                            append_code(txt, code_count)
                            code_count += 1
                        except Exception as e:
                            # print(i, j, i | or_mask2[0], j | or_mask2[1])
                            except_count += 1
                            # raise e
                            pass
    print("number of codes = ", code_count)
    print("number of code_exceptions = ", except_count)
    code_matrix = np.stack(code_matrix)  # the issue here is that is a sparse matrix but I'm working as if  dense ...

    return code_matrix, txt2code, code2txt, txt2num, num2txt


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def decode_multihot_utf8(code):
    """
    code
    :param code: a 4 bytes length element containing the code for
    :return:
    """

    pass

