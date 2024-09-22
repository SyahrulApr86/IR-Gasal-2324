import array
import time

import numpy as np

class StandardPostings:
    """ 
    Class dengan static methods, untuk mengubah representasi postings list
    yang awalnya adalah List of integer, berubah menjadi sequence of bytes.
    Kita menggunakan Library array di Python.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    Silakan pelajari:
        https://docs.python.org/3/library/array.html
    """

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        # Untuk yang standard, gunakan L untuk unsigned long, karena docID
        # tidak akan negatif. Dan kita asumsikan docID yang paling besar
        # cukup ditampung di representasi 4 byte unsigned.
        return array.array('L', postings_list).tobytes()

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = array.array('L')
        decoded_postings_list.frombytes(encoded_postings_list)
        return decoded_postings_list.tolist()


class VBEPostings:
    """ 
    Berbeda dengan StandardPostings, dimana untuk suatu postings list,
    yang disimpan di disk adalah sequence of integers asli dari postings
    list tersebut apa adanya.

    Pada VBEPostings, kali ini, yang disimpan adalah gap-nya, kecuali
    posting yang pertama. Barulah setelah itu di-encode dengan Variable-Byte
    Enconding algorithm ke bytestream.

    Contoh:
    postings list [34, 67, 89, 454] akan diubah dulu menjadi gap-based,
    yaitu [34, 33, 22, 365]. Barulah setelah itu di-encode dengan algoritma
    compression Variable-Byte Encoding, dan kemudian diubah ke bytesream.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    """

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes (dengan Variable-Byte
        Encoding). JANGAN LUPA diubah dulu ke gap-based list, sebelum
        di-encode dan diubah ke bytearray.

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        # Ubah ke gap-based list

        if not postings_list:
            return bytearray()

        gap_based_list = [postings_list[0]]

        for i in range(1, len(postings_list)):
            gap_based_list.append(postings_list[i] - postings_list[i-1])

        # Encode gap-based list
        return VBEPostings.vb_encode(gap_based_list)

    @staticmethod
    def vb_encode(list_of_numbers):
        """ 
        Melakukan encoding (tentunya dengan compression) terhadap
        list of numbers, dengan Variable-Byte Encoding
        """
        bytestream = bytearray()
        for n in list_of_numbers:
            bytes_array = VBEPostings.vb_encode_number(n)
            bytestream.extend(bytes_array)
        return bytestream

    @staticmethod
    def vb_encode_number(number):
        """
        Encodes a number using Variable-Byte Encoding
        Lihat buku teks kita!
        """
        bytes_array = bytearray()
        while True:
            bytes_array.insert(0, number % 128)
            if number < 128:
                break
            number = number // 128
        bytes_array[-1] += 128

        return bytes_array

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes. JANGAN LUPA
        bytestream yang di-decode dari encoded_postings_list masih berupa
        gap-based list.

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        gap_based_list = VBEPostings.vb_decode(encoded_postings_list)
        if not gap_based_list:
            return []

        postings_list = [gap_based_list[0]]

        for i in range(1, len(gap_based_list)):
            postings_list.append(postings_list[i-1] + gap_based_list[i])

        return postings_list

    @staticmethod
    def vb_decode(encoded_bytestream):
        """
        Decoding sebuah bytestream yang sebelumnya di-encode dengan
        variable-byte encoding.
        """
        numbers = []
        n = 0
        for byte in encoded_bytestream:
            if byte < 128:
                n = 128 * n + byte
            else:
                n = 128 * n + byte - 128
                numbers.append(n)
                n = 0
        return numbers

class EliasGammaPostings:
    """
    Elias Gamma Encoding untuk postings list.
    """

    @staticmethod
    def encode(postings_list):
        if not postings_list:
            return bytearray()

        gap_based_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_based_list.append(postings_list[i] - postings_list[i-1])

        # Encode gap-based list
        encoded_bits, _ = EliasGammaPostings.encode_array(np.array(gap_based_list))
        return encoded_bits.tobytes()

    @staticmethod
    def encode_array(arr):
        def gamma_encode(num):
            if num == 0:
                return '0'
            bin_rep = bin(num)[2:]  # remove the '0b' prefix
            length = len(bin_rep)
            return '0' * (length - 1) + bin_rep

        bitstring = ''.join([gamma_encode(x) for x in arr])
        extra_padding = 8 - (len(bitstring) % 8)
        bitstring += '0' * extra_padding  # pad the bitstring to make its length a multiple of 8

        byte_arr = [int(bitstring[i:i+8], 2) for i in range(0, len(bitstring), 8)]
        return np.array(byte_arr, dtype=np.uint8), len(bitstring) - extra_padding


    @staticmethod
    def decode(encoded_postings_list):
        encoded_bits = np.frombuffer(encoded_postings_list, dtype=np.uint8)
        n = len(encoded_postings_list) * 8
        gap_based_list = EliasGammaPostings.decode_array(encoded_bits, n)
        if not len(gap_based_list):
            return []

        postings_list = [gap_based_list[0]]
        for i in range(1, len(gap_based_list)):
            postings_list.append(postings_list[i-1] + gap_based_list[i])

        return postings_list

    @staticmethod
    def decode_array(b, n):
        b = np.unpackbits(b, count=n).view(bool)
        s = b.nonzero()[0]
        s = (s<<1).repeat(np.diff(s,prepend=-1))
        s -= np.arange(-1, len(s)-1)
        s = s.tolist()  # list has faster __getitem__
        ns = len(s)
        def gen():
            idx = 0
            yield idx
            while idx < ns:
                idx = s[idx]
                yield idx
        offs = np.fromiter(gen(), int)
        sz = np.diff(offs)>>1
        mx = sz.max() + 1 if sz.size > 0 else 0
        out = np.zeros(offs.size-1, int)
        for i in range(mx):
            out[b[offs[1:]-i-1] & (sz>=i)] += 1<<i
        return out

# referensi: https://gist.github.com/jogonba2/0a813e1b6a4d437a6dfe

if __name__ == '__main__':

    postings_list = [34, 67, 89, 454, 2345738, 191872978]
    for Postings in [StandardPostings, VBEPostings, EliasGammaPostings]:
        print(Postings.__name__)
        encoded_postings_list = Postings.encode(postings_list)
        print("byte hasil encode: ", encoded_postings_list)
        print("ukuran encoded postings: ", len(encoded_postings_list), "bytes")
        decoded_posting_list = Postings.decode(encoded_postings_list)
        print("hasil decoding: ", decoded_posting_list)
        assert decoded_posting_list == postings_list, "hasil decoding tidak sama dengan postings original pada {}".format(Postings.__name__)
        print()

    print("Uji coba dengan postings list yang lebih besar")
    postings_list_large = [i for i in range(10000000)]
    for Postings in [StandardPostings, VBEPostings, EliasGammaPostings]:
        start_time = time.time()
        print(Postings.__name__)

        encoded_postings_list = Postings.encode(postings_list_large)
        print("ukuran encoded postings: ", len(encoded_postings_list), "bytes")

        decoded_posting_list = Postings.decode(encoded_postings_list)

        print("Waktu indexing: ", time.time() - start_time, "detik")
        print()


