line 1 - 52 : import library
line 63: English-Hindi dictionary
line 69 - 76: dataset reading
line 81 - 82: making separate Hindi and English wordlist from dictionary
line 89 - 92: English dataset tokenization into sentences
line 95 - 98: Bengali dataset tokenization into sentences
line 101 - 104: Hindi dataset tokenization into sentences
line 112 - 117: word tokenization of the sentences of the 1st paragraph in Hindi dataset
line 123 - 128: word tokenization of the sentences of the 1st paragraph in English dataset
line 132 - 140: stemming in English word tokens
line 176 - 191: English translation of Hindi word tokens for each sentence of that paragraph from dictionary
line 195 - 207: making a list of the matched words between engish word tokens and translated words of hindi tokens
line 229 - 249: stf calculations of the matched pairs
line 290 - 347: idtf calculation
line 378 - 394: sum(stf*idtf) for each sentence of the paragraph
line 433 - 450: length penalty
line 468 - 491: dynamic programming