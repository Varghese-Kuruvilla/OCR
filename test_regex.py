#Script to test python regex
import re





def test_re():
    #Regular expression to be matched:\(*\b[^\s,;#&()]+[.,;)\n]* 
    raw_string = '(((ab c'
    print("raw_string",raw_string)
    re_tokens = re.compile(r'\(*\b[^s,]*')
    # re_tokens = re.compile(r'''\(*b[^s*]''')
    tokens = re_tokens.findall(raw_string)
    print("tokens",tokens)




if __name__ == '__main__':
    test_re()