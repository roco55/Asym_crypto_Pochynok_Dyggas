# %% [markdown]
# # Лабораторна робота 1 з "Асиметричних криптосистем та протоколів"
# ## Тема: Побудова тестів для перевірки якості випадкових та псевдовипадкових послідовностей.
# 
# **Виконали**\
# Дигас Богдан, ФІ-03\
# Починок Юрій, ФІ-03

# %% [markdown]
# ## Additional functional
# 

# %%
import random as rand
import numpy as np
import scipy

# %%
def bin_to_dec(bin_n):
    dec_n = 0
    res = 0
    for i in range(len(bin_n)):
        res = bin_n[len(bin_n) - i - 1] * 2 ** i
        dec_n += res
    return dec_n

def bin_add(a, b):
    len_a = len(a)
    result = [0]*32
    carry = 0
    for i in range(len_a):
        result[len_a - i - 1] = a[len_a - i - 1] + b[len_a - i - 1] + carry
        carry = int(result[len_a - i - 1] / 2)
        result[len_a - i - 1] = result[len_a - i - 1] % 2
    return result


def bin_mul(a,b):
    len_a = len(a)
    res_add = [0]*32
    for i in range(len_a):
        res_mul = [0]*32
        for j in range(len_a-i):
            res_mul[len_a-1-i-j] = a[len_a-1-i] & b[len_a-1-j]
        res_add = bin_add(res_add, res_mul)
        res_add
    return res_add

def generate_bit_seq(n):
    seq = [0]*n
    for i in range(n):
        seq[i] = rand.randint(0, 1)
    return seq

def bits_to_bytes(x):
    length = int((len(x)+7)/8)
    result = [0]*length
    x = [str(a) for a in x]
    for i in range(length):
        result[i] = ''.join((x[8*i : 8*(i+1)]))
    return result



# %% [markdown]
# ## Вбудований генератор

# %%
def built_in_randomizer(n):
    return(generate_bit_seq(n))

# %% [markdown]
# ## Lehmer low/high
# 
# 

# %% [markdown]
# ### Define constants $a, c, x_0, m, \alpha$

# %%
#constants
a = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
c = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1]
x0 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
m = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
alpha = 0.05 # Точність

# %%
def lehmer_low(x0,n):
    res = [0]*8
    length = 8
    result =[0]*n
    for i in range(n):
        x0 = bin_add((bin_mul(a,x0)),c)
        for j in range(length):
            res[j] = x0[32-length+j]
        result[i] = f'{(bin_to_dec(res)%256):08b}'

    return result

def lehmer_high(x0,n):
    res = [0]*8
    result =[0]*n
    result[0] = f'{(bin_to_dec(res)):08b}'
    length = 8
    for i in range(1, n):
        x0 = bin_add((bin_mul(a,x0)),c)
        for j in range(length):
            res[j] = x0[j]
        result[i] = f'{(bin_to_dec(res)%256):08b}'
    
    return result


# %% [markdown]
# ## L20
# 

# %%
def L20(n):
    seq = generate_bit_seq(20)
    result = [0]*n
    for i in range(20):
        result[i] = seq[i]
    for i in range(20,n):
        result[i] = result[i-3]^result[i-5]^result[i-9]^result[i-20]
    return result

# %% [markdown]
# ## L89

# %%
def L89(n):
    seq = generate_bit_seq(89)
    result = [0]*n
    for i in range(89):
        result[i] = seq[i]
    for i in range(89,n):
        result[i] = result[i-38]^result[i-89]
    return result

# %% [markdown]
# ## Geffe

# %%
def L11(n):
    seq = generate_bit_seq(11)
    result = [0]*n
    for i in range(11):
        result[i] = seq[i]
    for i in range(11,n):
        result[i] = result[i-11]^result[i-9]
    return result

def L9(n):
    seq = generate_bit_seq(9)
    result = [0]*n
    for i in range(9):
        result[i] = seq[i]
    for i in range(9,n):
        result[i] = result[i-9]^result[i-8]^result[i-6]^result[i-5]
    return result

def L10(n):
    seq = generate_bit_seq(10)
    result = [0]*n
    for i in range(10):
        result[i] = seq[i]
    for i in range(10,n):
        result[i] = result[i-10]^result[i-7]
    return result

def geffe(n):
    result = [0]*n
    x = L11(n)
    y = L9(n)
    s = L10(n)
    for i in range(n):
        result[i]=s[i]&x[i]^(1^s[i])&y[i]
    return result


# %% [markdown]
# ## Бібліотекар

# %%
import io

def Librarian(n):
    file = io.open("book-war-and-peace.txt", mode='r', encoding='utf-8')
    text = file.read()
    clean_text = ""
    for i in text:
        if ord(i) < 256:
            clean_text += i
    
    print("The text was already clean" if len(clean_text) == len(text) else "The text wasn't clean, we cleaned " + str(len(text) - len(clean_text)) + " symbols")
    
    max_bytes = len(clean_text)
    if n > max_bytes:
        print("Sorry, the maximum bytes here is: ", max_bytes, ", please try again.")
        return 1
    
    result = [0]*n
    for i in range(n):
        result[i] = f'{(ord(clean_text[i])):08b}'
    
    return result


# %% [markdown]
# ## Вольфрам

# %%
def cycle_shift_l(n, n_bits, shift):
    shift = shift % n_bits
    return ((n << shift) | (n >> (n_bits - shift))) % 2**n_bits

def cycle_shift_r(n, n_bits, shift):
    shift = shift % n_bits
    return ((n >> shift) | (n << (n_bits - shift))) % 2**n_bits

    

def Wolfram(n):
    r = np.uint32(rand.randint(0, 2**32-1))
    result = [0]*n

    for i in range(n):
        result[i] = r % 2
        r = cycle_shift_l(r, 32, 1) ^ (r | cycle_shift_r(r, 32, 1))
    return result

# %% [markdown]
# ## BM

# %%
p1 = int("CEA42B987C44FA642D80AD9F51F10457690DEF10C83D0BC1BCEE12FC3B6093E3",16)
a1 = int("5B88C41246790891C095E2878880342E88C79974303BD0400B090FE38A688356",16)
q1 = "675215CC3E227D3216C056CFA8F8822BB486F788641E85E0DE77097E1DB049F1"

def BM(n):
    comparing_number = (p1-1)/2
    t = rand.randrange(0,a1-1)
    result = [0]*n
    for i in range(n):
        t = pow(a1,t,p1)
        if(t<comparing_number):
            result[i]=1
    return result

# %% [markdown]
# ## BM bytes

# %%
p1 = int("CEA42B987C44FA642D80AD9F51F10457690DEF10C83D0BC1BCEE12FC3B6093E3",16)
a1 = int("5B88C41246790891C095E2878880342E88C79974303BD0400B090FE38A688356",16)
q1 = "675215CC3E227D3216C056CFA8F8822BB486F788641E85E0DE77097E1DB049F1"

def BM_bytes(n):
    t = rand.randrange(0,a1-1)
    result = [0]*n 
    for i in range(n):
        t = pow(a1,t,p1)
        result[i]=f'{(t*256 //(p1-1)):08b}'
    return result

# %% [markdown]
# ## BBS

# %%
p2 = int("D5BBB96D30086EC484EBA3D7F9CAEB07",16)
q2 = int("425D2B9BFDB25B9CF6C416CC6E37B59C1F",16)
n2 = p2*q2
def bbs(n):
    r = int(rand.randrange(2,n2))
    result = [0]*n
    for i in range(n):
        r = pow(r,2,n2)
        result[i] = r%2
    return result


# %% [markdown]
# ## BBS bytes

# %%
p3 = int("D5BBB96D30086EC484EBA3D7F9CAEB07",16)
q3 = int("425D2B9BFDB25B9CF6C416CC6E37B59C1F",16)
n3 = p3*q3
def bbs_bytes(n):
    r = int(rand.randrange(2,n3))
    result = [0]*n
    for i in range(n):
        r = pow(r,2,n3)
        result[i] = f'{(r%256):08b}'
    return result

# %% [markdown]
# ## Analysis
# 

# %%
#Критерій перевірки рівноімовірності знаків

def test1(x):
    freq = {}
    for i in x:
        if i in freq:
            freq[i] += 1
        else:
            freq[i] = 1

    length = len(x)/len(freq)

    xi = 0
    for i in freq:
        xi = xi + (freq[i]-length)**2/length
    #use scipy
    xi_alpha = scipy.stats.chi2.ppf(1 - alpha, 2**8 - 1)
    #return true/false
    print("\nActual Xi^2 = ", xi, "\nXi^2 boundary = ", xi_alpha)
    return (xi <= xi_alpha)


# %%
#Критерій перевірки незалежності знаків
def test2(x):
    pairs = np.zeros((2**8, 2**8), dtype=int)
    sum = 0
    for i in range(0, len(x), 2):
        pairs[int(x[i],2)][int(x[i+1],2)] = pairs[int(x[i],2)][int(x[i+1],2)] + 1
    for i in range(2**8):
        for j in range(2**8):
            dilnuk = np.sum(pairs[i,:])*np.sum(pairs[:,j])
            # print(dilnuk)
            if dilnuk != 0:
                sum += (pairs[i,j]**2)/dilnuk
    xi = (len(x)/2)*(sum-1)

    #use scipy
    xi_alpha = scipy.stats.chi2.ppf(1 - alpha, (2**8 - 1)**2)
    #return true/false
    print("\nActual Xi^2 = ", xi, "\nXi^2 boundary = ", xi_alpha)
    return (xi <= xi_alpha)




# %%
#Критерій перевірки однорідності двійкової послідовності
def test3(x):
    sum = 0   
    r = 200
    pairs = np.zeros((2**8, r), dtype=int)
    #розбиття на інтервали
    for j in range(0, r):
        for i in range(0, len(x) // r):
            c = int(x[r*j + i],2)
            pairs[c, j] += 1
    #хі квадрат
    for i in range(2**8):
        for j in range(r):
            dilnuk = np.sum(pairs[i,:])*np.sum(pairs[:,j])
            if dilnuk != 0:
                sum += (pairs[i,j]**2)/dilnuk
    xi = (len(x))*(sum-1)
    #use scipy
    xi_alpha = scipy.stats.chi2.ppf(1 - alpha, (2**8-1) * (r-1))
    #return true/false
    print("\nActual Xi^2 = ", xi, "\nXi^2 boundary = ", xi_alpha)
    return (xi <= xi_alpha)


# %% [markdown]
# ## Тестування

# %%
def test_bits(thing):
    print("Критерій перевірки рівноімовірності знаків: ", test1(bits_to_bytes(thing)))
    print("Критерій перевірки незалежності знаків: ", test2(bits_to_bytes(thing)))
    print("Критерій перевірки однорідності двійкової послідовності: ", test3(bits_to_bytes(thing)))
    
def test_bytes(thing):
    print("Критерій перевірки рівноімовірності знаків: ", test1(thing))
    print("Критерій перевірки незалежності знаків: ", test2(thing))
    print("Критерій перевірки однорідності двійкової послідовності: ", test3(thing))

# %% [markdown]
# ### Built-in-RNG
# 
# <a id='1'></a>

# %%
thing = built_in_randomizer(1000000)

test_bits(thing)

# %% [markdown]
# ### LehmerLow
# <a id='2'></a>

# %%
thing = lehmer_low(x0, 125000)

test_bytes(thing)

# %% [markdown]
# ### LehmerHigh
# <a id='3'></a>

# %%
thing = lehmer_high(x0, 125000)

test_bytes(thing)

# %% [markdown]
# ### L20
# <a id='4'></a>

# %%
thing = L20(8000000)

test_bits(thing)

# %% [markdown]
# ### L89
# <a id='5'></a>

# %%
thing = L89(1000000)

test_bits(thing)

# %% [markdown]
# ### Geffe
# <a id='6'></a>

# %%
thing = geffe(1000000)

test_bits(thing)

# %% [markdown]
# ### Wolfram
# <a id='7'></a>

# %%
thing = Wolfram(8000000)

test_bits(thing)

# %% [markdown]
# ### Librarian
# <a id='8'></a>

# %%
thing = Librarian(125000)

test_bytes(thing)

# %% [markdown]
# ### BM-Bits
# <a id='9'></a>

# %%
thing = BM(8000000)

test_bits(thing)

# %% [markdown]
# ### BM-Bytes
# <a id='10'></a>

# %%
thing = BM_bytes(250000)

test_bytes(thing)

# %% [markdown]
# ### BBS-Bits
# <a id='11'></a>

# %%
thing = bbs(1000000)

test_bits(thing)

# %% [markdown]
# ### BBS-Bytes
# <a id='12'></a>

# %%
thing = bbs_bytes(125000)

test_bytes(thing)

# %% [markdown]
# ## Результати в табличній формі

# %% [markdown]
# | Генератор | Рівноімовірність | Незалежність | Однорідність | $\chi^2$ / $\chi^2_{1-\alpha}$|
# | :-: | :-: | :-: | :-: | :-: |
# | Built-in-RNG | $\checkmark$ | $\checkmark$ | $\checkmark$ | <a href='#1'>Result</a> |
# | LehmerLow | $\checkmark$ | $\times$ | $\checkmark$ | <a href='#2'>Result</a> | 
# | LehmerHigh | $\checkmark$ | $\checkmark$ | $\checkmark$ | <a href='#3'>Result</a> | 
# | L20 | $\checkmark$ | $\checkmark$ | $\checkmark$ | <a href='#4'>Result</a> | 
# | L89 | $\checkmark$ | $\checkmark$ | $\checkmark$ | <a href='#5'>Result</a> | 
# | Geffe | $\checkmark$ | $\times$ | $\checkmark$ | <a href='#6'>Result</a> | 
# | Wolfram | $\times$ | $\times$ | $\checkmark$ |  <a href='#7'>Result</a>| 
# | Librarian | $\times$ | $\times$ | $\checkmark$ |  <a href='#8'>Result</a>| 
# | BM | $\checkmark$ | $\checkmark$ | $\checkmark$ |  <a href='#9'>Result</a>| 
# | BM_bytes | $\checkmark$ | $\checkmark$ | $\checkmark$ |  <a href='#10'>Result</a>| 
# | BBS | $\checkmark$ | $\checkmark$ | $\checkmark$ | <a href='#11'>Result</a>|
# | BBS_bytes | $\checkmark$ | $\checkmark$ | $\checkmark$ |  <a href='#12'>Result</a>| 

# %% [markdown]
# ## Висновок
# 
# - **Мета:** Розібратися з різноманітними генераторами випадкових чисел, дослідити хід їх роботи, проаналізувати їхні властивості та оцінити їх якість
# - **Хід роботи:** З допомогою методички написання генераторів та тестів не заставляло докладати особливих зусиль та прибігати до дослідження сторонніх джерел, так як ми просто напряму слідували методичним вказівкам. Проблеми виникли з генератором "Бібліотекар", так як вибраний текст французькою містив символ який не влізав в кодування utf-8, через що псувалась однорідність. Цей недолік виправлявся "очисткою" тексту, або вибором простого англомовного тексту (ми зробили обидва). 
# - **Результати:** Явно не варто використовувати генератори "Вольфрам" та "Бібліотекар". Щодо інших, варто звертати увагу на обраховані статистичні дані. Будьте обережні з середовищами на комп'ютері, а то простий акт завантажування scipy може знести вам голову.


